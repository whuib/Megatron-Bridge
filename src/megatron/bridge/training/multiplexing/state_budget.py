# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offload only as much Adam state as the copy bandwidth can hide behind the computation.

Parameters stay on the GPU. A module's FP32 gradients exist only from its first backward of the step
to its Adam step, which needs no copies. The Adam state (FP32 main parameters and both moments) of a
planned subset of modules lives in pinned host memory: a copy stream prefetches it before the
module's Adam step and writes it back right after. That step runs either inline, right after the
module's last backward of the step (inside the backward of the last micro-batch), or deferred to
right before the module's first forward of the next step, with the learning rate and step of the
step that produced the gradients. Neither window contains the activation peak at the end of a
micro-batch's forward.

Device synchronizations (:meth:`BudgetedStateOffload.synchronize`, and the end of the step) drain
the copy stream, so they split a step into segments whose copies must finish inside the segment.
The plan comes from the copy bandwidth, measured with every rank copying at once, and from the
compute timeline of a step: modules are offloaded, largest first, while a simulation of the copy
stream of every segment predicts no wait at a fraction ``margin`` of the bandwidth. The first
replayed step is the measurement (every Adam step then runs after the backward). Whenever the
computation of a planned step waited on a copy, the margin shrinks and the plan is rebuilt from that
step's timeline.

The plan hides copy time; it does not account for copies slowing the computation they overlap. On
systems where host-device DMA contends with NVLink collectives (an A100 SXM node with NCCL's Simple
protocol runs a large all-reduce 2-3x slower while a device-to-host copy is in flight), a planned
step without a single copy wait can still be slower than a step with all Adam state resident, in
proportion to the offloaded bytes. ``step_summary`` reports those bytes next to the copy wait.
"""

import bisect
import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .offload_plan import (
    CopyRates,
    GroupProfile,
    StepProfile,
    UnitProfile,
)
from .offload_plan import (
    plan_offload as plan_mixed_offload,
)
from .state_offload import (
    _BACKWARD,
    _FORWARD,
    _GRAD,
    _OPTIM,
    _PARAM,
    ModuleStateOffload,
    OffloadedModel,
    _allocate,
    _PinnedArena,
    _release,
    _Unit,
)


logger = logging.getLogger(__name__)

RESIDENT, INLINE, DEFERRED = "resident", "inline", "deferred"
_SYNC = (-2, 3)
# Each replan after a step that waited on a copy multiplies the bandwidth margin by this factor.
_MARGIN_DECAY = 0.9
_BANDWIDTH_PROBE_BYTES = 256 * 2**20
_CHUNK_BYTES = 16 * 2**20

_Hyperparameters = list[tuple[float, float, float, float, int, float]]


class _PacedCopier:
    """Host-device copies in chunks alternating between two streams, each waiting for the previous one.

    A copy engine serves copies of all streams in order, so a small copy of the compute stream (an
    ``.item()``, a tensor built from host data) would wait for a whole module's state; with one chunk
    queued at a time it waits for at most one chunk, at the same bandwidth.
    """

    def __init__(self) -> None:
        self._streams = (torch.cuda.Stream(), torch.cuda.Stream())
        self._turn = 0
        self._waits: list[torch.cuda.Event] = []

    @property
    def stream(self) -> torch.cuda.Stream:
        """Stream of the copy buffers' allocations; every copy is ordered after all earlier ones."""
        return self._streams[0]

    def wait_stream(self, stream: torch.cuda.Stream) -> None:
        """Order the next copies after the work queued on ``stream`` so far."""
        event = torch.cuda.Event()
        event.record(stream)
        self._waits.append(event)

    def copy(self, destination: torch.Tensor, source: torch.Tensor) -> torch.cuda.Event:
        """Queue ``destination.copy_(source)``; returns an event completing with it."""
        step = _CHUNK_BYTES // destination.element_size()
        for start in range(0, destination.numel(), step):
            stream = self._streams[self._turn]
            self._turn ^= 1
            for event in self._waits:
                stream.wait_event(event)
            with torch.cuda.stream(stream):
                destination[start : start + step].copy_(source[start : start + step], non_blocking=True)
                tail = torch.cuda.Event()
                tail.record()
            self._waits = [tail]
        return self.done()

    def done(self, *, timing: bool = False) -> torch.cuda.Event:
        """An event completing with every copy queued so far."""
        stream = self._streams[self._turn]
        for event in self._waits:
            stream.wait_event(event)
        event = torch.cuda.Event(enable_timing=timing)
        event.record(stream)
        self._waits = [event]
        return event


@dataclass(frozen=True)
class OffloadCandidate:
    """A module whose Adam state may be offloaded.

    Attributes:
        index: Module index.
        nbytes: Bytes of its Adam state; an offloaded step copies them in and back out.
        deferred_at: Seconds into the step at which its deferred Adam step runs (its first forward).
        inline_at: Seconds into the step at which its inline Adam step runs (right after its last
            backward), or None if it has none.
    """

    index: int
    nbytes: int
    deferred_at: float
    inline_at: float | None


def copy_slack(
    updates: list[tuple[float, int]], *, start: float, end: float, h2d_rate: float, d2h_rate: float, lookahead: int
) -> float:
    """Smallest lead on a simulated copy stream of a prefetch over its Adam step, or of ``end`` over the
    last write-back (negative: a wait).

    The copy stream is idle at ``start`` and must be idle again at ``end``. At ``start`` it prefetches
    the state of the first ``lookahead`` Adam steps ``(seconds into the step, bytes)``; after each Adam
    step it writes that module's state back, then prefetches the module ``lookahead`` Adam steps
    ahead. Rates are in bytes per second.
    """
    if not updates:
        return math.inf
    ordered = sorted(updates)
    count = len(ordered)
    ready = [0.0] * count
    copy_free = start
    for position in range(min(lookahead, count)):
        copy_free += ordered[position][1] / h2d_rate
        ready[position] = copy_free
    slack = math.inf
    for position, (at, nbytes) in enumerate(ordered):
        slack = min(slack, at - ready[position])
        copy_free = max(copy_free, at) + nbytes / d2h_rate
        following = position + lookahead
        if following < count:
            copy_free += ordered[following][1] / h2d_rate
            ready[following] = copy_free
    return min(slack, end - copy_free)


def plan_offload(
    candidates: list[OffloadCandidate],
    *,
    barriers: list[float],
    end: float,
    h2d_rate: float,
    d2h_rate: float,
    lookahead: int,
) -> dict[int, str]:
    """Offload Adam state greedily, largest first, while no copy makes the computation wait.

    ``barriers`` are the times of the device synchronizations inside the step and ``end`` the time of
    the one ending it; an Adam step at a synchronization runs before it. Each offloaded module takes
    whichever of its windows leaves the larger slack in its segment.

    Returns:
        ``RESIDENT``, ``INLINE`` or ``DEFERRED`` per candidate index.
    """
    bounds = [0.0, *barriers, end]
    plan = {candidate.index: RESIDENT for candidate in candidates}
    chosen: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda candidate: candidate.nbytes, reverse=True):
        best = None
        for mode, at in ((DEFERRED, candidate.deferred_at), (INLINE, candidate.inline_at)):
            if at is None:
                continue
            segment = bisect.bisect_left(barriers, at)
            slack = copy_slack(
                [*chosen[segment], (at, candidate.nbytes)],
                start=bounds[segment],
                end=bounds[segment + 1],
                h2d_rate=h2d_rate,
                d2h_rate=d2h_rate,
                lookahead=lookahead,
            )
            if slack >= 0 and (best is None or slack > best[0]):
                best = (slack, mode, segment, at)
        if best is not None:
            _, mode, segment, at = best
            plan[candidate.index] = mode
            chosen[segment].append((at, candidate.nbytes))
    return plan


def measure_copy_bandwidth(nbytes: int = _BANDWIDTH_PROBE_BYTES) -> tuple[float, float]:
    """Host-to-device and device-to-host bytes per second of pinned copies, every rank copying at once.

    Returns:
        The minimum over ranks of each rate.
    """
    arena = _PinnedArena(nbytes, torch.uint8)
    host = arena.take(nbytes)
    device = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    repeats = 4
    rates = []
    try:
        for to_device in (True, False):
            for timed in (False, True):
                torch.cuda.synchronize()
                if timed and dist.is_initialized():
                    dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    start.record()
                    for _ in range(repeats if timed else 1):
                        if to_device:
                            device.copy_(host, non_blocking=True)
                        else:
                            host.copy_(device, non_blocking=True)
                    end.record()
                end.synchronize()
            rates.append(repeats * nbytes / (start.elapsed_time(end) / 1e3))
    finally:
        torch.cuda.synchronize()
        arena.close()
    measured = torch.tensor(rates, dtype=torch.float64, device="cuda")
    if dist.is_initialized():
        dist.all_reduce(measured, op=dist.ReduceOp.MIN)
    h2d, d2h = measured.tolist()
    return h2d, d2h


def measure_copy_rates(nbytes: int = _BANDWIDTH_PROBE_BYTES) -> CopyRates:
    """Unidirectional and simultaneous pinned copy rates, minimum over ranks.

    Args:
        nbytes: Bytes of each timed copy.

    Returns:
        Host-to-device and device-to-host rates alone and while the other direction copies.
    """
    h2d, d2h = measure_copy_bandwidth(nbytes)
    arena = _PinnedArena(nbytes * 2, torch.uint8)
    host_in, host_out = arena.take(nbytes), arena.take(nbytes)
    device_in = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    device_out = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    shared = []
    try:
        for timed in (False, True):
            torch.cuda.synchronize()
            if timed and dist.is_initialized():
                dist.barrier()
            starts = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
            ends = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
            repeats = 4 if timed else 1
            with torch.cuda.stream(streams[0]):
                starts[0].record()
                for _ in range(repeats):
                    device_in.copy_(host_in, non_blocking=True)
                ends[0].record()
            with torch.cuda.stream(streams[1]):
                starts[1].record()
                for _ in range(repeats):
                    host_out.copy_(device_out, non_blocking=True)
                ends[1].record()
            ends[0].synchronize()
            ends[1].synchronize()
        for start, end in zip(starts, ends, strict=True):
            shared.append(repeats * nbytes / (start.elapsed_time(end) / 1e3))
    finally:
        torch.cuda.synchronize()
        arena.close()
    measured = torch.tensor(shared, dtype=torch.float64, device="cuda")
    if dist.is_initialized():
        dist.all_reduce(measured, op=dist.ReduceOp.MIN)
    h2d_shared, d2h_shared = measured.tolist()
    return CopyRates(h2d, d2h, h2d_shared, d2h_shared)


class _ActivationArena:
    """One pinned byte buffer carved into the saved tensors of a step."""

    def __init__(self, nbytes: int) -> None:
        self.storage = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.offset = 0

    def reset(self) -> None:
        """Reuse the buffer at the start of a step, after the previous step unpacked."""
        self.offset = 0

    def take(self, numel: int, dtype: torch.dtype) -> torch.Tensor | None:
        """A typed view of ``numel`` elements, or None when the buffer is full."""
        width = torch.empty((), dtype=dtype).element_size()
        start = (self.offset + 15) & ~15
        size = numel * width
        if start + size > self.storage.numel():
            return None
        view = self.storage[start : start + size].view(dtype)
        self.offset = start + size
        return view


def _is_param_hold(hold: tuple, unit: _Unit) -> bool:
    """Whether ``hold`` is a pending parameter release of ``unit``."""
    return len(hold) == 4 and hold[0] == "storage" and hold[2] is unit and hold[3] == _PARAM


def _offloadable_activation(tensor: torch.Tensor, param_ptrs: set[int]) -> bool:
    """Whether dropping ``tensor`` after a copy frees exactly its own bytes."""
    if not tensor.is_cuda or not tensor.is_contiguous() or tensor.numel() == 0:
        return False
    storage = tensor.untyped_storage()
    stored = storage.nbytes() if callable(storage.nbytes) else storage.nbytes
    return storage.data_ptr() not in param_ptrs and stored == tensor.numel() * tensor.element_size()


class _ActivationSession:
    """Saved-tensor hooks that count, or copy out, one module forward's activations."""

    def __init__(self, engine: "BudgetedStateOffload", quota: int | None, param_ptrs: set[int]) -> None:
        self.engine = engine
        self.quota = quota
        self.param_ptrs = param_ptrs
        self.nbytes = 0
        self.taken = 0
        self._seen: set[int] = set()

    def pack(self, tensor: torch.Tensor) -> torch.Tensor | tuple:
        """Count an owning saved tensor; copy it out while the byte quota is not yet met."""
        if not _offloadable_activation(tensor, self.param_ptrs):
            return tensor
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in self._seen:
            return tensor
        self._seen.add(ptr)
        self.nbytes += tensor.nbytes
        if self.quota is None or self.taken >= self.quota:
            return tensor
        host = (
            self.engine._activation_arena.take(tensor.numel(), tensor.dtype) if self.engine._activation_arena else None
        )
        if host is None:
            return tensor
        flat = tensor.detach().view(-1)
        event = self.engine._copy_out(host, flat)
        self.engine._holds.append(("tensor", event, [tensor]))
        self.taken += tensor.nbytes
        self.engine.step_traffic["d2h"] += tensor.nbytes
        return ("full", host, tuple(tensor.shape), tensor.dtype, event)

    def unpack(self, packed: torch.Tensor | tuple) -> torch.Tensor:
        """Reload a copied activation onto the compute device."""
        if isinstance(packed, torch.Tensor):
            return packed
        _, host, shape, dtype, event = packed
        event.synchronize()
        device = torch.device("cuda", torch.cuda.current_device())
        restored = torch.empty(shape, dtype=dtype, device=device)
        restored.view(-1).copy_(host)
        self.engine.step_traffic["h2d"] += host.numel() * host.element_size()
        return restored


class BudgetedStateOffload(ModuleStateOffload):
    """Keep parameters on the GPU and offload the Adam state the copy bandwidth can hide.

    With ``headroom_bytes``, a measured step instead records the allocated memory and the saved
    activations, and the plan offloads whatever mix of Adam state, saved activations and parameters
    keeps the predicted peak at least ``headroom_bytes`` under the memory PyTorch can use. Adam state
    and parameters are offloaded per module; activations are offloaded per saved tensor until the
    planned byte count of that forward is met.

    Used like :class:`ModuleStateOffload`; every device synchronization inside a step must go through
    :meth:`synchronize`, and the step must be followed by one (it drains the last write-backs).
    Without :meth:`final_microbatch` no module has an inline window, so only deferred Adam steps
    offload. Deferred steps complete in :meth:`close`.

    Args:
        models: The offloaded models; their parameters are moved into the module buffers.
        margin: Fraction of the measured bandwidth the first plan may use.
        lookahead: Prefetches in flight ahead of the next offloaded Adam step of a segment.
        bandwidth: ``(host-to-device, device-to-host)`` bytes per second, or a :class:`CopyRates`;
            measured when None.
        headroom_bytes: Free bytes to leave at the predicted peak. None keeps the bandwidth-only plan.
    """

    _HOST_COMPONENTS = (_OPTIM,)

    def __init__(
        self,
        models: list[OffloadedModel],
        *,
        margin: float = 0.9,
        lookahead: int = 2,
        bandwidth: tuple[float, float] | CopyRates | None = None,
        headroom_bytes: int | None = None,
    ) -> None:
        self._margin = margin
        self._lookahead = lookahead
        self._throttle = 1
        self._headroom = headroom_bytes
        self._param_ids: set[int] = set()
        self._activation_quota: list[int] = []
        self._activation_arena: _ActivationArena | None = None
        self._holds: list[tuple] = []
        self._group_records: list[list[int | None]] = []
        self._open_groups: dict[int, list[int]] = defaultdict(list)
        self._memory_samples: list[int] = []
        self._measuring = False
        self._forward_cursor = 0
        self._memory_budget = 0
        self._predicted_peak = 0
        self._offloaded_activation_bytes = 0
        self._offloaded_param_bytes = 0
        self._min_free: int | None = None
        if isinstance(bandwidth, CopyRates):
            self._copy_rates = bandwidth
            self._rates = (bandwidth.h2d, bandwidth.d2h)
        elif headroom_bytes is None:
            self._copy_rates = None
            self._rates = bandwidth if bandwidth is not None else measure_copy_bandwidth()
        else:
            self._copy_rates = (
                measure_copy_rates() if bandwidth is None else CopyRates(*bandwidth, bandwidth[0], bandwidth[1])
            )
            self._rates = (self._copy_rates.h2d, self._copy_rates.d2h)
        self._copier = _PacedCopier()
        super().__init__(models, overlap=False)
        # Until the first plan every module's Adam state is offloaded and steps after the backward.
        self._mode = {unit.index: INLINE for unit in self._units}
        # Offloaded modules of every segment in Adam-step order, and each one's (segment, position).
        self._segments: list[list[int]] = []
        self._slot: dict[int, tuple[int, int]] = {}
        # Event index of each offloaded module's Adam step.
        self._update_at: dict[int, int] = {}
        self._segment = 0
        self._pending: dict[int, tuple[_Hyperparameters, int]] = {}
        # (candidates, synchronization times, step time) of the last measurement step.
        self._measurement: tuple[list[OffloadCandidate], list[float], float] | None = None
        self._profiling = True
        self._replayed = False
        self._diverged = False
        self._marks: list[torch.cuda.Event] = []
        self._stalls: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None
        self.step_stall = 0.0
        self.step_timeline = 0.0

    def _log_setup(self) -> None:
        logger.info(
            "budgeted module state offload: %d modules, %.2f GiB Adam state, copy bandwidth %.1f GB/s "
            "host-to-device and %.1f GB/s device-to-host",
            len(self._units),
            self._adam_bytes(self._units) / 2**30,
            self._rates[0] / 1e9,
            self._rates[1] / 1e9,
        )

    @staticmethod
    def _adam_bytes(units: list[_Unit]) -> int:
        return sum(unit.nbytes(_OPTIM) for unit in units)

    def step_summary(self) -> dict[str, float]:
        """Metrics of the last training step, for logging."""
        offloaded = [unit for unit in self._units if self._mode[unit.index] != RESIDENT]
        summary = {
            **super().step_summary(),
            "offloaded_adam_gib": self._adam_bytes(offloaded) / 2**30,
            "timeline_s": self.step_timeline,
            "copy_wait_ms": self.step_stall * 1e3,
            "margin": self._margin,
        }
        if self._headroom is not None:
            summary["offloaded_activation_gib"] = self._offloaded_activation_bytes / 2**30
            summary["offloaded_param_gib"] = self._offloaded_param_bytes / 2**30
            summary["memory_budget_gib"] = self._memory_budget / 2**30
            summary["predicted_peak_gib"] = self._predicted_peak / 2**30
            summary["min_free_mib"] = 0.0 if self._min_free is None else self._min_free / 2**20
        return summary

    def _adopt(self, unit: _Unit) -> None:
        optim = unit.device[_OPTIM]
        unit.host[_OPTIM] = self._arenas[_OPTIM].take(optim.numel())
        unit.host[_OPTIM].copy_(optim)
        _release(optim)
        _release(unit.device[_GRAD])
        unit.resident = {_PARAM}

    @staticmethod
    def _timing_event(stream: torch.cuda.Stream | None = None) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def _advance(self, entry: tuple[int, int]) -> int:
        index = super()._advance(entry)
        self._marks.append(self._timing_event())
        return index

    def _diverge(self, index: int) -> None:
        self._diverged = True
        super()._diverge(index)

    @property
    def _planned(self) -> bool:
        """Whether this step follows the plan (inline and deferred Adam steps)."""
        return self._replaying and not self._profiling

    def begin_step(self) -> None:
        """Start a training step: gradients restart from zero."""
        super().begin_step()
        self._replayed = self._replaying
        self._diverged = False
        self._marks, self._stalls = [], []
        self._start = self._timing_event()
        self._segment = 0
        self._forward_cursor = 0
        self._measuring = self._headroom is not None and self._replayed and self._profiling
        self._group_records = []
        self._open_groups = defaultdict(list)
        self._memory_samples = []
        self._min_free = None
        if self._activation_arena is not None:
            self._activation_arena.reset()
        self._sample_memory()
        if self._planned:
            self._prefetch_segment()

    def synchronize(self) -> None:
        """Synchronize the device; inside a step the plan keeps the copies of a segment before it."""
        if not self._active:
            torch.cuda.synchronize()
            return
        index = self._advance(_SYNC)
        self._settle_previous(index)
        self._record_copy_tail(index)
        torch.cuda.synchronize()
        self._segment += 1
        if self._planned:
            self._prefetch_segment()
        self._reap_copies()
        self._sample_memory()

    def _record_copy_tail(self, index: int) -> None:
        """Record how long the copies run past the compute stream at a synchronization."""
        self._stalls.append((index, self._timing_event(), self._copier.done(timing=True)))

    def _prefetch_segment(self) -> None:
        if self._segment < len(self._segments):
            for unit_index in self._segments[self._segment][: self._lookahead]:
                if _OPTIM not in self._units[unit_index].resident:
                    self._prefetch(self._units[unit_index])

    def _settle_previous(self, index: int) -> None:
        previous = self._previous
        if previous is not None and not previous.stepped and self._planned and previous.final_backward == index - 1:
            self._settle(previous, index)

    def final_microbatch(self) -> None:
        """Mark the last micro-batch and sample memory when a headroom plan is being measured."""
        super().final_microbatch()
        self._reap_copies()
        self._sample_memory()

    def _forward_pre_hook(self, unit: _Unit):
        def hook(module, args):
            if _PARAM not in unit.resident:
                self._ensure_params(unit)
            self._event(unit, _FORWARD)
            self._begin_capture(unit)

        return hook

    def _forward_hook(self, unit: _Unit):
        base = super()._forward_hook(unit)

        def hook(module, args, output):
            base(module, args, output)
            self._end_capture(unit)
            if self._planned and unit.index in self._param_ids:
                self._offload_params(unit)
            self._reap_copies()
            self._sample_memory()

        return hook

    def _event(self, unit: _Unit, kind: int) -> None:
        if not self._active:
            raise RuntimeError(f"module state offload: {unit.name} ran outside a training step")
        if kind == _BACKWARD and _PARAM not in unit.resident:
            self._ensure_params(unit)
        index = self._advance((unit.index, kind))
        if unit.stepped:
            raise RuntimeError(f"module state offload: {unit.name} ran after its optimizer step")
        self._settle_previous(index)
        if kind == _FORWARD and unit.index in self._pending:
            self._step_module(unit, *self._pending.pop(unit.index), index)
        elif kind == _FORWARD and self._planned and self._update_at.get(unit.index) == index:
            # A deferred slot without a deferred step (the previous step was not planned).
            if _OPTIM in unit.resident:
                with torch.cuda.stream(self._copier.stream):
                    _release(unit.device[_OPTIM])
                unit.resident.discard(_OPTIM)
                unit.ready = None
            self._prefetch_following(unit)
        if kind == _BACKWARD and _GRAD not in unit.resident:
            self._allocate_grad(unit)
        if kind == _BACKWARD:
            self._note_backward(unit, index)
            self._reap_copies()
            self._sample_memory()
        self._previous = unit

    def _allocate_grad(self, unit: _Unit) -> None:
        grad = unit.device[_GRAD]
        _allocate(grad)
        grad.zero_()
        unit.resident.add(_GRAD)

    def _settle(self, unit: _Unit, index: int) -> None:
        """Synchronize ``unit``'s gradients after its last backward of the step, then step or defer."""
        if unit.index in self._pending:
            self._step_module(unit, *self._pending.pop(unit.index), index)
        if _GRAD not in unit.resident:
            self._allocate_grad(unit)
        self._reduce(unit)
        hyperparameters = self._hyperparameters(unit)
        if self._planned and self._mode[unit.index] == DEFERRED:
            self._pending[unit.index] = (hyperparameters, self._step)
        else:
            self._step_module(unit, hyperparameters, self._step, index)
        unit.stepped = True

    def _step_module(self, unit: _Unit, hyperparameters: _Hyperparameters, step: int, index: int) -> None:
        """Apply ``unit``'s Adam step, release its gradients and write offloaded state back."""
        if _OPTIM not in unit.resident:
            self._prefetch(unit)
        if unit.ready is not None:
            before = self._timing_event()
            torch.cuda.current_stream().wait_event(unit.ready)
            self._stalls.append((index, before, self._timing_event()))
            unit.ready = None
        self._apply(unit, hyperparameters, step)
        _release(unit.device[_GRAD])
        unit.resident.discard(_GRAD)
        if self._mode[unit.index] == RESIDENT:
            unit.dirty.add(_OPTIM)
            return
        self._write_back(unit)
        self._prefetch_following(unit)

    def _prefetch_following(self, unit: _Unit) -> None:
        """Prefetch the offloaded modules up to ``lookahead`` after ``unit`` in its segment."""
        slot = self._slot.get(unit.index)
        if slot is not None and self._active and self._planned:
            segment, position = slot
            for unit_index in self._segments[segment][position + 1 : position + 1 + self._lookahead]:
                if _OPTIM not in self._units[unit_index].resident:
                    self._prefetch(self._units[unit_index])

    def _prefetch(self, unit: _Unit) -> None:
        optim = unit.device[_OPTIM]
        with torch.cuda.stream(self._copier.stream):
            _allocate(optim)
        unit.ready = self._copier.copy(optim, unit.host[_OPTIM])
        unit.resident.add(_OPTIM)
        self.step_traffic["h2d"] += unit.nbytes(_OPTIM)

    def _write_back(self, unit: _Unit) -> None:
        # The copies first wait for the compute stream, so they read the updated state, and later
        # allocations of the copy stream reuse the released buffer only after the copies.
        self._copier.wait_stream(torch.cuda.current_stream())
        optim = unit.device[_OPTIM]
        self._copier.copy(unit.host[_OPTIM], optim)
        with torch.cuda.stream(self._copier.stream):
            _release(optim)
        unit.resident.discard(_OPTIM)
        unit.dirty.discard(_OPTIM)
        self.step_traffic["d2h"] += unit.nbytes(_OPTIM)

    def end_step(self) -> float:
        """Finish the step: synchronize the remaining gradients and step (or defer) every module.

        Returns:
            The global gradient norm over all offloaded models.

        Raises:
            FloatingPointError: if any gradient was non-finite (the step is fatal, as in native training).
        """
        if not self._active:
            raise RuntimeError("module state offload: end_step without begin_step")
        if self._replaying and self._cursor != len(self._sequence):
            self._diverge(self._cursor)
        index = self._cursor
        self._end = self._timing_event()
        self._settle_previous(index)
        remaining = [unit for unit in self._units if not unit.stepped]
        loading = [
            unit
            for unit in remaining
            if _OPTIM not in unit.resident and not (self._planned and self._mode[unit.index] == DEFERRED)
        ]
        position = {unit.index: rank for rank, unit in enumerate(loading)}
        issued = 0
        for unit in remaining:
            if unit.index in position:
                while issued < min(len(loading), position[unit.index] + self._lookahead + 1):
                    if _OPTIM not in loading[issued].resident:
                        self._prefetch(loading[issued])
                    issued += 1
            self._settle(unit, index)
        self._record_copy_tail(index)
        if dist.is_initialized():
            dist.all_reduce(self._nonfinite, op=dist.ReduceOp.MAX)
        for model, norm_sq in zip(self._models, self._norm_sq, strict=True):
            if model.tp_group is not None and model.tp_group.size() > 1:
                dist.all_reduce(norm_sq, group=model.tp_group)
        stats = torch.stack([self._nonfinite.to(torch.float32), *self._norm_sq]).tolist()
        if not self._replaying:
            self._record_sequence()
        self._active = False
        if stats[0]:
            raise FloatingPointError("module state offload: non-finite gradients")
        self._copier.done().synchronize()
        self._reap_copies()
        self._sample_memory()
        if self._headroom is not None and self._min_free is not None and dist.is_initialized():
            tightest = torch.tensor([self._min_free], dtype=torch.int64, device="cuda")
            dist.all_reduce(tightest, op=dist.ReduceOp.MIN)
            self._min_free = int(tightest.item())
        self._adapt()
        return math.sqrt(sum(stats[1:]))

    def _timeline(self) -> tuple[list[float], float, float]:
        """Seconds into the step of every event and of ``end_step`` without copy waits, and the waits."""
        stalls = sorted((index, max(0.0, before.elapsed_time(after) / 1e3)) for index, before, after in self._stalls)
        times, lost, cursor = [], 0.0, 0
        for index, mark in enumerate([*self._marks, self._end]):
            while cursor < len(stalls) and stalls[cursor][0] < index:
                lost += stalls[cursor][1]
                cursor += 1
            times.append(self._start.elapsed_time(mark) / 1e3 - lost)
        return times[:-1], times[-1], sum(stall for _, stall in stalls)

    def _sample_memory(self) -> None:
        """Allocated bytes of the current interval, without Adam state or gradients."""
        if self._headroom is not None:
            free, _ = torch.cuda.mem_get_info()
            self._min_free = free if self._min_free is None else min(self._min_free, free)
        if not self._measuring:
            return
        extra = 0
        for unit in self._units:
            if _OPTIM in unit.resident:
                extra += unit.nbytes(_OPTIM)
            if _GRAD in unit.resident:
                extra += unit.nbytes(_GRAD)
        self._memory_samples.append(max(0, torch.cuda.memory_allocated() - extra))

    def _param_storage_ptrs(self) -> set[int]:
        return {unit.device[_PARAM].untyped_storage().data_ptr() for unit in self._units if _PARAM in unit.resident}

    def _copy_out(self, host: torch.Tensor, source: torch.Tensor) -> torch.cuda.Event:
        """Queue a device-to-host copy ordered after the compute stream and earlier copies."""
        self._copier.wait_stream(torch.cuda.current_stream())
        return self._copier.copy(host, source)

    def _reap_copies(self) -> None:
        """Drop GPU tensors whose device-to-host copy has finished, and release finished parameters."""
        alive = []
        for hold in self._holds:
            if not hold[1].query():
                alive.append(hold)
                continue
            if hold[0] == "tensor":
                hold[2][0] = None
            elif _PARAM in hold[2].resident:
                _release(hold[2].device[_PARAM])
                hold[2].resident.discard(_PARAM)
        self._holds = alive

    def _ensure_params(self, unit: _Unit) -> None:
        """Bring ``unit``'s parameters back before a forward or backward that still needs them."""
        pending = [hold for hold in self._holds if not _is_param_hold(hold, unit)]
        if _PARAM in unit.resident:
            self._holds = pending
            return
        self._holds = pending
        param = unit.device[_PARAM]
        _allocate(param)
        self._copier.wait_stream(torch.cuda.current_stream())
        event = self._copier.copy(param, unit.host[_PARAM])
        torch.cuda.current_stream().wait_event(event)
        unit.resident.add(_PARAM)
        self.step_traffic["h2d"] += unit.nbytes(_PARAM)

    def _offload_params(self, unit: _Unit) -> None:
        """Copy ``unit``'s parameters to pinned memory and free them once the copy finishes."""
        if _PARAM not in unit.resident:
            return
        if _PARAM not in unit.host:
            unit.host[_PARAM] = torch.empty(unit.numel, dtype=unit.param_dtype, pin_memory=True)
        self._copier.wait_stream(torch.cuda.current_stream())
        event = self._copier.copy(unit.host[_PARAM], unit.device[_PARAM])
        self._holds.append(("storage", event, unit, _PARAM))
        self.step_traffic["d2h"] += unit.nbytes(_PARAM)

    def _begin_capture(self, unit: _Unit) -> None:
        """Count saved activations, or start copying out this forward's quota."""
        measure = self._measuring
        quota = None
        if not measure:
            if not (self._planned and self._headroom is not None and self._activation_quota):
                return
            if self._forward_cursor >= len(self._activation_quota):
                return
            quota = self._activation_quota[self._forward_cursor]
            self._forward_cursor += 1
            if quota <= 0:
                return
        session = _ActivationSession(self, quota, self._param_storage_ptrs())
        context = torch.autograd.graph.saved_tensors_hooks(session.pack, session.unpack)
        context.__enter__()
        unit._capture = (context, session, self._cursor - 1, measure)

    def _end_capture(self, unit: _Unit) -> None:
        capture = getattr(unit, "_capture", None)
        if not capture:
            return
        context, session, forward_index, measure = capture
        unit._capture = None
        context.__exit__(None, None, None)
        if measure:
            # Push even an empty forward so later backwards pop the matching record.
            self._group_records.append([unit.index, forward_index, session.nbytes, None])
            self._open_groups[unit.index].append(len(self._group_records) - 1)

    def _note_backward(self, unit: _Unit, index: int) -> None:
        if not self._measuring:
            return
        stack = self._open_groups.get(unit.index)
        if stack:
            self._group_records[stack.pop()][3] = index

    def _adapt(self) -> None:
        """Plan after the measurement step; shrink the margin and replan after a step waited on a copy."""
        times, end, stall = self._timeline()
        measured = torch.tensor([*times, end, -stall], dtype=torch.float64, device="cuda")
        if dist.is_initialized():
            lengths = torch.tensor([measured.numel(), -measured.numel()], device="cuda")
            dist.all_reduce(lengths, op=dist.ReduceOp.MAX)
            if lengths.sum().item() == 0:
                # Plan for the earliest event times and the longest wait over the ranks.
                dist.all_reduce(measured, op=dist.ReduceOp.MIN)
        *times, end, stall = measured.tolist()
        self.step_stall = -stall
        self.step_timeline = end
        if self._headroom is not None:
            self._adapt_headroom(times, end)
            return
        if self._diverged or not self._replayed:
            self._flush_pending()
            self._profiling = True
            return
        if self._profiling:
            # Planned steps run Adam steps and copies inside the step, so only this step measures
            # the computation alone.
            self._profiling = False
            self._measurement = (
                self._candidates(times, end),
                [times[index] for index, entry in enumerate(self._sequence) if entry == _SYNC],
                end,
            )
        elif self.step_stall > max(1e-3, 2e-3 * end):
            self._margin *= _MARGIN_DECAY
        else:
            return
        candidates, barriers, end = self._measurement
        plan = plan_offload(
            candidates,
            barriers=barriers,
            end=end,
            h2d_rate=self._rates[0] * self._margin,
            d2h_rate=self._rates[1] * self._margin,
            lookahead=self._lookahead,
        )
        self._flush_pending()
        self._activate(plan)
        offloaded = [unit for unit in self._units if self._mode[unit.index] != RESIDENT]
        logger.info(
            "budgeted module state offload plan at margin %.2f (step %.3f s, %d synchronizations, waited %.1f ms): "
            "%d of %d modules offload %.2f of %.2f GiB Adam state, %d inline and %d deferred",
            self._margin,
            end,
            self._sequence.count(_SYNC),
            self.step_stall * 1e3,
            len(offloaded),
            len(self._units),
            self._adam_bytes(offloaded) / 2**30,
            self._adam_bytes(self._units) / 2**30,
            sum(self._mode[unit.index] == INLINE for unit in offloaded),
            sum(self._mode[unit.index] == DEFERRED for unit in offloaded),
        )

    def _candidates(self, times: list[float], end: float) -> list[OffloadCandidate]:
        first_forward: dict[int, int] = {}
        for index, (unit_index, kind) in enumerate(self._sequence):
            if kind == _FORWARD:
                first_forward.setdefault(unit_index, index)
        candidates = []
        for unit in self._units:
            if unit.index not in first_forward:
                continue
            inline_at = None
            if unit.final_backward is not None:
                after = unit.final_backward + 1
                inline_at = times[after] if after < len(times) else end
            candidates.append(
                OffloadCandidate(unit.index, unit.nbytes(_OPTIM), times[first_forward[unit.index]], inline_at)
            )
        return candidates

    def _update_index(self, unit: _Unit) -> int:
        """Position in the recorded order at which ``unit``'s offloaded Adam step runs."""
        if self._mode[unit.index] == DEFERRED:
            return self._sequence.index((unit.index, _FORWARD))
        return unit.final_backward + 1

    def _activate(self, plan: dict[int, str]) -> None:
        """Switch to ``plan``: write back newly offloaded Adam state and load newly resident state."""
        compute = torch.cuda.current_stream()
        self._copier.wait_stream(compute)
        for unit in self._units:
            if plan.get(unit.index, RESIDENT) == RESIDENT or _OPTIM not in unit.resident:
                continue
            if _OPTIM in unit.dirty:
                self._write_back(unit)
            else:
                with torch.cuda.stream(self._copier.stream):
                    _release(unit.device[_OPTIM])
                unit.resident.discard(_OPTIM)
            unit.ready = None
        # Compute-stream allocations may reuse buffers released above only after their write-backs.
        compute.wait_event(self._copier.done())
        for unit in self._units:
            if plan.get(unit.index, RESIDENT) == RESIDENT and _OPTIM not in unit.resident:
                optim = unit.device[_OPTIM]
                _allocate(optim)
                optim.copy_(unit.host[_OPTIM], non_blocking=True)
                unit.resident.add(_OPTIM)
                self.step_traffic["h2d"] += unit.nbytes(_OPTIM)
        self._mode = {unit.index: plan.get(unit.index, RESIDENT) for unit in self._units}
        segment_at, segment = [], 0
        for entry in self._sequence:
            segment_at.append(segment)
            segment += entry == _SYNC
        self._segments = [[] for _ in range(segment + 1)]
        self._slot = {}
        offloaded = [unit for unit in self._units if self._mode[unit.index] != RESIDENT]
        self._update_at = {unit.index: self._update_index(unit) for unit in offloaded}
        for unit in sorted(offloaded, key=self._update_index):
            index = self._update_at[unit.index]
            segment = segment_at[index] if index < len(segment_at) else len(self._segments) - 1
            self._slot[unit.index] = (segment, len(self._segments[segment]))
            self._segments[segment].append(unit.index)

    def _adapt_headroom(self, times: list[float], end: float) -> None:
        """Fit one plan to the free-memory budget; keep it even when copies make the step wait."""
        if self._diverged or not self._replayed:
            self._flush_pending()
            self._profiling = True
            return
        if not self._profiling:
            return
        self._profiling = False
        count = len(self._sequence)
        if len(self._memory_samples) != count + 2 or len(times) != count:
            raise RuntimeError(
                "module state offload measured "
                f"{len(self._memory_samples)} memory samples and {len(times)} event times for {count} events"
            )
        samples = self._memory_samples
        records = [record for record in self._group_records if record[3] is not None and record[2]]
        if dist.is_initialized():
            counts = torch.tensor([len(records), -len(records)], dtype=torch.int64, device="cuda")
            dist.all_reduce(counts, op=dist.ReduceOp.MAX)
            if counts[0].item() != -counts[1].item():
                raise RuntimeError("module state offload measured a different activation count on each rank")
            reduced = torch.tensor(samples, dtype=torch.int64, device="cuda")
            dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
            samples = reduced.tolist()
            widths = torch.tensor([int(record[2]) for record in records], dtype=torch.int64, device="cuda")
            if widths.numel():
                dist.all_reduce(widths, op=dist.ReduceOp.MAX)
                for record, width in zip(records, widths.tolist(), strict=True):
                    record[2] = int(width)
        free, total = torch.cuda.mem_get_info()
        budget = min(free + torch.cuda.memory_reserved() - self._headroom, total)
        if dist.is_initialized():
            agreed = torch.tensor([budget], dtype=torch.int64, device="cuda")
            dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
            budget = int(agreed.item())
        units = []
        first_forward: dict[int, int] = {}
        first_backward: dict[int, int] = {}
        for index, (unit_index, kind) in enumerate(self._sequence):
            if kind == _FORWARD:
                first_forward.setdefault(unit_index, index)
            elif kind == _BACKWARD:
                first_backward.setdefault(unit_index, index)
        for unit in self._units:
            units.append(
                UnitProfile(
                    unit.index,
                    unit.nbytes(_OPTIM),
                    unit.nbytes(_GRAD),
                    unit.nbytes(_PARAM),
                    first_forward.get(unit.index),
                    first_backward.get(unit.index),
                    unit.final_backward,
                )
            )
        groups = tuple(
            GroupProfile(int(record[0]), int(record[1]), int(record[3]), int(record[2])) for record in records
        )
        profile = StepProfile(
            times=tuple(times),
            end=end,
            barriers=tuple(index for index, entry in enumerate(self._sequence) if entry == _SYNC),
            units=tuple(units),
            groups=groups,
            memory=tuple(int(sample) for sample in samples),
        )
        plan = plan_mixed_offload(
            profile,
            rates=self._copy_rates.scaled(self._margin),
            lookahead=self._lookahead,
            throttle=self._throttle,
            budget=budget,
        )
        self._flush_pending()
        self._activate(plan.adam)
        quotas = []
        group_at = {group.forward: position for position, group in enumerate(groups)}
        for index, (_, kind) in enumerate(self._sequence):
            if kind != _FORWARD:
                continue
            position = group_at.get(index)
            quotas.append(0 if position is None else plan.activations.get(position, 0))
        self._activation_quota = quotas
        self._param_ids = set(plan.params)
        self._offloaded_activation_bytes = sum(quotas)
        self._offloaded_param_bytes = sum(unit.nbytes(_PARAM) for unit in self._units if unit.index in self._param_ids)
        self._memory_budget = budget
        self._predicted_peak = 0 if plan.peak is None else plan.peak
        # One tensor can cross a forward's quota, so the arena covers every saved byte, not just the quota.
        capacity = sum(group.nbytes for group in groups)
        self._activation_arena = _ActivationArena(capacity + 4096) if capacity else None
        logger.info(
            "budgeted mixed offload: budget %.2f GiB, predicted peak %.2f GiB, copy wait %.1f ms, "
            "Adam %.2f GiB, activations %.2f GiB, parameters %.2f GiB across %d modules",
            budget / 2**30,
            self._predicted_peak / 2**30,
            plan.stall * 1e3,
            self._adam_bytes([unit for unit in self._units if self._mode[unit.index] != RESIDENT]) / 2**30,
            self._offloaded_activation_bytes / 2**30,
            self._offloaded_param_bytes / 2**30,
            len(self._param_ids),
        )
        if self._predicted_peak > budget:
            logger.warning(
                "budgeted mixed offload still predicts a peak %.2f GiB over the %.2f GiB budget",
                (self._predicted_peak - budget) / 2**30,
                budget / 2**30,
            )

    def _flush_pending(self) -> None:
        """Apply the deferred Adam steps now."""
        for unit_index in list(self._pending):
            self._step_module(self._units[unit_index], *self._pending.pop(unit_index), len(self._sequence))

    def close(self) -> None:
        """Apply the deferred Adam steps, write all Adam state back and release it and the hooks.

        The parameters stay on the GPU; the Adam state stays readable in the (unpinned) host buffers.
        """
        self._flush_pending()
        self._copier.done().synchronize()
        self._reap_copies()
        for unit in self._units:
            if _PARAM not in unit.resident:
                self._ensure_params(unit)
        self._copier.wait_stream(torch.cuda.current_stream())
        for unit in self._units:
            if _OPTIM in unit.dirty:
                self._copier.copy(unit.host[_OPTIM], unit.device[_OPTIM])
                unit.dirty.discard(_OPTIM)
        torch.cuda.synchronize()
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        for unit in self._units:
            for component in (_GRAD, _OPTIM):
                if component in unit.resident:
                    _release(unit.device[component])
                    unit.resident.discard(component)
            unit.ready = None
        for arena in self._arenas.values():
            arena.close()
