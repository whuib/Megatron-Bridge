# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Migrate module state between pinned host memory and the GPU around each module's computation.

A model is split into modules: every layer of a homogeneous ``nn.ModuleList`` (transformer layers,
DeepStack mergers) and every remaining leaf module that holds parameters (embeddings, norms, output
layer, patch embedding, ...). A module's state is its parameters, its FP32 main gradients and its
FP32 Adam state (main parameters, first and second moments). It lives in pinned host memory: the
parameters are copied to the GPU for the module's forward, parameters and gradients for its
backward, all of the state for its Adam step, and the GPU copy is written back and released as soon
as the next module starts.

Without overlap every copy runs on the compute stream, in line with the computation, and one module
is GPU resident at a time. With overlap, the module execution order recorded in the previous step
drives a copy stream: while a module computes, the previous module is written back and the next
one is prefetched. The current and next modules have transient state on the GPU, with a separate
byte allowance for priority uploads and a deferred Adam write-back. Selected weights or Adam
buffers can stay on the GPU across steps. With the last micro-batch marked
(:meth:`ModuleStateOffload.final_microbatch`), each module's Adam step also runs right after its last
backward of the step instead of after the whole backward, saving one more round trip of its state.

A module's parameters may only be used inside its own forward and the matching backward, and the
module execution order of the last micro-batch must not depend on the data.
"""

import logging
import math
import os
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


logger = logging.getLogger(__name__)

_FORWARD, _BACKWARD, _MARK = 0, 1, 2
_FINAL_MICROBATCH = (-1, _MARK)
_PARAM, _GRAD, _OPTIM = "param", "grad", "optim"
_COMPONENTS = (_PARAM, _GRAD, _OPTIM)
_FORWARD_NEEDS = frozenset({_PARAM})
_BACKWARD_NEEDS = frozenset({_PARAM, _GRAD})
_ADAM_NEEDS = frozenset({_OPTIM})
_STEP_NEEDS = frozenset(_COMPONENTS)
# Every parameter starts at a multiple of this many elements inside its module's flat buffers.
_ALIGN = 128


@dataclass
class OffloadedModel:
    """A model whose module state is migrated, with what its gradient sync and Adam step need.

    Attributes:
        name: Prefix of the model's module names.
        module: The unwrapped model.
        param_groups: ``(optimizer, group index)`` of every trainable parameter. The group (``lr``,
            ``betas``, ``eps``, ``weight_decay``, ``bias_correction``) is read at every step, so
            scheduler updates apply; the optimizer is a Transformer Engine ``FusedAdam`` without
            master weights, whose kernel and weight-decay mode the step uses.
        tp_group: Group of the replicated-parameter gradient reductions; None without TP.
        dp_group: Group of the data-parallel gradient reduction; None without DP.
        average_in_collective: Average over DP inside the collective instead of pre-scaling.
        sequence_parallel: The model config's ``sequence_parallel``.
        qk_layernorm: The model config's ``qk_layernorm``.
    """

    name: str
    module: nn.Module
    param_groups: dict[nn.Parameter, tuple[torch.optim.Optimizer, int]]
    tp_group: dist.ProcessGroup | None
    dp_group: dist.ProcessGroup | None
    average_in_collective: bool
    sequence_parallel: bool
    qk_layernorm: bool


def _is_layer_list(module: nn.Module) -> bool:
    if not isinstance(module, nn.ModuleList) or len(module) < 2 or len({type(child) for child in module}) != 1:
        return False
    return all(any(True for _ in child.parameters()) for child in module)


def discover_modules(root: nn.Module) -> list[tuple[str, nn.Module]]:
    """Split ``root`` into offloaded modules, in registration order.

    Every element of a homogeneous ``nn.ModuleList`` is a module; so is every other maximal submodule
    that holds parameters and no such list.

    Raises:
        ValueError: if a submodule containing such a list also owns parameters directly.
    """
    found: list[tuple[str, nn.Module]] = []

    def visit(name: str, module: nn.Module) -> None:
        if _is_layer_list(module):
            found.extend((f"{name}.{index}", child) for index, child in enumerate(module))
            return
        if not any(_is_layer_list(submodule) for submodule in module.modules()):
            if any(True for _ in module.parameters()):
                found.append((name, module))
            return
        direct = [param_name for param_name, _ in module.named_parameters(recurse=False)]
        if direct:
            raise ValueError(f"module state offload cannot place parameters {direct} of {name or 'the model'}")
        for child_name, child in module.named_children():
            visit(f"{name}.{child_name}" if name else child_name, child)

    visit("", root)
    return found


def _tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _allocate(tensor: torch.Tensor) -> None:
    tensor.untyped_storage().resize_(tensor.numel() * tensor.element_size())


def _release(tensor: torch.Tensor) -> None:
    tensor.untyped_storage().resize_(0)


def _device_local_cpus() -> set[int]:
    """CPUs of the current GPU's NUMA node that this thread may run on (empty if unknown)."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    path = f"/sys/bus/pci/devices/{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    try:
        with open(f"{path}/local_cpulist") as handle:
            spec = handle.read().strip()
    except OSError:
        return set()
    cpus = set()
    for part in spec.split(","):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus & os.sched_getaffinity(0)


class _PinnedArena:
    """One page-locked host allocation of exactly ``numel`` elements, carved into module buffers."""

    def __init__(self, numel: int, dtype: torch.dtype) -> None:
        self.tensor = torch.empty(numel, dtype=dtype)
        self.offset = 0
        self.registered = False
        if numel:
            nbytes = numel * self.tensor.element_size()
            # Registration faults the pages in from this thread, so pin it to the GPU's NUMA node:
            # remote pages roughly halve the copy bandwidth once every GPU of a node copies.
            local = _device_local_cpus() if hasattr(os, "sched_setaffinity") else set()
            saved = os.sched_getaffinity(0) if local else None
            if local:
                os.sched_setaffinity(0, local)
            try:
                torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(self.tensor.data_ptr(), nbytes, 0))
            finally:
                if saved is not None:
                    os.sched_setaffinity(0, saved)
            self.registered = True

    def take(self, numel: int) -> torch.Tensor:
        view = self.tensor[self.offset : self.offset + numel]
        self.offset += numel
        return view

    def close(self) -> None:
        if self.registered:
            torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(self.tensor.data_ptr()))
            self.registered = False


class _Unit:
    """One offloaded module: flat host and device buffers sharing one element layout, and residency."""

    def __init__(
        self,
        index: int,
        name: str,
        model_index: int,
        module: nn.Module,
        named_params: list[tuple[str, nn.Parameter]],
        model: OffloadedModel,
    ) -> None:
        self.index = index
        self.name = name
        self.model_index = model_index
        self.module = module
        dtypes = {param.dtype for _, param in named_params}
        if len(dtypes) != 1:
            raise ValueError(f"module state offload needs one parameter dtype per module; {name} has {dtypes}")
        self.param_dtype = dtypes.pop()
        # Parameters of one optimizer group are contiguous, so each group is one slice of the buffers.
        group_rank: dict[tuple[int, int], int] = {}
        ranks = []
        for _, param in named_params:
            optimizer, group = model.param_groups[param]
            ranks.append(group_rank.setdefault((id(optimizer), group), len(group_rank)))
        order = sorted(range(len(named_params)), key=ranks.__getitem__)
        self.names = [named_params[i][0] for i in order]
        self.params = [named_params[i][1] for i in order]
        self.offsets = []
        self.segments: list[tuple[int, int, torch.optim.Optimizer, int]] = []
        offset = 0
        for param in self.params:
            optimizer, group = model.param_groups[param]
            if not self.segments or self.segments[-1][2] is not optimizer or self.segments[-1][3] != group:
                self.segments.append((offset, offset, optimizer, group))
            self.offsets.append(offset)
            offset += -(-param.numel() // _ALIGN) * _ALIGN
            start, _, _, _ = self.segments[-1]
            self.segments[-1] = (start, offset, optimizer, group)
        self.numel = offset
        self.device: dict[str, torch.Tensor] = {}
        self.host: dict[str, torch.Tensor] = {}
        self.resident: set[str] = set()
        self.persistent: set[str] = set()
        self.dirty: set[str] = set()
        self.ready: torch.cuda.Event | None = None
        self.grad_is_zero = True
        self.stepped = False
        self.final_backward: int | None = None
        self.tp_reductions: list[tuple[dist.ReduceOp, list[torch.Tensor]]] = []
        self.norm_grads: list[torch.Tensor] | None = None

    def nbytes(self, component: str) -> int:
        """Host/device bytes of one state component."""
        return self.device[component].numel() * self.device[component].element_size()

    def optimizer_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Device FP32 main parameters, first moments and second moments."""
        return self.device[_OPTIM].split(self.numel)


class ModuleStateOffload:
    """Keep module state in pinned host memory and stage it on the GPU around each module's work.

    Call :meth:`begin_step` before and :meth:`end_step` after the forward and backward passes of a
    training step, and optionally :meth:`final_microbatch` right before the last micro-batch;
    :meth:`end_step` completes the gradient synchronization and Adam step of every module and
    replaces the models' gradient finalization and optimizer steps.

    Args:
        models: The offloaded models; their parameters are moved into the module buffers.
        overlap: Overlap the copies with computation (see the module docstring).
        resident_param_bytes: Maximum bytes of module weights to leave on the GPU across steps.
        resident_optim_bytes: Maximum bytes of FP32 Adam state to leave on the GPU across steps.
        prefetch_bytes: Maximum extra bytes to stage before the previous module writes back.
            A positive limit also gives backward and Adam separate readiness events.
        scoped_handoff: Wait for the compute stream at handoffs with local boundary transport.
        writeback_chunk_bytes: Send this many bytes of a small enough Adam write-back first and
            let the next transition's priority uploads precede the rest.
    """

    # State components backed by pinned host memory.
    _HOST_COMPONENTS: tuple[str, ...] = _COMPONENTS

    def __init__(
        self,
        models: list[OffloadedModel],
        *,
        overlap: bool,
        resident_param_bytes: int = 0,
        resident_optim_bytes: int = 0,
        prefetch_bytes: int = 0,
        scoped_handoff: bool = False,
        writeback_chunk_bytes: int = 0,
    ) -> None:
        if resident_param_bytes < 0 or (resident_param_bytes and not overlap):
            raise ValueError("resident_param_bytes must be nonnegative and requires overlap when positive")
        if resident_optim_bytes < 0 or (resident_optim_bytes and not overlap):
            raise ValueError("resident_optim_bytes must be nonnegative and requires overlap when positive")
        if prefetch_bytes < 0 or (prefetch_bytes and not overlap):
            raise ValueError("prefetch_bytes must be nonnegative and requires overlap when positive")
        if scoped_handoff and not overlap:
            raise ValueError("scoped_handoff requires overlap")
        if writeback_chunk_bytes and not 0 < writeback_chunk_bytes < prefetch_bytes:
            raise ValueError("writeback_chunk_bytes must be positive and smaller than the prefetch byte allowance")
        self._models = models
        self._overlap = overlap
        self._prefetch_bytes = prefetch_bytes
        self._writeback_chunk_bytes = writeback_chunk_bytes
        self._pending_writeback: tuple[_Unit, int] | None = None
        self._chunked_writebacks = 0
        self._scoped_handoff = scoped_handoff and os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1"
        if scoped_handoff and not self._scoped_handoff:
            logger.info("Module offload handoffs use device synchronization with CUDA_DEVICE_MAX_CONNECTIONS=1")
        self._copy_stream = torch.cuda.Stream() if overlap else None
        self._units: list[_Unit] = []
        seen: set[nn.Parameter] = set()
        for model_index, model in enumerate(models):
            for name, module in discover_modules(model.module):
                named = [(f"{name}.{param_name}", param) for param_name, param in module.named_parameters()]
                frozen = [param_name for param_name, param in named if not param.requires_grad]
                if frozen:
                    raise ValueError(f"module state offload does not support frozen parameters {frozen}")
                shared = [param_name for param_name, param in named if param in seen]
                if shared:
                    raise ValueError(f"module state offload does not support shared parameters {shared}")
                missing = [param_name for param_name, param in named if param not in model.param_groups]
                if missing:
                    raise ValueError(f"parameters {missing} have no optimizer param group")
                seen.update(param for _, param in named)
                self._units.append(_Unit(len(self._units), f"{model.name}.{name}", model_index, module, named, model))
            unplaced = [name for name, param in model.module.named_parameters() if param not in seen]
            if unplaced:
                raise ValueError(f"module state offload left parameters {unplaced} of {model.name} unplaced")
        param_dtypes = {unit.param_dtype for unit in self._units}
        if len(param_dtypes) != 1:
            raise ValueError(f"module state offload needs one parameter dtype, got {param_dtypes}")
        total = sum(unit.numel for unit in self._units)
        param_dtype = param_dtypes.pop()
        width = self._units[0].params[0].element_size()
        fp32_width = torch.finfo(torch.float32).bits // 8
        self.resident_param_bytes = self._reserve_resident_state(_PARAM, width, resident_param_bytes)
        self.resident_optim_bytes = self._reserve_resident_state(_OPTIM, 3 * fp32_width, resident_optim_bytes)
        sizes = {
            _PARAM: (total - self.resident_param_bytes // width, param_dtype),
            _GRAD: (total, torch.float32),
            _OPTIM: (3 * total - self.resident_optim_bytes // fp32_width, torch.float32),
        }
        self._arenas = {component: _PinnedArena(*sizes[component]) for component in self._HOST_COMPONENTS}
        for unit in self._units:
            self._build(unit)
            self._adopt(unit)
        self._hooks = []
        for unit in self._units:
            self._hooks.append(unit.module.register_forward_pre_hook(self._forward_pre_hook(unit)))
            self._hooks.append(unit.module.register_forward_hook(self._forward_hook(unit)))
            self._hooks.extend(param.register_post_accumulate_grad_hook(_accumulate) for param in unit.params)
        self._overflow = torch.zeros(1, dtype=torch.int, device="cuda")
        self._nonfinite = torch.zeros((), dtype=torch.int32, device="cuda")
        self._norm_sq = [torch.zeros((), dtype=torch.float32, device="cuda") for _ in models]
        self._sequence: list[tuple[int, int]] = []
        self._new_sequence: list[tuple[int, int]] = []
        self._replaying = False
        self._cursor = 0
        self._previous: _Unit | None = None
        self._active = False
        self._marked = False
        self._step = 0
        self._resident: dict[int, _Unit] = {}
        self._warned = False
        self.max_resident_modules = 0
        self.step_traffic = {"h2d": 0, "d2h": 0}
        self._log_setup()

    def _reserve_resident_state(self, component: str, width: int, budget: int) -> int:
        """Choose whole buffers within a per-rank allowance, largest first."""
        if budget == 0:
            return 0
        # Prefer later registered modules for equal-size Adam buffers, so the output projection's
        # optimizer state can be ready at the start of backward.
        units = reversed(self._units) if component == _OPTIM else self._units
        used = 0
        for unit in sorted(units, key=lambda unit: unit.numel, reverse=True):
            size = unit.numel * width
            if size <= budget - used:
                unit.persistent.add(component)
                used += size
        return used

    def _log_setup(self) -> None:
        largest = max(self._units, key=lambda unit: unit.numel)
        logger.info(
            "module state offload (%s): %d modules, %.2f GiB pinned host state, largest module %s %.1f MiB, "
            "%.1f MiB resident parameters, %.1f MiB resident Adam state",
            "overlapped" if self._overlap else "not overlapped",
            len(self._units),
            sum(arena.tensor.numel() * arena.tensor.element_size() for arena in self._arenas.values()) / 2**30,
            largest.name,
            sum(largest.nbytes(component) for component in _COMPONENTS) / 2**20,
            self.resident_param_bytes / 2**20,
            self.resident_optim_bytes / 2**20,
        )

    def step_summary(self) -> dict[str, float]:
        """Metrics of the last training step, for logging."""
        summary = {"h2d_gib": self.step_traffic["h2d"] / 2**30, "d2h_gib": self.step_traffic["d2h"] / 2**30}
        if self._writeback_chunk_bytes:
            summary["chunked_writebacks"] = self._chunked_writebacks
        return summary

    @classmethod
    def from_megatron(cls, models: list[tuple[str, list, Any, Any]], **options: Any) -> "ModuleStateOffload":
        """Adopt DDP-wrapped Megatron models and their distributed optimizers.

        Args:
            models: ``(name, model chunks, optimizer, pg_collection)`` per model, one chunk each.
            **options: Keyword arguments of the engine's constructor (e.g. ``overlap``).

        Returns:
            The offload engine. The DDP parameter and gradient buffers and the optimizers' FP32 main
            parameters are released; the optimizers keep only their param groups for the schedulers.
        """
        from megatron.core.utils import get_model_config, unwrap_model

        described = []
        for name, chunks, optimizer, pg_collection in models:
            if len(chunks) != 1:
                raise ValueError(f"module state offload needs one {name} model chunk, got {len(chunks)}")
            chunk = chunks[0]
            module = unwrap_model(chunk)
            config = get_model_config(chunk)
            # Expert gradients then reduce over the dense groups (same ranks as the expert ones).
            if any(not getattr(param, "allreduce", True) for param in module.parameters()) and (
                pg_collection.ep.size() != 1 or pg_collection.expt_tp.size() != pg_collection.tp.size()
            ):
                raise ValueError(f"module state offload needs {name} expert parallelism 1 and expert TP equal to TP")
            described.append(
                OffloadedModel(
                    name=name,
                    module=module,
                    param_groups=_megatron_param_groups(module, optimizer, pg_collection.dp_cp),
                    tp_group=pg_collection.tp,
                    dp_group=pg_collection.dp_cp,
                    average_in_collective=chunk.ddp_config.average_in_collective,
                    sequence_parallel=bool(config.sequence_parallel),
                    qk_layernorm=bool(config.qk_layernorm),
                )
            )
        engine = cls(described, **options)
        for _, chunks, optimizer, _ in models:
            _release_megatron_state(chunks[0], optimizer)
        torch.cuda.empty_cache()
        return engine

    def _build(self, unit: _Unit) -> None:
        """Move ``unit``'s parameters into GPU flat buffers and precompute its gradient reductions."""
        model = self._models[unit.model_index]
        param = torch.zeros(unit.numel, dtype=unit.param_dtype, device="cuda")
        grad = torch.zeros(unit.numel, dtype=torch.float32, device="cuda")
        optim = torch.zeros(3 * unit.numel, dtype=torch.float32, device="cuda")
        for tensor, offset in zip(unit.params, unit.offsets, strict=True):
            view = param[offset : offset + tensor.numel()].view_as(tensor)
            view.copy_(tensor.detach())
            tensor.data = view
            tensor.main_grad = grad[offset : offset + tensor.numel()].view_as(tensor)
        optim[: unit.numel].copy_(param)
        unit.device = {_PARAM: param, _GRAD: grad, _OPTIM: optim}
        tp_size = model.tp_group.size() if model.tp_group is not None else 1
        if tp_size > 1:
            summed, averaged = [], []
            for name, tensor in zip(unit.names, unit.params, strict=True):
                if getattr(tensor, "average_gradients_across_tp_domain", False):
                    averaged.append(tensor.main_grad)
                elif (
                    (model.sequence_parallel and getattr(tensor, "sequence_parallel", False))
                    or (model.qk_layernorm and ("q_layernorm" in name or "k_layernorm" in name))
                    or getattr(tensor, "sum_gradients_across_tp_domain", False)
                ):
                    summed.append(tensor.main_grad)
            unit.tp_reductions = [
                (op, grads) for op, grads in ((dist.ReduceOp.SUM, summed), (dist.ReduceOp.AVG, averaged)) if grads
            ]
        # Replicated parameters count once in the gradient norm (on TP rank 0).
        if tp_size > 1 and model.tp_group.rank() != 0:
            unit.norm_grads = [
                tensor.main_grad for tensor in unit.params if getattr(tensor, "tensor_model_parallel", False)
            ]

    def _adopt(self, unit: _Unit) -> None:
        """Copy ``unit``'s state to pinned host memory and release its GPU buffers."""
        for component, tensor in unit.device.items():
            if component in unit.persistent:
                unit.resident.add(component)
                continue
            unit.host[component] = self._arenas[component].take(tensor.numel())
            unit.host[component].copy_(tensor)
            _release(tensor)

    def _forward_pre_hook(self, unit: _Unit):
        def hook(module, args):
            self._event(unit, _FORWARD)

        return hook

    def _forward_hook(self, unit: _Unit):
        def hook(module, args, output):
            outputs = [tensor for tensor in _tensors(output) if tensor.requires_grad]
            if not outputs:
                return
            fired = [False]

            def on_gradient(grad):
                # The module's backward starts once the gradient of any of its outputs is ready.
                if not fired[0]:
                    fired[0] = True
                    self._event(unit, _BACKWARD)

            for tensor in outputs:
                tensor.register_hook(on_gradient)

        return hook

    def _needs(self, unit: _Unit, kind: int, index: int) -> frozenset[str]:
        if kind == _FORWARD:
            return _FORWARD_NEEDS
        if not self._prefetch_bytes and self._replaying and index == unit.final_backward:
            return _STEP_NEEDS
        return _BACKWARD_NEEDS

    def _advance(self, entry: tuple[int, int]) -> int:
        index = self._cursor
        self._cursor += 1
        if self._replaying and (index >= len(self._sequence) or self._sequence[index] != entry):
            self._diverge(index)
        if not self._replaying:
            self._new_sequence.append(entry)
        return index

    def _upcoming(self, index: int) -> tuple[_Unit | None, frozenset[str] | None]:
        """The module the replayed order runs after event ``index``, and what it needs."""
        if not (self._overlap and self._replaying):
            return None, None
        for position in range(index + 1, len(self._sequence)):
            unit_index, kind = self._sequence[position]
            if kind != _MARK:
                unit = self._units[unit_index]
                return unit, self._needs(unit, kind, position)
        return None, None

    def final_microbatch(self) -> None:
        """Mark that the step's last micro-batch starts.

        Backwards recorded after the mark may be a module's last one of the step, which then runs
        the module's Adam step inline. A step with another micro-batch count than the recorded one
        diverges from the recorded order at the mark, before any module could step early.
        """
        if not self._active:
            raise RuntimeError("module state offload: final_microbatch outside a training step")
        if self._marked:
            raise RuntimeError("module state offload: final_microbatch called twice in one step")
        self._marked = True
        self._advance(_FINAL_MICROBATCH)

    def _event(self, unit: _Unit, kind: int) -> None:
        if not self._active:
            raise RuntimeError(f"module state offload: {unit.name} ran outside a training step")
        index = self._advance((unit.index, kind))
        if unit.stepped:
            raise RuntimeError(f"module state offload: {unit.name} ran after its optimizer step")
        previous = self._previous
        if previous is not None and previous is not unit and self._replaying and previous.final_backward == index - 1:
            self._finish(previous)
        following, following_needs = self._upcoming(index)
        self._transition(unit, self._needs(unit, kind, index), following, following_needs)
        if kind == _BACKWARD:
            if self._prefetch_bytes and self._replaying and index == unit.final_backward:
                self._load(unit, _ADAM_NEEDS)
            unit.dirty.add(_GRAD)
            unit.grad_is_zero = False
        self._previous = unit

    def _record_sequence(self) -> None:
        """Replay the order recorded this step from the next step on."""
        self._sequence = self._new_sequence
        self._new_sequence = []
        for unit in self._units:
            unit.final_backward = None
        mark = self._sequence.index(_FINAL_MICROBATCH) if _FINAL_MICROBATCH in self._sequence else None
        if mark is not None:
            for index in range(mark + 1, len(self._sequence)):
                unit_index, kind = self._sequence[index]
                if kind == _BACKWARD:
                    self._units[unit_index].final_backward = index
        self._replaying = True

    def _diverge(self, index: int) -> None:
        if not self._warned:
            logger.warning(
                "module state offload: module execution order changed at event %d; loading on demand "
                "for the rest of this step and recording the new order",
                index,
            )
            self._warned = True
        self._replaying = False
        self._new_sequence = self._sequence[:index]

    def _transition(
        self,
        unit: _Unit,
        needs: frozenset[str],
        following: _Unit | None,
        following_needs: frozenset[str] | None,
    ) -> None:
        """Make ``unit`` ready in the compute stream and stage ``following``."""
        compute = torch.cuda.current_stream()
        if self._prefetch_bytes:
            # Uploads for the current and next modules do not depend on the previous module's
            # compute. Reserve their extra storage before queuing that module's write-back.
            budget = self._prefetch_bytes
            if self._pending_writeback is not None:
                budget -= self._pending_writeback[0].nbytes(_OPTIM)
            current_bytes = sum(unit.nbytes(component) for component in needs - unit.resident)
            if current_bytes <= budget:
                self._load(unit, needs)
                if unit.ready is not None:
                    compute.wait_event(unit.ready)
                    unit.ready = None
                if following is not None and following_needs is not None:
                    next_bytes = sum(following.nbytes(component) for component in following_needs - following.resident)
                    if current_bytes + next_bytes <= budget:
                        self._load(following, following_needs)
            self._flush_writeback()
        if self._overlap:
            # Copy work issued from here on runs after everything already queued on the compute
            # stream, so released buffers can be reused by copy-stream allocations.
            self._copy_stream.wait_stream(compute)
        keep = (unit, following)
        for other in list(self._resident.values()):
            if other not in keep:
                self._evict(other)
        self._load(unit, needs)
        if unit.ready is not None:
            compute.wait_event(unit.ready)
            unit.ready = None
        if following is not None and following_needs is not None:
            self._load(following, following_needs)
        self.max_resident_modules = max(self.max_resident_modules, len(self._resident))

    def _load(self, unit: _Unit, needs: frozenset[str]) -> None:
        if self._pending_writeback is not None and self._pending_writeback[0] is unit and _OPTIM in needs:
            self._flush_writeback()
        missing = [component for component in _COMPONENTS if component in needs and component not in unit.resident]
        if not missing:
            return
        with torch.cuda.stream(self._copy_stream) if self._overlap else nullcontext():
            for component in missing:
                tensor = unit.device[component]
                _allocate(tensor)
                if component == _GRAD and unit.grad_is_zero:
                    tensor.zero_()
                else:
                    tensor.copy_(unit.host[component], non_blocking=True)
                    self.step_traffic["h2d"] += unit.nbytes(component)
                unit.resident.add(component)
            if self._overlap:
                unit.ready = torch.cuda.Event()
                unit.ready.record()
        self._resident[unit.index] = unit

    def _evict(self, unit: _Unit) -> None:
        with torch.cuda.stream(self._copy_stream) if self._overlap else nullcontext():
            for component in _COMPONENTS:
                if component not in unit.resident or component in unit.persistent:
                    continue
                tensor = unit.device[component]
                release = True
                if component in unit.dirty:
                    nbytes = unit.nbytes(component)
                    if (
                        component == _OPTIM
                        and 0 < self._writeback_chunk_bytes < nbytes <= self._prefetch_bytes
                        and self._pending_writeback is None
                    ):
                        first = self._writeback_chunk_bytes // tensor.element_size()
                        unit.host[component][:first].copy_(tensor[:first], non_blocking=True)
                        self.step_traffic["d2h"] += first * tensor.element_size()
                        self._pending_writeback = (unit, first)
                        self._chunked_writebacks += 1
                        release = False
                    else:
                        unit.host[component].copy_(tensor, non_blocking=True)
                        self.step_traffic["d2h"] += nbytes
                if release:
                    _release(tensor)
                unit.resident.remove(component)
        unit.dirty.clear()
        unit.ready = None
        del self._resident[unit.index]

    def _flush_writeback(self) -> None:
        """Finish the deferred Adam copy before reusing its host or device storage."""
        if self._pending_writeback is None:
            return
        unit, first = self._pending_writeback
        tensor = unit.device[_OPTIM]
        with torch.cuda.stream(self._copy_stream):
            unit.host[_OPTIM][first:].copy_(tensor[first:], non_blocking=True)
            self.step_traffic["d2h"] += (tensor.numel() - first) * tensor.element_size()
            _release(tensor)
        self._pending_writeback = None

    def _finish(self, unit: _Unit) -> None:
        """Synchronize ``unit``'s gradients and apply its Adam step (all of its state is resident)."""
        self._reduce(unit)
        if unit.ready is not None:
            torch.cuda.current_stream().wait_event(unit.ready)
            unit.ready = None
        self._apply(unit, self._hyperparameters(unit), self._step)
        unit.dirty.update((_PARAM, _OPTIM))
        unit.dirty.discard(_GRAD)
        unit.grad_is_zero = True
        unit.stepped = True

    def _reduce(self, unit: _Unit) -> None:
        """Synchronize ``unit``'s gradients and add them to the finite flag and gradient norms."""
        model = self._models[unit.model_index]
        grad = unit.device[_GRAD]
        if model.dp_group is not None and model.dp_group.size() > 1:
            if model.average_in_collective:
                dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=model.dp_group)
            else:
                grad.mul_(1.0 / model.dp_group.size())
                dist.all_reduce(grad, group=model.dp_group)
        for op, grads in unit.tp_reductions:
            coalesced = _flatten_dense_tensors(grads)
            dist.all_reduce(coalesced, op=op, group=model.tp_group)
            for buffer, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads), strict=True):
                buffer.copy_(synced)
        self._nonfinite.add_(torch.isfinite(grad).logical_not().any().to(torch.int32))
        if unit.norm_grads is None:
            self._norm_sq[unit.model_index].add_(torch.linalg.vector_norm(grad).square())
        elif unit.norm_grads:
            self._norm_sq[unit.model_index].add_(torch.stack(torch._foreach_norm(unit.norm_grads)).square().sum())

    def _hyperparameters(self, unit: _Unit) -> list[tuple[float, float, float, float, int, float]]:
        """``(lr, beta1, beta2, eps, bias_correction, weight_decay)`` of each of ``unit``'s segments now."""
        values = []
        for _, _, optimizer, group_index in unit.segments:
            group = optimizer.param_groups[group_index]
            beta1, beta2 = group["betas"]
            values.append(
                (group["lr"], beta1, beta2, group["eps"], 1 if group["bias_correction"] else 0, group["weight_decay"])
            )
        return values

    def _apply(
        self, unit: _Unit, hyperparameters: list[tuple[float, float, float, float, int, float]], step: int
    ) -> None:
        """Apply Adam step ``step`` to ``unit`` (gradients, parameters and Adam state on the GPU)."""
        from transformer_engine.pytorch.optimizers import multi_tensor_applier

        grad = unit.device[_GRAD]
        main, exp_avg, exp_avg_sq = unit.optimizer_state()
        for (start, end, optimizer, _), (lr, beta1, beta2, eps, bias_correction, weight_decay) in zip(
            unit.segments, hyperparameters, strict=True
        ):
            multi_tensor_applier(
                optimizer.multi_tensor_adam,
                self._overflow,
                [[grad[start:end]], [main[start:end]], [exp_avg[start:end]], [exp_avg_sq[start:end]]],
                lr,
                beta1,
                beta2,
                eps,
                step,
                optimizer.adam_w_mode,
                bias_correction,
                weight_decay,
            )
        unit.device[_PARAM].copy_(main)

    def begin_step(self) -> None:
        """Start a training step: gradients restart from zero."""
        if self._active:
            raise RuntimeError("module state offload: begin_step called twice")
        self._active = True
        self._marked = False
        self._cursor = 0
        self._previous = None
        self._step += 1
        self.step_traffic = {"h2d": 0, "d2h": 0}
        self._chunked_writebacks = 0
        for unit in self._units:
            unit.stepped = False
            for param in unit.params:
                param.grad_added_to_main_grad = False
        self._nonfinite.zero_()
        for norm_sq in self._norm_sq:
            norm_sq.zero_()
        if not self._replaying:
            self._new_sequence = []

    def synchronize(self) -> None:
        """Synchronize the device (every stream, including the copy stream)."""
        self._flush_writeback()
        torch.cuda.synchronize()

    def synchronize_handoff(self) -> None:
        """Wait for a phase's results while allowing unrelated state copies to continue."""
        if self._scoped_handoff:
            torch.cuda.current_stream().synchronize()
        else:
            self.synchronize()

    def end_step(self) -> float:
        """Finish the step: synchronize the remaining gradients, step every module, write state back.

        Returns:
            The global gradient norm over all offloaded models.

        Raises:
            FloatingPointError: if any gradient was non-finite (the step is fatal, as in native training).
        """
        if not self._active:
            raise RuntimeError("module state offload: end_step without begin_step")
        if self._replaying and self._cursor != len(self._sequence):
            self._diverge(self._cursor)
        previous = self._previous
        if previous is not None and self._replaying and previous.final_backward == self._cursor - 1:
            self._finish(previous)
        pending = [unit for unit in self._units if not unit.stepped]
        for position, unit in enumerate(pending):
            following = pending[position + 1] if self._overlap and position + 1 < len(pending) else None
            self._transition(unit, _STEP_NEEDS, following, _STEP_NEEDS)
            self._finish(unit)
        compute = torch.cuda.current_stream()
        if self._overlap:
            self._copy_stream.wait_stream(compute)
        for unit in list(self._resident.values()):
            self._evict(unit)
        self._flush_writeback()
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
        first, first_needs = self._upcoming(-1)
        if first is not None:
            self._copy_stream.wait_stream(compute)
            self._load(first, first_needs)
        return math.sqrt(sum(stats[1:]))

    def close(self) -> None:
        """Remove the hooks, release the GPU buffers and unregister the pinned host memory."""
        self._flush_writeback()
        torch.cuda.synchronize()
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        for unit in self._units:
            for component in unit.resident:
                _release(unit.device[component])
            unit.resident.clear()
        self._resident.clear()
        for arena in self._arenas.values():
            arena.close()


def _accumulate(param: nn.Parameter) -> None:
    # Runs before the DDP hook, which then sees no gradient; TE gradient-accumulation fusion already
    # wrote into main_grad.
    if param.grad is not None and (
        not getattr(param, "grad_added_to_main_grad", False) or getattr(param, "zero_out_wgrad", False)
    ):
        param.main_grad.add_(param.grad)
    param.grad = None


def _megatron_param_groups(
    module: nn.Module, optimizer: Any, dp_group: dist.ProcessGroup
) -> dict[nn.Parameter, tuple[torch.optim.Optimizer, int]]:
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    optimizers = list(getattr(optimizer, "chained_optimizers", [optimizer]))
    for sub_optimizer in optimizers:
        if not isinstance(sub_optimizer, DistributedOptimizer):
            raise ValueError(f"module state offload needs the distributed optimizer, got {type(sub_optimizer)}")
        inner = sub_optimizer.optimizer
        if (
            not type(inner).__module__.startswith("transformer_engine")
            or type(inner).__name__ != "FusedAdam"
            or inner.master_weights
            or inner.capturable
            or inner.use_decoupled_grad
        ):
            raise ValueError("module state offload needs Transformer Engine FusedAdam without master weights")
    params = list(module.parameters())
    position = {param: index for index, param in enumerate(params)}
    # Each DP rank's optimizer maps only the parameters it holds a shard of; their union covers all.
    local = {}
    for optimizer_index, sub_optimizer in enumerate(optimizers):
        for param, (group_index, _) in sub_optimizer.model_param_group_index_map.items():
            if param in position:
                local[position[param]] = (optimizer_index, group_index)
    if dp_group.size() > 1:
        gathered = [None] * dp_group.size()
        dist.all_gather_object(gathered, local, group=dp_group)
        for item in gathered:
            local.update(item)
    return {
        params[index]: (optimizers[optimizer_index].optimizer, group_index)
        for index, (optimizer_index, group_index) in local.items()
    }


def _release_megatron_state(chunk: Any, optimizer: Any) -> None:
    for buffer in chunk.buffers + chunk.expert_parallel_buffers:
        for data in (buffer.param_data, buffer.grad_data):
            if data is not None:
                _release(data)
    for sub_optimizer in getattr(optimizer, "chained_optimizers", [optimizer]):
        for group in sub_optimizer.shard_fp32_from_float16_groups:
            for main in group:
                _release(main)
