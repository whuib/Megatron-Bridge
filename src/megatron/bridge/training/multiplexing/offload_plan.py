# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan what a training step offloads to host memory, from one measured step.

Three kinds of state can leave the GPU:

- a module's Adam state (FP32 main parameters and moments), for the whole step except around its Adam
  step, which runs inline (right after the module's last backward) or deferred (right before its first
  forward of the next step);
- the activations a module forward saves for backward, from the end of that forward until shortly before
  the matching backward, possibly only some of them;
- a module's parameters, between each of its forwards and the matching backward.

A step is a sequence of events: module forwards and backwards, device synchronizations (which drain every
copy) and its end. :class:`StepProfile` holds their compute times without copy waits, the memory each
interval between events used, and what each module and module forward could offload. The planner
simulates the two copy directions, which share the host link (:func:`simulate_copies`), predicts the
memory of each interval, and greedily picks what to offload: either as much Adam state as copies hide
without delaying the computation, or the least that keeps the predicted peak under a byte budget,
preferring copies that delay nothing.

Memory indices: index ``k`` of an ``N``-event step is the interval from the hook of event ``k - 1``
(including its own allocations) to the hook of event ``k``; index ``N`` ends at the step's end and
``N + 1`` is the processing at the end. Something allocated by hook ``i`` counts from index ``i + 1``;
something freed by hook ``j`` counts until index ``j + 1``.
"""

import bisect
import math
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


RESIDENT, INLINE, DEFERRED = "resident", "inline", "deferred"
H2D, D2H = 0, 1
# Where in a hook a copy is queued: before its waits (the previous module's forward just ended), after
# them (Adam steps, write-backs, prefetches, reloads), or after its device synchronization.
START, SETTLE, RESUME = 0, 1, 2
# Copies a hook queues, in this order within one (event, phase).
_REFRESH, _WRITE_BACK, _PREFETCH, _RELOAD = range(4)


@dataclass(frozen=True)
class CopyRates:
    """Pinned host-device copy bandwidth in bytes per second.

    Attributes:
        h2d: Host to device with no device-to-host copy running.
        d2h: Device to host with no host-to-device copy running.
        h2d_shared: Host to device while a device-to-host copy runs.
        d2h_shared: Device to host while a host-to-device copy runs.
    """

    h2d: float
    d2h: float
    h2d_shared: float
    d2h_shared: float

    def scaled(self, factor: float) -> "CopyRates":
        """These rates times ``factor``."""
        return CopyRates(self.h2d * factor, self.d2h * factor, self.h2d_shared * factor, self.d2h_shared * factor)


@dataclass(frozen=True)
class CopyJob:
    """One copy of a simulated step.

    Attributes:
        lane: ``H2D`` or ``D2H``; the copies of one direction run in the order they are queued.
        nbytes: Bytes copied.
        issue: ``(event, phase)`` of the hook queuing the copy.
        deadline: Event whose hook waits for the copy; None if only the next synchronization does.
        after: Index of a copy (of the other direction) this one starts after.
    """

    lane: int
    nbytes: int
    issue: tuple[int, int]
    deadline: int | None = None
    after: int | None = None


@dataclass(frozen=True)
class Simulation:
    """Outcome of :func:`simulate_copies`.

    Attributes:
        stall: Seconds the computation waited on copies.
        leads: Per copy, its lead over the hook waiting for it, or (if none) over the synchronization
            after it; negative: a wait.
        finish: Simulated finish time of every copy.
    """

    stall: float
    leads: tuple[float, ...]
    finish: tuple[float, ...]

    @property
    def slack(self) -> float:
        """Smallest lead; infinite without copies."""
        return min(self.leads, default=math.inf)


def simulate_copies(
    jobs: list[CopyJob], *, times: list[float], end: float, barriers: list[int], rates: CopyRates
) -> Simulation:
    """Run ``jobs`` against a step with event times ``times`` and its end at ``end``.

    A copy starts once queued, its direction's earlier copies finished and the copy it follows finished.
    While both directions copy they run at the shared rates. A hook waiting for an unfinished copy, and
    a synchronization (the events in ``barriers`` and the end) waiting for all copies, delay every later
    event by the wait.
    """
    count = len(times)
    moments = [*times, end]
    queued: dict[tuple[int, int], list[int]] = defaultdict(list)
    due: dict[int, list[int]] = defaultdict(list)
    for index, job in enumerate(jobs):
        if job.deadline is not None and job.deadline < job.issue[0]:
            raise ValueError(f"copy {index} is due at event {job.deadline}, before its hook {job.issue[0]}")
        queued[job.issue].append(index)
        if job.deadline is not None:
            due[job.deadline].append(index)
    syncs = set(barriers) | {count}
    lanes: tuple[deque[int], deque[int]] = (deque(), deque())
    left = [float(job.nbytes) for job in jobs]
    finish = [math.inf] * len(jobs)
    clock = 0.0

    def head(lane: int) -> int | None:
        if not lanes[lane]:
            return None
        index = lanes[lane][0]
        after = jobs[index].after
        return index if after is None or finish[after] <= clock else None

    def advance(limit: float | None, done: Callable[[], bool] | None = None) -> None:
        nonlocal clock
        while done is None or not done():
            active = [(lane, index) for lane in (H2D, D2H) if (index := head(lane)) is not None]
            if not active:
                if done is not None:
                    raise RuntimeError("copy simulation waits for a copy that cannot start")
                clock = max(clock, limit)
                return
            shared = len(active) == 2
            speed = (
                rates.h2d_shared if shared else rates.h2d,
                rates.d2h_shared if shared else rates.d2h,
            )
            step = min(left[index] / speed[lane] for lane, index in active)
            if done is None and clock + step >= limit:
                for lane, index in active:
                    left[index] -= (limit - clock) * speed[lane]
                clock = limit
                return
            clock += step
            for lane, index in active:
                left[index] -= step * speed[lane]
                if left[index] <= 1e-9 * jobs[index].nbytes + 1e-12:
                    left[index] = 0.0
                    finish[index] = clock
                    lanes[lane].popleft()

    shift = 0.0
    leads = [math.inf] * len(jobs)
    unbounded: list[int] = []
    for event in range(count + 1):
        now = max(moments[event] + shift, clock)
        advance(now)
        for phase in (START, SETTLE, RESUME):
            if phase == SETTLE:
                for index in due.get(event, ()):
                    advance(None, lambda index=index: finish[index] < math.inf)
                    leads[index] = now - finish[index]
                    if finish[index] > now:
                        shift += finish[index] - now
                        now = finish[index]
                    clock = max(clock, now)
            if phase == RESUME and event in syncs:
                advance(None, lambda: not lanes[H2D] and not lanes[D2H])
                for index in unbounded:
                    leads[index] = now - finish[index]
                unbounded = []
                if clock > now:
                    shift += clock - now
                    now = clock
                clock = now
            for index in queued.get((event, phase), ()):
                lanes[jobs[index].lane].append(index)
                if jobs[index].deadline is None:
                    unbounded.append(index)
    return Simulation(shift, tuple(leads), tuple(finish))


@dataclass(frozen=True)
class UnitProfile:
    """What one module can offload, and where its Adam step and gradients sit in the step.

    Attributes:
        index: Module index.
        adam: Bytes of its Adam state.
        grad: Bytes of its FP32 gradients (alive from its first backward to its Adam step).
        param: Bytes of its parameters.
        first_forward: Event of its first forward (where a deferred Adam step runs), None if none.
        first_backward: Event of its first backward, None if none.
        final_backward: Event of its last backward after the last micro-batch's mark (an inline Adam
            step runs at the next event), None if none.
    """

    index: int
    adam: int
    grad: int
    param: int
    first_forward: int | None
    first_backward: int | None
    final_backward: int | None


@dataclass(frozen=True)
class GroupProfile:
    """The activations one module forward saves for its backward.

    Attributes:
        unit: Module index.
        forward: Event of the forward.
        backward: Event of the backward that first reads them.
        nbytes: Bytes the forward may offload.
    """

    unit: int
    forward: int
    backward: int
    nbytes: int


@dataclass(frozen=True)
class StepProfile:
    """One measured step.

    Attributes:
        times: Seconds into the step of every event, without copy waits.
        end: Seconds into the step at its end.
        barriers: Events that synchronize the device.
        units: Every module.
        groups: Every module forward with offloadable activations.
        memory: Allocated bytes of every memory index (see the module docstring) without Adam state and
            gradients; None if unmeasured.
    """

    times: tuple[float, ...]
    end: float
    barriers: tuple[int, ...]
    units: tuple[UnitProfile, ...]
    groups: tuple[GroupProfile, ...] = ()
    memory: tuple[int, ...] | None = None


@dataclass(frozen=True)
class OffloadPlan:
    """What a step offloads.

    Attributes:
        adam: ``RESIDENT``, ``INLINE`` or ``DEFERRED`` per module index.
        activations: Bytes offloaded per position in :attr:`StepProfile.groups`.
        params: Modules whose parameters leave the GPU between each forward and its backward.
        peak: Predicted peak of the memory indices, None without a memory profile.
        stall: Predicted seconds of copy waits.
    """

    adam: dict[int, str]
    activations: dict[int, int] = field(default_factory=dict)
    params: frozenset[int] = frozenset()
    peak: int | None = None
    stall: float = 0.0


@dataclass
class Schedule:
    """Where the engine queues each copy of a plan, and what the copies do to memory.

    Attributes:
        jobs: The copies of one step.
        windows: ``(first, last, bytes)`` memory index ranges (inclusive) adding ``bytes``.
        release_job: For a window, the copy that must finish before a negative ``bytes`` applies.
        resident: Bytes of Adam state that never leaves the GPU.
        update_event: Event of each offloaded module's Adam step.
        prefetch_start: Modules whose Adam state is prefetched at the start of each segment.
        prefetch_after: Modules whose Adam state is prefetched right after each module's Adam step.
        reload_at: Group positions reloaded at each ``(event, phase)``; ``(0, START)`` is the step start.
    """

    jobs: list[CopyJob]
    windows: list[tuple[int, int, int]]
    release_job: list[int | None]
    resident: int
    update_event: dict[int, int]
    prefetch_start: dict[int, list[int]]
    prefetch_after: dict[int, list[int]]
    reload_at: dict[tuple[int, int], list[int]]


def update_event(unit: UnitProfile, mode: str, count: int) -> int:
    """Event at which ``unit``'s Adam step runs in ``mode`` (``count`` is the step's end)."""
    if mode == DEFERRED:
        return unit.first_forward
    return unit.final_backward + 1 if unit.final_backward is not None else count


def build_schedule(plan: OffloadPlan, profile: StepProfile, *, lookahead: int, throttle: int) -> Schedule:
    """Place the copies of ``plan``.

    Within each segment between synchronizations, the first ``lookahead`` offloaded Adam steps are
    prefetched at its start and each later one right after the Adam step ``lookahead`` earlier; every
    write-back is queued right after its Adam step and must finish ``throttle`` hooks later. A module
    forward's offloaded activations are copied out when it ends and must finish ``throttle`` hooks later.
    Reloads of those activations and of offloaded parameters are queued ``lookahead`` hooks before the
    backward, and not before the copy-out finished, so the bytes stay off the GPU until then.
    """
    count = len(profile.times)
    last = count + 1
    barriers = sorted(profile.barriers)
    barrier_set = set(barriers)
    units = {unit.index: unit for unit in profile.units}

    def segment(event: int, phase: int) -> int:
        position = bisect.bisect_left(barriers, event)
        return position + 1 if phase == RESUME and event in barrier_set else position

    def segment_start(index: int) -> tuple[int, int]:
        return (0, START) if index == 0 else (barriers[index - 1], RESUME)

    def segment_end(index: int) -> int:
        return barriers[index] if index < len(barriers) else count

    def first_index(issue: tuple[int, int]) -> int:
        return 0 if issue == (0, START) else issue[0] + 1

    keyed: list[tuple[tuple[int, int], int, int, CopyJob]] = []
    windows: list[tuple[int, int, int]] = []
    release_job: list[int | None] = []
    resident = 0
    updates: dict[int, int] = {}
    members: dict[int, list[int]] = defaultdict(list)

    def add(job: CopyJob, order: int) -> int:
        # A deadline is waited on at SETTLE, so a copy due at its own hook must be queued at START.
        if job.deadline is not None and job.issue >= (job.deadline, SETTLE):
            job = CopyJob(job.lane, job.nbytes, (job.deadline, START), job.deadline, job.after)
        keyed.append((job.issue, order, len(keyed), job))
        return len(keyed) - 1

    for unit in profile.units:
        mode = plan.adam.get(unit.index, RESIDENT) if unit.first_forward is not None else RESIDENT
        if mode == RESIDENT:
            resident += unit.adam
        else:
            updates[unit.index] = update_event(unit, mode, count)
            members[segment(updates[unit.index], SETTLE)].append(unit.index)
        if unit.first_backward is not None:
            settle = unit.final_backward + 1 if unit.final_backward is not None else count
            if mode == DEFERRED:
                windows.append((unit.first_backward + 1, last, unit.grad))
                windows.append((0, unit.first_forward + 1, unit.grad))
                release_job.extend((None, None))
            else:
                windows.append((unit.first_backward + 1, min(settle + 1, last), unit.grad))
                release_job.append(None)
        if unit.index in plan.params:
            settle = updates.get(unit.index, unit.final_backward + 1 if unit.final_backward is not None else count)
            add(CopyJob(D2H, unit.param, (settle, SETTLE)), _REFRESH)

    prefetch_start: dict[int, list[int]] = defaultdict(list)
    prefetch_after: dict[int, list[int]] = defaultdict(list)
    for index, chain in members.items():
        chain.sort(key=lambda unit_index: (updates[unit_index], unit_index))
        for position, unit_index in enumerate(chain):
            unit, event = units[unit_index], updates[unit_index]
            if position < lookahead:
                issue = segment_start(index)
                prefetch_start[index].append(unit_index)
            else:
                trigger = chain[position - lookahead]
                issue = (updates[trigger], SETTLE)
                prefetch_after[trigger].append(unit_index)
            add(CopyJob(H2D, unit.adam, issue, event), _PREFETCH)
            boundary = segment_end(index)
            deadline = event + throttle if event + throttle < boundary else None
            add(CopyJob(D2H, unit.adam, (event, SETTLE), deadline), _WRITE_BACK)
            windows.append(
                (first_index(issue), min((deadline if deadline is not None else boundary) + 1, last), unit.adam)
            )
            release_job.append(None)

    offloads: dict[int, int] = {}
    reloading: dict[int, list[int]] = defaultdict(list)
    for position, group in enumerate(profile.groups):
        nbytes = plan.activations.get(position, 0)
        params = group.unit in plan.params
        if not nbytes and not params:
            continue
        reloading[segment(group.backward, SETTLE)].append(position)
        if nbytes:
            boundary = segment_end(segment(group.forward + 1, START))
            deadline = group.forward + 1 + throttle
            offloads[position] = add(
                CopyJob(D2H, nbytes, (group.forward + 1, START), deadline if deadline < boundary else None), _REFRESH
            )
    reload_at: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, chain in reloading.items():
        chain.sort(key=lambda position: (profile.groups[position].backward, position))
        for rank, position in enumerate(chain):
            group = profile.groups[position]
            nbytes = plan.activations.get(position, 0)
            # Keep the bytes off the GPU until shortly before this backward. The first ``lookahead``
            # backwards in the segment have no earlier backward to chain from.
            candidate = (max(group.backward - lookahead, 0), START)
            if rank >= lookahead:
                candidate = max(candidate, (profile.groups[chain[rank - lookahead]].backward, SETTLE))
            earliest = (group.forward + 1 + (throttle if nbytes else 0), SETTLE)
            issue = min(max(candidate, earliest), (group.backward, SETTLE))
            reload_at[issue].append(position)
            if nbytes:
                add(CopyJob(H2D, nbytes, issue, group.backward, offloads[position]), _RELOAD)
                windows.append((group.forward + 2 + throttle, issue[0], -nbytes))
                release_job.append(offloads[position])
            if group.unit in plan.params:
                param = units[group.unit].param
                add(CopyJob(H2D, param, issue, group.backward), _RELOAD)
                windows.append((group.forward + 2, issue[0], -param))
                release_job.append(None)

    keyed.sort(key=lambda item: (item[0], item[1], item[2]))
    renumber = {original: position for position, (_, _, original, _) in enumerate(keyed)}
    jobs = [
        CopyJob(job.lane, job.nbytes, job.issue, job.deadline, None if job.after is None else renumber[job.after])
        for _, _, _, job in keyed
    ]
    return Schedule(
        jobs=jobs,
        windows=windows,
        release_job=[None if job is None else renumber[job] for job in release_job],
        resident=resident,
        update_event=updates,
        prefetch_start=dict(prefetch_start),
        prefetch_after=dict(prefetch_after),
        reload_at=dict(reload_at),
    )


def predict_memory(schedule: Schedule, profile: StepProfile, finish: tuple[float, ...] | None = None) -> np.ndarray:
    """Allocated bytes of every memory index under ``schedule``.

    ``finish`` is the simulated completion time of each copy. A negative window waits until that
    copy has finished, so bytes stay in the peak while they are still queued behind other copies.
    """
    base = np.asarray(profile.memory, dtype=np.int64)
    delta = np.zeros(len(base) + 1, dtype=np.int64)
    times = profile.times
    for (first, last, nbytes), job in zip(schedule.windows, schedule.release_job, strict=True):
        if nbytes < 0 and finish is not None and job is not None:
            # Memory index k follows event k - 1. The copy is gone from the first interval after
            # the first event whose hook is at or after the copy finishes.
            first = max(first, bisect.bisect_left(times, finish[job]) + 1)
        first, last = max(first, 0), min(last, len(base) - 1)
        if first <= last:
            delta[first] += nbytes
            delta[last + 1] -= nbytes
    return base + schedule.resident + np.cumsum(delta[:-1])


@dataclass
class _Evaluation:
    plan: OffloadPlan
    peak: int | None
    stall: float
    slack: float


def _evaluate(plan: OffloadPlan, profile: StepProfile, rates: CopyRates, lookahead: int, throttle: int) -> _Evaluation:
    schedule = build_schedule(plan, profile, lookahead=lookahead, throttle=throttle)
    simulation = simulate_copies(
        schedule.jobs, times=list(profile.times), end=profile.end, barriers=list(profile.barriers), rates=rates
    )
    peak = int(predict_memory(schedule, profile, simulation.finish).max()) if profile.memory is not None else None
    return _Evaluation(plan, peak, simulation.stall, simulation.slack)


def _with(plan: OffloadPlan, **changes) -> OffloadPlan:
    return OffloadPlan(
        adam=changes.get("adam", plan.adam),
        activations=changes.get("activations", plan.activations),
        params=changes.get("params", plan.params),
    )


def plan_offload(
    profile: StepProfile,
    *,
    rates: CopyRates,
    lookahead: int,
    throttle: int,
    budget: int | None = None,
    scan: int = 32,
) -> OffloadPlan:
    """Plan a step's offloads.

    Without ``budget``, offload Adam state greedily, largest module first, while no copy delays the
    computation; each module takes the Adam window that leaves the larger slack.

    With ``budget`` (and :attr:`StepProfile.memory`), start from everything resident and add offloads
    (a module's Adam state in either window, a module forward's activations or part of them, a module's
    parameters) until the predicted peak fits: each round takes the largest addition that lowers the peak
    without adding copy waits, or else, among the ``scan`` largest that lower it, the one adding the
    least wait per byte of peak. Stops early, over budget, when nothing lowers the peak.
    """
    count = len(profile.times)
    offloadable = [unit for unit in profile.units if unit.first_forward is not None]
    plan = OffloadPlan(adam={unit.index: RESIDENT for unit in profile.units})
    current = _evaluate(plan, profile, rates, lookahead, throttle)
    if budget is None or profile.memory is None:
        for unit in sorted(offloadable, key=lambda unit: unit.adam, reverse=True):
            best = None
            for mode in (DEFERRED, INLINE):
                if mode == INLINE and unit.final_backward is None:
                    continue
                trial = _evaluate(
                    _with(plan, adam={**plan.adam, unit.index: mode}), profile, rates, lookahead, throttle
                )
                if trial.stall <= 1e-12 and (best is None or trial.slack > best.slack):
                    best = trial
            if best is not None:
                plan, current = best.plan, best
        return OffloadPlan(plan.adam, peak=current.peak, stall=current.stall)

    groups_of: dict[int, list[int]] = defaultdict(list)
    for position, group in enumerate(profile.groups):
        groups_of[group.unit].append(position)
    options: list[tuple[int, str, int, str | None]] = []
    for unit in offloadable:
        options.append((unit.adam, "adam", unit.index, DEFERRED))
        if unit.final_backward is not None:
            options.append((unit.adam, "adam", unit.index, INLINE))
        if groups_of.get(unit.index) and unit.param:
            options.append((unit.param * len(groups_of[unit.index]), "param", unit.index, None))
    options.extend((group.nbytes, "act", position, None) for position, group in enumerate(profile.groups))
    options.sort(key=lambda option: option[0], reverse=True)

    def extend(option: tuple[int, str, int, str | None], over: int) -> OffloadPlan | None:
        _, kind, index, mode = option
        if kind == "adam":
            if plan.adam[index] != RESIDENT or (mode == DEFERRED and index in plan.params):
                return None
            return _with(plan, adam={**plan.adam, index: mode})
        if kind == "param":
            if index in plan.params or plan.adam[index] == DEFERRED:
                return None
            return _with(plan, params=plan.params | {index})
        done = plan.activations.get(index, 0)
        total = profile.groups[index].nbytes
        if done >= total:
            return None
        return _with(plan, activations={**plan.activations, index: min(total, done + max(over, 1))})

    while current.peak > budget:
        over = current.peak - budget
        chosen, best, scanned = None, None, 0
        for option in options:
            trial_plan = extend(option, over)
            if trial_plan is None:
                continue
            trial = _evaluate(trial_plan, profile, rates, lookahead, throttle)
            if trial.peak >= current.peak:
                continue
            if trial.stall <= current.stall + 1e-9:
                chosen = trial
                break
            score = (trial.stall - current.stall) / (current.peak - trial.peak)
            if best is None or score < best[0]:
                best = (score, trial)
            scanned += 1
            if scanned >= scan:
                break
        if chosen is None and best is not None:
            chosen = best[1]
        if chosen is None:
            break
        plan, current = chosen.plan, chosen
    del count
    return OffloadPlan(plan.adam, plan.activations, plan.params, peak=current.peak, stall=current.stall)
