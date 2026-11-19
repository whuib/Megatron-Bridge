# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan partial MLP replay and Adam residency offline from matched fixed-shape measurements."""

import argparse
import json
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


_MIB = 2**20


def resident_indices(sizes: Sequence[int], budget: int) -> Iterator[int]:
    """Yield whole buffers that fit the byte budget, largest first with stable ties."""
    if budget == 0:
        return
    for index in sorted(range(len(sizes)), key=sizes.__getitem__, reverse=True):
        size = sizes[index]
        if size <= budget:
            yield index
            budget -= size


@dataclass(frozen=True)
class RankCalibration:
    """Per-rank device high-water, Adam buffer sizes and steady-step CUDA trace metrics.

    Device high-water includes allocator reservation and allocations outside PyTorch. Copy
    bytes count only state migration; exposed time excludes overlap with compute or NCCL.
    """

    peak_device_mib: int
    adam_unit_bytes: tuple[int, ...]
    h2d_bytes: int
    d2h_bytes: int
    compute_seconds: float
    copy_seconds: float
    exposed_copy_seconds: float


@dataclass(frozen=True)
class RecomputeCalibration:
    """One fixed workload at a measured MLP prefix and optimizer residency cap."""

    recompute_mlp_layers: int
    resident_optim_mib: int
    step_seconds: float
    ranks: tuple[RankCalibration, ...]


@dataclass(frozen=True)
class RecomputePlan:
    """Static training knobs and predictions for selecting a candidate to validate."""

    recompute_mlp_layers: int
    resident_optim_mib: int
    peak_device_mib: int
    headroom_mib: int
    resident_optim_bytes_per_rank: tuple[int, ...]
    estimated_step_seconds: float
    worst_rank_serial_copy_seconds: float
    worst_rank_copy_excess_over_compute_seconds: float


def plan_recompute(
    calibrations: Sequence[RecomputeCalibration],
    *,
    min_device_mib: int,
    max_device_mib: int,
    headroom_mib: int,
    h2d_bytes_per_second: float,
    d2h_bytes_per_second: float,
) -> RecomputePlan:
    """Choose a measured MLP prefix and a whole-buffer Adam cap before training.

    Use calibrations for the same workload, layout, software and device placement. A single
    copy stream transfers the nonresident Adam buffers both ways once per step. Estimate
    how much of the avoided copy time was exposed, then rank candidates by projected step
    time. The model adds growth in resident bytes to each measured device peak. The requested
    headroom allows for changes in allocator reservation and workspace, which need validation
    against a training run. Serial copy excess compares the total copy time against an ideal
    compute window, which can be further restricted by per-layer dependencies.
    """
    peak_limit = max_device_mib - headroom_mib
    if not 0 < min_device_mib <= peak_limit or headroom_mib < 0:
        raise ValueError("Invalid device memory range or headroom")
    rates = (h2d_bytes_per_second, d2h_bytes_per_second)
    if not all(math.isfinite(rate) and rate > 0 for rate in rates):
        raise ValueError("Both copy bandwidths must be positive and finite")
    if not calibrations or not calibrations[0].ranks:
        raise ValueError("At least one calibration with rank measurements is required")

    inventories = tuple(rank.adam_unit_bytes for rank in calibrations[0].ranks)
    best = None
    for sample in calibrations:
        if tuple(rank.adam_unit_bytes for rank in sample.ranks) != inventories:
            raise ValueError("Calibrations must have matching per-rank Adam inventories")
        if (
            min(sample.recompute_mlp_layers, sample.resident_optim_mib) < 0
            or not math.isfinite(sample.step_seconds)
            or sample.step_seconds <= 0
            or any(
                not rank.adam_unit_bytes
                or min(rank.adam_unit_bytes) <= 0
                or min(rank.peak_device_mib, rank.h2d_bytes, rank.d2h_bytes) <= 0
                or not all(
                    math.isfinite(time)
                    for time in (rank.exposed_copy_seconds, rank.copy_seconds, rank.compute_seconds)
                )
                or not 0 <= rank.exposed_copy_seconds <= rank.copy_seconds
                or rank.copy_seconds <= 0
                or rank.compute_seconds <= 0
                for rank in sample.ranks
            )
        ):
            raise ValueError("Invalid recompute calibration metrics")

        def resident(inventory: tuple[int, ...], cap_mib: int) -> int:
            return sum(inventory[index] for index in resident_indices(inventory, cap_mib * _MIB))

        baseline = tuple(resident(inventory, sample.resident_optim_mib) for inventory in inventories)
        cap_limit = min((peak_limit - rank.peak_device_mib) + old // _MIB for rank, old in zip(sample.ranks, baseline))
        # Include the measured cap itself even if some of it was unusable due to granularity.
        for cap in range(sample.resident_optim_mib, max(cap_limit, sample.resident_optim_mib) + 1):
            selected = tuple(resident(inventory, cap) for inventory in inventories)
            growth = [new - old for new, old in zip(selected, baseline)]
            if min(growth) < 0:
                continue
            peak = max(rank.peak_device_mib + (delta + _MIB - 1) // _MIB for rank, delta in zip(sample.ranks, growth))
            if not min_device_mib <= peak <= peak_limit:
                continue
            copy_seconds, copy_excess, savings = [], [], []
            for rank, delta in zip(sample.ranks, growth):
                old_copy = rank.h2d_bytes / h2d_bytes_per_second + rank.d2h_bytes / d2h_bytes_per_second
                new_copy = (
                    max(0, rank.h2d_bytes - delta) / h2d_bytes_per_second
                    + max(0, rank.d2h_bytes - delta) / d2h_bytes_per_second
                )
                copy_seconds.append(new_copy)
                copy_excess.append(max(0.0, new_copy - rank.compute_seconds))
                saves = (old_copy - new_copy) * rank.exposed_copy_seconds / rank.copy_seconds
                savings.append(min(saves, rank.exposed_copy_seconds))
            candidate = RecomputePlan(
                sample.recompute_mlp_layers,
                cap,
                peak,
                max_device_mib - peak,
                selected,
                sample.step_seconds - min(savings),
                max(copy_seconds),
                max(copy_excess),
            )
            score = (candidate.estimated_step_seconds, candidate.recompute_mlp_layers, candidate.resident_optim_mib)
            if best is None or score < best[0]:
                best = score, candidate

    if best is None:
        raise ValueError("No calibrated MLP prefix and Adam cap fit the requested range and headroom")
    return best[1]


def main() -> None:
    """Write the plan for a JSON list of recompute calibrations to standard output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibrations", type=Path, help="JSON list of RecomputeCalibration records")
    parser.add_argument("--min-device-mib", type=int, required=True)
    parser.add_argument("--max-device-mib", type=int, required=True)
    parser.add_argument("--headroom-mib", type=int, required=True)
    parser.add_argument("--h2d-gb-per-second", type=float, required=True)
    parser.add_argument("--d2h-gb-per-second", type=float, required=True)
    args = parser.parse_args()
    try:
        calibrations = [
            RecomputeCalibration(**{**sample, "ranks": tuple(RankCalibration(**rank) for rank in sample["ranks"])})
            for sample in json.loads(args.calibrations.read_text())
        ]
        plan = plan_recompute(
            calibrations,
            min_device_mib=args.min_device_mib,
            max_device_mib=args.max_device_mib,
            headroom_mib=args.headroom_mib,
            h2d_bytes_per_second=args.h2d_gb_per_second * 1e9,
            d2h_bytes_per_second=args.d2h_gb_per_second * 1e9,
        )
    except (OSError, TypeError, KeyError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write(json.dumps(asdict(plan), indent=2) + "\n")


if __name__ == "__main__":
    main()
