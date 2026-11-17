# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarize throughput records and refuse to compare runs with different contracts."""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Arguments allowed to differ between compared runs: the execution mode, its layout and its own
# memory/schedule knobs (each mode is compared at its best feasible setting).
LAYOUT_ARGS = {
    "mode",
    "language_tp",
    "language_pp",
    "encoder_tp",
    "vision_gpus",
    "sequence_parallel",
    "vision_recompute",
    "mux_encoder_schedule",
}
# Megatron pads the vocabulary to a multiple of 128 * TP, so the logit FLOPs (and the model FLOPs per
# iteration) legitimately differ by layout. Anything beyond this relative gap is a different workload.
FLOPS_RELATIVE_TOLERANCE = 1e-3


def contract(record: dict[str, Any]) -> dict[str, Any]:
    """Everything that must be identical for two records to be comparable."""
    meta = record["metadata"]
    return {
        "args": {key: value for key, value in meta["args"].items() if key not in LAYOUT_ARGS},
        "workload": record["workload"],
        "world_size": record["world_size"],
        "measure_iters": record["measure_iters"],
        "warmup_iters": record["warmup_iters"],
        "versions": meta["versions"],
        "gpu": meta["gpu"],
        "git_commit": meta["git"]["commit"],
    }


def contract_differences(records: list[dict[str, Any]]) -> list[str]:
    """Return the contract fields that differ between records (empty when comparable)."""
    if not records:
        return []
    reference = contract(records[0])
    differences = []
    for record in records[1:]:
        current = contract(record)
        for key in reference:
            if current[key] != reference[key]:
                differences.append(f"{record['_source']}: {key} differs ({current[key]!r} != {reference[key]!r})")
        flops, reference_flops = record["flops_per_iteration"], records[0]["flops_per_iteration"]
        if abs(flops - reference_flops) > FLOPS_RELATIVE_TOLERANCE * reference_flops:
            differences.append(f"{record['_source']}: flops_per_iteration differs ({flops} != {reference_flops})")
    return differences


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median and range per mode/layout over repetitions, with speedup relative to native."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        meta = record["metadata"]
        layout = ",".join(f"{key}={value}" for key, value in sorted(meta["layout"].items()))
        groups.setdefault((meta["mode"], layout), []).append(record)
    rows = []
    for (mode, layout), group in sorted(groups.items()):
        steps = [record["step_seconds"] for record in group]
        rows.append(
            {
                "mode": mode,
                "layout": layout,
                "runs": len(group),
                "median_step_seconds": statistics.median(steps),
                "min_step_seconds": min(steps),
                "max_step_seconds": max(steps),
                "median_samples_per_second": statistics.median(r["samples_per_second"] for r in group),
                "median_model_tflops_per_gpu": statistics.median(r["model_tflops_per_gpu"] for r in group),
                "max_peak_allocated_gib": max(r["max_peak_allocated_gib"] for r in group),
                "max_foreign_mean_sm_percent": max((r.get("foreign_mean_sm_percent") or 0.0) for r in group),
                "all_finite": all(r["loss_and_grad_finite_on_all_ranks"] for r in group),
            }
        )
    native = [row["median_step_seconds"] for row in rows if row["mode"] == "native"]
    for row in rows:
        row["speedup_vs_native"] = min(native) / row["median_step_seconds"] if native else None
    return rows


def main() -> int:
    """Load ``*.json`` throughput records from a directory and print a comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--allow-contract-mismatch", action="store_true")
    parser.add_argument(
        "--max-foreign-sm",
        type=float,
        help="exclude runs whose foreign mean SM percent (gpu_activity.py monitor) exceeded this on any GPU",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = []
    for path in sorted(args.results_dir.glob("*.json")):
        if path.name == "summary.json" or path.name.endswith(".interference.json"):
            continue
        record = json.loads(path.read_text())
        record["_source"] = path.name
        interference = path.with_name(path.name[: -len(".json")] + ".interference.json")
        if interference.exists():
            record["foreign_mean_sm_percent"] = json.loads(interference.read_text())["max_foreign_mean_sm_percent"]
        if args.max_foreign_sm is not None and (record.get("foreign_mean_sm_percent") or 0.0) > args.max_foreign_sm:
            logger.warning("excluded %s: foreign mean SM %.2f%%", path.name, record["foreign_mean_sm_percent"])
            continue
        records.append(record)
    if not records:
        logger.error("no throughput records in %s", args.results_dir)
        return 1
    differences = contract_differences(records)
    for difference in differences:
        logger.error("NOT COMPARABLE: %s", difference)
    if differences and not args.allow_contract_mismatch:
        return 1
    rows = summarize(records)
    header = (
        "| mode | layout | runs | step s (median) | step s (min-max) | samples/s | model TFLOPS/GPU "
        "| speedup vs native | max peak alloc GiB | finite | max foreign SM % |"
    )
    logger.info(header)
    logger.info("|" + "---|" * 11)
    for row in rows:
        speedup = f"{row['speedup_vs_native']:.3f}" if row["speedup_vs_native"] else "n/a"
        logger.info(
            f"| {row['mode']} | {row['layout']} | {row['runs']} | {row['median_step_seconds']:.3f} | "
            f"{row['min_step_seconds']:.3f}-{row['max_step_seconds']:.3f} | {row['median_samples_per_second']:.3f} | "
            f"{row['median_model_tflops_per_gpu']:.1f} | {speedup} | {row['max_peak_allocated_gib']:.1f} | "
            f"{row['all_finite']} | {row['max_foreign_mean_sm_percent']:.2f} |"
        )
    reference = records[0]
    logger.info(
        "\nContract: %s; %d GPUs (%s); warmup %d + measured %d steps per run.",
        reference["metadata"]["label"],
        reference["world_size"],
        reference["metadata"]["gpu"],
        reference["warmup_iters"],
        reference["measure_iters"],
    )
    (args.results_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
