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

"""Tabulate a Qwen3-Omni Seq x GBS grid: throughput and MFU per mode, with OOM and failures kept.

Cases that ``plan_omni_grid.py`` did not launch appear as ``unsupported: <reason>`` or
``OOM (inferred: ...)`` rows.
"""

import argparse
import json
import logging
import re
import statistics
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qwen3_vl"))
from gpu_activity import foreign_at_oom_mib  # noqa: E402


logger = logging.getLogger(__name__)

# An OOM with more foreign memory than this on the failing GPU is not attributed to the case (the runner
# repeats such runs; this marks one whose repeat was not clean either).
OOM_FOREIGN_LIMIT_MIB = 2048

_POINT = re.compile(r"seq(\d+)_gbs(\d+)$")
_RUN_LOG = re.compile(r"^(.+)\.rep\d+\.log$")
# PyTorch's allocator, or NCCL failing to allocate device memory for its own buffers (NCCL allocates
# outside PyTorch's caching allocator, so a full device can fail there first).
TORCH_OOM = re.compile(r"OutOfMemoryError|CUDA out of memory")
NCCL_OOM = re.compile(r"Cuda failure 2 'out of memory'|Failed to CUDA calloc|cudaMalloc[^\n]*out of memory")
_FAILURES = (
    ("OOM", TORCH_OOM),
    ("OOM (NCCL allocation)", NCCL_OOM),
    ("NCCL error", re.compile(r"DistBackendError|NCCL error")),
)


def is_oom(text: str) -> bool:
    """Whether a run log ends in an out-of-memory failure (PyTorch or NCCL allocation)."""
    return bool(TORCH_OOM.search(text) or NCCL_OOM.search(text))


def is_clean_oom(text: str) -> bool:
    """An OOM that other processes' memory on the failing GPU cannot explain."""
    foreign = foreign_at_oom_mib(text)
    return is_oom(text) and (foreign is None or foreign <= OOM_FOREIGN_LIMIT_MIB)


_OOM_DETAIL = re.compile(
    r"this process has ([\d.]+) GiB memory in use\. Of the allocated memory ([\d.]+) GiB is allocated by PyTorch, "
    r"and ([\d.]+) (GiB|MiB) is reserved by PyTorch but unallocated"
)


def oom_detail(log: Path) -> dict[str, float]:
    """Memory at the first PyTorch OOM of a run: process total, PyTorch-allocated, and outside PyTorch (GiB)."""
    match = _OOM_DETAIL.search(log.read_text(errors="replace")) if log.exists() else None
    if not match:
        return {}
    in_use, allocated, unallocated = (float(value) for value in match.groups()[:3])
    unallocated /= 1024 if match.group(4) == "MiB" else 1
    foreign = foreign_at_oom_mib(log.read_text(errors="replace"))
    return {
        "oom_process_gib": in_use,
        "oom_allocated_gib": allocated,
        "oom_outside_pytorch_gib": round(in_use - allocated - unallocated, 2),
        "oom_foreign_gib": round(max(foreign or 0.0, 0.0) / 1024, 2),
    }


def _failure(log: Path) -> str:
    text = log.read_text(errors="replace") if log.exists() else ""
    if is_oom(text) and not is_clean_oom(text):
        return f"OOM with {foreign_at_oom_mib(text) / 1024:.1f} GiB foreign GPU memory present"
    for label, pattern in _FAILURES:
        if pattern.search(text):
            return label
    match = re.search(r"^\[rank\d+\]: (\w+Error: .{0,120})", text, re.MULTILINE)
    return f"failed: {match.group(1)}" if match else "failed (no throughput record)"


# Cases plan_omni_grid.py did not launch, and the status they get.
_MARKERS = {
    "unsupported": "unsupported: {reason}",
    "oom_inferred": "OOM (inferred: {reason})",
    "not_selected": "not selected: {reason}",
}


def collect(grid_dir: Path) -> list[dict[str, Any]]:
    """One row per (point, case) with either a measurement, the failure reason or why it was not launched."""
    rows = []
    for point in sorted(path for path in grid_dir.iterdir() if path.is_dir() and _POINT.search(path.name)):
        seq, gbs = (int(value) for value in _POINT.search(point.name).groups())
        # Training logs only (``<case>.rep<N>.log``), not the gate logs next to them.
        logs = sorted(log for log in point.glob("*.log") if _RUN_LOG.search(log.name))
        cases = sorted({_RUN_LOG.search(log.name).group(1) for log in logs})
        markers = [marker for kind in _MARKERS for marker in sorted(point.glob(f"*.rep1.{kind}"))]
        for marker in markers:
            case, kind = marker.name.split(".rep1.")
            status = _MARKERS[kind].format(reason=marker.read_text().strip())
            rows.append({"seq": seq, "gbs": gbs, "case": case, "status": status})
        for case in cases:
            records = []
            for log in (log for log in logs if _RUN_LOG.search(log.name).group(1) == case):
                record_path = log.with_suffix(".json")
                if record_path.exists():
                    records.append(json.loads(record_path.read_text()))
                else:
                    rows.append({"seq": seq, "gbs": gbs, "case": case, "status": _failure(log), **oom_detail(log)})
            if not records:
                continue
            meta = records[0]["metadata"]
            rows.append(
                {
                    "seq": seq,
                    "gbs": gbs,
                    "case": case,
                    "mode": meta["mode"],
                    "layout": meta["layout"],
                    "status": "ok",
                    "runs": len(records),
                    "step_seconds": statistics.median(r["step_seconds"] for r in records),
                    "samples_per_second": statistics.median(r["samples_per_second"] for r in records),
                    "padded_tokens_per_second": statistics.median(r["padded_tokens_per_second"] for r in records),
                    "model_tflops_per_gpu": statistics.median(r["model_tflops_per_gpu"] for r in records),
                    "mfu": statistics.median(r["mfu"] for r in records),
                    "max_peak_allocated_gib": max(r["max_peak_allocated_gib"] for r in records),
                    "max_outside_pytorch_gib": max(
                        max(r.get("outside_pytorch_gib_per_rank") or [0.0]) for r in records
                    ),
                    "finite": all(r["loss_and_grad_finite_on_all_ranks"] for r in records),
                }
            )
    return rows


def main() -> int:
    """Print the grid table and write ``grid_summary.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = collect(args.grid_dir)
    if not rows:
        logger.error("no grid points in %s", args.grid_dir)
        return 1
    logger.info("| Seq | GBS | case | step s | samples/s | tokens/s | TFLOPS/GPU | MFU | peak GiB | status |")
    logger.info("|" + "---|" * 10)
    for row in rows:
        if row["status"] != "ok":
            logger.info(f"| {row['seq']} | {row['gbs']} | {row['case']} | | | | | | | {row['status']} |")
            continue
        logger.info(
            f"| {row['seq']} | {row['gbs']} | {row['case']} | {row['step_seconds']:.3f} | "
            f"{row['samples_per_second']:.2f} | {row['padded_tokens_per_second']:.0f} | "
            f"{row['model_tflops_per_gpu']:.1f} | {100 * row['mfu']:.1f}% | {row['max_peak_allocated_gib']:.1f} | "
            f"{'ok' if row['finite'] else 'NON-FINITE'} |"
        )
    (args.grid_dir / "grid_summary.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
