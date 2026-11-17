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

"""Choose which cases of one Qwen3-Omni grid point to launch; record why the others are not launched.

A case is not launched, and a marker file takes the place of its run log, when

* its layout or workload is rejected before launch (``<case>.rep1.unsupported``, holding the reason), or
* it ran out of memory at a point whose sequence length and global batch size are both no larger
  (``<case>.rep1.oom_inferred``, naming that point). With micro-batch size 1 a rank's peak memory
  does not decrease when either grows: longer sequences carry more tokens and more media, and a larger
  global batch only adds micro-batches (the in-flight count is bounded or grows).

With ``--select-top K --screen-gbs G``, a point with a global batch larger than G launches only the K
fastest layouts of each mode measured at the same sequence length and GBS G; the others get a
``<case>.rep1.not_selected`` marker naming that screening point.

A case that already finished at this point (throughput record, OOM, or a log ending in a launcher
failure report) is kept and not launched again, so an interrupted grid can be resumed. The remaining cases are
written to ``--out`` in the runner's ``<name> <mode> <args>`` format.
"""

import argparse
import json
import logging
import re
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

from summarize_omni_grid import is_clean_oom, is_oom


logger = logging.getLogger(__name__)

_POINT = re.compile(r"seq(\d+)_gbs(\d+)$")
# torch.distributed.run's failure report at the end of a failed launch.
_LAUNCH_FAILED = re.compile(r"ChildFailedError|exitcode\s*:")


def read_cases(path: Path) -> list[tuple[str, str, str]]:
    """``(name, mode, layout args)`` per non-comment line of a cases file."""
    cases = []
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=2)
        if not parts or parts[0].startswith("#"):
            continue
        cases.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return cases


def measured_oom(grid_dir: Path, case: str, seq: int, gbs: int) -> tuple[int, int] | None:
    """The smallest earlier point (seq' <= seq, gbs' <= gbs) where ``case`` ran out of memory."""
    hits = []
    for point in grid_dir.iterdir():
        match = _POINT.search(point.name)
        if not point.is_dir() or not match:
            continue
        point_seq, point_gbs = (int(value) for value in match.groups())
        if (point_seq, point_gbs) == (seq, gbs) or point_seq > seq or point_gbs > gbs:
            continue
        for log in point.glob(f"{case}.rep*.log"):
            if not log.with_suffix(".json").exists() and is_clean_oom(log.read_text(errors="replace")):
                hits.append((point_seq, point_gbs))
    return min(hits) if hits else None


def finished(point: Path, case: str) -> bool:
    """Whether ``case`` already has a throughput record, an OOM or a failed launch at this point."""
    for log in point.glob(f"{case}.rep*.log"):
        text = log.read_text(errors="replace")
        if log.with_suffix(".json").exists() or is_oom(text) or _LAUNCH_FAILED.search(text):
            return True
    return False


def screened(grid_dir: Path, cases: list[tuple[str, str, str]], *, seq: int, gbs: int, top: int) -> set[str]:
    """The ``top`` fastest measured cases of each mode at ``(seq, gbs)``."""
    point = grid_dir / f"seq{seq}_gbs{gbs}"
    by_mode: dict[str, list[tuple[float, str]]] = {}
    for name, mode, _ in cases:
        record = point / f"{name}.rep1.json"
        if record.exists():
            by_mode.setdefault(mode, []).append((json.loads(record.read_text())["step_seconds"], name))
    return {name for timings in by_mode.values() for _, name in sorted(timings)[:top]}


def plan_point(
    grid_dir: Path,
    cases: list[tuple[str, str, str]],
    *,
    seq: int,
    gbs: int,
    workload_args: str,
    reason_fn: Callable[[list[str]], str | None],
    select_top: int | None = None,
    screen_gbs: int | None = None,
) -> list[tuple[str, str, str]]:
    """Write skip markers into ``grid_dir/seq{seq}_gbs{gbs}`` and return the cases to launch."""
    point = grid_dir / f"seq{seq}_gbs{gbs}"
    point.mkdir(parents=True, exist_ok=True)
    selected = None
    if select_top is not None and gbs > screen_gbs:
        selected = screened(grid_dir, cases, seq=seq, gbs=screen_gbs, top=select_top)
    launch = []
    for name, mode, layout_args in cases:
        if finished(point, name):
            continue
        reason = reason_fn(["--mode", mode, *shlex.split(layout_args), *shlex.split(workload_args)])
        if reason is not None:
            (point / f"{name}.rep1.unsupported").write_text(reason + "\n")
            continue
        oom = measured_oom(grid_dir, name, seq, gbs)
        if oom is not None:
            (point / f"{name}.rep1.oom_inferred").write_text(f"OOM measured at seq {oom[0]}, GBS {oom[1]}\n")
            continue
        if selected is not None and name not in selected:
            (point / f"{name}.rep1.not_selected").write_text(
                f"not among the {select_top} fastest {mode} layouts at seq {seq}, GBS {screen_gbs}\n"
            )
            continue
        launch.append((name, mode, layout_args))
    return launch


def main() -> int:
    """Plan one grid point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--gbs", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--workload-args", required=True, help="the workload arguments every case is launched with")
    parser.add_argument("--out", type=Path, required=True, help="cases file of the cases to launch")
    parser.add_argument("--select-top", type=int, help="launch only the K fastest layouts per mode above --screen-gbs")
    parser.add_argument("--screen-gbs", type=int, help="global batch whose measurements select the layouts")
    args = parser.parse_args()
    if (args.select_top is None) != (args.screen_gbs is None):
        parser.error("--select-top and --screen-gbs go together")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare_training import unsupported_reason

    launch = plan_point(
        args.grid_dir,
        read_cases(args.cases),
        seq=args.seq,
        gbs=args.gbs,
        workload_args=args.workload_args,
        reason_fn=lambda argv: unsupported_reason(argv, args.world_size),
        select_top=args.select_top,
        screen_gbs=args.screen_gbs,
    )
    args.out.write_text("".join(f"{name} {mode} {layout}\n" for name, mode, layout in launch))
    logger.info("seq=%d gbs=%d: launching %s", args.seq, args.gbs, [name for name, _, _ in launch] or "nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
