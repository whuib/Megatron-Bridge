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

"""Keep benchmark runs on shared GPUs free of foreign work, using ``nvidia-smi pmon``.

``wait-quiet`` blocks until no process has used more than ``--threshold`` percent SM on any
selected GPU for ``--window`` seconds (exit 0), or ``--max-wait`` passes (exit 1).

``monitor`` samples per-process SM utilization until ``--root-pid`` exits. Processes that do
not descend from ``--root-pid`` are foreign. It writes a JSON summary with each GPU's mean
foreign SM percent over the run (foreign SM-seconds / run seconds), the largest single foreign
sample, and the foreign processes seen, so a summarizer can reject contaminated runs.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path


logger = logging.getLogger(__name__)


def pmon_samples(interval: int) -> Iterator[tuple[float, int, int, int, int, str]]:
    """Yield ``(time, gpu, pid, sm_percent, fb_mib, command)`` for every active process row."""
    process = subprocess.Popen(
        ["nvidia-smi", "pmon", "-s", "um", "-d", str(interval)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    columns: list[str] = []
    try:
        for line in process.stdout:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "#":
                if not columns:
                    columns = fields[1:]
                continue
            if not columns or len(fields) < len(columns) or not fields[0].isdigit():
                continue
            row = dict(zip(columns, fields))
            if not row["pid"].isdigit():
                # An idle GPU still yields a row so that waiting loops observe the passing time.
                yield time.time(), int(row["gpu"]), 0, 0, 0, "-"
                continue
            sm = int(row["sm"]) if row.get("sm", "-").isdigit() else 0
            fb = int(row["fb"]) if row.get("fb", "-").isdigit() else 0
            yield time.time(), int(row["gpu"]), int(row["pid"]), sm, fb, row.get("command", "?")
    finally:
        process.kill()
        process.wait()


def _parent(pid: int) -> int:
    try:
        # The command name may contain spaces; fields after the closing parenthesis are fixed.
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return 0


def descends_from(pid: int, root: int) -> bool:
    """Whether ``pid`` is ``root`` or one of its descendants."""
    for _ in range(64):
        if pid == root:
            return True
        if pid <= 1:
            return False
        pid = _parent(pid)
    return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_quiet(*, gpus: set[int] | None, threshold: int, window: float, max_wait: float, interval: int) -> bool:
    """Return True once every selected GPU stayed under ``threshold`` SM for ``window`` seconds."""
    start = last_busy = time.time()
    for now, gpu, pid, sm, _, command in pmon_samples(interval):
        if (gpus is None or gpu in gpus) and sm > threshold:
            if now - last_busy > interval:
                logger.info("busy: gpu=%d pid=%d sm=%d%% command=%s", gpu, pid, sm, command)
            last_busy = now
        if now - last_busy >= window:
            return True
        if now - start >= max_wait:
            return False
    return False


def monitor(*, root_pid: int, interval: int, threshold: int) -> dict:
    """Account foreign SM time per GPU while ``root_pid`` is alive."""
    own: dict[int, bool] = {}
    foreign_sm_seconds: dict[int, float] = {}
    foreign_max: dict[int, int] = {}
    busy_samples: dict[int, int] = {}
    foreign_processes: dict[str, dict] = {}
    start = time.time()
    last_check = start
    for now, gpu, pid, sm, fb, command in pmon_samples(interval):
        if pid == 0:
            pass
        elif pid not in own:
            own[pid] = descends_from(pid, root_pid)
        if pid and not own[pid]:
            foreign_sm_seconds[gpu] = foreign_sm_seconds.get(gpu, 0.0) + sm * interval / 100.0
            foreign_max[gpu] = max(foreign_max.get(gpu, 0), sm)
            busy_samples[gpu] = busy_samples.get(gpu, 0) + int(sm > threshold)
            seen = foreign_processes.setdefault(str(pid), {"gpu": gpu, "command": command, "max_sm": 0, "max_fb": 0})
            seen["max_sm"] = max(seen["max_sm"], sm)
            seen["max_fb"] = max(seen["max_fb"], fb)
        if now - last_check >= interval:
            last_check = now
            if not _alive(root_pid):
                break
    duration = max(time.time() - start, 1e-9)
    gpus = sorted(set(foreign_sm_seconds) | set(foreign_max))
    mean = {str(gpu): 100.0 * foreign_sm_seconds.get(gpu, 0.0) / duration for gpu in gpus}
    return {
        "duration_seconds": duration,
        "interval_seconds": interval,
        "foreign_mean_sm_percent": mean,
        "foreign_max_sm_percent": {str(gpu): foreign_max.get(gpu, 0) for gpu in gpus},
        "foreign_busy_samples": {str(gpu): busy_samples.get(gpu, 0) for gpu in gpus},
        "max_foreign_mean_sm_percent": max(mean.values(), default=0.0),
        "foreign_processes": foreign_processes,
    }


_OOM_MESSAGE = re.compile(
    r"total capacity of ([\d.]+) GiB of which ([\d.]+) (MiB|GiB) is free\..*?this process has ([\d.]+) GiB"
)
_TO_MIB = {"MiB": 1.0, "GiB": 1024.0}


def foreign_at_oom_mib(log_text: str) -> float | None:
    """Largest memory (MiB) held by other processes on a failing GPU, from PyTorch's OOM messages.

    PyTorch reports the GPU's total and free memory and this process's usage at the failed allocation;
    the remainder belongs to other processes, including ones it cannot name. None without such a message.
    """
    values = [
        float(total) * 1024 - float(free) * _TO_MIB[unit] - float(own) * 1024
        for total, free, unit, own in _OOM_MESSAGE.findall(log_text)
    ]
    return max(values) if values else None


def oom_foreign_mib(log_path: Path, interference_path: Path) -> int:
    """Foreign GPU memory behind an OOM: from the OOM message, else the monitor's largest foreign process."""
    from_message = foreign_at_oom_mib(log_path.read_text(errors="replace")) if log_path.exists() else None
    if from_message is not None:
        return max(int(from_message), 0)
    if not interference_path.exists():
        return 0
    processes = json.loads(interference_path.read_text()).get("foreign_processes", {})
    return max((int(item["max_fb"]) for item in processes.values()), default=0)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    quiet = sub.add_parser("wait-quiet")
    quiet.add_argument("--gpus", help="comma-separated GPU indices (default: all)")
    quiet.add_argument("--threshold", type=int, default=1, help="SM percent that counts as busy")
    quiet.add_argument("--window", type=float, default=60.0, help="seconds of continuous quiet required")
    quiet.add_argument("--max-wait", type=float, default=7200.0)
    quiet.add_argument("--interval", type=int, default=1)
    watch = sub.add_parser("monitor")
    watch.add_argument("--root-pid", type=int, required=True)
    watch.add_argument("--out", required=True)
    watch.add_argument("--threshold", type=int, default=1)
    watch.add_argument("--interval", type=int, default=1)
    oom = sub.add_parser("oom-foreign", help="print the foreign GPU memory (MiB) behind a run's OOM")
    oom.add_argument("--log", type=Path, required=True)
    oom.add_argument("--interference", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "oom-foreign":
        sys.stdout.write(f"{oom_foreign_mib(args.log, args.interference)}\n")
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    if args.command == "wait-quiet":
        gpus = {int(item) for item in args.gpus.split(",")} if args.gpus else None
        ok = wait_quiet(
            gpus=gpus, threshold=args.threshold, window=args.window, max_wait=args.max_wait, interval=args.interval
        )
        sys.exit(0 if ok else 1)
    record = monitor(root_pid=args.root_pid, interval=args.interval, threshold=args.threshold)
    Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
