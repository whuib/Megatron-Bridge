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

"""Mode-agnostic steady-state throughput measurement for the Qwen3-VL comparison.

The same callback measures native, DistTrain and multiplexing training. A steady window
opens after ``warmup_iters`` completed steps and closes ``measure_iters`` steps later; both
boundaries synchronize the device and then every rank with a global barrier. No other
synchronization is added inside the window, so the measured steps run as in a normal job.
"""

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist

from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.utils.flop_utils import num_floating_point_operations, vit_flops
from megatron.bridge.utils.common_utils import print_rank_0


@dataclass(frozen=True)
class Workload:
    """Per-iteration work, identical for every execution mode of one comparison."""

    global_batch_size: int
    seq_length: int
    images_per_sample: int
    patches_per_image: int
    audios_per_sample: int = 0
    audio_frames_per_clip: int = 0


def flops_per_iteration(cfg: Any, workload: Workload) -> int:
    """Model training FLOPs for one global batch with Bridge's estimators.

    Language FLOPs use the padded sequence length (every sample is padded to it). ViT FLOPs
    use per-image attention over ``patches_per_image`` patches. Native training logs these
    two estimators too, so the meter's TFLOPS/GPU is directly comparable with its logs.
    """
    language = num_floating_point_operations(cfg, batch_size=workload.global_batch_size)
    vision = vit_flops(
        cfg,
        batch_size=workload.global_batch_size * workload.images_per_sample,
        num_patches=workload.patches_per_image,
    )
    return int(language) + int(vision)


def window_summary(
    *,
    window_seconds: float,
    measure_iters: int,
    workload: Workload,
    flops_per_iter: int,
    world_size: int,
    peak_tflops_per_gpu: float | None = None,
) -> dict[str, float]:
    """Throughput metrics for a steady window of ``measure_iters`` complete iterations.

    ``mfu`` (model FLOPs utilization) is reported when the per-GPU dense peak is given.
    """
    if window_seconds <= 0 or measure_iters <= 0 or world_size <= 0:
        raise ValueError("window_seconds, measure_iters and world_size must be positive")
    step_seconds = window_seconds / measure_iters
    tflops = flops_per_iter / step_seconds / world_size / 1e12
    summary = {
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "samples_per_second": workload.global_batch_size / step_seconds,
        "padded_tokens_per_second": workload.global_batch_size * workload.seq_length / step_seconds,
        "images_per_second": workload.global_batch_size * workload.images_per_sample / step_seconds,
        "audio_clips_per_second": workload.global_batch_size * workload.audios_per_sample / step_seconds,
        "model_tflops_per_gpu": tflops,
    }
    if peak_tflops_per_gpu is not None:
        summary["peak_tflops_per_gpu"] = peak_tflops_per_gpu
        summary["mfu"] = tflops / peak_tflops_per_gpu
    return summary


class ThroughputMeter(Callback):
    """Measure one steady window and write a JSON record on global rank 0."""

    def __init__(
        self,
        *,
        warmup_iters: int,
        measure_iters: int,
        workload: Workload,
        results_path: str | None,
        metadata: dict[str, Any],
        flops_fn: Callable[[Any, Workload], int] = flops_per_iteration,
        peak_tflops_per_gpu: float | None = None,
    ) -> None:
        if warmup_iters < 1 or measure_iters < 1:
            raise ValueError("warmup_iters and measure_iters must be at least 1")
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.workload = workload
        self.results_path = results_path
        self.metadata = metadata
        self.flops_fn = flops_fn
        self.peak_tflops_per_gpu = peak_tflops_per_gpu
        self.completed = 0
        self.flops_per_iter: int | None = None
        self._window_start: float | None = None
        self._last_step_end: float | None = None
        self._step_seconds_rank0: list[float] = []
        self._finite = True
        self._last_loss: float | None = None
        self._last_grad_norm: float | None = None

    @staticmethod
    def _synchronized_now() -> float:
        torch.cuda.synchronize()
        dist.barrier()
        return time.perf_counter()

    def on_train_start(self, context: CallbackContext) -> None:
        """Compute the per-iteration FLOPs from the finalized runtime config."""
        self.flops_per_iter = int(self.flops_fn(context.state.cfg, self.workload))

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Advance the step count and open or close the steady window."""
        self.completed += 1
        in_window = self.warmup_iters < self.completed <= self.warmup_iters + self.measure_iters
        if in_window:
            self._observe(context)
        if self.completed == self.warmup_iters:
            self._window_start = self._synchronized_now()
            self._last_step_end = self._window_start
        elif self.completed == self.warmup_iters + self.measure_iters:
            self._finish(self._synchronized_now())

    def on_train_end(self, context: CallbackContext) -> None:
        """Refuse to report silently when training ended before the window closed."""
        if self.completed < self.warmup_iters + self.measure_iters:
            raise RuntimeError(
                f"training ran {self.completed} steps; the throughput window needs "
                f"{self.warmup_iters} warmup + {self.measure_iters} measured steps"
            )

    def _observe(self, context: CallbackContext) -> None:
        # Rank-local, unsynchronized step-to-step times; used only for jitter, not throughput.
        now = time.perf_counter()
        if self._last_step_end is not None:
            self._step_seconds_rank0.append(now - self._last_step_end)
        self._last_step_end = now
        for value in (context.loss_dict or {}).values():
            tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
            if tensor.numel() == 2:  # [loss_sum, num_tokens] reporting format
                tensor = tensor[0] / tensor[1].clamp(min=1)
            self._finite = self._finite and bool(torch.isfinite(tensor.float()).all())
            self._last_loss = float(tensor.float().mean())
        if context.grad_norm is not None:
            self._finite = self._finite and math.isfinite(context.grad_norm)
            self._last_grad_norm = float(context.grad_norm)
        if context.skipped_iter:
            self._finite = False

    def _finish(self, window_end: float) -> None:
        if self._window_start is None or self.flops_per_iter is None:
            raise RuntimeError("throughput window closed before it was opened")
        world_size = dist.get_world_size()
        free, total = torch.cuda.mem_get_info()
        local = torch.tensor(
            [
                float(torch.cuda.max_memory_allocated()),
                float(torch.cuda.max_memory_reserved()),
                float(self._finite),
                # Device memory in use at the end of the window, and the part outside PyTorch's caching
                # allocator (CUDA context, NCCL communicator buffers; other processes on the GPU count too).
                float(total - free),
                float(total - free - torch.cuda.memory_reserved()),
            ],
            device="cuda",
            dtype=torch.float64,
        )
        gathered = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        if dist.get_rank() != 0:
            return
        per_rank = torch.stack(gathered).cpu()
        steps = sorted(self._step_seconds_rank0)
        record = {
            "metadata": self.metadata,
            "workload": asdict(self.workload),
            "warmup_iters": self.warmup_iters,
            "measure_iters": self.measure_iters,
            "world_size": world_size,
            "flops_per_iteration": self.flops_per_iter,
            **window_summary(
                window_seconds=window_end - self._window_start,
                measure_iters=self.measure_iters,
                workload=self.workload,
                flops_per_iter=self.flops_per_iter,
                world_size=world_size,
                peak_tflops_per_gpu=self.peak_tflops_per_gpu,
            ),
            "rank0_step_seconds_unsynchronized": {
                "min": steps[0] if steps else None,
                "median": steps[len(steps) // 2] if steps else None,
                "max": steps[-1] if steps else None,
            },
            "peak_allocated_gib_per_rank": [round(v / 2**30, 3) for v in per_rank[:, 0].tolist()],
            "peak_reserved_gib_per_rank": [round(v / 2**30, 3) for v in per_rank[:, 1].tolist()],
            "max_peak_allocated_gib": round(float(per_rank[:, 0].max()) / 2**30, 3),
            "max_peak_reserved_gib": round(float(per_rank[:, 1].max()) / 2**30, 3),
            "device_used_gib_per_rank": [round(v / 2**30, 3) for v in per_rank[:, 3].tolist()],
            "outside_pytorch_gib_per_rank": [round(v / 2**30, 3) for v in per_rank[:, 4].tolist()],
            "loss_and_grad_finite_on_all_ranks": bool(per_rank[:, 2].min() == 1.0),
            "last_reported_loss_rank0": self._last_loss,
            "last_reported_grad_norm_rank0": self._last_grad_norm,
        }
        print_rank_0(
            "THROUGHPUT mode={mode} step_seconds={step_seconds:.4f} samples_per_second={samples_per_second:.3f} "
            "model_tflops_per_gpu={model_tflops_per_gpu:.2f} mfu={mfu} max_peak_allocated_gib={max_peak_allocated_gib} "
            "finite={loss_and_grad_finite_on_all_ranks}".format(
                **{**record, "mode": self.metadata.get("mode"), "mfu": round(record.get("mfu", float("nan")), 4)}
            )
        )
        if self.results_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.results_path)), exist_ok=True)
            with open(self.results_path, "w") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
