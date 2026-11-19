# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU collation and loss scaling for fixed, unequal multiplexing microbatches."""

from collections.abc import Callable
from typing import Any

import torch


def collate_microbatches(samples: list, *, sizes: tuple[int, ...], collate: Callable[[list], dict]) -> list[dict]:
    """Split raw samples in order before collating each multimodal microbatch."""
    if len(samples) != sum(sizes):
        raise ValueError(f"Expected {sum(sizes)} samples for microbatch sizes {sizes}, got {len(samples)}")
    batches = []
    start = 0
    for size in sizes:
        batches.append(collate(samples[start : start + size]))
        start += size
    return batches


def scale_microbatch_loss(
    output: torch.Tensor, *, loss_func: Callable, count: int, step_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Keep the loss reports and compensate for Megatron's per-microbatch normalization."""
    loss_sum, tokens, report = loss_func(output)
    # The no-pipeline schedule next divides by max(tokens, 1) and count. The
    # resulting loss is the microbatch sum divided by the entire step's token count.
    scaled = loss_sum * (tokens.clamp(min=1) * (count / step_tokens))
    return scaled, tokens, report
