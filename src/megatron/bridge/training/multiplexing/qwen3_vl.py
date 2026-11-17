# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL adapters; native and DistTrain model code is shared unchanged."""

from collections import deque
from dataclasses import dataclass

import torch
from torch import nn

from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider

from .bridge.payload import BoundaryPayload


class BoundaryCache(nn.Module):
    """Supply already encoded features at the native model's vision boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.payloads: deque[BoundaryPayload] = deque()

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        """Return the next activation in language microbatch order."""
        if not self.payloads:
            raise RuntimeError("multiplexing vision boundary has no queued microbatch")
        payload = self.payloads.popleft()
        return payload.vision_embeds, payload.deepstack_features


@dataclass
class MultiplexingEncoderProvider(Qwen3VLModelProvider):
    """Build the repository's vision encoder with its own process groups."""

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> nn.Module:
        """Use the same encoder implementation and weights as native Qwen3-VL."""
        model = super().provide(pre_process=True, post_process=True, vp_stage=None)
        model.vision_model.share_embeddings_and_output_weights = False
        # In native Qwen the outer language module supplies the precision wrapper.
        # Standalone vision DDP must expose that same policy on its own config.
        model.vision_model.config.bf16 = self.bf16
        model.vision_model.config.fp16 = self.fp16
        model.vision_model.config.params_dtype = self.params_dtype
        return model.vision_model


def install_boundary_cache(models: list[nn.Module]) -> list[nn.Module]:
    """Install the parameter-free boundary before language DDP is constructed."""
    for model in models:
        if model.pre_process:
            model.vision_model = BoundaryCache()
    return models
