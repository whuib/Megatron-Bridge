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

# Adapted from Pai-Megatron-Patch multiplexing (Qwen3-VL transport).
"""Boundary tensor contract between Encoder and LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


# vision_embeds + 3 DeepStack features share one shape negotiation.
N_DEEPSTACK = 3


@dataclass
class BoundarySpec:
    """Describe one tensor that crosses the Encoder/LLM Bridge."""

    name: str
    dtype: torch.dtype
    ndim: int
    dynamic_dims: Tuple[int, ...] = ()
    requires_grad: bool = True


@dataclass
class BoundaryPayload:
    """Encoder → LLM boundary data.

    ``image_input_mask`` / ``video_input_mask`` / ``input_ids`` stay on the LLM
    side and do not travel with this payload.
    """

    vision_embeds: Optional[Tensor]
    deepstack_features: Optional[List[Tensor]]
    video_start_index: int = -1
    n_merged_tokens: int = 0

    def tensors(self) -> List[Tensor]:
        items: List[Tensor] = []
        if self.vision_embeds is not None:
            items.append(self.vision_embeds)
        if self.deepstack_features:
            items.extend(self.deepstack_features)
        return items

    def grad_tensors(self) -> List[Optional[Tensor]]:
        return [tensor.grad for tensor in self.tensors()]

    def has_vision(self) -> bool:
        return self.vision_embeds is not None and int(self.n_merged_tokens) > 0

    def n_deepstack(self) -> int:
        return 0 if not self.deepstack_features else len(self.deepstack_features)

    def detached_tensors(self) -> List[Tensor]:
        return [tensor.detach().contiguous() for tensor in self.tensors()]

    @classmethod
    def empty(cls, video_start_index: int = -1, n_deepstack: int = N_DEEPSTACK) -> "BoundaryPayload":
        del n_deepstack
        return cls(
            vision_embeds=None,
            deepstack_features=None,
            video_start_index=video_start_index,
            n_merged_tokens=0,
        )

    @classmethod
    def from_tensor_list(
        cls,
        tensors: List[Tensor],
        video_start_index: int,
        n_merged_tokens: int,
    ) -> "BoundaryPayload":
        if not tensors:
            return cls.empty(video_start_index=video_start_index)
        vision, *deepstack = tensors
        return cls(
            vision_embeds=vision,
            deepstack_features=list(deepstack) if deepstack else None,
            video_start_index=video_start_index,
            n_merged_tokens=n_merged_tokens,
        )

    @classmethod
    def from_encoder_output(cls, out, video_start_index: int = -1) -> "BoundaryPayload":
        vision_embeds = out.vision_embeds
        n_merged = 0 if vision_embeds is None else int(vision_embeds.shape[0])
        return cls(
            vision_embeds=vision_embeds,
            deepstack_features=out.deepstack_feature_lists,
            video_start_index=video_start_index,
            n_merged_tokens=n_merged,
        )
