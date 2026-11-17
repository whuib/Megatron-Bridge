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
"""Fixed-shape metadata handshake that precedes Bridge tensor traffic.

``n_merged_tokens`` varies per micro-batch. Destination ranks allocate receive
buffers only after this exchange. Metadata is always a length-4 int64 vector
so the handshake itself needs no shape negotiation.

When ``--multiplexing-static-shape`` is set the caller supplies the same
4-tuple on every rank and this step is skipped.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


META_N_TOKENS = 0
META_VIDEO_START = 1
META_HAS_VISION = 2
META_N_DEEPSTACK = 3
META_SIZE = 4

ShapeMeta = Tuple[int, int, int, int]


class ShapeNegotiator:
    """Pack / unpack the ``[n_merged_tokens, video_start_index, has_vision, n_deepstack]`` vector."""

    def pack(
        self,
        n_merged_tokens: int,
        video_start_index: int,
        has_vision: int,
        n_deepstack: int,
        *,
        device,
        dtype=torch.int64,
    ) -> Tensor:
        return torch.tensor(
            [int(n_merged_tokens), int(video_start_index), int(has_vision), int(n_deepstack)],
            dtype=dtype,
            device=device,
        )

    def unpack(self, meta: Tensor) -> ShapeMeta:
        values = meta.detach().to("cpu").reshape(-1).tolist()
        if len(values) != META_SIZE:
            raise ValueError(f"shape metadata must have {META_SIZE} entries, got {len(values)}")
        return tuple(int(v) for v in values)  # type: ignore[return-value]

    def allocate(
        self,
        n_merged_tokens: int,
        hidden_size: int,
        n_deepstack: int,
        *,
        device,
        dtype,
        requires_grad: bool = True,
    ) -> list:
        """Allocate ``1 + n_deepstack`` tensors of shape ``[n_merged_tokens, hidden_size]``."""
        tensors = []
        for _ in range(1 + int(n_deepstack)):
            tensor = torch.empty(
                (int(n_merged_tokens), int(hidden_size)),
                device=device,
                dtype=dtype,
            )
            if requires_grad and n_merged_tokens > 0:
                tensor.requires_grad_(True)
            tensors.append(tensor)
        return tensors


def static_meta_or_none(value) -> Optional[ShapeMeta]:
    """Parse an optional static boundary shape."""
    if value in (None, False):
        return None
    if value is True:
        raise ValueError(
            "--multiplexing-static-shape requires an explicit "
            "(n_merged_tokens, video_start_index, has_vision, n_deepstack) tuple"
        )
    if len(value) != META_SIZE:
        raise ValueError(f"static shape metadata must have {META_SIZE} entries, got {value!r}")
    return tuple(int(v) for v in value)  # type: ignore[return-value]
