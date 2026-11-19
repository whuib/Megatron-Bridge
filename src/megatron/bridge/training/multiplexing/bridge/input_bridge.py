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
"""Redistribute vision inputs from the canonical LLM DP split to Encoder DP.

The training dataloader is sharded by the LLM (global) DP group. Encoder DP
replicas that do not coincide with that split receive their vision tensors
through this input-side Bridge. Gradients are not transferred.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from ..layout import ModuleParallelLayout
from .layout_bridge import LayoutBridge, _broadcast
from .roles import DestRole, SrcRole


def _image_token_counts(grid_thw: Tensor) -> List[int]:
    if grid_thw.numel() == 0:
        return []
    return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(torch.long).tolist()


def _chunk_bounds(n_items: int, n_chunks: int) -> List[Tuple[int, int]]:
    base, rem = divmod(n_items, n_chunks)
    bounds = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def split_vision_by_images(
    vision_data: Tensor, vision_grid_thw: Tensor, n_chunks: int, chunk_id: int
) -> Tuple[Tensor, Tensor]:
    """Split complete images, preserving variable patch counts."""
    n_images = int(vision_grid_thw.shape[0])
    start, end = _chunk_bounds(n_images, n_chunks)[chunk_id]
    sub_grid = vision_grid_thw[start:end]
    counts = _image_token_counts(vision_grid_thw)
    tok_start = sum(counts[:start])
    tok_end = sum(counts[:end])
    return vision_data[tok_start:tok_end], sub_grid


def cat_vision(parts: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """Concatenate image shards in logical DP order."""
    data_parts = [item[0] for item in parts]
    grid_parts = [item[1] for item in parts]
    if data_parts:
        data = torch.cat(data_parts, dim=0)
    else:
        data = torch.zeros((0, 1))
    nonempty = [grid for grid in grid_parts if grid.numel() > 0]
    if nonempty:
        grid = torch.cat(nonempty, dim=0)
    else:
        grid = torch.zeros((0, 3), dtype=torch.long, device=data.device)
    return data, grid


class InputPartitioner:
    """LLM-DP vision shards → Encoder-DP vision shards.

    ``src_layout`` is the LLM layout (canonical batch). ``dest_layout`` is the
    Encoder layout. When ``DP_E == DP_L`` and rank mappings match this is a
    zero-communication identity.
    """

    def __init__(
        self,
        src_layout: ModuleParallelLayout,
        dest_layout: ModuleParallelLayout,
        ctx=None,
        *,
        device=None,
    ) -> None:
        self.src_layout = src_layout
        self.dest_layout = dest_layout
        self.ctx = ctx
        self._identity = src_layout.dp_size == dest_layout.dp_size and all(
            src_layout.local_rank("dp", rank) == dest_layout.local_rank("dp", rank)
            for rank in range(src_layout.world_size)
        )
        self._bridge: Optional[LayoutBridge] = None
        # An identity partition keeps every rank's own shard and issues no collective.
        self.needs_communication = not self._identity
        if not self._identity:
            if src_layout.dp_size % dest_layout.dp_size != 0 and dest_layout.dp_size % src_layout.dp_size != 0:
                raise ValueError(
                    f"DP_E={dest_layout.dp_size} and DP_L={src_layout.dp_size} must "
                    "divide evenly (整除); non-divisible DP pairing is Future Work"
                )
            self._bridge = LayoutBridge(
                src_layout,
                dest_layout,
                ctx,
                hidden_size=1,
                n_deepstack=0,
                dtype=torch.float32,
                device=device,
            )

    def scatter_vision_inputs(self, vision_data: Tensor, vision_grid_thw: Tensor) -> Tuple[Tensor, Tensor]:
        if self._identity:
            return vision_data, vision_grid_thw
        return self._scatter(vision_data, vision_grid_thw)

    def _scatter(self, vision_data: Tensor, vision_grid_thw: Tensor) -> Tuple[Tensor, Tensor]:
        if not dist.is_initialized():
            raise RuntimeError("InputPartitioner scatter requires torch.distributed")
        bridge = self._bridge
        assert bridge is not None
        me = dist.get_rank()
        info = bridge.info
        device = vision_data.device
        feat = int(vision_data.shape[-1]) if vision_data.ndim == 2 and vision_data.numel() else 1
        dtype = vision_data.dtype
        grid_dtype = vision_grid_thw.dtype

        chunks: List[Tuple[Tensor, Tensor]] = []
        if info.src_role == SrcRole.BOUNDARY_LEADER:
            n_targets = max(len(info.send_targets), 1)
            if n_targets == 1:
                chunks = [(vision_data, vision_grid_thw)]
            else:
                chunks = [split_vision_by_images(vision_data, vision_grid_thw, n_targets, i) for i in range(n_targets)]

        dest_parts: Dict[int, Tuple[Tensor, Tensor]] = {}
        if info.dest_role == DestRole.BOUNDARY_LEADER and info.src_role == SrcRole.BOUNDARY_LEADER:
            for src in info.recv_sources:
                if src != me:
                    continue
                chunk_id = info.send_targets.index(me)
                dest_parts[src] = chunks[chunk_id]

        def _send_pair(data: Tensor, grid: Tensor, dest: int, src: int) -> None:
            pg = bridge.comm_map[src].p2p_pgs[dest]
            meta = torch.tensor(
                [int(data.shape[0]), int(grid.shape[0]), int(data.shape[-1] if data.ndim == 2 else feat)],
                dtype=torch.int64,
                device=device,
            )
            _broadcast(meta, pg, root_global=src)
            if data.numel() > 0:
                _broadcast(data.contiguous(), pg, root_global=src)
            if grid.numel() > 0:
                _broadcast(grid.contiguous(), pg, root_global=src)

        def _recv_pair(src: int, dest: int) -> Tuple[Tensor, Tensor]:
            pg = bridge.comm_map[dest].p2p_pgs[src]
            meta = torch.empty(3, dtype=torch.int64, device=device)
            _broadcast(meta, pg, root_global=src)
            n_tokens, n_images, feat_dim = (int(v) for v in meta.tolist())
            feat_dim = feat_dim or feat
            data = torch.empty((n_tokens, feat_dim), device=device, dtype=dtype)
            grid = torch.empty((n_images, 3), device=device, dtype=grid_dtype)
            if n_tokens > 0:
                _broadcast(data, pg, root_global=src)
            if n_images > 0:
                _broadcast(grid, pg, root_global=src)
            return data, grid

        for src, dest in bridge.plan.p2p_pairs:
            if me == src:
                chunk_id = info.send_targets.index(dest)
                data, grid = chunks[chunk_id]
                _send_pair(data, grid, dest, src)
            elif me == dest:
                dest_parts[src] = _recv_pair(src, dest)

        if info.dest_role == DestRole.BOUNDARY_LEADER:
            ordered = [dest_parts[src] for src in info.recv_sources]
            data, grid = cat_vision(ordered)
        else:
            data = torch.empty((0, feat), device=device, dtype=dtype)
            grid = torch.empty((0, 3), device=device, dtype=grid_dtype)

        dest_pg = info.dest_broadcast_pg
        dest_ranks = info.dest_broadcast_ranks
        if dest_ranks:
            leader = info.dest_leader_rank
            meta = torch.tensor(
                [int(data.shape[0]), int(grid.shape[0]), int(data.shape[-1] if data.ndim == 2 else feat)],
                dtype=torch.int64,
                device=device,
            )
            if dest_pg is not None:
                _broadcast(meta, dest_pg, root_global=leader)
                n_tokens, n_images, feat_dim = (int(v) for v in meta.tolist())
                if info.dest_role != DestRole.BOUNDARY_LEADER:
                    data = torch.empty((n_tokens, feat_dim), device=device, dtype=dtype)
                    grid = torch.empty((n_images, 3), device=device, dtype=grid_dtype)
                if n_tokens > 0:
                    data = data.contiguous()
                    _broadcast(data, dest_pg, root_global=leader)
                if n_images > 0:
                    grid = grid.contiguous()
                    _broadcast(grid, dest_pg, root_global=leader)
            return data, grid
        if info.dest_role == DestRole.NON_BOUNDARY:
            return vision_data, vision_grid_thw
        return data, grid
