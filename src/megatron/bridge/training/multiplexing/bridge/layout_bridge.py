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
"""Layout-aware activation bridge.

``IdentityBridge`` is the Phase 3 same-layout passthrough. ``LayoutBridge``
moves Encoder activations to the LLM and gradients back, fully aware of
TP / PP / DP. Every rank may be a source and a destination at once.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from ..config import pg_timeout
from ..layout import ModuleParallelLayout
from .payload import N_DEEPSTACK, BoundaryPayload
from .roles import (
    BridgeCommPlan,
    DestRole,
    RankBridgeInfo,
    SrcRole,
    build_bridge_comm_plan,
)
from .shape_negotiation import ShapeNegotiator


logger = logging.getLogger(__name__)

_FROZEN_PG_MESSAGE = (
    "process groups are frozen after LayoutBridge startup; runtime must not create or destroy communicators"
)

_NEW_GROUP_HAS_DESC = "group_desc" in inspect.signature(dist.new_group).parameters

_FWD_PHASES = ("shape_p2p", "shape_bcast", "data_p2p", "data_bcast")
_BWD_PHASES = ("grad_allreduce", "grad_p2p", "grad_bcast")


def _new_group(ranks: Sequence[int], group_desc: Optional[str] = None) -> dist.ProcessGroup:
    kwargs = {}
    if group_desc is not None and _NEW_GROUP_HAS_DESC:
        kwargs["group_desc"] = group_desc
    return dist.new_group(list(ranks), timeout=pg_timeout(), **kwargs)


def _pg_ranks(pg: dist.ProcessGroup) -> Tuple[int, ...]:
    if hasattr(dist, "get_process_group_ranks"):
        return tuple(dist.get_process_group_ranks(pg))
    return tuple()


def _broadcast(tensor: Tensor, pg: dist.ProcessGroup, root_global: int) -> Tensor:
    """Broadcast ``tensor`` from ``root_global`` (a *global* rank).

    PyTorch 2.x ``dist.broadcast(..., src=)`` takes the global rank, not the
    in-group index. Passing a local index makes NCCL/Gloo look for the wrong
    root (``Global rank X is not part of group``) or hang.
    """
    tensor = tensor.contiguous()
    if dist.get_backend() == "nccl" and tensor.dtype == torch.int64:
        # Some NCCL builds mishandle int64; float32 is universally supported.
        tmp = tensor.to(torch.float32)
        dist.broadcast(tmp, src=int(root_global), group=pg)
        tensor.copy_(tmp.to(torch.int64))
        return tensor
    dist.broadcast(tensor, src=int(root_global), group=pg)
    return tensor


def _comm_device():
    if dist.is_initialized() and dist.get_backend() == "nccl" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _pack_boundary(tensors: Optional[Sequence[Tensor]]) -> Optional[Tensor]:
    """Stack equal-shaped boundary tensors for one collective transfer."""
    if not tensors:
        return None
    shape = tuple(tensors[0].shape)
    if any(tuple(tensor.shape) != shape for tensor in tensors):
        raise ValueError("boundary tensors must have identical shapes")
    return torch.stack([tensor.contiguous() for tensor in tensors], dim=0)


def _unpack_boundary(packed: Optional[Tensor], *, requires_grad: bool = False) -> List[Tensor]:
    """Restore independently differentiable tensors from a packed boundary."""
    if packed is None:
        return []
    return [item.detach().clone().requires_grad_(requires_grad) for item in packed.unbind(0)]


class IdentityBridge:
    """Same-layout passthrough used to prove the Phase 3 split did not change math.

    Layouts must match on every parallel size and dimension order. Heterogeneous
    TP/PP/DP is ``LayoutBridge``. Pipeline hops inside a module are handled by
    the module PP schedule, not by this bridge.
    """

    def __init__(
        self,
        src_layout: Optional[ModuleParallelLayout] = None,
        dest_layout: Optional[ModuleParallelLayout] = None,
    ) -> None:
        if src_layout is not None and dest_layout is not None:
            src_sizes = (
                src_layout.tp_size,
                src_layout.cp_size,
                src_layout.pp_size,
                src_layout.dp_size,
            )
            dest_sizes = (
                dest_layout.tp_size,
                dest_layout.cp_size,
                dest_layout.pp_size,
                dest_layout.dp_size,
            )
            if src_sizes != dest_sizes:
                raise ValueError(f"IdentityBridge requires identical layouts; src={src_sizes} dest={dest_sizes}")
            if src_layout.dimension_order != dest_layout.dimension_order:
                raise ValueError("IdentityBridge requires identical dimension_order")
        self.src_layout = src_layout
        self.dest_layout = dest_layout
        self._saved_payload: Optional[BoundaryPayload] = None
        self._saved_grads: Optional[List[Optional[Tensor]]] = None
        self.phase_log: List[str] = []

    def send_forward(self, payload: BoundaryPayload) -> BoundaryPayload:
        return payload

    def recv_forward(self, payload: Optional[BoundaryPayload] = None) -> Optional[BoundaryPayload]:
        return payload

    def forward(self, payload: BoundaryPayload) -> BoundaryPayload:
        return payload

    def forward_send(self, payload: Optional[BoundaryPayload]) -> None:
        self._saved_payload = payload

    def forward_recv(self) -> Optional[BoundaryPayload]:
        return self._saved_payload

    def backward_send(self, grads: Optional[List[Optional[Tensor]]]) -> None:
        self._saved_grads = grads

    def backward_recv(self) -> List[Optional[Tensor]]:
        if self._saved_grads is None:
            return []
        return self._saved_grads

    def needs_cross_grid_sync(self) -> bool:
        """Never: this bridge issues no collectives, so no communicator switch."""
        return False

    def describe(self) -> str:
        return "IdentityBridge"


class LayoutBridge:
    """Encoder → LLM activations and LLM → Encoder gradients.

    Every rank must call ``forward_send`` then ``forward_recv`` (and the
    backward pair) so the four communication phases stay deadlock-free.
    Source leaders pass a payload; every other rank may pass ``None``.
    """

    def __init__(
        self,
        src_layout: ModuleParallelLayout,
        dest_layout: ModuleParallelLayout,
        ctx=None,
        *,
        hidden_size: Optional[int] = None,
        n_deepstack: int = N_DEEPSTACK,
        dtype: torch.dtype = torch.float32,
        static_shape=None,
        barrier_debug: Optional[bool] = None,
        device=None,
        replicated_destination_grads: bool = False,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("LayoutBridge requires torch.distributed to be initialized")
        self.src_layout = src_layout
        self.dest_layout = dest_layout
        self.ctx = ctx
        self.replicated_destination_grads = replicated_destination_grads
        self.hidden_size = hidden_size
        self.n_deepstack = int(n_deepstack)
        self.dtype = dtype
        self.device = device if device is not None else _comm_device()
        args = None if ctx is None else getattr(ctx, "_args", None)
        self.static_shape = static_shape
        if self.static_shape is None and args is not None and getattr(args, "multiplexing_static_shape", False):
            self.static_shape = True
        self.barrier_debug = (
            bool(getattr(args, "multiplexing_barrier_debug", False)) if barrier_debug is None else bool(barrier_debug)
        )
        self.negotiator = ShapeNegotiator()
        self.rank = dist.get_rank()
        self.plan: BridgeCommPlan = build_bridge_comm_plan(src_layout, dest_layout)
        self.comm_map: Dict[int, RankBridgeInfo] = self.plan.comm_map
        self.info: RankBridgeInfo = self.comm_map[self.rank]
        self._owned_pgs: List[dist.ProcessGroup] = []
        self._pg_by_ranks: Dict[Tuple[int, ...], dist.ProcessGroup] = {}
        self._pg_frozen = False
        self._create_all_pgs()
        self._pg_frozen = True

        self._saved_payload: Optional[BoundaryPayload] = None
        self._recv_payload: Optional[BoundaryPayload] = None
        self._recv_grads: List[Optional[Tensor]] = []
        self._split_sizes: List[int] = []
        self._src_metas: List[Tuple[int, int, int, int]] = []
        self._forward_done = False
        self._backward_done = False
        self.phase_log: List[str] = []
        self._cross_grid = not self._pure_self_loop()

    # ------------------------------------------------------------------ PGs
    def _create_all_pgs(self) -> None:
        """Every rank walks the same sequence. Size-1 groups are skipped.

        P2P groups are never aliased onto dest/src broadcast groups even when
        the rank sets coincide. Overlapping two NCCL ops on one communicator
        deadlocks with ``CUDA_DEVICE_MAX_CONNECTIONS=1``.
        """
        world_ranks = tuple(range(dist.get_world_size()))
        for group in self.plan.dest_broadcast_groups:
            if tuple(group) == world_ranks:
                self._pg_by_ranks[("dest", tuple(group))] = dist.group.WORLD
            else:
                self._maybe_create("dest", group, f"mx.bridge.dest_bcast.{group[0]}")
        for group in self.plan.src_broadcast_groups:
            if tuple(group) == world_ranks:
                self._pg_by_ranks[("src", tuple(group))] = dist.group.WORLD
            else:
                self._maybe_create("src", group, f"mx.bridge.src_bcast.{group[0]}")
        for src, dest in self.plan.p2p_pairs:
            pair = tuple(sorted((src, dest)))
            self._maybe_create("p2p", pair, f"mx.bridge.p2p.{src}.{dest}")

        for rank, info in self.comm_map.items():
            if info.dest_broadcast_ranks and len(info.dest_broadcast_ranks) > 1:
                info.dest_broadcast_pg = self._pg_lookup("dest", info.dest_broadcast_ranks)
                if info.dest_broadcast_pg is None:
                    raise RuntimeError(
                        f"rank {rank}: dest broadcast group {info.dest_broadcast_ranks} was not created"
                    )
            if info.src_broadcast_ranks and len(info.src_broadcast_ranks) > 1:
                info.src_broadcast_pg = self._pg_lookup("src", info.src_broadcast_ranks)
                if info.src_broadcast_pg is None:
                    raise RuntimeError(f"rank {rank}: src broadcast group {info.src_broadcast_ranks} was not created")
            if info.src_role == SrcRole.BOUNDARY_LEADER:
                for dest in info.send_targets:
                    if dest == rank:
                        continue
                    pair = tuple(sorted((rank, dest)))
                    pg = self._pg_lookup("p2p", pair)
                    if pg is None:
                        raise RuntimeError(f"rank {rank}: missing P2P group to dest {dest}")
                    info.p2p_pgs[dest] = pg
            if info.dest_role == DestRole.BOUNDARY_LEADER:
                for src in info.recv_sources:
                    if src == rank:
                        continue
                    pair = tuple(sorted((src, rank)))
                    pg = self._pg_lookup("p2p", pair)
                    if pg is None:
                        raise RuntimeError(f"rank {rank}: missing P2P group from src {src}")
                    info.p2p_pgs[src] = pg
        self.info = self.comm_map[self.rank]

    def _pg_lookup(self, kind: str, ranks: Sequence[int]) -> Optional[dist.ProcessGroup]:
        return self._pg_by_ranks.get((kind, tuple(ranks)))

    def _maybe_create(self, kind: str, ranks: Sequence[int], desc: str) -> Optional[dist.ProcessGroup]:
        key = (kind, tuple(ranks))
        if len(ranks) <= 1:
            return None
        if key in self._pg_by_ranks:
            return self._pg_by_ranks[key]
        pg = _new_group(tuple(ranks), group_desc=desc)
        self._pg_by_ranks[key] = pg
        self._owned_pgs.append(pg)
        return pg

    def _get_or_create_broadcast_pg(self, *_args, **_kwargs):
        raise RuntimeError(_FROZEN_PG_MESSAGE)

    def _get_or_create_bridge_pg(self, *_args, **_kwargs):
        raise RuntimeError(_FROZEN_PG_MESSAGE)

    def _get_or_create(self, *_args, **_kwargs):
        raise RuntimeError(_FROZEN_PG_MESSAGE)

    # ------------------------------------------------------------------ helpers
    def _phase(self, name: str) -> None:
        self.phase_log.append(name)
        if self.barrier_debug:
            logger.info("[mx-bridge] rank=%s phase=%s", self.rank, name)
        if os.environ.get("MX_DEBUG"):
            logger.debug(
                f"[mx-bridge] rank={self.rank} phase={name} self_loop={self._pure_self_loop()}",
            )
        # Broadcasts are paired by plan.p2p_pairs (every rank derives the same
        # sequence). Cross-communicator ordering lives in runtime._handoff via
        # needs_cross_grid_sync -- a local device drain, not a WORLD barrier.

    def _pure_self_loop(self) -> bool:
        """Skip all Bridge traffic only when every dest replica is a self-loop.

        Fan-out layouts mix self edges with real P2P (B4). A dest replica that
        only consumes itself must still send to its other targets, so the
        shortcut is allowed only when ``p2p_pairs`` is empty (B1).
        """
        if self.plan.p2p_pairs:
            return False
        if self.info.dest_role == DestRole.NON_BOUNDARY:
            return False
        leader = self.info.dest_leader_rank
        if leader is None:
            return False
        recv = self.comm_map[leader].recv_sources
        return recv == [leader]

    def needs_cross_grid_sync(self) -> bool:
        """Whether this rank actually switches NCCL communicators at the boundary.

        ``_pure_self_loop()`` short-circuits every phase, so the homogeneous
        layout never opens a Bridge communicator and there is nothing to order
        against the surrounding module-TP collectives. Once the layouts differ
        the P2P and broadcast groups are real, and a CUDA event on the default
        stream does not wait for NCCL's internal stream, so the runtime has to
        device-synchronize between the encoder-TP, Bridge and LLM-TP groups.

        Constant for the life of the bridge: it is a property of ``self.plan``,
        which is frozen in ``__init__``.
        """
        return self._cross_grid

    def _hidden_size_from(self, payload: Optional[BoundaryPayload]) -> int:
        if payload is not None and payload.vision_embeds is not None:
            return int(payload.vision_embeds.shape[-1])
        if self.hidden_size is not None:
            return int(self.hidden_size)
        raise RuntimeError("LayoutBridge hidden_size is unknown; pass it to the constructor")

    def _p2p_pg(self, peer: int) -> dist.ProcessGroup:
        pg = self.info.p2p_pgs.get(peer)
        if pg is None:
            raise RuntimeError(f"rank {self.rank} has no P2P group to {peer}")
        return pg

    def describe(self) -> str:
        return (
            f"rank={self.rank} src={self.info.src_role.name} "
            f"dest={self.info.dest_role.name} send={self.info.send_targets} "
            f"recv={self.info.recv_sources}"
        )

    # ------------------------------------------------------------------ forward
    def forward_send(self, payload: Optional[BoundaryPayload]) -> None:
        if payload is not None:
            self._saved_payload = payload
            if payload.vision_embeds is not None:
                self.hidden_size = int(payload.vision_embeds.shape[-1])
                self.dtype = payload.vision_embeds.dtype
                self.n_deepstack = max(self.n_deepstack, payload.n_deepstack())
        self._run_forward_exchange()

    def forward_recv(self) -> Optional[BoundaryPayload]:
        if not self._forward_done:
            self._run_forward_exchange()
        self._forward_done = False
        self._backward_done = False
        return self._recv_payload

    def _run_forward_exchange(self) -> None:
        if self._forward_done:
            return
        self.phase_log = []
        if self._pure_self_loop():
            # No Bridge traffic: WORLD barriers here are redundant and can
            # deadlock after LLM TP collectives (NCCL WORLD vs module TP).
            self.phase_log.extend(["shape_p2p", "shape_bcast", "data_p2p", "data_bcast"])
            payload = self._saved_payload
            if self.info.dest_role == DestRole.NON_BOUNDARY:
                self._recv_payload = None
            else:
                self._recv_payload = payload
                if payload is not None and payload.has_vision():
                    self._split_sizes = [int(payload.n_merged_tokens)]
                    self._src_metas = [
                        (
                            int(payload.n_merged_tokens),
                            int(payload.video_start_index),
                            1,
                            payload.n_deepstack() or self.n_deepstack,
                        )
                    ]
            self._forward_done = True
            return

        self._forward_shape()
        self._forward_data()
        self._forward_done = True

    def _src_meta_tensor(self) -> Tensor:
        payload = self._saved_payload
        if payload is None or not payload.has_vision():
            n_tokens, video_start, has_vision, n_deep = 0, -1, 0, self.n_deepstack
        else:
            n_tokens = int(payload.n_merged_tokens)
            video_start = int(payload.video_start_index)
            has_vision = 1
            n_deep = payload.n_deepstack() or self.n_deepstack
        if self.static_shape not in (None, False, True):
            n_tokens, video_start, has_vision, n_deep = (int(v) for v in self.static_shape)
        return self.negotiator.pack(n_tokens, video_start, has_vision, n_deep, device=self.device)

    def _forward_shape(self) -> None:
        skip = self.static_shape not in (None, False)
        self._phase("shape_p2p")
        leader_stack: Optional[Tensor] = None
        if not skip:
            meta = self._src_meta_tensor() if self.info.src_role == SrcRole.BOUNDARY_LEADER else None
            parts_by_src: Dict[int, Tensor] = {}
            if self.info.dest_role == DestRole.BOUNDARY_LEADER:
                for src in self.info.recv_sources:
                    if src == self.rank:
                        parts_by_src[src] = meta if meta is not None else self._src_meta_tensor()
            for src, dest in self.plan.p2p_pairs:
                if self.rank == src:
                    _broadcast(meta.clone(), self._p2p_pg(dest), root_global=src)
                elif self.rank == dest:
                    buf = torch.empty(4, dtype=torch.int64, device=self.device)
                    _broadcast(buf, self._p2p_pg(src), root_global=src)
                    parts_by_src[src] = buf
            if self.info.dest_role == DestRole.BOUNDARY_LEADER:
                leader_stack = torch.stack([parts_by_src[src] for src in self.info.recv_sources], dim=0)
        else:
            if self.info.dest_role == DestRole.BOUNDARY_LEADER:
                n = max(len(self.info.recv_sources), 1)
                meta = self._src_meta_tensor()
                leader_stack = meta.unsqueeze(0).repeat(n, 1)

        self._phase("shape_bcast")
        dest_pg = self.info.dest_broadcast_pg
        dest_ranks = self.info.dest_broadcast_ranks
        if dest_ranks:
            k = len(self.comm_map[self.info.dest_leader_rank].recv_sources) or 1
            if dest_pg is None:
                stack = leader_stack
            else:
                if self.info.dest_role == DestRole.BOUNDARY_LEADER:
                    stack = leader_stack
                else:
                    stack = torch.empty((k, 4), dtype=torch.int64, device=self.device)
                if stack is None:
                    stack = torch.empty((k, 4), dtype=torch.int64, device=self.device)
                _broadcast(stack, dest_pg, root_global=self.info.dest_leader_rank)
            if stack is not None:
                self._src_metas = [self.negotiator.unpack(row) for row in stack]
                self._split_sizes = [meta[0] for meta in self._src_metas]
        elif leader_stack is not None:
            self._src_metas = [self.negotiator.unpack(row) for row in leader_stack]
            self._split_sizes = [meta[0] for meta in self._src_metas]

    def _meta_for_src(self, src: int) -> Tuple[int, int, int, int]:
        if self.info.dest_role == DestRole.BOUNDARY_LEADER:
            for peer, meta in zip(self.info.recv_sources, self._src_metas):
                if peer == src:
                    return meta
        leader = self.info.dest_leader_rank
        if leader is not None:
            sources = self.comm_map[leader].recv_sources
            for peer, meta in zip(sources, self._src_metas):
                if peer == src:
                    return meta
        return (0, -1, 0, self.n_deepstack)

    def _forward_data(self) -> None:
        self._phase("data_p2p")
        received: Dict[int, List[Tensor]] = {}
        hidden = self.hidden_size
        if hidden is None and self._saved_payload is not None and self._saved_payload.vision_embeds is not None:
            hidden = int(self._saved_payload.vision_embeds.shape[-1])
            self.hidden_size = hidden

        payload = self._saved_payload
        src_tensors = payload.detached_tensors() if payload is not None and payload.has_vision() else []
        for src, dest in self.plan.p2p_pairs:
            if self.rank == src:
                if not src_tensors:
                    continue
                pg = self._p2p_pg(dest)
                for tensor in src_tensors:
                    _broadcast(tensor, pg, root_global=src)
            elif self.rank == dest:
                n_tokens, _video, has_vision, n_deep = self._meta_for_src(src)
                if not has_vision or n_tokens == 0:
                    received[src] = []
                    continue
                if hidden is None:
                    raise RuntimeError("hidden_size required to receive Bridge tensors")
                bufs = self.negotiator.allocate(
                    n_tokens,
                    hidden,
                    n_deep,
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=False,
                )
                pg = self._p2p_pg(src)
                for buf in bufs:
                    _broadcast(buf, pg, root_global=src)
                received[src] = bufs

        gathered: List[List[Tensor]] = []
        if self.info.dest_role == DestRole.BOUNDARY_LEADER:
            for src, meta in zip(
                self.info.recv_sources,
                self._src_metas or [(0, -1, 0, self.n_deepstack)] * len(self.info.recv_sources),
            ):
                n_tokens, _video, has_vision, n_deep = meta
                if not has_vision or n_tokens == 0:
                    gathered.append([])
                    continue
                if src == self.rank:
                    gathered.append(self._saved_payload.detached_tensors())
                else:
                    gathered.append(received.get(src, []))

        self._phase("data_bcast")
        dest_pg = self.info.dest_broadcast_pg
        dest_ranks = self.info.dest_broadcast_ranks
        if not dest_ranks:
            self._recv_payload = None
            return

        total_tokens = int(sum(self._split_sizes)) if self._split_sizes else 0
        has_any = total_tokens > 0 and any(meta[2] for meta in self._src_metas)
        video_start = self._src_metas[0][1] if self._src_metas else -1
        n_deep = self._src_metas[0][3] if self._src_metas else self.n_deepstack
        hidden = self.hidden_size
        if not has_any:
            self._recv_payload = BoundaryPayload.empty(video_start_index=video_start)
            return
        if hidden is None:
            raise RuntimeError("hidden_size required to broadcast Bridge tensors")

        if self.info.dest_role == DestRole.BOUNDARY_LEADER:
            per_tensor: List[Tensor] = []
            n_tensors = 1 + n_deep
            for idx in range(n_tensors):
                chunks = [chunk[idx] for chunk in gathered if chunk]
                per_tensor.append(
                    torch.cat(chunks, dim=0)
                    if chunks
                    else torch.empty((0, hidden), device=self.device, dtype=self.dtype)
                )
        else:
            per_tensor = self.negotiator.allocate(
                total_tokens,
                hidden,
                n_deep,
                device=self.device,
                dtype=self.dtype,
                requires_grad=False,
            )

        if dest_pg is not None:
            root = self.info.dest_leader_rank
            for tensor in per_tensor:
                _broadcast(tensor, dest_pg, root_global=root)

        leaves = []
        for tensor in per_tensor:
            leaf = tensor.detach().contiguous()
            if leaf.numel() > 0:
                leaf.requires_grad_(True)
            leaves.append(leaf)
        self._recv_payload = BoundaryPayload.from_tensor_list(leaves, video_start, total_tokens)

    # ------------------------------------------------------------------ backward
    def backward_send(self, grads: Optional[List[Optional[Tensor]]]) -> None:
        self._saved_grads = grads
        self._run_backward_exchange()

    def backward_recv(self) -> List[Optional[Tensor]]:
        if not self._backward_done:
            self._run_backward_exchange()
        self._backward_done = False
        return self._recv_grads

    def _run_backward_exchange(self) -> None:
        if self._backward_done:
            return
        if os.environ.get("MX_DEBUG"):
            n_grads = 0
            supplied = getattr(self, "_saved_grads", None)
            if supplied:
                n_grads = sum(1 for item in supplied if item is not None)
            logger.debug(
                f"[mx-bridge] rank={self.rank} bwd-enter self_loop={self._pure_self_loop()} "
                f"p2p={len(self.plan.p2p_pairs)} n_grads={n_grads} "
                f"src={self.info.src_role.name} dest={self.info.dest_role.name}",
            )
        if self._pure_self_loop():
            self.phase_log.extend(["grad_allreduce", "grad_p2p", "grad_bcast"])
            supplied = getattr(self, "_saved_grads", None)
            if self.info.src_role == SrcRole.NON_BOUNDARY:
                self._recv_grads = [None]
            elif supplied is not None:
                self._recv_grads = list(supplied)
            else:
                payload = self._saved_payload
                if payload is None:
                    self._recv_grads = [None]
                else:
                    self._recv_grads = list(payload.grad_tensors())
            self._backward_done = True
            return

        reduced = self._backward_allreduce()
        src_grads = self._backward_p2p(reduced)
        self._recv_grads = self._backward_src_bcast(src_grads)
        self._backward_done = True

    def _materialize_dest_grads(self) -> List[Tensor]:
        payload = self._recv_payload
        supplied = getattr(self, "_saved_grads", None)
        if payload is None or not payload.tensors():
            return []
        if supplied is not None:
            if len(supplied) != len(payload.tensors()):
                raise ValueError("boundary gradient count must match its activation count")
            return [
                torch.zeros_like(tensor) if grad is None else grad.contiguous()
                for tensor, grad in zip(payload.tensors(), supplied, strict=True)
            ]
        out = []
        for tensor in payload.tensors():
            if tensor.grad is not None:
                out.append(tensor.grad.contiguous())
            else:
                out.append(torch.zeros_like(tensor))
        return out

    def _backward_allreduce(self) -> List[Tensor]:
        self._phase("grad_allreduce")
        grads = []
        if self.info.dest_role != DestRole.NON_BOUNDARY:
            grads = self._materialize_dest_grads()
            pg = self.info.dest_broadcast_pg
            if pg is not None:
                for grad in grads:
                    if self.replicated_destination_grads:
                        # Native Qwen3-VL's TP backward has already combined the
                        # input gradient. Average its replicas instead of summing twice.
                        grad.div_(self.dest_layout.tp_size)
                    dist.all_reduce(grad, group=pg)
        return grads

    def _backward_p2p(self, dest_grads: List[Tensor]) -> List[Tensor]:
        self._phase("grad_p2p")
        chunks_by_src: Dict[int, List[Tensor]] = {}
        local_chunk: Optional[List[Tensor]] = None
        if self.info.dest_role == DestRole.BOUNDARY_LEADER and dest_grads:
            sources = self.info.recv_sources
            sizes = self._split_sizes or [int(dest_grads[0].shape[0])]
            if len(sources) > 1:
                split_parts = [torch.split(grad, sizes, dim=0) for grad in dest_grads]
            else:
                split_parts = [(grad,) for grad in dest_grads]
            for idx, src in enumerate(sources):
                chunk = [parts[idx].contiguous() for parts in split_parts]
                chunks_by_src[src] = chunk
                if src == self.rank:
                    local_chunk = [tensor.clone() for tensor in chunk]

        received: Dict[int, List[Tensor]] = {}
        payload = self._saved_payload
        for src, dest in self.plan.p2p_pairs:
            if self.rank == dest:
                chunk = chunks_by_src.get(src)
                if not chunk:
                    continue
                pg = self._p2p_pg(src)
                for tensor in chunk:
                    _broadcast(tensor, pg, root_global=dest)
            elif self.rank == src:
                n_tokens = 0 if payload is None else int(payload.n_merged_tokens)
                if n_tokens == 0 or payload is None or not payload.has_vision():
                    continue
                n_deep = payload.n_deepstack() or self.n_deepstack
                hidden = self._hidden_size_from(payload)
                bufs = self.negotiator.allocate(
                    n_tokens,
                    hidden,
                    n_deep,
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=False,
                )
                pg = self._p2p_pg(dest)
                for buf in bufs:
                    _broadcast(buf, pg, root_global=dest)
                received[dest] = bufs

        if self.info.src_role != SrcRole.BOUNDARY_LEADER:
            return local_chunk or []
        acc = local_chunk
        for dest in self.info.send_targets:
            if dest == self.rank:
                continue
            bufs = received.get(dest)
            if not bufs:
                continue
            if acc is None:
                acc = [tensor.clone() for tensor in bufs]
            else:
                for dst, src_t in zip(acc, bufs):
                    dst.add_(src_t)
        if acc is None and payload is not None and payload.has_vision():
            acc = [torch.zeros_like(tensor) for tensor in payload.tensors()]
        return acc or []

    def _backward_src_bcast(self, src_grads: List[Tensor]) -> List[Optional[Tensor]]:
        self._phase("grad_bcast")
        if self.info.src_role == SrcRole.NON_BOUNDARY:
            return [None]
        ranks = self.info.src_broadcast_ranks
        if not ranks:
            return [None]
        pg = self.info.src_broadcast_pg
        root = self.info.src_leader_rank
        payload = self._saved_payload
        n_tokens = 0 if payload is None else int(payload.n_merged_tokens)
        n_deep = payload.n_deepstack() if payload is not None else self.n_deepstack
        if self.info.src_role == SrcRole.BOUNDARY_FOLLOWER:
            if n_tokens == 0 or payload is None or not payload.has_vision():
                return [None]
            hidden = self._hidden_size_from(payload)
            src_grads = self.negotiator.allocate(
                n_tokens,
                hidden,
                n_deep,
                device=self.device,
                dtype=self.dtype,
                requires_grad=False,
            )
        if pg is not None and src_grads:
            for tensor in src_grads:
                _broadcast(tensor, pg, root_global=root)
        if not src_grads:
            return [None]
        return list(src_grads)
