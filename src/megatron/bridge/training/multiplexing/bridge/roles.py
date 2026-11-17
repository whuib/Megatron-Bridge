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
"""Dual-role Bridge communication map.

MCore ``BridgeCommunicator`` assigns each rank a single ``CommRole``. Under
multiplexing every rank belongs to both grids, so a rank can be a source
leader and a destination follower at the same time. Source and destination
roles are therefore independent.

``build_bridge_comm_map`` is a pure function: every rank computes the same
map with no communication. Process groups are attached later by
``LayoutBridge`` at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..layout import ModuleParallelLayout


try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover
    dist = None


class SrcRole(Enum):
    """Participation in the source module boundary."""

    NON_BOUNDARY = "src_non_boundary"
    BOUNDARY_LEADER = "src_boundary_leader"
    BOUNDARY_FOLLOWER = "src_boundary_follower"


class DestRole(Enum):
    """Participation in the destination module boundary."""

    NON_BOUNDARY = "dest_non_boundary"
    BOUNDARY_LEADER = "dest_boundary_leader"
    BOUNDARY_FOLLOWER = "dest_boundary_follower"


@dataclass
class RankBridgeInfo:
    """One rank's full Bridge identity — source and destination are independent."""

    src_role: SrcRole
    dest_role: DestRole
    send_targets: List[int] = field(default_factory=list)
    recv_sources: List[int] = field(default_factory=list)
    src_broadcast_ranks: Optional[Tuple[int, ...]] = None
    dest_broadcast_ranks: Optional[Tuple[int, ...]] = None
    src_leader_rank: Optional[int] = None
    dest_leader_rank: Optional[int] = None
    src_broadcast_pg: Optional["dist.ProcessGroup"] = None
    dest_broadcast_pg: Optional["dist.ProcessGroup"] = None
    p2p_pgs: Dict[int, "dist.ProcessGroup"] = field(default_factory=dict)

    @property
    def recv_source(self):
        """Single source, ``None``, or the full list for fan-in."""
        if not self.recv_sources:
            return None
        if len(self.recv_sources) == 1:
            return self.recv_sources[0]
        return list(self.recv_sources)

    def is_self_loop_leader(self, my_rank: int) -> bool:
        return self.src_role == SrcRole.BOUNDARY_LEADER and self.send_targets == [my_rank]


@dataclass(frozen=True)
class BridgeCommPlan:
    """Deterministic plan derived from two layouts. No process groups."""

    comm_map: Dict[int, RankBridgeInfo]
    dest_broadcast_groups: Tuple[Tuple[int, ...], ...]
    src_broadcast_groups: Tuple[Tuple[int, ...], ...]
    p2p_pairs: Tuple[Tuple[int, int], ...]  # (src_leader, dest_leader), src != dest


def _require_divisible_dp(dp_src: int, dp_dest: int) -> None:
    if dp_src <= 0 or dp_dest <= 0:
        raise ValueError(f"DP_E={dp_src} and DP_L={dp_dest} must be positive")
    if dp_dest % dp_src != 0 and dp_src % dp_dest != 0:
        raise ValueError(
            f"DP_E={dp_src} and DP_L={dp_dest} must divide evenly (整除); non-divisible DP pairing is Future Work"
        )


def boundary_ranks(layout: ModuleParallelLayout, *, last: bool) -> List[int]:
    """Return global ranks on one module pipeline boundary."""
    pp_value = layout.pp_size - 1 if last else 0
    return [rank for rank in range(layout.world_size) if layout.local_rank("pp", rank) == pp_value]


def _leader_of_group(layout: ModuleParallelLayout, group: List[int]) -> int:
    """Unique ``tp_rank == 0`` (and ``cp_rank == 0``) member of a DP replica."""
    leaders = [rank for rank in group if layout.local_rank("tp", rank) == 0 and layout.local_rank("cp", rank) == 0]
    if len(leaders) != 1:
        raise RuntimeError(f"{layout.module_name}: expected one tp=0/cp=0 leader in {group}, got {leaders}")
    return leaders[0]


def leaders_by_dp(layout: ModuleParallelLayout, ranks: List[int]) -> List[int]:
    """Return DP_src/DP_dest leaders sorted by dp_rank. Each has ``tp_rank == 0``."""
    by_dp: Dict[int, List[int]] = {}
    for rank in ranks:
        by_dp.setdefault(layout.local_rank("dp", rank), []).append(rank)
    if len(by_dp) != layout.dp_size:
        raise RuntimeError(
            f"{layout.module_name}: expected {layout.dp_size} DP groups on the PP boundary, got {sorted(by_dp)}"
        )
    return [_leader_of_group(layout, by_dp[dp]) for dp in sorted(by_dp)]


def pair_leaders(src_leaders: List[int], dest_leaders: List[int]) -> Dict[int, List[int]]:
    """Map each src leader to the dest leaders it sends to.

    Fan-out when ``len(dest) % len(src) == 0``; fan-in when the converse holds.
    """
    n_src = len(src_leaders)
    n_dest = len(dest_leaders)
    _require_divisible_dp(n_src, n_dest)
    send_map: Dict[int, List[int]] = {src: [] for src in src_leaders}
    if n_dest % n_src == 0:
        ratio = n_dest // n_src
        for i, src in enumerate(src_leaders):
            send_map[src] = dest_leaders[i * ratio : (i + 1) * ratio]
    else:
        ratio = n_src // n_dest
        for j, dest in enumerate(dest_leaders):
            for src in src_leaders[j * ratio : (j + 1) * ratio]:
                send_map[src] = [dest]
    return send_map


def _tp_groups_on_pp(layout: ModuleParallelLayout, pp_value: int) -> List[Tuple[int, ...]]:
    groups: List[Tuple[int, ...]] = []
    for group in layout.group_ranks("tp"):
        if layout.local_rank("pp", group[0]) == pp_value:
            groups.append(tuple(group))
    return groups


def _leader_in_group(comm_map: Dict[int, RankBridgeInfo], group: Tuple[int, ...], *, src: bool):
    role_attr = "src_role" if src else "dest_role"
    want = SrcRole.BOUNDARY_LEADER if src else DestRole.BOUNDARY_LEADER
    leaders = [rank for rank in group if getattr(comm_map[rank], role_attr) == want]
    if len(leaders) != 1:
        raise RuntimeError(f"expected one leader in {group}, got {leaders}")
    return leaders[0]


def build_bridge_comm_plan(src_layout: ModuleParallelLayout, dest_layout: ModuleParallelLayout) -> BridgeCommPlan:
    """Build deterministic forward and reverse routes without communicating."""
    if src_layout.world_size != dest_layout.world_size:
        raise ValueError(f"src world_size={src_layout.world_size} != dest world_size={dest_layout.world_size}")
    world = src_layout.world_size
    _require_divisible_dp(src_layout.dp_size, dest_layout.dp_size)

    src_boundary = boundary_ranks(src_layout, last=True)
    dest_boundary = boundary_ranks(dest_layout, last=False)
    src_leaders = leaders_by_dp(src_layout, src_boundary)
    dest_leaders = leaders_by_dp(dest_layout, dest_boundary)
    if len(src_leaders) != src_layout.dp_size or len(dest_leaders) != dest_layout.dp_size:
        raise RuntimeError(
            f"leader counts src={len(src_leaders)} dest={len(dest_leaders)} "
            f"vs DP_E={src_layout.dp_size} DP_L={dest_layout.dp_size}"
        )

    send_map = pair_leaders(src_leaders, dest_leaders)
    recv_map: Dict[int, List[int]] = {dest: [] for dest in dest_leaders}
    for src, targets in send_map.items():
        for dest in targets:
            recv_map[dest].append(src)
    for dest, sources in recv_map.items():
        recv_map[dest] = sorted(sources, key=lambda rank: src_layout.local_rank("dp", rank))

    src_tp_groups = _tp_groups_on_pp(src_layout, src_layout.pp_size - 1)
    dest_tp_groups = _tp_groups_on_pp(dest_layout, 0)
    src_group_of = {rank: group for group in src_tp_groups for rank in group}
    dest_group_of = {rank: group for group in dest_tp_groups for rank in group}

    comm_map: Dict[int, RankBridgeInfo] = {}
    src_leader_set = set(src_leaders)
    dest_leader_set = set(dest_leaders)
    src_boundary_set = set(src_boundary)
    dest_boundary_set = set(dest_boundary)

    for rank in range(world):
        if rank in src_leader_set:
            src_role = SrcRole.BOUNDARY_LEADER
        elif rank in src_boundary_set:
            src_role = SrcRole.BOUNDARY_FOLLOWER
        else:
            src_role = SrcRole.NON_BOUNDARY
        if rank in dest_leader_set:
            dest_role = DestRole.BOUNDARY_LEADER
        elif rank in dest_boundary_set:
            dest_role = DestRole.BOUNDARY_FOLLOWER
        else:
            dest_role = DestRole.NON_BOUNDARY

        info = RankBridgeInfo(src_role=src_role, dest_role=dest_role)
        if src_role != SrcRole.NON_BOUNDARY:
            info.src_broadcast_ranks = src_group_of[rank]
            info.src_leader_rank = next(member for member in info.src_broadcast_ranks if member in src_leader_set)
        if dest_role != DestRole.NON_BOUNDARY:
            info.dest_broadcast_ranks = dest_group_of[rank]
            info.dest_leader_rank = next(member for member in info.dest_broadcast_ranks if member in dest_leader_set)
        if src_role == SrcRole.BOUNDARY_LEADER:
            info.send_targets = list(send_map[rank])
        if dest_role == DestRole.BOUNDARY_LEADER:
            info.recv_sources = list(recv_map[rank])
        comm_map[rank] = info

    p2p_pairs = []
    for src, targets in send_map.items():
        for dest in targets:
            if src != dest:
                p2p_pairs.append((src, dest))
    p2p_pairs.sort()

    return BridgeCommPlan(
        comm_map=comm_map,
        dest_broadcast_groups=tuple(sorted(dest_tp_groups)),
        src_broadcast_groups=tuple(sorted(src_tp_groups)),
        p2p_pairs=tuple(p2p_pairs),
    )


def build_bridge_comm_map(
    src_layout: ModuleParallelLayout, dest_layout: ModuleParallelLayout
) -> Dict[int, RankBridgeInfo]:
    """Pure function. Every rank must obtain a field-identical map."""
    return build_bridge_comm_plan(src_layout, dest_layout).comm_map
