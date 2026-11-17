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
"""Static rank layouts for multiplexed model modules."""

from dataclasses import dataclass
from functools import cached_property
from itertools import product
from math import prod
from typing import Dict, List, Optional, Sequence, Tuple, Union


_DIMENSIONS = ("tp", "cp", "pp", "dp")


@dataclass(frozen=True)
class ModuleParallelLayout:
    """Describe one module's logical parallel layout over all global ranks."""

    module_name: str
    world_size: int
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    dimension_order: str = "tp-cp-pp-dp"
    rank_mapping: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if not self.module_name or not self.module_name.isidentifier():
            raise ValueError(f"module_name must be a non-empty Python identifier, got {self.module_name!r}")

        factors = (self.tp_size, self.cp_size, self.pp_size, self.dp_size)
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in factors):
            raise ValueError(
                "parallel sizes must be positive integers: "
                f"tp_size={self.tp_size}, cp_size={self.cp_size}, "
                f"pp_size={self.pp_size}, dp_size={self.dp_size}"
            )
        if not isinstance(self.world_size, int) or isinstance(self.world_size, bool):
            raise ValueError(f"world_size must be a positive integer, got {self.world_size!r}")

        factor_product = prod(factors)
        if factor_product != self.world_size:
            raise ValueError(
                "parallel size product does not match world_size: "
                f"tp_size={self.tp_size} * cp_size={self.cp_size} * "
                f"pp_size={self.pp_size} * dp_size={self.dp_size} = "
                f"{factor_product}, world_size={self.world_size}"
            )

        dim_names = self.dimension_order.split("-")
        if len(dim_names) != len(_DIMENSIONS) or set(dim_names) != set(_DIMENSIONS):
            raise ValueError(f"dimension_order must be a permutation of {_DIMENSIONS}, got {self.dimension_order!r}")

        if self.rank_mapping is not None:
            if not isinstance(self.rank_mapping, tuple):
                raise ValueError("rank_mapping must be a tuple or None")
            if len(self.rank_mapping) != self.world_size:
                raise ValueError(
                    f"rank_mapping length must equal world_size={self.world_size}, got {len(self.rank_mapping)}"
                )
            if sorted(self.rank_mapping) != list(range(self.world_size)):
                raise ValueError(f"rank_mapping must be a permutation of range(world_size={self.world_size})")

    @property
    def dim_names(self) -> List[str]:
        return self.dimension_order.split("-")

    @property
    def sizes(self) -> Dict[str, int]:
        return {
            "tp": self.tp_size,
            "cp": self.cp_size,
            "pp": self.pp_size,
            "dp": self.dp_size,
        }

    @property
    def shape(self) -> List[int]:
        return [self.sizes[name] for name in self.dim_names]

    @cached_property
    def _stride(self) -> List[int]:
        strides: List[int] = []
        stride = 1
        for size in self.shape:
            strides.append(stride)
            stride *= size
        return strides

    @cached_property
    def _inverse_mapping(self) -> Tuple[int, ...]:
        if self.rank_mapping is None:
            return tuple(range(self.world_size))
        inverse = [0] * self.world_size
        for logical_rank, global_rank in enumerate(self.rank_mapping):
            inverse[global_rank] = logical_rank
        return tuple(inverse)

    def coord_to_logical_rank(self, coord: Dict[str, int]) -> int:
        if set(coord) != set(_DIMENSIONS):
            raise ValueError(f"coord must contain exactly {_DIMENSIONS}, got {sorted(coord)}")
        logical_rank = 0
        for name, stride in zip(self.dim_names, self._stride):
            value = coord[name]
            size = self.sizes[name]
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < size:
                raise ValueError(f"{name} coordinate must be in [0, {size}), got {value!r}")
            logical_rank += value * stride
        return logical_rank

    def logical_rank_to_coord(self, logical_rank: int) -> Dict[str, int]:
        self._validate_rank(logical_rank, "logical_rank")
        return {
            name: (logical_rank // stride) % size
            for name, size, stride in zip(self.dim_names, self.shape, self._stride)
        }

    def coord_to_global_rank(self, coord: Dict[str, int]) -> int:
        logical_rank = self.coord_to_logical_rank(coord)
        if self.rank_mapping is None:
            return logical_rank
        return self.rank_mapping[logical_rank]

    def global_rank_to_coord(self, global_rank: int) -> Dict[str, int]:
        self._validate_rank(global_rank, "global_rank")
        return self.logical_rank_to_coord(self._inverse_mapping[global_rank])

    def group_ranks(self, dims: Union[str, Sequence[str]]) -> List[List[int]]:
        selected = dims.split("-") if isinstance(dims, str) else list(dims)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError(f"dims must be a non-empty sequence of unique dimensions, got {dims!r}")
        unknown = set(selected) - set(_DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown dimensions: {sorted(unknown)}")

        # Preserve layout order. This matches MCore's masked mixed-radix
        # decomposition, including both group and in-group enumeration order.
        selected_set = set(selected)
        varying = [name for name in self.dim_names if name in selected_set]
        fixed = [name for name in self.dim_names if name not in selected_set]
        zero_coord = dict.fromkeys(_DIMENSIONS, 0)
        groups: List[List[int]] = []
        for fixed_values in self._mixed_radix_values(fixed):
            group: List[int] = []
            for varying_values in self._mixed_radix_values(varying):
                coord = dict(zero_coord)
                coord.update(zip(fixed, fixed_values))
                coord.update(zip(varying, varying_values))
                group.append(self.coord_to_global_rank(coord))
            groups.append(group)
        return groups

    def local_rank(self, dim: str, global_rank: Optional[int] = None) -> int:
        if dim not in _DIMENSIONS:
            raise ValueError(f"dim must be one of {_DIMENSIONS}, got {dim!r}")
        if global_rank is None:
            try:
                import torch.distributed as dist

                if not dist.is_initialized():
                    raise RuntimeError
                global_rank = dist.get_rank()
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError("global_rank is required when torch.distributed is not initialized") from exc
        return self.global_rank_to_coord(global_rank)[dim]

    def is_first_stage(self, global_rank: Optional[int] = None) -> bool:
        return self.local_rank("pp", global_rank) == 0

    def is_last_stage(self, global_rank: Optional[int] = None) -> bool:
        return self.local_rank("pp", global_rank) == self.pp_size - 1

    def non_dp_index(self, coord: Dict[str, int]) -> int:
        """Flatten non-DP coordinates. Unique per DP group, identical inside one."""
        index = 0
        stride = 1
        for name in self.dim_names:
            if name == "dp":
                continue
            index += coord[name] * stride
            stride *= self.sizes[name]
        return index

    def to_metadata(self) -> dict:
        return {
            "module_name": self.module_name,
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "cp_size": self.cp_size,
            "pp_size": self.pp_size,
            "dp_size": self.dp_size,
            "dimension_order": self.dimension_order,
            "rank_mapping": list(self.rank_mapping) if self.rank_mapping is not None else None,
        }

    @classmethod
    def from_metadata(cls, d: dict) -> "ModuleParallelLayout":
        metadata = dict(d)
        if metadata.get("rank_mapping") is not None:
            metadata["rank_mapping"] = tuple(metadata["rank_mapping"])
        return cls(**metadata)

    def describe(self) -> str:
        mapping = "identity" if self.rank_mapping is None else "permuted"
        return (
            f"{self.module_name}: world={self.world_size}, tp={self.tp_size}, "
            f"cp={self.cp_size}, pp={self.pp_size}, dp={self.dp_size}, "
            f"order={self.dimension_order}, mapping={mapping}"
        )

    def _mixed_radix_values(self, names: Sequence[str]):
        """Yield tuples with the first named dimension varying fastest."""
        if not names:
            yield ()
            return
        ranges = [range(self.sizes[name]) for name in reversed(names)]
        for reversed_values in product(*ranges):
            yield tuple(reversed(reversed_values))

    def _validate_rank(self, rank: int, label: str) -> None:
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < self.world_size:
            raise ValueError(f"{label} must be in [0, {self.world_size}), got {rank!r}")
