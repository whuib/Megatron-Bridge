# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoder-local random streams for overlapping parallel layouts."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed

from megatron.bridge.training.setup import _preserve_rng_state


class EncoderRNG:
    """Preserve language streams while initializing/executing the encoder grid.

    Encoder replicas span every language pipeline stage, whose global seeds differ
    by stage. The encoder therefore owns its CPU stream as well as its CUDA streams:
    modules initialized with the default CPU generator (e.g. ``nn.Conv3d`` and
    ``nn.Embedding`` in the Qwen3-VL vision tower) must be identical on every replica.
    """

    def __init__(self, *, seed: int, tp_rank: int) -> None:
        with _preserve_rng_state():
            torch.manual_seed(seed)
            self.cpu_state = torch.get_rng_state()
            model_parallel_cuda_manual_seed(
                seed,
                tp_rank=tp_rank,
                ep_rank=0,
                etp_rank=0,
                gtp_remat_rank=0,
                egtp_remat_rank=0,
                gtp_remat_world_size=1,
                egtp_remat_world_size=1,
            )
            self.states = get_cuda_rng_tracker().get_states()
            self.device_state = torch.cuda.get_rng_state()

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Advance only the encoder streams, including during recomputation."""
        with _preserve_rng_state():
            torch.set_rng_state(self.cpu_state)
            get_cuda_rng_tracker().set_states(self.states)
            torch.cuda.set_rng_state(self.device_state)
            try:
                yield
            finally:
                self.cpu_state = torch.get_rng_state()
                self.states = get_cuda_rng_tracker().get_states()
                self.device_state = torch.cuda.get_rng_state()
