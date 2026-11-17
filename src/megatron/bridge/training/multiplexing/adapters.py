# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model adapters: what the multiplexing runtime needs to know about each supported model.

An adapter names the encoder boundary streams (one ``LayoutBridge`` each), builds the encoder
provider, extracts per-stream encoder inputs as ``(rows, grid)`` pairs that the input partitioner
splits item by item, runs the encoder on its shard and hands received features to the language
model. Model code is shared unchanged with the native and DistTrain paths.
"""

import copy
from collections.abc import Callable
from dataclasses import fields
from typing import Any

import torch
from megatron.core.utils import unwrap_model

from .bridge.payload import BoundaryPayload


def _copy_provider(provider: Any, provider_class: type) -> Any:
    return provider_class(
        **{item.name: copy.deepcopy(getattr(provider, item.name)) for item in fields(provider) if item.init}
    )


class Qwen3VLAdapter:
    """Qwen3-VL: one vision stream (embeddings plus DeepStack features)."""

    def __init__(self, model: Any) -> None:
        from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step

        self.forward_step: Callable = forward_step
        self.streams: tuple[int, ...] = (len(model.vision_config.deepstack_visual_indexes),)

    def encoder_provider(self, model: Any) -> Any:
        """Build the standalone vision encoder provider from the language provider."""
        from .qwen3_vl import MultiplexingEncoderProvider

        return _copy_provider(model, MultiplexingEncoderProvider)

    def prepare_language(self, model: Any) -> None:
        """Replace the language model's vision tower with a parameter-free boundary cache."""
        from .qwen3_vl import install_boundary_cache

        model.add_encoder = False
        model.register_pre_wrap_hook(install_boundary_cache)

    def encoder_inputs(self, batch: dict, dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Return [(pixel rows, image grid)] or None when the batch is not image-only."""
        visual = batch.get("visual_inputs")
        inputs = visual.normalized_for_model() if visual is not None else {}
        if inputs.get("pixel_values") is None or inputs.get("image_grid_thw") is None:
            return None
        if inputs.get("pixel_values_videos") is not None:
            return None
        return [(inputs["pixel_values"].to(device="cuda", dtype=dtype), inputs["image_grid_thw"].cuda())]

    def run_encoder(self, encoder: list, inputs: list[tuple[torch.Tensor, torch.Tensor]]) -> list[list[torch.Tensor]]:
        """Encode this rank's image shard."""
        (pixels, grid) = inputs[0]
        config = unwrap_model(encoder[0]).config
        # The standalone encoder is a last PP stage. Keep Float16Module from upcasting its
        # boundary: native Qwen consumes BF16 features.
        precision_kwargs = {"fp32_output": False} if config.bf16 else {}
        vision, deepstack = encoder[0](hidden_states=pixels, grid_thw=grid, **precision_kwargs)
        return [[vision, *deepstack]]

    def deliver(self, language_module: Any, payloads: list[BoundaryPayload]) -> None:
        """Queue one micro-batch of received features for the language model."""
        language_module.vision_model.payloads.append(payloads[0])

    def pending(self, language_module: Any) -> int:
        """Number of queued, not yet consumed micro-batches."""
        return len(language_module.vision_model.payloads)

    def clear(self, language_module: Any) -> None:
        """Drop queued features (error paths)."""
        language_module.vision_model.payloads.clear()


class Qwen3OmniAdapter:
    """Qwen3-Omni thinker: a vision stream (with DeepStack) and an audio stream.

    The Hugging Face vision and audio towers cannot be tensor-parallel, so they need encoder TP 1;
    the Megatron towers (``encoder_implementation="megatron"``) take any encoder TP that divides their
    head counts. Audio clips travel through the input partitioner as items of ``(1, 1, frames)``
    rows of mel features. Batches must contain images and audio and no video.
    """

    def __init__(self, model: Any) -> None:
        from megatron.bridge.models.qwen_omni.qwen3_omni_step import forward_step

        self.forward_step: Callable = forward_step
        n_deepstack = len(model.thinker_config.vision_config.deepstack_visual_indexes or ())
        self.streams: tuple[int, ...] = (n_deepstack, 0)

    def encoder_provider(self, model: Any) -> Any:
        """The same provider, building only the vision and audio towers."""
        provider = _copy_provider(model, type(model))
        provider.add_encoder = True
        provider.add_decoder = False
        return provider

    def prepare_language(self, model: Any) -> None:
        """Build the language model without towers; it consumes queued encoder features."""
        model.add_encoder = False

    def encoder_inputs(self, batch: dict, dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Return [(pixel rows, image grid), (mel-frame rows, (1, 1, frames) per clip)] or None."""
        from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model import flatten_audio_features

        pixels, grid = batch.get("pixel_values"), batch.get("image_grid_thw")
        features = batch.get("input_features")
        if pixels is None or grid is None or features is None or batch.get("pixel_values_videos") is not None:
            return None
        if grid.dim() == 3:
            grid = grid.reshape(-1, grid.shape[-1])
        mask = batch.get("feature_attention_mask")
        lengths = batch.get("audio_feature_lengths")
        if lengths is None:
            lengths = mask.sum(dim=1)
        audio = flatten_audio_features(features, mask, lengths).transpose(0, 1).contiguous()
        audio_grid = torch.stack([torch.ones_like(lengths), torch.ones_like(lengths), lengths], dim=1)
        return [
            (pixels.to(device="cuda", dtype=dtype), grid.cuda()),
            (audio.to(device="cuda", dtype=dtype), audio_grid.to(device="cuda", dtype=torch.long)),
        ]

    def run_encoder(self, encoder: list, inputs: list[tuple[torch.Tensor, torch.Tensor]]) -> list[list[torch.Tensor]]:
        """Encode this rank's image and audio shards."""
        (pixels, grid), (audio, audio_grid) = inputs
        thinker = unwrap_model(encoder[0]).thinker
        features = thinker.encode_multimodal(
            pixel_values=pixels,
            image_grid_thw=grid,
            audio_features=audio.transpose(0, 1),
            audio_feature_lengths=audio_grid[:, 2],
        )
        return [[features.vision_embeds, *features.deepstack_embeds], [features.audio_embeds]]

    def deliver(self, language_module: Any, payloads: list[BoundaryPayload]) -> None:
        """Queue one micro-batch of received features on the decoder-only thinker."""
        from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model import OmniEncoderFeatures

        vision, audio = payloads
        language_module.thinker.external_features.append(
            OmniEncoderFeatures(vision.vision_embeds, list(vision.deepstack_features or []), audio.vision_embeds)
        )

    def pending(self, language_module: Any) -> int:
        """Number of queued, not yet consumed micro-batches."""
        return len(language_module.thinker.external_features)

    def clear(self, language_module: Any) -> None:
        """Drop queued features (error paths)."""
        language_module.thinker.external_features.clear()


def adapter_for(model: Any) -> Qwen3VLAdapter | Qwen3OmniAdapter:
    """Select the adapter for a model provider; unsupported models fail before setup."""
    from megatron.bridge.models.qwen_omni.qwen3_omni_provider import Qwen3OmniModelProvider
    from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider

    if type(model) is Qwen3VLModelProvider:
        return Qwen3VLAdapter(model)
    if type(model) is Qwen3OmniModelProvider:
        if model.multiplexing.encoder_tensor_model_parallel_size != 1 and model.encoder_implementation != "megatron":
            raise ValueError(
                "Qwen3-Omni multiplexing with encoder TP > 1 needs encoder_implementation='megatron' "
                "(the Hugging Face towers cannot be tensor-parallel)"
            )
        return Qwen3OmniAdapter(model)
    raise ValueError(f"multiplexing supports Qwen3VLModelProvider and Qwen3OmniModelProvider, got {type(model)}")
