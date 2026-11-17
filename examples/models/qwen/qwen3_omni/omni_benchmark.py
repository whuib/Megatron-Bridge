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

"""Qwen3-Omni throughput-benchmark models and their model FLOPs.

Two sizes (vision-audio-LLM parameters): ``1b-1b-3b`` and ``3b-3b-9b``. Each keeps Qwen3-Omni's tower
types, tokenizer, processor and DeepStack layers (8/16/24), and resizes the widths and depths. The LLM
is dense (Qwen3-Omni-30B-A3B's thinker is MoE): a "3B LLM" here means 3B total parameters. Both LLMs
use head dimension 128; the 9B LLM uses 8 KV groups so that it can run with TP 8.
"""

import math
from typing import Any

from transformers import AutoConfig, PretrainedConfig


MODEL_SIZES: dict[str, dict[str, dict[str, Any]]] = {
    # 0.97B vision, 0.98B audio, 3.04B LLM.
    "1b-1b-3b": {
        "vision": {
            "hidden_size": 1536,
            "num_heads": 16,
            "intermediate_size": 6144,
            "depth": 27,
            "out_hidden_size": 2048,
        },
        "audio": {
            "d_model": 1536,
            "encoder_attention_heads": 24,
            "encoder_ffn_dim": 6144,
            "encoder_layers": 34,
            "num_hidden_layers": 34,
            "output_dim": 2048,
        },
        "llm": {
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 5120,
        },
    },
    # 2.99B vision, 3.03B audio, 9.00B LLM. Tower head counts (24 vision, head dim 112; 32 audio, head
    # dim 80) are divisible by 8 so that tensor-parallel towers can shard up to TP 8; the first
    # 3b-3b-9b grid used 21 and 20 heads (same parameter counts).
    "3b-3b-9b": {
        "vision": {
            "hidden_size": 2688,
            "num_heads": 24,
            "intermediate_size": 10752,
            "depth": 27,
            "out_hidden_size": 4096,
        },
        "audio": {
            "d_model": 2560,
            "encoder_attention_heads": 32,
            "encoder_ffn_dim": 10240,
            "encoder_layers": 38,
            "num_hidden_layers": 38,
            "output_dim": 4096,
        },
        "llm": {
            "hidden_size": 4096,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 9728,
        },
    },
}
# Dense BF16 tensor-core peak per GPU, for MFU.
PEAK_BF16_TFLOPS = {"A100": 312.0, "H100": 989.0, "H200": 989.0, "B200": 2250.0}


def peak_tflops(gpu_name: str) -> float:
    """Dense BF16 peak of a GPU model name (e.g. ``NVIDIA A100-SXM4-80GB``)."""
    for key, value in PEAK_BF16_TFLOPS.items():
        if key in gpu_name:
            return value
    raise ValueError(f"unknown BF16 peak for GPU {gpu_name!r}; add it to PEAK_BF16_TFLOPS")


def benchmark_hf_config(base_path: str, model_size: str) -> PretrainedConfig:
    """Load a Qwen3-Omni config (processor-compatible tokens) and resize it to a benchmark model size."""
    size = MODEL_SIZES[model_size]
    config = AutoConfig.from_pretrained(base_path)
    config.enable_audio_output = False
    thinker = config.thinker_config
    for key, value in size["vision"].items():
        setattr(thinker.vision_config, key, value)
    for key, value in size["audio"].items():
        setattr(thinker.audio_config, key, value)
    text = thinker.text_config
    for key, value in size["llm"].items():
        setattr(text, key, value)
    # Every layer uses the dense MLP; the provider is made dense explicitly as well.
    text.mlp_only_layers = list(range(size["llm"]["num_hidden_layers"]))
    text.tie_word_embeddings = False
    return config


def make_dense(provider: Any, model_size: str) -> None:
    """Configure the Megatron language model as the dense benchmark LLM."""
    provider.num_moe_experts = None
    provider.moe_ffn_hidden_size = None
    provider.ffn_hidden_size = MODEL_SIZES[model_size]["llm"]["intermediate_size"]


def parameter_counts(provider: Any) -> dict[str, int]:
    """Exact parameter counts of the three components, from meta-device modules."""
    import torch
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
        Qwen3OmniMoeVisionEncoder,
    )

    thinker = provider.thinker_config
    with torch.device("meta"):
        vision = Qwen3OmniMoeVisionEncoder._from_config(thinker.vision_config)
        audio = Qwen3OmniMoeAudioEncoder._from_config(thinker.audio_config)
    h, layers, ffn = provider.hidden_size, provider.num_layers, provider.ffn_hidden_size
    q = provider.num_attention_heads * provider.kv_channels
    kv = provider.num_query_groups * provider.kv_channels
    layer = h * q + 2 * h * kv + q * h + 3 * h * ffn + 2 * h + 2 * provider.kv_channels
    embeddings = provider.vocab_size * h * (1 if provider.share_embeddings_and_output_weights else 2)
    return {
        "vision": sum(p.numel() for p in vision.parameters()),
        "audio": sum(p.numel() for p in audio.parameters()),
        "llm": layers * layer + embeddings + h,
    }


def audio_encoder_flops(audio_config: Any, frames_per_clip: int) -> int:
    """Training FLOPs (3x forward) of the Qwen3-Omni audio encoder for one clip of mel frames.

    Mirrors the Hugging Face encoder: three stride-2 3x3 convolutions per 2*n_window-frame chunk,
    a linear projection to d_model, transformer layers with attention inside n_window_infer
    windows, then proj1 and proj2. A multiply-add counts as two FLOPs.
    """
    d, ffn = audio_config.d_model, audio_config.encoder_ffn_dim
    channels, mel = audio_config.downsample_hidden_size, audio_config.num_mel_bins
    chunk = 2 * audio_config.n_window
    chunks = [chunk] * (frames_per_clip // chunk) + ([frames_per_clip % chunk] if frames_per_clip % chunk else [])
    conv, tokens_per_chunk = 0, []
    for length in chunks:
        m, t, c_in = mel, length, 1
        for _ in range(3):
            m, t = math.ceil(m / 2), math.ceil(t / 2)
            conv += 2 * 9 * c_in * channels * m * t
            c_in = channels
        conv += 2 * channels * m * d * t
        tokens_per_chunk.append(t)
    tokens = sum(tokens_per_chunk)
    window_chunks = audio_config.n_window_infer // chunk
    windows = [sum(tokens_per_chunk[i : i + window_chunks]) for i in range(0, len(tokens_per_chunk), window_chunks)]
    layers = audio_config.encoder_layers * (tokens * (8 * d * d + 4 * d * ffn) + sum(4 * w * w * d for w in windows))
    head = tokens * (2 * d * d + 2 * d * audio_config.output_dim)
    return 3 * (conv + layers + head)


def omni_flops_per_iteration(cfg: Any, workload: Any) -> int:
    """Model training FLOPs for one global batch: LLM (padded length) + vision ViT + audio encoder."""
    from megatron.bridge.training.utils.flop_utils import num_floating_point_operations, vit_flops_from_patch_stats

    images = workload.global_batch_size * workload.images_per_sample
    language = num_floating_point_operations(cfg, batch_size=workload.global_batch_size)
    merge = cfg.model.thinker_config.vision_config.spatial_merge_size**2
    vision = vit_flops_from_patch_stats(
        cfg,
        patch_sum=images * workload.patches_per_image,
        patch_squared_sum=images * workload.patches_per_image**2,
        merged_token_sum=images * workload.patches_per_image // merge,
    )
    clips = workload.global_batch_size * workload.audios_per_sample
    audio = clips * audio_encoder_flops(cfg.model.thinker_config.audio_config, workload.audio_frames_per_clip)
    return int(language) + int(vision) + int(audio)
