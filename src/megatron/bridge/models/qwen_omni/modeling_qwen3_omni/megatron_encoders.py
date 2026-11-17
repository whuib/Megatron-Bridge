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

"""Tensor-parallel Megatron implementations of the Qwen3-Omni vision and audio towers.

Bridge calls the Hugging Face towers by default (``encoder_implementation="huggingface"``). They
cannot be tensor-parallel, so every TP rank holds, updates and runs a full copy. This module is a
deliberate exception to calling Hugging Face encoders directly: ``encoder_implementation="megatron"``
computes the same functions with Transformer Engine layers sharded over the TP group of the process
groups the towers are built with.

* Vision: Bridge's Qwen3-VL ViT. The Qwen3-Omni vision encoder has the same patch embedding, learned
  and rotary position embeddings, blocks and DeepStack mergers; only its merger parameter names
  differ (``merger.ln_q``/``mlp.0``/``mlp.2`` vs ``norm``/``linear_fc1``/``linear_fc2``).
* Audio: the Hugging Face convolutional front end, sinusoidal positions and output projections
  (replicated on every TP rank; about 1% of the parameters) around a Megatron ``TransformerBlock``
  for the pre-LayerNorm transformer layers. The windowed attention is expressed as packed (THD)
  sequences with the Hugging Face ``cu_seqlens``.

Both towers attend only within an image or an audio window. Their core attention is
``PackedSDPAttention`` rather than Transformer Engine's: TE fuses THD attention only on sm90+ and
otherwise pads every sequence through an unfused path whose layout conversions dominate the tower
time (about 4x slower than the Hugging Face towers on A100).

``vision_state_dict_from_hf`` and ``audio_state_dict_from_hf`` convert unsharded Hugging Face tower
weights to these modules (TP 1); ``shard_for_tensor_parallel`` slices a full state dict for one TP rank.
"""

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from torch import nn
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    chunk_and_pad_features,
    get_audio_cu_seqlens,
    get_valid_indices,
)

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import get_vision_model_config
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import PatchMergerSubmodules
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel


class PackedSDPAttention(MegatronModule):
    """Core attention over packed (THD) sequences with PyTorch SDPA, one call per sequence length.

    The packed tokens are split at ``cu_seqlens``; sequences of equal length are stacked into one
    batched ``scaled_dot_product_attention`` call and the outputs are put back in token order. The
    split is computed once per ``PackedSeqParams`` (one host sync per micro-batch) and reused by every
    layer. Non-causal, no bias; inputs are this rank's attention heads ``[t, heads, head_dim]``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config=config)
        if config.context_parallel_size != 1:
            raise ValueError("PackedSDPAttention does not support context parallelism")
        self.softmax_scale = softmax_scale
        self.attention_dropout = config.attention_dropout if attention_dropout is None else attention_dropout

    @staticmethod
    def _plan(
        packed_seq_params: PackedSeqParams, device: torch.device
    ) -> tuple[list[tuple[int, torch.Tensor]], torch.Tensor | None]:
        """Per sequence length, the starts of its sequences (``[sequences, length]`` token indices), and the
        permutation back to token order (None when the groups already are in token order). Cached."""
        plan = getattr(packed_seq_params, "_sdpa_plan", None)
        if plan is None:
            if packed_seq_params.cu_seqlens_kv is not None and not torch.equal(
                packed_seq_params.cu_seqlens_q, packed_seq_params.cu_seqlens_kv
            ):
                raise ValueError("PackedSDPAttention needs identical query and key/value sequences")
            bounds = packed_seq_params.cu_seqlens_q.tolist()
            by_length: dict[int, list[int]] = {}
            for start, end in zip(bounds[:-1], bounds[1:]):
                if end > start:
                    by_length.setdefault(end - start, []).append(start)
            groups, flat = [], []
            for length, starts in sorted(by_length.items()):
                index = torch.tensor(starts, device=device)[:, None] + torch.arange(length, device=device)[None, :]
                groups.append((length, index))
                flat.extend(start + offset for start in starts for offset in range(length))
            in_order = flat == list(range(len(flat)))
            order = None if in_order else torch.argsort(torch.tensor(flat, device=device))
            plan = (groups, order)
            packed_seq_params._sdpa_plan = plan
        return plan

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        attn_mask_type=None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> torch.Tensor:
        """Attention within each packed sequence; returns ``[t, heads, head_dim]``."""
        if attention_bias is not None:
            raise NotImplementedError("PackedSDPAttention does not support attention bias")
        dropout = self.attention_dropout if self.training else 0.0
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            if attention_mask is not None:
                raise NotImplementedError("PackedSDPAttention attends over packed sequences or whole sequences")
            # [s, b, heads, dim] -> [b, heads, s, dim]
            q, k, v = (tensor.permute(1, 2, 0, 3) for tensor in (query, key, value))
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, scale=self.softmax_scale)
            return out.permute(2, 0, 1, 3).reshape(out.shape[2], out.shape[0], -1)
        groups, order = self._plan(packed_seq_params, query.device)
        outputs = []
        for length, index in groups:
            if order is None and len(groups) == 1:
                # One run of equal-length sequences in token order: a view, no gather.
                q, k, v = (
                    tensor.view(-1, length, *tensor.shape[1:]).transpose(1, 2) for tensor in (query, key, value)
                )
            else:
                q, k, v = (tensor[index].transpose(1, 2) for tensor in (query, key, value))
            # [sequences, heads, length, dim]
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, scale=self.softmax_scale)
            outputs.append(out.transpose(1, 2).reshape(-1, *query.shape[1:]))
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        return output if order is None else output[order]


def _tower_layer_spec(self_attention_module=None):
    """TE ViT layer spec with ``PackedSDPAttention`` as core attention."""
    spec = get_vit_layer_with_transformer_engine_spec()
    if self_attention_module is not None:
        spec.submodules.self_attention.module = self_attention_module
    spec.submodules.self_attention.submodules.core_attention = PackedSDPAttention
    return spec


def mark_replicated_params_for_tp_grad_sync(module: nn.Module) -> None:
    """Average the gradients of parameters every TP rank holds in full (not the TP shards)."""
    for param in module.parameters():
        if not getattr(param, "tensor_model_parallel", False):
            param.average_gradients_across_tp_domain = True


class Qwen3OmniVisionEncoder(Qwen3VLVisionModel):
    """The Qwen3-Omni vision encoder as Bridge's tensor-parallel Qwen3-VL ViT.

    ``forward(pixel_values, grid_thw)`` returns ``(merged_embeddings, deepstack_features)``.
    """

    def __init__(self, hf_config, language_config: TransformerConfig, pg_collection: ProcessGroupCollection) -> None:
        layer_spec = _tower_layer_spec(Qwen3VLSelfAttention)
        merger_spec = PatchMergerSubmodules(
            patch_norm=TENorm, linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
        )
        config = get_vision_model_config(hf_config, megatron_config=language_config)
        # The towers shard over the TP group they are built with (language TP for native training,
        # the encoder TP of DistTrain or multiplexing), not necessarily the language config's TP.
        config.tensor_model_parallel_size = pg_collection.tp.size()
        config.pipeline_model_parallel_size = 1
        config.params_dtype = language_config.params_dtype
        config.bf16 = language_config.bf16
        config.fp16 = language_config.fp16
        super().__init__(
            config, layer_spec, merger_spec, pre_process=True, post_process=True, pg_collection=pg_collection
        )
        mark_replicated_params_for_tp_grad_sync(self)

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype (inputs are cast to it)."""
        return self.patch_embed.proj.weight.dtype


def audio_transformer_config(hf_config, language_config: TransformerConfig, *, tp_size: int) -> TransformerConfig:
    """Megatron config of the Qwen3-Omni audio transformer layers (Whisper-style pre-LayerNorm)."""
    heads = hf_config.encoder_attention_heads
    if hf_config.d_model % heads:
        raise ValueError("audio d_model must be divisible by encoder_attention_heads")
    if hf_config.activation_function != "gelu":
        raise ValueError(
            f"Megatron audio encoder supports activation_function='gelu', got {hf_config.activation_function!r}"
        )
    return TransformerConfig(
        num_layers=hf_config.encoder_layers,
        hidden_size=hf_config.d_model,
        num_attention_heads=heads,
        num_query_groups=heads,
        kv_channels=hf_config.d_model // heads,
        ffn_hidden_size=hf_config.encoder_ffn_dim,
        add_bias_linear=True,
        add_qkv_bias=True,
        gated_linear_unit=False,
        activation_func=F.gelu,  # exact GELU, as Hugging Face's "gelu"
        bias_activation_fusion=False,  # the fused bias-GELU uses the tanh approximation
        bias_dropout_fusion=False,
        normalization="LayerNorm",
        layernorm_epsilon=1e-5,  # torch.nn.LayerNorm default used by the Hugging Face layers
        hidden_dropout=0.0,
        attention_dropout=0.0,
        apply_query_key_layer_scaling=False,
        attention_softmax_in_fp32=True,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        params_dtype=language_config.params_dtype,
        bf16=language_config.bf16,
        fp16=language_config.fp16,
        use_cpu_initialization=language_config.use_cpu_initialization,
    )


def _sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> torch.Tensor:
    """Hugging Face ``SinusoidsPositionEmbedding`` table."""
    log_timescale_increment = torch.log(torch.tensor(max_timescale)) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class Qwen3OmniAudioEncoder(MegatronModule):
    """The Qwen3-Omni audio encoder with tensor-parallel transformer layers.

    ``forward(input_features, feature_lens)`` takes the flattened ``[mel, frames]`` features of all
    clips and returns their ``[tokens, output_dim]`` embeddings, like the Hugging Face encoder's
    ``last_hidden_state``.
    """

    def __init__(self, hf_config, language_config: TransformerConfig, pg_collection: ProcessGroupCollection) -> None:
        config = audio_transformer_config(hf_config, language_config, tp_size=pg_collection.tp.size())
        super().__init__(config=config)
        self.n_window = hf_config.n_window
        self.n_window_infer = hf_config.n_window_infer
        self.conv_chunksize = hf_config.conv_chunksize
        channels = hf_config.downsample_hidden_size
        self.conv2d1 = nn.Conv2d(1, channels, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(channels, channels, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(channels, channels, 3, 2, padding=1)
        frequencies = (((hf_config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(channels * frequencies, hf_config.d_model, bias=False)
        self.register_buffer(
            "positional_embedding", _sinusoids(hf_config.max_source_positions, hf_config.d_model), persistent=False
        )
        self.decoder = TransformerBlock(
            config=config,
            spec=_tower_layer_spec(),
            post_layer_norm=True,  # Hugging Face ln_post
            pg_collection=pg_collection,
        )
        self.proj1 = nn.Linear(hf_config.d_model, hf_config.d_model)
        self.proj2 = nn.Linear(hf_config.d_model, hf_config.output_dim)
        mark_replicated_params_for_tp_grad_sync(self)

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype (inputs are cast to it)."""
        return self.conv2d1.weight.dtype

    def forward(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> torch.Tensor:
        """Encode ``[mel, frames]`` features of clips with ``feature_lens`` frames each."""
        padded, chunk_lengths = chunk_and_pad_features(input_features, feature_lens, self.n_window)
        valid_indices = get_valid_indices(chunk_lengths, self.n_window)
        cu_seqlens = get_audio_cu_seqlens(chunk_lengths, feature_lens, self.n_window_infer, self.n_window)
        padded = padded.unsqueeze(1).to(dtype=self.conv2d1.weight.dtype)
        convolved = []
        for chunk in padded.split(self.conv_chunksize, dim=0):
            chunk = F.gelu(self.conv2d1(chunk))
            chunk = F.gelu(self.conv2d2(chunk))
            convolved.append(F.gelu(self.conv2d3(chunk)))
        embedded = torch.cat(convolved, dim=0)
        batch, channels, frequencies, time = embedded.size()
        embedded = self.conv_out(embedded.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * frequencies))
        embedded = embedded + self.positional_embedding[:time].unsqueeze(0).to(embedded.dtype)
        hidden = torch.index_select(embedded.reshape(-1, embedded.shape[-1]), 0, valid_indices)
        window_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(window_lengths.max())
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
        hidden = self.decoder(hidden_states=hidden[:, None], attention_mask=None, packed_seq_params=packed)[:, 0]
        return self.proj2(F.gelu(self.proj1(hidden)))


def _interleave_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    """Megatron's fused QKV layout (per head: q, k, v rows) from separate projections."""
    parts = [tensor.reshape(heads, -1, *tensor.shape[1:]) for tensor in (q, k, v)]
    return torch.cat(parts, dim=1).reshape(-1, *q.shape[1:])


def vision_state_dict_from_hf(hf: dict[str, torch.Tensor], *, num_heads: int) -> dict[str, torch.Tensor]:
    """``Qwen3OmniVisionEncoder`` (TP 1) weights from a Hugging Face ``Qwen3OmniMoeVisionEncoder`` state dict."""
    out = {
        "patch_embed.proj.weight": hf["patch_embed.proj.weight"],
        "patch_embed.proj.bias": hf["patch_embed.proj.bias"],
    }
    out["pos_embed.weight"] = hf["pos_embed.weight"]
    layers = sorted({int(key.split(".")[1]) for key in hf if key.startswith("blocks.")})
    for i in layers:
        src, dst = f"blocks.{i}.", f"decoder.layers.{i}."
        for kind in ("weight", "bias"):
            q, k, v = hf[f"{src}attn.qkv.{kind}"].chunk(3, dim=0)
            out[f"{dst}self_attention.linear_qkv.{kind}"] = _interleave_qkv(q, k, v, num_heads)
            out[f"{dst}self_attention.linear_qkv.layer_norm_{kind}"] = hf[f"{src}norm1.{kind}"]
            out[f"{dst}self_attention.linear_proj.{kind}"] = hf[f"{src}attn.proj.{kind}"]
            out[f"{dst}mlp.linear_fc1.layer_norm_{kind}"] = hf[f"{src}norm2.{kind}"]
            out[f"{dst}mlp.linear_fc1.{kind}"] = hf[f"{src}mlp.linear_fc1.{kind}"]
            out[f"{dst}mlp.linear_fc2.{kind}"] = hf[f"{src}mlp.linear_fc2.{kind}"]
    n_deepstack = len({key.split(".")[1] for key in hf if key.startswith("merger_list.")})
    mergers = {"merger.": "merger."}
    mergers.update({f"merger_list.{i}.": f"decoder.deepstack_merger_list.{i}." for i in range(n_deepstack)})
    for src, dst in mergers.items():
        for kind in ("weight", "bias"):
            out[f"{dst}patch_norm.{kind}"] = hf[f"{src}ln_q.{kind}"]
            out[f"{dst}linear_fc1.{kind}"] = hf[f"{src}mlp.0.{kind}"]
            out[f"{dst}linear_fc2.{kind}"] = hf[f"{src}mlp.2.{kind}"]
    return out


def audio_state_dict_from_hf(hf: dict[str, torch.Tensor], *, num_heads: int) -> dict[str, torch.Tensor]:
    """``Qwen3OmniAudioEncoder`` (TP 1) weights from a Hugging Face ``Qwen3OmniMoeAudioEncoder`` state dict."""
    out = {key: hf[key] for key in hf if key.startswith(("conv2d", "conv_out.", "proj1.", "proj2."))}
    out["decoder.final_layernorm.weight"] = hf["ln_post.weight"]
    out["decoder.final_layernorm.bias"] = hf["ln_post.bias"]
    layers = sorted({int(key.split(".")[1]) for key in hf if key.startswith("layers.")})
    for i in layers:
        src, dst = f"layers.{i}.", f"decoder.layers.{i}."
        for kind in ("weight", "bias"):
            q, k, v = (hf[f"{src}self_attn.{name}_proj.{kind}"] for name in ("q", "k", "v"))
            out[f"{dst}self_attention.linear_qkv.{kind}"] = _interleave_qkv(q, k, v, num_heads)
            out[f"{dst}self_attention.linear_qkv.layer_norm_{kind}"] = hf[f"{src}self_attn_layer_norm.{kind}"]
            out[f"{dst}self_attention.linear_proj.{kind}"] = hf[f"{src}self_attn.out_proj.{kind}"]
            out[f"{dst}mlp.linear_fc1.layer_norm_{kind}"] = hf[f"{src}final_layer_norm.{kind}"]
            out[f"{dst}mlp.linear_fc1.{kind}"] = hf[f"{src}fc1.{kind}"]
            out[f"{dst}mlp.linear_fc2.{kind}"] = hf[f"{src}fc2.{kind}"]
    return out


def shard_for_tensor_parallel(module: nn.Module, full: dict[str, torch.Tensor], *, tp_rank: int, tp_size: int) -> dict:
    """Slice a full (TP 1) state dict for ``module``'s parameters on one TP rank.

    Tensor-parallel parameters carry Megatron's ``partition_dim``; every other parameter is replicated.
    """
    params = dict(module.named_parameters())
    out = {}
    for name, tensor in full.items():
        param = params.get(name)
        if param is not None and getattr(param, "tensor_model_parallel", False):
            tensor = tensor.chunk(tp_size, dim=param.partition_dim)[tp_rank]
        out[name] = tensor
    return out
