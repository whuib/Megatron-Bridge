"""Tensor-parallel Megatron Qwen3-Omni towers against the Hugging Face towers.

Run with ``pytest`` for TP 1, or with ``torch.distributed.run --nproc_per_node=2`` for TP 2: every rank
builds the Hugging Face reference, converts and shards its weights, and compares outputs.
"""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


pytestmark = [pytest.mark.unit, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (TE layers)")]

HEADS = 4


@pytest.fixture(scope="module")
def tp_groups():
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(
            "nccl",
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
            timeout=datetime.timedelta(minutes=5),
        )
        created = True
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    parallel_state.destroy_model_parallel()
    if created:
        dist.destroy_process_group()


def _language_config():
    from megatron.core.transformer import TransformerConfig

    return TransformerConfig(
        num_layers=1, hidden_size=64, num_attention_heads=HEADS, params_dtype=torch.bfloat16, bf16=True
    )


def _load(megatron, full, groups):
    from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.megatron_encoders import shard_for_tensor_parallel

    sharded = shard_for_tensor_parallel(megatron, full, tp_rank=groups.tp.rank(), tp_size=groups.tp.size())
    params = dict(megatron.named_parameters())
    assert set(params) == set(sharded), (sorted(set(params) - set(sharded)), sorted(set(sharded) - set(params)))
    with torch.no_grad():
        for name, tensor in sharded.items():
            assert params[name].shape == tensor.shape, name
            params[name].copy_(tensor)


def _assert_close(reference, actual, name):
    reference, actual = reference.float(), actual.float()
    assert reference.shape == actual.shape, name
    cosine = F.cosine_similarity(reference.flatten(), actual.flatten(), dim=0).item()
    assert cosine >= 0.999, f"{name}: cosine {cosine:.6f}, max |diff| {(reference - actual).abs().max().item():.3e}"


def _seeded(module_factory):
    torch.manual_seed(0)
    return module_factory().cuda().bfloat16()


def test_vision_matches_hugging_face(tp_groups):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeVisionEncoderConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

    from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.megatron_encoders import (
        Qwen3OmniVisionEncoder,
        vision_state_dict_from_hf,
    )

    config = Qwen3OmniMoeVisionEncoderConfig(
        depth=3,
        hidden_size=128,
        num_heads=HEADS,
        intermediate_size=256,
        out_hidden_size=96,
        deepstack_visual_indexes=[0, 2],
    )
    config._attn_implementation = "sdpa"
    hf = _seeded(lambda: Qwen3OmniMoeVisionEncoder._from_config(config))
    megatron = Qwen3OmniVisionEncoder(config, _language_config(), tp_groups).cuda().bfloat16()
    full = vision_state_dict_from_hf(hf.state_dict(), num_heads=HEADS)
    _load(megatron, full, tp_groups)
    grid = torch.tensor([[1, 8, 12], [1, 4, 4]], device="cuda")
    torch.manual_seed(1)
    pixels = torch.randn(int(grid.prod(-1).sum()), 3 * 2 * 16 * 16, device="cuda", dtype=torch.bfloat16)
    reference = hf(pixels, grid_thw=grid)
    merged, deepstack = megatron(pixels, grid_thw=grid)
    _assert_close(reference.pooler_output, merged, "merged embeddings")
    assert len(deepstack) == 2
    for index, (expected, actual) in enumerate(zip(reference.deepstack_features, deepstack, strict=True)):
        _assert_close(expected, actual, f"deepstack {index}")
    if tp_groups.tp.size() == 1:
        # Gradients of the same loss over every output agree too (backward through the packed THD attention).
        cotangents = [torch.randn_like(merged, dtype=torch.float32) for _ in range(1 + len(deepstack))]
        expected = [reference.pooler_output, *reference.deepstack_features]
        sum((out.float() * c).sum() for out, c in zip(expected, cotangents, strict=True)).backward()
        sum((out.float() * c).sum() for out, c in zip([merged, *deepstack], cotangents, strict=True)).backward()
        hf_grads = vision_state_dict_from_hf({k: p.grad for k, p in hf.named_parameters()}, num_heads=HEADS)
        for name, param in megatron.named_parameters():
            _assert_close(hf_grads[name], param.grad, f"grad {name}")


def test_audio_matches_hugging_face(tp_groups):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.megatron_encoders import (
        Qwen3OmniAudioEncoder,
        audio_state_dict_from_hf,
    )

    config = Qwen3OmniMoeAudioEncoderConfig(
        d_model=128,
        encoder_attention_heads=HEADS,
        encoder_ffn_dim=256,
        encoder_layers=3,
        num_hidden_layers=3,
        output_dim=96,
        downsample_hidden_size=32,
    )
    config._attn_implementation = "sdpa"
    hf = _seeded(lambda: Qwen3OmniMoeAudioEncoder._from_config(config))
    megatron = Qwen3OmniAudioEncoder(config, _language_config(), tp_groups).cuda().bfloat16()
    full = audio_state_dict_from_hf(hf.state_dict(), num_heads=HEADS)
    _load(megatron, full, tp_groups)
    # A 15 s clip spans one 800-frame attention window and a shorter one; the second clip is partial.
    lengths = torch.tensor([1500, 730], device="cuda")
    torch.manual_seed(2)
    features = torch.randn(config.num_mel_bins, int(lengths.sum()), device="cuda", dtype=torch.bfloat16)
    reference = hf(features, feature_lens=lengths).last_hidden_state
    actual = megatron(features, feature_lens=lengths)
    _assert_close(reference, actual, "audio embeddings")
    if tp_groups.tp.size() == 1:
        cotangent = torch.randn_like(actual, dtype=torch.float32)
        (reference.float() * cotangent).sum().backward()
        (actual.float() * cotangent).sum().backward()
        hf_grads = audio_state_dict_from_hf({k: p.grad for k, p in hf.named_parameters()}, num_heads=HEADS)
        for name, param in megatron.named_parameters():
            _assert_close(hf_grads[name], param.grad, f"grad {name}")


def test_tensor_parallel_shards_and_replicated_params_are_marked(tp_groups):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig

    from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.megatron_encoders import Qwen3OmniAudioEncoder

    config = Qwen3OmniMoeAudioEncoderConfig(
        d_model=128,
        encoder_attention_heads=HEADS,
        encoder_ffn_dim=256,
        encoder_layers=1,
        num_hidden_layers=1,
        output_dim=96,
    )
    megatron = Qwen3OmniAudioEncoder(config, _language_config(), tp_groups)
    tp = tp_groups.tp.size()
    params = dict(megatron.named_parameters())
    assert params["decoder.layers.0.self_attention.linear_qkv.weight"].shape == (3 * 128 // tp, 128)
    assert params["decoder.layers.0.mlp.linear_fc2.weight"].shape == (128, 256 // tp)
    # Only parameters every TP rank holds in full average their gradients across TP.
    for name, param in params.items():
        sharded = getattr(param, "tensor_model_parallel", False)
        assert getattr(param, "average_gradients_across_tp_domain", False) is not sharded, name


@pytest.mark.parametrize("lengths", [(3, 5, 3, 2), (4, 4, 4)])
def test_packed_sdpa_attention_matches_per_sequence_attention(lengths):
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer import TransformerConfig

    from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.megatron_encoders import PackedSDPAttention

    config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=2, attention_dropout=0.0)
    attention = PackedSDPAttention(config, layer_number=1, attn_mask_type=None, attention_type="self")
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lengths))
    torch.manual_seed(0)
    q, k, v = (torch.randn(sum(lengths), 2, 8, device="cuda", requires_grad=True) for _ in range(3))
    actual = attention(q, k, v, None, packed_seq_params=packed)
    expected = []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        sq, sk, sv = (t[start:end].transpose(0, 1) for t in (q, k, v))
        expected.append(F.scaled_dot_product_attention(sq, sk, sv).transpose(0, 1))
    torch.testing.assert_close(actual, torch.cat(expected))
    # The cached plan serves the next layer; gradients flow to every token.
    attention(q, k, v, None, packed_seq_params=packed).sum().backward()
    assert q.grad.abs().sum(dim=(1, 2)).min() > 0
