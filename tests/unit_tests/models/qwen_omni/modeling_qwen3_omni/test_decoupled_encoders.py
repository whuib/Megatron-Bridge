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

"""Decoupled Qwen3-Omni encoders: DistTrain shard packing, external features and multi-clip audio."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.rope import _get_feat_extract_output_lengths
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model import (
    OmniEncoderFeatures,
    Qwen3OmniThinkerModel,
    _audio_chunk_bounds,
    _image_chunk_bounds,
    flatten_audio_features,
)


pytestmark = pytest.mark.unit

HIDDEN = 8
N_DEEPSTACK = 3


def _thinker(ratio: int, received: torch.Tensor | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        vision_to_llm_dp_ratio=ratio,
        n_deepstack=N_DEEPSTACK,
        config=SimpleNamespace(spatial_merge_size=2),
        thinker_transformer_config=SimpleNamespace(audio_config=SimpleNamespace(n_window=50)),
        _dist_train_received=received,
        audio_model=None,
        _current_features=None,
    )


def test_chunk_bounds_split_items_equally_and_reject_uneven_shards():
    grid = torch.tensor([[1, 4, 4], [1, 2, 2], [1, 4, 4], [1, 2, 2]])
    assert _image_chunk_bounds(grid, 2, 1) == (2, 4, 20, 40)
    assert _audio_chunk_bounds(6, 3, 2) == (4, 6)
    with pytest.raises(ValueError, match="equal encoder shards"):
        _image_chunk_bounds(grid, 3, 0)
    with pytest.raises(ValueError, match="equal encoder shards"):
        _audio_chunk_bounds(5, 2, 0)


def test_flatten_audio_features_keeps_only_valid_frames():
    features = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    lengths = torch.tensor([5, 2])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    from_mask = flatten_audio_features(features, mask, lengths)
    from_lengths = flatten_audio_features(features, None, lengths)
    assert from_mask.shape == (3, 7)
    assert torch.equal(from_mask, from_lengths)
    assert torch.equal(from_mask[:, 5:], features[1, :, :2])


@pytest.mark.parametrize("ratio", [1, 2])
def test_dist_train_unpack_restores_encoder_shard_order(ratio):
    # Two images of 4x4 patches (4 merged tokens each) and two 2-chunk audio clips per micro-batch.
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    lengths = torch.tensor([200, 200])
    audio_tokens = int(_get_feat_extract_output_lengths(lengths[:1]).item())
    vision = torch.randn(8, HIDDEN)
    deepstack = [torch.randn(8, HIDDEN) for _ in range(N_DEEPSTACK)]
    audio = torch.randn(2 * audio_tokens, HIDDEN)
    shards = []
    for index in range(ratio):
        v_rows, a_rows = 8 // ratio, 2 * audio_tokens // ratio
        v_slice, a_slice = slice(index * v_rows, (index + 1) * v_rows), slice(index * a_rows, (index + 1) * a_rows)
        # The encoder rank's packing: vision, DeepStack features, audio, as [1, rows, hidden].
        parts = [vision[v_slice], *(feature[v_slice] for feature in deepstack), audio[a_slice]]
        shards.append(torch.cat(parts).unsqueeze(0))
    received = torch.cat(shards, dim=0)  # fan-in stacks the encoder shards on the batch dimension

    features = Qwen3OmniThinkerModel._dist_train_unpack(_thinker(ratio, received), grid, lengths)

    assert torch.equal(features.vision_embeds, vision)
    assert all(torch.equal(got, want) for got, want in zip(features.deepstack_embeds, deepstack, strict=True))
    assert torch.equal(features.audio_embeds, audio)


def test_dist_train_unpack_rejects_mismatched_shard_rows():
    grid = torch.tensor([[1, 4, 4]])
    received = torch.randn(1, 5, HIDDEN)
    with pytest.raises(ValueError, match="rows"):
        Qwen3OmniThinkerModel._dist_train_unpack(_thinker(1, received), grid, None)


def test_external_audio_features_allow_several_clips_per_sample():
    lengths = torch.tensor([100, 100, 100, 100])  # four clips, two per sample
    per_clip = int(_get_feat_extract_output_lengths(lengths[:1]).item())
    audio = torch.randn(4 * per_clip, HIDDEN)
    thinker = _thinker(1)
    thinker._current_features = OmniEncoderFeatures(None, [], audio)
    features = torch.zeros(4, 3, 100)

    got = Qwen3OmniThinkerModel.get_audio_features(
        thinker,
        features,
        audio_feature_lengths=lengths,
        expected_audio_token_counts=torch.tensor([2 * per_clip, 2 * per_clip]),
    )
    assert got is audio
    with pytest.raises(ValueError, match="several clips per sample"):
        Qwen3OmniThinkerModel.get_audio_features(
            thinker,
            features,
            audio_feature_lengths=lengths,
            expected_audio_token_counts=torch.tensor([2 * per_clip, 2 * per_clip - 1]),
        )
