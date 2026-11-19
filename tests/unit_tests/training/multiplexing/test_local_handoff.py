from types import SimpleNamespace

import pytest
import torch
from multiplexing_subject.config import MultiplexingConfig


pytestmark = pytest.mark.unit


def _runtime():
    """The runtime needs the full Bridge training stack; skip where it cannot be imported."""
    return pytest.importorskip("megatron.bridge.training.multiplexing.runtime")


def _adapters():
    return pytest.importorskip("megatron.bridge.training.multiplexing.adapters")


def test_skip_local_handoff_defaults_off():
    assert MultiplexingConfig().skip_local_handoff is False


def test_handoff_skip_never_touches_the_device(monkeypatch):
    runtime = _runtime()

    def fail():
        raise AssertionError("synchronized")

    monkeypatch.setattr(runtime.torch.cuda, "synchronize", fail)
    runtime._handoff(skip=True)
    runtime._handoff(SimpleNamespace(synchronize_handoff=fail), skip=True)


def test_handoff_without_skip_synchronizes(monkeypatch):
    runtime = _runtime()
    calls = []
    monkeypatch.setattr(runtime.torch.cuda, "synchronize", lambda: calls.append("device"))
    runtime._handoff()
    offload = SimpleNamespace(synchronize_handoff=lambda: calls.append("offload"))
    runtime._handoff(offload)
    assert calls == ["device", "offload"]


@pytest.mark.parametrize(
    "cross_grid,communicates,expected",
    [(False, [False], True), (True, [False], False), (False, [True], False), (False, [False, False], True)],
)
def test_boundaries_are_local(cross_grid, communicates, expected):
    runtime = _runtime()
    templates = [SimpleNamespace(needs_cross_grid_sync=lambda: cross_grid)]
    partitioner = [SimpleNamespace(needs_communication=value) for value in communicates]
    assert runtime._boundaries_are_local(templates, partitioner) is expected


def test_unknown_partitioner_is_assumed_to_communicate():
    runtime = _runtime()
    templates = [SimpleNamespace(needs_cross_grid_sync=lambda: False)]
    assert runtime._boundaries_are_local(templates, [object()]) is False


def test_fetched_batches_apply_the_language_transform():
    runtime = _runtime()
    batches = [{"x": 1}, {"x": 2}]
    iterator = runtime._FetchedBatches(batches, transform=lambda batch: {**batch, "seen": True})
    assert next(iterator) == {"x": 1, "seen": True}
    assert batches[0] == {"x": 1}  # the encoder's batch is untouched
    assert next(iterator) == {"x": 2, "seen": True}
    with pytest.raises(RuntimeError, match="has not consumed"):
        next(iterator)


def test_omni_language_batch_keeps_grids_and_replaces_raw_media():
    adapters = _adapters()
    adapter = adapters.Qwen3OmniAdapter.__new__(adapters.Qwen3OmniAdapter)
    batch = {
        "input_ids": torch.arange(4),
        "pixel_values": torch.ones(8, 3),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "input_features": torch.ones(1, 4, 6),
        "feature_attention_mask": torch.ones(1, 6, dtype=torch.long),
        "audio_feature_lengths": torch.tensor([6]),
        "pixel_values_videos": None,
    }
    out = adapter.language_batch(batch)
    assert out["pixel_values"].numel() == 0 and out["input_features"].numel() == 0
    for key in ("input_ids", "image_grid_thw", "feature_attention_mask", "audio_feature_lengths"):
        assert out[key] is batch[key]
    assert out["pixel_values_videos"] is None
    assert batch["pixel_values"].numel() == 24  # the encoder still sees the real tensor


def test_language_batch_is_a_noop_without_media():
    adapters = _adapters()
    adapter = adapters.Qwen3OmniAdapter.__new__(adapters.Qwen3OmniAdapter)
    batch = {"input_ids": torch.arange(4), "pixel_values": None, "input_features": None}
    assert adapter.language_batch(batch) is batch
