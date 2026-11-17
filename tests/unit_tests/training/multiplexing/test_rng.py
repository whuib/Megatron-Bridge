import pytest
import torch


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="EncoderRNG captures CUDA generator state"),
]


@pytest.fixture
def encoder_rng_cls():
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from multiplexing_subject.rng import EncoderRNG

    yield EncoderRNG
    get_cuda_rng_tracker().reset()


def _cpu_initialized_weight(rng) -> torch.Tensor:
    with rng.activate():
        return torch.nn.Embedding(8, 4).weight.detach().clone()


def test_cpu_initialized_encoder_modules_match_across_pipeline_stages(encoder_rng_cls):
    # Language pipeline stages seed and consume the global CPU generator differently.
    torch.manual_seed(1234)
    torch.rand(7)
    first_stage = _cpu_initialized_weight(encoder_rng_cls(seed=42, tp_rank=0))
    torch.manual_seed(1334)
    torch.rand(11)
    last_stage = _cpu_initialized_weight(encoder_rng_cls(seed=42, tp_rank=0))
    torch.testing.assert_close(first_stage, last_stage, rtol=0, atol=0)


def test_encoder_streams_leave_outer_streams_untouched(encoder_rng_cls):
    torch.manual_seed(5)
    expected_cpu = torch.rand(3)
    expected_cuda = torch.rand(3, device="cuda")
    torch.manual_seed(5)
    rng = encoder_rng_cls(seed=42, tp_rank=0)
    with rng.activate():
        torch.rand(100)
        torch.rand(100, device="cuda")
    torch.testing.assert_close(torch.rand(3), expected_cpu, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(3, device="cuda"), expected_cuda, rtol=0, atol=0)


def test_encoder_stream_advances_between_activations(encoder_rng_cls):
    rng = encoder_rng_cls(seed=42, tp_rank=0)
    with rng.activate():
        first = torch.rand(4)
    with rng.activate():
        second = torch.rand(4)
    assert not torch.equal(first, second)
