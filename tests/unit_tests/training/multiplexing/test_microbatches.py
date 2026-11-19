import pytest
import torch
from multiplexing_subject.microbatches import collate_microbatches, scale_microbatch_loss


pytestmark = pytest.mark.unit


def test_collate_splits_multimodal_samples_before_processing():
    samples = [{"image": sample, "audio": -sample} for sample in range(32)]

    def collate(group):
        return {key: [sample[key] for sample in group] for key in ("image", "audio")}

    first, second = collate_microbatches(samples, sizes=(24, 8), collate=collate)
    assert (first["image"], second["image"]) == (list(range(24)), list(range(24, 32)))
    assert first["audio"] + second["audio"] == [-sample for sample in range(32)]
    with pytest.raises(ValueError, match="Expected 32 samples"):
        collate_microbatches(samples[:-1], sizes=(24, 8), collate=collate)


def test_step_token_normalization_matches_unsplit_gradient():
    parameter = torch.tensor(2.0, requires_grad=True)
    groups = (torch.tensor([1.0, 1.5, 4.0]), torch.tensor([9.0]), torch.empty(0))
    step_tokens = sum(group.numel() for group in groups)
    expected = torch.autograd.grad((parameter * torch.cat(groups)).mean(), parameter)[0]
    for group in groups:
        report = {"lm loss": group.sum()}

        def loss_func(output):
            return output.sum(), torch.tensor(output.numel()), report

        loss, num_tokens, returned = scale_microbatch_loss(
            parameter * group, loss_func=loss_func, count=len(groups), step_tokens=step_tokens
        )
        (loss / num_tokens.clamp(min=1) / len(groups)).backward()
        assert returned is report
    torch.testing.assert_close(parameter.grad, expected)
