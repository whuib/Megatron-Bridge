import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from multiplexing_subject.bridge.input_bridge import InputPartitioner, split_vision_by_images
from multiplexing_subject.bridge.layout_bridge import LayoutBridge
from multiplexing_subject.bridge.payload import BoundaryPayload
from multiplexing_subject.layout import ModuleParallelLayout


pytestmark = pytest.mark.unit


def test_image_sharding_respects_patch_boundaries():
    grid = torch.tensor([[1, 2, 2], [1, 3, 2], [1, 2, 3]])
    data = torch.arange(16).reshape(16, 1)
    shards = [split_vision_by_images(data, grid, 2, rank) for rank in range(2)]
    assert [len(part[0]) for part in shards] == [10, 6]
    torch.testing.assert_close(torch.cat([part[0] for part in shards]), data)


@pytest.fixture(scope="module")
def world():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("launch this file with two distributed processes")
    backend = os.environ.get("MULTIPLEXING_TEST_BACKEND", "nccl")
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    yield device
    dist.destroy_process_group()


@pytest.mark.parametrize("language_pp", [1, 2])
@pytest.mark.parametrize("replicated", [False, True])
def test_forward_backward_and_dp_scaling(world, language_pp, replicated):
    rank = dist.get_rank()
    encoder = ModuleParallelLayout("encoder", 2, dp_size=2, dimension_order="tp-cp-dp-pp")
    language = ModuleParallelLayout(
        "llm", 2, tp_size=2 // language_pp, pp_size=language_pp, dimension_order="tp-cp-dp-pp"
    )
    partition = InputPartitioner(language, encoder, device=world)
    bridge = LayoutBridge(
        encoder, language, hidden_size=2, n_deepstack=1, device=world, replicated_destination_grads=replicated
    )
    # Repeat with two different shapes to detect stale transfer metadata.
    for offset, image_lengths in ((0, (2, 3)), (10, (4, 3))):
        # Two variable-size images in the one logical language batch.
        pixels = torch.arange(2 * sum(image_lengths), device=world, dtype=torch.float32).reshape(-1, 2)
        grid = torch.tensor([[1, 1, size] for size in image_lengths], device=world)
        local_pixels, local_grid = partition.scatter_vision_inputs(pixels, grid)
        expected_pixels = pixels[: image_lengths[0]] if rank == 0 else pixels[image_lengths[0] :]
        torch.testing.assert_close(local_pixels, expected_pixels)
        torch.testing.assert_close(local_grid, grid[rank : rank + 1])
        weight = torch.tensor(0.75, device=world, requires_grad=True)
        output = (local_pixels + offset) * weight
        deep = output * 2
        bridge.forward_send(BoundaryPayload(output.detach(), [deep.detach()], n_merged_tokens=len(output)))
        received = bridge.forward_recv()
        supplied = None
        if received is not None:
            expected = (pixels + offset) * weight.detach()
            torch.testing.assert_close(received.vision_embeds, expected)
            torch.testing.assert_close(received.deepstack_features[0], expected * 2)
            scale = 1 if replicated else 1 / language.tp_size
            # Model a TP-replicated or TP-partial destination derivative.
            (received.vision_embeds.square().sum() * scale).backward()
            supplied = received.grad_tensors()  # Unused DeepStack must remain a zero slot.
        bridge.backward_send(supplied)
        grads = bridge.backward_recv()
        assert len(grads) == 2
        torch.testing.assert_close(grads[0], 2 * output.detach())
        torch.testing.assert_close(grads[1], torch.zeros_like(deep))
        # Different DP layouts must reproduce a global-batch parameter update.
        torch.autograd.backward([output, deep], [grad * 2 for grad in grads])
        dist.all_reduce(weight.grad)
        weight.grad.div_(2)
        reference = torch.tensor(0.75, device=world, requires_grad=True)
        (((pixels + offset) * reference).square().sum()).backward()
        torch.testing.assert_close(weight.grad, reference.grad)
        torch.testing.assert_close(weight - 0.01 * weight.grad, reference - 0.01 * reference.grad)
    dist.barrier()
