"""Two-GPU Qwen3-VL multiplexing smoke case for a configured Bridge environment."""

import importlib.util
import json
import os
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples/models/qwen/qwen3_vl"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"qwen3_vl_{name}_under_test", EXAMPLE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    "encoder_schedule,vision_recompute",
    # Interleaving bounds encoder activations to one micro-batch, so it runs without vision recompute
    # (the DeepStack features then carry autograd history and reach the encoder backward).
    [("deferred", "inherit"), ("interleaved", "none")],
)
def test_qwen3_vl_multiplexing_smoke(tmp_path, encoder_schedule, vision_recompute):
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Run with two torch.distributed processes")
    from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.training.pretrain import pretrain

    case = _load("compare_training")
    args = Namespace(
        mode="multiplexing",
        hf_model="Qwen/Qwen3-VL-8B-Instruct",
        language_tp=2,
        language_pp=1,
        encoder_tp=1,
        sequence_parallel=False,
        vision_gpus=1,
        seq_length=1024,
        global_batch_size=4,
        micro_batch_size=1,
        train_iters=3,
        num_layers=4,
        num_images=2,
        image_size=128,
        seed=1234,
        recompute="full",
        vision_recompute=vision_recompute,
        mux_encoder_schedule=encoder_schedule,
        num_workers=1,
        log_interval=1,
        gradient_accumulation_fusion=importlib.util.find_spec("fused_weight_gradient_mlp_cuda") is not None,
    )
    cfg = case.build_config(args)

    events = Counter()

    class CountEvents(Callback):
        def on_train_start(self, context):
            events["on_train_start"] += 1

        def on_train_step_start(self, context):
            events["on_train_step_start"] += 1

        def on_train_step_end(self, context):
            # Native semantics: the step counter advances after on_train_step_end.
            assert context.state.train_state.step == events["on_train_step_end"]
            events["on_train_step_end"] += 1

        def on_train_end(self, context):
            events["on_train_end"] += 1

    results = tmp_path / "throughput.json"
    meter = case.ThroughputMeter(
        warmup_iters=1,
        measure_iters=2,
        workload=case.Workload(
            global_batch_size=args.global_batch_size,
            seq_length=args.seq_length,
            images_per_sample=args.num_images,
            patches_per_image=case.patches_per_image(cfg.dataset, args.hf_model),
        ),
        results_path=str(results),
        metadata={"mode": args.mode},
    )
    pretrain(cfg, forward_step, callbacks=[CountEvents(), meter])

    assert events == {"on_train_start": 1, "on_train_step_start": 3, "on_train_step_end": 3, "on_train_end": 1}
    # The runtime destroys a process group it created, so identify rank 0 from the launcher.
    if int(os.environ["RANK"]) == 0:
        record = json.loads(results.read_text())
        assert record["measure_iters"] == 2
        assert record["step_seconds"] > 0
        assert record["model_tflops_per_gpu"] > 0
        assert record["loss_and_grad_finite_on_all_ranks"]
        assert len(record["peak_allocated_gib_per_rank"]) == 2
