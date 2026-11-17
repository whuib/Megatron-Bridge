"""Unit tests for the Qwen3-VL native / DistTrain / multiplexing throughput comparison helpers."""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples/models/qwen/qwen3_vl"
pytestmark = pytest.mark.unit


def _load(name: str):
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"qwen3_vl_{name}_under_test", EXAMPLE_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))


@pytest.fixture(scope="module")
def compare():
    return _load("compare_training")


@pytest.fixture(scope="module")
def throughput():
    return _load("throughput")


def _args(**overrides):
    values = dict(
        mode="native",
        language_tp=1,
        language_pp=2,
        encoder_tp=1,
        vision_gpus=4,
        global_batch_size=32,
        micro_batch_size=2,
        num_images=8,
        sequence_parallel=False,
        vision_recompute="inherit",
        mux_encoder_schedule="deferred",
    )
    values.update(overrides)
    return Namespace(**values)


def test_window_summary(throughput):
    workload = throughput.Workload(global_batch_size=32, seq_length=8192, images_per_sample=8, patches_per_image=256)
    summary = throughput.window_summary(
        window_seconds=100.0, measure_iters=10, workload=workload, flops_per_iter=8 * 10**15, world_size=8
    )
    assert summary["step_seconds"] == pytest.approx(10.0)
    assert summary["samples_per_second"] == pytest.approx(3.2)
    assert summary["padded_tokens_per_second"] == pytest.approx(32 * 8192 / 10.0)
    assert summary["images_per_second"] == pytest.approx(25.6)
    assert summary["model_tflops_per_gpu"] == pytest.approx(100.0)
    with pytest.raises(ValueError):
        throughput.window_summary(
            window_seconds=0.0, measure_iters=10, workload=workload, flops_per_iter=1, world_size=8
        )


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("native", {"vision_dp": 4, "language_dp": 4, "vision_gpus": 8, "language_gpus": 8}),
        ("disttrain", {"vision_dp": 4, "language_dp": 2, "vision_gpus": 4, "language_gpus": 4}),
        ("multiplexing", {"vision_dp": 8, "language_dp": 4, "vision_gpus": 8, "language_gpus": 8}),
    ],
)
def test_parallel_layout_per_mode(compare, mode, expected):
    layout = compare.parallel_layout(_args(mode=mode), world_size=8)
    assert {key: layout[key] for key in expected} == expected


def test_parallel_layout_rejects_non_multiple_vision_dp(compare):
    # Language PP2 on 6 GPUs gives DP3; two vision GPUs cannot feed it evenly.
    with pytest.raises(ValueError, match="multiple of language DP"):
        compare.parallel_layout(_args(mode="disttrain", vision_gpus=2), world_size=8)


@pytest.mark.parametrize("mode", ["disttrain", "multiplexing"])
def test_workload_requires_whole_equal_image_shards(compare, mode):
    args = _args(mode=mode, micro_batch_size=1, num_images=1)
    with pytest.raises(ValueError, match="divisible by vision DP / language DP"):
        compare.validate_workload(args, compare.parallel_layout(args, world_size=8))
    args = _args(mode=mode)
    compare.validate_workload(args, compare.parallel_layout(args, world_size=8))


def test_sequence_parallel_requires_tensor_parallel(compare):
    with pytest.raises(ValueError, match="requires --language-tp > 1"):
        compare.parallel_layout(_args(sequence_parallel=True), world_size=8)
    layout = compare.parallel_layout(_args(language_tp=2, language_pp=1, sequence_parallel=True), world_size=8)
    assert layout["sequence_parallel"] == 1 and layout["language_dp"] == 4


def test_native_has_no_split_requirement(compare):
    args = _args(mode="native", micro_batch_size=1, num_images=1, global_batch_size=8)
    compare.validate_workload(args, compare.parallel_layout(args, world_size=8))


@pytest.mark.parametrize(
    "policy,granularity,method", [("full", "full", "uniform"), ("selective", "selective", None), ("none", None, None)]
)
def test_recompute_policies(compare, policy, granularity, method):
    model = SimpleNamespace(recompute_granularity="x", recompute_method="x", recompute_num_layers=3)
    compare.apply_recompute(model, policy)
    assert model.recompute_granularity == granularity
    assert model.recompute_method == method
    with pytest.raises(ValueError):
        compare.apply_recompute(model, "partial")


def test_interleaved_encoder_schedule_requires_single_language_stage(compare):
    with pytest.raises(ValueError, match="language-pp 1"):
        compare.parallel_layout(_args(mode="multiplexing", mux_encoder_schedule="interleaved"), world_size=8)
    layout = compare.parallel_layout(
        _args(mode="multiplexing", language_pp=1, mux_encoder_schedule="interleaved", vision_recompute="none"),
        world_size=8,
    )
    assert layout["mux_encoder_schedule"] == "interleaved" and layout["vision_recompute"] == "none"
    # The schedule is a multiplexing knob; other modes record it as not applicable.
    assert compare.parallel_layout(_args(mux_encoder_schedule="interleaved"), world_size=8)[
        "mux_encoder_schedule"
    ] == ("n/a")


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("inherit", ("inherit", None, None, None)),
        ("full", ("full", "uniform", 1, None)),
        ("selective", ("selective", None, None, ["core_attn"])),
        ("none", (None, None, None, None)),
    ],
)
def test_vision_recompute_policies(compare, policy, expected):
    model = SimpleNamespace(
        vision_recompute_granularity="x",
        vision_recompute_method="x",
        vision_recompute_num_layers=3,
        vision_recompute_modules=["mlp"],
    )
    compare.apply_vision_recompute(model, policy)
    assert (
        model.vision_recompute_granularity,
        model.vision_recompute_method,
        model.vision_recompute_num_layers,
        model.vision_recompute_modules,
    ) == expected
    with pytest.raises(ValueError):
        compare.apply_vision_recompute(model, "partial")


def _record(mode: str, step_seconds: float, **arg_overrides):
    args = {"mode": mode, "seq_length": 8192, "language_pp": 2, "num_layers": 12}
    args.update(arg_overrides)
    return {
        "_source": f"{mode}.json",
        "metadata": {
            "mode": mode,
            "label": "benchmark-only",
            "layout": {"language_pp": args["language_pp"]},
            "args": args,
            "versions": {"torch": "2.14.1"},
            "gpu": "A100",
            "git": {"commit": "abc", "dirty": False},
        },
        "workload": {"global_batch_size": 32},
        "world_size": 8,
        "flops_per_iteration": 10,
        "measure_iters": 40,
        "warmup_iters": 10,
        "step_seconds": step_seconds,
        "samples_per_second": 32 / step_seconds,
        "model_tflops_per_gpu": 1.0,
        "max_peak_allocated_gib": 30.0,
        "loss_and_grad_finite_on_all_ranks": True,
    }


def test_summary_allows_layout_differences_only():
    summarize = _load("summarize_throughput")
    records = [_record("native", 10.0), _record("multiplexing", 8.0, language_pp=1)]
    assert summarize.contract_differences(records) == []
    rows = {row["mode"]: row for row in summarize.summarize(records)}
    assert rows["multiplexing"]["speedup_vs_native"] == pytest.approx(1.25)
    mismatched = [_record("native", 10.0), _record("multiplexing", 8.0, seq_length=4096)]
    assert any("args differs" in item for item in summarize.contract_differences(mismatched))
    # TP-dependent vocabulary padding changes FLOPs slightly; a different workload changes them a lot.
    padded = _record("multiplexing", 8.0)
    padded["flops_per_iteration"] = 10 * (1 + 1e-5)
    assert summarize.contract_differences([_record("native", 10.0), padded]) == []
    padded["flops_per_iteration"] = 11
    assert any(
        "flops_per_iteration" in item for item in summarize.contract_differences([_record("native", 10.0), padded])
    )


def test_summary_attaches_and_filters_foreign_gpu_work(tmp_path, monkeypatch):
    import json

    summarize = _load("summarize_throughput")
    for name, step, foreign in (("native.rep1", 10.0, 0.2), ("native.rep2", 13.0, 7.5)):
        (tmp_path / f"{name}.json").write_text(json.dumps(_record("native", step)))
        (tmp_path / f"{name}.interference.json").write_text(json.dumps({"max_foreign_mean_sm_percent": foreign}))
    monkeypatch.setattr(sys, "argv", ["summarize", str(tmp_path), "--max-foreign-sm", "1.0"])
    assert summarize.main() == 0
    (row,) = json.loads((tmp_path / "summary.json").read_text())
    # The contaminated repetition is excluded; the interference record itself is not a throughput record.
    assert row["runs"] == 1 and row["median_step_seconds"] == 10.0
    assert row["max_foreign_mean_sm_percent"] == pytest.approx(0.2)


def test_gpu_activity_process_ancestry():
    import os

    activity = _load("gpu_activity")
    assert activity.descends_from(os.getpid(), os.getppid())
    assert activity.descends_from(os.getpid(), os.getpid())
    assert not activity.descends_from(os.getppid(), os.getpid())


def test_foreign_memory_at_oom_from_pytorch_message(tmp_path):
    activity = _load("gpu_activity")
    named = (
        "GPU 4 has a total capacity of 79.25 GiB of which 78.94 MiB is free. Process 904748 has 494.00 MiB memory "
        "in use. Including non-PyTorch memory, this process has 78.68 GiB memory in use."
    )
    unnamed = (
        "GPU 3 has a total capacity of 79.25 GiB of which 8.94 MiB is free. Including non-PyTorch memory, "
        "this process has 76.91 GiB memory in use."
    )
    assert round(activity.foreign_at_oom_mib(named)) == round(0.57 * 1024 - 78.94)
    assert round(activity.foreign_at_oom_mib(named + "\n" + unnamed)) == round(2.34 * 1024 - 8.94)
    assert activity.foreign_at_oom_mib("ncclUnhandledCudaError") is None
    log, monitor = tmp_path / "run.log", tmp_path / "run.interference.json"
    log.write_text("Cuda failure 2 'out of memory'")
    monitor.write_text('{"foreign_processes": {"1": {"max_fb": 3038}, "2": {"max_fb": 494}}}')
    assert activity.oom_foreign_mib(log, monitor) == 3038
