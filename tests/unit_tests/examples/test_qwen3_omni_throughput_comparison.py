"""Unit tests for the Qwen3-Omni native / DistTrain / multiplexing throughput comparison helpers."""

import importlib.util
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3] / "examples/models/qwen"
pytestmark = pytest.mark.unit


def _load(directory: str, name: str):
    paths = [str(ROOT / "qwen3_omni"), str(ROOT / "qwen3_vl")]
    sys.path[:0] = paths
    try:
        spec = importlib.util.spec_from_file_location(f"omni_{name}_under_test", ROOT / directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for path in paths:
            sys.path.remove(path)


@pytest.fixture(scope="module")
def compare():
    return _load("qwen3_omni", "compare_training")


@pytest.fixture(scope="module")
def benchmark():
    return _load("qwen3_omni", "omni_benchmark")


def _args(**overrides):
    values = dict(
        mode="native",
        model_size="1b-1b-3b",
        language_tp=1,
        language_pp=1,
        language_first_stage_layers=None,
        encoder_implementation="huggingface",
        encoder_tp=1,
        sequence_parallel=False,
        vision_gpus=4,
        mux_encoder_schedule="interleaved",
        mux_module_state_offload=False,
        mux_module_state_offload_overlap=False,
        mux_module_state_offload_budget=False,
        mux_module_state_offload_resident_param_mib=0,
        mux_module_state_offload_resident_optim_mib=0,
        mux_module_state_offload_prefetch_mib=0,
        mux_module_state_offload_writeback_chunk_mib=0,
        mux_module_state_offload_scoped_handoff=False,
        seq_length=4096,
        global_batch_size=32,
        micro_batch_size=1,
        media_per_sample=None,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "mode,expected_encoder_dp,expected_language_dp",
    # Native towers follow the language DP (TP ranks replicate them); multiplexing uses every GPU;
    # DistTrain uses its 4 encoder GPUs against the 4-GPU TP2 language model.
    [("native", 4, 4), ("multiplexing", 8, 4), ("disttrain", 4, 2)],
)
def test_encoder_and_language_dp_per_mode(compare, mode, expected_encoder_dp, expected_language_dp):
    layout = compare.parallel_layout(_args(mode=mode, language_tp=2, sequence_parallel=True), world_size=8)
    assert (layout["encoder_dp"], layout["language_dp"]) == (expected_encoder_dp, expected_language_dp)


@pytest.mark.parametrize(
    "mode,overrides,expected",
    # (encoder TP, encoder DP, language DP): native Megatron towers shard over the language TP; DistTrain and
    # multiplexing take --encoder-tp.
    [
        ("native", dict(language_tp=2), (2, 4, 4)),
        ("multiplexing", dict(language_tp=4, encoder_tp=2), (2, 4, 2)),
        ("disttrain", dict(language_tp=2, encoder_tp=2, vision_gpus=4), (2, 2, 2)),
    ],
)
def test_megatron_towers_shard_over_the_encoder_tp(compare, mode, overrides, expected):
    args = _args(mode=mode, model_size="3b-3b-9b", encoder_implementation="megatron", **overrides)
    layout = compare.parallel_layout(args, world_size=8)
    assert (layout["encoder_tp"], layout["encoder_dp"], layout["language_dp"]) == expected


def test_encoder_tp_needs_megatron_towers_and_divides_heads(compare):
    with pytest.raises(ValueError, match="megatron"):
        compare.parallel_layout(_args(mode="multiplexing", encoder_tp=2), world_size=8)
    with pytest.raises(ValueError, match="language TP"):
        compare.parallel_layout(_args(mode="native", encoder_tp=2, encoder_implementation="megatron"), world_size=8)
    # The 1b-1b-3b vision tower has 16 heads: TP 8 divides them, a 3-way split would not.
    layout = compare.parallel_layout(
        _args(mode="multiplexing", language_tp=8, encoder_tp=8, encoder_implementation="megatron"), world_size=8
    )
    assert layout["encoder_dp"] == 1


def test_media_scales_with_sequence_and_shards_must_be_equal(compare):
    assert compare.media_per_sample(_args(seq_length=8192)) == 8
    assert compare.media_per_sample(_args(media_per_sample=3)) == 3
    args = _args(mode="disttrain", vision_gpus=6, language_tp=1, global_batch_size=32, seq_length=4096)
    with pytest.raises(ValueError):
        compare.validate_workload(args, compare.parallel_layout(args, world_size=8))
    with pytest.raises(ValueError, match="language-pp 1"):
        compare.parallel_layout(_args(mode="multiplexing", language_pp=2), world_size=8)


def test_uneven_first_stage_split_is_validated(compare):
    layout = compare.parallel_layout(_args(language_pp=2, language_first_stage_layers=6), world_size=8)
    assert layout["language_first_stage_layers"] == 6 and layout["language_dp"] == 4
    with pytest.raises(ValueError, match="language-first-stage-layers"):
        compare.parallel_layout(_args(language_first_stage_layers=6), world_size=8)
    with pytest.raises(ValueError, match="divisible"):
        compare.parallel_layout(_args(language_pp=3, language_first_stage_layers=5), world_size=6)


def test_unsupported_reason_rejects_before_launch(compare):
    workload = ["--hf-model", "unused", "--model-size", "3b-3b-9b", "--seq-length", "4096"]
    # Multiplexing TP8 has language DP 1 and encoder DP 8: 4 images per micro-batch cannot fill 8 shards.
    reason = compare.unsupported_reason(["--mode", "multiplexing", "--language-tp", "8", *workload], world_size=8)
    assert reason is not None and "divisible by 8" in reason
    native = ["--mode", "native", "--language-pp", "2", "--language-first-stage-layers", "6", *workload]
    assert compare.unsupported_reason(native, world_size=8) is None


def test_scoped_offload_handoffs_require_matching_layouts(compare):
    args = _args(
        mode="multiplexing",
        language_tp=4,
        encoder_tp=2,
        encoder_implementation="megatron",
        mux_module_state_offload=True,
        mux_module_state_offload_overlap=True,
        mux_module_state_offload_scoped_handoff=True,
    )
    with pytest.raises(ValueError, match="identical encoder/language TP"):
        compare.parallel_layout(args, world_size=8)
    args.encoder_tp = 4
    assert compare.parallel_layout(args, world_size=8)["mux_module_state_offload_scoped_handoff"] == 1


def test_model_sizes_match_their_names(benchmark):
    torch = pytest.importorskip("torch")
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
        Qwen3OmniMoeVisionEncoderConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
        Qwen3OmniMoeVisionEncoder,
    )

    expected = {"1b-1b-3b": (0.97e9, 0.98e9, 3.04e9), "3b-3b-9b": (2.99e9, 3.03e9, 9.00e9)}
    for name, (vision_params, audio_params, llm_params) in expected.items():
        size = benchmark.MODEL_SIZES[name]
        with torch.device("meta"):
            vision = Qwen3OmniMoeVisionEncoder._from_config(Qwen3OmniMoeVisionEncoderConfig(**size["vision"]))
            audio = Qwen3OmniMoeAudioEncoder._from_config(Qwen3OmniMoeAudioEncoderConfig(**size["audio"]))
        llm = size["llm"]
        h, q, kv = (
            llm["hidden_size"],
            llm["num_attention_heads"] * llm["head_dim"],
            llm["num_key_value_heads"] * llm["head_dim"],
        )
        layer = 2 * h * q + 2 * h * kv + 3 * h * llm["intermediate_size"]
        llm_total = llm["num_hidden_layers"] * layer + 2 * 152064 * h  # untied embeddings, Qwen3-Omni vocabulary
        for actual, wanted in (
            (sum(p.numel() for p in vision.parameters()), vision_params),
            (sum(p.numel() for p in audio.parameters()), audio_params),
            (llm_total, llm_params),
        ):
            assert math.isclose(actual, wanted, rel_tol=0.02), (name, actual, wanted)


def test_planner_infers_oom_and_records_unsupported_cases(tmp_path):
    planner = _load("qwen3_omni", "plan_omni_grid")
    summarize = _load("qwen3_omni", "summarize_omni_grid")
    small = tmp_path / "seq1024_gbs16"
    small.mkdir()
    (small / "mux.rep1.log").write_text("torch.OutOfMemoryError: CUDA out of memory.")
    (small / "nat.rep1.log").write_text("THROUGHPUT ...")
    (small / "nat.rep1.json").write_text("{}")
    cases = [("nat", "native", "--language-pp 2"), ("mux", "multiplexing", "--language-tp 4"), ("dt", "disttrain", "")]

    def reason(argv):
        return "needs more images" if argv[1] == "disttrain" else None

    launch = planner.plan_point(tmp_path, cases, seq=2048, gbs=32, workload_args="--seq-length 2048", reason_fn=reason)
    assert [name for name, _, _ in launch] == ["nat"]
    point = tmp_path / "seq2048_gbs32"
    assert (point / "mux.rep1.oom_inferred").read_text().strip() == "OOM measured at seq 1024, GBS 16"
    assert (point / "dt.rep1.unsupported").read_text().strip() == "needs more images"
    # An OOM inside NCCL's own allocation counts as well; a finished run is not launched again.
    (small / "dt.rep1.log").write_text(
        "ncclUnhandledCudaError\nLast error:\nCuda failure 2 'out of memory'\nexitcode  : 1"
    )
    launch = planner.plan_point(tmp_path, cases, seq=4096, gbs=16, workload_args="", reason_fn=lambda argv: None)
    assert [name for name, _, _ in launch] == ["nat"]
    assert planner.plan_point(tmp_path, cases, seq=1024, gbs=16, workload_args="", reason_fn=lambda argv: None) == []
    # An OOM next to 3 GiB of another process's memory is neither inferred from nor reported as the case's.
    foreign = tmp_path / "seq2048_gbs16"
    foreign.mkdir()
    (foreign / "nat.rep1.log").write_text(
        "torch.OutOfMemoryError: CUDA out of memory. GPU 3 has a total capacity of 79.25 GiB of which 8.94 MiB is "
        "free. Including non-PyTorch memory, this process has 76.21 GiB memory in use."
    )
    assert planner.measured_oom(tmp_path, "nat", 4096, 64) is None
    assert summarize._failure(foreign / "nat.rep1.log") == "OOM with 3.0 GiB foreign GPU memory present"
    for path in foreign.iterdir():
        path.unlink()
    foreign.rmdir()
    # A smaller global batch is not implied by an OOM at a larger one.
    launch = planner.plan_point(tmp_path, cases, seq=2048, gbs=8, workload_args="", reason_fn=lambda argv: None)
    assert [name for name, _, _ in launch] == ["nat", "mux", "dt"]
    (small / "dt.rep1.log").unlink()

    summarize = _load("qwen3_omni", "summarize_omni_grid")
    for placeholder in small.glob("nat.rep1.*"):  # not a real throughput record
        placeholder.unlink()
    rows = {(row["seq"], row["case"]): row["status"] for row in summarize.collect(tmp_path) if row["gbs"] != 8}
    assert rows[(2048, "mux")] == "OOM (inferred: OOM measured at seq 1024, GBS 16)"
    assert rows[(2048, "dt")] == "unsupported: needs more images"
    assert rows[(1024, "mux")] == "OOM"


def test_audio_encoder_flops_match_hand_count(benchmark):
    config = SimpleNamespace(
        d_model=8,
        encoder_ffn_dim=16,
        downsample_hidden_size=4,
        num_mel_bins=16,
        n_window=4,
        n_window_infer=16,
        encoder_layers=2,
        output_dim=6,
    )
    # One 8-frame chunk: conv outputs 8x4 -> 4x2 -> 2x1 (mel x time), one token, one window.
    conv = 2 * 9 * 1 * 4 * 8 * 4 + 2 * 9 * 4 * 4 * 4 * 2 + 2 * 9 * 4 * 4 * 2 * 1 + 2 * 4 * 2 * 8 * 1
    layers = 2 * (1 * (8 * 64 + 4 * 8 * 16) + 4 * 1 * 1 * 8)
    head = 2 * 64 + 2 * 8 * 6
    assert benchmark.audio_encoder_flops(config, 8) == 3 * (conv + layers + head)
    # Two chunks form one attention window of two tokens.
    assert benchmark.audio_encoder_flops(config, 16) == 3 * (
        2 * conv + 2 * (2 * (8 * 64 + 4 * 8 * 16)) + 2 * (4 * 4 * 8) + 2 * head
    )


def test_peak_tflops_by_gpu_name(benchmark):
    assert benchmark.peak_tflops("NVIDIA A100-SXM4-80GB") == 312.0
    with pytest.raises(ValueError):
        benchmark.peak_tflops("Unknown GPU")


def test_grid_summary_keeps_oom_and_measurements(tmp_path):
    summarize = _load("qwen3_omni", "summarize_omni_grid")
    point = tmp_path / "seq4096_gbs32"
    point.mkdir()
    record = {
        "metadata": {"mode": "native", "layout": {"language_tp": 1}},
        "step_seconds": 2.0,
        "samples_per_second": 16.0,
        "padded_tokens_per_second": 65536.0,
        "model_tflops_per_gpu": 90.0,
        "mfu": 90.0 / 312.0,
        "max_peak_allocated_gib": 60.0,
        "loss_and_grad_finite_on_all_ranks": True,
    }
    (point / "nat.rep1.log").write_text("THROUGHPUT ...")
    (point / "nat.rep1.json").write_text(json.dumps(record))
    (point / "nat.rep1.gate.log").write_text("busy: gpu=0")
    (point / "mux.rep1.log").write_text(
        "[rank3]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total "
        "capacity of 79.25 GiB of which 2.50 MiB is free. Including non-PyTorch memory, this process has 79.23 GiB "
        "memory in use. Of the allocated memory 70.04 GiB is allocated by PyTorch, and 612.00 MiB is reserved by "
        "PyTorch but unallocated."
    )

    rows = {row["case"]: row for row in summarize.collect(tmp_path)}

    assert rows["mux"]["status"] == "OOM"
    assert rows["mux"]["oom_allocated_gib"] == 70.04 and rows["mux"]["oom_outside_pytorch_gib"] == 8.59
    assert rows["mux"]["oom_foreign_gib"] == 0.02
    assert rows["nat"]["status"] == "ok" and rows["nat"]["seq"] == 4096 and rows["nat"]["gbs"] == 32
    assert math.isclose(rows["nat"]["mfu"], 90.0 / 312.0)
    assert len(rows) == 2


def test_planner_keeps_the_fastest_layouts_per_mode_above_the_screening_batch(tmp_path):
    planner = _load("qwen3_omni", "plan_omni_grid")
    summarize = _load("qwen3_omni", "summarize_omni_grid")
    screen = tmp_path / "seq4096_gbs16"
    screen.mkdir()
    timings = {"nat_a": 3.0, "nat_b": 2.0, "nat_c": 4.0, "mux_a": 5.0}
    for name, seconds in timings.items():
        (screen / f"{name}.rep1.log").write_text("THROUGHPUT ...")
        (screen / f"{name}.rep1.json").write_text(json.dumps({"step_seconds": seconds}))
    cases = [(name, "native" if name.startswith("nat") else "multiplexing", "") for name in timings]
    launch = planner.plan_point(
        tmp_path, cases, seq=4096, gbs=32, workload_args="", reason_fn=lambda argv: None, select_top=2, screen_gbs=16
    )
    assert sorted(name for name, _, _ in launch) == ["mux_a", "nat_a", "nat_b"]
    marker = tmp_path / "seq4096_gbs32" / "nat_c.rep1.not_selected"
    assert marker.read_text().strip() == "not among the 2 fastest native layouts at seq 4096, GBS 16"
    for path in screen.iterdir():
        path.unlink()
    screen.rmdir()
    rows = {row["case"]: row["status"] for row in summarize.collect(tmp_path)}
    assert rows == {"nat_c": "not selected: not among the 2 fastest native layouts at seq 4096, GBS 16"}
