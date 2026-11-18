import copy
import math
from types import SimpleNamespace

import multiplexing_subject.state_budget as state_budget
import pytest
import torch
from multiplexing_subject.config import MultiplexingConfig
from multiplexing_subject.state_budget import (
    DEFERRED,
    INLINE,
    RESIDENT,
    BudgetedStateOffload,
    OffloadCandidate,
    copy_slack,
    plan_offload,
)
from multiplexing_subject.state_offload import ModuleStateOffload, OffloadedModel, discover_modules
from torch import nn


pytestmark = pytest.mark.unit


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.norm(x))


class Toy(nn.Module):
    def __init__(self, width: int = 64, layers: int = 4) -> None:
        super().__init__()
        self.inp = nn.Linear(16, width)
        self.blocks = nn.ModuleList(Block(width) for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_discover_modules_splits_layers_and_leaves():
    names = [name for name, _ in discover_modules(Toy(layers=3))]
    assert names == ["inp", "blocks.0", "blocks.1", "blocks.2", "head"]


def test_discover_modules_rejects_parameters_next_to_layers():
    model = Toy()
    model.scale = nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="scale"):
        discover_modules(model)


def test_overlap_requires_offload():
    with pytest.raises(ValueError, match="budget requires module_state_offload_overlap"):
        MultiplexingConfig(module_state_offload=True, module_state_offload_budget=True).validate(object())
    with pytest.raises(ValueError, match="requires module_state_offload"):
        MultiplexingConfig(module_state_offload_overlap=True).validate(object())
    with pytest.raises(ValueError, match="requires use_multiplexing"):
        MultiplexingConfig(module_state_offload=True).validate(object())


def test_offload_rejects_unsupported_settings():
    model = SimpleNamespace(
        pipeline_model_parallel_size=2,
        recompute_granularity=None,
        vit_gradient_checkpointing=False,
        cpu_offloading=False,
        share_embeddings_and_output_weights=True,
        bf16=True,
    )
    optimizer = SimpleNamespace(
        optimizer="adam",
        use_precision_aware_optimizer=False,
        optimizer_cpu_offload=False,
        optimizer_cuda_graph=False,
    )
    cfg = SimpleNamespace(model=model, optimizer=optimizer, ddp=SimpleNamespace(use_distributed_optimizer=True))
    config = MultiplexingConfig(True, 1, module_state_offload=True)
    with pytest.raises(ValueError, match="language pipeline parallelism, tied embeddings"):
        config.validate_module_state_offload(cfg)
    model.pipeline_model_parallel_size = 1
    model.share_embeddings_and_output_weights = False
    config.validate_module_state_offload(cfg)


def _adam(params: list[torch.Tensor]):
    optimizers = pytest.importorskip("transformer_engine.pytorch.optimizers")
    decay = [param for param in params if param.dim() > 1]
    no_decay = [param for param in params if param.dim() == 1]
    return optimizers.FusedAdam(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        adam_w_mode=True,
    )


def _loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return model(batch).float().square().mean()


def _reference(model: nn.Module, steps: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """Megatron's update: FP32 main gradients, FusedAdam on FP32 main params, BF16 copy back."""
    params = list(model.parameters())
    main_grads = [torch.zeros_like(param, dtype=torch.float32) for param in params]

    def accumulator(grad: torch.Tensor):
        def accumulate(param: nn.Parameter) -> None:
            grad.add_(param.grad)
            param.grad = None

        return accumulate

    for param, grad in zip(params, main_grads):
        param.register_post_accumulate_grad_hook(accumulator(grad))
    mains = [param.detach().float() for param in params]
    adam = _adam(mains)
    for batches in steps:
        for grad in main_grads:
            grad.zero_()
        for batch in batches:
            _loss(model, batch).backward()
        for main, grad in zip(mains, main_grads):
            main.grad = grad
        adam.step()
        for param, main in zip(params, mains):
            param.data.copy_(main)
    return [param.detach().clone() for param in params]


def _described(model: nn.Module) -> list[OffloadedModel]:
    params = list(model.parameters())
    adam = _adam([param.detach().float() for param in params])
    groups = {}
    for param in params:
        groups[param] = (adam, 0 if param.dim() > 1 else 1)
    return [
        OffloadedModel(
            name="toy",
            module=model,
            param_groups=groups,
            tp_group=None,
            dp_group=None,
            average_in_collective=False,
            sequence_parallel=False,
            qk_layernorm=False,
        )
    ]


def _train(engine: ModuleStateOffload, model: nn.Module, batches: list[torch.Tensor]) -> None:
    engine.begin_step()
    for index, batch in enumerate(batches):
        if index == len(batches) - 1:
            engine.final_microbatch()
        _loss(model, batch).backward()
        engine.synchronize_handoff()
    engine.end_step()


def _offloaded(model: nn.Module, steps: list[list[torch.Tensor]], **options):
    params = list(model.parameters())
    engine = ModuleStateOffload(_described(model), **options)
    for batches in steps:
        _train(engine, model, batches)
    torch.cuda.synchronize()
    values = {}
    for unit in engine._units:
        weights = unit.device["param"] if "param" in unit.persistent else unit.host["param"]
        for param, offset in zip(unit.params, unit.offsets):
            values[param] = weights[offset : offset + param.numel()].view_as(param).to("cuda", copy=True)
    if options.get("resident_param_bytes"):
        assert 0 < engine.resident_param_bytes <= options["resident_param_bytes"]
        assert any("param" not in unit.persistent for unit in engine._units)
    if options.get("resident_optim_bytes"):
        assert 0 < engine.resident_optim_bytes <= options["resident_optim_bytes"]
        assert any("optim" not in unit.persistent for unit in engine._units)
    if options.get("writeback_chunk_bytes"):
        assert engine.step_summary()["chunked_writebacks"] > 0
        assert engine._pending_writeback is None
    resident = engine.max_resident_modules
    engine.close()
    return [values[param] for param in params], resident


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    ("overlap", "resident_param_bytes", "prefetch_bytes", "resident_optim_bytes", "scoped_handoff", "chunk_bytes"),
    [
        (False, 0, 0, 0, False, 0),
        (True, 0, 0, 0, False, 0),
        (True, 18 * 1024, 0, 0, False, 0),
        (True, 18 * 1024, 64 * 1024, 0, False, 0),
        (True, 18 * 1024, 64 * 1024, 58 * 1024, False, 0),
        (True, 18 * 1024, 64 * 1024, 58 * 1024, True, 0),
        (True, 18 * 1024, 64 * 1024, 58 * 1024, True, 16 * 1024),
    ],
)
def test_offloaded_training_matches_resident_training(
    overlap, resident_param_bytes, prefetch_bytes, resident_optim_bytes, scoped_handoff, chunk_bytes
):
    torch.manual_seed(0)
    model = Toy().cuda().to(torch.bfloat16)
    reference_model = copy.deepcopy(model)
    # Changing micro-batch counts make the module execution order diverge from the recorded one.
    steps = [
        [torch.randn(32, 16, device="cuda", dtype=torch.bfloat16) for _ in range(count)]
        for count in (2, 2, 3, 1, 1, 2)
    ]
    expected = _reference(reference_model, steps)
    actual, resident = _offloaded(
        model,
        steps,
        overlap=overlap,
        resident_param_bytes=resident_param_bytes,
        prefetch_bytes=prefetch_bytes,
        resident_optim_bytes=resident_optim_bytes,
        scoped_handoff=scoped_handoff,
        writeback_chunk_bytes=chunk_bytes,
    )
    for want, got in zip(expected, actual):
        assert torch.equal(want, got)
    assert resident == (2 if overlap else 1)


def test_copy_slack_simulates_the_copy_stream():
    segment = {"start": 0.0, "end": 1.0, "h2d_rate": 20.0, "d2h_rate": 20.0}
    assert copy_slack([], lookahead=1, **segment) == math.inf
    # 4 bytes at 0.3 s, then 1 byte at 0.5 s: its prefetch queues behind the 0.2 s write-back.
    assert copy_slack([(0.3, 4), (0.5, 1)], lookahead=1, **segment) == pytest.approx(-0.05)
    assert copy_slack([(0.3, 4), (0.8, 1)], lookahead=1, **segment) == pytest.approx(0.1)
    # Both prefetched at the start of the segment.
    assert copy_slack([(0.3, 4), (0.5, 1)], lookahead=2, **segment) == pytest.approx(0.1)
    # The last write-back must finish before the synchronization ending the segment.
    assert copy_slack([(0.9, 4)], lookahead=1, **segment) == pytest.approx(-0.1)


def test_plan_offload_picks_windows_the_bandwidth_hides():
    candidates = [OffloadCandidate(0, 4, 0.3, None), OffloadCandidate(1, 1, 0.5, 0.8)]
    rates = {"h2d_rate": 20.0, "d2h_rate": 20.0, "lookahead": 1}
    assert plan_offload(candidates, barriers=[], end=1.0, **rates) == {0: DEFERRED, 1: INLINE}
    # A synchronization at 0.45 s cuts module 0's write-back; module 1 then has its segment alone.
    assert plan_offload(candidates, barriers=[0.45], end=1.0, **rates) == {0: RESIDENT, 1: INLINE}
    fast = plan_offload(candidates, barriers=[], end=1.0, h2d_rate=1e9, d2h_rate=1e9, lookahead=1)
    assert RESIDENT not in fast.values()
    slow = plan_offload(candidates, barriers=[], end=1.0, h2d_rate=1.0, d2h_rate=1.0, lookahead=1)
    assert set(slow.values()) == {RESIDENT}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("plan", ["fast", "slow", "mixed"])
def test_budgeted_training_matches_resident_training(plan, monkeypatch):
    # Copies of the toy modules span several chunks.
    monkeypatch.setattr(state_budget, "_CHUNK_BYTES", 4096)
    if plan == "mixed":
        modes = (RESIDENT, INLINE, DEFERRED)
        monkeypatch.setattr(
            state_budget,
            "plan_offload",
            lambda candidates, **_: {candidate.index: modes[candidate.index % 3] for candidate in candidates},
        )
    torch.manual_seed(0)
    model = Toy().cuda().to(torch.bfloat16)
    reference_model = copy.deepcopy(model)
    # Micro-batch count changes diverge from the recorded order, re-measure and re-plan.
    steps = [
        [torch.randn(32, 16, device="cuda", dtype=torch.bfloat16) for _ in range(count)]
        for count in (2, 2, 2, 2, 3, 3, 3, 1, 2, 2, 2)
    ]
    expected = _reference(reference_model, steps)
    engine = BudgetedStateOffload(_described(model), bandwidth=(1.0, 1.0) if plan == "slow" else (1e18, 1e18))
    planned_steps = 0
    for batches in steps:
        _train(engine, model, batches)
        if engine._profiling:
            continue
        planned_steps += 1
        offloaded = [unit for unit in engine._units if engine._mode[unit.index] != RESIDENT]
        if plan == "fast":
            # Only modules whose Adam steps run right at a segment boundary stay resident.
            assert 2 * len(offloaded) >= len(engine._units)
        elif plan == "slow":
            assert not offloaded
        assert sum("optim" in unit.resident for unit in offloaded) <= engine._lookahead
        assert {unit.index for unit in engine._units if "grad" in unit.resident} == set(engine._pending)
    assert planned_steps >= 4
    assert set(engine.step_summary()) == {
        "h2d_gib",
        "d2h_gib",
        "offloaded_adam_gib",
        "timeline_s",
        "copy_wait_ms",
        "margin",
    }
    engine.close()
    for want, got in zip(expected, model.parameters()):
        assert torch.equal(want, got)


def test_mixed_plan_offloads_activations_to_fit():
    offload_plan = pytest.importorskip("multiplexing_subject.offload_plan")
    # The forward at event 0 is not read again until event 6, so its saved bytes cover the peak.
    unit = offload_plan.UnitProfile(0, 100, 10, 50, 0, 6, 6)
    group = offload_plan.GroupProfile(0, 0, 6, 5000)
    profile = offload_plan.StepProfile(
        times=tuple(index * 0.05 for index in range(8)),
        end=0.4,
        barriers=(),
        units=(unit,),
        groups=(group,),
        memory=(200, 200, 200, 200, 7000, 7000, 200, 200, 200, 200),
    )
    rates = offload_plan.CopyRates(1e12, 1e12, 1e12, 1e12)
    plan = offload_plan.plan_offload(profile, rates=rates, lookahead=1, throttle=1, budget=3000)
    assert plan.peak is not None and plan.peak <= 3000
    assert plan.activations.get(0, 0) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_headroom_budget_matches_resident_training(monkeypatch):
    monkeypatch.setattr(state_budget, "_CHUNK_BYTES", 4096)
    torch.manual_seed(0)
    model = Toy().cuda().to(torch.bfloat16)
    reference_model = copy.deepcopy(model)
    steps = [[torch.randn(8, 16, device="cuda", dtype=torch.bfloat16) for _ in range(count)] for count in (2, 2, 2, 1)]
    expected = _reference(reference_model, steps)
    engine = BudgetedStateOffload(
        _described(model),
        bandwidth=(1e18, 1e18),
        headroom_bytes=1 << 60,
    )
    for batches in steps:
        _train(engine, model, batches)
    assert engine._memory_budget < 0 or engine._predicted_peak <= engine._memory_budget or engine._param_ids
    assert (
        engine._offloaded_activation_bytes
        or engine._offloaded_param_bytes
        or any(engine._mode[unit.index] != RESIDENT for unit in engine._units)
    )
    engine.close()
    for want, got in zip(expected, model.parameters()):
        assert torch.equal(want, got)
