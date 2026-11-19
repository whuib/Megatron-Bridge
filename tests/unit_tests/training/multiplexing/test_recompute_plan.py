from dataclasses import replace

import pytest
from multiplexing_subject.recompute_plan import RankCalibration, RecomputeCalibration, plan_recompute


pytestmark = pytest.mark.unit
MIB = 2**20


def _plan(*calibrations, **overrides):
    options = dict(
        min_device_mib=78 * 1024,
        max_device_mib=79 * 1024,
        headroom_mib=350,
        h2d_bytes_per_second=10_000 * MIB,
        d2h_bytes_per_second=8_000 * MIB,
    )
    return plan_recompute(calibrations, **(options | overrides))


def _calibration(layers: int, fast_rank_peak: int, slow_rank_peak: int, step: float):
    rank = RankCalibration(
        peak_device_mib=fast_rank_peak,
        adam_unit_bytes=(900 * MIB, 900 * MIB, 250 * MIB, 250 * MIB),
        h2d_bytes=24_000 * MIB,
        d2h_bytes=22_000 * MIB,
        compute_seconds=2.5,
        copy_seconds=5.0,
        exposed_copy_seconds=2.0,
    )
    return RecomputeCalibration(layers, 900, step, (rank, replace(rank, peak_device_mib=slow_rank_peak)))


def test_planning_uses_worst_rank_peak_and_whole_adam_buffers():
    original = _calibration(0, 80_400, 80_400, 7.0)
    replay = _calibration(8, 79_000, 79_370, 7.04)
    plan = _plan(original, replay)
    assert plan.recompute_mlp_layers == 8
    assert plan.peak_device_mib == 80_520
    assert plan.headroom_mib == 376
    assert plan.resident_optim_mib == 2_050
    assert plan.resident_optim_bytes_per_rank == (2_050 * MIB,) * 2
    assert plan.estimated_step_seconds == pytest.approx(6.9365)
    assert plan.worst_rank_copy_excess_over_compute_seconds == pytest.approx(2.39125)


def test_fast_copies_do_not_justify_a_slower_recompute_baseline():
    original = _calibration(0, 80_400, 80_400, 7.0)
    replay = _calibration(8, 79_000, 79_370, 7.04)
    plan = _plan(original, replay, h2d_bytes_per_second=100_000 * MIB, d2h_bytes_per_second=80_000 * MIB)
    assert plan.recompute_mlp_layers == 0
    assert plan.worst_rank_copy_excess_over_compute_seconds == 0


def test_no_plan_exceeds_the_memory_guard():
    sample = _calibration(8, 79_000, 81_000, 7.0)
    with pytest.raises(ValueError, match="No calibrated MLP prefix"):
        _plan(sample)
