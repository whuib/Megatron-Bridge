from types import SimpleNamespace

import pytest
from multiplexing_subject.bridge.roles import build_bridge_comm_plan
from multiplexing_subject.config import MultiplexingConfig
from multiplexing_subject.layout import ModuleParallelLayout


pytestmark = pytest.mark.unit


def test_disabled_is_inert():
    MultiplexingConfig().validate(object())


def test_conflicting_modes_fail_before_runtime():
    cfg = SimpleNamespace(model=SimpleNamespace(dist_train=SimpleNamespace(use_dist_train=True)))
    with pytest.raises(ValueError, match="mutually exclusive"):
        MultiplexingConfig(True, 1).validate(cfg)


def test_encoder_schedule_validation():
    def cfg(pp):
        return SimpleNamespace(model=SimpleNamespace(pipeline_model_parallel_size=pp))

    with pytest.raises(ValueError, match="encoder_schedule must be"):
        MultiplexingConfig(True, 1, encoder_schedule="eager").validate(cfg(1))
    with pytest.raises(ValueError, match="pipeline parallel size 1"):
        MultiplexingConfig(True, 1, encoder_schedule="interleaved").validate(cfg(2))


@pytest.mark.parametrize("world,tp,pp,encoder_tp", [(8, 2, 2, 1), (8, 1, 2, 2), (2, 2, 1, 1)])
def test_routes_cover_each_encoder_replica_once(world, tp, pp, encoder_tp):
    encoder, language = MultiplexingConfig(True, encoder_tp).layouts(world_size=world, language_tp=tp, language_pp=pp)
    plan = build_bridge_comm_plan(encoder, language)
    targets = [rank for rank, info in plan.comm_map.items() if info.recv_sources]
    sources = [source for rank in targets for source in plan.comm_map[rank].recv_sources]
    assert len(targets) == language.dp_size
    assert len(sources) == len(set(sources)) == encoder.dp_size


@pytest.mark.parametrize("world,tp,pp,encoder_tp", [(3, 2, 1, 1), (8, 1, 2, 0), (12, 2, 1, 3)])
def test_invalid_layouts(world, tp, pp, encoder_tp):
    with pytest.raises(ValueError):
        MultiplexingConfig(True, encoder_tp).layouts(world_size=world, language_tp=tp, language_pp=pp)


def test_a_smaller_encoder_dp_gathers_language_replicas():
    # Encoder DP 4 against language DP 8: each encoder replica serves two language replicas.
    encoder, language = MultiplexingConfig(True, 2).layouts(world_size=8, language_tp=1, language_pp=1)
    assert (encoder.dp_size, language.dp_size) == (4, 8)


def test_rank_mapping_roundtrip():
    layout = ModuleParallelLayout("encoder", 4, tp_size=2, dp_size=2, rank_mapping=(2, 3, 0, 1))
    for rank in range(4):
        assert layout.coord_to_global_rank(layout.global_rank_to_coord(rank)) == rank
