"""Runner tests: configuration, batching, seeding, optimisation, checkpointing and the trainer.

Ref: Algorithm 1, Sec. 4.6, Sec. 4.9.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from private_vascular_motion_generation.runner.batching import (
    BatchSpec,
    RoundSampler,
    batch_from_records,
    vessel_mask,
)
from private_vascular_motion_generation.runner.checkpoint import (
    CheckpointMeta,
    config_digest,
    digest_matches,
    load_checkpoint,
    privacy_tuple_from,
    save_checkpoint,
    seed_state_from,
)
from private_vascular_motion_generation.runner.config import (
    ConfigError,
    StudyConfig,
    apply_overrides,
    deep_merge,
    load_yaml,
    resolve_study,
)
from private_vascular_motion_generation.runner.distributed import (
    DistributedPlan,
    PrecisionPolicy,
    build_grad_scaler,
    initialise_process_group,
    shard_workload,
    uses_grad_scaler,
    wrap_model,
)
from private_vascular_motion_generation.runner.logging_utils import LoggingConfig, configure_logging, summarise_round
from private_vascular_motion_generation.runner.optim import (
    OptimisationPlan,
    apply_learning_rate,
    build_adamw,
    build_optimiser,
    learning_rate_at,
    schedule_ladder,
)
from private_vascular_motion_generation.runner.overfit import FitSpec, SingleBatchFit, branch_balance
from private_vascular_motion_generation.runner.seeding import (
    SeedState,
    generator_for,
    same_seed_state,
    set_seed,
)
from private_vascular_motion_generation.runner.session import split_reference_and_private
from private_vascular_motion_generation.runner.trainer import CAPSTrainer, TrainingSpec


def test_yaml_loading_and_merging(release_root: Path) -> None:
    payload = load_yaml(release_root / "configs" / "scheme" / "caps_shaped.yaml")
    assert payload["sigma_parallel"] == 5.4
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}}
    with pytest.raises(ConfigError):
        load_yaml(release_root / "configs" / "scheme" / "absent.yaml")


def test_override_and_validation() -> None:
    resolved = apply_overrides({"a": {"b": 1}}, ["a.b=7", "c=x"])
    assert resolved["a"]["b"] == 7 and resolved["c"] == "x"
    with pytest.raises(ConfigError):
        apply_overrides({"a": 1}, ["bad"])


def test_study_resolution_layers_fragments(release_root: Path) -> None:
    config = resolve_study("caps_main", root=release_root / "configs" / "study")
    assert config.value("scheme") == "caps_shaped"
    assert config.section("substrate")["sigma_parallel"] == 5.4
    assert config.section("substrate")["dimension"] == 1024
    assert config.estimator_spec().rank == 12
    assert config.appearance_config().dim == 1024
    assert config.adapter_spec().dimension == 1024
    assert config.latent_spec().dimension == 256
    assert len(config.order_grid().orders) == 15
    with pytest.raises(ConfigError):
        config.section("absent")


def test_study_inheritance_and_arms(release_root: Path) -> None:
    arm = resolve_study("factorial_resolution_conditioning", root=release_root / "configs" / "study")
    assert arm.value("base") == "caps_main"
    assert arm.section("factorial")["declared_budget"] == 2.0
    assert arm.section("substrate")["sigma_parallel"] == 5.4
    ablation = resolve_study("ablation_components", root=release_root / "configs" / "study")
    assert len(ablation.section("components")["removal_order"]) == 6


def test_unknown_study_raises(release_root: Path) -> None:
    with pytest.raises(ConfigError):
        resolve_study("nonexistent", root=release_root / "configs" / "study")


def test_study_config_rejects_missing_section(release_root: Path) -> None:
    config = StudyConfig(name="probe", payload={"a": 1}, source="probe")
    with pytest.raises(ConfigError):
        config.section("missing")
    assert config.value("a") == 1


def test_seeding_is_reproducible() -> None:
    first = set_seed(1234)
    draw_one = torch.randn(4, generator=generator_for(1234, 0))
    second = set_seed(1234)
    draw_two = torch.randn(4, generator=generator_for(1234, 0))
    assert torch.allclose(draw_one, draw_two)
    assert same_seed_state(first, second)
    assert len(SeedState(seed=5).offsets()) == 20
    assert len(SeedState(seed=5).training_seeds()) == 3


def test_optimisation_schedule_shape() -> None:
    plan = OptimisationPlan(rounds=100, learning_rate=0.01, warmup_fraction=0.1, minimum_learning_ratio=0.1)
    assert plan.warmup_rounds() == 10
    assert learning_rate_at(0, plan) < learning_rate_at(9, plan)
    assert learning_rate_at(99, plan) >= plan.learning_rate * plan.minimum_learning_ratio
    ladder = schedule_ladder(plan, 5)
    assert len(ladder) == 5 and ladder[0] < ladder[-1]
    parameters = [torch.nn.Parameter(torch.zeros(2))]
    optimiser = build_optimiser(parameters, plan)
    value = apply_learning_rate(optimiser, 50, plan)
    assert optimiser.param_groups[0]["lr"] == pytest.approx(value)
    assert isinstance(build_adamw(parameters, 1e-3), torch.optim.AdamW)


def test_batching_and_sampling(smoke_config, small_fixture) -> None:
    spec = BatchSpec(
        frames=int(smoke_config.section("branch", "appearance")["in_frames"]),
        height=int(smoke_config.section("run")["height"]),
        width=int(smoke_config.section("run")["width"]),
        latent_dim=int(smoke_config.section("coupling")["latent_dim"]),
    )
    batch = batch_from_records(small_fixture.records[:6], spec, seed=1)
    assert tuple(batch.frames.shape) == (6, 16, 32, 32)
    assert tuple(batch.mask.shape) == (6, 1, 32, 32)
    sampler = RoundSampler([batch], 0.5, seed=2)
    assert sampler.batch_size() == 3
    drawn = sampler.draw()
    assert drawn.size() == 3
    mask = vessel_mask(small_fixture.records[0].appearance)
    assert tuple(mask.shape) == (1, 32, 32)


def test_batching_validation() -> None:
    with pytest.raises(ValueError):
        BatchSpec(frames=0, height=8, width=8, latent_dim=4)
    with pytest.raises(ValueError):
        RoundSampler([], 0.5, 1)


def test_checkpoint_round_trip(smoke_session, tmp_path: Path) -> None:
    optimiser = torch.optim.SGD([smoke_session.pair.adapter.theta], lr=0.01)
    before = smoke_session.pair.adapter.snapshot()
    meta = CheckpointMeta(
        step=2,
        round_index=1,
        seed=smoke_session.config.seed(),
        initialisations=20,
        privacy_tuple={},
        config_digest=config_digest(smoke_session.config.payload),
    )
    path = save_checkpoint(tmp_path / "probe.pt", smoke_session.pair, optimiser, meta)
    smoke_session.pair.adapter.restore(torch.zeros_like(before))
    restored = load_checkpoint(path, smoke_session.pair, optimiser)
    assert torch.allclose(smoke_session.pair.adapter.snapshot(), before, atol=1e-12)
    assert seed_state_from(restored).seed == smoke_session.config.seed()
    assert digest_matches(restored, smoke_session.config.payload)
    assert privacy_tuple_from(restored) is None
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "absent.pt", smoke_session.pair)


def test_checkpoint_carries_the_privacy_statement_once_the_run_records_one(smoke_session, tmp_path: Path) -> None:
    from private_vascular_motion_generation.accounting.tuple import PrivacyTuple

    statement = PrivacyTuple(
        epsilon=2.0,
        delta=1e-5,
        accountant="renyi",
        order=16.0,
        steps_times_sampling=175.2,
        clipping_norm=0.85,
        sigma_parallel=5.4,
        sigma_perpendicular=2.6,
        rank=12,
        dimension=1024,
        rho=0.19,
    )
    optimiser = torch.optim.SGD([smoke_session.pair.adapter.theta], lr=0.01)
    meta = CheckpointMeta(
        step=3,
        round_index=2,
        seed=smoke_session.config.seed(),
        initialisations=20,
        privacy_tuple=statement.to_dict(),
        config_digest=config_digest(smoke_session.config.payload),
    )
    path = save_checkpoint(tmp_path / "statement.pt", smoke_session.pair, optimiser, meta)
    rebuilt = privacy_tuple_from(load_checkpoint(path, smoke_session.pair))
    assert rebuilt is not None
    assert rebuilt.epsilon == pytest.approx(2.0)
    assert rebuilt.rank == 12 and rebuilt.order == pytest.approx(16.0)


def test_precision_and_distribution_switches() -> None:
    policy = PrecisionPolicy(enabled=False)
    with policy.autocast():
        pass
    assert not uses_grad_scaler(policy)
    assert build_grad_scaler(policy).is_enabled() is False
    plan = DistributedPlan(enabled=False)
    assert not initialise_process_group(plan)
    assert isinstance(wrap_model(torch.nn.Linear(2, 2), plan), torch.nn.Linear)
    workload = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    assert shard_workload(workload, plan).shape == workload.shape
    with pytest.raises(ValueError):
        DistributedPlan(world_size=0)


def test_logging_configuration(tmp_path: Path) -> None:
    logger = configure_logging(LoggingConfig(log_file=tmp_path / "run.log"))
    logger.info("probe")
    assert (tmp_path / "run.log").exists()
    assert "round" in summarise_round(0, 4, 0.19, 2.5)


def test_split_keeps_reference_disjoint(small_fixture) -> None:
    reference, private = split_reference_and_private(small_fixture.records, 4)
    assert len(reference) == 4
    assert not ({record.procedure_id for record in reference} & {record.procedure_id for record in private})
    with pytest.raises(ValueError):
        split_reference_and_private(small_fixture.records[:2], 4)


def test_single_batch_fit_decreases_the_residual(smoke_session, small_fixture) -> None:
    batch = smoke_session.make_batch(
        small_fixture.records[: int(smoke_session.config.section("run")["reference_size"])], seed=5
    )
    fit = SingleBatchFit(smoke_session.pair, FitSpec(steps=30, learning_rate=5e-3), seed=6).run(batch)
    assert fit.relative_decrease(5) > 0.0
    assert fit.tail_mean(5) < fit.head_mean(5)
    balance = branch_balance(batch, smoke_session.pair, torch.Generator().manual_seed(7))
    assert 0.0 < balance["share_appearance"] < 1.0


def test_caps_trainer_produces_a_privacy_tuple(smoke_config, smoke_session, small_fixture) -> None:
    set_seed(smoke_config.seed())
    reference_records, private_records = split_reference_and_private(
        small_fixture.records, int(smoke_config.section("run")["reference_size"])
    )
    reference = smoke_session.make_batch(reference_records, seed=11)
    private = smoke_session.make_batch(private_records, seed=12)
    trainer = smoke_session.trainer(seed=13)
    assert isinstance(trainer, CAPSTrainer)
    run = trainer.run([reference], smoke_session.sampler([private], 13), {"study": "probe"})
    assert len(run.rounds) == smoke_config.section("optimiser")["rounds"]
    assert run.budget.epsilon > 0.0
    assert run.privacy_tuple.rank == smoke_config.estimator_spec().rank
    assert 0.0 <= run.measured_rho <= 1.0
    assert len(run.privacy_tuple.as_ordered_items()) == 10


def test_training_spec_validation() -> None:
    with pytest.raises(ValueError):
        TrainingSpec(rounds=0, delta=1e-5, clipping_norm=0.85, sampling_rate=0.5)
    with pytest.raises(ValueError):
        TrainingSpec(rounds=2, delta=1.0, clipping_norm=0.85, sampling_rate=0.5)
    with pytest.raises(ValueError):
        TrainingSpec(rounds=2, delta=1e-5, clipping_norm=0.85, sampling_rate=0.0)
