"""End-to-end smoke of the private training loop on the reduced-width configuration.

Ref: Algorithm 1, Sec. 4.4, Sec. 4.6.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from private_vascular_motion_generation.runner.overfit import FitSpec, SingleBatchFit
from private_vascular_motion_generation.runner.seeding import set_seed
from private_vascular_motion_generation.runner.session import build_session, split_reference_and_private
from private_vascular_motion_generation.substrate import release_substrate


@pytest.fixture()
def loop_inputs(smoke_config, small_fixture):
    set_seed(smoke_config.seed())
    session = build_session(smoke_config)
    reference_records, private_records = split_reference_and_private(
        small_fixture.records, int(smoke_config.section("run")["reference_size"])
    )
    reference = session.make_batch(reference_records, seed=901)
    private = session.make_batch(private_records, seed=902)
    return session, reference, private


def test_smoke_config_is_flagged_as_a_test_configuration(smoke_config) -> None:
    assert smoke_config.name == "_smoke"
    assert "not a reporting configuration" in smoke_config.payload["description"]
    assert smoke_config.section("substrate")["dimension"] < smoke_config.section("substrate")["window_length"] * 100


def test_two_rounds_of_the_private_pass(loop_inputs) -> None:
    session, reference, private = loop_inputs
    trainer = session.trainer(seed=int(session.config.section("run")["seed"]))
    before = session.pair.adapter.snapshot()
    run = trainer.run([reference], session.sampler([private], trainer.seed), {"study": "smoke"})
    after = session.pair.adapter.snapshot()
    assert len(run.rounds) == session.config.section("optimiser")["rounds"]
    assert not torch.allclose(before, after)
    assert run.budget.epsilon > 0.0
    assert torch.isfinite(run.loss_trace()).all()
    assert torch.isfinite(run.rho_trace()).all()


def test_estimator_is_frozen_before_the_private_pass(loop_inputs) -> None:
    session, reference, private = loop_inputs
    trainer = session.trainer(seed=3)
    result = trainer.estimate_subspace([reference])
    frozen = result.projector.basis.clone()
    trainer.run([reference], session.sampler([private], 3), {"study": "smoke"})
    assert torch.allclose(frozen, result.projector.basis)
    assert result.rank() == session.config.estimator_spec().rank


def test_clipping_is_applied_before_the_noise(loop_inputs) -> None:
    session, reference, private = loop_inputs
    trainer = session.trainer(seed=4)
    run = trainer.run([reference], session.sampler([private], 4), {"study": "smoke"})
    assert all(record.noise_norm > 0.0 for record in run.rounds)
    assert all(record.workload_norm > 0.0 for record in run.rounds)


def test_learning_signal_reaches_both_branches(loop_inputs) -> None:
    session, reference, _private = loop_inputs
    fit = SingleBatchFit(session.pair, FitSpec(steps=40, learning_rate=5e-3), seed=5).run(reference)
    assert fit.relative_decrease(5) >= 0.05
    assert fit.tail_mean(5) < fit.head_mean(5)


def test_checkpoint_round_trip_restores_the_run(smoke_config, tmp_path: Path, loop_inputs) -> None:
    from private_vascular_motion_generation.runner.checkpoint import (
        CheckpointMeta,
        config_digest,
        load_checkpoint,
        save_checkpoint,
    )

    session, reference, private = loop_inputs
    trainer = session.trainer(seed=6)
    optimiser = torch.optim.SGD([session.pair.adapter.theta], lr=0.01)
    meta = CheckpointMeta(
        step=1,
        round_index=0,
        seed=smoke_config.seed(),
        initialisations=20,
        privacy_tuple={},
        config_digest=config_digest(smoke_config.payload),
    )
    path = save_checkpoint(tmp_path / "smoke.pt", session.pair, optimiser, meta)
    trained = session.pair.adapter.snapshot()
    session.pair.adapter.restore(torch.zeros_like(trained))
    load_checkpoint(path, session.pair, optimiser)
    assert torch.allclose(session.pair.adapter.snapshot(), trained, atol=1e-12)
    del reference, private, trainer


def test_release_decodes_both_branches(loop_inputs) -> None:
    session, reference, _private = loop_inputs
    result = session.estimator.fit(session.pair, [reference])
    from private_vascular_motion_generation.branches.decoder import JointProcedureDecoder

    decoder = JointProcedureDecoder(session.config.decoder_config())
    codes = torch.randn(4, session.config.substrate_spec().dimension, generator=torch.Generator().manual_seed(7))
    appearance, motion = release_substrate(
        codes,
        result.projector,
        decoder,
        session.trainer(seed=7).run([reference], session.sampler([_private], 7), {"study": "smoke"}).budget,
    )
    assert tuple(appearance.shape) == (4, 16, 32, 32)
    assert tuple(motion.shape) == (4, 6, 16)
    assert bool(torch.isfinite(appearance).all().item())
