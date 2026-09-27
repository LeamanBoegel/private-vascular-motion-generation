"""Shaping-estimator tests, including the disjointness premise of Algorithm 2.

Ref: Algorithm 2, Sec. 4.4, Sec. 4.7.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from private_vascular_motion_generation.estimator.reference_estimator import (
    EstimatorSpec,
    ReferenceShapingEstimator,
    cost_estimate,
    subspace_from_private_records,
)
from private_vascular_motion_generation.geometry.projector import projector_residual


def test_estimator_returns_the_declared_rank(release_root: Path, smoke_config, smoke_session, small_fixture) -> None:
    reference_records, _private = (
        list(small_fixture.records[:6]),
        list(small_fixture.records[6:]),
    )
    batch = smoke_session.make_batch(reference_records, seed=21)
    estimator = ReferenceShapingEstimator(EstimatorSpec(reference_batch_size=6, rank=3), smoke_config.pairing_spec())
    result = estimator.fit(smoke_session.pair, [batch])
    assert result.rank() == 3
    assert result.subspace_dimension() == smoke_session.config.substrate_spec().dimension
    assert projector_residual(result.projector) < 1e-5
    assert 0.0 <= result.diagnostics["leading_correlation"] <= 1.0
    assert estimator.freeze(result) is result.projector


def test_estimator_requires_a_fit_before_freezing(smoke_config) -> None:
    estimator = ReferenceShapingEstimator(EstimatorSpec(reference_batch_size=4, rank=2), smoke_config.pairing_spec())
    with pytest.raises(RuntimeError):
        estimator.freeze(None)  # type: ignore[arg-type]


def test_estimator_rejects_an_oversized_reference_batch(smoke_config, smoke_session, small_fixture) -> None:
    batch = smoke_session.make_batch(list(small_fixture.records[:6]), seed=22)
    estimator = ReferenceShapingEstimator(EstimatorSpec(reference_batch_size=2, rank=2), smoke_config.pairing_spec())
    with pytest.raises(ValueError):
        estimator.fit(smoke_session.pair, [batch])


def test_estimator_needs_at_least_one_batch(smoke_config, smoke_session) -> None:
    estimator = ReferenceShapingEstimator(EstimatorSpec(reference_batch_size=4, rank=2), smoke_config.pairing_spec())
    with pytest.raises(ValueError):
        estimator.fit(smoke_session.pair, [])


def test_cost_model_follows_the_stated_orders() -> None:
    report = cost_estimate(8, 32, 2)
    assert report["per_minibatch"] == 8 * 32**2
    assert report["total_estimator"] == 2 * 8 * 32**2
    assert report["attention_multiplier_cost"] == 32**2


def test_private_record_subspace_is_a_distinct_control(smoke_session, small_fixture) -> None:
    batch = smoke_session.make_batch(list(small_fixture.records[6:12]), seed=23)
    projector = subspace_from_private_records(smoke_session.pair, [batch], 3)
    assert projector.rank == 3
    assert projector_residual(projector) < 1e-5
    with pytest.raises(ValueError):
        cost_estimate(0, 32, 1)


def test_reference_corpus_manifest_is_required(tmp_path: Path) -> None:
    from private_vascular_motion_generation.cohort.reference import (
        ReferenceCorpusSpec,
        assert_estimator_disjointness,
        load_reference_corpus,
    )

    spec = ReferenceCorpusSpec(root=tmp_path, clip_frames=4, appearance_extent=(8, 8))
    with pytest.raises(FileNotFoundError):
        load_reference_corpus(spec)
    corpus = _written_corpus(tmp_path, spec)
    assert len(corpus) == 2
    assert corpus.aligned()
    assert corpus.source_mix()["public_angiography"] == 1
    batches = list(corpus.batches(1))
    assert len(batches) == 2
    assert_estimator_disjointness(corpus, ["A::00000"])
    with pytest.raises(ValueError):
        assert_estimator_disjointness(corpus, ["ref-00000"])


def _written_corpus(tmp_path: Path, spec):  # type: ignore[no-untyped-def]
    import json

    from private_vascular_motion_generation.cohort.reference import load_reference_corpus

    entries = [
        {"index": 0, "source_kind": "public_angiography"},
        {"index": 1, "source_kind": "open_simulator"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({"clips": entries}), encoding="utf-8")
    generator = torch.Generator().manual_seed(24)
    for index in range(2):
        torch.save(
            {
                "appearance": torch.rand(
                    spec.appearance_extent[0], spec.appearance_extent[1], spec.clip_frames, generator=generator
                ),
                "motion": torch.randn(6, spec.clip_frames, generator=generator),
            },
            tmp_path / f"clip_{index:05d}.pt",
        )
    return load_reference_corpus(spec)
