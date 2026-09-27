"""Study-registry tests: every arm runs and reports its status.

Ref: Sec. 2.1-2.9, Sec. 4.8-4.10.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from private_vascular_motion_generation.runner.config import StudyConfig, resolve_study
from private_vascular_motion_generation.runner.studies import (
    STUDY_REGISTRY,
    StudyOutcome,
    execute_study,
)

SMOKE_OVERRIDES: tuple[str, ...] = (
    "substrate.dimension=32",
    "substrate.window_length=3",
    "substrate.rank=3",
    "substrate.rounds=1",
    "substrate.sampling_rate=0.5",
    "coupling.latent_dim=24",
    "estimator.reference_batch_size=6",
    "branch.appearance.dim=32",
    "branch.appearance.heads=4",
    "branch.appearance.state_dim=8",
    "branch.appearance.transformer_depth=1",
    "branch.appearance.latent_dim=24",
    "branch.motion.width=32",
    "branch.motion.depth=1",
    "branch.motion.latent_dim=24",
    "branch.decoder.latent_dim=32",
    "branch.decoder.hidden=32",
    "optimiser.rounds=1",
    "run.reference_size=6",
    "run.private_size=8",
)


def _arm(release_root: Path, name: str) -> StudyConfig:
    return resolve_study(name, list(SMOKE_OVERRIDES), release_root / "configs" / "study")


def test_registry_is_populated() -> None:
    assert len(STUDY_REGISTRY) >= 12
    assert "caps_main" in STUDY_REGISTRY
    assert all(callable(arm) for arm in STUDY_REGISTRY.values())


def test_unknown_arm_is_rejected(release_root: Path) -> None:
    with pytest.raises(KeyError):
        execute_study("not_a_study", release_root)
    with pytest.raises(TypeError):
        execute_study("null_results", release_root, unsupported_keyword=1)


def test_outcome_serialisation() -> None:
    outcome = StudyOutcome(name="probe", status="PASS", metrics={"a": 1.0}, notes=["n"])
    payload = outcome.to_dict()
    assert payload["study"] == "probe" and payload["metrics"]["a"] == 1.0


def test_caps_main_arm(release_root: Path) -> None:
    from private_vascular_motion_generation.runner.studies import run_caps_main

    outcome = run_caps_main(_arm(release_root, "caps_main"), max_rounds=1)
    assert outcome.status == "PASS"
    assert outcome.metrics["rounds"] == 1
    assert outcome.metrics["epsilon"] > 0.0
    assert 0.0 <= outcome.metrics["measured_rho"] <= 1.0
    assert outcome.metrics["subspace_rank"] == 3
    assert outcome.metrics["fixture_procedures"] >= 14


def test_overfit_arm(release_root: Path) -> None:
    outcome = execute_study("overfit_single_batch", release_root, overrides=list(SMOKE_OVERRIDES))
    assert outcome.status == "PASS"
    assert outcome.metrics["relative_decrease"] > 0.10


def test_component_ablation_arm(release_root: Path) -> None:
    from private_vascular_motion_generation.runner.studies import run_component_ablation

    outcome = run_component_ablation(_arm(release_root, "ablation_components"), max_rounds=1)
    assert outcome.status == "PASS"
    variants = outcome.metrics["variants"]
    assert len(variants) == 6
    assert variants["full_mechanism"]["epsilon"] > 0.0
    assert variants["minus_coupling_shaped_window_noise"]["epsilon"] != variants["full_mechanism"]["epsilon"]


def test_window_ladder_arm(release_root: Path) -> None:
    from private_vascular_motion_generation.runner.studies import run_window_ladder

    config = _arm(release_root, "ablation_window_ladder")
    config.payload["windows"] = [8, 16]
    outcome = run_window_ladder(config, max_rounds=1)
    assert outcome.status == "PASS"
    assert set(outcome.metrics["ladder"]) == {"k=8", "k=16"}


def test_factorial_arm(release_root: Path) -> None:
    from private_vascular_motion_generation.runner.studies import run_factorial

    config = _arm(release_root, "factorial_resolution_conditioning")
    config.payload["factorial"]["cells"] = {
        "on_1x": {"resolution": 1, "conditioning": "on"},
        "off_1x": {"resolution": 1, "conditioning": "off"},
    }
    outcome = run_factorial(config, max_rounds=1)
    assert outcome.status == "PASS"
    assert set(outcome.metrics["cells"]) == {"on_1x", "off_1x"}
    assert outcome.metrics["printed_contrasts"]["fid"]["conditioning_main_effect"] == pytest.approx(-9.65)


def test_audit_arm(release_root: Path) -> None:
    outcome = execute_study("audit_membership", release_root, overrides=list(SMOKE_OVERRIDES))
    assert outcome.status == "PASS"
    assert 0.0 <= outcome.metrics["plain_auc"]["auc"] <= 1.0
    assert "appearance_median" in outcome.metrics["distance_by_modality"]


def test_window_horizon_arm(release_root: Path) -> None:
    outcome = execute_study("scaling_window_horizon", release_root, overrides=list(SMOKE_OVERRIDES))
    assert outcome.status == "PASS"
    assert outcome.metrics["measured_ladder"]["128"]["group_multiplier"] > 2.0
    assert outcome.metrics["reported_monotone"] is True
    assert outcome.metrics["fit"]["r_squared"] > 0.95


def test_substrate_scaling_arm(release_root: Path) -> None:
    from private_vascular_motion_generation.runner.studies import run_substrate_scaling

    config = _arm(release_root, "scaling_substrate_volume")
    config.payload["sweep"]["rungs"] = [600, 1200]
    config.payload["run"]["private_size"] = 4
    outcome = run_substrate_scaling(config, max_rounds=1)
    assert outcome.status == "PASS"
    assert set(outcome.metrics["ladder"]) == {"600", "1200"}


def test_reference_corpus_arm(release_root: Path) -> None:
    from private_vascular_motion_generation.runner.studies import run_reference_corpus_scaling

    config = _arm(release_root, "scaling_reference_corpus")
    config.payload["sweep"]["rungs_thousands"] = [2, 4]
    config.payload["run"]["reference_size"] = 4
    config.payload["run"]["private_size"] = 6
    outcome = run_reference_corpus_scaling(config, max_rounds=1)
    assert outcome.status == "PASS"
    assert set(outcome.metrics["ladder"]) == {"2", "4"}


def test_perturbation_arm(release_root: Path) -> None:
    from private_vascular_motion_generation.runner.studies import run_perturbation_response

    config = _arm(release_root, "perturbation_response")
    config.payload["perturbation"]["families"] = {
        "intensity_noise": [0.01, 0.05],
        "frame_dropout": [0.05],
    }
    outcome = run_perturbation_response(config)
    assert outcome.status == "PASS"
    assert len(outcome.metrics["families"]["intensity_noise"]) == 2


def test_policy_arm(release_root: Path) -> None:
    from private_vascular_motion_generation.runner.studies import run_policy_comparison

    config = _arm(release_root, "policy_comparison")
    config.payload["run"]["private_size"] = 2
    outcome = run_policy_comparison(config, steps=12)
    assert outcome.status == "PASS"
    assert {"standard_of_care", "control_barrier_function", "reactive_potential_field"} <= set(
        outcome.metrics["strategies"]
    )
    assert outcome.metrics["force_margin"] > 0.0


def test_null_results_arm(release_root: Path) -> None:
    outcome = execute_study("null_results", release_root, overrides=list(SMOKE_OVERRIDES))
    assert outcome.status == "PASS"
    assert outcome.metrics["estimator_null"]["retained_null"] is True
    assert outcome.metrics["inventory_total"] == 23
    assert "interaction_permutation" in outcome.metrics


def test_interaction_arm(release_root: Path) -> None:
    outcome = execute_study("component_interaction", release_root, overrides=list(SMOKE_OVERRIDES))
    assert outcome.status == "PASS"
    pairwise = outcome.metrics["pairwise"]
    assert pairwise["all_super_additive"] is True
    assert pairwise["max_pair"] == "shaped noise"
    assert pairwise["max_interaction_ratio"] == pytest.approx(1.51, abs=0.01)
    assert pairwise["prerequisites_are_not_synergy"] is True
    assert len(pairwise["prerequisite_pairs"]) == 1


def test_metric_suite_arm(release_root: Path) -> None:
    outcome = execute_study("metric_suite", release_root, overrides=list(SMOKE_OVERRIDES))
    assert outcome.status == "PASS"
    assert 0.0 <= outcome.metrics["dice_projection"] <= 1.0
    assert outcome.metrics["ade_mm"] >= 0.0


def test_release_path_arm(release_root: Path) -> None:
    outcome = execute_study("release_path", release_root, overrides=list(SMOKE_OVERRIDES))
    assert outcome.status == "PASS"
    assert outcome.metrics["released_procedures"] == 6
    assert outcome.metrics["subspace_rank"] == 3
