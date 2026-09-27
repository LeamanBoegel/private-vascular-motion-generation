"""Metric tests: fidelity, geometry, motion, safety and clinical endpoints.

Ref: Sec. 4.10, Table 1, Table 5, Table 6.
"""

from __future__ import annotations

import pytest
import torch

from private_vascular_motion_generation.metrics.clinical import (
    cohen_kappa,
    discrimination_accuracy,
    reader_agreement_gain,
    task_success,
    three_way_reader_contrast,
    wilson_interval,
)
from private_vascular_motion_generation.metrics.fidelity import (
    frechet_distance,
    polynomial_mmd,
    structural_similarity,
)
from private_vascular_motion_generation.metrics.safety import (
    clears_mcid,
    contact_event_reduction,
    cross_region_bound_holds,
    force_envelope_margin,
    non_inferiority_margin,
    per_site_spread,
    safety_from_rollouts,
    within_non_inferiority_margin,
)
from private_vascular_motion_generation.metrics.segmentation import (
    dice_by_stratum,
    dice_overlap,
    geometry_report,
    percentile_hausdorff,
)
from private_vascular_motion_generation.metrics.trajectory import (
    PATH_EFFICIENCY_ARTIFACT_CEILING,
    average_displacement_error,
    deviation_from_centreline,
    dynamic_time_warping,
    final_displacement_error,
    motion_report,
    path_efficiency_is_artifact,
    path_length,
    path_length_efficiency,
    trajectory_smoothness,
)


def test_frechet_distance_vanishes_for_identical_clouds(torch_generator) -> None:
    cloud = torch.randn(24, 5, generator=torch_generator)
    assert frechet_distance(cloud, cloud) == pytest.approx(0.0, abs=1e-6)


def test_frechet_distance_reads_a_mean_shift(torch_generator) -> None:
    cloud = torch.randn(24, 5, generator=torch_generator)
    assert frechet_distance(cloud, cloud + 3.0) > 1.0


def test_frechet_distance_rejects_singletons() -> None:
    with pytest.raises(ValueError):
        frechet_distance(torch.randn(1, 4), torch.randn(1, 4))


def test_polynomial_mmd_orders_near_and_far(torch_generator) -> None:
    cloud = torch.randn(20, 4, generator=torch_generator)
    near = cloud + 0.02 * torch.randn(20, 4, generator=torch_generator)
    far = cloud + 4.0
    assert polynomial_mmd(cloud, near) < polynomial_mmd(cloud, far)


def test_ssim_rewards_similarity(torch_generator) -> None:
    image = torch.rand(1, 16, 16, generator=torch_generator)
    near = image + 0.01 * torch.randn(1, 16, 16, generator=torch_generator)
    far = torch.rand(1, 16, 16, generator=torch_generator)
    assert structural_similarity(image, near) > structural_similarity(image, far)


def test_ssim_rejects_a_window_larger_than_the_image() -> None:
    with pytest.raises(ValueError):
        structural_similarity(torch.rand(1, 4, 4), torch.rand(1, 4, 4), window=7)


def test_dice_and_hausdorff_on_a_simple_mask() -> None:
    mask = torch.zeros(16, 16)
    mask[4:12, 6:8] = 1.0
    assert dice_overlap(mask, mask) == pytest.approx(1.0)
    assert percentile_hausdorff(mask, mask) == pytest.approx(0.0, abs=1e-6)
    shifted = torch.roll(mask, 1, dims=0)
    assert 0.0 < dice_overlap(mask, shifted) < 1.0


def test_geometry_report_over_a_frame_stack() -> None:
    mask = torch.zeros(12, 12, 3)
    mask[3:9, 5:7, :] = 1.0
    report = geometry_report(mask, mask)
    assert report["dice"] == pytest.approx(1.0)
    assert report["hausdorff95"] == pytest.approx(0.0, abs=1e-6)


def test_dice_by_stratum_groups_procedures() -> None:
    masks = torch.zeros(3, 12, 12)
    masks[:, 3:9, 5:7] = 1.0
    resolved = dice_by_stratum(masks, masks, ["simple", "complex", "simple"])
    assert resolved["simple"] == pytest.approx(1.0)
    assert set(resolved) == {"simple", "complex"}


def test_trajectory_errors_and_efficiency(torch_generator) -> None:
    trajectory = torch.cumsum(torch.randn(20, 3, generator=torch_generator) * 0.1, dim=0)
    assert average_displacement_error(trajectory, trajectory) == pytest.approx(0.0, abs=1e-9)
    assert final_displacement_error(trajectory + 1.0, trajectory) == pytest.approx(3.0**0.5, rel=1e-6)
    assert dynamic_time_warping(trajectory, trajectory) == pytest.approx(0.0, abs=1e-9)
    assert path_length_efficiency(trajectory, trajectory) == pytest.approx(1.0, rel=1e-9)
    assert path_length(trajectory) > 0.0
    assert trajectory_smoothness(trajectory) > 0.0
    assert deviation_from_centreline(trajectory, trajectory) == pytest.approx(0.0, abs=1e-9)


def test_path_efficiency_artifact_ceiling() -> None:
    assert path_efficiency_is_artifact(PATH_EFFICIENCY_ARTIFACT_CEILING + 0.01)
    assert not path_efficiency_is_artifact(PATH_EFFICIENCY_ARTIFACT_CEILING - 0.01)


def test_motion_report_keys(torch_generator) -> None:
    trajectory = torch.cumsum(torch.randn(18, 3, generator=torch_generator) * 0.1, dim=0)
    report = motion_report(trajectory, trajectory + 0.05, trajectory)
    assert set(report) == {
        "ade_mm",
        "fde_mm",
        "dtw",
        "smoothness",
        "path_efficiency",
        "centreline_deviation",
    }


def test_safety_row_counts_planted_contacts() -> None:
    forces = torch.zeros(3, 50)
    forces[:, 10:14] = 0.25
    row = safety_from_rollouts(
        label="fixture",
        forces=forces,
        successful=torch.tensor([1.0, 1.0, 0.0]),
        fluoroscopy_min=11.0,
        contrast_ml=130.0,
        navigation_min=36.0,
        path_efficiency=0.8,
    )
    assert row.contacts_per_procedure == pytest.approx(1.0)
    assert row.peak_force_n == pytest.approx(0.25)
    assert row.contacts_per_thousand_frames() == pytest.approx(20.0)


def test_safety_thresholds() -> None:
    assert force_envelope_margin(0.41, 0.76) == pytest.approx(1.0 - 0.41 / 0.76)
    assert clears_mcid(contact_event_reduction(1.4, 2.4))
    assert not clears_mcid(0.2)
    assert within_non_inferiority_margin(non_inferiority_margin(0.902, 0.895))
    assert per_site_spread({"A": 0.905, "B": 0.889, "C": 0.871}) == pytest.approx(3.4, abs=0.05)
    assert cross_region_bound_holds(0.897, 0.871)


def test_wilson_interval_matches_the_score_definition() -> None:
    estimate = wilson_interval(386, 412)
    assert estimate.point == pytest.approx(386 / 412)
    assert estimate.low < estimate.point < estimate.high
    assert estimate.low == pytest.approx(0.909, abs=0.001)


def test_wilson_interval_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_task_success_reports_its_own_denominator() -> None:
    estimate = task_success(torch.tensor([1.0, 1.0, 0.0, 1.0]))
    assert estimate.denominator == 4
    assert estimate.point == pytest.approx(0.75)


def test_cohen_kappa_bounds() -> None:
    first = torch.tensor([0, 1, 1, 0, 1, 0])
    assert cohen_kappa(first, first, 2) == pytest.approx(1.0)
    second = 1 - first
    assert cohen_kappa(first, second, 2) == pytest.approx(-1.0)


def test_reader_contrast_and_gain() -> None:
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
    clinician = torch.tensor([0.6, 0.4, 0.7, 0.3])
    model = torch.tensor([0.2, 0.8, 0.9, 0.1])
    fused = torch.tensor([0.3, 0.7, 0.8, 0.2])
    report = three_way_reader_contrast(clinician, model, fused, labels)
    assert report["model_alone"] == pytest.approx(1.0)
    assert reader_agreement_gain(0.62, 0.74) == pytest.approx(0.12)
    assert discrimination_accuracy(model, labels) == pytest.approx(1.0)
