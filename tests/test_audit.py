"""Privacy-audit tests for the membership-inference statistic and the distance proxy.

Ref: Sec. 2.2, Table 2.
"""

from __future__ import annotations

import pytest
import torch

from private_vascular_motion_generation.analysis.delong import NO_ADVANTAGE_REFERENCE
from private_vascular_motion_generation.audit.distance import (
    dcr_against_marginals,
    dcr_summary,
    distance_to_closest_record,
    duplicate_fraction,
)
from private_vascular_motion_generation.audit.membership import (
    audit_gap,
    audit_is_attack_based,
    frequency_calibrated_scores,
    loss_scores,
    member_nonmember_auc,
    paired_audit_test,
)


def test_no_advantage_benchmark_is_one_half() -> None:
    assert NO_ADVANTAGE_REFERENCE == 0.5
    assert audit_is_attack_based()["primary_evidence"] == "membership-inference attack"


def test_separated_populations_give_a_high_area() -> None:
    generator = torch.Generator().manual_seed(51)
    members = torch.randn(200, generator=generator) + 2.0
    nonmembers = torch.randn(200, generator=generator) - 2.0
    estimate = member_nonmember_auc(members, nonmembers)
    assert estimate.auc > 0.9
    assert not (estimate.low <= 0.5 <= estimate.high)


def test_overlapping_populations_cover_the_benchmark() -> None:
    generator = torch.Generator().manual_seed(52)
    estimate = member_nonmember_auc(torch.randn(150, generator=generator), torch.randn(150, generator=generator))
    assert estimate.low <= 0.5 <= estimate.high


def test_frequency_calibration_recentres_bins() -> None:
    generator = torch.Generator().manual_seed(53)
    scores = torch.randn(120, generator=generator) + 0.5
    features = torch.randn(120, 4, generator=generator)
    calibrated = frequency_calibrated_scores(scores, features, bins=6)
    assert abs(float(calibrated.mean().item())) < 0.05


def test_frequency_calibration_rejects_bad_bins() -> None:
    with pytest.raises(ValueError):
        frequency_calibrated_scores(torch.randn(8), torch.randn(8, 2), bins=1)


def test_loss_scores_are_negative_log_likelihoods() -> None:
    model = torch.nn.Linear(3, 1)
    features = torch.randn(10, 3)
    targets = (torch.rand(10) > 0.5).to(torch.float32)
    scores = loss_scores(model, features, targets)
    assert tuple(scores.shape) == (10,)
    assert bool(torch.all(scores > 0.0).item())


def test_audit_gap_reports_both_intervals() -> None:
    generator = torch.Generator().manual_seed(54)
    private = member_nonmember_auc(torch.randn(120, generator=generator), torch.randn(120, generator=generator))
    non_private = member_nonmember_auc(
        torch.randn(120, generator=generator) + 1.0, torch.randn(120, generator=generator) - 1.0
    )
    report = audit_gap(non_private, private)
    assert report["delta_auc"] > 0.0
    assert report["private_includes_no_advantage"] == 1.0


def test_paired_audit_test_runs_on_aligned_vectors() -> None:
    generator = torch.Generator().manual_seed(55)
    labels = torch.cat([torch.ones(50), torch.zeros(50)])
    first = torch.randn(100, generator=generator) + labels
    second = torch.randn(100, generator=generator)
    report = paired_audit_test(first, second, labels)
    assert report["z"] > 0.0


def test_distance_to_closest_record_detects_duplicates() -> None:
    generator = torch.Generator().manual_seed(56)
    real = torch.randn(16, 4, generator=generator)
    generated = real[:5] + 0.001 * torch.randn(5, 4, generator=generator)
    distances = distance_to_closest_record(generated, real)
    assert float(distances.max().item()) < 0.01
    assert duplicate_fraction(distances, 0.01) == pytest.approx(1.0)


def test_distance_rejects_mismatched_extents() -> None:
    with pytest.raises(ValueError):
        distance_to_closest_record(torch.randn(4, 3), torch.randn(6, 4))


def test_dcr_summary_and_marginals() -> None:
    generator = torch.Generator().manual_seed(57)
    summary = dcr_summary(torch.tensor([1.0, 2.0, 3.0]))
    assert summary["min"] == 1.0 and summary["max"] == 3.0
    report = dcr_against_marginals(
        torch.randn(4, 2, 3, generator=generator),
        torch.randn(4, 6, 3, generator=generator),
        torch.randn(6, 2, 3, generator=generator),
        torch.randn(6, 6, 3, generator=generator),
    )
    assert "appearance_median" in report and "motion_median" in report
