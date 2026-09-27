"""Contrast, interval, multiplicity and resampling tests.

Ref: Sec. 2.8, Sec. 4.9, Tables 3 and 4, Supplementary Tables SE1, T7, SC2.
"""

from __future__ import annotations

import pytest
import torch

from private_vascular_motion_generation.analysis.delong import (
    auc,
    delong_interval,
    delong_paired_test,
    delong_variance,
)
from private_vascular_motion_generation.analysis.factorial import (
    CELLS,
    FactorialCell,
    budget_moves_with_conditioning,
    conditioning_main_effect,
    contrast,
    full_contrast_table,
    interaction,
    interaction_resolvable,
    resolution_main_effect,
    rho_invariance_across_resolution,
)
from private_vascular_motion_generation.analysis.interaction import (
    PairwiseInteraction,
    component_drop,
    largest_pair_is_design_pair,
    pairwise_from_full,
    prerequisites_are_not_synergy,
    report_pairwise,
)
from private_vascular_motion_generation.analysis.multiplicity import (
    benjamini_hochberg,
    declare_families,
    exploratory_inventory_size,
    holm_bonferroni,
    inventory_total,
)
from private_vascular_motion_generation.analysis.null_fence import (
    NULL_FENCE_RELATIVE,
    fence_verdict,
    markdown_stars,
    minimal_clinically_important_difference_met,
    retained_null_results,
    unmarked_because_inside_interval,
)
from private_vascular_motion_generation.analysis.powerlaw import (
    budget_floor_reached,
    fit_power_law,
    reference_corpus_saturation,
    residuals_in_units,
    sample_efficiency_exponent,
    scaling_is_monotone,
    systematic_at_size,
)
from private_vascular_motion_generation.analysis.resampling import (
    paired_bootstrap_delta,
    paired_permutation_test,
    two_proportion_z,
    wilcoxon_signed_rank,
)
from private_vascular_motion_generation.analysis.variance import (
    beats_unconstrained,
    coefficient_of_variation_verdict,
    dispersion,
    dispersion_table,
    parity_band_verdict,
    run_to_run_floor,
    worst_initialisation,
)


def _cells() -> dict[str, FactorialCell]:
    return {
        "on_1x": FactorialCell("on_1x", 45.3, 0.148, 0.842, 0.19, 2.0),
        "on_2x": FactorialCell("on_2x", 45.9, 0.156, 0.833, 0.19, 2.1),
        "off_1x": FactorialCell("off_1x", 52.4, 0.233, 0.762, 0.71, 5.6),
        "off_2x": FactorialCell("off_2x", 58.1, 0.271, 0.744, 0.70, 6.2),
    }


def test_delong_auc_matches_the_rank_definition() -> None:
    scores = torch.tensor([0.9, 0.8, 0.4, 0.7, 0.3, 0.5, 0.6, 0.1])
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    rank_definition = float((positives[:, None] > negatives[None, :]).to(torch.float64).mean().item())
    assert auc(scores, labels) == pytest.approx(rank_definition, abs=1e-12)
    assert auc(scores, labels) == pytest.approx(13.0 / 16.0, abs=1e-12)


def test_delong_interval_is_ordered_and_bounded() -> None:
    scores = torch.randn(200, generator=torch.Generator().manual_seed(31))
    labels = (torch.randn(200, generator=torch.Generator().manual_seed(32)) > 0).to(torch.float64)
    estimate = delong_interval(scores, labels)
    assert estimate.low <= estimate.auc <= estimate.high
    assert estimate.low >= 0.0 and estimate.high <= 1.0
    assert delong_variance(scores, labels) >= 0.0


def test_delong_paired_test_separates_informative_scores() -> None:
    labels = torch.cat([torch.ones(60), torch.zeros(60)])
    informative = torch.cat([torch.randn(60) + 1.5, torch.randn(60) - 1.5])
    useless = torch.randn(120, generator=torch.Generator().manual_seed(33))
    report = delong_paired_test(informative, useless, labels)
    assert report["delta_auc"] > 0.2
    assert report["z"] > 4.0


def test_factorial_contrasts_recompute_from_the_cells() -> None:
    cells = _cells()
    table = full_contrast_table(cells)
    assert conditioning_main_effect(cells, "fid") == pytest.approx(-9.65, abs=1e-6)
    assert resolution_main_effect(cells, "fid") == pytest.approx(3.15, abs=1e-6)
    assert interaction(cells, "fid") == pytest.approx(5.1, abs=1e-6)
    assert contrast(cells, "fid").dominance_ratio() == pytest.approx(3.063, abs=0.01)
    assert set(table) == {"fid", "dtw", "contact_event_f1"}


def test_factorial_auxiliary_reports() -> None:
    cells = _cells()
    assert set(cells) == set(CELLS)
    rho = rho_invariance_across_resolution(cells)
    assert abs(rho["on_1x_minus_on_2x"]) == pytest.approx(0.0, abs=1e-9)
    assert budget_moves_with_conditioning(cells)
    assert not interaction_resolvable(0.009, 0.01)
    assert interaction_resolvable(0.05, 0.01)


def test_pairwise_interaction_ratio_and_relation() -> None:
    entry = pairwise_from_full(0.842, 0.771, 0.812, 0.719, "coupling latent", "shaped noise")
    assert entry.interaction_ratio() == pytest.approx(0.123 / 0.101, rel=1e-3)
    assert entry.relation() == "super-additive"
    report = report_pairwise([entry])
    assert report["all_super_additive"] is True
    assert component_drop(0.842, 0.771) == pytest.approx(0.071)


def test_pairwise_prerequisite_and_largest_pair() -> None:
    prerequisite = PairwiseInteraction(
        left="public-reference estimator", right="shaped noise", drop_both=0.032, drop_left_plus_right=0.032
    )
    assert prerequisite.relation() == "prerequisite, not synergy"
    assert prerequisites_are_not_synergy([prerequisite])
    design = pairwise_from_full(0.842, 0.771, 0.812, 0.719, "coupling latent", "shaped noise")
    weaker = pairwise_from_full(0.842, 0.771, 0.819, 0.734, "coupling latent", "factorization")
    assert largest_pair_is_design_pair([prerequisite, design, weaker], ("coupling latent", "shaped noise"))


def test_holm_step_down_monotone_and_conservative() -> None:
    rows = holm_bonferroni(["a", "b", "c"], [0.001, 0.02, 0.9])
    assert rows[0].p_corrected == pytest.approx(0.003)
    assert all(row.correction == "holm" for row in rows)
    assert rows[0].passes and not rows[2].passes
    corrected = [row.p_corrected for row in rows]
    assert corrected == sorted(corrected)


def test_benjamini_hochberg_step_up() -> None:
    rows = benjamini_hochberg(["a", "b", "c", "d"], [0.001, 0.02, 0.3, 0.9])
    assert [row.p_corrected for row in rows] == [
        pytest.approx(0.004, rel=1e-9),
        pytest.approx(0.04, rel=1e-9),
        pytest.approx(0.4, rel=1e-9),
        pytest.approx(0.9, rel=1e-9),
    ]
    assert rows[0].passes
    assert not rows[-1].passes


def test_multiplicity_inventory_shape() -> None:
    families = declare_families()
    assert len(families) == 7
    assert inventory_total() == exploratory_inventory_size() == 23


def test_null_fence_marks_and_retention() -> None:
    retained = fence_verdict("estimator", 0.842, 0.840)
    assert retained.is_retained_null()
    assert retained.stars() == 0
    strong = fence_verdict("contact events", 2.4, 1.4)
    assert strong.stars() == 3
    assert markdown_stars(2) == "**"
    assert len(retained_null_results()) == 3
    assert pytest.approx(0.01) == NULL_FENCE_RELATIVE


def test_null_fence_auxiliary_rules() -> None:
    assert unmarked_because_inside_interval(-0.7, 1.2)
    assert minimal_clinically_important_difference_met(1.0, 0.5)
    with pytest.raises(ValueError):
        markdown_stars(4)


def test_power_law_recovers_a_planted_exponent() -> None:
    sizes = torch.tensor([100.0, 200.0, 400.0, 800.0, 1600.0], dtype=torch.float64)
    values = 3.0 * sizes**0.25
    fit = fit_power_law(sizes, values)
    assert fit.exponent == pytest.approx(0.25, abs=1e-9)
    assert fit.r_squared > 0.999
    assert float(residuals_in_units(fit, sizes, values).abs().max().item()) < 1e-6


def test_power_law_diagnostics() -> None:
    sizes = torch.tensor([600.0, 1200.0, 2400.0, 4800.0])
    values = torch.tensor([0.774, 0.801, 0.822, 0.836])
    assert scaling_is_monotone(sizes, values)
    assert sample_efficiency_exponent(sizes, values)["exponent"] > 0.0
    assert budget_floor_reached(sizes, torch.tensor([3.4, 2.9, 2.4, 2.0])).tolist() == [4800.0]
    assert budget_floor_reached(sizes, torch.tensor([3.4, 2.9, 2.4, 2.1])).numel() == 0
    assert reference_corpus_saturation(sizes, torch.tensor([0.827, 0.832, 0.836, 0.836]), tolerance=0.001)
    residuals = torch.tensor([1.0, 1.0, -1.0, -1.0])
    assert systematic_at_size(residuals, sizes, 600.0)


def test_resampling_tests() -> None:
    differences = torch.tensor([0.40, 0.35, 0.38, 0.42, 0.36, 0.41, 0.37, 0.39, 0.43, 0.34, 0.40, 0.38])
    first = torch.tensor([0.90, 0.85, 0.88, 0.92, 0.86, 0.91, 0.87, 0.89, 0.93, 0.84, 0.90, 0.88])
    second = first - differences
    assert paired_permutation_test(first, second, 1000, 41)["p_value"] < 0.05
    assert wilcoxon_signed_rank(first, second)["w"] == pytest.approx(0.0)
    interval = paired_bootstrap_delta(first, second, 500, 42)
    assert interval["low"] > 0.0 and interval["high"] > interval["low"]
    proportions = two_proportion_z(1133, 1204, 1077, 1204)
    assert proportions["z"] == pytest.approx(4.13, abs=0.05)


def test_resampling_rejects_misaligned_input() -> None:
    with pytest.raises(ValueError):
        paired_permutation_test(torch.zeros(3), torch.zeros(4), 10, 1)
    with pytest.raises(ValueError):
        two_proportion_z(1, 0, 1, 10)


def test_dispersion_and_verdicts() -> None:
    values = torch.tensor([45.1, 45.3, 45.5])
    report = dispersion("fid", values)
    assert report.coefficient_of_variation() > 0.0
    assert set(dispersion_table({"fid": values})) == {"fid"}
    assert beats_unconstrained(44.9, 44.9)
    assert not beats_unconstrained(46.2, 44.9)
    assert parity_band_verdict(0.7, (0.5, 1.0), 0.01) == "inside_parity_band"
    assert coefficient_of_variation_verdict(0.0127, 0.0127) == "within_tolerance"
    assert worst_initialisation(values) == pytest.approx(45.5)
    assert run_to_run_floor(values, 0.9) > 0.0
