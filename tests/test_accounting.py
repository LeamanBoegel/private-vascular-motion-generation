"""Accounting tests: Rényi divergence, composition, clipping and the privacy tuple.

Ref: Eq. (3), Eq. (4), Sec. 4.4, Sec. 4.6.
"""

from __future__ import annotations

import math

import pytest
import torch

from private_vascular_motion_generation.accounting.clipping import (
    clip_procedure_gradient,
    clip_workload,
    clipping_norm_report,
    compare_clipping_norms,
    no_post_noise_clipping,
)
from private_vascular_motion_generation.accounting.orders import (
    DEFAULT_ORDER_GRID,
    OrderGrid,
    integer_grid,
    interpolated_grid,
)
from private_vascular_motion_generation.accounting.renyi import (
    PrivacyBudget,
    composed_rdp,
    convert_to_delta,
    convert_to_epsilon,
    epsilon_at_parity,
    monotone_in_noise,
    prefix_amplification_term,
    rdp_binomial_integer,
    rdp_from_noise_multiplier,
    rdp_sampled_gaussian,
    rdp_shaped_mechanism,
    structure_versus_concentration,
)
from private_vascular_motion_generation.accounting.tuple import (
    TUPLE_FIELDS,
    PrivacyTuple,
    bare_budget_warning,
    conservative_ratio,
    tuple_is_complete,
)


def test_closed_form_matches_quadrature() -> None:
    for order in (2, 4, 8, 16):
        closed = rdp_binomial_integer(order, 0.05, 3.5)
        numeric = rdp_sampled_gaussian(float(order), 0.05, 3.5)
        assert closed == pytest.approx(numeric, rel=1e-8)


def test_unit_sampling_reduces_to_the_gaussian_closed_form() -> None:
    assert rdp_binomial_integer(6, 1.0, 2.0) == pytest.approx(6 / (2 * 2.0**2))
    assert rdp_sampled_gaussian(6.0, 1.0, 2.0) == pytest.approx(6 / (2 * 2.0**2))


def test_rdp_grows_with_order_at_fixed_noise() -> None:
    assert rdp_binomial_integer(32, 0.02, 4.0) > rdp_binomial_integer(2, 0.02, 4.0)


def test_order_grid_validation() -> None:
    with pytest.raises(ValueError):
        OrderGrid((1.5,))
    with pytest.raises(ValueError):
        OrderGrid((4.0, 2.0))
    assert len(DEFAULT_ORDER_GRID.orders) > 8
    assert integer_grid([2, 4, 8]).orders == (2.0, 4.0, 8.0)
    assert interpolated_grid(2.0, 8.0, 4).orders[0] == pytest.approx(2.0)


def test_composition_is_additive_in_the_round_count() -> None:
    grid = integer_grid([2, 4, 8, 16])
    single = composed_rdp(grid, 100, 0.01, 4.0)
    double = composed_rdp(grid, 200, 0.01, 4.0)
    assert torch.allclose(double, single * 2.0, rtol=1e-9)


def test_conversion_prefers_the_lowest_epsilon_over_the_grid() -> None:
    grid = integer_grid([2, 4, 8, 16, 32])
    accumulator = composed_rdp(grid, 500, 0.01, 3.0)
    window = 4
    budget = convert_to_epsilon(accumulator, grid, 1e-5, window_length=window)
    manual = min(
        float(accumulator[index].item())
        + math.log(1e5) / (grid.orders[index] - 1.0)
        + math.log2(window + 1) ** 2 / grid.orders[index]
        for index in range(len(grid.orders))
    )
    assert budget.epsilon == pytest.approx(manual, rel=1e-9)
    converted = [
        float(accumulator[index].item())
        + math.log(1e5) / (grid.orders[index] - 1.0)
        + math.log2(window + 1) ** 2 / grid.orders[index]
        for index in range(len(grid.orders))
    ]
    assert budget.order == grid.orders[converted.index(min(converted))]
    assert budget.delta == 1e-5


def test_convert_to_delta_inverts_the_conversion() -> None:
    grid = integer_grid([2, 4, 8, 16])
    accumulator = composed_rdp(grid, 200, 0.02, 4.0)
    budget = convert_to_epsilon(accumulator, grid, 1e-5)
    recovered = convert_to_delta(accumulator, grid, budget.epsilon)
    assert recovered.delta <= 1e-5 * (1.0 + 1e-6)


def test_more_noise_never_costs_more_privacy() -> None:
    assert monotone_in_noise(integer_grid([2, 4, 8, 16]), 400, 0.02, 2.0, 8.0, 1e-5)


def test_shaped_charge_uses_the_normalised_sensitivity() -> None:
    charging = rdp_shaped_mechanism(8, 0.01, 5.4, 2.6, 0.19, 0.85)
    isotropic = rdp_shaped_mechanism(8, 0.01, 3.9, 3.9, 0.41, 0.85)
    assert charging < isotropic


def test_prefix_slack_grows_with_the_window() -> None:
    assert prefix_amplification_term(128, 8.0) > prefix_amplification_term(4, 8.0)
    with pytest.raises(ValueError):
        prefix_amplification_term(0, 8.0)


def test_budget_union_keeps_the_stronger_epsilon() -> None:
    weak = PrivacyBudget(epsilon=4.0, delta=1e-5, order=8.0)
    strong = PrivacyBudget(epsilon=2.0, delta=1e-5, order=16.0)
    assert strong.conservative_union(weak).epsilon == 4.0


def test_clipping_bounds_every_procedure_norm() -> None:
    generator = torch.Generator().manual_seed(14)
    workload = torch.randn(6, 5, 4, generator=generator) * 3.0
    clipped, scales = clip_workload(workload, 0.85)
    norms = torch.linalg.norm(clipped.reshape(6, -1), dim=1)
    assert float(norms.max().item()) <= 0.85 + 1e-6
    assert bool(torch.all(scales <= 1.0).item())


def test_clipping_is_idempotent() -> None:
    generator = torch.Generator().manual_seed(15)
    workload = torch.randn(4, 6, generator=generator) * 2.0
    clipped, _ = clip_workload(workload, 0.85)
    twice, _ = clip_workload(clipped, 0.85)
    assert torch.allclose(clipped, twice, atol=1e-7)


def test_single_norm_clip_reports_the_scale() -> None:
    gradient = torch.full((4, 4), 4.0)
    outcome = clip_procedure_gradient(gradient, 0.8)
    assert outcome.was_clipped
    assert outcome.scale == pytest.approx(0.8 / outcome.observed_norm)
    assert no_post_noise_clipping(outcome.clipped, outcome.clipped)


def test_clipping_norm_report_and_comparison() -> None:
    report = clipping_norm_report(0.85, "procedure")
    assert report["per_procedure_single_norm"] is True
    assert compare_clipping_norms(0.85, 1.0)["ratio"] == pytest.approx(1.0 / 0.85)


def test_privacy_tuple_is_complete_and_ordered() -> None:
    entry = PrivacyTuple(
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
    payload = entry.to_dict()
    assert all(name in payload for name in TUPLE_FIELDS)
    assert tuple_is_complete(payload)
    assert bare_budget_warning({"epsilon": 2.0}) is not None
    assert bare_budget_warning(payload) is None
    assert entry.weaker_unit() == "example"


def test_privacy_tuple_validation() -> None:
    with pytest.raises(ValueError):
        PrivacyTuple(
            epsilon=2.0,
            delta=1e-5,
            accountant="gdp",
            order=16.0,
            steps_times_sampling=1.0,
            clipping_norm=0.85,
            sigma_parallel=5.4,
            sigma_perpendicular=2.6,
            rank=12,
            dimension=1024,
        )


def test_conservative_ratio_reads_downward() -> None:
    reading = conservative_ratio(2.0, 4.8)
    assert reading["quotient"] == pytest.approx(2.4)
    assert reading["reading"] == "conservative_lower_bound"


def test_structure_and_concentration_are_reported_apart() -> None:
    report = structure_versus_concentration(5.4, 2.6, 0.19)
    assert report["concentration_condition_gain_rho"] == 0.19
    assert "prefix" in str(report["structural_condition"])


def test_parity_budget_is_finite_over_the_grid() -> None:
    budget = epsilon_at_parity(integer_grid([2, 4, 8, 16, 32]), 12000, 0.0146, 2.4, 1e-5)
    assert 0.0 < budget.epsilon < 1e3
    assert budget.order in (2.0, 4.0, 8.0, 16.0, 32.0)


def test_rdp_entry_points_reject_invalid_input() -> None:
    with pytest.raises(ValueError):
        rdp_binomial_integer(1, 0.1, 3.0)
    with pytest.raises(ValueError):
        rdp_from_noise_multiplier(4.0, 0.0, 3.0)
    with pytest.raises(ValueError):
        prefix_amplification_term(8, 1.0)
