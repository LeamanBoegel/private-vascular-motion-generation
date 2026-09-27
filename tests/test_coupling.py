"""Coupling head, canonical-correlation solver and prefix-sum factorisation tests.

Ref: Sec. 4.4, Algorithm 1 step 7, Algorithm 2 steps 6-8.
"""

from __future__ import annotations

import pytest
import torch

from private_vascular_motion_generation.coupling.cca import (
    canonical_orthonormality,
    coupling_loss,
    generalised_eigen_residual,
    inverse_sqrt,
    paired_covariances,
    solve_canonical,
)
from private_vascular_motion_generation.coupling.head import CouplingHead, CouplingHeadSpec
from private_vascular_motion_generation.coupling.latent import (
    CouplingLatentSpec,
    draw_latent,
    encode_procedure_identity,
)
from private_vascular_motion_generation.coupling.prefix_sum import (
    REPORTED_COMPOSITION_MULTIPLIER,
    WINDOW_LADDER,
    WindowSpec,
    composition_multiplier_ladder,
    fit_log_ladder,
    group_multiplier,
    prefix_sum_inverse,
    prefix_sum_scan,
    truncation_rate,
    windowed_workload,
)


@pytest.fixture()
def paired_gradients() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    first = torch.randn(48, 10, generator=generator)
    mixing = torch.randn(10, 10, generator=generator)
    second = first @ mixing + 0.2 * torch.randn(48, 10, generator=generator)
    return first, second


def test_inverse_sqrt_inverts_its_square() -> None:
    generator = torch.Generator().manual_seed(4)
    matrix = torch.randn(8, 8, generator=generator)
    matrix = matrix @ matrix.transpose(0, 1) + 0.5 * torch.eye(8)
    root = inverse_sqrt(matrix)
    identity = root @ matrix @ root
    assert torch.allclose(identity, torch.eye(8), atol=1e-6)


def test_canonical_solution_solves_the_generalised_problem(paired_gradients) -> None:
    first, second = paired_gradients
    triple = paired_covariances(first, second)
    solution = solve_canonical(triple)
    for index in range(3):
        assert generalised_eigen_residual(triple, solution, index) < 1e-4
    assert canonical_orthonormality(solution, triple) < 1e-4


def test_canonical_correlations_are_bounded_and_ordered(paired_gradients) -> None:
    solution = solve_canonical(paired_covariances(*paired_gradients))
    assert bool(torch.all(solution.correlations <= 1.0 + 1e-9).item())
    assert bool(torch.all(solution.correlations[:-1] >= solution.correlations[1:] - 1e-9).item())


def test_coupling_loss_is_the_negative_spectral_density(paired_gradients) -> None:
    solution = solve_canonical(paired_covariances(*paired_gradients))
    loss = float(coupling_loss(solution, 4).item())
    assert loss == pytest.approx(-solution.spectral_density(4), abs=1e-9)


def test_head_reports_a_bounded_spectrum(paired_gradients) -> None:
    head = CouplingHead(spec=CouplingHeadSpec(rank=4)).fit(*paired_gradients)
    assert 0.0 <= head.spectral_density() <= 1.0
    assert head.require_projector().rank == 4
    quiet = CouplingHead(spec=CouplingHeadSpec(rank=4, reduce_head=True)).fit(*paired_gradients)
    assert float(quiet.loss().item()) == 0.0


def test_head_requires_a_fit_before_use() -> None:
    head = CouplingHead(spec=CouplingHeadSpec(rank=2))
    with pytest.raises(RuntimeError):
        head.loss()
    with pytest.raises(RuntimeError):
        head.require_projector()


def test_shared_latent_is_unit_norm_for_both_branches() -> None:
    spec = CouplingLatentSpec(dimension=12)
    codes = draw_latent(spec, 5, torch.Generator().manual_seed(6))
    norms = codes.codes.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    index = torch.arange(5)
    appearance, motion = codes.project(torch.zeros(5, 12), torch.zeros(5, 12), index)
    assert torch.allclose(appearance, motion)


def test_prefix_sum_round_trip(torch_generator) -> None:
    signal = torch.randn(9, 3, generator=torch_generator)
    assert torch.allclose(prefix_sum_inverse(prefix_sum_scan(signal)), signal, atol=1e-6)


def test_window_spec_requires_a_uniform_tiling() -> None:
    with pytest.raises(ValueError):
        WindowSpec(length=4, micro_steps=6)
    window = WindowSpec(length=4, micro_steps=8)
    assert window.stride == 2
    assert window.structural_condition_holds()


def test_windowed_workload_layout(torch_generator) -> None:
    payload = torch.randn(5, 6, generator=torch_generator)
    window = WindowSpec(length=3, micro_steps=3)
    reshaped = windowed_workload(payload, window)
    assert tuple(reshaped.shape) == (5, 18)


def test_group_multiplier_and_truncation_rate() -> None:
    assert group_multiplier(WindowSpec(4, 4)) < group_multiplier(WindowSpec(128, 128))
    lengths = torch.tensor([4.0, 8.0, 16.0])
    assert truncation_rate(lengths, 8) == pytest.approx(1.0 / 3.0)


def test_reported_window_ladder_is_polylogarithmic() -> None:
    ladder = composition_multiplier_ladder()
    assert ladder["cost_ladder_monotone"]
    fit = fit_log_ladder(WINDOW_LADDER, REPORTED_COMPOSITION_MULTIPLIER)
    assert fit["r_squared"] > 0.95


def test_encode_procedure_identity_shape(torch_generator) -> None:
    tokens = torch.randn(16, 8, generator=torch_generator)
    weights = torch.randn(8, 5, generator=torch_generator)
    code = encode_procedure_identity(tokens, CouplingLatentSpec(dimension=5), weights)
    assert tuple(code.shape) == (5,)
    assert float(code.abs().max().item()) <= 1.0
