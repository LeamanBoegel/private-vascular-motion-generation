"""Noise-geometry tests: projectors, shaped covariance, sampling and concentration.

Ref: Eq. (1), Eq. (2), Eq. (4), Sec. 4.4, Sec. 4.6.
"""

from __future__ import annotations

import pytest
import torch

from private_vascular_motion_generation.geometry.concentration import (
    concentration_from_gradients,
    energy_ratio_ci,
    gain_from_concentration,
    on_coupling_energy,
    per_procedure_rho,
    residual_energy_trace,
)
from private_vascular_motion_generation.geometry.projector import (
    CouplingProjector,
    complement_projector,
    projector_from_directions,
    projector_residual,
)
from private_vascular_motion_generation.geometry.shaped_covariance import (
    ShapedCovariance,
    anisotropy_admissible,
    covariance_from_projector,
    normalised_sensitivity,
    normalised_sensitivity_isotropic,
    sample_shaped_noise,
)

DIMENSION = 12
RANK = 3


@pytest.fixture()
def projector() -> CouplingProjector:
    generator = torch.Generator().manual_seed(8)
    basis = torch.linalg.qr(torch.randn(DIMENSION, RANK, generator=generator))[0]
    return projector_from_directions(basis, RANK)


def test_projector_is_orthonormal_and_idempotent(projector: CouplingProjector) -> None:
    assert projector_residual(projector) < 1e-6
    generator = torch.Generator().manual_seed(9)
    probe = torch.randn(DIMENSION, generator=generator)
    once = projector.apply(probe)
    assert torch.allclose(projector.apply(once), once, atol=1e-6)
    assert torch.allclose(projector.apply(probe) + complement_projector(projector, probe), probe, atol=1e-6)


def test_projector_rejects_a_rank_beyond_the_directions() -> None:
    generator = torch.Generator().manual_seed(10)
    with pytest.raises(ValueError):
        projector_from_directions(torch.randn(6, 2, generator=generator), 4)


def test_precision_places_the_two_multipliers(projector: CouplingProjector) -> None:
    covariance = ShapedCovariance(sigma_parallel=5.4, sigma_perpendicular=2.6, rank=RANK, dimension=DIMENSION)
    precision = covariance.precision(projector)
    on = precision @ projector.basis[:, 0]
    assert torch.allclose(on, projector.basis[:, 0] / 5.4**2, atol=1e-10)
    residual = projector.complement()[:, 0]
    assert torch.allclose(precision @ residual, residual / 2.6**2, atol=1e-10)


def test_covariance_is_the_inverse_of_the_precision(projector: CouplingProjector) -> None:
    covariance = covariance_from_projector(projector, 5.4, 2.6)
    product = covariance.covariance(projector) @ covariance.precision(projector)
    assert torch.allclose(product, torch.eye(DIMENSION), atol=1e-6)


def test_covariance_trace_matches_the_rank_weighting(projector: CouplingProjector) -> None:
    covariance = covariance_from_projector(projector, 5.4, 2.6)
    expected = RANK * 5.4**2 + (DIMENSION - RANK) * 2.6**2
    assert covariance.trace() == pytest.approx(expected, rel=1e-9)


def test_sampler_reproduces_the_analytic_covariance(projector: CouplingProjector) -> None:
    covariance = covariance_from_projector(projector, 5.4, 2.6)
    samples = sample_shaped_noise(
        covariance, projector, rows=3000, columns=4, generator=torch.Generator().manual_seed(11)
    )
    flat = samples.permute(0, 2, 1).reshape(-1, DIMENSION)
    empirical = flat.transpose(0, 1) @ flat / flat.shape[0]
    analytic = covariance.covariance(projector)
    assert float((empirical - analytic).abs().max().item()) < 2.0
    assert float(empirical.trace().item()) == pytest.approx(float(analytic.trace().item()), rel=0.02)


def test_equal_multipliers_make_the_precision_a_scaled_identity(projector: CouplingProjector) -> None:
    """The conditioning-off arm equalises the multipliers while keeping the frozen subspace.

    Ref: Sec. 2.3 (the shaped covariance is substituted with an isotropic one; the off-cells
    still report an off-coupling energy ratio, so the subspace itself survives).
    """
    isotropic = ShapedCovariance(
        sigma_parallel=2.6, sigma_perpendicular=2.6, rank=projector.rank, dimension=projector.dimension
    )
    assert not isotropic.anisotropic
    precision = isotropic.precision(projector)
    assert torch.allclose(precision, precision[0, 0] * torch.eye(DIMENSION), atol=1e-8)
    assert precision[0, 0] == pytest.approx(2.6**-2)


def test_admissibility_holds_for_the_reported_multipliers() -> None:
    assert anisotropy_admissible(5.4, 2.6)
    assert normalised_sensitivity(5.4, 2.6, 0.19) < normalised_sensitivity_isotropic(2.6, 0.19)
    assert normalised_sensitivity(2.6, 2.6, 0.4) == pytest.approx(normalised_sensitivity_isotropic(2.6, 0.4))


def test_shaped_sensitivity_beats_the_isotropic_joint_arm() -> None:
    shaped = normalised_sensitivity(5.4, 2.6, 0.19)
    isotropic = normalised_sensitivity(3.9, 3.9, 0.41)
    assert shaped < isotropic


def test_concentration_partitions_the_gradient_energy(projector: CouplingProjector) -> None:
    gradients = torch.randn(32, DIMENSION, generator=torch.Generator().manual_seed(12))
    rho = concentration_from_gradients(gradients, projector)
    report = residual_energy_trace(gradients, projector)
    assert 0.0 <= rho <= 1.0
    assert rho == pytest.approx(report["rho"], abs=1e-6)
    assert float(on_coupling_energy(gradients, projector).sum().item()) == pytest.approx(
        report["on_coupling"], rel=1e-6
    )
    assert report["on_coupling"] + report["off_coupling"] == pytest.approx(report["total"], rel=1e-6)


def test_per_procedure_ratio_is_bounded(projector: CouplingProjector) -> None:
    gradients = torch.randn(16, DIMENSION, generator=torch.Generator().manual_seed(13))
    ratios = per_procedure_rho(gradients, projector)
    assert bool(torch.all(ratios >= 0.0).item())
    assert bool(torch.all(ratios <= 1.0 + 1e-9).item())


def test_gain_is_bounded_by_the_isotropic_reference(projector: CouplingProjector) -> None:
    gain = gain_from_concentration(0.19, 5.4, 2.6)
    assert 0.0 < gain < 1.0
    assert gain_from_concentration(0.0, 5.4, 2.6) > gain


def test_energy_ratio_interval_brackets_the_mean() -> None:
    samples = torch.tensor([0.17, 0.19, 0.21, 0.20])
    mean, low, high = energy_ratio_ci(samples)
    assert low < mean < high


def test_concentration_rejects_a_degenerate_matrix(projector: CouplingProjector) -> None:
    with pytest.raises(ValueError):
        concentration_from_gradients(torch.zeros(4, DIMENSION), projector)
