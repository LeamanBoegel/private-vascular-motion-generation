"""Measured off-coupling energy ratio of the per-procedure gradient.

Ref: Sec. 4.4, Sec. 4.10 (rho governs the gain, the structural condition governs the bound).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .projector import CouplingProjector

RHO_ISOTROPIC_REFERENCE: float = 0.41
RHO_SHAPED: float = 0.19
RHO_CONDITIONING_OFF: float = 0.71


def on_coupling_energy(gradients: Tensor, projector: CouplingProjector) -> Tensor:
    """Per-procedure energy inside the frozen coupling subspace."""
    if gradients.dim() != 2 or gradients.shape[1] != projector.dimension:
        raise ValueError("gradient matrix must be P x d")
    projected = gradients @ projector.basis @ projector.basis.transpose(0, 1)
    return (projected**2).sum(dim=1)


def off_coupling_energy(gradients: Tensor, projector: CouplingProjector) -> Tensor:
    """Per-procedure energy in the complementary residual."""
    if gradients.dim() != 2 or gradients.shape[1] != projector.dimension:
        raise ValueError("gradient matrix must be P x d")
    residual = gradients - gradients @ projector.basis @ projector.basis.transpose(0, 1)
    return (residual**2).sum(dim=1)


def concentration_from_gradients(gradients: Tensor, projector: CouplingProjector) -> float:
    """rho = E[‖Π_S⊥ g‖²] / E[‖g‖²], evaluated at the frozen subspace.

    Ref: Sec. 4.4, Table 3 caption.
    """
    if gradients.dim() != 2:
        raise ValueError("gradient matrix must be two-dimensional")
    total = float((gradients**2).sum().item())
    if total <= 0.0:
        raise ValueError("degenerate gradient matrix carries no energy")
    residual = float(off_coupling_energy(gradients, projector).sum().item())
    return residual / total


def per_procedure_rho(gradients: Tensor, projector: CouplingProjector, floor: float = 1e-12) -> Tensor:
    numerator = off_coupling_energy(gradients, projector)
    denominator = (gradients**2).sum(dim=1).clamp_min(floor)
    return numerator / denominator


def residual_energy_trace(gradients: Tensor, projector: CouplingProjector) -> dict[str, float]:
    on = float(on_coupling_energy(gradients, projector).sum().item())
    off = float(off_coupling_energy(gradients, projector).sum().item())
    total = on + off
    if total <= 0.0:
        raise ValueError("degenerate gradient matrix carries no energy")
    return {"on_coupling": on, "off_coupling": off, "total": total, "rho": off / total}


def concentration_regime(rho: float, shaped: float = RHO_SHAPED, isotropic: float = RHO_ISOTROPIC_REFERENCE) -> str:
    """Classify the run against the reported concentration levels of the protocol."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if rho <= shaped + 0.05:
        return "shaped"
    if rho < isotropic - 0.05:
        return "intermediate"
    return "isotropic"


def gain_from_concentration(rho: float, sigma_parallel: float, sigma_perpendicular: float) -> float:
    """Relative reduction of noise-normalised sensitivity against the isotropic arm.

    Ref: Eq. (4).
    """
    from .shaped_covariance import normalised_sensitivity, normalised_sensitivity_isotropic

    numerator = normalised_sensitivity(sigma_parallel, sigma_perpendicular, rho)
    denominator = normalised_sensitivity_isotropic(sigma_perpendicular, rho)
    return 1.0 - numerator / denominator


def energy_ratio_ci(samples: Tensor, level: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap-free normal interval for the per-run ratio; reported per cell, never averaged.

    Ref: Table 3 caption ("a per-run quantity, measured per cell, never averaged across arms").
    """
    if samples.dim() != 1 or samples.numel() < 2:
        raise ValueError("at least two per-run ratios are required")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    mean = float(samples.mean().item())
    standard_error = float(samples.std(unbiased=True).item()) / samples.numel() ** 0.5
    z = torch.special.ndtri(torch.tensor(0.5 + level / 2.0)).item()
    return mean, mean - z * standard_error, mean + z * standard_error
