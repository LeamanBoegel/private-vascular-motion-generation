"""Coupling-shaped Gaussian geometry and its noise-normalised sensitivity.

Ref: Eq. (1), Eq. (2), Eq. (4).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .projector import CouplingProjector

MIN_NOISE_MULTIPLIER: float = 1e-3


@dataclass(frozen=True)
class ShapedCovariance:
    sigma_parallel: float
    sigma_perpendicular: float
    rank: int
    dimension: int

    def __post_init__(self) -> None:
        if self.sigma_parallel < MIN_NOISE_MULTIPLIER or self.sigma_perpendicular < MIN_NOISE_MULTIPLIER:
            raise ValueError("noise multipliers must be positive")
        if self.rank < 0 or self.rank > self.dimension:
            raise ValueError("rank must lie in [0, d]")

    @property
    def anisotropic(self) -> bool:
        return (
            self.rank > 0
            and not torch.isclose(torch.tensor(self.sigma_parallel), torch.tensor(self.sigma_perpendicular)).item()
        )

    def precision(self, projector: CouplingProjector) -> Tensor:
        """Σ^{-1} = σ∥^{-2} Π_S + σ⊥^{-2} Π_S⊥."""
        _check(projector, self)
        parallel = projector.basis @ projector.basis.transpose(0, 1)
        complement = torch.eye(self.dimension, dtype=parallel.dtype, device=parallel.device) - parallel
        return parallel / self.sigma_parallel**2 + complement / self.sigma_perpendicular**2

    def covariance(self, projector: CouplingProjector) -> Tensor:
        _check(projector, self)
        parallel = projector.basis @ projector.basis.transpose(0, 1)
        complement = torch.eye(self.dimension, dtype=parallel.dtype, device=parallel.device) - parallel
        return parallel * self.sigma_parallel**2 + complement * self.sigma_perpendicular**2

    def inverse_sqrt(self, projector: CouplingProjector) -> Tensor:
        _check(projector, self)
        parallel = projector.basis @ projector.basis.transpose(0, 1)
        complement = torch.eye(self.dimension, dtype=parallel.dtype, device=parallel.device) - parallel
        return parallel / self.sigma_parallel + complement / self.sigma_perpendicular

    def trace(self) -> float:
        return self.rank * self.sigma_parallel**2 + (self.dimension - self.rank) * self.sigma_perpendicular**2

    def to_dict(self) -> dict[str, object]:
        return {
            "sigma_parallel": self.sigma_parallel,
            "sigma_perpendicular": self.sigma_perpendicular,
            "rank": self.rank,
            "dimension": self.dimension,
        }


def _check(projector: CouplingProjector, covariance: ShapedCovariance) -> None:
    if projector.rank != covariance.rank or projector.dimension != covariance.dimension:
        raise ValueError("projector and covariance describe different subspaces")


def covariance_from_projector(
    projector: CouplingProjector,
    sigma_parallel: float,
    sigma_perpendicular: float,
) -> ShapedCovariance:
    return ShapedCovariance(
        sigma_parallel=sigma_parallel,
        sigma_perpendicular=sigma_perpendicular,
        rank=projector.rank,
        dimension=projector.dimension,
    )


def normalised_sensitivity(sigma_parallel: float, sigma_perpendicular: float, rho: float) -> float:
    """u = sqrt(σ∥^{-2} + ρ σ⊥^{-2}).

    Ref: Eq. (4).
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("off-coupling energy ratio must lie in [0, 1]")
    if sigma_parallel <= 0.0 or sigma_perpendicular <= 0.0:
        raise ValueError("noise multipliers must be positive")
    return float((sigma_parallel**-2 + rho * sigma_perpendicular**-2) ** 0.5)


def normalised_sensitivity_isotropic(sigma_perpendicular: float, rho: float) -> float:
    """u_iso = σ⊥^{-1} sqrt(1 + ρ).

    Ref: Eq. (4).
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("off-coupling energy ratio must lie in [0, 1]")
    if sigma_perpendicular <= 0.0:
        raise ValueError("noise multiplier must be positive")
    return float(sigma_perpendicular**-1 * (1.0 + rho) ** 0.5)


def anisotropy_admissible(sigma_parallel: float, sigma_perpendicular: float, grid: int = 201) -> bool:
    """Shaped noise is admissible when u never exceeds u_iso over the concentration range.

    Ref: Sec. 4.6 (admissibility, not merely permissibility).
    """
    if grid < 3:
        raise ValueError("grid must carry at least three rho samples")
    return all(
        normalised_sensitivity(sigma_parallel, sigma_perpendicular, step / (grid - 1))
        <= normalised_sensitivity_isotropic(sigma_perpendicular, step / (grid - 1)) + 1e-12
        for step in range(grid)
    )


def sample_shaped_noise(
    covariance: ShapedCovariance,
    projector: CouplingProjector,
    rows: int,
    columns: int,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Draw Z over a workload with procedures on rows and window micro-steps on columns.

    Ref: Eq. (1), Eq. (2).
    """
    _check(projector, covariance)
    if rows <= 0 or columns <= 0:
        raise ValueError("workload must have positive extent")
    standard = torch.randn(rows, projector.rank, columns, generator=generator, dtype=dtype)
    parallel = torch.einsum("dr,prc->pdc", projector.basis, standard) * covariance.sigma_parallel
    leftover = (
        torch.randn(rows, covariance.dimension, columns, generator=generator, dtype=dtype)
        * covariance.sigma_perpendicular
    )
    coefficients = torch.einsum("dr,pdc->prc", projector.basis, leftover)
    residual = leftover - torch.einsum("dr,prc->pdc", projector.basis, coefficients)
    return (parallel + residual).contiguous()
