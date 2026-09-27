"""The coupling head: canonical-correlation spectral density of the paired branch gradients.

Ref: Sec. 4.4, Algorithm 1 step 7, Table 4 rows 10-11.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..geometry.projector import CouplingProjector, projector_from_directions
from .cca import (
    CanonicalSolution,
    CovarianceTriple,
    coupling_loss,
    paired_covariances,
    solve_canonical,
)

DEFAULT_COUPLING_RANK: int = 12


@dataclass(frozen=True)
class CouplingHeadSpec:
    rank: int = DEFAULT_COUPLING_RANK
    reduce_shared_latent: bool = False
    reduce_head: bool = False

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("coupling rank must be positive")

    def active(self) -> bool:
        return not self.reduce_head


@dataclass
class CouplingHead:
    spec: CouplingHeadSpec
    solution: CanonicalSolution | None = None
    projector: CouplingProjector | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    def fit(self, appearance_gradients: Tensor, motion_gradients: Tensor) -> CouplingHead:
        triple: CovarianceTriple = paired_covariances(appearance_gradients, motion_gradients)
        solution = solve_canonical(triple)
        available = int(solution.correlations.numel())
        rank = min(self.spec.rank, available)
        left, _right = solution.leading(rank)
        self.solution = solution
        self.projector = projector_from_directions(left, rank)
        self.diagnostics = {
            "leading_correlation": float(solution.correlations[0].item()),
            "spectral_density": solution.spectral_density(rank),
            "available_directions": float(available),
        }
        return self

    def loss(self) -> Tensor:
        if self.solution is None:
            raise RuntimeError("coupling head must be fitted before its loss is evaluated")
        if not self.spec.active():
            return torch.zeros((), dtype=self.solution.correlations.dtype)
        return coupling_loss(self.solution, self.spec.rank)

    def spectral_density(self) -> float:
        if self.solution is None:
            raise RuntimeError("coupling head must be fitted before its spectrum is read")
        return self.solution.spectral_density(self.spec.rank)

    def require_projector(self) -> CouplingProjector:
        if self.projector is None:
            raise RuntimeError("coupling head must be fitted before its subspace is read")
        return self.projector

    def loss_drop_from_head_removal(self) -> float:
        """Removing the head only is the smaller of the two Tier-1 removals.

        Ref: Table 4 rows 10-11.
        """
        if self.solution is None:
            raise RuntimeError("coupling head must be fitted first")
        return self.solution.spectral_density(min(self.spec.rank, int(self.solution.correlations.numel())))


def head_removal_contrast(
    full: CouplingHead, latent_removed: CouplingHead, head_removed: CouplingHead
) -> dict[str, float]:
    """The latent and the head are separately attributable; the latent contributes more.

    Ref: Sec. 2.5, Table 4 rows 10-11.
    """
    return {
        "full_density": full.spectral_density(),
        "latent_removed_density": latent_removed.spectral_density(),
        "head_removed_density": head_removed.spectral_density(),
    }


def canonical_spectrum_report(solution: CanonicalSolution, rank: int) -> dict[str, object]:
    leading = solution.correlations[:rank]
    return {
        "rank": rank,
        "leading": [float(value) for value in leading.tolist()],
        "mean_squared": float((leading**2).mean().item()),
        "tail_energy": float((solution.correlations[rank:] ** 2).sum().item()),
    }
