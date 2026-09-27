"""Canonical-correlation coupling between the appearance and motion branch gradients.

Ref: Sec. 4.4 (head solves Σ12 Σ22^{-1} Σ21 u = λ² Σ11 u), Algorithm 2 step 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

RIDGE: float = 1e-6


@dataclass(frozen=True)
class CovarianceTriple:
    sigma_11: Tensor
    sigma_22: Tensor
    sigma_12: Tensor

    def __post_init__(self) -> None:
        first, second = self.sigma_11.shape
        if self.sigma_22.shape != (second, second):
            raise ValueError("within-branch second moments are inconsistent")
        if self.sigma_12.shape != (first, second):
            raise ValueError("cross second moment is inconsistent with the marginals")


@dataclass(frozen=True)
class CanonicalSolution:
    correlations: Tensor
    left: Tensor
    right: Tensor

    def leading(self, rank: int) -> tuple[Tensor, Tensor]:
        if rank < 1 or rank > int(self.correlations.numel()):
            raise ValueError("rank outside the available canonical directions")
        return self.left[:, :rank], self.right[:, :rank]

    def spectral_density(self, rank: int) -> float:
        if rank < 1:
            raise ValueError("rank must be positive")
        leading = self.correlations[:rank]
        return float((leading**2).mean().item())


def paired_covariances(appearance_gradients: Tensor, motion_gradients: Tensor) -> CovarianceTriple:
    """Accumulate Σ11, Σ22, Σ12 from the paired per-procedure gradients.

    Ref: Algorithm 2 steps 3-4.
    """
    if appearance_gradients.dim() != 2 or motion_gradients.dim() != 2:
        raise ValueError("gradient pairs must be two-dimensional")
    if appearance_gradients.shape[0] != motion_gradients.shape[0]:
        raise ValueError("paired gradients must share the procedure axis")
    first = appearance_gradients - appearance_gradients.mean(dim=0, keepdim=True)
    second = motion_gradients - motion_gradients.mean(dim=0, keepdim=True)
    count = max(1, int(first.shape[0]) - 1)
    sigma_11 = first.transpose(0, 1) @ first / count
    sigma_22 = second.transpose(0, 1) @ second / count
    sigma_12 = first.transpose(0, 1) @ second / count
    return CovarianceTriple(sigma_11=sigma_11, sigma_22=sigma_22, sigma_12=sigma_12)


def inverse_sqrt(matrix: Tensor, ridge: float = RIDGE) -> Tensor:
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("expected a square matrix")
    values, vectors = torch.linalg.eigh(matrix + ridge * torch.eye(matrix.shape[0], dtype=matrix.dtype))
    values = values.clamp_min(ridge)
    root: Tensor = (vectors * values.rsqrt().unsqueeze(0)) @ vectors.transpose(0, 1)
    return root


def solve_canonical(triple: CovarianceTriple, redundancy_rank: int | None = None) -> CanonicalSolution:
    """Solve the generalised eigenvalue problem through the whitened SVD of Σ11^{-1/2} Σ12 Σ22^{-1/2}.

    Ref: Sec. 4.4, Algorithm 2 steps 6-7.
    """
    first_whitener = inverse_sqrt(triple.sigma_11)
    second_whitener = inverse_sqrt(triple.sigma_22)
    reduced = first_whitener @ triple.sigma_12 @ second_whitener
    left_vectors, singular_values, right_transposed = torch.linalg.svd(reduced, full_matrices=False)
    left = first_whitener @ left_vectors
    right = second_whitener @ right_transposed.transpose(0, 1)
    correlations = singular_values.clamp(0.0, 1.0)
    if redundancy_rank is not None:
        if redundancy_rank < 1:
            raise ValueError("redundancy rank must be positive")
        correlations = correlations.clone()
        correlations[redundancy_rank:] = 0.0
    return CanonicalSolution(correlations=correlations, left=left, right=right)


def coupling_loss(solution: CanonicalSolution, rank: int) -> Tensor:
    """L_cc = -r^{-1} Σ_{j<=r} λ_j², maximised by the private optimiser.

    Ref: Algorithm 1 step 7.
    """
    if rank < 1:
        raise ValueError("rank must be positive")
    if int(solution.correlations.numel()) < rank:
        raise ValueError("fewer canonical directions than the declared rank")
    leading = solution.correlations[:rank]
    return -(leading**2).sum() / rank


def generalised_eigen_residual(triple: CovarianceTriple, solution: CanonicalSolution, index: int) -> float:
    """Residual of Σ12 Σ22^{-1} Σ21 u = λ² Σ11 u for one canonical pair."""
    if index < 0 or index >= int(solution.correlations.numel()):
        raise IndexError("canonical index out of range")
    left = solution.left[:, index]
    lam = solution.correlations[index]
    sigma_22_inverse = torch.linalg.pinv(triple.sigma_22)
    left_side = triple.sigma_12 @ sigma_22_inverse @ triple.sigma_12.transpose(0, 1) @ left
    right_side = (lam**2) * (triple.sigma_11 @ left)
    denominator = torch.linalg.norm(right_side).clamp_min(torch.finfo(right_side.dtype).tiny)
    return float((torch.linalg.norm(left_side - right_side) / denominator).item())


def canonical_orthonormality(solution: CanonicalSolution, triple: CovarianceTriple) -> float:
    """Canonical directions are orthonormal in the Σ-metric, not in the Euclidean one."""
    left_gram = solution.left.transpose(0, 1) @ triple.sigma_11 @ solution.left
    right_gram = solution.right.transpose(0, 1) @ triple.sigma_22 @ solution.right
    identity = torch.eye(left_gram.shape[0], dtype=left_gram.dtype)
    return float(
        max(
            torch.linalg.norm(left_gram - identity).item(),
            torch.linalg.norm(right_gram - identity).item(),
        )
    )
