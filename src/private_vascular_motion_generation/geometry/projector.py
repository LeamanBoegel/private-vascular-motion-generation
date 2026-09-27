"""Orthogonal projectors onto the coupling subspace and its complementary residual.

Ref: Eq. (1), Sec. 4.4 (tensorised with the coupling projection operator).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CouplingProjector:
    basis: Tensor
    rank: int
    dimension: int

    def __post_init__(self) -> None:
        if self.basis.dim() != 2:
            raise ValueError("coupling basis must be d x r")
        rows, columns = self.basis.shape
        if columns != self.rank or rows != self.dimension:
            raise ValueError("coupling basis shape disagrees with the declared rank and dimension")
        if self.rank > self.dimension:
            raise ValueError("rank cannot exceed the noise dimension")

    def apply(self, vector: Tensor) -> Tensor:
        """Π_S g."""
        return self.basis @ (self.basis.transpose(0, 1) @ vector)

    def apply_batch(self, matrix: Tensor) -> Tensor:
        """Π_S applied row-wise to a batch of d-dimensional vectors."""
        if matrix.dim() != 2 or matrix.shape[1] != self.dimension:
            raise ValueError("batch must be P x d")
        return matrix @ self.basis @ self.basis.transpose(0, 1)

    def complement(self) -> Tensor:
        """Π_S⊥ g = g − Π_S g, without materialising the complementary basis."""
        identity = torch.eye(self.dimension, dtype=self.basis.dtype, device=self.basis.device)
        return identity - self.basis @ self.basis.transpose(0, 1)

    def subspace_power(self, vector: Tensor) -> float:
        projected = self.apply(vector)
        return float(torch.dot(projected, projected).item())

    def residual_power(self, vector: Tensor) -> float:
        residual = vector - self.apply(vector)
        return float(torch.dot(residual, residual).item())

    def shift(self, offset: Tensor) -> CouplingProjector:
        """Rotate the basis inside the same subspace while preserving orthonormality.

        Ref: Sec. 4.10 (dependence on the structure of the protected boundary).
        """
        if offset.shape != (self.rank, self.rank):
            raise ValueError("offset must be r x r")
        rotated = self.basis @ torch.linalg.qr(offset)[0]
        return CouplingProjector(basis=rotated, rank=self.rank, dimension=self.dimension)


def projector_from_directions(directions: Tensor, rank: int, tol: float = 1e-8) -> CouplingProjector:
    """Orthonormalise the leading canonical directions into the coupling basis U.

    Ref: Algorithm 2 step 8.
    """
    if directions.dim() != 2:
        raise ValueError("directions must be d x m")
    dimension, available = directions.shape
    if rank < 1 or rank > min(available, dimension):
        raise ValueError("rank outside the available canonical directions")
    orthonormal, triangular = torch.linalg.qr(directions[:, :rank])
    signs = torch.sign(torch.diagonal(triangular))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    orthonormal = orthonormal * signs.unsqueeze(0)
    residual = float(torch.linalg.norm(orthonormal.transpose(0, 1) @ orthonormal - torch.eye(rank)).item())
    if residual > 1e-4:
        raise ValueError("canonical directions are numerically dependent")
    del tol
    return CouplingProjector(basis=orthonormal.contiguous(), rank=rank, dimension=dimension)


def complement_projector(projector: CouplingProjector, vector: Tensor) -> Tensor:
    return vector - projector.apply(vector)


def projector_residual(projector: CouplingProjector) -> float:
    gram = projector.basis.transpose(0, 1) @ projector.basis
    identity = torch.eye(projector.rank, dtype=gram.dtype, device=gram.device)
    return float(torch.linalg.norm(gram - identity).item())
