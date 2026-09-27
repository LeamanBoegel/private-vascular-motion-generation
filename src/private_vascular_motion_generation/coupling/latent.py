"""The shared coupling latent that carries procedure identity into both branches.

Ref: Sec. 4.4 (z_p is the only channel of procedure identity), Algorithm 1 step 16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

LATENT_DIM: int = 256


@dataclass(frozen=True)
class CouplingLatentSpec:
    dimension: int = LATENT_DIM
    branches: tuple[str, ...] = ("appearance", "motion")

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("latent dimension must be positive")
        if len(self.branches) != 2:
            raise ValueError("the mechanism is paired: exactly two branches")


@dataclass
class CouplingLatent:
    spec: CouplingLatentSpec
    codes: Tensor

    def __post_init__(self) -> None:
        if self.codes.dim() != 2 or self.codes.shape[1] != self.spec.dimension:
            raise ValueError("latent codes must be P x latent_dim")

    def select(self, index: Tensor) -> Tensor:
        return self.codes.index_select(0, index)

    def project(self, appearance_gradient: Tensor, motion_gradient: Tensor, index: Tensor) -> tuple[Tensor, Tensor]:
        """Condition both branches on the same per-procedure code.

        Ref: Sec. 4.4 (no re-encoding of the appearance branch is permitted).
        """
        codes = self.select(index)
        appearance_conditioned = appearance_gradient + codes
        motion_conditioned = motion_gradient + codes
        return appearance_conditioned, motion_conditioned

    def identity_leakage(self, appearance_gradient: Tensor, motion_gradient: Tensor, index: Tensor) -> float:
        """The appearance branch must not carry procedure identity beyond the shared latent.

        Ref: Sec. 4.4.
        """
        codes = self.select(index)
        appearance_residual = appearance_gradient - codes
        motion_residual = motion_gradient - codes
        shared = torch.linalg.norm(codes).clamp_min(torch.finfo(codes.dtype).tiny)
        unexplained = torch.linalg.norm(appearance_residual - motion_residual) + torch.linalg.norm(
            motion_residual - appearance_residual
        )
        return float((unexplained / shared).item())


def draw_latent(spec: CouplingLatentSpec, procedures: int, generator: torch.Generator) -> CouplingLatent:
    if procedures <= 0:
        raise ValueError("at least one procedure code is required")
    codes = torch.randn(procedures, spec.dimension, generator=generator)
    codes = codes / codes.norm(dim=1, keepdim=True).clamp_min(torch.finfo(codes.dtype).tiny)
    return CouplingLatent(spec=spec, codes=codes)


def latent_from_embeddings(embeddings: Tensor, spec: CouplingLatentSpec) -> CouplingLatent:
    if embeddings.shape[1] != spec.dimension:
        raise ValueError("embedding width disagrees with the latent specification")
    normalised = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(torch.finfo(embeddings.dtype).tiny)
    return CouplingLatent(spec=spec, codes=normalised)


def encode_procedure_identity(token_features: Tensor, spec: CouplingLatentSpec, weights: Tensor) -> Tensor:
    """Squeeze a per-procedure token stack into the single identity channel.

    Ref: Sec. 4.4.
    """
    if token_features.dim() != 2:
        raise ValueError("token features must be tokens x width")
    if weights.shape != (token_features.shape[1], spec.dimension):
        raise ValueError("identity projection has the wrong shape")
    pooled = token_features.mean(dim=0)
    return torch.tanh(pooled @ weights)
