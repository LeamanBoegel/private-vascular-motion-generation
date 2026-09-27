"""Pairing of the two branches and formation of the per-procedure gradient workload.

Ref: Algorithm 1 steps 5-10, Algorithm 2 steps 3-4, Eq. (2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..cohort.procedure import MOTION_DIM
from ..coupling.head import CouplingHead
from ..coupling.latent import CouplingLatentSpec
from .adapter import PrivateCouplingAdapter
from .appearance import AppearanceBranch
from .motion import FlowMatchingActionHead


@dataclass(frozen=True)
class PairingSpec:
    shared_latent: bool = True
    coupling_head: bool = True
    window_micro_steps: int = 1

    def __post_init__(self) -> None:
        if self.window_micro_steps < 1:
            raise ValueError("at least one window micro-step is required")

    def decoupled(self) -> bool:
        return not self.shared_latent


@dataclass(frozen=True)
class ProcedureBatch:
    frames: Tensor
    mask: Tensor
    motion: Tensor
    endpoints: Tensor
    latent_codes: Tensor

    def __post_init__(self) -> None:
        if self.frames.dim() != 4:
            raise ValueError("frames must be batch x frames x height x width")
        if self.mask.dim() != 4:
            raise ValueError("mask must be batch x 1 x height x width")
        if self.motion.dim() != 3 or self.motion.shape[1] != MOTION_DIM:
            raise ValueError("motion must be batch x 6 x frames")
        batch = self.frames.shape[0]
        if self.mask.shape[0] != batch or self.motion.shape[0] != batch:
            raise ValueError("batch axis must agree across the paired record")
        if self.endpoints.shape != (batch, 2):
            raise ValueError("endpoints must be batch x 2")
        if self.latent_codes.dim() != 2 or self.latent_codes.shape[0] != batch:
            raise ValueError("latent codes must carry one row per procedure")

    def size(self) -> int:
        return int(self.frames.shape[0])


class BranchPair(nn.Module):
    """Appearance and motion branches behind one shared privatised attention block."""

    def __init__(
        self,
        appearance: AppearanceBranch,
        motion: FlowMatchingActionHead,
        adapter: PrivateCouplingAdapter,
        latent_spec: CouplingLatentSpec,
        pairing: PairingSpec,
    ) -> None:
        super().__init__()
        self.appearance = appearance
        self.motion = motion
        self.adapter = adapter
        self.latent_spec = latent_spec
        self.pairing = pairing

    def _codes(self, batch: ProcedureBatch) -> Tensor:
        if self.pairing.shared_latent:
            return batch.latent_codes
        return torch.zeros_like(batch.latent_codes)

    def appearance_tokens(self, batch: ProcedureBatch) -> tuple[Tensor, Tensor]:
        """Appearance logits and the modulated token readout of every procedure."""
        codes = self._codes(batch)
        tokens = self.appearance.encode(batch.frames, codes)
        modulated = self.adapter.modulate(tokens)
        extent = (int(batch.frames.shape[2]), int(batch.frames.shape[3]))
        logits = self.appearance.segmentation_head(modulated, extent)
        return logits, modulated.mean(dim=1)

    def appearance_terms(self, logits: Tensor, mask: Tensor) -> Tensor:
        """Per-procedure segmentation residual: cross-entropy plus soft Dice."""
        terms = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask, reduction="none").mean(dim=(1, 2, 3))
        probability = torch.sigmoid(logits).reshape(logits.shape[0], -1)
        flat_target = mask.reshape(mask.shape[0], -1)
        intersection = (probability * flat_target).sum(dim=1)
        denominator = probability.sum(dim=1) + flat_target.sum(dim=1)
        return terms + 1.0 - 2.0 * intersection / denominator.clamp_min(torch.finfo(probability.dtype).tiny)

    def motion_tokens(self, state: Tensor, codes: Tensor) -> Tensor:
        """Motion tokens after the shared privatised attention block."""
        return self.adapter.modulate(self.motion.encode(state, codes))

    def motion_velocity(self, state: Tensor, time: Tensor, codes: Tensor) -> Tensor:
        return self.motion.head_from_tokens(self.motion_tokens(state, codes), time)

    def motion_readout(self, batch: ProcedureBatch) -> Tensor:
        codes = self._codes(batch)
        return self.motion_tokens(batch.motion, codes).mean(dim=1)

    def per_procedure_flow_residual(
        self, target: Tensor, codes: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        """Flow-matching residual per procedure, routed through the privatised block.

        Ref: Sec. 4.4.
        """
        if target.dim() != 3 or target.shape[1] != MOTION_DIM:
            raise ValueError("target motion must be batch x 6 x frames")
        batch = target.shape[0]
        noise = torch.randn(target.shape, generator=generator, dtype=target.dtype)
        time = torch.rand(batch, generator=generator, dtype=target.dtype)
        interpolated = time.view(batch, 1, 1) * target + (1.0 - time.view(batch, 1, 1)) * noise
        predicted = self.motion_velocity(interpolated, time, codes)
        return ((predicted - (target - noise)) ** 2).mean(dim=(1, 2))

    def per_procedure_residuals(
        self, batch: ProcedureBatch, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Separate appearance and motion residuals, one entry per procedure."""
        logits, _readout = self.appearance_tokens(batch)
        appearance_terms = self.appearance_terms(logits, batch.mask)
        motion_terms = self.per_procedure_flow_residual(batch.motion, self._codes(batch), generator)
        return appearance_terms, motion_terms

    def concatenated_residual(self, batch: ProcedureBatch, generator: torch.Generator | None = None) -> Tensor:
        appearance_terms, motion_terms = self.per_procedure_residuals(batch, generator)
        return appearance_terms + motion_terms

    def coupling_term(self, head: CouplingHead, batch: ProcedureBatch) -> Tensor:
        """L_cc enters the objective the private optimiser follows.

        Ref: Algorithm 1 step 7.
        """
        if not self.pairing.coupling_head:
            return torch.zeros((), dtype=torch.float32)
        appearance, motion = self.branch_gradient_views(batch)
        head.fit(appearance, motion)
        return head.loss()

    def branch_gradient_views(self, batch: ProcedureBatch) -> tuple[Tensor, Tensor]:
        """The paired per-procedure gradient views used by the coupling estimator.

        Both views live in the privatised attention space, so the canonical directions of
        the coupling head index the same d coordinates the shaped noise occupies.

        Ref: Algorithm 2 steps 3-4, Eq. (2).
        """
        logits, _readout = self.appearance_tokens(batch)
        appearance_terms = self.appearance_terms(logits, batch.mask)
        motion_terms = self.per_procedure_flow_residual(batch.motion, self._codes(batch))
        appearance_rows: list[Tensor] = []
        motion_rows: list[Tensor] = []
        for index in range(batch.size()):
            appearance_rows.append(
                torch.autograd.grad(appearance_terms[index], self.adapter.theta, retain_graph=True, allow_unused=False)[
                    0
                ].mean(dim=1)
            )
            motion_rows.append(
                torch.autograd.grad(motion_terms[index], self.adapter.theta, retain_graph=True, allow_unused=False)[
                    0
                ].mean(dim=1)
            )
        return torch.stack(appearance_rows, dim=0), torch.stack(motion_rows, dim=0)

    def per_procedure_workload(self, batch: ProcedureBatch, generator: torch.Generator | None = None) -> Tensor:
        """G_p for every procedure, shaped procedures x d x window micro-steps.

        Ref: Algorithm 1 step 6, Eq. (2).
        """
        terms = self.concatenated_residual(batch, generator)
        rows: list[Tensor] = []
        for index in range(batch.size()):
            rows.append(torch.autograd.grad(terms[index], self.adapter.theta, retain_graph=True, allow_unused=False)[0])
        return torch.stack(rows, dim=0)

    def aggregate_workload(self, batch: ProcedureBatch, generator: torch.Generator | None = None) -> Tensor:
        """AG: procedures on rows collapse into one window-by-dimension workload.

        Ref: Algorithm 1 step 10.
        """
        return self.per_procedure_workload(batch, generator).sum(dim=0)


def concatenated_residual(appearance_residual: Tensor, motion_residual: Tensor) -> Tensor:
    if appearance_residual.shape != motion_residual.shape:
        raise ValueError("branch residuals must share a shape to be concatenated")
    return appearance_residual + motion_residual
