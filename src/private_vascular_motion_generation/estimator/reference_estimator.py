"""Offline shaping-subspace estimation from the non-private reference corpora.

Ref: Algorithm 2, Sec. 4.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..branches.pairing import BranchPair, PairingSpec, ProcedureBatch
from ..coupling.cca import CovarianceTriple, paired_covariances, solve_canonical
from ..coupling.head import DEFAULT_COUPLING_RANK, CouplingHead, CouplingHeadSpec
from ..geometry.projector import CouplingProjector, projector_from_directions

REFERENCE_BATCH_SIZE: int = 8


@dataclass(frozen=True)
class EstimatorSpec:
    reference_batch_size: int = REFERENCE_BATCH_SIZE
    rank: int = DEFAULT_COUPLING_RANK
    max_minibatches: int | None = None

    def __post_init__(self) -> None:
        if self.reference_batch_size < 1:
            raise ValueError("reference batch size must be positive")
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if self.max_minibatches is not None and self.max_minibatches < 1:
            raise ValueError("minibatch budget must be positive when set")


@dataclass
class EstimatorResult:
    projector: CouplingProjector
    triple: CovarianceTriple
    canonical: object
    minibatches: int
    diagnostics: dict[str, float] = field(default_factory=dict)

    def rank(self) -> int:
        return self.projector.rank

    def subspace_dimension(self) -> int:
        return self.projector.dimension


class ReferenceShapingEstimator:
    """One offline pass over the reference corpora; the private cohort is never touched.

    Ref: Algorithm 2, Sec. 4.4 (estimator-disjointness).
    """

    def __init__(self, spec: EstimatorSpec, pairing: PairingSpec) -> None:
        self.spec = spec
        self.pairing = pairing
        self.fitted = False

    def _choose_rank(self, available: int) -> int:
        rank = min(self.spec.rank, available)
        if rank < 1:
            raise ValueError("no canonical direction is available for the shaped subspace")
        return rank

    def fit(self, pair: BranchPair, batches: list[ProcedureBatch]) -> EstimatorResult:
        """Accumulate Sigma11, Sigma22, Sigma12 over the reference minibatches.

        Ref: Algorithm 2 steps 1-5.
        """
        if not batches:
            raise ValueError("the estimator needs at least one reference minibatch")
        appearance_blocks: list[Tensor] = []
        motion_blocks: list[Tensor] = []
        budget = self.spec.max_minibatches
        for index, batch in enumerate(batches):
            if budget is not None and index >= budget:
                break
            if batch.size() > self.spec.reference_batch_size:
                raise ValueError("reference minibatch exceeds the declared reference batch size")
            first, second = pair.branch_gradient_views(batch)
            appearance_blocks.append(first)
            motion_blocks.append(second)
        appearance = torch.cat(appearance_blocks, dim=0)
        motion = torch.cat(motion_blocks, dim=0)
        triple = paired_covariances(appearance, motion)
        canonical = solve_canonical(triple)
        rank = self._choose_rank(int(canonical.correlations.numel()))
        left, _right = canonical.leading(rank)
        projector = projector_from_directions(left, rank)
        self.fitted = True
        return EstimatorResult(
            projector=projector,
            triple=triple,
            canonical=canonical,
            minibatches=len(appearance_blocks),
            diagnostics={
                "leading_correlation": float(canonical.correlations[0].item()),
                "spectral_density": canonical.spectral_density(rank),
                "reference_procedures": float(appearance.shape[0]),
                "appearance_width": float(appearance.shape[1]),
                "motion_width": float(motion.shape[1]),
            },
        )

    def freeze(self, result: EstimatorResult) -> CouplingProjector:
        """Freeze U before the private fine-tune begins.

        Ref: Algorithm 2 step 8, Sec. 4.4.
        """
        if not self.fitted:
            raise RuntimeError("the estimator must be fitted before its subspace is frozen")
        return result.projector


def cost_estimate(reference_batch_size: int, dimension: int, minibatches: int) -> dict[str, int]:
    """O(B_ref d^2) per minibatch, O(K(d^2 + dk)) for the private pass.

    Ref: Sec. 4.7, Sec. 4.4.
    """
    if min(reference_batch_size, dimension, minibatches) < 1:
        raise ValueError("cost terms must be positive")
    per_batch = reference_batch_size * dimension**2
    return {
        "per_minibatch": per_batch,
        "total_estimator": per_batch * minibatches,
        "attention_multiplier_cost": dimension**2,
    }


def subspace_from_private_records(pair: BranchPair, batches: list[ProcedureBatch], rank: int) -> CouplingProjector:
    """The invalid variant that selects the subspace on private records.

    Ref: Table 2 row 3, Table 4 row 14. Reported with its selection uncounted.
    """
    head = CouplingHead(spec=CouplingHeadSpec(rank=rank))
    appearance: list[Tensor] = []
    motion: list[Tensor] = []
    for batch in batches:
        first, second = pair.branch_gradient_views(batch)
        appearance.append(first)
        motion.append(second)
    head.fit(torch.cat(appearance, dim=0), torch.cat(motion, dim=0))
    return head.require_projector()
