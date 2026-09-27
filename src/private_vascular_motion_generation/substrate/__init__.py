"""Assembly and release of the private substrate.

Ref: Eq. (2), Algorithm 1 steps 10-16, Sec. 4.6 (release is pure post-processing).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..accounting.renyi import PrivacyBudget, prefix_amplification_term
from ..geometry.projector import CouplingProjector
from ..geometry.shaped_covariance import ShapedCovariance, sample_shaped_noise


@dataclass(frozen=True)
class SubstrateSpec:
    dimension: int
    window_length: int
    sigma_parallel: float
    sigma_perpendicular: float
    rank: int
    clipping_norm: float
    sampling_rate: float
    rounds: int
    delta: float

    def __post_init__(self) -> None:
        if min(self.dimension, self.window_length, self.rounds) < 1:
            raise ValueError("substrate extents must be positive")
        if not 0.0 < self.sampling_rate <= 1.0:
            raise ValueError("sampling rate must lie in (0, 1]")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")

    def steps_times_sampling(self) -> float:
        return float(self.rounds) * self.sampling_rate

    def covariance(self) -> ShapedCovariance:
        return ShapedCovariance(
            sigma_parallel=self.sigma_parallel,
            sigma_perpendicular=self.sigma_perpendicular,
            rank=self.rank,
            dimension=self.dimension,
        )


@dataclass
class Substrate:
    spec: SubstrateSpec
    appearance: Tensor
    motion: Tensor
    budget: PrivacyBudget
    diagnostics: dict[str, float] = field(default_factory=dict)

    def procedures(self) -> int:
        return int(self.motion.shape[0])


def assemble_workload(per_procedure_gradients: Tensor) -> Tensor:
    """AG: rows are procedures, columns are window micro-steps.

    Ref: Eq. (2), Algorithm 1 step 10.
    """
    if per_procedure_gradients.dim() != 3:
        raise ValueError("per-procedure gradients must be P x d x k")
    return per_procedure_gradients.sum(dim=0)


def add_shaped_noise(
    workload: Tensor,
    projector: CouplingProjector,
    covariance: ShapedCovariance,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Dtilde = AG + Z with Z drawn from the coupling-shaped covariance.

    Ref: Eq. (2), Algorithm 1 step 11.
    """
    if workload.dim() != 2 or workload.shape[0] != projector.dimension:
        raise ValueError("workload must be d x k and match the subspace dimension")
    noise = sample_shaped_noise(
        covariance=covariance,
        projector=projector,
        rows=1,
        columns=int(workload.shape[1]),
        generator=generator,
        dtype=workload.dtype,
    ).squeeze(0)
    return workload + noise, noise


def privatised_update(
    workload: Tensor,
    projector: CouplingProjector,
    covariance: ShapedCovariance,
    learning_rate: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """theta <- theta - gamma (AG + Z).

    Ref: Algorithm 1 step 12.
    """
    if learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    noisy, noise = add_shaped_noise(workload, projector, covariance, generator)
    return -learning_rate * noisy, noise


def budget_from_spec(spec: SubstrateSpec, convert: Callable[[SubstrateSpec], PrivacyBudget]) -> PrivacyBudget:
    """Convert the accumulated Rényi bound once per round, paid at O(K).

    Ref: Eq. (3), Sec. 4.4 (the accounting block pays O(K), not O(N)).
    """
    if spec.steps_times_sampling() <= 0.0:
        raise ValueError("steps x q must be positive")
    return convert(spec)


def prefix_window_slack(spec: SubstrateSpec, order: float) -> float:
    return prefix_amplification_term(spec.window_length, order)


def release_substrate(
    codes: Tensor,
    projector: CouplingProjector,
    decoder: Callable[[Tensor, CouplingProjector], tuple[Tensor, Tensor]],
    budget: PrivacyBudget,
) -> tuple[Tensor, Tensor]:
    """Sample z_p and decode both branches; no record and no gradient is touched.

    Ref: Algorithm 1 step 16, Sec. 4.6.
    """
    if codes.dim() != 2:
        raise ValueError("codes must be procedures x latent_dim")
    if not 0.0 < budget.delta < 1.0:
        raise ValueError("the released artifact must carry a valid delta")
    appearance, motion = decoder(codes, projector)
    if appearance.shape[0] != codes.shape[0] or motion.shape[0] != codes.shape[0]:
        raise ValueError("decoder must return one record per procedure")
    return appearance, motion


def release_is_post_processing(decoded_before: Tensor, decoded_after: Tensor, tolerance: float = 1e-6) -> bool:
    """Re-decoding with identical inputs must not move the released record.

    Ref: Sec. 4.6.
    """
    if decoded_before.shape != decoded_after.shape:
        raise ValueError("shapes must agree")
    return bool(torch.allclose(decoded_before, decoded_after, atol=tolerance, rtol=0.0))
