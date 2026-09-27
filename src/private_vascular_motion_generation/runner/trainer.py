"""The procedure-level coupling-anisotropic private training loop.

Ref: Algorithm 1 (CAPS-train), Eq. (1)-(4), Sec. 4.4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from ..accounting.clipping import clip_workload
from ..accounting.orders import OrderGrid
from ..accounting.renyi import PrivacyBudget, composed_rdp, convert_to_epsilon, rdp_from_noise_multiplier
from ..accounting.tuple import PrivacyTuple
from ..branches.pairing import BranchPair, ProcedureBatch
from ..coupling.head import CouplingHead
from ..estimator.reference_estimator import EstimatorResult, ReferenceShapingEstimator
from ..geometry.concentration import concentration_from_gradients
from ..geometry.projector import CouplingProjector
from ..geometry.shaped_covariance import ShapedCovariance, normalised_sensitivity
from ..substrate import add_shaped_noise, assemble_workload
from .batching import RoundSampler
from .checkpoint import CheckpointMeta, config_digest, save_checkpoint
from .logging_utils import get_logger, summarise_round
from .optim import OptimisationPlan, apply_learning_rate, build_optimiser

LOGGER = get_logger("trainer")


@dataclass(frozen=True)
class TrainingSpec:
    rounds: int
    delta: float
    clipping_norm: float
    sampling_rate: float
    checkpoint_every: int = 0
    measure_rho_every: int = 1

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if self.clipping_norm <= 0.0:
            raise ValueError("clipping norm must be positive")
        if not 0.0 < self.sampling_rate <= 1.0:
            raise ValueError("sampling rate must lie in (0, 1]")
        if self.measure_rho_every < 1:
            raise ValueError("rho measurement stride must be positive")


@dataclass
class RoundRecord:
    round_index: int
    loss: float
    rho: float
    workload_norm: float
    noise_norm: float
    learning_rate: float


@dataclass
class TrainingRun:
    rounds: tuple[RoundRecord, ...]
    budget: PrivacyBudget
    privacy_tuple: PrivacyTuple
    measured_rho: float
    estimator: EstimatorResult
    checkpoint_digest: str
    notes: dict[str, str] = field(default_factory=dict)

    def loss_trace(self) -> Tensor:
        return torch.tensor([record.loss for record in self.rounds], dtype=torch.float64)

    def rho_trace(self) -> Tensor:
        return torch.tensor([record.rho for record in self.rounds], dtype=torch.float64)

    def first_loss(self) -> float:
        if not self.rounds:
            raise ValueError("the run recorded no rounds")
        return self.rounds[0].loss

    def last_loss(self) -> float:
        if not self.rounds:
            raise ValueError("the run recorded no rounds")
        return self.rounds[-1].loss


class CAPSTrainer:
    """Runs the four-component mechanism end to end on a schema-compatible cohort."""

    def __init__(
        self,
        pair: BranchPair,
        head: CouplingHead,
        covariance: ShapedCovariance,
        plan: OptimisationPlan,
        training: TrainingSpec,
        estimator: ReferenceShapingEstimator,
        order_grid: OrderGrid,
        seed: int,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.pair = pair
        self.head = head
        self.covariance = covariance
        self.plan = plan
        self.training = training
        self.estimator = estimator
        self.order_grid = order_grid
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir

    def estimate_subspace(self, reference_batches: Sequence[ProcedureBatch]) -> EstimatorResult:
        """Freeze the shaping subspace before any private record is touched.

        Ref: Algorithm 1 step 1, Algorithm 2.
        """
        result = self.estimator.fit(self.pair, list(reference_batches))
        self.estimator.freeze(result)
        return result

    def run(
        self,
        reference_batches: Sequence[ProcedureBatch],
        sampler: RoundSampler,
        config_payload: dict[str, object],
    ) -> TrainingRun:
        estimator_result = self.estimate_subspace(reference_batches)
        projector: CouplingProjector = estimator_result.projector

        optimiser = build_optimiser([self.pair.adapter.theta], self.plan)
        digest = config_digest(config_payload)
        records: list[RoundRecord] = []
        rho_values: list[float] = []
        rdp_accumulator = torch.zeros(len(self.order_grid.orders), dtype=torch.float64)
        sigma_parallel = self.covariance.sigma_parallel
        sigma_perpendicular = self.covariance.sigma_perpendicular

        for round_index in range(self.training.rounds):
            learning_rate = apply_learning_rate(optimiser, round_index, self.plan)
            batch = sampler.draw()
            workload = self.pair.aggregate_workload(batch)
            clipped, _scales = clip_workload(workload.unsqueeze(0), self.training.clipping_norm)
            aggregated = assemble_workload(clipped)
            noisy, noise = add_shaped_noise(aggregated, projector, self.covariance)
            optimiser.zero_grad(set_to_none=True)
            self.pair.adapter.theta.grad = noisy
            optimiser.step()

            appearance_views, _motion_views = self.pair.branch_gradient_views(batch)
            rho = concentration_from_gradients(appearance_views.detach(), projector)
            rho_values.append(rho)

            multiplier = 1.0 / (
                self.training.clipping_norm * normalised_sensitivity(sigma_parallel, sigma_perpendicular, rho)
            )
            for index, order in enumerate(self.order_grid.orders):
                rdp_accumulator[index] += rdp_from_noise_multiplier(order, self.training.sampling_rate, multiplier)

            loss = float(self.pair.concatenated_residual(batch).mean().item())
            records.append(
                RoundRecord(
                    round_index=round_index,
                    loss=loss,
                    rho=rho,
                    workload_norm=float(torch.linalg.norm(aggregated).item()),
                    noise_norm=float(torch.linalg.norm(noise).item()),
                    learning_rate=learning_rate,
                )
            )
            if (
                self.checkpoint_dir is not None
                and self.training.checkpoint_every > 0
                and (round_index + 1) % self.training.checkpoint_every == 0
            ):
                self._write_checkpoint(round_index, optimiser, digest)

        budget = convert_to_epsilon(
            rdp_accumulator,
            self.order_grid,
            self.training.delta,
            window_length=self.pair.pairing.window_micro_steps,
        )
        privacy = PrivacyTuple(
            epsilon=budget.epsilon,
            delta=budget.delta,
            accountant="renyi",
            order=budget.order,
            steps_times_sampling=self.training.rounds * self.training.sampling_rate,
            clipping_norm=self.training.clipping_norm,
            sigma_parallel=sigma_parallel,
            sigma_perpendicular=sigma_perpendicular,
            rank=projector.rank,
            dimension=projector.dimension,
            rho=float(sum(rho_values) / len(rho_values)) if rho_values else None,
            notes={"sampling_rate": "engineering default, not stated in the manuscript"},
        )
        LOGGER.info(
            "run complete eps=%.4f order=%.1f rho=%.4f",
            budget.epsilon,
            budget.order,
            privacy.rho if privacy.rho is not None else float("nan"),
        )
        return TrainingRun(
            rounds=tuple(records),
            budget=budget,
            privacy_tuple=privacy,
            measured_rho=float(sum(rho_values) / len(rho_values)) if rho_values else float("nan"),
            estimator=estimator_result,
            checkpoint_digest=digest,
        )

    def _write_checkpoint(self, round_index: int, optimiser: torch.optim.Optimizer, digest: str) -> None:
        if self.checkpoint_dir is None:
            return
        meta = CheckpointMeta(
            step=round_index + 1,
            round_index=round_index,
            seed=self.seed,
            initialisations=1,
            privacy_tuple={},
            config_digest=digest,
        )
        path = self.checkpoint_dir / f"caps_round_{round_index + 1:06d}.pt"
        save_checkpoint(path, self.pair, optimiser, meta)
        LOGGER.info("checkpoint written at %s", summarise_round(round_index, self.training.rounds, 0.0, 0.0))


def accumulate_rdp_for_run(
    order_grid: OrderGrid,
    rounds: int,
    sampling_rate: float,
    clipping_norm: float,
    sigma_parallel: float,
    sigma_perpendicular: float,
    rho: float,
) -> Tensor:
    """Closed-form accumulation without the loop, used by the accounting checks."""
    multiplier = 1.0 / (clipping_norm * normalised_sensitivity(sigma_parallel, sigma_perpendicular, rho))
    return composed_rdp(order_grid, rounds, sampling_rate, multiplier)
