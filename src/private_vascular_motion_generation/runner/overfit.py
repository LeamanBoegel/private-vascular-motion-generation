"""Single-batch fit used to confirm that the optimisation signal reaches the branches.

Ref: Sec. 4.5 (the appearance and motion branches are trained jointly), Sec. 4.9.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..branches.pairing import BranchPair, ProcedureBatch
from .optim import build_adamw


@dataclass(frozen=True)
class FitSpec:
    steps: int
    learning_rate: float = 5e-3
    weight_decay: float = 0.0
    log_every: int = 0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")


@dataclass
class FitResult:
    losses: tuple[float, ...]
    steps: int

    def trace(self) -> torch.Tensor:
        return torch.tensor(self.losses, dtype=torch.float64)

    def head_mean(self, window: int) -> float:
        if window < 1 or window > len(self.losses):
            raise ValueError("window outside the recorded trace")
        return float(sum(self.losses[:window]) / window)

    def tail_mean(self, window: int) -> float:
        if window < 1 or window > len(self.losses):
            raise ValueError("window outside the recorded trace")
        return float(sum(self.losses[-window:]) / window)

    def relative_decrease(self, window: int) -> float:
        head = self.head_mean(window)
        if head == 0.0:
            raise ValueError("head of the trace carries no loss")
        return (head - self.tail_mean(window)) / head

    def monotone_fraction(self) -> float:
        comparisons = [float(later <= earlier + 1e-9) for earlier, later in zip(self.losses, self.losses[1:])]
        return sum(comparisons) / len(comparisons) if comparisons else 0.0


class SingleBatchFit:
    """Supervised descent on one fixed paired batch; carries no privacy account."""

    def __init__(self, pair: BranchPair, spec: FitSpec, seed: int) -> None:
        self.pair = pair
        self.spec = spec
        self.seed = seed

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.pair.parameters() if parameter.requires_grad]

    def run(self, batch: ProcedureBatch) -> FitResult:
        generator = torch.Generator().manual_seed(self.seed)
        optimiser = build_adamw(self.trainable_parameters(), self.spec.learning_rate, self.spec.weight_decay)
        losses: list[float] = []
        for _step in range(self.spec.steps):
            optimiser.zero_grad(set_to_none=True)
            terms = self.pair.concatenated_residual(batch, generator)
            loss: Tensor = terms.mean()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
        return FitResult(losses=tuple(losses), steps=self.spec.steps)


def disappears_under_fit(result: FitResult, min_relative_decrease: float, window: int) -> bool:
    """A single batch that is fit at all must show a clear decrease in the residual."""
    if not 0.0 < min_relative_decrease < 1.0:
        raise ValueError("required decrease must lie in (0, 1)")
    return result.relative_decrease(window) >= min_relative_decrease


def branch_balance(
    batch: ProcedureBatch, pair: BranchPair, generator: torch.Generator | None = None
) -> dict[str, float]:
    """Both branches must contribute to the concatenated residual."""
    appearance_terms, motion_terms = pair.per_procedure_residuals(batch, generator)
    appearance = float(appearance_terms.mean().item())
    motion = float(motion_terms.mean().item())
    return {
        "appearance": appearance,
        "motion": motion,
        "share_appearance": appearance / (appearance + motion) if appearance + motion > 0 else 0.0,
    }
