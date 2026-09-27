"""Optimiser and schedule construction for the private fine-tune.

Ref: Sec. 2.2 (the 12,000 x 0.0032 optimisation plan), Sec. 4.9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OptimisationPlan:
    rounds: int
    learning_rate: float
    warmup_fraction: float = 0.05
    minimum_learning_ratio: float = 0.02
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clip: float | None = None

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup fraction must lie in [0, 1)")
        if not 0.0 <= self.minimum_learning_ratio <= 1.0:
            raise ValueError("minimum learning ratio must lie in [0, 1]")

    def warmup_rounds(self) -> int:
        return int(round(self.warmup_fraction * self.rounds))

    def to_dict(self) -> dict[str, object]:
        return {
            "rounds": self.rounds,
            "learning_rate": self.learning_rate,
            "warmup_fraction": self.warmup_fraction,
            "minimum_learning_ratio": self.minimum_learning_ratio,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "gradient_clip": self.gradient_clip,
        }


def build_optimiser(parameters: list[nn.Parameter], plan: OptimisationPlan) -> torch.optim.Optimizer:
    """Plain gradient descent; the private pass applies theta <- theta - gamma (AG + Z).

    Ref: Algorithm 1 step 12.
    """
    if not parameters:
        raise ValueError("at least one parameter group is required")
    return torch.optim.SGD(
        parameters,
        lr=plan.learning_rate,
        momentum=0.0,
        weight_decay=plan.weight_decay,
    )


def build_adamw(
    parameters: list[nn.Parameter], learning_rate: float, weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    """Adaptive optimiser used by the policy baselines, which carry no privacy account."""
    if not parameters:
        raise ValueError("at least one parameter group is required")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def learning_rate_at(round_index: int, plan: OptimisationPlan) -> float:
    """Linear warmup followed by cosine decay to the floor ratio."""
    if round_index < 0:
        raise ValueError("round index must be non-negative")
    warmup = plan.warmup_rounds()
    if warmup > 0 and round_index < warmup:
        return plan.learning_rate * (round_index + 1) / warmup
    progress = (round_index - warmup) / max(1, plan.rounds - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = plan.minimum_learning_ratio
    return plan.learning_rate * (floor + (1.0 - floor) * cosine)


def apply_learning_rate(optimiser: torch.optim.Optimizer, round_index: int, plan: OptimisationPlan) -> float:
    value = learning_rate_at(round_index, plan)
    for group in optimiser.param_groups:
        group["lr"] = value
    return value


def clip_gradients(parameters: list[nn.Parameter], maximum: float) -> float:
    if maximum <= 0.0:
        raise ValueError("gradient clip must be positive")
    return float(torch.nn.utils.clip_grad_norm_(parameters, maximum).item())


def schedule_ladder(plan: OptimisationPlan, samples: int) -> list[float]:
    """Evenly spaced learning rates over the plan, endpoints included.

    A ladder of n samples has n - 1 intervals, so the stride is taken from the last round rather
    than by truncating a floor-divided range, which would drop the final sample.
    """
    if samples < 2:
        raise ValueError("at least two samples are required")
    if samples > plan.rounds:
        raise ValueError("cannot draw more samples than there are rounds")
    last = plan.rounds - 1
    indices = [round(index * last / (samples - 1)) for index in range(samples)]
    return [learning_rate_at(index, plan) for index in indices]
