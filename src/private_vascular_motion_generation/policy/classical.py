"""Classical constrained-control comparators.

Ref: Table 5 panel (a): control-barrier-function safety filter and reactive
potential-field avoider.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

SAFETY_MARGIN: float = 0.02
BARRIER_GAIN: float = 1.5


@dataclass(frozen=True)
class BarrierSolution:
    action: Tensor
    intervened: bool
    constraint_value: float


def barrier_value(wall_distance: Tensor, margin: float = SAFETY_MARGIN) -> Tensor:
    """h(x) = d_wall(x) - margin, positive inside the safe set."""
    if wall_distance.dim() != 1:
        raise ValueError("wall distances must be one-dimensional")
    return wall_distance - margin


def control_barrier_filter(
    nominal_action: Tensor,
    wall_distance: Tensor,
    wall_normal: Tensor,
    gain: float = BARRIER_GAIN,
    margin: float = SAFETY_MARGIN,
) -> BarrierSolution:
    """Project the nominal action onto the half-space that keeps h non-decreasing.

    The normal points from the centreline outwards, so a positive drift approaches the wall and
    the admissible half-space is drift <= gain * h; an already-violated h therefore admits only
    an inward correction.

    Ref: Table 5 panel (a).
    """
    if nominal_action.dim() != 2:
        raise ValueError("action must be batch x action_dim")
    if wall_normal.shape != nominal_action.shape:
        raise ValueError("wall normal must align with the action")
    if wall_distance.shape[0] != nominal_action.shape[0]:
        raise ValueError("one wall distance per batch entry is required")
    h = barrier_value(wall_distance, margin)
    drift = (wall_normal * nominal_action).sum(dim=1)
    slack = gain * h - drift
    needs_intervention = slack < 0.0
    correction = torch.zeros_like(nominal_action)
    if bool(needs_intervention.any().item()):
        normal = wall_normal[needs_intervention]
        denominator = (normal * normal).sum(dim=1, keepdim=True).clamp_min(torch.finfo(normal.dtype).tiny)
        magnitude = (-slack[needs_intervention]).unsqueeze(1)
        correction[needs_intervention] = normal * (magnitude / denominator)
    return BarrierSolution(
        action=nominal_action - correction,
        intervened=bool(needs_intervention.any().item()),
        constraint_value=float(h.min().item()),
    )


def potential_field_action(
    tip: Tensor,
    target: Tensor,
    obstacle_centres: Tensor,
    attractive_gain: float = 1.0,
    repulsive_gain: float = 0.05,
    influence_radius: float = 0.25,
) -> Tensor:
    """Attractive pull toward the endpoint plus a short-range repulsion from the wall.

    Ref: Table 5 panel (a).
    """
    if tip.dim() != 2 or tip.shape[1] != 3:
        raise ValueError("tip must be batch x 3")
    if target.shape != tip.shape:
        raise ValueError("target must align with the tip")
    if obstacle_centres.dim() != 2 or obstacle_centres.shape[1] != 3:
        raise ValueError("obstacle centres must be m x 3")
    attraction = attractive_gain * (target - tip)
    offsets = obstacle_centres.unsqueeze(0) - tip.unsqueeze(1)
    distances = torch.linalg.norm(offsets, dim=2).clamp_min(torch.finfo(tip.dtype).tiny)
    within = (distances < influence_radius).to(tip.dtype)
    magnitude = repulsive_gain * within * (1.0 / distances - 1.0 / influence_radius) / distances**2
    repulsion = (offsets * magnitude.unsqueeze(2)).sum(dim=1)
    combined: Tensor = attraction - repulsion
    return combined


def potential_field_rollout(
    start: Tensor, target: Tensor, obstacle_centres: Tensor, steps: int, step_size: float = 0.05
) -> Tensor:
    """Integrate the field to produce a full trajectory for the held-out procedure."""
    if steps < 1:
        raise ValueError("rollout needs at least one step")
    trajectory = [start]
    current = start
    for _ in range(steps):
        update = potential_field_action(current, target, obstacle_centres)
        current = current + step_size * update
        trajectory.append(current)
    return torch.stack(trajectory, dim=0).squeeze(1)
