"""Rolling every strategy over the same held-out procedure trajectories.

Ref: Sec. 2.6, Table 5.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ..cohort.procedure import ProcedureRecord
from ..metrics.safety import SafetyRow, safety_from_rollouts
from ..metrics.trajectory import path_length
from .classical import control_barrier_filter, potential_field_action
from .environment import EndovascularEnvironment, EnvironmentSpec, tube_from_centreline

NOMINAL_GAIN: float = 0.9


@dataclass(frozen=True)
class StrategyOutcome:
    label: str
    forces: Tensor
    successes: Tensor
    travelled: Tensor
    centreline_length: float


def environment_for(record: ProcedureRecord, step_size: float = 0.05) -> EndovascularEnvironment:
    centreline = record.tip_trajectory().to(torch.float64)
    radius = tube_from_centreline(centreline)
    spec = EnvironmentSpec(frames=int(centreline.shape[0]), step_size=step_size)
    return EndovascularEnvironment(spec=spec, centreline=centreline, radius_profile=radius)


def _path_following_action(environment: EndovascularEnvironment) -> Tensor:
    target_index = min(environment.progress + 1, environment.spec.frames - 1)
    target = environment.centreline[target_index]
    return NOMINAL_GAIN * (target - environment.tip)


def rollout_path_follower(
    environment: EndovascularEnvironment, steps: int, use_barrier: bool
) -> tuple[Tensor, Tensor, float]:
    """Nominal crossing strategy, optionally filtered by the control barrier function."""
    environment.reset()
    forces: list[float] = []
    success = torch.zeros(1, dtype=torch.float64)
    for _ in range(steps):
        nominal = _path_following_action(environment)
        if use_barrier:
            distance = torch.tensor([environment.wall_distance()], dtype=torch.float64)
            normal = (environment.tip - environment.centreline[environment.progress]).reshape(1, 3)
            filtered = control_barrier_filter(nominal.reshape(1, 3), distance, normal)
            action = filtered.action.squeeze(0)
        else:
            action = nominal
        outcome = environment.step(action)
        forces.append(outcome.contact_force_n)
        if outcome.done:
            success[0] = float(outcome.success)
            break
    return torch.tensor(forces, dtype=torch.float64), success, environment.travelled


def rollout_potential_field(
    environment: EndovascularEnvironment, obstacles: Tensor, steps: int
) -> tuple[Tensor, Tensor, float]:
    environment.reset()
    forces: list[float] = []
    success = torch.zeros(1, dtype=torch.float64)
    for _ in range(steps):
        target_index = min(environment.progress + 1, environment.spec.frames - 1)
        action = potential_field_action(
            environment.tip.reshape(1, 3), environment.centreline[target_index].reshape(1, 3), obstacles
        ).squeeze(0)
        outcome = environment.step(action)
        forces.append(outcome.contact_force_n)
        if outcome.done:
            success[0] = float(outcome.success)
            break
    return torch.tensor(forces, dtype=torch.float64), success, environment.travelled


def rollout_action_source(
    environment: EndovascularEnvironment, steps: int, source: Callable[[Tensor], Tensor]
) -> tuple[Tensor, Tensor, float]:
    """Generic rollout for any action source that maps the environment state to an action."""
    state = environment.reset()
    forces: list[float] = []
    success = torch.zeros(1, dtype=torch.float64)
    for _ in range(steps):
        action = source(state)
        if action.dim() != 1:
            action = action.reshape(-1)
        outcome = environment.step(action[:3])
        forces.append(outcome.contact_force_n)
        state = outcome.state
        if outcome.done:
            success[0] = float(outcome.success)
            break
    return torch.tensor(forces, dtype=torch.float64), success, environment.travelled


def aggregate_strategy(
    label: str,
    outcomes: list[tuple[Tensor, Tensor, float]],
    centreline: Tensor,
    fluoroscopy_min: float,
    contrast_ml: float,
    navigation_min: float,
) -> tuple[SafetyRow, StrategyOutcome]:
    if not outcomes:
        raise ValueError("at least one procedure rollout is required")
    length = min(int(forces.numel()) for forces, _success, _travelled in outcomes)
    stacked = torch.stack([forces[:length] for forces, _success, _travelled in outcomes], dim=0)
    successes = torch.cat([success for _forces, success, _travelled in outcomes])
    travelled = torch.tensor([travelled for _forces, _success, travelled in outcomes], dtype=torch.float64)
    reference_length = path_length(centreline.to(torch.float64))
    efficiency = reference_length / max(torch.finfo(torch.float64).tiny, float(travelled.mean().item()))
    row = safety_from_rollouts(
        label=label,
        forces=stacked,
        successful=successes,
        fluoroscopy_min=fluoroscopy_min,
        contrast_ml=contrast_ml,
        navigation_min=navigation_min,
        path_efficiency=efficiency,
    )
    return row, StrategyOutcome(
        label=label,
        forces=stacked,
        successes=successes,
        travelled=travelled,
        centreline_length=reference_length,
    )


def obstacles_from_centreline(centreline: Tensor, offset: float = 0.12) -> Tensor:
    """Wall samples along the centreline, used as repulsive sources."""
    if centreline.dim() != 2 or centreline.shape[0] < 2:
        raise ValueError("centreline must carry at least two frames")
    direction = centreline[1:] - centreline[:-1]
    perpendicular = torch.zeros_like(direction)
    perpendicular[:, 1] = -direction[:, 0]
    perpendicular[:, 0] = direction[:, 1]
    norm = torch.linalg.norm(perpendicular, dim=1, keepdim=True).clamp_min(torch.finfo(centreline.dtype).tiny)
    offset_centreline: Tensor = centreline[1:] + offset * perpendicular / norm
    return offset_centreline


def published_comparator_rows() -> tuple[dict[str, object], ...]:
    """Each published comparator reports task success alone.

    Ref: Table 5 panel (b).
    """
    return (
        {
            "label": "strongest 2026 endovascular world-model controller",
            "held_out_vasculatures": 10,
            "task_success": 0.90,
            "endpoints_reported": "task success only",
            "source": "published",
        },
        {
            "label": "base world-model controller",
            "held_out_vasculatures": 10,
            "task_success": 0.65,
            "endpoints_reported": "task success only",
            "source": "published",
        },
        {
            "label": "soft actor-critic",
            "held_out_vasculatures": None,
            "task_success": 0.37,
            "endpoints_reported": "task success only",
            "source": "reproduced",
        },
    )


def contact_events_per_procedure(force_rows: Tensor) -> Tensor:
    from ..cohort.force_channel import count_contact_events

    return torch.tensor([len(count_contact_events(row)) for row in force_rows], dtype=torch.float64)
