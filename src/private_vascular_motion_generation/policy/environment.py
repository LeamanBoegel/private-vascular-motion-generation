"""A tube-following environment over the held-out clinical centreline.

Ref: Sec. 4.8 (policy learning), Sec. 4.10 (contact events, force envelope).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

ACTION_DIM: int = 3
WALL_STIFFNESS_N_PER_UNIT: float = 6.5
PROGRESS_TOLERANCE: float = 0.05


@dataclass(frozen=True)
class EnvironmentSpec:
    frames: int
    step_size: float = 0.05
    action_limit: float = 1.0
    force_penalty: float = 0.35
    deviation_penalty: float = 1.0
    progress_reward: float = 1.0
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.frames < 2:
            raise ValueError("a centreline needs at least two frames")
        if self.step_size <= 0.0:
            raise ValueError("step size must be positive")
        if self.action_limit <= 0.0:
            raise ValueError("action limit must be positive")

    def horizon(self) -> int:
        return self.max_steps if self.max_steps is not None else self.frames - 1


@dataclass
class StepOutcome:
    state: Tensor
    reward: float
    contact_force_n: float
    deviation: float
    progress: int
    done: bool
    success: bool


class EndovascularEnvironment:
    """Deterministic tip-dynamics with a compliant wall and an axial force channel."""

    def __init__(self, spec: EnvironmentSpec, centreline: Tensor, radius_profile: Tensor) -> None:
        if centreline.dim() != 2 or centreline.shape[1] != 3:
            raise ValueError("centreline must be frames x 3")
        if radius_profile.dim() != 1 or radius_profile.shape[0] != centreline.shape[0]:
            raise ValueError("radius profile must carry one entry per centreline frame")
        if spec.frames != centreline.shape[0]:
            raise ValueError("specification and centreline disagree on the frame count")
        self.spec = spec
        self.centreline = centreline.to(torch.float64)
        self.radius_profile = radius_profile.to(torch.float64)
        self.progress = 0
        self.tip = self.centreline[0].clone()
        self.travelled = 0.0
        self.contact_forces: list[float] = []

    def reset(self) -> Tensor:
        self.progress = 0
        self.tip = self.centreline[0].clone()
        self.travelled = 0.0
        self.contact_forces = []
        return self.state()

    def state(self) -> Tensor:
        return torch.cat([self.tip, self.radius_profile[self.progress].reshape(1)])

    def wall_distance(self) -> float:
        offset = float(torch.linalg.norm(self.tip - self.centreline[self.progress]).item())
        return float(self.radius_profile[self.progress].item()) - offset

    def step(self, action: Tensor) -> StepOutcome:
        if action.dim() != 1 or action.shape[0] != ACTION_DIM:
            raise ValueError("action must be a three-dimensional tip displacement")
        limited = torch.clamp(action, -self.spec.action_limit, self.spec.action_limit)
        displacement = self.spec.step_size * limited
        self.tip = self.tip + displacement
        self.travelled += float(torch.linalg.norm(displacement).item())
        self.progress = min(self.progress + 1, self.spec.frames - 1)
        distance = self.wall_distance()
        force = max(0.0, -distance) * WALL_STIFFNESS_N_PER_UNIT
        self.contact_forces.append(force)
        deviation = float(torch.linalg.norm(self.tip - self.centreline[self.progress]).item())
        reward = self.spec.progress_reward - self.spec.deviation_penalty * deviation - self.spec.force_penalty * force
        success = distance >= 0.0
        done = self.progress >= self.spec.frames - 1
        return StepOutcome(
            state=self.state(),
            reward=reward,
            contact_force_n=force,
            deviation=deviation,
            progress=self.progress,
            done=done,
            success=success and done,
        )

    def force_trace(self) -> Tensor:
        return torch.tensor(self.contact_forces, dtype=torch.float64)

    def reached_endpoint(self, target: Tensor) -> bool:
        return float(torch.linalg.norm(self.tip - target).item()) <= PROGRESS_TOLERANCE

    def clip_to_vessel(self) -> None:
        """Keep the tip inside the tube so that the rollout stays physically meaningful."""
        direction = self.tip - self.centreline[self.progress]
        norm = float(torch.linalg.norm(direction).item())
        radius = float(self.radius_profile[self.progress].item())
        if norm > radius and norm > 0.0:
            self.tip = self.centreline[self.progress] + direction * (radius / norm)


def tube_from_centreline(centreline: Tensor, radius: float = 0.08, taper: float = 0.02) -> Tensor:
    """Radius profile narrowing toward the distal end."""
    if centreline.dim() != 2:
        raise ValueError("centreline must be frames x 3")
    frames = centreline.shape[0]
    return radius - taper * torch.linspace(0.0, 1.0, frames, dtype=centreline.dtype)
