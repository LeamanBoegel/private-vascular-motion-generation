"""Latent world-model controller in the TD-MPC2 style, reproduced in this study.

Ref: Table 5 panel (b) (base world-model controller and the strongest 2026 controller).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .environment import EndovascularEnvironment


@dataclass(frozen=True)
class WorldModelConfig:
    state_dim: int
    action_dim: int
    latent_dim: int = 32
    hidden: int = 64
    learning_rate: float = 3e-4
    planning_horizon: int = 3
    candidates: int = 32
    elite_fraction: float = 0.25


class LatentWorldModel(nn.Module):
    """Encoder, latent dynamics, reward head and terminal value head."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.latent_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_dim, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_dim, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 1),
        )

    def encode(self, state: Tensor) -> Tensor:
        return torch.tanh(self.encoder(state))

    def step(self, latent: Tensor, action: Tensor) -> Tensor:
        delta: Tensor = self.dynamics(torch.cat([latent, action], dim=-1))
        return torch.tanh(latent + delta)

    def reward(self, latent: Tensor, action: Tensor) -> Tensor:
        predicted: Tensor = self.reward_head(torch.cat([latent, action], dim=-1))
        return predicted.squeeze(-1)

    def value(self, latent: Tensor) -> Tensor:
        estimated: Tensor = self.value_head(latent)
        return estimated.squeeze(-1)

    def training_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        terminal_value: Tensor,
    ) -> Tensor:
        """Latent consistency plus reward and value regression.

        Ref: Table 5 panel (b).
        """
        latent = self.encode(states)
        with torch.no_grad():
            target_latent = self.encode(next_states)
        predicted_latent = self.step(latent, actions)
        consistency = torch.mean((predicted_latent - target_latent) ** 2)
        reward_error = torch.mean((self.reward(latent, actions) - rewards) ** 2)
        value_error = torch.mean((self.value(predicted_latent) - terminal_value) ** 2)
        return consistency + reward_error + value_error


class LatentPlanner:
    """Random-shooting planner over the latent model with elite refinement."""

    def __init__(self, model: LatentWorldModel, seed: int = 0) -> None:
        self.model = model
        self.generator = torch.Generator().manual_seed(seed)
        self.optimiser = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)

    @torch.no_grad()
    def plan(self, state: Tensor) -> Tensor:
        config = self.model.config
        latent = self.model.encode(state.unsqueeze(0))
        candidates = torch.randn(
            config.candidates,
            config.planning_horizon,
            config.action_dim,
            generator=self.generator,
            dtype=state.dtype,
        )
        candidates = torch.tanh(candidates)
        scores = self._score(latent, candidates)
        elite_count = max(1, int(config.candidates * config.elite_fraction))
        elite = torch.argsort(scores, descending=True)[:elite_count]
        mean = candidates[elite].mean(dim=0)
        return torch.tanh(mean[0])

    @torch.no_grad()
    def _score(self, latent: Tensor, candidates: Tensor) -> Tensor:
        repeated = latent.expand(candidates.shape[0], latent.shape[1])
        total = torch.zeros(candidates.shape[0], dtype=candidates.dtype)
        for step in range(candidates.shape[1]):
            action = candidates[:, step, :]
            total = total + self.model.reward(repeated, action)
            repeated = self.model.step(repeated, action)
        return total + self.model.value(repeated)

    def update(self, batch: dict[str, Tensor]) -> float:
        loss: Tensor = self.model.training_loss(
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["next_states"],
            batch["terminal_value"],
        )
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return float(loss.item())


def rollout_planner(planner: LatentPlanner, environment: EndovascularEnvironment, steps: int) -> tuple[Tensor, Tensor]:
    """Execute the planner on the held-out centreline and report its safety trace."""
    state = environment.reset()
    forces: list[float] = []
    success = torch.zeros(1, dtype=torch.float64)
    for _ in range(steps):
        action = planner.plan(state)
        outcome = environment.step(action[:3])
        forces.append(outcome.contact_force_n)
        state = outcome.state
        if outcome.done:
            success[0] = float(outcome.success)
            break
    return torch.tensor(forces, dtype=torch.float64), success


def latent_consistency_report(
    model: LatentWorldModel, states: Tensor, actions: Tensor, next_states: Tensor
) -> dict[str, float]:
    with torch.no_grad():
        predicted = model.step(model.encode(states), actions)
        target = model.encode(next_states)
        error = torch.linalg.norm(predicted - target, dim=1)
    return {
        "mean_consistency_error": float(error.mean().item()),
        "max_consistency_error": float(error.max().item()),
    }
