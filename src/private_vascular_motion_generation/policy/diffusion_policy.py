"""Diffusion policy learned on the released substrate.

Ref: Sec. 4.5 (the policy learned on the released substrate is a diffusion policy).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..cohort.procedure import MOTION_DIM


@dataclass(frozen=True)
class DiffusionPolicyConfig:
    state_dim: int
    action_dim: int = MOTION_DIM
    horizon: int = 4
    width: int = 64
    depth: int = 2
    diffusion_steps: int = 16


class NoiseSchedule(nn.Module):
    betas: Tensor
    alpha_bar: Tensor

    def __init__(self, steps: int, low: float = 1e-4, high: float = 2e-2) -> None:
        super().__init__()
        if steps < 2:
            raise ValueError("at least two diffusion steps are required")
        betas = torch.linspace(low, high, steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))

    def noisy(self, clean: Tensor, noise: Tensor, index: Tensor) -> Tensor:
        coefficient = self.alpha_bar[index].view(-1, 1, 1) ** 0.5
        spread = (1.0 - self.alpha_bar[index]).view(-1, 1, 1) ** 0.5
        return coefficient * clean + spread * noise


class ActionDenoiser(nn.Module):
    def __init__(self, config: DiffusionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.state_proj = nn.Linear(config.state_dim, config.width)
        self.action_proj = nn.Linear(config.horizon * config.action_dim, config.width)
        self.time_proj = nn.Linear(1, config.width)
        layers: list[nn.Module] = []
        for _ in range(config.depth):
            layers.extend([nn.Linear(config.width, config.width), nn.SiLU()])
        layers.append(nn.Linear(config.width, config.horizon * config.action_dim))
        self.body = nn.Sequential(*layers)

    def forward(self, state: Tensor, noisy_action: Tensor, time: Tensor) -> Tensor:
        if state.dim() != 3:
            raise ValueError("state must be batch x history x state_dim")
        pooled = self.state_proj(state).mean(dim=1)
        flat_action = noisy_action.reshape(noisy_action.shape[0], -1)
        hidden = pooled + self.action_proj(flat_action) + self.time_proj(time.view(-1, 1))
        decoded: Tensor = self.body(hidden)
        return decoded.reshape(-1, self.config.horizon, self.config.action_dim)


class DiffusionPolicy(nn.Module):
    """Denoises a four-step action chunk conditioned on the kinematic history."""

    schedule: NoiseSchedule
    denoiser: ActionDenoiser

    def __init__(self, config: DiffusionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.schedule = NoiseSchedule(config.diffusion_steps)
        self.denoiser = ActionDenoiser(config)

    def training_loss(self, state: Tensor, action: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Standard epsilon-prediction objective over the action chunk."""
        if action.shape[1] != self.config.horizon:
            raise ValueError("action chunk disagrees with the declared horizon")
        batch = action.shape[0]
        noise = torch.randn(action.shape, generator=generator, dtype=action.dtype)
        index = torch.randint(0, self.config.diffusion_steps, (batch,), generator=generator)
        time = index.to(action.dtype) / max(1, self.config.diffusion_steps - 1)
        noisy = self.schedule.noisy(action, noise, index)
        predicted = self.denoiser(state, noisy, time)
        return torch.mean((predicted - noise) ** 2)

    @torch.no_grad()
    def sample(self, state: Tensor, generator: torch.Generator | None = None) -> Tensor:
        batch = state.shape[0]
        action = torch.randn(batch, self.config.horizon, self.config.action_dim, generator=generator, dtype=state.dtype)
        for index in range(self.config.diffusion_steps - 1, -1, -1):
            time = torch.full((batch,), index / max(1, self.config.diffusion_steps - 1), dtype=state.dtype)
            predicted = self.denoiser(state, action, time)
            alpha_bar = self.schedule.alpha_bar[index]
            beta = self.schedule.betas[index]
            coefficient = 1.0 / (1.0 - beta) ** 0.5
            mean = coefficient * (action - beta / (1.0 - alpha_bar) ** 0.5 * predicted)
            if index > 0:
                variance = beta * (1.0 - self.schedule.alpha_bar[index - 1]) / (1.0 - alpha_bar)
                action = mean + variance**0.5 * torch.randn(action.shape, generator=generator, dtype=action.dtype)
            else:
                action = mean
        return action

    def action_dim(self) -> int:
        return self.config.action_dim
