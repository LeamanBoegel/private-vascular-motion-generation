"""Soft actor-critic reproduced within the study on the same held-out vasculatures.

Ref: Table 5 panel (b) (one soft actor-critic reproduced within this study).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .environment import EndovascularEnvironment

LOG_STD_BOUNDS: tuple[float, float] = (-5.0, 2.0)


@dataclass(frozen=True)
class SACConfig:
    state_dim: int
    action_dim: int
    hidden: int = 64
    learning_rate: float = 3e-4
    discount: float = 0.99
    tau: float = 0.02
    target_entropy_scale: float = 1.0


@dataclass
class ReplayBuffer:
    capacity: int
    state_dim: int
    action_dim: int
    states: Tensor = field(init=False)
    actions: Tensor = field(init=False)
    rewards: Tensor = field(init=False)
    next_states: Tensor = field(init=False)
    terminals: Tensor = field(init=False)
    cursor: int = field(init=False, default=0)
    size: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        self.states = torch.zeros(self.capacity, self.state_dim, dtype=torch.float32)
        self.actions = torch.zeros(self.capacity, self.action_dim, dtype=torch.float32)
        self.rewards = torch.zeros(self.capacity, dtype=torch.float32)
        self.next_states = torch.zeros(self.capacity, self.state_dim, dtype=torch.float32)
        self.terminals = torch.zeros(self.capacity, dtype=torch.float32)

    def push(self, state: Tensor, action: Tensor, reward: float, next_state: Tensor, terminal: bool) -> None:
        index = self.cursor
        self.states[index] = state.detach()
        self.actions[index] = action.detach()
        self.rewards[index] = float(reward)
        self.next_states[index] = next_state.detach()
        self.terminals[index] = float(terminal)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, generator: torch.Generator | None = None) -> tuple[Tensor, ...]:
        if self.size < batch:
            raise ValueError("not enough transitions to sample")
        indices = torch.randint(0, self.size, (batch,), generator=generator)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminals[indices],
        )


class GaussianPolicy(nn.Module):
    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(config.hidden, config.action_dim)
        self.log_std = nn.Linear(config.hidden, config.action_dim)

    def distribution(self, state: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.trunk(state)
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), *LOG_STD_BOUNDS)
        return mean, log_std

    def sample(self, state: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        mean, log_std = self.distribution(state)
        noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
        pre_tanh = mean + log_std.exp() * noise
        action = torch.tanh(pre_tanh)
        log_probability = _gaussian_log_prob(pre_tanh, mean, log_std) - _tanh_correction(pre_tanh)
        return action, log_probability.sum(dim=1, keepdim=True)

    def deterministic(self, state: Tensor) -> Tensor:
        mean, _log_std = self.distribution(state)
        return torch.tanh(mean)


class TwinCritic(nn.Module):
    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.first = _critic_trunk(config)
        self.second = _critic_trunk(config)


def _critic_trunk(config: SACConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.state_dim + config.action_dim, config.hidden),
        nn.ReLU(),
        nn.Linear(config.hidden, config.hidden),
        nn.ReLU(),
        nn.Linear(config.hidden, 1),
    )


def _gaussian_log_prob(sample: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    variance = (2.0 * log_std).exp()
    return -0.5 * ((sample - mean) ** 2 / variance + 2.0 * log_std + torch.log(torch.tensor(2.0 * torch.pi)))


def _tanh_correction(pre_tanh: Tensor) -> Tensor:
    return torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)


class SoftActorCritic:
    """Entropy-regularised actor-critic with twin critics and automatic temperature."""

    def __init__(self, config: SACConfig, seed: int = 0) -> None:
        self.config = config
        self.generator = torch.Generator().manual_seed(seed)
        self.actor = GaussianPolicy(config)
        self.critic = TwinCritic(config)
        self.target_critic = TwinCritic(config)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.target_entropy = -config.target_entropy_scale * config.action_dim

    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def update(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        states, actions, rewards, next_states, terminals = batch
        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(next_states)
            target_input = torch.cat([next_states, next_actions], dim=1)
            target_value = (
                torch.minimum(self.target_critic.first(target_input), self.target_critic.second(target_input))
                - self.alpha() * next_log_probability
            )
            backup = rewards.unsqueeze(1) + self.config.discount * (1.0 - terminals.unsqueeze(1)) * target_value
        critic_input = torch.cat([states, actions], dim=1)
        critic_loss: Tensor = torch.nn.functional.mse_loss(self.critic.first(critic_input), backup)
        critic_loss = critic_loss + torch.nn.functional.mse_loss(self.critic.second(critic_input), backup)
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        sampled_actions, log_probability = self.actor.sample(states)
        actor_input = torch.cat([states, sampled_actions], dim=1)
        actor_loss: Tensor = (
            self.alpha().detach() * log_probability
            - torch.minimum(self.critic.first(actor_input), self.critic.second(actor_input))
        ).mean()
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        alpha_loss: Tensor = -(self.log_alpha * (log_probability + self.target_entropy).detach()).mean()
        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.mul_(1.0 - self.config.tau)
                target_parameter.add_(self.config.tau * parameter)
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha().item()),
        }

    def act(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            dtype = next(self.actor.parameters()).dtype
            return self.actor.deterministic(state.to(dtype).unsqueeze(0)).squeeze(0)


def rollout_sac(
    agent: SoftActorCritic,
    environment: EndovascularEnvironment,
    steps: int,
) -> tuple[Tensor, Tensor]:
    """Roll out the deterministic actor and return the force trace and success flag."""
    state = environment.reset()
    forces: list[float] = []
    success = torch.zeros(1, dtype=torch.float64)
    for _ in range(steps):
        action = agent.act(state)
        outcome = environment.step(action)
        forces.append(outcome.contact_force_n)
        state = outcome.state
        if outcome.done:
            success[0] = float(outcome.success)
            break
    return torch.tensor(forces, dtype=torch.float64), success
