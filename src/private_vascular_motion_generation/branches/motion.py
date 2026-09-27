"""Flow-matching action head over the six-channel tip state and its axial force channel.

Ref: Sec. 4.4, Sec. 4.5 (flow-matching action head for the six-channel motion branch).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..cohort.procedure import FORCE_CHANNEL_INDEX, MOTION_DIM


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding width must be even")
        self.dim = dim
        self.scale = scale

    def forward(self, time: Tensor) -> Tensor:
        if time.dim() != 1:
            raise ValueError("time must be a one-dimensional batch")
        half = self.dim // 2
        frequencies = torch.exp(
            -torch.arange(half, dtype=time.dtype, device=time.device) * (torch.log(torch.tensor(self.scale)) / half)
        )
        angles = time.unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class FlowMatchingActionHead(nn.Module):
    """Velocity field trained by conditional flow matching on the motion branch.

    Ref: Sec. 4.4.
    """

    def __init__(self, frames: int, width: int, latent_dim: int, depth: int = 2) -> None:
        super().__init__()
        if min(frames, width, latent_dim) <= 0:
            raise ValueError("widths must be positive")
        self.frames = frames
        self.width = width
        self.time = SinusoidalTimeEmbedding(width)
        self.state_proj = nn.Linear(MOTION_DIM, width)
        self.latent_proj = nn.Linear(latent_dim, width)
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, MOTION_DIM))
        self.velocity = nn.Sequential(*layers)

    def encode(self, state: Tensor, latent: Tensor) -> Tensor:
        """Per-frame token stack of the motion branch at the attention width."""
        if state.dim() != 3 or state.shape[1] != MOTION_DIM:
            raise ValueError("state must be batch x 6 x frames")
        if latent.dim() != 2 or latent.shape[0] != state.shape[0]:
            raise ValueError("latent must carry one code per procedure")
        tokens: Tensor = self.state_proj(state.transpose(1, 2))
        projected: Tensor = self.latent_proj(latent)
        return tokens + projected.unsqueeze(1)

    def forward(self, state: Tensor, time: Tensor, latent: Tensor) -> Tensor:
        """Predict the constant velocity between the noise endpoint and the data endpoint."""
        return self.head_from_tokens(self.encode(state, latent), time)

    def head_from_tokens(self, tokens: Tensor, time: Tensor) -> Tensor:
        """Apply the velocity head to an already encoded token stack."""
        if tokens.dim() != 3:
            raise ValueError("tokens must be batch x frames x width")
        if time.dim() != 1 or time.shape[0] != tokens.shape[0]:
            raise ValueError("one flow time is required per procedure")
        velocity = self.velocity(tokens + self.time(time).unsqueeze(1))
        output: Tensor = velocity.transpose(1, 2)
        return output

    def per_procedure_flow_residual(
        self, target: Tensor, latent: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        if target.dim() != 3 or target.shape[1] != MOTION_DIM:
            raise ValueError("target motion must be batch x 6 x frames")
        batch = target.shape[0]
        noise = torch.randn(target.shape, generator=generator, dtype=target.dtype)
        time = torch.rand(batch, generator=generator, dtype=target.dtype)
        interpolated = time.view(batch, 1, 1) * target + (1.0 - time.view(batch, 1, 1)) * noise
        predicted = self.forward(interpolated, time, latent)
        return ((predicted - (target - noise)) ** 2).mean(dim=(1, 2))

    def flow_matching_loss(self, target: Tensor, latent: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """L = E[ || v_theta(x_t, t) - (x_1 - x_0) ||^2 ] with x_t = t x_1 + (1 - t) x_0.

        Ref: Sec. 4.4.
        """
        if target.dim() != 3 or target.shape[1] != MOTION_DIM:
            raise ValueError("target motion must be batch x 6 x frames")
        batch = target.shape[0]
        noise = torch.randn(target.shape, generator=generator, dtype=target.dtype)
        time = torch.rand(batch, generator=generator, dtype=target.dtype)
        interpolated = time.view(batch, 1, 1) * target + (1.0 - time.view(batch, 1, 1)) * noise
        predicted = self.forward(interpolated, time, latent)
        return torch.mean((predicted - (target - noise)) ** 2)

    def integrate(self, latent: Tensor, steps: int = 8, generator: torch.Generator | None = None) -> Tensor:
        """Euler integration from the noise endpoint; used for release-time decoding only.

        Ref: Sec. 4.4, Algorithm 1 step 16.
        """
        if steps < 1:
            raise ValueError("integration needs at least one step")
        batch = int(latent.shape[0])
        state = torch.randn(batch, MOTION_DIM, self.frames, generator=generator, dtype=latent.dtype)
        grid = torch.linspace(0.0, 1.0, steps + 1, dtype=latent.dtype)
        for start, stop in zip(grid[:-1], grid[1:]):
            time = start.expand(batch)
            state = state + (stop - start) * self.forward(state, time, latent)
        return state


def force_channel_from_motion(motion: Tensor) -> Tensor:
    if motion.dim() != 3 or motion.shape[1] != MOTION_DIM:
        raise ValueError("motion must be batch x 6 x frames")
    return motion[:, FORCE_CHANNEL_INDEX]
