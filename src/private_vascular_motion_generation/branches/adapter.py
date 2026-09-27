"""The privatised attention parameter block and the window assignment it is indexed by.

Ref: Sec. 4.4 (d is the dimension of the privatised attention mechanisms, k the window
length), Eq. (2), Algorithm 1 steps 6-12.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AdapterSpec:
    dimension: int
    micro_steps: int

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.micro_steps < 1:
            raise ValueError("dimension and micro-step count must be positive")

    def parameter_shape(self) -> tuple[int, int]:
        return (self.dimension, self.micro_steps)


class PrivateCouplingAdapter(nn.Module):
    """Carries the privatised attention parameters theta of shape d x k."""

    def __init__(self, spec: AdapterSpec) -> None:
        super().__init__()
        self.spec = spec
        self.theta = nn.Parameter(torch.zeros(spec.dimension, spec.micro_steps))
        self.scale = nn.Parameter(torch.ones(spec.dimension))

    def window_assignment(self, token_count: int) -> Tensor:
        """Assign each token position to one window micro-step, tiling the window."""
        if token_count < 1:
            raise ValueError("token count must be positive")
        return torch.arange(token_count, dtype=torch.long) % self.spec.micro_steps

    def modulate(self, tokens: Tensor, windows: Tensor | None = None) -> Tensor:
        """Gate a token stack of width d by the privatised block, one column per window.

        theta enters multiplicatively so that the per-procedure gradient carries the
        token content of that procedure rather than a data-independent mean.
        """
        if tokens.dim() != 3:
            raise ValueError("tokens must be batch x length x dimension")
        if tokens.shape[2] != self.spec.dimension:
            raise ValueError("token width disagrees with the privatised block")
        assignment = self.window_assignment(tokens.shape[1]) if windows is None else windows
        if assignment.dim() != 1 or assignment.shape[0] != tokens.shape[1]:
            raise ValueError("window assignment must carry one entry per token")
        block = self.theta.index_select(1, assignment).transpose(0, 1).unsqueeze(0)
        return tokens * (self.scale.unsqueeze(0).unsqueeze(0) + block)

    def total_noise_energy(self) -> float:
        return float((self.theta**2).sum().item())

    def parameter_block(self) -> Tensor:
        return self.theta

    def snapshot(self) -> Tensor:
        return self.theta.detach().clone()

    def restore(self, snapshot: Tensor) -> None:
        if snapshot.shape != self.theta.shape:
            raise ValueError("snapshot shape disagrees with the privatised block")
        with torch.no_grad():
            self.theta.copy_(snapshot)
