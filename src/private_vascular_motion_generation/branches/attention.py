"""The privatised attention modules of the latent video diffusion transformer.

Ref: Sec. 4.4, Sec. 4.5 (attention modules adjusted in the manner of DP-LDM).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PrivateAttentionBlock(nn.Module):
    """Cross-attention block whose parameters form the privatised set theta_1."""

    def __init__(self, dim: int, heads: int, context_dim: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by the head count")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.action_embedder = nn.Linear(context_dim, dim, bias=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def _split(self, tensor: Tensor) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: Tensor, action_context: Tensor | None = None) -> Tensor:
        if hidden.dim() != 3:
            raise ValueError("expected batch x length x dim")
        if hidden.shape[2] != self.dim:
            raise ValueError("hidden width disagrees with the block width")
        if action_context is None:
            conditioned = hidden
        else:
            if action_context.dim() != 2 or action_context.shape[0] != hidden.shape[0]:
                raise ValueError("action context must carry one row per procedure")
            conditioned = hidden + self.action_embedder(action_context).unsqueeze(1)
        normalised = self.norm(conditioned)
        query = self._split(self.to_q(normalised))
        key = self._split(self.to_k(normalised))
        value = self._split(self.to_v(normalised))
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        batch, heads, length, head_dim = attended.shape
        merged = attended.transpose(1, 2).reshape(batch, length, heads * head_dim)
        mixed: Tensor = self.to_out(merged)
        blend = torch.sigmoid(self.gate)
        return hidden + blend * mixed


class PrivatisedTransformerStack(nn.Module):
    """Stack of attention modules; every parameter outside the stack stays frozen.

    Ref: Sec. 4.4 (only the attention modules are privatised), Algorithm 1 step 2.
    """

    def __init__(self, depth: int, dim: int, heads: int, context_dim: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.dim = dim
        self.blocks = nn.ModuleList([PrivateAttentionBlock(dim, heads, context_dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)

    def frozen_parameters(self, keep: int = 1) -> list[nn.Parameter]:
        """A stand-in for the frozen backbone outside the attention modules."""
        embedding = nn.Parameter(torch.randn(self.dim, self.dim) * 0.02)
        return [embedding for _ in range(keep)]

    def forward(self, hidden: Tensor, action_context: Tensor | None = None) -> Tensor:
        tokens = hidden
        for block in self.blocks:
            tokens = block(tokens, action_context)
        normalised: Tensor = self.norm_out(tokens)
        return normalised

    def privatised_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.privatised_parameters())

    def noise_dimension(self) -> int:
        return self.dim
