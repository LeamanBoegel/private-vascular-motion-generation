"""Selective state-space and shifted-window attention blocks behind the appearance branch.

Ref: Sec. 4.5 (Swin-UMamba type encoder for state-space vessel segmentation).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SelectiveScanBlock(nn.Module):
    """Token-conditioned linear recurrence over the frame axis."""

    def __init__(self, dim: int, state_dim: int, dt_rank: int | None = None) -> None:
        super().__init__()
        if min(dim, state_dim) <= 0:
            raise ValueError("dimensions must be positive")
        rank = dt_rank if dt_rank is not None else max(1, dim // 16)
        self.dim = dim
        self.state_dim = state_dim
        self.dt_rank = rank
        self.in_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.dt_proj = nn.Linear(rank, dim, bias=True)
        self.dt_up = nn.Linear(dim, rank, bias=False)
        self.b_proj = nn.Linear(dim, state_dim, bias=False)
        self.c_proj = nn.Linear(dim, state_dim, bias=False)
        self.a_log = nn.Parameter(torch.zeros(dim, state_dim))
        self.d_skip = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.a_log.copy_(torch.linspace(-1.0, -3.0, dim).unsqueeze(1).expand(dim, state_dim))

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.dim() != 3:
            raise ValueError("expected batch x length x dim")
        batch, length, dim = tokens.shape
        if dim != self.dim:
            raise ValueError("token width disagrees with the block width")
        gated, gate = self.in_proj(tokens).chunk(2, dim=-1)
        gate = torch.nn.functional.silu(gate)
        dt = torch.nn.functional.softplus(self.dt_proj(self.dt_up(gated)))
        b_t = self.b_proj(gated)
        c_t = self.c_proj(gated)
        a = -torch.exp(self.a_log)
        state = tokens.new_zeros(batch, dim, self.state_dim)
        outputs = []
        for step in range(length):
            decay = torch.exp(dt[:, step].unsqueeze(-1) * a.unsqueeze(0))
            drive = dt[:, step].unsqueeze(-1) * b_t[:, step].unsqueeze(1) * gated[:, step].unsqueeze(-1)
            state = decay * state + drive
            read = (c_t[:, step].unsqueeze(1) * state).sum(dim=-1)
            outputs.append(read + self.d_skip.unsqueeze(0) * gated[:, step])
        stacked = torch.stack(outputs, dim=1)
        gated_output: Tensor = self.out_proj(stacked * gate)
        return gated_output


class WindowAttentionBlock(nn.Module):
    """Shifted-window self-attention over the appearance tokens."""

    def __init__(self, dim: int, heads: int, window: int, shift: bool = False) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by the head count")
        if window < 1:
            raise ValueError("window must be positive")
        self.dim = dim
        self.heads = heads
        self.window = window
        self.shift = shift
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def _rolled(self, tokens: Tensor) -> Tensor:
        if not self.shift or self.window <= 1:
            return tokens
        return torch.roll(tokens, shifts=self.window // 2, dims=1)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.dim() != 3:
            raise ValueError("expected batch x length x dim")
        batch, length, dim = tokens.shape
        if dim != self.dim:
            raise ValueError("token width disagrees with the block width")
        pad = (-length) % self.window
        padded = torch.nn.functional.pad(self._rolled(tokens), (0, 0, 0, pad))
        span = padded.shape[1]
        windows = padded.reshape(batch * (span // self.window), self.window, dim)
        normalised = self.norm(windows)
        attended, _weights = self.attention(normalised, normalised, normalised, need_weights=False)
        merged = (windows + attended).reshape(batch, span, dim)[:, :length]
        rolled_back = self._rolled(merged)
        mixed: Tensor = self.mlp(rolled_back)
        return tokens + mixed


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch: int) -> None:
        super().__init__()
        if patch < 1:
            raise ValueError("patch must be positive")
        self.patch = patch
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)

    def forward(self, frames: Tensor) -> Tensor:
        if frames.dim() != 4:
            raise ValueError("expected batch x channels x height x width")
        embedded = self.proj(frames)
        batch, dim, rows, columns = embedded.shape
        flattened: Tensor = embedded.reshape(batch, dim, rows * columns).transpose(1, 2)
        return flattened


class SwinUMambaEncoder(nn.Module):
    """Window attention followed by selective scanning at each stage."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        heads: int,
        window: int,
        stages: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError("at least one stage is required")
        self.patch = PatchEmbedding(in_channels, dim, patch=2)
        self.attentions = nn.ModuleList(
            [WindowAttentionBlock(dim, heads, window, shift=bool(stage % 2)) for stage in range(stages)]
        )
        self.scans = nn.ModuleList([SelectiveScanBlock(dim, state_dim) for _ in range(stages)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, frames: Tensor) -> Tensor:
        tokens = self.patch(frames)
        for attention, scan in zip(self.attentions, self.scans):
            attended: Tensor = attention(tokens)
            scanned: Tensor = scan(tokens)
            tokens = attended + scanned
        normalised: Tensor = self.norm(tokens)
        return normalised
