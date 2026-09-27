"""The appearance branch: state-space vessel segmentation over the angiographic frames.

Ref: Sec. 4.5 (the encoder's centreline output is used to achieve geometric consistency).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .attention import PrivatisedTransformerStack
from .ssm import SwinUMambaEncoder


@dataclass(frozen=True)
class AppearanceConfig:
    in_frames: int = 16
    dim: int = 32
    heads: int = 4
    window: int = 4
    stages: int = 2
    state_dim: int = 8
    transformer_depth: int = 2
    latent_dim: int = 256
    patch: int = 2


class AppearanceBranch(nn.Module):
    """Vessel segmentation head plus the privatised attention stack of the generator."""

    def __init__(self, config: AppearanceConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SwinUMambaEncoder(
            in_channels=config.in_frames,
            dim=config.dim,
            heads=config.heads,
            window=config.window,
            stages=config.stages,
            state_dim=config.state_dim,
        )
        self.transformer = PrivatisedTransformerStack(
            depth=config.transformer_depth,
            dim=config.dim,
            heads=config.heads,
            context_dim=config.latent_dim,
        )
        self.centreline = nn.Conv2d(config.dim, 1, kernel_size=1)

    def token_grid(self, height: int, width: int) -> tuple[int, int]:
        return (height // self.config.patch, width // self.config.patch)

    def encode(self, frames: Tensor, latent: Tensor | None = None) -> Tensor:
        """Encoded token stack at the attention width, before the privatised block."""
        if frames.dim() != 4:
            raise ValueError("expected batch x frames x height x width")
        if frames.shape[1] != self.config.in_frames:
            raise ValueError("frame stack disagrees with the appearance configuration")
        encoded = self.encoder(frames)
        decoded: Tensor = self.transformer(encoded, latent)
        return decoded

    def segmentation_head(self, tokens: Tensor, extent: tuple[int, int]) -> Tensor:
        """Centreline logits at the acquired extent, from a token stack of width d."""
        if tokens.dim() != 3:
            raise ValueError("tokens must be batch x length x dimension")
        batch, length, dim = tokens.shape
        rows = int(round(length**0.5))
        if rows * rows != length:
            raise ValueError("token grid must be square for the centreline head")
        grid = tokens.transpose(1, 2).reshape(batch, dim, rows, rows)
        coarse = self.centreline(grid)
        return torch.nn.functional.interpolate(coarse, size=extent, mode="bilinear", align_corners=False)

    def forward(self, frames: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return the segmentation logits at the acquired extent and the encoded tokens."""
        tokens = self.encode(frames, latent)
        extent = (int(frames.shape[2]), int(frames.shape[3]))
        return self.segmentation_head(tokens, extent), tokens

    def centreline_probability(self, frames: Tensor, latent: Tensor | None = None) -> Tensor:
        logits, _tokens = self.forward(frames, latent)
        return torch.sigmoid(logits)

    def segmentation_loss(self, logits: Tensor, target: Tensor, bce_weight: float = 1.0) -> Tensor:
        """Binary cross-entropy plus soft Dice on the vessel centreline.

        Ref: Sec. 2.5 (Dice as the vessel-geometry measure), Sec. 4.10.
        """
        if logits.shape != target.shape:
            raise ValueError("segmentation logits and target must share a shape")
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        probability = torch.sigmoid(logits).reshape(logits.shape[0], -1)
        flat_target = target.reshape(target.shape[0], -1)
        intersection = (probability * flat_target).sum(dim=1)
        denominator = probability.sum(dim=1) + flat_target.sum(dim=1)
        dice = 1.0 - (2.0 * intersection / denominator.clamp_min(torch.finfo(probability.dtype).tiny)).mean()
        return bce_weight * bce + dice

    def geometric_consistency(self, logits: Tensor, endpoints: Tensor) -> Tensor:
        """Centreline mass must concentrate near the declared endpoint.

        Ref: Sec. 4.5.
        """
        if endpoints.dim() != 2 or endpoints.shape[1] != 2:
            raise ValueError("endpoints must be batch x 2 in normalised coordinates")
        probability = torch.sigmoid(logits).squeeze(1)
        _batch, rows, columns = probability.shape
        yy = torch.arange(rows, dtype=probability.dtype, device=probability.device).view(1, rows, 1)
        xx = torch.arange(columns, dtype=probability.dtype, device=probability.device).view(1, 1, columns)
        weight = probability / probability.sum(dim=(1, 2), keepdim=True).clamp_min(torch.finfo(probability.dtype).tiny)
        centre_y = (weight * yy).sum(dim=(1, 2)) / max(1, rows)
        centre_x = (weight * xx).sum(dim=(1, 2)) / max(1, columns)
        predicted = torch.stack([centre_x, centre_y], dim=1)
        return torch.mean(torch.linalg.norm(predicted - endpoints, dim=1))

    def privatised_parameter_count(self) -> int:
        return self.transformer.parameter_count()
