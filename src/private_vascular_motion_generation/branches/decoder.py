"""Joint decoding of the procedure record from the released code.

Ref: Sec. 4.5 (a decoder capable of joint decoding of both the image and the motion),
Algorithm 1 step 16 (pure post-processing).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..cohort.procedure import MOTION_DIM
from ..geometry.projector import CouplingProjector


@dataclass(frozen=True)
class JointDecoderConfig:
    latent_dim: int
    appearance_extent: tuple[int, int]
    frames: int
    hidden: int = 64


class JointProcedureDecoder(nn.Module):
    """Conditioned on the coupling-projected code rather than on either marginal."""

    def __init__(self, config: JointDecoderConfig) -> None:
        super().__init__()
        self.config = config
        height, width = config.appearance_extent
        if min(height, width, config.frames, config.hidden) <= 0:
            raise ValueError("decoder extents must be positive")
        self.expand = nn.Linear(config.latent_dim, config.hidden)
        self.appearance_upsample = nn.Sequential(
            nn.Linear(config.hidden, config.hidden * 2),
            nn.SiLU(),
            nn.Linear(config.hidden * 2, config.frames * height * width),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(config.hidden + config.latent_dim, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, MOTION_DIM * config.frames),
        )

    def forward(self, code: Tensor, projector: CouplingProjector) -> tuple[Tensor, Tensor]:
        """Decode the appearance tensor and the six-channel motion tensor.

        Ref: Eq. (2), Algorithm 1 step 16.
        """
        if code.dim() != 2 or code.shape[1] != self.config.latent_dim:
            raise ValueError("code must be batch x latent_dim")
        if projector.dimension != self.config.latent_dim:
            raise ValueError("projector dimension disagrees with the decoder latent width")
        projected = projector.apply_batch(code)
        hidden = self.expand(projected)
        height, width = self.config.appearance_extent
        appearance = self.appearance_upsample(hidden).reshape(-1, self.config.frames, height, width)
        motion = self.motion_head(torch.cat([hidden, projected], dim=1)).reshape(-1, MOTION_DIM, self.config.frames)
        return torch.sigmoid(appearance), motion

    def decode_shaped_only(self, code: Tensor, projector: CouplingProjector) -> Tensor:
        """Code restricted to the coupling subspace, used by the conditioning-off arm."""
        if projector.dimension != self.config.latent_dim:
            raise ValueError("projector dimension disagrees with the decoder latent width")
        expanded: Tensor = self.expand(projector.apply_batch(code))
        return expanded

    def conditioning_dependence(self, code: Tensor, projector: CouplingProjector) -> float:
        """Fraction of decoded appearance energy attributable to the shared coupling code."""
        if projector.dimension != self.config.latent_dim:
            raise ValueError("projector dimension disagrees with the decoder latent width")
        with_code = self.decode_shaped_only(code, projector)
        without = self.decode_shaped_only(torch.zeros_like(code), projector)
        numerator = torch.linalg.norm(with_code - without).item()
        denominator = max(torch.linalg.norm(without).item(), torch.finfo(torch.float32).tiny)
        return float(numerator / denominator)
