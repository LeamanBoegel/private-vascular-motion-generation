"""Appearance and motion branches of the paired generator.

Ref: Sec. 4.4, Sec. 4.5.
"""

from __future__ import annotations

from .adapter import AdapterSpec, PrivateCouplingAdapter
from .appearance import AppearanceBranch, AppearanceConfig
from .attention import PrivateAttentionBlock, PrivatisedTransformerStack
from .decoder import JointDecoderConfig, JointProcedureDecoder
from .motion import FlowMatchingActionHead, SinusoidalTimeEmbedding, force_channel_from_motion
from .pairing import BranchPair, PairingSpec, ProcedureBatch, concatenated_residual
from .ssm import PatchEmbedding, SelectiveScanBlock, SwinUMambaEncoder, WindowAttentionBlock

__all__ = [
    "AdapterSpec",
    "AppearanceBranch",
    "AppearanceConfig",
    "BranchPair",
    "FlowMatchingActionHead",
    "JointDecoderConfig",
    "JointProcedureDecoder",
    "PairingSpec",
    "PatchEmbedding",
    "PrivateAttentionBlock",
    "PrivateCouplingAdapter",
    "PrivatisedTransformerStack",
    "ProcedureBatch",
    "SelectiveScanBlock",
    "SinusoidalTimeEmbedding",
    "SwinUMambaEncoder",
    "WindowAttentionBlock",
    "concatenated_residual",
    "force_channel_from_motion",
]
