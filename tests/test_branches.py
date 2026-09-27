"""Branch, pairing, adapter and decoder tests.

Ref: Sec. 4.4, Sec. 4.5, Algorithm 1 steps 5-10.
"""

from __future__ import annotations

import pytest
import torch

from private_vascular_motion_generation.branches.adapter import AdapterSpec, PrivateCouplingAdapter
from private_vascular_motion_generation.branches.appearance import AppearanceBranch, AppearanceConfig
from private_vascular_motion_generation.branches.attention import PrivateAttentionBlock, PrivatisedTransformerStack
from private_vascular_motion_generation.branches.decoder import JointDecoderConfig, JointProcedureDecoder
from private_vascular_motion_generation.branches.motion import FlowMatchingActionHead, SinusoidalTimeEmbedding
from private_vascular_motion_generation.branches.pairing import PairingSpec, ProcedureBatch
from private_vascular_motion_generation.branches.ssm import SelectiveScanBlock, SwinUMambaEncoder, WindowAttentionBlock
from private_vascular_motion_generation.cohort.procedure import MOTION_DIM
from private_vascular_motion_generation.geometry.projector import projector_from_directions

FRAMES = 16
EXTENT = 32
WIDTH = 32
LATENT = 24
MICRO_STEPS = 3


@pytest.fixture()
def batch() -> ProcedureBatch:
    generator = torch.Generator().manual_seed(21)
    frames = torch.rand(2, FRAMES, EXTENT, EXTENT, generator=generator)
    mask = (frames.mean(dim=1, keepdim=True) > 0.5).to(torch.float32)
    motion = torch.randn(2, MOTION_DIM, FRAMES, generator=generator)
    endpoints = torch.rand(2, 2, generator=generator)
    codes = torch.randn(2, LATENT, generator=generator)
    codes = codes / codes.norm(dim=1, keepdim=True)
    return ProcedureBatch(frames=frames, mask=mask, motion=motion, endpoints=endpoints, latent_codes=codes)


def _pair(smoke_session) -> object:
    return smoke_session.pair


def test_adapter_gates_with_the_window_assignment() -> None:
    adapter = PrivateCouplingAdapter(AdapterSpec(dimension=WIDTH, micro_steps=MICRO_STEPS))
    tokens = torch.ones(2, 6, WIDTH)
    gated = adapter.modulate(tokens)
    assert torch.allclose(gated, tokens, atol=1e-6)
    assignment = adapter.window_assignment(6)
    assert assignment.tolist() == [0, 1, 2, 0, 1, 2]
    assert adapter.parameter_block().shape == (WIDTH, MICRO_STEPS)


def test_adapter_requires_matching_width() -> None:
    adapter = PrivateCouplingAdapter(AdapterSpec(dimension=8, micro_steps=2))
    with pytest.raises(ValueError):
        adapter.modulate(torch.ones(1, 4, 4))


def test_selective_scan_preserves_the_token_grid() -> None:
    block = SelectiveScanBlock(dim=8, state_dim=4)
    tokens = torch.randn(2, 5, 8)
    assert tuple(block(tokens).shape) == (2, 5, 8)


def test_window_attention_is_residual_and_shape_preserving() -> None:
    block = WindowAttentionBlock(dim=8, heads=2, window=2)
    tokens = torch.randn(2, 7, 8)
    assert tuple(block(tokens).shape) == (2, 7, 8)


def test_swin_umamba_encoder_embeds_to_tokens() -> None:
    encoder = SwinUMambaEncoder(in_channels=FRAMES, dim=WIDTH, heads=4, window=4, stages=1, state_dim=8)
    tokens = encoder(torch.rand(2, FRAMES, EXTENT, EXTENT))
    assert tuple(tokens.shape) == (2, (EXTENT // 2) ** 2, WIDTH)


def test_appearance_branch_returns_logits_at_the_acquired_extent() -> None:
    branch = AppearanceBranch(
        AppearanceConfig(
            in_frames=FRAMES,
            dim=WIDTH,
            heads=4,
            window=4,
            stages=1,
            state_dim=8,
            transformer_depth=1,
            latent_dim=LATENT,
            patch=2,
        )
    )
    logits, tokens = branch(torch.rand(2, FRAMES, EXTENT, EXTENT), torch.randn(2, LATENT))
    assert tuple(logits.shape) == (2, 1, EXTENT, EXTENT)
    assert tuple(tokens.shape) == (2, (EXTENT // 2) ** 2, WIDTH)


def test_attention_stack_keeps_the_privatised_block() -> None:
    block = PrivateAttentionBlock(dim=WIDTH, heads=4, context_dim=LATENT)
    out = block(torch.randn(2, 5, WIDTH), torch.randn(2, LATENT))
    assert tuple(out.shape) == (2, 5, WIDTH)
    stack = PrivatisedTransformerStack(depth=2, dim=WIDTH, heads=4, context_dim=LATENT)
    assert stack.parameter_count() > 0
    assert stack.noise_dimension() == WIDTH


def test_time_embedding_is_even_and_bounded() -> None:
    embedding = SinusoidalTimeEmbedding(8)
    values = embedding(torch.tensor([0.0, 0.5, 1.0]))
    assert tuple(values.shape) == (3, 8)
    assert float(values.abs().max().item()) <= 1.0


def test_flow_matching_head_shape_and_loss() -> None:
    head = FlowMatchingActionHead(frames=FRAMES, width=WIDTH, latent_dim=LATENT, depth=1)
    target = torch.randn(3, MOTION_DIM, FRAMES)
    codes = torch.randn(3, LATENT)
    loss = head.flow_matching_loss(target, codes, torch.Generator().manual_seed(22))
    assert float(loss.item()) > 0.0
    assert tuple(head.integrate(codes, steps=2).shape) == (3, MOTION_DIM, FRAMES)


def test_branch_pair_workload_shapes(smoke_session, batch) -> None:
    pair = _pair(smoke_session)
    pair.pairing = PairingSpec(shared_latent=True, coupling_head=True, window_micro_steps=MICRO_STEPS)
    workload = pair.per_procedure_workload(batch, torch.Generator().manual_seed(71))
    aggregated = pair.aggregate_workload(batch, torch.Generator().manual_seed(71))
    assert tuple(workload.shape) == (2, WIDTH, MICRO_STEPS)
    assert tuple(aggregated.shape) == (WIDTH, MICRO_STEPS)
    assert torch.allclose(aggregated, workload.sum(dim=0), atol=1e-6)


def test_branch_pair_gradient_views_are_paired(smoke_session, batch) -> None:
    appearance, motion = _pair(smoke_session).branch_gradient_views(batch)
    assert tuple(appearance.shape) == (2, WIDTH)
    assert tuple(motion.shape) == (2, WIDTH)
    assert float(appearance.abs().sum().item()) > 0.0
    assert float(motion.abs().sum().item()) > 0.0


def test_decoupled_pairing_zeroes_the_shared_latent(smoke_session, batch) -> None:
    pair = _pair(smoke_session)
    pair.pairing = PairingSpec(shared_latent=False, coupling_head=True, window_micro_steps=MICRO_STEPS)
    assert pair._codes(batch).abs().sum().item() == 0.0
    pair.pairing = PairingSpec(shared_latent=True, coupling_head=True, window_micro_steps=MICRO_STEPS)


def test_joint_decoder_uses_the_coupling_projector(torch_generator) -> None:
    basis = torch.linalg.qr(torch.randn(WIDTH, 3, generator=torch_generator))[0]
    projector = projector_from_directions(basis, 3)
    decoder = JointProcedureDecoder(
        JointDecoderConfig(latent_dim=WIDTH, appearance_extent=(EXTENT, EXTENT), frames=FRAMES, hidden=16)
    )
    codes = torch.randn(2, WIDTH, generator=torch_generator)
    appearance, motion = decoder(codes, projector)
    assert tuple(appearance.shape) == (2, FRAMES, EXTENT, EXTENT)
    assert tuple(motion.shape) == (2, MOTION_DIM, FRAMES)
    assert decoder.conditioning_dependence(codes, projector) > 0.0


def test_batch_rejects_mismatched_masks(batch) -> None:
    with pytest.raises(ValueError):
        ProcedureBatch(
            frames=batch.frames,
            mask=batch.mask[:1],
            motion=batch.motion,
            endpoints=batch.endpoints,
            latent_codes=batch.latent_codes,
        )
