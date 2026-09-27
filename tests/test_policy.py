"""Policy, environment and rollout tests.

Ref: Sec. 4.8, Sec. 2.6, Table 5.
"""

from __future__ import annotations

import pytest
import torch

from private_vascular_motion_generation.cohort import fixture as fixture_module
from private_vascular_motion_generation.policy.classical import (
    barrier_value,
    control_barrier_filter,
    potential_field_action,
    potential_field_rollout,
)
from private_vascular_motion_generation.policy.diffusion_policy import (
    DiffusionPolicy,
    DiffusionPolicyConfig,
    NoiseSchedule,
)
from private_vascular_motion_generation.policy.environment import (
    EndovascularEnvironment,
    EnvironmentSpec,
    tube_from_centreline,
)
from private_vascular_motion_generation.policy.rollout import (
    aggregate_strategy,
    environment_for,
    obstacles_from_centreline,
    published_comparator_rows,
    rollout_path_follower,
    rollout_potential_field,
)
from private_vascular_motion_generation.policy.sac import (
    ReplayBuffer,
    SACConfig,
    SoftActorCritic,
    rollout_sac,
)
from private_vascular_motion_generation.policy.world_model import (
    LatentPlanner,
    LatentWorldModel,
    WorldModelConfig,
    latent_consistency_report,
)


@pytest.fixture()
def environment() -> EndovascularEnvironment:
    fixture = fixture_module.default_fixture(seed=61, per_site={"A": 2})
    return environment_for(fixture.records[0])


def test_barrier_value_sign() -> None:
    assert float(barrier_value(torch.tensor([0.2])).item()) > 0.0
    assert float(barrier_value(torch.tensor([0.0])).item()) < 0.0


def test_barrier_filter_intervenes_near_the_wall() -> None:
    filtered = control_barrier_filter(
        torch.tensor([[0.3, 0.3, 0.3]]), torch.tensor([0.001]), torch.tensor([[1.0, 0.0, 0.0]])
    )
    assert filtered.intervened
    assert float(filtered.action.norm().item()) < float(torch.tensor([0.3, 0.3, 0.3]).norm().item())


def test_barrier_filter_is_inactive_away_from_the_wall() -> None:
    filtered = control_barrier_filter(
        torch.tensor([[0.2, 0.0, 0.0]]), torch.tensor([0.4]), torch.tensor([[1.0, 0.0, 0.0]])
    )
    assert not filtered.intervened


def test_potential_field_combines_attraction_and_repulsion() -> None:
    free = potential_field_action(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]]), torch.zeros(0, 3))
    obstructed = potential_field_action(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.05, 0.0, 0.0]]),
    )
    assert float(obstructed[0, 0].item()) < float(free[0, 0].item())


def test_potential_field_rollout_shape() -> None:
    start = torch.zeros(1, 3)
    target = torch.ones(1, 3)
    obstacles = torch.zeros(0, 3)
    trajectory = potential_field_rollout(start, target, obstacles, steps=5)
    assert tuple(trajectory.shape) == (6, 3)


def test_environment_records_contacts_and_resets(environment: EndovascularEnvironment) -> None:
    environment.reset()
    for _ in range(6):
        environment.step(torch.tensor([0.05, 0.0, 0.0]))
    assert environment.progress > 0
    assert environment.travelled > 0.0
    assert environment.force_trace().numel() > 0
    environment.reset()
    assert environment.progress == 0
    assert torch.allclose(environment.tip, environment.centreline[0])


def test_environment_rejects_bad_actions(environment: EndovascularEnvironment) -> None:
    environment.reset()
    with pytest.raises(ValueError):
        environment.step(torch.zeros(2))


def test_tube_narrows_distally() -> None:
    fixture = fixture_module.default_fixture(seed=62, per_site={"A": 1})
    centreline = fixture.records[0].tip_trajectory()
    radius = tube_from_centreline(centreline)
    assert float(radius[-1].item()) < float(radius[0].item())


def test_rollout_strategies_produce_finite_traces(environment: EndovascularEnvironment) -> None:
    nominal = rollout_path_follower(environment, 20, use_barrier=False)
    filtered = rollout_path_follower(environment, 20, use_barrier=True)
    assert nominal[0].numel() > 0
    assert float(filtered[0].max().item()) <= float(nominal[0].max().item()) + 1e-9


def test_potential_field_rollout_on_environment(environment: EndovascularEnvironment) -> None:
    obstacles = obstacles_from_centreline(environment.centreline)
    forces, success, travelled = rollout_potential_field(environment, obstacles, 20)
    assert forces.numel() > 0
    assert success.numel() == 1
    assert travelled >= 0.0


def test_aggregate_strategy_builds_a_row(environment: EndovascularEnvironment) -> None:
    outcomes = [
        rollout_path_follower(environment, 18, use_barrier=False),
        rollout_path_follower(environment, 18, use_barrier=True),
    ]
    row, summary = aggregate_strategy(
        label="fixture",
        outcomes=outcomes,
        centreline=environment.centreline,
        fluoroscopy_min=12.0,
        contrast_ml=140.0,
        navigation_min=40.0,
    )
    assert row.label == "fixture"
    assert summary.centreline_length > 0.0
    assert row.path_efficiency > 0.0


def test_published_comparators_report_success_only() -> None:
    rows = published_comparator_rows()
    assert len(rows) == 3
    assert all(row["endpoints_reported"] == "task success only" for row in rows)


def test_soft_actor_critic_takes_a_finite_update() -> None:
    agent = SoftActorCritic(SACConfig(state_dim=4, action_dim=3, hidden=16), seed=7)
    buffer = ReplayBuffer(capacity=192, state_dim=4, action_dim=3)
    generator = torch.Generator().manual_seed(63)
    for _ in range(80):
        buffer.push(
            torch.randn(4, generator=generator),
            torch.randn(3, generator=generator) * 0.1,
            float(torch.rand(1, generator=generator).item()),
            torch.randn(4, generator=generator),
            False,
        )
    diagnostics = agent.update(buffer.sample(24, generator))
    assert diagnostics["critic_loss"] > 0.0
    assert diagnostics["alpha"] > 0.0


def test_replay_buffer_rejects_early_sampling() -> None:
    buffer = ReplayBuffer(capacity=8, state_dim=3, action_dim=2)
    with pytest.raises(ValueError):
        buffer.sample(4)


def test_sac_rollout_on_the_environment(environment: EndovascularEnvironment) -> None:
    agent = SoftActorCritic(SACConfig(state_dim=4, action_dim=3, hidden=16), seed=8)
    forces, success = rollout_sac(agent, environment, 12)
    assert forces.numel() > 0
    assert success.numel() == 1


def test_latent_world_model_consistency_and_planning() -> None:
    model = LatentWorldModel(WorldModelConfig(state_dim=4, action_dim=3, hidden=16, latent_dim=6))
    planner = LatentPlanner(model, seed=9)
    generator = torch.Generator().manual_seed(64)
    states = torch.randn(32, 4, generator=generator)
    actions = torch.randn(32, 3, generator=generator)
    next_states = torch.randn(32, 4, generator=generator)
    report = latent_consistency_report(model, states, actions, next_states)
    assert report["mean_consistency_error"] >= 0.0
    loss = planner.update(
        {
            "states": states,
            "actions": actions,
            "rewards": torch.randn(32, generator=generator),
            "next_states": next_states,
            "terminal_value": torch.randn(32, generator=generator),
        }
    )
    assert loss > 0.0
    plan = planner.plan(torch.randn(4, generator=generator))
    assert tuple(plan.shape) == (3,)
    assert float(plan.abs().max().item()) <= 1.0


def test_noise_schedule_monotone() -> None:
    schedule = NoiseSchedule(8)
    assert bool(torch.all(schedule.alpha_bar[1:] <= schedule.alpha_bar[:-1]).item())
    with pytest.raises(ValueError):
        NoiseSchedule(1)


def test_diffusion_policy_loss_and_sampling(torch_generator) -> None:
    policy = DiffusionPolicy(DiffusionPolicyConfig(state_dim=4, horizon=3, width=16, depth=1, diffusion_steps=3))
    states = torch.randn(6, 2, 4, generator=torch_generator)
    actions = torch.tanh(torch.randn(6, 3, 6, generator=torch_generator))
    loss = policy.training_loss(states, actions, torch_generator)
    assert float(loss.item()) > 0.0
    sampled = policy.sample(states, torch_generator)
    assert tuple(sampled.shape) == (6, 3, 6)
    assert policy.action_dim() == 6


def test_environment_spec_validation() -> None:
    with pytest.raises(ValueError):
        EnvironmentSpec(frames=1)
    with pytest.raises(ValueError):
        EnvironmentSpec(frames=8, step_size=0.0)
