from __future__ import annotations

import jax
import jax.numpy as jnp

from dgr.control.loop import make_mpc_actor, make_proportional_actor, run_episode, zero_actor
from dgr.control.mpc import PlannerConfig, cem
from dgr.control.true_rollout import make_consensus_rollout
from dgr.envs.suites.toy_graph_control.core import reset, step
from dgr.envs.suites.toy_graph_control.scenarios import make_consensus_config


def _consensus_rollout_for(cfg, state):
    return make_consensus_rollout(
        state.senders,
        state.receivers,
        state.node_mask,
        state.edge_mask,
        cfg.actuator_mask,
        cfg.dynamics.alpha,
        cfg.dynamics.beta,
    )


def test_consensus_rollout_matches_noise_free_env_step():
    cfg = make_consensus_config(5, n_max=8, horizon=10, noise_std=0.0)
    state, obs = reset(jax.random.PRNGKey(0), cfg)
    action = jnp.where(
        state.node_mask,
        jax.random.uniform(jax.random.PRNGKey(1), (8,), minval=-1.0, maxval=1.0),
        0.0,
    )
    rollout = _consensus_rollout_for(cfg, state)

    x_traj = rollout(obs.nodes, action[None, :])  # one-step plan
    next_state, _, _, _ = step(jax.random.PRNGKey(2), cfg, state, action)

    assert jnp.allclose(x_traj[0], next_state.x, atol=1e-5)


def test_true_mpc_beats_zero_action():
    cfg = make_consensus_config(5, n_max=8, horizon=20, noise_std=0.0)
    structure_state, _ = reset(jax.random.PRNGKey(0), cfg)
    planner_config = PlannerConfig(horizon=5, action_dim=8, population=256)
    action_mask = structure_state.node_mask & cfg.actuator_mask

    true_actor = make_mpc_actor(
        _consensus_rollout_for(cfg, structure_state),
        cem,
        planner_config,
        action_mask,
        structure_state.node_mask,
    )
    true_res = run_episode(jax.random.PRNGKey(1), cfg, true_actor)
    zero_res = run_episode(jax.random.PRNGKey(1), cfg, zero_actor())

    # Perfect-model MPC on an easy control task must beat doing nothing.
    assert float(true_res.episode_return) > float(zero_res.episode_return)
    assert float(true_res.final_mse) < float(zero_res.final_mse)


def test_true_mpc_is_competitive_with_proportional():
    cfg = make_consensus_config(6, n_max=8, horizon=20, noise_std=0.0)
    structure_state, _ = reset(jax.random.PRNGKey(0), cfg)
    planner_config = PlannerConfig(horizon=8, action_dim=8, population=512)
    action_mask = structure_state.node_mask & cfg.actuator_mask

    true_actor = make_mpc_actor(
        _consensus_rollout_for(cfg, structure_state),
        cem,
        planner_config,
        action_mask,
        structure_state.node_mask,
    )
    prop_actor = make_proportional_actor(cfg.actuator_mask)

    true_res = run_episode(jax.random.PRNGKey(3), cfg, true_actor)
    prop_res = run_episode(jax.random.PRNGKey(3), cfg, prop_actor)

    # With a perfect model, MPC should at least roughly match the hand-tuned controller
    # (returns are negative; "competitive" = not much worse than proportional).
    margin = 0.25 * abs(float(prop_res.episode_return))
    assert float(true_res.episode_return) > float(prop_res.episode_return) - margin
