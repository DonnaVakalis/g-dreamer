from __future__ import annotations

import jax
import jax.numpy as jnp

from dgr.control.wm_rollout import make_wm_rollout
from dgr.envs.suites.toy_graph_control.core import reset
from dgr.envs.suites.toy_graph_control.scenarios import make_consensus_config
from dgr.models.world_models import graph_rssm_wm


def _setup(horizon: int = 6, n_real: int = 5, n_max: int = 8):
    cfg = make_consensus_config(n_real, n_max=n_max, horizon=20)
    state, obs = reset(jax.random.PRNGKey(0), cfg)
    params = graph_rssm_wm.init_params(
        jax.random.PRNGKey(1), graph_rssm_wm.GraphRSSMConfig(node_dim=2)
    )
    action_seq = jax.random.uniform(
        jax.random.PRNGKey(2), (horizon, n_max), minval=-1.0, maxval=1.0
    )
    action_seq = jnp.where(state.node_mask, action_seq, 0.0)
    return state, obs, params, action_seq


def test_rollout_matches_manual_open_loop():
    state, obs, params, action_seq = _setup(horizon=6)
    rollout = make_wm_rollout(
        graph_rssm_wm.predict_next_nodes_single,
        params,
        state.senders,
        state.receivers,
        state.node_mask,
        state.edge_mask,
    )
    x_traj = rollout(obs.nodes, action_seq)

    # Manual open-loop: re-inject the static goal channel, feed prediction back.
    goal = obs.nodes[:, 1]
    nodes = obs.nodes
    manual = []
    for t in range(action_seq.shape[0]):
        pred = graph_rssm_wm.predict_next_nodes_single(
            params,
            nodes,
            action_seq[t],
            state.senders,
            state.receivers,
            state.node_mask,
            state.edge_mask,
        )
        x_next = jnp.where(state.node_mask, pred[:, 0], 0.0)
        nodes = jnp.stack([x_next, goal], axis=-1)
        manual.append(x_next)

    assert x_traj.shape == (action_seq.shape[0], state.node_mask.shape[0])
    assert jnp.allclose(x_traj, jnp.stack(manual), atol=1e-5)


def test_rollout_first_step_matches_one_step_predict():
    state, obs, params, action_seq = _setup(horizon=4)
    rollout = make_wm_rollout(
        graph_rssm_wm.predict_next_nodes_single,
        params,
        state.senders,
        state.receivers,
        state.node_mask,
        state.edge_mask,
    )
    x_traj = rollout(obs.nodes, action_seq)
    one_step = graph_rssm_wm.predict_next_nodes_single(
        params,
        obs.nodes,
        action_seq[0],
        state.senders,
        state.receivers,
        state.node_mask,
        state.edge_mask,
    )
    assert jnp.allclose(x_traj[0], jnp.where(state.node_mask, one_step[:, 0], 0.0), atol=1e-5)


def test_rollout_respects_node_mask_on_padding():
    state, obs, params, action_seq = _setup(horizon=5, n_real=5, n_max=8)
    rollout = make_wm_rollout(
        graph_rssm_wm.predict_next_nodes_single,
        params,
        state.senders,
        state.receivers,
        state.node_mask,
        state.edge_mask,
    )
    x_traj = rollout(obs.nodes, action_seq)
    padding = x_traj[:, ~state.node_mask]
    assert jnp.allclose(padding, 0.0, atol=1e-6)
