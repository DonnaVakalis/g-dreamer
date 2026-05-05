from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dgr.agents.graph_dreamerv3.data import collect_random_transitions
from dgr.agents.graph_dreamerv3.minimal_world_model import (
    MinimalWorldModelConfig,
    batch_loss,
    init_params,
    predict_next_nodes_single,
)
from dgr.envs.suites.toy_graph_control.core import reset
from dgr.envs.suites.toy_graph_control.scenarios import make_consensus_config


def test_collect_random_transitions_supports_multiple_sizes():
    dataset = collect_random_transitions(
        sizes=[4, 6],
        episodes_per_size=1,
        n_max=6,
        horizon=3,
        noise_std=0.0,
        seed=0,
    )

    assert dataset.size == 6
    assert dataset.nodes.shape == (6, 6, 2)
    assert dataset.next_nodes.shape == (6, 6, 2)
    assert dataset.actions.shape == (6, 6)
    assert sorted(np.unique(dataset.n_real).tolist()) == [4, 6]
    assert set(np.sum(dataset.node_mask, axis=1).tolist()) == {4, 6}


def test_minimal_world_model_ignores_padded_node_features():
    cfg = make_consensus_config(4, n_max=6, horizon=3, noise_std=0.0)
    _, obs = reset(jax.random.PRNGKey(0), cfg)

    params = init_params(jax.random.PRNGKey(1), MinimalWorldModelConfig(node_dim=2))
    action = jnp.array([0.2, -0.1, 0.4, 0.0, 0.0, 0.0], dtype=jnp.float32)

    pred = predict_next_nodes_single(
        params,
        obs.nodes,
        action,
        obs.senders,
        obs.receivers,
        obs.node_mask,
        obs.edge_mask,
    )

    padded = ~obs.node_mask
    noisy_nodes = jnp.where(
        padded[:, None],
        jnp.array([[7.0, -3.0]], dtype=jnp.float32),
        obs.nodes,
    )
    pred_noisy = predict_next_nodes_single(
        params,
        noisy_nodes,
        action,
        obs.senders,
        obs.receivers,
        obs.node_mask,
        obs.edge_mask,
    )

    assert jnp.allclose(pred[: cfg.n_real], pred_noisy[: cfg.n_real], atol=1e-6)


def test_batch_loss_returns_finite_metrics():
    dataset = collect_random_transitions(
        sizes=[4],
        episodes_per_size=1,
        n_max=4,
        horizon=2,
        noise_std=0.0,
        seed=0,
    )
    params = init_params(jax.random.PRNGKey(3), MinimalWorldModelConfig(node_dim=2))
    batch = {
        "nodes": jnp.asarray(dataset.nodes),
        "actions": jnp.asarray(dataset.actions),
        "next_nodes": jnp.asarray(dataset.next_nodes),
        "senders": jnp.asarray(dataset.senders),
        "receivers": jnp.asarray(dataset.receivers),
        "node_mask": jnp.asarray(dataset.node_mask),
        "edge_mask": jnp.asarray(dataset.edge_mask),
    }

    loss, metrics = batch_loss(params, batch)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["loss"])
    assert jnp.isfinite(metrics["x_mse"])
