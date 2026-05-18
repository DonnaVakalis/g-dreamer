from __future__ import annotations

import jax
import jax.numpy as jnp

from dgr.envs.suites.toy_graph_control.core import reset, step
from dgr.envs.suites.toy_graph_control.scenarios import get_scenario, make_consensus_config
from dgr.envs.suites.toy_graph_control.topologies import make_ring_topology

_TOPOLOGIES = ("ring", "grid", "kregular")


def _rollout(cfg, steps: int = 5):
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    state, obs = reset(reset_key, cfg)
    rewards = []
    for _ in range(steps):
        key, action_key, step_key = jax.random.split(key, 3)
        action = jnp.where(
            state.node_mask,
            jax.random.uniform(action_key, state.x.shape, minval=-1.0, maxval=1.0),
            0.0,
        )
        state, obs, reward, _ = step(step_key, cfg, state, action)
        rewards.append(float(reward))
    return state, obs, rewards


def test_each_topology_resets_steps_and_observes():
    # observe() runs validate_graph internally — reaching the asserts means it passed.
    for topology in _TOPOLOGIES:
        cfg = make_consensus_config(16, n_max=16, horizon=10, topology=topology)
        state, obs, rewards = _rollout(cfg, steps=5)
        assert obs.nodes.shape == (16, 2)
        assert jnp.all(jnp.isfinite(obs.nodes))
        assert all(r <= 0.0 for r in rewards)  # reward = -MSE-to-goal
        assert int(jnp.sum(state.edge_mask)) > 0


def test_topologies_have_distinct_edge_counts():
    counts = {}
    for topology in _TOPOLOGIES:
        cfg = make_consensus_config(16, n_max=16, topology=topology)
        state, _ = reset(jax.random.PRNGKey(0), cfg)
        counts[topology] = int(jnp.sum(state.edge_mask))
    # ring: 2n; grid 4x4: 2*(4*3 + 4*3); k-regular k=4: 4n.
    assert counts == {"ring": 32, "grid": 48, "kregular": 64}


def test_ring_through_new_config_path_matches_legacy_generator():
    cfg = make_consensus_config(12, n_max=16, topology="ring")
    state, _ = reset(jax.random.PRNGKey(0), cfg)
    senders, receivers, edge_mask = make_ring_topology(cfg.spec, 12)
    assert jnp.array_equal(state.senders, senders)
    assert jnp.array_equal(state.receivers, receivers)
    assert jnp.array_equal(state.edge_mask, edge_mask)


def test_legacy_scenario_config_falls_back_to_ring():
    # get_scenario() builds configs directly, without a topology — reset must still work.
    cfg = get_scenario("debug_ring_dense")
    assert cfg.topology is None
    state, _ = reset(jax.random.PRNGKey(0), cfg)
    assert int(jnp.sum(state.edge_mask)) == 2 * cfg.n_real


def test_kregular_topology_is_reproducible_across_config_builds():
    a, _ = reset(jax.random.PRNGKey(0), make_consensus_config(16, n_max=16, topology="kregular"))
    b, _ = reset(jax.random.PRNGKey(0), make_consensus_config(16, n_max=16, topology="kregular"))
    assert jnp.array_equal(a.senders, b.senders)
    assert jnp.array_equal(a.receivers, b.receivers)
