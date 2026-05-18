"""The consensus env dynamics expressed as a planning model (chunk 2d).

This is the upper-bound baseline: an MPC actor that plans with the *exact* (noise-free)
env dynamics isolates planner quality from world-model error. Mirrors ``_step_consensus``
in the env core, minus the process noise. Topology-agnostic — graph structure enters only
as senders / receivers / masks.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from dgr.control.wm_rollout import RolloutFn


def make_consensus_rollout(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    node_mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
    actuator_mask: jnp.ndarray,
    alpha: float,
    beta: float,
) -> RolloutFn:
    """Build an open-loop rollout using the true consensus dynamics on one fixed graph."""
    n_max = node_mask.shape[0]
    node_mask_b = node_mask.astype(jnp.bool_)
    edge_mask_f = edge_mask.astype(jnp.float32)
    actuator_mask_f = actuator_mask.astype(jnp.float32) * node_mask_b.astype(jnp.float32)

    def consensus_step(x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        u = action.astype(jnp.float32) * actuator_mask_f
        msgs = x[senders] * edge_mask_f
        agg = jnp.zeros((n_max,), dtype=jnp.float32).at[receivers].add(msgs)
        deg = jnp.zeros((n_max,), dtype=jnp.float32).at[receivers].add(edge_mask_f)
        neigh_mean = agg / jnp.maximum(deg, 1.0)
        x_next = x + alpha * (neigh_mean - x) + beta * u
        return jnp.where(node_mask_b, x_next, 0.0)

    def rollout(nodes0: jnp.ndarray, action_seq: jnp.ndarray) -> jnp.ndarray:
        def step(x: jnp.ndarray, action: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            x_next = consensus_step(x, action)
            return x_next, x_next

        _, x_traj = lax.scan(step, nodes0[:, 0], action_seq)
        return x_traj

    return rollout
