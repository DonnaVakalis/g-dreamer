"""
Reusable hand-coded baselines
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def zero_action(node_mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(node_mask, 0.0, 0.0).astype(jnp.float32)


def random_action(key: jax.Array, node_mask: jnp.ndarray, scale: float = 0.5) -> jnp.ndarray:
    u = scale * jax.random.uniform(key, shape=node_mask.shape, minval=-1.0, maxval=1.0)
    return jnp.where(node_mask, u, 0.0).astype(jnp.float32)


def proportional_action(
    x: jnp.ndarray,
    goal: jnp.ndarray,
    node_mask: jnp.ndarray,
    actuator_mask: jnp.ndarray,
    k: float = 0.5,
) -> jnp.ndarray:
    u = k * (goal - x)
    mask = node_mask & actuator_mask
    return jnp.where(mask, u, 0.0).astype(jnp.float32)


def masked_proportional_action(
    x: jnp.ndarray,
    goal: jnp.ndarray,
    node_mask: jnp.ndarray,
    actuator_mask: jnp.ndarray,
    goal_obs_mask: jnp.ndarray,
    k: float = 0.5,
) -> jnp.ndarray:
    # Only uses goal where it is visible.
    visible_goal = goal * goal_obs_mask.astype(jnp.float32)
    u = k * (visible_goal - x)
    mask = node_mask & actuator_mask & goal_obs_mask
    return jnp.where(mask, u, 0.0).astype(jnp.float32)


def mse_to_goal(x: jnp.ndarray, goal: jnp.ndarray, node_mask: jnp.ndarray) -> jnp.ndarray:
    mask_f = node_mask.astype(jnp.float32)
    err = (x - goal) * mask_f
    return jnp.sum(err * err) / jnp.maximum(jnp.sum(mask_f), 1.0)


# helper when debugging inferred goal proportional action
def mse_on_mask(x, goal, mask) -> jnp.ndarray:
    m = mask.astype(jnp.float32)
    err = (x - goal) * m
    return jnp.sum(err * err) / jnp.maximum(jnp.sum(m), 1.0)


def _neighbor_mean(values, senders, receivers, edge_mask):
    n_max = values.shape[0]
    edge_mask_f = edge_mask.astype(jnp.float32)
    msgs = values[senders] * edge_mask_f
    agg = jnp.zeros((n_max,), dtype=jnp.float32).at[receivers].add(msgs)
    deg = jnp.zeros((n_max,), dtype=jnp.float32).at[receivers].add(edge_mask_f)
    return agg / jnp.maximum(deg, 1.0)


def inferred_goal_proportional_action(
    x: jnp.ndarray,
    visible_goal: jnp.ndarray,
    node_mask: jnp.ndarray,
    actuator_mask: jnp.ndarray,
    goal_obs_mask: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    edge_mask: jnp.ndarray,
    *,
    iters: int = 8,
    k: float = 0.5,
) -> jnp.ndarray:
    """
    Infer a full goal field by diffusing the visible goals across the graph,
    clamping observed nodes each iteration.
    """
    nm = node_mask.astype(jnp.float32)
    obs = goal_obs_mask.astype(jnp.float32)

    # Start from visible values; unknown nodes start at 0.
    g_hat = visible_goal * obs * nm

    for _ in range(int(iters)):
        g_hat = _neighbor_mean(g_hat, senders, receivers, edge_mask)
        # Clamp observed nodes back to the known values.
        g_hat = jnp.where(goal_obs_mask, visible_goal, g_hat)
        g_hat = g_hat * nm

    u = k * (g_hat - x)
    mask = node_mask & actuator_mask
    return jnp.where(mask, u, 0.0).astype(jnp.float32)
