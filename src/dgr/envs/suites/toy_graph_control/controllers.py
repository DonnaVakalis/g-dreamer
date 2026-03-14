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
