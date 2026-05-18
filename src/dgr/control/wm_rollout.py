"""Adapt a world-model variant's one-step predictor into a multi-step rollout (chunk 2b).

Every variant exposes the same ``predict_next_nodes_single`` interface. ``make_wm_rollout``
closes over the (episode-static) graph structure and parameters and returns a pure
``rollout(nodes0, action_seq) -> x_traj`` function, suitable as the dynamics model inside
an MPC cost. The goal channel is static, observed context — so it is re-injected at every
step rather than trusted to the model (handoff decision 2).
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import lax

# (params, nodes, actions, senders, receivers, node_mask, edge_mask) -> next_nodes
PredictFn = Callable[..., jnp.ndarray]
# (nodes0, action_seq) -> x_traj
RolloutFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

_X, _GOAL = 0, 1  # node feature indices: [x, goal]


def make_wm_rollout(
    predict_next_nodes: PredictFn,
    params: dict,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    node_mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
) -> RolloutFn:
    """Build an open-loop rollout for one world model on one fixed graph.

    The returned ``rollout`` takes the current node features ``(n_max, 2)`` and an action
    sequence ``(horizon, n_max)``, and returns the predicted state trajectory
    ``x_traj`` of shape ``(horizon, n_max)``.
    """

    def rollout(nodes0: jnp.ndarray, action_seq: jnp.ndarray) -> jnp.ndarray:
        goal = nodes0[:, _GOAL]  # static — re-injected each step

        def step(nodes: jnp.ndarray, action: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            pred = predict_next_nodes(
                params, nodes, action, senders, receivers, node_mask, edge_mask
            )
            x_next = jnp.where(node_mask, pred[:, _X], 0.0)
            nodes_next = jnp.stack([x_next, goal], axis=-1)
            return nodes_next, x_next

        _, x_traj = lax.scan(step, nodes0, action_seq)
        return x_traj

    return rollout
