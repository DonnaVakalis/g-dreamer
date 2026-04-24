from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FlatWMConfig:
    node_dim: int
    n_max: int
    hidden_dim: int = 256


def _init_linear(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jnp.ndarray]:
    scale = jnp.sqrt(2.0 / max(in_dim, 1))
    return {
        "w": scale * jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32),
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _linear(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["w"] + params["b"]


def init_params(
    key: jax.Array,
    config: FlatWMConfig,
) -> dict[str, dict[str, jnp.ndarray]]:
    # Input: flatten(nodes * node_mask) + node_mask + actions
    # = n_max * node_dim  +  n_max  +  n_max
    in_dim = config.n_max * (config.node_dim + 2)
    out_dim = config.n_max * config.node_dim
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "layer1": _init_linear(k1, in_dim, config.hidden_dim),
        "layer2": _init_linear(k2, config.hidden_dim, config.hidden_dim),
        "out": _init_linear(k3, config.hidden_dim, out_dim),
    }


def predict_next_nodes_single(
    params: dict[str, dict[str, jnp.ndarray]],
    nodes: jnp.ndarray,
    actions: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    node_mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
) -> jnp.ndarray:
    node_mask = node_mask.astype(jnp.bool_)
    n_max, node_dim = nodes.shape

    masked_nodes = jnp.where(node_mask[:, None], nodes, jnp.zeros_like(nodes))
    flat_nodes = masked_nodes.reshape(-1)
    flat_mask = node_mask.astype(jnp.float32)
    flat_actions = actions.astype(jnp.float32)

    x = jnp.concatenate([flat_nodes, flat_mask, flat_actions], axis=0)
    h = jax.nn.relu(_linear(params["layer1"], x))
    h = jax.nn.relu(_linear(params["layer2"], h))
    pred_flat = _linear(params["out"], h)

    pred = pred_flat.reshape(n_max, node_dim)
    return jnp.where(node_mask[:, None], pred, jnp.zeros_like(pred))


def predict_next_nodes_batch(
    params: dict[str, dict[str, jnp.ndarray]],
    nodes: jnp.ndarray,
    actions: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    node_mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(
        predict_next_nodes_single,
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )(params, nodes, actions, senders, receivers, node_mask, edge_mask)


def masked_node_mse(
    pred_nodes: jnp.ndarray,
    target_nodes: jnp.ndarray,
    node_mask: jnp.ndarray,
    *,
    feature_index: int | None = None,
) -> jnp.ndarray:
    if feature_index is not None:
        pred_nodes = pred_nodes[..., feature_index : feature_index + 1]
        target_nodes = target_nodes[..., feature_index : feature_index + 1]
    mask = node_mask[..., :, None].astype(jnp.float32)
    sq = jnp.square(pred_nodes - target_nodes) * mask
    denom = jnp.maximum(jnp.sum(mask) * pred_nodes.shape[-1], 1.0)
    return jnp.sum(sq) / denom


def batch_loss(
    params: dict[str, dict[str, jnp.ndarray]],
    batch: dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    pred = predict_next_nodes_batch(
        params,
        batch["nodes"],
        batch["actions"],
        batch["senders"],
        batch["receivers"],
        batch["node_mask"],
        batch["edge_mask"],
    )
    loss = masked_node_mse(pred, batch["next_nodes"], batch["node_mask"])
    x_mse = masked_node_mse(pred, batch["next_nodes"], batch["node_mask"], feature_index=0)
    return loss, {"loss": loss, "x_mse": x_mse}
