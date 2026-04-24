from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dgr.interface.graph_spec import Graph
from dgr.models.message_passing import masked_message_passing


@dataclass(frozen=True)
class GraphEncDecWMConfig:
    node_dim: int
    latent_dim: int = 64
    hidden_dim: int = 128


def _init_linear(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jnp.ndarray]:
    scale = jnp.sqrt(2.0 / max(in_dim, 1))
    return {
        "w": scale * jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32),
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _linear(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["w"] + params["b"]


def _mask_nodes(x: jnp.ndarray, node_mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(node_mask[:, None], x, jnp.zeros_like(x))


def _mp(
    nodes: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    node_mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
) -> jnp.ndarray:
    g = Graph(
        nodes=nodes,
        edges=jnp.zeros((senders.shape[0], 0), dtype=nodes.dtype),
        senders=senders.astype(jnp.int32),
        receivers=receivers.astype(jnp.int32),
        node_mask=node_mask.astype(jnp.bool_),
        edge_mask=edge_mask.astype(jnp.bool_),
        globals=jnp.zeros((0,), dtype=nodes.dtype),
    )
    return masked_message_passing(g)


def init_params(
    key: jax.Array,
    config: GraphEncDecWMConfig,
) -> dict[str, dict[str, jnp.ndarray]]:
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    return {
        # encoder: node_dim -> hidden -> latent (per node), then masked mean pool
        "enc_in": _init_linear(k1, config.node_dim, config.hidden_dim),
        "enc_out": _init_linear(k2, config.hidden_dim, config.latent_dim),
        # dynamics: (latent_dim + 1) -> hidden -> latent_dim  [+1 for scalar action mean]
        "dyn_in": _init_linear(k3, config.latent_dim + 1, config.hidden_dim),
        "dyn_out": _init_linear(k4, config.hidden_dim, config.latent_dim),
        # decoder per node: (latent_dim + 1) -> hidden -> node_dim  [+1 for node position]
        "dec_in": _init_linear(k5, config.latent_dim + 1, config.hidden_dim),
        "dec_out": _init_linear(k6, config.hidden_dim, config.node_dim),
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
    edge_mask = edge_mask.astype(jnp.bool_)
    n_max = nodes.shape[0]

    # Encoder: per-node embed → message passing → latent per node
    h = jax.nn.relu(_linear(params["enc_in"], _mask_nodes(nodes, node_mask)))
    h = _mask_nodes(h, node_mask)
    h = jax.nn.relu(_mp(h, senders, receivers, node_mask, edge_mask))
    h = _mask_nodes(jax.nn.relu(_linear(params["enc_out"], h)), node_mask)

    # Pool: masked mean over real nodes → global latent z
    n_real = jnp.maximum(jnp.sum(node_mask.astype(jnp.float32)), 1.0)
    z = jnp.sum(h, axis=0) / n_real  # (latent_dim,)

    # Action: masked mean of per-node scalar actions
    action_mean = jnp.sum(actions.astype(jnp.float32) * node_mask.astype(jnp.float32)) / n_real

    # Dynamics: MLP on global latent + scalar action
    dyn_in = jnp.concatenate([z, action_mean[None]], axis=0)
    z_next = jax.nn.relu(_linear(params["dyn_in"], dyn_in))
    z_next = _linear(params["dyn_out"], z_next)  # (latent_dim,)

    # Decoder: broadcast z_next to each node, concat with normalized node index
    node_pos = jnp.arange(n_max, dtype=jnp.float32) / jnp.maximum(n_max - 1, 1)  # (n_max,)
    z_broadcast = jnp.tile(z_next[None, :], (n_max, 1))  # (n_max, latent_dim)
    dec_in = jnp.concatenate([z_broadcast, node_pos[:, None]], axis=-1)
    pred = jax.nn.relu(_linear(params["dec_in"], dec_in))
    pred = _linear(params["dec_out"], pred)

    return _mask_nodes(pred, node_mask)


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
