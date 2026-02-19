# a tiny masked message-passing block

from __future__ import annotations  # for JAX/ML code organization; see graph_spec.py

import jax.numpy as jnp

from dgr.interface.graph_spec import Graph


def masked_message_passing(g: Graph) -> jnp.ndarray:
    """
    Minimal permutation-equivariant message passing:
      msg_e = nodes[senders[e]]
      agg_i = sum_{e: receivers[e]=i} msg_e
      out_i = nodes[i] + agg_i
    Masks:
      - edge_mask gates messages
      - node_mask gates outputs (masked nodes -> 0)
    Returns:
      out_nodes: (N_max, F_n)
    """
    nodes = g.nodes
    n_max, f_n = nodes.shape

    msgs = nodes[g.senders]  # (E_max, F_n)
    msgs = msgs * g.edge_mask[:, None].astype(nodes.dtype)

    agg = jnp.zeros((n_max, f_n), dtype=nodes.dtype)
    agg = agg.at[g.receivers].add(msgs)

    out = nodes + agg
    out = jnp.where(g.node_mask[:, None], out, jnp.zeros_like(out))
    return out
