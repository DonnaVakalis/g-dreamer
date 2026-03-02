from __future__ import annotations

import jax.numpy as jnp

from dgr.interface.graph_spec import Graph, GraphSpec


def flat_dim(
    spec: GraphSpec,
    *,
    include_edges: bool = True,
    include_indices: bool = True,
    include_masks: bool = True,
    include_globals: bool = True,
) -> int:
    d = spec.n_max * spec.f_n
    if include_edges:
        d += spec.e_max * spec.f_e
    if include_indices:
        d += 2 * spec.e_max  # senders + receivers (normalized floats)
    if include_masks:
        d += spec.n_max + spec.e_max
    if include_globals:
        d += spec.f_g
    return int(d)


def _norm_index(idx: jnp.ndarray, n_max: int) -> jnp.ndarray:
    """Map int indices in [0, n_max-1] to float in [-1, 1]."""
    if n_max <= 1:
        return jnp.zeros_like(idx, dtype=jnp.float32)
    return (idx.astype(jnp.float32) / jnp.float32(n_max - 1)) * 2.0 - 1.0


def flatten_graph(
    g: Graph,
    spec: GraphSpec,
    *,
    include_edges: bool = True,
    include_indices: bool = True,
    include_masks: bool = True,
    include_globals: bool = True,
) -> jnp.ndarray:
    """
    Flatten a padded Graph into a 1D float32 vector with **no padding leakage**.
    Everything that corresponds to masked nodes/edges is zeroed before flattening.
    """
    # Mask features so padded values can't leak.
    nodes = g.nodes * g.node_mask[:, None].astype(jnp.float32)
    parts = [nodes.reshape(-1)]

    if include_edges and spec.f_e:
        edges = g.edges * g.edge_mask[:, None].astype(jnp.float32)
        parts.append(edges.reshape(-1))

    if include_indices:
        # Zero out indices for masked edges, then normalize.
        senders = jnp.where(g.edge_mask, g.senders, jnp.zeros_like(g.senders))
        receivers = jnp.where(g.edge_mask, g.receivers, jnp.zeros_like(g.receivers))
        parts.append(_norm_index(senders, spec.n_max))
        parts.append(_norm_index(receivers, spec.n_max))

    if include_masks:
        parts.append(g.node_mask.astype(jnp.float32))
        parts.append(g.edge_mask.astype(jnp.float32))

    if include_globals and spec.f_g:
        parts.append(g.globals.astype(jnp.float32))

    vec = jnp.concatenate([p.reshape(-1).astype(jnp.float32) for p in parts], axis=0)

    expected = flat_dim(
        spec,
        include_edges=include_edges,
        include_indices=include_indices,
        include_masks=include_masks,
        include_globals=include_globals,
    )
    if vec.shape != (expected,):
        raise ValueError(f"Flatten produced shape {vec.shape}, expected {(expected,)}")
    return vec
