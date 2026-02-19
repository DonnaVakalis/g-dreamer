# Graph observation contract and shape helpers

# A note about this first import:
# This is especially useful in JAX/ML bc we
# have complex type relationships and want to avoid import-time muck :)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp


@dataclass(frozen=True)
class GraphSpec:
    """Static-shape graph spec for JIT/batching."""

    n_max: int
    e_max: int
    f_n: int
    f_e: int = 0
    f_g: int = 0


@dataclass(frozen=True)
class Graph:
    """
    Canonical static-shape graph dict (padded + masked).

    nodes:      (N_max, F_n) float32
    edges:      (E_max, F_e) float32 (F_e may be 0)
    senders:    (E_max,) int32
    receivers:  (E_max,) int32
    node_mask:  (N_max,) bool
    edge_mask:  (E_max,) bool
    globals:    (F_g,) float32 (F_g may be 0)
    """

    nodes: jnp.ndarray
    edges: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray
    node_mask: jnp.ndarray
    edge_mask: jnp.ndarray
    globals: jnp.ndarray

    @property
    def n_max(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def e_max(self) -> int:
        return int(self.senders.shape[0])


def zeros_graph(spec: GraphSpec, *, dtype=jnp.float32) -> Graph:
    nodes = jnp.zeros((spec.n_max, spec.f_n), dtype=dtype)
    edges = jnp.zeros((spec.e_max, spec.f_e), dtype=dtype)
    senders = jnp.zeros((spec.e_max,), dtype=jnp.int32)
    receivers = jnp.zeros((spec.e_max,), dtype=jnp.int32)
    node_mask = jnp.zeros((spec.n_max,), dtype=jnp.bool_)
    edge_mask = jnp.zeros((spec.e_max,), dtype=jnp.bool_)
    glb = jnp.zeros((spec.f_g,), dtype=dtype)
    return Graph(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        node_mask=node_mask,
        edge_mask=edge_mask,
        globals=glb,
    )


def validate_graph(g: Graph, spec: Optional[GraphSpec] = None) -> None:
    if g.nodes.ndim != 2:
        raise ValueError(f"nodes must be 2D, got {g.nodes.shape}")
    if g.edges.ndim != 2:
        raise ValueError(f"edges must be 2D, got {g.edges.shape}")
    if g.senders.shape != g.receivers.shape:
        raise ValueError("senders and receivers must have same shape")
    if g.senders.ndim != 1:
        raise ValueError("senders must be 1D")
    if g.node_mask.ndim != 1 or g.edge_mask.ndim != 1:
        raise ValueError("masks must be 1D")
    if g.senders.dtype != jnp.int32 or g.receivers.dtype != jnp.int32:
        raise ValueError("senders/receivers must be int32")
    if g.node_mask.dtype != jnp.bool_ or g.edge_mask.dtype != jnp.bool_:
        raise ValueError("masks must be bool")

    n_max = g.nodes.shape[0]
    e_max = g.senders.shape[0]
    if g.node_mask.shape[0] != n_max:
        raise ValueError("node_mask must have length N_max")
    if g.edge_mask.shape[0] != e_max:
        raise ValueError("edge_mask must have length E_max")

    # Indices must be in-range; masked-off edges are allowed to be anything,
    # but we still encourage in-range for easier debugging.
    if jnp.any(g.senders < 0) or jnp.any(g.senders >= n_max):
        raise ValueError("senders contains out-of-range indices")
    if jnp.any(g.receivers < 0) or jnp.any(g.receivers >= n_max):
        raise ValueError("receivers contains out-of-range indices")

    if spec is not None:
        if (n_max, e_max) != (spec.n_max, spec.e_max):
            raise ValueError(f"Graph N/E mismatch: {(n_max, e_max)} vs {(spec.n_max, spec.e_max)}")
        if g.nodes.shape[1] != spec.f_n:
            raise ValueError("node feature dim mismatch")
        if g.edges.shape[1] != spec.f_e:
            raise ValueError("edge feature dim mismatch")
        if g.globals.shape != (spec.f_g,):
            raise ValueError("globals feature dim mismatch")


def permute_nodes(g: Graph, perm: jnp.ndarray) -> Graph:
    """
    Permute node order. Updates senders/receivers so the *graph* is unchanged,
    just re-indexed.

    Convention: nodes_perm[i] = nodes[perm[i]] (perm maps new_index -> old_index).
    Therefore old_index j becomes new_index inv_perm[j].
    """
    perm = perm.astype(jnp.int32)
    n_max = g.nodes.shape[0]
    inv_perm = jnp.empty((n_max,), dtype=jnp.int32).at[perm].set(jnp.arange(n_max, dtype=jnp.int32))

    nodes_p = g.nodes[perm]
    node_mask_p = g.node_mask[perm]
    senders_p = inv_perm[g.senders]
    receivers_p = inv_perm[g.receivers]

    return Graph(
        nodes=nodes_p,
        edges=g.edges,  # edge features remain aligned with edge slots
        senders=senders_p,
        receivers=receivers_p,
        node_mask=node_mask_p,
        edge_mask=g.edge_mask,  # unchanged mask per edge slot
        globals=g.globals,
    )
