"""Graph topology generators for the toy graph control suite (experiment matrix, Stage A1).

Each generator returns ``(senders, receivers, edge_mask)`` padded to ``spec.e_max``.
Undirected edges are stored as two directed edges. Topologies are static per (kind, size):
the ring and grid are deterministic; the random k-regular graph is sampled once from a
fixed seed and reused. ``make_topology`` is the dispatcher; ``Topology`` is the immutable
holder stored on ``ToyGraphControlConfig``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from dgr.interface.graph_spec import GraphSpec

_KREGULAR_MAX_ATTEMPTS = 5000


@dataclass(frozen=True)
class Topology:
    """Precomputed graph structure for one (kind, size); padding-masked to ``e_max``."""

    kind: str
    senders: jnp.ndarray
    receivers: jnp.ndarray
    edge_mask: jnp.ndarray


def _pad_edges(
    spec: GraphSpec, senders_real: list[int], receivers_real: list[int]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pad real directed edges to ``spec.e_max``; padding slots are index 0 and masked off."""
    e_real = len(senders_real)
    if e_real > spec.e_max:
        raise ValueError(f"Need e_max >= {e_real}, got e_max={spec.e_max}")
    senders = (
        jnp.zeros((spec.e_max,), dtype=jnp.int32)
        .at[:e_real]
        .set(jnp.asarray(senders_real, dtype=jnp.int32))
    )
    receivers = (
        jnp.zeros((spec.e_max,), dtype=jnp.int32)
        .at[:e_real]
        .set(jnp.asarray(receivers_real, dtype=jnp.int32))
    )
    edge_mask = jnp.arange(spec.e_max) < e_real
    return senders, receivers, edge_mask


def _both_ways(undirected: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Turn undirected edges into directed senders/receivers (each edge both ways)."""
    senders = [u for u, _ in undirected] + [v for _, v in undirected]
    receivers = [v for _, v in undirected] + [u for u, _ in undirected]
    return senders, receivers


def make_ring_topology(
    spec: GraphSpec, n_real: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Directed ring: node i connects to i±1 (mod n_real). Degree 2."""
    e_real = 2 * n_real
    if e_real > spec.e_max:
        raise ValueError(f"Need e_max >= 2*n_real, got e_max={spec.e_max}, n_real={n_real}")

    i = jnp.arange(n_real, dtype=jnp.int32)
    s1 = i
    r1 = (i + 1) % n_real
    s2 = i
    r2 = (i - 1) % n_real

    senders_real = jnp.concatenate([s1, s2], axis=0)
    receivers_real = jnp.concatenate([r1, r2], axis=0)

    senders = jnp.zeros((spec.e_max,), dtype=jnp.int32).at[:e_real].set(senders_real)
    receivers = jnp.zeros((spec.e_max,), dtype=jnp.int32).at[:e_real].set(receivers_real)
    edge_mask = jnp.arange(spec.e_max) < e_real
    return senders, receivers, edge_mask


def grid_dims(n_real: int) -> tuple[int, int]:
    """Most-square factorisation rows×cols == n_real (rows <= cols, rows maximal)."""
    best_rows = 1
    for rows in range(1, math.isqrt(n_real) + 1):
        if n_real % rows == 0:
            best_rows = rows
    return best_rows, n_real // best_rows


def make_grid_topology(
    spec: GraphSpec, n_real: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """2-D grid (4-neighbour / von Neumann), node id = row * cols + col."""
    rows, cols = grid_dims(n_real)
    if rows < 2 or cols < 2:
        raise ValueError(
            f"grid needs a rows×cols factorisation with both >= 2; n_real={n_real} has none"
        )
    undirected: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                undirected.append((u, u + 1))
            if r + 1 < rows:
                undirected.append((u, u + cols))
    return _pad_edges(spec, *_both_ways(undirected))


def _is_connected(n_real: int, undirected: list[tuple[int, int]]) -> bool:
    adj: dict[int, list[int]] = {i: [] for i in range(n_real)}
    for u, v in undirected:
        adj[u].append(v)
        adj[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == n_real


def make_kregular_topology(
    spec: GraphSpec, n_real: int, *, k: int = 4, seed: int = 0
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Connected random k-regular graph, sampled once from ``seed`` (configuration model)."""
    if n_real < k + 1:
        raise ValueError(f"k-regular needs n_real >= k+1; n_real={n_real}, k={k}")
    if (n_real * k) % 2 != 0:
        raise ValueError(f"k-regular needs n_real*k even; n_real={n_real}, k={k}")

    rng = np.random.default_rng(seed)
    for _ in range(_KREGULAR_MAX_ATTEMPTS):
        stubs = np.repeat(np.arange(n_real), k)
        rng.shuffle(stubs)
        pairs = stubs.reshape(-1, 2)
        edges: set[tuple[int, int]] = set()
        ok = True
        for a, b in pairs:
            u, v = int(a), int(b)
            if u == v:
                ok = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                ok = False
                break
            edges.add(edge)
        if ok:
            undirected = sorted(edges)
            if _is_connected(n_real, undirected):
                return _pad_edges(spec, *_both_ways(undirected))
    raise RuntimeError(
        f"failed to sample a connected {k}-regular graph on {n_real} nodes "
        f"in {_KREGULAR_MAX_ATTEMPTS} attempts"
    )


def make_topology(kind: str, spec: GraphSpec, n_real: int, *, seed: int = 0) -> Topology:
    """Dispatch to a generator and wrap the result as a ``Topology``."""
    if kind == "ring":
        senders, receivers, edge_mask = make_ring_topology(spec, n_real)
    elif kind == "grid":
        senders, receivers, edge_mask = make_grid_topology(spec, n_real)
    elif kind == "kregular":
        senders, receivers, edge_mask = make_kregular_topology(spec, n_real, seed=seed)
    else:
        raise ValueError(f"unknown topology kind: {kind!r} (expected ring/grid/kregular)")
    return Topology(kind=kind, senders=senders, receivers=receivers, edge_mask=edge_mask)
