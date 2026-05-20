"""Graph-theoretic invariants per (topology, size) — the W1 theoretical handles.

Currently exposed:
- **algebraic connectivity** ``λ₁(L)`` of the (symmetric) combinatorial Laplacian: the canonical
  measure of how fast diffusion mixes on the graph (small = slow, ring; large = fast,
  k-regular expanders). Hypothesised to drive C's open-loop divergence rate.
- **diameter** ``D(G)``: longest shortest-path between any two nodes.
- **degree statistics**: mean / std / min / max.

The env stores each undirected edge as two directed entries; this module treats the graph
as undirected (symmetrises the adjacency) before computing invariants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GraphInvariants:
    n_real: int
    n_edges_undirected: int
    mean_degree: float
    degree_std: float
    min_degree: int
    max_degree: int
    diameter: int  # -1 if disconnected
    algebraic_connectivity: float  # λ₁(L) — the spectral gap; 0 if disconnected
    laplacian_largest: float  # λ_max(L)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_graph_invariants(
    senders: jnp.ndarray | np.ndarray,
    receivers: jnp.ndarray | np.ndarray,
    edge_mask: jnp.ndarray | np.ndarray,
    n_real: int,
) -> GraphInvariants:
    """Compute graph invariants on the real-node subgraph.

    Ignores self-loops and treats the graph as undirected (symmetric adjacency).
    """
    if n_real < 1:
        raise ValueError("n_real must be >= 1")
    s = np.asarray(senders)[np.asarray(edge_mask)]
    r = np.asarray(receivers)[np.asarray(edge_mask)]
    keep = (s < n_real) & (r < n_real) & (s != r)
    s, r = s[keep], r[keep]

    A = np.zeros((n_real, n_real), dtype=np.float64)
    A[s, r] = 1.0
    A = np.maximum(A, A.T)  # symmetrise
    np.fill_diagonal(A, 0.0)

    degrees = A.sum(axis=1).astype(np.int64)
    n_und = int(A.sum() // 2)

    if n_real == 1:
        return GraphInvariants(
            n_real=1,
            n_edges_undirected=0,
            mean_degree=0.0,
            degree_std=0.0,
            min_degree=0,
            max_degree=0,
            diameter=0,
            algebraic_connectivity=0.0,
            laplacian_largest=0.0,
        )

    laplacian = np.diag(degrees.astype(np.float64)) - A
    eigvals = np.sort(np.linalg.eigvalsh(laplacian))
    algebraic_connectivity = float(eigvals[1])  # second-smallest; 0 iff disconnected
    laplacian_largest = float(eigvals[-1])

    diameter = _diameter(A)
    return GraphInvariants(
        n_real=n_real,
        n_edges_undirected=n_und,
        mean_degree=float(degrees.mean()),
        degree_std=float(degrees.std()),
        min_degree=int(degrees.min()),
        max_degree=int(degrees.max()),
        diameter=diameter,
        algebraic_connectivity=algebraic_connectivity,
        laplacian_largest=laplacian_largest,
    )


def _diameter(adjacency: np.ndarray) -> int:
    """All-pairs BFS diameter; returns -1 if the graph is disconnected."""
    n = adjacency.shape[0]
    if n <= 1:
        return 0
    neighbours = [np.where(adjacency[i] > 0)[0].tolist() for i in range(n)]
    longest = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in neighbours[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if any(d == -1 for d in dist):
            return -1
        longest = max(longest, max(dist))
    return longest
