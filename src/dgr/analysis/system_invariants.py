"""Classical (A, B) controllability and goal-information geometry — interface (I) handles.

Two distinct theoretical concepts, one per interface axis:

- **I2 actuation: controllability rank of (A, B).** Given the linearised dynamics matrix `A`
  (function of W1 topology + W2 dynamics) and the actuator selector `B` (from I2's
  `actuator_mask`), the rank of the Kalman controllability matrix
  ``[B, AB, A²B, …, A^{n-1}B]`` tells us *whether* the full state can be driven, and the
  controllability Gramian eigenvalues tell us *how cheaply*. Rank deficit = some modes are
  uncontrollable from the actuator set.

- **I1 observability of goals: graph distance from hidden goals to nearest visible.** Our
  state observation is always full; the I1 axis varies which *goals* the agent sees. The
  classical handle is therefore not (A, C)-observability but **harmonic reconstruction on
  the graph**: given visible-goal "boundary values", the un-observed goals are best
  inferred by neighbour averaging — and the inference quality is bounded by the average
  graph distance from each hidden node to the nearest visible one. We expose that average
  and max as scalar summaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SystemInvariants:
    n_real: int
    controllability_rank: int
    controllability_gramian_min_eig: float  # smallest eigenvalue — bottleneck mode
    controllability_gramian_condition: float  # max / min eigenvalue (np.inf if rank deficit)
    n_actuators: int
    n_visible_goals: int
    goal_info_mean_distance: float  # mean graph distance from hidden goal to nearest visible
    goal_info_max_distance: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def linearised_consensus_dynamics(
    senders: np.ndarray,
    receivers: np.ndarray,
    edge_mask: np.ndarray,
    n_real: int,
    alpha: float,
) -> np.ndarray:
    """``A_lin = (1-α)·I + α·P`` where ``P`` is the random-walk operator on the real-node graph."""
    s = senders[edge_mask]
    r = receivers[edge_mask]
    keep = (s < n_real) & (r < n_real) & (s != r)
    s, r = s[keep], r[keep]
    adjacency = np.zeros((n_real, n_real), dtype=np.float64)
    adjacency[s, r] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    degree = adjacency.sum(axis=1)
    walk = np.zeros_like(adjacency)
    nz = degree > 0
    walk[nz] = adjacency[nz] / degree[nz, None]
    return (1.0 - alpha) * np.eye(n_real) + alpha * walk


def linearised_node_independent_dynamics(n_real: int, alpha: float) -> np.ndarray:
    """``A_lin = (1-α)·I`` — diagonal, no coupling."""
    return (1.0 - alpha) * np.eye(n_real)


def _controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    cols = [B]
    AkB = B
    for _ in range(n - 1):
        AkB = A @ AkB
        cols.append(AkB)
    return np.concatenate(cols, axis=1)


def _gramian(A: np.ndarray, B: np.ndarray, steps: int | None = None) -> np.ndarray:
    n = A.shape[0]
    if steps is None:
        steps = n
    G = np.zeros((n, n), dtype=np.float64)
    AkB = B
    for _ in range(steps):
        G = G + AkB @ AkB.T
        AkB = A @ AkB
    return G


def compute_system_invariants(
    A_lin: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    edge_mask: np.ndarray,
    actuator_mask: np.ndarray,
    goal_obs_mask: np.ndarray,
    n_real: int,
) -> SystemInvariants:
    """Combine controllability of (A, B) and goal-information distances on the graph."""
    actuator_idx = np.where(np.asarray(actuator_mask)[:n_real])[0]
    n_actuators = int(actuator_idx.size)
    B = np.zeros((n_real, max(n_actuators, 1)), dtype=np.float64)
    for col, node_i in enumerate(actuator_idx):
        B[node_i, col] = 1.0
    if n_actuators == 0:
        ctrb_rank = 0
        min_eig = 0.0
        condition = float("inf")
    else:
        ctrb = _controllability_matrix(A_lin, B)
        ctrb_rank = int(np.linalg.matrix_rank(ctrb))
        gramian = _gramian(A_lin, B, steps=n_real)
        eigs = np.linalg.eigvalsh((gramian + gramian.T) / 2)
        eigs = np.clip(eigs, 0.0, None)
        min_eig = float(eigs.min())
        max_eig = float(eigs.max())
        condition = float(max_eig / min_eig) if min_eig > 0 else float("inf")

    visible_idx = np.where(np.asarray(goal_obs_mask)[:n_real])[0]
    n_visible = int(visible_idx.size)

    # Graph distance from each hidden goal node to its nearest visible goal node.
    s = np.asarray(senders)[np.asarray(edge_mask)]
    r = np.asarray(receivers)[np.asarray(edge_mask)]
    keep = (s < n_real) & (r < n_real)
    s, r = s[keep], r[keep]
    A_adj = np.zeros((n_real, n_real), dtype=np.float64)
    A_adj[s, r] = 1.0
    A_adj = np.maximum(A_adj, A_adj.T)
    np.fill_diagonal(A_adj, 0.0)
    neighbours = [np.where(A_adj[i] > 0)[0].tolist() for i in range(n_real)]

    if n_visible == 0 or n_visible == n_real:
        goal_info_mean = 0.0
        goal_info_max = 0
    else:
        visible_set = set(int(v) for v in visible_idx)
        hidden = [i for i in range(n_real) if i not in visible_set]
        distances: list[int] = []
        for src in hidden:
            dist = [-1] * n_real
            dist[src] = 0
            queue = [src]
            head = 0
            best = -1
            while head < len(queue):
                u = queue[head]
                head += 1
                if u in visible_set:
                    best = dist[u]
                    break
                for v in neighbours[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            distances.append(best if best >= 0 else n_real)  # unreachable → diameter cap
        goal_info_mean = float(np.mean(distances))
        goal_info_max = int(max(distances))

    return SystemInvariants(
        n_real=n_real,
        controllability_rank=ctrb_rank,
        controllability_gramian_min_eig=min_eig,
        controllability_gramian_condition=condition,
        n_actuators=n_actuators,
        n_visible_goals=n_visible,
        goal_info_mean_distance=goal_info_mean,
        goal_info_max_distance=goal_info_max,
    )
