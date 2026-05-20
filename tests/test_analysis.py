from __future__ import annotations

import numpy as np

from dgr.analysis.graph_invariants import compute_graph_invariants
from dgr.analysis.system_invariants import (
    compute_system_invariants,
    linearised_consensus_dynamics,
    linearised_node_independent_dynamics,
)


def _ring_edges(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both-ways directed edges of an n-cycle padded to e_max = 2n."""
    e_real = 2 * n
    s = np.zeros(e_real, dtype=np.int32)
    r = np.zeros(e_real, dtype=np.int32)
    for i in range(n):
        s[2 * i] = i
        r[2 * i] = (i + 1) % n
        s[2 * i + 1] = i
        r[2 * i + 1] = (i - 1) % n
    mask = np.ones(e_real, dtype=bool)
    return s, r, mask


def _complete_edges(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    s = np.asarray([a for a, _ in pairs], dtype=np.int32)
    r = np.asarray([b for _, b in pairs], dtype=np.int32)
    mask = np.ones(len(pairs), dtype=bool)
    return s, r, mask


def test_ring_invariants_match_analytical_values():
    # n=6 cycle: every node degree 2; diameter n/2 = 3; λ₁(L) = 2 − 2 cos(2π/n) = 1 for n=6.
    s, r, mask = _ring_edges(6)
    inv = compute_graph_invariants(s, r, mask, n_real=6)
    assert inv.n_edges_undirected == 6
    assert inv.mean_degree == 2.0
    assert inv.degree_std == 0.0
    assert inv.min_degree == 2 and inv.max_degree == 2
    assert inv.diameter == 3
    assert abs(inv.algebraic_connectivity - 1.0) < 1e-9


def test_complete_graph_invariants_match_analytical_values():
    # K_n: every node degree n−1; diameter 1; Laplacian eigenvalues {0, n, n, …, n}.
    for n in (3, 5):
        s, r, mask = _complete_edges(n)
        inv = compute_graph_invariants(s, r, mask, n_real=n)
        assert inv.mean_degree == n - 1
        assert inv.diameter == 1
        assert abs(inv.algebraic_connectivity - n) < 1e-9
        assert abs(inv.laplacian_largest - n) < 1e-9


def test_node_independent_controllability_matches_actuator_count():
    # Node-independent dynamics: only the actuated nodes are controllable.
    n = 6
    A = linearised_node_independent_dynamics(n, alpha=0.2)
    s, r, mask = _ring_edges(n)
    # Sparse actuation: only nodes {0, 3}.
    actuator_mask = np.zeros(n, dtype=bool)
    actuator_mask[[0, 3]] = True
    goal_obs_mask = np.ones(n, dtype=bool)
    inv = compute_system_invariants(A, s, r, mask, actuator_mask, goal_obs_mask, n_real=n)
    assert inv.n_actuators == 2
    assert inv.controllability_rank == 2  # only the 2 actuated nodes


def test_dense_actuation_gives_full_controllability_on_consensus():
    n = 5
    A = linearised_consensus_dynamics(*_ring_edges(n), n_real=n, alpha=0.3)
    s, r, mask = _ring_edges(n)
    actuator_mask = np.ones(n, dtype=bool)
    goal_obs_mask = np.ones(n, dtype=bool)
    inv = compute_system_invariants(A, s, r, mask, actuator_mask, goal_obs_mask, n_real=n)
    assert inv.controllability_rank == n


def test_single_leader_on_symmetric_ring_has_eigenvalue_limited_rank():
    # Classical result: with single-input control, the controllable subspace has dimension at
    # most the number of *distinct* eigenvalues of A. The C_n ring with random-walk consensus
    # has a circulant Laplacian whose eigenvalues come in conjugate pairs — so the n=5 ring
    # has only 3 distinct eigenvalues {0.7+0.3, 0.7+0.3 cos(2π/5), 0.7+0.3 cos(4π/5)} and
    # rank is capped at 3, not 5. This is a feature of symmetry, not a bug.
    n = 5
    A = linearised_consensus_dynamics(*_ring_edges(n), n_real=n, alpha=0.3)
    s, r, mask = _ring_edges(n)
    actuator_mask = np.zeros(n, dtype=bool)
    actuator_mask[0] = True
    goal_obs_mask = np.ones(n, dtype=bool)
    inv = compute_system_invariants(A, s, r, mask, actuator_mask, goal_obs_mask, n_real=n)
    assert inv.controllability_rank < n
    assert inv.controllability_rank == len(np.unique(np.round(np.linalg.eigvals(A).real, 8)))


def test_goal_info_distance_zero_when_all_goals_visible():
    n = 5
    A = linearised_consensus_dynamics(*_ring_edges(n), n_real=n, alpha=0.3)
    s, r, mask = _ring_edges(n)
    actuator_mask = np.ones(n, dtype=bool)
    goal_obs_mask = np.ones(n, dtype=bool)
    inv = compute_system_invariants(A, s, r, mask, actuator_mask, goal_obs_mask, n_real=n)
    assert inv.goal_info_mean_distance == 0.0
    assert inv.goal_info_max_distance == 0


def test_goal_info_distance_on_ring_with_one_visible_leader():
    # n=6 ring, only node 0 visible: hidden nodes 1..5 have distances {1,2,3,2,1}.
    n = 6
    A = linearised_consensus_dynamics(*_ring_edges(n), n_real=n, alpha=0.3)
    s, r, mask = _ring_edges(n)
    actuator_mask = np.ones(n, dtype=bool)
    goal_obs_mask = np.zeros(n, dtype=bool)
    goal_obs_mask[0] = True
    inv = compute_system_invariants(A, s, r, mask, actuator_mask, goal_obs_mask, n_real=n)
    expected = (1 + 2 + 3 + 2 + 1) / 5
    assert abs(inv.goal_info_mean_distance - expected) < 1e-9
    assert inv.goal_info_max_distance == 3
