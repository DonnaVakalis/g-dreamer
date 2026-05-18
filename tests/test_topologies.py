from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dgr.envs.suites.toy_graph_control.topologies import (
    grid_dims,
    make_grid_topology,
    make_kregular_topology,
)
from dgr.interface.graph_spec import Graph, GraphSpec, validate_graph

_SPEC = GraphSpec(n_max=16, e_max=64, f_n=2)


def _degrees(senders: jnp.ndarray, edge_mask: jnp.ndarray, n_real: int) -> np.ndarray:
    real_senders = np.asarray(senders)[np.asarray(edge_mask)]
    return np.bincount(real_senders, minlength=n_real)[:n_real]


def _undirected_set(senders: jnp.ndarray, receivers: jnp.ndarray, edge_mask: jnp.ndarray):
    s = np.asarray(senders)[np.asarray(edge_mask)]
    r = np.asarray(receivers)[np.asarray(edge_mask)]
    return {(min(a, b), max(a, b)) for a, b in zip(s, r)}


def _as_graph(spec: GraphSpec, senders, receivers, edge_mask, n_real: int) -> Graph:
    return Graph(
        nodes=jnp.zeros((spec.n_max, spec.f_n), dtype=jnp.float32),
        edges=jnp.zeros((spec.e_max, spec.f_e), dtype=jnp.float32),
        senders=senders,
        receivers=receivers,
        node_mask=jnp.arange(spec.n_max) < n_real,
        edge_mask=edge_mask,
        globals=jnp.zeros((spec.f_g,), dtype=jnp.float32),
    )


def test_grid_dims_are_most_square():
    assert grid_dims(4) == (2, 2)
    assert grid_dims(6) == (2, 3)
    assert grid_dims(9) == (3, 3)
    assert grid_dims(12) == (3, 4)
    assert grid_dims(16) == (4, 4)


def test_grid_degrees_match_von_neumann_neighbourhood():
    # 3x3 grid: 4 corners (deg 2), 4 edges (deg 3), 1 centre (deg 4).
    senders, receivers, edge_mask = make_grid_topology(_SPEC, 9)
    deg = _degrees(senders, edge_mask, 9)
    assert sorted(deg.tolist()) == [2, 2, 2, 2, 3, 3, 3, 3, 4]
    # senders and receivers are reverses of each other (undirected stored both ways).
    assert deg.tolist() == _degrees(receivers, edge_mask, 9).tolist()


def test_grid_edge_count_and_symmetry():
    rows, cols = grid_dims(12)
    senders, receivers, edge_mask = make_grid_topology(_SPEC, 12)
    n_undirected = rows * (cols - 1) + cols * (rows - 1)
    assert int(jnp.sum(edge_mask)) == 2 * n_undirected
    # Every directed edge has its reverse.
    real = list(
        zip(
            np.asarray(senders)[np.asarray(edge_mask)].tolist(),
            np.asarray(receivers)[np.asarray(edge_mask)].tolist(),
        )
    )
    assert {(v, u) for u, v in real} == set(real)


def test_kregular_every_node_has_degree_k():
    for n_real in (6, 10, 16):
        senders, _, edge_mask = make_kregular_topology(_SPEC, n_real, k=4, seed=0)
        deg = _degrees(senders, edge_mask, n_real)
        assert deg.tolist() == [4] * n_real


def test_kregular_simple_and_connected():
    senders, receivers, edge_mask = make_kregular_topology(_SPEC, 16, k=4, seed=1)
    undirected = _undirected_set(senders, receivers, edge_mask)
    # Simple graph: no self-loops, no multi-edges (one entry per undirected pair).
    assert all(u != v for u, v in undirected)
    assert len(undirected) == 16 * 4 // 2
    # Connected: BFS reaches every node.
    adj: dict[int, list[int]] = {i: [] for i in range(16)}
    for u, v in undirected:
        adj[u].append(v)
        adj[v].append(u)
    seen, stack = {0}, [0]
    while stack:
        for y in adj[stack.pop()]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    assert len(seen) == 16


def test_kregular_is_deterministic_for_a_seed():
    a = make_kregular_topology(_SPEC, 12, k=4, seed=7)
    b = make_kregular_topology(_SPEC, 12, k=4, seed=7)
    c = make_kregular_topology(_SPEC, 12, k=4, seed=8)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b))
    assert not all(jnp.array_equal(x, y) for x, y in zip(a, c))


def test_topologies_satisfy_the_graph_contract():
    for senders, receivers, edge_mask, n_real in (
        (*make_grid_topology(_SPEC, 16), 16),
        (*make_kregular_topology(_SPEC, 16, k=4, seed=0), 16),
    ):
        assert senders.dtype == jnp.int32 and receivers.dtype == jnp.int32
        assert edge_mask.dtype == jnp.bool_
        assert senders.shape == receivers.shape == edge_mask.shape == (_SPEC.e_max,)
        # Padding slots are index 0 and masked off.
        e_real = int(jnp.sum(edge_mask))
        assert jnp.all(senders[e_real:] == 0) and jnp.all(receivers[e_real:] == 0)
        assert not bool(jnp.any(edge_mask[e_real:]))
        validate_graph(_as_graph(_SPEC, senders, receivers, edge_mask, n_real), _SPEC)
