from __future__ import annotations

import jax
import jax.numpy as jnp

from dgr.interface.graph_spec import Graph, GraphSpec, permute_nodes, validate_graph, zeros_graph
from dgr.models.message_passing import masked_message_passing


def _random_padded_graph(spec: GraphSpec, n_real: int, e_real: int, seed: int) -> Graph:
    assert n_real <= spec.n_max
    assert e_real <= spec.e_max

    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    nodes = jax.random.normal(k1, (spec.n_max, spec.f_n), dtype=jnp.float32)

    # masks
    node_mask = jnp.arange(spec.n_max) < n_real
    edge_mask = jnp.arange(spec.e_max) < e_real

    # Real edges only connect within [0, n_real)
    senders_real = jax.random.randint(k2, (e_real,), 0, n_real, dtype=jnp.int32)
    receivers_real = jax.random.randint(k3, (e_real,), 0, n_real, dtype=jnp.int32)

    senders = jnp.zeros((spec.e_max,), dtype=jnp.int32).at[:e_real].set(senders_real)
    receivers = jnp.zeros((spec.e_max,), dtype=jnp.int32).at[:e_real].set(receivers_real)

    # Keep edge features present (even if unused yet)
    edges = jnp.zeros((spec.e_max, spec.f_e), dtype=jnp.float32)
    glb = jnp.zeros((spec.f_g,), dtype=jnp.float32)

    g = Graph(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        node_mask=node_mask.astype(jnp.bool_),
        edge_mask=edge_mask.astype(jnp.bool_),
        globals=glb,
    )
    validate_graph(g, spec)
    return g


def test_zeros_graph_shapes_and_masks():
    spec = GraphSpec(n_max=3, e_max=4, f_n=2, f_e=1, f_g=5)
    g = zeros_graph(spec)
    validate_graph(g, spec)
    # catches “contract drift” early (e.g., if we accidentally change shapes or mask semantics)
    assert g.nodes.shape == (3, 2)
    assert g.edges.shape == (4, 1)
    assert g.senders.shape == (4,)
    assert g.receivers.shape == (4,)
    assert g.node_mask.shape == (3,)
    assert g.edge_mask.shape == (4,)
    assert g.globals.shape == (5,)


def test_permutation_equivariance_message_passing():
    spec = GraphSpec(n_max=6, e_max=10, f_n=5, f_e=0, f_g=0)
    g = _random_padded_graph(spec, n_real=4, e_real=7, seed=0)

    out = masked_message_passing(g)

    perm = jnp.array([2, 0, 5, 1, 3, 4], dtype=jnp.int32)  # arbitrary permutation of N_max
    g_p = permute_nodes(g, perm)
    out_p = masked_message_passing(g_p)

    # Equivariance: out_p[i] == out[perm[i]]
    assert jnp.allclose(out_p, out[perm], atol=1e-5)


def test_padding_invariance_masked_nodes_do_not_matter():
    spec = GraphSpec(n_max=8, e_max=12, f_n=4, f_e=0, f_g=0)
    n_real, e_real = 5, 9

    g1 = _random_padded_graph(spec, n_real=n_real, e_real=e_real, seed=1)
    g2 = _random_padded_graph(spec, n_real=n_real, e_real=e_real, seed=1)

    # Same real nodes/edges, but scramble padded node values in g2
    # (masked nodes should not affect outputs)
    padded = jnp.arange(spec.n_max) >= n_real
    noise = jax.random.normal(jax.random.PRNGKey(999), (spec.n_max, spec.f_n), dtype=jnp.float32)
    nodes2 = jnp.where(padded[:, None], noise, g2.nodes)

    g2 = Graph(
        nodes=nodes2,
        edges=g2.edges,
        senders=g2.senders,
        receivers=g2.receivers,
        node_mask=g2.node_mask,
        edge_mask=g2.edge_mask,
        globals=g2.globals,
    )

    out1 = masked_message_passing(g1)
    out2 = masked_message_passing(g2)

    # Compare only real nodes
    assert jnp.allclose(out1[:n_real], out2[:n_real], atol=1e-5)
