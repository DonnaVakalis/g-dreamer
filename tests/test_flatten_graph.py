from __future__ import annotations

import jax
import jax.numpy as jnp

from dgr.envs.wrappers.flatten_graph import flat_dim, flatten_graph
from dgr.interface.graph_spec import Graph, GraphSpec, validate_graph


def _random_graph(spec: GraphSpec, n_real: int, e_real: int, seed: int) -> Graph:
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    nodes = jax.random.normal(k1, (spec.n_max, spec.f_n), dtype=jnp.float32)
    node_mask = jnp.arange(spec.n_max) < n_real

    senders_real = jax.random.randint(k2, (e_real,), 0, n_real, dtype=jnp.int32)
    receivers_real = jax.random.randint(k3, (e_real,), 0, n_real, dtype=jnp.int32)
    senders = jnp.zeros((spec.e_max,), dtype=jnp.int32).at[:e_real].set(senders_real)
    receivers = jnp.zeros((spec.e_max,), dtype=jnp.int32).at[:e_real].set(receivers_real)
    edge_mask = jnp.arange(spec.e_max) < e_real

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


def test_flat_dim_matches_output_length():
    spec = GraphSpec(n_max=6, e_max=10, f_n=2, f_e=0, f_g=0)
    g = _random_graph(spec, n_real=4, e_real=7, seed=0)
    vec = flatten_graph(g, spec)
    assert vec.shape == (flat_dim(spec),)


def test_flatten_padding_invariance():
    spec = GraphSpec(n_max=8, e_max=12, f_n=3, f_e=0, f_g=0)
    n_real, e_real = 5, 9
    g1 = _random_graph(spec, n_real=n_real, e_real=e_real, seed=1)
    vec1 = flatten_graph(g1, spec)

    # Make a copy with junk in padded nodes + padded edges indices.
    key = jax.random.PRNGKey(999)
    junk_nodes = jax.random.normal(key, (spec.n_max, spec.f_n), dtype=jnp.float32)

    padded_nodes = jnp.arange(spec.n_max) >= n_real
    nodes2 = jnp.where(padded_nodes[:, None], junk_nodes, g1.nodes)

    # Corrupt padded edge indices too; should be masked away by flatten_graph.
    senders2 = g1.senders.at[e_real:].set(jnp.arange(spec.e_max - e_real, dtype=jnp.int32))
    receivers2 = g1.receivers.at[e_real:].set(jnp.arange(spec.e_max - e_real, dtype=jnp.int32))

    g2 = Graph(
        nodes=nodes2,
        edges=g1.edges,
        senders=senders2,
        receivers=receivers2,
        node_mask=g1.node_mask,
        edge_mask=g1.edge_mask,
        globals=g1.globals,
    )
    vec2 = flatten_graph(g2, spec)

    assert jnp.allclose(vec1, vec2, atol=1e-6)
