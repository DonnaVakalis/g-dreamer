"""
Consensus / diffusion toy environment:
A group of agents (nodes) on a ring topology need to reach consensus
on their states while being influenced by their neighbors.

Example: If you have 3 agents with goals [1.0, 2.0, 3.0],
they need to coordinate through their ring connections to
reach these target values despite random noise and their own
control actions.

Environment Overview
Problem: A group of agents (nodes) on a ring topology need to
reach consensus on their states while being influenced by
their neighbors.


Graph Structure
Ring topology: Each node connects to its immediate neighbors
(i → i+1 and i → i-1)
Padded representation: Uses fixed-size arrays with masks for
variable numbers of real nodes

State & Observations
Node features: [current_value, goal_value] for each node
Graph observation: Full graph with nodes, edges, and
connectivity information

Dynamics (Step Function)
x_next = x + alpha*(neighbor_mean - x) + beta*action + noise
Where:
alpha: Diffusion rate - how strongly neighbors influence each other
beta: Control strength - how much agent actions affect their state
neighbor_mean: Average of connected neighbors' current values
noise: Small random perturbations

Reward
Negative MSE: -∑(current_value - goal_value)²

This is a minimal testbed for:
Graph neural networks (GNNs) in RL
Multi-agent coordination algorithms
Consensus/diffusion problems
Message passing on graphs
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dgr.interface.graph_spec import Graph, GraphSpec, validate_graph


@dataclass(frozen=True)
class ConsensusConfig:
    spec: GraphSpec
    n_real: int
    horizon: int = 50
    alpha: float = 0.2  # diffusion rate
    beta: float = 0.5  # control strength
    noise_std: float = 0.01


@dataclass(frozen=True)
class EnvState:
    t: jnp.ndarray  # int32 scalar
    x: jnp.ndarray  # (N_max,) float32
    goal: jnp.ndarray  # (N_max,) float32


def _ring_edges(
    n_max: int, n_real: int, e_max: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Directed ring edges for the first n_real nodes:
      i -> i+1 and i -> i-1 (mod n_real)
    Total real edges: 2*n_real, padded to e_max.
    """
    e_real = 2 * n_real
    if e_real > e_max:
        raise ValueError(f"Need e_max >= 2*n_real, got e_max={e_max}, n_real={n_real}")

    i = jnp.arange(n_real, dtype=jnp.int32)
    s1 = i
    r1 = (i + 1) % n_real
    s2 = i
    r2 = (i - 1) % n_real

    senders_real = jnp.concatenate([s1, s2], axis=0)
    receivers_real = jnp.concatenate([r1, r2], axis=0)

    senders = jnp.zeros((e_max,), dtype=jnp.int32).at[:e_real].set(senders_real)
    receivers = jnp.zeros((e_max,), dtype=jnp.int32).at[:e_real].set(receivers_real)
    edge_mask = jnp.arange(e_max) < e_real
    return senders, receivers, edge_mask


def observe(cfg: ConsensusConfig, state: EnvState) -> Graph:
    spec = cfg.spec
    if spec.f_n != 2:
        raise ValueError("This toy env uses node features [x, goal], so GraphSpec.f_n must be 2.")

    node_mask = jnp.arange(spec.n_max) < cfg.n_real
    senders, receivers, edge_mask = _ring_edges(spec.n_max, cfg.n_real, spec.e_max)

    # Node features: [x, goal]
    nodes = jnp.stack([state.x, state.goal], axis=-1).astype(jnp.float32)  # (N_max, 2)
    nodes = jnp.where(node_mask[:, None], nodes, jnp.zeros_like(nodes))

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


def reset(key: jax.Array, cfg: ConsensusConfig) -> tuple[EnvState, Graph]:
    spec = cfg.spec
    if cfg.n_real > spec.n_max:
        raise ValueError("n_real must be <= n_max")

    k1, k2 = jax.random.split(key, 2)
    x0 = jax.random.normal(k1, (spec.n_max,), dtype=jnp.float32)
    goal = jax.random.normal(k2, (spec.n_max,), dtype=jnp.float32)

    node_mask = jnp.arange(spec.n_max) < cfg.n_real
    x0 = jnp.where(node_mask, x0, jnp.zeros_like(x0))
    goal = jnp.where(node_mask, goal, jnp.zeros_like(goal))

    state = EnvState(t=jnp.array(0, dtype=jnp.int32), x=x0, goal=goal)
    obs = observe(cfg, state)
    return state, obs


def step(
    key: jax.Array,
    cfg: ConsensusConfig,
    state: EnvState,
    action: jnp.ndarray,
) -> tuple[EnvState, Graph, jnp.ndarray, jnp.ndarray]:
    """
    action: (N_max,) float32 (controls each node; padded entries ignored via mask)
    returns: (new_state, obs, reward, done)
    """
    spec = cfg.spec
    if action.shape != (spec.n_max,):
        raise ValueError(f"Expected action shape {(spec.n_max,)}, got {action.shape}")

    node_mask = jnp.arange(spec.n_max) < cfg.n_real
    senders, receivers, edge_mask = _ring_edges(spec.n_max, cfg.n_real, spec.e_max)

    x = state.x
    u = jnp.where(node_mask, action.astype(jnp.float32), jnp.zeros_like(action, dtype=jnp.float32))

    # Neighbor mean via masked scatter-add
    msgs = x[senders] * edge_mask.astype(jnp.float32)
    agg = jnp.zeros((spec.n_max,), dtype=jnp.float32).at[receivers].add(msgs)
    deg = (
        jnp.zeros((spec.n_max,), dtype=jnp.float32).at[receivers].add(edge_mask.astype(jnp.float32))
    )
    neigh_mean = agg / jnp.maximum(deg, 1.0)

    k_noise = key
    noise = cfg.noise_std * jax.random.normal(k_noise, (spec.n_max,), dtype=jnp.float32)

    x_next = x + cfg.alpha * (neigh_mean - x) + cfg.beta * u + noise
    x_next = jnp.where(node_mask, x_next, jnp.zeros_like(x_next))

    t_next = state.t + jnp.array(1, dtype=jnp.int32)
    done = t_next >= jnp.array(cfg.horizon, dtype=jnp.int32)

    # Reward: negative MSE to goal over real nodes
    err = (x_next - state.goal) * node_mask.astype(jnp.float32)
    reward = -jnp.sum(err * err) / jnp.maximum(jnp.sum(node_mask.astype(jnp.float32)), 1.0)

    new_state = EnvState(t=t_next, x=x_next, goal=state.goal)
    obs = observe(cfg, new_state)
    return new_state, obs, reward, done
