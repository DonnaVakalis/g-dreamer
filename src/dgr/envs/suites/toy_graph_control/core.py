"""
Generalizes the toy graph control environment to different scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from dgr.interface.graph_spec import Graph, GraphSpec, validate_graph


@dataclass(frozen=True)
class DynamicsConfig:
    horizon: int = 50
    alpha: float = 0.2
    beta: float = 0.5
    noise_std: float = 0.01


@dataclass(frozen=True)
class GoalConfig:
    mode: str = "iid"  # "iid" or "smooth"
    smooth_steps: int = 0  # number of smoothing iterations
    residual_std: float = 0.0  # add iid noise after smoothing (keeps it imperfect)


@dataclass(frozen=True)
class ToyGraphControlConfig:
    spec: GraphSpec
    n_real: int
    dynamics: DynamicsConfig
    actuator_mask: jnp.ndarray  # (N_max,) bool
    goal_obs_mask: jnp.ndarray  # (N_max,) bool  — which nodes' goal is vis  in observation
    goal: GoalConfig = field(default_factory=GoalConfig)


@dataclass(frozen=True)
class EnvState:
    t: jnp.ndarray
    x: jnp.ndarray  # (N_max,)
    goal: jnp.ndarray  # (N_max,)
    senders: jnp.ndarray  # (E_max,)
    receivers: jnp.ndarray  # (E_max,)
    edge_mask: jnp.ndarray  # (E_max,)
    node_mask: jnp.ndarray  # (N_max,)


def make_ring_topology(
    spec: GraphSpec, n_real: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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


def _make_goal(
    key: jax.Array,
    spec: GraphSpec,
    node_mask: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    edge_mask: jnp.ndarray,
    goal_cfg: GoalConfig,
) -> jnp.ndarray:
    key0, key_res = jax.random.split(key, 2)

    goal = jax.random.normal(key0, (spec.n_max,), dtype=jnp.float32)
    goal = jnp.where(node_mask, goal, 0.0)

    if goal_cfg.mode == "iid":
        return goal

    if goal_cfg.mode == "smooth":
        g = goal
        for _ in range(int(goal_cfg.smooth_steps)):
            g = _neighbor_mean(g, senders, receivers, edge_mask)
            g = jnp.where(node_mask, g, 0.0)
        if goal_cfg.residual_std > 0:
            eps = goal_cfg.residual_std * jax.random.normal(
                key_res, (spec.n_max,), dtype=jnp.float32
            )
            g = jnp.where(node_mask, g + eps, 0.0)
        return g

    raise ValueError(f"Unknown goal_cfg.mode: {goal_cfg.mode!r}")


def _neighbor_mean(
    values: jnp.ndarray, senders: jnp.ndarray, receivers: jnp.ndarray, edge_mask: jnp.ndarray
) -> jnp.ndarray:
    n_max = values.shape[0]
    edge_mask_f = edge_mask.astype(jnp.float32)
    msgs = values[senders] * edge_mask_f
    agg = jnp.zeros((n_max,), dtype=jnp.float32).at[receivers].add(msgs)
    deg = jnp.zeros((n_max,), dtype=jnp.float32).at[receivers].add(edge_mask_f)
    return agg / jnp.maximum(deg, 1.0)


def observe(cfg: ToyGraphControlConfig, state: EnvState) -> Graph:
    spec = cfg.spec
    if spec.f_n != 2:
        raise ValueError("This env uses node features [x, goal], so GraphSpec.f_n must be 2.")

    visible_goal = (
        state.goal * cfg.goal_obs_mask.astype(jnp.float32) * state.node_mask.astype(jnp.float32)
    )
    nodes = jnp.stack([state.x, visible_goal], axis=-1).astype(jnp.float32)
    nodes = jnp.where(state.node_mask[:, None], nodes, jnp.zeros_like(nodes))

    edges = jnp.zeros((spec.e_max, spec.f_e), dtype=jnp.float32)
    glb = jnp.zeros((spec.f_g,), dtype=jnp.float32)

    g = Graph(
        nodes=nodes,
        edges=edges,
        senders=state.senders,
        receivers=state.receivers,
        node_mask=state.node_mask,
        edge_mask=state.edge_mask,
        globals=glb,
    )
    validate_graph(g, spec)
    return g


def reset(key: jax.Array, cfg: ToyGraphControlConfig) -> tuple[EnvState, Graph]:
    spec = cfg.spec
    if cfg.n_real > spec.n_max:
        raise ValueError("n_real must be <= n_max")
    if cfg.actuator_mask.shape != (spec.n_max,):
        raise ValueError(
            f"Expected actuator_mask shape {(spec.n_max,)}, got {cfg.actuator_mask.shape}"
        )
    if cfg.goal_obs_mask.shape != (spec.n_max,):
        raise ValueError(
            f"Expected goal_obs_mask shape {(spec.n_max,)}, got {cfg.goal_obs_mask.shape}"
        )

    node_mask = jnp.arange(spec.n_max) < cfg.n_real
    senders, receivers, edge_mask = make_ring_topology(spec, cfg.n_real)

    k1, k2 = jax.random.split(key, 2)
    x0 = jax.random.normal(k1, (spec.n_max,), dtype=jnp.float32)
    x0 = jnp.where(node_mask, x0, jnp.zeros_like(x0))
    goal = _make_goal(k2, spec, node_mask, senders, receivers, edge_mask, cfg.goal)

    state = EnvState(
        t=jnp.array(0, dtype=jnp.int32),
        x=x0,
        goal=goal,
        senders=senders,
        receivers=receivers,
        edge_mask=edge_mask.astype(jnp.bool_),
        node_mask=node_mask.astype(jnp.bool_),
    )
    return state, observe(cfg, state)


def step(
    key: jax.Array,
    cfg: ToyGraphControlConfig,
    state: EnvState,
    action: jnp.ndarray,
) -> tuple[EnvState, Graph, jnp.ndarray, jnp.ndarray]:
    spec = cfg.spec
    dyn = cfg.dynamics

    if action.shape != (spec.n_max,):
        raise ValueError(f"Expected action shape {(spec.n_max,)}, got {action.shape}")

    node_mask_f = state.node_mask.astype(jnp.float32)
    edge_mask_f = state.edge_mask.astype(jnp.float32)
    actuator_mask_f = cfg.actuator_mask.astype(jnp.float32) * node_mask_f

    x = state.x
    u = action.astype(jnp.float32) * actuator_mask_f

    msgs = x[state.senders] * edge_mask_f
    agg = jnp.zeros((spec.n_max,), dtype=jnp.float32).at[state.receivers].add(msgs)
    deg = jnp.zeros((spec.n_max,), dtype=jnp.float32).at[state.receivers].add(edge_mask_f)
    neigh_mean = agg / jnp.maximum(deg, 1.0)

    noise = dyn.noise_std * jax.random.normal(key, (spec.n_max,), dtype=jnp.float32)

    x_next = x + dyn.alpha * (neigh_mean - x) + dyn.beta * u + noise
    x_next = jnp.where(state.node_mask, x_next, jnp.zeros_like(x_next))

    t_next = state.t + jnp.array(1, dtype=jnp.int32)
    done = t_next >= jnp.array(dyn.horizon, dtype=jnp.int32)

    err = (x_next - state.goal) * node_mask_f
    reward = -jnp.sum(err * err) / jnp.maximum(jnp.sum(node_mask_f), 1.0)

    new_state = EnvState(
        t=t_next,
        x=x_next,
        goal=state.goal,
        senders=state.senders,
        receivers=state.receivers,
        edge_mask=state.edge_mask,
        node_mask=state.node_mask,
    )
    return new_state, observe(cfg, new_state), reward, done
