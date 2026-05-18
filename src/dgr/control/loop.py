"""Closed-loop MPC machinery (chunks 2c–2d): actors and an episode runner.

An *actor* maps ``(key, state, obs) -> action``. MPC actors plan with a dynamics model
each step and execute the first action of the plan; reactive actors (zero / random /
proportional) are the model-free baselines. ``run_episode`` drives any actor through one
episode of the real (stochastic) env.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from dgr.control.mpc import CostFn, PlannerConfig, PlannerResult
from dgr.control.wm_rollout import RolloutFn
from dgr.envs.suites.toy_graph_control.controllers import mse_to_goal, proportional_action
from dgr.envs.suites.toy_graph_control.core import EnvState, ToyGraphControlConfig, reset, step
from dgr.interface.graph_spec import Graph

Actor = Callable[[jax.Array, EnvState, Graph], jnp.ndarray]
Planner = Callable[[jax.Array, CostFn, PlannerConfig, jnp.ndarray | None], PlannerResult]


class EpisodeResult(NamedTuple):
    episode_return: jnp.ndarray  # sum of env reward over the episode
    final_mse: jnp.ndarray  # MSE-to-goal at the final step


def make_mpc_actor(
    rollout_fn: RolloutFn,
    planner: Planner,
    planner_config: PlannerConfig,
    action_mask: jnp.ndarray,
    node_mask: jnp.ndarray,
) -> Actor:
    """Receding-horizon actor: plan each step, execute the first action.

    The plan minimises the summed per-step MSE-to-goal of the predicted trajectory — the
    same quantity (negated) the env rewards. The goal is read from the observed nodes.
    """

    def plan(key: jax.Array, nodes: jnp.ndarray) -> jnp.ndarray:
        goal = nodes[:, 1]

        def cost_fn(action_seq: jnp.ndarray) -> jnp.ndarray:
            x_traj = rollout_fn(nodes, action_seq)
            return jnp.sum(jax.vmap(lambda x: mse_to_goal(x, goal, node_mask))(x_traj))

        return planner(key, cost_fn, planner_config, action_mask).actions[0]

    jitted = jax.jit(plan)

    def actor(key: jax.Array, state: EnvState, obs: Graph) -> jnp.ndarray:
        return jitted(key, obs.nodes)

    return actor


def zero_actor() -> Actor:
    def actor(key: jax.Array, state: EnvState, obs: Graph) -> jnp.ndarray:
        return jnp.zeros_like(state.x)

    return actor


def random_actor() -> Actor:
    def actor(key: jax.Array, state: EnvState, obs: Graph) -> jnp.ndarray:
        u = jax.random.uniform(key, state.x.shape, minval=-1.0, maxval=1.0, dtype=jnp.float32)
        return jnp.where(state.node_mask, u, 0.0)

    return actor


def make_proportional_actor(actuator_mask: jnp.ndarray, k: float = 0.5) -> Actor:
    def actor(key: jax.Array, state: EnvState, obs: Graph) -> jnp.ndarray:
        return proportional_action(
            obs.nodes[:, 0], obs.nodes[:, 1], state.node_mask, actuator_mask, k
        )

    return actor


def run_episode(key: jax.Array, cfg: ToyGraphControlConfig, actor: Actor) -> EpisodeResult:
    """Run one episode of the real env under ``actor``; report return and final error."""
    reset_key, key = jax.random.split(key)
    state, obs = reset(reset_key, cfg)
    total = jnp.array(0.0, dtype=jnp.float32)
    for _ in range(int(cfg.dynamics.horizon)):
        key, action_key, step_key = jax.random.split(key, 3)
        action = actor(action_key, state, obs)
        state, obs, reward, _ = step(step_key, cfg, state, action)
        total = total + reward
    final_mse = mse_to_goal(state.x, state.goal, state.node_mask)
    return EpisodeResult(episode_return=total, final_mse=final_mse)
