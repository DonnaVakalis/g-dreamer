"""Model-agnostic receding-horizon planners (Part 2, chunk 2a).

A planner optimises an action sequence of shape ``(horizon, action_dim)`` against a
scalar ``cost_fn``. It knows nothing about graphs, world models, or rewards — the
caller supplies ``cost_fn`` (typically a closure that rolls a world model forward and
scores the trajectory). The MPC loop executes only the first action of the returned
plan, then replans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

# Maps an action sequence (horizon, action_dim) to a scalar cost.
CostFn = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int
    action_dim: int
    population: int = 512
    action_low: float = -1.0
    action_high: float = 1.0
    cem_iters: int = 5
    elite_frac: float = 0.1
    cem_init_std: float = 0.5
    cem_min_std: float = 0.05


class PlannerResult(NamedTuple):
    actions: jnp.ndarray  # (horizon, action_dim)
    cost: jnp.ndarray  # scalar


def _apply_action_mask(actions: jnp.ndarray, action_mask: jnp.ndarray | None) -> jnp.ndarray:
    # action_mask has shape (action_dim,); masked coordinates are forced to zero so the
    # planner never proposes control on padding / non-actuator nodes.
    if action_mask is None:
        return actions
    return jnp.where(action_mask, actions, 0.0)


def random_shooting(
    key: jax.Array,
    cost_fn: CostFn,
    config: PlannerConfig,
    action_mask: jnp.ndarray | None = None,
) -> PlannerResult:
    """Sample ``population`` action sequences uniformly and return the cheapest."""
    samples = jax.random.uniform(
        key,
        (config.population, config.horizon, config.action_dim),
        minval=config.action_low,
        maxval=config.action_high,
        dtype=jnp.float32,
    )
    samples = _apply_action_mask(samples, action_mask)
    costs = jax.vmap(cost_fn)(samples)
    best = jnp.argmin(costs)
    return PlannerResult(actions=samples[best], cost=costs[best])


def cem(
    key: jax.Array,
    cost_fn: CostFn,
    config: PlannerConfig,
    action_mask: jnp.ndarray | None = None,
) -> PlannerResult:
    """Cross-entropy method: refine a per-timestep Gaussian over ``cem_iters`` rounds."""
    n_elite = max(1, int(config.population * config.elite_frac))
    shape = (config.horizon, config.action_dim)
    init_mean = jnp.zeros(shape, dtype=jnp.float32)
    init_std = jnp.full(shape, config.cem_init_std, dtype=jnp.float32)

    def body(
        carry: tuple[jax.Array, jnp.ndarray, jnp.ndarray], _: None
    ) -> tuple[tuple[jax.Array, jnp.ndarray, jnp.ndarray], None]:
        key, mean, std = carry
        key, sample_key = jax.random.split(key)
        noise = jax.random.normal(sample_key, (config.population, *shape), dtype=jnp.float32)
        samples = jnp.clip(mean + std * noise, config.action_low, config.action_high)
        samples = _apply_action_mask(samples, action_mask)
        costs = jax.vmap(cost_fn)(samples)
        elite = samples[jnp.argsort(costs)[:n_elite]]
        mean = elite.mean(axis=0)
        std = jnp.maximum(elite.std(axis=0), config.cem_min_std)
        return (key, mean, std), None

    (_, mean, _), _ = lax.scan(body, (key, init_mean, init_std), None, length=config.cem_iters)
    plan = _apply_action_mask(jnp.clip(mean, config.action_low, config.action_high), action_mask)
    return PlannerResult(actions=plan, cost=cost_fn(plan))
