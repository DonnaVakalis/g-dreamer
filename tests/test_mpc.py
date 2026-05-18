from __future__ import annotations

import jax
import jax.numpy as jnp

from dgr.control.mpc import CostFn, PlannerConfig, cem, random_shooting


def _quadratic_cost(target: jnp.ndarray) -> CostFn:
    def cost_fn(actions: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((actions - target) ** 2)

    return cost_fn


def test_random_shooting_beats_zero_plan():
    config = PlannerConfig(horizon=4, action_dim=5, population=4096)
    target = jnp.full((config.horizon, config.action_dim), 0.4, dtype=jnp.float32)
    cost_fn = _quadratic_cost(target)

    result = random_shooting(jax.random.PRNGKey(0), cost_fn, config)

    zero_cost = cost_fn(jnp.zeros((config.horizon, config.action_dim)))
    assert float(result.cost) < float(zero_cost)
    assert jnp.allclose(result.cost, cost_fn(result.actions), atol=1e-5)
    assert jnp.all(result.actions >= config.action_low)
    assert jnp.all(result.actions <= config.action_high)


def test_cem_beats_random_shooting_and_approaches_optimum():
    config = PlannerConfig(horizon=4, action_dim=5, population=512, cem_iters=8)
    target = jnp.full((config.horizon, config.action_dim), 0.4, dtype=jnp.float32)
    cost_fn = _quadratic_cost(target)
    key = jax.random.PRNGKey(0)

    rs = random_shooting(key, cost_fn, config)
    ce = cem(key, cost_fn, config)

    assert float(ce.cost) < float(rs.cost)
    assert float(ce.cost) < 1e-2  # convex problem — CEM should nearly solve it


def test_planner_respects_action_mask():
    config = PlannerConfig(horizon=3, action_dim=6, population=256, cem_iters=5)
    target = jnp.full((config.horizon, config.action_dim), 0.5, dtype=jnp.float32)
    cost_fn = _quadratic_cost(target)
    mask = jnp.array([True, True, True, False, False, False])

    for planner in (random_shooting, cem):
        result = planner(jax.random.PRNGKey(1), cost_fn, config, action_mask=mask)
        assert jnp.allclose(result.actions[:, ~mask], 0.0, atol=1e-6)
        assert jnp.allclose(result.cost, cost_fn(result.actions), atol=1e-5)


def test_planner_is_jittable():
    config = PlannerConfig(horizon=3, action_dim=4, population=128, cem_iters=3)
    target = jnp.full((config.horizon, config.action_dim), 0.2, dtype=jnp.float32)
    cost_fn = _quadratic_cost(target)

    for planner in (random_shooting, cem):
        jitted = jax.jit(lambda key, p=planner: p(key, cost_fn, config).cost)
        assert jnp.isfinite(jitted(jax.random.PRNGKey(0)))
