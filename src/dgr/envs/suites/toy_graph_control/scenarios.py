"""
Here are some named reproducible presets
"""

from __future__ import annotations

import jax.numpy as jnp

from dgr.interface.graph_spec import GraphSpec

from .core import DynamicsConfig, GoalConfig, ToyGraphControlConfig


def _dense_actuation(spec: GraphSpec, n_real: int) -> jnp.ndarray:
    return jnp.arange(spec.n_max) < n_real


def _sparse_actuation_even(spec: GraphSpec, n_real: int) -> jnp.ndarray:
    idx = jnp.arange(spec.n_max)
    return (idx < n_real) & ((idx % 2) == 0)


def _all_goals_visible(spec: GraphSpec, n_real: int) -> jnp.ndarray:
    return jnp.arange(spec.n_max) < n_real


def _leader_goals_visible(spec: GraphSpec, n_real: int, leaders: int = 1) -> jnp.ndarray:
    # Only the first `leaders` nodes (among real nodes) have goal visible.
    idx = jnp.arange(spec.n_max)
    return idx < leaders  # leaders are always within real nodes if leaders <= n_real


def _leaders_spaced(spec: GraphSpec, n_real: int, leaders: int) -> jnp.ndarray:
    # Choose roughly evenly spaced leaders among real nodes.
    if leaders <= 0:
        return jnp.zeros((spec.n_max,), dtype=jnp.bool_)
    idx = jnp.linspace(0, max(n_real - 1, 0), leaders).round().astype(jnp.int32)
    mask = jnp.zeros((spec.n_max,), dtype=jnp.bool_).at[idx].set(True)
    return mask


def _leaders_spaced_on_parity(
    spec: GraphSpec, n_real: int, leaders: int, parity: int
) -> jnp.ndarray:
    """
    Choose `leaders` visible-goal nodes among indices < n_real with given parity (0 even, 1 odd),
    roughly evenly spaced. Returns bool mask of shape (N_max,).
    """
    if leaders <= 0:
        return jnp.zeros((spec.n_max,), dtype=jnp.bool_)

    parity = int(parity) & 1
    # candidate indices: parity, parity+2, ...
    num = (n_real - parity + 1) // 2  # how many candidates exist
    num = max(int(num), 0)

    if num == 0:
        return jnp.zeros((spec.n_max,), dtype=jnp.bool_)

    cand = parity + 2 * jnp.arange(num, dtype=jnp.int32)  # shape (num,)

    # pick evenly spaced indices into cand
    sel = jnp.linspace(0, num - 1, leaders).round().astype(jnp.int32)
    sel = jnp.clip(sel, 0, num - 1)
    idx = cand[sel]  # leader node indices

    mask = jnp.zeros((spec.n_max,), dtype=jnp.bool_)
    mask = mask.at[idx].set(True)
    return mask


def make_consensus_config(
    n_real: int,
    *,
    n_max: int | None = None,
    horizon: int = 50,
    alpha: float = 0.2,
    beta: float = 0.5,
    noise_std: float = 0.01,
) -> ToyGraphControlConfig:
    """
    Build a simple dense-actuation, full-goal-visibility consensus config for arbitrary graph sizes.

    This is the minimal setting used by the graph world model sprint so we can train on
    multiple graph sizes without minting a separate named scenario for each one.
    """
    if n_real <= 0:
        raise ValueError(f"n_real must be positive, got {n_real}")
    n_max = n_real if n_max is None else n_max
    if n_max < n_real:
        raise ValueError(f"n_max must be >= n_real, got n_max={n_max}, n_real={n_real}")

    spec = GraphSpec(n_max=n_max, e_max=2 * n_max, f_n=2, f_e=0, f_g=0)
    return ToyGraphControlConfig(
        spec=spec,
        n_real=n_real,
        dynamics=DynamicsConfig(
            mode="consensus",
            horizon=horizon,
            alpha=alpha,
            beta=beta,
            noise_std=noise_std,
        ),
        actuator_mask=_dense_actuation(spec, n_real),
        goal_obs_mask=_all_goals_visible(spec, n_real),
    )


def get_scenario(name: str) -> ToyGraphControlConfig:
    spec = GraphSpec(n_max=8, e_max=16, f_n=2, f_e=0, f_g=0)

    if name == "debug_ring_dense":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=20, alpha=0.2, beta=0.5, noise_std=0.0
            ),
            actuator_mask=_dense_actuation(spec, n_real),
            goal_obs_mask=_all_goals_visible(spec, n_real),
        )

    if name == "debug_ring_sparse":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=20, alpha=0.2, beta=0.5, noise_std=0.0
            ),
            actuator_mask=_sparse_actuation_even(spec, n_real),
            goal_obs_mask=_all_goals_visible(spec, n_real),
        )

    if name == "train_ring_dense":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=50, alpha=0.2, beta=0.5, noise_std=0.01
            ),
            actuator_mask=_dense_actuation(spec, n_real),
            goal_obs_mask=_all_goals_visible(spec, n_real),
        )

    if name == "debug_ring_hidden_goal_leader1":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=20, alpha=0.2, beta=0.5, noise_std=0.0
            ),
            actuator_mask=_dense_actuation(spec, n_real),
            goal_obs_mask=_leader_goals_visible(spec, n_real, leaders=1),
        )

    if name == "train_ring_hidden_goal_leader1":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=50, alpha=0.2, beta=0.5, noise_std=0.01
            ),
            actuator_mask=_dense_actuation(spec, n_real),
            goal_obs_mask=_leader_goals_visible(spec, n_real, leaders=1),
        )

    if name == "debug_ring_hidden_goal_leader3_smooth":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=10, alpha=0.2, beta=0.5, noise_std=0.0
            ),
            actuator_mask=_dense_actuation(spec, n_real),
            goal_obs_mask=_leaders_spaced(spec, n_real, leaders=3),
            goal=GoalConfig(mode="smooth", smooth_steps=8, residual_std=0.1),
        )

    if name == "train_ring_hidden_goal_leader3_smooth":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=50, alpha=0.2, beta=0.5, noise_std=0.01
            ),
            actuator_mask=_dense_actuation(spec, n_real),
            goal_obs_mask=_leaders_spaced(spec, n_real, leaders=3),
            goal=GoalConfig(mode="smooth", smooth_steps=8, residual_std=0.1),
        )

    if name == "debug_ring_sparse_hidden_goal_leader3_smooth":
        n_real = 8
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=50, alpha=0.2, beta=1.0, noise_std=0.0
            ),
            actuator_mask=_sparse_actuation_even(spec, n_real),
            goal_obs_mask=_leaders_spaced(spec, n_real, leaders=3),
            goal=GoalConfig(mode="smooth", smooth_steps=8, residual_std=0.05),
        )

    if name == "train_ring_sparse_hidden_goal_leader3_smooth":
        n_real = 8
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=50, alpha=0.2, beta=0.5, noise_std=0.01
            ),
            actuator_mask=_sparse_actuation_even(spec, n_real),
            goal_obs_mask=_leaders_spaced(spec, n_real, leaders=3),
            goal=GoalConfig(mode="smooth", smooth_steps=8, residual_std=0.1),
        )

    if name == "debug_ring_sparse_hidden_smooth_aligned":
        n_real = 8
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=30, alpha=0.2, beta=1.0, noise_std=0.0
            ),
            actuator_mask=_sparse_actuation_even(spec, n_real),  # actuated = evens
            goal_obs_mask=_leaders_spaced_on_parity(
                spec, n_real, leaders=3, parity=0
            ),  # leaders on evens
            goal=GoalConfig(mode="clamped_smooth", smooth_steps=8, residual_std=0.05),
        )

    if name == "debug_ring_sparse_hidden_smooth_misaligned":
        n_real = 8
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=30, alpha=0.2, beta=1.0, noise_std=0.0
            ),
            actuator_mask=_sparse_actuation_even(spec, n_real),  # actuated = evens
            goal_obs_mask=_leaders_spaced_on_parity(
                spec, n_real, leaders=3, parity=1
            ),  # leaders on odds
            goal=GoalConfig(mode="clamped_smooth", smooth_steps=8, residual_std=0.05),
        )

    if name == "train_ring_sparse_hidden_smooth_aligned":
        n_real = 8
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=50, alpha=0.2, beta=0.5, noise_std=0.01
            ),
            actuator_mask=_sparse_actuation_even(spec, n_real),
            goal_obs_mask=_leaders_spaced_on_parity(spec, n_real, leaders=3, parity=0),
            goal=GoalConfig(mode="clamped_smooth", smooth_steps=8, residual_std=0.1),
        )

    if name == "train_ring_sparse_hidden_smooth_misaligned":
        n_real = 8
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=50, alpha=0.2, beta=0.5, noise_std=0.01
            ),
            actuator_mask=_sparse_actuation_even(spec, n_real),
            goal_obs_mask=_leaders_spaced_on_parity(spec, n_real, leaders=3, parity=1),
            goal=GoalConfig(mode="clamped_smooth", smooth_steps=8, residual_std=0.1),
        )

    if name == "eval_ring_dense":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=50, alpha=0.2, beta=0.5, noise_std=0.01
            ),
            actuator_mask=_dense_actuation(spec, n_real),
            goal_obs_mask=_all_goals_visible(spec, n_real),
        )

    if name == "eval_ring_dense_ood":
        n_real = 7
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(
                mode="consensus", horizon=50, alpha=0.2, beta=0.5, noise_std=0.01
            ),
            actuator_mask=_dense_actuation(spec, n_real),
            goal_obs_mask=_all_goals_visible(spec, n_real),
        )

    raise ValueError(f"Unknown scenario: {name}")


def scenario_stats(cfg: ToyGraphControlConfig) -> dict:
    """Useful diagnostics for interpreting masked vs inferred controllers."""
    node_mask = jnp.arange(cfg.spec.n_max) < cfg.n_real
    a = cfg.actuator_mask & node_mask
    v = cfg.goal_obs_mask & node_mask
    o = a & v
    n_act = int(jnp.sum(a))
    n_vis = int(jnp.sum(v))
    n_ovl = int(jnp.sum(o))
    return {
        "n_real": int(cfg.n_real),
        "n_actuated": n_act,
        "n_visible": n_vis,
        "n_overlap": n_ovl,
        "overlap_ratio_actuated": float(n_ovl / max(n_act, 1)),
        "overlap_ratio_visible": float(n_ovl / max(n_vis, 1)),
    }
