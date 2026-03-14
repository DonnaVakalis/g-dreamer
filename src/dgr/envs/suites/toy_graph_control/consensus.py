from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from dgr.interface.graph_spec import GraphSpec

from .core import (
    DynamicsConfig,
    EnvState,
    ToyGraphControlConfig,
)
from .core import (
    observe as core_observe,
)
from .core import (
    reset as core_reset,
)
from .core import (
    step as core_step,
)


@dataclass(frozen=True)
class ConsensusConfig:
    spec: GraphSpec
    n_real: int
    horizon: int = 50
    alpha: float = 0.2
    beta: float = 0.5
    noise_std: float = 0.01


def _to_core(cfg: ConsensusConfig) -> ToyGraphControlConfig:
    actuator_mask = jnp.arange(cfg.spec.n_max) < cfg.n_real
    goal_obs_mask = getattr(cfg, "goal_obs_mask", None)
    if goal_obs_mask is None:
        goal_obs_mask = jnp.arange(cfg.spec.n_max) < cfg.n_real
    if goal_obs_mask.shape != (cfg.spec.n_max,):
        raise ValueError(
            f"Expected goal_obs_mask shape {(cfg.spec.n_max,)}, got {goal_obs_mask.shape}"
        )
    return ToyGraphControlConfig(
        spec=cfg.spec,
        n_real=cfg.n_real,
        dynamics=DynamicsConfig(
            horizon=cfg.horizon,
            alpha=cfg.alpha,
            beta=cfg.beta,
            noise_std=cfg.noise_std,
        ),
        actuator_mask=actuator_mask,
        goal_obs_mask=goal_obs_mask,
    )


def reset(key, cfg: ConsensusConfig):
    return core_reset(key, _to_core(cfg))


def step(key, cfg: ConsensusConfig, state: EnvState, action):
    return core_step(key, _to_core(cfg), state, action)


def observe(cfg: ConsensusConfig, state: EnvState):
    return core_observe(_to_core(cfg), state)
