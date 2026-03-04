"""
Here are some named reproducible presets
"""

from __future__ import annotations

import jax.numpy as jnp

from dgr.interface.graph_spec import GraphSpec

from .core import DynamicsConfig, ToyGraphControlConfig


def _dense_actuation(spec: GraphSpec, n_real: int) -> jnp.ndarray:
    return jnp.arange(spec.n_max) < n_real


def _sparse_actuation_even(spec: GraphSpec, n_real: int) -> jnp.ndarray:
    idx = jnp.arange(spec.n_max)
    return (idx < n_real) & ((idx % 2) == 0)


def get_scenario(name: str) -> ToyGraphControlConfig:
    spec = GraphSpec(n_max=8, e_max=16, f_n=2, f_e=0, f_g=0)

    if name == "debug_ring_dense":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(horizon=10, alpha=0.2, beta=0.5, noise_std=0.0),
            actuator_mask=_dense_actuation(spec, n_real),
        )

    if name == "debug_ring_sparse":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(horizon=10, alpha=0.2, beta=0.5, noise_std=0.0),
            actuator_mask=_sparse_actuation_even(spec, n_real),
        )

    if name == "train_ring_dense":
        n_real = 5
        return ToyGraphControlConfig(
            spec=spec,
            n_real=n_real,
            dynamics=DynamicsConfig(horizon=50, alpha=0.2, beta=0.5, noise_std=0.01),
            actuator_mask=_dense_actuation(spec, n_real),
        )

    raise ValueError(f"Unknown scenario: {name}")
