"""
Tiny legacy-Gym shim around the pure JAX toy env.
"""

from __future__ import annotations

from dataclasses import dataclass

import gym
import jax
import jax.numpy as jnp
import numpy as np

from dgr.envs.suites.toy_graph_control.consensus import (
    ConsensusConfig,
)
from dgr.envs.suites.toy_graph_control.consensus import (
    reset as jax_reset,
)
from dgr.envs.suites.toy_graph_control.consensus import (
    step as jax_step,
)
from dgr.envs.wrappers.flatten_graph import flat_dim, flatten_graph
from dgr.interface.graph_spec import GraphSpec

ENV_ID = "DGRToyConsensus-v0"


def _make_default_config() -> ConsensusConfig:
    spec = GraphSpec(
        n_max=8,
        e_max=16,  # enough for ring with n_real <= 8 (needs 2 * n_real)
        f_n=2,
        f_e=0,
        f_g=0,
    )
    return ConsensusConfig(
        spec=spec,
        n_real=5,
        horizon=50,
        alpha=0.2,
        beta=0.5,
        noise_std=0.01,
    )


@dataclass
class ToyConsensusGymEnv(gym.Env):
    """
    Legacy-Gym shim around the pure JAX toy env.

    Observation:
      flat float32 vector from flatten_graph(...)
    Action:
      float32 vector of shape (n_max,)
    """

    seed_value: int = 0

    def __post_init__(self) -> None:
        self.cfg = _make_default_config()
        d = flat_dim(self.cfg.spec)

        self.observation_space = gym.spaces.Dict(
            {"vector": gym.spaces.Box(-np.inf, np.inf, (d,), np.float32)}
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.spec.n_max,),
            dtype=np.float32,
        )

        self._seed = int(self.seed_value)
        self._state = None

    def _next_key(self):
        self._seed += 1
        return jax.random.PRNGKey(self._seed)

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.seed_value = int(seed)
            self._seed = int(seed)
        return [self.seed_value]

    def reset(self):
        with jax.transfer_guard("allow"):
            k = self._next_key()
            self._state, g = jax_reset(k, self.cfg)
            vec = flatten_graph(g, self.cfg.spec)
            return {"vector": np.asarray(vec, dtype=np.float32)}

    def step(self, action):
        with jax.transfer_guard("allow"):
            if self._state is None:
                _ = self.reset()
            k = self._next_key()
            act = jnp.asarray(np.asarray(action, dtype=np.float32), dtype=jnp.float32)
            self._state, g, reward, done = jax_step(k, self.cfg, self._state, act)
            vec = flatten_graph(g, self.cfg.spec)
            return (
                {"vector": np.asarray(vec, dtype=np.float32)},
                float(reward),
                bool(done),
                {"is_terminal": bool(done)},
            )

    def render(self, mode="rgb_array"):
        # Minimal placeholder image; enough if anything calls render().
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        return None


def register_toy_consensus_env() -> None:
    from gym.envs.registration import register

    try:
        register(
            id=ENV_ID,
            entry_point=ToyConsensusGymEnv,
            disable_env_checker=True,
            apply_api_compatibility=False,
        )
    except Exception as exc:  # already-registered is fine
        msg = str(exc).lower()
        if "already registered" not in msg and "cannot re-register id" not in msg:
            raise

    # Touch registry once so import-time failures surface early.
    _ = gym.spec(ENV_ID)
