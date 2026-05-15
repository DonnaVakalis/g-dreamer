"""
Tiny legacy-Gym shim around the pure JAX toy env.
"""

from __future__ import annotations

from dataclasses import dataclass

import gym
import jax
import jax.numpy as jnp
import numpy as np

from dgr.envs.suites.toy_graph_control.core import (
    reset as jax_reset,
)
from dgr.envs.suites.toy_graph_control.core import (
    step as jax_step,
)
from dgr.envs.suites.toy_graph_control.scenarios import (
    get_scenario,
)
from dgr.envs.wrappers.flatten_graph import flat_dim, flatten_graph

SCENARIO_TO_ENV_ID = {
    "debug_ring_dense": "DGRToyConsensusDebugDense-v0",
    "debug_ring_sparse": "DGRToyConsensusDebugSparse-v0",
    "train_ring_dense": "DGRToyConsensusTrainDense-v0",
    "train_ring_sparse_hidden_smooth_aligned": "DGRToyConsensusTrainSparseHiddenSmoothAligned-v0",
    "train_ring_sparse_hidden_smooth_misaligned": (
        "DGRToyConsensusTrainSparseHiddenSmoothMisaligned-v0"
    ),
}


@dataclass
class ToyConsensusGymEnv(gym.Env):
    """
    Legacy-Gym shim around the pure JAX toy env.

    Observation:
      flat float32 vector from flatten_graph(...)
    Action:
      float32 vector of shape (n_max,)
    """

    metadata = {"render_modes": ["rgb_array"]}  # minimal placeholder
    seed_value: int = 0
    scenario_name: str = "debug_ring_dense"  # e.g., train_ring_dense, debug_ring_sparse
    include_graph_obs: bool = False

    def __post_init__(self) -> None:
        # DreamerV3 enables jax_transfer_guard=disallow; allow explicit transfers
        # during env construction where scenario config builds JAX arrays.
        with jax.transfer_guard("allow"):
            self.cfg = get_scenario(self.scenario_name)
        d = flat_dim(self.cfg.spec)

        obs_spaces = {
            "vector": gym.spaces.Box(-np.inf, np.inf, (d,), np.float32),
            "log/end_mse": gym.spaces.Box(-np.inf, np.inf, (), np.float32),
        }
        if self.include_graph_obs:
            obs_spaces.update(
                {
                    "nodes": gym.spaces.Box(
                        -np.inf,
                        np.inf,
                        (self.cfg.spec.n_max, self.cfg.spec.f_n),
                        np.float32,
                    ),
                    "senders": gym.spaces.Box(
                        low=0,
                        high=self.cfg.spec.n_max - 1,
                        shape=(self.cfg.spec.e_max,),
                        dtype=np.int32,
                    ),
                    "receivers": gym.spaces.Box(
                        low=0,
                        high=self.cfg.spec.n_max - 1,
                        shape=(self.cfg.spec.e_max,),
                        dtype=np.int32,
                    ),
                    "node_mask": gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.cfg.spec.n_max,),
                        dtype=np.float32,
                    ),
                    "edge_mask": gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.cfg.spec.e_max,),
                        dtype=np.float32,
                    ),
                }
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)
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
            obs = {"vector": np.asarray(vec, dtype=np.float32), "log/end_mse": np.float32(0.0)}
            if self.include_graph_obs:
                obs.update(self._graph_obs(g))
            return obs

    def step(self, action):
        with jax.transfer_guard("allow"):
            if self._state is None:
                _ = self.reset()
            k = self._next_key()
            act = jnp.asarray(np.asarray(action, dtype=np.float32), dtype=jnp.float32)
            self._state, g, reward, done = jax_step(k, self.cfg, self._state, act)
            vec = flatten_graph(g, self.cfg.spec)
            obs = {
                "vector": np.asarray(vec, dtype=np.float32),
                "log/end_mse": np.float32(-reward if done else 0.0),
            }
            if self.include_graph_obs:
                obs.update(self._graph_obs(g))
            return (obs, float(reward), bool(done), {"is_terminal": bool(done)})

    def render(self, mode="rgb_array"):
        # Minimal placeholder image; enough if anything calls render().
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        return None

    def _graph_obs(self, g) -> dict[str, np.ndarray]:
        # The structured graph fields are optional because the legacy baseline
        # only consumes the flattened vector. The graph-encoder Dreamer variant
        # requests these keys explicitly so it can perform message passing while
        # still keeping the reconstruction target identical to the baseline.
        return {
            "nodes": np.asarray(g.nodes, dtype=np.float32),
            "senders": np.asarray(g.senders, dtype=np.int32),
            "receivers": np.asarray(g.receivers, dtype=np.int32),
            "node_mask": np.asarray(g.node_mask, dtype=np.float32),
            "edge_mask": np.asarray(g.edge_mask, dtype=np.float32),
        }


def env_id_for_scenario(scenario_name: str) -> str:
    try:
        return SCENARIO_TO_ENV_ID[scenario_name]
    except KeyError as exc:
        raise ValueError(f"Unknown scenario_name: {scenario_name}") from exc


def register_toy_consensus_envs() -> None:
    from gym.envs.registration import register

    for scenario_name, env_id in SCENARIO_TO_ENV_ID.items():
        try:
            register(
                id=env_id,
                entry_point=ToyConsensusGymEnv,
                kwargs={"scenario_name": scenario_name},
                disable_env_checker=True,
                apply_api_compatibility=False,
            )
        except Exception as exc:  # already-registered is fine
            msg = str(exc).lower()
            if "already registered" not in msg and "cannot re-register id" not in msg:
                raise

    _ = gym.spec(env_id)
