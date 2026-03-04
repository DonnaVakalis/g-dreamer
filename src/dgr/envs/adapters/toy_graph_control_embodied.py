"""
Minimal Embodied-style env adapter for the toy graph control environment.
to actually run Dreamer on the toy env: Dreamer’s env loop expects obs
dict keys like reward/is_first/is_last/is_terminal and action dict includes reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from dgr.envs.suites.toy_graph_control.consensus import ConsensusConfig, EnvState
from dgr.envs.suites.toy_graph_control.consensus import reset as jax_reset
from dgr.envs.suites.toy_graph_control.consensus import step as jax_step
from dgr.envs.wrappers.flatten_graph import flat_dim, flatten_graph


@dataclass
class ToyGraphControlEmbodied:
    """
    Minimal Embodied-style env adapter.

    - step(action_dict) where action_dict has: {"action": np.ndarray, "reset": bool}
    - returns obs dict with:
        vector (float32[D]), reward (float32), is_first/is_last/is_terminal (bool)
    """

    cfg: ConsensusConfig
    seed: int = 0

    def __post_init__(self) -> None:
        # Safe under global jax_transfer_guard=disallow setups.
        with jax.transfer_guard("allow"):
            self._key = jax.random.PRNGKey(self.seed)
        self._state: EnvState | None = None
        self._done = True

    @property
    def obs_space(self):
        import elements  # optional dependency

        d = flat_dim(self.cfg.spec)
        return {
            "vector": elements.Space(np.float32, (d,), -np.inf, np.inf),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    @property
    def act_space(self):
        import elements  # optional dependency

        n = self.cfg.spec.n_max
        return {
            "action": elements.Space(np.float32, (n,), -1.0, 1.0),
            "reset": elements.Space(bool),
        }

    def close(self):
        return

    def step(self, action: dict):
        with jax.transfer_guard("allow"):
            do_reset = bool(action.get("reset", False))
            act = action.get("action", np.zeros((self.cfg.spec.n_max,), np.float32))
            act = jnp.asarray(act, dtype=jnp.float32)

            if do_reset or self._done or self._state is None:
                self._key, k = jax.random.split(self._key)
                self._state, g = jax_reset(k, self.cfg)
                self._done = False
                vec = flatten_graph(g, self.cfg.spec)
                return {
                    "vector": np.asarray(vec, dtype=np.float32),
                    "reward": np.float32(0.0),
                    "is_first": True,
                    "is_last": False,
                    "is_terminal": False,
                }

            self._key, k = jax.random.split(self._key)
            self._state, g, rew, done = jax_step(k, self.cfg, self._state, act)
            self._done = bool(done)
            vec = flatten_graph(g, self.cfg.spec)
            return {
                "vector": np.asarray(vec, dtype=np.float32),
                "reward": np.asarray(rew, dtype=np.float32),
                "is_first": False,
                "is_last": bool(done),
                "is_terminal": bool(done),
            }
