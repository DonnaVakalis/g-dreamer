from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gym")

from dgr.envs.adapters.toy_graph_control_gym import ENV_ID, register_toy_consensus_env  # noqa: E402


def test_toy_gym_env_reset_step():
    register_toy_consensus_env()
    env = gym.make(ENV_ID, disable_env_checker=True)

    obs = env.reset()
    assert obs.ndim == 1
    assert obs.dtype == np.float32

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs2, reward, done, info = env.step(action)

    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "is_terminal" in info
