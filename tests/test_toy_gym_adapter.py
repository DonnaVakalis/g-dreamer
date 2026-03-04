from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gym")

from dgr.envs.adapters.toy_graph_control_gym import (  # noqa: E402
    env_id_for_scenario,
    register_toy_consensus_envs,
)


def test_toy_gym_env_reset_step():
    register_toy_consensus_envs()
    env = gym.make(
        env_id_for_scenario("debug_ring_dense"),
        disable_env_checker=True,
    )

    obs = env.reset()
    assert isinstance(obs, dict)
    assert "vector" in obs
    assert obs["vector"].ndim == 1
    assert obs["vector"].dtype == np.float32

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs2, reward, done, info = env.step(action)

    assert isinstance(obs2, dict)
    assert "vector" in obs2
    assert obs2["vector"].shape == obs["vector"].shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "is_terminal" in info
