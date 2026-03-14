import jax
import jax.numpy as jnp

from dgr.envs.suites.toy_graph_control.core import reset
from dgr.envs.suites.toy_graph_control.scenarios import get_scenario


def test_goal_mask_hides_goals_in_observation():
    cfg = get_scenario("debug_ring_hidden_goal_leader1")
    key = jax.random.PRNGKey(0)
    state, obs = reset(key, cfg)

    # obs.nodes[:, 1] is visible goal
    visible_goal = obs.nodes[:, 1]
    # state.goal is true goal
    true_goal = state.goal

    # For nodes where goal_obs_mask is False (but node is real), visible goal should be 0.
    hidden_real = (~cfg.goal_obs_mask) & obs.node_mask
    assert jnp.all(visible_goal[hidden_real] == 0.0)

    # For visible goal nodes, it should match true goal (for real nodes).
    visible_real = cfg.goal_obs_mask & obs.node_mask
    assert jnp.allclose(visible_goal[visible_real], true_goal[visible_real])
