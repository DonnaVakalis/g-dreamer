from __future__ import annotations

import jax
import jax.numpy as jnp

from dgr.envs.suites.toy_graph_control.consensus import ConsensusConfig, reset, step
from dgr.interface.graph_spec import GraphSpec


def proportional_action(x: jnp.ndarray, goal: jnp.ndarray, node_mask: jnp.ndarray, k: float = 0.5):
    u = k * (goal - x)
    return jnp.where(node_mask, u, jnp.zeros_like(u))


def main():
    spec = GraphSpec(n_max=8, e_max=16, f_n=2, f_e=0, f_g=0)
    cfg = ConsensusConfig(
        spec=spec,
        n_real=5,
        horizon=10,
        alpha=0.2,
        beta=0.5,
        noise_std=0.0,  # deterministic
    )

    key = jax.random.PRNGKey(0)
    state, obs = reset(key, cfg)

    node_mask = obs.node_mask
    print("Initial x:", state.x)
    print("Goal     :", state.goal)
    print("Node mask:", node_mask)
    print("Obs vector shape:", obs.nodes.shape)

    total_reward = 0.0

    for t in range(cfg.horizon):
        action = proportional_action(state.x, state.goal, node_mask, k=0.5)

        key, subkey = jax.random.split(key)
        state, obs, reward, done = step(subkey, cfg, state, action)

        err = (state.x - state.goal) * node_mask.astype(jnp.float32)
        mse = jnp.sum(err * err) / jnp.maximum(jnp.sum(node_mask.astype(jnp.float32)), 1.0)

        total_reward += float(reward)

        print(f"\nStep {t + 1}")
        print("action :", action)
        print("x      :", state.x)
        print("reward :", float(reward))
        print("mse    :", float(mse))
        print("done   :", bool(done))

        if done:
            break

    print("\nTotal reward:", total_reward)


if __name__ == "__main__":
    main()
