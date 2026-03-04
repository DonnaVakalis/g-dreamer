from __future__ import annotations

import jax
import jax.numpy as jnp

from dgr.envs.suites.toy_graph_control.consensus import ConsensusConfig, reset, step
from dgr.interface.graph_spec import GraphSpec


def zero_action(x: jnp.ndarray, goal: jnp.ndarray, node_mask: jnp.ndarray) -> jnp.ndarray:
    del x, goal
    return jnp.where(node_mask, jnp.zeros_like(node_mask, dtype=jnp.float32), 0.0)


def random_action(
    key: jax.Array,
    x: jnp.ndarray,
    goal: jnp.ndarray,
    node_mask: jnp.ndarray,
    scale: float = 0.5,
) -> jnp.ndarray:
    del x, goal
    u = scale * jax.random.uniform(key, shape=node_mask.shape, minval=-1.0, maxval=1.0)
    return jnp.where(node_mask, u, jnp.zeros_like(u))


def proportional_action(
    x: jnp.ndarray,
    goal: jnp.ndarray,
    node_mask: jnp.ndarray,
    k: float = 0.5,
) -> jnp.ndarray:
    u = k * (goal - x)
    return jnp.where(node_mask, u, jnp.zeros_like(u))


def mse_to_goal(x: jnp.ndarray, goal: jnp.ndarray, node_mask: jnp.ndarray) -> jnp.ndarray:
    mask_f = node_mask.astype(jnp.float32)
    err = (x - goal) * mask_f
    return jnp.sum(err * err) / jnp.maximum(jnp.sum(mask_f), 1.0)


def rollout_controller(name: str, cfg: ConsensusConfig, reset_key: jax.Array) -> dict:
    # Reset with the SAME key for each controller so x0/goal are identical.
    state, obs = reset(reset_key, cfg)
    node_mask = obs.node_mask

    print(f"\n{'=' * 20} {name.upper()} {'=' * 20}")
    print("Initial x:", state.x)
    print("Goal     :", state.goal)
    print("Node mask:", node_mask)

    total_reward = 0.0
    start_mse = float(mse_to_goal(state.x, state.goal, node_mask))

    # Separate policy RNG for the random controller.
    policy_key = jax.random.PRNGKey(123)

    history = []

    for t in range(cfg.horizon):
        if name == "zero":
            action = zero_action(state.x, state.goal, node_mask)
        elif name == "random":
            policy_key, k_act = jax.random.split(policy_key)
            action = random_action(k_act, state.x, state.goal, node_mask)
        elif name == "proportional":
            action = proportional_action(state.x, state.goal, node_mask, k=0.5)
        else:
            raise ValueError(f"Unknown controller: {name}")

        # Env key is deterministic/reproducible too.
        step_key = jax.random.PRNGKey(t + 1)

        state, obs, reward, done = step(step_key, cfg, state, action)
        mse = mse_to_goal(state.x, state.goal, node_mask)

        total_reward += float(reward)
        history.append(
            {
                "t": t,
                "reward": float(reward),
                "mse": float(mse),
                "done": bool(done),
            }
        )

        print(
            f"Step {t + 1:02d} | "
            f"reward {float(reward):8.3f} | "
            f"mse {float(mse):8.3f} | "
            f"done {bool(done)}"
        )

        if done:
            break

    end_mse = float(mse_to_goal(state.x, state.goal, node_mask))

    return {
        "name": name,
        "start_mse": start_mse,
        "end_mse": end_mse,
        "total_reward": total_reward,
        "steps": len(history),
        "history": history,
    }


def main():
    spec = GraphSpec(n_max=8, e_max=16, f_n=2, f_e=0, f_g=0)
    cfg = ConsensusConfig(
        spec=spec,
        n_real=5,
        horizon=10,
        alpha=0.2,
        beta=0.5,
        noise_std=0.0,  # deterministic debugging
    )

    reset_key = jax.random.PRNGKey(0)

    results = [
        rollout_controller("zero", cfg, reset_key),
        rollout_controller("random", cfg, reset_key),
        rollout_controller("proportional", cfg, reset_key),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(
            f"{r['name']:>12} | "
            f"start_mse={r['start_mse']:8.3f} | "
            f"end_mse={r['end_mse']:8.3f} | "
            f"total_reward={r['total_reward']:9.3f} | "
            f"steps={r['steps']}"
        )


if __name__ == "__main__":
    main()
