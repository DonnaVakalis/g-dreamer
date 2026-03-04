from __future__ import annotations

import json
from pathlib import Path

import jax

from dgr.envs.suites.toy_graph_control.controllers import (
    mse_to_goal,
    proportional_action,
    random_action,
    zero_action,
)
from dgr.envs.suites.toy_graph_control.core import reset, step
from dgr.envs.suites.toy_graph_control.scenarios import get_scenario


def rollout(name: str, scenario_name: str, seed: int = 0):
    cfg = get_scenario(scenario_name)
    key = jax.random.PRNGKey(seed)
    state, obs = reset(key, cfg)

    history = []
    total_reward = 0.0
    start_mse = float(mse_to_goal(state.x, state.goal, state.node_mask))

    policy_key = jax.random.PRNGKey(123)

    for t in range(cfg.dynamics.horizon):
        if name == "zero":
            action = zero_action(state.node_mask)
        elif name == "random":
            policy_key, k = jax.random.split(policy_key)
            action = random_action(k, state.node_mask)
        elif name == "proportional":
            action = proportional_action(
                state.x, state.goal, state.node_mask, cfg.actuator_mask, k=0.5
            )
        else:
            raise ValueError(name)

        step_key = jax.random.PRNGKey(t + 1)
        state, obs, reward, done = step(step_key, cfg, state, action)
        mse = mse_to_goal(state.x, state.goal, state.node_mask)

        history.append({"t": t, "reward": float(reward), "mse": float(mse), "done": bool(done)})
        total_reward += float(reward)
        if done:
            break

    return {
        "controller": name,
        "scenario": scenario_name,
        "start_mse": start_mse,
        "end_mse": float(mse_to_goal(state.x, state.goal, state.node_mask)),
        "total_reward": total_reward,
        "steps": len(history),
        "history": history,
    }


def main():
    scenario_name = "debug_ring_sparse"  # debug_ring_dense, debug_ring_sparse
    outdir = Path("experiments/toy_debug") / scenario_name / "seed_000"
    outdir.mkdir(parents=True, exist_ok=True)

    results = [
        rollout("zero", scenario_name, seed=0),
        rollout("random", scenario_name, seed=0),
        rollout("proportional", scenario_name, seed=0),
    ]

    for r in results:
        with (outdir / f"{r['controller']}.json").open("w") as f:
            json.dump(r, f, indent=2)

    summary = {
        "scenario": scenario_name,
        "results": [
            {
                "controller": r["controller"],
                "start_mse": r["start_mse"],
                "end_mse": r["end_mse"],
                "total_reward": r["total_reward"],
                "steps": r["steps"],
            }
            for r in results
        ],
    }

    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
