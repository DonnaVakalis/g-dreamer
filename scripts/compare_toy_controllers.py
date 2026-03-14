"""
Compares the performance of different toy controllers
on the toy graph control environment.

Run any scenario + seed from the command line,
and optionally write outputs to a predictable folder.

Example usage:
poetry run python scripts/compare_toy_controllers.py debug_ring_sparse \
    --seed 0 --outdir experiments/toy_debug/debug_ring_sparse/seed_000/

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from dgr.envs.suites.toy_graph_control.controllers import (
    inferred_goal_proportional_action,
    masked_proportional_action,
    mse_to_goal,
    proportional_action,
    random_action,
    zero_action,
)
from dgr.envs.suites.toy_graph_control.core import reset, step
from dgr.envs.suites.toy_graph_control.scenarios import get_scenario


def rollout(name: str, scenario_name: str, seed: int = 0, k_prop: float = 0.5):
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
                state.x, state.goal, state.node_mask, cfg.actuator_mask, k=k_prop
            )
        elif name == "masked_proportional":
            action = masked_proportional_action(
                state.x, state.goal, state.node_mask, cfg.actuator_mask, cfg.goal_obs_mask, k=k_prop
            )
        elif name == "inferred_proportional":
            # visible_goal is what the agent sees (0 where hidden)
            visible_goal = obs.nodes[:, 1]
            action = inferred_goal_proportional_action(
                state.x,
                visible_goal,
                state.node_mask,
                cfg.actuator_mask,
                cfg.goal_obs_mask,
                state.senders,
                state.receivers,
                state.edge_mask,
                iters=8,
                k=k_prop,
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
        "seed": seed,
        "k_prop": k_prop,
        "start_mse": start_mse,
        "end_mse": float(mse_to_goal(state.x, state.goal, state.node_mask)),
        "total_reward": total_reward,
        "steps": len(history),
        "history": history,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenario", help="Scenario name from scenarios.get_scenario()")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-prop", type=float, default=0.5)
    p.add_argument(
        "--outdir",
        type=str,
        default="experiments/toy_debug",
        help="Base output dir (scenario/seed_xxx will be created under this).",
    )
    p.add_argument("--no-write", action="store_true", help="Do not write json files.")
    args = p.parse_args()

    scenario_name = args.scenario
    seed = args.seed
    k_prop = args.k_prop

    results = [
        rollout("zero", scenario_name, seed=seed, k_prop=k_prop),
        rollout("random", scenario_name, seed=seed, k_prop=k_prop),
        rollout("proportional", scenario_name, seed=seed, k_prop=k_prop),
        rollout("masked_proportional", scenario_name, seed=seed, k_prop=k_prop),
        rollout("inferred_proportional", scenario_name, seed=seed, k_prop=k_prop),
    ]

    summary = {
        "scenario": scenario_name,
        "seed": seed,
        "k_prop": k_prop,
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

    print(json.dumps(summary, indent=2))

    if not args.no_write:
        outdir = Path(args.outdir) / scenario_name / f"seed_{seed:03d}"
        outdir.mkdir(parents=True, exist_ok=True)

        for r in results:
            (outdir / f"{r['controller']}.json").write_text(json.dumps(r, indent=2))
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
