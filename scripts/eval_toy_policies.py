"""
This runs zero/random/proportional on a named scenario and writes:

scores.jsonl (same spirit as Dreamer)
summary.json (mean/std/etc.)

Example usage:
poetry run python scripts/eval_toy_policies.py \
train_ring_dense --episodes 50 --seed 0

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from dgr.envs.suites.toy_graph_control.controllers import (
    mse_to_goal,
    proportional_action,
    random_action,
    zero_action,
)
from dgr.envs.suites.toy_graph_control.core import reset, step
from dgr.envs.suites.toy_graph_control.scenarios import get_scenario


def run_episode(cfg, controller: str, seed: int, k_prop: float = 0.5) -> dict:
    key = jax.random.PRNGKey(seed)
    state, obs = reset(key, cfg)

    score = 0.0
    length = 0

    # policy rng for random controller
    policy_key = jax.random.PRNGKey(seed + 10_000)

    for t in range(cfg.dynamics.horizon):
        if controller == "zero":
            action = zero_action(state.node_mask)
        elif controller == "random":
            policy_key, k = jax.random.split(policy_key)
            action = random_action(k, state.node_mask)
        elif controller == "proportional":
            action = proportional_action(
                state.x, state.goal, state.node_mask, cfg.actuator_mask, k=k_prop
            )
        else:
            raise ValueError(controller)

        # deterministic per-step env key (reproducible)
        step_key = jax.random.PRNGKey(seed * 1_000_000 + (t + 1))
        state, obs, reward, done = step(step_key, cfg, state, action)

        score += float(reward)
        length += 1
        if bool(done):
            break

    start_mse = float(mse_to_goal(obs.nodes[:, 0], obs.nodes[:, 1], obs.node_mask))  # x vs goal
    end_mse = float(mse_to_goal(state.x, state.goal, state.node_mask))
    return {
        "episode/score": score,
        "episode/length": length,
        "start_mse": start_mse,
        "end_mse": end_mse,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenario", help="Scenario name (e.g., train_ring_dense, debug_ring_sparse)")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-prop", type=float, default=0.5)
    p.add_argument("--outdir", type=str, default="experiments/toy_eval")
    args = p.parse_args()

    cfg = get_scenario(args.scenario)

    outdir = Path(args.outdir) / args.scenario / f"seed_{args.seed:03d}"
    outdir.mkdir(parents=True, exist_ok=True)

    controllers = ["zero", "random", "proportional"]
    summary = {
        "scenario": args.scenario,
        "episodes": args.episodes,
        "seed": args.seed,
        "k_prop": args.k_prop,
    }

    for ctrl in controllers:
        rows = []
        scores = []
        lengths = []
        end_mses = []
        for ep in range(args.episodes):
            ep_seed = args.seed + ep
            row = run_episode(cfg, ctrl, ep_seed, k_prop=args.k_prop)
            rows.append(row)
            scores.append(row["episode/score"])
            lengths.append(row["episode/length"])
            end_mses.append(row["end_mse"])

        # write per-episode jsonl similar to Dreamer
        scores_path = outdir / f"{ctrl}__scores.jsonl"
        with scores_path.open("w") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {"episode/score": r["episode/score"], "episode/length": r["episode/length"]}
                    )
                    + "\n"
                )

        summary[ctrl] = {
            "mean_score": float(jnp.mean(jnp.array(scores))),
            "std_score": float(jnp.std(jnp.array(scores))),
            "mean_length": float(jnp.mean(jnp.array(lengths))),
            "mean_end_mse": float(jnp.mean(jnp.array(end_mses))),
            "scores_jsonl": str(scores_path),
        }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
