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
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

import wandb
from dgr.envs.suites.toy_graph_control.controllers import (
    inferred_goal_proportional_action,
    masked_proportional_action,
    mse_on_mask,
    mse_to_goal,
    proportional_action,
    random_action,
    zero_action,
)
from dgr.envs.suites.toy_graph_control.core import reset, step
from dgr.envs.suites.toy_graph_control.scenarios import get_scenario, scenario_stats
from dgr.experiments.metadata import ExperimentMetadata
from dgr.experiments.naming import canonical_policy_name, controller_eval_dir
from dgr.experiments.wandb_utils import wandb_init_kwargs

_EPISODE_LOG_KEYS = (
    "controller",
    "scenario",
    "seed",
    "episode",
    "k_prop",
    "start_mse",
    "end_mse",
    "start_mse_ctrl",
    "end_mse_ctrl",
    "start_mse_unact",
    "end_mse_unact",
    "total_reward",
    "steps",
)


def compact_episode_row(row: dict) -> dict:
    """Per-episode metrics without step history (smaller JSON, publication-friendly)."""
    return {k: row[k] for k in _EPISODE_LOG_KEYS if k in row}


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rollout(name: str, scenario_name: str, cfg, seed: int = 0, k_prop: float = 0.5):
    key = jax.random.PRNGKey(seed)
    state, obs = reset(key, cfg)

    history = []
    total_reward = 0.0
    start_mse = float(mse_to_goal(state.x, state.goal, state.node_mask))
    start_mse_ctrl = float(mse_on_mask(state.x, state.goal, state.node_mask & cfg.actuator_mask))
    start_mse_unact = float(mse_to_goal(state.x, state.goal, state.node_mask & ~cfg.actuator_mask))

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
        mse_ctrl = mse_on_mask(state.x, state.goal, state.node_mask & cfg.actuator_mask)
        mse_unact = mse_to_goal(state.x, state.goal, state.node_mask & ~cfg.actuator_mask)

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
        "end_mse": float(mse),  # float(mse_to_goal(state.x, state.goal, state.node_mask)),
        "start_mse_ctrl": start_mse_ctrl,
        "end_mse_ctrl": float(mse_ctrl),
        "start_mse_unact": float(start_mse_unact),
        "end_mse_unact": float(mse_unact),
        "total_reward": total_reward,
        "steps": len(history),
        "history": history,
    }


def aggregate_rollouts(rows):
    return {
        "episodes": len(rows),
        "start_mse_mean": float(np.mean([r["start_mse"] for r in rows])),
        "start_mse_std": float(np.std([r["start_mse"] for r in rows])),
        "end_mse_mean": float(np.mean([r["end_mse"] for r in rows])),
        "end_mse_std": float(np.std([r["end_mse"] for r in rows])),
        "start_mse_ctrl_mean": float(np.mean([r["start_mse_ctrl"] for r in rows])),
        "start_mse_ctrl_std": float(np.std([r["start_mse_ctrl"] for r in rows])),
        "end_mse_ctrl_mean": float(np.mean([r["end_mse_ctrl"] for r in rows])),
        "end_mse_ctrl_std": float(np.std([r["end_mse_ctrl"] for r in rows])),
        "start_mse_unact_mean": float(np.mean([r["start_mse_unact"] for r in rows])),
        "start_mse_unact_std": float(np.std([r["start_mse_unact"] for r in rows])),
        "end_mse_unact_mean": float(np.mean([r["end_mse_unact"] for r in rows])),
        "end_mse_unact_std": float(np.std([r["end_mse_unact"] for r in rows])),
        "total_reward_mean": float(np.mean([r["total_reward"] for r in rows])),
        "total_reward_std": float(np.std([r["total_reward"] for r in rows])),
        "steps_mean": float(np.mean([r["steps"] for r in rows])),
        "steps_std": float(np.std([r["steps"] for r in rows])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenario", help="Scenario name from scenarios.get_scenario()")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--k-prop", type=float, default=0.5)
    p.add_argument(
        "--outdir",
        type=str,
        default="experiments/controller_eval",
        help="Base output dir for controller evaluation artifacts).",
    )
    p.add_argument("--no-write", action="store_true", help="Do not write json files.")
    p.add_argument("--wandb", action="store_true", help="Log results to Weights & Biases.")
    args = p.parse_args()

    scenario_name = args.scenario
    scenario_cfg = get_scenario(scenario_name)
    print("Scenario stats:", scenario_stats(scenario_cfg))

    seed = args.seed
    episodes = args.episodes
    k_prop = args.k_prop
    if episodes < 1:
        raise ValueError("--episodes must be >= 1")

    controllers = [
        "zero",
        "random",
        "proportional",
        "masked_proportional",
        "inferred_proportional",
    ]
    results_by_controller = {}
    for name in controllers:
        rows = []
        for episode in range(episodes):
            row = rollout(name, scenario_name, scenario_cfg, seed=seed + episode, k_prop=k_prop)
            row["episode"] = episode
            rows.append(row)
        results_by_controller[name] = rows

    summary = {
        "scenario": scenario_name,
        "seed": seed,
        "episodes": episodes,
        "k_prop": k_prop,
        "results": [
            {
                "controller": name,
                **aggregate_rollouts(rows),
            }
            for name, rows in results_by_controller.items()
        ],
    }

    summary_by_controller = {row["controller"]: row for row in summary["results"]}

    print(json.dumps(summary, indent=2))

    git_rev = _git_revision()
    run_metadata_base = {
        "scenario": scenario_name,
        "git_revision": git_rev,
        "argv": sys.argv,
    }

    if not args.no_write:
        # Scenario-level summary dir
        scenario_dir = Path(args.outdir) / scenario_name / f"seed={seed:03d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_record = {**summary, **run_metadata_base}
        (scenario_dir / "scenario_summary.json").write_text(json.dumps(scenario_record, indent=2))

        # Per-controller dirs with metadata
        for name, rows in results_by_controller.items():
            agg = summary_by_controller[name]
            compact_episodes = [compact_episode_row(r) for r in rows]

            meta = ExperimentMetadata(
                run_type="controller_eval",
                scenario=scenario_name,
                policy_or_agent=canonical_policy_name(name),
                seed=seed,
                variant=f"kprop{args.k_prop}",
                episodes=args.episodes,
            )

            outdir = controller_eval_dir(args.outdir, meta)
            outdir.mkdir(parents=True, exist_ok=True)

            controller_record = {
                "controller": name,
                "aggregate": agg,
                "episodes": compact_episodes,
            }
            (outdir / "summary.json").write_text(json.dumps(controller_record, indent=2))
            (outdir / "episodes.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in compact_episodes)
            )
            meta_dict = {**meta.to_dict(), **run_metadata_base}
            (outdir / "metadata.json").write_text(json.dumps(meta_dict, indent=2))

    if args.wandb:
        for name, rows in results_by_controller.items():
            agg = summary_by_controller[name]
            meta = ExperimentMetadata(
                run_type="controller_eval",
                scenario=scenario_name,
                policy_or_agent=canonical_policy_name(name),
                seed=seed,
                variant=f"kprop{args.k_prop}",
                episodes=args.episodes,
            )
            stats = scenario_stats(scenario_cfg)
            wb_kwargs = wandb_init_kwargs(meta)
            wb_kwargs["config"] = {
                **wb_kwargs["config"],
                **stats,
                **({} if git_rev is None else {"git_revision": git_rev}),
            }
            with wandb.init(**wb_kwargs) as run:
                for row in rows:
                    ep = int(row["episode"])
                    c = compact_episode_row(row)
                    run.log(
                        {
                            "episode/total_reward": c["total_reward"],
                            "episode/end_mse": c["end_mse"],
                            "episode/end_mse_ctrl": c["end_mse_ctrl"],
                            "episode/end_mse_unact": c["end_mse_unact"],
                            "episode/steps": c["steps"],
                            "episode/start_mse": c["start_mse"],
                        },
                        step=ep,
                    )
                run.log(
                    {
                        "aggregate/end_mse_mean": agg["end_mse_mean"],
                        "aggregate/end_mse_std": agg["end_mse_std"],
                        "aggregate/total_reward_mean": agg["total_reward_mean"],
                        "aggregate/total_reward_std": agg["total_reward_std"],
                        "aggregate/start_mse_mean": agg["start_mse_mean"],
                        "aggregate/start_mse_std": agg["start_mse_std"],
                        "aggregate/start_mse_ctrl_mean": agg["start_mse_ctrl_mean"],
                        "aggregate/start_mse_ctrl_std": agg["start_mse_ctrl_std"],
                        "aggregate/end_mse_ctrl_mean": agg["end_mse_ctrl_mean"],
                        "aggregate/end_mse_ctrl_std": agg["end_mse_ctrl_std"],
                        "aggregate/start_mse_unact_mean": agg["start_mse_unact_mean"],
                        "aggregate/start_mse_unact_std": agg["start_mse_unact_std"],
                        "aggregate/end_mse_unact_mean": agg["end_mse_unact_mean"],
                        "aggregate/end_mse_unact_std": agg["end_mse_unact_std"],
                        "aggregate/steps_mean": agg["steps_mean"],
                        "aggregate/steps_std": agg["steps_std"],
                    },
                    step=episodes,
                )


if __name__ == "__main__":
    main()
