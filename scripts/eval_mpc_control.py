"""Closed-loop MPC control evaluation (chunks 2c–2d).

Runs receding-horizon MPC with each world-model variant as the dynamics model, a
perfect-model upper bound (``true``), and reactive baselines (``proportional`` / ``zero``
/ ``random``). Reports closed-loop episode return (= summed env reward) and final
MSE-to-goal per model × graph size × planning horizon.

The pipeline is topology-agnostic (handoff decision 6): graph structure enters only as
senders / receivers / masks. ``--topology`` is wired for the experiment matrix; only
``ring`` is available until experiment_matrix.md Stage A lands.

Usage:
    python scripts/eval_mpc_control.py \\
        --checkpoint-dir experiments/world_model --run-prefix full \\
        --models flat,graph_enc_dec,graph_rssm,true,proportional,zero \\
        --sizes 5,10,16 --horizons 1,5,10 --episodes 10 \\
        --out experiments/world_model/mpc_control.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import jax
import numpy as np

from dgr.agents.graph_dreamerv3.checkpoints import load_checkpoint
from dgr.agents.graph_dreamerv3.data import parse_sizes
from dgr.control.loop import (
    Actor,
    make_mpc_actor,
    make_proportional_actor,
    random_actor,
    run_episode,
    zero_actor,
)
from dgr.control.mpc import PlannerConfig, cem, random_shooting
from dgr.control.true_rollout import make_consensus_rollout
from dgr.control.wm_rollout import make_wm_rollout
from dgr.envs.suites.toy_graph_control.core import reset
from dgr.envs.suites.toy_graph_control.scenarios import make_consensus_config

_WM_MODULES = {
    "flat": "dgr.models.world_models.flat_wm",
    "graph_enc_dec": "dgr.models.world_models.graph_enc_dec_wm",
    "graph_rssm": "dgr.models.world_models.graph_rssm_wm",
}
_PLANNED = set(_WM_MODULES) | {"true"}
_REACTIVE = {"zero", "random", "proportional"}
_PLANNERS = {"shooting": random_shooting, "cem": cem}


def _load_wm(checkpoint_path: Path):
    payload = load_checkpoint(checkpoint_path)
    module = importlib.import_module(_WM_MODULES[payload["model_name"]])
    return module.predict_next_nodes_single, payload["params"]


def _build_actor(model, horizon, *, cfg, structure_state, planner, population, wm) -> Actor:
    node_mask = structure_state.node_mask
    if model in _REACTIVE:
        if model == "zero":
            return zero_actor()
        if model == "random":
            return random_actor()
        return make_proportional_actor(cfg.actuator_mask)

    if model == "true":
        rollout_fn = make_consensus_rollout(
            structure_state.senders,
            structure_state.receivers,
            node_mask,
            structure_state.edge_mask,
            cfg.actuator_mask,
            cfg.dynamics.alpha,
            cfg.dynamics.beta,
        )
    else:
        predict_fn, params = wm
        rollout_fn = make_wm_rollout(
            predict_fn,
            params,
            structure_state.senders,
            structure_state.receivers,
            node_mask,
            structure_state.edge_mask,
        )

    planner_config = PlannerConfig(
        horizon=horizon, action_dim=int(node_mask.shape[0]), population=population
    )
    action_mask = node_mask & cfg.actuator_mask
    return make_mpc_actor(rollout_fn, planner, planner_config, action_mask, node_mask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("experiments/world_model"))
    parser.add_argument("--run-prefix", default="full")
    parser.add_argument(
        "--topology",
        default="ring",
        choices=["ring", "grid", "kregular"],
        help="Graph topology (match the checkpoints' training topology).",
    )
    parser.add_argument("--models", default="flat,graph_enc_dec,graph_rssm,true,proportional,zero")
    parser.add_argument("--sizes", default="5,10,16")
    parser.add_argument("--horizons", default="1,5,10", help="Planning horizons (planned models).")
    parser.add_argument("--planner", choices=sorted(_PLANNERS), default="cem")
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--env-horizon", type=int, default=50, help="Episode length.")
    parser.add_argument("--n-max", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/world_model/mpc_control.json")
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = parse_sizes(args.sizes)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    planner = _PLANNERS[args.planner]

    results: dict[str, dict] = {}
    for model in models:
        wm = None
        if model in _WM_MODULES:
            ckpt = args.checkpoint_dir / f"{args.run_prefix}_{model}" / f"{model}_world_model.pkl"
            if not ckpt.exists():
                print(f"skipping {model} — no checkpoint at {ckpt}")
                continue
            wm = _load_wm(ckpt)
        print(f"=== {model} ===")
        results[model] = {}

        for size in sizes:
            cfg = make_consensus_config(
                size,
                n_max=args.n_max,
                horizon=args.env_horizon,
                alpha=args.alpha,
                beta=args.beta,
                noise_std=args.noise_std,
                topology=args.topology,
            )
            structure_state, _ = reset(jax.random.PRNGKey(0), cfg)  # topology is deterministic
            results[model][str(size)] = {}

            for horizon in horizons if model in _PLANNED else [0]:
                actor = _build_actor(
                    model,
                    horizon,
                    cfg=cfg,
                    structure_state=structure_state,
                    planner=planner,
                    population=args.population,
                    wm=wm,
                )
                returns, mses = [], []
                for ep in range(args.episodes):
                    res = run_episode(jax.random.PRNGKey(args.seed + ep), cfg, actor)
                    returns.append(float(res.episode_return))
                    mses.append(float(res.final_mse))
                cell = {
                    "return_mean": float(np.mean(returns)),
                    "return_std": float(np.std(returns)),
                    "final_mse_mean": float(np.mean(mses)),
                    "final_mse_std": float(np.std(mses)),
                    "episodes": args.episodes,
                }
                results[model][str(size)][str(horizon)] = cell
                label = f"n={size}" + (f" H={horizon}" if model in _PLANNED else "")
                print(
                    f"  {label:14}  return={cell['return_mean']:+8.3f}  "
                    f"final_mse={cell['final_mse_mean']:.4f}"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "topology": args.topology,
            "planner": args.planner,
            "population": args.population,
            "episodes": args.episodes,
            "env_horizon": args.env_horizon,
            "sizes": sizes,
            "horizons": horizons,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved results → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
