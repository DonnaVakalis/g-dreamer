"""Collect on-policy transitions under MPC with the *true* dynamics model (P1′ data axis).

Roll out an MPC actor that plans with the env's true dynamics (the "true-model" upper bound
from Result 3) and record every (state, action, next_state) transition. The resulting dataset
is the *on-policy* distribution the expert controller visits — the natural training
distribution for the P1′ spoke (trajectory-divergence × on-policy data).

Schema matches ``TransitionDataset`` (same shape as ``collect_consensus_world_model_data.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from dgr.agents.graph_dreamerv3.data import TransitionDataset, parse_sizes, save_transition_dataset
from dgr.control.loop import collect_episode, make_mpc_actor
from dgr.control.mpc import PlannerConfig, cem, random_shooting
from dgr.control.true_rollout import make_consensus_rollout, make_node_independent_rollout
from dgr.envs.suites.toy_graph_control.core import reset
from dgr.envs.suites.toy_graph_control.scenarios import make_consensus_config

_PLANNERS = {"cem": cem, "shooting": random_shooting}


def _build_true_actor(cfg, structure_state, *, planner, population, plan_horizon):
    if cfg.dynamics.mode == "consensus":
        rollout_fn = make_consensus_rollout(
            structure_state.senders,
            structure_state.receivers,
            structure_state.node_mask,
            structure_state.edge_mask,
            cfg.actuator_mask,
            cfg.dynamics.alpha,
            cfg.dynamics.beta,
        )
    elif cfg.dynamics.mode == "node_independent":
        rollout_fn = make_node_independent_rollout(
            structure_state.node_mask,
            cfg.actuator_mask,
            cfg.dynamics.alpha,
            cfg.dynamics.beta,
        )
    else:
        raise ValueError(f"No true rollout for dynamics {cfg.dynamics.mode!r}")

    planner_config = PlannerConfig(
        horizon=plan_horizon,
        action_dim=int(structure_state.node_mask.shape[0]),
        population=population,
    )
    action_mask = structure_state.node_mask & cfg.actuator_mask
    return make_mpc_actor(
        rollout_fn, planner, planner_config, action_mask, structure_state.node_mask
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="ring", choices=["ring", "grid", "kregular"])
    parser.add_argument(
        "--dynamics", default="consensus", choices=["consensus", "node_independent"]
    )
    parser.add_argument("--sizes", default="4,5,6", help="Graph sizes to collect.")
    parser.add_argument("--episodes-per-size", type=int, default=500)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=50, help="Env horizon (episode length).")
    parser.add_argument("--plan-horizon", type=int, default=1, help="MPC planning horizon.")
    parser.add_argument("--planner", choices=sorted(_PLANNERS), default="cem")
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/world_model/on_policy/transitions.npz"),
    )
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    n_max = max(sizes) if args.n_max is None else args.n_max
    planner = _PLANNERS[args.planner]

    chunks: list[dict[str, np.ndarray]] = []
    n_reals: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    episode_counter = 0
    key = jax.random.PRNGKey(args.seed)

    for size in sizes:
        cfg = make_consensus_config(
            size,
            n_max=n_max,
            horizon=args.horizon,
            alpha=args.alpha,
            beta=args.beta,
            noise_std=args.noise_std,
            topology=args.topology,
            dynamics=args.dynamics,
        )
        structure_state, _ = reset(jax.random.PRNGKey(0), cfg)  # deterministic topology
        actor = _build_true_actor(
            cfg,
            structure_state,
            planner=planner,
            population=args.population,
            plan_horizon=args.plan_horizon,
        )
        print(f"  collecting n={size} × {args.episodes_per_size} episodes …")
        for ep in range(args.episodes_per_size):
            key, ep_key = jax.random.split(key)
            _, transitions = collect_episode(ep_key, cfg, actor)
            chunks.append(transitions)
            h = transitions["nodes"].shape[0]
            n_reals.append(np.full((h,), size, dtype=np.int32))
            episode_ids.append(np.full((h,), episode_counter, dtype=np.int32))
            episode_counter += 1

    keys = (
        "nodes",
        "actions",
        "next_nodes",
        "senders",
        "receivers",
        "node_mask",
        "edge_mask",
        "step_id",
    )
    merged = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}
    n_real_arr = np.concatenate(n_reals, axis=0)
    episode_id_arr = np.concatenate(episode_ids, axis=0)

    dataset = TransitionDataset(
        nodes=merged["nodes"],
        actions=merged["actions"],
        next_nodes=merged["next_nodes"],
        senders=merged["senders"],
        receivers=merged["receivers"],
        node_mask=merged["node_mask"],
        edge_mask=merged["edge_mask"],
        n_real=n_real_arr,
        episode_id=episode_id_arr,
        step_id=merged["step_id"],
    )
    metadata = {
        "policy": "true_mpc",
        "topology": args.topology,
        "dynamics": args.dynamics,
        "sizes": sizes,
        "episodes_per_size": args.episodes_per_size,
        "n_max": n_max,
        "horizon": args.horizon,
        "plan_horizon": args.plan_horizon,
        "planner": args.planner,
        "population": args.population,
        "alpha": args.alpha,
        "beta": args.beta,
        "noise_std": args.noise_std,
        "seed": args.seed,
        "num_transitions": dataset.size,
    }
    save_transition_dataset(dataset, args.out, metadata=metadata)
    print(f"Saved {dataset.size} on-policy transitions → {args.out}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
