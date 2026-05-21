"""Iterative DAgger orchestrator (P1″ spoke).

The textbook DAgger move: at each round, collect data under the *learner's* current
policy (here: MPC planning with the learner's world model) in the true env, aggregate
with all prior rounds, retrain the world model warm-started from the previous round.
Result 6 showed that one-shot expert-on-policy data catastrophically breaks B's OOD
control via the BC state-distribution mismatch; DAgger fixes that mismatch by definition
because the training distribution *is* the learner's deployment distribution.

The orchestrator runs collection in-process (cheap — true env) and shells out to
``scripts/train_minimal_graph_world_model.py`` for training, so the standard
checkpointing / val-split code path is reused.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dgr.agents.graph_dreamerv3.data import (
    TransitionDataset,
    load_transition_dataset,
    save_transition_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _concat_datasets(datasets: list[TransitionDataset]) -> TransitionDataset:
    """Concatenate datasets, reindexing episode_id so episodes stay disjoint."""
    out: dict[str, np.ndarray] = {}
    fields = (
        "nodes",
        "actions",
        "next_nodes",
        "senders",
        "receivers",
        "node_mask",
        "edge_mask",
        "n_real",
        "step_id",
    )
    for f in fields:
        out[f] = np.concatenate([getattr(d, f) for d in datasets], axis=0)

    eps: list[np.ndarray] = []
    offset = 0
    for d in datasets:
        eps.append(d.episode_id.astype(np.int32) + offset)
        offset += int(d.episode_id.max()) + 1
    out["episode_id"] = np.concatenate(eps, axis=0)

    return TransitionDataset(**out)


def _collect_learner_round(
    *,
    checkpoint: Path,
    out_npz: Path,
    sizes: str,
    episodes_per_size: int,
    n_max: int,
    horizon: int,
    plan_horizon: int,
    population: int,
    seed: int,
    topology: str,
    dynamics: str,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "collect_learner_on_policy_data.py"),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out_npz),
        "--sizes",
        sizes,
        "--episodes-per-size",
        str(episodes_per_size),
        "--n-max",
        str(n_max),
        "--horizon",
        str(horizon),
        "--plan-horizon",
        str(plan_horizon),
        "--population",
        str(population),
        "--seed",
        str(seed),
        "--topology",
        topology,
        "--dynamics",
        dynamics,
    ]
    print("[dagger] collecting:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _train_round(
    *,
    model: str,
    dataset_path: Path,
    init_checkpoint: Path | None,
    outdir: Path,
    epochs: int,
    rollout_horizon: int,
    loss_aggregation: str,
    train_sizes: str,
    seed: int,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_minimal_graph_world_model.py"),
        "--model",
        model,
        "--dataset",
        str(dataset_path),
        "--outdir",
        str(outdir),
        "--epochs",
        str(epochs),
        "--rollout-horizon",
        str(rollout_horizon),
        "--loss-aggregation",
        loss_aggregation,
        "--train-sizes",
        train_sizes,
        "--seed",
        str(seed),
    ]
    if init_checkpoint is not None:
        cmd += ["--init-checkpoint", str(init_checkpoint)]
    print("[dagger] training:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["graph_enc_dec", "graph_rssm", "flat"], required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        required=True,
        help="Starting world model checkpoint (typically the offline-trained one).",
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        required=True,
        help="Original training dataset (.npz); aggregated with each round's on-policy data.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="Directory for DAgger artifacts (per-round datasets, checkpoints, logs).",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--episodes-per-size", type=int, default=200)
    parser.add_argument("--sizes", default="4,5,6")
    parser.add_argument("--n-max", type=int, default=16)
    parser.add_argument("--env-horizon", type=int, default=50)
    parser.add_argument("--plan-horizon", type=int, default=10)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--rollout-horizon", type=int, default=1)
    parser.add_argument("--loss-aggregation", choices=["mean", "final", "max"], default="mean")
    parser.add_argument("--train-sizes", default="4,5,6")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    base = load_transition_dataset(args.base_dataset)
    print(f"[dagger] loaded base dataset {args.base_dataset} (size={base.size})")

    current_ckpt: Path = args.init_checkpoint
    onpol_datasets: list[TransitionDataset] = []
    summary: list[dict] = []

    for k in range(1, args.iterations + 1):
        round_dir = workdir / f"iter_{k:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        onpol_npz = round_dir / "onpolicy_transitions.npz"

        _collect_learner_round(
            checkpoint=current_ckpt,
            out_npz=onpol_npz,
            sizes=args.sizes,
            episodes_per_size=args.episodes_per_size,
            n_max=args.n_max,
            horizon=args.env_horizon,
            plan_horizon=args.plan_horizon,
            population=args.population,
            seed=args.seed * 1000 + k,
            topology="ring",
            dynamics="consensus",
        )
        new_data = load_transition_dataset(onpol_npz)
        onpol_datasets.append(new_data)

        # Aggregate base + every prior round's on-policy data (DAgger's "D ← D ∪ ...").
        merged = _concat_datasets([base, *onpol_datasets])
        merged_npz = round_dir / "aggregated_dataset.npz"
        save_transition_dataset(
            merged,
            merged_npz,
            metadata={
                "round": k,
                "base_size": int(base.size),
                "onpolicy_sizes": [int(d.size) for d in onpol_datasets],
                "total_size": int(merged.size),
            },
        )
        print(f"[dagger] aggregated dataset size={merged.size}")

        # Train: warm-start from previous checkpoint.
        round_outdir = round_dir / "model"
        _train_round(
            model=args.model,
            dataset_path=merged_npz,
            init_checkpoint=current_ckpt,
            outdir=round_outdir,
            epochs=args.epochs,
            rollout_horizon=args.rollout_horizon,
            loss_aggregation=args.loss_aggregation,
            train_sizes=args.train_sizes,
            seed=args.seed,
        )
        next_ckpt = round_outdir / f"{args.model}_world_model.pkl"
        assert next_ckpt.exists(), f"expected checkpoint at {next_ckpt}"
        current_ckpt = next_ckpt

        summary.append(
            {
                "iter": k,
                "onpolicy_size": int(new_data.size),
                "aggregated_size": int(merged.size),
                "checkpoint": str(next_ckpt),
            }
        )

    (workdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[dagger] done. Final checkpoint: {current_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
