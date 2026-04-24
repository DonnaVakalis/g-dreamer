"""Collect random consensus-environment transitions across graph sizes for world-model training.

WHAT "SIZE" MEANS
-----------------
Each "size" is the number of real nodes (n_real) in a ring-topology consensus graph.
A size-4 graph has 4 nodes connected in a directed ring; size-5 has 5 nodes; etc.
These are genuinely different topologies — the generalization experiment trains on small
rings and tests whether the world model transfers to larger rings it has never seen.

All transitions are padded to n_max = max(sizes) nodes so they can be stored in a single
array and batched with static shapes (required by JAX/JIT).

RECOMMENDED COLLECTION COMMAND (large dataset for all experiments)
-------------------------------------------------------------------
    poetry run python scripts/collect_consensus_world_model_data.py \\
        --sizes 3,4,5,6,8,10,12,16 \\
        --episodes-per-size 2000 \\
        --horizon 50 \\
        --seed 0 \\
        --out experiments/world_model/consensus_transitions_large.npz

Dataset contents:

    Size  | Episodes | Steps/ep | Transitions
    ------|----------|----------|------------
    3     | 2000     | 50       | 100 000    (below training range — extrapolation check)
    4     | 2000     | 50       | 100 000    \\
    5     | 2000     | 50       | 100 000     > training sizes
    6     | 2000     | 50       | 100 000    /
    8     | 2000     | 50       | 100 000    \\
    10    | 2000     | 50       | 100 000     \\
    12    | 2000     | 50       | 100 000      > OOD eval sizes (never used in training)
    16    | 2000     | 50       | 100 000     /
    ------|----------|----------|------------
    TOTAL | 16 000   |          | 800 000

TRAIN / OOD SPLIT (used at training time, not collection time)
--------------------------------------------------------------
The dataset stores n_real per transition, so splits are applied in the training script
via --train-sizes. The canonical splits are:

    Train:       --train-sizes 4,5,6
    In-dist eval: n_real in {4, 5, 6}
    OOD eval:     n_real in {3, 8, 10, 12, 16}  (never seen during training)

These splits are non-overlapping by design.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dgr.agents.graph_dreamerv3.data import (
    collect_random_transitions,
    parse_sizes,
    save_transition_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="4,5,6,8,10", help="Comma-separated graph sizes.")
    parser.add_argument("--episodes-per-size", type=int, default=64)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/world_model/consensus_transitions.npz"),
    )
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    n_max = max(sizes) if args.n_max is None else args.n_max
    dataset = collect_random_transitions(
        sizes=sizes,
        episodes_per_size=args.episodes_per_size,
        n_max=n_max,
        horizon=args.horizon,
        alpha=args.alpha,
        beta=args.beta,
        noise_std=args.noise_std,
        action_scale=args.action_scale,
        seed=args.seed,
    )
    metadata = {
        "sizes": sizes,
        "episodes_per_size": args.episodes_per_size,
        "n_max": n_max,
        "horizon": args.horizon,
        "alpha": args.alpha,
        "beta": args.beta,
        "noise_std": args.noise_std,
        "action_scale": args.action_scale,
        "seed": args.seed,
        "num_transitions": dataset.size,
    }
    save_transition_dataset(dataset, args.out, metadata=metadata)
    print(f"Saved {dataset.size} transitions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
