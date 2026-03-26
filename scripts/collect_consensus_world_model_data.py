"""Collect random consensus-environment transitions across graph sizes for world-model training."""

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
