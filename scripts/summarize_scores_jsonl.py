"""
Dreamer already writes scores.jsonl under its logdir...
To make comparison trivial with our controllers,
we need this summarizer:

Example usage:
poetry run python scripts/summarize_scores_jsonl.py \
experiments/runs/<your_dreamer_run>/scores.jsonl

poetry run python scripts/summarize_scores_jsonl.py \
experiments/toy_eval/train_ring_dense/seed_000/proportional__scores.jsonl

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=str, help="Path to scores.jsonl")
    args = p.parse_args()

    scores = []
    lengths = []
    for line in Path(args.path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores.append(row["episode/score"])
        lengths.append(row["episode/length"])

    scores = np.array(scores, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.float32)

    print(
        f"episodes={len(scores)} mean_score={scores.mean():.3f} \
        std_score={scores.std():.3f} mean_len={lengths.mean():.2f}"
    )


if __name__ == "__main__":
    main()
