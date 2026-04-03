"""Evaluate the minimal graph world model and save a train-vs-unseen generalization figure."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dgr.agents.graph_dreamerv3.checkpoints import load_checkpoint
from dgr.agents.graph_dreamerv3.data import parse_sizes
from dgr.agents.graph_dreamerv3.minimal_world_model import (
    MinimalWorldModelConfig,
    predict_next_nodes_single,
)
from dgr.envs.suites.toy_graph_control.core import reset, step
from dgr.envs.suites.toy_graph_control.scenarios import make_consensus_config


def _rollout_episode(
    params,
    *,
    size: int,
    n_max: int,
    horizon: int,
    seed: int,
    alpha: float,
    beta: float,
    noise_std: float,
    action_scale: float,
) -> dict[str, np.ndarray | float | int]:
    cfg = make_consensus_config(
        size,
        n_max=n_max,
        horizon=horizon,
        alpha=alpha,
        beta=beta,
        noise_std=noise_std,
    )
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state, obs = reset(reset_key, cfg)

    gt_x = []
    pred_x = []
    for _ in range(horizon):
        key, action_key, step_key = jax.random.split(key, 3)
        action = action_scale * jax.random.uniform(
            action_key,
            shape=(cfg.spec.n_max,),
            minval=-1.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
        action = jnp.where(state.node_mask, action, 0.0)
        pred_next = predict_next_nodes_single(
            params,
            obs.nodes,
            action,
            state.senders,
            state.receivers,
            state.node_mask,
            state.edge_mask,
        )
        next_state, next_obs, _, done = step(step_key, cfg, state, action)
        gt_x.append(np.asarray(next_obs.nodes[:size, 0], dtype=np.float32))
        pred_x.append(np.asarray(pred_next[:size, 0], dtype=np.float32))
        state, obs = next_state, next_obs
        if bool(done):
            break

    gt_x_np = np.stack(gt_x, axis=0)
    pred_x_np = np.stack(pred_x, axis=0)
    mse = float(np.mean((gt_x_np - pred_x_np) ** 2))
    return {"size": size, "gt_x": gt_x_np, "pred_x": pred_x_np, "mse": mse}


def _plot_panels(panels: list[dict], out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(panels)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, panel in zip(axes_flat, panels):
        gt_x = panel["gt_x"]
        pred_x = panel["pred_x"]
        times = np.arange(1, gt_x.shape[0] + 1)
        palette = plt.cm.tab10(np.linspace(0.0, 1.0, gt_x.shape[1]))
        group_color = "#dceeff" if panel["group"] == "train" else "#ffe8d6"
        ax.set_facecolor(group_color)
        for node_idx, color in enumerate(palette):
            label_gt = "ground truth" if node_idx == 0 else None
            label_pred = "predicted" if node_idx == 0 else None
            ax.plot(times, gt_x[:, node_idx], color=color, linewidth=2.0, alpha=0.9, label=label_gt)
            ax.plot(
                times,
                pred_x[:, node_idx],
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.85,
                label=label_pred,
            )
        ax.set_title(f"{panel['group']} size n={panel['size']} | x-MSE={panel['mse']:.4f}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Node state x")
        ax.grid(alpha=0.2)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Consensus Dynamics Generalization: One-Step Predictions", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-sizes", default="4,5,6")
    parser.add_argument("--eval-sizes", default="8,10")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/assets/generalization_result.png"))
    args = parser.parse_args()

    payload = load_checkpoint(args.checkpoint)
    model_config = MinimalWorldModelConfig(**payload["model_config"])
    _ = model_config
    params = payload["params"]

    train_sizes = parse_sizes(args.train_sizes)
    eval_sizes = parse_sizes(args.eval_sizes)
    n_max = max(
        max(train_sizes + eval_sizes),
        int(payload.get("train_config", {}).get("n_max", 0)),
    )

    panels: list[dict[str, object]] = []
    metrics: dict[str, list[dict[str, float | int]]] = {"train": [], "unseen": []}
    for offset, size in enumerate(train_sizes):
        result = _rollout_episode(
            params,
            size=size,
            n_max=n_max,
            horizon=args.horizon,
            seed=args.seed + offset,
            alpha=args.alpha,
            beta=args.beta,
            noise_std=args.noise_std,
            action_scale=args.action_scale,
        )
        result["group"] = "train"
        panels.append(result)
        metrics["train"].append({"size": size, "x_mse": result["mse"]})

    for offset, size in enumerate(eval_sizes, start=len(train_sizes)):
        result = _rollout_episode(
            params,
            size=size,
            n_max=n_max,
            horizon=args.horizon,
            seed=args.seed + offset,
            alpha=args.alpha,
            beta=args.beta,
            noise_std=args.noise_std,
            action_scale=args.action_scale,
        )
        result["group"] = "unseen"
        panels.append(result)
        metrics["unseen"].append({"size": size, "x_mse": result["mse"]})

    _plot_panels(panels, args.out)
    metrics_path = args.out.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved generalization figure to {args.out}")
    print(f"Saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
