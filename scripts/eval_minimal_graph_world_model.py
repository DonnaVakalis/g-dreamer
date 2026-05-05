"""Evaluate a trained world model checkpoint and measure generalization across graph sizes.

Loads a checkpoint produced by train_minimal_graph_world_model.py, rolls out the consensus
environment with a random policy, and compares one-step predictions against ground truth
for each requested graph size.

Produces:
  - A trajectory figure (ground truth vs predicted node states per size)
  - A per-size MSE table printed to stdout
  - A JSON metrics file alongside the figure

Canonical eval sizes:
    In-distribution: --train-sizes 4,5,6   (same sizes seen during training)
    OOD:             --eval-sizes 3,8,10,12,16  (never seen during training)

    python scripts/eval_minimal_graph_world_model.py \\
        --checkpoint experiments/world_model/graph_rssm/graph_rssm_world_model.pkl \\
        --train-sizes 4,5,6 \\
        --eval-sizes 3,8,10,12,16 \\
        --out experiments/world_model/graph_rssm/generalization.png
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from dgr.agents.graph_dreamerv3.checkpoints import load_checkpoint
from dgr.agents.graph_dreamerv3.data import parse_sizes
from dgr.envs.suites.toy_graph_control.core import reset, step
from dgr.envs.suites.toy_graph_control.scenarios import make_consensus_config
from dgr.experiments.metadata import ExperimentMetadata
from dgr.experiments.wandb_utils import wandb_init_kwargs

_MODEL_MODULES = {
    "flat": "dgr.models.world_models.flat_wm",
    "graph_enc_dec": "dgr.models.world_models.graph_enc_dec_wm",
    "graph_rssm": "dgr.models.world_models.graph_rssm_wm",
}

_CONFIG_CLASSES = {
    "flat": ("dgr.models.world_models.flat_wm", "FlatWMConfig"),
    "graph_enc_dec": ("dgr.models.world_models.graph_enc_dec_wm", "GraphNodeIndepWMConfig"),
    "graph_rssm": ("dgr.models.world_models.graph_rssm_wm", "GraphRSSMConfig"),
}


def _load_model_module(model_name: str) -> types.ModuleType:
    if model_name not in _MODEL_MODULES:
        raise ValueError(f"Unknown model: {model_name!r}")
    return importlib.import_module(_MODEL_MODULES[model_name])


def _reconstruct_config(model_name: str, config_dict: dict):
    module_path, class_name = _CONFIG_CLASSES[model_name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**config_dict)


def _rollout_episode(
    mod: types.ModuleType,
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
) -> dict:
    cfg = make_consensus_config(
        size, n_max=n_max, horizon=horizon, alpha=alpha, beta=beta, noise_std=noise_std
    )
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state, obs = reset(reset_key, cfg)

    gt_x, pred_x = [], []
    for _ in range(horizon):
        key, action_key, step_key = jax.random.split(key, 3)
        action = action_scale * jax.random.uniform(
            action_key, shape=(cfg.spec.n_max,), minval=-1.0, maxval=1.0, dtype=jnp.float32
        )
        action = jnp.where(state.node_mask, action, 0.0)
        pred_next = mod.predict_next_nodes_single(
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
    return {
        "size": size,
        "gt_x": gt_x_np,
        "pred_x": pred_x_np,
        "mse": float(np.mean((gt_x_np - pred_x_np) ** 2)),
    }


def _plot_panels(panels: list[dict], out_path: Path, model_name: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(panels)
    n_cols = min(n_panels, 3)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, panel in zip(axes_flat, panels):
        gt_x, pred_x = panel["gt_x"], panel["pred_x"]
        times = np.arange(1, gt_x.shape[0] + 1)
        palette = plt.cm.tab10(np.linspace(0.0, 1.0, gt_x.shape[1]))
        ax.set_facecolor("#dceeff" if panel["group"] == "train" else "#ffe8d6")
        for node_idx, color in enumerate(palette):
            ax.plot(
                times,
                gt_x[:, node_idx],
                color=color,
                linewidth=2.0,
                alpha=0.9,
                label="ground truth" if node_idx == 0 else None,
            )
            ax.plot(
                times,
                pred_x[:, node_idx],
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.85,
                label="predicted" if node_idx == 0 else None,
            )
        group_label = "train" if panel["group"] == "train" else "OOD"
        ax.set_title(f"{group_label}  n={panel['size']}  MSE={panel['mse']:.4f}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Node state x")
        ax.grid(alpha=0.2)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"One-step prediction generalization — {model_name}", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _print_table(train_sizes: list[int], eval_sizes: list[int], metrics: dict) -> None:
    train_label = "{" + ",".join(str(s) for s in train_sizes) + "}"
    print(f"\n{'size':>6}  {'group':>10}  {'x_mse':>10}")
    print("-" * 32)
    for row in metrics["train"]:
        print(f"{row['size']:>6}  {'in-dist':>10}  {row['x_mse']:>10.6f}")
    for row in metrics["unseen"]:
        print(f"{row['size']:>6}  {'OOD':>10}  {row['x_mse']:>10.6f}")
    print(f"\n  Trained on n ∈ {train_label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-sizes", default="4,5,6", help="Sizes used during training (for labelling)."
    )
    parser.add_argument(
        "--eval-sizes",
        default="3,8,10,12,16",
        help="Sizes to evaluate (should include both in-dist and OOD).",
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--episodes", type=int, default=1, help="Episodes per size to average over."
    )
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path. Defaults to <checkpoint_dir>/generalization.png",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    args = parser.parse_args()

    payload = load_checkpoint(args.checkpoint)
    model_name: str = payload.get("model_name", "graph_rssm")
    model_config = _reconstruct_config(model_name, payload["model_config"])
    _ = model_config
    params = payload["params"]
    n_max: int = int(payload.get("train_config", {}).get("n_max", 16))

    mod = _load_model_module(model_name)
    train_sizes = parse_sizes(args.train_sizes)
    eval_sizes = parse_sizes(args.eval_sizes)
    all_sizes = sorted(set(train_sizes) | set(eval_sizes))
    train_set = set(train_sizes)

    out_path = args.out or args.checkpoint.parent / "generalization.png"

    run = None
    if args.wandb:
        meta = ExperimentMetadata(
            run_type="wm_eval",
            scenario="consensus",
            policy_or_agent="world_model",
            variant=model_name,
            seed=args.seed,
        ).with_timestamp()
        init_kwargs = wandb_init_kwargs(meta)
        init_kwargs["config"]["eval_sizes"] = eval_sizes
        init_kwargs["config"]["train_sizes"] = train_sizes
        init_kwargs["config"]["checkpoint"] = str(args.checkpoint)
        run = wandb.init(**init_kwargs)  # type: ignore[attr-defined]

    panels: list[dict] = []
    metrics: dict[str, list[dict]] = {"train": [], "unseen": []}

    for offset, size in enumerate(all_sizes):
        episode_mses = []
        first_result = None
        for ep in range(args.episodes):
            result = _rollout_episode(
                mod,
                params,
                size=size,
                n_max=n_max,
                horizon=args.horizon,
                seed=args.seed + offset * 100 + ep,
                alpha=args.alpha,
                beta=args.beta,
                noise_std=args.noise_std,
                action_scale=args.action_scale,
            )
            episode_mses.append(result["mse"])
            if first_result is None:
                first_result = result

        assert first_result is not None
        mean_mse = float(np.mean(episode_mses))
        first_result["mse"] = mean_mse
        group = "train" if size in train_set else "unseen"
        first_result["group"] = group
        panels.append(first_result)
        metrics[group].append({"size": size, "x_mse": mean_mse})

        if run is not None:
            run.log({f"mse/n{size}": mean_mse, f"mse_group/{group}_n{size}": mean_mse})

    _print_table(train_sizes, eval_sizes, metrics)
    _plot_panels(panels, out_path, model_name)

    metrics_path = out_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    if run is not None:
        run.log({"generalization_figure": wandb.Image(str(out_path))})  # type: ignore[attr-defined]
        run.finish()

    print(f"\nSaved figure  → {out_path}")
    print(f"Saved metrics → {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
