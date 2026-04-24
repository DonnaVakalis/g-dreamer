"""Train a graph world model to predict next-step node features.

Three variants available via --model:
  flat         Variant A: MLP on flattened padded node observations
  graph_enc_dec Variant B: GNN encoder/decoder with flat latent dynamics
  graph_rssm   Variant C: Full graph-structured RSSM (message-passing through dynamics)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import types
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from dgr.agents.graph_dreamerv3.checkpoints import save_checkpoint
from dgr.agents.graph_dreamerv3.data import load_transition_dataset, parse_sizes
from dgr.experiments.metadata import ExperimentMetadata
from dgr.experiments.wandb_utils import wandb_init_kwargs

_MODEL_MODULES = {
    "flat": "dgr.models.world_models.flat_wm",
    "graph_enc_dec": "dgr.models.world_models.graph_enc_dec_wm",
    "graph_rssm": "dgr.models.world_models.graph_rssm_wm",
}


def _load_model_module(model_name: str) -> types.ModuleType:
    if model_name not in _MODEL_MODULES:
        raise ValueError(f"Unknown model: {model_name!r}")
    return importlib.import_module(_MODEL_MODULES[model_name])


def _make_config(model_name: str, node_dim: int, n_max: int, latent_dim: int, hidden_dim: int):
    if model_name == "flat":
        from dgr.models.world_models.flat_wm import FlatWMConfig

        return FlatWMConfig(node_dim=node_dim, n_max=n_max, hidden_dim=hidden_dim)
    elif model_name == "graph_enc_dec":
        from dgr.models.world_models.graph_enc_dec_wm import GraphEncDecWMConfig

        return GraphEncDecWMConfig(node_dim=node_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    elif model_name == "graph_rssm":
        from dgr.models.world_models.graph_rssm_wm import GraphRSSMConfig

        return GraphRSSMConfig(node_dim=node_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


def _make_batch(dataset, indices: np.ndarray) -> dict[str, jnp.ndarray]:
    return {
        "nodes": jnp.asarray(dataset.nodes[indices]),
        "actions": jnp.asarray(dataset.actions[indices]),
        "next_nodes": jnp.asarray(dataset.next_nodes[indices]),
        "senders": jnp.asarray(dataset.senders[indices]),
        "receivers": jnp.asarray(dataset.receivers[indices]),
        "node_mask": jnp.asarray(dataset.node_mask[indices]),
        "edge_mask": jnp.asarray(dataset.edge_mask[indices]),
    }


def _split_indices(
    dataset,
    train_sizes: list[int],
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    size_mask = np.isin(dataset.n_real, np.asarray(train_sizes, dtype=np.int32))
    candidate_indices = np.nonzero(size_mask)[0]
    if candidate_indices.size == 0:
        raise ValueError(f"No transitions found for train_sizes={train_sizes}")

    episode_ids = np.unique(dataset.episode_id[candidate_indices])
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    if val_frac <= 0.0 or episode_ids.size < 2:
        return candidate_indices, np.asarray([], dtype=np.int32)

    n_val_episodes = max(1, int(round(val_frac * episode_ids.size)))
    n_val_episodes = min(n_val_episodes, episode_ids.size - 1)
    val_episode_ids = episode_ids[:n_val_episodes]
    is_val = size_mask & np.isin(dataset.episode_id, val_episode_ids)
    is_train = size_mask & (~np.isin(dataset.episode_id, val_episode_ids))
    return np.nonzero(is_train)[0], np.nonzero(is_val)[0]


def _epoch_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {"loss": float("nan"), "x_mse": float("nan")}
    return {
        "loss": float(np.mean([row["loss"] for row in values])),
        "x_mse": float(np.mean([row["x_mse"] for row in values])),
    }


def _plot_loss(history: list[dict[str, float]], out_path: Path, model_name: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, label="train", linewidth=2.0, color="#1f77b4")
    if not all(np.isnan(val_loss)):
        ax.plot(epochs, val_loss, label="val", linewidth=2.0, color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked node MSE")
    ax.set_title(f"World Model Training — {model_name}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["flat", "graph_enc_dec", "graph_rssm"],
        default="graph_rssm",
        help="World model variant to train.",
    )
    parser.add_argument(
        "--train-sizes",
        default="4,5,6",
        help="Comma-separated graph sizes to train on.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to experiments/world_model/{model}.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    args = parser.parse_args()

    outdir = args.outdir or Path(f"experiments/world_model/{args.model}")

    dataset = load_transition_dataset(args.dataset)
    train_sizes = parse_sizes(args.train_sizes)
    train_indices, val_indices = _split_indices(dataset, train_sizes, args.val_frac, args.seed)

    node_dim = int(dataset.nodes.shape[-1])
    n_max = int(dataset.nodes.shape[1])
    model_config = _make_config(args.model, node_dim, n_max, args.latent_dim, args.hidden_dim)
    mod = _load_model_module(args.model)

    params = mod.init_params(jax.random.PRNGKey(args.seed), model_config)
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)

    run = None
    if args.wandb:
        meta = ExperimentMetadata(
            run_type="wm_train",
            scenario="consensus",
            policy_or_agent="world_model",
            variant=args.model,
            seed=args.seed,
        ).with_timestamp()
        extra_config: dict[str, Any] = {
            "dataset": str(args.dataset),
            "train_sizes": train_sizes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "val_frac": args.val_frac,
            "n_max": n_max,
            "node_dim": node_dim,
        }
        init_kwargs = wandb_init_kwargs(meta)
        init_kwargs["config"].update(extra_config)
        run = wandb.init(**init_kwargs)  # type: ignore[attr-defined]

    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, metrics), grads = jax.value_and_grad(mod.batch_loss, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    @jax.jit
    def eval_step(params, batch):
        _, metrics = mod.batch_loss(params, batch)
        return metrics

    history: list[dict[str, float]] = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        shuffled = np.array(train_indices, copy=True)
        rng.shuffle(shuffled)
        train_metrics = []
        for start in range(0, shuffled.size, args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            batch = _make_batch(dataset, batch_indices)
            params, opt_state, metrics = train_step(params, opt_state, batch)
            train_metrics.append({k: float(v) for k, v in metrics.items()})

        val_metrics = []
        for start in range(0, val_indices.size, args.batch_size):
            batch_indices = val_indices[start : start + args.batch_size]
            batch = _make_batch(dataset, batch_indices)
            metrics = eval_step(params, batch)
            val_metrics.append({k: float(v) for k, v in metrics.items()})

        epoch_train = _epoch_metrics(train_metrics)
        epoch_val = _epoch_metrics(val_metrics)
        row = {
            "epoch": epoch,
            "train_loss": epoch_train["loss"],
            "train_x_mse": epoch_train["x_mse"],
            "val_loss": epoch_val["loss"],
            "val_x_mse": epoch_val["x_mse"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"train_x_mse={row['train_x_mse']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_x_mse={row['val_x_mse']:.6f}"
        )
        if run is not None:
            run.log(
                {
                    "train/loss": row["train_loss"],
                    "train/x_mse": row["train_x_mse"],
                    "val/loss": row["val_loss"],
                    "val/x_mse": row["val_x_mse"],
                    "epoch": epoch,
                }
            )

    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / f"{args.model}_world_model.pkl"
    loss_curve_path = outdir / "loss_curve.png"
    metrics_path = outdir / "train_metrics.json"

    save_checkpoint(
        checkpoint_path,
        {
            "params": params,
            "model_name": args.model,
            "model_config": {k: v for k, v in vars(model_config).items()},
            "train_config": {
                "dataset": str(args.dataset),
                "train_sizes": train_sizes,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "val_frac": args.val_frac,
                "seed": args.seed,
                "n_max": n_max,
            },
            "history": history,
        },
    )
    metrics_path.write_text(json.dumps(history, indent=2) + "\n")
    _plot_loss(history, loss_curve_path, args.model)

    if run is not None:
        run.log({"loss_curve": wandb.Image(str(loss_curve_path))})  # type: ignore[attr-defined]
        run.finish()

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved loss curve to {loss_curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
