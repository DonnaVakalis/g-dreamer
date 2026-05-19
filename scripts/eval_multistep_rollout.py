"""Evaluate world model variants under open-loop multi-step rollout.

At each timestep the model receives its own previous prediction as input rather than
the ground-truth observation. Error compounds over the horizon, revealing which
architectures have stable self-prediction loops.

Produces:
  <out>.png               — per-timestep MSE curves for each model × selected sizes
  <out>_data.json         — raw arrays for re-styling

Usage:
    python scripts/eval_multistep_rollout.py \\
        --checkpoint-dir experiments/world_model \\
        --run-prefix full \\
        --sizes 5,10,16 \\
        --episodes 10 \\
        --out docs/assets/multistep_rollout.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dgr.agents.graph_dreamerv3.checkpoints import load_checkpoint
from dgr.agents.graph_dreamerv3.data import parse_sizes
from dgr.envs.suites.toy_graph_control.core import reset, step
from dgr.envs.suites.toy_graph_control.scenarios import make_consensus_config

_MODEL_NAMES = ["flat", "graph_enc_dec", "graph_rssm"]
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
COLORS = {"flat": "#E69F00", "graph_enc_dec": "#56B4E9", "graph_rssm": "#009E73"}
LABELS = {
    "flat": "A: Flat MLP",
    "graph_enc_dec": "B: Node-indep. GNN",
    "graph_rssm": "C: Graph RSSM",
}

# Open-loop rollout error is heavy-tailed: a few episodes diverge by ~30 orders of
# magnitude. Each episode's per-step error is clipped to [_FLOOR, _CEIL] *before*
# aggregation so the reported mean ± SD is finite and a diverged episode contributes
# the ceiling rather than infinity. _DIVERGE_THRESHOLD defines a diverged episode.
_FLOOR, _CEIL = 1e-4, 1e4


def _load_model(checkpoint_path: Path):
    import importlib

    payload = load_checkpoint(checkpoint_path)
    model_name: str = payload["model_name"]

    module_path, class_name = _CONFIG_CLASSES[model_name]
    mod = importlib.import_module(module_path)
    config = getattr(mod, class_name)(**payload["model_config"])
    _ = config

    pred_mod = importlib.import_module(_MODEL_MODULES[model_name])
    n_max = int(payload["train_config"]["n_max"])
    return pred_mod, payload["params"], n_max


def _rollout_episode(
    mod,
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
    topology: str,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = make_consensus_config(
        size,
        n_max=n_max,
        horizon=horizon,
        alpha=alpha,
        beta=beta,
        noise_std=noise_std,
        topology=topology,
    )
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state, obs = reset(reset_key, cfg)

    model_nodes = obs.nodes
    x_errs: list[float] = []
    goal_errs: list[float] = []

    for _ in range(horizon):
        key, action_key, step_key = jax.random.split(key, 3)
        action = action_scale * jax.random.uniform(
            action_key, shape=(n_max,), minval=-1.0, maxval=1.0, dtype=jnp.float32
        )
        action = jnp.where(state.node_mask, action, 0.0)

        pred_next = mod.predict_next_nodes_single(
            params,
            model_nodes,
            action,
            state.senders,
            state.receivers,
            state.node_mask,
            state.edge_mask,
        )

        next_state, next_obs, _, done = step(step_key, cfg, state, action)

        gt = np.asarray(next_obs.nodes[:size], dtype=np.float32)
        pr = np.asarray(pred_next[:size], dtype=np.float32)
        x_errs.append(float(np.mean((gt[:, 0] - pr[:, 0]) ** 2)))
        goal_errs.append(float(np.mean((gt[:, 1] - pr[:, 1]) ** 2)))

        state = next_state
        model_nodes = pred_next  # open-loop: feed own prediction back
        if bool(done):
            break

    return np.array(x_errs, dtype=np.float32), np.array(goal_errs, dtype=np.float32)


def _run_size(
    mod,
    params,
    *,
    size,
    n_max,
    horizon,
    episodes,
    seed,
    alpha,
    beta,
    noise_std,
    action_scale,
    diverge_threshold,
    topology,
):
    all_x = []
    all_g = []
    for ep in range(episodes):
        x_e, g_e = _rollout_episode(
            mod,
            params,
            size=size,
            n_max=n_max,
            horizon=horizon,
            seed=seed + ep * 1000,
            alpha=alpha,
            beta=beta,
            noise_std=noise_std,
            action_scale=action_scale,
            topology=topology,
        )
        all_x.append(x_e)
        all_g.append(g_e)
    x_arr = np.stack(all_x)  # (episodes, T)
    g_arr = np.stack(all_g)

    # An episode has diverged if its combined error ever exceeds the threshold or goes
    # non-finite. Measured on raw values, before clipping.
    combined = x_arr + g_arr
    diverged = ~np.isfinite(combined) | (combined > diverge_threshold)
    diverge_rate = float(diverged.any(axis=1).mean())

    # Clip per-episode so the mean ± SD bands are finite (see _FLOOR/_CEIL note).
    x_clip = np.clip(np.nan_to_num(x_arr, nan=_CEIL, posinf=_CEIL, neginf=_CEIL), _FLOOR, _CEIL)
    g_clip = np.clip(np.nan_to_num(g_arr, nan=_CEIL, posinf=_CEIL, neginf=_CEIL), _FLOOR, _CEIL)
    return {
        "x_mean": x_clip.mean(0),
        "x_std": x_clip.std(0),
        "goal_mean": g_clip.mean(0),
        "goal_std": g_clip.std(0),
        "diverge_rate": np.asarray(diverge_rate, dtype=np.float32),
        "n_episodes": np.asarray(episodes, dtype=np.int32),
    }


def _plot(results: dict, sizes: list[int], train_sizes: set[int], out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    n_sizes = len(sizes)
    fig = plt.figure(figsize=(4.5 * n_sizes, 7))
    gs = gridspec.GridSpec(
        2,
        n_sizes,
        figure=fig,
        hspace=0.45,
        wspace=0.28,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.09,
    )

    feature_rows = [
        ("x_mean", "x_std", "State  x_mse"),
        ("goal_mean", "goal_std", "Goal  goal_mse"),
    ]

    # Open-loop rollout error compounds over many orders of magnitude, so the y-axis
    # is logarithmic. Non-finite / diverged values are clipped to a display ceiling
    # so a blown-up model still reads as "pinned at the ceiling" rather than erasing
    # the axis scale for every other curve.
    floor, ceil = 1e-4, 1e4

    for col, size in enumerate(sizes):
        group = "in-dist" if size in train_sizes else "OOD"
        for row, (mean_key, std_key, ylabel) in enumerate(feature_rows):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#f0f4f8" if group == "in-dist" else "#fff8f0")

            for model in _MODEL_NAMES:
                if model not in results or size not in results[model]:
                    continue
                r = results[model][size]
                T = len(r[mean_key])
                t = np.arange(1, T + 1)
                mean = np.nan_to_num(r[mean_key], nan=ceil, posinf=ceil, neginf=ceil)
                std = np.nan_to_num(r[std_key], nan=0.0, posinf=0.0, neginf=0.0)
                mean = np.clip(mean, floor, ceil)
                lo = np.clip(mean - std, floor, ceil)
                hi = np.clip(mean + std, floor, ceil)
                ax.plot(t, mean, color=COLORS[model], linewidth=2.0, label=LABELS[model])
                ax.fill_between(t, lo, hi, color=COLORS[model], alpha=0.15)

            ax.set_yscale("log")
            ax.set_ylim(floor, ceil)
            ax.axhline(ceil, color="#999999", linewidth=0.8, linestyle=":")
            ax.set_xlabel("Timestep", labelpad=3)
            if col == 0:
                ax.set_ylabel(ylabel, labelpad=4)
            if row == 0:
                ax.set_title(f"n={size}  ({group})", pad=6)
            ax.grid(alpha=0.2, which="both")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=COLORS[m], linewidth=2, label=LABELS[m]) for m in _MODEL_NAMES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.97),
    )
    fig.suptitle(
        "Open-loop rollout error over horizon  —  mean ± 1 SD across episodes "
        "(per-episode error clipped to the display range)",
        fontsize=10,
        y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {out_path}")


def _plot_divergence_rate(
    results: dict,
    sizes: list[int],
    train_sizes: set[int],
    threshold: float,
    out_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    models = [m for m in _MODEL_NAMES if m in results]
    fig, ax = plt.subplots(figsize=(1.7 * len(sizes) + 2.5, 4.2))

    x = np.arange(len(sizes))
    width = 0.8 / max(len(models), 1)
    n_eps = next(
        (int(results[m][s]["n_episodes"]) for m in models for s in sizes if s in results[m]),
        0,
    )

    for col, size in enumerate(sizes):
        if size not in train_sizes:
            ax.axvspan(col - 0.5, col + 0.5, color="#fff8f0", zorder=0)

    for i, model in enumerate(models):
        rates = [
            float(results[model][s]["diverge_rate"]) if s in results[model] else 0.0 for s in sizes
        ]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, rates, width, color=COLORS[model], label=LABELS[model], zorder=3)
        for b, r in zip(bars, rates):
            ax.text(
                b.get_x() + b.get_width() / 2,
                r + 0.02,
                f"{r:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={s}\n({'in-dist' if s in train_sizes else 'OOD'})" for s in sizes])
    ax.set_ylabel("Episodes that diverged")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(
        f"Open-loop divergence rate  —  {n_eps} episodes/size, "
        f"diverged = combined MSE ever exceeds {threshold:g}",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9, ncol=len(models))
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("experiments/world_model"),
        help="Parent directory containing per-model subdirectories.",
    )
    parser.add_argument(
        "--run-prefix",
        default="full",
        help="Subdirectory prefix, e.g. 'full' → full_flat/, full_graph_enc_dec/, ...",
    )
    parser.add_argument("--sizes", default="5,10,16", help="Graph sizes to evaluate.")
    parser.add_argument("--train-sizes", default="4,5,6", help="Training sizes (for panel labels).")
    parser.add_argument(
        "--topology",
        default="ring",
        choices=["ring", "grid", "kregular"],
        help="Graph topology of the eval environment (match the checkpoints' training topology).",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--diverge-threshold",
        type=float,
        default=10.0,
        help="An episode counts as diverged once its combined (x+goal) MSE exceeds this.",
    )
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/multistep_rollout.png"),
    )
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    train_sizes = set(parse_sizes(args.train_sizes))

    results: dict[str, dict[int, dict]] = {}
    for model_name in _MODEL_NAMES:
        ckpt_dir = args.checkpoint_dir / f"{args.run_prefix}_{model_name}"
        ckpt_path = ckpt_dir / f"{model_name}_world_model.pkl"
        if not ckpt_path.exists():
            print(f"  skipping {model_name} — checkpoint not found at {ckpt_path}")
            continue
        print(f"Loading {model_name} from {ckpt_path}")
        mod, params, n_max = _load_model(ckpt_path)
        results[model_name] = {}
        for size in sizes:
            print(f"  rolling out n={size} × {args.episodes} episodes …")
            results[model_name][size] = _run_size(
                mod,
                params,
                size=size,
                n_max=n_max,
                horizon=args.horizon,
                episodes=args.episodes,
                seed=args.seed,
                alpha=args.alpha,
                beta=args.beta,
                noise_std=args.noise_std,
                action_scale=args.action_scale,
                diverge_threshold=args.diverge_threshold,
                topology=args.topology,
            )

    _plot(results, sizes, train_sizes, args.out)
    div_stem = args.out.stem
    div_stem = (
        div_stem.replace("rollout", "divergence_rate")
        if "rollout" in div_stem
        else div_stem + "_divergence_rate"
    )
    _plot_divergence_rate(
        results,
        sizes,
        train_sizes,
        args.diverge_threshold,
        args.out.with_name(div_stem + ".png"),
    )

    data_path = args.out.with_name(args.out.stem + "_data.json")
    serialisable = {
        model: {str(size): {k: v.tolist() for k, v in r.items()} for size, r in by_size.items()}
        for model, by_size in results.items()
    }
    data_path.write_text(json.dumps(serialisable, indent=2) + "\n")
    print(f"Saved data  → {data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
