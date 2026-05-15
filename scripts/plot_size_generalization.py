"""Generate the Layer 2 size generalization figure.

Reads generalization_figure_data.json outputs from eval_minimal_graph_world_model.py
for all three model variants and produces:
  docs/assets/size_generalization.png
  docs/assets/size_generalization_data.json   (source data for re-styling)

Layout:
  Row 0: x_mse bar chart (2/3 width) | goal_mse bar chart (1/3 width)
  Row 1: trajectory at n=16 for each of the three models
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

EVAL_DATA = {
    "flat": "experiments/world_model/full_flat/generalization_figure_data.json",
    "graph_enc_dec": "experiments/world_model/full_graph_enc_dec/generalization_figure_data.json",
    "graph_rssm": "experiments/world_model/full_graph_rssm/generalization_figure_data.json",
}

TRAIN_SIZES = {4, 5, 6}
ALL_SIZES = [3, 4, 5, 6, 8, 10, 12, 16]
TRAJ_SIZE = 16
N_TRAJ_NODES = 4

COLORS = {
    "flat": "#E69F00",
    "graph_enc_dec": "#56B4E9",
    "graph_rssm": "#009E73",
}
LABELS = {
    "flat": "A: Flat MLP",
    "graph_enc_dec": "B: Node-indep. GNN",
    "graph_rssm": "C: Graph RSSM",
}
NODE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

OUT_PNG = Path("docs/assets/size_generalization.png")
OUT_DATA = Path("docs/assets/size_generalization_data.json")


def _load(path: str) -> dict[int, dict]:
    with open(path) as f:
        panels = json.load(f)
    return {p["size"]: p for p in panels}


def _bar_axes(ax, metrics_by_model: dict, metric: str, title: str, ylabel: str) -> None:
    models = list(metrics_by_model.keys())
    n = len(models)
    x = np.arange(len(ALL_SIZES))
    width = 0.22
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    train_idx = [i for i, s in enumerate(ALL_SIZES) if s in TRAIN_SIZES]
    ax.axvspan(min(train_idx) - 0.5, max(train_idx) + 0.5, alpha=0.07, color="gray", zorder=0)
    ax.text(
        (min(train_idx) + max(train_idx)) / 2,
        1.01,
        "train",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    for model, offset in zip(models, offsets):
        vals = [metrics_by_model[model].get(s, {}).get(metric, np.nan) for s in ALL_SIZES]
        ax.bar(
            x + offset,
            vals,
            width,
            color=COLORS[model],
            label=LABELS[model],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in ALL_SIZES])
    ax.set_xlabel("Graph size  n", labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_title(title, pad=7)
    ax.grid(axis="y", alpha=0.2, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _traj_axes(ax, panel: dict, model_name: str) -> None:
    gt_x = np.array(panel["gt_x"])
    pred_x = np.array(panel["pred_x"])
    T, n = gt_x.shape
    times = np.arange(1, T + 1)
    node_idx = np.round(np.linspace(0, n - 1, min(N_TRAJ_NODES, n))).astype(int)

    for i, (ni, color) in enumerate(zip(node_idx, NODE_COLORS)):
        ax.plot(
            times,
            gt_x[:, ni],
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label="ground truth" if i == 0 else "_nolegend_",
        )
        ax.plot(
            times,
            pred_x[:, ni],
            color=color,
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
            label="predicted" if i == 0 else "_nolegend_",
        )

    ax.set_xlabel("Timestep", labelpad=4)
    ax.set_ylabel("Node state  x", labelpad=4)
    ax.set_title(
        f"{LABELS[model_name]}\n"
        f"n={panel['size']} OOD  ·  x={panel['x_mse']:.3f}  goal={panel['goal_mse']:.3f}",
        pad=6,
    )
    ax.grid(alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("experiments/world_model/.mplconfig")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    all_data = {name: _load(path) for name, path in EVAL_DATA.items()}
    metrics = {
        model: {s: {"x_mse": p["x_mse"], "goal_mse": p["goal_mse"]} for s, p in by_size.items()}
        for model, by_size in all_data.items()
    }

    fig = plt.figure(figsize=(14, 8.5))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.52,
        wspace=0.34,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.09,
    )

    ax_x = fig.add_subplot(gs[0, 0:2])
    ax_g = fig.add_subplot(gs[0, 2])
    ax_ta = fig.add_subplot(gs[1, 0])
    ax_tb = fig.add_subplot(gs[1, 1])
    ax_tc = fig.add_subplot(gs[1, 2])

    _bar_axes(ax_x, metrics, "x_mse", "State prediction  (x_mse)", "MSE  (log scale)")
    _bar_axes(ax_g, metrics, "goal_mse", "Goal prediction  (goal_mse)", "MSE  (log scale)")

    for ax, model in zip([ax_ta, ax_tb, ax_tc], EVAL_DATA):
        _traj_axes(ax, all_data[model][TRAJ_SIZE], model)

    ax_x.legend(loc="upper left", frameon=False, fontsize=9)
    handles, labels = ax_ta.get_legend_handles_labels()
    ax_tc.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)

    fig.suptitle(
        "World model size generalization  —  trained on n ∈ {4, 5, 6},  evaluated across sizes",
        fontsize=11,
        y=0.98,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {OUT_PNG}")

    figure_source = {
        "metrics": {
            model: {str(s): v for s, v in by_size.items()} for model, by_size in metrics.items()
        },
        "trajectories": {
            model: {
                "size": TRAJ_SIZE,
                "x_mse": all_data[model][TRAJ_SIZE]["x_mse"],
                "goal_mse": all_data[model][TRAJ_SIZE]["goal_mse"],
                "gt_x": all_data[model][TRAJ_SIZE]["gt_x"],
                "pred_x": all_data[model][TRAJ_SIZE]["pred_x"],
            }
            for model in all_data
            if TRAJ_SIZE in all_data[model]
        },
        "train_sizes": sorted(TRAIN_SIZES),
        "all_sizes": ALL_SIZES,
        "traj_size": TRAJ_SIZE,
        "labels": LABELS,
        "colors": COLORS,
    }
    OUT_DATA.write_text(json.dumps(figure_source, indent=2) + "\n")
    print(f"Saved source data → {OUT_DATA}")


if __name__ == "__main__":
    main()
