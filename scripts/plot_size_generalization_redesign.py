#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

MODEL_ORDER = ["flat", "graph_enc_dec", "graph_rssm"]
MODEL_LABELS = {
    "flat": "A  Flat MLP",
    "graph_enc_dec": "B  Node-indep. GNN",
    "graph_rssm": "C  Graph RSSM",
}
MODEL_SHORT = {
    "flat": "A",
    "graph_enc_dec": "B",
    "graph_rssm": "C",
}
MODEL_STORY = {
    "flat": "memorizes train sizes, collapses OOD",
    "graph_enc_dec": "state generalizes, goal still weak",
    "graph_rssm": "best overall generalization",
}
MODEL_COLORS = {
    "flat": "#C9782A",
    "graph_enc_dec": "#4C78A8",
    "graph_rssm": "#2A9D8F",
}
TRAIN_SIZES = [4, 5, 6]
SIZE_ORDER = [3, 4, 5, 6, 8, 10, 12, 16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/assets/size_generalization_data.json"),
        help="Path to JSON figure data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/size_generalization_redesign.png"),
        help="Path to save the rendered figure.",
    )
    return parser.parse_args()


def load_data(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def style_axes(ax: plt.Axes, *, log_y: bool = False) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", which="both", color="#D8D8D8", alpha=0.55, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if log_y:
        ax.set_yscale("log")


def annotate_train_region(ax: plt.Axes) -> None:
    ax.axvspan(3.5, 6.5, color="#BDBDBD", alpha=0.12, zorder=0)
    ax.text(
        5.0,
        1.01,
        "train: n in {4, 5, 6}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.text(
        11.8,
        1.01,
        "OOD",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )


def plot_metric_panel(ax: plt.Axes, metrics: dict, metric_key: str, title: str) -> None:
    style_axes(ax, log_y=True)
    annotate_train_region(ax)

    x = np.array(SIZE_ORDER)
    for model in MODEL_ORDER:
        y = np.array([metrics[model][str(size)][metric_key] for size in SIZE_ORDER], dtype=float)
        ax.plot(
            x,
            y,
            color=MODEL_COLORS[model],
            linewidth=2.4,
            marker="o",
            markersize=5.5,
            label=MODEL_LABELS[model],
        )

    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("Graph size n")
    ax.set_ylabel("MSE (log scale)")
    ax.set_xticks(SIZE_ORDER)


def plot_error_heatmap(ax: plt.Axes, traj: dict, model: str, norm: LogNorm, cmap: str):
    style_axes(ax, log_y=False)
    ax.grid(False)
    gt = np.array(traj["gt_x"], dtype=float)
    pred = np.array(traj["pred_x"], dtype=float)
    err = np.abs(pred - gt).T
    im = ax.imshow(
        err,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        norm=norm,
        cmap=cmap,
    )

    ax.set_title(
        f"{MODEL_SHORT[model]}  {MODEL_LABELS[model].split('  ', 1)[1]}\n{MODEL_STORY[model]}",
        fontsize=11.5,
        color=MODEL_COLORS[model],
        pad=8,
    )
    ax.text(
        0.02,
        0.97,
        f"n=16   x={traj['x_mse']:.3f}   goal={traj['goal_mse']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8),
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Node index")
    ax.set_xticks([0, 10, 20, 30, 40, 49])
    ax.set_yticks([0, 5, 10, 15])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(MODEL_COLORS[model])
        spine.set_linewidth(1.2)
    return im


def build_figure(data: dict) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(15.5, 10.5))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.0])

    ax_x = fig.add_subplot(gs[0, 0:3])
    ax_goal = fig.add_subplot(gs[0, 3:6])
    ax_a = fig.add_subplot(gs[1, 0:2])
    ax_b = fig.add_subplot(gs[1, 2:4])
    ax_c = fig.add_subplot(gs[1, 4:6])

    plot_metric_panel(ax_x, data["metrics"], "x_mse", "State prediction")
    plot_metric_panel(ax_goal, data["metrics"], "goal_mse", "Goal prediction")

    all_errors = []
    for model in MODEL_ORDER:
        gt = np.array(data["trajectories"][model]["gt_x"], dtype=float)
        pred = np.array(data["trajectories"][model]["pred_x"], dtype=float)
        all_errors.append(np.abs(pred - gt))
    all_errors = np.concatenate([err.reshape(-1) for err in all_errors])
    norm = LogNorm(
        vmin=max(1e-3, float(np.percentile(all_errors, 1))), vmax=float(np.max(all_errors))
    )
    cmap = "cividis"

    im = plot_error_heatmap(ax_a, data["trajectories"]["flat"], "flat", norm, cmap)
    plot_error_heatmap(ax_b, data["trajectories"]["graph_enc_dec"], "graph_enc_dec", norm, cmap)
    plot_error_heatmap(ax_c, data["trajectories"]["graph_rssm"], "graph_rssm", norm, cmap)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker="o",
            linewidth=2.4,
            markersize=5.5,
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
    )

    fig.suptitle(
        "World model size generalization",
        fontsize=27,
        y=0.972,
    )
    fig.text(
        0.5,
        0.935,
        "trained on n in {4, 5, 6}, evaluated across graph sizes",
        ha="center",
        va="center",
        fontsize=15,
        color="#4F4F4F",
    )
    fig.text(
        0.5,
        0.485,
        "OOD rollout at n=16 shown as absolute state error |pred - gt| on a shared log color scale",
        ha="center",
        va="center",
        fontsize=11,
        color="#5A5A5A",
    )
    cax = fig.add_axes([0.972, 0.08, 0.015, 0.30])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("|prediction error|")
    fig.subplots_adjust(left=0.06, right=0.955, bottom=0.08, top=0.87, wspace=0.55, hspace=0.55)
    return fig


def main() -> None:
    args = parse_args()
    data = load_data(args.input)
    fig = build_figure(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
