"""Plot closed-loop MPC control results (chunk 2e).

Reads the JSON produced by ``eval_mpc_control.py`` and renders a two-panel figure:
  Panel A — control cost vs graph size, each model at its best planning horizon.
  Panel B — control cost vs planning horizon, at one OOD graph size.

Control cost = negated episode return = cumulative MSE-to-goal over the episode
(lower is better). Also writes ``<out>_data.json`` with the plotted arrays.

Usage:
    python scripts/plot_mpc_control.py \\
        --results experiments/world_model/mpc_control.json \\
        --ood-size 16 \\
        --out docs/assets/mpc_control.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_WM = ["flat", "graph_enc_dec", "graph_rssm"]
_PLANNED = _WM + ["true"]
_ORDER = _WM + ["true", "proportional", "zero", "random"]

COLORS = {
    "flat": "#E69F00",
    "graph_enc_dec": "#56B4E9",
    "graph_rssm": "#009E73",
    "true": "#000000",
    "proportional": "#CC79A7",
    "zero": "#999999",
    "random": "#BBBBBB",
}
LABELS = {
    "flat": "A: Flat MLP",
    "graph_enc_dec": "B: Node-indep. GNN",
    "graph_rssm": "C: Graph RSSM",
    "true": "True model (MPC upper bound)",
    "proportional": "Proportional (model-free)",
    "zero": "Zero action",
    "random": "Random action",
}
STYLES = {"true": (0, (6, 2)), "proportional": (0, (1, 1.5))}


def _cost(results: dict, model: str, size: int, horizon: int) -> tuple[float, float] | None:
    """Return (cost mean, cost std) for one cell, or None if absent. Cost = -return."""
    cell = results.get(model, {}).get(str(size), {}).get(str(horizon))
    if cell is None:
        return None
    return -cell["return_mean"], cell["return_std"]


def _best_cost(
    results: dict, model: str, size: int, horizons: list[int]
) -> tuple[float, float, int] | None:
    """Return (cost mean, cost std, best horizon) minimising cost over horizons."""
    best: tuple[float, float, int] | None = None
    for h in horizons:
        c = _cost(results, model, size, h)
        if c is None:
            continue
        if best is None or c[0] < best[0]:
            best = (c[0], c[1], h)
    return best


def _plot(data: dict, args: argparse.Namespace) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", str(args.out.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = data["results"]
    sizes: list[int] = data["config"]["sizes"]
    horizons: list[int] = data["config"]["horizons"]
    train_sizes = {int(s) for s in args.train_sizes.split(",")}
    models = [m for m in _ORDER if m in results]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
    plotted: dict = {"panel_a": {}, "panel_b": {}}

    # --- Panel A: cost vs graph size, each model at its best planning horizon ---
    for model in models:
        xs, ys, es, best_h = [], [], [], []
        for size in sizes:
            if model in _PLANNED:
                bc = _best_cost(results, model, size, horizons)
                if bc is None:
                    continue
                cost, std, h = bc
            else:
                c = _cost(results, model, size, 0)
                if c is None:
                    continue
                cost, std, h = c[0], c[1], 0
            xs.append(size)
            ys.append(cost)
            es.append(std)
            best_h.append(h)
        if not xs:
            continue
        ax_a.errorbar(
            xs,
            ys,
            yerr=es,
            label=LABELS[model],
            color=COLORS[model],
            linestyle=STYLES.get(model, "-"),
            linewidth=2.0,
            marker="o",
            markersize=4,
            capsize=3,
        )
        plotted["panel_a"][model] = {
            "sizes": xs,
            "cost": ys,
            "cost_std": es,
            "best_horizon": best_h,
        }

    in_dist = [s for s in sizes if s in train_sizes]
    ood = [s for s in sizes if s not in train_sizes]
    if in_dist and ood:
        divider = 0.5 * (max(in_dist) + min(ood))
        ax_a.axvline(divider, color="#888888", linewidth=1.0, linestyle=":")
        ax_a.text(divider, ax_a.get_ylim()[1], "  OOD →", va="top", fontsize=8, color="#888888")

    ax_a.set_yscale("log")
    ax_a.set_xlabel("Graph size (n)")
    ax_a.set_ylabel("Control cost  (cumulative MSE-to-goal, lower better)")
    ax_a.set_title("A · Cost vs graph size  (best planning horizon per model)", fontsize=10)
    ax_a.grid(alpha=0.2, which="both")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # --- Panel B: cost vs planning horizon, fixed OOD size ---
    for model in models:
        if model in _PLANNED:
            xs, ys, es = [], [], []
            for h in horizons:
                c = _cost(results, model, args.ood_size, h)
                if c is None:
                    continue
                xs.append(h)
                ys.append(c[0])
                es.append(c[1])
            if not xs:
                continue
            ax_b.errorbar(
                xs,
                ys,
                yerr=es,
                label=LABELS[model],
                color=COLORS[model],
                linestyle=STYLES.get(model, "-"),
                linewidth=2.0,
                marker="o",
                markersize=4,
                capsize=3,
            )
            plotted["panel_b"][model] = {"horizons": xs, "cost": ys, "cost_std": es}
        else:
            c = _cost(results, model, args.ood_size, 0)
            if c is None:
                continue
            ax_b.axhline(c[0], color=COLORS[model], linestyle=STYLES.get(model, "-"), linewidth=1.5)
            plotted["panel_b"][model] = {"cost": c[0]}

    ax_b.set_yscale("log")
    ax_b.set_xlabel("Planning horizon (H)")
    ax_b.set_ylabel("Control cost  (cumulative MSE-to-goal, lower better)")
    ax_b.set_title(f"B · Cost vs planning horizon  (n={args.ood_size}, OOD)", fontsize=10)
    ax_b.grid(alpha=0.2, which="both")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.0),
        title="Closed-loop MPC control — world models as the planner's dynamics model",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {args.out}")
    return plotted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("experiments/world_model/mpc_control.json")
    )
    parser.add_argument("--ood-size", type=int, default=16, help="OOD graph size for panel B.")
    parser.add_argument("--train-sizes", default="4,5,6")
    parser.add_argument("--out", type=Path, default=Path("docs/assets/mpc_control.png"))
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    plotted = _plot(data, args)

    data_path = args.out.with_name(args.out.stem + "_data.json")
    data_path.write_text(json.dumps(plotted, indent=2) + "\n")
    print(f"Saved data  → {data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
