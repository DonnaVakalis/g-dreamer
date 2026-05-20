"""Compute the theoretical-empirical bridge — graph invariants, model Jacobians, system ranks.

Iterates over every world-model checkpoint we have plus every (topology, size) the
experiments touch, computes:

1. **Graph invariants** for each (topology, size): algebraic connectivity ``λ₁(L)``, diameter,
   degree stats. Hypothesis: C's divergence rises with ``λ₁(L)`` (faster mixing = faster error
   spread).
2. **Per-step rollout-Jacobian spectral radius** of each trained checkpoint at each eval
   size: directly tests the architectural error-amplification hypothesis.
3. **Controllability rank + Gramian conditioning** for each (W₁, W₂, I₂) cell, and goal-
   information graph distance for each I₁ pattern. The "easy corner" everything has run in is
   trivially full-rank with distance 0; the script also computes a few partial-interface
   examples that predict where graph bias should pay off in the future I-spoke.

Outputs ``experiments/world_model/analysis.json`` and a divergence-vs-``λ₁(L)`` plot.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from dgr.agents.graph_dreamerv3.checkpoints import load_checkpoint
from dgr.agents.graph_dreamerv3.data import load_transition_dataset
from dgr.analysis.graph_invariants import compute_graph_invariants
from dgr.analysis.model_jacobian import compute_rollout_jacobian_stats
from dgr.analysis.system_invariants import (
    compute_system_invariants,
    linearised_consensus_dynamics,
    linearised_node_independent_dynamics,
)
from dgr.envs.suites.toy_graph_control.scenarios import make_consensus_config

_WM_MODULES = {
    "flat": "dgr.models.world_models.flat_wm",
    "graph_enc_dec": "dgr.models.world_models.graph_enc_dec_wm",
    "graph_rssm": "dgr.models.world_models.graph_rssm_wm",
}

# Topology × sizes seen in the experiments so far (for graph-invariants table).
_TOPOLOGY_SIZES = {
    "ring": [3, 4, 5, 6, 8, 10, 12, 16],
    "grid": [4, 6, 9, 12, 16],
    "kregular": [6, 8, 10, 12, 16],
}

# Eval sizes per topology used by E2/E3 — where Jacobian stats are computed.
_EVAL_SIZES = {"ring": [5, 10, 16], "grid": [6, 12, 16], "kregular": [8, 12, 16]}


def _graph_invariants_table() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for topology, sizes in _TOPOLOGY_SIZES.items():
        rows: list[dict[str, Any]] = []
        for size in sizes:
            cfg = make_consensus_config(size, n_max=16, topology=topology)
            topo = cfg.topology
            assert topo is not None
            inv = compute_graph_invariants(
                topo.senders, topo.receivers, topo.edge_mask, n_real=size
            )
            row = {"size": size, **inv.as_dict()}
            rows.append(row)
        out[topology] = rows
    return out


def _system_invariants_table() -> dict[str, dict[str, Any]]:
    """Easy-corner (dense + full obs) + a couple of partial-interface predictions."""
    out: dict[str, dict[str, Any]] = {}
    for topology in ("ring", "grid", "kregular"):
        for size in _EVAL_SIZES[topology]:
            for dyn in ("consensus", "node_independent"):
                cfg = make_consensus_config(size, n_max=16, topology=topology, dynamics=dyn)
                topo = cfg.topology
                assert topo is not None
                A = (
                    linearised_consensus_dynamics(
                        np.asarray(topo.senders),
                        np.asarray(topo.receivers),
                        np.asarray(topo.edge_mask),
                        n_real=size,
                        alpha=cfg.dynamics.alpha,
                    )
                    if dyn == "consensus"
                    else linearised_node_independent_dynamics(size, cfg.dynamics.alpha)
                )
                actuator_full = np.asarray(cfg.actuator_mask)[:size]
                goal_full = np.asarray(cfg.goal_obs_mask)[:size]
                # Easy-corner cell (the regime everything has run in so far).
                inv_easy = compute_system_invariants(
                    A,
                    np.asarray(topo.senders),
                    np.asarray(topo.receivers),
                    np.asarray(topo.edge_mask),
                    actuator_full,
                    goal_full,
                    n_real=size,
                )
                # Partial-interface predictions: single-leader (node 0) actuation + only
                # node 0 visible. These are predictions for the I-spoke.
                leader = np.zeros(size, dtype=bool)
                leader[0] = True
                inv_leader = compute_system_invariants(
                    A,
                    np.asarray(topo.senders),
                    np.asarray(topo.receivers),
                    np.asarray(topo.edge_mask),
                    leader,
                    leader,
                    n_real=size,
                )
                key = f"{topology}_n{size}_{dyn}"
                out[key] = {
                    "topology": topology,
                    "size": size,
                    "dynamics": dyn,
                    "easy_corner": inv_easy.as_dict(),
                    "single_leader_prediction": inv_leader.as_dict(),
                }
    return out


def _checkpoint_cells(checkpoint_dir: Path) -> list[tuple[str, str, Path]]:
    """Walk ``checkpoint_dir`` for every ``<prefix>_<model>/<model>_world_model.pkl``."""
    cells: list[tuple[str, str, Path]] = []
    for p in sorted(checkpoint_dir.iterdir()):
        if not p.is_dir():
            continue
        for model in _WM_MODULES:
            ckpt = p / f"{model}_world_model.pkl"
            if ckpt.exists() and p.name.endswith(f"_{model}"):
                prefix = p.name[: -(len(model) + 1)]
                cells.append((prefix, model, ckpt))
    return cells


def _infer_topology_dynamics(dataset_path: Path) -> tuple[str, str]:
    meta_path = dataset_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return (meta.get("topology", "ring"), meta.get("dynamics", "consensus"))
    # Fallback: infer from the directory name.
    name = dataset_path.parent.name
    if name == "grid":
        return ("grid", "consensus")
    if name == "kregular":
        return ("kregular", "consensus")
    if name == "nodeindep":
        return ("ring", "node_independent")
    return ("ring", "consensus")


def _model_jacobian_table(
    cells: list[tuple[str, str, Path]], *, n_states: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for prefix, model_name, ckpt in cells:
        payload = load_checkpoint(ckpt)
        dataset_path = Path(payload["train_config"]["dataset"])
        if not dataset_path.exists():
            continue
        dataset = load_transition_dataset(dataset_path)
        topology, dynamics = _infer_topology_dynamics(dataset_path)
        rollout_horizon = int(payload["train_config"].get("rollout_horizon", 1))
        best_epoch = int(payload["train_config"].get("best_epoch", -1))

        predict_module = importlib.import_module(_WM_MODULES[model_name])
        predict_fn = predict_module.predict_next_nodes_single
        params = payload["params"]

        eval_sizes_present = sorted(
            s for s in _EVAL_SIZES.get(topology, []) if s in set(int(x) for x in dataset.n_real)
        )
        for size in eval_sizes_present:
            mask = dataset.n_real == size
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            chosen = rng.choice(idx, size=min(n_states, idx.size), replace=False)
            states = jnp.asarray(dataset.nodes[chosen])
            actions = jnp.asarray(dataset.actions[chosen])
            senders = jnp.asarray(dataset.senders[chosen[0]])
            receivers = jnp.asarray(dataset.receivers[chosen[0]])
            node_mask = jnp.asarray(dataset.node_mask[chosen[0]])
            edge_mask = jnp.asarray(dataset.edge_mask[chosen[0]])
            stats = compute_rollout_jacobian_stats(
                predict_fn,
                params,
                states,
                actions,
                senders,
                receivers,
                node_mask,
                edge_mask,
            )
            rows.append(
                {
                    "prefix": prefix,
                    "model": model_name,
                    "topology": topology,
                    "dynamics": dynamics,
                    "rollout_horizon": rollout_horizon,
                    "best_epoch": best_epoch,
                    "size": size,
                    **stats.as_dict(),
                }
            )
            print(
                f"  {prefix}/{model_name:14} n={size:>2}  "
                f"ρ(J) mean={stats.spectral_radius_mean:.3f} max={stats.spectral_radius_max:.3f} "
                f"frac>1={stats.frac_above_one:.0%}"
            )
    return rows


def _plot_divergence_vs_lambda1(
    invariants: dict[str, list[dict[str, Any]]],
    divergence_paths: dict[str, Path],
    out_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"flat": "#E69F00", "graph_enc_dec": "#56B4E9", "graph_rssm": "#009E73"}
    labels = {
        "flat": "A: Flat MLP",
        "graph_enc_dec": "B: Node-indep. GNN",
        "graph_rssm": "C: Graph RSSM",
    }
    markers = {"ring": "o", "grid": "s", "kregular": "^"}

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    plotted_models: set[str] = set()
    for topology, path in divergence_paths.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        lookup = {row["size"]: row["algebraic_connectivity"] for row in invariants[topology]}
        for model in ("flat", "graph_enc_dec", "graph_rssm"):
            xs, ys = [], []
            for size_str, cell in data[model].items():
                size = int(size_str)
                if size not in lookup:
                    continue
                xs.append(lookup[size])
                ys.append(float(cell["diverge_rate"]))
            if not xs:
                continue
            ax.scatter(
                xs,
                ys,
                color=colors[model],
                marker=markers[topology],
                s=80,
                edgecolor="black",
                linewidth=0.5,
                label=labels[model] if model not in plotted_models else None,
                zorder=3,
            )
            plotted_models.add(model)

    topo_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[t],
            color="w",
            markerfacecolor="#cccccc",
            markeredgecolor="black",
            markersize=8,
            label=t,
        )
        for t in markers
    ]
    leg1 = ax.legend(loc="upper left", title="model", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=topo_handles, loc="lower right", title="topology", frameon=False)

    ax.set_xscale("log")
    ax.set_xlabel(r"Algebraic connectivity  $\lambda_1(L)$  (log scale)")
    ax.set_ylabel("Open-loop divergence rate")
    ax.set_title("Divergence rises with how fast the graph mixes  —  empirical dose-response")
    ax.grid(alpha=0.25, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("experiments/world_model"))
    parser.add_argument("--n-states", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("experiments/world_model/analysis.json"))
    parser.add_argument(
        "--plot-out", type=Path, default=Path("docs/assets/divergence_vs_spectral_gap.png")
    )
    args = parser.parse_args()

    print("=== graph invariants ===")
    inv = _graph_invariants_table()
    for topology, rows in inv.items():
        for row in rows:
            print(
                f"  {topology:8} n={row['size']:>2}  λ₁(L)={row['algebraic_connectivity']:.4f}  "
                f"diam={row['diameter']:>2}  ⟨deg⟩={row['mean_degree']:.2f}"
            )

    print("\n=== system invariants (easy corner + single-leader prediction) ===")
    sys_inv = _system_invariants_table()
    for key, val in sys_inv.items():
        easy = val["easy_corner"]
        leader = val["single_leader_prediction"]
        print(
            f"  {key:24} easy(rank/{val['size']}={easy['controllability_rank']})  "
            f"leader(rank/{val['size']}={leader['controllability_rank']}, "
            f"goal-info dist={leader['goal_info_mean_distance']:.2f})"
        )

    print("\n=== model rollout-Jacobian spectral radius ===")
    cells = _checkpoint_cells(args.checkpoint_dir)
    jac_rows = _model_jacobian_table(cells, n_states=args.n_states)

    payload = {
        "graph_invariants": inv,
        "system_invariants": sys_inv,
        "model_jacobians": jac_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSaved analysis → {args.out}")

    divergence_paths = {
        "ring": Path("docs/assets/multistep_rollout_data.json"),
        "grid": Path("docs/assets/multistep_rollout_grid_data.json"),
        "kregular": Path("docs/assets/multistep_rollout_kregular_data.json"),
    }
    _plot_divergence_vs_lambda1(inv, divergence_paths, args.plot_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
