# G-Dreamer Sprint: Graph World Model Handoff

> **Purpose:** Build a minimal but working graph world model on top of the existing G-Dreamer codebase, demonstrating that graph-structured world models generalize to unseen graph sizes. Each layer is independently shippable. Work in order.
>
> **Repo:** `https://github.com/DonnaVakalis/g-dreamer`
>
> **Why this matters:** This is a research artifact for a conversation with AMI Labs (Yann LeCun's world-model startup). The punchline is: physical systems are natively graph-structured, and world models that operate on graphs can generalize across topologies — something flat-vector world models cannot do.

---

## Orientation: What Already Exists

Read these files before writing any code:

| File | What it contains |
|------|-----------------|
| `docs/graph_obs_contract.md` | The canonical graph observation spec. Read this first. |
| `src/dgr/interface/graph_spec.py` | `GraphSpec`, `Graph` dataclasses, `validate_graph()`, `permute_nodes()` |
| `src/dgr/models/message_passing.py` | Masked message-passing primitive (the GNN building block) |
| `src/dgr/envs/suites/toy_graph_control/consensus.py` | Ring-topology consensus environment |
| `src/dgr/envs/suites/toy_graph_control/core.py` | Base env class, graph construction, step logic |
| `third_party/` | DreamerV3 upstream (reference for RSSM architecture) |
| `pyproject.toml` | Dependencies (Poetry). Python 3.11. |

### The consensus environment

- Nodes on a directed ring. Each node has state `x_i` and goal `goal_i`.
- Node features: `[x_i, goal_i]` (f_n = 2). No edge features (f_e = 0). No globals (f_g = 0).
- Actions: per-node continuous values.
- Dynamics: each node's state evolves based on its neighbors' states and the applied action (diffusion + control).
- The env is JAX/JIT compatible and produces `Graph` observations with padded/masked fixed-size arrays.
- `n_real` (actual number of nodes) can vary; `n_max` (padded capacity) is fixed.

### Key constraint

All operations **must respect `node_mask` and `edge_mask`**. Padding slots must never influence real-node computations. This is the central invariant of the codebase. Use the existing message-passing primitive — it already handles masking.

---

## Implementation Status (as of 2026-05-05)

### What has been built (deviations from original handoff noted)

**Decisions made:**
- All models use **JAX + Optax** (not PyTorch as originally suggested) — existing infrastructure already JAX
- Script names kept as-is (not renamed to match handoff)
- World model code lives in `src/dgr/models/world_models/` (as handoff specified)
- **Variant B replaced (2026-04-30):** Original `graph_enc_dec_wm.py` was stuck at ~0.5 loss due to collapsing per-node actions to a scalar mean, making the dynamics blind to spatial action structure. Replaced with Variant B2: GNN encoder → per-node latents → per-node-independent MLP dynamics (no cross-node message passing) → per-node decoder. This creates a cleaner ablation: A (no graph) → B (graph everywhere except dynamics coupling) → C (graph everywhere). See "Future ablations" note below for the discarded mean-pool variant.

**Layer 1 — COMPLETE:**
- `src/dgr/models/world_models/flat_wm.py` — Variant A ✓
- `src/dgr/models/world_models/graph_enc_dec_wm.py` — Variant B ✓
- `src/dgr/models/world_models/graph_rssm_wm.py` — Variant C (refactored from `agents/graph_dreamerv3/minimal_world_model.py`) ✓
- `scripts/train_minimal_graph_world_model.py` — extended with `--model {flat,graph_enc_dec,graph_rssm}` and `--wandb` ✓
- `scripts/eval_minimal_graph_world_model.py` — extended with per-size MSE table, `--episodes`, `--wandb` ✓
- Dataset collected: `experiments/world_model/consensus_transitions_large.npz` (800k transitions, gitignored) ✓

**Layer 1 — COMPLETE** (cluster scripts deprioritised; full runs done locally)

**Layer 2 — COMPLETE:**
- `scripts/eval_minimal_graph_world_model.py` — extended with per-feature (x_mse + goal_mse) breakdown, `_figure_data.json` output for re-styling ✓
- `scripts/plot_size_generalization.py` — generates Layer 2 figure ✓
- `docs/assets/size_generalization.png` — committed ✓
- `docs/assets/size_generalization_data.json` — raw figure source data committed ✓

**Key results (60 epochs, train n ∈ {4,5,6}, eval n ∈ {3,8,10,12,16}):**
```
Variant   in-dist loss   OOD x_mse (n=16)   OOD goal_mse (n=16)
A (flat)     ~0.000          4.985              5.316   ← collapses
B (node-indep GNN)  0.347   0.061              0.703   ← x ok, goal broken everywhere
C (graph RSSM)      0.108   0.099              0.128   ← graceful on both
```
Finding: graph structure in encoder alone gives size invariance on x (B and C both flat OOD).
Cross-node coupling in dynamics is needed for correct multi-feature prediction (C 3× lower total loss than B).

**Layer 3 — COMPLETE:**
- README rewrite ✓

### Dataset spec
```
File: experiments/world_model/consensus_transitions_large.npz  (82MB compressed, gitignored)
Sizes: 3, 4, 5, 6, 8, 10, 12, 16  (100k transitions each = 800k total)
n_max: 16  (all transitions padded to 16 nodes)
Episodes/size: 2000 × 50 steps
Policy: random (uniform [-1, 1] per node, masked to real nodes)
Seed: 0

Train split:    --train-sizes 4,5,6
OOD eval:       n_real in {3, 8, 10, 12, 16}  (non-overlapping by design)
```

### Early training observations (5-epoch smoke test)
- **flat**: converges very fast (loss ~0.0003) — memorizes fixed topology, will fail OOD
- **graph_enc_dec**: stuck (~0.500) — mean-pool bottleneck may need higher LR or capacity
- **graph_rssm**: converging steadily — expected to generalize

---

## LAYER 1: Three World Model Variants

### Shared infrastructure (built)

#### 1a. Data collection
`scripts/collect_consensus_world_model_data.py` — working, documented

#### 1b. Training harness
`scripts/train_minimal_graph_world_model.py`:
- `--model {flat,graph_enc_dec,graph_rssm}`
- `--train-sizes` (default `4,5,6`)
- `--wandb` flag
- Saves checkpoint as `{model}_world_model.pkl`, loss curve PNG, metrics JSON

#### 1c. Evaluation harness
`scripts/eval_minimal_graph_world_model.py`:
- `--train-sizes` / `--eval-sizes` (labels in-dist vs OOD)
- `--episodes` for averaging
- `--wandb` flag
- Prints per-size MSE table, saves figure + JSON

### Variant A: Flat-vector baseline
`src/dgr/models/world_models/flat_wm.py`
```
Input:  concat(flatten(nodes * node_mask), node_mask, actions)  →  fixed vector
Model:  MLP(hidden_dims=[256, 256])
Output: reshape to (n_max, f_n), apply node_mask
```

### Variant B: GNN encoder/decoder, per-node-independent dynamics
`src/dgr/models/world_models/graph_enc_dec_wm.py`
```
Encoder:   node_dim → hidden → message passing → latent per node h_i
Dynamics:  MLP(concat(h_i, action_i)) → h_i_next   [per node, NO cross-node communication]
Decoder:   MLP per node: h_i_next → predicted node features[i]
```
Ablation intent: isolates whether cross-node coupling in the dynamics specifically is necessary,
vs. graph thinking only at the encoder/decoder boundary. Dynamics here are node-independent —
each node transitions based only on its own latent and its own action, with no neighbor influence.

> **Future ablation — B1 (mean-pool variant):** A fourth variant worth adding for a paper-quality
> ablation table would be: GNN encoder → masked **mean pool** → global latent z → flat MLP
> dynamics → broadcast z_next to each node → per-node decoder. This would add a B1→B2 step that
> isolates per-node vs. global latent representation, independent of the dynamics coupling question.
> Discarded for the AMI demo because B1→B2 conflates three simultaneous changes (global vs per-node
> latent, scalar vs per-node action, broadcast vs per-node decoder), weakening the ablation story.

### Variant C: Full graph-structured RSSM
`src/dgr/models/world_models/graph_rssm_wm.py`
```
Encoder:   node_dim → hidden → message passing → latent per node h_t
Dynamics:  concat(h_t, action) per node → message passing → h_{t+1}
Decoder:   MLP per node: h_{t+1}[i] → predicted node features[i]
```

---

## LAYER 2: Generalization Experiment and Figure

### Experiment protocol
1. All three models trained **only** on n_real ∈ {4, 5, 6}
2. Evaluate on:
   - In-distribution: n_real ∈ {4, 5, 6}
   - OOD: n_real ∈ {3, 8, 10, 12, 16}
3. Metrics: mean per-step MSE on real nodes

### Figure spec
Save as `docs/assets/size_generalization.png`

**Panel A:** Bar/scatter — x: graph size, y: MSE, 3 series (A/B/C), dashed line separating train/OOD sizes
**Panel B:** Trajectory comparison at one OOD size (n=10 or n=16), ground truth vs predicted

Style: matplotlib, clean (no gray bg), colorblind-friendly, legible at 700px wide

---

## LAYER 3: README

Rewrite `README.md` with sections:
1. Hook (1-2 sentences)
2. Key Result (figure + caption)
3. Architecture (diagram)
4. Quick Start
5. Project Structure
6. Research Context (world models + graph structure, no AMI mention)
7. Status + Next steps
8. Citation/License

---

## General Conventions

- JAX throughout (no PyTorch)
- `src/dgr/models/world_models/` for model code
- `scripts/` for runnable scripts
- `experiments/world_model/` for outputs (gitignored via `experiments/**`)
- `docs/assets/` for figures committed to repo
- Use existing `Graph`/`GraphSpec` types and `masked_message_passing` primitive
- `--wandb` flag pattern: off by default, uses `ExperimentMetadata` + `wandb_init_kwargs`
- All pre-commit checks must pass: ruff, ruff-format, mypy
