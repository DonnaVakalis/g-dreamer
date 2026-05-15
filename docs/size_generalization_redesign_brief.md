# Size Generalization Figure Redesign Brief

Use this brief to remake `docs/assets/size_generalization.png` from the data in `docs/assets/size_generalization_data.json`.

## Goal

Produce a clearer, prettier figure that communicates the progression from **A -> B -> C**:

- **A: Flat MLP** memorizes training sizes and collapses out of distribution.
- **B: Node-independent GNN** generalizes state prediction across sizes, but goal prediction is still poor.
- **C: Graph RSSM** is the only model that generalizes gracefully on both state and goal prediction.

The new figure should make that story obvious at a glance.

## Main problem with the current figure

The current image reuses color for two different meanings:

- In the **top row**, color means **model identity** (`A`, `B`, `C`).
- In the **bottom row**, color means **different node trajectories**.

That semantic reuse is confusing. In the remake, **model colors must mean model identity everywhere**. If another visual element needs encoding in the lower panels, use neutrals, line style, alpha, or a completely separate non-conflicting palette.

## Inputs

- Current figure: `docs/assets/size_generalization.png`
- Source data: `docs/assets/size_generalization_data.json`

The JSON has two main sections:

- `metrics`
  - keys: `flat`, `graph_enc_dec`, `graph_rssm`
  - sizes: `3, 4, 5, 6, 8, 10, 12, 16`
  - metrics per size: `x_mse`, `goal_mse`
- `trajectories`
  - one entry per model
  - each entry includes:
    - `size` (currently `16`)
    - `x_mse`
    - `goal_mse`
    - `gt_x`: shape `50 x 16`
    - `pred_x`: shape `50 x 16`

Training sizes are `n in {4, 5, 6}`. The others are evaluation-only, especially OOD sizes `8, 10, 12, 16`.

## Required narrative emphasis

These are the key points the figure should support:

1. `A` looks great in-distribution and then fails badly as graph size increases.
2. `B` fixes size generalization for `x_mse` but still has consistently weak `goal_mse`.
3. `C` is the best overall because it stays stable on both metrics.

Useful anchor numbers at `n=16`:

- `A`: `x_mse=4.985`, `goal_mse=5.316`
- `B`: `x_mse=0.061`, `goal_mse=0.703`
- `C`: `x_mse=0.099`, `goal_mse=0.128`

## Design direction

Prefer a clean, publication-style figure rather than a dashboard. Make it legible in README context and also when scaled down.

### Visual semantics

- Reserve one consistent color per model across the whole figure.
- Do not reuse those model colors to mean node identity, time series identity, or error magnitude.
- Suggested model palette:
  - `A / Flat MLP`: warm amber or rust
  - `B / Node-indep. GNN`: muted blue
  - `C / Graph RSSM`: teal or green
- Use gray or black for non-model semantics:
  - ground truth
  - predicted overlays
  - annotations
  - train/OOD separators

### Recommended layout

Use a **2-row layout** with a strong top story and a simpler bottom explanation.

#### Top row: two main metric panels

- Left: `State prediction (x_mse)` vs graph size
- Right: `Goal prediction (goal_mse)` vs graph size
- Use lines with markers instead of grouped bars
  - lines make the A -> B -> C progression easier to read
  - log y-scale is fine and probably still necessary
- Shade or bracket the training region `n in {4,5,6}`
- Label OOD region clearly
- Keep model legend simple and prominent

#### Bottom row: one panel per model, but avoid the spaghetti problem

The lower row should help explain *why* A, B, and C differ at `n=16`, without turning into 16-color line chaos.

Preferred option:

- Show **three model-specific panels**, one for `A`, one for `B`, one for `C`
- In each panel:
  - use the model color only for that panel title or border/accent
  - plot a small number of representative node trajectories, not all 16
  - use **solid dark gray** for ground truth
  - use the **model color** for prediction
  - use 3 to 4 fixed node indices across all three panels so comparisons are fair
  - keep those node traces thin and semi-transparent

This preserves model-color meaning and makes each panel much easier to read.

Acceptable alternative if trajectory overlays still look too busy:

- Replace the lower row with **absolute error heatmaps** (`|pred_x - gt_x|`) for A, B, C at `n=16`
- Use a single shared sequential colormap for error intensity
- Keep model identity in titles, panel labels, or frame accents only

If choosing between the two, prefer the option that looks cleanest and most interpretable.

## What to avoid

- No grouped bar charts unless there is a strong reason.
- No full 16-color spaghetti plots.
- No gray matplotlib background.
- No duplicate legends if one shared legend is enough.
- No semantic color collisions.
- No tiny captions embedded inside panel titles.

## Annotations and labels

Use concise text that reinforces the story:

- Main title should emphasize **world model size generalization**
- Subtitle or caption can state:
  - trained on `n in {4,5,6}`
  - evaluated across `n in {3,4,5,6,8,10,12,16}`
- Panel text can call out:
  - `A: memorizes train sizes, collapses OOD`
  - `B: state generalizes, goal does not`
  - `C: best overall generalization`

## Output expectations

- Save the remade figure over `docs/assets/size_generalization.png` or provide a candidate alongside it first.
- If you write a plotting script, keep it in the repo and make it reusable.
- The result should look good both full-size and at README width.

## Success criteria

The remake is successful if:

1. A reader can immediately tell what changed from `A` to `B` to `C`.
2. Color meaning stays consistent across the entire figure.
3. The top-row metrics tell the quantitative story quickly.
4. The bottom row adds intuition without clutter.
5. The overall figure feels cleaner and more intentional than the current version.
