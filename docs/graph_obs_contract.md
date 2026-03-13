# Graph Observation Contract

This document defines the canonical padded-and-masked graph observation used by `dgr`.
The contract is designed for static shapes so observations can be batched and JIT-compiled
without ragged tensors.

The source of truth for the runtime objects is [src/dgr/interface/graph_spec.py](../src/dgr/interface/graph_spec.py).

## Goals

- Keep graph observations shape-stable across episodes and environments.
- Separate "real structure" from "padding slots" with explicit masks.
- Prevent padded values from leaking into models or flattened observations.
- Support permutation-equivariant graph operations with simple array semantics.

## Canonical Types

`GraphSpec` defines the static capacity of a graph observation:

```python
GraphSpec(
    n_max: int,  # max node slots
    e_max: int,  # max edge slots
    f_n: int,    # node feature width
    f_e: int,    # edge feature width
    f_g: int,    # global feature width
)
```

`Graph` is the actual observation payload:

```python
Graph(
    nodes,      # (N_max, F_n) float32
    edges,      # (E_max, F_e) float32
    senders,    # (E_max,) int32
    receivers,  # (E_max,) int32
    node_mask,  # (N_max,) bool
    edge_mask,  # (E_max,) bool
    globals,    # (F_g,) float32
)
```

Throughout this document:

- `N_max == spec.n_max`
- `E_max == spec.e_max`
- `F_n == spec.f_n`
- `F_e == spec.f_e`
- `F_g == spec.f_g`

## Field Semantics

### `nodes`

- Shape: `(N_max, F_n)`
- Dtype: `float32`
- Meaning: per-node features for each node slot
- Rule: entries where `node_mask[i] == False` are padding slots

### `edges`

- Shape: `(E_max, F_e)`
- Dtype: `float32`
- Meaning: per-edge features for each edge slot
- Rule: entries where `edge_mask[e] == False` are padding slots
- Note: `F_e` may be `0`, so `edges.shape` can be `(E_max, 0)`

### `senders` and `receivers`

- Shape: `(E_max,)`
- Dtype: `int32`
- Meaning: directed edge list in COO-style form
- Interpretation: edge slot `e` connects `senders[e] -> receivers[e]`
- Rule: indices must stay in `[0, N_max - 1]`
- Rule: even masked edges should preferably carry in-range indices for easier debugging

### `node_mask`

- Shape: `(N_max,)`
- Dtype: `bool`
- Meaning: `True` for real nodes, `False` for padded node slots

### `edge_mask`

- Shape: `(E_max,)`
- Dtype: `bool`
- Meaning: `True` for real edges, `False` for padded edge slots

### `globals`

- Shape: `(F_g,)`
- Dtype: `float32`
- Meaning: graph-level features
- Note: `F_g` may be `0`, so `globals.shape` can be `(0,)`

## Core Invariants

Any valid graph observation must satisfy:

1. Shapes match the static `GraphSpec`.
2. `senders.shape == receivers.shape == (E_max,)`.
3. `node_mask.shape == (N_max,)`.
4. `edge_mask.shape == (E_max,)`.
5. `senders` and `receivers` use `int32`.
6. `node_mask` and `edge_mask` use `bool`.
7. Every sender and receiver index is in range for `N_max`.

The project currently validates these conditions in
[src/dgr/interface/graph_spec.py](../src/dgr/interface/graph_spec.py).

## Padding Rules

Padding is explicit. Real graph content occupies a prefix or subset of the fixed-size arrays,
and masks declare which slots are meaningful.

### Padded nodes

If `node_mask[i] == False`:

- `nodes[i]` should usually be zeroed by producers
- downstream consumers must treat the slot as invalid regardless of stored values

### Padded edges

If `edge_mask[e] == False`:

- `edges[e]` should usually be zeroed by producers
- `senders[e]` and `receivers[e]` are ignored semantically
- indices should still remain in-range to satisfy validation

### Why masks are mandatory

Static shapes alone are not enough. Two observations that differ only in the junk values stored in
padded slots should behave identically after masking. That is the contract downstream code relies on.

## Consumer Expectations

### Message passing

[src/dgr/models/message_passing.py](../src/dgr/models/message_passing.py) assumes:

- messages are gated by `edge_mask`
- outputs are gated by `node_mask`
- masked edges contribute nothing
- masked nodes produce zero output

In other words, padded structure must not affect real-node outputs.

### Flattening to vectors

[src/dgr/envs/wrappers/flatten_graph.py](../src/dgr/envs/wrappers/flatten_graph.py) assumes:

- node features are multiplied by `node_mask`
- edge features are multiplied by `edge_mask`
- indices for masked edges are zeroed before flattening
- masks themselves may be included as explicit features

This wrapper exists to guarantee "no padding leakage" when adapting graph observations to
flat-vector agents.

## Recommended Producer Conventions

Environment and adapter code should follow these conventions when constructing a `Graph`:

1. Fill real node slots first, then padded node slots.
2. Fill real edge slots first, then padded edge slots.
3. Zero node features in padded node slots.
4. Zero edge features in padded edge slots.
5. Zero global features unless they are intentionally used.
6. Keep masked-edge indices in-range, typically `0`.
7. Call `validate_graph(g, spec)` before returning a new observation.

These are stronger than the minimum validator in a few places, but they make bugs easier to spot.

## Toy Control Example

The toy consensus environment uses:

- `f_n = 2`
- node features `nodes[i] = [x_i, goal_i]`
- `f_e = 0`
- `f_g = 0`

For `n_real = 4`, `n_max = 8`, and a directed ring topology:

```text
node_mask  = [1, 1, 1, 1, 0, 0, 0, 0]
edge_mask  = [1, 1, 1, 1, 1, 1, 1, 1, 0, ..., 0]
senders    = [0, 1, 2, 3, 0, 1, 2, 3, 0, ..., 0]
receivers  = [1, 2, 3, 0, 3, 0, 1, 2, 0, ..., 0]
nodes[i]   = [x_i, goal_i] for real nodes, zero for padded nodes
edges[e]   = empty feature vectors because F_e = 0
globals    = empty because F_g = 0
```

This is implemented in
[src/dgr/envs/suites/toy_graph_control/core.py](../src/dgr/envs/suites/toy_graph_control/core.py).

## Permutation Convention

Node order is not semantically meaningful.
If nodes are permuted, the graph should represent the same structure after sender and receiver indices
are updated consistently.

The helper `permute_nodes()` uses this convention:

- `nodes_perm[i] = nodes[perm[i]]`
- `perm` maps `new_index -> old_index`
- sender and receiver indices are remapped with the inverse permutation

See [src/dgr/interface/graph_spec.py](../src/dgr/interface/graph_spec.py) for the implementation.

## Non-Goals

This contract does not currently define:

- heterogeneous node or edge types
- variable feature widths within one graph
- sparse tensor storage formats beyond sender/receiver index arrays
- temporal stacking inside the graph object itself

Those can be added later without weakening the padded-and-masked core contract.
