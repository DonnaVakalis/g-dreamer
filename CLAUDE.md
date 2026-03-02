# g_dreamer — Claude Code guide

## Commands

```bash
poetry install --with dev,upstream   # full install
pytest                               # run tests
pre-commit run --all-files           # lint + format + typecheck
```

## Architecture

**Package:** `dgr` (src/dgr/)

**Core modules (implemented):**
- `interface/graph_spec.py` — `GraphSpec` (static shape spec) + `Graph` (padded/masked data)
- `models/message_passing.py` — masked, permutation-equivariant aggregation
- `envs/suites/toy_graph_control/consensus.py` — ring-topology consensus env (main testbed)
- `train.py` — orchestrates runs; invokes upstream DreamerV3 via subprocess

Most files under `agents/`, `envs/adapters/`, `envs/wrappers/`, and several env suites
are **stubs (1 LOC)**. Don't over-engineer them — match the minimal skeleton style.

**Upstream DreamerV3** lives in `third_party/dreamerv3/` as a git submodule.
Never modify it. It is invoked as a subprocess from `train.py`, not imported.

## Graph contract

All graph data uses static shapes + padding + boolean masks:

```
Graph.nodes       (n_max, f_n)  float32
Graph.edges       (e_max, f_e)  float32
Graph.senders     (e_max,)      int32
Graph.receivers   (e_max,)      int32
Graph.node_mask   (n_max,)      bool
Graph.edge_mask   (e_max,)      bool
Graph.globals     (f_g,)        float32   (optional)
```

Real elements occupy the first `n_real` / `e_real` positions; the rest are zeroed and
masked. Never assume the full array is real — always respect masks.

## JAX conventions

- All functions must be pure (no side effects, no Python mutation).
- Use `jnp.where()` and `.at[].add()` for conditional/scatter ops — no Python loops over arrays.
- Split RNG keys before use: `key, subkey = jax.random.split(key)`.
- Static shapes are required for JIT — enforce via GraphSpec validation.

## Code style

- Domain objects: `@dataclass(frozen=True)`.
- Always include `from __future__ import annotations` at the top of every module.
- Imports: stdlib → third-party → local.
- Line length: 100 (ruff enforced).
- Numerical assertions in tests: `assert jnp.allclose(a, b, atol=1e-5)`.
- Don't add docstrings, comments, or type annotations to code you didn't change.

## Tooling notes

- Ruff excludes `third_party/**` — don't add it to lint scope.
- mypy runs via pre-commit; `jaxtyping` is installed but not yet used in source.
- Experiments log to `experiments/runs/`; don't commit run artifacts.
