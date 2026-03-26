"""
This is a bridge that registers the env, then calls
upstream DreamerV3 on the toy graph control environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _parse_seed(argv: list[str]) -> int:
    for i, tok in enumerate(argv):
        if tok == "--seed" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                pass
    return 0


def _maybe_init_wandb(argv: list[str]) -> None:
    scenario = os.environ.get("DGR_WANDB_SCENARIO")
    if not scenario:
        return
    variant = os.environ.get("DGR_WANDB_VARIANT", "custom")
    seed = _parse_seed(argv)

    from dgr.experiments.metadata import ExperimentMetadata
    from dgr.experiments.naming import canonical_policy_name
    from dgr.experiments.wandb_utils import wandb_init_kwargs

    meta = ExperimentMetadata(
        run_type="dreamer_train",
        scenario=scenario,
        policy_or_agent=canonical_policy_name("dreamer_flat"),
        seed=seed,
        variant=variant,
    ).with_timestamp()

    import wandb

    wandb.init(**wandb_init_kwargs(meta))  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    repo_root = Path(__file__).resolve().parents[2]
    upstream_root = repo_root / "third_party" / "dreamerv3"

    if not upstream_root.exists():
        raise FileNotFoundError(f"Expected upstream checkout at {upstream_root}")

    # Make vendored upstream importable as `dreamerv3`.
    sys.path.insert(0, str(upstream_root))

    from dgr.envs.adapters.toy_graph_control_gym import register_toy_consensus_envs

    register_toy_consensus_envs()

    # Resolve --logdir to an absolute path before chdir changes the working directory.
    for i, tok in enumerate(argv):
        if tok == "--logdir" and i + 1 < len(argv):
            argv[i + 1] = str(Path(argv[i + 1]).resolve())
            break

    # Pre-init wandb with full metadata so dreamer's WandBOutput re-uses this run.
    _maybe_init_wandb(argv)

    # Optional: mimic running from inside the upstream repo.
    os.chdir(upstream_root)

    from dreamerv3 import main as upstream_main

    upstream_main.main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
