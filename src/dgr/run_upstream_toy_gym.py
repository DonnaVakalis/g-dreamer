"""
This is a bridge that registers the env, then calls
upstream DreamerV3 on the toy graph control environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


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

    # Optional: mimic running from inside the upstream repo.
    os.chdir(upstream_root)

    from dreamerv3 import main as upstream_main

    upstream_main.main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
