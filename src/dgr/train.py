"""
This file serves as the main training orchestrator for the g_dreamer project...

The file is essentially a wrapper and launcher that will evolve to support
the project's graph-based variants,
(once they're implemented in the agents/ directory).

Current usage examples:

# Run baseline DreamerV3 on crafter environment
python -m dgr.train agent=baseline env=crafter_debug steps=2000

# With custom log directory
python -m dgr.train agent=baseline env=crafter_debug steps=2000 logdir=/path/to/logs

# Pass extra args to upstream DreamerV3 after the key=value pairs (e.g., set seeds)
poetry run python -m dgr.train agent=baseline env=crafter_debug --seed 0
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class RunSpec:
    agent: str
    env: str
    steps: int
    logdir: Path
    extra_upstream_args: List[str]


def _parse_kv(argv: List[str]) -> Dict[str, str]:
    """Parse CLI tokens like key=value into a dict; leave others untouched."""
    out: Dict[str, str] = {}
    for tok in argv:
        if "=" in tok and not tok.startswith("--"):
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _strip_kv(argv: List[str]) -> List[str]:
    """Remove key=value tokens (our mini-config), keep all other tokens."""
    return [tok for tok in argv if not ("=" in tok and not tok.startswith("--"))]


def _env_to_upstream_configs(env: str) -> List[str]:
    """
    Map our env names to upstream DreamerV3 config blocks.

    env=crafter_debug -> --configs crafter debug
    """
    mapping = {
        "crafter_debug": ["crafter", "debug"],
        # add later:
        # "dmc_debug": ["dmc", "debug"],
        # "atari_debug": ["atari", "debug"],
    }
    if env not in mapping:
        raise ValueError(f"Unknown env={env!r}. Supported: {', '.join(sorted(mapping))}")
    return mapping[env]


def _default_logdir(agent: str, env: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("experiments") / "runs" / f"{env}__{agent}__{stamp}"


def _run_upstream_dreamerv3(spec: RunSpec) -> int:
    """
    Runs upstream DreamerV3 unchanged via subprocess.
    Assumes dreamerv3 is vendored at third_party/dreamerv3.
    """
    repo_root = Path(__file__).resolve().parents[2]
    upstream_root = repo_root / "third_party" / "dreamerv3"
    main_py = upstream_root / "dreamerv3" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Expected upstream at: {main_py}")

    spec.logdir.mkdir(parents=True, exist_ok=True)

    configs = _env_to_upstream_configs(spec.env)

    cmd = [
        sys.executable,
        str(main_py),
        "--logdir",
        str(spec.logdir),
        "--configs",
        *configs,
        "--run.steps",
        str(spec.steps),
        *spec.extra_upstream_args,
    ]

    # Mimic how upstream expects to be run (like your Group 1 script)
    env = os.environ.copy()
    env["PYTHONPATH"] = "." if "PYTHONPATH" not in env else f".:{env['PYTHONPATH']}"

    print("\n[dgr.train] Running upstream DreamerV3 baseline")
    print(f"[dgr.train] cwd: {upstream_root}")
    print(f"[dgr.train] logdir: {spec.logdir}")
    print(f"[dgr.train] cmd: {' '.join(cmd)}\n")

    proc = subprocess.run(cmd, cwd=str(upstream_root), env=env)
    return proc.returncode


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    kv = _parse_kv(argv)
    rest = _strip_kv(argv)

    agent = kv.get("agent", "baseline")
    env = kv.get("env", "crafter_debug")
    steps = int(kv.get("steps", "2000"))
    logdir = Path(kv["logdir"]) if "logdir" in kv else _default_logdir(agent, env)

    spec = RunSpec(
        agent=agent,
        env=env,
        steps=steps,
        logdir=logdir,
        extra_upstream_args=rest,
    )

    if spec.agent == "baseline":
        return _run_upstream_dreamerv3(spec)

    raise ValueError(f"Unknown agent={spec.agent!r}. Supported: baseline (for now).")


if __name__ == "__main__":
    raise SystemExit(main())
