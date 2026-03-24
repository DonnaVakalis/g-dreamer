from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from dgr.experiments.metadata import ExperimentMetadata
from dgr.experiments.naming import canonical_policy_name
from dgr.experiments.wandb_utils import wandb_init_kwargs

_SIZE_SIGNATURES = {
    (512, 4, 64): "size1m",
    (2048, 16, 256): "size12m",
    (3072, 24, 384): "size25m",
    (4096, 32, 512): "size50m",
    (6144, 48, 768): "size100m",
    (8192, 64, 1024): "size200m",
    (12288, 96, 1536): "size400m",
}

_ENV_ID_TO_SCENARIO = {
    "DGRToyConsensusDebugDense-v0": "debug_ring_dense",
    "DGRToyConsensusDebugSparse-v0": "debug_ring_sparse",
    "DGRToyConsensusTrainDense-v0": "train_ring_dense",
}


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _known_scenarios() -> list[str]:
    path = (
        Path(__file__).resolve().parents[1] / "src/dgr/envs/suites/toy_graph_control/scenarios.py"
    )
    text = path.read_text()
    names = re.findall(r'if name == "([^"]+)"', text)
    return sorted(set(names), key=len, reverse=True)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _infer_scenario(run_dir: Path, cfg: dict[str, Any]) -> str:
    task = str(cfg.get("task", ""))
    logdir = str(cfg.get("logdir", ""))
    haystacks = [task, logdir, run_dir.name, str(run_dir)]

    for candidate in _known_scenarios():
        if any(candidate in h for h in haystacks):
            return candidate

    if task.startswith("gym_"):
        env_id = task[len("gym_") :]
        if env_id in _ENV_ID_TO_SCENARIO:
            scenario = _ENV_ID_TO_SCENARIO[env_id]
            if any(name.startswith(f"{scenario}_") for name in _known_scenarios()):
                raise ValueError(
                    f"Task {task!r} only resolves to coarse scenario {scenario!r}, "
                    "which is ambiguous in this repo. Pass --scenario with the full "
                    "scenario name, for example debug_ring_sparse_hidden_smooth_aligned."
                )
            return scenario

    for candidate in sorted(_ENV_ID_TO_SCENARIO.values(), key=len, reverse=True):
        if candidate in logdir:
            return candidate
    raise ValueError(
        f"Could not infer scenario from task={task!r}, logdir={logdir!r}, run_dir={run_dir!r}. "
        "Pass --scenario with the full scenario name."
    )


def _infer_run_type(cfg: dict[str, Any]) -> str:
    script = str(cfg.get("script", ""))
    if script == "eval_only":
        return "dreamer_eval"
    return "dreamer_train"


def _infer_size_label(cfg: dict[str, Any]) -> str:
    agent = cfg.get("agent", {})
    try:
        deter = int(agent["dyn"]["rssm"]["deter"])
        depth = int(agent["enc"]["simple"]["depth"])
        units = int(agent["enc"]["simple"]["units"])
    except (KeyError, TypeError, ValueError):
        return "custom"
    return _SIZE_SIGNATURES.get((deter, depth, units), "custom")


def _infer_agent_name(cfg: dict[str, Any]) -> str:
    random_agent = bool(cfg.get("random_agent", False))
    if random_agent:
        return "random"
    return canonical_policy_name("dreamer_flat")


def _infer_timestamp(run_dir: Path) -> str | None:
    name = run_dir.name
    parts = name.rsplit("__", 1)
    if len(parts) == 2 and len(parts[1]) == 15 and "_" in parts[1]:
        return parts[1]
    return None


def _build_metadata(
    run_dir: Path,
    cfg: dict[str, Any],
    *,
    scenario_override: str | None,
    agent_override: str | None,
    variant_override: str | None,
) -> ExperimentMetadata:
    scenario = scenario_override or _infer_scenario(run_dir, cfg)
    agent_name = canonical_policy_name(agent_override or _infer_agent_name(cfg))
    variant = variant_override or _infer_size_label(cfg)
    run_steps = cfg.get("run", {}).get("steps")
    run_steps = int(float(run_steps)) if run_steps is not None else None
    seed = int(cfg.get("seed", 0))
    return ExperimentMetadata(
        run_type=_infer_run_type(cfg),
        scenario=scenario,
        policy_or_agent=agent_name,
        seed=seed,
        variant=variant,
        run_steps=run_steps,
        timestamp=_infer_timestamp(run_dir),
    )


def _iter_metric_rows(path: Path):
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{lineno}")
        yield row


def _scenario_stats_or_empty(name: str) -> dict[str, Any]:
    try:
        from dgr.envs.suites.toy_graph_control.scenarios import get_scenario, scenario_stats

        return scenario_stats(get_scenario(name))
    except Exception:
        return {}


def _upload_run(
    run_dir: Path,
    *,
    scenario_override: str | None,
    agent_override: str | None,
    variant_override: str | None,
    dry_run: bool,
) -> None:
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {run_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.jsonl in {run_dir}")

    cfg = _load_yaml(config_path)
    meta = _build_metadata(
        run_dir,
        cfg,
        scenario_override=scenario_override,
        agent_override=agent_override,
        variant_override=variant_override,
    )

    rows = list(_iter_metric_rows(metrics_path))
    stats = _scenario_stats_or_empty(meta.scenario)
    wb_kwargs = wandb_init_kwargs(meta)
    wb_kwargs["config"] = {
        **wb_kwargs["config"],
        **stats,
        "dreamer_logdir": str(run_dir),
        "dreamer_task": cfg.get("task"),
        "dreamer_script": cfg.get("script"),
        "dreamer_logger_outputs": cfg.get("logger", {}).get("outputs"),
        "jax_platform": cfg.get("jax", {}).get("platform"),
        "jax_compute_dtype": cfg.get("jax", {}).get("compute_dtype"),
        "run_envs": cfg.get("run", {}).get("envs"),
        "run_eval_envs": cfg.get("run", {}).get("eval_envs"),
        "run_eval_eps": cfg.get("run", {}).get("eval_eps"),
        "run_train_ratio": cfg.get("run", {}).get("train_ratio"),
        "git_revision": _git_revision(),
    }

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "wandb_project": wb_kwargs["project"],
                "wandb_group": wb_kwargs["group"],
                "wandb_job_type": wb_kwargs["job_type"],
                "wandb_name": wb_kwargs["name"],
                "rows": len(rows),
                "metadata": meta.to_dict(),
            },
            indent=2,
        )
    )
    if dry_run:
        return

    import wandb

    with wandb.init(**wb_kwargs) as run:  # type: ignore[attr-defined]
        for row in rows:
            step = row.get("step")
            if step is None:
                run.log(_sanitize(row))
            else:
                run.log(_sanitize(row), step=int(step))


def _resolve_run_dirs(paths: list[str], pattern: str | None) -> list[Path]:
    run_dirs: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            run_dirs.append(p)
        else:
            raise FileNotFoundError(p)
    if pattern:
        root = Path("third_party/dreamerv3/experiments/runs")
        run_dirs.extend(sorted(root.glob(pattern)))
    unique = []
    seen = set()
    for path in run_dirs:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "run_dirs",
        nargs="*",
        help="Dreamer run directories to upload (each containing config.yaml and metrics.jsonl).",
    )
    p.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Optional glob under third_party/dreamerv3/experiments/runs, \
            e.g. 'toy_consensus_train_dense__baseline__20260323_*'.",
    )
    p.add_argument("--scenario", type=str, default=None, help="Override inferred scenario name.")
    p.add_argument("--agent", type=str, default=None, help="Override inferred agent name.")
    p.add_argument("--variant", type=str, default=None, help="Override inferred variant name.")
    p.add_argument(
        "--dry-run", action="store_true", help="Print inferred metadata without uploading."
    )
    args = p.parse_args()

    run_dirs = _resolve_run_dirs(args.run_dirs, args.glob)
    if not run_dirs:
        raise SystemExit("No run directories provided.")

    for run_dir in run_dirs:
        _upload_run(
            run_dir,
            scenario_override=args.scenario,
            agent_override=args.agent,
            variant_override=args.variant,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
