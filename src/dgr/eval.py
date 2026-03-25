"""
Evaluates a trained DreamerV3 checkpoint on a toy scenario.
Logs per-episode metrics to wandb in the same format as compare_toy_controllers.py.

Usage:
    poetry run python -m dgr.eval \
        --logdir experiments/runs/toy_consensus_train_dense__baseline__20260325_... \
        --episodes 20 --seed 0 [--wandb] [--scenario override]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
    "DGRToyConsensusTrainSparseHiddenSmoothAligned-v0": "train_ring_sparse_hidden_smooth_aligned",
    "DGRToyConsensusTrainSparseHiddenSmoothMisaligned-v0": (
        "train_ring_sparse_hidden_smooth_misaligned"
    ),
}


def _known_scenarios() -> list[str]:
    path = (
        Path(__file__).resolve().parent / "envs" / "suites" / "toy_graph_control" / "scenarios.py"
    )
    names = re.findall(r'if name == "([^"]+)"', path.read_text())
    return sorted(set(names), key=len, reverse=True)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
                    f"Task {task!r} resolves to ambiguous scenario {scenario!r}. Pass --scenario."
                )
            return scenario
    for candidate in sorted(_ENV_ID_TO_SCENARIO.values(), key=len, reverse=True):
        if candidate in logdir:
            return candidate
    raise ValueError(
        f"Could not infer scenario from task={task!r}, logdir={logdir!r}. Pass --scenario."
    )


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
    if bool(cfg.get("random_agent", False)):
        return "random"
    return canonical_policy_name("dreamer_flat")


def _ensure_dreamer_on_path() -> None:
    upstream_root = Path(__file__).resolve().parents[2] / "third_party" / "dreamerv3"
    if not upstream_root.exists():
        raise FileNotFoundError(f"Expected upstream at {upstream_root}")
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))


def _load_dreamer_config(logdir: Path):
    _ensure_dreamer_on_path()
    import elements

    path = logdir / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {logdir}")
    return elements.Config.load(str(path))


def _make_obs_act_spaces(scenario_name: str, dreamer_config):
    import jax
    from dreamerv3.main import wrap_env
    from embodied.envs.from_gym import FromGym

    from dgr.envs.adapters.toy_graph_control_gym import (
        ToyConsensusGymEnv,
        register_toy_consensus_envs,
    )

    with jax.transfer_guard("allow"):
        register_toy_consensus_envs()
        gym_env = ToyConsensusGymEnv(scenario_name=scenario_name)

    env = wrap_env(FromGym(gym_env), dreamer_config)

    def _not_log_key(k: str) -> bool:
        return not k.startswith("log/")

    obs_space = {k: v for k, v in env.obs_space.items() if _not_log_key(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
    env.close()
    return obs_space, act_space


def _make_agent(logdir: Path, obs_space, act_space, dreamer_config):
    import elements
    from dreamerv3.agent import Agent

    agent_config = elements.Config(
        **dreamer_config.agent,
        logdir=str(logdir),
        seed=dreamer_config.seed,
        jax=dreamer_config.jax,
        batch_size=dreamer_config.batch_size,
        batch_length=dreamer_config.batch_length,
        replay_context=dreamer_config.replay_context,
        report_length=dreamer_config.report_length,
        replica=getattr(dreamer_config, "replica", 0),
        replicas=getattr(dreamer_config, "replicas", 1),
    )
    agent = Agent(obs_space, act_space, agent_config)
    cp = elements.Checkpoint(logdir / "ckpt")
    cp.agent = agent
    cp.load(keys=["agent"])
    return agent


def _run_episode(agent, gym_env, episode_seed: int) -> dict[str, Any]:
    gym_env.seed(episode_seed)
    raw_obs = gym_env.reset()

    def _fmt(raw: dict, reward: float, is_first: bool, is_last: bool, is_terminal: bool) -> dict:
        return {
            "vector": raw["vector"][np.newaxis],
            "reward": np.array([reward], dtype=np.float32),
            "is_first": np.array([is_first]),
            "is_last": np.array([is_last]),
            "is_terminal": np.array([is_terminal]),
        }

    carry = agent.init_policy(batch_size=1)
    carry, acts, _ = agent.policy(carry, _fmt(raw_obs, 0.0, True, False, False), mode="eval")

    total_reward = 0.0
    steps = 0
    last_reward = 0.0

    while True:
        action = np.array(acts["action"][0])
        raw_obs, reward, done, info = gym_env.step(action)
        steps += 1
        total_reward += reward
        last_reward = reward
        is_terminal = bool(info.get("is_terminal", done))
        carry, acts, _ = agent.policy(
            carry, _fmt(raw_obs, float(reward), False, done, is_terminal), mode="eval"
        )
        if done:
            break

    return {"total_reward": total_reward, "end_mse": float(-last_reward), "steps": steps}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "end_mse_mean": float(np.mean([r["end_mse"] for r in rows])),
        "end_mse_std": float(np.std([r["end_mse"] for r in rows])),
        "total_reward_mean": float(np.mean([r["total_reward"] for r in rows])),
        "total_reward_std": float(np.std([r["total_reward"] for r in rows])),
        "steps_mean": float(np.mean([r["steps"] for r in rows])),
    }


def _log_to_wandb(
    rows: list[dict[str, Any]],
    agg: dict[str, Any],
    meta: ExperimentMetadata,
    logdir: Path,
    episodes: int,
) -> None:
    import wandb

    from dgr.envs.suites.toy_graph_control.scenarios import get_scenario, scenario_stats

    stats: dict[str, Any] = {}
    try:
        stats = scenario_stats(get_scenario(meta.scenario))
    except Exception:
        pass

    git_rev = _git_revision()
    wb_kwargs = wandb_init_kwargs(meta)
    wb_kwargs["config"] = {
        **wb_kwargs["config"],
        **stats,
        "dreamer_logdir": str(logdir),
        **({"git_revision": git_rev} if git_rev else {}),
    }

    with wandb.init(**wb_kwargs) as run:  # type: ignore[attr-defined]
        for i, row in enumerate(rows):
            run.log(
                {
                    "episode/total_reward": row["total_reward"],
                    "episode/end_mse": row["end_mse"],
                    "episode/steps": row["steps"],
                },
                step=i,
            )
        run.log(
            {
                "aggregate/end_mse_mean": agg["end_mse_mean"],
                "aggregate/end_mse_std": agg["end_mse_std"],
                "aggregate/total_reward_mean": agg["total_reward_mean"],
                "aggregate/total_reward_std": agg["total_reward_std"],
            },
            step=episodes,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--scenario", default=None)
    args = p.parse_args()

    logdir = Path(args.logdir).resolve()
    dreamer_config = _load_dreamer_config(logdir)

    scenario = args.scenario or _infer_scenario(logdir, dreamer_config)
    variant = _infer_size_label(dreamer_config)
    agent_name = canonical_policy_name(_infer_agent_name(dreamer_config))
    seed = int(getattr(dreamer_config, "seed", args.seed))

    obs_space, act_space = _make_obs_act_spaces(scenario, dreamer_config)
    agent = _make_agent(logdir, obs_space, act_space, dreamer_config)

    from dgr.envs.adapters.toy_graph_control_gym import ToyConsensusGymEnv

    gym_env = ToyConsensusGymEnv(scenario_name=scenario)
    rows = []
    for i in range(args.episodes):
        row = _run_episode(agent, gym_env, episode_seed=seed + i)
        rows.append(row)
        print(f"ep {i:3d}: end_mse={row['end_mse']:.4f}  total_reward={row['total_reward']:.3f}")
    gym_env.close()

    agg = _aggregate(rows)
    print(json.dumps(agg, indent=2))

    if args.wandb:
        meta = ExperimentMetadata(
            run_type="dreamer_eval",
            scenario=scenario,
            policy_or_agent=agent_name,
            seed=seed,
            variant=variant,
            episodes=args.episodes,
        ).with_timestamp()
        _log_to_wandb(rows, agg, meta, logdir, args.episodes)


if __name__ == "__main__":
    main()
