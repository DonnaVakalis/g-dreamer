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


def _checkpoint_metadata(logdir: Path) -> dict[str, str]:
    ckpt_dir = logdir / "ckpt"
    meta = {"dreamer_checkpoint_dir": str(ckpt_dir)}
    latest = ckpt_dir / "latest"
    if not latest.exists():
        return meta
    try:
        ref = latest.read_text().strip()
    except OSError:
        return meta
    if not ref:
        return meta
    meta["dreamer_checkpoint_ref"] = ref
    meta["dreamer_checkpoint_path"] = str(ckpt_dir / ref)
    return meta


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


def _infer_agent_name(cfg: dict[str, Any], run_dir: Path | None = None) -> str:
    if bool(cfg.get("random_agent", False)):
        return "random"
    enc_typ = str(cfg.get("agent", {}).get("enc", {}).get("typ", "simple"))
    if enc_typ == "graph":
        return canonical_policy_name("dreamer_gnnenc")
    if run_dir and "graph_encoder" in run_dir.name:
        return canonical_policy_name("dreamer_gnnenc")
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


def _use_graph_obs(logdir: Path, dreamer_config) -> bool:
    enc_typ = str(getattr(getattr(dreamer_config.agent, "enc", {}), "typ", "simple"))
    return enc_typ == "graph" or "graph_flat" in logdir.name or "graph_encoder" in logdir.name


def _make_obs_act_spaces(scenario_name: str, dreamer_config, *, include_graph_obs: bool):
    import jax
    from dreamerv3.main import wrap_env
    from embodied.envs.from_gym import FromGym

    from dgr.envs.adapters.toy_graph_control_gym import (
        ToyConsensusGymEnv,
        register_toy_consensus_envs,
    )

    with jax.transfer_guard("allow"):
        register_toy_consensus_envs()
        gym_env = ToyConsensusGymEnv(
            scenario_name=scenario_name,
            include_graph_obs=include_graph_obs,
        )

    env = wrap_env(FromGym(gym_env), dreamer_config)

    def _not_log_key(k: str) -> bool:
        return not k.startswith("log/")

    obs_space = {k: v for k, v in env.obs_space.items() if _not_log_key(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
    env.close()
    return obs_space, act_space


def _make_agent(logdir: Path, obs_space, act_space, dreamer_config):
    import elements

    from dgr.agents.graph_dreamerv3.agent import Agent

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
        obs = {key: raw[key][np.newaxis] for key in raw if not str(key).startswith("log/")}
        obs.update(
            {
                "reward": np.array([reward], dtype=np.float32),
                "is_first": np.array([is_first]),
                "is_last": np.array([is_last]),
                "is_terminal": np.array([is_terminal]),
            }
        )
        return obs

    carry = agent.init_policy(batch_size=1)
    carry, acts, _ = agent.policy(carry, _fmt(raw_obs, 0.0, True, False, False), mode="eval")

    total_reward = 0.0
    steps = 0
    last_reward = 0.0
    history: list[dict[str, Any]] = []

    while True:
        action = np.array(acts["action"][0])
        raw_obs, reward, done, info = gym_env.step(action)
        steps += 1
        total_reward += reward
        last_reward = reward
        history.append(
            {
                "t": steps,
                "reward": float(reward),
                "mse": float(-reward),
                "done": bool(done),
            }
        )
        is_terminal = bool(info.get("is_terminal", done))
        carry, acts, _ = agent.policy(
            carry, _fmt(raw_obs, float(reward), False, done, is_terminal), mode="eval"
        )
        if done:
            break

    return {
        "total_reward": total_reward,
        "end_mse": float(-last_reward),
        "steps": steps,
        "history": history,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "end_mse_mean": float(np.mean([r["end_mse"] for r in rows])),
        "end_mse_std": float(np.std([r["end_mse"] for r in rows])),
        "total_reward_mean": float(np.mean([r["total_reward"] for r in rows])),
        "total_reward_std": float(np.std([r["total_reward"] for r in rows])),
        "steps_mean": float(np.mean([r["steps"] for r in rows])),
    }


def _within_episode_profile(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    max_steps = max((len(row.get("history", [])) for row in rows), default=0)
    profile: list[dict[str, float | int]] = []
    for idx in range(max_steps):
        mses = [row["history"][idx]["mse"] for row in rows if idx < len(row.get("history", []))]
        rewards = [
            row["history"][idx]["reward"] for row in rows if idx < len(row.get("history", []))
        ]
        if not mses:
            continue
        profile.append(
            {
                "t": idx + 1,
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "count": len(mses),
            }
        )
    return profile


def _log_to_wandb(
    rows: list[dict[str, Any]],
    agg: dict[str, Any],
    meta: ExperimentMetadata,
    logdir: Path,
    episodes: int,
    train_seed: int = 0,
) -> None:
    import wandb

    from dgr.envs.suites.toy_graph_control.scenarios import get_scenario, scenario_stats

    stats: dict[str, Any] = {}
    try:
        stats = scenario_stats(get_scenario(meta.scenario))
    except Exception:
        pass

    git_rev = _git_revision()
    checkpoint_meta = _checkpoint_metadata(logdir)
    within_episode = _within_episode_profile(rows)
    wb_kwargs = wandb_init_kwargs(meta)
    wb_kwargs["config"] = {
        **wb_kwargs["config"],
        **stats,
        "train_seed": train_seed,
        "dreamer_logdir": str(logdir),
        **checkpoint_meta,
        **({"git_revision": git_rev} if git_rev else {}),
    }

    with wandb.init(**wb_kwargs) as run:  # type: ignore[attr-defined]
        run.define_metric("episode/index")
        run.define_metric("episode/*", step_metric="episode/index")
        run.define_metric("within_episode/t")
        run.define_metric("within_episode/*", step_metric="within_episode/t")
        wandb_table = getattr(wandb, "Table")

        for i, row in enumerate(rows):
            run.log(
                {
                    "episode/index": i,
                    "episode/total_reward": row["total_reward"],
                    "episode/end_mse": row["end_mse"],
                    "episode/steps": row["steps"],
                }
            )

        raw_step_rows = []
        for i, row in enumerate(rows):
            for step_row in row.get("history", []):
                raw_step_rows.append(
                    [
                        i,
                        int(step_row["t"]),
                        float(step_row["mse"]),
                        float(step_row["reward"]),
                        bool(step_row["done"]),
                    ]
                )
        if raw_step_rows:
            run.log(
                {
                    "within_episode/raw_table": wandb_table(
                        columns=["episode", "t", "mse", "reward", "done"],
                        data=raw_step_rows,
                    )
                }
            )

        for row in within_episode:
            run.log(
                {
                    "within_episode/t": int(row["t"]),
                    "within_episode/mse_mean": row["mse_mean"],
                    "within_episode/mse_std": row["mse_std"],
                    "within_episode/reward_mean": row["reward_mean"],
                    "within_episode/reward_std": row["reward_std"],
                    "within_episode/count": row["count"],
                }
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
    agent_name = canonical_policy_name(_infer_agent_name(dreamer_config, logdir))
    train_seed = int(getattr(dreamer_config, "seed", 0))
    eval_seed = args.seed
    include_graph_obs = _use_graph_obs(logdir, dreamer_config)

    obs_space, act_space = _make_obs_act_spaces(
        scenario,
        dreamer_config,
        include_graph_obs=include_graph_obs,
    )
    agent = _make_agent(logdir, obs_space, act_space, dreamer_config)

    from dgr.envs.adapters.toy_graph_control_gym import ToyConsensusGymEnv

    gym_env = ToyConsensusGymEnv(
        scenario_name=scenario,
        include_graph_obs=include_graph_obs,
    )
    rows = []
    for i in range(args.episodes):
        row = _run_episode(agent, gym_env, episode_seed=eval_seed + i)
        row["episode"] = i
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
            seed=eval_seed,
            variant=variant,
            episodes=args.episodes,
        ).with_timestamp()
        _log_to_wandb(rows, agg, meta, logdir, args.episodes, train_seed=train_seed)


if __name__ == "__main__":
    main()
