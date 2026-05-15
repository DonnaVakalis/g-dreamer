"""
Launch DreamerV3 on the toy graph environment using the encoder-swap agent.

This runner exists so we can stage the Dreamer integration carefully:
  - keep upstream Dreamer's training loop, replay, RSSM, and losses
  - swap only the encoder implementation
  - expose structured graph observations without changing the baseline runner

We monkey-patch only the pieces that need to know about graph observations:
  - `make_env()` to request the structured graph keys from the toy env
  - `make_agent()` to instantiate the thin encoder-swap Dreamer wrapper
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


def _scenario_from_task(task: str) -> str | None:
    from dgr.envs.adapters.toy_graph_control_gym import SCENARIO_TO_ENV_ID

    if not task.startswith("gym_"):
        return None
    env_id = task[len("gym_") :]
    inv = {v: k for k, v in SCENARIO_TO_ENV_ID.items()}
    return inv.get(env_id)


def _maybe_init_wandb(argv: list[str]) -> None:
    scenario = os.environ.get("DGR_WANDB_SCENARIO")
    if not scenario:
        return
    if "--logger.outputs" in argv:
        idx = argv.index("--logger.outputs")
        if idx + 1 < len(argv) and "wandb" not in argv[idx + 1].split(","):
            return
    variant = os.environ.get("DGR_WANDB_VARIANT", "custom")
    seed = _parse_seed(argv)
    agent_name = os.environ.get("DGR_WANDB_AGENT", "dreamer_gnnenc")

    from dgr.experiments.metadata import ExperimentMetadata
    from dgr.experiments.naming import canonical_policy_name
    from dgr.experiments.wandb_utils import wandb_init_kwargs

    meta = ExperimentMetadata(
        run_type="dreamer_train",
        scenario=scenario,
        policy_or_agent=canonical_policy_name(agent_name),
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

    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    from dreamerv3 import main as upstream_main

    from dgr.agents.graph_dreamerv3.agent import Agent
    from dgr.envs.adapters.toy_graph_control_gym import ToyConsensusGymEnv

    original_make_env = upstream_main.make_env

    def make_env(config, index, **overrides):
        scenario = _scenario_from_task(config.task)
        if scenario is None:
            return original_make_env(config, index, **overrides)

        import jax
        from embodied.envs.from_gym import FromGym

        with jax.transfer_guard("allow"):
            env = ToyConsensusGymEnv(
                scenario_name=scenario,
                include_graph_obs=True,
            )
        return upstream_main.wrap_env(FromGym(env), config)

    def make_agent(config):
        import elements
        import embodied

        env = make_env(config, 0)

        def notlog(key):
            return not key.startswith("log/")

        obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
        act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
        env.close()
        if config.random_agent:
            return embodied.RandomAgent(obs_space, act_space)
        return Agent(
            obs_space,
            act_space,
            elements.Config(
                **config.agent,
                logdir=config.logdir,
                seed=config.seed,
                jax=config.jax,
                batch_size=config.batch_size,
                batch_length=config.batch_length,
                replay_context=config.replay_context,
                report_length=config.report_length,
                replica=config.replica,
                replicas=config.replicas,
            ),
        )

    for i, tok in enumerate(argv):
        if tok == "--logdir" and i + 1 < len(argv):
            argv[i + 1] = str(Path(argv[i + 1]).resolve())
            break

    _maybe_init_wandb(argv)
    os.chdir(upstream_root)
    upstream_main.make_env = make_env
    upstream_main.make_agent = make_agent
    upstream_main.main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
