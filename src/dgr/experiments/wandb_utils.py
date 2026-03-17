from __future__ import annotations

from dgr.experiments.metadata import ExperimentMetadata


def wandb_init_kwargs(meta: ExperimentMetadata) -> dict:
    return {
        "project": "g-dreamer",
        "group": meta.scenario,
        "job_type": meta.run_type,
        "name": f"{meta.policy_or_agent}__{meta.variant}__seed{meta.seed}",
        "config": meta.to_dict(),
    }
