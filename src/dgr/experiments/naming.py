from __future__ import annotations

from pathlib import Path

from dgr.experiments.metadata import ExperimentMetadata


def canonical_policy_name(name: str) -> str:
    mapping = {
        "zero": "zero",
        "random": "random",
        "proportional": "prop_oracle",
        "masked_proportional": "prop_masked",
        "inferred_proportional": "prop_inferred",
        "dreamer_flat": "dreamer_flat",
        "dreamer_gnnenc": "dreamer_gnnenc",
        "dreamer_gnnrssm": "dreamer_gnnrssm",
    }
    return mapping.get(name, name)


def controller_eval_dir(base: str | Path, meta: ExperimentMetadata) -> Path:
    return Path(base) / meta.scenario / f"policy={meta.policy_or_agent}" / f"seed={meta.seed:03d}"


def dreamer_run_name(meta: ExperimentMetadata) -> str:
    ts = meta.timestamp or "notimestamp"
    return (
        f"scenario={meta.scenario}"
        f"__agent={meta.policy_or_agent}"
        f"__variant={meta.variant}"
        f"__seed={meta.seed}"
        f"__ts={ts}"
    )
