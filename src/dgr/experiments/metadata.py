from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ExperimentMetadata:
    run_type: str  # controller_eval | dreamer_train | dreamer_eval
    scenario: str
    policy_or_agent: str
    seed: int
    variant: str = "default"
    episodes: int | None = None
    run_steps: int | None = None
    timestamp: str | None = None

    def with_timestamp(self) -> "ExperimentMetadata":
        return replace(self, timestamp=self.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
