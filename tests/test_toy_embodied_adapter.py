"""
Test the toy graph control embodied adapter.
(skippable if you don't have elements installed)
to run the test:
poetry install --with dev,upstream
poetry run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

elements = pytest.importorskip("elements")  # skip unless you installed --with upstream

from dgr.envs.adapters.toy_graph_control_embodied import ToyGraphControlEmbodied  # noqa: E402
from dgr.envs.suites.toy_graph_control.consensus import ConsensusConfig  # noqa: E402
from dgr.interface.graph_spec import GraphSpec  # noqa: E402


def test_embodied_adapter_step_contract():
    spec = GraphSpec(n_max=8, e_max=16, f_n=2, f_e=0, f_g=0)
    cfg = ConsensusConfig(spec=spec, n_real=5, horizon=10)

    env = ToyGraphControlEmbodied(cfg)

    a = {"action": np.zeros((spec.n_max,), np.float32), "reset": True}
    obs = env.step(a)
    assert obs["is_first"] is True
    assert obs["vector"].shape[0] > 0

    a["reset"] = False
    obs2 = env.step(a)
    assert obs2["is_first"] is False
    assert isinstance(obs2["reward"].item(), float)
