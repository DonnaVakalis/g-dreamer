from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

nj = pytest.importorskip("ninjax")
elements = pytest.importorskip("elements")
pytest.importorskip("gym")

upstream_root = Path(__file__).resolve().parents[1] / "third_party" / "dreamerv3"
if not upstream_root.exists():
    pytest.skip("Vendored DreamerV3 checkout is not available", allow_module_level=True)

if str(upstream_root) not in sys.path:
    sys.path.insert(0, str(upstream_root))

rssm = pytest.importorskip("dreamerv3.rssm")  # noqa: E402

from dgr.agents.graph_dreamerv3.encoders.gnn import GraphEncoder  # noqa: E402
from dgr.envs.adapters.toy_graph_control_gym import ToyConsensusGymEnv  # noqa: E402


def test_toy_gym_env_can_emit_graph_observation_fields():
    env = ToyConsensusGymEnv(scenario_name="debug_ring_dense", include_graph_obs=True)
    obs = env.reset()

    assert "vector" in obs
    assert "nodes" in obs
    assert "senders" in obs
    assert "receivers" in obs
    assert "node_mask" in obs
    assert "edge_mask" in obs
    assert obs["nodes"].ndim == 2
    assert obs["senders"].dtype == np.int32
    env.close()


def test_graph_encoder_matches_baseline_token_interface():
    env = ToyConsensusGymEnv(scenario_name="debug_ring_dense", include_graph_obs=True)
    obs = env.reset()

    vector_obs_space = {"vector": elements.Space(np.float32, obs["vector"].shape, -np.inf, np.inf)}
    graph_obs_space = {
        "nodes": elements.Space(np.float32, obs["nodes"].shape, -np.inf, np.inf),
        "senders": elements.Space(np.int32, obs["senders"].shape, 0, obs["nodes"].shape[0]),
        "receivers": elements.Space(np.int32, obs["receivers"].shape, 0, obs["nodes"].shape[0]),
        "node_mask": elements.Space(np.float32, obs["node_mask"].shape, 0.0, 1.0),
        "edge_mask": elements.Space(np.float32, obs["edge_mask"].shape, 0.0, 1.0),
    }
    act_space = {"action": elements.Space(np.float32, (obs["nodes"].shape[0],), -1.0, 1.0)}

    enc_cfg = dict(
        depth=2,
        mults=(2, 3, 4, 4),
        layers=1,
        units=8,
        act="silu",
        norm="rms",
        winit="trunc_normal_in",
        symlog=True,
        outer=False,
        kernel=5,
        strided=False,
    )
    dyn_cfg = dict(
        deter=8,
        hidden=3,
        stoch=2,
        classes=4,
        act="silu",
        norm="rms",
        unimix=0.01,
        outscale=1.0,
        winit="trunc_normal_in",
        imglayers=1,
        obslayers=1,
        dynlayers=1,
        absolute=False,
        blocks=4,
        free_nats=1.0,
    )

    reset = jnp.array([True])
    prev_action = {"action": jnp.zeros((1, obs["nodes"].shape[0]), dtype=jnp.float32)}
    vector_batch = {"vector": jnp.asarray(obs["vector"])[None]}
    graph_batch = {
        "nodes": jnp.asarray(obs["nodes"])[None],
        "senders": jnp.asarray(obs["senders"])[None],
        "receivers": jnp.asarray(obs["receivers"])[None],
        "node_mask": jnp.asarray(obs["node_mask"])[None],
        "edge_mask": jnp.asarray(obs["edge_mask"])[None],
    }

    def rollout(enc_type):
        enc = (
            rssm.Encoder(vector_obs_space, **enc_cfg, name="enc")
            if enc_type == "simple"
            else GraphEncoder(graph_obs_space, **enc_cfg, name="enc")
        )
        dyn = rssm.RSSM(act_space, **dyn_cfg, name="dyn")

        obs_batch = vector_batch if enc_type == "simple" else graph_batch

        def fn():
            enc_carry = enc.initial(1)
            dyn_carry = dyn.initial(1)
            _, _, tokens = enc(enc_carry, obs_batch, reset, training=False, single=True)
            _, _, feat = dyn.observe(
                dyn_carry,
                tokens,
                prev_action,
                reset,
                training=False,
                single=True,
            )
            return tokens, feat

        params = nj.init(fn)({}, seed=0)
        _, (tokens, feat) = nj.pure(fn)(params, seed=0)
        return tokens, feat

    baseline_tokens, baseline_feat = rollout("simple")
    graph_tokens, graph_feat = rollout("graph")

    assert baseline_tokens.shape == graph_tokens.shape
    assert baseline_feat["deter"].shape == graph_feat["deter"].shape
    assert baseline_feat["stoch"].shape == graph_feat["stoch"].shape
    env.close()
