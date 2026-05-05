"""
Graph encoder used for the first Dreamer integration stage.

Why this exists:
  The research question in this stage is intentionally narrow:
  "Does a graph-aware observation encoder help before we change Dreamer's
  latent dynamics model?"

To keep that ablation clean, this module only replaces the observation encoder.
It preserves the downstream interface expected by upstream Dreamer:
  graph observation -> fixed-width token vector

That lets us hold the RSSM, decoder, losses, actor, and critic constant while
swapping the representation from:
  flattened graph -> MLP token
to:
  message passing over structured graph -> pooled token
"""

from __future__ import annotations

import math
from pathlib import Path

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dgr.interface.graph_spec import Graph
from dgr.models.message_passing import masked_message_passing


def _ensure_upstream_on_path() -> None:
    import sys

    upstream_root = Path(__file__).resolve().parents[5] / "third_party" / "dreamerv3"
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))


_ensure_upstream_on_path()
import elements  # noqa: E402


class GraphEncoder(nj.Module):
    units = 1024
    norm = "rms"
    act = "gelu"
    layers = 3
    symlog = True

    def __init__(self, obs_space, **kw):
        required = {"nodes", "senders", "receivers", "node_mask", "edge_mask"}
        missing = required - set(obs_space)
        if missing:
            raise ValueError(f"GraphEncoder requires graph obs keys, missing {sorted(missing)}")
        self.obs_space = obs_space
        self.units = int(kw.get("units", self.units))
        self.norm = str(kw.get("norm", self.norm))
        self.act = str(kw.get("act", self.act))
        self.layers = int(kw.get("layers", self.layers))
        self.symlog = bool(kw.get("symlog", self.symlog))
        # `enc.simple` contains many fields that make sense for the baseline MLP
        # encoder but not for individual linear layers. We intentionally reuse
        # only the layer-safe subset here so the graph encoder tracks comparable
        # initialization behavior without accidentally forwarding irrelevant
        # config keys like `depth`, `kernel`, or `mults` into `nn.Linear`.
        self.linear_kw = {
            key: value for key, value in kw.items() if key in {"bias", "winit", "binit", "outscale"}
        }
        self.n_max, self.node_dim = obs_space["nodes"].shape
        self.e_max = obs_space["senders"].shape[0]

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        return {}

    def truncate(self, entries, carry=None):
        return {}

    def __call__(self, carry, obs, reset, training, single=False):
        del training
        bshape = reset.shape
        batch = math.prod(bshape)

        nodes = obs["nodes"].reshape((batch, self.n_max, self.node_dim))
        senders = obs["senders"].reshape((batch, self.e_max)).astype(jnp.int32)
        receivers = obs["receivers"].reshape((batch, self.e_max)).astype(jnp.int32)
        node_mask = obs["node_mask"].reshape((batch, self.n_max)) > 0.5
        edge_mask = obs["edge_mask"].reshape((batch, self.e_max)) > 0.5

        def encode_one(*xs):
            return self._encode_graph(*xs)

        tokens = jax.vmap(encode_one)(nodes, senders, receivers, node_mask, edge_mask)
        return carry, {}, tokens.reshape((*bshape, tokens.shape[-1]))

    def _encode_graph(self, nodes, senders, receivers, node_mask, edge_mask):
        # We intentionally pool back down to a single token vector because this
        # stage is about swapping only the encoder while preserving the RSSM API.
        x = nn.symlog(nodes) if self.symlog else nodes
        x = self._mask_nodes(nn.cast(x), node_mask)
        for i in range(self.layers):
            x = self.sub(f"node{i}", nn.Linear, self.units, **self.linear_kw)(x)
            x = nn.act(self.act)(self.sub(f"node{i}norm", nn.Norm, self.norm)(x))
            x = self._mask_nodes(x, node_mask)
            x = self._message_passing(x, senders, receivers, node_mask, edge_mask)
            x = self._mask_nodes(x, node_mask)
        pooled = self._masked_mean(x, node_mask)
        return pooled

    @staticmethod
    def _mask_nodes(x, node_mask):
        return jnp.where(node_mask[:, None], x, jnp.zeros_like(x))

    @staticmethod
    def _masked_mean(x, node_mask):
        weights = node_mask.astype(x.dtype)[:, None]
        denom = jnp.maximum(weights.sum(), 1.0)
        return (x * weights).sum(axis=0) / denom

    def _message_passing(self, nodes, senders, receivers, node_mask, edge_mask):
        graph = Graph(
            nodes=nodes,
            edges=jnp.zeros((self.e_max, 0), dtype=nodes.dtype),
            senders=senders.astype(jnp.int32),
            receivers=receivers.astype(jnp.int32),
            node_mask=node_mask.astype(jnp.bool_),
            edge_mask=edge_mask.astype(jnp.bool_),
            globals=jnp.zeros((0,), dtype=nodes.dtype),
        )
        return masked_message_passing(graph)


def graph_obs_space(spec) -> dict[str, elements.Space]:
    return {
        "nodes": elements.Space(np.float32, (spec.n_max, spec.f_n), -np.inf, np.inf),
        "senders": elements.Space(np.int32, (spec.e_max,), 0, spec.n_max),
        "receivers": elements.Space(np.int32, (spec.e_max,), 0, spec.n_max),
        "node_mask": elements.Space(np.float32, (spec.n_max,), 0.0, 1.0),
        "edge_mask": elements.Space(np.float32, (spec.e_max,), 0.0, 1.0),
    }
