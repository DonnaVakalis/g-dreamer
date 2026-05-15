from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dgr.envs.suites.toy_graph_control.core import reset, step
from dgr.envs.suites.toy_graph_control.scenarios import make_consensus_config


@dataclass(frozen=True)
class TransitionDataset:
    nodes: np.ndarray
    actions: np.ndarray
    next_nodes: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    node_mask: np.ndarray
    edge_mask: np.ndarray
    n_real: np.ndarray
    episode_id: np.ndarray
    step_id: np.ndarray

    @property
    def size(self) -> int:
        return int(self.nodes.shape[0])


def parse_sizes(text: str) -> list[int]:
    sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not sizes:
        raise ValueError("Expected at least one graph size")
    if any(size <= 0 for size in sizes):
        raise ValueError(f"Graph sizes must be positive, got {sizes}")
    return sizes


def collect_random_transitions(
    *,
    sizes: list[int],
    episodes_per_size: int,
    n_max: int | None = None,
    horizon: int = 50,
    alpha: float = 0.2,
    beta: float = 0.5,
    noise_std: float = 0.01,
    action_scale: float = 1.0,
    seed: int = 0,
) -> TransitionDataset:
    n_max = max(sizes) if n_max is None else n_max
    if n_max < max(sizes):
        raise ValueError(f"n_max must cover all sizes, got n_max={n_max}, sizes={sizes}")
    if episodes_per_size <= 0:
        raise ValueError(f"episodes_per_size must be positive, got {episodes_per_size}")

    nodes = []
    actions = []
    next_nodes = []
    senders = []
    receivers = []
    node_masks = []
    edge_masks = []
    n_reals = []
    episode_ids = []
    step_ids = []

    key = jax.random.PRNGKey(seed)
    episode_counter = 0
    for size in sizes:
        cfg = make_consensus_config(
            size,
            n_max=n_max,
            horizon=horizon,
            alpha=alpha,
            beta=beta,
            noise_std=noise_std,
        )
        for _ in range(episodes_per_size):
            key, reset_key = jax.random.split(key)
            state, obs = reset(reset_key, cfg)
            for t in range(horizon):
                key, action_key, step_key = jax.random.split(key, 3)
                action = action_scale * jax.random.uniform(
                    action_key,
                    shape=(cfg.spec.n_max,),
                    minval=-1.0,
                    maxval=1.0,
                    dtype=jnp.float32,
                )
                action = jnp.where(state.node_mask, action, 0.0)
                next_state, next_obs, _, done = step(step_key, cfg, state, action)

                nodes.append(np.asarray(obs.nodes, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
                next_nodes.append(np.asarray(next_obs.nodes, dtype=np.float32))
                senders.append(np.asarray(state.senders, dtype=np.int32))
                receivers.append(np.asarray(state.receivers, dtype=np.int32))
                node_masks.append(np.asarray(state.node_mask, dtype=np.bool_))
                edge_masks.append(np.asarray(state.edge_mask, dtype=np.bool_))
                n_reals.append(size)
                episode_ids.append(episode_counter)
                step_ids.append(t)

                state, obs = next_state, next_obs
                if bool(done):
                    break
            episode_counter += 1

    return TransitionDataset(
        nodes=np.stack(nodes, axis=0),
        actions=np.stack(actions, axis=0),
        next_nodes=np.stack(next_nodes, axis=0),
        senders=np.stack(senders, axis=0),
        receivers=np.stack(receivers, axis=0),
        node_mask=np.stack(node_masks, axis=0),
        edge_mask=np.stack(edge_masks, axis=0),
        n_real=np.asarray(n_reals, dtype=np.int32),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        step_id=np.asarray(step_ids, dtype=np.int32),
    )


def save_transition_dataset(
    dataset: TransitionDataset,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        nodes=dataset.nodes,
        actions=dataset.actions,
        next_nodes=dataset.next_nodes,
        senders=dataset.senders,
        receivers=dataset.receivers,
        node_mask=dataset.node_mask,
        edge_mask=dataset.edge_mask,
        n_real=dataset.n_real,
        episode_id=dataset.episode_id,
        step_id=dataset.step_id,
    )
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_transition_dataset(path: str | Path) -> TransitionDataset:
    data = np.load(Path(path))
    return TransitionDataset(
        nodes=data["nodes"],
        actions=data["actions"],
        next_nodes=data["next_nodes"],
        senders=data["senders"],
        receivers=data["receivers"],
        node_mask=data["node_mask"].astype(np.bool_),
        edge_mask=data["edge_mask"].astype(np.bool_),
        n_real=data["n_real"],
        episode_id=data["episode_id"],
        step_id=data["step_id"],
    )
