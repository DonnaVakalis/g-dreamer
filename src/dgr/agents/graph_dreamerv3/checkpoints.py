from __future__ import annotations

import pickle
from pathlib import Path

import jax
import numpy as np


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = jax.tree_util.tree_map(
        lambda x: np.asarray(x) if hasattr(x, "shape") else x,
        payload,
    )
    with path.open("wb") as f:
        pickle.dump(safe_payload, f)


def load_checkpoint(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)
