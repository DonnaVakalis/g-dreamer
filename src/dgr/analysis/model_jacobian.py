"""Rollout-Jacobian spectral radius of a trained world model — a model-side theoretical
handle on rollout stability.

Under our open-loop rollout (goal channel re-injected each step), the iterated map for the
state channel is::

    x_{t+1} = predict_next_nodes_single(params, nodes=[x_t, goal], action_t, ...)[:, 0]

So the **per-step Jacobian governing rollout stability** is ``∂(predict[:,0]) / ∂(nodes[:,0])``
— an ``(n_real, n_real)`` matrix. Its spectral radius bounds local rollout amplification:
``ρ ≤ 1`` is locally contractive; ``ρ > 1`` is locally expanding. We hypothesise
``ρ(J_C) > 1`` (architectural amplification via message-passing) and ``ρ(J_B) ≤ 1`` (no
cross-node coupling in dynamics).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

PredictFn = Callable[..., jnp.ndarray]


@dataclass(frozen=True)
class JacobianStats:
    spectral_radius_mean: float
    spectral_radius_median: float
    spectral_radius_p95: float  # 95th-percentile (heavy-tail-aware)
    spectral_radius_max: float
    frac_above_one: float  # fraction of samples whose ρ > 1
    off_diagonal_mass_mean: float  # mean ‖J − diag(J)‖_1 / ‖J‖_1
    n_samples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_rollout_jacobian_stats(
    predict_fn: PredictFn,
    params,
    states: jnp.ndarray,  # (B, n_max, f_n)
    actions: jnp.ndarray,  # (B, n_max)
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    node_mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
) -> JacobianStats:
    """Compute the spectral-radius distribution of the per-step rollout Jacobian.

    Evaluates ``∂(predict[:,0]) / ∂(nodes[:,0])`` at each (state, action) pair, restricts to
    real nodes, and aggregates the spectral radii across the batch.
    """

    def x_next_partial(nodes_single, action_single):
        return predict_fn(
            params, nodes_single, action_single, senders, receivers, node_mask, edge_mask
        )[:, 0]

    # Jacobian wrt nodes (arg 0 of the closure): (n_max,) → shape (n_max, n_max, f_n).
    jac_fn = jax.jacrev(x_next_partial, argnums=0)
    jac_batch = jax.vmap(jac_fn, in_axes=(0, 0))
    full_jac = jac_batch(states, actions)  # (B, n_max, n_max, f_n)
    full_jac = np.asarray(full_jac)
    # Partial wrt the x-channel only (input channel 0); shape (B, n_max, n_max).
    J = full_jac[..., 0]

    real_idx = np.where(np.asarray(node_mask))[0]
    J_real = J[:, real_idx][:, :, real_idx]  # (B, n_real, n_real)

    radii = np.empty(J_real.shape[0], dtype=np.float64)
    offdiag = np.empty(J_real.shape[0], dtype=np.float64)
    for i, j_mat in enumerate(J_real):
        eigs = np.linalg.eigvals(j_mat)
        radii[i] = float(np.max(np.abs(eigs)))
        total = float(np.sum(np.abs(j_mat)))
        diag = float(np.sum(np.abs(np.diag(j_mat))))
        offdiag[i] = (total - diag) / total if total > 0 else 0.0

    return JacobianStats(
        spectral_radius_mean=float(radii.mean()),
        spectral_radius_median=float(np.median(radii)),
        spectral_radius_p95=float(np.percentile(radii, 95)),
        spectral_radius_max=float(radii.max()),
        frac_above_one=float((radii > 1.0).mean()),
        off_diagonal_mass_mean=float(offdiag.mean()),
        n_samples=int(J_real.shape[0]),
    )
