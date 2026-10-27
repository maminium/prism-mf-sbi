from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from sbi.utils import BoxUniform
except Exception:  # pragma: no cover
    BoxUniform = None  # type: ignore[assignment]

from prism_vertex.experiments.task import Task
from prism_vertex.inference.safe_sampling import safe_sample_tensor
from prism_vertex.inference.sbi_posterior_utils import proposal_log_prob_unnorm_if_restricted
from prism_vertex.utils.seeding import derive_seed


def _make_prior_from_task(task: Task, seed: int) -> torch.distributions.Distribution:
    if hasattr(task, "get_prior_distribution"):
        try:
            prior = task.get_prior_distribution()  # type: ignore[attr-defined]
            if isinstance(prior, torch.distributions.Distribution):
                return prior
        except Exception:
            pass
    # Build bounded prior from sampled support to keep compatibility across tasks.
    theta = task.sample_prior(5000, seed=derive_seed(seed, "prior", "prior_bounds"))
    if BoxUniform is not None:
        # Use a generous support expansion around sampled prior values.
        # Estimating bounds from finite samples can otherwise make the BoxUniform too tight,
        # causing later true-prior draws to fall outside and trigger SNPE-C prior-eval failures.
        lo_np = np.min(theta, axis=0).astype(np.float32)
        hi_np = np.max(theta, axis=0).astype(np.float32)
        span = np.maximum(hi_np - lo_np, 1e-6).astype(np.float32)
        pad = (0.05 * span + 1e-3).astype(np.float32)
        lo = torch.as_tensor(lo_np - pad, dtype=torch.float32)
        hi = torch.as_tensor(hi_np + pad, dtype=torch.float32)
        return BoxUniform(low=lo, high=hi)
    theta_t = torch.as_tensor(theta, dtype=torch.float32)
    mean = theta_t.mean(dim=0)
    std = theta_t.std(dim=0).clamp_min(1e-6)
    return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)


def _round_allocations(total_hf: int, num_rounds: int) -> list[int]:
    base = total_hf // num_rounds
    rem = total_hf % num_rounds
    out = [base] * num_rounds
    for i in range(rem):
        out[i] += 1
    return out


def _round_allocations_floor(total_hf: int, num_rounds: int) -> list[int]:
    n_rounds = max(1, int(num_rounds))
    per_round = int(total_hf) // n_rounds
    return [per_round] * n_rounds


def _ensure_x_batch(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _sanitize_theta_with_prior(
    *,
    theta: np.ndarray,
    prior: torch.distributions.Distribution,
    seed: int,
) -> np.ndarray:
    """
    Keep only finite theta rows that are inside prior support (finite log_prob).
    Replace invalid rows with fresh prior samples to prevent simulator crashes.
    """
    arr = np.asarray(theta, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[0] == 0:
        return arr
    t = torch.as_tensor(arr, dtype=torch.float32)
    with torch.no_grad():
        lp = prior.log_prob(t).detach().cpu().numpy().reshape(-1)
    valid = np.isfinite(arr).all(axis=1) & np.isfinite(lp)
    if np.all(valid):
        return arr
    out = arr.copy()
    bad_idx = np.where(~valid)[0]
    if bad_idx.size > 0:
        repl = prior.sample((int(bad_idx.size),)).detach().cpu().numpy().astype(np.float32)
        out[bad_idx] = repl
    return out


def _sample_from_proposal_truncated(
    proposal: Any,
    n: int,
    *,
    epsilon: float,
    seed: int,
    pool_factor: int = 20,
    theta_min: np.ndarray | None = None,
    theta_max: np.ndarray | None = None,
) -> np.ndarray:
    # Approximate density truncation by sampling from proposal and keeping high-density region.
    pool_n = max(n * pool_factor, n)
    theta_pool_t = safe_sample_tensor(dist=proposal, n=pool_n, fallback_dist=proposal)

    with torch.no_grad():
        # log_prob can be -inf outside support; we must keep it finite for thresholding.
        logp_t = proposal_log_prob_unnorm_if_restricted(proposal, theta_pool_t)
        logp = logp_t.detach().cpu().numpy().reshape(-1)

    theta_pool = theta_pool_t.detach().cpu().numpy()

    is_theta_finite = np.isfinite(theta_pool).all(axis=1)
    is_logp_finite = np.isfinite(logp)
    valid = is_theta_finite & is_logp_finite

    if not np.any(valid):
        if theta_min is not None and theta_max is not None:
            rng = np.random.default_rng(seed)
            theta_min_f = np.asarray(theta_min, dtype=np.float32)
            theta_max_f = np.asarray(theta_max, dtype=np.float32)
            range_mag = np.maximum(np.abs(theta_max_f - theta_min_f), 1e-6)
            margin = 1e-3 * range_mag + 1e-6
            lo = theta_min_f + margin
            hi = theta_max_f - margin
            lo = np.minimum(lo, hi)
            theta = rng.uniform(lo, hi, size=(n, theta_min_f.shape[0])).astype(np.float32)
            return theta
        return theta_pool[:n].astype(np.float32)

    logp_valid = logp.copy()
    logp_valid[~valid] = -np.inf

    max_lp = float(np.max(logp_valid))
    threshold = max_lp + np.log(max(float(epsilon), 1e-12))
    keep = np.where(logp_valid >= threshold)[0]
    if len(keep) < n:
        keep = np.argsort(logp_valid)[-n:]
    rng = np.random.default_rng(seed)
    if len(keep) > n:
        keep = rng.choice(keep, size=n, replace=False)
    theta = theta_pool[keep]

    if theta_min is not None and theta_max is not None:
        theta_min_f = np.asarray(theta_min, dtype=np.float32)
        theta_max_f = np.asarray(theta_max, dtype=np.float32)
        range_mag = np.maximum(np.abs(theta_max_f - theta_min_f), 1e-6)
        margin = 1e-3 * range_mag + 1e-6
        lo = theta_min_f + margin
        hi = theta_max_f - margin
        lo = np.minimum(lo, hi)
        theta = np.clip(theta, lo, hi)

    theta = theta[np.isfinite(theta).all(axis=1)]
    if theta.shape[0] < n and theta_min is not None and theta_max is not None:
        rng = np.random.default_rng(seed)
        missing = n - theta.shape[0]
        theta_min_f = np.asarray(theta_min, dtype=np.float32)
        theta_max_f = np.asarray(theta_max, dtype=np.float32)
        range_mag = np.maximum(np.abs(theta_max_f - theta_min_f), 1e-6)
        margin = 1e-3 * range_mag + 1e-6
        lo = theta_min_f + margin
        hi = theta_max_f - margin
        lo = np.minimum(lo, hi)
        extra = rng.uniform(lo, hi, size=(missing, theta_min_f.shape[0])).astype(np.float32)
        theta = np.vstack([theta, extra])

    return theta[:n].astype(np.float32)
