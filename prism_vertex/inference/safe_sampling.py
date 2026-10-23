from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _sample_shape(n: int) -> tuple[int, ...]:
    return (int(n),)


def safe_sample_tensor(
    *,
    dist: Any,
    n: int,
    fallback_dist: Any | None = None,
) -> torch.Tensor:
    """
    Robust distribution sampling for SBI/nflows posteriors.
    Falls back to fallback_dist (or dist itself) when sampling raises numerical assertions.
    """
    try:
        return dist.sample(_sample_shape(n))
    except (AssertionError, RuntimeError, ValueError, FloatingPointError):
        fb = fallback_dist if fallback_dist is not None else dist
        return fb.sample(_sample_shape(n))


def safe_sample_numpy(
    *,
    dist: Any,
    n: int,
    fallback_dist: Any | None = None,
) -> np.ndarray:
    t = safe_sample_tensor(dist=dist, n=n, fallback_dist=fallback_dist)
    return t.detach().cpu().numpy()

