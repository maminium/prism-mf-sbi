from __future__ import annotations

from typing import Any

import numpy as np
import torch

from prism_vertex.inference.sbi_posterior_utils import posterior_log_prob_unnorm


def _top_b_by_uncertainty(
    *,
    candidates_t: torch.Tensor,
    ensemble_posteriors: list[Any],
    b: int,
) -> np.ndarray:
    if b <= 0:
        return np.empty((0, candidates_t.shape[-1]), dtype=np.float32)
    if not ensemble_posteriors:
        idx = np.arange(min(b, candidates_t.shape[0]))
        return candidates_t[idx].detach().cpu().numpy().astype(np.float32)
    logps = []
    with torch.no_grad():
        for post in ensemble_posteriors:
            lp = posterior_log_prob_unnorm(post, candidates_t).detach().cpu().numpy().reshape(-1)
            logps.append(lp)
    arr = np.stack(logps, axis=0)  # [E, N]
    score = np.var(arr, axis=0)
    idx = np.argsort(score)[-min(b, len(score)) :]
    return candidates_t[idx].detach().cpu().numpy().astype(np.float32)
