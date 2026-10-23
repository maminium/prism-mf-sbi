"""
Exact NLTP = -log q(θ* | x_o) under the learned neural posterior (paper definition).

We evaluate the posterior log-density with ``norm_posterior=False`` (see
``posterior_log_prob_unnorm``) so sbi does not run expensive / fragile Monte Carlo
normalization. Failed normalized ``log_prob`` calls were dropping ``nltp_exact``
from ``diagnostics.json`` silently on some tasks (e.g. exp09 MF-truncated-proposal).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from prism_vertex.inference.sbi_posterior_utils import posterior_log_prob_unnorm


def add_nltp_exact_neural(
    diagnostics: dict[str, Any],
    posterior: Any,
    theta_true: np.ndarray | None,
    x_o: np.ndarray | None,
) -> None:
    """
    Mutates ``diagnostics`` with ``nltp_exact`` when computable, else
    ``nltp_exact_status`` and optional ``nltp_exact_error`` / ``nltp_exact_note``.
    """
    if theta_true is None:
        diagnostics["nltp_exact_status"] = "skipped_no_theta_true"
        return
    tt = torch.as_tensor(np.asarray(theta_true, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
    x_t: torch.Tensor | None = None
    if x_o is not None:
        xa = np.asarray(x_o, dtype=np.float32)
        if xa.ndim == 1:
            xa = xa.reshape(1, -1)
        x_t = torch.as_tensor(xa, dtype=torch.float32)
    try:
        with torch.no_grad():
            lp = posterior_log_prob_unnorm(posterior, tt, x=x_t)
        nltp = float((-lp.reshape(-1)[0]).item())
        if np.isfinite(nltp):
            diagnostics["nltp_exact"] = float(nltp)
            diagnostics["nltp_exact_status"] = "ok"
        else:
            diagnostics["nltp_exact_status"] = "non_finite"
    except Exception as e:
        diagnostics["nltp_exact_status"] = "failed"
        diagnostics["nltp_exact_error"] = f"{type(e).__name__}: {e}"


def add_nltp_exact_smc_abc_na(diagnostics: dict[str, Any]) -> None:
    """SMC-ABC outputs particles only; parametric q(θ|x) is not defined."""
    diagnostics["nltp_exact_status"] = "not_applicable_nonparametric"
    diagnostics["nltp_exact_note"] = (
        "hf_smc_abc has no parametric posterior density; use nltp_kde / nltp_gaussian from samples."
    )
