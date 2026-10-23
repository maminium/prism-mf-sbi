"""MF-SBI: multi-fidelity simulation-based inference with SNPE proposals."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from prism_vertex.experiments.task import Task
from prism_vertex.inference.mf_sbi_core import MfSbiResult, run_mf_sbi_core


def run_mf_sbi(
    *,
    task: Task,
    x_o: np.ndarray,
    budget_lf: int,
    budget_hf: int,
    seed: int,
    num_posterior_samples: int,
    config: Mapping[str, Any] | None = None,
    theta_true: np.ndarray | None = None,
) -> MfSbiResult:
    """Run MF-SBI (LF warmup, active LF acquisition, mixed LF/HF rounds)."""
    cfg = dict(config or {})
    cfg["use_snpe_proposal"] = True
    cfg.setdefault("diagnostics_method_name", "mf_sbi")
    return run_mf_sbi_core(
        task=task,
        x_o=x_o,
        budget_lf=budget_lf,
        budget_hf=budget_hf,
        seed=seed,
        num_posterior_samples=num_posterior_samples,
        config=cfg,
        theta_true=theta_true,
    )
