"""Shared inference-time context: x_o, penalty scales, failure registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from prism_vertex.simulator.inference_failure_registry import (
    InferenceFailureRegistry,
    init_inference_failure_registry,
)
from prism_vertex.inference.penalty_summaries import penalty_summary_vector

_X_O: np.ndarray | None = None
_SCALES: np.ndarray | None = None


def _env_flag(name: str, legacy: str | None = None) -> bool:
    for key in (name, legacy) if legacy else (name,):
        if key is None:
            continue
        val = os.environ.get(key, "").strip()
        if val in {"1", "true", "True", "yes", "YES"}:
            return True
        if val in {"0", "false", "False", "no", "NO"}:
            return False
    return False


def init_inference_simulation(
    *,
    x_o: np.ndarray,
    run_dir: Path,
    scales: np.ndarray | None = None,
) -> InferenceFailureRegistry:
    """Call once at the start of each inference run."""
    global _X_O, _SCALES
    _X_O = np.asarray(x_o, dtype=np.float64).reshape(-1).copy()
    if scales is not None:
        _SCALES = np.asarray(scales, dtype=np.float64).reshape(-1).copy()
    else:
        _SCALES = np.maximum(np.abs(_X_O), 1e-6)
    return init_inference_failure_registry(run_dir)


def get_inference_x_o() -> np.ndarray | None:
    return None if _X_O is None else _X_O.copy()


def get_penalty_scales() -> np.ndarray:
    global _SCALES
    if _SCALES is None:
        raise RuntimeError("init_inference_simulation not called")
    return _SCALES.copy()


def make_penalty_summary() -> np.ndarray:
    x_o = get_inference_x_o()
    if x_o is None:
        raise RuntimeError("init_inference_simulation not called")
    return penalty_summary_vector(x_o, scales=get_penalty_scales())


def get_failure_registry() -> InferenceFailureRegistry:
    return InferenceFailureRegistry.from_env()


def baseline_drop_on_fail_enabled() -> bool:
    """M2/M3 baselines: omit failed sims from training."""
    return _env_flag("PRISM_VERTEX_INFERENCE_BASELINE_DROP_ON_FAIL", "PRISM_VERTEX_BASELINE_DROP_ON_FAIL")


def penalty_mode_enabled() -> bool:
    """If set, failed sims keep a penalty summary instead of dropping the row."""
    return _env_flag("PRISM_VERTEX_INFERENCE_PENALTY_MODE", "PRISM_VERTEX_PENALTY_MODE")


def smoke_final_samples(task: Any, n: int, seed: int, *, inference_smoke: bool) -> np.ndarray | None:
    """Skip slow SNPE rejection sampling in smoke runs (wiring check only)."""
    if not inference_smoke:
        return None
    return np.asarray(task.sample_prior(int(n), seed=int(seed)), dtype=np.float32)


def smoke_prior_candidates(task: Any, n: int, seed: int, *, inference_smoke: bool) -> np.ndarray | None:
    """Smoke runs: never draw round proposals from an untrained posterior."""
    if not inference_smoke:
        return None
    return np.asarray(task.sample_prior(int(n), seed=int(seed)), dtype=np.float32)
