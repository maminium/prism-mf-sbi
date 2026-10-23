"""Observation-opposed penalty summaries for failed Vertex tissue Chaste runs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prism_vertex.simulator.chaste_summaries import SUMMARY_DIM

# Area-ratio dims (0–4) and elongation (5) — sensible clip ranges for penalties.
_RATIO_LOW = 0.0
_RATIO_HIGH = 2.5
_ELONG_LOW = 0.0
_ELONG_HIGH = 3.0


def default_penalty_scales(observations_dir: Path | None = None) -> np.ndarray:
    """Per-dimension scales for κ·σ offsets (fallback if s_obs_std missing)."""
    if observations_dir is not None:
        p = Path(observations_dir) / "s_obs_std.npy"
        if p.exists():
            scales = np.asarray(np.load(p), dtype=np.float64).reshape(-1)
            if scales.shape == (SUMMARY_DIM,) and np.all(np.isfinite(scales) & (scales > 0)):
                return scales
    return np.full(SUMMARY_DIM, 0.05, dtype=np.float64)


def penalty_summary_vector(
    x_o: np.ndarray,
    *,
    mode: str | None = None,
    kappa: float | None = None,
    scales: np.ndarray | None = None,
    fill: float = 1.0e9,
) -> np.ndarray:
    """
    Synthetic 6D summary marking simulator failure at θ.

    ``observation_offset`` (default): x_o + κ·σ with per-dim sign pushing away from x_o.
    ``constant_fill``: legacy ABC-style 1e9 vector.
    """
    mode = (mode or os.environ.get("PRISM_VERTEX_PENALTY_MODE", "observation_offset")).strip().lower()
    if mode in {"constant", "constant_fill", "fill"}:
        return np.full(SUMMARY_DIM, float(fill), dtype=np.float64)

    x_o = np.asarray(x_o, dtype=np.float64).reshape(-1)
    if x_o.shape != (SUMMARY_DIM,):
        raise ValueError(f"x_o must be shape ({SUMMARY_DIM},), got {x_o.shape}")

    kappa = float(kappa if kappa is not None else os.environ.get("PRISM_VERTEX_PENALTY_KAPPA", "8.0"))
    if scales is None:
        scales = default_penalty_scales()
    else:
        scales = np.asarray(scales, dtype=np.float64).reshape(-1)
    scales = np.maximum(scales, 1e-6)

    # Push each coordinate away from x_o (alternate sign pattern is stable across runs).
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    out = x_o + kappa * scales * signs

    # Clip to physically plausible ranges for Kursawe summaries.
    out[:5] = np.clip(out[:5], _RATIO_LOW, _RATIO_HIGH)
    out[5] = np.clip(out[5], _ELONG_LOW, _ELONG_HIGH)
    return out.astype(np.float64)


def is_penalty_summary(
    x: np.ndarray,
    x_o: np.ndarray,
    *,
    fill: float = 1.0e9,
    kappa: float = 8.0,
    scales: np.ndarray | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """True if x matches the configured penalty (offset or constant fill)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if np.allclose(x, fill, rtol=0, atol=0) and np.all(x >= fill * 0.99):
        return True
    ref = penalty_summary_vector(
        x_o,
        mode="observation_offset",
        kappa=kappa,
        scales=scales,
    )
    return bool(np.allclose(x, ref, rtol=rtol, atol=atol))
