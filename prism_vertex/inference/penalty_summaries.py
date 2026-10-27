from __future__ import annotations

import os

import numpy as np


def penalty_summary_vector(
    x_o: np.ndarray,
    *,
    mode: str | None = None,
    kappa: float | None = None,
    scales: np.ndarray | None = None,
    fill: float = 1.0e9,
) -> np.ndarray:
    """
    Synthetic summary marking simulator failure at θ.

    ``observation_offset`` (default): x_o + κ·σ with alternating signs away from x_o.
    ``constant_fill``: legacy ABC-style large constant vector.
    """
    mode = (mode or os.environ.get("PRISM_VERTEX_INFERENCE_PENALTY_MODE", "observation_offset")).strip().lower()
    if mode in {"constant", "constant_fill", "fill"}:
        x_o = np.asarray(x_o, dtype=np.float64).reshape(-1)
        return np.full(x_o.shape[0], float(fill), dtype=np.float64)

    x_o = np.asarray(x_o, dtype=np.float64).reshape(-1)
    d = int(x_o.shape[0])
    kappa = float(
        kappa
        if kappa is not None
        else os.environ.get("PRISM_VERTEX_INFERENCE_PENALTY_KAPPA", os.environ.get("PRISM_VERTEX_PENALTY_KAPPA", "8.0"))
    )
    if scales is None:
        scales = np.maximum(np.abs(x_o), 1e-6)
    else:
        scales = np.asarray(scales, dtype=np.float64).reshape(-1)
        if scales.shape != x_o.shape:
            raise ValueError(f"scales shape {scales.shape} != x_o shape {x_o.shape}")
    scales = np.maximum(scales, 1e-6)

    signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(d)], dtype=np.float64)
    return (x_o + kappa * scales * signs).astype(np.float64)


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
    if x.size > 0 and np.allclose(x, fill, rtol=0, atol=0) and np.all(x >= fill * 0.99):
        return True
    ref = penalty_summary_vector(
        x_o,
        mode="observation_offset",
        kappa=kappa,
        scales=scales,
    )
    return bool(np.allclose(x, ref, rtol=rtol, atol=atol))
