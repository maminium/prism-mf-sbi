from __future__ import annotations

import numpy as np

# Operational prior used in prism_vertex task + ABC wrapper.
LAM_MIN, LAM_MAX = -1.45, 0.25
GAM_MIN, GAM_MAX = 0.01, 0.20


def in_operational_box(theta: np.ndarray) -> bool:
    t = np.asarray(theta, dtype=float).reshape(-1)
    lam, gam = float(t[0]), float(t[1])
    return (LAM_MIN <= lam <= LAM_MAX) and (GAM_MIN <= gam <= GAM_MAX)


def gamma_ab_boundary(lam: np.ndarray | float) -> np.ndarray:
    """
    Proxy A|B boundary matching the paper's Fig.3 geometry.

    The exact analytical inequalities from Staple/Kursawe are not implemented in the
    Chaste driver; we therefore use a visual proxy for coverage diagnostics only.
    """
    x = np.asarray(lam, dtype=float)
    # Approximate line through (-1.5, 0.16) and (0, 0).
    return -0.1066666667 * x


def lambda_bc_boundary(gam: np.ndarray | float) -> np.ndarray:
    """
    Proxy B|C boundary matching the paper's Fig.3 right-hand curve.
    """
    y = np.asarray(gam, dtype=float)
    y_clip = np.clip(y, 0.0, 0.15)
    return 0.25 * np.sqrt(np.maximum(0.0, 1.0 - y_clip / 0.15))


def in_region_b_proxy(theta: np.ndarray) -> bool:
    """
    Region-B proxy for plotting and point design.

    B is above A|B line and left of B|C curve, inside the operational box.
    """
    t = np.asarray(theta, dtype=float).reshape(-1)
    lam, gam = float(t[0]), float(t[1])
    if not in_operational_box(t):
        return False
    if gam < max(0.0, float(gamma_ab_boundary(lam))):
        return False
    if lam > float(lambda_bc_boundary(gam)):
        return False
    return True
