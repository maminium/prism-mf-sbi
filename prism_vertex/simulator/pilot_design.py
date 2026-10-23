from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prism_vertex.simulator.prior_region import (
    GAM_MAX,
    GAM_MIN,
    LAM_MAX,
    LAM_MIN,
    in_region_b_proxy,
)
from prism_vertex.simulator.task import THETA_TRUE
from prism_vertex.utils.seeding import derive_seed

N_PILOT_THETA = 20
N_PILOT_SEEDS = 10
PILOT_THETA_SEED = 20260527


def scoreboard_sim_seed(theta_index: int, seed_k: int) -> int:
    """Same seed for HF and all LFs at (theta_index, seed_k)."""
    return int(derive_seed("prism_vertex", "scoreboard", theta_index, seed_k))


def _fixed_stratified_points() -> list[tuple[str, np.ndarray]]:
    """Hand-picked: centre, near-boundaries, and interior anchors in region-B proxy."""
    pts: list[tuple[str, np.ndarray]] = [
        ("theta_true", THETA_TRUE.copy()),
        # Corners (inset ~10% from box edges)
        ("corner_ll", np.array([-1.35, 0.03])),
        ("corner_lh", np.array([-1.35, 0.16])),
        ("corner_hl", np.array([0.18, 0.03])),
        ("corner_hh", np.array([0.05, 0.16])),
        # Border midpoints (stress unstable / sloppy edges)
        ("border_lam_low", np.array([-1.42, 0.10])),
        ("border_lam_high", np.array([0.23, 0.03])),
        ("border_gam_low", np.array([0.00, 0.015])),
        ("border_gam_high", np.array([-0.15, 0.19])),
        # Near theoretical A|B boundary (negative Λ, moderate Γ)
        ("near_A_B", np.array([-1.20, 0.08])),
        # Interior “safe” anchors
        ("interior_a", np.array([-0.35, 0.09])),
        ("interior_b", np.array([0.05, 0.06])),
        ("interior_c", np.array([-0.70, 0.12])),
    ]
    return pts


def build_pilot_theta_list(
    n: int = N_PILOT_THETA,
    *,
    rng_seed: int = PILOT_THETA_SEED,
) -> tuple[np.ndarray, list[dict]]:
    """
    Build N_θ pilot parameters: fixed stratified + LHS fill.

    Returns theta array (n, 2) and metadata list with ``tag`` per row.
    """
    fixed = _fixed_stratified_points()
    meta: list[dict] = []
    rows: list[np.ndarray] = []

    for tag, theta in fixed:
        if not in_region_b_proxy(theta):
            continue
        rows.append(theta.astype(np.float64))
        meta.append({"tag": tag, "source": "fixed"})

    rng = np.random.default_rng(rng_seed)
    while len(rows) < n:
        lam = rng.uniform(LAM_MIN, LAM_MAX)
        gam = rng.uniform(GAM_MIN, GAM_MAX)
        theta = np.array([lam, gam], dtype=np.float64)
        if not in_region_b_proxy(theta):
            continue
        # Dedupe roughly
        if any(np.linalg.norm(theta - r) < 1e-4 for r in rows):
            continue
        rows.append(theta)
        meta.append({"tag": f"lhs_{len(rows):02d}", "source": "lhs"})

    theta_arr = np.stack(rows[:n], axis=0)
    return theta_arr, meta[:n]


def save_pilot_design(out_dir: Path, n_theta: int = N_PILOT_THETA) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theta, theta_meta = build_pilot_theta_list(n_theta)
    np.save(out_dir / "theta_pilot_list.npy", theta)

    seed_map: dict[str, dict] = {}
    for i in range(theta.shape[0]):
        for k in range(N_PILOT_SEEDS):
            seed_map[f"{i:02d}_{k:02d}"] = {
                "theta_index": i,
                "seed_k": k,
                "sim_seed": scoreboard_sim_seed(i, k),
            }

    design = {
        "n_theta": int(theta.shape[0]),
        "n_seeds": N_PILOT_SEEDS,
        "lam_range": [LAM_MIN, LAM_MAX],
        "gam_range": [GAM_MIN, GAM_MAX],
        "theta_meta": theta_meta,
        "seed_map": seed_map,
        "theta_true_index": int(np.argmin(np.linalg.norm(theta - THETA_TRUE, axis=1))),
    }
    path = out_dir / "pilot_design.json"
    path.write_text(json.dumps(design, indent=2))
    return path


def load_pilot_theta(out_dir: Path) -> np.ndarray:
    p = Path(out_dir) / "theta_pilot_list.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run scoreboard with --init-design only first.")
    return np.load(p)
