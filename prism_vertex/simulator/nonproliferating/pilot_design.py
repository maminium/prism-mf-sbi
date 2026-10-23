"""Pilot θ design for nonproliferating PRISM LF scoreboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prism_vertex.simulator.nonproliferating.constants import (
    N_PILOT_SEEDS,
    N_PILOT_THETA,
    PRIOR_HIGH,
    PRIOR_LOW,
    THETA_NAMES,
    THETA_TRUE,
)
from prism_vertex.utils.seeding import derive_seed

PILOT_THETA_SEED = 20260720


def scoreboard_sim_seed(theta_index: int, seed_k: int) -> int:
    return int(derive_seed("nonproliferating_vertex", "scoreboard", theta_index, seed_k))


def in_prior_box(theta: np.ndarray) -> bool:
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    return bool(np.all(t >= PRIOR_LOW) and np.all(t <= PRIOR_HIGH))


def _fixed_points() -> list[tuple[str, np.ndarray]]:
    lo, hi = PRIOR_LOW, PRIOR_HIGH
    corners = []
    for mask in range(8):
        corners.append(
            np.array(
                [
                    lo[0] if mask & 1 else hi[0],
                    lo[1] if mask & 2 else hi[1],
                    lo[2] if mask & 4 else hi[2],
                ],
                dtype=np.float64,
            )
        )
    mids = [
        ("mid_a_r", np.array([0.275, 0.225, 5.0])),
        ("mid_b_0", np.array([0.275, 0.075, 5.0])),
        ("mid_k", np.array([0.275, 0.225, 7.25])),
        ("interior", np.array([0.15, 0.12, 4.0])),
    ]
    out: list[tuple[str, np.ndarray]] = [("theta_true", THETA_TRUE.copy())]
    for i, c in enumerate(corners):
        out.append((f"corner_{i:02d}", c))
    out.extend(mids)
    return out


def build_pilot_theta_list(n: int = N_PILOT_THETA, *, rng_seed: int = PILOT_THETA_SEED):
    rows: list[np.ndarray] = []
    meta: list[dict] = []
    for tag, theta in _fixed_points():
        if not in_prior_box(theta):
            continue
        if any(np.linalg.norm(theta - r) < 1e-5 for r in rows):
            continue
        rows.append(theta.astype(np.float64))
        meta.append({"tag": tag, "source": "fixed"})
    rng = np.random.default_rng(rng_seed)
    while len(rows) < n:
        theta = rng.uniform(PRIOR_LOW, PRIOR_HIGH).astype(np.float64)
        if any(np.linalg.norm(theta - r) < 1e-4 for r in rows):
            continue
        rows.append(theta)
        meta.append({"tag": f"rand_{len(rows):02d}", "source": "lhs"})
    return np.stack(rows[:n], axis=0), meta[:n]


def save_pilot_design(out_dir: Path, n_theta: int = N_PILOT_THETA) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    theta, theta_meta = build_pilot_theta_list(n_theta)
    np.save(out_dir / "theta_pilot_list.npy", theta)
    seed_map = {}
    for i in range(theta.shape[0]):
        for k in range(N_PILOT_SEEDS):
            seed_map[f"{i:02d}_{k:02d}"] = {
                "theta_index": i,
                "seed_k": k,
                "sim_seed": scoreboard_sim_seed(i, k),
            }
    payload = {
        "experiment": "nonproliferating_vertex",
        "n_theta": int(theta.shape[0]),
        "n_seeds": N_PILOT_SEEDS,
        "theta_names": list(THETA_NAMES),
        "theta_meta": theta_meta,
        "seed_map": seed_map,
        "prior_low": PRIOR_LOW.tolist(),
        "prior_high": PRIOR_HIGH.tolist(),
        "theta_true": THETA_TRUE.tolist(),
        "hf_protocol": {
            "init_mode": "voronoi",
            "voronoi_relax": 0,
            "end_time": 5.0,
            "dt": 0.001,
            "initial_size": 12,
        },
    }
    path = out_dir / "pilot_design.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_pilot_design(out_dir: Path) -> dict:
    p = Path(out_dir) / "pilot_design.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Initialize the LF scoreboard design first.")
    return json.loads(p.read_text())
