"""Best HF-simulator θ matching Yan w165 x_o (pseudo-truth for metrics).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism_vertex.simulator.nonproliferating.constants import (
    INFER_OUTPUT_SUBSPACE_5D,
    INFER_OUTPUT_SUBSPACE_6D,
    THETA_NAMES,
)
from prism_vertex.simulator.nonproliferating.output_subspace import slice_summary
from prism_vertex.simulator.nonproliferating.pilot_design import build_pilot_theta_list

SUMMARY_NAMES_8 = (
    "mean_log_ell",
    "std_log_ell",
    "mean_c2",
    "mean_s2",
    "ell_c2_cov",
    "mean_log_A",
    "std_log_A",
    "frac_hex",
)


@dataclass(frozen=True)
class SimRecord:
    source: str
    theta: np.ndarray
    x8: np.ndarray
    dist6: float
    dist5: float
    dist8: float


def _exp_root(results_root: Path) -> Path:
    root = Path(results_root)
    if (root / "observations").is_dir():
        return root
    cand = root / "nonproliferating_vertex"
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f"No nonproliferating results under {results_root}")


def load_x_o(exp_root: Path) -> np.ndarray:
    return np.asarray(np.load(exp_root / "observations" / "x_o_w165_mean.npy"), dtype=float).reshape(-1)


def _row_from_arrays(source: str, theta: np.ndarray, x8: np.ndarray, x_o8: np.ndarray) -> SimRecord:
    th = np.asarray(theta, dtype=float).reshape(-1)
    x = np.asarray(x8, dtype=float).reshape(-1)
    if th.size < 3 or x.size < 8:
        raise ValueError(f"Expected 3D theta and 8D x for {source}")
    x6 = slice_summary(x, INFER_OUTPUT_SUBSPACE_6D)
    x5 = slice_summary(x, INFER_OUTPUT_SUBSPACE_5D)
    o6 = slice_summary(x_o8, INFER_OUTPUT_SUBSPACE_6D)
    o5 = slice_summary(x_o8, INFER_OUTPUT_SUBSPACE_5D)
    return SimRecord(
        source=source,
        theta=th[:3],
        x8=x,
        dist6=float(np.linalg.norm(x6 - o6)),
        dist5=float(np.linalg.norm(x5 - o5)),
        dist8=float(np.linalg.norm(x - x_o8)),
    )


def _iter_train_sims(exp_root: Path) -> list[SimRecord]:
    out: list[SimRecord] = []
    hf_inf = exp_root / "hf_inference"
    if not hf_inf.is_dir():
        return out
    x_o8 = load_x_o(exp_root)
    for npz in sorted(hf_inf.rglob("train_sims.npz")):
        data = np.load(npz)
        theta = np.asarray(data["theta"], dtype=float)
        x8 = np.asarray(data["x8"], dtype=float)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        if theta.shape[1] < 3:
            continue
        rel = npz.relative_to(exp_root)
        for i in range(theta.shape[0]):
            out.append(_row_from_arrays(f"train_sims:{rel.parent}:{i}", theta[i], x8[i], x_o8))
    return out


def _iter_scoreboard_hf(exp_root: Path) -> list[SimRecord]:
    out: list[SimRecord] = []
    cells_path = exp_root / "lf_scoreboard" / "analysis" / "cells.csv"
    if not cells_path.is_file():
        return out
    thetas, meta = build_pilot_theta_list()
    x_o8 = load_x_o(exp_root)
    cells = pd.read_csv(cells_path)
    hf = cells[(cells["fidelity_id"] == "hf_single") & (cells["status"] == "ok")]
    for _, r in hf.iterrows():
        ti = int(r["theta_index"])
        tag = meta[ti]["tag"]
        x8 = np.array(
            [
                r["mean_log_ell"],
                r["std_log_ell"],
                r["mean_c2"],
                r["mean_s2"],
                r["s4"],
                r["mean_log_A"],
                np.nan,
                r["frac_hex"],
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(x8[[0, 1, 2, 3, 5, 7]])):
            continue
        src = f"scoreboard_hf:{tag}:seed_{int(r['seed_k'])}"
        rec = _row_from_arrays(src, thetas[ti], x8, x_o8)
        out.append(rec)
    return out


def collect_sim_pool(exp_root: Path) -> list[SimRecord]:
    pool = _iter_train_sims(exp_root) + _iter_scoreboard_hf(exp_root)
    if not pool:
        raise FileNotFoundError(f"No (theta, x) HF simulation records under {exp_root}")
    return pool


def find_theta_hat(
    exp_root: Path,
    *,
    subspace: str = "out6d",
) -> dict[str, Any]:
    """Return θ̂ = argmin ||x_HF(θ) - x_o|| over pooled HF sims."""
    exp_root = _exp_root(exp_root)
    x_o8 = load_x_o(exp_root)
    pool = collect_sim_pool(exp_root)
    key = {"out6d": "dist6", "out5d": "dist5", "out8d": "dist8"}[subspace]
    best = min(pool, key=lambda r: getattr(r, key))
    x_hat6 = slice_summary(best.x8, INFER_OUTPUT_SUBSPACE_6D)
    x_hat5 = slice_summary(best.x8, INFER_OUTPUT_SUBSPACE_5D)
    x_o6 = slice_summary(x_o8, INFER_OUTPUT_SUBSPACE_6D)
    x_o5 = slice_summary(x_o8, INFER_OUTPUT_SUBSPACE_5D)
    return {
        "definition": "theta_hat = HF simulator argmin L2 distance to w165 mean x_o",
        "subspace_primary": subspace,
        "theta_hat": best.theta.tolist(),
        "theta_names": list(THETA_NAMES),
        "best_source": best.source,
        "pool_size": len(pool),
        "pool_breakdown": {
            "train_sims": sum(1 for r in pool if r.source.startswith("train_sims:")),
            "scoreboard_hf": sum(1 for r in pool if r.source.startswith("scoreboard_hf:")),
        },
        "dist6": best.dist6,
        "dist5": best.dist5,
        "dist8": best.dist8 if np.isfinite(best.dist8) else None,
        "x_o6": x_o6.tolist(),
        "x_hat6": x_hat6.tolist(),
        "residual6": (x_hat6 - x_o6).tolist(),
        "x_o5": x_o5.tolist(),
        "x_hat5": x_hat5.tolist(),
        "residual5": (x_hat5 - x_o5).tolist(),
        "summary_names_6d": [SUMMARY_NAMES_8[i] for i in INFER_OUTPUT_SUBSPACE_6D],
        "note": (
            "θ̂ is the best HF forward-model match in archived sim pools, not Yan's MCMC truth. "
            "MF inference/ campaign sims were not saved as train_sims.npz locally."
        ),
    }


def theta_hat_path(exp_root: Path) -> Path:
    return _exp_root(exp_root) / "inference" / "analysis" / "theta_hat_w165.json"


def save_theta_hat(exp_root: Path, payload: dict[str, Any] | None = None) -> Path:
    exp_root = _exp_root(exp_root)
    out = theta_hat_path(exp_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = payload or find_theta_hat(exp_root)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def load_theta_hat(exp_root: Path) -> np.ndarray:
    path = theta_hat_path(exp_root)
    if not path.is_file():
        save_theta_hat(exp_root)
    data = json.loads(path.read_text())
    return np.asarray(data["theta_hat"], dtype=float)


def load_theta_hat_payload(exp_root: Path) -> dict[str, Any]:
    path = theta_hat_path(exp_root)
    if not path.is_file():
        save_theta_hat(exp_root)
    return json.loads(path.read_text())
