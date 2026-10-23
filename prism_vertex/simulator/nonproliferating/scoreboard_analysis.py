"""nonproliferating scoreboard frame + PRISM helpers (6D primary subspace)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prism_vertex.simulator.nonproliferating.fidelities import (
    FIDELITY_AXIS,
    LF_SCOREBOARD_IDS,
    SCOREBOARD_FIDELITY_IDS,
)
from prism_vertex.simulator.nonproliferating.pilot_design import N_PILOT_SEEDS
from prism_vertex.simulator.nonproliferating.scoreboard_io import raw_run_dir
from prism_vertex.utils.scoreboard_failure_registry import classify_error

SUMMARY_NAMES = (
    "mean_log_ell",
    "std_log_ell",
    "mean_c2",
    "mean_s2",
    "mean_log_A",
    "frac_hex",
)
SUMMARY_DIM = len(SUMMARY_NAMES)

HF_ID = "hf_single"
LF_ORDER = list(LF_SCOREBOARD_IDS)
FIDELITY_ORDER = list(SCOREBOARD_FIDELITY_IDS)

ERROR_CLASS_COLORS = {
    "none": "#2ca02c",
    "crash": "#d62728",
    "stall": "#ff7f0e",
    "timeout": "#9467bd",
    "mpi_launch": "#8c564b",
    "launch": "#e377c2",
    "missing_summary": "#7f7f7f",
    "invalid_summary": "#bcbd22",
    "unphysical": "#17becf",
    "other": "#1f77b4",
    "absent": "#cccccc",
    "unknown": "#aaaaaa",
    "missing": "#cccccc",
}

FIDELITY_COLORS = {
    "lf_T0125": "#1b9e77",
    "lf_T025": "#2ca25f",
    "lf_T05": "#99d8c9",
    "lf_DT2": "#d95f02",
    "lf_DT5": "#fc8d62",
    "lf_DT10": "#fdcdac",
    "lf_MESH8": "#7570b3",
    "lf_MESH4": "#b2abd2",
    "lf_HONEY": "#7f7f7f",
    "lf_VORO_RELAX5": "#525252",
    "lf_R1": "#bdd7e7",
    "lf_R2": "#6baed6",
    "lf_R5": "#2171b5",
}


def cells_csv_path(root: Path) -> Path:
    return Path(root) / "analysis" / "cells.csv"


def _summary_6d_from_meta(meta: dict) -> np.ndarray:
    s = meta.get("summary_6d")
    if s is None:
        return np.full(SUMMARY_DIM, np.nan, dtype=float)
    arr = np.asarray(s, dtype=float).reshape(-1)
    out = np.full(SUMMARY_DIM, np.nan, dtype=float)
    out[: min(SUMMARY_DIM, arr.size)] = arr[:SUMMARY_DIM]
    return out


def build_cells_frame(root: Path, n_theta: int, n_seeds: int = N_PILOT_SEEDS) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(n_theta):
        for k in range(n_seeds):
            for fid in SCOREBOARD_FIDELITY_IDS:
                run = raw_run_dir(root, i, k, fid)
                meta_p = run / "run_meta.json"
                if not meta_p.exists():
                    rows.append(
                        {
                            "theta_index": i,
                            "seed_k": k,
                            "fidelity_id": fid,
                            "axis": FIDELITY_AXIS.get(fid, ""),
                            "status": "missing",
                            "error_class": "missing",
                            "wall_seconds": np.nan,
                            **{f"s{j}": np.nan for j in range(SUMMARY_DIM)},
                        }
                    )
                    continue
                meta = json.loads(meta_p.read_text())
                status = str(meta.get("status", "unknown"))
                wall = meta.get("wall_seconds_python", meta.get("wall_seconds", np.nan))
                err = meta.get("error", "") or ""
                error_class = str(meta.get("error_class") or classify_error(err, status))
                summary = _summary_6d_from_meta(meta)
                row = {
                    "theta_index": i,
                    "seed_k": k,
                    "fidelity_id": fid,
                    "axis": str(meta.get("axis", FIDELITY_AXIS.get(fid, ""))),
                    "status": status,
                    "error_class": error_class,
                    "wall_seconds": float(wall) if wall is not None else np.nan,
                }
                for j in range(SUMMARY_DIM):
                    row[f"s{j}"] = float(summary[j]) if j < summary.size else np.nan
                for name, j in zip(SUMMARY_NAMES, range(SUMMARY_DIM)):
                    row[name] = row[f"s{j}"]
                rows.append(row)
    return pd.DataFrame(rows)


def load_cells_frame(root: Path) -> pd.DataFrame:
    p = cells_csv_path(root)
    if p.exists():
        return pd.read_csv(p)
    theta = np.load(Path(root) / "theta_pilot_list.npy")
    return build_cells_frame(root, int(theta.shape[0]))


def tensor_for_fidelity(
    df: pd.DataFrame, fidelity_id: str, n_theta: int, n_seeds: int = N_PILOT_SEEDS
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.full((n_theta, n_seeds, SUMMARY_DIM), np.nan, dtype=float)
    ok = np.zeros((n_theta, n_seeds), dtype=bool)
    sub = df[df["fidelity_id"] == fidelity_id]
    for _, row in sub.iterrows():
        i, k = int(row["theta_index"]), int(row["seed_k"])
        if not (0 <= i < n_theta and 0 <= k < n_seeds):
            continue
        vals = np.array([row.get(f"s{j}", np.nan) for j in range(SUMMARY_DIM)], dtype=float)
        arr[i, k] = vals
        ok[i, k] = str(row["status"]) == "ok" and np.isfinite(vals).all()
    return arr, ok


def hf_mean_and_std(
    df: pd.DataFrame, n_theta: int, n_seeds: int = N_PILOT_SEEDS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hf, ok = tensor_for_fidelity(df, HF_ID, n_theta, n_seeds)
    mean = np.full((n_theta, SUMMARY_DIM), np.nan, dtype=float)
    std = np.full((n_theta, SUMMARY_DIM), np.nan, dtype=float)
    count = np.zeros(n_theta, dtype=int)
    for i in range(n_theta):
        idxs = np.where(ok[i])[0]
        count[i] = int(idxs.size)
        if idxs.size == 0:
            continue
        mean[i] = np.mean(hf[i, idxs], axis=0)
        std[i] = np.std(hf[i, idxs], axis=0, ddof=0) if idxs.size > 1 else 0.0
    return mean, std, count
