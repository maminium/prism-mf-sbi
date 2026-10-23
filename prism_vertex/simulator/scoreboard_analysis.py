"""Shared helpers for proliferating scoreboard post-run analysis.

One foundation used by all analysis scripts:
- ``build_cells_frame`` / ``load_cells_frame``: one row per (theta_index, seed_k, fidelity_id)
- ``classify_error``: map a Chaste error string + status to an error class
- ``tensor_for_fidelity``: (n_theta, n_seeds, 6) summary array + ok mask from the frame
- plotting conventions (fidelity order, colors, linestyles) and small stats utilities
  (RBF-MMD, PCA via SVD) so figures stay consistent.

"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prism_vertex.simulator.chaste_summaries import SUMMARY_NAMES
from prism_vertex.simulator.fidelities import SCOREBOARD_FIDELITY_IDS
from prism_vertex.simulator.pilot_design import N_PILOT_SEEDS
from prism_vertex.simulator.scoreboard_io import existing_cell_dir, raw_run_dir

# ----------------------------------------------------------------------------
# Conventions
# ----------------------------------------------------------------------------
HF_ID = "hf_single"
LF_S_ID = "lf_S"  # legacy pilot id; analysis uses lf_R1 (reads lf_S cells if needed)
# 9 LF candidates: 3×T, 3×ΔT, 3×R (K=1,2,5 replicates at HF physics).
LF_ORDER: tuple[str, ...] = (
    "lf_T70",
    "lf_T140",
    "lf_T350",
    "lf_DT02",
    "lf_DT05",
    "lf_DT10",
    "lf_R1",
    "lf_R2",
    "lf_R5",
)
FIDELITY_ORDER: tuple[str, ...] = (HF_ID,) + LF_ORDER

S_COLS: tuple[str, ...] = tuple(f"s{j}" for j in range(len(SUMMARY_NAMES)))

# Consistent colours: HF black; LF-S grey (variance-only); T* blues; DT* warm; R* greens.
FIDELITY_COLORS: dict[str, str] = {
    "hf_single": "#000000",
    "lf_T70": "#CCECE6",
    "lf_T140": "#66C2A4",
    "lf_T350": "#238B45",
    "lf_DT10": "#FCAE91",
    "lf_DT05": "#FB6A4A",
    "lf_DT02": "#CB181D",
    "lf_R1": "#BDD7E7",
    "lf_R2": "#6BAED6",
    "lf_R5": "#2171B5",
    "lf_S": "#CCECE6",
}
FIDELITY_LINESTYLES: dict[str, str] = {
    "hf_single": "-",
    "lf_T70": "-",
    "lf_T140": "-",
    "lf_T350": "-",
    "lf_DT02": "--",
    "lf_DT05": "--",
    "lf_DT10": "--",
    "lf_R1": ":",
    "lf_R2": ":",
    "lf_R5": ":",
    "lf_S": ":",
}
# Error-class colours for failure maps / heatmaps.
ERROR_CLASS_COLORS: dict[str, str] = {
    "none": "#2ca02c",
    "stall": "#d62728",
    "crash": "#9467bd",
    "timeout": "#ff7f0e",
    "launch": "#8c564b",
    "missing_summary": "#7f7f7f",
    "invalid_summary": "#bcbd22",
    "absent": "#e8e8e8",
    "unknown": "#17becf",
    "other": "#1f77b4",
}

# Non-ok classes shown in failure breakdown plots (registry blocks roll into stall/crash).
FAILURE_ERROR_CLASSES: tuple[str, ...] = (
    "stall", "crash", "timeout", "launch", "other", "unknown", "absent",
)

HF_REPLICATE_MIN = 7  # protocol: require >= 7/10 ok HF seeds per theta


# ----------------------------------------------------------------------------
# Error classification
# ----------------------------------------------------------------------------
def classify_error(error: str | None, status: str) -> str:
    """Map (error string, status) to a coarse, scientifically meaningful class."""
    if status == "ok":
        return "none"
    if status == "absent":
        return "absent"
    e = error or ""
    if status == "skipped_known_failed_point":
        if not e.strip():
            return "other"
        if "No output-dir growth" in e:
            return "stall"
        if "Hard timeout" in e or "rc=-9" in e:
            return "timeout"
        if any(k in e for k in ("PerformIntersectionSwap", "terminate called", "libstdc++.so.6(+0x", "EXCEPTION")):
            return "crash"
        return "other"
    if not e.strip():
        return "unknown"
    if "No output-dir growth" in e:
        return "stall"
    if "Hard timeout" in e or "rc=-9" in e:
        return "timeout"
    if any(k in e for k in ("PerformIntersectionSwap", "terminate called", "libstdc++.so.6(+0x", "EXCEPTION")):
        return "crash"
    if any(k in e for k in ("cannot open shared object", "rc=127", "No such file")):
        return "launch"
    if "Missing" in e:
        return "missing_summary"
    if "Invalid summary" in e or "prism_vertex_summary" in e or "Corrupt or incomplete" in e:
        return "invalid_summary"
    return "other"


def run_log_tag(meta: dict) -> str:
    """Human-readable tag for scoreboard runner progress lines."""
    status = str(meta.get("status", ""))
    err = meta.get("error") or meta.get("last_error") or ""
    if meta.get("skipped") and status == "ok":
        return "KEEP"  # already ok — resume, not a failure
    if status == "ok":
        return "OK"
    if status == "failed":
        ec = classify_error(err, status)
        return ec.upper() if ec in ("stall", "crash", "timeout", "launch") else "FAIL"
    if status == "skipped_known_failed_point":
        ec = classify_error(err, status)
        return ec.upper() if ec in ("stall", "crash", "timeout") else "BLOCKED"
    return status.upper()


def load_failure_registry(root: Path) -> dict:
    p = Path(root) / "failure_registry.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def resolve_cell_error(
    meta: dict | None,
    *,
    registry: dict,
    root: Path,
    theta_index: int,
    fidelity_id: str,
) -> str:
    """Error text for a cell; registry-blocked rows inherit stall/crash from registry or siblings."""
    if meta is None:
        return ""
    err = str(meta.get("error") or meta.get("last_error") or "")
    if str(meta.get("status")) != "skipped_known_failed_point" or err.strip():
        return err
    rec = registry.get(f"{theta_index:02d}:{fidelity_id}", {})
    err = str(rec.get("last_error") or "")
    if err.strip():
        return err
    for sk in range(N_PILOT_SEEDS):
        sib = _read_meta(raw_run_dir(root, theta_index, sk, fidelity_id) / "run_meta.json")
        if sib and sib.get("status") == "failed":
            err = str(sib.get("error") or "")
            if err.strip():
                return err
    return ""


# ----------------------------------------------------------------------------
# Frame build / load
# ----------------------------------------------------------------------------
def _read_meta(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def build_cells_frame(root: Path, n_theta: int) -> pd.DataFrame:
    """One row per (theta_index, seed_k, fidelity_id) across the full design grid."""
    theta = np.load(Path(root) / "theta_pilot_list.npy")
    registry = load_failure_registry(root)
    rows: list[dict] = []
    for i in range(n_theta):
        lam, gam = float(theta[i, 0]), float(theta[i, 1])
        for k in range(N_PILOT_SEEDS):
            for fid in SCOREBOARD_FIDELITY_IDS:
                cell_dir = existing_cell_dir(Path(root), i, k, fid)
                meta = _read_meta(cell_dir / "run_meta.json")
                summ_npy = cell_dir / "summary_6d.npy"

                if meta is None:
                    status = "absent"
                    err = ""
                    wall = np.nan
                    n_cells = np.nan
                else:
                    status = str(meta.get("status", "unknown"))
                    err = resolve_cell_error(
                        meta, registry=registry, root=Path(root), theta_index=i, fidelity_id=fid,
                    )
                    wall = float(meta.get("wall_seconds_python", meta.get("wall_seconds", np.nan)) or np.nan)
                    n_cells = meta.get("n_cells", np.nan)

                s_vec = np.full(len(SUMMARY_NAMES), np.nan)
                has_summary = False
                if summ_npy.exists():
                    try:
                        v = np.load(summ_npy)
                        if v.shape[0] == len(SUMMARY_NAMES) and np.isfinite(v).all():
                            s_vec = v.astype(float)
                            has_summary = True
                    except Exception:
                        pass
                if not has_summary and meta is not None and meta.get("summary_6d") is not None:
                    v = np.asarray(meta["summary_6d"], dtype=float).reshape(-1)
                    if v.shape[0] == len(SUMMARY_NAMES) and np.isfinite(v).all():
                        s_vec = v
                        has_summary = True

                row = {
                    "theta_index": i,
                    "seed_k": k,
                    "fidelity_id": fid,
                    "lambda": lam,
                    "gamma": gam,
                    "status": status,
                    "error_class": classify_error(err, status),
                    "wall_seconds": wall,
                    "n_cells": float(n_cells) if n_cells is not None else np.nan,
                    "has_summary": has_summary,
                }
                for j, col in enumerate(S_COLS):
                    row[col] = float(s_vec[j])
                rows.append(row)
    return pd.DataFrame(rows)


def cells_csv_path(root: Path) -> Path:
    return Path(root) / "analysis" / "cells.csv"


def load_cells_frame(root: Path) -> pd.DataFrame:
    p = cells_csv_path(root)
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Build the LF scoreboard frame first.")
    return pd.read_csv(p)


# ----------------------------------------------------------------------------
# Tensors / aggregates derived from the frame
# ----------------------------------------------------------------------------
def tensor_for_fidelity(
    df: pd.DataFrame, fidelity_id: str, n_theta: int, n_seeds: int = N_PILOT_SEEDS
) -> tuple[np.ndarray, np.ndarray]:
    """Return (data[n_theta, n_seeds, 6], ok_mask[n_theta, n_seeds])."""
    data = np.full((n_theta, n_seeds, len(S_COLS)), np.nan)
    ok = np.zeros((n_theta, n_seeds), dtype=bool)
    sub = df[df["fidelity_id"] == fidelity_id]
    for _, r in sub.iterrows():
        i, k = int(r["theta_index"]), int(r["seed_k"])
        if i >= n_theta or k >= n_seeds:
            continue
        if bool(r["has_summary"]) and str(r["status"]) == "ok":
            data[i, k] = [r[c] for c in S_COLS]
            ok[i, k] = True
    return data, ok


def hf_mean_and_std(df: pd.DataFrame, n_theta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HF mean, std (over ok seeds) and count of ok seeds per theta."""
    hf, ok = tensor_for_fidelity(df, HF_ID, n_theta)
    mean = np.full((n_theta, len(S_COLS)), np.nan)
    std = np.full((n_theta, len(S_COLS)), np.nan)
    count = ok.sum(axis=1)
    for i in range(n_theta):
        if count[i] >= 1:
            mean[i] = np.nanmean(hf[i, ok[i]], axis=0)
        if count[i] >= 2:
            std[i] = np.nanstd(hf[i, ok[i]], axis=0, ddof=1)
    return mean, std, count


# ----------------------------------------------------------------------------
# Small stats helpers (no sklearn dependency)
# ----------------------------------------------------------------------------
def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores[n,2], explained_variance_ratio[2]) via SVD on centred X."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, :2] * S[:2]
    var = (S**2) / max(1, (X.shape[0] - 1))
    ratio = var[:2] / var.sum() if var.sum() > 0 else np.zeros(2)
    return scores, ratio


def rbf_mmd2(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased-ish RBF MMD^2 with median-heuristic bandwidth on pooled data."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape[0] < 2 or Y.shape[0] < 2:
        return float("nan")
    Z = np.vstack([X, Y])
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / (med + 1e-12)

    def _k(A, B):
        da = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * da)

    kxx = _k(X, X).mean()
    kyy = _k(Y, Y).mean()
    kxy = _k(X, Y).mean()
    return float(kxx + kyy - 2 * kxy)


def holm_correction(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    prev = 0.0
    for rank, (k, p) in enumerate(items):
        a = min(1.0, (m - rank) * p)
        a = max(a, prev)  # enforce monotonicity
        adj[k] = a
        prev = a
    for k, v in pvals.items():
        if k not in adj:
            adj[k] = float("nan")
    return adj
