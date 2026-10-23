"""Scoreboard-derived features for Vertex tissue pre-inference LF / budget selection."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prism_vertex.simulator.scoreboard_analysis import (
    LF_ORDER,
    S_COLS,
    hf_mean_and_std,
    pca_2d,
    rbf_mmd2,
    tensor_for_fidelity,
)

HF_LABELS = 500
COST_CAP_HF_EQUIV = 1000.0
NOMINAL_BUDGETS = (500, 1000, 5000, 10000)
R_COST_VETO = 0.85
R_AFFORD_VETO = 2500


def _theta0_stats(cells: pd.DataFrame, lf_id: str) -> dict[str, float]:
    s_cols = [c for c in cells.columns if re.fullmatch(r"s\d+", c)]
    sub = cells[(cells["fidelity_id"] == lf_id) & (cells["theta_index"] == 0) & (cells["status"] == "ok")]
    if sub.empty or not s_cols:
        return {}
    hf = cells[(cells["fidelity_id"] == "hf_single") & (cells["theta_index"] == 0) & (cells["status"] == "ok")]
    if hf.empty:
        return {}
    lf_s = sub[s_cols].to_numpy(dtype=float)
    hf_m = hf[s_cols].to_numpy(dtype=float).mean(axis=0)
    err = lf_s - hf_m.reshape(1, -1)
    return {
        "rmse_theta0": float(np.sqrt(np.mean(err**2))),
        "bias_theta0": float(np.linalg.norm(err.mean(axis=0))),
        "var_theta0": float(np.mean(np.var(lf_s, axis=0))),
    }


def _level_a_arrays(cells: pd.DataFrame, lf_id: str, n_theta: int) -> tuple[np.ndarray, np.ndarray]:
    hf_mean, _, hf_count = hf_mean_and_std(cells, n_theta)
    lf, lf_ok = tensor_for_fidelity(cells, lf_id, n_theta)
    lf_list, ref_list = [], []
    for i in range(n_theta):
        if hf_count[i] < 1:
            continue
        for k in range(lf.shape[1]):
            if not lf_ok[i, k]:
                continue
            lf_list.append(lf[i, k])
            ref_list.append(hf_mean[i])
    if not lf_list:
        return np.empty((0, len(S_COLS))), np.empty((0, len(S_COLS)))
    return np.stack(lf_list), np.stack(ref_list)


def _per_dim_rho(lf_arr: np.ndarray, ref_arr: np.ndarray) -> np.ndarray:
    rhos = []
    for j in range(lf_arr.shape[1]):
        if lf_arr.shape[0] < 3:
            rhos.append(float("nan"))
            continue
        rhos.append(float(np.corrcoef(lf_arr[:, j], ref_arr[:, j])[0, 1]))
    return np.asarray(rhos, dtype=float)


def _pca_mean_distance(lf_arr: np.ndarray, ref_arr: np.ndarray) -> float:
    if lf_arr.shape[0] < 3:
        return float("nan")
    pool = np.vstack([ref_arr, lf_arr])
    scores, _ = pca_2d(pool)
    n_ref = ref_arr.shape[0]
    ref_c = scores[:n_ref].mean(axis=0)
    lf_c = scores[n_ref:].mean(axis=0)
    return float(np.linalg.norm(lf_c - ref_c))


def _noise_fraction(lf_arr: np.ndarray, ref_arr: np.ndarray) -> float:
    if lf_arr.shape[0] < 2:
        return float("nan")
    resid = lf_arr - ref_arr
    bias_vec = resid.mean(axis=0)
    var_res = np.var(resid, axis=0, ddof=1).mean()
    bias_sq = float(np.sum(bias_vec**2))
    denom = bias_sq + var_res
    return float(var_res / denom) if denom > 0 else float("nan")


def _lf_axis(lf_id: str) -> str:
    if lf_id.startswith("lf_T"):
        return "T"
    if lf_id.startswith("lf_DT"):
        return "DT"
    if lf_id.startswith("lf_R"):
        return "R"
    return "other"


def _affordable_nominals(c_lf: float, cap: float = COST_CAP_HF_EQUIV) -> list[int]:
    out = []
    for b in NOMINAL_BUDGETS:
        if HF_LABELS + b * c_lf <= cap + 1e-6:
            out.append(b)
    return out


def _nearest_nominal(n: int) -> int:
    if not NOMINAL_BUDGETS:
        return n
    return min(NOMINAL_BUDGETS, key=lambda b: abs(b - n))


def build_extended_features(
    sb: pd.DataFrame,
    cells: pd.DataFrame,
    cost_json: dict,
    *,
    n_theta: int = 20,
) -> pd.DataFrame:
    c_map = {f["fidelity_id"]: float(f["empirical_ratio_median"]) for f in cost_json["fidelities"]}
    sb_idx = sb.set_index("fidelity_id")
    rows: list[dict] = []

    for lf in LF_ORDER:
        r = sb_idx.loc[lf]
        c = float(r["cost_ratio_vs_hf"])
        c_lf = c_map.get(lf, c)
        r2 = float(r["R2_A"])
        rmse = float(r["RMSE_A_vs_hf_mean"])
        rho = float(r["rho_bar"])
        bias = float(r["bias_norm"])
        fail = float(r["fail_pct"])
        t0 = _theta0_stats(cells, lf)
        lf_arr, ref_arr = _level_a_arrays(cells, lf, n_theta)
        rhos = _per_dim_rho(lf_arr, ref_arr)
        rho_min = float(np.nanmin(rhos)) if rhos.size else float("nan")
        mmd2 = rbf_mmd2(lf_arr, ref_arr) if lf_arr.shape[0] else float("nan")
        ref_scale = float(np.mean(np.sum(ref_arr**2, axis=1))) if ref_arr.shape[0] else 1.0
        mmd_norm = float(mmd2 / (ref_scale + 1e-12)) if np.isfinite(mmd2) else float("nan")
        pca_dist = _pca_mean_distance(lf_arr, ref_arr)
        noise_frac = _noise_fraction(lf_arr, ref_arr)
        n_afford = int((COST_CAP_HF_EQUIV - HF_LABELS) / max(c_lf, 1e-9))
        afford_nominals = _affordable_nominals(c_lf)
        var0 = t0.get("var_theta0", float("nan"))
        rmse0 = t0.get("rmse_theta0", float("nan"))
        bias0 = t0.get("bias_theta0", float("nan"))

        # --- original C1–C10 ---
        n_knee = 10000
        if np.isfinite(noise_frac) and np.isfinite(rmse0) and np.isfinite(var0) and var0 > 0:
            # Noise-limited knee: more noise → more LF calls help; high bias → saturates early.
            scale = 5000.0 * (rmse0**2) / (var0 + 1e-12)
            if bias > 0.006:
                scale = min(scale, 5000.0)
            n_knee = int(np.clip(scale, 500, 10000))
        n_rec = int(min(NOMINAL_BUDGETS[-1], n_afford, n_knee))

        row: dict = {
            "lf_id": lf,
            "lf_axis": _lf_axis(lf),
            "Q": float(r["Q_R2_over_cost"]),
            "R2_A": r2,
            "rho_bar": rho,
            "rho_min": rho_min,
            "RMSE_A": rmse,
            "bias_norm": bias,
            "fail_pct": fail,
            "cost_ratio": c,
            "mmd2": mmd2,
            "mmd_norm": mmd_norm,
            "pca_dist": pca_dist,
            "noise_frac": noise_frac,
            "n_afford_cap": n_afford,
            "afford_nominals": ",".join(str(x) for x in afford_nominals),
            "n_knee_proxy": n_knee,
            "gate_soft": int(rho >= 0.80 and r2 > 0),
            "gate_hard": int(rho >= 0.80 and r2 > 0 and fail <= 10.0),
            "gate_R_veto": int(_lf_axis(lf) == "R" and (c > R_COST_VETO or n_afford < R_AFFORD_VETO)),
            "C1_Q": float(r["Q_R2_over_cost"]),
            "C2_Q_bias": r2 / (c * (1.0 + bias**2)) if c > 0 and r2 > 0 else float("nan"),
            "C3_Q_rho": (rho**2) / c if c > 0 else float("nan"),
            "C4_KOH": r2 / (c * (1.0 + rmse**2)) if c > 0 and r2 > 0 else float("nan"),
            "C5_log_comp": np.log(max(r2, 1e-6)) - np.log(c) - np.log(1.0 + rmse) if c > 0 else float("nan"),
            "C6_info_cost": (rho * r2) / c if c > 0 else float("nan"),
            "C7_neg_rmse_cost": -rmse / c if c > 0 else float("nan"),
            "C8_afford_weighted": r2 * min(1.0, n_afford / 5000.0) / c if c > 0 and r2 > 0 else float("nan"),
            "C9_theta0_precision": (rho**2) / (c * (1.0 + rmse0**2)) if c > 0 and np.isfinite(rmse0) else float("nan"),
            "C10_two_stage": float(r["Q_R2_over_cost"]) if rho >= 0.80 and r2 > 0 else float("nan"),
            **t0,
        }

        # --- new N1–N10 (multi-term, distribution / PCA / axis-aware) ---
        inv_mmd = 1.0 / (1.0 + mmd_norm) if np.isfinite(mmd_norm) else float("nan")
        inv_pca = 1.0 / (1.0 + pca_dist) if np.isfinite(pca_dist) else float("nan")
        cost_pen_exp = c**1.25 if c > 0 else float("nan")

        row["N1_mmd_efficiency"] = (inv_mmd * r2) / c if c > 0 and r2 > 0 else float("nan")
        row["N2_pca_rho_min"] = (inv_pca * rho_min**2) / c if c > 0 and np.isfinite(rho_min) else float("nan")
        row["N3_noise_bias_balance"] = (r2 * noise_frac) / (c * (1.0 + bias**2)) if c > 0 and r2 > 0 else float("nan")
        row["N4_weakest_link"] = (rho_min * r2) / c if c > 0 and r2 > 0 and np.isfinite(rho_min) else float("nan")
        row["N5_superlinear_cost"] = (rho * r2) / cost_pen_exp if c > 0 else float("nan")
        row["N6_R_veto_score"] = float("nan") if row["gate_R_veto"] else row["C4_KOH"]
        row["N7_DT_axis_bonus"] = row["C4_KOH"] * (1.15 if _lf_axis(lf) == "DT" else 1.0) if np.isfinite(row["C4_KOH"]) else float("nan")
        row["N8_mmd_over_rmse"] = (inv_mmd * r2) / (c * (1.0 + rmse)) if c > 0 and r2 > 0 else float("nan")
        row["N9_fail_penalized"] = (rho_min**2 * r2) / (c * (1.0 + fail / 100.0)) if c > 0 and r2 > 0 else float("nan")
        row["N10_mega_product"] = (
            (r2 * rho_min * inv_mmd) / (c * (1.0 + bias**2) * max(1.0, c / 0.30))
            if c > 0 and r2 > 0 and np.isfinite(rho_min)
            else float("nan")
        )

        # --- three-stage LF score (A gate, B rank term, C afford) ---
        row["S3_full"] = row["N10_mega_product"] if row["gate_soft"] and not row["gate_R_veto"] else float("nan")

        # --- budget heuristics (nominal LF count) ---
        b_afford = max(afford_nominals) if afford_nominals else 500
        b_knee = _nearest_nominal(n_knee)
        b_bias_cap = 5000 if bias > 0.006 else 10000
        b_mmd_cap = 5000 if (np.isfinite(mmd_norm) and mmd_norm < 0.02 and r2 > 0.97) else 10000
        b_r_cap = 1000 if c > R_COST_VETO else 10000
        b_t_cap = 1000 if (_lf_axis(lf) == "T" or rho < 0.85) else 10000
        b_composite = min(b_afford, b_knee, b_bias_cap, b_mmd_cap, b_r_cap, b_t_cap)
        b_hybrid = min(b_afford, b_bias_cap, b_mmd_cap, b_r_cap, b_t_cap, max(n_knee, 1000))

        # proxy argmin: bias^2 + var0/N + rmse0^2/N over feasible nominals
        best_proxy, best_b = float("inf"), b_composite
        for b in afford_nominals or [500]:
            if not np.isfinite(var0):
                score = bias**2 + rmse**2 / b
            else:
                score = bias**2 + var0 / b + (rmse0**2 if np.isfinite(rmse0) else rmse**2) / b
            if score < best_proxy:
                best_proxy, best_b = score, b

        row["B1_max_afford"] = b_afford
        row["B2_knee"] = b_knee
        row["B3_bias_cap"] = min(b_afford, b_bias_cap)
        row["B4_mmd_plateau_cap"] = min(b_afford, b_mmd_cap)
        row["B5_R_cap_1k"] = min(b_afford, b_r_cap)
        row["B6_composite"] = b_composite
        row["B7_proxy_argmin"] = best_b
        row["B8_hybrid"] = _nearest_nominal(b_hybrid)
        row["n_rec_budget"] = row["B8_hybrid"]
        rows.append(row)

    return pd.DataFrame(rows)


def load_cost_json(analysis_dir: Path) -> dict:
    return json.loads((analysis_dir / "lf_cost_ratios.json").read_text())
