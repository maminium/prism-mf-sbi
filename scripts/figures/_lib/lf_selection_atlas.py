#!/usr/bin/env python3
"""LF selection atlas v2 — three panels (A–C).

  (A) S₃ scatter on top row
  (B) S₃ rank and (C) affordability on bottom row

Example:
  python -m scripts.figures._lib.lf_selection_atlas
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from prism_vertex.benchmark.comparison_paths import results_dir, fig4_comparison_dir
from prism_vertex.benchmark.lf_selection_features import (
    build_extended_features,
    load_cost_json,
)
from prism_vertex.simulator.scoreboard_analysis import (
    FIDELITY_COLORS,
    LF_ORDER,
)

SHORT: dict[str, str] = {
    "lf_T70": "T70",
    "lf_T140": "T140",
    "lf_T350": "T350",
    "lf_DT02": r"$\Delta T$0.2",
    "lf_DT05": r"$\Delta T$0.5",
    "lf_DT10": r"$\Delta T$1.0",
    "lf_R1": "R1",
    "lf_R2": "R2",
    "lf_R5": "R5",
}

AXIS_MARKERS = {"T": "o", "DT": "s", "R": "^"}
N_LF_CAP_LINE = 10_000
COST_CAP_HF = 1000.0
HF_LABELS = 500.0

NUM_FORMULA = (
    r"$\tilde{n}_j=\dfrac{R^2_A\,\rho_{\min}}{1+\mathrm{MMD}_{\mathrm{norm}}}$"
    r"$\;\;(\times{-1}\;\mathrm{if}\;R^2_A{<}0\;\vee\;\rho_{\min}{<}0)$"
)
DEN_LOG_FORMULA = (
    r"$\log_{10}\!\left[c_j(1+\|\mathrm{Bias}\|^2)\max\!\left(1,\dfrac{c_j}{0.3}\right)\right]$"
)
S3_FORMULA = (
    r"$S_3(j)=\dfrac{R^2_A\,\rho_{\min}/(1+\mathrm{MMD}_{\mathrm{norm}})}"
    r"{c_j(1+\|\mathrm{Bias}\|^2)\max(1,\,c_j/0.3)}$"
)


def _signed_numerator(r2: float, rho_min: float, inv_mmd: float) -> float:
    raw = r2 * rho_min * inv_mmd
    if not np.isfinite(raw):
        return float("nan")
    if r2 < 0 or rho_min < 0:
        return -abs(raw)
    return float(raw)


def _s3_parts(row: pd.Series) -> tuple[float, float, float]:
    c = float(row["cost_ratio"])
    r2 = float(row["R2_A"])
    rho_min = float(row["rho_min"])
    mmd_norm = float(row["mmd_norm"])
    bias = float(row["bias_norm"])
    if c <= 0 or not np.isfinite(rho_min) or not np.isfinite(mmd_norm):
        return float("nan"), float("nan"), float("nan")
    inv_mmd = 1.0 / (1.0 + mmd_norm)
    num = _signed_numerator(r2, rho_min, inv_mmd)
    den = c * (1.0 + bias**2) * max(1.0, c / 0.30)
    s3 = num / den if den > 0 else float("nan")
    return float(num), float(den), float(s3)


def _attach_s3(feat: pd.DataFrame) -> pd.DataFrame:
    parts = feat.apply(lambda r: pd.Series(_s3_parts(r), index=["S3_num", "S3_den", "S3_val"]), axis=1)
    out = feat.copy()
    out[["S3_num", "S3_den", "S3_val"]] = parts
    out["log10_den"] = np.log10(out["S3_den"].clip(lower=1e-12))
    out["log10_cost"] = np.log10(out["cost_ratio"].clip(lower=1e-12))
    return out


def _load_features(exp_dir: Path) -> pd.DataFrame:
    analysis = exp_dir / "lf_selection" / "analysis"
    cached = analysis / "lf_selection_features_and_truth.csv"
    if cached.is_file():
        feat = pd.read_csv(cached)
    else:
        sb = pd.read_csv(analysis / "scoreboard_table.csv")
        cells = pd.read_csv(exp_dir / "lf_selection" / "cells.csv")
        cost = load_cost_json(analysis)
        feat = build_extended_features(sb, cells, cost)
    order = {lf: i for i, lf in enumerate(LF_ORDER)}
    return _attach_s3(feat.sort_values("lf_id", key=lambda s: s.map(order)))


def _lf_axis(lf_id: str) -> str:
    if lf_id.startswith("lf_T"):
        return "T"
    if lf_id.startswith("lf_DT"):
        return "DT"
    if lf_id.startswith("lf_R"):
        return "R"
    return "other"


def _lf_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=AXIS_MARKERS.get(_lf_axis(lf), "o"),
            color="w",
            markerfacecolor=FIDELITY_COLORS[lf],
            markeredgecolor="0.15",
            markeredgewidth=0.7,
            markersize=7,
            label=SHORT[lf],
        )
        for lf in LF_ORDER
    ]


def _scatter_lf(ax, feat: pd.DataFrame, *, xkey: str, ykey: str, size: float = 55) -> None:
    for _, row in feat.iterrows():
        lf = row["lf_id"]
        ax.scatter(
            row[xkey],
            row[ykey],
            s=size,
            c=FIDELITY_COLORS[lf],
            marker=AXIS_MARKERS.get(row["lf_axis"], "o"),
            edgecolors="0.12",
            linewidths=0.7,
            zorder=3,
            alpha=0.95,
        )


def _smooth01(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _add_panel_a_zone_shading(
    ax: plt.Axes,
    feat: pd.DataFrame,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> None:
    """Box-aligned fades: blue dark below, lightens upward; gray dark right, lightens leftward."""
    dt = feat.loc[feat["lf_id"] == "lf_DT05"].iloc[0]
    x_ref = float(dt["log10_den"])
    y_ref = float(dt["S3_num"])

    nx, ny = 420, 420
    X, Y = np.meshgrid(np.linspace(x_lo, x_hi, nx), np.linspace(y_lo, y_hi, ny))
    extent = (x_lo, x_hi, y_lo, y_hi)

    # Blue box: darkest at bottom; arrow ↑ → gradual lighten and vanish toward ΔT0.5 height.
    blue_top = y_ref + 0.04
    blue_span = max(blue_top - y_lo, 0.2)
    t_blue = (blue_top - Y) / blue_span
    blue_strength = _smooth01(t_blue)
    blue_a = np.clip(blue_strength * 0.56, 0, 0.56)
    blue_rgb = np.dstack(
        [
            0.86 - 0.46 * blue_strength,
            0.94 - 0.36 * blue_strength,
            0.99 - 0.12 * blue_strength,
            blue_a,
        ]
    )
    ax.imshow(blue_rgb, extent=extent, origin="lower", aspect="auto", interpolation="bilinear", zorder=0)

    # Gray box: darkest at right; arrow ← → gradual lighten and vanish toward ΔT0.5 x.
    gray_left = x_ref
    gray_span = max(x_hi - gray_left, 0.25)
    t_gray_x = (X - gray_left) / gray_span
    t_gray_y = (Y - 0.0) / max(y_hi, 0.2)
    gray_strength = _smooth01(t_gray_x) * _smooth01(t_gray_y)
    gray_a = np.clip(gray_strength * 0.44, 0, 0.44)
    gray_rgb = np.dstack(
        [
            0.72 - 0.22 * gray_strength,
            0.72 - 0.22 * gray_strength,
            0.72 - 0.22 * gray_strength,
            gray_a,
        ]
    )
    ax.imshow(gray_rgb, extent=extent, origin="lower", aspect="auto", interpolation="bilinear", zorder=0)


def _panel_s3_scatter(ax: plt.Axes, feat: pd.DataFrame) -> None:
    x_lo = float(feat["log10_den"].min()) - 0.12
    x_hi = float(feat["log10_den"].max()) + 0.12
    nums = feat["S3_num"].to_numpy(dtype=float)
    y_lo, y_hi = float(np.nanmin(nums)), float(np.nanmax(nums))
    pad = 0.1 * max(y_hi - y_lo, 0.15)
    y_lo -= pad
    y_hi += pad

    _add_panel_a_zone_shading(ax, feat, x_lo, x_hi, y_lo, y_hi)
    ax.axhline(0, color="0.35", lw=0.6, zorder=1)
    _scatter_lf(ax, feat, xkey="log10_den", ykey="S3_num", size=58)

    ax.set_xlabel(DEN_LOG_FORMULA, fontsize=8)
    ax.set_ylabel(NUM_FORMULA, fontsize=8)
    ax.set_title(r"(A) $S_3$ decomposition", fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, color="0.92", lw=0.5, zorder=1)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)


def _panel_s3_rank(ax: plt.Axes, feat: pd.DataFrame) -> None:
    s3_arr = feat["S3_val"].to_numpy(dtype=float)
    order_idx = np.argsort(-np.nan_to_num(s3_arr, nan=-np.inf))
    lf_sorted = [feat.iloc[i]["lf_id"] for i in order_idx]
    vals = s3_arr[order_idx]
    x = np.arange(len(feat))
    colors = [FIDELITY_COLORS[lf] for lf in lf_sorted]

    ax.bar(x, vals, color=colors, edgecolor="0.2", width=0.72, linewidth=0.5, zorder=2)
    ax.axhline(0, color="0.35", lw=0.8, zorder=1)
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            continue
        va = "bottom" if v >= 0 else "top"
        off = 0.15 if v >= 0 else -0.15
        ax.text(i, v + off, f"{v:.2f}", ha="center", va=va, fontsize=7, color="0.15")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[lf] for lf in lf_sorted], fontsize=8, rotation=35, ha="right")
    ax.set_ylabel(S3_FORMULA, fontsize=8)
    ax.set_title(r"(B) $S_3$ rank", fontsize=10, fontweight="bold", loc="left")
    y_lo = float(np.nanmin(vals[np.isfinite(vals)]))
    y_hi = float(np.nanmax(vals[np.isfinite(vals)]))
    pad = 0.12 * max(y_hi - y_lo, 1.0)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.grid(axis="y", color="0.92", lw=0.5)


def _panel_n_lf_max(ax: plt.Axes, feat: pd.DataFrame) -> None:
    n_raw = feat["n_afford_cap"].to_numpy(dtype=float)
    n_plot = np.minimum(n_raw, N_LF_CAP_LINE)
    truncated = n_raw > N_LF_CAP_LINE
    x = np.arange(len(feat))
    colors = [FIDELITY_COLORS[lf] for lf in feat["lf_id"]]

    bars = ax.bar(x, n_plot, color=colors, edgecolor="0.25", linewidth=0.6, width=0.72, zorder=2)
    ax.axhline(N_LF_CAP_LINE, color="#a63603", ls="--", lw=1.2, zorder=3, label=rf"cap ${N_LF_CAP_LINE:,}$")

    for i, (raw, show, bar) in enumerate(zip(n_raw, n_plot, bars)):
        if truncated[i]:
            bar.set_hatch("///")
            ax.text(i, show * 1.04, rf"$\geq${raw:,.0f}", ha="center", va="bottom", fontsize=6, color="0.4", rotation=90)

    ax.set_yscale("log")
    ax.set_ylim(400, N_LF_CAP_LINE * 2.2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[lf] for lf in feat["lf_id"]], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(r"$N_{LF}^{\max}=\lfloor(1000{-}500)/c_j\rfloor$", fontsize=9)
    ax.set_title(
        rf"(C) Affordability (cap ${COST_CAP_HF:.0f}$, ${HF_LABELS:.0f}$ HF labels)",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    ax.grid(axis="y", color="0.92", lw=0.5, which="both")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.95)


def plot_lf_selection_atlas_v2(
    exp_dir: Path,
    out_path: Path,
    *,
    dpi: int = 220,
) -> Path:
    feat = _load_features(exp_dir)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
        }
    )

    fig = plt.figure(figsize=(14.0, 8.5), facecolor="white")

    gs = GridSpec(
        2,
        2,
        figure=fig,
        left=0.07,
        right=0.98,
        top=0.88,
        bottom=0.10,
        height_ratios=[1.05, 1.0],
        hspace=0.32,
        wspace=0.28,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    _panel_s3_scatter(ax_a, feat)
    _panel_s3_rank(ax_b, feat)
    _panel_n_lf_max(ax_c, feat)

    fig.suptitle(
        r"LF scoreboard atlas — signed $S_3$ rank and cost ceiling",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    fig.legend(
        handles=_lf_legend_handles(),
        loc="upper center",
        ncol=9,
        fontsize=7.5,
        framealpha=0.95,
        title="LF candidates (colour + marker)",
        bbox_to_anchor=(0.5, 0.93),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exp_dir = results_dir(args.results_root)
    out = args.out or (
        fig4_comparison_dir(exp_dir) / "plots" / "max_budget_comparison" / "lf_selection_decision_atlas_v2.png"
    )
    p = plot_lf_selection_atlas_v2(exp_dir, out, dpi=args.dpi)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
