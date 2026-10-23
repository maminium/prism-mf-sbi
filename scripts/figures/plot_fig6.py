#!/usr/bin/env python3
"""Paper Figure 6 — non-proliferating PRISM scoreboard (Mo 8 inch layout).

  (A) Scoreboard regimes
  (B) Scoreboard predicts best NLTP
  (C)–(E) Max-budget NRMSE, CRPS, mean posterior STD
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scripts.figures._lib.color_palette import HF_COLORS, METHOD_COLORS, REGION_COLORS
from scripts.figures._lib.fig3_style import (
    MANUSCRIPT_REGION_LINE_COLOR,
    MANUSCRIPT_REGION_LINE_LW,
    lf_manuscript_marker,
    plot_manuscript_regime_scatter,
    style_axes_like_fig4,
)
from scripts.figures._lib.output_formats import (
    manuscript_fig_stem,
    save_matplotlib_manuscript_png,
    save_manuscript_submission_formats,
)
from scripts.figures._lib.panels_fig2 import (
    DPI,
    FIG_PAD_IN,
    _add_axes_inches,
    _fig_height_in,
    apply_rcparams,
)
from scripts.figures._lib.panels_fig3 import (
    FIG4_ABC_HEIGHT_SCALE,
    FIG4_ABC_WIDTH_SCALE,
    FIG4_BOTTOM_ROW_GAP_IN,
    FIG4_LEGEND_PAD_IN,
    FIG4_LEGEND_REGION_H_IN,
    FIG4_TOP_ROW_GAP_IN,
    FIG4_WIDTH_SCALE,
    _fig4_axes_geometry,
    _fig4_row_title_y_frac,
)

DATA_ROOT = PROJECT_ROOT / "data" / "nonproliferating_vertex"
FEATURES = DATA_ROOT / "lf_selection" / "lf_selection_features_out5d.csv"
METRICS = DATA_ROOT / "results" / "metrics" / "runs_metrics.csv"

REGIME_ORDER = ("Efficiency-optimal", "Fidelity-bound", "Affordability-bound")
HF_SERIES = ("hf_snpe", "hf_npe", "hf_smc_abc")
HF_LABELS = {
    "hf_snpe": "HF-SNPE",
    "hf_npe": "HF-NPE",
    "hf_smc_abc": "HF-SMC-ABC",
}
SHORT = {
    "lf_T05": r"$T_{50\%}$",
    "lf_T025": r"$T_{25\%}$",
    "lf_T0125": r"$T_{12\%}$",
    "lf_DT2": r"$\Delta T_{2}$",
    "lf_DT5": r"$\Delta T_{5}$",
    "lf_DT10": r"$\Delta T_{10}$",
    "lf_MESH4": r"$M_4$",
    "lf_MESH8": r"$M_8$",
    "lf_HONEY": "Honey",
    "lf_VORO_RELAX5": r"Voro$+5$",
    "lf_R1": r"$K_1$",
    "lf_R2": r"$K_2$",
    "lf_R5": r"$K_5$",
}
TOP_TITLES = {
    "a": r"$\mathbf{(A)}$ Scoreboard regimes",
    "b": r"$\mathbf{(B)}$ Scoreboard predicts best NLTP",
}
METRIC_SPECS = (
    ("nrmse", r"$\mathbf{(C)}$ NRMSE"),
    ("crps", r"$\mathbf{(D)}$ CRPS"),
    ("mean_std", r"$\mathbf{(E)}$ Mean posterior STD"),
)
NUM_FORMULA = r"$R^2\,\rho_{\min}/(1+\mathrm{MMD}_{\mathrm{norm}})$"


def _load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES)
    if "regime" in df.columns:
        df["regime"] = df["regime"].astype(str)
    if "log10_den" not in df.columns and "cost_ratio" in df.columns:
        df = df.copy()
        df["log10_den"] = np.log10(np.clip(df["cost_ratio"].to_numpy(float), 1e-12, None))
    if "S3_num" not in df.columns and "PRISM" in df.columns and "cost_ratio" in df.columns:
        # Recover numerator ≈ PRISM * c_j when only score is shipped
        df = df.copy()
        df["S3_num"] = df["PRISM"].to_numpy(float) * df["cost_ratio"].to_numpy(float)
    return df


def _load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS)
    if "nltp" not in df.columns and "nltp_kde" in df.columns:
        df = df.copy()
        df["nltp"] = df["nltp_kde"]
    return df


def _best_nltp_by_lf(metrics: pd.DataFrame) -> pd.DataFrame:
    mf = metrics[metrics["method"] == "mf_sbi"].copy()
    g = (
        mf.groupby(["lf_id", "budget_lf", "budget_hf"], as_index=False)["nltp"]
        .agg(nltp_mean="mean", nltp_std="std")
    )
    idx = g.groupby("lf_id")["nltp_mean"].idxmin()
    return g.loc[idx].reset_index(drop=True)


def _panel_a(ax: plt.Axes, feat: pd.DataFrame) -> None:
    xs = feat["log10_den"].to_numpy(float)
    ys = feat["S3_num"].to_numpy(float)
    x_pad = 0.12 * max(float(np.nanmax(xs) - np.nanmin(xs)), 0.5)
    y_pad = 0.10 * max(float(np.nanmax(ys) - np.nanmin(ys)), 0.3)
    x_lo = float(np.nanmin(xs) - x_pad)
    x_hi = float(np.nanmax(xs) + x_pad)
    y_lo = float(min(0.0, np.nanmin(ys)) - y_pad)
    y_hi = float(np.nanmax(ys) + y_pad)
    ax.axhline(0.0, color=MANUSCRIPT_REGION_LINE_COLOR, lw=MANUSCRIPT_REGION_LINE_LW, zorder=1)
    for _, row in feat.iterrows():
        lf = str(row["lf_id"])
        regime = str(row.get("regime", "Fidelity-bound"))
        color = REGION_COLORS.get(regime, "#777777")
        plot_manuscript_regime_scatter(
            ax,
            [row["log10_den"]],
            [row["S3_num"]],
            color=color,
            marker=lf_manuscript_marker(lf),
            zorder=4,
        )
        ax.annotate(
            SHORT.get(lf, lf),
            xy=(float(row["log10_den"]), float(row["S3_num"])),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color="0.22",
            zorder=8,
            clip_on=True,
        )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"$\log_{10} c_j$", fontsize=8)
    ax.set_ylabel(NUM_FORMULA, fontsize=8)
    style_axes_like_fig4(ax)


def _panel_b(ax: plt.Axes, feat: pd.DataFrame, metrics: pd.DataFrame) -> None:
    best = _best_nltp_by_lf(metrics)
    score_col = "PRISM" if "PRISM" in feat.columns else "S3_val"
    merged = feat.merge(best, on="lf_id", how="inner")
    for _, row in merged.iterrows():
        lf = str(row["lf_id"])
        regime = str(row.get("regime", "Fidelity-bound"))
        color = REGION_COLORS.get(regime, "#777777")
        x = float(row[score_col])
        y = float(row["nltp_mean"])
        yerr = float(row["nltp_std"]) if np.isfinite(row["nltp_std"]) else 0.0
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=lf_manuscript_marker(lf),
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            SHORT.get(lf, lf),
            xy=(x, y),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color="0.22",
            zorder=8,
            clip_on=True,
        )
    ax.set_xlabel("PRISM score", fontsize=8)
    ax.set_ylabel(r"Best NLTP over LF budgets (mean$\pm$SD)", fontsize=8)
    style_axes_like_fig4(ax)


def _max_budget_rows(metrics: pd.DataFrame, feat: pd.DataFrame, col: str) -> list[dict]:
    regime_of = feat.set_index("lf_id")["regime"].astype(str).to_dict() if "regime" in feat.columns else {}
    rows: list[dict] = []
    mf = metrics[metrics["method"] == "mf_sbi"]
    for lf_id, g in mf.groupby("lf_id"):
        g2 = g[g["budget_hf"] == 500] if "budget_hf" in g.columns else g
        if g2.empty:
            continue
        bmax = g2["budget_lf"].max()
        sub = g2[g2["budget_lf"] == bmax]
        rows.append(
            {
                "id": str(lf_id),
                "mean": float(sub[col].mean()),
                "std": float(sub[col].std(ddof=0)),
                "color": REGION_COLORS.get(regime_of.get(str(lf_id), ""), "#777777"),
            }
        )
    for method in HF_SERIES:
        sub = metrics[metrics["method"] == method]
        if "budget_hf" in sub.columns:
            sub = sub[sub["budget_hf"] == 1000]
        if sub.empty or col not in sub.columns:
            continue
        rows.append(
            {
                "id": method,
                "mean": float(sub[col].mean()),
                "std": float(sub[col].std(ddof=0)),
                "color": METHOD_COLORS.get(method, HF_COLORS.get(method, "#888888")),
            }
        )
    return rows


def _draw_metric_bars(ax: plt.Axes, metrics: pd.DataFrame, feat: pd.DataFrame, col: str) -> None:
    rows = _max_budget_rows(metrics, feat, col)
    if not rows:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    x = np.arange(len(rows))
    ax.bar(
        x,
        [r["mean"] for r in rows],
        yerr=[r["std"] for r in rows],
        color=[r["color"] for r in rows],
        edgecolor="0.25",
        linewidth=0.6,
        width=0.72,
        capsize=2,
    )
    labels = []
    for r in rows:
        i = r["id"]
        if i.startswith("hf_"):
            labels.append(HF_LABELS.get(i, i))
        else:
            labels.append(SHORT.get(i, i))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    style_axes_like_fig4(ax)


def _draw_legends(fig: plt.Figure, *, fig_h: float, region_y: float) -> None:
    handles = [
        Patch(facecolor=REGION_COLORS[lab], edgecolor="0.25", label=lab) for lab in REGIME_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, region_y / fig_h),
    )


def plot_fig6(*, dpi: int = DPI) -> tuple[Path, Path]:
    apply_rcparams()
    feat = _load_features()
    metrics = _load_metrics()

    panel_h_de = _fig_height_in()
    panel_w_de = 2.35 * FIG4_WIDTH_SCALE
    panel_w_abc = panel_w_de * FIG4_ABC_WIDTH_SCALE
    panel_h_abc = panel_h_de * FIG4_ABC_HEIGHT_SCALE

    n_bot = len(METRIC_SPECS)
    bot_content_w = n_bot * panel_w_de + max(0, n_bot - 1) * FIG4_BOTTOM_ROW_GAP_IN
    top_content_w = 2 * panel_w_abc + FIG4_TOP_ROW_GAP_IN
    content_w = max(bot_content_w, top_content_w)
    bot_x0 = FIG_PAD_IN + 0.5 * (content_w - bot_content_w)
    top_x0 = FIG_PAD_IN + 0.5 * (content_w - top_content_w)

    fig_w = content_w + 2 * FIG_PAD_IN
    bar_xlabel_pad_in = 0.78
    bot_y0 = FIG_PAD_IN + bar_xlabel_pad_in
    region_legend_y0 = bot_y0 + panel_h_de + FIG4_LEGEND_PAD_IN
    top_y0 = region_legend_y0 + FIG4_LEGEND_REGION_H_IN + FIG4_LEGEND_PAD_IN
    fig_h = top_y0 + panel_h_abc + FIG_PAD_IN

    grid_y0_abc, grid_h_abc = _fig4_axes_geometry(panel_h_abc)
    grid_y0_de, grid_h_de = _fig4_axes_geometry(panel_h_de)
    ax_w_abc = panel_w_abc - 2 * FIG_PAD_IN
    ax_w_de = panel_w_de - 2 * FIG_PAD_IN

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    top_panels = (
        ("a", lambda ax: _panel_a(ax, feat)),
        ("b", lambda ax: _panel_b(ax, feat, metrics)),
    )
    for i, (panel, draw) in enumerate(top_panels):
        slot_x = top_x0 + i * (panel_w_abc + FIG4_TOP_ROW_GAP_IN)
        ax = _add_axes_inches(fig, slot_x + FIG_PAD_IN, top_y0 + grid_y0_abc, ax_w_abc, grid_h_abc)
        draw(ax)
        fig.text(
            (slot_x + 0.5 * panel_w_abc) / fig_w,
            _fig4_row_title_y_frac(top_y0, panel_h_abc, fig_h),
            TOP_TITLES[panel],
            ha="center",
            va="bottom",
            fontsize=11,
            color="0.12",
        )

    for i, (metric_key, title) in enumerate(METRIC_SPECS):
        slot_x = bot_x0 + i * (panel_w_de + FIG4_BOTTOM_ROW_GAP_IN)
        ax = _add_axes_inches(fig, slot_x + FIG_PAD_IN, bot_y0 + grid_y0_de, ax_w_de, grid_h_de)
        _draw_metric_bars(ax, metrics, feat, metric_key)
        fig.text(
            (slot_x + 0.5 * panel_w_de) / fig_w,
            _fig4_row_title_y_frac(bot_y0, panel_h_de, fig_h),
            title,
            ha="center",
            va="bottom",
            fontsize=11,
            color="0.12",
        )

    _draw_legends(fig, fig_h=fig_h, region_y=region_legend_y0 + 0.5 * FIG4_LEGEND_REGION_H_IN)

    stem = manuscript_fig_stem(PROJECT_ROOT / "data" / "proliferating_vertex", 6)
    png = stem.with_suffix(".png")
    save_matplotlib_manuscript_png(fig, png, dpi=dpi, save_kwargs=dict(bbox_inches="tight", pad_inches=0.02))
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.02)
    save_manuscript_submission_formats(fig, stem, dpi=dpi)
    plt.close(fig)
    return png, stem.with_suffix(".pdf")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=DPI)
    args = ap.parse_args(argv)
    png, pdf = plot_fig6(dpi=args.dpi)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
