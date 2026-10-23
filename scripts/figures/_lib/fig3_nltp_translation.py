#!/usr/bin/env python3
"""Link Vertex tissue LF scoreboard regions to NLTP-vs-cost behavior.

Outputs:
  - scoreboard_nltp_region_translation.png
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
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from prism_vertex.benchmark.comparison_paths import results_dir, fig4_comparison_dir
from prism_vertex.simulator.scoreboard_analysis import LF_ORDER
from scripts.figures._lib.fig4_feasibility_grid import (
    METHOD_LABELS,
    _hf_nltp_at_max_budget,
    _prepare_overall_with_cost,
    _safe_set_log_xscale,
    _to_cost_map,
)
from scripts.figures._lib.lf_selection_atlas import (
    DEN_LOG_FORMULA,
    NUM_FORMULA,
    SHORT,
    _load_features,
)
from scripts.figures._lib.panels_fig2 import SHORT as FIG3_LF_LABELS
from scripts.figures._lib.fig3_style import (
    MANUSCRIPT_REGION_LINE_COLOR,
    MANUSCRIPT_REGION_LINE_LW,
    MANUSCRIPT_CURVE_MS_T_PANEL_B,
    lf_manuscript_linestyle,
    lf_manuscript_marker,
    plot_manuscript_curve,
    plot_manuscript_regime_scatter,
)

# Unified interpretive regime palette (all panels).
REGIME_EFFICIENCY = "Efficiency-optimal"
REGIME_AFFORDABILITY = "Affordability-bound"
REGIME_FIDELITY = "Fidelity-bound"

REGIME_INCONSISTENT = "Score-inference inconsistent"

from scripts.figures._lib.color_palette import REGION_COLORS  # noqa: E402

REGION_LABEL_SHORT: dict[str, str] = {
    REGIME_EFFICIENCY: "Efficiency-optimal (LF)",
    REGIME_AFFORDABILITY: "Affordability-bound (LF)",
    REGIME_FIDELITY: "Fidelity-bound (LF)",
    REGIME_INCONSISTENT: "Score–inference inconsistent",
}

REGION_BY_LF: dict[str, str] = {
    "lf_T70": REGIME_FIDELITY,
    "lf_T140": REGIME_FIDELITY,
    "lf_T350": REGIME_FIDELITY,
    "lf_DT02": REGIME_AFFORDABILITY,
    "lf_R1": REGIME_AFFORDABILITY,
    "lf_R2": REGIME_AFFORDABILITY,
    "lf_R5": REGIME_AFFORDABILITY,
    "lf_DT05": REGIME_EFFICIENCY,
    "lf_DT10": REGIME_EFFICIENCY,
}

HF_LINE_COLOR = "#334155"
HF_LINE_STYLES: dict[str, str] = {
    "hf_snpe": "--",
    "hf_npe": ":",
    "hf_smc_abc": "-.",
}
SHADE_ALPHA = 0.12

# Panel A — exact partition [xmin, xmax, ymin, ymax].
PANEL_A_XLIM = (-2.5, 2.0)
PANEL_A_YLIM = (-0.95, 1.15)
PANEL_A_RECTS: tuple[tuple[float, float, float, float, str], ...] = (
    (-2.5, -0.15, 0.75, 1.15, REGIME_EFFICIENCY),
    (-2.5, -0.15, -0.95, 0.75, REGIME_FIDELITY),
    (-0.15, 2.0, -0.95, 1.15, REGIME_AFFORDABILITY),
)

# Panel B — horizontal bands aligned to HF reference lines.
PANEL_B_XLIM = (450.0, 1050.0)
PANEL_B_YLIM = (-9.2, 2.2)
PANEL_B_RECTS: tuple[tuple[float, float, float, float, str], ...] = (
    (450.0, 1050.0, -9.2, -6.5, REGIME_EFFICIENCY),
    (450.0, 1050.0, -6.5, -5.2, REGIME_AFFORDABILITY),
    (450.0, 1050.0, -5.2, 2.2, REGIME_FIDELITY),
)

# Panel C — gray inconsistent zones + clean regime geometry.
PANEL_C_XLIM = (-19.0, 6.0)
PANEL_C_YLIM = (-9.5, -0.5)
PANEL_C_GRAY_RECTS: tuple[tuple[float, float, float, float], ...] = (
    (2.0, 6.0, -6.6, -0.5),
    (-19.0, 2.0, -9.5, -6.6),
)
PANEL_C_YELLOW_RECT = (-5.0, 2.0, -6.6, -5.0)
PANEL_C_BLUE_RECT = (2.0, 6.0, -9.5, -6.6)
PANEL_C_FIDELITY_POLY: tuple[tuple[float, float], ...] = (
    (-19.0, -0.5),
    (2.0, -0.5),
    (2.0, -5.0),
    (-5.0, -5.0),
    (-5.0, -6.6),
    (-19.0, -6.6),
)

# Point-label anchor offsets from scatter centre (dx, dy, ha, va); shared by panels A and C.
LF_LABEL_TOP_RIGHT = (0.06, 0.04, "left", "bottom")
LF_LABEL_TOP_LEFT = (-0.14, 0.04, "right", "bottom")
LF_LABEL_BOTTOM_RIGHT = (0.08, -0.10, "left", "top")
LF_LABEL_BOTTOM_LEFT = (-0.08, -0.10, "right", "top")

PANEL_A_LABEL_STYLE: dict[str, tuple[float, float, str, str]] = {
    "lf_R5": LF_LABEL_TOP_LEFT,
    "lf_DT05": LF_LABEL_BOTTOM_RIGHT,
    "lf_DT10": LF_LABEL_TOP_LEFT,
    "lf_T70": LF_LABEL_BOTTOM_RIGHT,
}
PANEL_A_LABEL_DEFAULT = LF_LABEL_TOP_RIGHT
PANEL_A_LABEL_DISTANCE_SCALE = 0.82

PANEL_C_LABEL_STYLE: dict[str, tuple[float, float, str, str]] = {
    "lf_DT10": LF_LABEL_BOTTOM_LEFT,
    "lf_DT02": LF_LABEL_BOTTOM_LEFT,
    "lf_R2": LF_LABEL_BOTTOM_LEFT,
}
PANEL_C_LABEL_DEFAULT = LF_LABEL_TOP_RIGHT

PANEL_B_HF_LABEL_OFFSET = 0.18
LF_LABEL_ZORDER = 8
LF_LABEL_FONTSIZE = 7


def _lf_point_label(lf_id: str, *, standalone: bool) -> str:
    if standalone:
        return FIG3_LF_LABELS.get(lf_id, lf_id)
    return SHORT.get(lf_id, lf_id)


def _annotate_lf_point(
    ax: plt.Axes,
    x: float,
    y: float,
    lf_id: str,
    *,
    standalone: bool,
    style_map: dict[str, tuple[float, float, str, str]],
    default_style: tuple[float, float, str, str],
    color: str,
) -> None:
    label = _lf_point_label(lf_id, standalone=standalone)
    if standalone:
        dx, dy, ha, va = style_map.get(lf_id, default_style)
    else:
        dx, dy, ha, va = 0.05, 0.03, "left", "bottom"
    ax.text(
        x + dx,
        y + dy,
        label,
        fontsize=LF_LABEL_FONTSIZE,
        fontweight="bold",
        color=color,
        ha=ha,
        va=va,
        zorder=LF_LABEL_ZORDER,
    )


def _annotate_panel_a_point(
    ax: plt.Axes, x: float, y: float, lf_id: str, *, standalone: bool
) -> None:
    label = _lf_point_label(lf_id, standalone=standalone)
    if standalone:
        dx, dy, ha, va = PANEL_A_LABEL_STYLE.get(lf_id, PANEL_A_LABEL_DEFAULT)
        dx *= PANEL_A_LABEL_DISTANCE_SCALE
        dy *= PANEL_A_LABEL_DISTANCE_SCALE
    else:
        dx, dy, ha, va = 0.05, 0.03, "left", "bottom"
    ax.text(
        x + dx,
        y + dy,
        label,
        fontsize=LF_LABEL_FONTSIZE,
        fontweight="bold",
        color="0.22",
        ha=ha,
        va=va,
        zorder=LF_LABEL_ZORDER,
    )


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.11,
        1.04,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _stat_box(ax: plt.Axes, text: str, *, x: float = 0.97, y: float = 0.97) -> None:
    box = FancyBboxPatch(
        (x - 0.34, y - 0.11),
        0.33,
        0.10,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="0.78",
        linewidth=0.9,
        alpha=0.92,
        zorder=6,
    )
    ax.add_patch(box)
    ax.text(x - 0.175, y - 0.06, text, transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.15")


def _load_nltp_curves(exp_dir: Path) -> pd.DataFrame:
    comparison_dir = fig4_comparison_dir(exp_dir)
    acc = pd.read_csv(comparison_dir / "accuracy_metrics.csv")
    eff = pd.read_csv(comparison_dir / "efficiency_metrics.csv")
    out = _prepare_overall_with_cost(acc, _to_cost_map(eff))
    out = out[out["metric_name"] == "nltp_exact"].copy()
    out["region"] = out["method"].map(REGION_BY_LF)
    return out[np.isfinite(out["x_cost"])].sort_values(["method", "x_cost"])


def _summarize_best_nltp(feat: pd.DataFrame, nltp_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for lf_id, g in nltp_df.groupby("method"):
        g = g.sort_values("x_cost")
        best = g.loc[g["metric_value_mean"].idxmin()]
        last = g.iloc[-1]
        rows.append(
            {
                "lf_id": lf_id,
                "best_nltp": float(best["metric_value_mean"]),
                "best_cost": float(best["x_cost"]),
                "max_cost_nltp": float(last["metric_value_mean"]),
                "nltp_rebound": float(last["metric_value_mean"] - best["metric_value_mean"]),
                "region": str(REGION_BY_LF[lf_id]),
            }
        )
    return feat.merge(pd.DataFrame(rows), on="lf_id", how="left")


def _add_rect_patch(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    regime: str,
    *,
    alpha: float | None = None,
) -> None:
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=REGION_COLORS[regime],
            edgecolor="none",
            alpha=SHADE_ALPHA if alpha is None else alpha,
            zorder=0,
        )
    )


def _add_rect_shading(
    ax: plt.Axes,
    rects: tuple[tuple[float, float, float, float, str], ...],
) -> None:
    for x0, x1, y0, y1, regime in rects:
        _add_rect_patch(ax, x0, x1, y0, y1, regime)


def _add_panel_c_shading(ax: plt.Axes) -> None:
    for x0, x1, y0, y1 in PANEL_C_GRAY_RECTS:
        _add_rect_patch(ax, x0, x1, y0, y1, REGIME_INCONSISTENT, alpha=SHADE_ALPHA + 0.04)
    ax.add_patch(
        Polygon(
            PANEL_C_FIDELITY_POLY,
            closed=True,
            facecolor=REGION_COLORS[REGIME_FIDELITY],
            edgecolor="none",
            alpha=SHADE_ALPHA,
            zorder=0,
        )
    )
    x0, x1, y0, y1 = PANEL_C_YELLOW_RECT
    _add_rect_patch(ax, x0, x1, y0, y1, REGIME_AFFORDABILITY)
    x0, x1, y0, y1 = PANEL_C_BLUE_RECT
    _add_rect_patch(ax, x0, x1, y0, y1, REGIME_EFFICIENCY)


def _add_panel_c_region_lines(ax: plt.Axes) -> None:
    """Dotted regime separators (same style as panel A)."""
    line_kw = dict(color=MANUSCRIPT_REGION_LINE_COLOR, lw=MANUSCRIPT_REGION_LINE_LW, linestyle=":", zorder=1)
    ax.axvline(2.0, **line_kw)
    ax.axvline(-5.0, **line_kw)
    ax.axhline(-0.5, **line_kw)
    ax.axhline(-5.0, **line_kw)
    ax.axhline(-6.6, **line_kw)


def _panel_scoreboard_regions(ax: plt.Axes, feat: pd.DataFrame, *, standalone: bool = False) -> None:
    ax.set_xlim(*PANEL_A_XLIM)
    ax.set_ylim(*PANEL_A_YLIM)
    _add_rect_shading(ax, PANEL_A_RECTS)
    ax.axhline(0, color=MANUSCRIPT_REGION_LINE_COLOR, lw=MANUSCRIPT_REGION_LINE_LW, zorder=1)
    ax.axvline(-0.15, color=MANUSCRIPT_REGION_LINE_COLOR, lw=MANUSCRIPT_REGION_LINE_LW, linestyle=":", zorder=1)
    ax.axhline(0.75, color=MANUSCRIPT_REGION_LINE_COLOR, lw=MANUSCRIPT_REGION_LINE_LW, linestyle=":", zorder=1)

    for _, row in feat.iterrows():
        lf = str(row["lf_id"])
        regime = REGION_BY_LF[lf]
        plot_manuscript_regime_scatter(
            ax,
            [row["log10_den"]],
            [row["S3_num"]],
            color=REGION_COLORS[regime],
            marker=lf_manuscript_marker(lf),
            zorder=4,
        )
    for _, row in feat.iterrows():
        _annotate_panel_a_point(
            ax, float(row["log10_den"]), float(row["S3_num"]), str(row["lf_id"]), standalone=standalone
        )

    ax.set_xlabel(DEN_LOG_FORMULA, fontsize=8)
    ax.set_ylabel(NUM_FORMULA, fontsize=8)
    if not standalone:
        ax.set_title("Scoreboard regimes", fontsize=10, fontweight="bold", loc="left", pad=8)
        _panel_letter(ax, "A")
        ax.grid(True, color="0.92", lw=0.5, zorder=1)


def _plot_lf_curve_regime(ax: plt.Axes, g: pd.DataFrame, lf_id: str) -> None:
    regime = REGION_BY_LF[lf_id]
    mx = g["x_cost"].to_numpy(dtype=float)
    my = g["metric_value_mean"].to_numpy(dtype=float)
    plot_manuscript_curve(
        ax,
        mx,
        my,
        color=REGION_COLORS[regime],
        linestyle=lf_manuscript_linestyle(lf_id),
        marker=lf_manuscript_marker(lf_id),
        markersize=MANUSCRIPT_CURVE_MS_T_PANEL_B if lf_id.startswith("lf_T") else None,
        zorder=5,
    )


def _panel_nltp_regions(
    ax: plt.Axes, exp_dir: Path, nltp_df: pd.DataFrame, *, obs_id: int, seed: int, standalone: bool = False
) -> None:
    x_lo, x_hi = PANEL_B_XLIM
    _add_rect_shading(ax, PANEL_B_RECTS)

    hf_labels: list[tuple[str, float]] = []
    hf_methods = ("hf_snpe", "hf_npe", "hf_smc_abc")
    for method in hf_methods:
        nltp = _hf_nltp_at_max_budget(exp_dir, method, obs_id, seed)
        linestyle = HF_LINE_STYLES[method]
        ax.hlines(
            nltp,
            x_lo,
            x_hi,
            colors=HF_LINE_COLOR,
            linewidth=2.05,
            linestyles=linestyle,
            alpha=0.95,
            zorder=2,
        )
        hf_labels.append((METHOD_LABELS[method], nltp))

    for lf_id in LF_ORDER:
        g = nltp_df[nltp_df["method"] == lf_id]
        if not g.empty:
            _plot_lf_curve_regime(ax, g.sort_values("x_cost"), lf_id)

    _safe_set_log_xscale(ax, np.array([x_lo, x_hi], dtype=float))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(*PANEL_B_YLIM)
    ax.set_xlabel("Total cost")
    ax.set_ylabel("nltp_exact")
    if not standalone:
        ax.set_title("NLTP budget regimes", fontsize=10, fontweight="bold", loc="left", pad=8)
        _panel_letter(ax, "B")
        ax.grid(alpha=0.25, zorder=1)

    label_offset = PANEL_B_HF_LABEL_OFFSET if standalone else 0.15
    for i, (label, yv) in enumerate(hf_labels):
        y_text = yv + label_offset + (0.0 if standalone else (0.15 if i == 0 else 0.0))
        ax.text(
            0.985,
            y_text,
            label,
            transform=ax.get_yaxis_transform(),
            fontsize=8,
            color=HF_LINE_COLOR,
            va="bottom",
            ha="right",
            clip_on=False,
        )


def _annotate_points(ax: plt.Axes, df: pd.DataFrame, xcol: str, ycol: str, *, standalone: bool = False) -> None:
    for _, row in df.iterrows():
        _annotate_lf_point(
            ax,
            float(row[xcol]),
            float(row[ycol]),
            str(row["lf_id"]),
            standalone=standalone,
            style_map=PANEL_C_LABEL_STYLE,
            default_style=PANEL_C_LABEL_DEFAULT,
            color="0.20",
        )


def _panel_score_vs_best_nltp(
    ax: plt.Axes, summary: pd.DataFrame, *, standalone: bool = False, show_pearson: bool = True
) -> None:
    xs = summary["S3_val"].to_numpy(float)
    ys = summary["best_nltp"].to_numpy(float)
    ax.set_xlim(*PANEL_C_XLIM)
    ax.set_ylim(*PANEL_C_YLIM)
    _add_panel_c_shading(ax)
    _add_panel_c_region_lines(ax)

    for region in (REGIME_EFFICIENCY, REGIME_AFFORDABILITY, REGIME_FIDELITY):
        g = summary[summary["region"] == region]
        for _, row in g.iterrows():
            lf = str(row["lf_id"])
            plot_manuscript_regime_scatter(
                ax,
                [row["S3_val"]],
                [row["best_nltp"]],
                color=REGION_COLORS[region],
                marker=lf_manuscript_marker(lf),
                zorder=4,
            )
    _annotate_points(ax, summary, "S3_val", "best_nltp", standalone=standalone)
    ax.set_xlabel(r"$S_3$ scoreboard score")
    ax.set_ylabel("best nltp_exact over LF budgets")
    if not standalone:
        ax.set_title("Scoreboard predicts best NLTP", fontsize=10, fontweight="bold", loc="left", pad=8)
        _panel_letter(ax, "C")
        ax.grid(alpha=0.25)
    if show_pearson:
        corr = float(np.corrcoef(xs, ys)[0, 1])
        _stat_box(ax, rf"Pearson $r \approx {corr:.2f}$")


def _regime_legend_handles() -> list[Line2D]:
    order = (REGIME_FIDELITY, REGIME_AFFORDABILITY, REGIME_EFFICIENCY)
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=REGION_COLORS[regime],
            markeredgecolor="none",
            markersize=10,
            alpha=0.85,
            label=REGION_LABEL_SHORT[regime],
        )
        for regime in order
    ]


def plot_scoreboard_nltp_region_translation(
    exp_dir: Path,
    out_path: Path,
    *,
    obs_id: int = 0,
    seed: int = 0,
    dpi: int = 220,
) -> Path:
    feat = _load_features(exp_dir)
    nltp_df = _load_nltp_curves(exp_dir)
    summary = _summarize_best_nltp(feat, nltp_df)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.14, wspace=0.30)

    _panel_scoreboard_regions(axes[0], feat)
    _panel_nltp_regions(axes[1], exp_dir, nltp_df, obs_id=obs_id, seed=seed)
    _panel_score_vs_best_nltp(axes[2], summary)

    fig.suptitle(
        "From scoreboard regimes to inference regimes: Vertex tissue LF selection vs NLTP budget behavior",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.legend(
        handles=_regime_legend_handles(),
        loc="upper center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.90),
    )
    fig.text(
        0.5,
        0.03,
        "Shaded bands are interpretive operating regimes (not strict boundaries). "
        "Gray = score–inference inconsistent (unreachable under the selection story).",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--obs_id", type=int, default=0)
    ap.add_argument("--seed_id", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exp_dir = results_dir(args.results_root)
    out = args.out or (
        fig4_comparison_dir(exp_dir) / "plots" / "max_budget_comparison" / "scoreboard_nltp_region_translation.png"
    )
    p = plot_scoreboard_nltp_region_translation(
        exp_dir,
        out,
        obs_id=args.obs_id,
        seed=args.seed_id,
        dpi=args.dpi,
    )
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
