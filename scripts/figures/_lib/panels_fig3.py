"""Shared style and plotting helpers for Figure 4 panels (a–h)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D

from prism_vertex.benchmark.comparison_paths import fig4_comparison_dir
from scripts.figures._lib.panels_fig2 import (
    DPI,
    FIG_PAD_IN,
    LF_COLORS,
    MAX_X_TICKS,
    SHORT as FIG3_LF_LABELS,
    _add_axes_inches,
    _bar_panel_draw_kwargs,
    _fig_height_in,
    _grid_geometry_inches,
    _measure_bar_width_in,
    _save_figure,
    _title_y_frac_for_row,
    apply_rcparams,
    format_cost_minutes,
    load_cost_minutes,
    load_features,
)
from scripts.figures._lib.fig4_feasibility_grid import (
    FIG4_GRID,
    METHOD_LABELS,
    _collect_metrics,
    _load_posterior_xy,
    _resolve_series_run,
)
from scripts.figures._lib.lf_selection_atlas import _load_features as _load_scoreboard_features
from scripts.figures._lib.fig3_nltp_translation import (
    REGION_COLORS,
    REGION_LABEL_SHORT,
    REGIME_AFFORDABILITY,
    REGIME_EFFICIENCY,
    REGIME_FIDELITY,
    REGIME_INCONSISTENT,
    _load_nltp_curves,
    _panel_nltp_regions,
    _panel_score_vs_best_nltp,
    _panel_scoreboard_regions,
    _summarize_best_nltp,
)
from scripts.figures._lib.color_palette import HF_COLORS

FIG4_OUT_DIR = "fig 4"
FIG4_WIDTH_SCALE = 1.2
FIG4_ABC_WIDTH_SCALE = 1.66
FIG4_ABC_HEIGHT_SCALE = 1.5
FIG4_TOP_ROW_GAP_IN = 0.5
FIG4_BOTTOM_ROW_GAP_IN = 0.4
FIG4_LEGEND_REGION_H_IN = 0.18
FIG4_LEGEND_SERIES_H_IN = 0.40
FIG4_LEGEND_PAD_IN = 0.05

FIG4_REGION_LEGEND_ORDER: tuple[str, ...] = (
    REGIME_EFFICIENCY,
    REGIME_FIDELITY,
    REGIME_AFFORDABILITY,
)
FIG4_DT_SERIES_ORDER: tuple[str, ...] = ("lf_DT10", "lf_DT05", "lf_DT02")
FIG4_SERIES_FLAT: tuple[str, ...] = (
    *FIG4_GRID[0],
    *FIG4_GRID[1],
    *FIG4_DT_SERIES_ORDER,
    *FIG4_GRID[3],
)
FIG4_LEGEND_COLUMNS: tuple[tuple[str, ...], ...] = (
    FIG4_GRID[0],
    FIG4_GRID[1],
    FIG4_DT_SERIES_ORDER,
    FIG4_GRID[3],
)
FIG4_GROUP_SEPARATORS = (2.5, 5.5, 8.5)

FIG4_HF_GRAY_COLORS: dict[str, str] = HF_COLORS

FIG4_METRIC_SPECS: dict[str, tuple[str, str]] = {
    "d": ("nrmse", r"$\mathbf{(D)}$ NRMSE"),
    "e": ("crps", r"$\mathbf{(E)}$ CRPS"),
    "f": ("tarp_error", r"$\mathbf{(F)}$ TARP coverage error"),
    "g": ("mean_std", r"$\mathbf{(G)}$ Mean posterior STD"),
    "h": ("mean_ci_width_95", r"$\mathbf{(H)}$ Mean 95% CI width"),
}

FIG4_ABC_TITLES: dict[str, str] = {
    "a": r"$\mathbf{(A)}$ Scoreboard regimes",
    "b": r"$\mathbf{(B)}$ NLTP budget regimes",
    "c": r"$\mathbf{(C)}$ Scoreboard predicts best NLTP",
}


def fig4_panel_fig_size_in(exp_dir: Path) -> tuple[float, float]:
    """Match standalone Figure 3 panel (e) width and height."""
    feat = load_features(exp_dir)
    feat = feat.assign(cost_min=load_cost_minutes(exp_dir)["cost_min"].to_numpy())
    _, _, row_h = _grid_geometry_inches()
    e_kw = _bar_panel_draw_kwargs("cost_min")
    fig_w = _measure_bar_width_in(
        feat,
        "cost_min",
        signed=False,
        value_fmt=format_cost_minutes,
        outer_pad=True,
        **e_kw,
    )
    return fig_w * FIG4_WIDTH_SCALE, _fig_height_in()


def fig4_abc_panel_fig_size_in(exp_dir: Path) -> tuple[float, float]:
    """Figure 4 panels (a)–(c): 1.66× bar-panel width, 1.5× bar-panel height."""
    bar_w, bar_h = fig4_panel_fig_size_in(exp_dir)
    return bar_w * FIG4_ABC_WIDTH_SCALE, bar_h * FIG4_ABC_HEIGHT_SCALE


def _fig4_axes_geometry(fig_h_in: float) -> tuple[float, float]:
    scale = fig_h_in / _fig_height_in()
    grid_y0_in, grid_h_in, _ = _grid_geometry_inches()
    return grid_y0_in * scale, grid_h_in * scale


def _fig4_abc_title_y_frac(fig_h_in: float) -> float:
    """Title just above the scaled plot area (not inside it)."""
    grid_y0_in, grid_h_in = _fig4_axes_geometry(fig_h_in)
    return (grid_y0_in + grid_h_in + 0.12 + 0.20) / fig_h_in


def load_max_budget_metrics(
    exp_dir: Path,
    *,
    obs_id: int = 0,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    metrics_by_id: dict[str, dict[str, float]] = {}
    for series_id in FIG4_SERIES_FLAT:
        try:
            run_dir, _budget = _resolve_series_run(exp_dir, series_id, obs_id, seed)
            samples = _load_posterior_xy(run_dir / "posterior_samples.csv")
            metrics_by_id[series_id] = _collect_metrics(run_dir, samples)
        except FileNotFoundError:
            continue
    if not metrics_by_id:
        raise FileNotFoundError("No max-budget posteriors found for Figure 4 metric panels.")
    return metrics_by_id


def _series_color(series_id: str) -> str:
    if series_id in FIG4_HF_GRAY_COLORS:
        return FIG4_HF_GRAY_COLORS[series_id]
    return LF_COLORS.get(series_id, "#333333")


def _format_metric_val(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return f"{v:.2f}"


def _draw_fig4_metric_bars_on_axes(
    ax: plt.Axes,
    metrics_by_id: dict[str, dict[str, float]],
    metric_key: str,
    *,
    show_bar_values: bool = False,
) -> None:
    series = list(FIG4_SERIES_FLAT)
    x = np.arange(len(series), dtype=float)
    vals = np.array([metrics_by_id.get(sid, {}).get(metric_key, np.nan) for sid in series], dtype=float)
    colors = [_series_color(sid) for sid in series]

    ax.bar(x, vals, color=colors, width=0.72, zorder=2, align="center")
    for sep in FIG4_GROUP_SEPARATORS:
        ax.axvline(sep, color="0.78", linestyle=":", linewidth=0.8, zorder=0)

    finite = vals[np.isfinite(vals)]
    ymax = float(np.nanmax(finite)) if finite.size else 1.0
    ax.set_ylim(0.0, ymax * 1.18)

    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=MAX_X_TICKS, prune=None))
    ax.set_xlim(-0.6, len(series) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["left"].set_color("0.25")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("0.25")
    ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.tick_params(axis="y", labelsize=9, length=3, width=0.6, pad=2)

    if show_bar_values:
        for xi, v in zip(x, vals):
            if not np.isfinite(v):
                continue
            ax.text(xi, v, _format_metric_val(v), ha="center", va="bottom", fontsize=7, color="0.12", zorder=3)


def plot_fig4_metric_panel(
    metrics_by_id: dict[str, dict[str, float]],
    panel: str,
    *,
    out_stem: Path,
    fig_w_in: float,
    fig_h_in: float,
    dpi: int = DPI,
) -> tuple[Path, Path]:
    metric_key, title = FIG4_METRIC_SPECS[panel]
    apply_rcparams()
    grid_y0_in, grid_h_in = _fig4_axes_geometry(fig_h_in)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")
    ax = _add_axes_inches(fig, FIG_PAD_IN, grid_y0_in, fig_w_in - 2 * FIG_PAD_IN, grid_h_in)
    _draw_fig4_metric_bars_on_axes(ax, metrics_by_id, metric_key)
    fig.text(0.5, _title_y_frac_for_row(0, fig_h_in), title, ha="center", va="bottom", fontsize=11, color="0.12")

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    kw = dict(facecolor="white", bbox_inches="tight", pad_inches=FIG_PAD_IN)
    fig.savefig(png, dpi=dpi, **kw)
    fig.savefig(pdf, **kw)
    plt.close(fig)
    return png, pdf


def plot_fig4_abc_panel(
    exp_dir: Path,
    panel: str,
    *,
    out_stem: Path,
    fig_w_in: float,
    fig_h_in: float,
    obs_id: int = 0,
    seed: int = 0,
    dpi: int = DPI,
) -> tuple[Path, Path]:
    title = FIG4_ABC_TITLES[panel]
    apply_rcparams()
    grid_y0_in, grid_h_in = _fig4_axes_geometry(fig_h_in)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")
    ax = _add_axes_inches(fig, FIG_PAD_IN, grid_y0_in, fig_w_in - 2 * FIG_PAD_IN, grid_h_in)

    if panel == "a":
        feat = _load_scoreboard_features(exp_dir)
        _panel_scoreboard_regions(ax, feat, standalone=True)
    elif panel == "b":
        nltp_df = _load_nltp_curves(exp_dir)
        _panel_nltp_regions(ax, exp_dir, nltp_df, obs_id=obs_id, seed=seed, standalone=True)
    elif panel == "c":
        feat = _load_scoreboard_features(exp_dir)
        nltp_df = _load_nltp_curves(exp_dir)
        summary = _summarize_best_nltp(feat, nltp_df)
        _panel_score_vs_best_nltp(ax, summary, standalone=True, show_pearson=False)
    else:
        raise ValueError(f"Unknown Figure 4 ABC panel: {panel}")

    fig.text(0.5, _fig4_abc_title_y_frac(fig_h_in), title, ha="center", va="bottom", fontsize=11, color="0.12")

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    kw = dict(facecolor="white", bbox_inches="tight", pad_inches=FIG_PAD_IN)
    fig.savefig(png, dpi=dpi, **kw)
    fig.savefig(pdf, **kw)
    plt.close(fig)
    return png, pdf


def _fig4_row_title_y_frac(row_base_y_in: float, panel_h_in: float, fig_h_in: float) -> float:
    grid_y0_in, grid_h_in = _fig4_axes_geometry(panel_h_in)
    return (row_base_y_in + grid_y0_in + grid_h_in + 0.12 + 0.20) / fig_h_in


def _fig4_series_label(series_id: str) -> str:
    if series_id in METHOD_LABELS and series_id.startswith("hf_"):
        return METHOD_LABELS[series_id]
    return FIG3_LF_LABELS.get(series_id, series_id)


def _fig4_region_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=REGION_COLORS[regime],
            markeredgecolor="none",
            markersize=7,
            alpha=0.88,
            label=REGION_LABEL_SHORT[regime],
        )
        for regime in FIG4_REGION_LEGEND_ORDER
    ]


def _fig4_series_legend_order() -> tuple[str, ...]:
    """4 columns (HF, T, ΔT, R) × 3 rows; matplotlib ncol fills each column top-to-bottom."""
    cols = FIG4_LEGEND_COLUMNS
    return tuple(cols[col][row] for col in range(len(cols)) for row in range(3))


def _fig4_series_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=_series_color(sid),
            markeredgecolor="0.35",
            markeredgewidth=0.4,
            markersize=6,
            label=_fig4_series_label(sid),
        )
        for sid in _fig4_series_legend_order()
    ]


def _draw_fig4_composite_legends(
    fig: plt.Figure,
    *,
    fig_h_in: float,
    region_center_y_in: float,
    series_center_y_in: float,
) -> None:
    region_y = region_center_y_in / fig_h_in
    series_y = series_center_y_in / fig_h_in
    region_leg = fig.legend(
        handles=_fig4_region_legend_handles(),
        loc="center",
        ncol=3,
        fontsize=7,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, region_y),
    )
    fig.add_artist(region_leg)
    fig.legend(
        handles=_fig4_series_legend_handles(),
        loc="center",
        ncol=4,
        fontsize=6.5,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.85,
        bbox_to_anchor=(0.5, series_y),
    )


def plot_fig4_abcdefgh(
    exp_dir: Path,
    out_stem: Path,
    *,
    obs_id: int = 0,
    seed: int = 0,
    dpi: int = DPI,
) -> tuple[Path, Path]:
    """Figure 4 composite: (a)–(c) top row, (d)–(h) bottom row."""
    apply_rcparams()
    panel_w_de, panel_h_de = fig4_panel_fig_size_in(exp_dir)
    panel_w_abc = panel_w_de * FIG4_ABC_WIDTH_SCALE
    panel_h_abc = panel_h_de * FIG4_ABC_HEIGHT_SCALE

    bot_content_w = 5 * panel_w_de + 4 * FIG4_BOTTOM_ROW_GAP_IN
    top_content_w = 3 * panel_w_abc + 2 * FIG4_TOP_ROW_GAP_IN
    content_w = max(bot_content_w, top_content_w)
    bot_x0 = FIG_PAD_IN + 0.5 * (content_w - bot_content_w)
    top_x0 = FIG_PAD_IN + 0.5 * (content_w - top_content_w)

    fig_w = content_w + 2 * FIG_PAD_IN
    bot_y0 = FIG_PAD_IN
    series_legend_y0 = bot_y0 + panel_h_de + FIG4_LEGEND_PAD_IN
    region_legend_y0 = series_legend_y0 + FIG4_LEGEND_SERIES_H_IN + FIG4_LEGEND_PAD_IN
    top_y0 = region_legend_y0 + FIG4_LEGEND_REGION_H_IN + FIG4_LEGEND_PAD_IN
    fig_h = top_y0 + panel_h_abc + FIG_PAD_IN

    series_legend_center = series_legend_y0 + 0.5 * FIG4_LEGEND_SERIES_H_IN
    region_legend_center = region_legend_y0 + 0.5 * FIG4_LEGEND_REGION_H_IN

    feat = _load_scoreboard_features(exp_dir)
    nltp_df = _load_nltp_curves(exp_dir)
    summary = _summarize_best_nltp(feat, nltp_df)
    metrics_by_id = load_max_budget_metrics(exp_dir, obs_id=obs_id, seed=seed)

    grid_y0_abc, grid_h_abc = _fig4_axes_geometry(panel_h_abc)
    grid_y0_de, grid_h_de = _fig4_axes_geometry(panel_h_de)
    ax_w_abc = panel_w_abc - 2 * FIG_PAD_IN
    ax_w_de = panel_w_de - 2 * FIG_PAD_IN

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    for i, panel in enumerate("abc"):
        slot_x = top_x0 + i * (panel_w_abc + FIG4_TOP_ROW_GAP_IN)
        ax = _add_axes_inches(fig, slot_x + FIG_PAD_IN, top_y0 + grid_y0_abc, ax_w_abc, grid_h_abc)
        if panel == "a":
            _panel_scoreboard_regions(ax, feat, standalone=True)
        elif panel == "b":
            _panel_nltp_regions(ax, exp_dir, nltp_df, obs_id=obs_id, seed=seed, standalone=True)
        else:
            _panel_score_vs_best_nltp(ax, summary, standalone=True, show_pearson=False)
        fig.text(
            (slot_x + 0.5 * panel_w_abc) / fig_w,
            _fig4_row_title_y_frac(top_y0, panel_h_abc, fig_h),
            FIG4_ABC_TITLES[panel],
            ha="center",
            va="bottom",
            fontsize=11,
            color="0.12",
        )

    for i, panel in enumerate("defgh"):
        slot_x = bot_x0 + i * (panel_w_de + FIG4_BOTTOM_ROW_GAP_IN)
        metric_key, title = FIG4_METRIC_SPECS[panel]
        ax = _add_axes_inches(fig, slot_x + FIG_PAD_IN, bot_y0 + grid_y0_de, ax_w_de, grid_h_de)
        _draw_fig4_metric_bars_on_axes(ax, metrics_by_id, metric_key, show_bar_values=False)
        fig.text(
            (slot_x + 0.5 * panel_w_de) / fig_w,
            _fig4_row_title_y_frac(bot_y0, panel_h_de, fig_h),
            title,
            ha="center",
            va="bottom",
            fontsize=11,
            color="0.12",
        )

    _draw_fig4_composite_legends(
        fig,
        fig_h_in=fig_h,
        region_center_y_in=region_legend_center,
        series_center_y_in=series_legend_center,
    )

    return _save_figure(fig, out_stem, dpi=dpi)


def default_out_stem(exp_dir: Path, panel: str) -> Path:
    names = {
        "a": "fig4_a_scoreboard",
        "b": "fig4_b_nltp",
        "c": "fig4_c_score_nltp",
        "d": "fig4_d_nrmse",
        "e": "fig4_e_crps",
        "f": "fig4_f_tarp",
        "g": "fig4_g_posterior_std",
        "h": "fig4_h_ci_width",
    }
    return fig4_comparison_dir(exp_dir) / "plots" / "max_budget_comparison" / FIG4_OUT_DIR / names[panel]
