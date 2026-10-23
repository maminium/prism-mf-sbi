"""Line weights, bar styling, and axis appearance shared with fig4_abcdefgh."""
from __future__ import annotations

import matplotlib.pyplot as plt

from scripts.figures._lib.panels_fig2 import FIG3_TICK_LABELSIZE, style_axes_like_fig3

# Panel (b) NLTP curves — plot_prism_vertex_scoreboard_nltp_translation._plot_lf_curve_regime
FIG4_CURVE_LW = 1.8
FIG4_CURVE_MS = 4.5
FIG4_CURVE_ALPHA = 0.92

# HF reference lines in composite fig4 panel (b)
FIG4_HF_HLINE_COLOR = "#3D4852"
FIG4_HF_HLINE_LW = 2.4
FIG4_HF_HLINE_ALPHA = 0.95
FIG4_HF_HLINE_LS = (0, (6, 4))

FIG4_GRID_ALPHA = 0.22
FIG4_GRID_ZORDER = 1

# Manuscript curve / scatter panels (Fig 3a–c, Fig 5a–c) — line & marker polish only.
MANUSCRIPT_CURVE_LW = 2.35
MANUSCRIPT_CURVE_MS = 5.8
MANUSCRIPT_CURVE_MS_T_PANEL_B = 7.2
MANUSCRIPT_CURVE_MEW = 1.05
MANUSCRIPT_CURVE_MEC = "white"
MANUSCRIPT_CURVE_ALPHA = 0.96
MANUSCRIPT_HF_HLINE_LW = 2.05
MANUSCRIPT_SCATTER_SIZE = 82
MANUSCRIPT_SCATTER_EDGE_LW = 1.05
MANUSCRIPT_SCATTER_EDGE_COLOR = "white"
MANUSCRIPT_SCATTER_ALPHA = 0.97
MANUSCRIPT_REGION_LINE_LW = 0.85
MANUSCRIPT_REGION_LINE_COLOR = "0.42"

# Panels (d)–(h) metric bars — fig4_panel_common._draw_fig4_metric_bars_on_axes
FIG4_BAR_WIDTH = 0.72
FIG4_SPINE_LW = 0.8
FIG4_SPINE_COLOR = "0.25"
FIG4_TICK_LENGTH = 3
FIG4_TICK_WIDTH = 0.6
FIG4_TICK_PAD = 2

# Feasibility / posterior maps — plot_prism_vertex_fig4_max_budget_comparison legend
FIG4_FEAS_STAR_SIZE = 140
FIG4_FEAS_STAR_EDGE_LW = 0.6
FIG4_FEAS_MESH_N = 200


def style_axes_like_fig4(ax: plt.Axes, *, y_ticks: bool = True, show_x_ticks: bool = True) -> None:
    """Fig3 typography + fig4 spine/tick weights."""
    style_axes_like_fig3(ax, y_ticks=y_ticks)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(FIG4_SPINE_LW)
        ax.spines[side].set_color(FIG4_SPINE_COLOR)
    ax.tick_params(
        axis="x",
        labelsize=FIG3_TICK_LABELSIZE,
        length=FIG4_TICK_LENGTH,
        width=FIG4_TICK_WIDTH,
        pad=FIG4_TICK_PAD,
        labelbottom=show_x_ticks,
    )
    ax.tick_params(
        axis="y",
        labelsize=FIG3_TICK_LABELSIZE,
        length=FIG4_TICK_LENGTH,
        width=FIG4_TICK_WIDTH,
        pad=FIG4_TICK_PAD,
        labelleft=y_ticks,
    )


def style_fig4_bar_axes(ax: plt.Axes) -> None:
    style_axes_like_fig4(ax)
    ax.tick_params(axis="x", length=0, labelbottom=False)


def plot_fig4_curve(
    ax: plt.Axes,
    x,
    y,
    *,
    color: str,
    label: str | None = None,
    zorder: int = 5,
) -> None:
    ax.plot(
        x,
        y,
        color=color,
        marker="o",
        label=label,
        linewidth=FIG4_CURVE_LW,
        markersize=FIG4_CURVE_MS,
        linestyle="-",
        solid_capstyle="round",
        alpha=FIG4_CURVE_ALPHA,
        zorder=zorder,
    )


def plot_fig4_hf_hline(ax: plt.Axes, y: float, x_lo: float, x_hi: float, *, color: str | None = None) -> None:
    ax.hlines(
        y,
        x_lo,
        x_hi,
        colors=color or FIG4_HF_HLINE_COLOR,
        linewidth=FIG4_HF_HLINE_LW,
        linestyles=FIG4_HF_HLINE_LS,
        alpha=FIG4_HF_HLINE_ALPHA,
        zorder=2,
    )


def add_fig4_grid(ax: plt.Axes) -> None:
    ax.grid(alpha=FIG4_GRID_ALPHA, zorder=FIG4_GRID_ZORDER, linewidth=0.55, color="0.82")


def lf_manuscript_linestyle(lf_id: str) -> str | tuple[int, tuple[int, ...]]:
    if lf_id.startswith("lf_DT"):
        return "-"
    if lf_id.startswith("lf_T"):
        return "--"
    return (0, (5, 2.5))


def lf_manuscript_marker(lf_id: str) -> str:
    if lf_id.startswith("lf_DT"):
        return "o"
    if lf_id.startswith("lf_T"):
        return "^"
    return "s"


def plot_manuscript_curve(
    ax: plt.Axes,
    x,
    y,
    *,
    color: str,
    label: str | None = None,
    linestyle: str | tuple[int, tuple[int, ...]] = "-",
    marker: str = "o",
    markersize: float | None = None,
    zorder: int = 5,
) -> None:
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        label=label,
        linewidth=MANUSCRIPT_CURVE_LW,
        markersize=MANUSCRIPT_CURVE_MS if markersize is None else markersize,
        markeredgecolor=MANUSCRIPT_CURVE_MEC,
        markeredgewidth=MANUSCRIPT_CURVE_MEW,
        linestyle=linestyle,
        solid_capstyle="round",
        alpha=MANUSCRIPT_CURVE_ALPHA,
        zorder=zorder,
    )


def plot_manuscript_hf_hline(ax: plt.Axes, y: float, x_lo: float, x_hi: float, *, linestyle, color: str | None = None) -> None:
    ax.hlines(
        y,
        x_lo,
        x_hi,
        colors=color or FIG4_HF_HLINE_COLOR,
        linewidth=MANUSCRIPT_HF_HLINE_LW,
        linestyles=linestyle,
        alpha=FIG4_HF_HLINE_ALPHA,
        zorder=2,
    )


def plot_manuscript_regime_scatter(
    ax: plt.Axes,
    x,
    y,
    *,
    color: str,
    marker: str = "o",
    zorder: int = 4,
) -> None:
    ax.scatter(
        x,
        y,
        s=MANUSCRIPT_SCATTER_SIZE,
        c=color,
        marker=marker,
        edgecolors=MANUSCRIPT_SCATTER_EDGE_COLOR,
        linewidths=MANUSCRIPT_SCATTER_EDGE_LW,
        alpha=MANUSCRIPT_SCATTER_ALPHA,
        zorder=zorder,
    )
