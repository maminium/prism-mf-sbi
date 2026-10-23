"""Shared style and paths for paper Figure 7 methods panels (build artifacts)."""
from __future__ import annotations

from pathlib import Path

from prism_vertex.benchmark.comparison_paths import results_dir, fig4_comparison_dir
from scripts.figures._lib.panels_fig2 import (
    DPI,
    FIG3_AXIS_LABEL_FONTSIZE,
    FIG3_LEGEND_FONTSIZE,
    FIG3_TICK_LABELSIZE,
    FIG3_TITLE_COLOR,
    add_fig_panel_title_above_ax,
    apply_rcparams,
)
from scripts.figures._lib.fig3_style import (
    FIG4_CURVE_LW,
    FIG4_FEAS_STAR_EDGE_LW,
    FIG4_FEAS_STAR_SIZE,
    style_axes_like_fig4,
)
from scripts.figures._lib.output_formats import manuscript_build_dir
FIG2_PANEL_SIZE = (6.4, 4.9)
FIG2_TABLE_SIZE = (10.5, 3.6)
FIG2_D_HEATMAP_SIZE = (6.8, 4.9)

# Kursawe et al. (2018) Table 2 lists 13 candidate summary statistics; we use the best 6.
KURSAWE_SUMMARY_CANDIDATES = 13
OBS_SUMMARY_DIM = 6


def fig2_out_dir(results_root: Path | None = None) -> Path:
    """Build directory for Figure 7 panel PNGs (paper fig7 = methods composite)."""
    exp = results_dir(results_root or Path(__file__).resolve().parents[3] / "data")
    return manuscript_build_dir(exp)


def fig2_panel_title_above(fig, ax, letter: str) -> None:
    """Place panel label above the axes block (fig3/fig4 style)."""
    add_fig_panel_title_above_ax(fig, ax, letter)
