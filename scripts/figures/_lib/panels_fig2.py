"""Shared style and plotting helpers for Figure 3 single-panel charts."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from prism_vertex.benchmark.lf_selection_features import _level_a_arrays, _per_dim_rho
from scripts.figures._lib.color_palette import LF_COLORS, correlation_cmap

# ΔT rows: 0.1 → 0.05 → 0.02 (colours densify along this order, like T and R).
FIG3_LF_ORDER: tuple[str, ...] = (
    "lf_T70",
    "lf_T140",
    "lf_T350",
    "lf_DT10",
    "lf_DT05",
    "lf_DT02",
    "lf_R1",
    "lf_R2",
    "lf_R5",
)

SHORT: dict[str, str] = {
    "lf_T70": "T70",
    "lf_T140": "T140",
    "lf_T350": "T350",
    "lf_DT02": r"$\Delta T$0.02",
    "lf_DT05": r"$\Delta T$0.05",
    "lf_DT10": r"$\Delta T$0.1",
    "lf_R1": "K1",
    "lf_R2": "K2",
    "lf_R5": "K5",
}

FIG_SIZE = (2.6, 3.0)
HEATMAP_CELL_WIDTH_SCALE = 1.5
HEATMAP_EXTRA_HEIGHT_IN = 0.38
DPI = 600
MAX_X_TICKS = 5
SEC_PER_MIN = 60.0
N_LF = len(FIG3_LF_ORDER)

MARGIN_LEFT = 0.14
MARGIN_RIGHT = 0.96
MARGIN_TOP = 0.90
MARGIN_BOTTOM = 0.14

SUMMARY_LABELS = ("S1", "S2", "S3", "S4", "S5", "S6")

FIG3_OUT_DIR = "fig 3"
FIG_PAD_IN = 0.015
ROW_PANEL_GAP_IN = 0.2
ROW_GAP_IN = ROW_PANEL_GAP_IN
ANNOTATE_MIN_CD = 1e-2
FIG3_BAR_WIDTH_SCALE = 0.5
FIG3_BAR_WIDTH_SCALE_COMPACT = 0.25
FIG3_COMPACT_BAR_COLS = frozenset({"mmd_norm", "bias_norm"})

PANEL_TITLES = {
    "a": r"$\mathbf{(A)}$ $\rho$",
    "b": r"$\mathbf{(B)}$ $R^2$",
    "c": r"$\mathbf{(C)}$ $\mathrm{MMD}_{\mathrm{norm}}$",
    "d": r"$\mathbf{(D)}$ $\|\mathrm{Bias}\|_2$",
    "e": r"$\mathbf{(E)}$ $c_j$",
    "f": r"$\mathbf{(F)}$ PRISM",
    "g": r"$\mathbf{(G)}$ $N_{\mathrm{LF}}^{\max}$",
}

PRISM_FORMULA = (
    r"$\mathrm{PRISM}(j)=\dfrac{R^2\,\rho_{\min}/(1+\mathrm{MMD}_{\mathrm{norm}})}"
    r"{c_j(1+\|\mathrm{Bias}\|_2^2)\max(1,\,c_j/0.3)}$"
)
S3_FORMULA = PRISM_FORMULA  # legacy alias

S3_X_TICKS = 6
N_LF_CAP = 10_000
AFFORD_X_TICKS = (10, 100, 1000, 10_000)
COMPACT_BAR_X_TICKS = 3


def apply_rcparams() -> None:
    """Exact Mo 8 manuscript Arial-10 setup (registered TTF + custom mathtext)."""
    arial = Path.home() / ".local/share/fonts/arial/arial.ttf"
    if arial.is_file():
        fm.fontManager.addfont(str(arial))
    for name in ("arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
        bold = arial.with_name(name) if arial.is_file() else None
        if bold is not None and bold.is_file():
            fm.fontManager.addfont(str(bold))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
        }
    )


FIG3_TITLE_FONTSIZE = 11
FIG3_TICK_LABELSIZE = 9
FIG3_AXIS_LABEL_FONTSIZE = 10
FIG3_LEGEND_FONTSIZE = 8
FIG3_TITLE_COLOR = "0.12"


def panel_title(letter: str, subtitle: str = "") -> str:
    """Panel label: bold (A), optional subtitle."""
    label = letter.upper()
    if subtitle:
        return rf"$\mathbf{{({label})}}$ {subtitle}"
    return rf"$\mathbf{{({label})}}$"


def style_axes_like_fig3(ax: plt.Axes, *, y_ticks: bool = True) -> None:
    """Match fig3 tick/spine styling."""
    ax.tick_params(axis="x", labelsize=FIG3_TICK_LABELSIZE, length=3, width=0.6, pad=2)
    ax.tick_params(
        axis="y",
        labelsize=FIG3_TICK_LABELSIZE,
        length=3,
        width=0.6,
        pad=2,
        labelleft=y_ticks,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_fig_panel_title_above_ax(
    fig: plt.Figure,
    ax: plt.Axes,
    letter: str,
    subtitle: str = "",
    *,
    y_pad: float = 0.008,
) -> None:
    """Place a fig3-style panel title just above an axes block."""
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + 0.5 * bbox.width,
        bbox.y1 + y_pad,
        panel_title(letter, subtitle),
        ha="center",
        va="bottom",
        fontsize=FIG3_TITLE_FONTSIZE,
        color=FIG3_TITLE_COLOR,
        transform=fig.transFigure,
    )


def add_fig_panel_label_top_left(
    fig: plt.Figure,
    ax: plt.Axes,
    letter: str,
    *,
    x_pad: float = 0.0,
    y_pad: float = 0.006,
) -> None:
    """Place a bold panel letter at the top-left corner above an axes block."""
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + x_pad,
        bbox.y1 + y_pad,
        panel_title(letter),
        ha="left",
        va="bottom",
        fontsize=FIG3_AXIS_LABEL_FONTSIZE,
        color=FIG3_TITLE_COLOR,
        transform=fig.transFigure,
    )


def add_fig_panel_title_above_block(
    fig: plt.Figure,
    axes,
    letter: str,
    subtitle: str = "",
    *,
    y_pad: float = 0.008,
) -> None:
    """Centered panel title above a multi-axes block (e.g. triangle + ECDF column)."""
    bboxes = [ax.get_position() for ax in np.asarray(axes).ravel()]
    x_center = 0.5 * (min(b.x0 for b in bboxes) + max(b.x1 for b in bboxes))
    y_top = max(b.y1 for b in bboxes)
    fig.text(
        x_center,
        y_top + y_pad,
        panel_title(letter, subtitle),
        ha="center",
        va="bottom",
        fontsize=FIG3_TITLE_FONTSIZE,
        color=FIG3_TITLE_COLOR,
        transform=fig.transFigure,
    )


def style_legend_like_fig3(legend: plt.Legend) -> None:
    legend.set_frame_on(False)
    for text in legend.get_texts():
        text.set_fontsize(FIG3_LEGEND_FONTSIZE)


def load_features(exp_dir: Path) -> pd.DataFrame:
    path = exp_dir / "lf_selection" / "analysis" / "lf_selection_features_and_truth.csv"
    feat = pd.read_csv(path)
    order = {lf: i for i, lf in enumerate(FIG3_LF_ORDER)}
    feat = feat.sort_values("lf_id", key=lambda s: s.map(order)).reset_index(drop=True)
    return attach_s3_values(feat)


def _signed_s3_numerator(r2: float, rho_min: float, inv_mmd: float) -> float:
    raw = r2 * rho_min * inv_mmd
    if not np.isfinite(raw):
        return float("nan")
    if r2 < 0 or rho_min < 0:
        return -abs(raw)
    return float(raw)


def attach_s3_values(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()

    def _row_s3(row: pd.Series) -> float:
        c = float(row["cost_ratio"])
        r2 = float(row["R2_A"])
        rho_min = float(row["rho_min"])
        mmd_norm = float(row["mmd_norm"])
        bias = float(row["bias_norm"])
        if c <= 0 or not np.isfinite(rho_min) or not np.isfinite(mmd_norm):
            return float("nan")
        inv_mmd = 1.0 / (1.0 + mmd_norm)
        num = _signed_s3_numerator(r2, rho_min, inv_mmd)
        den = c * (1.0 + bias**2) * max(1.0, c / 0.30)
        return float(num / den) if den > 0 else float("nan")

    out["S3_val"] = out.apply(_row_s3, axis=1)
    return out


def load_cost_minutes(exp_dir: Path) -> pd.DataFrame:
    sb = pd.read_csv(exp_dir / "lf_selection" / "analysis" / "scoreboard_table.csv")
    idx = sb.set_index("fidelity_id")
    rows = [
        {"lf_id": lf, "cost_min": float(idx.loc[lf, "median_wall_s"]) / SEC_PER_MIN}
        for lf in FIG3_LF_ORDER
    ]
    return pd.DataFrame(rows)


def format_small_or_decimal(v: float) -> str:
    if not np.isfinite(v):
        return ""
    if abs(v) < 0.01:
        return f"{v:.2e}"
    return f"{v:.3f}"


def format_cost_minutes(v: float) -> str:
    if not np.isfinite(v):
        return ""
    if abs(v) < 0.01:
        return f"{v:.2e}"
    if abs(v) < 10.0:
        return f"{v:.2f}"
    return f"{v:.1f}"


def format_signed_r2(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return f"{v:.2f}"


def format_s3_val(v: float) -> str:
    if not np.isfinite(v):
        return ""
    if abs(v) >= 10.0:
        return f"{v:.1f}"
    return f"{v:.2f}"


def format_n_lf(v: float) -> str:
    if not np.isfinite(v):
        return ""
    if v >= 1000:
        return f"{v:,.0f}"
    return f"{v:.0f}"


def _label_text_color(hex_color: str) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 140 else "0.12"


def _fig_height_in() -> float:
    return FIG_SIZE[1] + HEATMAP_EXTRA_HEIGHT_IN


def _title_y_frac_for_row(row_idx: int, fig_total_h_in: float) -> float:
    grid_y0, grid_h, _ = _grid_geometry_inches()
    row_base = FIG_PAD_IN + row_idx * (_fig_height_in() + ROW_GAP_IN)
    return (row_base + grid_y0 + grid_h + 0.12 + 0.20) / fig_total_h_in


def _title_y_frac() -> float:
    return _title_y_frac_for_row(0, _fig_height_in())


def _row_base_y_in(row_idx: int) -> float:
    """Row 0 = bottom (f, g); row 1 = top (a–e)."""
    return FIG_PAD_IN + row_idx * (_fig_height_in() + ROW_GAP_IN)


def _fig3_abcde_bar_specs() -> tuple:
    return (
        ("R2_A", True, format_signed_r2, None),
        ("mmd_norm", False, format_small_or_decimal, ANNOTATE_MIN_CD),
        ("bias_norm", False, format_small_or_decimal, ANNOTATE_MIN_CD),
        ("cost_min", False, format_cost_minutes, None),
    )


def _fig3_panel_widths(feat: pd.DataFrame, row_h_in: float) -> list[float]:
    panel_widths = [_panel_a_width_in(row_h_in)]
    for col, signed, fmt, amin in _fig3_abcde_bar_specs():
        draw_kw = _bar_panel_draw_kwargs(col)
        panel_widths.append(
            _measure_bar_width_in(
                feat,
                col,
                signed=signed,
                value_fmt=fmt,
                annotate_min=amin,
                **draw_kw,
            )
        )
    return panel_widths


def _fig3_fg_slot_widths(panel_widths: list[float]) -> tuple[float, float]:
    f_slot = panel_widths[0] + panel_widths[1] + panel_widths[2] + 2 * ROW_PANEL_GAP_IN
    g_slot = panel_widths[3] + panel_widths[4] + ROW_PANEL_GAP_IN
    return f_slot, g_slot


def _panel_a_width_in(row_h_in: float) -> float:
    cell_w_in = row_h_in * HEATMAP_CELL_WIDTH_SCALE
    return (1 + len(SUMMARY_LABELS)) * cell_w_in


def _lf_label_width_in(row_h_in: float) -> float:
    return row_h_in * HEATMAP_CELL_WIDTH_SCALE


def _panel_f_content_width_in(feat: pd.DataFrame, row_h_in: float) -> float:
    """Total content width for panel (f): panels (a) + (b) + (c)."""
    b_kw = _bar_panel_draw_kwargs("R2_A")
    c_kw = _bar_panel_draw_kwargs("mmd_norm")
    return (
        _panel_a_width_in(row_h_in)
        + _measure_bar_width_in(
            feat,
            "R2_A",
            signed=True,
            value_fmt=format_signed_r2,
            **b_kw,
        )
        + _measure_bar_width_in(
            feat,
            "mmd_norm",
            signed=False,
            value_fmt=format_small_or_decimal,
            annotate_min=ANNOTATE_MIN_CD,
            **c_kw,
        )
    )


def _panel_g_content_width_in(feat: pd.DataFrame, row_h_in: float) -> float:
    """Total content width for panel (g): panels (d) + (e)."""
    d_kw = _bar_panel_draw_kwargs("bias_norm")
    e_kw = _bar_panel_draw_kwargs("cost_min")
    return _measure_bar_width_in(
        feat,
        "bias_norm",
        signed=False,
        value_fmt=format_small_or_decimal,
        annotate_min=ANNOTATE_MIN_CD,
        **d_kw,
    ) + _measure_bar_width_in(
        feat,
        "cost_min",
        signed=False,
        value_fmt=format_cost_minutes,
        **e_kw,
    )


def _save_figure(fig: plt.Figure, out_stem: Path, *, dpi: int) -> tuple[Path, Path]:
    from scripts.figures._lib.output_formats import (
        save_manuscript_submission_formats,
        save_matplotlib_manuscript_png,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    kw = dict(facecolor="white", bbox_inches="tight", pad_inches=FIG_PAD_IN)
    save_matplotlib_manuscript_png(fig, png, save_kwargs=kw)
    fig.savefig(pdf, **kw)
    save_manuscript_submission_formats(fig, out_stem, dpi=dpi, save_kwargs=dict(pad_inches=FIG_PAD_IN))
    plt.close(fig)
    return png, pdf


def _axes_bbox_width_in(ax: plt.Axes, fig: plt.Figure) -> float:
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.0)
    return bbox.width / fig.dpi


def _measure_bar_width_in(
    feat: pd.DataFrame,
    col: str,
    *,
    signed: bool,
    value_fmt: Callable[[float], str],
    annotate_min: float | None = None,
    outer_pad: bool = False,
    width_scale: float = 1.0,
    xlim_right_scale: float = 1.22,
) -> float:
    apply_rcparams()
    fig_h = _fig_height_in()
    grid_y0, grid_h, _ = _grid_geometry_inches()
    fig = plt.figure(figsize=(3.5, fig_h), dpi=100, facecolor="white")
    ax = _add_axes_inches(fig, 0.05, grid_y0, 3.35, grid_h)
    _draw_bar_on_axes(
        ax,
        feat,
        col,
        signed=signed,
        value_fmt=value_fmt,
        annotate_min=annotate_min,
        xlim_right_scale=xlim_right_scale,
    )
    width_in = _axes_bbox_width_in(ax, fig) * width_scale
    if outer_pad:
        width_in += 2 * FIG_PAD_IN
    plt.close(fig)
    return width_in


def _bar_panel_draw_kwargs(col: str) -> dict[str, float]:
    scale = (
        FIG3_BAR_WIDTH_SCALE_COMPACT
        if col in FIG3_COMPACT_BAR_COLS
        else FIG3_BAR_WIDTH_SCALE
    )
    return {"width_scale": scale, "xlim_right_scale": 1.22}


def _cells_path(exp_dir: Path) -> Path:
    for rel in ("lf_selection/analysis/cells.csv", "lf_selection/cells.csv"):
        p = exp_dir / rel
        if p.is_file():
            return p
    raise FileNotFoundError(f"No cells.csv under {exp_dir / 'lf_selection'}")


def load_rho_matrix(exp_dir: Path, *, n_theta: int = 20) -> np.ndarray:
    cells = pd.read_csv(_cells_path(exp_dir))
    return np.vstack([_per_dim_rho(*_level_a_arrays(cells, lf, n_theta)) for lf in FIG3_LF_ORDER])


def plot_rho_heatmap_panel(
    rho_mat: np.ndarray,
    *,
    out_stem: Path,
    dpi: int = DPI,
    title: str = PANEL_TITLES["a"],
) -> tuple[Path, Path]:
    """Panel (a): Pearson ρ heatmap with LF row labels and colorbar below."""
    apply_rcparams()
    grid_y0_in, grid_h_in, row_h_in = _grid_geometry_inches()
    content_w = _panel_a_width_in(row_h_in)
    fig_w = content_w + 2 * FIG_PAD_IN
    fig_h = _fig_height_in()
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    _draw_panel_a_on_fig(
        fig,
        rho_mat,
        slot_x_in=FIG_PAD_IN,
        grid_y0_in=grid_y0_in,
        grid_h_in=grid_h_in,
        row_h_in=row_h_in,
    )
    fig.text(
        (FIG_PAD_IN + 0.5 * content_w) / fig_w,
        _title_y_frac(),
        title,
        ha="center",
        va="bottom",
        fontsize=11,
        color="0.12",
    )
    return _save_figure(fig, out_stem, dpi=dpi)


def _grid_geometry_inches() -> tuple[float, float, float]:
    row_h_in = FIG_SIZE[1] * (MARGIN_TOP - MARGIN_BOTTOM) / N_LF
    grid_h_in = N_LF * row_h_in
    grid_y0_in = MARGIN_BOTTOM * FIG_SIZE[1]
    return grid_y0_in, grid_h_in, row_h_in


def _add_axes_inches(
    fig: plt.Figure,
    x_in: float,
    y_in: float,
    w_in: float,
    h_in: float,
) -> plt.Axes:
    fig_w, fig_h = fig.get_size_inches()
    return fig.add_axes([x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h])


def _draw_lf_label_column(
    fig: plt.Figure,
    *,
    slot_x_in: float,
    grid_y0_in: float,
    grid_h_in: float,
    row_h_in: float,
    n_rows: int,
) -> float:
    """LF colour boxes matching panel (a); returns label column width in inches."""
    label_w_in = _lf_label_width_in(row_h_in)
    ax_labels = _add_axes_inches(fig, slot_x_in, grid_y0_in, label_w_in, grid_h_in)
    ax_labels.set_xlim(0.0, 1.0)
    ax_labels.set_ylim(float(n_rows), 0.0)
    ax_labels.axis("off")
    cell_h_frac = grid_h_in / n_rows / fig.get_size_inches()[1]
    label_pt = max(5.0, min(6.5, cell_h_frac * fig.get_size_inches()[1] * 72 * 0.36))
    for i, lf in enumerate(FIG3_LF_ORDER):
        color = LF_COLORS[lf]
        ax_labels.add_patch(
            Rectangle(
                (0.0, float(i)),
                1.0,
                1.0,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                clip_on=False,
            )
        )
        ax_labels.text(
            0.5,
            i + 0.5,
            SHORT[lf],
            ha="center",
            va="center",
            fontsize=label_pt,
            color=_label_text_color(color),
            fontweight="medium",
            clip_on=False,
        )
    return label_w_in


def _draw_s3_lollipop_on_axes(ax: plt.Axes, feat: pd.DataFrame) -> None:
    s3 = feat["S3_val"].to_numpy(dtype=float)
    n = len(s3)
    y = np.arange(n, dtype=float)

    finite = s3[np.isfinite(s3)]
    if finite.size:
        x_lo = min(0.0, float(np.nanmin(finite)))
        x_hi = max(0.0, float(np.nanmax(finite)))
    else:
        x_lo, x_hi = -1.0, 1.0
    span = max(x_hi - x_lo, 1e-6)
    pad = 0.05 * span
    xlim = (x_lo - pad, x_hi + pad)
    ax.set_xlim(xlim)
    ax.set_xticks(np.linspace(xlim[0], xlim[1], S3_X_TICKS))

    ax.axvline(0, color="0.35", lw=1.0, zorder=1)
    x_span = xlim[1] - xlim[0]
    for yi, lf, v in zip(y, feat["lf_id"], s3):
        if not np.isfinite(v):
            continue
        color = LF_COLORS[lf]
        ax.hlines(yi, 0.0, v, color=color, lw=2.0, zorder=2)
        ax.scatter([v], [yi], s=42, color=color, edgecolors="0.15", linewidths=0.35, zorder=5)
        off = 0.02 * x_span
        if lf == "lf_T140":
            ax.text(
                v,
                yi + 0.38,
                format_s3_val(v),
                va="top",
                ha="center",
                fontsize=8,
                color="0.12",
                zorder=6,
            )
        elif v >= 0:
            ax.text(v + off, yi, format_s3_val(v), va="center", ha="left", fontsize=8, color="0.12", zorder=6)
        else:
            ax.text(v - off, yi, format_s3_val(v), va="center", ha="right", fontsize=8, color="0.12", zorder=6)

    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("0.25")
    ax.tick_params(axis="x", labelsize=9, length=3, width=0.6, pad=2)
    ax.tick_params(axis="y", length=0, labelleft=False, left=False)

    t140_y = float(list(FIG3_LF_ORDER).index("lf_T140"))
    y_formula = 0.5 * (t140_y + (n - 0.5))
    x_formula = -10.0
    if xlim[0] <= x_formula <= 0.0:
        ax.text(
            x_formula,
            y_formula,
            PRISM_FORMULA,
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.12",
            zorder=6,
        )


def _draw_panel_f_on_fig(
    fig: plt.Figure,
    feat: pd.DataFrame,
    *,
    slot_x_in: float,
    grid_y0_in: float,
    grid_h_in: float,
    row_h_in: float,
    content_w_in: float,
) -> None:
    n_rows = len(feat)
    label_w_in = _draw_lf_label_column(
        fig,
        slot_x_in=slot_x_in,
        grid_y0_in=grid_y0_in,
        grid_h_in=grid_h_in,
        row_h_in=row_h_in,
        n_rows=n_rows,
    )
    chart_w_in = content_w_in - label_w_in
    ax = _add_axes_inches(fig, slot_x_in + label_w_in, grid_y0_in, chart_w_in, grid_h_in)
    _draw_s3_lollipop_on_axes(ax, feat)


def plot_s3_lollipop_panel(
    feat: pd.DataFrame,
    *,
    out_stem: Path,
    dpi: int = DPI,
    title: str = PANEL_TITLES["f"],
) -> tuple[Path, Path]:
    """Panel (f): signed PRISM lollipop chart with heatmap-matched LF labels."""
    apply_rcparams()
    grid_y0_in, grid_h_in, row_h_in = _grid_geometry_inches()
    content_w = _panel_f_content_width_in(feat, row_h_in)
    fig_w = content_w + 2 * FIG_PAD_IN
    fig_h = _fig_height_in()
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    _draw_panel_f_on_fig(
        fig,
        feat,
        slot_x_in=FIG_PAD_IN,
        grid_y0_in=grid_y0_in,
        grid_h_in=grid_h_in,
        row_h_in=row_h_in,
        content_w_in=content_w,
    )
    fig.text(
        (FIG_PAD_IN + 0.5 * content_w) / fig_w,
        _title_y_frac(),
        title,
        ha="center",
        va="bottom",
        fontsize=11,
        color="0.12",
    )
    return _save_figure(fig, out_stem, dpi=dpi)


def _apply_bar_axis_style(ax: plt.Axes, n: int) -> None:
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(n, dtype=float))
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("0.25")
    ax.tick_params(axis="x", labelsize=9, length=3, width=0.6, pad=2)
    ax.tick_params(axis="y", length=0, labelleft=False, left=False)


def _draw_n_lf_max_on_axes(ax: plt.Axes, feat: pd.DataFrame) -> None:
    n_raw = feat["n_afford_cap"].to_numpy(dtype=float)
    n_plot = np.minimum(n_raw, float(N_LF_CAP))
    truncated = n_raw > N_LF_CAP
    colors = [LF_COLORS[lf] for lf in feat["lf_id"]]
    n = len(n_raw)
    y = np.arange(n, dtype=float)

    bars = ax.barh(y, n_plot, color=colors, height=0.72, zorder=2, align="center")
    for bar, is_trunc in zip(bars, truncated):
        if is_trunc:
            bar.set_hatch("///")
            bar.set_edgecolor("0.25")
            bar.set_linewidth(0.55)

    ax.axvline(N_LF_CAP, color=LF_COLORS["lf_DT02"], ls="--", lw=1.2, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(8.0, N_LF_CAP * 1.14)
    ax.set_xticks(list(AFFORD_X_TICKS))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: format_n_lf(v)))
    _apply_bar_axis_style(ax, n)

    label_x = N_LF_CAP * 1.06
    for yi, raw, is_trunc in zip(y, n_raw, truncated):
        if not np.isfinite(raw):
            continue
        txt = rf"$\geq${format_n_lf(raw)}" if is_trunc else format_n_lf(raw)
        ax.text(label_x, yi, txt, va="center", ha="left", fontsize=8, color="0.12", zorder=6)


def plot_n_lf_max_panel(
    feat: pd.DataFrame,
    *,
    out_stem: Path,
    dpi: int = DPI,
    title: str = PANEL_TITLES["g"],
) -> tuple[Path, Path]:
    """Panel (g): log-scale :math:`N_{LF}^{max}` bars with display cap at 10,000."""
    apply_rcparams()
    grid_y0_in, grid_h_in, row_h_in = _grid_geometry_inches()
    content_w = _panel_g_content_width_in(feat, row_h_in)
    fig_w = content_w + 2 * FIG_PAD_IN
    fig_h = _fig_height_in()
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = _add_axes_inches(fig, FIG_PAD_IN, grid_y0_in, content_w, grid_h_in)
    _draw_n_lf_max_on_axes(ax, feat)
    fig.text(
        (FIG_PAD_IN + 0.5 * content_w) / fig_w,
        _title_y_frac(),
        title,
        ha="center",
        va="bottom",
        fontsize=11,
        color="0.12",
    )
    return _save_figure(fig, out_stem, dpi=dpi)


def _draw_panel_a_on_fig(
    fig: plt.Figure,
    rho_mat: np.ndarray,
    *,
    slot_x_in: float,
    grid_y0_in: float,
    grid_h_in: float,
    row_h_in: float,
) -> None:
    n_rows, n_cols = rho_mat.shape
    cell_w_in = row_h_in * HEATMAP_CELL_WIDTH_SCALE
    label_w_in = cell_w_in
    heat_w_in = n_cols * cell_w_in
    header_h_in = 0.12
    cbar_h_in = 0.15

    x_label_in = slot_x_in
    x_heat_in = x_label_in + label_w_in
    y_header_in = grid_y0_in + grid_h_in + 0.02
    y_cbar_in = max(0.08, grid_y0_in - cbar_h_in - 0.04)

    _draw_lf_label_column(
        fig,
        slot_x_in=x_label_in,
        grid_y0_in=grid_y0_in,
        grid_h_in=grid_h_in,
        row_h_in=row_h_in,
        n_rows=n_rows,
    )
    ax_heat = _add_axes_inches(fig, x_heat_in, grid_y0_in, heat_w_in, grid_h_in)

    norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = correlation_cmap()
    ax_heat.imshow(
        rho_mat,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        extent=(0.0, float(n_cols), float(n_rows), 0.0),
        interpolation="nearest",
        zorder=1,
    )
    for i in range(n_rows + 1):
        ax_heat.axhline(i, color="white", linewidth=1.0, zorder=2)
    for j in range(n_cols + 1):
        ax_heat.axvline(j, color="white", linewidth=1.0, zorder=2)

    cell_h_frac = grid_h_in / n_rows / fig.get_size_inches()[1]
    annot_pt = max(5.0, min(7.0, cell_h_frac * fig.get_size_inches()[1] * 72 * 0.42))
    for i in range(n_rows):
        for j in range(n_cols):
            v = rho_mat[i, j]
            if not np.isfinite(v):
                continue
            fg = "white" if abs(v) > 0.55 else "0.12"
            ax_heat.text(
                j + 0.5,
                i + 0.5,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=annot_pt,
                color=fg,
                fontweight="medium",
                zorder=3,
            )

    ax_heat.set_xlim(0.0, float(n_cols))
    ax_heat.set_ylim(float(n_rows), 0.0)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    ax_header = _add_axes_inches(fig, x_heat_in, y_header_in, heat_w_in, header_h_in)
    ax_header.set_xlim(0.0, float(n_cols))
    ax_header.set_ylim(0.0, 1.0)
    ax_header.axis("off")
    header_pt = max(5.0, min(7.0, cell_w_in * 72 * 0.34))
    for j, label in enumerate(SUMMARY_LABELS):
        ax_header.text(
            j + 0.5,
            0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=header_pt,
            color="0.12",
        )

    ax_cbar = _add_axes_inches(fig, x_heat_in, y_cbar_in, heat_w_in, cbar_h_in)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=7, length=2, width=0.5, pad=1)
    cbar.outline.set_linewidth(0.35)
    cbar.outline.set_edgecolor("0.35")


def _draw_bar_on_axes(
    ax: plt.Axes,
    feat: pd.DataFrame,
    col: str,
    *,
    signed: bool = False,
    value_fmt: Callable[[float], str] = format_small_or_decimal,
    annotate_min: float | None = None,
    xlim_right_scale: float = 1.22,
) -> None:
    vals = feat[col].to_numpy(dtype=float)
    colors = [LF_COLORS[lf] for lf in feat["lf_id"]]
    n = len(vals)
    y = np.arange(n, dtype=float)

    ax.barh(y, vals, color=colors, height=0.72, zorder=2, align="center")
    ax.axvline(0, color="0.35", lw=1.0, zorder=1)

    if signed:
        vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
        pad = max(vmax * 0.30, 0.15)
        ax.set_xlim(-vmax - pad, vmax + pad)
    else:
        vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
        ax.set_xlim(0.0, vmax * xlim_right_scale)

    ax.xaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(
            nbins=COMPACT_BAR_X_TICKS if col in FIG3_COMPACT_BAR_COLS else MAX_X_TICKS,
            prune=None,
        )
    )
    _apply_bar_axis_style(ax, n)

    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for yi, v in zip(y, vals):
        if not np.isfinite(v):
            continue
        if annotate_min is not None and abs(v) < annotate_min:
            continue
        if signed and v < 0:
            ha, off = "right", -0.02 * x_span
        else:
            ha, off = "left", 0.02 * x_span
        ax.text(v + off, yi, value_fmt(v), va="center", ha=ha, fontsize=8, color="0.12")


def plot_fig3_abcde_row(
    exp_dir: Path,
    out_stem: Path,
    *,
    dpi: int = DPI,
) -> tuple[Path, Path]:
    """Panels (a)–(e) in one row with aligned LF rows and titles."""
    apply_rcparams()
    feat = load_features(exp_dir)
    feat = feat.assign(cost_min=load_cost_minutes(exp_dir)["cost_min"].to_numpy())
    rho_mat = load_rho_matrix(exp_dir)

    grid_y0_in, grid_h_in, row_h_in = _grid_geometry_inches()
    fig_h_in = _fig_height_in()
    panel_widths = _fig3_panel_widths(feat, row_h_in)
    bar_specs = _fig3_abcde_bar_specs()

    n_gaps = len(panel_widths) - 1
    fig_w_in = sum(panel_widths) + 2 * FIG_PAD_IN + n_gaps * ROW_PANEL_GAP_IN
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")

    slot_x = FIG_PAD_IN
    _draw_panel_a_on_fig(
        fig,
        rho_mat,
        slot_x_in=slot_x,
        grid_y0_in=grid_y0_in,
        grid_h_in=grid_h_in,
        row_h_in=row_h_in,
    )
    slot_x += panel_widths[0] + ROW_PANEL_GAP_IN

    for (col, signed, fmt, amin), panel_w in zip(bar_specs, panel_widths[1:]):
        draw_kw = _bar_panel_draw_kwargs(col)
        ax = _add_axes_inches(fig, slot_x, grid_y0_in, panel_w, grid_h_in)
        _draw_bar_on_axes(
            ax,
            feat,
            col,
            signed=signed,
            value_fmt=fmt,
            annotate_min=amin,
            xlim_right_scale=draw_kw["xlim_right_scale"],
        )
        slot_x += panel_w + ROW_PANEL_GAP_IN

    title_y = _title_y_frac_for_row(0, fig_h_in)
    slot_x = FIG_PAD_IN
    for panel_w, title in zip(panel_widths, (PANEL_TITLES[k] for k in "abcde")):
        fig.text(
            (slot_x + 0.5 * panel_w) / fig_w_in,
            title_y,
            title,
            ha="center",
            va="bottom",
            fontsize=11,
            color="0.12",
        )
        slot_x += panel_w + ROW_PANEL_GAP_IN

    return _save_figure(fig, out_stem, dpi=dpi)


def plot_fig3_abcdefg(
    exp_dir: Path,
    out_stem: Path,
    *,
    dpi: int = DPI,
) -> tuple[Path, Path]:
    """Panels (a)–(g): ABCDE row on top, F and G aligned below."""
    apply_rcparams()
    feat = load_features(exp_dir)
    feat = feat.assign(cost_min=load_cost_minutes(exp_dir)["cost_min"].to_numpy())
    rho_mat = load_rho_matrix(exp_dir)

    grid_y0_in, grid_h_in, row_h_in = _grid_geometry_inches()
    row_h = _fig_height_in()
    panel_widths = _fig3_panel_widths(feat, row_h_in)
    bar_specs = _fig3_abcde_bar_specs()
    f_slot_w, g_slot_w = _fig3_fg_slot_widths(panel_widths)

    n_gaps = len(panel_widths) - 1
    fig_w_in = sum(panel_widths) + 2 * FIG_PAD_IN + n_gaps * ROW_PANEL_GAP_IN
    fig_total_h_in = 2 * row_h + ROW_GAP_IN + 2 * FIG_PAD_IN
    fig = plt.figure(figsize=(fig_w_in, fig_total_h_in), facecolor="white")

    # --- top row: (a)–(e) ---
    top_y0 = _row_base_y_in(1)
    slot_x = FIG_PAD_IN
    _draw_panel_a_on_fig(
        fig,
        rho_mat,
        slot_x_in=slot_x,
        grid_y0_in=top_y0 + grid_y0_in,
        grid_h_in=grid_h_in,
        row_h_in=row_h_in,
    )
    slot_x += panel_widths[0] + ROW_PANEL_GAP_IN

    for (col, signed, fmt, amin), panel_w in zip(bar_specs, panel_widths[1:]):
        draw_kw = _bar_panel_draw_kwargs(col)
        ax = _add_axes_inches(fig, slot_x, top_y0 + grid_y0_in, panel_w, grid_h_in)
        _draw_bar_on_axes(
            ax,
            feat,
            col,
            signed=signed,
            value_fmt=fmt,
            annotate_min=amin,
            xlim_right_scale=draw_kw["xlim_right_scale"],
        )
        slot_x += panel_w + ROW_PANEL_GAP_IN

    top_title_y = _title_y_frac_for_row(1, fig_total_h_in)
    slot_x = FIG_PAD_IN
    for panel_w, title in zip(panel_widths, (PANEL_TITLES[k] for k in "abcde")):
        fig.text(
            (slot_x + 0.5 * panel_w) / fig_w_in,
            top_title_y,
            title,
            ha="center",
            va="bottom",
            fontsize=11,
            color="0.12",
        )
        slot_x += panel_w + ROW_PANEL_GAP_IN

    # --- bottom row: (f), (g) ---
    bot_y0 = _row_base_y_in(0)
    f_x = FIG_PAD_IN
    _draw_panel_f_on_fig(
        fig,
        feat,
        slot_x_in=f_x,
        grid_y0_in=bot_y0 + grid_y0_in,
        grid_h_in=grid_h_in,
        row_h_in=row_h_in,
        content_w_in=f_slot_w,
    )
    g_x = f_x + f_slot_w + ROW_PANEL_GAP_IN
    ax_g = _add_axes_inches(fig, g_x, bot_y0 + grid_y0_in, g_slot_w, grid_h_in)
    _draw_n_lf_max_on_axes(ax_g, feat)

    bot_title_y = _title_y_frac_for_row(0, fig_total_h_in)
    fig.text((f_x + 0.5 * f_slot_w) / fig_w_in, bot_title_y, PANEL_TITLES["f"], ha="center", va="bottom", fontsize=11, color="0.12")
    fig.text((g_x + 0.5 * g_slot_w) / fig_w_in, bot_title_y, PANEL_TITLES["g"], ha="center", va="bottom", fontsize=11, color="0.12")

    return _save_figure(fig, out_stem, dpi=dpi)


def plot_lf_bar_panel(
    feat: pd.DataFrame,
    col: str,
    *,
    title: str,
    out_stem: Path,
    dpi: int = DPI,
    signed: bool = False,
    value_fmt: Callable[[float], str] = format_small_or_decimal,
    annotate_min: float | None = None,
    width_scale: float | None = None,
    xlim_right_scale: float | None = None,
) -> tuple[Path, Path]:
    apply_rcparams()
    draw_kw = _bar_panel_draw_kwargs(col)
    if width_scale is None:
        width_scale = draw_kw["width_scale"]
    if xlim_right_scale is None:
        xlim_right_scale = draw_kw["xlim_right_scale"]
    grid_y0_in, grid_h_in, _ = _grid_geometry_inches()
    fig_w_in = _measure_bar_width_in(
        feat,
        col,
        signed=signed,
        value_fmt=value_fmt,
        annotate_min=annotate_min,
        outer_pad=True,
        width_scale=width_scale,
        xlim_right_scale=xlim_right_scale,
    )
    fig_h_in = _fig_height_in()
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")
    ax = _add_axes_inches(fig, FIG_PAD_IN, grid_y0_in, fig_w_in - 2 * FIG_PAD_IN, grid_h_in)
    _draw_bar_on_axes(
        ax,
        feat,
        col,
        signed=signed,
        value_fmt=value_fmt,
        annotate_min=annotate_min,
        xlim_right_scale=xlim_right_scale,
    )
    fig.text(0.5, _title_y_frac(), title, ha="center", va="bottom", fontsize=11, color="0.12")
    return _save_figure(fig, out_stem, dpi=dpi)
