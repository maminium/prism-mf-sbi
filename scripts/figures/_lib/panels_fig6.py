"""Figure 5 — MF-SBI posterior triangle + TARP/SBC at ΔT_0.05."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from prism_vertex.benchmark.comparison_paths import fig4_comparison_dir, method_results_dir
from prism_vertex.simulator.constants import THETA_TRUE
from prism_vertex.metrics import benchmark_unified as bu
from scripts.figures._lib.panels_fig2 import (
    DPI,
    FIG3_AXIS_LABEL_FONTSIZE,
    FIG3_LEGEND_FONTSIZE,
    FIG3_TITLE_FONTSIZE,
    _save_figure,
    add_fig_panel_title_above_block,
    apply_rcparams,
)
from scripts.figures._lib.fig3_style import (
    FIG4_CURVE_LW,
    FIG4_GRID_ALPHA,
    style_axes_like_fig4,
)
from scripts.figures._lib.fig6_calibration import load_calibration_sbc_ranks
from scripts.figures._lib.color_palette import (
    DKW_BAND_COLOR,
    IDEAL_LINE_COLOR,
    METHOD_COLORS,
    THETA_STAR_COLOR,
)

def calibration_theta_true(exp_dir: Path, obs_id: int) -> np.ndarray:
    for cand in (
        exp_dir / "observations" / f"obs_{obs_id:02d}" / "theta_true.npy",
        exp_dir / "observations" / f"theta_{obs_id:02d}.npy",
    ):
        if cand.is_file():
            return np.load(cand).reshape(-1)
    if obs_id == 0 and (exp_dir / "observations" / "theta_true.npy").is_file():
        return np.load(exp_dir / "observations" / "theta_true.npy").reshape(-1)
    from prism_vertex.simulator.calibration_observations import theta_for_obs
    return theta_for_obs(obs_id)


def _load_posteriors(exp_dir: Path) -> pd.DataFrame:
    from prism_vertex.benchmark.calibration import iter_calibration_run_dirs
    rows: list[dict[str, object]] = []
    for run_dir, method, obs_id, _seed in iter_calibration_run_dirs(exp_dir):
        post = run_dir / "posterior_samples.csv"
        if not post.is_file():
            continue
        df = pd.read_csv(post)
        theta_cols = [c for c in df.columns if c in ("lambda", "gamma") or c.startswith("theta_")]
        if not theta_cols:
            continue
        rows.append({"method": method, "obs_id": obs_id, "samples": df[theta_cols].to_numpy(dtype=float)})
    return pd.DataFrame(rows)


FIG4_OUT_DIR = "fig 4"
TRIANGLE_BLOCK_SIZE = (6.8, 6.8)
OPERATING_LF = "lf_DT05"
OPERATING_BUDGET = "budget_lf_010000_hf_000500"

METHOD_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("mf_sbi", "MF-SBI", METHOD_COLORS["mf_sbi"], "fig5_posterior_triangle_mf_sbi"),
)
STACKED_STEM = "fig5_mf_sbi_calibration"
FIG6_BLOCK_TITLES: dict[str, str] = {
    "a": "MF-SBI posterior",
    "b": "MF-SBI SBC",
}
PARAM_LABELS = (r"$\Lambda$", r"$\Gamma$")
UNIT_TICKS = [0.0, 0.5, 1.0]
TARP_ALPHAS = np.linspace(0.02, 0.98, 24)

MethodSampleRow = tuple[str, str, str, str, np.ndarray]


def default_out_stem(exp_dir: Path, stem: str) -> Path:
    return fig4_comparison_dir(exp_dir) / "plots" / "max_budget_comparison" / FIG4_OUT_DIR / stem


def _posterior_run_dir(exp_dir: Path, method: str, obs_id: int, seed: int) -> Path:
    return (
        method_results_dir(exp_dir, method)
        / OPERATING_LF
        / OPERATING_BUDGET
        / f"obs_{obs_id:02d}"
        / f"seed_{seed:02d}"
    )


def _load_posterior_xy(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [c for c in ("lambda", "gamma") if c in df.columns]
    if len(cols) != 2:
        raise ValueError(f"Expected lambda,gamma columns in {path}")
    return df[cols].to_numpy(dtype=float)


def _kde_1d(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.zeros_like(grid, dtype=float)
    return gaussian_kde(x, bw_method="silverman")(grid)


def _kde_2d_contour(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n_levels: int = 6,
    alpha: float = 0.95,
) -> None:
    xy = np.vstack([x, y])
    m = np.isfinite(xy).all(axis=0)
    xy = xy[:, m]
    if xy.shape[1] < 10:
        return
    kde = gaussian_kde(xy, bw_method="silverman")
    pad_x = 0.10 * max(xlim[1] - xlim[0], 1e-6)
    pad_y = 0.10 * max(ylim[1] - ylim[0], 1e-6)
    gx = np.linspace(xlim[0] - pad_x, xlim[1] + pad_x, 100)
    gy = np.linspace(ylim[0] - pad_y, ylim[1] + pad_y, 100)
    gx_grid, gy_grid = np.meshgrid(gx, gy)
    positions = np.vstack([gx_grid.ravel(), gy_grid.ravel()])
    z = kde(positions).reshape(gx_grid.shape)
    levels = np.linspace(z.max() * 0.12, z.max(), n_levels)
    ax.contour(gx_grid, gy_grid, z, levels=levels, colors=[color], linewidths=FIG4_CURVE_LW, alpha=alpha)
    ax.contourf(gx_grid, gy_grid, z, levels=levels, colors=[color], alpha=0.12)


def _style_posterior_axes(ax: plt.Axes, *, y_ticks: bool = True) -> None:
    style_axes_like_fig4(ax, y_ticks=y_ticks)
    ax.xaxis.label.set_size(FIG3_AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(FIG3_AXIS_LABEL_FONTSIZE)


def _apply_unit_square_ticks(ax: plt.Axes, *, show_x: bool = True, show_y: bool = True) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if show_x:
        ax.set_xticks(UNIT_TICKS)
    else:
        ax.set_xticks(UNIT_TICKS)
        ax.set_xticklabels([])
    if show_y:
        ax.set_yticks(UNIT_TICKS)
    else:
        ax.set_yticks(UNIT_TICKS)
        ax.set_yticklabels([])
    _style_posterior_axes(ax, y_ticks=show_y)


def _set_x_tick_display(ax: plt.Axes, *, show: bool) -> None:
    if show:
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)


def _load_pooled_tarp_curve(exp_dir: Path, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Paper Fig 5 TARP: Lemos curve from calibration/pooled_curves.csv."""

    csv_path = exp_dir / "results" / "calibration" / "pooled_curves.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing Lemos TARP table: {csv_path}")
    df = pd.read_csv(csv_path)
    if "alpha" not in df.columns or "tarp_lemos" not in df.columns:
        raise ValueError(f"Expected columns alpha,tarp_lemos in {csv_path}")

    alphas_src = df["alpha"].to_numpy(dtype=float)
    curve_src = df["tarp_lemos"].to_numpy(dtype=float)
    mu = np.interp(TARP_ALPHAS, alphas_src, curve_src)

    n_obs = 20
    notes = exp_dir / "results" / "calibration" / "run_notes.json"
    if notes.is_file():
        n_obs = int(json.loads(notes.read_text()).get("n_obs", n_obs))

    return TARP_ALPHAS, mu, mu.copy(), mu.copy(), n_obs


def _shared_posterior_limits(all_samples: list[np.ndarray], *, pad_frac: float = 0.08) -> dict[str, tuple[float, float]]:
    combined = np.vstack(all_samples)
    lam = combined[:, 0]
    gam = combined[:, 1]

    def _pad(lo: float, hi: float) -> tuple[float, float]:
        p = pad_frac * max(hi - lo, 1e-6)
        return float(lo - p), float(hi + p)

    lam_xlim = _pad(float(np.nanmin(lam)), float(np.nanmax(lam)))
    gam_xlim = _pad(float(np.nanmin(gam)), float(np.nanmax(gam)))
    return {
        "lam_xlim": lam_xlim,
        "gam_xlim": gam_xlim,
        "joint_xlim": gam_xlim,
        "joint_ylim": lam_xlim,
    }


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    xs = np.sort(values)
    ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
    x_plot = np.concatenate(([0.0], xs, [1.0]))
    y_plot = np.concatenate(([0.0], ys, [1.0]))
    return x_plot, y_plot


def _dkw_band(n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return 1.0
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))


def _four_ticks(xlim: tuple[float, float]) -> np.ndarray:
    lo, hi = xlim
    return np.linspace(lo, hi, 4)


def _apply_fixed_xticks(ax: plt.Axes, ticks: np.ndarray, *, show_labels: bool) -> None:
    ax.set_xticks(ticks)
    if show_labels:
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", labelbottom=False)


def _draw_marginal_panel(
    ax: plt.Axes,
    samples: np.ndarray,
    theta_true: np.ndarray,
    *,
    dim: int,
    color: str,
    xlim: tuple[float, float],
    x_ticks: np.ndarray,
    show_ylabel: bool,
    show_xlabel: bool = False,
    show_x_ticklabels: bool = False,
) -> None:
    vals = samples[:, dim]
    grid = np.linspace(xlim[0], xlim[1], 220)
    density = _kde_1d(vals, grid)
    ax.fill_between(grid, density, color=color, alpha=0.22, linewidth=0)
    ax.plot(grid, density, color=color, linewidth=FIG4_CURVE_LW)
    ax.axvline(float(theta_true[dim]), color=THETA_STAR_COLOR, linewidth=FIG4_CURVE_LW, zorder=5)
    ax.set_xlim(xlim)
    _apply_fixed_xticks(ax, x_ticks, show_labels=show_x_ticklabels)
    if show_xlabel and show_x_ticklabels:
        ax.set_xlabel(PARAM_LABELS[dim], fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=FIG4_GRID_ALPHA)
    _style_posterior_axes(ax, y_ticks=False)


def _draw_joint_panel(
    ax: plt.Axes,
    samples: np.ndarray,
    theta_true: np.ndarray,
    *,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_ticks: np.ndarray,
    show_ylabel: bool,
    show_xlabel: bool,
    show_x_ticklabels: bool = False,
) -> None:
    _kde_2d_contour(ax, samples[:, 1], samples[:, 0], color=color, xlim=xlim, ylim=ylim)
    ax.scatter(
        samples[:, 1],
        samples[:, 0],
        s=4,
        c=color,
        alpha=0.08,
        rasterized=True,
        linewidths=0,
    )
    ax.scatter(
        [float(theta_true[1])],
        [float(theta_true[0])],
        c=THETA_STAR_COLOR,
        s=70,
        marker="*",
        edgecolors="black",
        linewidths=0.5,
        zorder=6,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _apply_fixed_xticks(ax, x_ticks, show_labels=show_x_ticklabels)
    if show_ylabel:
        ax.set_ylabel(PARAM_LABELS[0], fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    if show_xlabel and show_x_ticklabels:
        ax.set_xlabel(PARAM_LABELS[1], fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    else:
        ax.set_xlabel("")
    ax.grid(alpha=FIG4_GRID_ALPHA)
    _style_posterior_axes(ax, y_ticks=show_ylabel)


def _draw_tarp_panel(
    ax: plt.Axes,
    samples: np.ndarray,
    theta_true: np.ndarray,
    *,
    color: str,
    method_label: str,
    show_legend: bool = True,
) -> None:
    curve = bu.tarp_coverage_curve(samples, theta_true, alphas=TARP_ALPHAS)
    ax.plot([0, 1], [0, 1], color=IDEAL_LINE_COLOR, linestyle="--", linewidth=FIG4_CURVE_LW)
    ax.plot(TARP_ALPHAS, curve, color=color, linewidth=FIG4_CURVE_LW)
    _apply_unit_square_ticks(ax)
    ax.set_xlabel(r"Credibility $\alpha$", fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Coverage", fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    ax.grid(alpha=FIG4_GRID_ALPHA)
    _style_posterior_axes(ax)
    if show_legend:
        ax.legend([method_label], fontsize=FIG3_LEGEND_FONTSIZE, loc="upper left", frameon=False)


def _draw_tarp_panel_pooled(
    ax: plt.Axes,
    alphas: np.ndarray,
    mu: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    color: str,
    n_obs: int,
) -> None:
    ax.plot([0, 1], [0, 1], color=IDEAL_LINE_COLOR, linestyle="--", linewidth=FIG4_CURVE_LW)
    ax.plot(alphas, mu, color=color, linewidth=FIG4_CURVE_LW)
    _apply_unit_square_ticks(ax)
    ax.set_xlabel(r"Credibility $\alpha$", fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Coverage", fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    ax.grid(alpha=FIG4_GRID_ALPHA)
    _style_posterior_axes(ax)


def _draw_triangle_block(
    axes: np.ndarray,
    samples: np.ndarray,
    theta_true: np.ndarray,
    *,
    method_label: str,
    color: str,
    limits: dict[str, tuple[float, float]],
    lam_ticks: np.ndarray,
    gam_ticks: np.ndarray,
    tarp_pooled: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None = None,
    show_gamma_x_ticklabels: bool = False,
) -> None:
    """Fill a 2×2 block: Λ marginal, joint, TARP (bottom-left), Γ marginal."""
    _draw_marginal_panel(
        axes[0, 0],
        samples,
        theta_true,
        dim=0,
        color=color,
        xlim=limits["lam_xlim"],
        x_ticks=lam_ticks,
        show_ylabel=True,
        show_x_ticklabels=False,
    )
    _draw_joint_panel(
        axes[0, 1],
        samples,
        theta_true,
        color=color,
        xlim=limits["joint_xlim"],
        ylim=limits["joint_ylim"],
        x_ticks=gam_ticks,
        show_ylabel=True,
        show_xlabel=False,
        show_x_ticklabels=False,
    )
    if tarp_pooled is not None:
        alphas, mu, lo, hi, _n_obs = tarp_pooled
        _draw_tarp_panel_pooled(axes[1, 0], alphas, mu, lo, hi, color=color, n_obs=_n_obs)
    else:
        _draw_tarp_panel(
            axes[1, 0],
            samples,
            theta_true,
            color=color,
            method_label=method_label,
            show_legend=True,
        )
    _draw_marginal_panel(
        axes[1, 1],
        samples,
        theta_true,
        dim=1,
        color=color,
        xlim=limits["gam_xlim"],
        x_ticks=gam_ticks,
        show_ylabel=False,
        show_xlabel=show_gamma_x_ticklabels,
        show_x_ticklabels=show_gamma_x_ticklabels,
    )


def _draw_sbc_ecdf_block(
    axes: list[plt.Axes],
    sbc_df: pd.DataFrame,
    *,
    method: str,
    color: str,
    show_xlabel: bool,
) -> None:
    xx = np.linspace(0.0, 1.0, 200)
    for ax, param_index in zip(axes, (0, 1)):
        ranks = sbc_df[(sbc_df["method"] == method) & (sbc_df["parameter_index"] == param_index)][
            "rank_normalized"
        ].to_numpy(dtype=float)
        n = ranks.size
        x_ecdf, y_ecdf = _ecdf(ranks)
        band = _dkw_band(n)
        ax.plot(xx, xx, color=IDEAL_LINE_COLOR, linestyle="--", linewidth=FIG4_CURVE_LW)
        ax.fill_between(
            xx,
            np.clip(xx - band, 0, 1),
            np.clip(xx + band, 0, 1),
            color=DKW_BAND_COLOR,
            alpha=0.45,
            linewidth=0,
        )
        ax.plot(x_ecdf, y_ecdf, color=color, linewidth=FIG4_CURVE_LW, drawstyle="steps-post")
        _apply_unit_square_ticks(ax, show_x=show_xlabel and param_index == 1, show_y=True)
        ax.grid(alpha=FIG4_GRID_ALPHA)
        if show_xlabel and param_index == 1:
            ax.set_xlabel("SBC rank (normalized)", fontsize=FIG3_AXIS_LABEL_FONTSIZE)
        else:
            ax.set_xlabel("")
            if param_index == 0:
                _set_x_tick_display(ax, show=False)


def _block_centered_title(fig: plt.Figure, axes, letter: str, subtitle: str) -> None:
    add_fig_panel_title_above_block(fig, axes, letter, subtitle)


def plot_posterior_triangle(
    samples: np.ndarray,
    theta_true: np.ndarray,
    *,
    method_label: str,
    color: str,
    limits: dict[str, tuple[float, float]] | None,
    out_stem: Path,
    dpi: int = DPI,
) -> tuple[Path, Path]:
    """2×2 corner layout: Λ marginal, joint, TARP inset, Γ marginal."""
    apply_rcparams()
    if limits is None:
        limits = _shared_posterior_limits([samples])
    lam_ticks = _four_ticks(limits["lam_xlim"])
    gam_ticks = _four_ticks(limits["gam_xlim"])
    fig, axes = plt.subplots(2, 2, figsize=TRIANGLE_BLOCK_SIZE, facecolor="white")
    _draw_triangle_block(
        axes,
        samples,
        theta_true,
        method_label=method_label,
        color=color,
        limits=limits,
        lam_ticks=lam_ticks,
        gam_ticks=gam_ticks,
        show_gamma_x_ticklabels=True,
    )
    fig.suptitle(
        rf"{method_label} posterior ($\Delta T$0.05, $B_{{LF}}=10^4$, $B_{{HF}}=500$, obs 0)",
        fontsize=FIG3_TITLE_FONTSIZE,
        y=1.01,
    )
    handles = [
        Line2D([0], [0], color=color, linewidth=FIG4_CURVE_LW, label=f"{method_label} posterior"),
        Line2D(
            [0],
            [0],
            marker="*",
            color=THETA_STAR_COLOR,
            linestyle="None",
            markersize=9,
            markeredgecolor="black",
            label=r"$\theta^*$",
        ),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.02), frameon=False, fontsize=FIG3_LEGEND_FONTSIZE)
    fig.subplots_adjust(hspace=0.12, wspace=0.12)
    return _save_figure(fig, out_stem, dpi=dpi)


def plot_posterior_triangle_stacked(
    exp_dir: Path,
    method_samples: list[MethodSampleRow],
    theta_true: np.ndarray,
    *,
    out_stem: Path,
    dpi: int = DPI,
) -> tuple[Path, Path]:
    """MF-SBI calibration stacked; shared posterior axes; SBC ECDF panels on the right."""
    apply_rcparams()
    all_samples = [samples for *_rest, samples in method_samples]
    limits = _shared_posterior_limits(all_samples)
    lam_ticks = _four_ticks(limits["lam_xlim"])
    gam_ticks = _four_ticks(limits["gam_xlim"])
    sbc_df = load_calibration_sbc_ranks(exp_dir)
    tarp_by_method: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for method, _label, _color, _stem, _samples in method_samples:
        tarp_by_method[method] = _load_pooled_tarp_curve(exp_dir, method)

    block_w, block_h = TRIANGLE_BLOCK_SIZE
    n_blocks = max(len(method_samples), 1)
    fig = plt.figure(figsize=(block_w * 2, block_h * n_blocks), facecolor="white")
    outer = GridSpec(
        n_blocks,
        2,
        figure=fig,
        width_ratios=[1.04, 0.90],
        height_ratios=[1] * n_blocks,
        wspace=0.22,
        hspace=0.22,
    )

    n_blocks = len(method_samples)

    block_letters = ("a", "b", "c", "d")
    block_pairs: list[tuple[np.ndarray, list[plt.Axes], str, str]] = []

    for block_idx, (method, method_label, color, _stem, samples) in enumerate(method_samples):
        is_last = block_idx == n_blocks - 1
        tri_gs = outer[block_idx, 0].subgridspec(2, 2, wspace=0.14, hspace=0.14)
        tri_axes = np.array([[fig.add_subplot(tri_gs[i, j]) for j in range(2)] for i in range(2)])
        _draw_triangle_block(
            tri_axes,
            samples,
            theta_true,
            method_label=method_label,
            color=color,
            limits=limits,
            lam_ticks=lam_ticks,
            gam_ticks=gam_ticks,
            tarp_pooled=tarp_by_method[method],
            show_gamma_x_ticklabels=is_last,
        )

        sbc_gs = outer[block_idx, 1].subgridspec(2, 1, hspace=0.14)
        sbc_axes = [fig.add_subplot(sbc_gs[i, 0]) for i in range(2)]
        _draw_sbc_ecdf_block(
            sbc_axes,
            sbc_df,
            method=method,
            color=color,
            show_xlabel=is_last,
        )
        for sbc_ax in sbc_axes:
            sbc_ax.set_ylabel("")
        block_pairs.append((tri_axes, sbc_axes, block_letters[2 * block_idx], block_letters[2 * block_idx + 1]))

    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.16)

    for tri_axes, sbc_axes, tri_letter, sbc_letter in block_pairs:
        _block_centered_title(fig, tri_axes, tri_letter, FIG6_BLOCK_TITLES[tri_letter])
        _block_centered_title(fig, sbc_axes, sbc_letter, FIG6_BLOCK_TITLES[sbc_letter])

    all_sbc_axes = [ax for _tri, sbc_axes, _tl, _sl in block_pairs for ax in sbc_axes]
    if all_sbc_axes:
        bboxes = [ax.get_position() for ax in all_sbc_axes]
        x_label = min(b.x0 for b in bboxes) - 0.022
        y_mid = 0.5 * (min(b.y0 for b in bboxes) + max(b.y1 for b in bboxes))
        fig.text(
            x_label,
            y_mid,
            "ECDF",
            rotation=90,
            ha="center",
            va="center",
            fontsize=FIG3_AXIS_LABEL_FONTSIZE,
            color="0.12",
            transform=fig.transFigure,
        )

    handles = [
        Line2D([0], [0], color=METHOD_COLORS["mf_sbi"], linewidth=FIG4_CURVE_LW, label="MF-SBI"),
        Line2D(
            [0],
            [0],
            marker="*",
            color=THETA_STAR_COLOR,
            linestyle="None",
            markersize=9,
            markeredgecolor="black",
            label=r"$\theta^*$",
        ),
        Line2D([0], [0], color=IDEAL_LINE_COLOR, linestyle="--", linewidth=FIG4_CURVE_LW, label="ideal / uniform"),
        Patch(facecolor=DKW_BAND_COLOR, edgecolor="none", alpha=0.45, label="DKW 95% band"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=FIG3_LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 0.01))
    return _save_figure(fig, out_stem, dpi=dpi)


def plot_fig4_posterior_triangles(
    exp_dir: Path,
    *,
    obs_id: int = 0,
    seed: int = 0,
    dpi: int = DPI,
) -> list[tuple[Path, Path]]:
    theta_true = THETA_TRUE.reshape(-1)
    outputs: list[tuple[Path, Path]] = []
    method_samples: list[MethodSampleRow] = []
    for method, label, color, stem in METHOD_SPECS:
        run_dir = _posterior_run_dir(exp_dir, method, obs_id, seed)
        post_path = run_dir / "posterior_samples.csv"
        if not post_path.is_file():
            raise FileNotFoundError(f"Missing posterior samples: {post_path}")
        samples = _load_posterior_xy(post_path)
        method_samples.append((method, label, color, stem, samples))

    limits = _shared_posterior_limits([samples for *_r, samples in method_samples])

    for method, label, color, stem, samples in method_samples:
        out_stem = default_out_stem(exp_dir, stem)
        outputs.append(
            plot_posterior_triangle(
                samples,
                theta_true,
                method_label=label,
                color=color,
                limits=limits,
                out_stem=out_stem,
                dpi=dpi,
            )
        )

    stacked_stem = default_out_stem(exp_dir, STACKED_STEM)
    outputs.append(
        plot_posterior_triangle_stacked(
            exp_dir,
            method_samples,
            theta_true,
            out_stem=stacked_stem,
            dpi=dpi,
        )
    )
    return outputs


def plot_manuscript_fig5(
    exp_dir: Path,
    *,
    obs_id: int = 0,
    seed: int = 0,
    dpi: int = DPI,
    out_stem: Path | None = None,
) -> tuple[Path, Path]:
    """Paper Figure 5 — MF-SBI posterior + TARP/SBC at ΔT_0.05."""
    from scripts.figures._lib.output_formats import manuscript_fig_stem

    theta_true = THETA_TRUE.reshape(-1)
    method_samples: list[MethodSampleRow] = []
    for method, label, color, stem in METHOD_SPECS:
        # Primary observation posterior may live under lf_DT05 (not calibration/)
        run_dir = _posterior_run_dir(exp_dir, method, obs_id, seed)
        post_path = run_dir / "posterior_samples.csv"
        if not post_path.is_file():
            # Fall back to calibration tree for obs_00
            from prism_vertex.benchmark.calibration import calibration_method_dir
            cal = calibration_method_dir(exp_dir, method) / f"obs_{obs_id:02d}" / f"seed_{seed:02d}"
            post_path = cal / "posterior_samples.csv"
            run_dir = cal
        if not post_path.is_file():
            raise FileNotFoundError(f"Missing posterior samples: {post_path}")
        samples = _load_posterior_xy(post_path)
        method_samples.append((method, label, color, stem, samples))

    stem = out_stem or manuscript_fig_stem(exp_dir, 5)
    return plot_posterior_triangle_stacked(
        exp_dir,
        method_samples,
        theta_true,
        out_stem=stem,
        dpi=dpi,
    )


# Back-compat alias
plot_manuscript_fig6_stacked = plot_manuscript_fig5
