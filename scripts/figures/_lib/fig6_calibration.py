"""Shared style and plotting helpers for Figure 5 (calibration at operating point)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from prism_vertex.benchmark.calibration import CALIBRATION_BUDGET_DIR, CALIBRATION_LF_ID, calibration_comparison_dir
from prism_vertex.benchmark.comparison_paths import fig4_comparison_dir, method_results_dir
from prism_vertex.simulator.calibration_observations import (
    CALIBRATION_OBS_IDS,
    theta_for_obs,
)
from prism_vertex.simulator.prior_region import (
    GAM_MAX,
    GAM_MIN,
    LAM_MAX,
    LAM_MIN,
    gamma_ab_boundary,
    lambda_bc_boundary,
)
from prism_vertex.simulator.constants import THETA_TRUE
from prism_vertex.metrics import benchmark_unified as bu
from scripts.figures._lib.panels_fig2 import DPI, FIG_PAD_IN, _save_figure, apply_rcparams
from scripts.figures._lib.color_palette import METHOD_COLORS, REGION_B_EDGE, THETA_STAR_COLOR

FIG5_OUT_DIR = "fig 5"
FIG5_PANEL_SIZE = (6.4, 5.4)
FIG5_SBC_PANEL_SIZE = (6.4, 7.2)

METHOD_ORDER = ("mf_sbi",)
METHOD_LABELS = {
    "mf_sbi": "MF-SBI",
    }
PARAM_LABELS = {
    "lambda": r"$\Lambda$",
    "gamma": r"$\Gamma$",
}
PARAM_INDEX_TO_NAME = {0: "lambda", 1: "gamma"}

PANEL_TITLES = {
    "a": r"$\mathbf{(a)}$ Calibration observations",
    "b": r"$\mathbf{(b)}$ TARP coverage",
    "c": r"$\mathbf{(c)}$ TARP coverage error",
    "d": r"$\mathbf{(d)}$ SBC rank ECDF",
}


def default_out_stem(exp_dir: Path, panel: str) -> Path:
    names = {
        "a": "fig5_a_obs_map",
        "b": "fig5_b_tarp_curve",
        "c": "fig5_c_tarp_coverage_error",
        "d": "fig5_d_sbc_ecdf",
        "abcd": "fig5_abcd",
    }
    return fig4_comparison_dir(exp_dir) / "plots" / "max_budget_comparison" / FIG5_OUT_DIR / names[panel]


def _cal_run_dir(exp_dir: Path, method: str, obs_id: int, seed: int = 0) -> Path:
    return (
        method_results_dir(exp_dir, method)
        / "calibration"
        / CALIBRATION_LF_ID
        / CALIBRATION_BUDGET_DIR
        / f"obs_{obs_id:02d}"
        / f"seed_{seed:02d}"
    )


def calibration_obs_ids(exp_dir: Path) -> list[int]:
    """Observations where MF-SBI produced posterior samples."""
    ok: list[int] = []
    for obs_id in CALIBRATION_OBS_IDS:
        if all((_cal_run_dir(exp_dir, method, obs_id) / "posterior_samples.csv").is_file() for method in METHOD_ORDER):
            ok.append(obs_id)
    return ok


def _theta_true(exp_dir: Path, obs_id: int) -> np.ndarray:
    for p in (
        exp_dir / "observations" / f"obs_{obs_id:02d}" / "theta_true.npy",
        exp_dir / "observations" / f"theta_{obs_id:02d}.npy",
    ):
        if p.is_file():
            return np.load(p).reshape(-1)
    if obs_id == 0 and (exp_dir / "observations" / "theta_true.npy").is_file():
        return np.load(exp_dir / "observations" / "theta_true.npy").reshape(-1)
    return theta_for_obs(obs_id)


def _load_posterior_samples(run_dir: Path) -> np.ndarray:
    df = pd.read_csv(run_dir / "posterior_samples.csv")
    cols = [c for c in ("lambda", "gamma") if c in df.columns]
    if len(cols) != 2:
        theta_cols = [c for c in df.columns if c.startswith("theta_") or c in ("lambda", "gamma")]
        if not theta_cols:
            theta_cols = [c for c in df.columns if c not in ("log_weight",)]
        cols = theta_cols[:2]
    return df[cols].to_numpy(dtype=float)


def load_calibration_posteriors(exp_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for obs_id in calibration_obs_ids(exp_dir):
        for method in METHOD_ORDER:
            run_dir = _cal_run_dir(exp_dir, method, obs_id)
            rows.append(
                {
                    "method": method,
                    "obs_id": obs_id,
                    "samples": _load_posterior_samples(run_dir),
                    "theta_true": _theta_true(exp_dir, obs_id),
                }
            )
    return pd.DataFrame(rows)


def _sbc_metrics_dir(exp_dir: Path) -> Path:
    cal = calibration_comparison_dir(exp_dir)
    if (cal / "sbc_rank_metrics.csv").is_file():
        return cal
    
    return dt05_mf_comparison_dir(exp_dir)


def load_calibration_sbc_ranks(exp_dir: Path) -> pd.DataFrame:
    p = _sbc_metrics_dir(exp_dir) / "sbc_rank_metrics.csv"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. Run: python scripts/inference/compute_metrics.py "
            "--experiment proliferating_vertex --calibration_only"
        )
    df = pd.read_csv(p)
    df["parameter"] = df["parameter_index"].map(PARAM_INDEX_TO_NAME)
    keep = set(calibration_obs_ids(exp_dir))
    df = df[df["obs_id"].isin(keep) & df["method"].isin(METHOD_ORDER)].copy()
    return df


def _draw_region_b(ax: plt.Axes) -> None:
    xx, yy = np.meshgrid(np.linspace(-1.5, 0.5, 450), np.linspace(0.0, 0.2, 300))
    mask = (yy >= np.maximum(0.0, gamma_ab_boundary(xx))) & (xx <= lambda_bc_boundary(yy))
    ax.contourf(xx, yy, mask.astype(float), levels=[0.5, 1.5], colors=[REGION_B_EDGE], alpha=0.95)
    x = np.linspace(-1.5, 0.5, 500)
    ax.plot(x, gamma_ab_boundary(x), color="black", lw=1.2)
    y = np.linspace(0.0, 0.2, 400)
    ax.plot(lambda_bc_boundary(y), y, color="black", lw=1.2)
    ax.plot(
        [LAM_MIN, LAM_MAX, LAM_MAX, LAM_MIN, LAM_MIN],
        [GAM_MIN, GAM_MIN, GAM_MAX, GAM_MAX, GAM_MIN],
        color="#444444",
        lw=1.2,
        ls="--",
    )
    ax.set_xlim(-1.5, 0.35)
    ax.set_ylim(0.0, 0.205)


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


def plot_fig5_a_obs_map(exp_dir: Path, *, out_stem: Path, dpi: int = DPI) -> tuple[Path, Path]:
    apply_rcparams()
    obs_ids = calibration_obs_ids(exp_dir)
    n_obs = len(obs_ids)
    theta = np.asarray([theta_for_obs(i) for i in obs_ids], dtype=float)

    fig, ax = plt.subplots(figsize=FIG5_PANEL_SIZE, facecolor="white")
    _draw_region_b(ax)
    ax.text(-1.2, 0.060, "A", fontsize=11, color="#888888")
    ax.text(-0.42, 0.10, "B", fontsize=11)
    ax.text(0.18, 0.10, "C", fontsize=11, color="#888888")

    ax.scatter(
        theta[:, 0],
        theta[:, 1],
        s=38,
        c="#1F77B4",
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
    )
    ax.scatter(
        THETA_TRUE[0],
        THETA_TRUE[1],
        s=85,
        c=THETA_STAR_COLOR,
        marker="x",
        linewidths=2,
        zorder=5,
    )

    legend_handles = [
        Patch(facecolor=REGION_B_EDGE, edgecolor="none", alpha=0.95, label="Region B (stable packings)"),
        Line2D([0], [0], color="#444444", lw=1.2, ls="--", label="Operational prior box"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1F77B4",
            markeredgecolor="white",
            markersize=7,
            label=rf"Calibration scenarios ($N={n_obs}$)",
        ),
        Line2D([0], [0], marker="x", color=THETA_STAR_COLOR, linestyle="None", markersize=8, label=r"$\theta^*$"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.15)
    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel(r"$\Gamma$")
    ax.set_title(PANEL_TITLES["a"] + rf" ($N={n_obs}$)", fontsize=11, pad=8)
    fig.tight_layout()
    return _save_figure(fig, out_stem, dpi=dpi)


def plot_fig5_b_tarp_curve(exp_dir: Path, *, out_stem: Path, dpi: int = DPI) -> tuple[Path, Path]:
    apply_rcparams()
    posts = load_calibration_posteriors(exp_dir)
    alphas = np.linspace(0.02, 0.98, 24)
    n_obs = len(calibration_obs_ids(exp_dir))

    fig, ax = plt.subplots(figsize=FIG5_PANEL_SIZE, facecolor="white")
    ax.plot([0, 1], [0, 1], color="0.25", linestyle="--", linewidth=1.2, label="ideal")

    for method in METHOD_ORDER:
        sub = posts[posts["method"] == method]
        if sub.empty:
            continue
        curves: list[np.ndarray] = []
        for _, row in sub.iterrows():
            curves.append(bu.tarp_coverage_curve(row["samples"], row["theta_true"], alphas=alphas))
        arr = np.asarray(curves, dtype=float)
        mu = np.nanmean(arr, axis=0)
        lo = np.nanquantile(arr, 0.20, axis=0)
        hi = np.nanquantile(arr, 0.80, axis=0)
        color = METHOD_COLORS[method]
        ax.fill_between(alphas, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(alphas, mu, color=color, linewidth=2.4, label=METHOD_LABELS[method])

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Credibility level $\alpha$")
    ax.set_ylabel("Empirical coverage probability")
    ax.set_title(PANEL_TITLES["b"], fontsize=11, pad=8)
    ax.text(
        0.98,
        0.04,
        rf"Pooled over $N={n_obs}$ scenarios",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    return _save_figure(fig, out_stem, dpi=dpi)


def plot_fig5_c_tarp_coverage_error(exp_dir: Path, *, out_stem: Path, dpi: int = DPI) -> tuple[Path, Path]:
    apply_rcparams()
    posts = load_calibration_posteriors(exp_dir)
    n_obs = len(calibration_obs_ids(exp_dir))

    fig, ax = plt.subplots(figsize=FIG5_PANEL_SIZE, facecolor="white")
    x = np.arange(len(METHOD_ORDER), dtype=float)
    means: list[float] = []
    stds: list[float] = []
    for method in METHOD_ORDER:
        sub = posts[posts["method"] == method]
        errs = [
            bu.tarp_coverage_error_mean(row["samples"], row["theta_true"])
            for _, row in sub.iterrows()
        ]
        means.append(float(np.nanmean(errs)) if errs else np.nan)
        stds.append(float(np.nanstd(errs)) if errs else np.nan)

    bars = ax.bar(
        x,
        means,
        width=0.55,
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        edgecolor="0.15",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="0.2", elinewidth=1.0, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Mean TARP coverage error")
    ax.set_ylim(0.0, max(0.35, float(np.nanmax(means)) * 1.25 + 0.04))
    ax.set_title(PANEL_TITLES["c"], fontsize=11, pad=8)
    ax.text(
        0.98,
        0.96,
        rf"Mean $\pm$ s.d. over $N={n_obs}$ scenarios",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.35",
    )
    ax.grid(axis="y", alpha=0.22)
    for rect, val in zip(bars, means):
        if np.isfinite(val):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.008,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    return _save_figure(fig, out_stem, dpi=dpi)


def plot_fig5_d_sbc_ecdf(exp_dir: Path, *, out_stem: Path, dpi: int = DPI) -> tuple[Path, Path]:
    apply_rcparams()
    df = load_calibration_sbc_ranks(exp_dir)
    n_obs = len(calibration_obs_ids(exp_dir))

    fig, axs = plt.subplots(2, 1, figsize=FIG5_SBC_PANEL_SIZE, facecolor="white", sharex=True, sharey=True)
    xx = np.linspace(0.0, 1.0, 200)

    for row, param_index in enumerate((0, 1)):
        ax = axs[row]
        param_name = PARAM_INDEX_TO_NAME[param_index]
        n_pool = 0
        for method in METHOD_ORDER:
            ranks = df[(df["method"] == method) & (df["parameter_index"] == param_index)]["rank_normalized"].to_numpy(
                dtype=float
            )
            n_pool = max(n_pool, ranks.size)
            x_ecdf, y_ecdf = _ecdf(ranks)
            ax.plot(
                x_ecdf,
                y_ecdf,
                color=METHOD_COLORS[method],
                linewidth=2.2,
                drawstyle="steps-post",
                label=METHOD_LABELS[method],
            )
        band = _dkw_band(n_pool)
        ax.plot(xx, xx, color="0.25", linestyle="--", linewidth=1.0, label="uniform" if row == 0 else None)
        ax.fill_between(
            xx,
            np.clip(xx - band, 0, 1),
            np.clip(xx + band, 0, 1),
            color="0.82",
            alpha=0.45,
            linewidth=0,
            label="DKW 95% band" if row == 0 else None,
        )
        ax.set_ylabel("ECDF")
        ax.set_title(PARAM_LABELS[param_name], fontsize=10, loc="left", pad=6)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)

    axs[-1].set_xlabel("SBC rank (normalized)")
    axs[0].set_title(PANEL_TITLES["d"], fontsize=11, pad=10)
    axs[0].text(
        0.98,
        0.04,
        rf"$N={n_obs}$ scenarios $\times$ 2 parameters",
        transform=axs[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    handles = [
        Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.0, label="uniform ECDF"),
        Patch(facecolor="0.82", edgecolor="none", alpha=0.45, label="DKW 95% band"),
        *[
            Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2.2, label=METHOD_LABELS[m])
            for m in METHOD_ORDER
        ],
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.90, hspace=0.28)
    return _save_figure(fig, out_stem, dpi=dpi)


def plot_fig5_abcd(exp_dir: Path, *, out_stem: Path, dpi: int = DPI) -> tuple[Path, Path]:
    apply_rcparams()
    fig = plt.figure(figsize=(12.8, 10.8), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    # Panel (a)
    ax_a = fig.add_subplot(gs[0, 0])
    obs_ids = calibration_obs_ids(exp_dir)
    theta = np.asarray([theta_for_obs(i) for i in obs_ids], dtype=float)
    _draw_region_b(ax_a)
    ax_a.scatter(theta[:, 0], theta[:, 1], s=30, c="#1F77B4", edgecolors="white", linewidths=0.5, zorder=4)
    ax_a.scatter(THETA_TRUE[0], THETA_TRUE[1], s=70, c=THETA_STAR_COLOR, marker="x", linewidths=1.8, zorder=5)
    ax_a.set_title(PANEL_TITLES["a"] + rf" ($N={len(obs_ids)}$)", fontsize=10)
    ax_a.grid(alpha=0.15)

    # Panel (b)
    ax_b = fig.add_subplot(gs[0, 1])
    posts = load_calibration_posteriors(exp_dir)
    alphas = np.linspace(0.02, 0.98, 24)
    ax_b.plot([0, 1], [0, 1], color="0.25", linestyle="--", linewidth=1.0)
    for method in METHOD_ORDER:
        sub = posts[posts["method"] == method]
        curves = [bu.tarp_coverage_curve(r["samples"], r["theta_true"], alphas=alphas) for _, r in sub.iterrows()]
        arr = np.asarray(curves, dtype=float)
        ax_b.plot(alphas, np.nanmean(arr, axis=0), color=METHOD_COLORS[method], lw=2.0, label=METHOD_LABELS[method])
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.set_title(PANEL_TITLES["b"], fontsize=10)
    ax_b.legend(fontsize=7, loc="upper left")
    ax_b.grid(alpha=0.2)

    # Panel (c)
    ax_c = fig.add_subplot(gs[1, 0])
    x = np.arange(len(METHOD_ORDER))
    means, stds = [], []
    for method in METHOD_ORDER:
        sub = posts[posts["method"] == method]
        errs = [bu.tarp_coverage_error_mean(r["samples"], r["theta_true"]) for _, r in sub.iterrows()]
        means.append(float(np.nanmean(errs)))
        stds.append(float(np.nanstd(errs)))
    ax_c.bar(x, means, width=0.55, color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.9)
    ax_c.errorbar(x, means, yerr=stds, fmt="none", ecolor="0.2", capsize=3)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8)
    ax_c.set_title(PANEL_TITLES["c"], fontsize=10)
    ax_c.grid(axis="y", alpha=0.2)

    # Panel (d)
    ax_d = fig.add_subplot(gs[1, 1])
    df = load_calibration_sbc_ranks(exp_dir)
    xx = np.linspace(0.0, 1.0, 200)
    for method in METHOD_ORDER:
        ranks = df[(df["method"] == method)]["rank_normalized"].to_numpy(dtype=float)
        x_ecdf, y_ecdf = _ecdf(ranks)
        ax_d.plot(x_ecdf, y_ecdf, color=METHOD_COLORS[method], lw=2.0, drawstyle="steps-post", label=METHOD_LABELS[method])
    n_pool = len(df)
    band = _dkw_band(n_pool)
    ax_d.plot(xx, xx, color="0.25", linestyle="--", linewidth=1.0)
    ax_d.fill_between(xx, np.clip(xx - band, 0, 1), np.clip(xx + band, 0, 1), color="0.82", alpha=0.45, linewidth=0)
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.set_title(PANEL_TITLES["d"] + " (pooled)", fontsize=10)
    ax_d.legend(fontsize=7, loc="lower right")
    ax_d.grid(alpha=0.2)

    for ax, xlab, ylab in (
        (ax_a, r"$\Lambda$", r"$\Gamma$"),
        (ax_b, r"$\alpha$", "Coverage"),
        (ax_c, "", "TARP error"),
        (ax_d, "SBC rank", "ECDF"),
    ):
        if xlab:
            ax.set_xlabel(xlab, fontsize=9)
        if ylab:
            ax.set_ylabel(ylab, fontsize=9)

    return _save_figure(fig, out_stem, dpi=dpi)
