#!/usr/bin/env python3
"""Max-budget posterior comparison: 3 HF + 9 MF-SBI LF fidelities.

Outputs under ``data/results/main_method_comparison/`` and root ``figures/``:
  - posterior_reference_grid_4x3.png   — feasibility/posterior panels at max budget
  - metrics_bars_max_budget_6panel.png — NRMSE, CRPS, NLTP, TARP, mean std, 95% CI width
  - nltp_budget_comparison.png         — nltp_exact vs total cost (9 LF curves + HF @ 1000)
  - lf_selection_decision_atlas.png    — Stages A–C pre-inference LF + budget selection

Layout (4 rows × 3 cols):
  row 0: hf_snpe, hf_npe, hf_smc_abc
  row 1: lf_T70, lf_T140, lf_T350
  row 2: lf_DT02, lf_DT05, lf_DT10
  row 3: lf_R1, lf_R2, lf_R5

Example:
  python -m scripts.figures.plot_fig4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

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
from matplotlib.patches import Patch  # noqa: E402

from prism_vertex.benchmark.comparison_paths import (
    fig4_comparison_dir,
    iter_fig4_mf_sbi_runs,
    method_results_dir,
    results_dir,
)
from prism_vertex.simulator.scoreboard_analysis import (
    FIDELITY_COLORS,
    FIDELITY_LINESTYLES,
    LF_ORDER,
)
from prism_vertex.simulator.constants import THETA_TRUE  # noqa: E402
from prism_vertex.metrics.benchmark_unified import (  # noqa: E402
    crps_mean_multivariate,
    tarp_coverage_error_mean,
)
from scripts.figures._lib.panels_fig2 import (  # noqa: E402
    DPI,
    FIG3_AXIS_LABEL_FONTSIZE,
    FIG3_LEGEND_FONTSIZE,
    FIG3_TICK_LABELSIZE,
    FIG3_TITLE_COLOR,
    add_fig_panel_title_above_ax,
    apply_rcparams,
)
from scripts.figures._lib.output_formats import (
    manuscript_fig_stem,
    save_manuscript_submission_formats,
    save_matplotlib_manuscript_png,
)  # noqa: E402
from scripts.figures._lib.color_palette import (  # noqa: E402
    FEAS_HF_BLOCKED_EDGE,
    FEAS_HF_BLOCKED_FILL,
    FEAS_LF_BLOCKED_EDGE,
    FEAS_LF_BLOCKED_FILL,
    FEAS_OK_FILL,
    FEAS_POSTERIOR_EDGE,
    FEAS_POSTERIOR_FILL,
    HF_COLORS,
    OBS_STAR_COLOR,
)
from scripts.figures._lib.fig3_style import (  # noqa: E402
    FIG4_FEAS_MESH_N,
    FIG4_FEAS_STAR_EDGE_LW,
    FIG4_FEAS_STAR_SIZE,
    style_axes_like_fig4,
)

# S1 Fig: full twelve-configuration grid
FIG4_GRID: tuple[tuple[str, ...], ...] = (
    ("hf_snpe", "hf_npe", "hf_smc_abc"),
    ("lf_T70", "lf_T140", "lf_T350"),
    ("lf_DT02", "lf_DT05", "lf_DT10"),
    ("lf_R1", "lf_R2", "lf_R5"),
)

# Main-text Fig 4: three HF baselines + best MF-SBI LF (ΔT_0.05)
# Panel order: (A) HF-SMC-ABC, (B) HF-NPE, (C) HF-SNPE, (D) MF-SBI ΔT_0.05
FIG4_MAIN_GRID: tuple[tuple[str, ...], ...] = (
    ("hf_smc_abc", "hf_npe"),
    ("hf_snpe", "lf_DT05"),
)

ROW_LABELS = ()  # unused; row labels removed from posterior grid

HF_METHODS = frozenset({"hf_snpe", "hf_npe", "hf_smc_abc"})

METHOD_LABELS: dict[str, str] = {
    "hf_snpe": "HF-SNPE",
    "hf_npe": "HF-NPE",
    "hf_smc_abc": "HF-SMC-ABC",
    "lf_T70": "T70",
    "lf_T140": "T140",
    "lf_T350": "T350",
    "lf_DT02": r"$\Delta T$ 0.02",
    "lf_DT05": r"$\Delta T$ 0.05",
    "lf_DT10": r"$\Delta T$ 0.1",
    "lf_R1": "K1",
    "lf_R2": "K2",
    "lf_R5": "K5",
}

HF_MAX_TOTAL_COST = 1000.0


def _to_cost_map(df_eff: pd.DataFrame) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    ccol = "total_cost_metric_value_mean"
    if ccol not in df_eff.columns:
        return out
    for _, r in df_eff.iterrows():
        out[(str(r["method"]), str(r["budget"]))] = float(r[ccol])
    return out


def _prepare_overall_with_cost(acc: pd.DataFrame, c_map: dict[tuple[str, str], float]) -> pd.DataFrame:
    acc = acc[acc["parameter"] == "overall"].copy()
    if acc.empty:
        return acc
    acc["x_cost"] = acc.apply(lambda r: c_map.get((str(r["method"]), str(r["budget"])), np.nan), axis=1)
    return acc[np.isfinite(acc["x_cost"])]


def _safe_set_log_xscale(ax: plt.Axes, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size > 0 and np.all(x > 0):
        ax.set_xscale("log")

META_COLS = {"experiment_id", "experiment_name", "method", "obs_id", "seed", "budget_id", "log_weight"}

XLIM = (-1.45, 0.28)
YLIM = (0.01, 0.205)
GRID_LAM_TICKS = np.linspace(XLIM[0], XLIM[1], 4)
GRID_GAM_TICKS = np.linspace(YLIM[0], YLIM[1], 4)


def _load_feas_module() -> Any:
    mod_path = PROJECT_ROOT / "scripts" / "figures" / "_lib" / "posterior_feasibility.py"
    spec = importlib.util.spec_from_file_location("posterior_feasibility", mod_path)
    assert spec and spec.loader
    feas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feas)
    return feas


def _budget_sort_key(budget: str) -> tuple[int, int]:
    m = re.search(r"budget_hf_(\d+)", budget)
    if m:
        return (0, int(m.group(1)))
    m = re.search(r"budget_lf_(\d+)_hf_(\d+)", budget)
    if m:
        return (1, int(m.group(1)))
    return (9, 0)


def _max_budget_hf_run(exp_dir: Path, method: str, obs_id: int, seed: int) -> tuple[Path, str]:
    method_dir = method_results_dir(exp_dir, method)
    best: tuple[tuple[int, int], Path, str] | None = None
    for budget_dir in sorted(method_dir.glob("budget_hf_*")):
        run_dir = budget_dir / f"obs_{obs_id:02d}" / f"seed_{seed:02d}"
        if not (run_dir / "posterior_samples.csv").is_file():
            continue
        budget = budget_dir.name
        key = _budget_sort_key(budget)
        if best is None or key > best[0]:
            best = (key, run_dir, budget)
    if best is None:
        raise FileNotFoundError(f"No HF posterior for {method} obs={obs_id} seed={seed}")
    return best[1], best[2]


def _max_budget_mf_sbi_lf_run(exp_dir: Path, lf_id: str, obs_id: int, seed: int) -> tuple[Path, str]:
    method_dir = method_results_dir(exp_dir, "mf_sbi")
    best: tuple[tuple[int, int], Path, str] | None = None
    for run_dir, lf, budget in iter_fig4_mf_sbi_runs(method_dir):
        if lf != lf_id:
            continue
        if run_dir.parent.name != f"obs_{obs_id:02d}" or run_dir.name != f"seed_{seed:02d}":
            continue
        if not (run_dir / "posterior_samples.csv").is_file():
            continue
        key = _budget_sort_key(budget)
        if best is None or key > best[0]:
            best = (key, run_dir, budget)
    if best is None:
        raise FileNotFoundError(f"No mf_sbi posterior for {lf_id} obs={obs_id} seed={seed}")
    return best[1], best[2]


def _resolve_series_run(exp_dir: Path, series_id: str, obs_id: int, seed: int) -> tuple[Path, str]:
    if series_id in HF_METHODS:
        return _max_budget_hf_run(exp_dir, series_id, obs_id, seed)
    return _max_budget_mf_sbi_lf_run(exp_dir, series_id, obs_id, seed)


def _series_color(series_id: str) -> str:
    if series_id in HF_COLORS:
        return HF_COLORS[series_id]
    return FIDELITY_COLORS.get(series_id, "#333333")


def _load_posterior_xy(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [c for c in ("lambda", "gamma") if c in df.columns]
    if len(cols) != 2:
        theta_cols = [c for c in df.columns if c not in META_COLS]
        cols = theta_cols[:2]
    return df[cols].to_numpy(dtype=float)


def _load_reference_posterior(exp_dir: Path, obs_id: int) -> np.ndarray | None:
    p_gold = exp_dir / "reference_gold" / f"obs_{obs_id:03d}" / "reference_posterior_hf.csv"
    if p_gold.is_file():
        return _load_posterior_xy(p_gold)
    try:
        run_dir, _ = _max_budget_hf_run(exp_dir, "hf_snpe", obs_id, 0)
        return _load_posterior_xy(run_dir / "posterior_samples.csv")
    except FileNotFoundError:
        return None


def _draw_feasibility_panel(ax: plt.Axes, run_dir: Path, *, ref_samples: np.ndarray | None, subtitle: str = "") -> None:
    feas = _load_feas_module()
    post = feas._load_posterior(run_dir)
    blocked_hf, blocked_lf, _ = feas._load_blocked_by_fidelity(run_dir)
    lam = np.linspace(-1.45, 0.25, FIG4_FEAS_MESH_N)
    gam = np.linspace(0.01, 0.20, FIG4_FEAS_MESH_N)
    xx, yy = np.meshgrid(lam, gam)

    feas._draw_prior_region(ax)
    if ref_samples is not None and ref_samples.shape[0] >= 8:
        z_ref = feas._kde_on_grid(ref_samples, xx, yy)
        if z_ref is not None:
            zmax = float(np.nanmax(z_ref))
            if zmax > 0:
                ax.contour(
                    xx,
                    yy,
                    z_ref / zmax,
                    levels=[0.25, 0.5, 0.75],
                    colors=[FEAS_OK_FILL],
                    linewidths=[0.7, 1.0, 1.2],
                    linestyles="--",
                    alpha=0.75,
                    zorder=4,
                )
    feas._draw_blocked_layer(ax, blocked_hf, xx=xx, yy=yy, style=feas.HF_STYLE, zorder=2)
    feas._draw_blocked_layer(ax, blocked_lf, xx=xx, yy=yy, style=feas.LF_STYLE, zorder=3)
    feas._draw_posterior_layer(ax, post, xx=xx, yy=yy, zorder=5)
    ax.scatter(
        THETA_TRUE[0],
        THETA_TRUE[1],
        marker="*",
        s=FIG4_FEAS_STAR_SIZE,
        c=OBS_STAR_COLOR,
        edgecolors="black",
        linewidths=FIG4_FEAS_STAR_EDGE_LW,
        zorder=7,
    )
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    if subtitle:
        ax.text(
            0.02,
            0.98,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FIG3_TICK_LABELSIZE,
            color="0.35",
        )
    style_axes_like_fig4(ax)


def _apply_grid_feas_ticks(ax: plt.Axes, *, show_x_labels: bool, show_y_labels: bool) -> None:
    ax.set_xticks(GRID_LAM_TICKS)
    ax.set_yticks(GRID_GAM_TICKS)
    ax.set_xticklabels([f"{v:.2f}" for v in GRID_LAM_TICKS] if show_x_labels else [])
    ax.set_yticklabels([f"{v:.3f}" for v in GRID_GAM_TICKS] if show_y_labels else [])
    ax.set_xlabel(r"$\Lambda$", fontsize=FIG3_AXIS_LABEL_FONTSIZE) if show_x_labels else ax.set_xlabel("")
    ax.set_ylabel(r"$\Gamma$", fontsize=FIG3_AXIS_LABEL_FONTSIZE) if show_y_labels else ax.set_ylabel("")
    ax.tick_params(axis="x", labelbottom=show_x_labels)
    ax.tick_params(axis="y", labelleft=show_y_labels)


def _nltp_exact(run_dir: Path, samples: np.ndarray) -> float:
    diag_path = run_dir / "diagnostics.json"
    if diag_path.is_file():
        try:
            diag = json.loads(diag_path.read_text())
            v = diag.get("nltp_exact")
            if v is not None and np.isfinite(float(v)):
                return float(v)
        except Exception:
            pass
    mu = samples.mean(axis=0)
    cov = np.cov(samples.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    diff = THETA_TRUE - mu
    _, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    quad = float(diff.T @ inv @ diff)
    return float(0.5 * (2.0 * np.log(2.0 * np.pi) + logdet + quad))


def _nrmse(samples: np.ndarray) -> float:
    mean_est = samples.mean(axis=0)
    denom = np.maximum(np.abs(THETA_TRUE), 1e-8)
    rel = (mean_est - THETA_TRUE) / denom
    return float(np.sqrt(np.mean(rel**2)))


def _mean_std(samples: np.ndarray) -> float:
    return float(np.mean(np.std(samples, axis=0)))


def _mean_ci_width_95(samples: np.ndarray) -> float:
    widths = []
    for j in range(samples.shape[1]):
        lo, hi = np.quantile(samples[:, j], [0.025, 0.975])
        widths.append(float(hi - lo))
    return float(np.mean(widths))


def _collect_metrics(run_dir: Path, samples: np.ndarray) -> dict[str, float]:
    return {
        "nrmse": _nrmse(samples),
        "crps": crps_mean_multivariate(samples, THETA_TRUE),
        "nltp_exact": _nltp_exact(run_dir, samples),
        "tarp_error": tarp_coverage_error_mean(samples, THETA_TRUE),
        "mean_std": _mean_std(samples),
        "mean_ci_width_95": _mean_ci_width_95(samples),
    }


def plot_posterior_reference_grid(
    exp_dir: Path,
    out_path: Path,
    *,
    obs_id: int = 0,
    seed: int = 0,
    dpi: int = DPI,
    grid: tuple[tuple[str, ...], ...] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Path:
    apply_rcparams()
    grid = grid or FIG4_GRID
    n_rows = len(grid)
    n_cols = len(grid[0])
    if figsize is None:
        figsize = (11.5, 13.5) if n_rows >= 4 else (9.5, 8.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=False, facecolor="white")
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    fig.subplots_adjust(hspace=0.38, wspace=0.28, top=0.90, bottom=0.08, left=0.08, right=0.98)

    panel_letters = "abcdefghijkl"
    panel_idx = 0

    for row_idx, row_ids in enumerate(grid):
        for col_idx, series_id in enumerate(row_ids):
            ax = axes[row_idx, col_idx]
            letter = panel_letters[panel_idx]
            panel_idx += 1
            show_x = row_idx == n_rows - 1
            show_y = col_idx == 0
            try:
                run_dir, _budget = _resolve_series_run(exp_dir, series_id, obs_id, seed)
            except FileNotFoundError:
                ax.text(0.5, 0.5, f"No data\n{series_id}", ha="center", va="center", transform=ax.transAxes)
                add_fig_panel_title_above_ax(fig, ax, letter, METHOD_LABELS.get(series_id, series_id))
                continue
            label = METHOD_LABELS.get(series_id, series_id)
            _draw_feasibility_panel(ax, run_dir, ref_samples=None, subtitle="")
            _apply_grid_feas_ticks(ax, show_x_labels=show_x, show_y_labels=show_y)
            add_fig_panel_title_above_ax(fig, ax, letter, label)

    legend_handles = [
        Patch(facecolor=FEAS_POSTERIOR_FILL, edgecolor=FEAS_POSTERIOR_EDGE, alpha=0.7, label="Posterior"),
        Patch(facecolor=FEAS_HF_BLOCKED_FILL, edgecolor=FEAS_HF_BLOCKED_EDGE, alpha=0.7, label="HF blocked θ"),
        Patch(facecolor=FEAS_LF_BLOCKED_FILL, edgecolor=FEAS_LF_BLOCKED_EDGE, alpha=0.7, label="LF blocked θ"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=OBS_STAR_COLOR,
            markeredgecolor="black",
            markersize=14,
            label=r"$\theta^{*}$",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        fontsize=11,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.995),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_matplotlib_manuscript_png(
        fig,
        out_path,
        save_kwargs=dict(bbox_inches="tight"),
    )
    save_manuscript_submission_formats(fig, out_path.with_suffix(""), dpi=dpi)
    plt.close(fig)
    return out_path


def _hf_nltp_at_max_budget(exp_dir: Path, method: str, obs_id: int, seed: int) -> float:
    run_dir, _ = _max_budget_hf_run(exp_dir, method, obs_id, seed)
    samples = _load_posterior_xy(run_dir / "posterior_samples.csv")
    return _nltp_exact(run_dir, samples)


def _plot_lf_nltp_series(ax: plt.Axes, m: pd.DataFrame, lf_id: str) -> None:
    mx = m["x_cost"].to_numpy(dtype=float)
    my = m["metric_value_mean"].to_numpy(dtype=float)
    kwargs: dict[str, Any] = {
        "label": lf_id,
        "linewidth": 1.6,
        "alpha": 0.95,
        "linestyle": FIDELITY_LINESTYLES.get(lf_id, "-"),
        "color": FIDELITY_COLORS.get(lf_id, "#333333"),
    }
    if "metric_value_std" in m.columns:
        std = m["metric_value_std"].fillna(0.0).to_numpy(dtype=float)
        nrep = m.get("n_replicates", pd.Series([1] * len(m))).fillna(1.0).to_numpy(dtype=float)
        se = std / np.sqrt(np.maximum(nrep, 1.0))
        ax.errorbar(mx, my, yerr=se, fmt="o", capsize=2, **kwargs)
    else:
        ax.plot(mx, my, "o", **kwargs)


def plot_nltp_budget_comparison(
    exp_dir: Path,
    out_path: Path,
    *,
    obs_id: int = 0,
    seed: int = 0,
    dpi: int = 170,
) -> Path:
    comparison_dir = fig4_comparison_dir(exp_dir)
    p_acc = comparison_dir / "accuracy_metrics.csv"
    p_eff = comparison_dir / "efficiency_metrics.csv"
    if not p_acc.is_file() or not p_eff.is_file():
        raise FileNotFoundError(f"Missing Fig.4 metrics under {comparison_dir}")

    acc = pd.read_csv(p_acc)
    eff = pd.read_csv(p_eff)
    acc = _prepare_overall_with_cost(acc, _to_cost_map(eff))
    sub = acc[acc["metric_name"] == "nltp_exact"]
    if sub.empty:
        raise FileNotFoundError("No nltp_exact rows in accuracy_metrics.csv")

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for lf_id in LF_ORDER:
        m = sub[sub["method"] == lf_id].sort_values("x_cost")
        if not m.empty:
            _plot_lf_nltp_series(ax, m, lf_id)

    xvals = sub["x_cost"].to_numpy(dtype=float)
    x_lo = float(np.nanmin(xvals)) * 0.98 if np.isfinite(xvals).any() else 400.0
    x_hi = HF_MAX_TOTAL_COST
    for method in ("hf_snpe", "hf_npe", "hf_smc_abc"):
        try:
            nltp = _hf_nltp_at_max_budget(exp_dir, method, obs_id, seed)
        except FileNotFoundError:
            continue
        if not np.isfinite(nltp):
            continue
        ax.hlines(
            nltp,
            x_lo,
            x_hi,
            colors=HF_COLORS[method],
            linewidth=1.8,
            linestyles="--",
            alpha=0.95,
            label=METHOD_LABELS[method],
        )

    _safe_set_log_xscale(ax, np.append(xvals, x_hi))
    ax.set_xlim(left=x_lo * 0.95, right=x_hi * 1.02)
    ax.set_xlabel("Total cost")
    ax.set_ylabel("nltp_exact")
    ax.set_title("nltp_exact vs cost")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metrics_bars_max_budget(
    exp_dir: Path,
    out_path: Path,
    *,
    obs_id: int = 0,
    seed: int = 0,
    dpi: int = 170,
) -> Path:
    series_flat = [sid for row in FIG4_GRID for sid in row]
    metrics_by_id: dict[str, dict[str, float]] = {}
    budgets: dict[str, str] = {}
    for series_id in series_flat:
        try:
            run_dir, budget = _resolve_series_run(exp_dir, series_id, obs_id, seed)
            samples = _load_posterior_xy(run_dir / "posterior_samples.csv")
            metrics_by_id[series_id] = _collect_metrics(run_dir, samples)
            budgets[series_id] = budget
        except FileNotFoundError:
            continue

    if not metrics_by_id:
        raise FileNotFoundError("No max-budget posteriors found for any of the 12 series.")

    metric_panels = [
        ("nrmse", "NRMSE", "posterior mean vs truth"),
        ("crps", "CRPS", "mean over dims"),
        ("nltp_exact", r"NLTP$_{\mathrm{exact}}$", r"$-\log p(\theta^*\mid x_o)$"),
        ("tarp_error", "TARP coverage error", "mean over dims"),
        ("mean_std", "Mean posterior std", r"avg($\Lambda$,$\Gamma$)"),
        ("mean_ci_width_95", "Mean 95% CI width", "mean over dims"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), constrained_layout=True)
    axes_flat = axes.ravel()
    x = np.arange(len(series_flat))
    colors = [_series_color(sid) for sid in series_flat]
    labels = [METHOD_LABELS.get(sid, sid) for sid in series_flat]

    for ax, (key, title, ylab) in zip(axes_flat, metric_panels):
        vals = [metrics_by_id.get(sid, {}).get(key, np.nan) for sid in series_flat]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.35, width=0.78)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=6)

        # Row separators (HF | T | DT | R)
        for sep in (2.5, 5.5, 8.5):
            ax.axvline(sep, color="0.75", linestyle=":", linewidth=0.9)

    fig.suptitle(
        "Max-budget posterior metrics — 3 HF + 9 mf_sbi LF fidelities",
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.01,
        " | ".join(f"{METHOD_LABELS.get(s, s)}: {budgets[s]}" for s in series_flat if s in budgets),
        ha="center",
        fontsize=6.5,
        color="0.4",
        wrap=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--obs_id", type=int, default=0)
    ap.add_argument("--seed_id", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument(
        "--with-extras",
        action="store_true",
        help="Also write auxiliary comparison plots outside all/fig4",
    )
    args = ap.parse_args()

    exp_dir = results_dir(args.results_root)
    out_dir = fig4_comparison_dir(exp_dir) / "plots" / "max_budget_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = plot_posterior_reference_grid(
        exp_dir,
        manuscript_fig_stem(exp_dir, 4).with_suffix(".png"),
        obs_id=args.obs_id,
        seed=args.seed_id,
        dpi=args.dpi,
    )
    print(f"Wrote {p1}")

    if not args.with_extras:
        return

    p2 = plot_metrics_bars_max_budget(
        exp_dir,
        out_dir / "metrics_bars_max_budget_6panel.png",
        obs_id=args.obs_id,
        seed=args.seed_id,
        dpi=args.dpi,
    )
    p_nltp = plot_nltp_budget_comparison(
        exp_dir,
        out_dir / "nltp_budget_comparison.png",
        obs_id=args.obs_id,
        seed=args.seed_id,
        dpi=args.dpi,
    )
    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from scripts.figures._lib.lf_selection_atlas import plot_lf_selection_atlas_v2

    p3 = plot_lf_selection_atlas_v2(
        exp_dir,
        out_dir / "lf_selection_decision_atlas.png",
        dpi=args.dpi,
    )
    print(f"Wrote {p2}")
    print(f"Wrote {p_nltp}")
    print(f"Wrote {p3}")


if __name__ == "__main__":
    main()
