#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.figures._lib.color_palette import FEAS_POSTERIOR_EDGE, FEAS_POSTERIOR_FILL, OBS_STAR_COLOR
from scripts.figures._lib.fig3_style import FIG4_FEAS_STAR_EDGE_LW, style_axes_like_fig4
from scripts.figures._lib.output_formats import (
    manuscript_fig_stem,
    save_matplotlib_manuscript_png,
    save_manuscript_submission_formats,
)
from scripts.figures._lib.panels_fig2 import DPI, apply_rcparams, add_fig_panel_title_above_ax

DATA_ROOT = PROJECT_ROOT / "data" / "nonproliferating_vertex"
RUNS_CSV = DATA_ROOT / "results" / "metrics" / "fig7_series_index.csv"
THETA_HAT = DATA_ROOT / "reference" / "theta_hat_w165.json"
PARAM_TEX = (r"$a_r$", r"$b_0$", r"$k$")
PAIR_COLS = ((0, 1), (0, 2), (1, 2))


def _load_theta_hat() -> np.ndarray:
    data = json.loads(THETA_HAT.read_text())
    th = np.asarray(data["theta_hat"], dtype=float).reshape(-1)
    if th.size < 3:
        raise ValueError(f"theta_hat must have 3 entries: {THETA_HAT}")
    return th[:3]


def _load_runs() -> list[dict[str, str]]:
    with RUNS_CSV.open() as f:
        return list(csv.DictReader(f))


def _load_samples(run_dir: Path) -> np.ndarray:
    path = run_dir / "posterior_samples.csv"
    df = pd.read_csv(path)
    cols = [c for c in ("a_r", "b_0", "k") if c in df.columns]
    if len(cols) != 3:
        # fallback: first three non-meta numeric columns
        skip = {"experiment_id", "experiment_name", "method", "obs_id", "seed", "budget_id", "log_weight"}
        cols = [c for c in df.columns if c not in skip][:3]
    return df[cols].to_numpy(dtype=float)


def _kde_contour(ax, samples: np.ndarray, i: int, j: int, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    xy = samples[:, [i, j]]
    if len(xy) < 20:
        ax.scatter(xy[:, 0], xy[:, 1], s=4, c=FEAS_POSTERIOR_EDGE, alpha=0.4)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return
    try:
        kde = gaussian_kde(xy.T)
        xx, yy = np.meshgrid(np.linspace(*xlim, 90), np.linspace(*ylim, 90))
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=6, cmap="Greens", alpha=0.85)
        ax.contour(xx, yy, zz, levels=6, colors=FEAS_POSTERIOR_EDGE, linewidths=0.4)
    except Exception:
        ax.scatter(xy[:, 0], xy[:, 1], s=4, c=FEAS_POSTERIOR_EDGE, alpha=0.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _best_t50_samples() -> np.ndarray:
    """Lowest-NLTP seed of MF-SBI with T50% at its largest low-fidelity budget."""
    metrics = pd.read_csv(DATA_ROOT / "results" / "metrics" / "runs_metrics.csv")
    sub = metrics[(metrics["method"] == "mf_sbi") & (metrics["lf_id"] == "lf_T05")].copy()
    if sub.empty:
        raise FileNotFoundError("No MF-SBI T50% rows in runs_metrics.csv")
    bmax = sub["budget_lf"].max()
    sub = sub[sub["budget_lf"] == bmax]
    row = sub.loc[sub["nltp"].idxmin()]
    budget = int(row["budget_lf"])
    seed = int(row["seed"])
    run_dir = (
        DATA_ROOT
        / "results"
        / "posterior_runs"
        / "mf_sbi"
        / "lf_T05"
        / f"budget_lf_{budget:06d}_hf_000500"
        / "obs_00"
        / f"seed_{seed:02d}"
    )
    return _load_samples(run_dir)


def _pair_limits(samples_list: list[np.ndarray], i: int, j: int, theta: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = np.concatenate([s[:, i] for s in samples_list])
    ys = np.concatenate([s[:, j] for s in samples_list])
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    x0, x1 = min(x0, theta[i]), max(x1, theta[i])
    y0, y1 = min(y0, theta[j]), max(y1, theta[j])
    px = 0.06 * (x1 - x0 + 1e-9)
    py = 0.06 * (y1 - y0 + 1e-9)
    return (float(x0 - px), float(x1 + px)), (float(y0 - py), float(y1 + py))


def plot_fig7(*, dpi: int = DPI) -> tuple[Path, Path]:
    apply_rcparams()
    theta_hat = _load_theta_hat()
    runs = _load_runs()
    # Stable row order matching paper
    order = ["hf_smc_abc", "hf_npe", "hf_snpe", "lf_T05"]
    by_id = {r["series_id"]: r for r in runs}
    rows = [by_id[k] for k in order if k in by_id]
    if len(rows) != 4:
        raise FileNotFoundError(f"Expected 4 series in {RUNS_CSV}, got {list(by_id)}")

    fig, axes = plt.subplots(4, 3, figsize=(10.8, 12.4), facecolor="white")
    fig.subplots_adjust(hspace=0.42, wspace=0.32, top=0.96, bottom=0.08, left=0.20, right=0.98)
    row_titles = ["HF-SMC-ABC", "HF-NPE", "HF-SNPE", r"MF-SBI, $T_{50\%}$"]
    loaded: list[np.ndarray] = []
    for row in rows:
        if row["series_id"] == "lf_T05":
            loaded.append(_best_t50_samples())
        else:
            loaded.append(_load_samples(PROJECT_ROOT / row["run_dir"]))
    limits = [_pair_limits(loaded, i, j, theta_hat) for i, j in PAIR_COLS]
    letters = "abcdefghijkl"
    idx = 0
    for r_i, samples in enumerate(loaded):
        for c_i, (i, j) in enumerate(PAIR_COLS):
            ax = axes[r_i, c_i]
            xlim, ylim = limits[c_i]
            _kde_contour(ax, samples, i, j, xlim, ylim)
            ax.plot(
                theta_hat[i],
                theta_hat[j],
                marker="*",
                markersize=12,
                color=OBS_STAR_COLOR,
                markeredgecolor="black",
                markeredgewidth=FIG4_FEAS_STAR_EDGE_LW,
                linestyle="None",
                zorder=5,
            )
            style_axes_like_fig4(ax)
            ax.set_box_aspect(1)
            if r_i == 3:
                ax.set_xlabel(PARAM_TEX[i], fontsize=10)
            if c_i == 0:
                ax.set_ylabel(PARAM_TEX[j], fontsize=10)
            add_fig_panel_title_above_ax(fig, ax, letters[idx], "")
            idx += 1
        axes[r_i, 0].annotate(
            row_titles[r_i],
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-78, 0),
            textcoords="offset points",
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="0.12",
            annotation_clip=False,
        )
    fig.legend(
        handles=[
            Patch(facecolor=FEAS_POSTERIOR_FILL, edgecolor=FEAS_POSTERIOR_EDGE, label="Posterior"),
            Line2D(
                [0], [0], marker="*", color="none", linestyle="None",
                markerfacecolor=OBS_STAR_COLOR, markeredgecolor="black",
                markersize=12, label=r"$\hat\theta$",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.55, 0.005),
    )

    stem = manuscript_fig_stem(PROJECT_ROOT / "data" / "proliferating_vertex", 7)
    png = stem.with_suffix(".png")
    save_matplotlib_manuscript_png(fig, png, save_kwargs=dict(bbox_inches="tight"))
    save_manuscript_submission_formats(fig, stem, dpi=dpi)
    plt.close(fig)
    return png, stem.with_suffix(".pdf")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=DPI)
    args = ap.parse_args(argv)
    png, pdf = plot_fig7(dpi=args.dpi)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
