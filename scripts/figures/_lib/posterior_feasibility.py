#!/usr/bin/env python3
"""Posterior (green) overlaid on inference-time blocked θ by fidelity.

Paper methods: ``mf_sbi`` and ``fuse`` (Fig. 4 / Fig. 5 feasibility panels).

Reads per-run artifacts:
  posterior_samples.csv
  failure_registry_inference.json   # blocked_theta with per-entry fidelity (hf/lf)
  diagnostics.json                  # round n_fail stats (annotation only)

Layers:
  orange  = HF blocked θ (KDE + scatter)
  blue    = LF blocked θ (KDE + scatter)
  green   = final posterior samples
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from prism_vertex.benchmark.inference_run_dirs import (
    budget_key_from_run_dir,
    method_name_from_run_dir,
)
from prism_vertex.benchmark.comparison_paths import method_results_dir
from prism_vertex.simulator.prior_region import (  # noqa: E402
    GAM_MAX,
    GAM_MIN,
    LAM_MAX,
    LAM_MIN,
    gamma_ab_boundary,
    lambda_bc_boundary,
)
from prism_vertex.simulator.constants import THETA_TRUE  # noqa: E402
from scripts.figures._lib.color_palette import (  # noqa: E402
    FEAS_HF_BLOCKED_EDGE,
    FEAS_HF_BLOCKED_FILL,
    FEAS_LF_BLOCKED_EDGE,
    FEAS_LF_BLOCKED_FILL,
    FEAS_OK_FILL,
    FEAS_POSTERIOR_EDGE,
    FEAS_POSTERIOR_FILL,
    OBS_STAR_COLOR,
)

SUPPORTED_METHODS = ("mf_sbi")

HF_STYLE = {
    "cmap": "Oranges",
    "scatter": FEAS_HF_BLOCKED_EDGE,
    "patch": FEAS_HF_BLOCKED_FILL,
    "edge": FEAS_HF_BLOCKED_EDGE,
    "label": "HF blocked θ",
}
LF_STYLE = {
    "cmap": "Blues",
    "scatter": FEAS_LF_BLOCKED_EDGE,
    "patch": FEAS_LF_BLOCKED_FILL,
    "edge": FEAS_LF_BLOCKED_EDGE,
    "label": "LF blocked θ",
}
POST_STYLE = {
    "cmap": "Greens",
    "scatter": FEAS_POSTERIOR_EDGE,
    "patch": FEAS_POSTERIOR_FILL,
    "edge": FEAS_POSTERIOR_EDGE,
    "label": "Posterior samples",
}


def _find_exp_dir(results_root: Path) -> Path:
    for d in results_root.glob("prism_vertex"):
        if d.is_dir():
            return d
    raise FileNotFoundError(f"No prism_vertex_* under {results_root}")


def _resolve_run_dir(
    *,
    results_root: Path,
    run_dir: Path | None,
    method: str,
    budget_hf: int | None,
    budget_lf: int | None,
    lf_fidelity: str | None,
    obs_id: int,
    seed_id: int,
) -> Path:
    if run_dir is not None:
        return run_dir.resolve()
    exp_dir = _find_exp_dir(results_root)
    method_dir = method_results_dir(exp_dir, method)
    if budget_lf is not None and budget_lf > 0:
        budget_name = f"budget_lf_{budget_lf:06d}_hf_{(budget_hf or 500):06d}"
    elif budget_hf is not None:
        budget_name = f"budget_hf_{budget_hf:06d}"
    else:
        raise ValueError("Provide --run_dir or --budget_hf (and optional --budget_lf).")
    tail = method_dir / budget_name / f"obs_{obs_id:02d}" / f"seed_{seed_id:02d}"
    if lf_fidelity:
        return method_dir / lf_fidelity / budget_name / f"obs_{obs_id:02d}" / f"seed_{seed_id:02d}"
    if tail.exists():
        return tail
    # New layout: method_<m>/lf_*/budget_*/obs_*/seed_*
    if budget_lf is not None and budget_lf > 0:
        matches = sorted(method_dir.glob(f"lf_*/{budget_name}/obs_{obs_id:02d}/seed_{seed_id:02d}"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple LF folders match {budget_name}: {[m.parent.parent.name for m in matches]}. "
                "Pass --lf_fidelity or --run_dir."
            )
    return tail


def _iter_run_dirs_with_posterior(method_dir: Path) -> list[Path]:
    patterns = (
        "budget_*/obs_*/seed_*/posterior_samples.csv",
        "lf_*/budget_*/obs_*/seed_*/posterior_samples.csv",
    )
    run_dirs: list[Path] = []
    for pat in patterns:
        run_dirs.extend(p.parent for p in method_dir.glob(pat))
    return sorted(set(run_dirs))


def _draw_prior_region(ax: plt.Axes) -> None:
    xx, yy = np.meshgrid(np.linspace(-1.5, 0.5, 320), np.linspace(0.0, 0.2, 240))
    mask = (yy >= np.maximum(0.0, gamma_ab_boundary(xx))) & (xx <= lambda_bc_boundary(yy))
    ax.contourf(xx, yy, mask.astype(float), levels=[0.5, 1.5], colors=["#f4f6f4"], alpha=1.0)
    x = np.linspace(-1.5, 0.5, 300)
    ax.plot(x, gamma_ab_boundary(x), color="0.55", lw=0.8)
    y = np.linspace(0.0, 0.2, 300)
    ax.plot(lambda_bc_boundary(y), y, color="0.55", lw=0.8)
    ax.plot(
        [LAM_MIN, LAM_MAX, LAM_MAX, LAM_MIN, LAM_MIN],
        [GAM_MIN, GAM_MIN, GAM_MAX, GAM_MAX, GAM_MIN],
        color="0.45",
        lw=0.9,
        ls="--",
    )


def _load_posterior(run_dir: Path) -> np.ndarray:
    path = run_dir / "posterior_samples.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    cols = [c for c in ("lambda", "gamma") if c in df.columns]
    if len(cols) != 2:
        raise ValueError(f"Expected lambda,gamma columns in {path}")
    return df[cols].to_numpy(dtype=float)


def _load_blocked_by_fidelity(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    path = run_dir / "failure_registry_inference.json"
    if not path.exists():
        empty = np.empty((0, 2), dtype=float)
        return empty, empty, {}
    data = json.loads(path.read_text())
    blocked = data.get("blocked_theta", {})
    hf_rows: list[np.ndarray] = []
    lf_rows: list[np.ndarray] = []
    for entry in blocked.values():
        theta = np.asarray(entry["theta"], dtype=float).reshape(2)
        fid = str(entry.get("fidelity", "hf")).lower()
        if fid == "lf":
            lf_rows.append(theta)
        else:
            hf_rows.append(theta)
    blocked_hf = np.vstack(hf_rows) if hf_rows else np.empty((0, 2), dtype=float)
    blocked_lf = np.vstack(lf_rows) if lf_rows else np.empty((0, 2), dtype=float)
    return blocked_hf, blocked_lf, data.get("stats", {})


def _load_diagnostics(run_dir: Path) -> dict:
    path = run_dir / "diagnostics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _kde_on_grid(points: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    pts = points.T
    rng = np.random.default_rng(0)
    if points.shape[0] >= 2:
        pts = pts + rng.normal(0.0, 1e-5, size=pts.shape)
    try:
        kde = gaussian_kde(pts, bw_method="scott")
    except np.linalg.LinAlgError:
        return None
    grid = np.vstack([xx.ravel(), yy.ravel()])
    z = kde(grid).reshape(xx.shape)
    return z


def _draw_blocked_layer(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    xx: np.ndarray,
    yy: np.ndarray,
    style: dict[str, str],
    zorder: int,
) -> None:
    z_block = _kde_on_grid(points, xx, yy)
    if z_block is not None:
        z_norm = z_block / np.nanmax(z_block)
        ax.contourf(
            xx,
            yy,
            z_norm,
            levels=12,
            cmap=style["cmap"],
            alpha=0.50,
            antialiased=True,
            zorder=zorder,
        )
    if points.shape[0] > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=24 if style is HF_STYLE else 18,
            c=style["scatter"],
            alpha=0.40,
            edgecolors="none",
            zorder=zorder + 1,
        )


def _draw_posterior_layer(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    xx: np.ndarray,
    yy: np.ndarray,
    zorder: int,
) -> None:
    """Blue posterior KDE + scatter (always drawn on top of blocked layers)."""
    z_post = _kde_on_grid(points, xx, yy)
    if z_post is not None:
        z_norm = z_post / max(np.nanmax(z_post), 1e-12)
        ax.contourf(
            xx,
            yy,
            z_norm,
            levels=10,
            cmap=POST_STYLE["cmap"],
            alpha=0.62,
            antialiased=True,
            zorder=zorder,
        )
    else:
        # Degenerate / tiny sample: hexbin so posterior is still visible
        ax.hexbin(
            points[:, 0],
            points[:, 1],
            gridsize=28,
            cmap=POST_STYLE["cmap"],
            mincnt=1,
            alpha=0.55,
            linewidths=0.0,
            zorder=zorder,
        )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=8,
        c=POST_STYLE["scatter"],
        alpha=0.28,
        edgecolors="none",
        zorder=zorder + 1,
    )


def plot_posterior_feasibility(
    run_dir: Path,
    *,
    out_path: Path | None = None,
    dpi: int = 160,
) -> Path:
    run_dir = run_dir.resolve()
    post = _load_posterior(run_dir)
    blocked_hf, blocked_lf, reg_stats = _load_blocked_by_fidelity(run_dir)
    diag = _load_diagnostics(run_dir)

    lam = np.linspace(-1.45, 0.25, 200)
    gam = np.linspace(0.01, 0.20, 200)
    xx, yy = np.meshgrid(lam, gam)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    _draw_prior_region(ax)

    _draw_blocked_layer(ax, blocked_hf, xx=xx, yy=yy, style=HF_STYLE, zorder=2)
    _draw_blocked_layer(ax, blocked_lf, xx=xx, yy=yy, style=LF_STYLE, zorder=3)
    _draw_posterior_layer(ax, post, xx=xx, yy=yy, zorder=5)

    ax.scatter(
        THETA_TRUE[0],
        THETA_TRUE[1],
        marker="*",
        s=220,
        c=OBS_STAR_COLOR,
        edgecolors="black",
        linewidths=0.8,
        zorder=6,
    )

    ax.set_xlim(-1.45, 0.28)
    ax.set_ylim(0.01, 0.205)
    ax.set_xlabel(r"$\Lambda$ (lambda)")
    ax.set_ylabel(r"$\Gamma$ (gamma)")

    n_fail_rounds = sum(int(r.get("n_fail", 0)) for r in diag.get("round_logs", []))
    blocked_n = int(diag.get("prism_vertex_failure_registry", {}).get("blocked_count", blocked_hf.shape[0] + blocked_lf.shape[0]))
    try:
        method = diag.get("method") or method_name_from_run_dir(run_dir)
    except ValueError:
        method = run_dir.parents[3].name.replace("method_", "")
    budget_id = diag.get("meta", {}).get("budget_id")
    if not budget_id:
        try:
            budget_id = budget_key_from_run_dir(run_dir)
        except ValueError:
            budget_id = run_dir.parents[2].name
    n_pen_hf = int(reg_stats.get("n_penalty_hf", 0))
    n_pen_lf = int(reg_stats.get("n_penalty_lf", 0))

    ax.set_title(
        f"Posterior vs inference-blocked θ ({method})\n"
        f"{budget_id} · unique blocked={blocked_n} "
        f"(HF={blocked_hf.shape[0]}, LF={blocked_lf.shape[0]}) · "
        f"penalties HF/LF={n_pen_hf}/{n_pen_lf} · round fails={n_fail_rounds}",
        fontsize=10,
    )

    legend_handles = [
        Patch(facecolor=HF_STYLE["patch"], edgecolor=HF_STYLE["edge"], alpha=0.7, label=f"{HF_STYLE['label']} (n={blocked_hf.shape[0]})"),
        Patch(facecolor=LF_STYLE["patch"], edgecolor=LF_STYLE["edge"], alpha=0.7, label=f"{LF_STYLE['label']} (n={blocked_lf.shape[0]})"),
        Patch(facecolor=POST_STYLE["patch"], edgecolor=POST_STYLE["edge"], alpha=0.7, label=f"{POST_STYLE['label']} (n={post.shape[0]})"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=OBS_STAR_COLOR, markeredgecolor="black", markersize=12, label=r"$\theta_{true}$"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.text(
        0.02,
        0.02,
        "Orange/purple = unique θ with Chaste fail or penalty (failure_registry_inference.json).\n"
        "Blue = final posterior. Registry stores blocked points by fidelity, not learned s_ψ(x).",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    if out_path is None:
        out_dir = run_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "posterior_feasibility_map.png"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--run_dir", type=Path, default=None, help="Full path to obs_XX/seed_YY run folder")
    ap.add_argument("--method", type=str, default="mf_sbi", choices=SUPPORTED_METHODS)
    ap.add_argument("--budget_hf", type=int, default=None)
    ap.add_argument("--budget_lf", type=int, default=None)
    ap.add_argument("--lf_fidelity", type=str, default=None, help="e.g. lf_DT10 (required if multiple LF folders match)")
    ap.add_argument("--obs_id", type=int, default=0)
    ap.add_argument("--seed_id", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path")
    ap.add_argument(
        "--all_budgets",
        action="store_true",
        help="Plot all runs with posterior_samples.csv for --method",
    )
    args = ap.parse_args()

    if args.all_budgets:
        exp_dir = _find_exp_dir(args.results_root)
        base = method_results_dir(exp_dir, args.method)
        run_dirs = _iter_run_dirs_with_posterior(base)
        if not run_dirs:
            raise SystemExit(f"No runs under {base}")
        for rd in run_dirs:
            out = plot_posterior_feasibility(rd)
            print(f"Wrote {out}")
        return

    run_dir = _resolve_run_dir(
        results_root=args.results_root,
        run_dir=args.run_dir,
        method=args.method,
        budget_hf=args.budget_hf,
        budget_lf=args.budget_lf,
        lf_fidelity=args.lf_fidelity,
        obs_id=args.obs_id,
        seed_id=args.seed_id,
    )
    out = plot_posterior_feasibility(run_dir, out_path=args.out)
    print(f"Wrote {out}")
    print(f"Run dir: {run_dir}")
    blocked_hf, blocked_lf, stats = _load_blocked_by_fidelity(run_dir)
    print(
        f"Blocked unique θ: HF={blocked_hf.shape[0]} LF={blocked_lf.shape[0]}  "
        f"registry stats: {stats}"
    )


if __name__ == "__main__":
    main()
