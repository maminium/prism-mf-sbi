#!/usr/bin/env python3
"""Methods Figure 2 — separate panel PNGs (a, c, d) plus composite assembly.

Panel (c) merges the prior-region schematic and HF failure map.
Panel (e) is supplied externally (fig 2/e.png); panel (f) is omitted.

Example:
  python scripts/plot_prism_vertex_methods_fig2_panels.py
  python scripts/plot_fig2_composite.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from prism_vertex.simulator.prior_region import (  # noqa: E402
    GAM_MAX,
    GAM_MIN,
    LAM_MAX,
    LAM_MIN,
    gamma_ab_boundary,
    lambda_bc_boundary,
)
from prism_vertex.simulator.scoreboard_analysis import (  # noqa: E402
    LF_ORDER,
    load_cells_frame,
)
from prism_vertex.simulator.scoreboard_io import scoreboard_root  # noqa: E402
from prism_vertex.simulator.constants import THETA_TRUE  # noqa: E402
from prism_vertex.benchmark.comparison_paths import results_dir  # noqa: E402
from scripts.figures._lib.panels_fig7 import (  # noqa: E402
    DPI,
    FIG2_D_HEATMAP_SIZE,
    FIG2_PANEL_SIZE,
    FIG2_TABLE_SIZE,
    KURSAWE_SUMMARY_CANDIDATES,
    OBS_SUMMARY_DIM,
    apply_rcparams,
    fig2_out_dir,
    style_axes_like_fig4,
)
from scripts.figures._lib.output_formats import FIG7_BUILD_PANELS  # noqa: E402
from scripts.figures._lib.panels_fig2 import (  # noqa: E402
    FIG3_AXIS_LABEL_FONTSIZE,
    FIG3_LEGEND_FONTSIZE,
    FIG3_TICK_LABELSIZE,
)
from scripts.figures._lib.fig3_style import (  # noqa: E402
    FIG4_CURVE_LW,
    FIG4_FEAS_STAR_EDGE_LW,
    FIG4_FEAS_STAR_SIZE,
)
from scripts.figures._lib.color_palette import (  # noqa: E402
    OBS_STAR_COLOR,
    PRIOR_BOX_COLOR,
    REGION_B_FILL,
    failure_field_cmap,
    failure_heatmap_cmap,
)

ATTEMPTED = ("ok", "failed", "skipped_known_failed_point")

FID_SHORT: dict[str, str] = {
    "hf_single": "HF",
    "lf_T70": "T70",
    "lf_T140": "T140",
    "lf_T350": "T350",
    "lf_DT02": r"$\Delta T_{0.02}$",
    "lf_DT05": r"$\Delta T_{0.05}$",
    "lf_DT10": r"$\Delta T_{0.10}$",
    "lf_R1": "K1",
    "lf_R2": "K2",
    "lf_R5": "K5",
}


def _draw_prior_regions_and_box(ax: plt.Axes, *, region_fill: str = REGION_B_FILL, region_alpha: float = 0.85) -> None:
    xx, yy = np.meshgrid(np.linspace(-1.5, 0.5, 450), np.linspace(0.0, 0.2, 300))
    mask_b = (yy >= np.maximum(0.0, gamma_ab_boundary(xx))) & (xx <= lambda_bc_boundary(yy))
    ax.contourf(xx, yy, mask_b.astype(float), levels=[0.5, 1.5], colors=[region_fill], alpha=region_alpha)

    x = np.linspace(-1.5, 0.5, 500)
    ax.plot(x, gamma_ab_boundary(x), color="black", lw=FIG4_CURVE_LW)
    y = np.linspace(0.0, 0.2, 400)
    ax.plot(lambda_bc_boundary(y), y, color="black", lw=FIG4_CURVE_LW)
    ax.plot(
        [LAM_MIN, LAM_MAX, LAM_MAX, LAM_MIN, LAM_MIN],
        [GAM_MIN, GAM_MIN, GAM_MAX, GAM_MAX, GAM_MIN],
        color=PRIOR_BOX_COLOR,
        lw=FIG4_CURVE_LW,
        ls="--",
    )

    ax.text(-1.2, 0.060, "A", fontsize=FIG3_AXIS_LABEL_FONTSIZE, color="#888888")
    ax.text(-0.42, 0.10, "B", fontsize=FIG3_AXIS_LABEL_FONTSIZE, color="0.12")
    ax.text(0.18, 0.10, "C", fontsize=FIG3_AXIS_LABEL_FONTSIZE, color="#888888")

    ax.set_xlim(-1.5, 0.35)
    ax.set_ylim(0.0, 0.205)
    ax.set_xlabel(r"$\Lambda$", fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\Gamma$", fontsize=FIG3_AXIS_LABEL_FONTSIZE)


def plot_panel_a(out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG2_TABLE_SIZE, facecolor="white")
    ax.axis("off")

    col_labels = ["", "Kursawe et al. (2018)", "This study"]
    rows = [
        (
            "Inference",
            "ABC on precomputed library (~$10^5$ sims)",
            "HF-SNPE, HF-NPE, HF-SMC-ABC, MF-SBI (~$10^3$ HF sims)",
        ),
        (
            "Prior",
            "Uniform on Region B only",
            "Uniform on operational prior box",
        ),
        (
            "Observation",
            f"{KURSAWE_SUMMARY_CANDIDATES} candidate summaries screened; adopted best {OBS_SUMMARY_DIM}",
            f"Same {OBS_SUMMARY_DIM}D summaries",
        ),
        (
            "Low fidelity",
            "Not considered",
            "Nine LF simulators (horizon, timestep, replicates)",
        ),
        (
            "Predictive performance",
            "Contours, descriptive spreads",
            "NLTP, CRPS (+ NRMSE secondary)",
        ),
        (
            "Calibration",
            "—",
            "TARP, SBC; mean posterior STD and 95% CI width",
        ),
    ]

    table_data = [[r[0], r[1], r[2]] for r in rows]
    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colWidths=[0.17, 0.39, 0.44],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FIG3_TICK_LABELSIZE)
    tbl.scale(1.0, 1.65)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E0E0E0")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#E8EEF4")
            cell.set_text_props(fontweight="bold", fontsize=FIG3_AXIS_LABEL_FONTSIZE)
        elif col == 0:
            cell.set_facecolor("#F8F8F8")
            cell.set_text_props(fontweight="bold", fontsize=FIG3_TICK_LABELSIZE)
        elif row % 2 == 0:
            cell.set_facecolor("#FFFFFF")
        else:
            cell.set_facecolor("#FAFAFA")

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _hf_failure_at_theta(df, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_theta = theta.shape[0]
    sub = df[df["fidelity_id"] == "hf_single"]
    fail_rate = np.full(n_theta, np.nan)
    n_att = np.zeros(n_theta)
    for i in range(n_theta):
        g = sub[sub["theta_index"] == i]
        att = int(g["status"].isin(ATTEMPTED).sum())
        ok = int((g["status"] == "ok").sum())
        n_att[i] = att
        fail_rate[i] = (att - ok) / att if att > 0 else np.nan
    return fail_rate, n_att


def _pct_text_color(value: float) -> str:
    return "white" if value >= 0.55 else "black"


def plot_panel_c_merged(df, theta: np.ndarray, out: Path) -> None:
    """Merged prior regions + HF failure-rate field (scoreboard locations)."""
    fig, ax = plt.subplots(figsize=FIG2_PANEL_SIZE, facecolor="white")
    _draw_prior_regions_and_box(ax)

    fail_rate, n_att = _hf_failure_at_theta(df, theta)
    cmap = failure_field_cmap()
    norm = plt.Normalize(0.0, 1.0)
    for i in range(theta.shape[0]):
        fr = fail_rate[i]
        if not np.isfinite(fr):
            ax.scatter(
                theta[i, 0],
                theta[i, 1],
                marker="s",
                s=36,
                c="white",
                edgecolors="0.5",
                linewidths=0.6,
                zorder=3,
            )
            continue
        size = 36 + 8 * float(n_att[i])
        ax.scatter(
            theta[i, 0],
            theta[i, 1],
            s=size,
            c=[cmap(norm(fr))],
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    ax.scatter(
        THETA_TRUE[0],
        THETA_TRUE[1],
        marker="*",
        s=FIG4_FEAS_STAR_SIZE,
        c=OBS_STAR_COLOR,
        edgecolors="black",
        linewidths=FIG4_FEAS_STAR_EDGE_LW,
        zorder=5,
        label=r"$\mathbf{s}_{\mathrm{obs}}$ ($\theta^*$)",
    )

    style_axes_like_fig4(ax)
    ax.grid(alpha=0.15)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("failure rate", fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=FIG3_TICK_LABELSIZE, length=3, width=0.6)

    legend_handles = [
        Patch(facecolor=REGION_B_FILL, edgecolor="black", alpha=0.85, linewidth=0.6, label="Region B"),
        plt.Line2D([0], [0], color=PRIOR_BOX_COLOR, lw=FIG4_CURVE_LW, ls="--", label="Operational prior box"),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=OBS_STAR_COLOR,
            markeredgecolor="black",
            markersize=FIG4_FEAS_STAR_SIZE * 0.07,
            linestyle="None",
            label=r"$\mathbf{s}_{\mathrm{obs}}$ ($\theta^*$)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.10),
        fontsize=FIG3_LEGEND_FONTSIZE,
        frameon=False,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_panel_d(df, out: Path) -> None:
    heatmap_fids = LF_ORDER
    n_seeds = int(df["seed_k"].max()) + 1
    fail_rate = np.full((n_seeds, len(heatmap_fids)), np.nan)
    for c, fid in enumerate(heatmap_fids):
        sub = df[df["fidelity_id"] == fid]
        for k in range(n_seeds):
            g = sub[sub["seed_k"] == k]
            att = int(g["status"].isin(ATTEMPTED).sum())
            ok = int((g["status"] == "ok").sum())
            fail_rate[k, c] = (att - ok) / att if att > 0 else np.nan

    col_labels = [FID_SHORT.get(fid, fid) for fid in heatmap_fids]

    fig, ax = plt.subplots(figsize=FIG2_D_HEATMAP_SIZE, facecolor="white")
    im = ax.imshow(fail_rate, aspect="auto", cmap=failure_heatmap_cmap(), vmin=0, vmax=1)
    ax.set_xticks(range(len(heatmap_fids)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=FIG3_TICK_LABELSIZE)
    ax.set_yticks(range(n_seeds))
    ax.set_yticklabels([f"$k={k}$" for k in range(n_seeds)], fontsize=FIG3_TICK_LABELSIZE)
    ax.set_ylabel("Chaste seed index $k$", fontsize=FIG3_AXIS_LABEL_FONTSIZE)

    for k in range(n_seeds):
        for c in range(len(heatmap_fids)):
            v = fail_rate[k, c]
            if not np.isfinite(v):
                ax.text(c, k, "—", ha="center", va="center", fontsize=FIG3_TICK_LABELSIZE, color="0.45")
                continue
            ax.text(
                c,
                k,
                f"{v:.0%}",
                ha="center",
                va="center",
                fontsize=FIG3_TICK_LABELSIZE,
                color=_pct_text_color(v),
            )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Failure rate", fontsize=FIG3_AXIS_LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=FIG3_TICK_LABELSIZE, length=3, width=0.6)

    style_axes_like_fig4(ax)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scoreboard-root", type=Path, default=None, help="lf_selection root")
    ap.add_argument("--results-root", type=Path, default=None, help="results/ parent")
    ap.add_argument("--out-dir", type=Path, default=None, help="output directory for PNGs")
    ap.add_argument("--composite", action="store_true", help="also assemble fig2_acde composite")
    args = ap.parse_args()

    apply_rcparams()
    exp_dir = results_dir(args.results_root)
    out_dir = args.out_dir or fig2_out_dir(args.results_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    sb_root = scoreboard_root(args.scoreboard_root)
    df = load_cells_frame(sb_root)
    theta = np.load(sb_root / "theta_pilot_list.npy")

    panels = {
        FIG7_BUILD_PANELS["a"]: lambda: plot_panel_a(out_dir / FIG7_BUILD_PANELS["a"]),
        FIG7_BUILD_PANELS["c"]: lambda: plot_panel_c_merged(
            df, theta, out_dir / FIG7_BUILD_PANELS["c"]
        ),
        FIG7_BUILD_PANELS["d"]: lambda: plot_panel_d(df, out_dir / FIG7_BUILD_PANELS["d"]),
    }

    for name, fn in panels.items():
        fn()
        print(f"Wrote {out_dir / name}")

    if args.composite:
        from scripts.figures.plot_fig7 import plot_fig7_manuscript_composite

        png, pdf = plot_fig7_manuscript_composite(exp_dir, dpi=DPI)
        print(f"Wrote {png}")
        print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
