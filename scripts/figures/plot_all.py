#!/usr/bin/env python3
"""Regenerate paper figures 1–8 from shipped processed data.

Mapping (Mo v6):
  fig1  — workflow schematic (shipped PDF/PNG)
  fig2  — PRISM screening (proliferating)
  fig3  — fidelity–cost scoreboard + post-training metrics (proliferating)
  fig4  — HF-SMC-ABC, HF-NPE, HF-SNPE, MF-SBI ΔT_0.05
  fig5  — MF-SBI calibration (TARP/SBC) at ΔT_0.05
  fig6  — PRISM + post-training metrics (non-proliferating)
  fig7  — pairwise posteriors (non-proliferating)
  fig8  — prior / failures / method allocation (proliferating)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism_vertex.benchmark.comparison_paths import results_dir
from scripts.figures._lib.panels_fig2 import DPI as FIG_DPI
from scripts.figures._lib.panels_fig2 import plot_fig3_abcdefg
from scripts.figures._lib.panels_fig3 import plot_fig4_abcdefgh
from scripts.figures._lib.panels_fig6 import plot_manuscript_fig5
from scripts.figures._lib.fig4_feasibility_grid import (
    FIG4_MAIN_GRID,
    plot_posterior_reference_grid,
)
from scripts.figures._lib.output_formats import (
    export_external_png_submission_formats,
    manuscript_fig_stem,
    manuscript_figures_dir,
)
from scripts.figures.plot_fig8 import plot_fig8_manuscript_composite


def _parse_want(s: str) -> set[str]:
    out: set[str] = set()
    for part in s.split(","):
        part = part.strip().lower()
        if not part:
            continue
        out.add(part)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--obs_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=FIG_DPI)
    ap.add_argument(
        "--figures",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated figure ids (1..8)",
    )
    ap.add_argument(
        "--rebuild-fig8",
        action="store_true",
        help="Rebuild fig8 from panels + e.png instead of exporting the shipped Mo 8 PNG",
    )
    args = ap.parse_args()

    want = _parse_want(args.figures)
    prolif_dir = results_dir(args.results_root)
    out_dir = manuscript_figures_dir(prolif_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "1" in want:
        # Mo 8 fig1 is the cropped PNG; never overwrite it from the widescreen PDF.
        from scripts.figures.plot_fig1 import main as plot_fig1_main

        plot_fig1_main(argv=[])

    if "2" in want:
        stem = manuscript_fig_stem(prolif_dir, 2)
        png, pdf = plot_fig3_abcdefg(prolif_dir, stem, dpi=args.dpi)
        print(f"Wrote {png}")
        print(f"Wrote {pdf}")

    if "3" in want:
        stem = manuscript_fig_stem(prolif_dir, 3)
        png, pdf = plot_fig4_abcdefgh(
            prolif_dir, stem, obs_id=args.obs_id, seed=args.seed, dpi=args.dpi
        )
        print(f"Wrote {png}")
        print(f"Wrote {pdf}")

    if "4" in want:
        stem = manuscript_fig_stem(prolif_dir, 4)
        png = plot_posterior_reference_grid(
            prolif_dir,
            stem.with_suffix(".png"),
            obs_id=args.obs_id,
            seed=args.seed,
            dpi=args.dpi,
            grid=FIG4_MAIN_GRID,
            figsize=(9.5, 8.5),
        )
        print(f"Wrote {png}")

    if "5" in want:
        png, pdf = plot_manuscript_fig5(
            prolif_dir, obs_id=args.obs_id, seed=args.seed, dpi=args.dpi
        )
        print(f"Wrote {png}")
        print(f"Wrote {pdf}")

    if "6" in want:
        from scripts.figures.plot_fig6 import main as plot_fig6_main

        plot_fig6_main(argv=[])

    if "7" in want:
        from scripts.figures.plot_fig7 import main as plot_fig7_main

        plot_fig7_main(argv=[])

    if "8" in want:
        # Mo 8 fig8 ships a high-res composite; rebuild only when panels + e.png need refresh.
        fig8_png = out_dir / "fig8.png"
        if fig8_png.is_file() and not getattr(args, "rebuild_fig8", False):
            export_external_png_submission_formats(fig8_png)
            print(f"Exported shipped {fig8_png}")
        else:
            png, pdf = plot_fig8_manuscript_composite(prolif_dir, dpi=args.dpi)
            print(f"Wrote {png}")
            print(f"Wrote {pdf}")



if __name__ == "__main__":
    main()
