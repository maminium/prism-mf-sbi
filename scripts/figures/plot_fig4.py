#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism_vertex.benchmark.comparison_paths import results_dir
from scripts.figures._lib.fig4_feasibility_grid import FIG4_MAIN_GRID, plot_posterior_reference_grid
from scripts.figures._lib.output_formats import manuscript_figures_dir, manuscript_fig_stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--obs-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    exp_dir = results_dir(args.results_root)
    out_dir = manuscript_figures_dir(exp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = manuscript_fig_stem(exp_dir, 4)
    png = plot_posterior_reference_grid(
        exp_dir,
        stem.with_suffix(".png"),
        obs_id=args.obs_id,
        seed=args.seed,
        dpi=args.dpi,
        grid=FIG4_MAIN_GRID,
        figsize=(9.5, 8.5),
    )
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
