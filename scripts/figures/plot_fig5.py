#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism_vertex.benchmark.comparison_paths import results_dir
from scripts.figures._lib.panels_fig2 import DPI
from scripts.figures._lib.panels_fig6 import plot_manuscript_fig5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--obs_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=DPI)
    args = ap.parse_args()

    exp_dir = results_dir(args.results_root)
    png, pdf = plot_manuscript_fig5(
        exp_dir, obs_id=args.obs_id, seed=args.seed, dpi=args.dpi
    )
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
