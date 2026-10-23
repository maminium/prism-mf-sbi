#!/usr/bin/env python3
"""Figure 3 panels (a)–(g): ABCDE row on top, F and G below.

Example:
  python -m scripts.figures.plot_fig2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism_vertex.benchmark.comparison_paths import results_dir
from scripts.figures._lib.output_formats import manuscript_fig_stem
from scripts.figures._lib.panels_fig2 import DPI, plot_fig3_abcdefg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--out_stem", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=DPI)
    args = ap.parse_args()

    exp_dir = results_dir(args.results_root)
    out_stem = args.out_stem or manuscript_fig_stem(exp_dir, 2)
    png, pdf = plot_fig3_abcdefg(exp_dir, out_stem, dpi=args.dpi)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
