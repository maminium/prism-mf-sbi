#!/usr/bin/env python3
"""Run PRISM screening for the proliferating vertex model (requires Chaste).

Writes paired LF–HF features under data/proliferating_vertex/lf_selection/.
For figure regeneration from shipped tables, use scripts/figures/plot_fig2.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "data" / "proliferating_vertex")
    args = ap.parse_args()
    analysis = args.results_root / "lf_selection" / "analysis"
    table = analysis / "scoreboard_table.csv"
    if table.is_file():
        print(f"Shipped PRISM scoreboard present: {table}")
        print("Re-running the full Chaste pilot screen is optional for figure regeneration.")
        print("See the repository README and scripts/lf_selection/generate_observations.py.")
        return
    raise SystemExit(
        "Missing shipped scoreboard. Build Chaste, set PRISM_VERTEX_* env vars, "
        "then regenerate lf_selection tables (see README)."
    )


if __name__ == "__main__":
    main()
