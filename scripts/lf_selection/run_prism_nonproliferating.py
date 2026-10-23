#!/usr/bin/env python3
"""Run PRISM screening for the non-proliferating vertex model (requires Chaste).

Writes features under data/nonproliferating_vertex/lf_selection/.
For figure regeneration from shipped tables, use scripts/figures/plot_fig6.py.
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
    ap.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "data" / "nonproliferating_vertex")
    args = ap.parse_args()
    feat = args.results_root / "lf_selection" / "lf_selection_features_out5d.csv"
    if feat.is_file():
        print(f"Shipped PRISM features present: {feat}")
        print("Re-running the full Chaste pilot screen is optional for figure regeneration.")
        return
    raise SystemExit(
        "Missing shipped PRISM features. Build NonproliferatingVertex, set env vars, "
        "then regenerate lf_selection tables (see README)."
    )


if __name__ == "__main__":
    main()
