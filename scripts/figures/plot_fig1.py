#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures._lib.output_formats import (
    export_external_pdf_submission_formats,
    export_external_png_submission_formats,
    rasterise_pdf_cropped_png,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures")
    ap.add_argument(
        "--from-pdf",
        action="store_true",
        help="Rebuild fig1.png from fig1.pdf with whitespace cropped (overwrites PNG).",
    )
    args = ap.parse_args(argv)
    pdf = args.figures_dir / "fig1.pdf"
    png = args.figures_dir / "fig1.png"

    if args.from_pdf:
        if not pdf.is_file():
            raise SystemExit(f"Missing {pdf}")
        rasterise_pdf_cropped_png(pdf, png)
        export_external_png_submission_formats(png)
        # Also refresh EPS/TIFF from PDF vector when available
        export_external_pdf_submission_formats(pdf, update_png=False)
    elif png.is_file():
        export_external_png_submission_formats(png)
    elif pdf.is_file():
        # No shipped PNG: crop from PDF once, then export.
        rasterise_pdf_cropped_png(pdf, png)
        export_external_png_submission_formats(png)
        export_external_pdf_submission_formats(pdf, update_png=False)
    else:
        raise SystemExit(f"Missing {png} and {pdf}")

    print(f"Wrote {png}")
    print(f"Wrote {args.figures_dir / 'EPS' / 'fig1.eps'}")
    print(f"Wrote {args.figures_dir / 'TIFF' / 'fig1.tiff'}")


if __name__ == "__main__":
    main()
