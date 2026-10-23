#!/usr/bin/env bash
# Regenerate manuscript figures 1–8 (PNG + TIFF + EPS) from shipped processed data.
# Uses Mo 8 Arial-10 rcParams. Fig1 ships the cropped PNG (not the widescreen PDF).
# Fig8 exports the shipped Mo 8 composite unless --rebuild-fig8 is passed.
# Panel C schematic is figures/e.png (whitespace-cropped from fig8-c).
set -euo pipefail
cd "$(dirname "$0")"
python3 -m scripts.figures.plot_all --figures 1,2,3,4,5,6,7,8 "$@"
