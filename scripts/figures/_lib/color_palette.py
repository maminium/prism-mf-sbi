"""Unified manuscript color palette
"""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Inference methods (Fig 4–6 curves, bars, posterior triangles)
# ---------------------------------------------------------------------------
METHOD_COLORS: dict[str, str] = {
    "mf_sbi": "#C25700",
    "hf_snpe": "#5C6770",
    "hf_npe": "#868E96",
    "hf_smc_abc": "#ADB5BD",
}

THETA_STAR_COLOR = "#B2182B"

HF_COLORS: dict[str, str] = {
    "hf_snpe": METHOD_COLORS["hf_snpe"],
    "hf_npe": METHOD_COLORS["hf_npe"],
    "hf_smc_abc": METHOD_COLORS["hf_smc_abc"],
}

HF_HLINE_COLOR = "#3D4852"

# ---------------------------------------------------------------------------
# Low-fidelity families — T (green) · ΔT (red) · K-replicates (blue)
# ΔT display order is 0.1 → 0.05 → 0.02; colours lighten → darken along that order.
# ---------------------------------------------------------------------------
LF_COLORS: dict[str, str] = {
    "lf_T70": "#CCECE6",
    "lf_T140": "#66C2A4",
    "lf_T350": "#238B45",
    "lf_DT10": "#FCAE91",
    "lf_DT05": "#FB6A4A",
    "lf_DT02": "#CB181D",
    "lf_R1": "#BDD7E7",
    "lf_R2": "#6BAED6",
    "lf_R5": "#2171B5",
}

FIDELITY_COLORS: dict[str, str] = {
    "hf_single": "#000000",
    **LF_COLORS,
}

# ---------------------------------------------------------------------------
# Scoreboard regimes (Fig 2–3)
# ---------------------------------------------------------------------------
REGION_COLORS: dict[str, str] = {
    "Efficiency-optimal": "#CB181D",
    "Affordability-bound": "#2171B5",
    "Fidelity-bound": "#238B45",
    "Score-inference inconsistent": "#BDBDBD",
}

# ---------------------------------------------------------------------------
# Spatial accents (Fig 7 failure map)
# ---------------------------------------------------------------------------
REGION_B_FILL = "#E8F4E5"
REGION_B_EDGE = "#74C476"
PRIOR_BOX_COLOR = "#525252"
OBS_STAR_COLOR = "#E6AB02"

# ---------------------------------------------------------------------------
# Neutrals & reference geometry
# ---------------------------------------------------------------------------
IDEAL_LINE_COLOR = "#404040"
DKW_BAND_COLOR = "#D9D9D9"
GRID_COLOR = "#E0E0E0"

# Posterior grid feasibility layers (Fig 4): posterior=green, LF blocked=blue, HF blocked=warm
FEAS_POSTERIOR_FILL = "#CCECE6"
FEAS_POSTERIOR_EDGE = "#238B45"
FEAS_HF_BLOCKED_FILL = "#FCAE91"
FEAS_HF_BLOCKED_EDGE = "#A63603"
FEAS_LF_BLOCKED_FILL = "#BDD7E7"
FEAS_LF_BLOCKED_EDGE = "#2171B5"
FEAS_OK_FILL = "#238B45"

CMAP_FAILURE_FIELD_STOPS: tuple[str, ...] = ("#1A9850", "#FEE08B", "#D73027")
CMAP_FAILURE_HEATMAP_STOPS: tuple[str, ...] = ("#FEE5D9", "#FC9272", "#A50F15")
CMAP_CORRELATION_STOPS: tuple[str, ...] = ("#2166AC", "#F7F7F7", "#B2182B")


def make_linear_cmap(stops: tuple[str, ...], name: str = "manuscript") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, stops, N=256)


def failure_field_cmap() -> LinearSegmentedColormap:
    return make_linear_cmap(CMAP_FAILURE_FIELD_STOPS, "manuscript_failure_field")


def failure_heatmap_cmap() -> LinearSegmentedColormap:
    return make_linear_cmap(CMAP_FAILURE_HEATMAP_STOPS, "manuscript_failure_heatmap")


def correlation_cmap() -> LinearSegmentedColormap:
    return make_linear_cmap(CMAP_CORRELATION_STOPS, "manuscript_correlation")
