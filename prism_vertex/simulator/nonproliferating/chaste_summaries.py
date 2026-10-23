from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

SUMMARY_DIM = 8
SUMMARY_NAMES: tuple[str, ...] = (
    "mean_log_ell",
    "std_log_ell",
    "mean_c2",
    "mean_s2",
    "ell_c2_cov",
    "mean_log_A",
    "std_log_A",
    "frac_hex",
)


def _sanitize_chaste_json_raw(raw: str) -> str:
    raw = raw.replace("-nan", "0").replace("-NaN", "0")
    raw = re.sub(r"(?<=[:\s\[,])-?(?:inf|Inf|INF)(?=[,\s\]\}])", "0", raw)
    raw = re.sub(r"\bnan\b", "0", raw, flags=re.IGNORECASE)
    return raw


def _load_summary_json(summary_path: Path, *, retries: int = 5) -> dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Chaste summary JSON: {summary_path}")
    last_err: Exception | None = None
    for attempt in range(retries):
        raw = summary_path.read_text()
        if not raw.strip():
            last_err = ValueError(f"Empty summary file: {summary_path}")
            time.sleep(0.2)
            continue
        try:
            return json.loads(_sanitize_chaste_json_raw(raw))
        except json.JSONDecodeError as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(0.25)
                continue
    raise ValueError(f"Corrupt or incomplete {summary_path}: {last_err}") from last_err


def _summary_path(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    primary = run_dir / "prism_vertex_summary.json"
    if primary.is_file():
        return primary
    raise FileNotFoundError(f"Missing Chaste summary JSON: {primary}")


def load_summary_vector(run_dir: Path) -> np.ndarray:
    data = _load_summary_json(_summary_path(run_dir))
    if "s" in data:
        s = np.asarray(data["s"], dtype=np.float64).reshape(-1)
        if s.size == SUMMARY_DIM and np.isfinite(s).all():
            return s
    parts = [float(data.get(name, np.nan)) for name in SUMMARY_NAMES]
    x = np.asarray(parts, dtype=np.float64)
    if x.shape[0] != SUMMARY_DIM or not np.isfinite(x).all():
        raise ValueError(f"nonproliferating summary invalid in {run_dir}: {x}")
    return x


def load_summary_metadata(run_dir: Path) -> dict:
    path = Path(run_dir) / "prism_vertex_summary.json"
    if not path.exists():
        return {}
    data = _load_summary_json(path)
    keys = (
        "n_cells",
        "n_internal_cells",
        "n_internal_edges",
        "wall_seconds",
        "end_time",
        "simulation_time",
        "dt",
        "a_r",
        "b_0",
        "k",
        "minimal_io",
        "median_area",
        "length_scale",
    )
    return {k: data[k] for k in keys if k in data}
