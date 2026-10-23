from __future__ import annotations

def _resolve_summary_path(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    candidate = run_dir / "prism_vertex_summary.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"No summary JSON in {run_dir} (expected prism_vertex_summary.json)"
    )



import json
import re
import time
from pathlib import Path

import numpy as np

SUMMARY_DIM = 6
SUMMARY_NAMES: tuple[str, ...] = (
    "area_ratio_n4",
    "area_ratio_n5",
    "area_ratio_n6",
    "area_ratio_n7",
    "area_ratio_n8",
    "mean_elongation",
)


def _sanitize_chaste_json_raw(raw: str) -> str:
    """C++ streams non-finite doubles as bare nan/inf tokens — invalid strict JSON."""
    # Order matters: -nan before bare nan.
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
            # Parallel workers may read while Chaste is still flushing the file.
            if attempt + 1 < retries:
                time.sleep(0.25)
                continue
    raise ValueError(f"Corrupt or incomplete {summary_path}: {last_err}") from last_err


def load_summary_vector(run_dir: Path) -> np.ndarray:
    """
    Build the Kursawe 6D observation vector from the Chaste summary JSON.

    Order: area ratios for polygon classes 4–8 (bulk cells), then mean elongation.
    Prefer the precomputed ``s`` array when present; otherwise read named fields.
    """
    run_dir = Path(run_dir)
    data = _load_summary_json(_resolve_summary_path(run_dir))

    if "s" in data:
        s = np.asarray(data["s"], dtype=np.float64).reshape(-1)
        if s.size == SUMMARY_DIM and np.isfinite(s).all():
            return s

    parts = [float(data.get(name, np.nan)) for name in SUMMARY_NAMES]
    x = np.asarray(parts, dtype=np.float64)
    if x.shape[0] != SUMMARY_DIM:
        raise ValueError(f"summary has wrong dimension: {x.shape[0]} != {SUMMARY_DIM}")
    if not np.isfinite(x).all():
        missing = [name for name, val in zip(SUMMARY_NAMES, x, strict=True) if not np.isfinite(val)]
        raise ValueError(
            "summary contains non-finite values. "
            f"Missing or invalid fields: {missing}. "
            "Rebuild Chaste with PRISM_VERTEX_PAPER_MODE=1."
        )
    return x


def load_summary_metadata(run_dir: Path) -> dict[str, float | int | bool | None]:
    """Optional run metadata for smoke tests and logging."""
    run_dir = Path(run_dir)
    summary_path = _resolve_summary_path(run_dir)
    if not summary_path.exists():
        return {}
    data = _load_summary_json(summary_path)
    out: dict[str, float | int | bool | None] = {}
    for key in (
        "n_cells",
        "n_bulk_cells",
        "wall_seconds",
        "end_time",
        "simulation_time",
        "dt",
        "lambda",
        "gamma",
        "paper_mode",
        "minimal_io",
    ):
        if key in data:
            out[key] = data[key]
    return out


def failed_summary_vector(*, fill: float = 1.0e9) -> np.ndarray:
    """Sentinel summary returned when Chaste fails after retries (ABC/SBI-safe)."""
    return np.full(SUMMARY_DIM, float(fill), dtype=np.float64)
