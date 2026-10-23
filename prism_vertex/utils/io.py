from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RunMeta:
    experiment_id: str
    experiment_name: str
    method: str
    obs_id: int
    seed: int
    budget_id: str | None = None


@dataclass(frozen=True)
class CostTracking:
    n_hf_calls: int
    n_lf_calls: int
    c_hf: float
    c_lf: float
    lambda_l_over_h: float
    total_cost: float
    wallclock_seconds_total: float
    train_seconds: float | None = None
    sim_seconds_hf: float | None = None
    sim_seconds_lf: float | None = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_run_config(path: str | Path, meta: RunMeta, config: Mapping[str, Any]) -> None:
    payload = {"meta": asdict(meta), "config": dict(config)}
    write_json(path, payload)


def write_cost_tracking_csv(path: str | Path, meta: RunMeta, cost: CostTracking) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    row = {**asdict(meta), **asdict(cost)}
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_posterior_samples_csv(
    path: str | Path,
    meta: RunMeta,
    theta_names: Sequence[str],
    samples: Sequence[Sequence[float]],
    log_weights: Sequence[float] | None = None,
) -> None:
    """
    Writes posterior samples to CSV.

    Schema:
    - columns: meta fields + theta columns + optional log_weight
    - one row per sample
    """
    p = Path(path)
    ensure_dir(p.parent)

    if log_weights is not None and len(log_weights) != len(samples):
        raise ValueError("log_weights length must match samples length")

    fieldnames = list(asdict(meta).keys()) + list(theta_names) + (["log_weight"] if log_weights is not None else [])

    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        meta_dict = asdict(meta)
        for i, s in enumerate(samples):
            if len(s) != len(theta_names):
                raise ValueError("sample dimension mismatch with theta_names")
            row = {**meta_dict, **{k: float(v) for k, v in zip(theta_names, s)}}
            if log_weights is not None:
                row["log_weight"] = float(log_weights[i])
            writer.writerow(row)


def write_metrics_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    """
    Generic tidy-metrics writer. Provide rows as dicts with identical keys.
    """
    p = Path(path)
    ensure_dir(p.parent)
    rows = list(rows)
    if not rows:
        raise ValueError("rows must be non-empty")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))

