"""nonproliferating LF scoreboard I/O — record failures; never abort the campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prism_vertex.simulator.nonproliferating.chaste_runner import (
    ChasteRunError,
    ChasteRunResult,
    run_chaste,
)
from prism_vertex.simulator.nonproliferating.constants import INFER_OUTPUT_SUBSPACE
from prism_vertex.simulator.nonproliferating.fidelities import FidelitySpec
from prism_vertex.simulator.nonproliferating.pilot_design import scoreboard_sim_seed
from prism_vertex.utils.scoreboard_failure_registry import (
    ScoreboardFailureRegistry,
    classify_error,
)
from prism_vertex.utils.seeding import derive_seed


def scoreboard_root(project_root: Path | None = None) -> Path:
    return Path(project_root or "data/nonproliferating_vertex/lf_selection")


def raw_run_dir(root: Path, theta_index: int, seed_k: int, fidelity_id: str) -> Path:
    return root / "raw" / f"theta_{theta_index:02d}" / f"seed_{seed_k:02d}" / fidelity_id


def slice_summary_6d(summary_8d: np.ndarray) -> np.ndarray:
    s = np.asarray(summary_8d, dtype=np.float64).reshape(-1)
    return s[list(INFER_OUTPUT_SUBSPACE)]


def save_run_artifact(out: Path, result: ChasteRunResult) -> None:
    out.mkdir(parents=True, exist_ok=True)
    s8 = np.asarray(result.summary, dtype=np.float64)
    s6 = slice_summary_6d(s8)
    np.save(out / "summary_8d.npy", s8)
    np.save(out / "summary_6d.npy", s6)
    meta = dict(result.meta)
    meta["summary_8d"] = s8.tolist()
    meta["summary_6d"] = s6.tolist()
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))


def _write_failed_meta(out: Path, meta: dict[str, Any]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def run_scoreboard_cell(
    *,
    root: Path,
    theta: np.ndarray,
    theta_index: int,
    seed_k: int,
    spec: FidelitySpec,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    out = raw_run_dir(root, theta_index, seed_k, spec.id)
    meta_json = out / "run_meta.json"
    summary_6d = out / "summary_6d.npy"
    registry = ScoreboardFailureRegistry(root)
    theta_list = np.asarray(theta, dtype=float).reshape(-1).tolist()

    if meta_json.exists() and not force:
        try:
            meta = json.loads(meta_json.read_text())
        except json.JSONDecodeError:
            meta = {}
        st = str(meta.get("status", ""))
        if st == "ok" and summary_6d.exists():
            meta["skipped"] = True
            return meta
        if st == "failed":
            meta["skipped"] = True
            meta.setdefault("error_class", classify_error(meta.get("error"), st))
            return meta

    batch_seed = scoreboard_sim_seed(theta_index, seed_k)
    n_rep = max(1, int(spec.n_replicates))

    try:
        summaries_8: list[np.ndarray] = []
        replicate_meta: list[dict[str, Any]] = []
        wall_total = 0.0
        for rep in range(n_rep):
            sim_seed = int(derive_seed("nonproliferating_vertex", "scoreboard_rep", batch_seed, spec.id, rep))
            subdir = f"sb_t{theta_index:02d}_k{seed_k:02d}_{spec.id}_r{rep}"
            result = run_chaste(
                theta,
                sim_seed,
                spec,
                output_subdir=subdir,
                cleanup_chaste_output=True,
                write_mesh=False,
            )
            summaries_8.append(np.asarray(result.summary, dtype=np.float64))
            rep_meta = dict(result.meta)
            rep_meta["replicate_k"] = rep
            replicate_meta.append(rep_meta)
            wall_total += float(rep_meta.get("wall_seconds", 0.0) or 0.0)

        mean_8 = np.mean(np.stack(summaries_8, axis=0), axis=0)
        mean_6 = slice_summary_6d(mean_8)
        meta = {
            "status": "ok",
            "fidelity_id": spec.id,
            "axis": spec.axis,
            "theta": theta_list,
            "theta_index": int(theta_index),
            "seed_k": int(seed_k),
            "sim_seed": batch_seed,
            "n_replicates": n_rep,
            "summary_8d": mean_8.tolist(),
            "summary_6d": mean_6.tolist(),
            "wall_seconds_python": wall_total,
            "skipped": False,
            "source": "chaste",
            "replicate_meta": replicate_meta,
            "error_class": "none",
            "protocol": {
                "init_mode": spec.init_mode,
                "voronoi_relax": spec.voronoi_relax,
                "end_time": spec.end_time,
                "dt": spec.dt,
                "initial_size": spec.initial_size,
            },
        }
        save_run_artifact(out, ChasteRunResult(summary=mean_8, meta=meta, run_dir=None))
        registry.record_success(
            theta_index=theta_index, seed_k=seed_k, fidelity_id=spec.id, theta=theta_list
        )
        return meta
    except ChasteRunError as exc:
        err = str(exc)[-1500:]
        ec = classify_error(err, "failed")
        rec = registry.record_failure(
            theta_index=theta_index,
            seed_k=seed_k,
            fidelity_id=spec.id,
            theta=theta_list,
            error=err,
            error_class=ec,
        )
        meta = {
            "status": "failed",
            "fidelity_id": spec.id,
            "axis": spec.axis,
            "theta": theta_list,
            "theta_index": int(theta_index),
            "seed_k": int(seed_k),
            "sim_seed": batch_seed,
            "n_replicates": n_rep,
            "error": err,
            "error_class": ec,
            "skipped": False,
            "registry_blocked": False,
            "fail_count": rec.get("fail_count") if isinstance(rec, dict) else None,
        }
        return _write_failed_meta(out, meta)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"[-1500:]
        ec = classify_error(err, "failed")
        rec = registry.record_failure(
            theta_index=theta_index,
            seed_k=seed_k,
            fidelity_id=spec.id,
            theta=theta_list,
            error=err,
            error_class=ec,
        )
        meta = {
            "status": "failed",
            "fidelity_id": spec.id,
            "axis": spec.axis,
            "theta": theta_list,
            "theta_index": int(theta_index),
            "seed_k": int(seed_k),
            "sim_seed": batch_seed,
            "n_replicates": n_rep,
            "error": err,
            "error_class": ec,
            "skipped": False,
            "registry_blocked": False,
            "fail_count": rec.get("fail_count") if isinstance(rec, dict) else None,
        }
        return _write_failed_meta(out, meta)
