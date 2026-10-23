from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prism_vertex.simulator.chaste_runner import (
    ChasteRunError,
    ChasteRunResult,
    run_chaste,
    save_run_artifact,
)
from prism_vertex.simulator.fidelities import (
    ALL_FIDELITIES,
    FidelitySpec,
    HF_SINGLE,
    get_fidelity,
)
from prism_vertex.simulator.pilot_design import (
    N_PILOT_SEEDS,
    scoreboard_sim_seed,
)
from prism_vertex.simulator.task import THETA_TRUE
from prism_vertex.utils.seeding import derive_seed


def scoreboard_root(project_root: Path | None = None) -> Path:
    root = project_root or Path("data/proliferating_vertex/lf_selection")
    return Path(root)


def raw_run_dir(root: Path, theta_index: int, seed_k: int, fidelity_id: str) -> Path:
    return root / "raw" / f"theta_{theta_index:02d}" / f"seed_{seed_k:02d}" / fidelity_id


def resolve_storage_fidelity_id(fidelity_id: str) -> str:
    """Canonical on-disk folder id (lf_R1 may read legacy lf_S)."""
    from prism_vertex.simulator.fidelities import FIDELITY_STORAGE_ALIASES

    aliases = FIDELITY_STORAGE_ALIASES.get(fidelity_id, (fidelity_id,))
    return aliases[0]


def existing_cell_dir(root: Path, theta_index: int, seed_k: int, fidelity_id: str) -> Path | None:
    """Return first existing raw cell dir for fidelity_id (supports storage aliases)."""
    from prism_vertex.simulator.fidelities import FIDELITY_STORAGE_ALIASES

    for fid in FIDELITY_STORAGE_ALIASES.get(fidelity_id, (fidelity_id,)):
        cell = raw_run_dir(root, theta_index, seed_k, fid)
        if (cell / "summary_6d.npy").exists() or (cell / "run_meta.json").exists():
            return cell
    return raw_run_dir(root, theta_index, seed_k, resolve_storage_fidelity_id(fidelity_id))


def try_load_observation_replicate(seed_k: int) -> np.ndarray | None:
    """Reuse HF replicate from s_obs build at θ_true when available."""
    p = Path("data/proliferating_vertex/observations/replicates") / (
        f"replicate_{seed_k:02d}"
    ) / "summary_6d.npy"
    if p.exists():
        return np.load(p)
    return None


def is_theta_true(theta: np.ndarray, tol: float = 1e-8) -> bool:
    return float(np.linalg.norm(np.asarray(theta) - THETA_TRUE)) < tol


def run_scoreboard_cell(
    *,
    root: Path,
    theta: np.ndarray,
    theta_index: int,
    seed_k: int,
    spec: FidelitySpec,
    force: bool,
) -> dict[str, Any]:
    storage_id = resolve_storage_fidelity_id(spec.id)
    out = raw_run_dir(root, theta_index, seed_k, storage_id)
    summary_npy = out / "summary_6d.npy"
    meta_json = out / "run_meta.json"

    if summary_npy.exists() and meta_json.exists() and not force:
        meta = json.loads(meta_json.read_text())
        meta["skipped"] = True
        meta.setdefault("fidelity_id", spec.id)
        if storage_id != spec.id:
            meta["storage_fidelity_id"] = storage_id
        return meta

    batch_seed = scoreboard_sim_seed(theta_index, seed_k)
    n_rep = max(1, int(spec.n_replicates))

    if n_rep == 1 and spec.id == HF_SINGLE.id and is_theta_true(theta):
        cached = try_load_observation_replicate(seed_k)
        if cached is not None and not force:
            meta = {
                "status": "ok",
                "fidelity_id": spec.id,
                "theta": theta.tolist(),
                "sim_seed": batch_seed,
                "summary_6d": cached.tolist(),
                "source": "observation_replicate_cache",
                "skipped": False,
                "n_replicates": 1,
            }
            save_run_artifact(out, ChasteRunResult(summary_6d=cached, meta=meta, run_dir=None))
            return meta

    try:
        summaries: list[np.ndarray] = []
        replicate_meta: list[dict[str, Any]] = []
        wall_total = 0.0
        for rep in range(n_rep):
            sim_seed = int(derive_seed("proliferating_vertex", "scoreboard_rep", batch_seed, spec.id, rep))
            result = run_chaste(theta, sim_seed, spec)
            summaries.append(np.asarray(result.summary_6d, dtype=np.float64))
            rep_meta = dict(result.meta)
            rep_meta["replicate_k"] = rep
            rep_meta["sim_seed"] = sim_seed
            replicate_meta.append(rep_meta)
            wall_total += float(rep_meta.get("wall_seconds_python", rep_meta.get("wall_seconds", 0.0)) or 0.0)

        if not summaries:
            raise ChasteRunError(f"No successful replicates for {spec.id}")

        mean_summary = np.mean(np.stack(summaries, axis=0), axis=0).astype(np.float32)
        meta = {
            "status": "ok",
            "fidelity_id": spec.id,
            "storage_fidelity_id": storage_id,
            "theta": np.asarray(theta).tolist(),
            "theta_index": theta_index,
            "seed_k": seed_k,
            "sim_seed": batch_seed,
            "n_replicates": n_rep,
            "replicate_sim_seeds": [m.get("sim_seed") for m in replicate_meta],
            "replicate_meta": replicate_meta,
            "summary_6d": mean_summary.tolist(),
            "wall_seconds_python": wall_total,
            "skipped": False,
            "source": "chaste",
        }
        save_run_artifact(out, ChasteRunResult(summary_6d=mean_summary, meta=meta, run_dir=None))
        return meta
    except ChasteRunError as exc:
        meta = {
            "status": "failed",
            "fidelity_id": spec.id,
            "storage_fidelity_id": storage_id,
            "theta": np.asarray(theta).tolist(),
            "theta_index": theta_index,
            "seed_k": seed_k,
            "sim_seed": batch_seed,
            "n_replicates": n_rep,
            "error": str(exc)[-2000:],
            "skipped": False,
        }
        out.mkdir(parents=True, exist_ok=True)
        meta_json.write_text(json.dumps(meta, indent=2))
        return meta


def list_jobs(
    theta_list: np.ndarray,
    fidelity_ids: list[str],
) -> list[tuple[int, int, str]]:
    jobs: list[tuple[int, int, str]] = []
    for i in range(theta_list.shape[0]):
        for k in range(N_PILOT_SEEDS):
            for fid in fidelity_ids:
                jobs.append((i, k, fid))
    return jobs
