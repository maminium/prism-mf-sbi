#!/usr/bin/env python3
"""
Build the single Vertex tissue synthetic observation.

  theta_true = (0.12, 0.04)
  K HF runs (default 10), different RNG seeds, parallel with --jobs
  s_obs = mean of K six-dimensional summaries

Each replicate is stored under:
  data/proliferating_vertex/observations/replicates/replicate_XX/

Terminal example (10 cores, paper HF t=700):
  source ~/venvs/main/bin/activate
  export PYTHONPATH="$PRISM_VERTEX_ROOT"
  source ~/chaste/projects/ProliferatingVertex/scripts/setup_paper_mode_env.sh
  cd "$PRISM_VERTEX_ROOT"
  python scripts/lf_selection/generate_observations.py --jobs 10

Then visualize:
  python scripts/lf_selection/generate_observations.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism_vertex.simulator.chaste_summaries import (  # noqa: E402
    SUMMARY_DIM,
    SUMMARY_NAMES,
)
from prism_vertex.simulator.task import (  # noqa: E402
    THETA_TRUE,
    K_HF_REPLICATES,
    TissueVertexTask,
    obs_replicate_seed,
)


def _setup_paper_env() -> None:
    os.environ.setdefault("PRISM_VERTEX_PAPER_MODE", "1")
    os.environ.setdefault("PRISM_VERTEX_MINIMAL_IO", "1")
    os.environ.setdefault("PRISM_VERTEX_END_TIME", "700")
    os.environ.setdefault("PRISM_VERTEX_END_TIME_HF", "700")
    # Observation build: keep Chaste run dirs under replicates/ for audit.
    os.environ.setdefault("PRISM_VERTEX_CLEANUP", "0")


def _replicate_dir(observations_dir: Path, replicate_k: int) -> Path:
    return observations_dir / "replicates" / f"replicate_{replicate_k:02d}"


def _run_one_replicate(
    replicate_k: int,
    observations_dir: str,
    force: bool,
    theta_true: np.ndarray,
) -> dict[str, Any]:
    """Worker: one HF Chaste call at theta_true (runs in a separate process)."""
    _setup_paper_env()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from prism_vertex.simulator.task import (  # noqa: E402
        TissueVertexTask,
        obs_replicate_seed,
    )

    theta_true = np.asarray(theta_true, dtype=np.float64).reshape(-1)
    obs_path = Path(observations_dir)
    rep_dir = _replicate_dir(obs_path, replicate_k)
    summary_npy = rep_dir / "summary_6d.npy"
    meta_json = rep_dir / "run_meta.json"

    if summary_npy.exists() and meta_json.exists() and not force:
        x = np.load(summary_npy)
        meta = json.loads(meta_json.read_text())
        meta["skipped"] = True
        return meta

    rep_dir.mkdir(parents=True, exist_ok=True)
    sim_seed = obs_replicate_seed(replicate_k)
    task = TissueVertexTask()

    t0 = time.perf_counter()
    x = task.simulate_hf_x_strict(theta_true, seed=sim_seed)
    wall_s = time.perf_counter() - t0

    np.save(summary_npy, x)
    meta = {
        "replicate_k": replicate_k,
        "sim_seed": int(sim_seed),
        "theta_true": theta_true.tolist(),
        "summary_6d": x.tolist(),
        "summary_names": list(SUMMARY_NAMES),
        "wall_seconds_python": wall_s,
        "end_time_hf": task.end_time_hf,
        "skipped": False,
    }
    meta_json.write_text(json.dumps(meta, indent=2))
    return meta


def _aggregate_replicates(observations_dir: Path, k: int) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    run_metas: list[dict] = []
    for i in range(k):
        rep_dir = _replicate_dir(observations_dir, i)
        summary_npy = rep_dir / "summary_6d.npy"
        if not summary_npy.exists():
            raise FileNotFoundError(f"Missing replicate output: {summary_npy}")
        rows.append(np.load(summary_npy))
        meta_path = rep_dir / "run_meta.json"
        if meta_path.exists():
            run_metas.append(json.loads(meta_path.read_text()))

    stack = np.stack(rows, axis=0)  # (K, 6)
    mean = stack.mean(axis=0)
    var = stack.var(axis=0, ddof=1) if k > 1 else np.zeros(SUMMARY_DIM)
    std = np.sqrt(var)
    sem = std / np.sqrt(k) if k > 0 else std

    stats = {
        "k": k,
        "summary_names": list(SUMMARY_NAMES),
        "mean": mean.tolist(),
        "var": var.tolist(),
        "std": std.tolist(),
        "sem": sem.tolist(),
        "min": stack.min(axis=0).tolist(),
        "max": stack.max(axis=0).tolist(),
        "replicate_matrix": stack.tolist(),
        "runs": run_metas,
    }
    return stats


def _generate_one_obs(
    obs_id: int,
    theta_true: np.ndarray,
    *,
    k: int,
    jobs: int,
    force: bool,
    sequential: bool,
) -> Path:
    task = TissueVertexTask()
    if obs_id == 0 and (task.observations_dir / "s_obs.npy").exists() and not force:
        out_dir = task.observations_dir
    else:
        out_dir = task.observations_dir / f"obs_{obs_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [obs_replicate_seed(i) for i in range(k)]
    jobs = max(1, min(jobs, k))
    theta_true = np.asarray(theta_true, dtype=np.float64).reshape(-1)

    print("=" * 60)
    print(f"Vertex tissue observation obs_id={obs_id}")
    print(f"theta_true     = {theta_true.tolist()}")
    print(f"K replicates   = {k}")
    print(f"parallel jobs  = {jobs}")
    print(f"output_dir     = {out_dir.resolve()}")
    print("=" * 60)

    if not task.chaste_exe.exists():
        sys.exit(f"ERROR: Chaste executable not found: {task.chaste_exe}")

    t_all = time.perf_counter()
    completed = 0
    errors: list[str] = []

    def _report(meta: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        rk = meta["replicate_k"]
        tag = "SKIP" if meta.get("skipped") else "DONE"
        wall = meta.get("wall_seconds_python", float("nan"))
        print(f"[{completed:2d}/{k}] {tag} replicate {rk:02d}  wall={wall:.1f}s", flush=True)

    if sequential:
        for i in range(k):
            try:
                meta = _run_one_replicate(i, str(out_dir), force, theta_true)
                _report(meta)
            except Exception as exc:
                errors.append(f"replicate {i:02d}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(_run_one_replicate, i, str(out_dir), force, theta_true): i for i in range(k)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    meta = fut.result()
                    _report(meta)
                except Exception as exc:
                    errors.append(f"replicate {i:02d}: {exc}")

    if errors:
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    stats = _aggregate_replicates(out_dir, k)
    s_obs = np.asarray(stats["mean"], dtype=np.float64)
    np.save(out_dir / "theta_true.npy", theta_true)
    np.save(out_dir / f"theta_{obs_id:02d}.npy", theta_true)
    np.save(out_dir / "s_obs.npy", s_obs)
    np.save(out_dir / f"obs_{obs_id:02d}.npy", s_obs)
    np.save(out_dir / "replicate_matrix.npy", np.asarray(stats["replicate_matrix"]))
    np.save(out_dir / "s_obs_std.npy", np.asarray(stats["std"]))
    np.save(out_dir / "s_obs_sem.npy", np.asarray(stats["sem"]))
    stats["obs_id"] = obs_id
    stats["s_obs"] = s_obs.tolist()
    stats["theta_true"] = theta_true.tolist()
    stats["total_elapsed_s"] = time.perf_counter() - t_all
    (out_dir / "aggregate_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_dir}/s_obs.npy  elapsed={stats['total_elapsed_s']:.1f}s")
    return out_dir


def main() -> None:
    from prism_vertex.simulator.calibration_observations import (  # noqa: E402
        CALIBRATION_OBS_IDS,
        theta_for_obs,
    )

    ap = argparse.ArgumentParser(description="Generate Vertex tissue s_obs (mean of K HF replicates).")
    ap.add_argument("--obs_id", type=int, default=0, help="Observation index 0–9")
    ap.add_argument("--all_calibration", action="store_true", help="Generate obs 0–9 for calibration study")
    ap.add_argument("--k", type=int, default=K_HF_REPLICATES, help="Number of HF replicates (default 10)")
    ap.add_argument("--jobs", type=int, default=10, help="Parallel worker processes (default 10)")
    ap.add_argument("--force", action="store_true", help="Re-run even if replicate files exist")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sequential", action="store_true", help="Disable parallelism (debug)")
    args = ap.parse_args()

    _setup_paper_env()

    if args.dry_run:
        ids = list(CALIBRATION_OBS_IDS) if args.all_calibration else [args.obs_id]
        for oid in ids:
            print(f"obs_{oid:02d}: theta={theta_for_obs(oid).tolist()}")
        return

    if args.all_calibration:
        for oid in CALIBRATION_OBS_IDS:
            _generate_one_obs(
                oid,
                theta_for_obs(oid),
                k=args.k,
                jobs=args.jobs,
                force=args.force,
                sequential=args.sequential,
            )
        return

    theta = theta_for_obs(args.obs_id) if args.obs_id in CALIBRATION_OBS_IDS else THETA_TRUE
    _generate_one_obs(
        args.obs_id,
        theta,
        k=args.k,
        jobs=args.jobs,
        force=args.force,
        sequential=args.sequential,
    )


if __name__ == "__main__":
    main()
