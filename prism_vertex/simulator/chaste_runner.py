from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prism_vertex.simulator.chaste_summaries import (
    SUMMARY_DIM,
    SUMMARY_NAMES,
    load_summary_metadata,
    load_summary_vector,
)
from prism_vertex.simulator.fidelities import FidelitySpec
from prism_vertex.utils.process_watchdog import run_with_watchdog
from prism_vertex.utils.seeding import derive_seed


class ChasteRunError(Exception):
    pass


@dataclass
class ChasteRunResult:
    summary_6d: np.ndarray
    meta: dict[str, Any]
    run_dir: Path | None


def _resolve_exe(explicit: str | None, candidates: tuple[str, ...]) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    for raw in candidates:
        path = Path(raw).expanduser()
        if path.is_file():
            return path
    return Path(candidates[0]).expanduser()


def _chaste_paths() -> tuple[Path, Path, Path]:
    exe = _resolve_exe(
        os.environ.get("PRISM_VERTEX_CHASTE_EXE"),
        (
            "~/chaste/build/projects/ProliferatingVertex/test/TestPaperCommandLineSpeedSimulation",
        ),
    )
    cwd = Path(os.environ.get("PRISM_VERTEX_CHASTE_CWD", "~/chaste")).expanduser()
    testoutput = Path(
        os.environ.get("PRISM_VERTEX_TESTOUTPUT_ROOT", "~/chaste/build/testoutput")
    ).expanduser()
    return exe, cwd, testoutput


_SLURM_STEP_KEYS = (
    "SLURM_STEP_ID",
    "SLURM_STEPID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_GTIDS",
    "SLURM_NODEID",
    "SLURM_NTASKS",
    "SLURM_NPROCS",
    "SLURM_TASKS_PER_NODE",
    "SLURM_STEP_NUM_TASKS",
    "SLURM_STEP_TASKS_PER_NODE",
    "SLURM_STEP_NODELIST",
    "SLURM_STEP_NUM_NODES",
    "SLURM_SRUN_COMM_HOST",
    "SLURM_SRUN_COMM_PORT",
    "SLURM_LAUNCH_NODE_IPADDR",
    "SLURM_TASK_PID",
    "SLURM_PTY_PORT",
    "SLURM_CPU_BIND",
    "SLURM_CPU_BIND_TYPE",
    "SLURM_CPU_BIND_LIST",
    "SLURM_CPU_BIND_VERBOSE",
)


def _sanitize_chaste_mpi_env(env: dict[str, str]) -> dict[str, str]:
    """Chaste is a serial PETSc binary. Strip Slurm/PMIx so OpenMPI does not
    treat it as `srun`-direct-launched (PMIX INVALID-NAMESPACE / pml cm)."""
    out = dict(env)
    drop_prefixes = ("PMIX_", "PMI_", "OMPI_COMM_WORLD_", "I_MPI_")
    for key in list(out):
        if key.startswith(drop_prefixes):
            out.pop(key, None)
    for key in _SLURM_STEP_KEYS:
        out.pop(key, None)
    out["OMPI_MCA_ess"] = "singleton"
    out["OMPI_MCA_pml"] = "ob1"
    out["OMPI_MCA_btl"] = "tcp,self"
    out["OMPI_MCA_btl_openib_allow_ib"] = "0"
    out["OMPI_MCA_mpi_warn_on_fork"] = "0"
    out["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"
    out["OMPI_MCA_pmix"] = "isolated"
    return out


def run_chaste(
    theta: np.ndarray,
    sim_seed: int,
    spec: FidelitySpec,
    *,
    output_subdir: str | None = None,
    cleanup_chaste_output: bool | None = None,
) -> ChasteRunResult:
    """
    One Chaste call at ``theta`` with fidelity ``spec``.

    ``sim_seed`` drives Chaste RNG (-opt5 chain); keep identical across HF/LF at same (θ, k).
    """
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    if t.size != 2:
        raise ValueError(f"theta must be (lambda, gamma), got {t.shape}")

    lam, gamma = float(t[0]), float(t[1])
    exe, cwd, testoutput_root = _chaste_paths()
    if not exe.exists():
        raise ChasteRunError(f"Chaste executable not found: {exe}")

    if cleanup_chaste_output is None:
        cleanup_chaste_output = os.environ.get("PRISM_VERTEX_CLEANUP", "1") not in {
            "0",
            "false",
            "False",
        }

    sim_id = int(derive_seed(sim_seed, spec.id, "sim_id") % 1_000_000)
    out_name = output_subdir or f"prolif_{spec.id}_s{sim_seed}_{int(time.time_ns() % 10**9)}"
    run_dir = testoutput_root / out_name

    env = spec.apply_to_env()
    # Optional ultra-short horizon for local wiring smoke tests only.
    smoke_t = os.environ.get("PRISM_VERTEX_SMOKE_END_TIME", "").strip()
    if smoke_t:
        env["PRISM_VERTEX_END_TIME"] = smoke_t
        env["PRISM_VERTEX_END_TIME_HF"] = smoke_t
    env["CHASTE_TEST_OUTPUT"] = str(testoutput_root)
    env["PRISM_VERTEX_OUTPUT_SUBDIR"] = out_name
    env = _sanitize_chaste_mpi_env(env)

    # -opt5 is now used DIRECTLY as the Chaste RNG seed (see TestPaperCommandLineSpeedSimulation.hpp).
    # Use a wide range (~1e9, within int32/stoul limits) so each (theta, seed_k) gets a unique,
    # parameter-independent seed; the old % 10_000 risked collisions across replicates.
    opt5 = int(derive_seed(sim_seed, "chaste", "opt5") % 1_000_000_000)
    args = [
        str(exe),
        "--timeout",
        "1",
        "-opt1",
        str(lam),
        "-opt2",
        str(gamma),
        "-opt3",
        "1",
        "-opt4",
        str(sim_id),
        "-opt5",
        str(opt5),
    ]

    timeout = float(os.environ.get("PRISM_VERTEX_TIMEOUT_SECONDS", "3600"))
    use_watchdog = os.environ.get("PRISM_VERTEX_ENABLE_STALL_WATCHDOG", "1") not in {"0", "false", "False"}
    stall_seconds = float(os.environ.get("PRISM_VERTEX_STALL_SECONDS", "900"))
    stall_poll_seconds = float(os.environ.get("PRISM_VERTEX_STALL_POLL_SECONDS", "60"))
    stall_min_age = float(os.environ.get("PRISM_VERTEX_STALL_MIN_AGE_SECONDS", "120"))
    t0 = time.perf_counter()
    status = "ok"
    err_msg = ""

    try:
        run = run_with_watchdog(
            args=args,
            cwd=str(cwd),
            env=env,
            hard_timeout_s=timeout,
            watch_dir=run_dir,
            enable_stall_watchdog=use_watchdog,
            stall_seconds=stall_seconds,
            stall_poll_seconds=stall_poll_seconds,
            stall_min_age_seconds=stall_min_age,
        )
        if run.returncode != 0:
            status = "timeout" if run.status == "timeout" else ("stalled" if run.status == "stalled" else "chaste_failed")
            err_msg = (run.message + "\n" + (run.stderr or run.stdout or ""))[-2000:]
            raise ChasteRunError(f"rc={run.returncode}: {err_msg}")

        summary_json = run_dir / "prism_vertex_summary.json"
        if not summary_json.exists():
            status = "missing_summary"
            raise ChasteRunError(f"Missing {summary_json}")

        try:
            x = load_summary_vector(run_dir)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            status = "invalid_summary"
            raise ChasteRunError(f"Invalid {summary_json}: {exc}") from exc
        if x.shape[0] != SUMMARY_DIM or not np.isfinite(x).all():
            status = "invalid_summary"
            raise ChasteRunError(f"Invalid summary shape={x.shape}")

        chaste_meta = load_summary_metadata(run_dir)
        wall = time.perf_counter() - t0
        meta = {
            "status": status,
            "fidelity_id": spec.id,
            "theta": [lam, gamma],
            "sim_seed": int(sim_seed),
            "sim_id": sim_id,
            "opt5": opt5,
            "summary_6d": x.tolist(),
            "summary_names": list(SUMMARY_NAMES),
            "wall_seconds_python": wall,
            "end_time": spec.end_time,
            "dt": spec.dt,
            "error": err_msg,
            **{k: chaste_meta.get(k) for k in ("n_cells", "n_bulk_cells", "wall_seconds", "dt")},
        }
        return ChasteRunResult(summary_6d=x, meta=meta, run_dir=run_dir if not cleanup_chaste_output else None)
    finally:
        if cleanup_chaste_output and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


def save_run_artifact(out_dir: Path, result: ChasteRunResult) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "summary_6d.npy", result.summary_6d)
    (out_dir / "run_meta.json").write_text(json.dumps(result.meta, indent=2))


def fidelity_spec_to_dict(spec: FidelitySpec) -> dict:
    return asdict(spec)
