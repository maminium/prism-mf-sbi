from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prism_vertex.simulator.nonproliferating.chaste_summaries import (
    SUMMARY_DIM,
    load_summary_metadata,
    load_summary_vector,
)
from prism_vertex.simulator.nonproliferating.fidelities import FidelitySpec
from prism_vertex.utils.process_watchdog import run_with_watchdog
from prism_vertex.utils.seeding import derive_seed


class ChasteRunError(Exception):
    pass


@dataclass
class ChasteRunResult:
    summary: np.ndarray
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
        os.environ.get("PRISM_VERTEX_NONPROLIF_CHASTE_EXE")
        or os.environ.get("PRISM_VERTEX_YAN_CHASTE_EXE"),
        (
            "~/chaste/build/projects/NonproliferatingVertex/test/TestYanVertexCommandLine",
        ),
    )
    cwd = Path(
        os.environ.get("PRISM_VERTEX_CHASTE_CWD", os.environ.get("PRISM_VERTEX_YAN_CHASTE_CWD", "~/chaste"))
    ).expanduser()
    testoutput = Path(
        os.environ.get(
            "PRISM_VERTEX_TESTOUTPUT_ROOT",
            os.environ.get("PRISM_VERTEX_YAN_TESTOUTPUT_ROOT", "~/chaste/build/testoutput"),
        )
    ).expanduser()
    return exe, cwd, testoutput


def run_chaste(
    theta: np.ndarray,
    sim_seed: int,
    spec: FidelitySpec,
    *,
    output_subdir: str | None = None,
    cleanup_chaste_output: bool | None = None,
    write_mesh: bool | None = None,
) -> ChasteRunResult:
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    if t.size != 3:
        raise ValueError(f"theta must be (a_r, b_0, k), got shape {t.shape}")

    a_r, b_0, k = float(t[0]), float(t[1]), float(t[2])
    exe, cwd, testoutput_root = _chaste_paths()
    if not exe.exists():
        raise ChasteRunError(f"Chaste executable not found: {exe}")

    if cleanup_chaste_output is None:
        cleanup_chaste_output = os.environ.get("PRISM_VERTEX_NONPROLIF_CLEANUP", "1") not in {
            "0",
            "false",
            "False",
        }

    sim_id = int(derive_seed(sim_seed, spec.id, "sim_id") % 1_000_000)
    out_name = output_subdir or f"nonprolif_{spec.id}_s{sim_seed}_{int(time.time_ns() % 10**9)}"
    run_dir = testoutput_root / out_name

    mesh_flag = spec.write_mesh if write_mesh is None else write_mesh
    env = spec.apply_to_env()
    env["CHASTE_TEST_OUTPUT"] = str(testoutput_root)
    env["PRISM_VERTEX_YAN_OUTPUT_SUBDIR"] = out_name
    env["PRISM_VERTEX_YAN_WRITE_MESH"] = "1" if mesh_flag else "0"

    opt5 = int(derive_seed(sim_seed, "chaste", "opt5") % 1_000_000_000)
    args = [
        str(exe),
        "--timeout",
        "1",
        "-opt1",
        str(a_r),
        "-opt2",
        str(b_0),
        "-opt3",
        str(k),
        "-opt4",
        str(sim_id),
        "-opt5",
        str(opt5),
    ]

    timeout = float(os.environ.get("PRISM_VERTEX_NONPROLIF_TIMEOUT_SECONDS", "3600"))
    use_watchdog = os.environ.get("PRISM_VERTEX_NONPROLIF_ENABLE_STALL_WATCHDOG", "1") not in {
        "0",
        "false",
        "False",
    }
    stall_seconds = float(os.environ.get("PRISM_VERTEX_NONPROLIF_STALL_SECONDS", "900"))
    stall_poll_seconds = float(os.environ.get("PRISM_VERTEX_NONPROLIF_STALL_POLL_SECONDS", "60"))
    stall_min_age = float(os.environ.get("PRISM_VERTEX_NONPROLIF_STALL_MIN_AGE_SECONDS", "120"))

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
            raise ChasteRunError(
                f"Chaste exited {run.returncode}: {(run.stderr or run.stdout or '')[-2000:]}"
            )
        summary = load_summary_vector(run_dir)
        if summary.shape[0] != SUMMARY_DIM or not np.isfinite(summary).all():
            raise ChasteRunError(f"Invalid nonproliferating summary: {summary}")
        meta = load_summary_metadata(run_dir)
        meta["run_dir"] = str(run_dir)
        result = ChasteRunResult(summary=summary, meta=meta, run_dir=run_dir)
    except Exception as exc:
        if cleanup_chaste_output and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        if isinstance(exc, ChasteRunError):
            raise
        raise ChasteRunError(str(exc)) from exc

    if cleanup_chaste_output and run_dir.exists() and not mesh_flag:
        # Keep mesh runs for figures; otherwise clean.
        shutil.rmtree(run_dir, ignore_errors=True)
        result.run_dir = None
    return result
