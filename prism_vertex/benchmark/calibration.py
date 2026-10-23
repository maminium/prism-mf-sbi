"""MF-SBI calibration paths for the paper operating point (Fig 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from prism_vertex.benchmark.comparison_paths import method_results_dir

CALIBRATION_METHODS: tuple[str, ...] = ("mf_sbi",)
CALIBRATION_BUDGET_DIR = "budget_lf_010000_hf_000500"
CALIBRATION_LF_ID = "lf_DT05"
CALIBRATION_RUN_GROUP = "calibration"
CALIBRATION_COMPARISON_SUBDIR = Path("results/calibration")


def calibration_method_dir(exp_dir: Path, method: str) -> Path:
    """Posterior run directory for the calibration observations."""
    return (
        method_results_dir(exp_dir, method)
        / CALIBRATION_RUN_GROUP
        / CALIBRATION_LF_ID
        / CALIBRATION_BUDGET_DIR
    )


def calibration_comparison_dir(exp_dir: Path) -> Path:
    return exp_dir / CALIBRATION_COMPARISON_SUBDIR


def iter_calibration_run_dirs(
    exp_dir: Path,
    *,
    methods: tuple[str, ...] = CALIBRATION_METHODS,
    lf_id: str = CALIBRATION_LF_ID,
    budget_dir: str = CALIBRATION_BUDGET_DIR,
) -> Iterator[tuple[Path, str, int, int]]:
    """Yield ``(run_dir, method, obs_id, seed)``."""
    for method in methods:
        method_dir = calibration_method_dir(exp_dir, method)
        if not method_dir.is_dir():
            continue
        for obs_dir in sorted(method_dir.glob("obs_*")):
            if not obs_dir.is_dir():
                continue
            obs_id = int(obs_dir.name.replace("obs_", ""))
            for seed_dir in sorted(obs_dir.glob("seed_*")):
                if not seed_dir.is_dir():
                    continue
                seed = int(seed_dir.name.replace("seed_", ""))
                yield seed_dir, method, obs_id, seed
