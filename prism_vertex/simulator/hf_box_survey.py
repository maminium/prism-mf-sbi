"""Uniform high-fidelity survey of the operational box.

Two hundred parameters, each with ten independent high-fidelity replicates.
A location can support a calibration observation only when all ten succeed.
The ten outcomes are all stored, so the failure fraction is available as well.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prism_vertex.simulator.prior_region import (
    GAM_MAX,
    GAM_MIN,
    LAM_MAX,
    LAM_MIN,
)
from prism_vertex.utils.seeding import derive_seed

SURVEY_SEED = 20261014
N_POINTS = 200
K_HF = 10
N_TASKS = 20
POINTS_PER_TASK = N_POINTS // N_TASKS


def theta_for_point(point_index: int, *, survey_seed: int = SURVEY_SEED) -> np.ndarray:
    rng = np.random.default_rng(derive_seed(survey_seed, "hf_box_survey", int(point_index)))
    lam = float(rng.uniform(LAM_MIN, LAM_MAX))
    gam = float(rng.uniform(GAM_MIN, GAM_MAX))
    return np.array([lam, gam], dtype=np.float64)


def replicate_seed(point_index: int, replicate_k: int, *, survey_seed: int = SURVEY_SEED) -> int:
    return int(derive_seed(survey_seed, "hf_box_survey_rep", int(point_index), int(replicate_k)))


def task_point_ids(task_id: int) -> range:
    if task_id < 0 or task_id >= N_TASKS:
        raise ValueError(f"task_id must be in 0..{N_TASKS - 1}")
    start = int(task_id) * POINTS_PER_TASK
    return range(start, start + POINTS_PER_TASK)


def survey_dir(results_root: Path) -> Path:
    return Path(results_root) / "proliferating_vertex" / "hf_k10_box_survey"


def point_dir(results_root: Path, point_index: int) -> Path:
    return survey_dir(results_root) / "points" / f"point_{int(point_index):04d}"
