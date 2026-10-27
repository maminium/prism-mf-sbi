"""nonproliferating shared constants"""

from __future__ import annotations

import numpy as np

from prism_vertex.utils.seeding import derive_seed

# θ* = (a_r, b_0, k) — Stage A generative lock
THETA_TRUE = np.array([0.25, 0.15, 5.0], dtype=np.float64)
THETA_NAMES = ("a_r", "b_0", "k")

PRIOR_LOW = np.array([0.05, 0.05, 2.5], dtype=np.float64)
PRIOR_HIGH = np.array([0.50, 0.40, 12.0], dtype=np.float64)

K_HF_REPLICATES = 10

# Legacy Stage-A honeycomb lock (closeness ranking); not MF scoreboard HF.
HF_END_TIME = 80.0
HF_DT = 0.001
HF_INITIAL_SIZE = 12
HF_T1_THRESHOLD = 0.10

HF_MATCH_SUBSPACE = (0, 2, 3, 5)

# Paper / real-data inference + PRISM scoreboard HF reference.
INFER_INPUTS = 3
INFER_OUTPUTS = 6
# 8D indices: mean_log_ell, std_log_ell, mean_c2, mean_s2, ell_c2_cov,
#             mean_log_A, std_log_A, frac_hex
INFER_OUTPUT_SUBSPACE_6D = (0, 1, 2, 3, 5, 7)  # includes mean_s2
INFER_OUTPUT_SUBSPACE_5D = (0, 1, 2, 5, 7)  # drop mean_s2
INFER_OUTPUT_SUBSPACE = INFER_OUTPUT_SUBSPACE_6D
INFER_METHOD = "snpe"
INFER_INIT_MODE = "voronoi"
INFER_VORONOI_RELAX = 0
INFER_END_TIME = 5.0
INFER_BUDGET = 1000
INFER_SEED = 0
INFER_OBS = "w165"
INFER_SIM_TAG = "voronoi_relax0_t5"

# Scoreboard aliases (same as inference HF unless overridden in a fidelity spec).
SCOREBOARD_HF_END_TIME = INFER_END_TIME
SCOREBOARD_HF_DT = HF_DT
SCOREBOARD_HF_INIT_MODE = INFER_INIT_MODE
SCOREBOARD_HF_VORONOI_RELAX = INFER_VORONOI_RELAX
SCOREBOARD_HF_INITIAL_SIZE = HF_INITIAL_SIZE
SCOREBOARD_HF_T1 = HF_T1_THRESHOLD

N_PILOT_THETA = 20
N_PILOT_SEEDS = 10


def obs_replicate_seed(replicate_k: int) -> int:
    return int(derive_seed("nonproliferating_vertex", "obs_replicate", replicate_k))
