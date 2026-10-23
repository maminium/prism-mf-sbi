"""shared constants (import-safe: no task/simulator dependencies)."""

from __future__ import annotations

import numpy as np

from prism_vertex.utils.seeding import derive_seed

# Kursawe et al. (2018) reference truth (region B).
THETA_TRUE = np.array([0.12, 0.04], dtype=np.float64)
# Paper: average summary over 10 independent tissues at the same parameters.
K_HF_REPLICATES = 10


def obs_replicate_seed(replicate_k: int) -> int:
    """Fixed RNG seeds for the single s_obs (10 HF runs at theta_true)."""
    return int(derive_seed("proliferating_vertex", "obs_replicate", replicate_k))
