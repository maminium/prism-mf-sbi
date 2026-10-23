"""Twenty calibration observation scenarios for SBC / TARP (paper Fig 5)."""

from __future__ import annotations

import numpy as np

from prism_vertex.simulator.constants import THETA_TRUE

# obs_id -> (lambda, gamma) in Kursawe region B (matches shipped data/observations).
CALIBRATION_THETAS: dict[int, np.ndarray] = {
    0: np.array([-0.09884828856595007, 0.10936307742984773], dtype=np.float64),
    1: np.array([-0.47132135585132595, 0.18630575501663654], dtype=np.float64),
    2: np.array([-0.28245665450503865, 0.1089434929505439], dtype=np.float64),
    3: np.array([-0.3479132137094276, 0.12800103915444735], dtype=np.float64),
    4: np.array([-0.4331848978781758, 0.1621501396090093], dtype=np.float64),
    5: np.array([0.10951964214475773, 0.06276799717253986], dtype=np.float64),
    6: np.array([0.05827056032251554, 0.10005146677689049], dtype=np.float64),
    7: np.array([-0.11410738221629058, 0.058502543888820806], dtype=np.float64),
    8: np.array([-0.11191352591705073, 0.16543400778280154], dtype=np.float64),
    9: np.array([-0.0849803955115731, 0.04467505648136631], dtype=np.float64),
    10: np.array([0.08971670388255948, 0.033103204847235716], dtype=np.float64),
    11: np.array([-0.32731209344277, 0.1856697553279954], dtype=np.float64),
    12: np.array([0.05532170237126066, 0.01177209327210467], dtype=np.float64),
    13: np.array([-0.09350423669226338, 0.13533805431179471], dtype=np.float64),
    14: THETA_TRUE.copy(),
    15: np.array([-0.2566317527999491, 0.16312883506687914], dtype=np.float64),
    16: np.array([-0.2756099101361529, 0.1470452747605304], dtype=np.float64),
    17: np.array([0.04526726350782284, 0.04913276330691172], dtype=np.float64),
    18: np.array([-0.6038112770680362, 0.19906032318171218], dtype=np.float64),
    19: np.array([-0.04216558245122792, 0.03023202360228123], dtype=np.float64),
}

CALIBRATION_OBS_IDS: tuple[int, ...] = tuple(sorted(CALIBRATION_THETAS))


def theta_for_obs(obs_id: int) -> np.ndarray:
    if obs_id not in CALIBRATION_THETAS:
        raise KeyError(f"Unknown calibration obs_id={obs_id}")
    return CALIBRATION_THETAS[obs_id].copy()
