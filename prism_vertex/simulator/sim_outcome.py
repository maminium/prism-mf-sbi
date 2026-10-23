"""Per-simulation outcome metadata (status, feasibility label)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SimulationDropped(Exception):
    """Raised when baseline drop-on-fail policy omits a failed simulation."""


@dataclass(frozen=True)
class SimOutcome:
    theta: np.ndarray
    summary_6d: np.ndarray
    m_ok: bool
    status: str
    fidelity: str  # "hf" | "lf"
    is_penalty: bool
    skipped_registry: bool = False


FIDELITY_HF = 0
FIDELITY_LF = 1
