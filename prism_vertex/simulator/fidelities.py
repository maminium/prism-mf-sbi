from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

HF_END_TIME = 700.0
HF_DT = 0.01


@dataclass(frozen=True)
class FidelitySpec:
    """
    One Chaste run configuration (paper-mode vertex model).

    Used by the scoreboard and (later) MF inference: HF reference label = mean of
    K=10 ``hf_single`` runs; LF training = summary from a chosen ``lf_*`` spec.

    Replicate LFs (``lf_R*``): same physics as HF single-run; ``n_replicates`` Chaste
    calls per scoreboard cell, stored summary = mean over replicates.
    """

    id: str
    description: str
    end_time: float = HF_END_TIME
    dt: float | None = HF_DT  # None → Chaste paper default (0.01)
    minimal_io: bool = True
    n_replicates: int = 1

    def apply_to_env(self, env: Mapping[str, str] | None = None) -> dict[str, str]:
        out = dict(env or os.environ)
        out["PRISM_VERTEX_PAPER_MODE"] = "1"
        out["PRISM_VERTEX_MINIMAL_IO"] = "1" if self.minimal_io else "0"
        out["PRISM_VERTEX_END_TIME"] = str(self.end_time)
        out["PRISM_VERTEX_END_TIME_HF"] = str(self.end_time)
        if self.dt is not None:
            out["PRISM_VERTEX_DT"] = str(self.dt)
        else:
            out.pop("PRISM_VERTEX_DT", None)
        return out


# Scoreboard + inference catalogue (8 runnable specs).
HF_SINGLE = FidelitySpec(
    id="hf_single",
    description="Full horizon single run (one of K=10 for HF mean)",
    end_time=HF_END_TIME,
    dt=HF_DT,
)

LF_T70 = FidelitySpec(id="lf_T70", description="t_end=70 (0.1×700)", end_time=70.0, dt=HF_DT)
LF_T140 = FidelitySpec(id="lf_T140", description="t_end=140 (0.2×700)", end_time=140.0, dt=HF_DT)
LF_T350 = FidelitySpec(id="lf_T350", description="t_end=350 (0.5×700)", end_time=350.0, dt=HF_DT)

LF_DT02 = FidelitySpec(id="lf_DT02", description="Δt=0.02 (2×)", end_time=HF_END_TIME, dt=0.02)
LF_DT05 = FidelitySpec(id="lf_DT05", description="Δt=0.05 (5×)", end_time=HF_END_TIME, dt=0.05)
LF_DT10 = FidelitySpec(id="lf_DT10", description="Δt=0.10 (10×)", end_time=HF_END_TIME, dt=0.10)

# Replicate family: HF physics, K ∈ {1, 2, 5} (HF reference uses K=10 mean).
LF_R1 = FidelitySpec(
    id="lf_R1",
    description="HF physics, mean of K=1 replicate",
    end_time=HF_END_TIME,
    dt=HF_DT,
    n_replicates=1,
)
LF_R2 = FidelitySpec(
    id="lf_R2",
    description="HF physics, mean of K=2 replicates",
    end_time=HF_END_TIME,
    dt=HF_DT,
    n_replicates=2,
)
LF_R5 = FidelitySpec(
    id="lf_R5",
    description="HF physics, mean of K=5 replicates",
    end_time=HF_END_TIME,
    dt=HF_DT,
    n_replicates=5,
)

# Legacy id (pilot scoreboard); same physics as lf_R1.
LF_S = FidelitySpec(
    id="lf_S",
    description="(legacy) full horizon single run — use lf_R1",
    end_time=HF_END_TIME,
    dt=HF_DT,
    n_replicates=1,
)

ALL_FIDELITIES: dict[str, FidelitySpec] = {
    HF_SINGLE.id: HF_SINGLE,
    LF_T70.id: LF_T70,
    LF_T140.id: LF_T140,
    LF_T350.id: LF_T350,
    LF_DT02.id: LF_DT02,
    LF_DT05.id: LF_DT05,
    LF_DT10.id: LF_DT10,
    LF_R1.id: LF_R1,
    LF_R2.id: LF_R2,
    LF_R5.id: LF_R5,
    LF_S.id: LF_S,
}

LF_FIDELITIES: dict[str, FidelitySpec] = {
    k: v for k, v in ALL_FIDELITIES.items() if k.startswith("lf_") and k != "lf_S"
}

SCOREBOARD_FIDELITY_IDS: tuple[str, ...] = (
    "hf_single",
    "lf_T70",
    "lf_T140",
    "lf_T350",
    "lf_DT02",
    "lf_DT05",
    "lf_DT10",
    "lf_R1",
    "lf_R2",
    "lf_R5",
)

# When building analysis frames: lf_R1 may reuse pilot cells stored as lf_S.
FIDELITY_STORAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "lf_R1": ("lf_R1", "lf_S"),
}


def get_fidelity(fidelity_id: str) -> FidelitySpec:
    key = fidelity_id.strip()
    if key not in ALL_FIDELITIES:
        raise KeyError(f"Unknown fidelity_id={fidelity_id!r}. Choose from {list(ALL_FIDELITIES)}")
    return ALL_FIDELITIES[key]


def lf_fidelity_from_env(default: str = "lf_T140") -> FidelitySpec:
    return get_fidelity(os.environ.get("PRISM_VERTEX_LF_FIDELITY", default))
