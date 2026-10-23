from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from prism_vertex.simulator.nonproliferating.constants import (
    HF_DT,
    HF_END_TIME,
    HF_INITIAL_SIZE,
    HF_T1_THRESHOLD,
    INFER_INIT_MODE,
    INFER_VORONOI_RELAX,
    SCOREBOARD_HF_END_TIME,
    SCOREBOARD_HF_INIT_MODE,
    SCOREBOARD_HF_VORONOI_RELAX,
)


@dataclass(frozen=True)
class FidelitySpec:
    id: str
    description: str
    end_time: float = SCOREBOARD_HF_END_TIME
    dt: float | None = HF_DT
    initial_size: int = HF_INITIAL_SIZE
    t1_threshold: float = HF_T1_THRESHOLD
    init_mode: str = SCOREBOARD_HF_INIT_MODE
    voronoi_relax: int = SCOREBOARD_HF_VORONOI_RELAX
    minimal_io: bool = True
    n_replicates: int = 1
    axis: str = "hf"
    write_mesh: bool = False

    def apply_to_env(self, env: Mapping[str, str] | None = None) -> dict[str, str]:
        out = dict(env or os.environ)
        out["PRISM_VERTEX_YAN_MINIMAL_IO"] = "1" if self.minimal_io else "0"
        out["PRISM_VERTEX_YAN_END_TIME"] = str(self.end_time)
        out["PRISM_VERTEX_YAN_INITIAL_SIZE"] = str(self.initial_size)
        out["PRISM_VERTEX_YAN_T1_THRESHOLD"] = str(self.t1_threshold)
        out["PRISM_VERTEX_YAN_INIT_MODE"] = str(self.init_mode)
        out["PRISM_VERTEX_YAN_VORONOI_RELAX"] = str(int(self.voronoi_relax))
        out["PRISM_VERTEX_YAN_CHECK_INTERSECTIONS"] = "0"
        out["PRISM_VERTEX_YAN_DISABLE_ADAPTIVE_DT"] = "1"
        out["PRISM_VERTEX_YAN_INIT_VERTEX_NOISE"] = out.get("PRISM_VERTEX_YAN_INIT_VERTEX_NOISE", "0")
        out["PRISM_VERTEX_YAN_RANDOMIZE_T1"] = out.get("PRISM_VERTEX_YAN_RANDOMIZE_T1", "0")
        if self.write_mesh:
            out["PRISM_VERTEX_YAN_WRITE_MESH"] = "1"
        else:
            out.setdefault("PRISM_VERTEX_YAN_WRITE_MESH", "0")
        if self.dt is not None:
            out["PRISM_VERTEX_YAN_DT"] = str(self.dt)
        return out


# --- Scoreboard HF (Voronoi T=5; matches inference lock) ---
HF_SINGLE = FidelitySpec(
    id="hf_single",
    description="HF Voronoi relax=0, T=5 (inference protocol)",
    axis="hf",
)

# --- Axis T (fractions of SCOREBOARD_HF_END_TIME=5) ---
LF_T05 = FidelitySpec(
    id="lf_T05",
    description="t_end=0.5×HF (2.5)",
    end_time=0.5 * SCOREBOARD_HF_END_TIME,
    axis="T",
)
LF_T025 = FidelitySpec(
    id="lf_T025",
    description="t_end=0.25×HF (1.25)",
    end_time=0.25 * SCOREBOARD_HF_END_TIME,
    axis="T",
)
LF_T0125 = FidelitySpec(
    id="lf_T0125",
    description="t_end=0.125×HF (0.625)",
    end_time=0.125 * SCOREBOARD_HF_END_TIME,
    axis="T",
)

# --- Axis DT ---
LF_DT2 = FidelitySpec(id="lf_DT2", description="Δt=2×HF", dt=2.0 * HF_DT, axis="DT")
LF_DT5 = FidelitySpec(id="lf_DT5", description="Δt=5×HF", dt=5.0 * HF_DT, axis="DT")
LF_DT10 = FidelitySpec(id="lf_DT10", description="Δt=10×HF", dt=10.0 * HF_DT, axis="DT")

# --- Axis MESH ---
LF_MESH8 = FidelitySpec(id="lf_MESH8", description="8×8 mesh", initial_size=8, axis="MESH")
LF_MESH4 = FidelitySpec(id="lf_MESH4", description="4×4 mesh", initial_size=4, axis="MESH")

# --- Axis INIT (nonproliferating-specific) ---
LF_HONEY = FidelitySpec(
    id="lf_HONEY",
    description="Honeycomb init, same T=5",
    init_mode="honeycomb",
    voronoi_relax=0,
    axis="INIT",
)
LF_VORO_RELAX5 = FidelitySpec(
    id="lf_VORO_RELAX5",
    description="Voronoi + 5 Lloyd relax steps",
    init_mode="voronoi",
    voronoi_relax=5,
    axis="INIT",
)

# --- Axis R ---
LF_R1 = FidelitySpec(id="lf_R1", description="HF physics, K=1", n_replicates=1, axis="R")
LF_R2 = FidelitySpec(id="lf_R2", description="HF physics, K=2", n_replicates=2, axis="R")
LF_R5 = FidelitySpec(id="lf_R5", description="HF physics, K=5", n_replicates=5, axis="R")

# Legacy honeycomb long-quench (Stage-A only; not in scoreboard catalogue).
HF_HONEY_T80 = FidelitySpec(
    id="hf_honey_t80",
    description="Legacy honeycomb T=80",
    end_time=HF_END_TIME,
    init_mode="honeycomb",
    voronoi_relax=0,
    axis="hf",
)

ALL_FIDELITIES: dict[str, FidelitySpec] = {
    s.id: s
    for s in (
        HF_SINGLE,
        LF_T0125,
        LF_T025,
        LF_T05,
        LF_DT2,
        LF_DT5,
        LF_DT10,
        LF_MESH8,
        LF_MESH4,
        LF_HONEY,
        LF_VORO_RELAX5,
        LF_R1,
        LF_R2,
        LF_R5,
        HF_HONEY_T80,
    )
}

SCOREBOARD_FIDELITY_IDS: tuple[str, ...] = (
    "hf_single",
    "lf_T0125",
    "lf_T025",
    "lf_T05",
    "lf_DT2",
    "lf_DT5",
    "lf_DT10",
    "lf_MESH8",
    "lf_MESH4",
    "lf_HONEY",
    "lf_VORO_RELAX5",
    "lf_R1",
    "lf_R2",
    "lf_R5",
)

LF_SCOREBOARD_IDS: tuple[str, ...] = tuple(
    fid for fid in SCOREBOARD_FIDELITY_IDS if fid != "hf_single"
)


def get_fidelity(fidelity_id: str) -> FidelitySpec:
    key = fidelity_id.strip()
    if key not in ALL_FIDELITIES:
        raise KeyError(f"Unknown nonproliferating fidelity: {fidelity_id!r}")
    return ALL_FIDELITIES[key]


FIDELITY_AXIS: dict[str, str] = {fid: get_fidelity(fid).axis for fid in SCOREBOARD_FIDELITY_IDS}


def lf_fidelity_from_env(default: str = "lf_DT2") -> FidelitySpec:
    return get_fidelity(os.environ.get("PRISM_VERTEX_NONPROLIF_LF_FIDELITY", default))
