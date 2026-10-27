from __future__ import annotations

import os
from typing import Sequence

from prism_vertex.simulator.nonproliferating.chaste_summaries import SUMMARY_NAMES
from prism_vertex.simulator.nonproliferating.constants import (
    INFER_OUTPUT_SUBSPACE,
    INFER_OUTPUT_SUBSPACE_5D,
    INFER_OUTPUT_SUBSPACE_6D,
)


def resolve_output_subspace(outputs: int | None = None) -> tuple[int, ...]:
    """Return 8D index tuple for inference summaries.

    Env ``PRISM_VERTEX_NONPROLIF_OUTPUTS`` = ``5`` or ``6`` (default 6).
    """
    if outputs is None:
        raw = os.environ.get("PRISM_VERTEX_NONPROLIF_OUTPUTS", "").strip()
        if raw:
            outputs = int(raw)
        else:
            return tuple(int(i) for i in INFER_OUTPUT_SUBSPACE)
    if int(outputs) == 5:
        return tuple(int(i) for i in INFER_OUTPUT_SUBSPACE_5D)
    if int(outputs) == 6:
        return tuple(int(i) for i in INFER_OUTPUT_SUBSPACE_6D)
    raise ValueError(f"Unsupported nonproliferating outputs={outputs!r}; use 5 or 6")


def slice_summary(x: Sequence[float] | object, subspace: Sequence[int] | None = None):
    import numpy as np

    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    idx = list(subspace if subspace is not None else resolve_output_subspace())
    if arr.size < max(idx) + 1:
        raise ValueError(f"summary len={arr.size} cannot slice indices {idx}")
    return arr[idx].astype(np.float64, copy=False)


def subspace_names(subspace: Sequence[int] | None = None) -> tuple[str, ...]:
    idx = subspace if subspace is not None else resolve_output_subspace()
    return tuple(SUMMARY_NAMES[i] for i in idx)
