"""Inference-time context: observation, penalty scales, failure registry (proliferating wrapper)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prism_vertex.simulator.inference_failure_registry import (
    InferenceFailureRegistry,
)
from prism_vertex.simulator.penalty_summaries import default_penalty_scales
from prism_vertex.inference import inference_context


def init_inference_simulation(
    *,
    x_o: np.ndarray,
    run_dir: Path,
    observations_dir: Path | None = None,
) -> InferenceFailureRegistry:
    """Call once at the start of each proliferating inference run."""
    return inference_context.init_inference_simulation(
        x_o=x_o,
        run_dir=run_dir,
        scales=default_penalty_scales(observations_dir),
    )


def get_inference_x_o() -> np.ndarray | None:
    return inference_context.get_inference_x_o()


def get_penalty_scales() -> np.ndarray:
    return inference_context.get_penalty_scales()


def make_penalty_summary() -> np.ndarray:
    return inference_context.make_penalty_summary()


def get_failure_registry() -> InferenceFailureRegistry:
    return inference_context.get_failure_registry()


def baseline_drop_on_fail_enabled() -> bool:
    return inference_context.baseline_drop_on_fail_enabled()


def penalty_mode_enabled() -> bool:
    return inference_context.penalty_mode_enabled()
