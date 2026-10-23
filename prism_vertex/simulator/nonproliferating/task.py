from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from prism_vertex.simulator.nonproliferating.chaste_runner import ChasteRunError
from prism_vertex.simulator.nonproliferating.chaste_summaries import (
    SUMMARY_DIM,
    SUMMARY_NAMES,
)
from prism_vertex.simulator.nonproliferating.constants import (
    K_HF_REPLICATES,
    PRIOR_HIGH,
    PRIOR_LOW,
    THETA_NAMES,
    THETA_TRUE,
    obs_replicate_seed,
)
from prism_vertex.simulator.nonproliferating.output_subspace import (
    resolve_output_subspace,
    slice_summary,
    subspace_names,
)
from prism_vertex.simulator.nonproliferating.simulator import NonproliferatingVertexSimulator
from prism_vertex.experiments.task import register_task
from prism_vertex.utils.seeding import derive_seed

logger = logging.getLogger(__name__)


class NonproliferatingSimulationDropped(RuntimeError):
    """Raised when a Chaste sim fails and baseline drop-on-fail should omit the row."""


@dataclass(frozen=True)
class NonproliferatingVertexTask:
    """
    Yan/Ogita Stage-A epithelial vertex quench via Chaste NonproliferatingVertex.

    θ = (a_r, b_0, k). Simulator emits 8D geometry summaries; inference uses a
    5D (no mean_s2) or 6D slice controlled by ``PRISM_VERTEX_NONPROLIF_OUTPUTS``.
    LF fidelity is selected by ``PRISM_VERTEX_NONPROLIF_LF_FIDELITY``.
    """

    experiment_id: str = "nonproliferating_vertex"
    experiment_name: str = "yan_epithelial_vertex"
    theta_names: Sequence[str] = THETA_NAMES
    observations_dir: Path = Path("data/nonproliferating_vertex/observations")
    reference_dir: Path = Path("data/nonproliferating_vertex/reference")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        return int(os.environ.get(name, str(default)))

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw not in {"0", "false", "False", "no", "NO"}

    @property
    def k_hf_replicates(self) -> int:
        return max(1, self._env_int("PRISM_VERTEX_NONPROLIF_K_HF_REPLICATES", K_HF_REPLICATES))

    @property
    def use_k_mean_for_hf(self) -> bool:
        return self._env_bool("PRISM_VERTEX_NONPROLIF_HF_K_MEAN", True)

    @property
    def output_subspace(self) -> tuple[int, ...]:
        return resolve_output_subspace()

    @property
    def x_names(self) -> Sequence[str]:
        return subspace_names(self.output_subspace)

    @property
    def observation_dim(self) -> int:
        return len(self.output_subspace)

    @property
    def theta_dim(self) -> int:
        return 3

    @property
    def lf_fidelity_id(self) -> str:
        return os.environ.get("PRISM_VERTEX_NONPROLIF_LF_FIDELITY", "lf_DT2")

    @property
    def sim_jobs(self) -> int:
        return max(1, self._env_int("PRISM_VERTEX_NONPROLIF_SIM_JOBS", 1))

    @property
    def drop_on_fail(self) -> bool:
        return self._env_bool("PRISM_VERTEX_INFERENCE_BASELINE_DROP_ON_FAIL", False) or self._env_bool(
            "PRISM_VERTEX_NONPROLIF_BASELINE_DROP_ON_FAIL", False
        )

    def sample_prior(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(derive_seed("nonproliferating_vertex", "prior", seed)))
        return rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(n, 3)).astype(np.float64)

    def get_prior_distribution(self):
        import torch

        low = torch.as_tensor(PRIOR_LOW, dtype=torch.float32)
        high = torch.as_tensor(PRIOR_HIGH, dtype=torch.float32)
        return torch.distributions.Independent(torch.distributions.Uniform(low, high), 1)

    def get_true_theta(self) -> np.ndarray:
        return THETA_TRUE.copy()

    def get_true_parameters(self, obs_id: int = 0) -> np.ndarray:
        del obs_id
        return THETA_TRUE.copy()

    def _slice(self, x: np.ndarray) -> np.ndarray:
        return slice_summary(x, self.output_subspace)

    def _simulate_one(self, theta: np.ndarray, seed: int, *, hf: bool) -> np.ndarray:
        sim = NonproliferatingVertexSimulator()
        t = np.asarray(theta, dtype=np.float64).reshape(-1)
        try:
            if hf:
                if self.use_k_mean_for_hf:
                    x8 = sim.run_hf_k_mean(t, seed, k=self.k_hf_replicates)
                else:
                    x8 = sim.run_hf_single(t, seed)
            else:
                x8 = sim.run_lf(t, seed, fidelity_id=self.lf_fidelity_id)
            return self._slice(x8)
        except ChasteRunError as exc:
            logger.warning(
                "nonproliferating Chaste failure (%s) theta=%s: %s",
                "hf" if hf else self.lf_fidelity_id,
                t,
                exc,
            )
            if self.drop_on_fail:
                raise NonproliferatingSimulationDropped(str(exc)) from exc
            raise

    def simulate_hf_x(self, theta: np.ndarray, seed: int) -> np.ndarray:
        return self._simulate_one(theta, seed, hf=True)

    def simulate_lf_x(self, theta: np.ndarray, seed: int) -> np.ndarray:
        return self._simulate_one(theta, seed, hf=False)

    def get_observation(self, obs_id: int = 0) -> np.ndarray:
        """Load Yan w165 mean observation (obs_id=0) and slice to inference dim."""
        subspace = self.output_subspace
        w165_mean = self.observations_dir / "x_o_w165_mean.npy"
        if int(obs_id) == 0 and w165_mean.exists():
            return self._slice(np.load(w165_mean))

        if int(obs_id) == 0 and len(subspace) == 6:
            out6 = self.observations_dir / "x_o_w165_out6d.npy"
            if out6.exists():
                return np.asarray(np.load(out6), dtype=np.float64).reshape(-1)

        path = self.observations_dir / f"x_o_seed{obs_id:02d}.npy"
        if path.exists():
            x = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
            if x.size == SUMMARY_DIM:
                return self._slice(x)
            if x.size == len(subspace):
                return x
            raise ValueError(f"Unexpected observation length {x.size} in {path}")

        # Synthetic observation at θ* (one HF run).
        seed = obs_replicate_seed(obs_id)
        os.environ.setdefault("PRISM_VERTEX_NONPROLIF_HF_K_MEAN", "0")
        x8 = NonproliferatingVertexSimulator().run_hf_single(THETA_TRUE, seed)
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, x8)
        return self._slice(x8)

    def prior_log_prob(self, theta) -> Any:
        import torch

        t = theta
        low = torch.as_tensor(PRIOR_LOW, dtype=t.dtype, device=t.device)
        high = torch.as_tensor(PRIOR_HIGH, dtype=t.dtype, device=t.device)
        inside = ((t >= low) & (t <= high)).all(dim=-1)
        vol = float(np.prod(PRIOR_HIGH - PRIOR_LOW))
        log_dens = -float(np.log(vol))
        return torch.where(
            inside,
            torch.full_like(t[..., 0], fill_value=log_dens),
            torch.full_like(t[..., 0], -1e10),
        )


def _factory(config: dict | None = None) -> NonproliferatingVertexTask:
    del config
    return NonproliferatingVertexTask()


register_task("nonproliferating_vertex", _factory)
