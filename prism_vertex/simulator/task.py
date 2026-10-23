from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from concurrent.futures import ProcessPoolExecutor, as_completed

from prism_vertex.simulator.chaste_runner import ChasteRunError
from prism_vertex.simulator.chaste_summaries import (
    SUMMARY_DIM,
    SUMMARY_NAMES,
)
from prism_vertex.simulator.inference_simulation import (
    baseline_drop_on_fail_enabled,
    get_failure_registry,
    make_penalty_summary,
)
from prism_vertex.simulator.sim_outcome import SimulationDropped
from prism_vertex.simulator.constants import (
    K_HF_REPLICATES,
    THETA_TRUE,
    obs_replicate_seed,
)
from prism_vertex.simulator.simulator import ProliferatingVertexSimulator
from prism_vertex.experiments.task import register_task
from prism_vertex.utils.seeding import derive_seed

logger = logging.getLogger(__name__)


class SimulationFailed(Exception):
    """Chaste run failed, timed out, or produced an invalid 6D summary."""


def _prolif_hf_batch_worker(args: tuple[np.ndarray, int]) -> np.ndarray:
    theta, seed = args
    task = ProliferatingVertexTask()
    if task.use_k_mean_for_hf:
        return task.simulate_hf_x_k_mean(theta, seed=seed)
    return task._simulate_hf_k_mean_one(theta, seed=seed, k=1)


def _prolif_lf_batch_worker(args: tuple[np.ndarray, int]) -> np.ndarray:
    theta, seed = args
    return ProliferatingVertexTask()._simulate_lf_one(theta, seed)


@dataclass(frozen=True)
class ProliferatingVertexTask:
    """
    Kursawe et al. (2018) vertex tissue model via Chaste ProliferatingVertex.

    θ = (lambda, gamma) — line tension and perimeter contractility (Farhadifar).
    x = 6D Kursawe summary: area ratios (polygon classes 4–8) + mean elongation.
    HF label: mean of K=10 runs at full paper settings (``HF_SINGLE``).
    LF label: one run at ``PRISM_VERTEX_LF_FIDELITY`` (e.g. ``lf_DT10``).

    Failed or timed-out Chaste runs are retried; persistent failures return an
    observation-opposed penalty summary so MF-SBI training continues and learns
    to avoid fault regions (see penalty summaries).
    """

    experiment_id: str = "proliferating_vertex"
    experiment_name: str = "bayesian_tissue_vertex"
    theta_names: Sequence[str] = ("lambda", "gamma")
    x_names: Sequence[str] = SUMMARY_NAMES
    observations_dir: Path = Path("data/proliferating_vertex/observations")
    reference_dir: Path = Path("data/proliferating_vertex/reference")

    @property
    def chaste_exe(self) -> Path:
        return Path(
            os.environ.get(
                "PRISM_VERTEX_CHASTE_EXE",
                "~/chaste/build/projects/ProliferatingVertex/test/TestPaperCommandLineSpeedSimulation",
            )
        ).expanduser()

    @property
    def chaste_cwd(self) -> Path:
        return Path(os.environ.get("PRISM_VERTEX_CHASTE_CWD", "~/chaste")).expanduser()

    @property
    def chaste_testoutput_root(self) -> Path:
        return Path(
            os.environ.get("PRISM_VERTEX_TESTOUTPUT_ROOT", "~/chaste/build/testoutput")
        ).expanduser()

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        return float(os.environ.get(name, str(default)))

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
    def paper_mode(self) -> bool:
        return self._env_bool("PRISM_VERTEX_PAPER_MODE", True)

    @property
    def minimal_io(self) -> bool:
        if "PRISM_VERTEX_MINIMAL_IO" in os.environ:
            return self._env_bool("PRISM_VERTEX_MINIMAL_IO", True)
        return self.paper_mode

    @property
    def end_time_hf(self) -> float:
        if "PRISM_VERTEX_END_TIME_HF" in os.environ:
            return self._env_float("PRISM_VERTEX_END_TIME_HF", 700.0)
        return self._env_float("PRISM_VERTEX_END_TIME", 700.0)

    @property
    def end_time_lf(self) -> float:
        return self._env_float("PRISM_VERTEX_END_TIME_LF", 50.0)

    @property
    def timeout_seconds(self) -> float:
        return self._env_float("PRISM_VERTEX_TIMEOUT_SECONDS", 3600.0)

    @property
    def max_sim_retries(self) -> int:
        return max(1, self._env_int("PRISM_VERTEX_MAX_SIM_RETRIES", 3))

    @property
    def failed_x_fill(self) -> float:
        return self._env_float("PRISM_VERTEX_FAILED_X_FILL", 1.0e9)

    @property
    def cleanup_testoutput(self) -> bool:
        return self._env_bool("PRISM_VERTEX_CLEANUP", True)

    @property
    def cleanup_on_fail(self) -> bool:
        return self._env_bool("PRISM_VERTEX_CLEANUP_ON_FAIL", False)

    @property
    def log_sim_failures(self) -> bool:
        return self._env_bool("PRISM_VERTEX_LOG_FAILURES", True)

    @property
    def hf_strict_any_fail(self) -> bool:
        return self._env_bool("PRISM_VERTEX_HF_STRICT_ANY_FAIL", True)

    def failed_summary(self) -> np.ndarray:
        return make_penalty_summary()

    def _failure_result(
        self,
        theta: np.ndarray,
        *,
        fidelity: str,
        failure_class: str = "sim_failure",
        skipped: bool = False,
    ) -> np.ndarray:
        """Penalty label for penalty-label mode, or raise for baseline drop-on-fail."""
        self._register_sim_failure(
            theta,
            fidelity=fidelity,
            failure_class=failure_class,
            skipped=skipped,
        )
        if baseline_drop_on_fail_enabled():
            raise SimulationDropped(
                f"{fidelity} simulation failed at theta=({float(theta[0]):.4f}, {float(theta[1]):.4f})"
            )
        return self.failed_summary()

    def _register_sim_failure(
        self,
        theta: np.ndarray,
        *,
        fidelity: str,
        failure_class: str = "sim_failure",
        skipped: bool = False,
    ) -> None:
        reg = get_failure_registry()
        reg.register_failure(
            theta,
            fidelity=fidelity,
            failure_class=failure_class,
            skipped=skipped,
        )

    def sample_prior(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(derive_seed(seed, "proliferating_vertex", "prior"))
        # Region B–style ranges (see ExampleCommandLineCSV.csv / Kursawe 2018).
        lam = rng.uniform(-1.45, 0.25, size=n)
        gamma = rng.uniform(0.01, 0.20, size=n)
        return np.stack([lam, gamma], axis=1).astype(np.float64)

    def get_prior_distribution(self) -> torch.distributions.Distribution:
        low = torch.as_tensor([-1.45, 0.01], dtype=torch.float32)
        high = torch.as_tensor([0.25, 0.20], dtype=torch.float32)
        return torch.distributions.Independent(torch.distributions.Uniform(low, high), 1)

    @property
    def lf_fidelity_id(self) -> str:
        return os.environ.get("PRISM_VERTEX_LF_FIDELITY", "lf_T140")

    @property
    def sim_jobs(self) -> int:
        return max(1, self._env_int("PRISM_VERTEX_SIM_JOBS", 1))

    def _simulator(self) -> ProliferatingVertexSimulator:
        return ProliferatingVertexSimulator()

    def _simulate_lf_one(self, theta: np.ndarray, seed: int) -> np.ndarray:
        t = np.asarray(theta, dtype=np.float64).reshape(-1)
        reg = get_failure_registry()
        if reg.is_blocked(t):
            if self.log_sim_failures:
                logger.info(
                    "LF (%s) skipped known-failed theta=(%.4f, %.4f); penalty.",
                    self.lf_fidelity_id,
                    float(t[0]),
                    float(t[1]),
                )
            return self._failure_result(
                t, fidelity="lf", failure_class="skipped_known_failed", skipped=True
            )

        last_err: str | None = None
        for attempt in range(self.max_sim_retries):
            try:
                attempt_seed = int(derive_seed(seed, "proliferating_vertex", "lf", "retry", attempt))
                return self._simulator().run_lf(t, attempt_seed, fidelity_id=self.lf_fidelity_id)
            except ChasteRunError as exc:
                last_err = str(exc)
                continue
        if self.log_sim_failures:
            logger.warning(
                "LF (%s) failed after %d retries for theta=(%.4f, %.4f); penalty. %s",
                self.lf_fidelity_id,
                self.max_sim_retries,
                float(t[0]),
                float(t[1]),
                (last_err or "")[:500],
            )
        return self._failure_result(t, fidelity="lf", failure_class="sim_failure")

    def _simulate_hf_k_mean_one(self, theta: np.ndarray, seed: int, k: int | None = None) -> np.ndarray:
        t = np.asarray(theta, dtype=np.float64).reshape(-1)
        k = int(k if k is not None else self.k_hf_replicates)
        reg = get_failure_registry()
        if reg.is_blocked(t):
            if self.log_sim_failures:
                logger.info(
                    "HF skipped known-failed theta=(%.4f, %.4f); penalty.",
                    float(t[0]),
                    float(t[1]),
                )
            return self._failure_result(
                t, fidelity="hf", failure_class="skipped_known_failed", skipped=True
            )

        last_err: str | None = None
        for attempt in range(self.max_sim_retries):
            try:
                batch_seed = int(derive_seed(seed, "proliferating_vertex", "hf", "retry", attempt))
                return self._simulator().run_hf_k_mean(
                    t,
                    batch_seed=batch_seed,
                    k=k,
                    strict_any_fail=self.hf_strict_any_fail,
                )
            except ChasteRunError as exc:
                last_err = str(exc)
                continue
        if self.log_sim_failures:
            logger.warning(
                "HF K-mean failed after %d retries for theta=(%.4f, %.4f); penalty. %s",
                self.max_sim_retries,
                float(t[0]),
                float(t[1]),
                (last_err or "")[:500],
            )
        return self._failure_result(t, fidelity="hf", failure_class="sim_failure")

    def _map_parallel(self, worker_fn, items: list[tuple[np.ndarray, int]], *, label: str) -> list[np.ndarray]:
        n_jobs = min(self.sim_jobs, max(1, len(items)))
        if n_jobs <= 1 or len(items) <= 1:
            return [worker_fn(args) for args in items]
        out: list[np.ndarray | None] = [None] * len(items)
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(worker_fn, args): idx for idx, args in enumerate(items)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    out[idx] = fut.result()
                except SimulationDropped:
                    raise
                except Exception as exc:
                    logger.warning("parallel %s worker failed at idx=%d: %s", label, idx, exc)
                    if baseline_drop_on_fail_enabled():
                        raise SimulationDropped(str(exc)) from exc
                    out[idx] = self.failed_summary()
        return [x if x is not None else self.failed_summary() for x in out]

    @property
    def k_hf_replicates(self) -> int:
        return max(1, self._env_int("PRISM_VERTEX_K_HF_REPLICATES", K_HF_REPLICATES))

    @property
    def use_k_mean_for_hf(self) -> bool:
        return self._env_bool("PRISM_VERTEX_HF_K_MEAN", True)

    def hf_replicate_seed(self, batch_seed: int, replicate_k: int) -> int:
        """K seeds for HF training at one θ (independent tissues, paper protocol)."""
        return int(derive_seed("proliferating_vertex", "hf_train", batch_seed, replicate_k))

    def simulate_hf_x_k_mean(self, theta: np.ndarray, seed: int, k: int | None = None) -> np.ndarray:
        """Paper HF: mean of k independent HF summaries at theta (default K=10)."""
        return self._simulate_hf_k_mean_one(np.asarray(theta, dtype=np.float64).reshape(-1), seed, k=k)

    def simulate_hf_x(self, theta: np.ndarray, seed: int) -> np.ndarray:
        arr = np.asarray(theta, dtype=np.float64)
        if arr.ndim == 1:
            if self.use_k_mean_for_hf:
                return self.simulate_hf_x_k_mean(arr, seed=derive_seed(seed, "proliferating_vertex", "hf"))
            return self._simulate_hf_k_mean_one(arr, seed=derive_seed(seed, "proliferating_vertex", "hf"), k=1)
        items = [
            (row, derive_seed(seed, "proliferating_vertex", "hf", i))
            for i, row in enumerate(arr)
        ]
        xs = self._map_parallel(_prolif_hf_batch_worker, items, label="hf")
        return np.stack(xs, axis=0)

    def simulate_lf_x(self, theta: np.ndarray, seed: int) -> np.ndarray:
        arr = np.asarray(theta, dtype=np.float64)
        if arr.ndim == 1:
            return self._simulate_lf_one(arr, seed=derive_seed(seed, "proliferating_vertex", "lf"))
        items = [(row, derive_seed(seed, "proliferating_vertex", "lf", i)) for i, row in enumerate(arr)]
        xs = self._map_parallel(_prolif_lf_batch_worker, items, label="lf")
        return np.stack(xs, axis=0)

    def simulate_hf_x_strict(self, theta: np.ndarray, seed: int) -> np.ndarray:
        """Single HF run without sentinel fallback (for smoke tests)."""
        arr = np.asarray(theta, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("simulate_hf_x_strict expects a single theta vector")
        return self._simulator().run_hf_single(arr, int(derive_seed(seed, "proliferating_vertex", "hf", "strict")))

    def get_observation(self, obs_id: int = 0) -> np.ndarray:
        obs_dir = self._observation_dir(obs_id)
        for name in (f"obs_{obs_id:02d}.npy", "s_obs.npy"):
            p = obs_dir / name
            if p.exists():
                return np.load(p)
        raise FileNotFoundError(
            f"Missing observation for obs_id={obs_id} in {obs_dir}. Run:\n"
            f"  python scripts/scripts/lf_selection/generate_observations.py --obs_id {obs_id}"
        )

    def get_true_parameters(self, obs_id: int) -> np.ndarray:
        import json

        from prism_vertex.simulator.real_observations import (
            REAL_DATA_OBS_IDS,
        )

        obs_dir = self._observation_dir(obs_id)
        source_p = obs_dir / "source.json"
        if obs_id in REAL_DATA_OBS_IDS or (
            source_p.exists()
            and bool(json.loads(source_p.read_text()).get("no_true_theta", False))
        ):
            raise LookupError(
                f"obs_id={obs_id} is real-data (no theta_true); skip NLTP/CRPS-to-truth"
            )

        for name in (f"theta_{obs_id:02d}.npy", "theta_true.npy"):
            p = obs_dir / name
            if p.exists():
                th = np.load(p)
                if np.any(~np.isfinite(th)):
                    raise LookupError(f"obs_id={obs_id} has non-finite theta_true")
                return th
        from prism_vertex.simulator.calibration_observations import (
            theta_for_obs,
        )

        try:
            return theta_for_obs(obs_id)
        except KeyError:
            return THETA_TRUE.copy()

    def _observation_dir(self, obs_id: int) -> Path:
        sub = self.observations_dir / f"obs_{obs_id:02d}"
        if sub.is_dir() and any(sub.glob("*.npy")):
            return sub
        if obs_id == 0:
            return self.observations_dir
        return sub

    def distance(self, x: np.ndarray, x_o: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_o = np.asarray(x_o, dtype=float)
        if x.ndim == 1:
            return np.array([float(np.linalg.norm(x - x_o))])
        return np.linalg.norm(x - x_o[None, :], axis=1)


def _factory(config: dict | None) -> ProliferatingVertexTask:
    _ = config
    return ProliferatingVertexTask()


register_task("proliferating_vertex", _factory)

register_task("prism_vertex", _factory)  # alias
