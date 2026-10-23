from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from sbi.inference import SMCABC

from prism_vertex.experiments.task import Task
from prism_vertex.inference.nltp_exact_diag import add_nltp_exact_smc_abc_na
from prism_vertex.utils.seeding import derive_seed
from prism_vertex.utils.timing import Timer


def _coerce_x_o_vector_for_smcabc(
    task: Task, x_o: np.ndarray, *, seed: int
) -> tuple[np.ndarray, str]:
    """Return a 1D observation vector matching one HF simulator output.

    sbi's SMC-ABC treats a leading dimension > 1 in ``x_o`` as multiple IID
    conditioning observations, which L2 distance rejects. On-disk ``obs_*.npy``
    files sometimes use a column vector ``(d, 1)`` or a mistaken ``(n, d)``
    stack; we normalize to a single summary vector of length ``d``."""
    x = np.asarray(x_o, dtype=np.float32)

    d_hint = getattr(task, "observation_dim", None)
    if callable(d_hint):
        d_sim = int(d_hint())
        note_base = "task_observation_dim"
    elif isinstance(d_hint, int):
        d_sim = int(d_hint)
        note_base = "task_observation_dim"
    else:
        theta_p = task.sample_prior(1, seed=derive_seed(seed, "smcabc_xo_probe", "theta"))[0]
        ref = np.asarray(
            task.simulate_hf_x(theta_p, seed=derive_seed(seed, "smcabc_xo_probe", "sim")),
            dtype=np.float32,
        )
        d_sim = int(ref.reshape(-1).shape[0])
        note_base = "as_loaded_1d"

    note = note_base
    if x.ndim == 1:
        v = x.reshape(-1)
        if v.size == d_sim:
            return v, note
        raise ValueError(
            f"SMC-ABC: observation length {v.size} != HF summary dimension {d_sim}. "
            "Regenerate or fix the observation file for this experiment."
        )

    if x.ndim == 2:
        if x.shape in ((1, d_sim), (d_sim, 1)):
            return x.reshape(-1), "reshaped_singleton_matrix"
        if x.shape[1] == d_sim:
            if x.shape[0] > 1:
                note = "used_first_row_of_shape_n_by_d"
            return x[0].reshape(-1), note
        if x.shape[0] == d_sim and x.shape[1] == 1:
            return x.reshape(-1), "reshaped_column_vector"
        v = x.reshape(-1)
        if v.size == d_sim:
            return v, "flattened_2d_to_match_d"
        raise ValueError(
            f"SMC-ABC: observation shape {x.shape} incompatible with HF summary dim {d_sim}."
        )

    v = x.reshape(-1)
    if v.size == d_sim:
        return v, "flattened_nd_to_match_d"
    raise ValueError(
        f"SMC-ABC: observation shape {x.shape} (raveled size {v.size}) != HF summary dim {d_sim}."
    )


@dataclass(frozen=True)
class SmcAbcResult:
    posterior_samples: np.ndarray
    diagnostics: dict[str, Any]
    n_hf_calls: int
    total_seconds: float


def run_hf_smc_abc(
    *,
    task: Task,
    x_o: np.ndarray,
    budget_hf: int,
    seed: int,
    num_posterior_samples: int,
    config: Mapping[str, Any] | None = None,
    theta_true: np.ndarray | None = None,
) -> SmcAbcResult:
    """
    HF-only SMC-ABC baseline via `sbi.inference.SMCABC`.

    ``theta_true`` is accepted for API symmetry with neural methods; SMC-ABC does not define
    a parametric posterior density, so diagnostics record ``nltp_exact_status=not_applicable_*``.
    """
    cfg = dict(config or {})
    population_size = int(cfg.get("population_size", min(500, budget_hf)))
    initial_round_factor = int(cfg.get("initial_round_factor", 5))
    epsilon_decay = float(cfg.get("epsilon_decay", 0.2))
    distance_based_decay = bool(cfg.get("distance_based_decay", True))
    batch_size = int(cfg.get("simulation_batch_size", 200))
    kernel = str(cfg.get("kernel", "gaussian"))
    kernel_variance_scale = float(cfg.get("kernel_variance_scale", 0.5))
    distance = str(cfg.get("distance", "l2"))

    timer = Timer()
    timer.start()

    torch.manual_seed(derive_seed(seed, "smcabc", "torch"))

    # simulator wrapper expected by sbi: theta -> x
    call_id = 0
    x_o_arr, x_o_coercion = _coerce_x_o_vector_for_smcabc(task, x_o, seed=seed)
    d_x = int(x_o_arr.shape[0])

    def simulator(theta_t: torch.Tensor) -> torch.Tensor:
        nonlocal call_id
        this_call = call_id
        call_id += 1

        # sbi SMCABC passes batches shaped (n, d_theta). nonproliferating.simulate_hf_x only
        # accepts a single θ vector (it reshape(-1)s); a batch used to raise and we
        # returned all-1e9 sentinels — so nonproliferating SMC finished in <2s with no Chaste.
        # Simulate row-wise (parallel for proliferating/14) and keep per-row sentinels.
        theta = np.asarray(theta_t.detach().cpu().numpy(), dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        n = int(theta.shape[0])
        out = np.full((n, d_x), 1.0e9, dtype=np.float32)
        batch_seed = derive_seed(seed, "hf_sim_batch", this_call)

        eid = str(getattr(task, "experiment_id", ""))
        if eid in {"proliferating_vertex", "nonproliferating_vertex"}:
            from prism_vertex.inference.batch_simulation import _simulate_rows

            rows = _simulate_rows(
                task,
                theta,
                seed=int(batch_seed),
                seed_tag="smc",
                fid_str="hf",
            )
            for i, row_res in enumerate(rows):
                if row_res.dropped or row_res.x is None:
                    continue
                x_i = np.asarray(row_res.x, dtype=np.float32).reshape(-1)
                if x_i.size == d_x:
                    out[i] = x_i
            return torch.as_tensor(out, dtype=torch.float32)

        for i, row in enumerate(theta):
            try:
                x = np.asarray(
                    task.simulate_hf_x(row, seed=derive_seed(batch_seed, i)),
                    dtype=np.float32,
                ).reshape(-1)
                if x.size == d_x:
                    out[i] = x
                elif x.ndim == 1 and x.size > d_x:
                    out[i] = x[:d_x]
            except Exception:
                # Invalid θ / Chaste abort → reject by huge distance (exp08+).
                pass
        return torch.as_tensor(out, dtype=torch.float32)

    # Prior passed to SBI's SMCABC controls where particles are proposed from.
    # For exp08, use a strict bounded prior to avoid invalid regions that crash Chaste.
    if getattr(task, "experiment_id", "") == "exp08":
        low = torch.as_tensor([1e-2, 1e-4, 12.0], dtype=torch.float32)
        high = torch.as_tensor([1e2, 1e-1, 48.0], dtype=torch.float32)
        prior = torch.distributions.Independent(torch.distributions.Uniform(low, high), 1)
    elif getattr(task, "experiment_id", "") == "proliferating_vertex":
        low = torch.as_tensor([-1.45, 0.01], dtype=torch.float32)
        high = torch.as_tensor([0.25, 0.20], dtype=torch.float32)
        prior = torch.distributions.Independent(torch.distributions.Uniform(low, high), 1)
    elif getattr(task, "experiment_id", "") == "nonproliferating_vertex":
        from prism_vertex.simulator.nonproliferating.constants import (
            PRIOR_HIGH,
            PRIOR_LOW,
        )

        low = torch.as_tensor(PRIOR_LOW, dtype=torch.float32)
        high = torch.as_tensor(PRIOR_HIGH, dtype=torch.float32)
        prior = torch.distributions.Independent(torch.distributions.Uniform(low, high), 1)
    elif getattr(task, "wang_stage_a", False) and hasattr(task, "get_prior_distribution"):
        # Exp12 Stage A: exact Uniform(κ) — Normal fit would leak outside [0.01, 0.08].
        prior = task.get_prior_distribution()
    else:
        # rough prior approximation for sbi's SMCABC kernel machinery (baseline harness only)
        theta_prior = task.sample_prior(
            n=min(10_000, max(1000, population_size * 5)),
            seed=derive_seed(seed, "prior_fit"),
        )
        theta_t = torch.as_tensor(theta_prior, dtype=torch.float32)
        mean = theta_t.mean(dim=0)
        std = theta_t.std(dim=0).clamp_min(1e-6)
        prior = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)

    inference_method = SMCABC(
        simulator=simulator,
        prior=prior,
        simulation_batch_size=batch_size,
        distance=distance,
        kernel=kernel,
        show_progress_bars=False,
        algorithm_variant=str(cfg.get("algorithm_variant", "C")),
    )

    # sbi SMCABC asserts: num_initial_pop > num_particles (not >=).
    # If budget_hf is smaller than the configured population_size (e.g. budget_hf=50 with population_size=300),
    # we must down-clamp population_size to keep the constraint feasible.
    if budget_hf <= 1:
        population_size = 1
    else:
        population_size = min(population_size, budget_hf - 1)

    num_initial_pop_candidate = max(population_size + 1, initial_round_factor * population_size)
    num_initial_pop = min(int(budget_hf), int(num_initial_pop_candidate))
    if num_initial_pop <= population_size:
        # Absolute fallback for pathological budgets.
        num_initial_pop = min(int(budget_hf), population_size + 1)

    output, summary = inference_method(
        x_o=torch.as_tensor(x_o_arr, dtype=torch.float32).unsqueeze(0),
        num_particles=population_size,
        num_initial_pop=num_initial_pop,
        num_simulations=budget_hf,
        epsilon_decay=epsilon_decay,
        distance_based_decay=distance_based_decay,
        kernel_variance_scale=kernel_variance_scale,
        return_summary=True,
    )

    # output is samples tensor
    if isinstance(output, torch.Tensor):
        raw = output.detach().cpu().numpy()
        if raw.shape[0] >= num_posterior_samples:
            idx = np.random.default_rng(derive_seed(seed, "posterior_subsample")).choice(raw.shape[0], size=num_posterior_samples, replace=False)
            samples = raw[idx]
        else:
            samples = raw
    else:
        # Some sbi versions return KDE posterior; fallback
        samples = output.sample(num_posterior_samples).detach().cpu().numpy()

    timer.stop()

    diagnostics: dict[str, Any] = {
        "method": "hf_smc_abc",
        "budget_hf": budget_hf,
        "population_size": population_size,
        "num_initial_pop": num_initial_pop,
        "initial_round_factor": initial_round_factor,
        "epsilon_decay": epsilon_decay,
        "distance_based_decay": distance_based_decay,
        "kernel": kernel,
        "kernel_variance_scale": kernel_variance_scale,
        "summary_keys": sorted(list(summary.keys())) if isinstance(summary, dict) else None,
        "x_o_coercion": x_o_coercion,
    }
    _ = theta_true  # reserved for future ABC-density proxies; not used.
    add_nltp_exact_smc_abc_na(diagnostics)

    return SmcAbcResult(
        posterior_samples=samples,
        diagnostics=diagnostics,
        n_hf_calls=int(budget_hf),
        total_seconds=float(timer.elapsed_s),
    )

