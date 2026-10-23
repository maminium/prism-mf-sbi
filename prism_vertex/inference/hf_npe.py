from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from sbi import inference as sbi_inference
from prism_vertex.inference.sbi_compat import posterior_nn

from prism_vertex.simulator.inference_simulation import (
    baseline_drop_on_fail_enabled,
)
from prism_vertex.experiments.task import Task
from prism_vertex.inference.nltp_exact_diag import add_nltp_exact_neural
from prism_vertex.inference.safe_sampling import safe_sample_numpy
from prism_vertex.utils.seeding import derive_seed
from prism_vertex.utils.timing import Timer


@dataclass(frozen=True)
class NpeResult:
    posterior_samples: np.ndarray
    diagnostics: dict[str, Any]
    n_hf_calls: int
    train_seconds: float
    total_seconds: float


def run_hf_npe(
    *,
    task: Task,
    x_o: np.ndarray,
    budget_hf: int,
    seed: int,
    num_posterior_samples: int,
    config: Mapping[str, Any] | None = None,
    theta_true: np.ndarray | None = None,
) -> NpeResult:
    """
    HF-only NPE baseline via `sbi`.
    This is single-round (non-sequential) training on HF simulations.
    """
    cfg = dict(config or {})
    neural_net = str(cfg.get("neural_net", "nsf")).lower()
    hidden_features = int(cfg.get("hidden_features", 50))
    simulation_batch_size = int(cfg.get("simulation_batch_size", 1000))
    training_batch_size = int(cfg.get("training_batch_size", 2000))
    max_num_epochs = int(cfg.get("max_num_epochs", 200))
    z_score_x = cfg.get("z_score_x", "independent")
    z_score_theta = cfg.get("z_score_theta", "independent")

    timer_total = Timer()
    timer_total.start()

    torch.manual_seed(derive_seed(seed, "npe", "torch"))

    # Build dataset (vectorized where possible)
    theta_seed = derive_seed(seed, "npe", "prior")
    sim_seed = derive_seed(seed, "npe", "hf_sim_batch")
    theta = task.sample_prior(n=budget_hf, seed=theta_seed).astype(np.float64)
    if getattr(task, "experiment_id", None) in {"proliferating_vertex", "nonproliferating_vertex"} and baseline_drop_on_fail_enabled():
        from prism_vertex.inference.batch_simulation import simulate_theta_batch

        sim = simulate_theta_batch(
            task,
            theta,
            seed=int(sim_seed),
            fidelity="hf",
            seed_tag="hf_npe",
        )
        theta = sim.theta.astype(np.float64)
        x = sim.x.astype(np.float32)
    else:
        if hasattr(task, "simulate_hf_paired"):
            theta, x = task.simulate_hf_paired(theta, seed=int(sim_seed))  # type: ignore[attr-defined]
            x = np.asarray(x, dtype=np.float32)
        else:
            x = task.simulate_hf_x(theta, seed=sim_seed).astype(np.float32)

    theta_t = torch.as_tensor(theta, dtype=torch.float32)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    n_keep = min(int(theta_t.shape[0]), int(x_t.shape[0]))
    if n_keep < 2:
        raise RuntimeError(
            f"HF-NPE received n={n_keep} simulations (budget_hf={budget_hf}); "
            "cannot fit a Normal prior or train. Usually every HF launch failed (MPI)."
        )
    theta_t = theta_t[:n_keep]
    x_t = x_t[:n_keep]
    finite = torch.isfinite(theta_t).all(dim=-1) & torch.isfinite(x_t).all(dim=-1)
    theta_t = theta_t[finite]
    x_t = x_t[finite]
    if int(theta_t.shape[0]) < 2:
        raise RuntimeError(
            f"HF-NPE had no finite (theta, x) rows after filtering (budget_hf={budget_hf})."
        )

    # sbi prior as independent Uniform over bounding box is not correct.
    # We treat prior implicitly by training on prior samples and using SNPE_C with prior samples only.
    # For sampling, we use a simple Gaussian prior fit on theta samples (baseline plumbing only).
    theta_mean = theta_t.mean(dim=0)
    theta_std = theta_t.std(dim=0)
    theta_std = torch.where(torch.isfinite(theta_std), theta_std, torch.ones_like(theta_std))
    theta_std = theta_std.clamp_min(1e-6)
    prior = torch.distributions.Independent(torch.distributions.Normal(theta_mean, theta_std), 1)

    density_estimator_fun = posterior_nn(
        model=neural_net,
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )

    timer_train = Timer()
    timer_train.start()
    inference_method = sbi_inference.SNPE_C(prior, density_estimator=density_estimator_fun)
    density_estimator = inference_method.append_simulations(theta_t, x_t, proposal=prior).train(
        training_batch_size=training_batch_size,
        max_num_epochs=max_num_epochs,
        show_train_summary=False,
    )
    posterior = inference_method.build_posterior(density_estimator).set_default_x(
        torch.as_tensor(x_o, dtype=torch.float32)
    )
    timer_train.stop()

    smoke = bool(cfg.get("inference_smoke", False))
    from prism_vertex.inference.inference_context import smoke_final_samples

    fast = smoke_final_samples(task, num_posterior_samples, seed, inference_smoke=smoke)
    if fast is not None:
        samples = fast
    else:
        samples = safe_sample_numpy(dist=posterior, n=num_posterior_samples, fallback_dist=prior)
    timer_total.stop()

    diagnostics: dict[str, Any] = {
        "method": "hf_npe",
        "neural_net": neural_net,
        "hidden_features": hidden_features,
        "budget_hf": budget_hf,
        "training_batch_size": training_batch_size,
        "simulation_batch_size": simulation_batch_size,
        "max_num_epochs": max_num_epochs,
    }
    add_nltp_exact_neural(diagnostics, posterior, theta_true, x_o)

    return NpeResult(
        posterior_samples=samples,
        diagnostics=diagnostics,
        n_hf_calls=int(budget_hf),
        train_seconds=float(timer_train.elapsed_s),
        total_seconds=float(timer_total.elapsed_s),
    )

