from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from sbi.inference import SNPE as NPE
from prism_vertex.inference.sbi_compat import posterior_nn

from prism_vertex.inference.inference_context import baseline_drop_on_fail_enabled, smoke_final_samples
from prism_vertex.experiments.task import Task
from prism_vertex.inference.batch_simulation import simulate_theta_batch
from prism_vertex.inference.sbi_priors import (
    _ensure_x_batch,
    _make_prior_from_task,
    _round_allocations_floor,
    _sanitize_theta_with_prior,
)
from prism_vertex.inference.nltp_exact_diag import add_nltp_exact_neural
from prism_vertex.inference.safe_sampling import safe_sample_numpy
from prism_vertex.utils.seeding import derive_seed
from prism_vertex.utils.timing import Timer


@dataclass(frozen=True)
class HfSnpeResult:
    posterior_samples: np.ndarray
    diagnostics: dict[str, Any]
    n_hf_calls: int
    train_seconds: float
    total_seconds: float


def _sample_snpe_mixture(
    *,
    posterior: Any,
    prior: Any,
    n: int,
    posterior_mix_alpha: float,
) -> np.ndarray:
    mix_alpha = float(np.clip(posterior_mix_alpha, 0.0, 1.0))
    n_post = max(1, int(n * (1.0 - mix_alpha)))
    n_prior = max(0, n - n_post)
    c_post = safe_sample_numpy(dist=posterior, n=n_post, fallback_dist=prior).astype(np.float32)
    if n_prior > 0:
        c_prior = safe_sample_numpy(dist=prior, n=n_prior).astype(np.float32)
        return np.vstack([c_post, c_prior])[:n].astype(np.float32)
    return np.atleast_2d(c_post)[:n].astype(np.float32)


def _simulate_hf_training_batch(
    task: Task,
    theta: np.ndarray,
    *,
    seed: int,
    seed_tag: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (theta_kept, x_kept, n_attempted). Uses row-wise drop on baseline methods."""
    if _use_drop_batch_sim(task):
        sim = simulate_theta_batch(task, theta, seed=seed, fidelity="hf", seed_tag=seed_tag)
        return sim.theta, sim.x, sim.n_attempted
    x = _ensure_x_batch(task.simulate_hf_x(theta, seed=seed))
    return np.asarray(theta, dtype=np.float32), x, int(np.asarray(theta).shape[0])


def _use_drop_batch_sim(task: Task) -> bool:
    if not baseline_drop_on_fail_enabled():
        return False
    eid = str(getattr(task, "experiment_id", ""))
    return eid in {"proliferating_vertex", "nonproliferating_vertex"}


def run_hf_snpe(
    *,
    task: Task,
    x_o: np.ndarray,
    budget_hf: int,
    seed: int,
    num_posterior_samples: int,
    config: Mapping[str, Any] | None = None,
    theta_true: np.ndarray | None = None,
) -> HfSnpeResult:
    cfg = dict(config or {})
    neural_net = str(cfg.get("neural_net", "nsf")).lower()
    hidden_features = int(cfg.get("hidden_features", 50))
    z_score_x = cfg.get("z_score_x", "independent")
    z_score_theta = cfg.get("z_score_theta", "independent")
    if int(budget_hf) <= 100 and str(z_score_x).lower() == "independent":
        z_score_x = "none"
    if int(budget_hf) <= 100 and str(z_score_theta).lower() == "independent":
        z_score_theta = "none"
    training_batch_size = max(1, int(cfg.get("hf_training_batch_size", cfg.get("training_batch_size", 200))))
    max_num_epochs = int(cfg.get("hf_max_num_epochs", cfg.get("max_num_epochs", 200)))
    num_rounds = max(1, int(cfg.get("num_rounds", 5)))
    posterior_mix_alpha = float(cfg.get("posterior_mix_alpha", 0.1))
    min_train_samples = max(2, int(cfg.get("min_train_samples", 2)))
    diagnostics_method_name = str(cfg.get("diagnostics_method_name", "hf_snpe"))

    timer_total = Timer()
    timer_total.start()
    train_seconds = 0.0
    n_hf_calls = 0
    round_logs: list[dict[str, Any]] = []

    prior = _make_prior_from_task(task, seed)
    density_estimator_fun = posterior_nn(
        model=neural_net,
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    inference = NPE(prior=prior, density_estimator=density_estimator_fun)
    x_o_t = torch.as_tensor(x_o, dtype=torch.float32)
    posterior: Any = None
    training_proposal: Any = prior

    round_sizes = _round_allocations_floor(int(budget_hf), num_rounds)

    for r, batch_n in enumerate(round_sizes, start=1):
        if batch_n <= 0:
            continue
        if r == 1 or posterior is None:
            theta_prop = task.sample_prior(
                batch_n,
                seed=derive_seed(seed, "hf_snpe", "theta_round", r),
            )
        else:
            theta_prop = _sample_snpe_mixture(
                posterior=posterior,
                prior=prior,
                n=batch_n,
                posterior_mix_alpha=posterior_mix_alpha,
            )
        theta_prop = _sanitize_theta_with_prior(
            theta=theta_prop,
            prior=prior,
            seed=derive_seed(seed, "hf_snpe", "theta_support", r),
        )

        theta_kept, x_kept, n_attempted = _simulate_hf_training_batch(
            task,
            theta_prop,
            seed=derive_seed(seed, "hf_snpe", "sim", r),
            seed_tag=f"round_{r}",
        )
        n_hf_calls += n_attempted

        if int(theta_kept.shape[0]) < min_train_samples:
            round_logs.append(
                {
                    "round": r,
                    "n_requested": batch_n,
                    "n_kept": int(theta_kept.shape[0]),
                    "trained": False,
                }
            )
            continue

        theta_t = torch.as_tensor(theta_kept, dtype=torch.float32)
        x_t = torch.as_tensor(x_kept, dtype=torch.float32)
        timer = Timer()
        timer.start()
        de = inference.append_simulations(theta_t, x_t, proposal=training_proposal).train(
            training_batch_size=max(1, min(training_batch_size, int(theta_t.shape[0]))),
            max_num_epochs=max_num_epochs,
            show_train_summary=False,
            validation_fraction=0.1,
            force_first_round_loss=(r > 1),
        )
        timer.stop()
        train_seconds += float(timer.elapsed_s)
        posterior = inference.build_posterior(de).set_default_x(x_o_t)
        training_proposal = posterior

        round_logs.append(
            {
                "round": r,
                "n_requested": batch_n,
                "n_kept": int(theta_kept.shape[0]),
                "trained": True,
            }
        )

    smoke = bool(cfg.get("inference_smoke", False))
    fast = smoke_final_samples(task, num_posterior_samples, seed, inference_smoke=smoke)
    if fast is not None:
        samples = fast
    elif posterior is None:
        samples = safe_sample_numpy(dist=prior, n=num_posterior_samples)
    else:
        samples = safe_sample_numpy(dist=posterior, n=num_posterior_samples, fallback_dist=prior)
    timer_total.stop()

    diagnostics: dict[str, Any] = {
        "method": diagnostics_method_name,
        "neural_net": neural_net,
        "hidden_features": hidden_features,
        "budget_hf": int(budget_hf),
        "num_rounds": num_rounds,
        "use_snpe_proposal": True,
        "truncation": False,
        "baseline_drop_on_fail": bool(
            getattr(task, "experiment_id", None) == "proliferating_vertex" and baseline_drop_on_fail_enabled()
        ),
        "round_logs": round_logs,
        "n_hf_attempted": n_hf_calls,
    }
    if posterior is not None:
        add_nltp_exact_neural(diagnostics, posterior, theta_true, x_o)

    return HfSnpeResult(
        posterior_samples=samples,
        diagnostics=diagnostics,
        n_hf_calls=n_hf_calls,
        train_seconds=train_seconds,
        total_seconds=float(timer_total.elapsed_s),
    )
