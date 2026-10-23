from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from sbi import inference as sbi_inference
from sbi.inference import SNPE as NPE
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from prism_vertex.inference.sbi_compat import posterior_nn

from prism_vertex.experiments.task import Task
from prism_vertex.inference.mf_sbi_acquisition import _top_b_by_uncertainty
from prism_vertex.inference.sbi_priors import (
    _make_prior_from_task,
    _round_allocations_floor,
    _sample_from_proposal_truncated,
    _sanitize_theta_with_prior,
)
from prism_vertex.inference.inference_context import smoke_final_samples, smoke_prior_candidates
from prism_vertex.inference.nltp_exact_diag import add_nltp_exact_neural
from prism_vertex.inference.safe_sampling import safe_sample_numpy
from prism_vertex.inference.sbi_posterior_utils import build_posterior_with_sampling, posterior_log_prob_unnorm
from prism_vertex.utils.seeding import derive_seed
from prism_vertex.utils.timing import Timer


def _use_drop_batch_sim(task: Task) -> bool:
    from prism_vertex.inference.inference_context import baseline_drop_on_fail_enabled

    if not baseline_drop_on_fail_enabled():
        return False
    eid = str(getattr(task, "experiment_id", ""))
    return eid in {"proliferating_vertex", "nonproliferating_vertex"}


def _baseline_drop(task: Task) -> bool:
    return _use_drop_batch_sim(task)


def _simulate_lf_paired(
    task: Task,
    theta: np.ndarray,
    *,
    seed: int,
    seed_tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    if _baseline_drop(task):
        from prism_vertex.inference.batch_simulation import simulate_theta_batch

        sim = simulate_theta_batch(task, theta, seed=seed, fidelity="lf", seed_tag=seed_tag)
        return sim.theta, _ensure_x_batch(sim.x)
    if hasattr(task, "simulate_lf_paired"):
        th, x = task.simulate_lf_paired(theta, seed=seed)  # type: ignore[attr-defined]
        return np.asarray(th, dtype=np.float32), _ensure_x_batch(x)
    return np.asarray(theta, dtype=np.float32), _ensure_x_batch(task.simulate_lf_x(theta, seed=seed))


def _simulate_hf_paired(
    task: Task,
    theta: np.ndarray,
    *,
    seed: int,
    seed_tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    if _baseline_drop(task):
        from prism_vertex.inference.batch_simulation import simulate_theta_batch

        sim = simulate_theta_batch(task, theta, seed=seed, fidelity="hf", seed_tag=seed_tag)
        return sim.theta, _ensure_x_batch(sim.x)
    if hasattr(task, "simulate_hf_paired"):
        th, x = task.simulate_hf_paired(theta, seed=seed)  # type: ignore[attr-defined]
        return np.asarray(th, dtype=np.float32), _ensure_x_batch(x)
    return np.asarray(theta, dtype=np.float32), _ensure_x_batch(task.simulate_hf_x(theta, seed=seed))


def _ensemble_variance_scores(*, ensemble_posteriors: list[Any], candidates_t: torch.Tensor) -> np.ndarray:
    if len(ensemble_posteriors) == 0:
        return np.zeros(int(candidates_t.shape[0]), dtype=np.float32)

    logps: list[np.ndarray] = []
    with torch.no_grad():
        for post in ensemble_posteriors:
            lp = posterior_log_prob_unnorm(post, candidates_t).detach().cpu().numpy().reshape(-1)
            logps.append(lp.astype(np.float32))
    arr = np.stack(logps, axis=0)  # [E, N]
    return np.var(arr, axis=0)


def _sanitize_theta_batch(
    *,
    theta: np.ndarray,
    prior: Any,
    theta_min: np.ndarray | None,
    theta_max: np.ndarray | None,
) -> np.ndarray:
    """
    Ensure theta rows are finite and (when bounds available) inside prior support.
    Replace invalid rows by prior samples to avoid SNPE prior-eval NaN/Inf assertions.
    """
    if theta.shape[0] == 0:
        return theta.astype(np.float32, copy=False)

    out = np.asarray(theta, dtype=np.float32).copy()
    valid = np.isfinite(out).all(axis=1)

    lo = hi = None
    if theta_min is not None and theta_max is not None:
        lo = np.asarray(theta_min, dtype=np.float32)
        hi = np.asarray(theta_max, dtype=np.float32)
        valid &= ((out >= lo).all(axis=1) & (out <= hi).all(axis=1))

    n_bad = int((~valid).sum())
    if n_bad > 0:
        repl = prior.sample((n_bad,)).detach().cpu().numpy().astype(np.float32)
        if lo is not None and hi is not None:
            # Pull slightly into interior, away from hard boundaries.
            span = np.maximum(np.abs(hi - lo), 1e-6)
            margin = 1e-2 * span + 1e-5
            lo_i = np.minimum(lo + margin, hi)
            hi_i = np.maximum(hi - margin, lo_i)
            repl = np.clip(repl, lo_i, hi_i)
        out[~valid] = repl

    if lo is not None and hi is not None:
        span = np.maximum(np.abs(hi - lo), 1e-6)
        margin = 1e-2 * span + 1e-5
        lo_i = np.minimum(lo + margin, hi)
        hi_i = np.maximum(hi - margin, lo_i)
        out = np.clip(out, lo_i, hi_i)

    return out.astype(np.float32, copy=False)


def _ensure_x_batch(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _usable_train_tensors(
    theta_t: torch.Tensor,
    x_t: torch.Tensor,
    min_n: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Drop empty/non-finite rows so sbi append_simulations never sees shape [0, -1]."""
    if theta_t is None or x_t is None:
        return None
    if theta_t.numel() == 0 or x_t.numel() == 0:
        return None
    if theta_t.ndim < 2 or x_t.ndim < 2:
        return None
    n = min(int(theta_t.shape[0]), int(x_t.shape[0]))
    if n < int(min_n):
        return None
    theta_t = theta_t[:n]
    x_t = x_t[:n]
    finite = torch.isfinite(theta_t).all(dim=-1) & torch.isfinite(x_t).all(dim=-1)
    theta_t = theta_t[finite]
    x_t = x_t[finite]
    if int(theta_t.shape[0]) < int(min_n):
        return None
    return theta_t, x_t


def _sample_snpe_mixture_candidates(
    *,
    ensemble_posteriors: list[Any],
    prior: Any,
    n: int,
    posterior_mix_alpha: float,
    theta_min: np.ndarray | None,
    theta_max: np.ndarray | None,
) -> np.ndarray:
    """
    SNPE-style candidate pool: posterior (+ optional ensemble) mixed with prior.
    No truncated-proposal truncation / RestrictedPrior rejection loop.
    """
    if n <= 0:
        return np.empty((0, int(getattr(prior, "low", torch.zeros(1)).shape[-1])), dtype=np.float32)

    mix_alpha = float(np.clip(posterior_mix_alpha, 0.0, 1.0))
    n_post = max(1, int(n * (1.0 - mix_alpha)))
    n_prior = max(0, n - n_post)

    if ensemble_posteriors:
        sampling_dist: Any = NeuralPosteriorEnsemble(ensemble_posteriors)
    else:
        sampling_dist = prior

    c_post = safe_sample_numpy(dist=sampling_dist, n=n_post, fallback_dist=prior).astype(np.float32)
    if theta_min is not None and theta_max is not None:
        c_post = np.clip(c_post, theta_min, theta_max)

    if n_prior > 0:
        c_prior = safe_sample_numpy(dist=prior, n=n_prior).astype(np.float32)
        if theta_min is not None and theta_max is not None:
            c_prior = np.clip(c_prior, theta_min, theta_max)
        candidates = np.vstack([c_post, c_prior])[:n]
    else:
        candidates = np.atleast_2d(c_post)[:n]

    return candidates.astype(np.float32, copy=False)


@dataclass(frozen=True)
class MfSbiResult:
    posterior_samples: np.ndarray
    diagnostics: dict[str, Any]
    n_hf_calls: int
    n_lf_calls: int
    train_seconds: float
    total_seconds: float


def run_mf_sbi_core(
    *,
    task: Task,
    x_o: np.ndarray,
    budget_lf: int,
    budget_hf: int,
    seed: int,
    num_posterior_samples: int,
    config: Mapping[str, Any] | None = None,
    theta_true: np.ndarray | None = None,
) -> MfSbiResult:
    """
    MF-SBI: LF active acquisition + truncated-proposal-style truncated proposals,
    with paper-1 `k%` cost-aware adaptive HF/LF allocation inside each mixed round.
    """
    cfg = dict(config or {})
    smoke = bool(cfg.get("inference_smoke", False))
    neural_net = str(cfg.get("neural_net", "nsf")).lower()
    hidden_features = int(cfg.get("hidden_features", 50))
    z_score_x = cfg.get("z_score_x", "independent")
    z_score_theta = cfg.get("z_score_theta", "independent")
    # Tiny-budget robustness: avoid zero-std standardization failures in sbi/nflows on some environments.
    if (int(budget_lf) <= 100 or int(budget_hf) <= 100) and str(z_score_x).lower() == "independent":
        z_score_x = "none"
    if (int(budget_lf) <= 100 or int(budget_hf) <= 100) and str(z_score_theta).lower() == "independent":
        z_score_theta = "none"

    lf_training_batch_size = max(1, int(cfg.get("lf_training_batch_size", 200)))
    hf_training_batch_size = max(1, int(cfg.get("hf_training_batch_size", 200)))
    min_train_samples = max(2, int(cfg.get("min_train_samples", 2)))
    lf_max_num_epochs = int(cfg.get("lf_max_num_epochs", 200))
    hf_max_num_epochs = int(cfg.get("hf_max_num_epochs", 200))

    num_rounds = int(cfg.get("num_rounds", 5))
    proposal_quantile = float(cfg.get("proposal_quantile", 1e-6))
    disable_truncation = bool(cfg.get("disable_truncation", False))
    posterior_sample_with = str(cfg.get("posterior_sample_with", "mcmc"))
    use_snpe_proposal = bool(cfg.get("use_snpe_proposal", False))
    posterior_mix_alpha = float(cfg.get("posterior_mix_alpha", 0.1))

    ensemble_size = int(cfg.get("ensemble_size", 5))
    active_fraction_lf = float(cfg.get("active_fraction_lf", cfg.get("active_fraction", 0.2)))
    candidate_pool_size = int(cfg.get("candidate_pool_size", 1000))

    lf_warmup_fraction = float(cfg.get("lf_warmup_fraction", 0.5))
    lf_active_fraction = float(cfg.get("lf_active_fraction", 0.2))
    lf_active_rounds = int(cfg.get("lf_active_rounds", 2))

    hf_top_k_percent = float(cfg.get("hf_top_k_percent", 100.0))
    use_phase2_lf_active = bool(cfg.get("use_phase2_lf_active", True))
    force_min_hf_per_mixed_round = bool(cfg.get("force_min_hf_per_mixed_round", True))
    diagnostics_method_name = str(cfg.get("diagnostics_method_name", "mf_sbi"))
    phase2_acquisition_fidelity = str(cfg.get("phase2_acquisition_fidelity", "lf")).lower().strip()
    if phase2_acquisition_fidelity not in ("lf", "hf"):
        raise ValueError(
            f"phase2_acquisition_fidelity must be 'lf' or 'hf', got {phase2_acquisition_fidelity!r}"
        )
    phase2_hf_match_mode = str(cfg.get("phase2_hf_match_mode", "cost")).lower().strip()
    if phase2_hf_match_mode not in ("cost", "calls"):
        raise ValueError(
            f"phase2_hf_match_mode must be 'cost' or 'calls', got {phase2_hf_match_mode!r}"
        )
    c_hf_ledger = float(cfg.get("c_hf", 1.0))
    c_lf_ledger = float(cfg.get("c_lf", 0.1))
    if c_hf_ledger <= 0:
        raise ValueError("c_hf must be > 0 for HF acquisition ledger")

    timer_total = Timer()
    timer_total.start()
    train_seconds = 0.0

    prior = _make_prior_from_task(task, seed)
    theta_min = None
    theta_max = None
    if hasattr(prior, "low") and hasattr(prior, "high"):
        try:
            theta_min = prior.low.detach().cpu().numpy().astype(np.float32)  # type: ignore[attr-defined]
            theta_max = prior.high.detach().cpu().numpy().astype(np.float32)  # type: ignore[attr-defined]
        except Exception:
            theta_min = None
            theta_max = None
    theta_clamp_lo_t: torch.Tensor | None = None
    theta_clamp_hi_t: torch.Tensor | None = None
    if theta_min is not None and theta_max is not None:
        range_mag = np.maximum(np.abs(theta_max - theta_min), 1e-12).astype(np.float32)
        # Bigger interior margin to avoid boundary effects producing -inf prior eval.
        # Cap the margin so lo/hi never collapse (important for strict-interior resampling).
        margin = np.minimum(1e-2 * range_mag + 1e-7, 0.49 * range_mag)
        lo = (theta_min + margin).astype(np.float32)
        hi = (theta_max - margin).astype(np.float32)
        lo = np.minimum(lo, hi).astype(np.float32)
        theta_clamp_lo_t = torch.as_tensor(lo, dtype=torch.float32)
        theta_clamp_hi_t = torch.as_tensor(hi, dtype=torch.float32)
    density_estimator_fun = posterior_nn(
        model=neural_net,
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )

    # ------------------------
    # Phase 1: LF warmup
    # ------------------------
    n_warmup = int(round(lf_warmup_fraction * int(budget_lf)))
    n_active_lf_requested = int(round(lf_active_fraction * int(budget_lf)))
    n_warmup = max(0, min(n_warmup, int(budget_lf)))
    n_active_lf_requested = max(0, min(n_active_lf_requested, int(budget_lf) - n_warmup))

    n_active_lf = 0
    n_active_hf_phase2 = 0
    if use_phase2_lf_active and n_active_lf_requested > 0:
        if phase2_acquisition_fidelity == "lf":
            n_active_lf = int(n_active_lf_requested)
        elif phase2_hf_match_mode == "calls":
            # Same acquisition *call count* as LF-acq, capped by HF budget (matched total N_H).
            n_active_hf_phase2 = min(int(budget_hf), int(n_active_lf_requested))
        else:
            # Same acquisition *cost* as LF-acq (n_active_lf * c_lf), spent as HF calls.
            cost_acq = float(n_active_lf_requested) * c_lf_ledger
            n_active_hf_phase2 = int(round(cost_acq / c_hf_ledger))
            n_active_hf_phase2 = max(0, min(int(budget_hf), n_active_hf_phase2))

    n_mix_lf = int(budget_lf) - n_warmup - n_active_lf

    theta_lf = task.sample_prior(n_warmup, seed=derive_seed(seed, "mf_sbi", "theta_lf_warmup"))
    theta_lf, x_lf = _simulate_lf_paired(
        task,
        theta_lf,
        seed=derive_seed(seed, "mf_sbi", "x_lf_warmup"),
        seed_tag="warmup",
    )
    theta_lf_t = torch.as_tensor(theta_lf, dtype=torch.float32)
    x_lf_t = torch.as_tensor(x_lf, dtype=torch.float32)
    warmup_pair = _usable_train_tensors(theta_lf_t, x_lf_t, min_train_samples)
    if warmup_pair is None:
        raise RuntimeError(
            f"MF-SBI LF warmup produced no usable simulations "
            f"(n_theta={int(theta_lf_t.shape[0])}, n_x={int(x_lf_t.shape[0])}). "
            "This is usually an MPI/Chaste launch failure, not a bad observation."
        )
    theta_lf_t, x_lf_t = warmup_pair

    inf_lf = NPE(prior=prior, density_estimator=density_estimator_fun)
    timer = Timer()
    timer.start()
    de_lf = inf_lf.append_simulations(theta_lf_t, x_lf_t, proposal=prior).train(
        training_batch_size=max(1, min(lf_training_batch_size, int(theta_lf_t.shape[0]))),
        max_num_epochs=lf_max_num_epochs,
        show_train_summary=False,
        validation_fraction=0.5,
    )
    timer.stop()
    train_seconds += float(timer.elapsed_s)

    # Ensemble inference objects, transferred from the warmup estimator.
    ensemble_inferences: list[Any] = []
    for _e in range(ensemble_size):

        def density_estimator_transfer(
            batch_theta: torch.Tensor, batch_x: torch.Tensor
        ) -> torch.nn.Module:
            model = density_estimator_fun(batch_theta, batch_x)
            model.load_state_dict(de_lf.state_dict(), strict=False)
            return model

        ensemble_inferences.append(NPE(prior=prior, density_estimator=density_estimator_transfer))

    x_o_t = torch.as_tensor(x_o, dtype=torch.float32)
    ensemble_posteriors: list[Any] = [
        build_posterior_with_sampling(
            inf_e,
            de_lf,
            sample_with=posterior_sample_with,
        ).set_default_x(x_o_t)
        for inf_e in ensemble_inferences
    ]

    # ------------------------
    # Stage 2: active acquisition (LF by default)
    # ------------------------
    n_lf_calls = 0
    n_lf_calls += int(n_warmup)
    total_hf = 0

    phase2_n = int(n_active_lf if phase2_acquisition_fidelity == "lf" else n_active_hf_phase2)
    lf_active_batches: list[int] = []
    if phase2_n > 0:
        lf_active_rounds_eff = max(1, lf_active_rounds)
        lf_active_batches = _round_allocations_floor(int(phase2_n), lf_active_rounds_eff)

    lf_active_logs: list[dict[str, Any]] = []
    for b_idx, b_size in enumerate(lf_active_batches):
        if b_size <= 0:
            continue
        # Candidate pool for ranking (uncertainty via ensemble variance).
        theta_pool = task.sample_prior(
            candidate_pool_size,
            seed=derive_seed(seed, "mf_sbi", "theta_pool", b_idx),
        )
        candidates_t = torch.as_tensor(theta_pool, dtype=torch.float32)

        # paper-1-style floor active count
        b_active = max(1, int(active_fraction_lf * b_size)) if b_size > 0 else 0
        b_active = min(b_active, b_size)
        b_rest = b_size - b_active

        theta_active = _top_b_by_uncertainty(
            candidates_t=candidates_t,
            ensemble_posteriors=ensemble_posteriors,
            b=b_active,
        )

        if b_rest > 0:
            theta_rest = task.sample_prior(
                b_rest,
                seed=derive_seed(seed, "mf_sbi", "theta_rest", b_idx),
            )
            theta_batch = np.vstack([theta_active, theta_rest])
        else:
            theta_batch = theta_active

        if phase2_acquisition_fidelity == "hf":
            theta_batch, x_batch = _simulate_hf_paired(
                task,
                theta_batch,
                seed=derive_seed(seed, "mf_sbi", "x_hf_active_batch", b_idx),
                seed_tag=f"hf_active_{b_idx}",
            )
            train_bs = hf_training_batch_size
            train_epochs = hf_max_num_epochs
        else:
            theta_batch, x_batch = _simulate_lf_paired(
                task,
                theta_batch,
                seed=derive_seed(seed, "mf_sbi", "x_lf_active_batch", b_idx),
                seed_tag=f"lf_active_{b_idx}",
            )
            train_bs = lf_training_batch_size
            train_epochs = lf_max_num_epochs
        theta_batch_t = torch.as_tensor(theta_batch, dtype=torch.float32)
        x_batch_t = torch.as_tensor(x_batch, dtype=torch.float32)
        usable_batch = _usable_train_tensors(theta_batch_t, x_batch_t, min_train_samples)
        if usable_batch is None:
            if phase2_acquisition_fidelity == "hf":
                total_hf += int(b_size)
            else:
                n_lf_calls += int(b_size)
            lf_active_logs.append(
                {
                    "batch": b_idx,
                    "n_in_batch": int(b_size),
                    "n_active": int(b_active),
                    "fidelity": phase2_acquisition_fidelity,
                    "trained": False,
                }
            )
            continue

        theta_batch_t, x_batch_t = usable_batch

        # Train each ensemble member on the new acquisition batch.
        new_posteriors: list[Any] = []
        for e, inf_e in enumerate(ensemble_inferences):
            timer = Timer()
            timer.start()
            de_e = inf_e.append_simulations(theta_batch_t, x_batch_t, proposal=prior).train(
                training_batch_size=max(1, min(train_bs, int(theta_batch_t.shape[0]))),
                max_num_epochs=train_epochs,
                show_train_summary=False,
                validation_fraction=0.5,
                # We append simulations sampled from `prior` (proposal=prior => round 0),
                # but the network has already been trained in earlier batches.
                # Setting this flag avoids SBI's assertion about missing/incorrect proposal handling.
                force_first_round_loss=True,
            )
            timer.stop()
            train_seconds += float(timer.elapsed_s)
            new_posteriors.append(
                build_posterior_with_sampling(
                    inf_e,
                    de_e,
                    sample_with=posterior_sample_with,
                ).set_default_x(x_o_t)
            )

        ensemble_posteriors = new_posteriors
        if phase2_acquisition_fidelity == "hf":
            total_hf += int(b_size)
        else:
            n_lf_calls += int(b_size)
        lf_active_logs.append(
            {
                "batch": b_idx,
                "n_in_batch": int(b_size),
                "n_active": int(b_active),
                "fidelity": phase2_acquisition_fidelity,
                "trained": True,
            }
        )

    # ------------------------
    # Phase 3: Mixed LF/HF sequential rounds
    # ------------------------
    remaining_hf = int(budget_hf) - int(n_active_hf_phase2)
    remaining_lf = int(n_mix_lf)
    total_lf_mix = 0
    # total_hf already includes phase-2 HF acquisition (0 for default LF-acq)

    # total mixed points to split across num_rounds
    total_mix_points = remaining_hf + remaining_lf
    mix_round_sizes = _round_allocations_floor(int(total_mix_points), max(1, num_rounds))

    proposal_for_sampling: Any = prior
    hf_lf_logs: list[dict[str, Any]] = []

    def _draw_mixed_round_candidates(n_draw: int) -> np.ndarray:
        fast = smoke_prior_candidates(
            task,
            int(n_draw),
            derive_seed(seed, "mf_sbi", "smoke_candidates", n_draw),
            inference_smoke=smoke,
        )
        if fast is not None:
            return fast.astype(np.float32, copy=False)
        if use_snpe_proposal:
            return _sample_snpe_mixture_candidates(
                ensemble_posteriors=ensemble_posteriors,
                prior=prior,
                n=int(n_draw),
                posterior_mix_alpha=posterior_mix_alpha,
                theta_min=theta_min,
                theta_max=theta_max,
            )
        return safe_sample_numpy(dist=proposal_for_sampling, n=int(n_draw), fallback_dist=prior).astype(np.float32)

    training_proposal: Any = prior if use_snpe_proposal else proposal_for_sampling

    for r, m_size in enumerate(mix_round_sizes, start=1):
        if remaining_hf < 0 or remaining_lf < 0:
            raise RuntimeError("Negative remaining budgets encountered.")
        if m_size <= 0:
            continue

        # Candidate parameters: truncated proposal or SNPE posterior+prior mix (mf_sbi).
        theta_candidates = _draw_mixed_round_candidates(int(m_size))

        candidates_t = torch.as_tensor(theta_candidates, dtype=torch.float32)
        scores = _ensemble_variance_scores(ensemble_posteriors=ensemble_posteriors, candidates_t=candidates_t)
        order = np.argsort(scores)[::-1]

        # paper-1 k% allocation using remaining budgets + HF share cap
        if remaining_hf <= 0:
            n_hf_this_round = 0
        elif remaining_lf <= 0:
            n_hf_this_round = int(m_size)
        else:
            effective_k_percent = min(hf_top_k_percent, 100.0 * remaining_hf / (remaining_hf + remaining_lf))
            n_hf_want = max(0, int(round(effective_k_percent / 100.0 * int(m_size))))
            if n_hf_want == 0 and m_size > 0 and force_min_hf_per_mixed_round:
                n_hf_want = 1

            n_hf_this_round = min(n_hf_want, remaining_hf)
            # Ensure we can fill the rest with LF without exceeding remaining_lf.
            n_lf_this_round = int(m_size) - n_hf_this_round
            if n_lf_this_round > remaining_lf:
                n_lf_this_round = remaining_lf
                n_hf_this_round = int(m_size) - n_lf_this_round
        n_lf_this_round = int(m_size) - int(n_hf_this_round)

        hf_idx = order[:n_hf_this_round] if n_hf_this_round > 0 else np.array([], dtype=int)
        lf_idx = order[n_hf_this_round : n_hf_this_round + n_lf_this_round] if n_lf_this_round > 0 else np.array([], dtype=int)

        theta_hf = theta_candidates[hf_idx] if n_hf_this_round > 0 else np.empty((0, theta_candidates.shape[1]), dtype=np.float32)
        theta_lf = theta_candidates[lf_idx] if n_lf_this_round > 0 else np.empty((0, theta_candidates.shape[1]), dtype=np.float32)
        theta_hf = _sanitize_theta_batch(theta=theta_hf, prior=prior, theta_min=theta_min, theta_max=theta_max)
        theta_lf = _sanitize_theta_batch(theta=theta_lf, prior=prior, theta_min=theta_min, theta_max=theta_max)
        theta_hf = _sanitize_theta_with_prior(
            theta=theta_hf,
            prior=prior,
            seed=derive_seed(seed, "mf_sbi", "theta_hf_support", r),
        )
        theta_lf = _sanitize_theta_with_prior(
            theta=theta_lf,
            prior=prior,
            seed=derive_seed(seed, "mf_sbi", "theta_lf_support", r),
        )

        # simulate both fidelities for this mixed batch
        if theta_hf.shape[0] > 0:
            theta_hf, x_hf = _simulate_hf_paired(
                task,
                theta_hf,
                seed=derive_seed(seed, "mf_sbi", "x_hf_round", r),
                seed_tag=f"mix_hf_{r}",
            )
        else:
            x_hf = np.empty((0, int(x_o_t.shape[-1])), dtype=np.float32)

        if theta_lf.shape[0] > 0:
            theta_lf, x_lf_mix = _simulate_lf_paired(
                task,
                theta_lf,
                seed=derive_seed(seed, "mf_sbi", "x_lf_round", r),
                seed_tag=f"mix_lf_{r}",
            )
        else:
            x_lf_mix = np.empty((0, int(x_o_t.shape[-1])), dtype=np.float32)

        # concat for training (HF points first, then LF points)
        theta_round = np.vstack([theta_hf, theta_lf]) if theta_hf.shape[0] + theta_lf.shape[0] > 0 else np.empty((0, theta_candidates.shape[1]), dtype=np.float32)
        x_round = np.vstack([x_hf, x_lf_mix]) if x_hf.shape[0] + x_lf_mix.shape[0] > 0 else np.empty((0, 1), dtype=np.float32)

        theta_round_t = torch.as_tensor(theta_round, dtype=torch.float32)
        x_round_t = torch.as_tensor(x_round, dtype=torch.float32)
        if theta_clamp_lo_t is not None and theta_clamp_hi_t is not None and theta_round_t.numel() > 0:
            # Ensure theta stays strictly inside prior support so SNPE-C prior.eval never hits -inf.
            theta_round_t = torch.max(torch.min(theta_round_t, theta_clamp_hi_t), theta_clamp_lo_t)

        # Final hardening: ensure prior.log_prob(theta) is finite for every row.
        if theta_round_t.numel() > 0:
            theta_round_t = theta_round_t.clone()
            n_dim = int(theta_round_t.shape[1])
            for repair_pass in range(5):
                with torch.no_grad():
                    lp = prior.log_prob(theta_round_t).reshape(-1)
                bad = ~torch.isfinite(lp)
                if not bool(bad.any().item()):
                    break
                n_bad = int(bad.sum().item())
                if n_bad <= 0:
                    break

                repl_np = _draw_mixed_round_candidates(n_bad)
                if repl_np.shape[0] != n_bad:
                    repl_np = safe_sample_numpy(dist=prior, n=n_bad).astype(np.float32)
                if repl_np.shape[0] != n_bad:
                    repl_np = prior.sample((n_bad,)).detach().cpu().numpy().astype(np.float32)
                repl = torch.as_tensor(repl_np, dtype=theta_round_t.dtype, device=theta_round_t.device)

                theta_round_t[bad] = repl
                if theta_clamp_lo_t is not None and theta_clamp_hi_t is not None:
                    theta_round_t = torch.max(torch.min(theta_round_t, theta_clamp_hi_t), theta_clamp_lo_t)

            # Final fallback: if any row is still non-finite under prior eval,
            # set it to the strict midpoint.
        with torch.no_grad():
            lp_final = prior.log_prob(theta_round_t).reshape(-1)
        bad_final = ~torch.isfinite(lp_final)
        if bool(bad_final.any().item()):
            n_bad_final = int(bad_final.sum().item())
            repl_np = _draw_mixed_round_candidates(n_bad_final)
            if repl_np.shape[0] != n_bad_final:
                repl_np = prior.sample((n_bad_final,)).detach().cpu().numpy().astype(np.float32)
            theta_round_t[bad_final] = torch.as_tensor(
                repl_np,
                dtype=theta_round_t.dtype,
                device=theta_round_t.device,
            )
        usable_round = _usable_train_tensors(theta_round_t, x_round_t, min_train_samples)
        if usable_round is None:
            remaining_hf -= int(n_hf_this_round)
            remaining_lf -= int(n_lf_this_round)
            total_hf += int(n_hf_this_round)
            total_lf_mix += int(n_lf_this_round)
            hf_lf_logs.append(
                {
                    "round": int(r),
                    "n_mixed": int(m_size),
                    "n_hf_in_round": int(n_hf_this_round),
                    "n_lf_in_round": int(n_lf_this_round),
                    "remaining_hf_after": int(remaining_hf),
                    "remaining_lf_after": int(remaining_lf),
                    "effective_k_percent": None,
                    "trained": False,
                }
            )
            continue

        theta_round_t, x_round_t = usable_round

        new_posteriors = []
        for e, inf_e in enumerate(ensemble_inferences):
            timer = Timer()
            timer.start()
            de_e = inf_e.append_simulations(theta_round_t, x_round_t, proposal=training_proposal).train(
                training_batch_size=max(1, min(hf_training_batch_size, int(theta_round_t.shape[0]))),
                max_num_epochs=hf_max_num_epochs,
                show_train_summary=False,
                validation_fraction=0.5,
                # In the first mixed round we sample from the prior, so SBI treats this as "round 0".
                # We need to allow training with first-round loss even though the network already exists.
                force_first_round_loss=True,
            )
            timer.stop()
            train_seconds += float(timer.elapsed_s)
            new_posteriors.append(
                build_posterior_with_sampling(
                    inf_e,
                    de_e,
                    sample_with=posterior_sample_with,
                ).set_default_x(x_o_t)
            )
        ensemble_posteriors = new_posteriors

        remaining_hf -= int(n_hf_this_round)
        remaining_lf -= int(n_lf_this_round)
        total_hf += int(n_hf_this_round)
        total_lf_mix += int(n_lf_this_round)

        # Update proposal for next round (truncated-proposal only; SNPE uses posterior+prior mix each round).
        if not use_snpe_proposal:
            if disable_truncation:
                proposal_for_sampling = prior
                training_proposal = prior
            else:
                try:
                    ensemble_posterior = NeuralPosteriorEnsemble(ensemble_posteriors)
                    accept_reject_fn = get_density_thresholder(
                        ensemble_posterior,
                        quantile=proposal_quantile,
                        num_samples_to_estimate_support=50_000,
                    )
                    proposal_for_sampling = RestrictedPrior(prior, accept_reject_fn, sample_with="mcmc")
                    training_proposal = proposal_for_sampling
                except Exception:
                    proposal_for_sampling = prior
                    training_proposal = prior

        hf_lf_logs.append(
            {
                "round": int(r),
                "n_mixed": int(m_size),
                "n_hf_in_round": int(n_hf_this_round),
                "n_lf_in_round": int(n_lf_this_round),
                "remaining_hf_after": int(remaining_hf),
                "remaining_lf_after": int(remaining_lf),
                "effective_k_percent": float(
                    min(hf_top_k_percent, 100.0 * (remaining_hf + int(n_hf_this_round)) / ((remaining_hf + int(n_hf_this_round)) + (remaining_lf + int(n_lf_this_round))))
                )
                if (remaining_hf + int(n_hf_this_round)) > 0 and (remaining_lf + int(n_lf_this_round)) > 0
                else None,
                "trained": True,
            }
        )

    # Consume any leftover quotas introduced by floor-based round sizing.
    if remaining_hf > 0 or remaining_lf > 0:
        theta_hf_topup = prior.sample((int(remaining_hf),)).detach().cpu().numpy().astype(np.float32) if remaining_hf > 0 else np.empty((0, int(theta_lf_t.shape[-1])), dtype=np.float32)
        theta_lf_topup = prior.sample((int(remaining_lf),)).detach().cpu().numpy().astype(np.float32) if remaining_lf > 0 else np.empty((0, int(theta_lf_t.shape[-1])), dtype=np.float32)
        theta_hf_topup = _sanitize_theta_batch(theta=theta_hf_topup, prior=prior, theta_min=theta_min, theta_max=theta_max)
        theta_lf_topup = _sanitize_theta_batch(theta=theta_lf_topup, prior=prior, theta_min=theta_min, theta_max=theta_max)
        theta_hf_topup = _sanitize_theta_with_prior(
            theta=theta_hf_topup,
            prior=prior,
            seed=derive_seed(seed, "mf_sbi", "theta_hf_topup_support"),
        )
        theta_lf_topup = _sanitize_theta_with_prior(
            theta=theta_lf_topup,
            prior=prior,
            seed=derive_seed(seed, "mf_sbi", "theta_lf_topup_support"),
        )

        if theta_hf_topup.shape[0] > 0:
            theta_hf_topup, x_hf_topup = _simulate_hf_paired(
                task,
                theta_hf_topup,
                seed=derive_seed(seed, "mf_sbi", "x_hf_topup"),
                seed_tag="hf_topup",
            )
        else:
            x_hf_topup = np.empty((0, int(x_o_t.shape[-1])), dtype=np.float32)
        if theta_lf_topup.shape[0] > 0:
            theta_lf_topup, x_lf_topup = _simulate_lf_paired(
                task,
                theta_lf_topup,
                seed=derive_seed(seed, "mf_sbi", "x_lf_topup"),
                seed_tag="lf_topup",
            )
        else:
            x_lf_topup = np.empty((0, int(x_o_t.shape[-1])), dtype=np.float32)
        theta_round_t = torch.as_tensor(np.vstack([theta_hf_topup, theta_lf_topup]).astype(np.float32), dtype=torch.float32)
        x_round_t = torch.as_tensor(np.vstack([x_hf_topup, x_lf_topup]).astype(np.float32), dtype=torch.float32)
        usable_topup = _usable_train_tensors(theta_round_t, x_round_t, min_train_samples)
        if usable_topup is not None:
            theta_round_t, x_round_t = usable_topup
            new_posteriors = []
            for inf_e in ensemble_inferences:
                timer = Timer()
                timer.start()
                de_e = inf_e.append_simulations(theta_round_t, x_round_t, proposal=training_proposal).train(
                    training_batch_size=max(1, min(hf_training_batch_size, int(theta_round_t.shape[0]))),
                    max_num_epochs=hf_max_num_epochs,
                    show_train_summary=False,
                    validation_fraction=0.5,
                    force_first_round_loss=True,
                )
                timer.stop()
                train_seconds += float(timer.elapsed_s)
                new_posteriors.append(
                    build_posterior_with_sampling(
                        inf_e,
                        de_e,
                        sample_with=posterior_sample_with,
                    ).set_default_x(x_o_t)
                )
            ensemble_posteriors = new_posteriors

        total_hf += int(remaining_hf)
        total_lf_mix += int(remaining_lf)
        hf_lf_logs.append(
            {
                "round": int(max(1, len(hf_lf_logs) + 1)),
                "n_mixed": int(remaining_hf + remaining_lf),
                "n_hf_in_round": int(remaining_hf),
                "n_lf_in_round": int(remaining_lf),
                "remaining_hf_after": 0,
                "remaining_lf_after": 0,
                "effective_k_percent": None,
                "trained": bool(usable_topup is not None),
            }
        )
        remaining_hf = 0
        remaining_lf = 0

    if int(total_hf) != int(budget_hf) or int(total_lf_mix + n_active_lf + n_warmup) != int(budget_lf):
        # Hard consistency check: budgets must be consumed strictly.
        raise RuntimeError(
            f"Budget mismatch: expected (lf={budget_lf}, hf={budget_hf}), "
            f"got (lf={int(n_warmup) + int(n_active_lf) + int(total_lf_mix)}, hf={int(total_hf)})."
        )

    # Final posterior samples: aggregate ensemble samples.
    E = len(ensemble_posteriors)
    fast = smoke_final_samples(task, num_posterior_samples, seed, inference_smoke=smoke)
    if fast is not None:
        posterior_samples = fast
    else:
        per_e = max(1, int(num_posterior_samples) // max(1, E))
        parts = [safe_sample_numpy(dist=post, n=per_e, fallback_dist=prior) for post in ensemble_posteriors]
        posterior_samples = np.concatenate(parts, axis=0)[: int(num_posterior_samples)]

    timer_total.stop()

    diagnostics: dict[str, Any] = {
        "method": diagnostics_method_name,
        "budget_lf": int(budget_lf),
        "budget_hf": int(budget_hf),
        "num_rounds": int(num_rounds),
        "proposal_quantile": proposal_quantile,
        "disable_truncation": disable_truncation,
        "posterior_sample_with": posterior_sample_with,
        "use_snpe_proposal": use_snpe_proposal,
        "posterior_mix_alpha": posterior_mix_alpha,
        "neural_net": neural_net,
        "hidden_features": hidden_features,
        "z_score_x": z_score_x,
        "z_score_theta": z_score_theta,
        "lf_training_batch_size": lf_training_batch_size,
        "hf_training_batch_size": hf_training_batch_size,
        "lf_max_num_epochs": lf_max_num_epochs,
        "hf_max_num_epochs": hf_max_num_epochs,
        "ensemble_size": ensemble_size,
        "lf_warmup_fraction": lf_warmup_fraction,
        "lf_active_fraction": lf_active_fraction,
        "lf_active_rounds": lf_active_rounds,
        "use_phase2_lf_active": use_phase2_lf_active,
        "phase2_acquisition_fidelity": phase2_acquisition_fidelity,
        "phase2_hf_match_mode": phase2_hf_match_mode,
        "force_min_hf_per_mixed_round": force_min_hf_per_mixed_round,
        "n_warmup": int(n_warmup),
        "n_active_lf_requested": int(n_active_lf_requested),
        "n_active_lf": int(n_active_lf),
        "n_active_hf_phase2": int(n_active_hf_phase2),
        "n_mix_lf": int(n_mix_lf),
        "active_fraction_lf": active_fraction_lf,
        "candidate_pool_size": candidate_pool_size,
        "hf_top_k_percent": hf_top_k_percent,
        "phase2_logs": lf_active_logs,
        "phase3_logs": hf_lf_logs,
    }
    if ensemble_posteriors:
        add_nltp_exact_neural(
            diagnostics, NeuralPosteriorEnsemble(ensemble_posteriors), theta_true, x_o
        )
    elif theta_true is not None:
        diagnostics["nltp_exact_status"] = "skipped_empty_ensemble"

    return MfSbiResult(
        posterior_samples=posterior_samples.astype(np.float32),
        diagnostics=diagnostics,
        n_hf_calls=int(total_hf),
        n_lf_calls=int(budget_lf),
        train_seconds=float(train_seconds),
        total_seconds=float(timer_total.elapsed_s),
    )

