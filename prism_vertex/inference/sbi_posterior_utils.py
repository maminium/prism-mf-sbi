"""
Fast, stable log-probability helpers for sbi posteriors and restricted priors.

SBI defaults normalize densities via Monte Carlo (rejection sampling inside the flow for
``DirectPosterior``, prior acceptance for ``RestrictedPrior``). That can dominate wall
time and trigger nflows spline issues. For training heuristics (ensemble disagreement,
truncation scoring) we only need unnormalized log-densities — use the ``norm_*=False``
flags below.

This is separate from ``safe_sampling.py`` (try/except around ``.sample()``) and from
``_sanitize_theta_with_prior`` (prior support checks before simulators).
"""

from __future__ import annotations

from typing import Any

import torch
from sbi.utils import RestrictedPrior


def build_posterior_with_sampling(
    inference: Any,
    density_estimator: Any,
    *,
    sample_with: str = "mcmc",
) -> Any:
    """
    Build an SBI posterior with an explicit sampling backend.

    ``sample_with="mcmc"`` is the robust default for tightly truncated truncated-proposal
    posteriors. ``sample_with="direct"`` / ``"rejection"`` request SBI's fast
    ``DirectPosterior`` path (flow sampling). In current sbi, that backend is
    selected by ``sample_with="rejection"`` *without* a custom proposal.
    """
    requested = str(sample_with).lower().strip()
    # sbi>=0.26 accepts sample_with="direct". Older docs/configs alias "direct"
    # to rejection sampling; keep that as a fallback only if direct fails.
    if requested in {"mcmc", "vi", "importance", "direct", "rejection"}:
        sbi_sample_with = requested
    else:
        raise ValueError(
            f"posterior_sample_with must be one of "
            f"{{'mcmc','direct','rejection','vi','importance'}}, got {sample_with!r}"
        )
    try:
        return inference.build_posterior(density_estimator, sample_with=sbi_sample_with)
    except (TypeError, ValueError, NotImplementedError):
        # Older sbi without "direct", or rejection without prior: try aliases.
        if requested == "direct":
            try:
                return inference.build_posterior(density_estimator, sample_with="rejection")
            except (TypeError, ValueError, NotImplementedError):
                return inference.build_posterior(density_estimator)
        try:
            return inference.build_posterior(density_estimator)
        except (TypeError, NotImplementedError):
            return inference.build_posterior(density_estimator)


def build_posterior_mcmc(inference: Any, density_estimator: Any) -> Any:
    """
    Build an SBI posterior that samples with MCMC by default.

    DirectPosterior rejection sampling hangs under tight truncated-proposal truncation
    (``Only 0.000% proposal samples are accepted``). MCMC targets the same
    density and is what SBI recommends in that warning.
    """
    return build_posterior_with_sampling(inference, density_estimator, sample_with="mcmc")


def posterior_log_prob_unnorm(
    posterior: Any,
    theta: torch.Tensor,
    *,
    x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unnormalized log-density when supported; fall back for MCMCPosterior / older sbi."""
    kwargs: dict[str, Any] = {}
    if x is not None:
        kwargs["x"] = x
    try:
        return posterior.log_prob(theta, norm_posterior=False, **kwargs)
    except TypeError:
        # MCMCPosterior (sample_with=\"mcmc\") has no norm_posterior kwarg.
        return posterior.log_prob(theta, **kwargs)


def proposal_log_prob_unnorm_if_restricted(proposal: Any, theta: torch.Tensor) -> torch.Tensor:
    """
    ``RestrictedPrior.log_prob`` with ``norm_restricted_prior=False`` when applicable.

    Default ``norm_restricted_prior=True`` estimates acceptance via extra sampling
    (see sbi ``prior_acceptance``), which is unnecessary for relative log-density ranking
    in truncation heuristics.
    """
    if isinstance(proposal, RestrictedPrior):
        return proposal.log_prob(theta, norm_restricted_prior=False)
    return proposal.log_prob(theta)
