from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prism_vertex.benchmark.comparison_paths import method_dir_name
from prism_vertex.experiments import get_task
from prism_vertex.inference.hf_npe import run_hf_npe
from prism_vertex.inference.hf_smc_abc import run_hf_smc_abc
from prism_vertex.inference.hf_snpe import run_hf_snpe
from prism_vertex.inference.mf_sbi import run_mf_sbi
from prism_vertex.utils.io import (
    CostTracking,
    RunMeta,
    write_cost_tracking_csv,
    write_json,
    write_posterior_samples_csv,
    write_run_config,
)

PAPER_METHODS = frozenset({"hf_smc_abc", "hf_npe", "hf_snpe", "mf_sbi"})
BASELINE_DROP_METHODS = frozenset({"hf_npe", "hf_snpe", "mf_sbi"})


def _lf_fidelity_id(task: Any) -> str | None:
    raw = os.environ.get("PRISM_VERTEX_LF_FIDELITY", "").strip()
    if raw:
        return raw
    if hasattr(task, "lf_fidelity_id"):
        try:
            return str(task.lf_fidelity_id)
        except Exception:
            pass
    return None


def _run_group() -> str | None:
    raw = os.environ.get("PRISM_VERTEX_RUN_GROUP", "").strip()
    if not raw:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(ch not in allowed for ch in raw):
        raise ValueError(f"Invalid PRISM_VERTEX_RUN_GROUP={raw!r}")
    return raw


def _resolve_inference_run_dir(
    *,
    exp_dir: Path,
    method: str,
    budget_hf: int,
    budget_lf: int,
    obs_id: int,
    seed: int,
    task: Any,
) -> tuple[Path, str, str | None]:
    budget_dir = (
        f"budget_lf_{int(budget_lf):06d}_hf_{int(budget_hf):06d}"
        if int(budget_lf) > 0
        else f"budget_hf_{int(budget_hf):06d}"
    )
    lf_id: str | None = _lf_fidelity_id(task) if int(budget_lf) > 0 else None

    parts: list[str] = ["results", "posterior_runs", method_dir_name(method)]
    run_group = _run_group()
    if run_group:
        parts.append(run_group)
    if lf_id:
        parts.append(lf_id)
    parts.extend([budget_dir, f"obs_{obs_id:02d}", f"seed_{seed:02d}"])
    return exp_dir.joinpath(*parts), budget_dir, lf_id


def _configure_simulation_env(*, method: str) -> None:
    if method in BASELINE_DROP_METHODS:
        os.environ["PRISM_VERTEX_BASELINE_DROP_ON_FAIL"] = "1"
    else:
        os.environ.pop("PRISM_VERTEX_BASELINE_DROP_ON_FAIL", None)
    os.environ.pop("PRISM_VERTEX_FEASIBILITY_PENALTY_MODE", None)


def run_method(
    *,
    experiment_id: str,
    method: str,
    obs_id: int,
    seed: int,
    budget_hf: int,
    num_posterior_samples: int,
    budget_lf: int = 0,
    method_config: Mapping[str, Any] | None = None,
    experiment_config: Mapping[str, Any] | None = None,
    results_root: str | Path = "data",
) -> Path:
    if experiment_id not in {"proliferating_vertex", "nonproliferating_vertex", "prism_vertex"}:
        raise ValueError(
            "Supported experiment_id: proliferating_vertex | nonproliferating_vertex "
            "(alias: prism_vertex → proliferating_vertex)."
        )
    if experiment_id == "prism_vertex":
        experiment_id = "proliferating_vertex"
    if method not in PAPER_METHODS:
        raise ValueError(f"Unknown method: {method}")

    results_root = Path(results_root)
    task = get_task(experiment_id, config=dict(experiment_config or {}))
    # Default data layout: data/<experiment_id>/
    if results_root.name == "data" or results_root == Path("data"):
        exp_data = results_root / experiment_id
        if exp_data.is_dir():
            results_root = exp_data
    x_o = task.get_observation(obs_id)
    try:
        theta_true = task.get_true_parameters(obs_id)
    except Exception:
        theta_true = None

    exp_dir = results_root
    run_dir, budget_dir, lf_fidelity_id = _resolve_inference_run_dir(
        exp_dir=exp_dir,
        method=method,
        budget_hf=budget_hf,
        budget_lf=budget_lf,
        obs_id=obs_id,
        seed=seed,
        task=task,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    from prism_vertex.simulator.inference_simulation import init_inference_simulation

    _configure_simulation_env(method=method)
    init_inference_simulation(
        x_o=np.asarray(x_o, dtype=np.float64),
        run_dir=run_dir,
        observations_dir=getattr(task, "observations_dir", None),
    )

    run_group = _run_group()
    meta = RunMeta(
        experiment_id=task.experiment_id,
        experiment_name=task.experiment_name,
        method=method,
        obs_id=obs_id,
        seed=seed,
        budget_id=(
            "/".join(p for p in (run_group, lf_fidelity_id, budget_dir) if p) or budget_dir
        ),
    )

    cfg_payload = {
        "experiment_config": dict(experiment_config or {}),
        "method_config": dict(method_config or {}),
        "budget_lf": budget_lf,
        "budget_hf": budget_hf,
        "num_posterior_samples": num_posterior_samples,
    }
    if lf_fidelity_id:
        cfg_payload["lf_fidelity_id"] = lf_fidelity_id
    if run_group:
        cfg_payload["run_group"] = run_group
    write_run_config(run_dir / "run_config.json", meta, cfg_payload)

    if method == "hf_npe":
        res = run_hf_npe(
            task=task,
            x_o=x_o,
            budget_hf=budget_hf,
            seed=seed,
            num_posterior_samples=num_posterior_samples,
            config=method_config,
            theta_true=theta_true,
        )
        samples = res.posterior_samples
        diagnostics = res.diagnostics
        c_hf = float(dict(method_config or {}).get("c_hf", 1.0))
        cost = CostTracking(
            n_hf_calls=res.n_hf_calls,
            n_lf_calls=0,
            c_hf=c_hf,
            c_lf=0.0,
            lambda_l_over_h=0.0,
            total_cost=float(res.n_hf_calls) * c_hf,
            wallclock_seconds_total=res.total_seconds,
            train_seconds=res.train_seconds,
            sim_seconds_hf=None,
            sim_seconds_lf=None,
        )
    elif method == "hf_smc_abc":
        res = run_hf_smc_abc(
            task=task,
            x_o=x_o,
            budget_hf=budget_hf,
            seed=seed,
            num_posterior_samples=num_posterior_samples,
            config=method_config,
            theta_true=theta_true,
        )
        samples = res.posterior_samples
        diagnostics = res.diagnostics
        c_hf = float(dict(method_config or {}).get("c_hf", 1.0))
        cost = CostTracking(
            n_hf_calls=res.n_hf_calls,
            n_lf_calls=0,
            c_hf=c_hf,
            c_lf=0.0,
            lambda_l_over_h=0.0,
            total_cost=float(res.n_hf_calls) * c_hf,
            wallclock_seconds_total=res.total_seconds,
            train_seconds=None,
            sim_seconds_hf=None,
            sim_seconds_lf=None,
        )
    elif method == "mf_sbi":
        cfg = dict(method_config or {})
        c_hf = float(cfg.get("c_hf", 1.0))
        c_lf = float(cfg.get("c_lf", 0.1))
        res = run_mf_sbi(
            task=task,
            x_o=x_o,
            budget_lf=int(budget_lf),
            budget_hf=budget_hf,
            seed=seed,
            num_posterior_samples=num_posterior_samples,
            config=cfg,
            theta_true=theta_true,
        )
        samples = res.posterior_samples
        diagnostics = res.diagnostics
        total_cost = float(res.n_hf_calls) * c_hf + float(res.n_lf_calls) * c_lf
        cost = CostTracking(
            n_hf_calls=res.n_hf_calls,
            n_lf_calls=res.n_lf_calls,
            c_hf=c_hf,
            c_lf=c_lf,
            lambda_l_over_h=(c_lf / c_hf) if c_hf > 0 else 0.0,
            total_cost=total_cost,
            wallclock_seconds_total=res.total_seconds,
            train_seconds=res.train_seconds,
            sim_seconds_hf=None,
            sim_seconds_lf=None,
        )
        samples = res.posterior_samples
        diagnostics = res.diagnostics
        total_cost = float(res.n_hf_calls) * c_hf + float(res.n_lf_calls) * c_lf
        cost = CostTracking(
            n_hf_calls=res.n_hf_calls,
            n_lf_calls=res.n_lf_calls,
            c_hf=c_hf,
            c_lf=c_lf,
            lambda_l_over_h=(c_lf / c_hf) if c_hf > 0 else 0.0,
            total_cost=total_cost,
            wallclock_seconds_total=res.total_seconds,
            train_seconds=res.train_seconds,
            sim_seconds_hf=None,
            sim_seconds_lf=None,
        )
    else:
        cfg = dict(method_config or {})
        c_hf = float(cfg.get("c_hf", 1.0))
        res = run_hf_snpe(
            task=task,
            x_o=x_o,
            budget_hf=budget_hf,
            seed=seed,
            num_posterior_samples=num_posterior_samples,
            config=cfg,
            theta_true=theta_true,
        )
        samples = res.posterior_samples
        diagnostics = res.diagnostics
        cost = CostTracking(
            n_hf_calls=res.n_hf_calls,
            n_lf_calls=0,
            c_hf=c_hf,
            c_lf=0.0,
            lambda_l_over_h=0.0,
            total_cost=float(res.n_hf_calls) * c_hf,
            wallclock_seconds_total=res.total_seconds,
            train_seconds=res.train_seconds,
            sim_seconds_hf=None,
            sim_seconds_lf=None,
        )

    diagnostics_out = dict(diagnostics)
    from prism_vertex.simulator.inference_failure_registry import InferenceFailureRegistry

    reg = InferenceFailureRegistry.from_env()
    diagnostics_out["failure_registry"] = {
        "blocked_count": reg.blocked_count(),
        "stats": reg.stats(),
        "registry_path": str(reg.path) if reg.path is not None else None,
    }

    write_posterior_samples_csv(
        run_dir / "posterior_samples.csv",
        meta,
        theta_names=list(task.theta_names),
        samples=samples.tolist(),
    )
    write_cost_tracking_csv(run_dir / "cost_tracking.csv", meta, cost)
    write_json(run_dir / "diagnostics.json", {"meta": asdict(meta), **diagnostics_out})
    return run_dir


def run_method_for_obs_ids(
    *,
    experiment_id: str,
    method: str,
    obs_ids: list[int],
    seed: int,
    budget_hf: int,
    num_posterior_samples: int,
    budget_lf: int = 0,
    method_config: Mapping[str, Any] | None = None,
    experiment_config: Mapping[str, Any] | None = None,
    results_root: str | Path = "data",
) -> list[Path]:
    return [
        run_method(
            experiment_id=experiment_id,
            method=method,
            obs_id=int(obs_id),
            seed=seed,
            budget_hf=budget_hf,
            budget_lf=budget_lf,
            num_posterior_samples=num_posterior_samples,
            method_config=method_config,
            experiment_config=experiment_config,
            results_root=results_root,
        )
        for obs_id in obs_ids
    ]
