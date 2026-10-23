"""
Single evaluation entry point for proliferating_vertex benchmarks.

Paper metrics (NLTP, CRPS, NRMSE, TARP, SBC, posterior STD) are computed here.
Primitives live in `prism_vertex.metrics.benchmark_unified`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kstest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism_vertex.benchmark.config import load_benchmark_config
from prism_vertex.benchmark.calibration import (
    CALIBRATION_COMPARISON_SUBDIR,
    iter_calibration_run_dirs,
)
from prism_vertex.benchmark.comparison_paths import (
    MAIN_METHOD_COMPARISON_DIR,
    iter_fig4_mf_sbi_runs,
    load_fig4_manifest,
    manifest_by_results_path,
    manifest_nominal_lf_budget,
    method_name_from_dir_name,
    method_results_dir,
    posterior_runs_dir,
    run_dir_results_path,
)
from prism_vertex.benchmark.inference_run_dirs import (
    budget_dir_name,
    is_paper_experiment_dir,
    iter_paper_method_comparison_run_dirs,
)
from prism_vertex.benchmark.registry import PAPER_METHODS_STR
from prism_vertex.metrics import benchmark_unified as bu

META_COLS = {"experiment_id", "experiment_name", "method", "obs_id", "seed", "budget_id"}


def _load_samples(path: Path) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    theta_cols = [c for c in fields if c not in META_COLS and c != "log_weight"]
    arr = np.asarray([[float(r[c]) for c in theta_cols] for r in rows], dtype=float)
    log_w = None
    if "log_weight" in fields:
        log_w = np.asarray([float(r["log_weight"]) for r in rows], dtype=float)
    return arr, theta_cols, log_w


def _load_reference(exp_dir: Path, obs_id: int, use_reference_posterior: bool) -> Path | None:
    if use_reference_posterior:
        p1 = exp_dir / "reference_gold" / f"obs_{obs_id:03d}" / "reference_posterior_hf.csv"
        p2 = exp_dir / "reference" / f"obs_{obs_id:03d}" / "reference_posterior_samples.csv"
        if p1.exists():
            return p1
        if p2.exists():
            return p2
    return None


def _load_cost(path: Path) -> dict[str, float]:
    with path.open() as f:
        row = next(csv.DictReader(f))
    out: dict[str, float] = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def _effective_unit_costs_hf1(
    experiment_id: str,
    exp_dir: Path,
    *,
    lf_cost_per_hf: float = 0.1,
) -> tuple[float, float] | None:
    """Return (c_hf, c_lf) with c_hf normalized to 1 for proliferating_vertex metrics."""
    if experiment_id.strip().lower() not in {"proliferating_vertex", "prism_vertex"}:
        return None
    cl = float(lf_cost_per_hf)
    if not np.isfinite(cl) or cl < 0:
        cl = 0.1
    return (1.0, cl)


def _adjust_cost_tracking_row(
    c: dict[str, float], unit_costs: tuple[float, float] | None
) -> dict[str, float]:
    """Recompute total_cost (and related fields) from call counts when unit_costs is set."""
    if unit_costs is None:
        return c
    nh = c.get("n_hf_calls")
    nl = c.get("n_lf_calls")
    if nh is None or nl is None:
        return c
    ch, cl = unit_costs
    out = dict(c)
    out["c_hf"] = float(ch)
    out["c_lf"] = float(cl)
    out["lambda_l_over_h"] = float(cl / ch) if ch > 0 else float("nan")
    out["total_cost"] = float(ch) * float(nh) + float(cl) * float(nl)
    return out


def _load_obs_x(exp_dir: Path, obs_id: int) -> np.ndarray | None:
    candidates = [
        exp_dir / "observations" / f"obs_{obs_id:02d}" / "s_obs.npy",
        exp_dir / "observations" / f"obs_{obs_id:02d}.npy",
    ]
    if obs_id == 0:
        candidates.extend(
            [
                exp_dir / "observations" / "s_obs.npy",
                exp_dir / "observations" / "obs_00.npy",
            ]
        )
    for p in candidates:
        if p.exists():
            return np.asarray(np.load(p), dtype=float).reshape(-1)
    return None


def _load_theta_true(exp_dir: Path, obs_id: int) -> np.ndarray | None:
    for p in (
        exp_dir / "observations" / f"obs_{obs_id:02d}" / "theta_true.npy",
        exp_dir / "observations" / f"theta_{obs_id:02d}.npy",
    ):
        if p.exists():
            return np.asarray(np.load(p), dtype=float).reshape(-1)
    if obs_id == 0 and (exp_dir / "observations" / "theta_true.npy").exists():
        return np.asarray(np.load(exp_dir / "observations" / "theta_true.npy"), dtype=float).reshape(-1)
    try:
        from prism_vertex.simulator.calibration_observations import (
            theta_for_obs,
        )

        return theta_for_obs(obs_id)
    except KeyError:
        return None


def _load_predictive_matrix(path: Path) -> np.ndarray | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    cols = [c for c in df.columns if c.startswith("x_") or c.startswith("y_") or c.startswith("pred_")]
    if not cols:
        num = [c for c in df.columns if c not in META_COLS and c != "log_weight"]
        cols = num
    if not cols:
        return None
    return df[sorted(cols, key=str)].to_numpy(dtype=float)




def _nrmse(samples: np.ndarray, theta_true: np.ndarray) -> float:
    mean_est = samples.mean(axis=0)
    denom = np.maximum(np.abs(theta_true), 1e-8)
    rel = (mean_est - theta_true) / denom
    return float(np.sqrt(np.mean(rel**2)))


def _nltp_gaussian(samples: np.ndarray, theta_true: np.ndarray) -> float:
    mu = samples.mean(axis=0)
    cov = np.cov(samples.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    diff = theta_true - mu
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + np.eye(cov.shape[0]) * 1e-3
        sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    k = float(cov.shape[0])
    quad = float(diff.T @ inv @ diff)
    return float(0.5 * (k * np.log(2.0 * np.pi) + logdet + quad))


def _plausibility_rate(samples: np.ndarray, bounds: list[tuple[float, float]]) -> float:
    inside = np.ones(len(samples), dtype=bool)
    for j, (lo, hi) in enumerate(bounds):
        inside = inside & (samples[:, j] >= lo) & (samples[:, j] <= hi)
    return float(np.mean(inside))


def _sbc_rank_stats(samples: np.ndarray, theta_true: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(samples)
    for j in range(samples.shape[1]):
        rank = int(np.sum(samples[:, j] <= theta_true[j]))
        rows.append(
            {
                "parameter_index": j,
                "rank": rank,
                "rank_normalized": float(rank / max(n, 1)),
            }
        )
    return rows


def _aggregate(df: pd.DataFrame, group_cols: list[str], value_col: str = "value") -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)[value_col]
    out = g.agg(["mean", "std", "median", "count"]).reset_index()
    out = out.rename(
        columns={
            "mean": "metric_value_mean",
            "std": "metric_value_std",
            "median": "metric_value_median",
            "count": "n_replicates",
        }
    )
    out["metric_value_std"] = out["metric_value_std"].fillna(0.0)
    return out


def _budget_matches_lf_tag(budget_name: str, lf_budget_tag: str, method: str) -> bool:
    budget_name = budget_dir_name(budget_name)
    if lf_budget_tag == "all":
        return True
    if method in {"hf_smc_abc", "hf_npe", "hf_snpe"} and "budget_hf_" in budget_name:
        return True
    if lf_budget_tag == "lf1e4":
        return "budget_lf_010000_" in budget_name
    if lf_budget_tag == "lf1e5":
        return "budget_lf_100000_" in budget_name
    if lf_budget_tag == "lf1k":
        return "budget_lf_001000_" in budget_name
    raise ValueError(f"Unsupported lf_budget_tag: {lf_budget_tag}")


def _append_acc(
    acc_rows_raw: list[dict[str, Any]],
    method: str,
    budget: str,
    metric_name: str,
    value: float,
    obs_id: int,
    seed: int,
) -> None:
    acc_rows_raw.append(
        {
            "method": method,
            "budget": budget,
            "parameter": "overall",
            "metric_name": metric_name,
            "value": value,
            "obs_id": obs_id,
            "seed": seed,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute benchmark metrics for one experiment (unified)")
    ap.add_argument("--experiment", type=str, required=True, help="proliferating_vertex (alias: prism_vertex)")
    ap.add_argument(
        "--methods",
        type=str,
        default=PAPER_METHODS_STR,
        help="Comma-separated paper methods",
    )
    ap.add_argument("--results_root", type=str, default="data")
    ap.add_argument("--comparison_subdir", type=str, default=None)
    ap.add_argument(
        "--lf_budget_tag",
        type=str,
        default="all",
        choices=["all", "lf1e4", "lf1e5", "lf1k"],
        help="LF slice for metric generation.",
    )
    ap.add_argument("--max_metric_samples", type=int, default=4000)
    ap.add_argument(
        "--c2st_target_threshold",
        type=float,
        default=0.55,
        help="For metrics_cost_to_target.csv (C2ST below this).",
    )
    ap.add_argument(
        "--lf_cost_per_hf",
        type=float,
        default=0.1,
        help=(
            "Relative LF cost per HF unit cost (c_hf fixed at 1). "
            "Default 0.1 matches the tissue LF/HF workload ratio used in the paper."
        ),
    )
    ap.add_argument(
        "--calibration_only",
        action="store_true",
        help="Calibration metrics for MF-SBI @ lf_DT05 10k LF / 500 HF.",
    )
    ap.add_argument(
        "--mf_sbi_per_lf",
        action="store_true",
        help=(
            "Paper method-comparison LF matrix: mf_sbi across all nine LF fidelities "
            f"(writes {MAIN_METHOD_COMPARISON_DIR}/; method column = lf_id)."
        ),
    )
    args = ap.parse_args()

    if args.experiment == "prism_vertex":
        args.experiment = "proliferating_vertex"
    if args.experiment != "proliferating_vertex":
        raise ValueError("Only --experiment proliferating_vertex is supported in this release.")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.mf_sbi_per_lf:
        if args.experiment not in ("proliferating_vertex",):
            raise ValueError("--mf_sbi_per_lf requires --experiment proliferating_vertex")
        methods = ["mf_sbi"]
    elif args.calibration_only:
        if args.experiment not in ("proliferating_vertex",):
            raise ValueError("--calibration_only requires --experiment proliferating_vertex")
        methods = ["mf_sbi"]

    cfg = load_benchmark_config(experiment_id=args.experiment, methods=methods)
    exp_cfg = cfg["experiment"]
    exp_dir = Path(args.results_root)

    use_reference = bool(exp_cfg.get("use_reference_posterior", False))
    comparison_subdir = args.comparison_subdir
    if comparison_subdir is None:
        if args.mf_sbi_per_lf:
            comparison_subdir = MAIN_METHOD_COMPARISON_DIR
        elif args.calibration_only:
            comparison_subdir = CALIBRATION_COMPARISON_SUBDIR
        else:
            comparison_subdir = MAIN_METHOD_COMPARISON_DIR

    comparison_dir = exp_dir / comparison_subdir
    comparison_dir.mkdir(parents=True, exist_ok=True)
    if args.mf_sbi_per_lf:
        for stale in comparison_dir.glob("metrics_*.csv"):
            stale.unlink(missing_ok=True)
        (comparison_dir / "fig4_run_index.csv").unlink(missing_ok=True)
    if args.calibration_only:
        for stale in comparison_dir.glob("metrics_*.csv"):
            stale.unlink(missing_ok=True)

    unit_costs = _effective_unit_costs_hf1(
        args.experiment, exp_dir, lf_cost_per_hf=float(args.lf_cost_per_hf)
    )
    if unit_costs is not None:
        print(
            f"Using cost model for metrics: c_hf={unit_costs[0]}, c_lf={unit_costs[1]} "
            f"(total_cost = c_hf*n_hf + c_lf*n_lf per run)"
        )

    acc_rows_raw: list[dict[str, Any]] = []
    eff_rows_raw: list[dict[str, Any]] = []
    uq_rows_raw: list[dict[str, Any]] = []
    cal_rows_raw: list[dict[str, Any]] = []
    sbc_rows_raw: list[dict[str, Any]] = []
    validity_rows_raw: list[dict[str, Any]] = []
    sbc_ranks_long: list[dict[str, Any]] = []

    proliferating_vertex_comparison = (
        args.experiment == "proliferating_vertex"
        and not args.mf_sbi_per_lf
        and not args.calibration_only
            )
    fig4_manifest = load_fig4_manifest(PROJECT_ROOT) if args.mf_sbi_per_lf else []
    fig4_manifest_map = manifest_by_results_path(fig4_manifest)
    fig4_run_index: list[dict[str, Any]] = []

    run_iter: list[tuple[Path, str, str, int, int]] = []
    if args.calibration_only:
        budget_label = "budget_lf_010000_hf_000500"
        for run_dir, method, obs_id, seed in iter_calibration_run_dirs(exp_dir):
            run_iter.append((run_dir, method, budget_label, obs_id, seed))
    elif args.mf_sbi_per_lf:
        method_dir = method_results_dir(exp_dir, "mf_sbi")
        for run_dir, lf_id, budget_label in iter_fig4_mf_sbi_runs(method_dir):
            obs_id = int(run_dir.parent.name.replace("obs_", ""))
            seed = int(run_dir.name.replace("seed_", ""))
            run_iter.append((run_dir, lf_id, budget_label, obs_id, seed))
            rp = run_dir_results_path(exp_dir, run_dir)
            mrow = fig4_manifest_map.get(rp, {})
            fig4_run_index.append(
                {
                    "lf_id": lf_id,
                    "budget": budget_label,
                    "nominal_lf_budget": manifest_nominal_lf_budget(mrow) if mrow else "",
                    "cost_hf_equiv": mrow.get("cost_hf_equiv", ""),
                    "results_path": rp,
                    "obs_id": obs_id,
                    "seed": seed,
                }
            )
    else:
        method_dirs = sorted([d for d in posterior_runs_dir(exp_dir).iterdir() if d.is_dir()])
        for method_dir in method_dirs:
            method = method_name_from_dir_name(method_dir.name)
            if method not in methods:
                continue
            if proliferating_vertex_comparison:
                run_pairs: Iterable[tuple[Path, str]] = iter_paper_method_comparison_run_dirs(method, method_dir)
            else:
                run_pairs = (
                    (run_dir, budget_dir.name)
                    for budget_dir in sorted(method_dir.iterdir())
                    if budget_dir.is_dir() and budget_dir.name.startswith("budget_")
                    for run_dir in sorted(budget_dir.glob("obs_*/seed_*"))
                    if run_dir.is_dir()
                )
            for run_dir, budget_key in run_pairs:
                budget_label = budget_dir_name(budget_key)
                if not _budget_matches_lf_tag(budget_label, args.lf_budget_tag, method):
                    continue
                obs_id = int(run_dir.parent.name.replace("obs_", ""))
                seed = int(run_dir.name.replace("seed_", ""))
                run_iter.append((run_dir, method, budget_label, obs_id, seed))

    for run_dir, method, budget_label, obs_id, seed in run_iter:
        post = run_dir / "posterior_samples.csv"
        cost = run_dir / "cost_tracking.csv"
        if not post.exists():
            continue
        approx, theta_cols, log_w = _load_samples(post)
        rng = np.random.default_rng(seed + obs_id * 101)
        if len(approx) > args.max_metric_samples:
            approx_eval = approx[rng.choice(len(approx), size=args.max_metric_samples, replace=False)]
            log_w_eval = None
        else:
            approx_eval = approx
            log_w_eval = log_w

        ref_eval = None
        ref_path = _load_reference(exp_dir, obs_id, use_reference)
        if ref_path is not None and ref_path.exists():
            ref, _, _ = _load_samples(ref_path)
            if len(ref) > args.max_metric_samples:
                ref_eval = ref[rng.choice(len(ref), size=args.max_metric_samples, replace=False)]
            else:
                ref_eval = ref

        theta_true = _load_theta_true(exp_dir, obs_id)
        if theta_true is None:
            p_theta = exp_dir / "observations" / f"theta_{obs_id:02d}.npy"
            if p_theta.exists():
                theta_true = np.asarray(np.load(p_theta), dtype=float).reshape(-1)
        if theta_true is not None:
            _append_acc(acc_rows_raw, method, budget_label, "nrmse", _nrmse(approx, theta_true), obs_id, seed)
            diag_path = run_dir / "diagnostics.json"
            if diag_path.exists():
                try:
                    diag = json.loads(diag_path.read_text())
                    v = diag.get("nltp_exact")
                    if v is not None:
                        fv = float(v)
                        if np.isfinite(fv):
                            _append_acc(acc_rows_raw, method, budget_label, "nltp_exact", fv, obs_id, seed)
                except Exception:
                    pass
            _append_acc(
                acc_rows_raw,
                method,
                budget_label,
                "nltp_gaussian",
                _nltp_gaussian(approx, theta_true),
                obs_id,
                seed,
            )
            _append_acc(
                acc_rows_raw,
                method,
                budget_label,
                "nltp_kde",
                bu.nltp_kde(approx, theta_true),
                obs_id,
                seed,
            )
            _append_acc(
                acc_rows_raw,
                method,
                budget_label,
                "crps_mean",
                bu.crps_mean_multivariate(approx, theta_true),
                obs_id,
                seed,
            )
            _append_acc(
                acc_rows_raw,
                method,
                budget_label,
                "tarp_coverage_error",
                bu.tarp_coverage_error_mean(approx, theta_true),
                obs_id,
                seed,
            )

            for j, c in enumerate(theta_cols):
                vals = approx[:, j]
                uq_rows_raw.append(
                    {
                        "method": method,
                        "budget": budget_label,
                        "parameter": c,
                        "metric_name": "posterior_std",
                        "value": float(np.std(vals)),
                        "obs_id": obs_id,
                        "seed": seed,
                    }
                )
                uq_rows_raw.append(
                    {
                        "method": method,
                        "budget": budget_label,
                        "parameter": c,
                        "metric_name": "posterior_var",
                        "value": float(np.var(vals)),
                        "obs_id": obs_id,
                        "seed": seed,
                    }
                )
                for nl in (0.5, 0.683, 0.9, 0.95, 0.99):
                    alpha = (1.0 - nl) / 2.0
                    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
                    covered = 1.0 if theta_true[j] >= lo and theta_true[j] <= hi else 0.0
                    if nl in (0.5, 0.95):
                        uq_rows_raw.append(
                            {
                                "method": method,
                                "budget": budget_label,
                                "parameter": c,
                                "metric_name": f"ci_width_{nl}",
                                "value": float(hi - lo),
                                "obs_id": obs_id,
                                "seed": seed,
                            }
                        )
                    cal_rows_raw.append(
                        {
                            "method": method,
                            "budget": budget_label,
                            "parameter": c,
                            "metric_name": f"coverage_{nl}",
                            "value": covered,
                            "obs_id": obs_id,
                            "seed": seed,
                        }
                    )
            for rr in _sbc_rank_stats(approx, theta_true):
                sbc_rows_raw.append(
                    {
                        "method": method,
                        "budget": budget_label,
                        "parameter_index": rr["parameter_index"],
                        "metric_name": "sbc_rank_normalized",
                        "value": rr["rank_normalized"],
                        "obs_id": obs_id,
                        "seed": seed,
                    }
                )
                sbc_ranks_long.append(
                    {
                        "method": method,
                        "budget": budget_label,
                        "obs_id": obs_id,
                        "seed": seed,
                        "parameter_index": rr["parameter_index"],
                        "rank_normalized": rr["rank_normalized"],
                    }
                )

            prior_draw = None
            try:
                from prism_vertex.experiments import get_task

                task_cfg = {}
                task = get_task(args.experiment, config=task_cfg)
                prior_draw = task.sample_prior(2000, seed=seed + 7 * obs_id)
            except Exception:
                prior_draw = None
            if prior_draw is not None and prior_draw.shape[1] == approx.shape[1]:
                bounds = [
                    (float(np.min(prior_draw[:, j])), float(np.max(prior_draw[:, j])))
                    for j in range(prior_draw.shape[1])
                ]
                validity_rows_raw.append(
                    {
                        "method": method,
                        "budget": budget_label,
                        "parameter": "overall",
                        "metric_name": "plausibility_rate",
                        "value": _plausibility_rate(approx, bounds),
                        "obs_id": obs_id,
                        "seed": seed,
                    }
                )


        if cost.exists():
            c = _adjust_cost_tracking_row(_load_cost(cost), unit_costs)
            for key in ("n_hf_calls", "n_lf_calls", "total_cost", "wallclock_seconds_total", "train_seconds"):
                if key in c:
                    eff_rows_raw.append(
                        {
                            "method": method,
                            "budget": budget_label,
                            "metric_name": key,
                            "value": float(c[key]),
                            "obs_id": obs_id,
                            "seed": seed,
                        }
                    )


    # --- write CSVs ---
    if acc_rows_raw:
        df = pd.DataFrame(acc_rows_raw)
        acc = _aggregate(df, ["method", "budget", "parameter", "metric_name"])
        acc.to_csv(comparison_dir / "accuracy_metrics.csv", index=False)
        conv = acc[acc["parameter"] == "overall"].pivot_table(
            index=["method", "budget", "n_replicates"],
            columns="metric_name",
            values=["metric_value_mean", "metric_value_std"],
            aggfunc="first",
        )
        conv.columns = [f"{b}_{a}" for a, b in conv.columns]
        conv = conv.reset_index()
        conv.to_csv(comparison_dir / "metrics_convergence.csv", index=False)


    if eff_rows_raw:
        df = pd.DataFrame(eff_rows_raw)
        eff = _aggregate(df, ["method", "budget", "metric_name"])
        eff.to_csv(comparison_dir / "metrics_efficiency_long.csv", index=False)
        eff_wide = eff.pivot_table(
            index=["method", "budget", "n_replicates"],
            columns="metric_name",
            values=["metric_value_mean", "metric_value_std"],
            aggfunc="first",
        )
        eff_wide.columns = [f"{b}_{a}" for a, b in eff_wide.columns]
        eff_wide = eff_wide.reset_index()
        eff_wide.to_csv(comparison_dir / "efficiency_metrics.csv", index=False)

    if uq_rows_raw:
        df = pd.DataFrame(uq_rows_raw)
        uq = _aggregate(df, ["method", "budget", "parameter", "metric_name"])
        uq.to_csv(comparison_dir / "metrics_uq.csv", index=False)

    if cal_rows_raw:
        df = pd.DataFrame(cal_rows_raw)
        cal = _aggregate(df, ["method", "budget", "parameter", "metric_name"])
        cal.to_csv(comparison_dir / "metrics_calibration.csv", index=False)

    if sbc_rows_raw:
        df = pd.DataFrame(sbc_rows_raw)
        sbc = _aggregate(df, ["method", "budget", "parameter_index", "metric_name"])
        ks_rows: list[dict[str, Any]] = []
        for (method, budget), g in df.groupby(["method", "budget"]):
            vals = g["value"].to_numpy(dtype=float)
            if len(vals) >= 3:
                ks = kstest(vals, "uniform")
                ks_rows.append(
                    {
                        "method": method,
                        "budget": budget,
                        "parameter_index": -1,
                        "metric_name": "sbc_rank_uniform_ks_pvalue_all_dims_pooled",
                        "metric_value_mean": float(ks.pvalue),
                        "metric_value_std": 0.0,
                        "metric_value_median": float(ks.pvalue),
                        "n_replicates": len(vals),
                    }
                )
        if ks_rows:
            sbc = pd.concat([sbc, pd.DataFrame(ks_rows)], ignore_index=True)
        sbc.to_csv(comparison_dir / "metrics_sbc.csv", index=False)

    if sbc_ranks_long:
        df_long = pd.DataFrame(sbc_ranks_long)
        df_long.to_csv(comparison_dir / "sbc_rank_metrics.csv", index=False)
        pooled = bu.dataframe_sbc_pooled_ks(df_long)
        if not pooled.empty:
            pooled.to_csv(comparison_dir / "metrics_sbc_pooled.csv", index=False)

    if validity_rows_raw:
        df = pd.DataFrame(validity_rows_raw)
        validity = _aggregate(df, ["method", "budget", "parameter", "metric_name"])
        validity.to_csv(comparison_dir / "metrics_validity.csv", index=False)



    if fig4_run_index:
        # Keep only runs that produced a posterior (failed HPC jobs may leave empty dirs).
        completed = {run_dir_results_path(exp_dir, rd) for rd, _, _, _, _ in run_iter if (rd / "posterior_samples.csv").is_file()}
        fig4_run_index = [r for r in fig4_run_index if r.get("results_path") in completed]
        if fig4_run_index:
            pd.DataFrame(fig4_run_index).to_csv(comparison_dir / "fig4_run_index.csv", index=False)

    if args.mf_sbi_per_lf:
        n_done = len({rd for rd, _, _, _, _ in run_iter if (rd / "posterior_samples.csv").is_file()})
        n_total = len(run_iter)
        print(f"Paper MF-SBI LF matrix: {n_done}/{n_total} runs with posterior_samples.csv -> {comparison_dir}")
    if args.calibration_only:
        n_done = len({rd for rd, _, _, _, _ in run_iter if (rd / "posterior_samples.csv").is_file()})
        n_total = len(run_iter)
        print(f"Paper calibration: {n_done}/{n_total} runs -> {comparison_dir}")


if __name__ == "__main__":
    main()
