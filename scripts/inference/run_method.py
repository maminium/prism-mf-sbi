from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", "--experiment", dest="exp", required=True, help="proliferating_vertex | nonproliferating_vertex")
    ap.add_argument(
        "--method",
        required=True,
        choices=["hf_smc_abc", "hf_npe", "hf_snpe", "mf_sbi"],
    )
    ap.add_argument("--obs_id", type=int, default=0)
    ap.add_argument("--obs_ids", default=None, help="comma-separated obs ids (overrides --obs_id)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget_hf", type=int, required=True)
    ap.add_argument("--budget_lf", type=int, default=0)
    ap.add_argument("--num_posterior_samples", type=int, default=2000)
    ap.add_argument("--method_config", default=None)
    ap.add_argument("--experiment_config", default=None)
    ap.add_argument("--results_root", default="data")
    args = ap.parse_args()

    from prism_vertex.config.loaders import load_yaml
    from prism_vertex.runners.run_method import run_method, run_method_for_obs_ids

    method_cfg = load_yaml(args.method_config) if args.method_config else {}
    exp_cfg = load_yaml(args.experiment_config) if args.experiment_config else {}

    if args.obs_ids:
        obs_ids = [int(v.strip()) for v in str(args.obs_ids).split(",") if v.strip()]
        run_dirs = run_method_for_obs_ids(
            experiment_id=args.exp,
            method=args.method,
            obs_ids=obs_ids,
            seed=args.seed,
            budget_hf=args.budget_hf,
            budget_lf=args.budget_lf,
            num_posterior_samples=args.num_posterior_samples,
            method_config=method_cfg,
            experiment_config=exp_cfg,
            results_root=args.results_root,
        )
        for d in run_dirs:
            print(str(d))
    else:
        run_dir = run_method(
            experiment_id=args.exp,
            method=args.method,
            obs_id=args.obs_id,
            seed=args.seed,
            budget_hf=args.budget_hf,
            budget_lf=args.budget_lf,
            num_posterior_samples=args.num_posterior_samples,
            method_config=method_cfg,
            experiment_config=exp_cfg,
            results_root=args.results_root,
        )
        print(str(run_dir))


if __name__ == "__main__":
    main()

