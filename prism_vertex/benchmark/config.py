from __future__ import annotations

from pathlib import Path
from typing import Any

from prism_vertex.config.loaders import deep_merge, load_yaml


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _default_experiment_path(experiment_id: str) -> Path:
    cand = Path("configs") / experiment_id / "experiment.yaml"
    if cand.is_file():
        return cand
    return Path("configs/proliferating_vertex/experiment.yaml")


def load_benchmark_config(
    *,
    experiment_id: str,
    defaults_path: str | Path = "configs/defaults.yaml",
    experiment_path: str | Path | None = None,
    method_dir: str | Path = "configs/methods",
    methods: list[str],
) -> dict[str, Any]:
    defaults = load_yaml(defaults_path)
    exp_path = Path(experiment_path) if experiment_path is not None else _default_experiment_path(experiment_id)
    exp_cfg = load_yaml(exp_path)
    if exp_cfg.get("experiment_id") not in (None, experiment_id):
        exp_cfg = dict(exp_cfg)
        exp_cfg["experiment_id"] = experiment_id

    method_cfgs: dict[str, dict[str, Any]] = {}
    method_defaults = defaults.get("method_defaults", {})
    for method in methods:
        base = method_defaults.get(method, {})
        override_path = Path(method_dir) / f"{method}.yaml"
        override = load_yaml(override_path) if override_path.exists() else {}
        method_cfgs[method] = deep_merge(base, override)

    merged_exp = dict(exp_cfg)
    merged_exp["budgets_hf"] = [int(v) for v in _as_list(merged_exp.get("budgets_hf", []))]
    merged_exp["budgets_lf"] = [int(v) for v in _as_list(merged_exp.get("budgets_lf", []))]
    merged_exp["n_obs"] = int(merged_exp.get("n_obs", 1))
    merged_exp["n_seeds"] = int(merged_exp.get("n_seeds", 1))
    merged_exp["use_reference_posterior"] = bool(merged_exp.get("use_reference_posterior", False))
    return {"experiment": merged_exp, "methods": method_cfgs}
