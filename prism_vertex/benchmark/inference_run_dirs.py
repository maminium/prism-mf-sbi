"""Discover inference run folders (flat budget_* or lf_*/budget_* layouts)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

PAPER_METHOD_COMPARISON_LF = "lf_DT10"
PAPER_MF_COMPARISON_METHODS = frozenset({"mf_sbi"})
PAPER_HF_COMPARISON_METHODS = frozenset({"hf_smc_abc", "hf_npe", "hf_snpe"})


def is_paper_experiment_dir(exp_dir: Path) -> bool:
    return exp_dir.name == "data" or "prism_vertex" in exp_dir.name.lower()


def budget_dir_name(budget_key: str) -> str:
    if "/" in budget_key:
        return budget_key.rsplit("/", 1)[-1]
    return budget_key


def iter_method_run_dirs(
    method_dir: Path,
    *,
    lf_fidelity: str | None = None,
) -> Iterator[tuple[Path, str]]:
    if not method_dir.is_dir():
        return
    for child in sorted(method_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("budget_"):
            for run_dir in sorted(child.glob("obs_*/seed_*")):
                if run_dir.is_dir():
                    yield run_dir, child.name
        elif child.name.startswith("lf_"):
            if lf_fidelity is not None and child.name != lf_fidelity:
                continue
            for bd in sorted(child.glob("budget_*")):
                if not bd.is_dir():
                    continue
                budget_key = f"{child.name}/{bd.name}"
                for run_dir in sorted(bd.glob("obs_*/seed_*")):
                    if run_dir.is_dir():
                        yield run_dir, budget_key


def iter_paper_method_comparison_run_dirs(method: str, method_dir: Path) -> Iterator[tuple[Path, str]]:
    if method in PAPER_MF_COMPARISON_METHODS:
        lf_dir = method_dir / PAPER_METHOD_COMPARISON_LF
        if lf_dir.is_dir():
            yield from iter_method_run_dirs(method_dir, lf_fidelity=PAPER_METHOD_COMPARISON_LF)
            return
    elif method not in PAPER_HF_COMPARISON_METHODS:
        return
    for child in sorted(method_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("budget_"):
            continue
        for run_dir in sorted(child.glob("obs_*/seed_*")):
            if run_dir.is_dir():
                yield run_dir, child.name


def method_name_from_run_dir(run_dir: Path) -> str:
    from prism_vertex.benchmark.comparison_paths import method_name_from_dir_name

    for p in run_dir.parents:
        if p.parent.name == "posterior_runs":
            return method_name_from_dir_name(p.name)
    raise ValueError(f"Cannot infer method from {run_dir}")


def budget_key_from_run_dir(run_dir: Path) -> str:
    obs = run_dir.parent
    budget_dir = obs.parent
    if not budget_dir.name.startswith("budget_"):
        raise ValueError(f"Cannot infer budget key from {run_dir}")
    lf_dir = budget_dir.parent
    if lf_dir.name.startswith("lf_"):
        return f"{lf_dir.name}/{budget_dir.name}"
    return budget_dir.name
