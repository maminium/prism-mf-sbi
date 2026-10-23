"""Paper data paths for method comparison results and posterior runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

LF_ORDER: tuple[str, ...] = (
    "lf_T70",
    "lf_T140",
    "lf_T350",
    "lf_DT02",
    "lf_DT05",
    "lf_DT10",
    "lf_R1",
    "lf_R2",
    "lf_R5",
)

MAIN_METHOD_COMPARISON_DIR = Path("results/main_method_comparison")
POSTERIOR_RUNS_DIR = Path("results/posterior_runs")
METHOD_DIR_NAMES: dict[str, str] = {
    "hf_smc_abc": "hf_smc_abc",
    "hf_npe": "hf_npe",
    "hf_snpe": "hf_snpe",
    "mf_sbi": "mf_sbi",
    }


def results_dir(results_root: Path) -> Path:
    root = Path(results_root)
    candidates = [
        root,
        root / "proliferating_vertex",
        root / "data" / "proliferating_vertex",
    ]
    # If caller passed repo root or data/, prefer proliferating_vertex.
    for cand in candidates:
        if (cand / "observations").is_dir() and (cand / "results").is_dir():
            return cand
    raise FileNotFoundError(
        f"Expected paper data directory with observations/ and results/: {root}"
    )


def fig4_comparison_dir(exp_dir: Path) -> Path:
    """Main method-comparison metric directory (paper figs 3-4)."""
    return exp_dir / MAIN_METHOD_COMPARISON_DIR


def method_dir_name(method: str) -> str:
    return METHOD_DIR_NAMES.get(method, method)


def method_name_from_dir_name(dirname: str) -> str:
    inverse = {v: k for k, v in METHOD_DIR_NAMES.items()}
    return inverse.get(dirname, dirname)


def posterior_runs_dir(exp_dir: Path) -> Path:
    return exp_dir / POSTERIOR_RUNS_DIR


def method_results_dir(exp_dir: Path, method: str) -> Path:
    return posterior_runs_dir(exp_dir) / method_dir_name(method)


def load_fig4_manifest(repo_root: Path | None = None) -> list[dict[str, Any]]:
    return []


def manifest_by_results_path(manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in manifest:
        rp = str(row.get("results_path", "")).strip()
        if rp:
            out[rp] = row
    return out


def manifest_nominal_lf_budget(row: dict[str, Any]) -> int:
    nom = row.get("budget_lf_nominal_list") or []
    if nom:
        return int(nom[0])
    return int(row.get("budget_lf_actual", 0))


def iter_fig4_mf_sbi_runs(
    method_dir: Path,
    *,
    lf_order: tuple[str, ...] = LF_ORDER,
) -> Iterator[tuple[Path, str, str]]:
    """
    Yield ``(run_dir, lf_id, budget_label)`` for Fig.4 LF trees only.

    ``budget_label`` is the ``budget_*`` folder name (no ``lf_*/`` prefix).
    Skips flat ``budget_*`` dirs at the MF-SBI root.
    """
    if not method_dir.is_dir():
        return
    for lf_id in lf_order:
        lf_dir = method_dir / lf_id
        if not lf_dir.is_dir():
            continue
        for bd in sorted(lf_dir.glob("budget_*")):
            if not bd.is_dir():
                continue
            budget_label = bd.name
            for run_dir in sorted(bd.glob("obs_*/seed_*")):
                if run_dir.is_dir():
                    yield run_dir, lf_id, budget_label


def run_dir_results_path(exp_dir: Path, run_dir: Path) -> str:
    rel = run_dir.relative_to(exp_dir).as_posix()
    return rel
