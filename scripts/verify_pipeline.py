#!/usr/bin/env python3
"""End-to-end smoke verification for prism-mf-sbi (sims, methods, figures)."""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Fast defaults for verification (do not affect paper shipped configs).
os.environ.setdefault("PRISM_VERTEX_HF_K_MEAN", "1")
os.environ.setdefault("PRISM_VERTEX_K_HF_REPLICATES", "1")
os.environ.setdefault("PRISM_VERTEX_CLEANUP", "1")
os.environ.setdefault("PRISM_VERTEX_ENABLE_STALL_WATCHDOG", "0")
os.environ.setdefault("PRISM_VERTEX_TIMEOUT_SECONDS", "600")
os.environ.setdefault("PRISM_VERTEX_MAX_SIM_RETRIES", "1")

PASS = 0
FAIL = 0
SKIP = 0


def _ok(name: str, detail: str = "") -> None:
    global PASS
    PASS += 1
    print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))


def _fail(name: str, err: BaseException | str) -> None:
    global FAIL
    FAIL += 1
    print(f"[FAIL] {name}: {err}")


def _skip(name: str, reason: str) -> None:
    global SKIP
    SKIP += 1
    print(f"[SKIP] {name} — {reason}")


def check_imports() -> None:
    try:
        from prism_vertex.experiments import get_task
        from prism_vertex.inference.hf_npe import run_hf_npe
        from prism_vertex.inference.hf_snpe import run_hf_snpe
        from prism_vertex.inference.hf_smc_abc import run_hf_smc_abc
        from prism_vertex.inference.mf_sbi import run_mf_sbi
        from prism_vertex.benchmark.config import load_benchmark_config
        from prism_vertex.benchmark.registry import PAPER_METHODS
        assert set(PAPER_METHODS) == {"hf_smc_abc", "hf_npe", "hf_snpe", "mf_sbi"}
        cfg = load_benchmark_config(experiment_id="proliferating_vertex", methods=list(PAPER_METHODS))
        assert cfg["experiment"]["budgets_lf"] == [500, 1000, 5000, 10000]
        _ok("imports+configs", f"methods={list(PAPER_METHODS)}")
    except Exception as exc:
        _fail("imports+configs", exc)
        traceback.print_exc()


def check_tasks() -> None:
    try:
        from prism_vertex.experiments import get_task

        prolif = get_task("proliferating_vertex")
        non = get_task("nonproliferating_vertex")
        assert prolif.k_hf_replicates >= 1
        assert non.k_hf_replicates >= 1
        assert prolif.use_k_mean_for_hf
        assert non.use_k_mean_for_hf
        th = prolif.sample_prior(4, seed=0)
        assert th.shape == (4, 2)
        th2 = non.sample_prior(4, seed=0)
        assert th2.shape == (4, 3)
        x0 = prolif.get_observation(0)
        assert np.asarray(x0).size == 6
        x1 = non.get_observation(0)
        assert np.asarray(x1).size in (5, 6)
        _ok("tasks", f"obs dims prolif={np.asarray(x0).size} non={np.asarray(x1).size}")
    except Exception as exc:
        _fail("tasks", exc)
        traceback.print_exc()


def check_fidelities_replicates() -> None:
    try:
        from prism_vertex.simulator.fidelities import LF_R2, LF_R5, get_fidelity
        from prism_vertex.simulator.nonproliferating.fidelities import (
            LF_R2 as NR2,
            LF_R5 as NR5,
            get_fidelity as get_f_np,
        )

        assert LF_R2.n_replicates == 2 and LF_R5.n_replicates == 5
        assert NR2.n_replicates == 2 and NR5.n_replicates == 5
        assert get_fidelity("lf_DT05").id == "lf_DT05"
        assert get_f_np("lf_T05").id == "lf_T05"
        _ok("fidelities R2/R5")
    except Exception as exc:
        _fail("fidelities R2/R5", exc)


def check_prolif_sim() -> None:
    try:
        from prism_vertex.simulator.chaste_runner import run_chaste, ChasteRunError
        from prism_vertex.simulator.fidelities import FidelitySpec

        exe, _, _ = __import__(
            "prism_vertex.simulator.chaste_runner", fromlist=["_chaste_paths"]
        )._chaste_paths()
        if not exe.exists():
            _skip("prolif Chaste sim", f"missing exe {exe}")
            return
        spec = FidelitySpec(
            id="lf_smoke",
            description="verification short run",
            end_time=70.0,
            dt=0.01,
            n_replicates=1,
        )
        theta = np.array([0.12, 0.04], dtype=float)
        result = run_chaste(theta, sim_seed=7, spec=spec, cleanup_chaste_output=True)
        assert result.summary_6d.shape == (6,)
        assert np.isfinite(result.summary_6d).all()
        assert float(np.abs(result.summary_6d).sum()) > 0.0
        _ok("prolif Chaste sim", f"x={np.round(result.summary_6d, 3).tolist()} exe={exe.name}")
    except Exception as exc:
        _fail("prolif Chaste sim", exc)
        traceback.print_exc()


def check_prolif_replicate_mean() -> None:
    try:
        from prism_vertex.simulator.simulator import TissueVertexSimulator
        from prism_vertex.simulator.chaste_runner import _chaste_paths
        from prism_vertex.simulator.fidelities import FidelitySpec, ALL_FIDELITIES

        exe, _, _ = _chaste_paths()
        if not exe.exists():
            _skip("prolif lf_R2 mean", f"missing exe {exe}")
            return
        # Temporarily register a short R2-like fidelity via monkeypatch of get_fidelity path
        short = FidelitySpec(
            id="lf_R2",
            description="short R2 for verify",
            end_time=70.0,
            dt=0.01,
            n_replicates=2,
        )
        ALL_FIDELITIES["lf_R2"] = short
        sim = TissueVertexSimulator()
        x = sim.run_lf(np.array([0.12, 0.04]), sim_seed=11, fidelity_id="lf_R2")
        assert x.shape == (6,) and np.isfinite(x).all()
        _ok("prolif lf_R2 mean-of-2", f"x={np.round(x, 3).tolist()}")
    except Exception as exc:
        _fail("prolif lf_R2 mean-of-2", exc)
        traceback.print_exc()


def check_nonprolif_sim() -> None:
    try:
        # Do not inherit prolif PRISM_VERTEX_CHASTE_EXE.
        os.environ.pop("PRISM_VERTEX_CHASTE_EXE", None)
        from prism_vertex.simulator.nonproliferating.chaste_runner import run_chaste, _chaste_paths
        from prism_vertex.simulator.nonproliferating.fidelities import LF_T05
        from prism_vertex.simulator.nonproliferating.constants import THETA_TRUE

        exe, _, _ = _chaste_paths()
        if not exe.exists():
            _skip("nonprolif Chaste sim", f"missing exe {exe}")
            return
        result = run_chaste(THETA_TRUE, sim_seed=0, spec=LF_T05, cleanup_chaste_output=True)
        assert result.summary.size >= 5
        assert np.isfinite(result.summary).all()
        _ok("nonprolif Chaste sim", f"dim={result.summary.size} exe={exe.name}")
    except Exception as exc:
        _fail("nonprolif Chaste sim", exc)
        traceback.print_exc()


def check_methods_smoke() -> None:
    """Tiny budgets via run_method (paper entrypoint) with short Chaste horizons."""
    import tempfile
    from prism_vertex.runners.run_method import run_method
    from prism_vertex.simulator.chaste_runner import _chaste_paths
    from prism_vertex.simulator import fidelities as fid

    if not os.environ.get("PRISM_VERTEX_CHASTE_EXE"):
        for cand in (
            Path.home() / "chaste/build/projects/ProliferatingVertex/test/TestPaperCommandLineSpeedSimulation",
            Path.home() / "chaste/build/projects/ProliferatingVertex/test/TestPaperCommandLineSpeedSimulation",
        ):
            if cand.exists():
                os.environ["PRISM_VERTEX_CHASTE_EXE"] = str(cand)
                break

    exe, _, _ = _chaste_paths()
    if not exe.exists():
        _skip("methods smoke", f"missing prolif exe {exe}")
        return

    os.environ["PRISM_VERTEX_LF_FIDELITY"] = "lf_T70"
    os.environ["PRISM_VERTEX_HF_STRICT_ANY_FAIL"] = "0"
    os.environ["PRISM_VERTEX_MAX_SIM_RETRIES"] = "2"
    os.environ["PRISM_VERTEX_K_HF_REPLICATES"] = "1"
    os.environ["PRISM_VERTEX_HF_K_MEAN"] = "1"
    os.environ["PRISM_VERTEX_TIMEOUT_SECONDS"] = "300"

    short = {}
    for key in list(fid.ALL_FIDELITIES):
        s = fid.ALL_FIDELITIES[key]
        short[key] = fid.FidelitySpec(
            id=s.id,
            description=s.description,
            end_time=70.0,
            dt=0.01,
            n_replicates=1 if s.id.startswith("hf") else min(s.n_replicates, 1),
        )
    fid.ALL_FIDELITIES.update(short)
    fid.HF_SINGLE = short["hf_single"]
    fid.LF_T70 = short["lf_T70"]

    smoke_cfg = {
        "inference_smoke": True,
        "num_rounds": 1,
        "ensemble_size": 1,
        "lf_max_num_epochs": 2,
        "hf_max_num_epochs": 2,
        "max_num_epochs": 2,
        "lf_training_batch_size": 32,
        "hf_training_batch_size": 32,
        "training_batch_size": 32,
        "simulation_batch_size": 8,
        "neural_net": "maf",
        "hidden_features": 10,
        "use_snpe_proposal": True,
        "lf_warmup_fraction": 1.0,
        "lf_active_fraction": 0.0,
        "active_fraction_lf": 0.0,
        "use_active_lf": False,
        "population_size": 6,
        "epsilon_schedule": [50.0, 20.0],
    }

    with tempfile.TemporaryDirectory(prefix="prism_vertex_verify_") as tmp:
        results_root = Path(tmp)
        (results_root / "proliferating_vertex").mkdir()
        # Point task observations at shipped data.
        os.environ.setdefault(
            "PRISM_VERTEX_OBS_DIR",
            str(ROOT / "data" / "proliferating_vertex" / "observations"),
        )
        for method, budget_hf, budget_lf in (
            ("hf_npe", 6, 0),
            ("hf_snpe", 6, 0),
            ("hf_smc_abc", 8, 0),
            ("mf_sbi", 4, 8),
        ):
            try:
                run_dir = run_method(
                    experiment_id="proliferating_vertex",
                    method=method,
                    obs_id=0,
                    seed=0,
                    budget_hf=budget_hf,
                    budget_lf=budget_lf,
                    num_posterior_samples=40,
                    method_config=smoke_cfg,
                    experiment_config={
                        "observations_dir": str(ROOT / "data" / "proliferating_vertex" / "observations"),
                    },
                    results_root=results_root / "proliferating_vertex",
                )
                samples = run_dir / "posterior_samples.csv"
                assert samples.is_file(), f"missing {samples}"
                _ok(f"method {method}", f"run_dir={run_dir.name}")
            except Exception as exc:
                _fail(f"method {method}", exc)
                traceback.print_exc()


def check_figures() -> None:
    try:
        from scripts.figures.plot_all import main as plot_main
        # plot_all uses argparse from sys.argv
        old = sys.argv
        sys.argv = ["plot_all", "--figures", "1,2,3,4,5,6,7,8"]
        try:
            plot_main()
        finally:
            sys.argv = old
        missing = [f"fig{i}.png" for i in range(1, 9) if not (ROOT / "figures" / f"fig{i}.png").is_file()]
        if missing:
            _fail("figures 1–8", f"missing {missing}")
        else:
            tiff = list((ROOT / "figures" / "TIFF").glob("fig*.tiff"))
            eps = list((ROOT / "figures" / "EPS").glob("fig*.eps"))
            _ok("figures 1–8", f"tiff={len(tiff)} eps={len(eps)}")
    except Exception as exc:
        _fail("figures 1–8", exc)
        traceback.print_exc()


def check_cli_help() -> None:
    try:
        import subprocess

        for script in (
            "scripts/inference/run_method.py",
            "scripts/inference/compute_metrics.py",
            "scripts/lf_selection/run_prism_proliferating.py",
            "scripts/lf_selection/run_prism_nonproliferating.py",
        ):
            r = subprocess.run(
                [sys.executable, script, "--help"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode != 0:
                raise RuntimeError(f"{script} --help failed: {r.stderr[-400:]}")
        _ok("CLI --help")
    except Exception as exc:
        _fail("CLI --help", exc)


def main() -> int:
    print("=== prism-mf-sbi verification ===")
    check_imports()
    check_tasks()
    check_fidelities_replicates()
    check_cli_help()
    check_prolif_sim()
    check_prolif_replicate_mean()
    check_nonprolif_sim()
    check_methods_smoke()
    check_figures()
    print(f"\n=== summary: {PASS} passed, {FAIL} failed, {SKIP} skipped ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
