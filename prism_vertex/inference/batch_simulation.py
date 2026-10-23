"""Batch HF/LF simulation helpers for paper inference methods."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from prism_vertex.simulator.sim_outcome import (
    FIDELITY_HF,
    FIDELITY_LF,
    SimulationDropped,
    SimOutcome,
)
from prism_vertex.experiments.task import Task
from prism_vertex.inference.inference_context import (
    baseline_drop_on_fail_enabled,
    penalty_mode_enabled,
    get_inference_x_o,
    make_penalty_summary,
)
from prism_vertex.inference.penalty_summaries import is_penalty_summary
from prism_vertex.utils.seeding import derive_seed


@dataclass
class SimBatchResult:
    theta: np.ndarray
    x: np.ndarray
    m_ok: np.ndarray
    fidelity_id: np.ndarray
    outcomes: list[SimOutcome]
    n_attempted: int
    n_dropped: int


@dataclass(frozen=True)
class _RowSimResult:
    x: np.ndarray | None
    dropped: bool


def _is_penalty_x(x: np.ndarray) -> bool:
    x_o = get_inference_x_o()
    if x_o is None:
        return False
    from prism_vertex.inference.inference_context import get_penalty_scales

    try:
        scales = get_penalty_scales()
    except RuntimeError:
        scales = None
    return is_penalty_summary(np.asarray(x, dtype=np.float64), x_o, scales=scales)


def _sim_jobs_for_task(task: Task) -> int:
    return max(1, int(getattr(task, "sim_jobs", 1)))


def _use_parallel_pool(task: Task, n_rows: int) -> bool:
    if n_rows <= 1:
        return False
    if _sim_jobs_for_task(task) <= 1:
        return False
    from prism_vertex.simulator.task import ProliferatingVertexTask
    from prism_vertex.simulator.nonproliferating.task import NonproliferatingVertexTask

    return isinstance(task, (ProliferatingVertexTask, NonproliferatingVertexTask))


def _row_sim_dropped(exc: BaseException) -> bool:
    from prism_vertex.simulator.nonproliferating.task import NonproliferatingSimulationDropped

    return isinstance(exc, (SimulationDropped, NonproliferatingSimulationDropped))


def _simulate_one_row(task: Task, row: np.ndarray, row_seed: int, fid_str: str) -> _RowSimResult:
    """In-process single-θ sim (mocks, tests, or sim_jobs=1)."""
    sim_fn = task.simulate_hf_x if fid_str == "hf" else task.simulate_lf_x
    try:
        x = np.asarray(
            sim_fn(np.asarray(row, dtype=np.float64).reshape(-1), seed=int(row_seed)),
            dtype=np.float64,
        ).reshape(-1)
        return _RowSimResult(x=x, dropped=False)
    except Exception as exc:
        if _row_sim_dropped(exc):
            return _RowSimResult(x=None, dropped=True)
        raise


def _prolif_row_worker(args: tuple[np.ndarray, int, str]) -> _RowSimResult:
    """Picklable ProcessPool worker: one θ, same seeds as the serial loop."""
    row, row_seed, fid_str = args
    from prism_vertex.simulator.task import ProliferatingVertexTask

    task = ProliferatingVertexTask()
    sim_fn = task.simulate_hf_x if fid_str == "hf" else task.simulate_lf_x
    try:
        x = np.asarray(
            sim_fn(np.asarray(row, dtype=np.float64).reshape(-1), seed=int(row_seed)),
            dtype=np.float64,
        ).reshape(-1)
        return _RowSimResult(x=x, dropped=False)
    except SimulationDropped:
        return _RowSimResult(x=None, dropped=True)


def _nonprolif_row_worker(args: tuple[np.ndarray, int, str]) -> _RowSimResult:
    row, row_seed, fid_str = args
    from prism_vertex.simulator.nonproliferating.task import (
        NonproliferatingSimulationDropped,
        NonproliferatingVertexTask,
    )

    task = NonproliferatingVertexTask()
    sim_fn = task.simulate_hf_x if fid_str == "hf" else task.simulate_lf_x
    try:
        x = np.asarray(
            sim_fn(np.asarray(row, dtype=np.float64).reshape(-1), seed=int(row_seed)),
            dtype=np.float64,
        ).reshape(-1)
        return _RowSimResult(x=x, dropped=False)
    except NonproliferatingSimulationDropped:
        return _RowSimResult(x=None, dropped=True)


def _simulate_rows(
    task: Task,
    arr: np.ndarray,
    *,
    seed: int,
    seed_tag: str,
    fid_str: str,
) -> list[_RowSimResult]:
    items = [
        (
            np.asarray(row, dtype=np.float64).reshape(-1),
            int(derive_seed(seed, "sim", seed_tag, fid_str, i)),
            fid_str,
        )
        for i, row in enumerate(arr)
    ]
    if not _use_parallel_pool(task, len(items)):
        return [_simulate_one_row(task, row, row_seed, fid_str) for row, row_seed, fid_str in items]

    n_jobs = min(_sim_jobs_for_task(task), len(items))
    out: list[_RowSimResult | None] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        from prism_vertex.simulator.nonproliferating.task import NonproliferatingVertexTask

        if isinstance(task, NonproliferatingVertexTask):
            futures = {pool.submit(_nonprolif_row_worker, item): idx for idx, item in enumerate(items)}
        else:
            futures = {pool.submit(_prolif_row_worker, item): idx for idx, item in enumerate(items)}
        for fut in as_completed(futures):
            idx = futures[fut]
            out[idx] = fut.result()
    return [r for r in out if r is not None]


def _process_row(
    row: np.ndarray,
    result: _RowSimResult,
    *,
    fid_str: str,
    fid_id: int,
) -> tuple[np.ndarray | None, np.ndarray | None, bool | None, int, SimOutcome]:
    """Return (theta, x, m_ok, n_dropped_delta, outcome)."""
    if result.dropped:
        if penalty_mode_enabled() and not baseline_drop_on_fail_enabled():
            x_pen = make_penalty_summary()
            outcome = SimOutcome(
                theta=row.copy(),
                summary_6d=x_pen.copy(),
                m_ok=False,
                status="dropped",
                fidelity=fid_str,
                is_penalty=True,
            )
            return row, x_pen, False, 0, outcome
        outcome = SimOutcome(
            theta=row.copy(),
            summary_6d=make_penalty_summary(),
            m_ok=False,
            status="dropped",
            fidelity=fid_str,
            is_penalty=True,
        )
        return None, None, None, 1, outcome

    x = np.asarray(result.x, dtype=np.float64).reshape(-1)
    penalty = _is_penalty_x(x)
    m_ok = not penalty
    status = "ok" if m_ok else "sim_failure"
    outcome = SimOutcome(
        theta=row.copy(),
        summary_6d=x.copy(),
        m_ok=m_ok,
        status=status,
        fidelity=fid_str,
        is_penalty=penalty,
    )
    if baseline_drop_on_fail_enabled() and not m_ok:
        return None, None, None, 1, outcome
    return row, x, m_ok, 0, outcome


def simulate_theta_batch(
    task: Task,
    theta: np.ndarray,
    *,
    seed: int,
    fidelity: str,
    seed_tag: str,
) -> SimBatchResult:
    """Simulate a batch; parallel when ``PRISM_VERTEX_SIM_JOBS`` > 1."""
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    fid_str = fidelity.lower()
    fid_id = FIDELITY_HF if fid_str == "hf" else FIDELITY_LF

    kept_theta: list[np.ndarray] = []
    kept_x: list[np.ndarray] = []
    kept_m: list[bool] = []
    kept_fid: list[int] = []
    outcomes: list[SimOutcome] = []
    n_dropped = 0

    row_results = _simulate_rows(task, arr, seed=seed, seed_tag=seed_tag, fid_str=fid_str)
    for row, result in zip(arr, row_results, strict=True):
        th, x, m_ok, drop_delta, outcome = _process_row(row, result, fid_str=fid_str, fid_id=fid_id)
        outcomes.append(outcome)
        n_dropped += drop_delta
        if th is None:
            continue
        kept_theta.append(th)
        kept_x.append(x)
        kept_m.append(bool(m_ok))
        kept_fid.append(fid_id)

    if not kept_theta:
        d = int(arr.shape[1]) if arr.shape[0] else 1
        x_dim = int(np.asarray(make_penalty_summary()).shape[0]) if get_inference_x_o() is not None else d
        return SimBatchResult(
            theta=np.empty((0, d), dtype=np.float32),
            x=np.empty((0, x_dim), dtype=np.float32),
            m_ok=np.empty((0,), dtype=bool),
            fidelity_id=np.empty((0,), dtype=np.int64),
            outcomes=outcomes,
            n_attempted=int(arr.shape[0]),
            n_dropped=n_dropped,
        )

    return SimBatchResult(
        theta=np.stack(kept_theta, axis=0).astype(np.float32),
        x=np.stack(kept_x, axis=0).astype(np.float32),
        m_ok=np.asarray(kept_m, dtype=bool),
        fidelity_id=np.asarray(kept_fid, dtype=np.int64),
        outcomes=outcomes,
        n_attempted=int(arr.shape[0]),
        n_dropped=n_dropped,
    )
