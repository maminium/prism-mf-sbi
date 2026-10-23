from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessRunResult:
    returncode: int
    stdout: str
    stderr: str
    status: str  # ok | timeout | stalled | launch_error
    message: str


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except Exception:
        # Watchdog should never crash the run path.
        return 0
    return total


def run_with_watchdog(
    *,
    args: list[str],
    cwd: str,
    env: dict[str, str],
    hard_timeout_s: float,
    watch_dir: Path | None = None,
    enable_stall_watchdog: bool = True,
    stall_seconds: float = 900.0,
    stall_poll_seconds: float = 60.0,
    stall_min_age_seconds: float = 120.0,
) -> ProcessRunResult:
    """
    Run a subprocess with:
      1) hard timeout kill
      2) optional no-progress watchdog on output-dir size growth.
    """
    try:
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, OSError) as exc:
        return ProcessRunResult(
            returncode=127,
            stdout="",
            stderr="",
            status="launch_error",
            message=str(exc),
        )

    lock = threading.Lock()
    stalled = {"value": False, "message": ""}
    stop_evt = threading.Event()
    started_at = time.monotonic()

    def _watchdog() -> None:
        if not enable_stall_watchdog or watch_dir is None:
            return
        last_size = _dir_size_bytes(watch_dir)
        last_change = time.monotonic()
        while not stop_evt.is_set():
            time.sleep(max(1.0, stall_poll_seconds))
            if stop_evt.is_set():
                return
            if proc.poll() is not None:
                return
            now = time.monotonic()
            size_now = _dir_size_bytes(watch_dir)
            if size_now != last_size:
                last_size = size_now
                last_change = now
                continue
            age = now - started_at
            stagnant = now - last_change
            if age >= stall_min_age_seconds and stagnant >= stall_seconds:
                with lock:
                    stalled["value"] = True
                    stalled["message"] = (
                        f"No output-dir growth for {stagnant:.0f}s "
                        f"(watch_dir={watch_dir})"
                    )
                try:
                    proc.terminate()
                    time.sleep(5.0)
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
                return

    th = threading.Thread(target=_watchdog, daemon=True)
    th.start()
    try:
        try:
            stdout, stderr = proc.communicate(timeout=max(1.0, hard_timeout_s))
            rc = int(proc.returncode or 0)
            if rc == 0:
                return ProcessRunResult(rc, stdout or "", stderr or "", "ok", "")
            with lock:
                if stalled["value"]:
                    return ProcessRunResult(rc, stdout or "", stderr or "", "stalled", stalled["message"])
            return ProcessRunResult(rc, stdout or "", stderr or "", "ok", f"nonzero returncode={rc}")
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                time.sleep(5.0)
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
            stdout, stderr = proc.communicate()
            return ProcessRunResult(
                returncode=124,
                stdout=stdout or "",
                stderr=stderr or "",
                status="timeout",
                message=f"Hard timeout exceeded ({hard_timeout_s:.0f}s)",
            )
    finally:
        stop_evt.set()
        th.join(timeout=1.0)
