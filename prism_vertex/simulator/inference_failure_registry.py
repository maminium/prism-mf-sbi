"""Per-inference-run registry of parameter points with simulator failures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_THETA_ROUND = 6


def theta_key(theta: np.ndarray) -> str:
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    rounded = np.round(t, _THETA_ROUND)
    return ",".join(f"{float(v):g}" for v in rounded)


def _empty_registry() -> dict[str, Any]:
    return {
        "version": 1,
        "blocked_theta": {},
        "stats": {
            "n_penalty_lf": 0,
            "n_penalty_hf": 0,
            "n_skipped_lf": 0,
            "n_skipped_hf": 0,
        },
    }


class InferenceFailureRegistry:
    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._data = _empty_registry()
        if self.path is not None and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict) and "blocked_theta" in loaded:
                    self._data = loaded
                    self._data.setdefault("stats", _empty_registry()["stats"])
            except (json.JSONDecodeError, OSError):
                pass

    @classmethod
    def from_env(cls) -> InferenceFailureRegistry:
        for env_name in (
            "PRISM_VERTEX_INFERENCE_FAILURE_REGISTRY_PATH",
            "PRISM_VERTEX_FAILURE_REGISTRY_PATH",
        ):
            raw = os.environ.get(env_name, "").strip()
            if raw:
                return cls(Path(raw))
        return cls(None)

    def is_enabled(self) -> bool:
        return self.path is not None and os.environ.get("PRISM_VERTEX_FAILURE_REGISTRY", "1") not in {
            "0",
            "false",
            "False",
        }

    def is_blocked(self, theta: np.ndarray) -> bool:
        if not self.is_enabled():
            return False
        return theta_key(theta) in self._data.get("blocked_theta", {})

    def register_failure(
        self,
        theta: np.ndarray,
        *,
        fidelity: str,
        failure_class: str = "sim_failure",
        skipped: bool = False,
    ) -> None:
        if not self.is_enabled():
            return
        key = theta_key(theta)
        entry = {
            "theta": np.asarray(theta, dtype=np.float64).reshape(-1).tolist(),
            "fidelity": fidelity,
            "failure_class": failure_class,
            "skipped": bool(skipped),
        }
        blocked = self._data.setdefault("blocked_theta", {})
        prev = blocked.get(key)
        if prev is None:
            blocked[key] = entry
        else:
            prev["failure_class"] = failure_class
            prev["skipped"] = prev.get("skipped", False) or skipped

        stats = self._data.setdefault("stats", _empty_registry()["stats"])
        if skipped:
            stats[f"n_skipped_{fidelity}"] = int(stats.get(f"n_skipped_{fidelity}", 0)) + 1
        else:
            stats[f"n_penalty_{fidelity}"] = int(stats.get(f"n_penalty_{fidelity}", 0)) + 1
        self._flush()

    def stats(self) -> dict[str, Any]:
        return dict(self._data.get("stats", {}))

    def blocked_count(self) -> int:
        return len(self._data.get("blocked_theta", {}))

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))


def init_inference_failure_registry(run_dir: Path) -> InferenceFailureRegistry:
    path = Path(run_dir) / "failure_registry_inference.json"
    os.environ["PRISM_VERTEX_INFERENCE_FAILURE_REGISTRY_PATH"] = str(path)
    os.environ["PRISM_VERTEX_FAILURE_REGISTRY_PATH"] = str(path)
    return InferenceFailureRegistry(path)
