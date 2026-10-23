from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


class Task(Protocol):
    """
    Minimal experiment interface required by runners + methods.

    All arrays are NumPy. Shapes:
    - theta: (n, d_theta)
    - x: (n, d_x)
    """

    experiment_id: str
    experiment_name: str
    theta_names: Sequence[str]
    x_names: Sequence[str]

    def sample_prior(self, n: int, seed: int) -> np.ndarray: ...
    
    def get_prior_distribution(self) -> torch.distributions.Distribution: ...

    def simulate_hf_x(self, theta: np.ndarray, seed: int) -> np.ndarray: ...

    def simulate_lf_x(self, theta: np.ndarray, seed: int) -> np.ndarray: ...

    def get_observation(self, obs_id: int) -> np.ndarray: ...

    def get_true_parameters(self, obs_id: int) -> np.ndarray: ...

    def distance(self, x: np.ndarray, x_o: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RegisteredTask:
    experiment_id: str
    factory: Callable[[dict[str, Any] | None], Task]


_TASKS: dict[str, RegisteredTask] = {}
_BUILTIN_TASKS_LOADED = False


def _ensure_builtin_tasks() -> None:
    global _BUILTIN_TASKS_LOADED
    if not _BUILTIN_TASKS_LOADED:
        import prism_vertex.experiments._builtin_tasks  # noqa: F401

        _BUILTIN_TASKS_LOADED = True


def register_task(experiment_id: str, factory: Callable[[dict[str, Any] | None], Task]) -> None:
    if experiment_id in _TASKS:
        raise KeyError(f"Task already registered: {experiment_id}")
    _TASKS[experiment_id] = RegisteredTask(experiment_id=experiment_id, factory=factory)


def get_task(experiment_id: str, config: dict[str, Any] | None = None) -> Task:
    _ensure_builtin_tasks()
    if experiment_id not in _TASKS:
        raise KeyError(f"Unknown experiment_id: {experiment_id}. Registered: {sorted(_TASKS.keys())}")
    return _TASKS[experiment_id].factory(config)

