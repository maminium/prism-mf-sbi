from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Timer:
    seconds: float = 0.0
    _t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._t0 is None:
            return
        self.seconds += time.perf_counter() - self._t0
        self._t0 = None

    @property
    def elapsed_s(self) -> float:
        if self._t0 is None:
            return float(self.seconds)
        return float(self.seconds + (time.perf_counter() - self._t0))


@contextmanager
def time_block(timer: Timer) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timer.seconds += time.perf_counter() - t0

