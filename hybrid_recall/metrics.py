"""Quality and cost metrics."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import numpy as np
import psutil


def recall_at_k(predicted: Sequence[str], truth: Sequence[str], k: int) -> float:
    """|predicted[:k] ∩ truth[:k]| / |truth[:k]|. Order-insensitive set overlap."""
    truth_set = set(truth[:k])
    if not truth_set:
        return 1.0 if not predicted else 0.0
    pred_set = set(predicted[:k])
    return len(pred_set & truth_set) / len(truth_set)


def percentiles(values: Sequence[float], ps: Sequence[int] = (50, 95, 99)) -> dict[str, float]:
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


class PeakRSS:
    """Sample this process's RSS in a background thread; report the peak (MB).

    Captures the high-water mark across an index build or a query phase. Sampling cost is
    negligible (one psutil read every `interval` seconds)."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self._proc = psutil.Process()
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakRSS":
        self._peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return self._peak / (1024 * 1024)


def time_call(fn, *args, **kwargs) -> tuple[object, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
