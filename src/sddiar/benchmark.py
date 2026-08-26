"""Dependency-free performance measurement helpers; measurements only."""
from __future__ import annotations

import resource
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class StageMeasurement:
    name: str
    elapsed_seconds: float
    audio_seconds: float | None = None
    peak_rss_bytes: int | None = None

    @property
    def rtf(self) -> float | None:
        return None if not self.audio_seconds or self.audio_seconds <= 0 else self.elapsed_seconds / self.audio_seconds


@dataclass(frozen=True, slots=True)
class RunMeasurement:
    elapsed_seconds: float
    peak_rss_bytes: int | None
    result: Any = None


def peak_rss_bytes() -> int | None:
    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        return int(raw if __import__('sys').platform == "darwin" else raw * 1024)
    except (AttributeError, OSError):
        return None


def measure_stage(name: str, fn: Callable[[], Any], *, audio_seconds: float | None = None) -> tuple[StageMeasurement, Any]:
    start = time.perf_counter(); result = fn(); elapsed = time.perf_counter() - start
    return StageMeasurement(name, elapsed, audio_seconds, peak_rss_bytes()), result


def run_repeated(fn: Callable[[], Any], *, repeats: int = 1) -> tuple[RunMeasurement, ...]:
    if repeats < 1: raise ValueError("repeats must be positive")
    out = []
    for _ in range(repeats):
        start = time.perf_counter(); result = fn()
        out.append(RunMeasurement(time.perf_counter() - start, peak_rss_bytes(), result))
    return tuple(out)


def account_long_file(stages: Iterable[StageMeasurement], *, audio_seconds: float, max_audio_seconds: float | None = None) -> dict[str, Any]:
    """Summarize bounded accounting without asserting a performance target."""
    if audio_seconds <= 0: raise ValueError("audio_seconds must be positive")
    bounded = max_audio_seconds is not None and audio_seconds > max_audio_seconds
    total = sum(s.elapsed_seconds for s in stages)
    return {"audio_seconds": audio_seconds, "elapsed_seconds": total, "rtf": total / audio_seconds,
            "bounded": bounded, "max_audio_seconds": max_audio_seconds,
            "peak_rss_bytes": max((s.peak_rss_bytes or 0 for s in stages), default=0) or None}


measure = measure_stage
