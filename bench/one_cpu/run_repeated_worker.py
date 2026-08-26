#!/usr/bin/env python3
"""Run a persistent local diarizer repeatedly under the one-CPU contract.

This is intentionally a small, dependency-free harness around
``LocalOnnxDiarizer``.  The model object is constructed exactly once and is
then used for every input/iteration in order.  The output is an aggregate
health record: it contains no source paths, audio names, transcripts, or raw
spans.  The functional result is represented by a digest of a canonical span
timeline and by aggregate counts only.

The callable API accepts readers and clocks as keyword arguments.  That makes
the safety checks unit-testable without starting a process, making a network
request, or depending on a particular host cgroup layout.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # ``resource`` is unavailable on Windows; this harness remains importable.
    import resource as _resource
except ImportError:  # pragma: no cover - platform dependent
    _resource = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sddiar.onnx_diarization import LocalOnnxDiarizationConfig, LocalOnnxDiarizer  # noqa: E402
from sddiar.media import WavPcmAccessor  # noqa: E402
from sddiar.runtime_env import delta_cpu_snapshots, read_cpu_snapshot  # noqa: E402
from sddiar.service import atomic_publish  # noqa: E402


class RepeatedWorkerError(RuntimeError):
    """Base error for a failed repeated-worker gate."""


class DigestDriftError(RepeatedWorkerError):
    """The same input produced different canonical timelines."""


class QuotaMismatchError(RepeatedWorkerError):
    """The observed cgroup was not exactly the requested CPU quota."""


class RssGrowthError(RepeatedWorkerError):
    """Resident memory grew beyond the configured repeated-run limit."""


class FallbackError(RepeatedWorkerError):
    """A model/runtime fallback was observed."""


class RuntimeLimitError(RepeatedWorkerError):
    """A declared RTF, memory, throttle, or evidence gate was exceeded."""


def _json_default(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, (tuple, list)):
        return list(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=_json_default).encode("utf-8")


def config_digest(config: Any) -> str:
    """Return a deterministic configuration fingerprint without file paths."""

    if config is None:
        payload: Any = {}
    elif isinstance(config, Mapping):
        payload = dict(config)
    elif hasattr(config, "__dict__"):
        payload = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    else:
        payload = str(config)
    # A caller may pass a full CLI mapping.  Model/audio locations are not
    # needed to identify algorithm settings and must never reach the record.
    if isinstance(payload, dict):
        payload = {k: v for k, v in payload.items()
                   if "path" not in str(k).lower() and str(k).lower() not in
                   {"audio", "audio_path", "silero_model", "wespeaker_model"}}
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _span_field(span: Any, name: str, default: Any = None) -> Any:
    value = _field(span, name, default)
    if value is None:
        return default
    if isinstance(value, (tuple, list)):
        return list(value)
    return value


def canonical_timeline(result: Any) -> list[dict[str, Any]]:
    """Extract only deterministic, redacted timeline fields from a result."""

    spans = _field(result, "spans", ()) or ()
    timeline: list[dict[str, Any]] = []
    for span in spans:
        # Deliberately omit span_id: implementations may derive it from an
        # input identifier.  Temporal coordinates and assignment evidence are
        # sufficient for a deterministic drift check.
        timeline.append({
            "start_us": int(_span_field(span, "start_us", 0)),
            "end_us": int(_span_field(span, "end_us", 0)),
            "speaker_id": str(_span_field(span, "speaker_id", "UNKNOWN")),
            "attribution_status": str(_span_field(span, "attribution_status", "")),
            "evidence_ids": list(_span_field(span, "evidence_ids", ()) or ()),
            "reason_codes": list(_span_field(span, "reason_codes", ()) or ()),
        })
    return timeline


def timeline_digest(result: Any) -> str:
    """Hash a canonical timeline; no raw spans are returned by this harness."""

    return hashlib.sha256(_json_bytes(canonical_timeline(result))).hexdigest()


def _result_metrics(result: Any) -> Mapping[str, Any]:
    metrics = _field(result, "metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def result_counts(result: Any) -> dict[str, int]:
    """Return speaker-neutral counts suitable for an aggregate record."""

    timeline = canonical_timeline(result)
    metrics = _result_metrics(result)
    speakers = [str(span["speaker_id"]) for span in timeline]
    return {
        "span_count": len(timeline),
        "assigned_span_count": sum(s not in {"UNKNOWN", "OVERLAP"} for s in speakers),
        "unknown_span_count": sum(s == "UNKNOWN" for s in speakers),
        "overlap_span_count": sum(s == "OVERLAP" for s in speakers),
        "tracklet_count": int(metrics.get("tracklet_count", 0) or 0),
        "anchor_count": int(metrics.get("anchor_count", 0) or 0),
        "support_count": int(metrics.get("support_count", 0) or 0),
        "deferred_count": int(metrics.get("deferred_count", 0) or 0),
        "valid_embedding_count": int(metrics.get("valid_embedding_count", 0) or 0),
    }


def _parse_size(value: str) -> int | None:
    parts = value.strip().split()
    if not parts:
        return None
    try:
        number = float(parts[0])
    except ValueError:
        return None
    unit = parts[1].lower() if len(parts) > 1 else "b"
    factor = {"b": 1, "kb": 1024, "kib": 1024, "mb": 1024**2,
              "mib": 1024**2, "gb": 1024**3, "gib": 1024**3}.get(unit)
    if factor is None or not math.isfinite(number) or number < 0:
        return None
    return int(number * factor)


def read_proc_status(path: str | os.PathLike[str] = "/proc/self/status") -> dict[str, int | None]:
    """Read the small process status subset used by the benchmark."""

    values: dict[str, int | None] = {"VmRSS_bytes": None, "VmHWM_bytes": None, "Threads": None}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return values
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key in {"VmRSS", "VmHWM"}:
            values[f"{key}_bytes"] = _parse_size(raw)
        elif key == "Threads":
            try:
                values[key] = int(raw.strip())
            except ValueError:
                values[key] = None
    return values


def read_smaps_rollup(path: str | os.PathLike[str] = "/proc/self/smaps_rollup") -> dict[str, int | None]:
    """Read RSS/PSS from Linux smaps_rollup when permitted."""

    values: dict[str, int | None] = {"Rss_bytes": None, "Pss_bytes": None}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return values
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if sep and key.strip() in {"Rss", "Pss"}:
            values[f"{key.strip()}_bytes"] = _parse_size(raw)
    return values


def _proc_status_fields(path: Path) -> dict[str, int | None]:
    values: dict[str, int | None] = {"ppid": None, "rss_bytes": None, "threads": None}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return values
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key == "PPid":
            try:
                values["ppid"] = int(raw.strip())
            except ValueError:
                pass
        elif key == "VmRSS":
            values["rss_bytes"] = _parse_size(raw)
        elif key == "Threads":
            try:
                values["threads"] = int(raw.strip())
            except ValueError:
                pass
    return values


def _proc_io_fields(path: Path) -> dict[str, int | None]:
    values: dict[str, int | None] = {"read_bytes": None, "write_bytes": None}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return values
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if sep and key.strip() in values:
            try:
                values[key.strip()] = int(raw.strip())
            except ValueError:
                pass
    return values


def read_process_tree_resources(
    proc_root: str | os.PathLike[str] = "/proc",
    *,
    root_pid: int | None = None,
    platform_name: str | None = None,
) -> dict[str, int | None]:
    """Aggregate RSS/PSS/thread/I/O evidence for this process and descendants.

    The function emits counts only, never PIDs or command lines.  Processes
    that race with the scan are ignored; missing PSS or I/O remains ``None``
    rather than being reported as zero.
    """

    result: dict[str, int | None] = {
        "process_count": None,
        "thread_count": None,
        "process_tree_rss_bytes": None,
        "process_tree_pss_bytes": None,
        "read_bytes": None,
        "write_bytes": None,
    }
    if not (platform_name or sys.platform).startswith("linux"):
        return result
    proc = Path(proc_root)
    target = os.getpid() if root_pid is None else int(root_pid)
    records: dict[int, dict[str, int | None]] = {}
    try:
        entries = tuple(proc.iterdir())
    except OSError:
        return result
    for entry in entries:
        if not entry.name.isdigit() or not entry.is_dir():
            continue
        fields = _proc_status_fields(entry / "status")
        if fields["ppid"] is not None:
            records[int(entry.name)] = fields
    selected = {target}
    changed = True
    while changed:
        changed = False
        for pid, fields in records.items():
            if pid not in selected and fields["ppid"] in selected:
                selected.add(pid)
                changed = True
    selected.intersection_update(records)
    if not selected:
        return result

    rss_total = pss_total = threads_total = read_total = write_total = 0
    rss_seen = pss_seen = threads_seen = read_seen = write_seen = False
    for pid in selected:
        fields = records[pid]
        if fields["rss_bytes"] is not None:
            rss_total += int(fields["rss_bytes"])
            rss_seen = True
        if fields["threads"] is not None:
            threads_total += int(fields["threads"])
            threads_seen = True
        smaps = read_smaps_rollup(proc / str(pid) / "smaps_rollup")
        if smaps["Rss_bytes"] is not None:
            # smaps RSS is more complete than VmRSS; replace this process's
            # status contribution rather than double-counting it.
            rss_total += int(smaps["Rss_bytes"]) - int(fields["rss_bytes"] or 0)
            rss_seen = True
        if smaps["Pss_bytes"] is not None:
            pss_total += int(smaps["Pss_bytes"])
            pss_seen = True
        io_values = _proc_io_fields(proc / str(pid) / "io")
        if io_values["read_bytes"] is not None:
            read_total += int(io_values["read_bytes"])
            read_seen = True
        if io_values["write_bytes"] is not None:
            write_total += int(io_values["write_bytes"])
            write_seen = True
    result.update({
        "process_count": len(selected),
        "thread_count": threads_total if threads_seen else None,
        "process_tree_rss_bytes": rss_total if rss_seen else None,
        "process_tree_pss_bytes": pss_total if pss_seen else None,
        "read_bytes": read_total if read_seen else None,
        "write_bytes": write_total if write_seen else None,
    })
    return result


def read_cgroup_cpuacct_usage_usec(
    root: str | os.PathLike[str] = "/sys/fs/cgroup",
    proc_cgroup_path: str | os.PathLike[str] = "/proc/self/cgroup",
    platform_name: str | None = None,
) -> int | None:
    """Read the cgroup-v1 ``cpuacct.usage`` nanosecond counter as usec."""

    if not (platform_name or sys.platform).startswith("linux"):
        return None
    root_path = Path(root)
    _, cpu_rel, _ = _proc_cgroup_paths(Path(proc_cgroup_path))
    if cpu_rel is None or (root_path / "cgroup.controllers").exists():
        return None
    rel = cpu_rel.lstrip("/")
    candidates = (
        root_path / "cpu,cpuacct" / rel / "cpuacct.usage",
        root_path / "cpuacct" / rel / "cpuacct.usage",
        root_path / "cpu" / rel / "cpuacct.usage",
        root_path / rel / "cpuacct.usage",
    )
    for path in candidates:
        try:
            raw = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError, UnicodeError):
            continue
        return raw // 1000 if raw >= 0 else None
    return None


def _proc_cgroup_paths(path: Path) -> tuple[str | None, str | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None, None, None
    v2 = cpu = memory = None
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, rel = fields
        if not controllers:
            v2 = rel
        if "cpu" in controllers.split(",") or "cpuacct" in controllers.split(","):
            cpu = rel
        if "memory" in controllers.split(","):
            memory = rel
    return v2, cpu, memory


def read_cgroup_memory(root: str | os.PathLike[str] = "/sys/fs/cgroup",
                       proc_cgroup_path: str | os.PathLike[str] = "/proc/self/cgroup",
                       platform_name: str | None = None) -> dict[str, int | None]:
    """Read cgroup current/peak memory without invoking a subprocess."""

    values: dict[str, int | None] = {"current_bytes": None, "peak_bytes": None}
    if (platform_name or sys.platform) != "linux":
        return values
    root_path = Path(root)
    v2, _, v1_memory = _proc_cgroup_paths(Path(proc_cgroup_path))
    if (root_path / "cgroup.controllers").exists():
        base = root_path / (v2.lstrip("/") if v2 and v2 != "/" else "")
        for key, filename in (("current_bytes", "memory.current"), ("peak_bytes", "memory.peak")):
            try:
                raw = (base / filename).read_text(encoding="utf-8")
                values[key] = None if raw.strip() == "max" else int(raw.strip())
            except (OSError, ValueError):
                pass
        return values
    if v1_memory is not None:
        base = root_path / "memory" / v1_memory.lstrip("/")
        for key, filename in (("current_bytes", "memory.usage_in_bytes"),
                              ("peak_bytes", "memory.max_usage_in_bytes")):
            try:
                values[key] = int((base / filename).read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pass
    return values


def _rusage() -> dict[str, int | float | None]:
    if _resource is None:
        return {"ru_maxrss_bytes": None, "user_cpu_sec": None, "system_cpu_sec": None}
    usage = _resource.getrusage(_resource.RUSAGE_SELF)
    raw_maxrss = float(usage.ru_maxrss)
    # Linux reports KiB, macOS reports bytes.
    maxrss_bytes = int(raw_maxrss * 1024) if sys.platform.startswith("linux") else int(raw_maxrss)
    return {"ru_maxrss_bytes": maxrss_bytes, "user_cpu_sec": float(usage.ru_utime),
            "system_cpu_sec": float(usage.ru_stime)}


def _gc_snapshot(collect: Callable[[], Any], counts: Callable[[], Any]) -> dict[str, Any]:
    collected = collect()
    raw_counts = counts()
    return {"collected": int(collected) if isinstance(collected, int) else None,
            "counts": list(raw_counts) if isinstance(raw_counts, (tuple, list)) else None,
            "enabled": bool(gc.isenabled())}


def _cpu_effective(snapshot: Any) -> float | None:
    value = _field(snapshot, "effective_cpu_equivalent", None)
    if value is None and isinstance(snapshot, Mapping):
        value = snapshot.get("effective_cpu")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _counter(value: Any, key: str) -> int | None:
    raw = _field(value, key, None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _cpu_record(before: Any, after: Any, delta: Any) -> dict[str, Any]:
    stat = _field(delta, "cpu_stat", None)
    return {
        "cgroup_version": _field(delta, "cgroup_version", _field(after, "cgroup_version", None)),
        "effective_cpu_equivalent": _cpu_effective(after),
        "cpuset_cpu_count": _field(delta, "cpuset_cpu_count", _field(after, "cpuset_cpu_count", None)),
        "usage_usec": _counter(stat, "usage_usec"),
        "user_usec": _counter(stat, "user_usec"),
        "system_usec": _counter(stat, "system_usec"),
        "nr_periods": _counter(stat, "nr_periods"),
        "nr_throttled": _counter(stat, "nr_throttled"),
        "throttled_usec": _counter(stat, "throttled_usec"),
        "counter_reset_detected": bool(_field(stat, "reset_detected", False)),
    }


def _fallback_detected(diarizer: Any, result: Any) -> bool:
    """Recognize an explicit fallback marker only; never infer from quality."""

    for item in (diarizer, result):
        for key in ("fallback_used", "fallback", "used_fallback", "backend_fallback"):
            if bool(_field(item, key, False)):
                return True
        decision = str(_field(item, "decision", ""))
        if "FALLBACK" in decision.upper():
            return True
        config = _field(item, "runtime_config", {})
        if isinstance(config, Mapping) and any(bool(config.get(k, False)) for k in
                                               ("fallback", "fallback_used", "backend_fallback")):
            return True
    return False


def _reuse_record(diarizer: Any) -> dict[str, Any]:
    """Expose reuse as public evidence, without reading private model state."""

    def public_count(names: Sequence[str]) -> int | None:
        for name in names:
            value = _field(diarizer, name, None)
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int) and value >= 0:
                return value
        return None

    session_count = public_count(("session_reuse_count", "session_count"))
    backend_count = public_count(("backend_reuse_count", "backend_count"))
    explicit = _field(diarizer, "session_backend_reused", None)
    if explicit is None:
        explicit = _field(diarizer, "session_reused", None)
    return {
        "single_diarizer_instance": True,
        "session_backend_reused": bool(explicit) if explicit is not None else True,
        "session_reuse_count": session_count,
        "backend_reuse_count": backend_count,
        "evidence": "public_counter" if session_count is not None or backend_count is not None or explicit is not None
        else "single_diarizer_instance_contract",
    }


def read_wav_duration_us(path: str | os.PathLike[str]) -> int:
    layout = WavPcmAccessor(path).layout
    frames = layout.frame_count
    rate = layout.sample_rate_hz
    if rate <= 0:
        raise ValueError("WAV sample rate must be positive")
    return (frames * 1_000_000 + rate // 2) // rate


def _memory_growth_bytes(baseline: Mapping[str, Any], after: Mapping[str, Any]) -> int | None:
    candidates: list[int] = []
    for key in ("VmRSS_bytes", "Rss_bytes", "process_tree_rss_bytes", "process_tree_pss_bytes"):
        before = baseline.get(key)
        current = after.get(key)
        if isinstance(before, (int, float)) and isinstance(current, (int, float)):
            candidates.append(max(0, int(current - before)))
    return max(candidates) if candidates else None


def _resource_snapshot(*, status_reader: Callable[[], Mapping[str, Any]],
                       smaps_reader: Callable[[], Mapping[str, Any]],
                       process_tree_reader: Callable[[], Mapping[str, Any]],
                       memory_reader: Callable[[], Mapping[str, Any]],
                       rusage_reader: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    status = dict(status_reader())
    smaps = dict(smaps_reader())
    process_tree = dict(process_tree_reader())
    memory = dict(memory_reader())
    usage = dict(rusage_reader())
    return {
        **{k: status.get(k) for k in ("VmRSS_bytes", "VmHWM_bytes", "Threads")},
        **{k: smaps.get(k) for k in ("Rss_bytes", "Pss_bytes")},
        **{k: process_tree.get(k) for k in ("process_count", "thread_count", "process_tree_rss_bytes",
                                             "process_tree_pss_bytes", "read_bytes", "write_bytes")},
        **{k: memory.get(k) for k in ("current_bytes", "peak_bytes")},
        **{k: usage.get(k) for k in ("ru_maxrss_bytes", "user_cpu_sec", "system_cpu_sec")},
    }


def _largest_memory_bytes(snapshot: Mapping[str, Any]) -> int | None:
    values = [snapshot.get(key) for key in (
        "process_tree_rss_bytes", "process_tree_pss_bytes", "current_bytes", "peak_bytes",
        "Rss_bytes", "Pss_bytes", "VmRSS_bytes", "VmHWM_bytes", "ru_maxrss_bytes",
    )]
    valid = [int(value) for value in values if isinstance(value, (int, float)) and value >= 0]
    return max(valid) if valid else None


def _resident_memory_bytes(snapshot: Mapping[str, Any]) -> int | None:
    """Return current resident memory, excluding cumulative high-water marks.

    ``memory.peak``, ``VmHWM``, and ``ru_maxrss`` are appropriate for the hard
    peak cap, but they are monotonic high-water counters and cannot measure a
    warm-baseline leak.  The warm-growth gate therefore uses only current
    process-tree/cgroup resident observations.
    """

    values = [snapshot.get(key) for key in (
        "process_tree_rss_bytes", "process_tree_pss_bytes", "current_bytes",
        "Rss_bytes", "Pss_bytes", "VmRSS_bytes",
    )]
    valid = [int(value) for value in values if isinstance(value, (int, float)) and value >= 0]
    return max(valid) if valid else None


def _nonnegative_delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> int | None:
    left, right = before.get(key), after.get(key)
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)) or right < left:
        return None
    return int(right - left)


def run_repeated_worker(
    audio_paths: Iterable[str | os.PathLike[str]],
    diarizer_factory: Callable[[], Any],
    *,
    repetitions: int = 3,
    config: Any = None,
    require_cpu_equivalent: float | None = 1.0,
    rss_growth_limit_mb: float | None = 64.0,
    warm_rss_growth_limit_percent: float | None = 10.0,
    max_memory_mb: float | None = 256.0,
    max_rtf: float | None = 0.35,
    max_throttled_wall_ratio: float | None = 0.01,
    min_total_audio_minutes: float = 0.0,
    evidence_mode: str = "local_proxy",
    require_cgroup_version: str | None = None,
    duration_reader: Callable[[str | os.PathLike[str]], int] = read_wav_duration_us,
    cpu_snapshot_reader: Callable[[], Any] = read_cpu_snapshot,
    cpu_delta_reader: Callable[[Any, Any], Any] = delta_cpu_snapshots,
    status_reader: Callable[[], Mapping[str, Any]] = read_proc_status,
    smaps_reader: Callable[[], Mapping[str, Any]] = read_smaps_rollup,
    process_tree_reader: Callable[[], Mapping[str, Any]] = read_process_tree_resources,
    memory_reader: Callable[[], Mapping[str, Any]] = read_cgroup_memory,
    cpuacct_usage_reader: Callable[[], int | None] = read_cgroup_cpuacct_usage_usec,
    rusage_reader: Callable[[], Mapping[str, Any]] = _rusage,
    clock: Callable[[], float] = time.perf_counter,
    process_clock: Callable[[], float] = time.process_time,
    gc_collect: Callable[[], Any] = gc.collect,
    gc_counts: Callable[[], Any] = gc.get_count,
    fallback_detector: Callable[[Any, Any], bool] = _fallback_detected,
) -> dict[str, Any]:
    """Run all local WAVs sequentially through one persistent diarizer."""

    paths = tuple(audio_paths)
    if not paths:
        raise ValueError("at least one local WAV is required")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if require_cpu_equivalent is not None and (not math.isfinite(require_cpu_equivalent) or require_cpu_equivalent <= 0):
        raise ValueError("require_cpu_equivalent must be positive or None")
    if rss_growth_limit_mb is not None and (not math.isfinite(rss_growth_limit_mb) or rss_growth_limit_mb < 0):
        raise ValueError("rss_growth_limit_mb must be non-negative or None")
    for name, value in (("warm_rss_growth_limit_percent", warm_rss_growth_limit_percent),
                        ("max_memory_mb", max_memory_mb), ("max_rtf", max_rtf),
                        ("max_throttled_wall_ratio", max_throttled_wall_ratio)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be non-negative or None")
    if not math.isfinite(min_total_audio_minutes) or min_total_audio_minutes < 0:
        raise ValueError("min_total_audio_minutes must be non-negative")
    if evidence_mode not in {"local_proxy", "target"}:
        raise ValueError("evidence_mode must be local_proxy or target")
    if evidence_mode == "target":
        require_cgroup_version = require_cgroup_version or "v1"
        if min_total_audio_minutes < 60.0:
            raise ValueError("target evidence requires at least 60 processed audio minutes")

    durations = tuple(int(duration_reader(path)) for path in paths)
    if any(value <= 0 for value in durations):
        raise ValueError("audio duration must be positive")
    processed_audio_us = sum(durations) * repetitions
    if processed_audio_us < min_total_audio_minutes * 60.0 * 1_000_000:
        raise RuntimeLimitError("processed audio duration is below the declared long-form gate")

    # The one and only model construction point.  A factory also keeps tests
    # independent from real ONNX files and network/subprocess state.
    diarizer = diarizer_factory()
    reuse = _reuse_record(diarizer)
    baseline_resources = _resource_snapshot(status_reader=status_reader, smaps_reader=smaps_reader,
                                             process_tree_reader=process_tree_reader,
                                             memory_reader=memory_reader, rusage_reader=rusage_reader)
    baseline_memory = _largest_memory_bytes(baseline_resources)
    if max_memory_mb is not None and baseline_memory is not None and baseline_memory > max_memory_mb * 1024 * 1024:
        raise RuntimeLimitError("process-tree/cgroup memory exceeded the configured cap before inference")
    expected: dict[int, str] = {}
    runs: list[dict[str, Any]] = []
    fallback_seen = _fallback_detected(diarizer, None)
    warm_baseline_resources: dict[str, Any] | None = None
    warm_resident_growth_percent_max = 0.0

    for iteration in range(repetitions):
        for input_index, audio_path in enumerate(paths):
            # Keep the path local to this call; it is never copied into the
            # output.  Duration is read by an injectable parser for tests.
            duration_us = durations[input_index]
            cpu_before = cpu_snapshot_reader()
            if require_cgroup_version is not None and _field(cpu_before, "cgroup_version", None) != require_cgroup_version:
                raise QuotaMismatchError(f"required cgroup version {require_cgroup_version}")
            effective_before = _cpu_effective(cpu_before)
            if require_cpu_equivalent is not None and (effective_before is None or
                    not math.isclose(effective_before, require_cpu_equivalent, rel_tol=0.0, abs_tol=1e-9)):
                raise QuotaMismatchError(f"required CPU-equivalent {require_cpu_equivalent:.2f}")
            cpuacct_before = cpuacct_usage_reader()
            pre = _resource_snapshot(status_reader=status_reader, smaps_reader=smaps_reader,
                                     process_tree_reader=process_tree_reader,
                                     memory_reader=memory_reader, rusage_reader=rusage_reader)
            gc_pre = _gc_snapshot(gc_collect, gc_counts)
            wall_start = clock()
            cpu_start = process_clock()
            result = diarizer.process(audio_path)
            elapsed = float(clock() - wall_start)
            process_elapsed = float(process_clock() - cpu_start)
            cpu_after = cpu_snapshot_reader()
            if require_cgroup_version is not None and _field(cpu_after, "cgroup_version", None) != require_cgroup_version:
                raise QuotaMismatchError(f"required cgroup version {require_cgroup_version}")
            effective_after = _cpu_effective(cpu_after)
            if require_cpu_equivalent is not None and (effective_after is None or
                    not math.isclose(effective_after, require_cpu_equivalent, rel_tol=0.0, abs_tol=1e-9)):
                raise QuotaMismatchError(f"required CPU-equivalent {require_cpu_equivalent:.2f}")
            cpuacct_after = cpuacct_usage_reader()
            post = _resource_snapshot(status_reader=status_reader, smaps_reader=smaps_reader,
                                      process_tree_reader=process_tree_reader,
                                      memory_reader=memory_reader, rusage_reader=rusage_reader)
            gc_post = _gc_snapshot(gc_collect, gc_counts)
            fallback_seen = fallback_seen or bool(fallback_detector(diarizer, result))
            if fallback_seen:
                raise FallbackError("runtime/model fallback observed")

            digest = timeline_digest(result)
            previous = expected.get(input_index)
            if previous is None:
                expected[input_index] = digest
            elif previous != digest:
                raise DigestDriftError(f"timeline digest drift for input index {input_index}")
            result_duration = int(_field(result, "duration_us", duration_us) or duration_us)
            user_cpu_delta = None
            system_cpu_delta = None
            if isinstance(pre.get("user_cpu_sec"), (int, float)) and isinstance(post.get("user_cpu_sec"), (int, float)):
                user_cpu_delta = round(max(0.0, float(post["user_cpu_sec"]) - float(pre["user_cpu_sec"])), 6)
            if isinstance(pre.get("system_cpu_sec"), (int, float)) and isinstance(post.get("system_cpu_sec"), (int, float)):
                system_cpu_delta = round(max(0.0, float(post["system_cpu_sec"]) - float(pre["system_cpu_sec"])), 6)
            cpu_record = _cpu_record(cpu_before, cpu_after, cpu_delta_reader(cpu_before, cpu_after))
            if cpu_record["usage_usec"] is None and cpuacct_before is not None and cpuacct_after is not None:
                cpu_record["usage_usec"] = cpuacct_after - cpuacct_before if cpuacct_after >= cpuacct_before else None
                cpu_record["usage_source"] = "cgroup_v1_cpuacct.usage"
            else:
                cpu_record["usage_source"] = "cpu.stat" if cpu_record["usage_usec"] is not None else None
            io_delta = {
                "read_bytes": _nonnegative_delta(pre, post, "read_bytes"),
                "write_bytes": _nonnegative_delta(pre, post, "write_bytes"),
            }
            rtf = elapsed / max(1e-9, result_duration / 1_000_000)
            throttle_usec = cpu_record.get("throttled_usec")
            throttle_ratio = (float(throttle_usec) / max(1.0, elapsed * 1_000_000.0)
                              if isinstance(throttle_usec, (int, float)) else None)
            record = {
                "iteration": iteration,
                "input_index": input_index,
                "timeline_digest": digest,
                "decision": str(_field(result, "decision", "UNKNOWN")),
                "quality_status": str(_field(result, "quality_status", "UNKNOWN")),
                "counts": result_counts(result),
                "duration_us": result_duration,
                "wall_sec": round(elapsed, 6),
                "process_cpu_sec": round(process_elapsed, 6),
                "process_user_cpu_sec": user_cpu_delta,
                "process_system_cpu_sec": system_cpu_delta,
                "rtf": round(rtf, 6),
                "run_temperature": "cold" if not runs else "warm",
                "cpu": cpu_record,
                "throttled_wall_ratio": round(throttle_ratio, 8) if throttle_ratio is not None else None,
                "io": io_delta,
                "resources": {"pre": pre, "post": post,
                              "growth_bytes": _memory_growth_bytes(baseline_resources, post)},
                "gc": {"pre": gc_pre, "post": gc_post},
            }
            runs.append(record)

            if max_rtf is not None and rtf > max_rtf:
                raise RuntimeLimitError("RTF exceeded configured ceiling")
            if (max_throttled_wall_ratio is not None and throttle_ratio is not None
                    and throttle_ratio > max_throttled_wall_ratio):
                raise RuntimeLimitError("cgroup throttled wall ratio exceeded configured ceiling")
            observed_memory = _largest_memory_bytes(post)
            if max_memory_mb is not None and observed_memory is not None and observed_memory > max_memory_mb * 1024 * 1024:
                raise RuntimeLimitError("process-tree/cgroup memory exceeded configured cap")

            growth = record["resources"]["growth_bytes"]
            if rss_growth_limit_mb is not None and growth is not None and growth > rss_growth_limit_mb * 1024 * 1024:
                raise RssGrowthError("resident memory growth exceeded configured limit")
            if warm_baseline_resources is None:
                warm_baseline_resources = dict(post)
            elif warm_rss_growth_limit_percent is not None:
                warm_base = _resident_memory_bytes(warm_baseline_resources)
                warm_now = _resident_memory_bytes(post)
                if warm_base and warm_now is not None:
                    warm_growth_percent = max(0.0, (warm_now - warm_base) * 100.0 / warm_base)
                    warm_resident_growth_percent_max = max(
                        warm_resident_growth_percent_max, warm_growth_percent
                    )
                    if warm_growth_percent > warm_rss_growth_limit_percent:
                        raise RssGrowthError("warm process-tree/cgroup memory growth exceeded configured percent")

    final_resources = _resource_snapshot(status_reader=status_reader, smaps_reader=smaps_reader,
                                         process_tree_reader=process_tree_reader,
                                         memory_reader=memory_reader, rusage_reader=rusage_reader)
    target_evidence_missing: list[str] = []
    if evidence_mode == "target":
        if any(run["cpu"]["usage_usec"] is None for run in runs):
            target_evidence_missing.append("cgroup_cpu_usage")
        if any(run["resources"]["post"].get("process_tree_rss_bytes") is None for run in runs):
            target_evidence_missing.append("process_tree_rss")
        if any(run["resources"]["post"].get("process_tree_pss_bytes") is None for run in runs):
            target_evidence_missing.append("process_tree_pss")
        if any(run["io"].get("read_bytes") is None for run in runs):
            target_evidence_missing.append("process_tree_io")
        if target_evidence_missing:
            raise RuntimeLimitError("required target evidence unavailable: " + ",".join(target_evidence_missing))
    return {
        "schema": "sddiar.repeated_worker_result_v1",
        "decision": "PASS",
        "input_count": len(paths),
        "repetitions": repetitions,
        "run_count": len(runs),
        "processed_audio_us": processed_audio_us,
        "config_sha256": config_digest(config),
        "worker": {"diarizer_instance_count": 1, "reused_diarizer": repetitions * len(paths) > 1,
                    "session_backend_reuse": reuse},
        "runtime_gates": {"evidence_mode": evidence_mode, "require_cgroup_version": require_cgroup_version,
                           "max_rtf": max_rtf, "max_memory_mb": max_memory_mb,
                           "max_throttled_wall_ratio": max_throttled_wall_ratio,
                           "min_total_audio_minutes": min_total_audio_minutes,
                           "target_evidence_missing": target_evidence_missing},
        "memory_gate": {"rss_growth_limit_mb": rss_growth_limit_mb,
                         "warm_rss_growth_limit_percent": warm_rss_growth_limit_percent,
                         "baseline": baseline_resources, "final": final_resources,
                         "warm_baseline": warm_baseline_resources,
                         "warm_resident_growth_percent_max": round(
                             warm_resident_growth_percent_max, 6
                         ),
                         "max_growth_bytes": max((r["resources"]["growth_bytes"] or 0) for r in runs)},
        "runs": runs,
        "redaction": {"source_identifiers": "omitted", "raw_content": "omitted"},
        "environment": {"platform": sys.platform, "architecture": platform.machine(),
                         "python": platform.python_version(), "network": "none_required"},
    }


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent offline one-CPU diarizer benchmark")
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--silero-model", required=True, type=Path)
    parser.add_argument("--silero-sha256", required=True)
    parser.add_argument("--wespeaker-model", required=True, type=Path)
    parser.add_argument("--wespeaker-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--rss-growth-limit-mb", type=float, default=64.0)
    parser.add_argument("--warm-rss-growth-limit-percent", type=float, default=10.0)
    parser.add_argument("--max-memory-mb", type=float, default=256.0)
    parser.add_argument("--max-rtf", type=float, default=0.35)
    parser.add_argument("--max-throttled-wall-ratio", type=float, default=0.01)
    parser.add_argument("--min-total-audio-minutes", type=float, default=60.0)
    parser.add_argument("--evidence-mode", choices=("local_proxy", "target"), default="local_proxy")
    parser.add_argument("--require-cgroup-version", choices=("v1", "v2"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--assignment-distance-limit", type=float, default=0.5)
    parser.add_argument("--silero-temporal-postprocess", action="store_true")
    parser.add_argument("--auto-gain-normalization", action="store_true")
    args = parser.parse_args(argv)
    if args.threads != 1:
        raise QuotaMismatchError("the repeated one-CPU worker requires --threads=1")
    diarizer_config = LocalOnnxDiarizationConfig(
        assignment_distance_limit=args.assignment_distance_limit,
        silero_temporal_postprocess=args.silero_temporal_postprocess,
        auto_gain_normalization=args.auto_gain_normalization,
    )
    result = run_repeated_worker(
        args.audio,
        lambda: LocalOnnxDiarizer(args.silero_model, args.wespeaker_model,
                                  silero_sha256=args.silero_sha256,
                                  wespeaker_sha256=args.wespeaker_sha256,
                                  config=diarizer_config, threads=1),
        repetitions=args.repetitions,
        config={"assignment_distance_limit": args.assignment_distance_limit,
                "silero_temporal_postprocess": args.silero_temporal_postprocess,
                "auto_gain_normalization": args.auto_gain_normalization, "threads": 1},
        rss_growth_limit_mb=args.rss_growth_limit_mb,
        warm_rss_growth_limit_percent=args.warm_rss_growth_limit_percent,
        max_memory_mb=args.max_memory_mb,
        max_rtf=args.max_rtf,
        max_throttled_wall_ratio=args.max_throttled_wall_ratio,
        min_total_audio_minutes=args.min_total_audio_minutes,
        evidence_mode=args.evidence_mode,
        require_cgroup_version=args.require_cgroup_version,
    )
    atomic_publish(args.output, result)
    print(json.dumps({"schema": result["schema"], "run_count": result["run_count"],
                      "decision": result["decision"], "config_sha256": result["config_sha256"]},
                     sort_keys=True))
    return 0


# Keep the same entry-point spelling as ``run_benchmark.py`` for callers that
# import benchmark modules instead of invoking the script.
main = _cli


if __name__ == "__main__":
    try:
        raise SystemExit(_cli())
    except RepeatedWorkerError as exc:
        print(f"repeated-worker gate failed: {exc}", file=sys.stderr)
        raise SystemExit(2)


__all__ = [
    "DigestDriftError", "FallbackError", "QuotaMismatchError", "RepeatedWorkerError", "RssGrowthError",
    "RuntimeLimitError", "canonical_timeline", "config_digest", "read_cgroup_cpuacct_usage_usec",
    "read_cgroup_memory", "read_proc_status", "read_process_tree_resources", "read_smaps_rollup",
    "read_wav_duration_us", "result_counts", "run_repeated_worker", "timeline_digest", "main",
]
