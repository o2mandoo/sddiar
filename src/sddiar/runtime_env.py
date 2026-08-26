"""Read-only CPU limits and throttling counters for local benchmarks.

The process may see many host CPUs while its cgroup is limited to a fraction
of one CPU.  This module deliberately reads the kernel's cgroup files instead
of using ``os.cpu_count()`` as a resource estimate.  It has no dependencies,
does not execute commands, and treats unavailable or malformed files as
unknown values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import sys
from types import MappingProxyType
from typing import Mapping, Optional, Union


PathLike = Union[str, os.PathLike[str]]


def _int(value: str) -> Optional[int]:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _read_int(path: Path) -> Optional[int]:
    text = _read(path)
    return _int(text) if text is not None else None


def _parse_key_values(text: Optional[str]) -> Mapping[str, int]:
    values: dict[str, int] = {}
    if text is None:
        return MappingProxyType(values)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            value = _int(parts[1])
            if value is not None:
                values[parts[0]] = value
    return MappingProxyType(values)


def _parse_cpuset(text: Optional[str]) -> Optional[tuple[int, ...]]:
    """Parse Linux cpuset syntax (for example ``0-3,8``) safely."""
    if text is None:
        return None
    cpus: set[int] = set()
    try:
        for item in text.replace("\n", "").split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start_s, end_s = item.split("-", 1)
                start, end = int(start_s), int(end_s)
                if start < 0 or end < start:
                    return None
                cpus.update(range(start, end + 1))
            else:
                cpu = int(item)
                if cpu < 0:
                    return None
                cpus.add(cpu)
    except (TypeError, ValueError):
        return None
    return tuple(sorted(cpus))


@dataclass(frozen=True)
class CgroupCpuStat:
    """A normalized subset of ``cpu.stat`` plus all parsed counters.

    ``throttled_usec`` is normalized across cgroup versions.  In cgroup v1,
    the kernel calls the source field ``throttled_time`` and reports it in
    nanoseconds; ``throttled_time_ns`` retains that original counter.
    """

    usage_usec: Optional[int] = None
    user_usec: Optional[int] = None
    system_usec: Optional[int] = None
    nr_periods: Optional[int] = None
    nr_throttled: Optional[int] = None
    throttled_usec: Optional[int] = None
    throttled_time_ns: Optional[int] = None
    values: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_text(cls, text: Optional[str]) -> Optional["CgroupCpuStat"]:
        if text is None:
            return None
        values = _parse_key_values(text)
        throttled_time_ns = values.get("throttled_time")
        throttled_usec = values.get("throttled_usec")
        if throttled_usec is None and throttled_time_ns is not None:
            throttled_usec = throttled_time_ns // 1000
        return cls(
            usage_usec=values.get("usage_usec"),
            user_usec=values.get("user_usec"),
            system_usec=values.get("system_usec"),
            nr_periods=values.get("nr_periods"),
            nr_throttled=values.get("nr_throttled"),
            throttled_usec=throttled_usec,
            throttled_time_ns=throttled_time_ns,
            values=values,
        )


@dataclass(frozen=True)
class RuntimeCpuSnapshot:
    """One point-in-time, read-only view of the process CPU cgroup."""

    platform: str
    cgroup_version: Optional[str] = None
    cgroup_path: Optional[Path] = None
    quota_us: Optional[int] = None
    period_us: Optional[int] = None
    cpuset_cpus: Optional[tuple[int, ...]] = None
    cpu_stat: Optional[CgroupCpuStat] = None
    unavailable_reason: Optional[str] = None

    @property
    def cpuset_cpu_count(self) -> Optional[int]:
        return len(self.cpuset_cpus) if self.cpuset_cpus is not None else None

    @property
    def effective_cpu_equivalent(self) -> Optional[float]:
        """CPU-equivalent after applying quota and cpuset limits.

        An unlimited quota is represented by ``None``.  If neither a quota
        nor a cpuset is readable, this remains unknown rather than falling
        back to the host's (often misleading) logical CPU count.
        """
        quota: Optional[float] = None
        if self.quota_us is not None and self.period_us and self.period_us > 0:
            if self.quota_us >= 0:
                quota = self.quota_us / self.period_us
        cpuset = self.cpuset_cpu_count
        if quota is None:
            return float(cpuset) if cpuset is not None else None
        if cpuset is None:
            return quota
        return min(quota, float(cpuset))


@dataclass(frozen=True)
class CgroupCpuStatDelta:
    """Non-negative counter differences between two snapshots.

    A field is ``None`` when it was not present in both snapshots or when a
    cgroup counter reset/decreased between reads (for example after a move).
    """

    usage_usec: Optional[int] = None
    user_usec: Optional[int] = None
    system_usec: Optional[int] = None
    nr_periods: Optional[int] = None
    nr_throttled: Optional[int] = None
    throttled_usec: Optional[int] = None
    reset_detected: bool = False
    values: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeCpuDelta:
    """Delta suitable for attaching to a benchmark result."""

    cpu_stat: Optional[CgroupCpuStatDelta]
    effective_cpu_equivalent: Optional[float]
    cpuset_cpu_count: Optional[int]
    cgroup_version: Optional[str]


def _counter_delta(before: Optional[int], after: Optional[int]) -> tuple[Optional[int], bool]:
    if before is None or after is None:
        return None, False
    if after < before:
        return None, True
    return after - before, False


def delta_cpu_stat(before: Optional[CgroupCpuStat], after: Optional[CgroupCpuStat]) -> Optional[CgroupCpuStatDelta]:
    if before is None or after is None:
        return None
    names = ("usage_usec", "user_usec", "system_usec", "nr_periods", "nr_throttled", "throttled_usec")
    deltas: dict[str, Optional[int]] = {}
    reset = False
    for name in names:
        value, did_reset = _counter_delta(getattr(before, name), getattr(after, name))
        deltas[name] = value
        reset = reset or did_reset
    raw: dict[str, int] = {}
    for key in set(before.values).intersection(after.values):
        value, did_reset = _counter_delta(before.values[key], after.values[key])
        reset = reset or did_reset
        if value is not None:
            raw[key] = value
    return CgroupCpuStatDelta(**deltas, reset_detected=reset, values=MappingProxyType(raw))


def delta_cpu_snapshots(before: RuntimeCpuSnapshot, after: RuntimeCpuSnapshot) -> RuntimeCpuDelta:
    """Return cgroup counter deltas without raising on unavailable data."""
    return RuntimeCpuDelta(
        cpu_stat=delta_cpu_stat(before.cpu_stat, after.cpu_stat),
        effective_cpu_equivalent=after.effective_cpu_equivalent,
        cpuset_cpu_count=after.cpuset_cpu_count,
        cgroup_version=after.cgroup_version,
    )


def _proc_cgroup(text: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (v2 relative path, v1 CPU path, v1 cpuset path)."""
    v2_path: Optional[str] = None
    cpu_path: Optional[str] = None
    cpuset_path: Optional[str] = None
    if text is None:
        return None, None, None
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, path = fields
        controller_set = set(controllers.split(","))
        if not controllers:
            v2_path = path
        elif "cpu" in controller_set or "cpuacct" in controller_set:
            cpu_path = path
        if "cpuset" in controller_set:
            cpuset_path = path
    return v2_path, cpu_path, cpuset_path


def _under(root: Path, relative: Optional[str]) -> Path:
    if not relative or relative == "/":
        return root
    return root / relative.lstrip("/")


def _v1_controller_dir(root: Path, relative: Optional[str], controller: str) -> Optional[Path]:
    names = (controller, "cpu,cpuacct", "cpuacct") if controller == "cpu" else (controller,)
    for name in names:
        candidate = _under(root / name, relative)
        if candidate.exists():
            return candidate
    # Small test fixtures and some bind-mounted layouts put controller files
    # directly below the supplied root.
    direct = _under(root, relative)
    if any((direct / file).exists() for file in ("cpu.cfs_quota_us", "cpu.cfs_period_us", "cpu.stat")):
        return direct
    return None


def read_cpu_snapshot(
    *,
    cgroup_root: PathLike = "/sys/fs/cgroup",
    proc_cgroup_path: PathLike = "/proc/self/cgroup",
    platform_name: Optional[str] = None,
) -> RuntimeCpuSnapshot:
    """Capture the current cgroup CPU limits using only filesystem reads."""
    platform = platform_name or sys.platform
    if not platform.startswith("linux"):
        return RuntimeCpuSnapshot(platform=platform, unavailable_reason="non_linux")

    root = Path(cgroup_root)
    proc_text = _read(Path(proc_cgroup_path))
    if proc_text is None or not root.exists():
        return RuntimeCpuSnapshot(platform=platform, unavailable_reason="cgroup_unavailable")

    v2_rel, v1_rel, cpuset_rel = _proc_cgroup(proc_text)
    v2 = (root / "cgroup.controllers").exists()
    if v2:
        base = _under(root, v2_rel)
        max_text = _read(base / "cpu.max")
        quota_us: Optional[int] = None
        period_us: Optional[int] = None
        if max_text is not None:
            parts = max_text.split()
            if len(parts) >= 2:
                period_us = _int(parts[1])
                quota_us = None if parts[0] == "max" else _int(parts[0])
        cpuset = _parse_cpuset(_read(base / "cpuset.cpus.effective"))
        if cpuset is None:
            cpuset = _parse_cpuset(_read(base / "cpuset.cpus"))
        stat = CgroupCpuStat.from_text(_read(base / "cpu.stat"))
        return RuntimeCpuSnapshot(platform, "v2", base, quota_us, period_us, cpuset, stat)

    if v1_rel is not None or cpuset_rel is not None:
        cpu_dir = _v1_controller_dir(root, v1_rel, "cpu") if v1_rel is not None else None
        cpuset_dir = _v1_controller_dir(root, cpuset_rel, "cpuset") if cpuset_rel is not None else None
        if cpu_dir is not None or cpuset_dir is not None:
            quota = _read_int(cpu_dir / "cpu.cfs_quota_us") if cpu_dir else None
            period = _read_int(cpu_dir / "cpu.cfs_period_us") if cpu_dir else None
            cpuset = _parse_cpuset(_read(cpuset_dir / "cpuset.cpus")) if cpuset_dir else None
            stat = CgroupCpuStat.from_text(_read(cpu_dir / "cpu.stat")) if cpu_dir else None
            return RuntimeCpuSnapshot(platform, "v1", cpu_dir or cpuset_dir, quota, period, cpuset, stat)

    return RuntimeCpuSnapshot(platform=platform, unavailable_reason="cgroup_unavailable")


# Short aliases make the API convenient for benchmark harnesses while keeping
# the implementation's canonical name explicit.
capture_cpu_snapshot = read_cpu_snapshot
cpu_snapshot_delta = delta_cpu_snapshots


__all__ = [
    "CgroupCpuStat",
    "RuntimeCpuSnapshot",
    "CgroupCpuStatDelta",
    "RuntimeCpuDelta",
    "read_cpu_snapshot",
    "capture_cpu_snapshot",
    "delta_cpu_stat",
    "delta_cpu_snapshots",
    "cpu_snapshot_delta",
]
