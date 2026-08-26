"""Offline, default-off oracle harness for a draft/refiner STT cascade.

This module is an evaluation contract, not an STT runtime.  Callers provide
time-only segment metrics (or redacted character counts) that were computed
elsewhere.  The harness never receives, stores, or emits transcript text and
does not download or execute a model.  Its result is deliberately marked
``REVIEW_REQUIRED`` and has no release authority.

The artifact pack in this module follows the same fail-closed shape as the
production local-STT identity, but is intentionally not imported by the
production orchestrator.  It exists to make an experimental VAD/OpenVINO/
CT2 comparison reproducible without granting those artifacts runtime
authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .offline import reject_url


SCHEMA = "stt_cascade_oracle_v2"
QUALITY_STATUS = "REVIEW_REQUIRED"
RELEASE_AUTHORITY = "none"
DEFAULT_BUDGETS: tuple[int, ...] = (10, 20, 30, 40)
DEFAULT_MAX_SEGMENTS = 256
DEFAULT_MAX_TOTAL_DURATION_US = 4 * 60 * 60 * 1_000_000
DEFAULT_MAX_SOLVER_STATES = 100_000
MAX_MITM_CANDIDATES = 28
MAX_CLI_INPUT_BYTES = 16 * 1024 * 1024
MAX_STT_PACK_ARTIFACTS = 128
MAX_STT_PACK_FILES = 10_000
MAX_STT_PACK_TREE_ENTRIES = 20_000
MAX_STT_PACK_BYTES = 8 * 1024 * 1024 * 1024
SUPPORTED_PACK_STRATEGIES = frozenset({"whispercpp_openvino", "ctranslate2"})
SUPPORTED_ERROR_METRICS = frozenset({"character_edit_count", "word_edit_count"})
REQUIRED_ARTIFACT_ROLES = frozenset({"vad", "openvino", "ct2"})  # legacy names only
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")
_PACK_TOKEN = object()


class SttCascadeExperimentalError(ValueError):
    """Base error for malformed experimental input."""


class SttCascadeContractError(SttCascadeExperimentalError):
    """Input does not satisfy the timing/metric contract."""


class SttCascadeArtifactError(SttCascadeExperimentalError):
    """A local experiment artifact is missing, mutable, or hash-invalid."""


def _freeze(value: Any) -> Any:
    """Recursively freeze public identity data."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze(v) for v in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"non-JSON identity value: {type(value).__name__}")


def _sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    before = path.stat()
    if max_bytes is not None and before.st_size > max_bytes:
        raise SttCascadeArtifactError("artifact byte resource bound exceeded")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
            if max_bytes is not None and total > max_bytes:
                raise SttCascadeArtifactError("artifact grew beyond byte resource bound")
            digest.update(block)
    after = path.stat()
    if (
        total != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or getattr(after, "st_ino", None) != getattr(before, "st_ino", None)
    ):
        raise SttCascadeArtifactError("artifact changed while hashing")
    return digest.hexdigest()


def _number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SttCascadeContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise SttCascadeContractError(f"{name} must be a finite non-negative number")
    return result


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise SttCascadeContractError(f"{name} must be a non-negative integer count")
    return value


def _time(row: Mapping[str, Any], stem: str) -> int:
    micros = row.get(f"{stem}_us")
    if micros is not None:
        if isinstance(micros, bool) or not isinstance(micros, int):
            raise SttCascadeContractError(f"{stem}_us must be an integer")
        value = micros
    else:
        seconds = row.get(f"{stem}_sec", row.get(stem))
        if seconds is None:
            raise SttCascadeContractError(f"segment has no {stem} time")
        value = round(_number(seconds, stem, nonnegative=True) * 1_000_000)
    return value


def _first(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _metric(row: Mapping[str, Any], side: str, reference_chars: float | None) -> tuple[int, float | None]:
    """Return an externally calculated error and an optional length count.

    Character-count difference is not an ASR error metric: equal-length text
    can be completely wrong, while different-length text can be correct after
    normalization.  Therefore an explicit error is mandatory.  Redacted
    counts remain optional stitch-risk diagnostics and never drive selection.
    """
    error = _first(row, (f"{side}_error_count", f"{side}_edit_count"))
    count_value = _first(row, (
        f"{side}_redacted_chars", f"{side}_redacted_char_count",
        f"{side}_chars", f"{side}_char_count",
    ))
    count = None if count_value is None else _number(count_value, f"{side} count")
    if error is None:
        raise SttCascadeContractError(f"segment needs an externally calculated {side}_error_count")
    return _count(error, f"{side} error"), count


@dataclass(frozen=True, slots=True)
class SttCascadeSegment:
    """One time-ordered, text-free segment observation."""

    start_us: int
    end_us: int
    draft_error: int
    refiner_error: int
    draft_chars: float | None = field(default=None, repr=False, compare=False)
    refiner_chars: float | None = field(default=None, repr=False, compare=False)
    reference_chars: float | None = field(default=None, repr=False, compare=False)
    hard: bool | None = field(default=None, repr=False, compare=False)
    duplicate_risk: float | None = field(default=None, repr=False, compare=False)
    deletion_risk: float | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.start_us) is not int or type(self.end_us) is not int or self.start_us < 0 or self.end_us <= self.start_us:
            raise SttCascadeContractError("segment timing must satisfy 0 <= start_us < end_us")
        for name in ("draft_error", "refiner_error"):
            _count(getattr(self, name), name)
        for name in ("draft_chars", "refiner_chars", "reference_chars", "duplicate_risk", "deletion_risk"):
            value = getattr(self, name)
            if value is not None:
                _number(value, name)
        for name in ("duplicate_risk", "deletion_risk"):
            value = getattr(self, name)
            if value is not None and value > 1:
                raise SttCascadeContractError(f"{name} must be between 0 and 1")

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SttCascadeSegment":
        if not isinstance(row, Mapping):
            raise SttCascadeContractError("each segment must be an object")
        # ``segment_*`` is accepted for callers that keep the segment prefix
        # on all timing fields; it is still reduced to source timing only.
        start_row = row if "start_us" in row or "start" in row or "start_sec" in row else {"start_us": row.get("segment_start_us")}
        end_row = row if "end_us" in row or "end" in row or "end_sec" in row else {"end_us": row.get("segment_end_us")}
        start, end = _time(start_row, "start"), _time(end_row, "end")
        reference_count_value = _first(row, (
            "reference_redacted_chars", "reference_redacted_char_count",
            "reference_chars", "reference_char_count", "ref_chars",
        ))
        reference_count = None if reference_count_value is None else _number(reference_count_value, "reference count")
        draft_error, draft_count = _metric(row, "draft", reference_count)
        refiner_error, refiner_count = _metric(row, "refiner", reference_count)
        hard_value = row.get("hard", row.get("hard_segment", row.get("router_hard")))
        hard = None if hard_value is None else (bool(hard_value) if type(hard_value) is bool else bool(_number(hard_value, "hard")))
        duplicate = _first(row, ("stitch_duplicate_risk", "duplicate_risk"))
        deletion = _first(row, ("stitch_deletion_risk", "deletion_risk"))
        return cls(
            start, end, draft_error, refiner_error,
            draft_count, refiner_count, reference_count, hard,
            None if duplicate is None else _number(duplicate, "duplicate risk"),
            None if deletion is None else _number(deletion, "deletion risk"),
        )


def _segments(value: Any, *, max_segments: int) -> tuple[SttCascadeSegment, ...]:
    rows = value
    if isinstance(value, Mapping):
        rows = _first(value, ("segments", "observations", "rows"))
    if not isinstance(rows, (list, tuple)) or not rows:
        raise SttCascadeContractError("segments must be a non-empty array")
    if len(rows) > max_segments:
        raise SttCascadeContractError("segment resource bound exceeded")
    parsed = tuple(SttCascadeSegment.from_mapping(row) for row in rows)
    ordered = tuple(sorted(parsed, key=lambda item: (item.start_us, item.end_us)))
    if any(right.start_us < left.end_us for left, right in zip(ordered, ordered[1:])):
        raise SttCascadeContractError("oracle core segments must not overlap")
    return ordered


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _risk(rows: Sequence[SttCascadeSegment]) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    rows = tuple(sorted(rows, key=lambda row: (row.start_us, row.end_us)))
    explicit_dup = [row.duplicate_risk for row in rows if row.duplicate_risk is not None]
    explicit_del = [row.deletion_risk for row in rows if row.deletion_risk is not None]
    overlap = sum(max(0, min(left.end_us, right.end_us) - max(left.start_us, right.start_us)) for left, right in zip(rows, rows[1:]))
    span = sum(row.duration_us for row in rows)
    duplicate = max(explicit_dup, default=_ratio(overlap, span))
    with_counts = [row for row in rows if row.reference_chars is not None and row.refiner_chars is not None]
    if with_counts:
        missing = sum(max(0.0, row.reference_chars - row.refiner_chars) for row in with_counts)
        expected = sum(row.reference_chars for row in with_counts)
        deletion = _ratio(missing, expected)
    else:
        # A refiner regression is the only text-free deletion proxy available
        # when callers supplied scalar errors rather than counts.
        deletion = _ratio(sum(max(0.0, row.refiner_error - row.draft_error) for row in rows), sum(max(row.draft_error, row.refiner_error) for row in rows))
    if explicit_del:
        deletion = max(deletion, max(explicit_del))
    return min(1.0, max(0.0, duplicate)), min(1.0, max(0.0, deletion))


def _budget_values(budgets: Iterable[int | float]) -> tuple[int, ...]:
    normalized: list[int] = []
    for item in budgets:
        if isinstance(item, float) and 0 < item < 1:
            item *= 100
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise SttCascadeContractError("budgets must be percentages in 1..100")
        if float(item) != int(item):
            raise SttCascadeContractError("budgets must be whole percentages")
        normalized.append(int(item))
    result = tuple(normalized)
    if not result or any(item <= 0 or item > 100 for item in result) or tuple(sorted(set(result))) != result:
        raise SttCascadeContractError("budgets must be strictly increasing percentages in 1..100")
    return result


def _better_subset(left: tuple[float, tuple[int, ...]], right: tuple[float, tuple[int, ...]]) -> tuple[float, tuple[int, ...]]:
    """Stable maximum-gain/timing-independent subset tie break."""
    if left[0] > right[0] + 1e-12:
        return left
    if right[0] > left[0] + 1e-12:
        return right
    return left if left[1] < right[1] else right


def _knapsack_mitm(rows: Sequence[SttCascadeSegment], capacity: int) -> tuple[int, ...]:
    """Exact bounded subset solver for the normal 20--30 segment workload."""
    candidates = tuple((index, row) for index, row in enumerate(rows) if row.draft_error > row.refiner_error)
    if not candidates or capacity <= 0:
        return ()
    split = len(candidates) // 2

    def enumerate_half(items: Sequence[tuple[int, SttCascadeSegment]]) -> list[tuple[int, float, tuple[int, ...]]]:
        out: list[tuple[int, float, tuple[int, ...]]] = []
        for mask in range(1 << len(items)):
            duration, gain, selected = 0, 0.0, []
            for offset, (index, row) in enumerate(items):
                if mask & (1 << offset):
                    duration += row.duration_us
                    if duration > capacity:
                        break
                    gain += row.draft_error - row.refiner_error
                    selected.append(index)
            else:
                out.append((duration, gain, tuple(selected)))
        return out

    right = enumerate_half(candidates[split:])
    right.sort(key=lambda item: (item[0], item[2]))
    durations: list[int] = []
    prefixes: list[tuple[float, tuple[int, ...]]] = []
    best = (0.0, ())
    for duration, gain, selected in right:
        if not durations or duration != durations[-1]:
            durations.append(duration)
            best = _better_subset(best, (gain, selected))
            prefixes.append(best)
        else:
            best = _better_subset(best, (gain, selected))
            prefixes[-1] = best
    import bisect
    result = (0.0, ())
    for duration, gain, selected in enumerate_half(candidates[:split]):
        remaining = capacity - duration
        position = bisect.bisect_right(durations, remaining) - 1
        if position >= 0:
            right_gain, right_selected = prefixes[position]
            result = _better_subset(result, (gain + right_gain, tuple(sorted(selected + right_selected))))
    return result[1]


def _knapsack_sparse(rows: Sequence[SttCascadeSegment], capacity: int, max_states: int) -> tuple[int, ...]:
    """Exact DP fallback with an explicit state bound for larger inputs."""
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, row in enumerate(rows):
        gain = row.draft_error - row.refiner_error
        if gain <= 0:
            continue
        updates: dict[int, tuple[float, tuple[int, ...]]] = {}
        for duration, (old_gain, selected) in states.items():
            new_duration = duration + row.duration_us
            if new_duration > capacity:
                continue
            candidate = (old_gain + gain, selected + (index,))
            updates[new_duration] = _better_subset(updates.get(new_duration, (-1.0, ())), candidate)
        for duration, candidate in updates.items():
            states[duration] = _better_subset(states.get(duration, (-1.0, ())), candidate)
        if len(states) > max_states:
            raise SttCascadeContractError("oracle solver state bound exceeded")
    return max(states.values(), key=lambda item: (item[0], tuple(-index for index in item[1])))[1]


def _duration_selection(rows: Sequence[SttCascadeSegment], capacity: int, max_states: int) -> tuple[int, ...]:
    positive = sum(1 for row in rows if row.draft_error > row.refiner_error)
    # Meet-in-the-middle enumerates 2^(N/2) subsets. Keep that branch under
    # an explicit cap; larger inputs use the state-bounded sparse solver.
    if positive <= MAX_MITM_CANDIDATES:
        return _knapsack_mitm(rows, capacity)
    return _knapsack_sparse(rows, capacity, max_states)


def _curve_row(rows: Sequence[SttCascadeSegment], selected_indices: Sequence[int], percent: int, target: int, draft_error: int, full_error: int, hard_rows: Sequence[SttCascadeSegment], total_duration: int, reference_units: int) -> dict[str, Any]:
    selected = tuple(rows[index] for index in selected_indices)
    selected_duration = sum(row.duration_us for row in selected)
    selected_gain = sum(row.draft_error - row.refiner_error for row in selected)
    oracle_error = draft_error - selected_gain
    selected_hard = sum(1 for row in selected if row in hard_rows)
    selected_dup, selected_del = _risk(selected)
    return {
        "budget_percent": percent,
        "target_duration_us": target,
        "selected_segment_count": len(selected),
        "selected_duration_us": selected_duration,
        "refined_duration_ratio": _ratio(selected_duration, total_duration),
        "oracle_error_count": oracle_error,
        "oracle_error_rate": _ratio(oracle_error, reference_units),
        "delta_vs_draft_error_count": oracle_error - draft_error,
        "delta_vs_draft_error_rate": _ratio(oracle_error - draft_error, reference_units),
        "delta_vs_full_refiner_error_count": oracle_error - full_error,
        "delta_vs_full_refiner_error_rate": _ratio(oracle_error - full_error, reference_units),
        "router_hard_segment_recall": _ratio(selected_hard, len(hard_rows)),
        "router_hard_duration_recall": _ratio(sum(row.duration_us for row in selected if row in hard_rows), sum(row.duration_us for row in hard_rows)),
        "stitch_duplicate_risk": selected_dup,
        "stitch_deletion_risk": selected_del,
        "stitch_risk": {"duplicate": selected_dup, "deletion": selected_del},
    }


def analyze_stt_cascade_oracle(value: Any, *, budgets: Iterable[int | float] = DEFAULT_BUDGETS,
                               max_segments: int = DEFAULT_MAX_SEGMENTS,
                               max_total_duration_us: int = DEFAULT_MAX_TOTAL_DURATION_US,
                               max_solver_states: int = DEFAULT_MAX_SOLVER_STATES,
                               include_segment_count_diagnostic: bool = True) -> dict[str, Any]:
    """Compute an exact duration-budget oracle upper bound.

    Each budget is a percentage of total source duration.  Only positive-gain
    segments are eligible, and a deterministic bounded knapsack selects a
    subset whose duration never exceeds the budget.  The previous count-based
    curve is retained under ``segment_count_curve`` solely as a diagnostic.
    """
    if type(max_segments) is not int or max_segments <= 0:
        raise SttCascadeContractError("segment resource bound exceeded")
    if type(max_total_duration_us) is not int or max_total_duration_us <= 0:
        raise SttCascadeContractError("max_total_duration_us must be a positive integer")
    if type(max_solver_states) is not int or max_solver_states <= 0:
        raise SttCascadeContractError("max_solver_states must be a positive integer")
    if not isinstance(value, Mapping) or value.get("metric_kind") not in SUPPORTED_ERROR_METRICS:
        raise SttCascadeContractError("metric_kind must identify additive character or word edit counts")
    metric_kind = str(value["metric_kind"])
    rows = _segments(value, max_segments=max_segments)
    budget_values = _budget_values(budgets)
    total_duration = sum(row.duration_us for row in rows)
    if total_duration > max_total_duration_us:
        raise SttCascadeContractError("total duration resource bound exceeded")
    draft_error = sum(row.draft_error for row in rows)
    reference_units = _count(value.get("reference_unit_count"), "reference_unit_count")
    if reference_units <= 0:
        raise SttCascadeContractError("reference_unit_count must be positive")
    full_error = sum(row.refiner_error for row in rows)
    marked_hard_rows = tuple(row for row in rows if row.hard is not None)
    hard_rows = tuple(row for row in marked_hard_rows if row.hard) if marked_hard_rows else tuple(row for row in rows if row.refiner_error < row.draft_error)
    full_dup, full_del = _risk(rows)
    curve: list[dict[str, Any]] = []
    for percent in budget_values:
        target = (total_duration * percent) // 100
        selected_indices = _duration_selection(rows, target, max_solver_states)
        curve.append(_curve_row(rows, selected_indices, percent, target, draft_error, full_error, hard_rows, total_duration, reference_units))
    result = {
        "schema": SCHEMA,
        "experimental": True,
        "default_enabled": False,
        "production_approved": False,
        "quality_status": QUALITY_STATUS,
        "release_authority": RELEASE_AUTHORITY,
        "privacy": {"redacted_metrics_only": True, "payload_kind": "scalar_metrics"},
        "metric_kind": metric_kind,
        "segment_count": len(rows),
        "budget_basis": "duration_us",
        "resource_bounds": {"max_segments": max_segments, "max_total_duration_us": max_total_duration_us, "max_solver_states": max_solver_states, "solver": "exact_bounded_knapsack"},
        "total_duration_us": total_duration,
        "reference_unit_count": reference_units,
        "draft_error_count": draft_error,
        "draft_error_rate": _ratio(draft_error, reference_units),
        "full_refiner_error_count": full_error,
        "full_refiner_error_rate": _ratio(full_error, reference_units),
        "full_refiner": {"error_count": full_error, "error_rate": _ratio(full_error, reference_units), "refined_duration_ratio": 1.0, "stitch_duplicate_risk": full_dup, "stitch_deletion_risk": full_del, "stitch_risk": {"duplicate": full_dup, "deletion": full_del}},
        "oracle_curve": curve,
        "limitations": [
            "oracle_selection_uses_supplied_segment_metrics",
            "only_additive_edit_error_counts_are_accepted",
            "oracle_curve_is_an_upper_bound_not_release_evidence",
            "stitch_risk_is_a_redacted_boundary_proxy",
            "no_runtime_model_or_network_execution",
        ],
    }
    if include_segment_count_diagnostic:
        ranked_indices = tuple(sorted(
            (index for index in range(len(rows)) if rows[index].draft_error > rows[index].refiner_error),
            key=lambda index: (-(rows[index].draft_error - rows[index].refiner_error), -rows[index].draft_error, rows[index].start_us, rows[index].end_us),
        ))
        result["segment_count_curve"] = [
            _curve_row(rows, ranked_indices[:math.ceil(len(rows) * percent / 100)], percent,
                       (total_duration * percent) // 100, draft_error, full_error, hard_rows, total_duration, reference_units)
            for percent in budget_values
        ]
    return result


# Short aliases make the contract convenient for notebook and CLI callers.
analyze_oracle_curve = analyze_stt_cascade_oracle
compute_oracle_curve = analyze_stt_cascade_oracle
run_oracle_curve = analyze_stt_cascade_oracle
analyze_stt_cascade = analyze_stt_cascade_oracle


@dataclass(frozen=True, slots=True)
class VerifiedExperimentalArtifact:
    """One hash-verified local file or directory; path is never public."""

    artifact_id: str
    group: str
    path: Path = field(repr=False, compare=False)
    sha256: str
    bytes: int
    file_count: int = 1

    @property
    def role(self) -> str:
        """Compatibility view; new contracts use explicit artifact groups."""
        return self.group

    def assert_unchanged(self) -> None:
        actual, size, count = _hash_local_artifact(self.path)
        if actual != self.sha256 or size != self.bytes or count != self.file_count:
            raise SttCascadeArtifactError(f"experimental {self.group} artifact changed")


@dataclass(frozen=True, slots=True, init=False)
class VerifiedLocalSttPack:
    """Sealed, strategy-aware multi-artifact identity for the experiment."""

    pack_id: str
    strategy: str
    runtime_abi: str
    platform: str
    artifacts: Mapping[str, VerifiedExperimentalArtifact]
    identity_sha256: str
    _token: object = field(repr=False, compare=False)

    def __init__(self, pack_id: str, strategy: str, runtime_abi: str, platform: str, artifacts: Mapping[str, VerifiedExperimentalArtifact], identity_sha256: str, *, _token: object | None = None) -> None:
        if _token is not _PACK_TOKEN:
            raise TypeError("VerifiedLocalSttPack must be created by verify_local_stt_pack")
        object.__setattr__(self, "pack_id", pack_id)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "runtime_abi", runtime_abi)
        object.__setattr__(self, "platform", platform)
        object.__setattr__(self, "artifacts", MappingProxyType(dict(artifacts)))
        object.__setattr__(self, "identity_sha256", identity_sha256)
        object.__setattr__(self, "_token", _token)

    def assert_artifacts_unchanged(self) -> None:
        for artifact in self.artifacts.values():
            artifact.assert_unchanged()

    def public_identity(self) -> Mapping[str, Any]:
        return _freeze({
            "schema": "verified-local-stt-pack-v2",
            "pack_id": self.pack_id,
            "strategy": self.strategy,
            "runtime_abi": self.runtime_abi,
            "platform": self.platform,
            "artifacts": {artifact_id: {"group": artifact.group, "sha256": artifact.sha256, "bytes": artifact.bytes, "file_count": artifact.file_count} for artifact_id, artifact in self.artifacts.items()},
            "identity_sha256": self.identity_sha256,
            "production_approved": False,
            "release_authority": RELEASE_AUTHORITY,
        })

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe identity without local paths."""
        return _jsonable(self.public_identity())


def _hash_local_artifact(path: Path) -> tuple[str, int, int]:
    """Hash a regular file or a symlink-free directory tree deterministically."""
    if path.is_symlink() or not path.exists():
        raise SttCascadeArtifactError("artifact must exist and cannot be a symlink")
    if path.is_file():
        size = path.stat().st_size
        if size > MAX_STT_PACK_BYTES:
            raise SttCascadeArtifactError("artifact byte resource bound exceeded")
        digest = _sha256_file(path, max_bytes=size)
        if path.stat().st_size != size:
            raise SttCascadeArtifactError("artifact changed while hashing")
        return digest, size, 1
    if not path.is_dir():
        raise SttCascadeArtifactError("artifact must be a regular file or directory")
    digest = hashlib.sha256()
    total, count = 0, 0
    children: list[Path] = []
    entry_count = 0
    for child in path.rglob("*"):
        entry_count += 1
        if entry_count > MAX_STT_PACK_TREE_ENTRIES:
            raise SttCascadeArtifactError("artifact tree entry resource bound exceeded")
        if child.is_symlink():
            raise SttCascadeArtifactError("artifact directory cannot contain symlinks")
        if child.is_file():
            children.append(child)
            if len(children) > MAX_STT_PACK_FILES:
                raise SttCascadeArtifactError("artifact file-count resource bound exceeded")
    for child in sorted(children, key=lambda item: item.relative_to(path).as_posix()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        child_size = child.stat().st_size
        if total + child_size > MAX_STT_PACK_BYTES:
            raise SttCascadeArtifactError("artifact byte resource bound exceeded")
        child_digest = _sha256_file(child, max_bytes=child_size)
        if child.stat().st_size != child_size:
            raise SttCascadeArtifactError("artifact changed while hashing")
        digest.update(relative + b"\0" + child_digest.encode("ascii") + b"\0" + str(child_size).encode("ascii") + b"\n")
        total += child_size
        count += 1
    if not count:
        raise SttCascadeArtifactError("artifact directory is empty")
    return digest.hexdigest(), total, count


def _identity_string(value: Any, name: str) -> str:
    if isinstance(value, Mapping):
        if not value or any(not isinstance(key, str) or not _SAFE_ID.fullmatch(key) for key in value):
            raise SttCascadeArtifactError(f"{name} mapping keys are unsafe")
        if any(not isinstance(item, (str, int, float, bool)) for item in value.values()):
            raise SttCascadeArtifactError(f"{name} mapping values are unsafe")
        parts = []
        for key, item in sorted(value.items()):
            rendered = str(item)
            if not _SAFE_IDENTITY.fullmatch(rendered):
                raise SttCascadeArtifactError(f"{name} mapping value is unsafe")
            parts.append(f"{key}={rendered}")
        value = ",".join(parts)
    if not isinstance(value, str) or not _SAFE_IDENTITY.fullmatch(value):
        raise SttCascadeArtifactError(f"{name} must be a safe opaque identity")
    return value


def verify_local_stt_pack(pack: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, pack_id: str | None = None,
                          strategy: str | None = None, runtime_abi: str | None = None,
                          platform: str | None = None,
                          artifact_root: str | os.PathLike[str] | None = None) -> VerifiedLocalSttPack:
    """Verify a strategy-aware local pack without loading or executing it.

    ``whispercpp_openvino`` requires engine/model/vad/encoder/runtime groups;
    ``ctranslate2`` requires engine/model/tokenizer and accepts optional vad.
    Each group may contain multiple safe artifact IDs, allowing OpenVINO XML+
    BIN and CTranslate2 directory packs to retain their complete hash.
    """
    if isinstance(pack, Mapping):
        rows = pack.get("artifacts", pack.get("files"))
        pack_id = pack.get("pack_id", pack_id)
        strategy = pack.get("strategy", strategy)
        runtime_abi = pack.get("runtime_abi", runtime_abi)
        platform = pack.get("platform", pack.get("platform_identity", platform))
        artifact_root = pack.get("artifact_root", artifact_root)
    else:
        rows = pack
    if not isinstance(pack_id, str) or not _SAFE_ID.fullmatch(pack_id):
        raise SttCascadeArtifactError("pack_id must be a safe identifier")
    if strategy not in SUPPORTED_PACK_STRATEGIES:
        raise SttCascadeArtifactError("strategy must be whispercpp_openvino or ctranslate2")
    runtime_abi = _identity_string(runtime_abi, "runtime_abi")
    platform = _identity_string(platform, "platform")
    if not isinstance(artifact_root, (str, os.PathLike)):
        raise SttCascadeArtifactError("artifact_root is required")
    try:
        reject_url(artifact_root)
    except Exception as exc:
        raise SttCascadeArtifactError("artifact_root must be local") from exc
    root_input = Path(os.path.abspath(Path(artifact_root)))
    if root_input.is_symlink() or not root_input.is_dir():
        raise SttCascadeArtifactError("artifact_root must be a regular directory")
    root = root_input.resolve()
    if root == Path(root.anchor):
        raise SttCascadeArtifactError("filesystem root cannot be an artifact_root")
    if not isinstance(rows, (list, tuple)):
        raise SttCascadeArtifactError("pack needs an artifacts array")
    if len(rows) > MAX_STT_PACK_ARTIFACTS:
        raise SttCascadeArtifactError("artifact-count resource bound exceeded")
    artifacts: dict[str, VerifiedExperimentalArtifact] = {}
    identity_rows: list[dict[str, Any]] = []
    pack_bytes = 0
    pack_files = 0
    aliases = {"voice_activity": "vad", "silero_vad": "vad", "openvino_runtime": "runtime", "stt_openvino": "engine", "ctranslate2": "engine", "c_translate2": "engine", "stt_ct2": "engine", "ct2": "engine", "ctranslate2_engine": "engine"}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SttCascadeArtifactError("artifact descriptor must be an object")
        artifact_id = row.get("artifact_id", row.get("file_id"))
        if not isinstance(artifact_id, str) or not _SAFE_ID.fullmatch(artifact_id) or artifact_id in artifacts:
            raise SttCascadeArtifactError("artifact_id must be unique and safe")
        group_value = str(row.get("group", row.get("role", ""))).lower()
        group = aliases.get(group_value, group_value)
        if group not in {"engine", "model", "vad", "encoder", "runtime", "tokenizer"}:
            raise SttCascadeArtifactError("artifact group is not supported")
        path_value = row.get("path", row.get("local_path", row.get("relative_path")))
        if not isinstance(path_value, (str, os.PathLike)):
            raise SttCascadeArtifactError(f"{group} artifact location is missing")
        try:
            reject_url(path_value)
        except Exception as exc:
            raise SttCascadeArtifactError(f"{group} artifact must be local") from exc
        supplied_path = Path(path_value)
        candidate = supplied_path if supplied_path.is_absolute() else root_input / supplied_path
        candidate = Path(os.path.abspath(candidate))
        try:
            relative = candidate.relative_to(root_input)
        except ValueError:
            raise SttCascadeArtifactError(f"{group} artifact is outside artifact_root") from None
        current = root_input
        for part in relative.parts:
            if part in {"", ".", ".."}:
                raise SttCascadeArtifactError(f"{group} artifact path is unsafe")
            current = current / part
            if current.is_symlink():
                raise SttCascadeArtifactError(f"{group} artifact path contains a symlink")
        path = candidate.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise SttCascadeArtifactError(f"{group} artifact resolves outside artifact_root") from None
        expected = str(row.get("sha256", "")).lower()
        if not _SHA256.fullmatch(expected):
            raise SttCascadeArtifactError(f"{group} artifact hash is invalid")
        try:
            actual, size, file_count = _hash_local_artifact(path)
            declared_size = None if "bytes" not in row else int(row["bytes"])
        except (OSError, TypeError, ValueError) as exc:
            raise SttCascadeArtifactError(f"{group} artifact unreadable") from exc
        if actual != expected or (declared_size is not None and declared_size != size):
            raise SttCascadeArtifactError(f"{group} artifact hash or size mismatch")
        pack_bytes += size
        pack_files += file_count
        if pack_bytes > MAX_STT_PACK_BYTES or pack_files > MAX_STT_PACK_FILES:
            raise SttCascadeArtifactError("STT pack resource bound exceeded")
        artifacts[artifact_id] = VerifiedExperimentalArtifact(artifact_id, group, path.resolve(), actual, size, file_count)
        identity_rows.append({"group": group, "artifact_id": artifact_id, "sha256": actual, "bytes": size, "file_count": file_count})
    groups = {artifact.group for artifact in artifacts.values()}
    required = {"engine", "model", "vad", "encoder", "runtime"} if strategy == "whispercpp_openvino" else {"engine", "model", "tokenizer"}
    if not required.issubset(groups):
        raise SttCascadeArtifactError(f"strategy {strategy} is missing required artifact groups")
    encoded = json.dumps({"pack_id": pack_id, "strategy": strategy, "runtime_abi": runtime_abi, "platform": platform, "artifacts": sorted(identity_rows, key=lambda row: row["artifact_id"])}, sort_keys=True, separators=(",", ":")).encode("ascii")
    identity = hashlib.sha256(encoded).hexdigest()
    return VerifiedLocalSttPack(pack_id, strategy, runtime_abi, platform, artifacts, identity, _token=_PACK_TOKEN)


__all__ = [
    "DEFAULT_BUDGETS", "DEFAULT_MAX_SEGMENTS", "DEFAULT_MAX_TOTAL_DURATION_US", "DEFAULT_MAX_SOLVER_STATES", "MAX_MITM_CANDIDATES", "MAX_CLI_INPUT_BYTES", "MAX_STT_PACK_ARTIFACTS", "MAX_STT_PACK_FILES", "MAX_STT_PACK_TREE_ENTRIES", "MAX_STT_PACK_BYTES", "SUPPORTED_PACK_STRATEGIES", "SUPPORTED_ERROR_METRICS", "REQUIRED_ARTIFACT_ROLES", "SCHEMA", "QUALITY_STATUS",
    "SttCascadeExperimentalError", "SttCascadeContractError", "SttCascadeArtifactError",
    "SttCascadeSegment", "analyze_stt_cascade_oracle", "analyze_stt_cascade", "analyze_oracle_curve", "compute_oracle_curve", "run_oracle_curve",
    "VerifiedExperimentalArtifact", "VerifiedLocalSttPack", "verify_local_stt_pack",
]
