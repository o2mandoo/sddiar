"""Offline, bounded builder for a sealed diarization annotation pack.

The pack has two deliberately different views:

* ``manifest.json`` is an evaluator artifact.  It contains source-time,
  category, audit, and hash facts and is verified with an externally retained
  SHA-256 digest.
* ``annotator/manifest.json`` and ``annotator/label_template.jsonl`` are the
  blind handoff.  They contain only shuffled opaque IDs, clip-relative times,
  and empty label fields.  The audio exists exactly once below
  ``annotator/clips`` and is referenced by both views.

No network, subprocess, model, transcript, or speaker identity is needed.
All input artifacts are copied through a single ``O_NOFOLLOW`` file
descriptor into a private snapshot before they are parsed or decoded.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import wave
from typing import Any, Iterable, Mapping, Sequence

from .media import MediaError, WavPcmAccessor


SCHEMA_VERSION = "sddiar-blind-annotation/v2"
ANNOTATOR_SCHEMA_VERSION = "sddiar-blind-annotation-annotator/v1"
DEFAULT_SEED = 20260826
DEFAULT_CLIP_SECONDS = 10.0
DEFAULT_UNIFORM_COUNT = 24
DEFAULT_BOUNDARY_COUNT = 12
DEFAULT_STRESS_COUNT = 12
EXPECTED_CLIP_COUNT = DEFAULT_UNIFORM_COUNT + DEFAULT_BOUNDARY_COUNT + DEFAULT_STRESS_COUNT
MAX_REFERENCE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
MAX_CLIP_FRAMES = 2_000_000
COPY_CHUNK_FRAMES = 16_384
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 200_000
MAX_JSON_ROW_KEYS = 48
MAX_EVENTS = 100_000
MAX_RTTM_LINE_BYTES = 8 * 1024
_PATH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

CATEGORY_UNIFORM = "UNIFORM_RANDOM_NONOVERLAP_REMAINDER"
CATEGORY_BOUNDARY = "REFERENCE_BOUNDARY_TIME_ONLY"
CATEGORY_STRESS = "SYSTEM_DISAGREEMENT_UNKNOWN_STRESS"
CATEGORIES = (CATEGORY_UNIFORM, CATEGORY_BOUNDARY, CATEGORY_STRESS)
CATEGORY_COUNTS = {CATEGORY_UNIFORM: 24, CATEGORY_BOUNDARY: 12, CATEGORY_STRESS: 12}
TEMPLATE_SEGMENT_LABELS = (
    "HUMAN_SPK_0",
    "HUMAN_SPK_1",
    "SILENCE",
    "OVERLAP",
    "UNCLEAR",
)
EVALUATOR_TOP_KEYS = frozenset({"schema_version", "source", "provenance", "policy", "selection", "clips"})
ANNOTATOR_TOP_KEYS = frozenset({"schema_version", "clips"})
ANNOTATOR_ROW_KEYS = frozenset({"annotation_id", "audio", "audio_sha256", "duration_us"})
TEMPLATE_ROW_KEYS = frozenset({
    "annotation_id", "audio", "clip_relative_start_us", "clip_relative_end_us",
    "segments", "change_boundaries_us", "allowed_segment_labels", "allowed_boundary_value",
})


class BlindAnnotationError(ValueError):
    """Raised when the blind-pack contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class _TimingEvent:
    start_us: int
    end_us: int
    marker: str | None = None


@dataclass(frozen=True, slots=True)
class ClipSpec:
    internal_index: int
    category: str
    start_frame: int
    end_frame: int
    source_time_start_us: int
    source_time_end_us: int
    audit_slot: str | None


@dataclass(frozen=True, slots=True)
class BlindPackResult:
    """Paths are local handoff handles; ``evidence`` contains no paths."""

    manifest_path: Path
    label_template_path: Path
    clip_paths: tuple[Path, ...]
    evidence: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.evidence)


def _sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _reject_dotdot(path_value: str | os.PathLike[str], label: str) -> Path:
    raw = os.fspath(path_value)
    if not isinstance(raw, str) or "\x00" in raw:
        raise BlindAnnotationError(f"{label} contains an invalid path")
    path = Path(raw)
    if not path.is_absolute():
        raise BlindAnnotationError(f"{label} must be an absolute path")
    if ".." in path.parts:
        raise BlindAnnotationError(f"{label} may not contain '..'")
    return path


def _reject_existing_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                # macOS exposes the system temporary tree through the
                # conventional /var alias.  Treat that fixed OS alias as a
                # canonicalization detail, while still rejecting every
                # user-created component below it.
                if current in {Path("/var"), Path("/tmp")}:  # pragma: no cover - platform-specific
                    continue
                raise BlindAnnotationError(f"{label} contains a symlink component")
        except OSError as exc:
            raise BlindAnnotationError(f"cannot inspect {label}") from exc


def _regular_file(path_value: str | os.PathLike[str], label: str, *, max_bytes: int) -> Path:
    path = _reject_dotdot(path_value, label)
    _reject_existing_symlink_components(path, label)
    if path.is_symlink() or not path.is_file():
        raise BlindAnnotationError(f"{label} must be a regular non-symlink file")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise BlindAnnotationError(f"cannot stat {label}") from exc
    if size > max_bytes:
        raise BlindAnnotationError(f"{label} exceeds bounded size limit")
    return path


def _chmod(path: Path, mode: int) -> None:
    try:
        path.chmod(mode, follow_symlinks=False)
    except (NotImplementedError, TypeError):
        path.chmod(mode)
    except OSError as exc:
        raise BlindAnnotationError(f"cannot set private permissions: {path.name}") from exc


def _mkdir_private(path: Path, *, parents: bool = False) -> None:
    if path.exists() and path.is_symlink():
        raise BlindAnnotationError("private directory may not be a symlink")
    try:
        path.mkdir(parents=parents, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise BlindAnnotationError("private output component is not a directory")
        _chmod(path, 0o700)
    except BlindAnnotationError:
        raise
    except OSError as exc:
        raise BlindAnnotationError("cannot create private directory") from exc


def _canonical_output_target(
    path_value: str | os.PathLike[str],
    repo_root: str | os.PathLike[str] | None,
) -> tuple[Path, Path]:
    """Return ``(final_target, canonical_root)`` without creating final target."""
    output = _reject_dotdot(path_value, "output root")
    _reject_existing_symlink_components(output, "output root")
    if output.exists() or output.is_symlink():
        raise BlindAnnotationError("final output target must be fresh and absent")

    if repo_root is not None:
        repo = _reject_dotdot(repo_root, "repository root")
        _reject_existing_symlink_components(repo, "repository root")
        if not repo.is_dir() or repo.is_symlink():
            raise BlindAnnotationError("repository root must be a regular directory")
        repo_resolved = repo.resolve(strict=True)
        canonical = repo_resolved / ".private" / "blind-annotation"
        try:
            relative = output.resolve(strict=False).relative_to(canonical)
        except ValueError as exc:
            raise BlindAnnotationError("output must be below repo/.private/blind-annotation") from exc
        if relative.parts and (len(relative.parts) != 1 or not _PATH_ID.fullmatch(relative.parts[0])):
            raise BlindAnnotationError("output must be canonical root or one opaque child below it")
    else:
        parts = output.parts
        matches = [index for index in range(len(parts) - 1) if parts[index] == ".private" and parts[index + 1] == "blind-annotation"]
        if not matches:
            raise BlindAnnotationError("output must contain .private/blind-annotation")
        canonical = Path(*parts[: matches[-1] + 2])
        relative = output.relative_to(canonical)
        if relative.parts and (len(relative.parts) != 1 or not _PATH_ID.fullmatch(relative.parts[0])):
            raise BlindAnnotationError("output must be canonical root or one opaque child below it")

    _reject_existing_symlink_components(canonical.parent, "canonical output root")
    _mkdir_private(canonical.parent, parents=True)
    if canonical.exists() and canonical.is_symlink():
        raise BlindAnnotationError("canonical output root may not be a symlink")
    output_parent_resolved = output.parent.resolve(strict=False)
    if output_parent_resolved != canonical and output_parent_resolved != canonical.parent.resolve(strict=False):
        raise BlindAnnotationError("output parent is outside canonical root")
    _mkdir_private(output.parent, parents=True)
    return output, canonical


def _open_nofollow(path: Path, flags: int, mode: int = 0o600) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags | nofollow, mode)
    except OSError as exc:
        raise BlindAnnotationError("input/output path could not be opened safely") from exc


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise BlindAnnotationError("private snapshot write made no progress")
        offset += written


def _snapshot_file(source: Path, destination_dir: Path, *, label: str, max_bytes: int) -> tuple[Path, str]:
    """Copy one opened, no-follow input into a private immutable snapshot."""
    destination = destination_dir / ("input-" + hashlib.sha256((label + ":" + str(source)).encode("utf-8")).hexdigest()[:16] + ".bin")
    source_fd = _open_nofollow(source, os.O_RDONLY)
    destination_fd: int | None = None
    digest = hashlib.sha256()
    try:
        source_stat = os.fstat(source_fd)
        if not (source_stat.st_mode & 0o170000) == 0o100000:
            raise BlindAnnotationError(f"{label} is not a regular file")
        if source_stat.st_size > max_bytes:
            raise BlindAnnotationError(f"{label} exceeds bounded size limit")
        destination_fd = _open_nofollow(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        while True:
            block = os.read(source_fd, 1 << 20)
            if not block:
                break
            digest.update(block)
            _write_all(destination_fd, block)
        os.fsync(destination_fd)
        if os.fstat(destination_fd).st_size != source_stat.st_size:
            raise BlindAnnotationError(f"{label} snapshot size changed")
    except BlindAnnotationError:
        raise
    except OSError as exc:
        raise BlindAnnotationError(f"failed to snapshot {label}") from exc
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)
    _chmod(destination, 0o600)
    return destination, digest.hexdigest()


def _remove_private_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        try:
            path.unlink()
        except OSError:
            pass
        return
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            try:
                child.unlink()
            except OSError:
                pass
        elif child.is_dir():
            _remove_private_tree(child)
    try:
        path.rmdir()
    except OSError:
        pass


def _number(value: Any, unit: str) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        return None
    return int(round(converted * {"us": 1.0, "ms": 1_000.0, "s": 1_000_000.0}[unit]))


def _timing_pair(row: Mapping[str, Any]) -> tuple[int, int] | None:
    aliases = (
        (("start_us", "end_us"), "us"),
        (("start_ms", "end_ms"), "ms"),
        (("start_seconds", "end_seconds"), "s"),
        (("start_sec", "end_sec"), "s"),
        (("start_time_us", "end_time_us"), "us"),
        (("start_time_ms", "end_time_ms"), "ms"),
        (("start_time", "end_time"), "s"),
        (("start", "end"), "s"),
        (("startTime", "endTime"), "s"),
    )
    for (start_key, end_key), unit in aliases:
        if start_key in row and end_key in row:
            start, end = _number(row[start_key], unit), _number(row[end_key], unit)
            if start is not None and end is not None and end > start:
                return start, end
    return None


def _validate_json_shape(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise BlindAnnotationError("timing JSON exceeds node limit")
        if depth > MAX_JSON_DEPTH:
            raise BlindAnnotationError("timing JSON exceeds depth limit")
        if isinstance(current, Mapping):
            if len(current) > MAX_JSON_ROW_KEYS:
                raise BlindAnnotationError("timing JSON object exceeds key limit")
            for key, child in current.items():
                if not isinstance(key, str) or len(key) > 256:
                    raise BlindAnnotationError("timing JSON contains an invalid key")
                if isinstance(child, (Mapping, list, tuple)):
                    stack.append((child, depth + 1))
        elif isinstance(current, (list, tuple)):
            if len(current) > MAX_EVENTS:
                raise BlindAnnotationError("timing JSON list exceeds event limit")
            for child in current:
                if isinstance(child, (Mapping, list, tuple)):
                    stack.append((child, depth + 1))


def _json_rows(raw: bytes, *, allowed_containers: frozenset[str]) -> list[Mapping[str, Any]]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise BlindAnnotationError("timing artifact is not supported UTF-8 JSON") from exc
    _validate_json_shape(value)
    if isinstance(value, list):
        rows_value: Any = value
    elif isinstance(value, Mapping):
        keys = [key for key in allowed_containers if key in value]
        if len(keys) != 1:
            raise BlindAnnotationError("timing JSON must use exactly one supported top-level container")
        rows_value = value[keys[0]]
    else:
        raise BlindAnnotationError("timing JSON root must be a row list or supported container object")
    if not isinstance(rows_value, list) or len(rows_value) > MAX_EVENTS:
        raise BlindAnnotationError("timing JSON rows exceed event limit")
    rows: list[Mapping[str, Any]] = []
    for row in rows_value:
        if not isinstance(row, Mapping) or len(row) > MAX_JSON_ROW_KEYS:
            raise BlindAnnotationError("timing JSON contains an invalid row")
        rows.append(row)
    return rows


def _normalize_system_marker(row: Mapping[str, Any]) -> str | None:
    def text(key: str) -> str:
        value = row.get(key)
        return value.strip().upper() if isinstance(value, str) else ""

    speaker = text("speaker_id") or text("speaker")
    attribution = text("attribution_status") or text("status")
    reason = text("conflict_reason") or text("reason") or text("overlap_reason")
    unknown = speaker.startswith("UNKNOWN") or attribution.startswith("UNKNOWN")
    overlap = any(token in reason for token in ("OVERLAP", "CONFLICT", "DISAGREEMENT", "AMBIGUOUS"))
    overlap = overlap or row.get("overlap") is True or row.get("overlap_flag") is True
    conflict = any(token in reason for token in ("CONFLICT", "DISAGREEMENT", "AMBIGUOUS"))
    conflict = conflict or row.get("disagreement") is True or row.get("is_disagreement") is True
    if unknown and overlap:
        return "UNKNOWN_OVERLAP"
    if unknown:
        return "UNKNOWN"
    if overlap and conflict:
        return "OVERLAP_CONFLICT"
    if overlap:
        return "OVERLAP"
    if conflict:
        return "CONFLICT"
    return None


def _parse_rttm(path: Path, *, duration_us: int, system: bool) -> tuple[_TimingEvent, ...]:
    events: list[_TimingEvent] = []
    with path.open("rb") as handle:
        line_number = 0
        while True:
            raw_line = handle.readline(MAX_RTTM_LINE_BYTES + 1)
            if not raw_line:
                break
            line_number += 1
            if len(raw_line) > MAX_RTTM_LINE_BYTES:
                raise BlindAnnotationError("RTTM line exceeds bounded size")
            try:
                fields = raw_line.decode("utf-8", errors="strict").split()
            except UnicodeDecodeError as exc:
                raise BlindAnnotationError("RTTM must be UTF-8") from exc
            if not fields or fields[0] != "SPEAKER" or len(fields) < 5:
                continue
            try:
                start = _number(float(fields[3]), "s")
                length = _number(float(fields[4]), "s")
            except (TypeError, ValueError):
                continue
            if start is None or length is None or length <= 0:
                continue
            marker: str | None = None
            if system:
                speaker = fields[7].upper() if len(fields) > 7 else ""
                marker = "UNKNOWN" if speaker.startswith("UNKNOWN") else None
                marker = "OVERLAP" if speaker == "OVERLAP" else marker
            if start < duration_us and start + length <= duration_us + 1:
                events.append(_TimingEvent(start, start + length, marker))
            if line_number > MAX_EVENTS:
                raise BlindAnnotationError("RTTM exceeds event limit")
    if not events:
        kind = "system" if system else "reference"
        raise BlindAnnotationError(f"{kind} RTTM contains no usable source-time intervals")
    return tuple(events)


def _parse_timing_file(path: Path, *, duration_us: int, system: bool) -> tuple[_TimingEvent, ...]:
    if path.stat().st_size > MAX_REFERENCE_BYTES:
        raise BlindAnnotationError("timing artifact exceeds bounded size limit")
    # Avoid materialising RTTM at all: its parser is intentionally streaming.
    with path.open("rb") as probe:
        prefix = probe.read(256).lstrip()
    if not prefix or prefix[:1] not in {b"{", b"["}:
        return _parse_rttm(path, duration_us=duration_us, system=system)
    raw = path.read_bytes()
    try:
        rows = _json_rows(raw, allowed_containers=frozenset({"turns", "segments", "spans", "events", "intervals"}))
    except BlindAnnotationError as json_error:
        raise json_error
    events: list[_TimingEvent] = []
    for row in rows:
        pair = _timing_pair(row)
        if pair is None:
            continue
        start, end = pair
        if end <= duration_us + 1:
            events.append(_TimingEvent(start, end, _normalize_system_marker(row) if system else None))
        if len(events) > MAX_EVENTS:
            raise BlindAnnotationError("timing artifact exceeds event limit")
    if not events:
        kind = "system" if system else "reference"
        raise BlindAnnotationError(f"{kind} timing artifact contains no usable source-time intervals")
    return tuple(sorted(set(events), key=lambda item: (item.start_us, item.end_us, item.marker or "")))


def _clip_window_for_frame(time_us: int, *, rate: int, clip_frames: int, source_frames: int) -> int:
    frame = (max(0, time_us) * rate) // 1_000_000
    return max(0, min(source_frames - clip_frames, frame - clip_frames // 2))


def _window_from_start(start_frame: int, *, rate: int, clip_frames: int) -> tuple[int, int]:
    return ((start_frame * 1_000_000 + rate // 2) // rate, ((start_frame + clip_frames) * 1_000_000 + rate // 2) // rate)


def _nonoverlap(start: int, intervals: Sequence[tuple[int, int]], clip_frames: int) -> bool:
    return all(start + clip_frames <= left or start >= right for left, right in intervals)


def _choose_nonoverlap(candidates: Iterable[int], *, count: int, clip_frames: int, intervals: list[tuple[int, int]], rng: random.Random) -> list[int]:
    available = sorted({int(item) for item in candidates})
    # Equal-duration intervals admit a simple maximum-cardinality schedule:
    # scan starts in order and retain the earliest compatible candidate.  Only
    # after finding that compatible set do we apply the seeded shuffle.  This
    # avoids a random early boundary blocking enough later boundary evidence.
    maximum: list[int] = []
    for start in available:
        if _nonoverlap(start, intervals, clip_frames) and all(_nonoverlap(start, ((item, item + clip_frames),), clip_frames) for item in maximum):
            maximum.append(start)
    if len(maximum) >= count:
        rng.shuffle(maximum)
        chosen = maximum[:count]
        intervals.extend((item, item + clip_frames) for item in chosen)
        return chosen
    raise BlindAnnotationError("not enough non-overlapping timing clips for the requested pack")


def _grid_starts_containing_points(
    points_us: Iterable[int], *, rate: int, clip_frames: int, source_frames: int,
) -> set[int]:
    """Return only grid windows whose half-open interval contains a boundary."""
    max_start = source_frames - clip_frames
    result: set[int] = set()
    for point_us in points_us:
        point_frame = (max(0, int(point_us)) * rate) // 1_000_000
        if point_frame >= source_frames:
            continue
        lower = max(0, point_frame - clip_frames + 1)
        upper = min(max_start, point_frame)
        first = ((lower + clip_frames - 1) // clip_frames) * clip_frames
        for start in range(first, upper + 1, clip_frames):
            result.add(start)
    return result


def _grid_starts_overlapping_events(
    events: Sequence[_TimingEvent], *, rate: int, clip_frames: int, source_frames: int,
) -> set[int]:
    """Return only grid windows with positive overlap with marked events."""
    max_start = source_frames - clip_frames
    result: set[int] = set()
    for event in events:
        event_start = max(0, (event.start_us * rate) // 1_000_000)
        event_end = min(source_frames, (event.end_us * rate + 999_999) // 1_000_000)
        if event_end <= event_start:
            continue
        lower = max(0, event_start - clip_frames + 1)
        upper = min(max_start, event_end - 1)
        first = ((lower + clip_frames - 1) // clip_frames) * clip_frames
        for start in range(first, upper + 1, clip_frames):
            result.add(start)
    return result


def _choose_boundary_stress_pair(
    boundary_candidates: set[int], stress_candidates: set[int], *, count: int,
    clip_frames: int, rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Choose two disjoint category sets before filling uniform slots.

    Candidate windows are snapped to a ten-second metric lattice.  This turns
    the category quota into a small deterministic set-allocation problem and
    prevents one category's random early choices from starving the other.
    """
    if len(boundary_candidates) < count or len(stress_candidates) < count:
        raise BlindAnnotationError("not enough distinct boundary/stress timing candidates")
    boundary = sorted(boundary_candidates)
    stress = sorted(stress_candidates)
    for _attempt in range(128):
        b_order = list(boundary)
        rng.shuffle(b_order)
        b_choice = b_order[:count]
        remaining_stress = [item for item in stress if item not in set(b_choice)]
        if len(remaining_stress) < count:
            continue
        rng.shuffle(remaining_stress)
        return b_choice, remaining_stress[:count]
    # Rarest-first deterministic fallback, useful when the two sets overlap
    # almost completely.
    b_choice = sorted(boundary, key=lambda item: (item in stress_candidates, item))[:count]
    remaining_stress = [item for item in stress if item not in set(b_choice)]
    if len(remaining_stress) >= count:
        return b_choice, remaining_stress[:count]
    raise BlindAnnotationError("boundary/stress timing candidates cannot satisfy non-overlap quotas")


def _select_specs(source_frames: int, rate: int, reference: Sequence[_TimingEvent], system: Sequence[_TimingEvent], *, seed: int, clip_seconds: float) -> tuple[ClipSpec, ...]:
    clip_frames = int(round(rate * clip_seconds))
    if clip_frames <= 0 or clip_frames > MAX_CLIP_FRAMES:
        raise BlindAnnotationError("clip duration exceeds bounded frame limit")
    if source_frames < EXPECTED_CLIP_COUNT * clip_frames:
        raise BlindAnnotationError("source is too short for 48 non-overlapping metric clips")
    marked = [event for event in system if event.marker is not None]
    if not marked:
        raise BlindAnnotationError("system timing has no normalized UNKNOWN/overlap/conflict stress events")
    boundary, stress = _choose_boundary_stress_pair(
        _grid_starts_containing_points((point for event in reference for point in (event.start_us, event.end_us)), rate=rate, clip_frames=clip_frames, source_frames=source_frames),
        _grid_starts_overlapping_events(marked, rate=rate, clip_frames=clip_frames, source_frames=source_frames),
        count=DEFAULT_BOUNDARY_COUNT, clip_frames=clip_frames, rng=random.Random(seed ^ 0xB0A7D),
    )
    intervals: list[tuple[int, int]] = [(item, item + clip_frames) for item in boundary + stress]
    uniform = _choose_nonoverlap(
        (item for item in range(0, source_frames - clip_frames + 1, clip_frames) if item not in set(boundary) and item not in set(stress)),
        count=DEFAULT_UNIFORM_COUNT, clip_frames=clip_frames, intervals=intervals, rng=random.Random(seed),
    )
    starts = [(CATEGORY_UNIFORM, item) for item in uniform] + [(CATEGORY_BOUNDARY, item) for item in boundary] + [(CATEGORY_STRESS, item) for item in stress]
    result: list[ClipSpec] = []
    for index, (category, start) in enumerate(starts):
        start_us, end_us = _window_from_start(start, rate=rate, clip_frames=clip_frames)
        result.append(ClipSpec(index, category, start, start + clip_frames, start_us, end_us, f"AUDIT_{index // 4:02d}" if index % 4 == 0 else None))
    if len(result) != EXPECTED_CLIP_COUNT or len({(item.start_frame, item.end_frame) for item in result}) != EXPECTED_CLIP_COUNT:
        raise BlindAnnotationError("clip selection cardinality/uniqueness invariant failed")
    reference_points = tuple(point for event in reference for point in (event.start_us, event.end_us))
    for spec in result:
        if spec.category == CATEGORY_BOUNDARY:
            if not any(spec.start_frame <= (point * rate) // 1_000_000 < spec.end_frame for point in reference_points):
                raise BlindAnnotationError("boundary clip does not contain a reference boundary")
        elif spec.category == CATEGORY_STRESS:
            if not any(
                spec.start_frame < min(source_frames, (event.end_us * rate + 999_999) // 1_000_000)
                and spec.end_frame > max(0, (event.start_us * rate) // 1_000_000)
                for event in marked
            ):
                raise BlindAnnotationError("stress clip does not overlap a marked system interval")
    return tuple(result)


def _write_file(path: Path, payload: bytes) -> str:
    if path.exists() or path.is_symlink():
        raise BlindAnnotationError("refusing to overwrite private output")
    try:
        path.write_bytes(payload)
        _chmod(path, 0o600)
    except OSError as exc:
        raise BlindAnnotationError("cannot write private output") from exc
    return hashlib.sha256(payload).hexdigest()


def _copy_clip(source: Path, destination: Path, spec: ClipSpec, *, rate: int, source_layout: Any) -> str:
    if destination.exists() or destination.is_symlink():
        raise BlindAnnotationError("refusing to overwrite an existing clip")
    frame_bytes = source_layout.sample_width_bytes * source_layout.channel_count
    temp = destination.with_name(f".{destination.name}.tmp")
    if temp.exists() or temp.is_symlink():
        raise BlindAnnotationError("temporary clip path already exists")
    try:
        with source.open("rb") as input_handle:
            input_handle.seek(source_layout.data_offset + spec.start_frame * frame_bytes)
            with wave.open(str(temp), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(rate)
                output.setcomptype("NONE", "not compressed")
                remaining = spec.end_frame - spec.start_frame
                while remaining:
                    block_frames = min(remaining, COPY_CHUNK_FRAMES)
                    payload = input_handle.read(block_frames * frame_bytes)
                    if len(payload) != block_frames * frame_bytes:
                        raise BlindAnnotationError("source snapshot ended before requested clip")
                    output.writeframesraw(payload)
                    remaining -= block_frames
        _chmod(temp, 0o600)
        os.replace(temp, destination)
        _chmod(destination, 0o600)
    except Exception:
        if temp.exists() and not temp.is_symlink():
            try:
                temp.unlink()
            except OSError:
                pass
        raise
    return _sha256_file(destination)


def _opaque_ids(specs: Sequence[ClipSpec], nonce: bytes) -> tuple[dict[int, str], tuple[int, ...]]:
    identifiers = {spec.internal_index: "item-" + hashlib.sha256(nonce + b"/id/" + str(spec.internal_index).encode("ascii")).hexdigest()[:24] for spec in specs}
    order = list(range(len(specs)))
    random.Random(int.from_bytes(hashlib.sha256(nonce + b"/shuffle").digest()[:8], "big")).shuffle(order)
    return identifiers, tuple(order)


def _selection_evidence(specs: Sequence[ClipSpec], *, rate: int, clip_hashes: Sequence[str]) -> dict[str, Any]:
    ordered = sorted((item.start_frame, item.end_frame) for item in specs)
    union_frames = 0
    overlap_count = 0
    previous_end = -1
    for start, end in ordered:
        if start < previous_end:
            overlap_count += 1
        union_frames += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return {
        "metric_union_duration_us": (union_frames * 1_000_000 + rate // 2) // rate,
        "metric_overlap_count": overlap_count,
        "metric_excluded_overlap_duration_us": 0,
        "qc_duplicate_audio_count": len(clip_hashes) - len(set(clip_hashes)),
        "second_annotator_slot_count": sum(item.audit_slot is not None for item in specs),
    }


def _fixed_public_evidence(*, manifest_sha: str, annotator_sha: str, template_sha: str, selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_sha,
        "annotator_manifest_sha256": annotator_sha,
        "label_template_sha256": template_sha,
        "clip_count": EXPECTED_CLIP_COUNT,
        "category_counts": dict(CATEGORY_COUNTS),
        "metric_union_duration_us": int(selection["metric_union_duration_us"]),
        "metric_overlap_count": int(selection["metric_overlap_count"]),
        "metric_excluded_overlap_duration_us": int(selection["metric_excluded_overlap_duration_us"]),
        "qc_duplicate_audio_count": int(selection["qc_duplicate_audio_count"]),
        "second_annotator_slot_count": int(selection["second_annotator_slot_count"]),
    }


def build_blind_annotation_pack(source_path: str | os.PathLike[str], output_root: str | os.PathLike[str], *, reference_path: str | os.PathLike[str], system_path: str | os.PathLike[str], seed: int = DEFAULT_SEED, clip_seconds: float = DEFAULT_CLIP_SECONDS, repo_root: str | os.PathLike[str] | None = None, presentation_nonce: str | bytes | None = None) -> BlindPackResult:
    """Build a fresh atomic pack below the canonical private root."""
    if not math.isfinite(float(clip_seconds)) or clip_seconds <= 0 or clip_seconds > 60:
        raise BlindAnnotationError("clip_seconds must be finite and in (0, 60]")
    source = _regular_file(source_path, "source WAV", max_bytes=MAX_SOURCE_BYTES)
    reference_file = _regular_file(reference_path, "reference timing", max_bytes=MAX_REFERENCE_BYTES)
    system_file = _regular_file(system_path, "system timing", max_bytes=MAX_REFERENCE_BYTES)
    final_target, _canonical_root = _canonical_output_target(output_root, repo_root)
    stage_token = hashlib.sha256(f"{os.getpid()}:{seed}:{source}".encode("utf-8")).hexdigest()[:20]
    stage = final_target.parent / f".{final_target.name}.staging-{stage_token}"
    if stage.exists() or stage.is_symlink():
        raise BlindAnnotationError("private staging target already exists")
    try:
        _mkdir_private(stage, parents=True)
    except Exception:
        _remove_private_tree(stage)
        raise
    try:
        snapshots = Path(tempfile.mkdtemp(prefix=".sddiar-input-", dir=str(final_target.parent)))
    except Exception:
        _remove_private_tree(stage)
        raise BlindAnnotationError("cannot create private input snapshot directory") from None
    _chmod(snapshots, 0o700)
    try:
        source_snapshot, source_sha = _snapshot_file(source, snapshots, label="source WAV", max_bytes=MAX_SOURCE_BYTES)
        reference_snapshot, reference_sha = _snapshot_file(reference_file, snapshots, label="reference timing", max_bytes=MAX_REFERENCE_BYTES)
        system_snapshot, system_sha = _snapshot_file(system_file, snapshots, label="system timing", max_bytes=MAX_REFERENCE_BYTES)
        try:
            layout = WavPcmAccessor(source_snapshot).layout
        except (OSError, EOFError, MediaError) as exc:
            raise BlindAnnotationError("invalid source WAV") from exc
        if layout.sample_width_bytes != 2 or layout.channel_count != 1:
            raise BlindAnnotationError("source must be PCM16 mono WAV")
        if layout.sample_rate_hz < 8_000 or layout.sample_rate_hz > 192_000:
            raise BlindAnnotationError("source sample rate is outside the bounded PCM contract")
        if layout.data_bytes > MAX_SOURCE_BYTES:
            raise BlindAnnotationError("source PCM exceeds bounded size limit")
        duration_us = (layout.frame_count * 1_000_000 + layout.sample_rate_hz // 2) // layout.sample_rate_hz
        reference = _parse_timing_file(reference_snapshot, duration_us=duration_us, system=False)
        system = _parse_timing_file(system_snapshot, duration_us=duration_us, system=True)
        specs = _select_specs(layout.frame_count, layout.sample_rate_hz, reference, system, seed=int(seed), clip_seconds=float(clip_seconds))
        nonce = (presentation_nonce.encode("utf-8") if isinstance(presentation_nonce, str) else presentation_nonce) or hashlib.sha256(b"sddiar-presentation-v1:" + source_sha.encode("ascii") + reference_sha.encode("ascii") + system_sha.encode("ascii") + str(seed).encode("ascii")).digest()
        if not nonce or len(nonce) > 256:
            raise BlindAnnotationError("presentation nonce must be 1..256 bytes")
        identifiers, order = _opaque_ids(specs, nonce)
        annotator_dir = stage / "annotator"
        clips_dir = annotator_dir / "clips"
        _mkdir_private(annotator_dir, parents=False)
        _mkdir_private(clips_dir, parents=False)
        evaluator_rows: list[dict[str, Any]] = []
        annotator_rows: list[dict[str, Any]] = []
        template_rows: list[dict[str, Any]] = []
        clip_paths: list[Path] = []
        clip_hashes: list[str] = []
        for spec in specs:
            opaque = identifiers[spec.internal_index]
            destination = clips_dir / f"{opaque}.wav"
            clip_sha = _copy_clip(source_snapshot, destination, spec, rate=layout.sample_rate_hz, source_layout=layout)
            clip_paths.append(destination)
            clip_hashes.append(clip_sha)
            duration = spec.source_time_end_us - spec.source_time_start_us
            evaluator_rows.append({"clip_id": opaque, "category": spec.category, "audio": f"annotator/clips/{opaque}.wav", "audio_sha256": clip_sha, "sample_rate_hz": layout.sample_rate_hz, "sample_width_bytes": 2, "channel_count": 1, "frame_count": spec.end_frame - spec.start_frame, "source_time_start_us": spec.source_time_start_us, "source_time_end_us": spec.source_time_end_us, "audit_slot": spec.audit_slot})
            annotator_rows.append({
                "annotation_id": opaque,
                "audio": f"clips/{opaque}.wav",
                "audio_sha256": clip_sha,
                "duration_us": duration,
            })
            template_rows.append({"annotation_id": opaque, "audio": f"clips/{opaque}.wav", "clip_relative_start_us": 0, "clip_relative_end_us": duration, "segments": [], "change_boundaries_us": [], "allowed_segment_labels": list(TEMPLATE_SEGMENT_LABELS), "allowed_boundary_value": "change boundary"})
        annotator_rows = [annotator_rows[index] for index in order]
        template_rows = [template_rows[index] for index in order]
        selection = _selection_evidence(specs, rate=layout.sample_rate_hz, clip_hashes=clip_hashes)
        annotator_path = annotator_dir / "manifest.json"
        annotator_sha = _write_file(annotator_path, _canonical_json({"schema_version": ANNOTATOR_SCHEMA_VERSION, "clips": annotator_rows}) + b"\n")
        template_path = annotator_dir / "label_template.jsonl"
        template_sha = _write_file(template_path, b"".join(_canonical_json(row) + b"\n" for row in template_rows))
        evaluator_manifest = {"schema_version": SCHEMA_VERSION, "source": {"audio_sha256": source_sha, "sample_rate_hz": layout.sample_rate_hz, "frame_count": layout.frame_count, "duration_us": duration_us}, "provenance": {"reference_timing_sha256": reference_sha, "system_timing_sha256": system_sha, "speaker_labels_used": False, "transcript_text_used": False, "selection_inputs_are_source_time_only": True, "presentation_nonce_sha256": hashlib.sha256(nonce).hexdigest(), "annotator_manifest_sha256": annotator_sha, "label_template_sha256": template_sha}, "policy": {"offline": True, "seed": int(seed), "clip_seconds": float(clip_seconds), "clip_count": EXPECTED_CLIP_COUNT, "category_counts_are_sampling_metadata_only": True, "do_not_sum_categories_for_metrics": True, "metric_intervals_non_overlapping": True}, "selection": selection, "clips": evaluator_rows}
        manifest_path = stage / "manifest.json"
        manifest_sha = _write_file(manifest_path, _canonical_json(evaluator_manifest) + b"\n")
        _write_file(stage / "manifest.sha256", (manifest_sha + "  manifest.json\n").encode("ascii"))
        _chmod(stage, 0o700)
        if final_target.exists() or final_target.is_symlink():
            raise BlindAnnotationError("final output target appeared during build")
        os.replace(stage, final_target)
        evidence = _fixed_public_evidence(manifest_sha=manifest_sha, annotator_sha=annotator_sha, template_sha=template_sha, selection=selection)
        final_clips = tuple(final_target / "annotator" / "clips" / path.name for path in clip_paths)
        return BlindPackResult(final_target / "manifest.json", final_target / "annotator" / "label_template.jsonl", final_clips, evidence)
    except Exception:
        _remove_private_tree(stage)
        raise
    finally:
        _remove_private_tree(snapshots)


def _read_json_file(path: Path, *, max_bytes: int = MAX_REFERENCE_BYTES) -> Any:
    try:
        if path.stat().st_size > max_bytes:
            raise BlindAnnotationError("JSON artifact exceeds bounded size")
        value = json.loads(path.read_text(encoding="utf-8"))
    except BlindAnnotationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise BlindAnnotationError("invalid sealed JSON artifact") from exc
    _validate_json_shape(value)
    return value


def _strict_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise BlindAnnotationError(f"invalid {label} hash")
    return value


def _verify_permissions(path: Path, *, directory: bool) -> None:
    if path.is_symlink() or (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise BlindAnnotationError("pack contains a missing or symlinked artifact")
    expected = 0o700 if directory else 0o600
    if path.stat().st_mode & 0o777 != expected:
        raise BlindAnnotationError("pack permissions are not owner-only")


def _verify_template(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    _verify_permissions(path, directory=False)
    parsed: list[Mapping[str, Any]] = []
    if path.stat().st_size > MAX_REFERENCE_BYTES:
        raise BlindAnnotationError("label template exceeds bounded size")
    digest = hashlib.sha256()
    line_cap = MAX_RTTM_LINE_BYTES * 2
    with path.open("rb") as handle:
        while True:
            line = handle.readline(line_cap + 1)
            if not line:
                break
            if len(line) > line_cap:
                raise BlindAnnotationError("label template line exceeds bounded size")
            if len(line) == line_cap and not line.endswith(b"\n"):
                # Distinguish an exact-size final line from an overlong line
                # without reading an unbounded buffer.
                if handle.read(1):
                    raise BlindAnnotationError("label template line exceeds bounded size")
            digest.update(line)
            line = line.rstrip(b"\r\n")
            if not line:
                raise BlindAnnotationError("invalid label template line")
            try:
                item = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
                raise BlindAnnotationError("invalid label template JSONL") from exc
            _validate_json_shape(item)
            if not isinstance(item, Mapping) or frozenset(item) != TEMPLATE_ROW_KEYS or item["segments"] != [] or item["change_boundaries_us"] != []:
                raise BlindAnnotationError("label template schema/content mismatch")
            if tuple(item["allowed_segment_labels"]) != TEMPLATE_SEGMENT_LABELS or item["allowed_boundary_value"] != "change boundary":
                raise BlindAnnotationError("label template allowed labels mismatch")
            if item["clip_relative_start_us"] != 0 or not isinstance(item["clip_relative_end_us"], int) or item["clip_relative_end_us"] <= 0:
                raise BlindAnnotationError("label template time range mismatch")
            parsed.append(item)
            if len(parsed) > MAX_EVENTS:
                raise BlindAnnotationError("label template exceeds event limit")
    if len(parsed) != len(rows):
        raise BlindAnnotationError("label template row count mismatch")
    if [(row["annotation_id"], row["audio"], row["duration_us"]) for row in rows] != [(row["annotation_id"], row["audio"], row["clip_relative_end_us"]) for row in parsed]:
        raise BlindAnnotationError("label template ordering/content mismatch")
    return digest.hexdigest()


def verify_pack(pack_root: str | os.PathLike[str], expected_manifest_sha256: str) -> dict[str, Any]:
    """Strictly verify a sealed pack and return fixed-key public evidence."""
    root = _reject_dotdot(pack_root, "pack root")
    _reject_existing_symlink_components(root, "pack root")
    if not ((root.name == "blind-annotation" and root.parent.name == ".private")
            or (root.parent.name == "blind-annotation" and root.parent.parent.name == ".private")):
        raise BlindAnnotationError("pack is outside canonical .private/blind-annotation root")
    _verify_permissions(root, directory=True)
    expected = _strict_hash(expected_manifest_sha256, "expected manifest")
    if {item.name for item in root.iterdir()} != {"manifest.json", "manifest.sha256", "annotator"}:
        raise BlindAnnotationError("pack contains an unexpected root artifact")
    manifest_path = root / "manifest.json"
    _verify_permissions(manifest_path, directory=False)
    actual_manifest_sha = _sha256_file(manifest_path)
    if actual_manifest_sha != expected:
        raise BlindAnnotationError("sealed manifest hash mismatch")
    manifest = _read_json_file(manifest_path)
    if not isinstance(manifest, Mapping) or frozenset(manifest) != EVALUATOR_TOP_KEYS or manifest.get("schema_version") != SCHEMA_VERSION:
        raise BlindAnnotationError("evaluator manifest schema mismatch")
    clips = manifest.get("clips")
    if not isinstance(clips, list) or len(clips) != EXPECTED_CLIP_COUNT:
        raise BlindAnnotationError("evaluator clip count mismatch")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or set(source) != {"audio_sha256", "sample_rate_hz", "frame_count", "duration_us"}:
        raise BlindAnnotationError("evaluator source schema mismatch")
    _strict_hash(source.get("audio_sha256"), "source audio")
    if (not isinstance(source["sample_rate_hz"], int) or source["sample_rate_hz"] <= 0
            or not isinstance(source["frame_count"], int) or source["frame_count"] <= 0
            or not isinstance(source["duration_us"], int) or source["duration_us"] <= 0):
        raise BlindAnnotationError("evaluator source values are invalid")
    provenance = manifest.get("provenance")
    expected_provenance = {
        "reference_timing_sha256", "system_timing_sha256", "speaker_labels_used",
        "transcript_text_used", "selection_inputs_are_source_time_only",
        "presentation_nonce_sha256", "annotator_manifest_sha256", "label_template_sha256",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected_provenance:
        raise BlindAnnotationError("evaluator provenance schema mismatch")
    for key in ("reference_timing_sha256", "system_timing_sha256", "presentation_nonce_sha256", "annotator_manifest_sha256", "label_template_sha256"):
        _strict_hash(provenance[key], key)
    if provenance["speaker_labels_used"] is not False or provenance["transcript_text_used"] is not False or provenance["selection_inputs_are_source_time_only"] is not True:
        raise BlindAnnotationError("evaluator privacy provenance mismatch")
    policy = manifest.get("policy")
    expected_policy = {"offline", "seed", "clip_seconds", "clip_count", "category_counts_are_sampling_metadata_only", "do_not_sum_categories_for_metrics", "metric_intervals_non_overlapping"}
    if not isinstance(policy, Mapping) or set(policy) != expected_policy or policy["offline"] is not True or not isinstance(policy["seed"], int) or not isinstance(policy["clip_seconds"], (int, float)) or not math.isfinite(float(policy["clip_seconds"])) or policy["clip_seconds"] <= 0 or policy["clip_count"] != EXPECTED_CLIP_COUNT or policy["category_counts_are_sampling_metadata_only"] is not True or policy["do_not_sum_categories_for_metrics"] is not True or policy["metric_intervals_non_overlapping"] is not True:
        raise BlindAnnotationError("evaluator policy mismatch")
    categories: dict[str, int] = {name: 0 for name in CATEGORIES}
    intervals: list[tuple[int, int]] = []
    clip_ids: set[str] = set()
    annotator_dir = root / "annotator"
    _verify_permissions(annotator_dir, directory=True)
    annotator_manifest_path, template_path, clips_dir = annotator_dir / "manifest.json", annotator_dir / "label_template.jsonl", annotator_dir / "clips"
    if {item.name for item in annotator_dir.iterdir()} != {"manifest.json", "label_template.jsonl", "clips"}:
        raise BlindAnnotationError("annotator bundle contains an unexpected artifact")
    _verify_permissions(annotator_manifest_path, directory=False)
    _verify_permissions(template_path, directory=False)
    _verify_permissions(clips_dir, directory=True)
    annotator_manifest = _read_json_file(annotator_manifest_path)
    if not isinstance(annotator_manifest, Mapping) or frozenset(annotator_manifest) != ANNOTATOR_TOP_KEYS or annotator_manifest.get("schema_version") != ANNOTATOR_SCHEMA_VERSION:
        raise BlindAnnotationError("annotator manifest schema mismatch")
    annotator_rows = annotator_manifest.get("clips")
    if not isinstance(annotator_rows, list) or len(annotator_rows) != EXPECTED_CLIP_COUNT:
        raise BlindAnnotationError("annotator clip count mismatch")
    for row in annotator_rows:
        if not isinstance(row, Mapping) or frozenset(row) != ANNOTATOR_ROW_KEYS:
            raise BlindAnnotationError("annotator row schema mismatch")
        if not isinstance(row["annotation_id"], str) or not row["annotation_id"].startswith("item-") or not _PATH_ID.fullmatch(row["annotation_id"]):
            raise BlindAnnotationError("annotator ID is not opaque")
        if not isinstance(row["audio"], str) or row["audio"] != f"clips/{row['annotation_id']}.wav" or ".." in Path(row["audio"]).parts or not _PATH_ID.fullmatch(Path(row["audio"]).stem):
            raise BlindAnnotationError("annotator audio path is not allowlisted")
        if not isinstance(row["duration_us"], int) or row["duration_us"] <= 0:
            raise BlindAnnotationError("annotator duration is invalid")
        _strict_hash(row["audio_sha256"], "annotator clip audio")
    annotator_ids = [row["annotation_id"] for row in annotator_rows]
    if len(set(annotator_ids)) != EXPECTED_CLIP_COUNT:
        raise BlindAnnotationError("annotator IDs are not unique")
    annotator_by_id = {row["annotation_id"]: row for row in annotator_rows}
    template_sha = _verify_template(template_path, annotator_rows)
    annotator_sha = _sha256_file(annotator_manifest_path)
    if provenance["annotator_manifest_sha256"] != annotator_sha or provenance["label_template_sha256"] != template_sha:
        raise BlindAnnotationError("sealed annotator artifact hash mismatch")
    expected_audio_names: set[str] = set()
    expected_row_keys = {"clip_id", "category", "audio", "audio_sha256", "sample_rate_hz", "sample_width_bytes", "channel_count", "frame_count", "source_time_start_us", "source_time_end_us", "audit_slot"}
    for row in clips:
        if not isinstance(row, Mapping) or set(row) != expected_row_keys or not isinstance(row["clip_id"], str) or not _PATH_ID.fullmatch(row["clip_id"]):
            raise BlindAnnotationError("evaluator clip row schema mismatch")
        clip_id = row["clip_id"]
        if clip_id in clip_ids or clip_id not in annotator_ids:
            raise BlindAnnotationError("evaluator/annotator clip mapping mismatch")
        clip_ids.add(clip_id)
        category = row["category"]
        if category not in CATEGORY_COUNTS:
            raise BlindAnnotationError("unexpected evaluator category")
        categories[category] += 1
        if not isinstance(row["source_time_start_us"], int) or not isinstance(row["source_time_end_us"], int) or row["source_time_end_us"] <= row["source_time_start_us"]:
            raise BlindAnnotationError("invalid evaluator source interval")
        if row["source_time_start_us"] < 0 or row["source_time_end_us"] > source["duration_us"]:
            raise BlindAnnotationError("evaluator source interval is outside source")
        intervals.append((row["source_time_start_us"], row["source_time_end_us"]))
        if row["audit_slot"] is not None and (not isinstance(row["audit_slot"], str) or not re.fullmatch(r"AUDIT_[0-9]{2}", row["audit_slot"])):
            raise BlindAnnotationError("invalid audit slot")
        if annotator_by_id[clip_id]["duration_us"] != row["source_time_end_us"] - row["source_time_start_us"]:
            raise BlindAnnotationError("annotator/evaluator duration mismatch")
        if annotator_by_id[clip_id]["audio_sha256"] != row["audio_sha256"]:
            raise BlindAnnotationError("annotator/evaluator audio hash mismatch")
        audio = row["audio"]
        if audio != f"annotator/clips/{clip_id}.wav":
            raise BlindAnnotationError("evaluator audio path mismatch")
        _strict_hash(row["audio_sha256"], "clip audio")
        if (not isinstance(row["sample_rate_hz"], int) or row["sample_rate_hz"] <= 0
                or row["sample_width_bytes"] != 2 or row["channel_count"] != 1
                or not isinstance(row["frame_count"], int) or row["frame_count"] <= 0):
            raise BlindAnnotationError("evaluator clip metadata is invalid")
        expected_audio_names.add(f"{clip_id}.wav")
        audio_path = root / audio
        _verify_permissions(audio_path, directory=False)
        try:
            layout = WavPcmAccessor(audio_path).layout
        except (MediaError, OSError, EOFError) as exc:
            raise BlindAnnotationError("invalid clip WAV") from exc
        if (layout.sample_rate_hz, layout.sample_width_bytes, layout.channel_count, layout.frame_count) != (row["sample_rate_hz"], row["sample_width_bytes"], row["channel_count"], row["frame_count"]):
            raise BlindAnnotationError("clip WAV metadata mismatch")
        if _sha256_file(audio_path) != row["audio_sha256"]:
            raise BlindAnnotationError("clip WAV hash mismatch")
    if categories != CATEGORY_COUNTS:
        raise BlindAnnotationError("category counts mismatch")
    audit_slots = [row["audit_slot"] for row in clips if row["audit_slot"] is not None]
    if len(audit_slots) != 12 or set(audit_slots) != {f"AUDIT_{index:02d}" for index in range(12)}:
        raise BlindAnnotationError("audit slot cardinality mismatch")
    ordered = sorted(intervals)
    if any(start < ordered[index - 1][1] for index, (start, _end) in enumerate(ordered) if index):
        raise BlindAnnotationError("metric clips overlap")
    if {item.name for item in clips_dir.iterdir()} != expected_audio_names:
        raise BlindAnnotationError("clip directory contains an unexpected artifact")
    sidecar_path = root / "manifest.sha256"
    _verify_permissions(sidecar_path, directory=False)
    if sidecar_path.read_text(encoding="ascii") != expected + "  manifest.json\n":
        raise BlindAnnotationError("manifest sidecar mismatch")
    selection = manifest["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {"metric_union_duration_us", "metric_overlap_count", "metric_excluded_overlap_duration_us", "qc_duplicate_audio_count", "second_annotator_slot_count"}:
        raise BlindAnnotationError("selection evidence schema mismatch")
    if any(type(selection[key]) is not int or selection[key] < 0 for key in selection):
        raise BlindAnnotationError("selection evidence values are invalid")
    if selection["metric_overlap_count"] != 0 or selection["metric_excluded_overlap_duration_us"] != 0 or selection["second_annotator_slot_count"] != 12:
        raise BlindAnnotationError("selection QC evidence mismatch")
    metric_union = sum(end - start for start, end in intervals)
    if selection["metric_union_duration_us"] != metric_union:
        raise BlindAnnotationError("selection union duration mismatch")
    actual_clip_hashes = [row["audio_sha256"] for row in clips]
    if selection["qc_duplicate_audio_count"] != len(actual_clip_hashes) - len(set(actual_clip_hashes)):
        raise BlindAnnotationError("selection duplicate audio evidence mismatch")
    return _fixed_public_evidence(manifest_sha=actual_manifest_sha, annotator_sha=annotator_sha, template_sha=template_sha, selection=selection)


def public_evidence(manifest_path: str | os.PathLike[str], expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Return fixed-key hash/count evidence after strict pack verification."""
    path = _regular_file(manifest_path, "manifest", max_bytes=MAX_REFERENCE_BYTES)
    return verify_pack(path.parent, expected_manifest_sha256 or _sha256_file(path))


build_pack = build_blind_annotation_pack
