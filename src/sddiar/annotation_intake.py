"""Strict, offline intake checks for independently annotated diarization data.

The intake boundary is deliberately boring: it reads a JSONL manifest and the
three files referenced by each row, but never modifies the dataset and never
loads an audio decoder or network client.  The public report contains only
aggregate counts and opaque record positions; in particular it does not echo
filesystem paths, transcript text, or annotation names.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .evaluation import (
    EvaluationError,
    RecordingManifest,
    SplitLeakageError,
    WordAnnotation,
    validate_recording_session_splits,
)
from .media import MediaError, WavPcmAccessor


REQUIRED_FIELDS = frozenset({
    "audio_id", "audio_sha256", "session_id", "split", "sample_rate_hz",
    "speaker_count", "gender_pair", "conditions", "audio", "rttm", "uem",
})
OPTIONAL_FIELDS = frozenset({
    "recording_id", "source_recording_id", "augmentation_group",
    "words", "words_sha256", "words_timebase",
})
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
REQUIRED_SPLITS = ("CALIBRATION", "DEVELOPMENT_HOLDOUT", "RELEASE_HOLDOUT")
GENDER_PAIRS = frozenset({"MM", "FF", "MF", "M", "F", "UNKNOWN"})
SAMPLE_RATES = frozenset({8000, 16000})
PATH_FIELDS = ("audio", "rttm", "uem")
WORD_ARTIFACT_FIELDS = frozenset({"words", "words_sha256", "words_timebase"})
WORD_ROW_REQUIRED_FIELDS = frozenset({
    "recording_id", "word_id", "start_us", "end_us", "text", "ref_speaker_id",
    "attributable", "overlap_flag", "boundary_crossing_flag",
})
WORD_ROW_OPTIONAL_FIELDS = frozenset({"micro_flag"})
WORD_ROW_ALLOWED_FIELDS = WORD_ROW_REQUIRED_FIELDS | WORD_ROW_OPTIONAL_FIELDS
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_KNOWN_BAD_IDS = frozenset({
    "unknown", "overlap", "anonymous", "anonymized", "name", "realname",
    "customer", "client", "transcript", "speaker", "person",
})
_NON_SPEECH_LABEL = re.compile(r"^(?:UNKNOWN|OVERLAP|SIL|SILENCE|NON[_-]?SPEECH)(?:[_-]?\d+)?$")


@dataclass(frozen=True)
class IntakeIssue:
    """A redacted issue; values that could identify a source are omitted."""

    code: str
    record_index: int | None = None
    field: str | None = None
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "severity": self.severity}
        if self.record_index is not None:
            out["record_index"] = self.record_index
        if self.field is not None:
            out["field"] = self.field
        return out


@dataclass
class _RecordStats:
    index: int
    audio_id: str
    session_id: str
    recording_id: str
    source_recording_id: str | None
    augmentation_group: str | None
    split: str
    sample_rate_hz: int
    speaker_count: int
    gender_pair: str
    conditions: tuple[str, ...]
    duration_us: int = 0
    speakers: set[str] = field(default_factory=set)
    segment_count: int = 0
    short_turn: bool = False
    overlap_us: int = 0
    valid_audio: bool = False
    valid_annotations: bool = False
    words_declared: bool = False
    words_valid: bool = False
    words_count: int = 0
    micro_flagged_words: int = 0


@dataclass(frozen=True)
class WordsArtifact:
    """Typed words payload for the UEM scorer after intake validation.

    The artifact loader intentionally returns the typed evaluation objects only
    to the caller that explicitly requests them.  The aggregate intake report
    never includes word IDs, timestamps, or text.
    """

    recording_id: str
    timebase: str
    sha256: str
    words: tuple[WordAnnotation, ...]
    micro_flags: tuple[tuple[str, bool], ...] = ()


@dataclass
class AnnotationValidationReport:
    """Redacted aggregate result returned by :func:`validate_annotation_dataset`."""

    errors: list[IntakeIssue] = field(default_factory=list)
    warnings: list[IntakeIssue] = field(default_factory=list)
    records_seen: int = 0
    records_valid: int = 0
    readiness: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
            "records_seen": self.records_seen,
            "records_valid": self.records_valid,
            "readiness": self.readiness,
        }

    # A convenient alias for callers that use the common report convention.
    to_dict = as_dict


def _issue(report: AnnotationValidationReport, code: str, index: int | None = None,
           field: str | None = None, *, warning: bool = False) -> None:
    item = IntakeIssue(code, index, field, "warning" if warning else "error")
    (report.warnings if warning else report.errors).append(item)


def _opaque_identifier(value: Any) -> bool:
    """Return true for a machine identifier, never a path or a personal name.

    Requiring a digit is intentional.  It makes the privacy contract
    deterministic (``session-001`` and ``REF_00`` pass) and rejects casual
    person-name labels without attempting a language-specific name database.
    """
    if not isinstance(value, str) or not value or len(value) > 128:
        return False
    if not _ID.fullmatch(value) or not any(ch.isdigit() for ch in value):
        return False
    if value.lower() in _KNOWN_BAD_IDS:
        return False
    return True


def _check_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    # Backslash is rejected even on POSIX so a manifest cannot change meaning
    # when copied to a Windows validation host.
    if "\\" in value or _URI_SCHEME.match(value):
        return None
    candidate = Path(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    # Resolve without following a missing leaf, then ensure every existing
    # component remains below the dataset root.  Symlinked files/components
    # are rejected explicitly rather than merely checking the final target.
    root_real = root.resolve()
    path = root / candidate
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(root_real)
    except (OSError, ValueError):
        return None
    current = root
    for part in candidate.parts:
        current = current / part
        try:
            if current.is_symlink():
                return None
        except OSError:
            return None
    if not path.is_file():
        return None
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_us(raw: str, *, positive: bool = False) -> int:
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError):
        raise ValueError("time") from None
    if not value.is_finite() or value < 0 or (positive and value <= 0):
        raise ValueError("time")
    # Decimal avoids platform-specific float rounding at annotation borders.
    return int((value * Decimal(1_000_000)).to_integral_value())


def _parse_uem(text: str, audio_id: str, duration_us: int) -> tuple[list[tuple[int, int]], list[str]]:
    intervals: list[tuple[int, int]] = []
    codes: list[str] = []
    previous_end = -1
    for line_no, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 4:
            codes.append("UEM_SCHEMA")
            continue
        id_ok = fields[0] == audio_id
        channel_ok = fields[1] == "1"
        if not id_ok:
            codes.append("FILE_ID_MISMATCH")
        if not channel_ok:
            codes.append("CHANNEL_MISMATCH")
        try:
            start, end = _time_us(fields[2]), _time_us(fields[3])
        except ValueError:
            codes.append("UEM_TIMING")
            continue
        if end <= start:
            codes.append("UEM_OUTSIDE_AUDIO")
            continue
        if end > duration_us:
            codes.append("UEM_OUTSIDE_AUDIO")
        if previous_end >= 0 and start < previous_end:
            codes.append("UEM_OVERLAP_OR_UNSORTED")
            continue
        if id_ok and channel_ok and end <= duration_us:
            intervals.append((start, end))
        previous_end = end
    if not intervals:
        codes.append("UEM_EMPTY")
    return intervals, codes


def _parse_rttm(text: str, audio_id: str, duration_us: int) -> tuple[list[tuple[str, int, int]], list[str]]:
    segments: list[tuple[str, int, int]] = []
    codes: list[str] = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 10 or fields[0].upper() != "SPEAKER":
            codes.append("RTTM_SCHEMA")
            continue
        if fields[1] != audio_id:
            codes.append("FILE_ID_MISMATCH")
        if fields[2] != "1":
            codes.append("CHANNEL_MISMATCH")
        speaker = fields[7]
        if not _opaque_identifier(speaker) or _NON_SPEECH_LABEL.fullmatch(speaker.upper()):
            codes.append("PRIVACY_ID")
        try:
            start = _time_us(fields[3])
            duration = _time_us(fields[4], positive=True)
        except ValueError:
            codes.append("RTTM_TIMING")
            continue
        end = start + duration
        if end > duration_us:
            codes.append("RTTM_OUTSIDE_AUDIO")
        segments.append((speaker, start, end))
    if not segments:
        codes.append("RTTM_EMPTY")
    return segments, codes


def _contained(start: int, end: int, uem: Iterable[tuple[int, int]]) -> bool:
    return any(start >= left and end <= right for left, right in uem)


def _parse_words_text(text: str, *, expected_recording_id: str,
                      duration_us: int | None = None,
                      uem: Iterable[tuple[int, int]] = (),
                      reference_speakers: Iterable[str] = ()) -> tuple[
                          tuple[WordAnnotation, ...], tuple[tuple[str, bool], ...], list[str]
                      ]:
    """Parse and validate words without putting row values into diagnostics."""
    words: list[WordAnnotation] = []
    micro_flags: list[tuple[str, bool]] = []
    codes: list[str] = []
    seen_word_ids: set[str] = set()
    reference = set(reference_speakers)
    scored_uem = tuple(uem)
    saw_row = False
    for line in text.splitlines():
        if not line.strip():
            continue
        saw_row = True
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            codes.append("WORDS_JSON")
            continue
        if not isinstance(value, dict):
            codes.append("WORDS_ROW_SCHEMA")
            continue
        if WORD_ROW_REQUIRED_FIELDS - set(value) or set(value) - WORD_ROW_ALLOWED_FIELDS:
            codes.append("WORDS_ROW_SCHEMA")
            continue
        # All times are integer source microseconds; accepting floats here
        # would make the artifact's source-timebase contract ambiguous.
        word_id = value["word_id"]
        row_recording_id = value["recording_id"]
        start = value["start_us"]
        end = value["end_us"]
        text_value = value["text"]
        ref_speaker_id = value["ref_speaker_id"]
        flags = tuple(value[name] for name in (
            "attributable", "overlap_flag", "boundary_crossing_flag"))
        micro_flag = value.get("micro_flag", False)
        row_ok = True
        if not _opaque_identifier(row_recording_id) or row_recording_id != expected_recording_id:
            codes.append("WORDS_RECORD_ID")
            row_ok = False
        if not _opaque_identifier(word_id):
            codes.append("WORDS_ID_SCHEMA")
            row_ok = False
        if word_id in seen_word_ids:
            codes.append("WORDS_DUPLICATE_ID")
            row_ok = False
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end)):
            codes.append("WORDS_TIMING_SCHEMA")
            row_ok = False
        else:
            if start < 0 or end <= start:
                codes.append("WORDS_TIMING_SCHEMA")
                row_ok = False
            if duration_us is not None and (start < 0 or end > duration_us):
                codes.append("WORDS_OUTSIDE_AUDIO")
                row_ok = False
            if scored_uem and not _contained(start, end, scored_uem):
                codes.append("WORDS_OUTSIDE_UEM")
                row_ok = False
        if not isinstance(text_value, str):
            codes.append("WORDS_TEXT_SCHEMA")
            row_ok = False
        if any(not isinstance(flag, bool) for flag in (*flags, micro_flag)):
            codes.append("WORDS_FLAG_SCHEMA")
            row_ok = False
        if ref_speaker_id is not None:
            if ref_speaker_id != "REF_OTHER" and not _opaque_identifier(ref_speaker_id):
                codes.append("WORDS_REF_SPEAKER_SCHEMA")
                row_ok = False
            elif reference and ref_speaker_id != "REF_OTHER" and ref_speaker_id not in reference:
                codes.append("WORDS_REF_SPEAKER_UNKNOWN")
                row_ok = False
        elif flags[0] and not flags[1] and not flags[2]:
            # An eligible word must have a reference speaker.  Explicitly
            # un-attributable/overlap/boundary rows may use null.
            codes.append("WORDS_REF_SPEAKER_REQUIRED")
            row_ok = False
        if not row_ok:
            continue
        seen_word_ids.add(word_id)
        try:
            word = WordAnnotation(
                word_id=word_id,
                start_us=start,
                end_us=end,
                text=text_value,
                ref_speaker_id=ref_speaker_id,
                attributable=flags[0],
                overlap_flag=flags[1],
                boundary_crossing_flag=flags[2],
            )
        except (EvaluationError, TypeError, ValueError):
            codes.append("WORDS_ROW_SCHEMA")
            continue
        words.append(word)
        if micro_flag:
            micro_flags.append((word_id, True))
    if not saw_row:
        codes.append("WORDS_EMPTY")
    return tuple(words), tuple(micro_flags), codes


def _words_path_is_safe(path: Path) -> bool:
    """Validate a direct loader path without revealing it in errors."""
    try:
        raw = str(path)
        if "\\" in raw or any(part == ".." for part in raw.split("/")):
            return False
        return (not path.is_symlink() and path.is_file() and path.suffix.lower() == ".jsonl"
                and path.resolve(strict=True).parent == path.parent.resolve(strict=True))
    except OSError:
        return False


def load_words_artifact(path: str | Path, *, expected_recording_id: str | None = None,
                        expected_sha256: str | None = None, recording_id: str | None = None,
                        timebase: str = "microseconds",
                        duration_us: int | None = None,
                        uem: Iterable[tuple[int, int]] = (),
                        reference_speakers: Iterable[str] = ()) -> WordsArtifact:
    """Load a validated words JSONL artifact for the UEM scorer.

    This is the explicit export seam: callers receive typed evaluation objects,
    while intake reports remain aggregate and redacted.  All failures use
    stable, value-free messages so a caller cannot accidentally log a path or
    transcript through this boundary.
    """
    target = Path(path)
    if expected_recording_id is None:
        expected_recording_id = recording_id
    if (not _opaque_identifier(expected_recording_id)
            or not isinstance(expected_sha256, str) or _HEX64.fullmatch(expected_sha256) is None
            or timebase != "microseconds" or not _words_path_is_safe(target)):
        raise ValueError("invalid words artifact contract")
    try:
        raw = target.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("unable to read words artifact") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest.lower() != expected_sha256.lower():
        raise ValueError("words artifact hash mismatch")
    words, micro_flags, codes = _parse_words_text(
        text, expected_recording_id=expected_recording_id, duration_us=duration_us,
        uem=uem, reference_speakers=reference_speakers)
    if codes:
        raise ValueError("invalid words artifact")
    return WordsArtifact(expected_recording_id, timebase, digest, words, micro_flags)


load_words_jsonl_artifact = load_words_artifact
read_words_artifact = load_words_artifact


def _annotation_checks(record: _RecordStats, rttm_path: Path, uem_path: Path,
                       report: AnnotationValidationReport) -> None:
    try:
        uem_text = uem_path.read_text(encoding="utf-8")
        rttm_text = rttm_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        _issue(report, "ANNOTATION_READ", record.index)
        return
    uem, uem_codes = _parse_uem(uem_text, record.audio_id, record.duration_us)
    segments, rttm_codes = _parse_rttm(rttm_text, record.audio_id, record.duration_us)
    for code in (*uem_codes, *rttm_codes):
        _issue(report, code, record.index)
    if not uem or not segments:
        return
    speakers = {speaker for speaker, _, _ in segments}
    record.speakers = speakers
    record.segment_count = len(segments)
    record.short_turn = any((end - start) <= 1_000_000 for _, start, end in segments)
    if len(speakers) != record.speaker_count:
        _issue(report, "SPEAKER_COUNT_MISMATCH", record.index, "speaker_count")
    if any(not _contained(start, end, uem) for _, start, end in segments):
        _issue(report, "RTTM_OUTSIDE_UEM", record.index)
    # Measure the union of regions with at least two distinct active speakers;
    # a three-way overlap is counted once rather than once per speaker pair.
    boundaries = sorted({point for _, start, end in segments for point in (start, end)})
    overlap = 0
    for left, right in zip(boundaries, boundaries[1:]):
        if len({speaker for speaker, start, end in segments if start < right and end > left}) >= 2:
            overlap += right - left
    record.overlap_us = overlap
    record.valid_annotations = not any(
        issue.record_index == record.index and issue.code in {
            "ANNOTATION_READ", "UEM_SCHEMA", "UEM_TIMING", "UEM_OUTSIDE_AUDIO",
            "UEM_EMPTY", "RTTM_SCHEMA", "RTTM_TIMING", "RTTM_OUTSIDE_AUDIO",
            "RTTM_EMPTY", "FILE_ID_MISMATCH", "CHANNEL_MISMATCH", "PRIVACY_ID",
            "RTTM_OUTSIDE_UEM", "SPEAKER_COUNT_MISMATCH",
        } for issue in report.errors
    )


def _words_checks(record: _RecordStats, words_path: Path, row: Mapping[str, Any],
                  report: AnnotationValidationReport, *, uem: Iterable[tuple[int, int]]) -> None:
    """Validate a declared words artifact and retain only aggregate counters."""
    expected_hash = row.get("words_sha256")
    if not isinstance(expected_hash, str) or _HEX64.fullmatch(expected_hash) is None:
        return
    try:
        raw = words_path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError):
        _issue(report, "WORDS_READ", record.index)
        return
    hash_ok = hashlib.sha256(raw).hexdigest().lower() == expected_hash.lower()
    if not hash_ok:
        _issue(report, "WORDS_HASH_MISMATCH", record.index, "words_sha256")
    words, micro_flags, codes = _parse_words_text(
        text,
        expected_recording_id=record.audio_id,
        duration_us=record.duration_us if record.valid_audio else None,
        uem=uem,
        reference_speakers=record.speakers,
    )
    for code in codes:
        _issue(report, code, record.index)
    record.words_count = len(words)
    record.micro_flagged_words = len(micro_flags)
    record.words_valid = not codes and hash_ok and row.get("words_timebase") == "microseconds"


def _record_from_row(row: Mapping[str, Any], index: int, report: AnnotationValidationReport) -> _RecordStats | None:
    if set(row) - ALLOWED_FIELDS:
        _issue(report, "MANIFEST_UNKNOWN_FIELD", index)
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        _issue(report, "MANIFEST_REQUIRED_FIELD", index)
    if missing or set(row) - ALLOWED_FIELDS:
        return None
    words_present = bool(WORD_ARTIFACT_FIELDS & set(row))
    if words_present and WORD_ARTIFACT_FIELDS - set(row):
        _issue(report, "WORDS_FIELDS", index)
    scalar_ids = ("audio_id", "session_id", "recording_id", "source_recording_id", "augmentation_group")
    for name in scalar_ids:
        if name in row and row[name] is not None and not _opaque_identifier(row[name]):
            _issue(report, "PRIVACY_ID", index, name)
    audio_hash = row["audio_sha256"]
    if not isinstance(audio_hash, str) or _HEX64.fullmatch(audio_hash) is None:
        _issue(report, "AUDIO_HASH_SCHEMA", index, "audio_sha256")
    if words_present:
        words_hash = row.get("words_sha256")
        if not isinstance(words_hash, str) or _HEX64.fullmatch(words_hash) is None:
            _issue(report, "WORDS_HASH_SCHEMA", index, "words_sha256")
        if row.get("words_timebase") != "microseconds":
            _issue(report, "WORDS_TIMEBASE_SCHEMA", index, "words_timebase")
    sample_rate = row["sample_rate_hz"]
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate not in SAMPLE_RATES:
        _issue(report, "SAMPLE_RATE_SCHEMA", index, "sample_rate_hz")
    count = row["speaker_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count not in {1, 2}:
        _issue(report, "SPEAKER_COUNT_SCHEMA", index, "speaker_count")
    gender = row["gender_pair"]
    if not isinstance(gender, str) or gender.upper() not in GENDER_PAIRS:
        _issue(report, "GENDER_PAIR_SCHEMA", index, "gender_pair")
    conditions = row["conditions"]
    if not isinstance(conditions, list) or not conditions or any(
        not isinstance(item, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", item.lower())
        for item in conditions
    ):
        _issue(report, "CONDITIONS_SCHEMA", index, "conditions")
        conditions = []
    elif len({item.lower() for item in conditions}) != len(conditions):
        _issue(report, "CONDITIONS_DUPLICATE", index, "conditions")
    split = row["split"]
    if not isinstance(split, str) or split.upper() not in REQUIRED_SPLITS:
        _issue(report, "SPLIT_SCHEMA", index, "split")
        split = "INVALID"
    # Keep a redacted placeholder for malformed IDs so independent checks
    # (hash, path and WAV layout) are still collected in the same invocation.
    audio_id = row["audio_id"] if _opaque_identifier(row["audio_id"]) else f"invalid-{index}"
    session_id = row["session_id"] if _opaque_identifier(row["session_id"]) else f"invalid-{index}"
    recording_raw = row.get("recording_id") or audio_id
    recording_id = recording_raw if _opaque_identifier(recording_raw) else f"invalid-{index}"
    source_raw = row.get("source_recording_id")
    source_recording_id = source_raw if _opaque_identifier(source_raw) else None
    augmentation_raw = row.get("augmentation_group")
    augmentation_group = augmentation_raw if _opaque_identifier(augmentation_raw) else None
    # A malformed scalar field remains an error, but returning a record allows
    # leakage and required-split diagnostics to be collected in one run.
    return _RecordStats(
        index=index,
        audio_id=audio_id,
        session_id=session_id,
        recording_id=recording_id,
        source_recording_id=source_recording_id,
        augmentation_group=augmentation_group,
        split=split,
        sample_rate_hz=sample_rate if isinstance(sample_rate, int) else 0,
        speaker_count=count if isinstance(count, int) else 0,
        gender_pair=gender.upper() if isinstance(gender, str) else "INVALID",
        conditions=tuple(str(item).lower() for item in conditions),
        words_declared=words_present,
    )


def _readiness(records: list[_RecordStats], report: AnnotationValidationReport) -> dict[str, Any]:
    bad_indexes = {issue.record_index for issue in report.errors if issue.record_index is not None}
    usable = [record for record in records
              if record.valid_audio and record.valid_annotations and record.index not in bad_indexes]
    split_counts = Counter(record.split for record in usable if record.split in REQUIRED_SPLITS)
    gender_counts = Counter(record.gender_pair for record in usable)
    rate_counts = Counter(str(record.sample_rate_hz) for record in usable)
    condition_counts: Counter[str] = Counter()
    for record in usable:
        for condition in record.conditions:
            # The guide permits qualifiers such as ``quiet-secondary`` while
            # the readiness inventory intentionally reports stable buckets.
            if condition.startswith("quiet"):
                condition_counts["quiet"] += 1
            elif condition.startswith("noisy"):
                condition_counts["noisy"] += 1
            elif condition in {"near", "far"}:
                condition_counts[condition] += 1
        if record.short_turn:
            condition_counts["short-turn"] += 1
        if record.overlap_us > 0:
            condition_counts["overlap"] += 1
    required = {split: {"count": split_counts.get(split, 0), "present": split_counts.get(split, 0) > 0}
                for split in REQUIRED_SPLITS}
    gender = {key: gender_counts.get(key, 0) for key in ("MM", "FF", "MF")}
    rates = {key: rate_counts.get(key, 0) for key in ("8000", "16000")}
    condition_keys = ("near", "far", "noisy", "quiet", "short-turn", "overlap")
    conditions = {key: condition_counts.get(key, 0) for key in condition_keys}
    tranche = {
        "one_speaker_files": sum(r.speaker_count == 1 for r in usable),
        "two_speaker_files": sum(r.speaker_count == 2 for r in usable),
        "minimum_one_speaker": sum(r.speaker_count == 1 for r in usable) >= 2,
        "minimum_two_speaker": sum(r.speaker_count == 2 for r in usable) >= 8,
    }
    checks = [*required.values(), *({"count": n, "present": n > 0} for n in gender.values()),
              *({"count": n, "present": n > 0} for n in rates.values()),
              *({"count": n, "present": n > 0} for n in conditions.values())]
    independent_ready = not report.errors and all(item["present"] for item in checks) and all(
        tranche[key] for key in ("minimum_one_speaker", "minimum_two_speaker")
    )
    for split, state in required.items():
        if not state["present"]:
            _issue(report, "REQUIRED_SPLIT_MISSING", field=split)
    if not all(item["present"] for item in checks):
        _issue(report, "SUBGROUP_INCOMPLETE", warning=True)
    if not tranche["minimum_one_speaker"] or not tranche["minimum_two_speaker"]:
        _issue(report, "MINIMUM_TRANCHE_INCOMPLETE", warning=True)
    return {
        "required_splits": required,
        "gender_pair": gender,
        "sample_rate_hz": rates,
        "conditions": conditions,
        "overlap_duration_seconds": round(sum(r.overlap_us for r in usable) / 1_000_000, 6),
        "records_with_overlap": sum(r.overlap_us > 0 for r in usable),
        "words": {
            "records_with_artifact": sum(r.words_declared and r.words_valid for r in usable),
            "rows": sum(r.words_count for r in usable),
            "micro_flagged_rows": sum(r.micro_flagged_words for r in usable),
        },
        "minimum_tranche": tranche,
        "ready_for_independent_validation": independent_ready,
    }


def validate_annotation_dataset(manifest_path: str | Path, *, dataset_root: str | Path | None = None) -> AnnotationValidationReport:
    """Validate a manifest and its local annotation files without side effects."""
    manifest = Path(manifest_path)
    root = Path(dataset_root) if dataset_root is not None else manifest.parent
    report = AnnotationValidationReport()
    records: list[_RecordStats] = []
    audio_ids_seen: set[str] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        _issue(report, "MANIFEST_READ")
        report.readiness = _readiness(records, report)
        return report
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        report.records_seen += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            _issue(report, "MANIFEST_JSON", index)
            continue
        if not isinstance(row, dict):
            _issue(report, "MANIFEST_ROW_SCHEMA", index)
            continue
        record = _record_from_row(row, index, report)
        if record is None:
            continue
        if record.audio_id in audio_ids_seen:
            _issue(report, "DUPLICATE_AUDIO_ID", index, "audio_id")
        audio_ids_seen.add(record.audio_id)
        records.append(record)
        paths: dict[str, Path] = {}
        for name in PATH_FIELDS:
            path = _check_path(root, row[name])
            if path is None:
                _issue(report, "PATH_INVALID", index, name)
            else:
                paths[name] = path
        if record.words_declared:
            words_path = _check_path(root, row.get("words"))
            if words_path is None:
                _issue(report, "PATH_INVALID", index, "words")
            elif words_path.suffix.lower() != ".jsonl":
                _issue(report, "WORDS_PATH_SCHEMA", index, "words")
            else:
                paths["words"] = words_path
        if "audio" not in paths:
            if "words" in paths:
                _words_checks(record, paths["words"], row, report, uem=())
            continue
        try:
            if _sha256(paths["audio"]).lower() != str(row["audio_sha256"]).lower():
                _issue(report, "AUDIO_HASH_MISMATCH", index, "audio_sha256")
            layout = WavPcmAccessor(paths["audio"]).layout
            if layout.channel_count != 1 or layout.sample_width_bytes not in {1, 2, 3, 4}:
                _issue(report, "WAV_LAYOUT", index, "audio")
            if layout.sample_rate_hz != record.sample_rate_hz:
                _issue(report, "WAV_RATE_MISMATCH", index, "sample_rate_hz")
            if layout.frame_count <= 0 or layout.sample_rate_hz <= 0:
                _issue(report, "WAV_DURATION", index, "audio")
            record.duration_us = round(layout.frame_count * 1_000_000 / layout.sample_rate_hz)
            record.valid_audio = True
        except (OSError, EOFError, MediaError):
            _issue(report, "WAV_INVALID", index, "audio")
            continue
        if "rttm" in paths and "uem" in paths:
            _annotation_checks(record, paths["rttm"], paths["uem"], report)
        if "words" in paths:
            scored_uem: tuple[tuple[int, int], ...] = ()
            if "uem" in paths:
                try:
                    uem_text = paths["uem"].read_text(encoding="utf-8")
                    scored_uem, _ = _parse_uem(uem_text, record.audio_id, record.duration_us)
                except (OSError, UnicodeError):
                    scored_uem = ()
            _words_checks(record, paths["words"], row, report, uem=scored_uem)
        if record.valid_audio and record.valid_annotations and not any(
            issue.record_index == record.index for issue in report.errors
        ):
            report.records_valid += 1
    # Existing evaluation guard is the authoritative split-leakage rule.
    try:
        validate_recording_session_splits(RecordingManifest(
            recording_id=r.recording_id, session_id=r.session_id, split=r.split,
            source_recording_id=r.source_recording_id,
            augmentation_group=r.augmentation_group,
        ) for r in records)
    except SplitLeakageError:
        _issue(report, "SPLIT_LEAKAGE")
    report.readiness = _readiness(records, report)
    return report


# Explicit name for callers that prefer the document terminology.
validate_annotation_manifest = validate_annotation_dataset


__all__ = [
    "AnnotationValidationReport", "IntakeIssue", "WordsArtifact", "REQUIRED_SPLITS",
    "load_words_artifact", "load_words_jsonl_artifact", "read_words_artifact",
    "validate_annotation_dataset", "validate_annotation_manifest",
]
