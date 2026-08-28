"""Offline, content-addressed runner for the Korean diarization benchmark.

This module is intentionally a small I/O boundary around :mod:`sddiar.evaluation`.
The annotation intake validator remains the authority for the reference dataset
schema.  Once that validator succeeds, this module snapshots the selected WAV
header/hash plus reference RTTM/UEM and prediction RTTM, then passes only typed
annotations to the deterministic scorer.  PCM samples are never decoded.

The public result is deliberately aggregate-only.  Paths, transcript text,
speaker labels, and record identifiers are never included in a result or an
error message.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from .annotation_intake import CONDITION_ALLOWLIST, validate_annotation_dataset
from .evaluation import (
    EvaluationError,
    EvaluationRecording,
    RecordingManifest,
    ScoringConfig,
    SplitLeakageError,
    score_corpus,
    parse_rttm,
    parse_uem,
    validate_recording_session_splits,
)
from .korean_benchmark import (
    KoreanBenchmarkError,
    KoreanReleasePolicy,
    evaluate_benchmark_eligibility,
    evaluate_korean_release_gate,
    parse_korean_corpus_lock,
)
from .media import MediaError, _parse_wav_layout


_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_KNOWN_BAD_IDS = frozenset({
    "unknown", "overlap", "anonymous", "anonymized", "name", "realname",
    "customer", "client", "transcript", "speaker", "person",
})
_PREDICTION_FIELDS = frozenset({"audio_id", "rttm", "rttm_sha256", "quality_status"})
RESULT_SCHEMA_VERSION = "sddiar-korean-benchmark-v1"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ANNOTATION_BYTES = 64 * 1024 * 1024
MAX_AUDIO_BYTES = 8 * 1024 * 1024 * 1024
MAX_BOOTSTRAP_ITERATIONS = 100_000
MAX_BOOTSTRAP_WORK = 200_000_000
_BOOTSTRAP_METRIC_FACTOR = 16
_SPLITS = frozenset({"CALIBRATION", "DEVELOPMENT_HOLDOUT", "RELEASE_HOLDOUT"})
_GENDER_PAIRS = frozenset({"MM", "FF", "MF", "M", "F", "UNKNOWN"})
_SAMPLE_RATES = frozenset({8_000, 16_000})
_QUALITY_STATUSES = frozenset({
    "PASS_HIGH", "PASS_STANDARD", "PASS_WITH_UNATTRIBUTED", "REVIEW_REQUIRED", "UNSUPPORTED",
})
_RESERVED_HYPOTHESIS_LABELS = frozenset({
    "UNKNOWN", "UNKNOWN_SHORT", "OVERLAP", "NON_SPEECH", "SIL", "SILENCE",
})


class EvaluationIOError(ValueError):
    """A redacted, stable input-contract error."""

    def __init__(self, code: str, *, record_index: int | None = None,
                 field: str | None = None) -> None:
        self.code = code
        self.record_index = record_index
        self.field = field
        super().__init__(code)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "severity": "error"}
        if self.record_index is not None:
            out["record_index"] = self.record_index
        if self.field is not None:
            out["field"] = self.field
        return out


@dataclass(frozen=True)
class _ReferenceRow:
    index: int
    audio_id: str
    recording_id: str
    session_id: str
    source_recording_id: str | None
    augmentation_group: str | None
    rttm_path: Path
    uem_path: Path
    rttm_bytes: bytes
    uem_bytes: bytes
    split: str
    gender_pair: str
    sample_rate_hz: int
    audio_duration_us: int | None
    speaker_count: int
    conditions: tuple[str, ...]
    speaker_group_ids: tuple[str, ...]
    reference_status: str
    uem_policy: str
    conversion_evidence_sha256: str


@dataclass(frozen=True)
class _PredictionRow:
    index: int
    audio_id: str
    rttm_path: Path
    rttm_bytes: bytes
    quality_status: str


def _safe_root(value: str | Path | None, fallback: Path) -> Path:
    root = fallback if value is None else Path(value)
    try:
        if root.is_symlink():
            raise EvaluationIOError("ROOT_INVALID")
        resolved = root.resolve(strict=True)
        if not resolved.is_dir() or resolved.is_symlink():
            raise EvaluationIOError("ROOT_INVALID")
        return resolved
    except (OSError, RuntimeError):
        raise EvaluationIOError("ROOT_INVALID") from None


def _safe_relative_path(root: Path, value: Any, *, field: str,
                        suffix: str | None = None) -> Path:
    """Resolve a manifest path while rejecting traversal and symlinks."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise EvaluationIOError("PATH_INVALID", field=field)
    # Reject Windows separators and URI syntax on every host for portable
    # manifests.  A path containing a dot component is not canonical enough for
    # a content-addressed run.
    if "\\" in value or _URI_SCHEME.match(value):
        raise EvaluationIOError("PATH_INVALID", field=field)
    candidate = Path(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise EvaluationIOError("PATH_INVALID", field=field)
    target = root / candidate
    try:
        target.resolve(strict=False).relative_to(root)
    except (OSError, ValueError, RuntimeError):
        raise EvaluationIOError("PATH_INVALID", field=field) from None
    current = root
    try:
        for part in candidate.parts:
            current = current / part
            if current.is_symlink():
                raise EvaluationIOError("PATH_INVALID", field=field)
        if not target.is_file():
            raise EvaluationIOError("PATH_MISSING", field=field)
        if suffix is not None and target.suffix.lower() != suffix:
            raise EvaluationIOError("PATH_SCHEMA", field=field)
    except OSError:
        raise EvaluationIOError("PATH_INVALID", field=field) from None
    return target


def _read_bytes(path: Path, *, field: str, max_bytes: int = MAX_ANNOTATION_BYTES) -> bytes:
    fd: int | None = None
    try:
        metadata = os.lstat(path)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if (path.is_symlink() or not stat.S_ISREG(metadata.st_mode)
                or (reparse and attributes & reparse)):
            raise EvaluationIOError("FILE_TYPE_INVALID", field=field)
        if metadata.st_size > max_bytes:
            raise EvaluationIOError("FILE_SIZE_LIMIT", field=field)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > max_bytes:
            raise EvaluationIOError("FILE_TYPE_INVALID", field=field)
        if ((metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino)
                and metadata.st_ino and opened.st_ino):
            raise EvaluationIOError("FILE_CHANGED_DURING_OPEN", field=field)
        with os.fdopen(fd, "rb", buffering=0) as handle:
            fd = None
            chunks: list[bytes] = []
            total = 0
            while True:
                block = handle.read(min(1 << 20, max_bytes - total + 1))
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise EvaluationIOError("FILE_SIZE_LIMIT", field=field)
                chunks.append(block)
            return b"".join(chunks)
    except EvaluationIOError:
        raise
    except (OSError, UnicodeError):
        raise EvaluationIOError("FILE_READ", field=field) from None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _snapshot_wav(path: Path) -> tuple[Any, str]:
    """Inspect and hash one regular WAV through the same open descriptor."""
    # Keep package import Windows-safe in environments that probe without the
    # platform's native ``_winapi`` module; the evaluator needs tempfile only
    # when an actual benchmark WAV is snapshotted.
    import tempfile

    fd: int | None = None
    try:
        metadata = os.lstat(path)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if (path.is_symlink() or not stat.S_ISREG(metadata.st_mode)
                or (reparse and attributes & reparse) or metadata.st_size > MAX_AUDIO_BYTES):
            raise EvaluationIOError("AUDIO_FILE_INVALID", field="audio")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_size > MAX_AUDIO_BYTES
                or ((metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino)
                    and metadata.st_ino and opened.st_ino)):
            raise EvaluationIOError("AUDIO_FILE_CHANGED", field="audio")
        with os.fdopen(fd, "rb", buffering=0) as handle, tempfile.SpooledTemporaryFile(
            max_size=8 * 1024 * 1024, mode="w+b",
        ) as snapshot:
            fd = None
            digest = hashlib.sha256()
            total = 0
            while True:
                block = handle.read(1 << 20)
                if not block:
                    break
                total += len(block)
                if total > MAX_AUDIO_BYTES:
                    raise EvaluationIOError("AUDIO_FILE_SIZE_LIMIT", field="audio")
                digest.update(block)
                snapshot.write(block)
            final = os.fstat(handle.fileno())
            if (total != opened.st_size or final.st_size != opened.st_size
                    or (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino)):
                raise EvaluationIOError("AUDIO_FILE_CHANGED", field="audio")
            snapshot.seek(0)
            layout = _parse_wav_layout(snapshot, total)
            return layout, digest.hexdigest()
    except EvaluationIOError:
        raise
    except (MediaError, OSError, EOFError, ValueError, ZeroDivisionError):
        raise EvaluationIOError("AUDIO_FILE_INVALID", field="audio") from None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json(raw: bytes, *, code: str) -> Any:
    try:
        def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in values:
                if key in out:
                    raise ValueError("duplicate")
                out[key] = value
            return out

        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, RecursionError):
        raise EvaluationIOError(code) from None


def _opaque_id(value: Any) -> bool:
    return (isinstance(value, str) and bool(value) and _ID.fullmatch(value) is not None
            and any(char.isdigit() for char in value) and value.lower() not in _KNOWN_BAD_IDS)


def _strict_rttm_shape(text: str) -> bool:
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 10 or fields[0].upper() != "SPEAKER" or fields[2] != "1":
            return False
    return True


def _uem_is_sorted_nonoverlapping(intervals: tuple[Any, ...]) -> bool:
    previous_end = -1
    for interval in intervals:
        if interval.start_us < previous_end:
            return False
        previous_end = interval.end_us
    return True


def _inside_reference_uem(record: Any, intervals: tuple[Any, ...]) -> bool:
    return any(
        interval.channel == record.channel
        and interval.start_us <= record.start_us
        and record.end_us <= interval.end_us
        for interval in intervals
    )


def _read_jsonl(path: Path, *, code_prefix: str) -> list[tuple[int, Mapping[str, Any]]]:
    raw = _read_bytes(path, field="manifest", max_bytes=MAX_MANIFEST_BYTES)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise EvaluationIOError(f"{code_prefix}_MANIFEST_ENCODING") from None
    rows: list[tuple[int, Mapping[str, Any]]] = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = _strict_json(line.encode("utf-8"), code=f"{code_prefix}_MANIFEST_JSON")
        except EvaluationIOError:
            raise EvaluationIOError(f"{code_prefix}_MANIFEST_JSON", record_index=index) from None
        if not isinstance(value, dict):
            raise EvaluationIOError(f"{code_prefix}_MANIFEST_ROW", record_index=index) from None
        rows.append((index, value))
    return rows


def _read_reference_rows(
    manifest_path: Path, root: Path, *, selected_split: str
) -> tuple[list[_ReferenceRow], bytes]:
    manifest_bytes = _read_bytes(manifest_path, field="reference_manifest", max_bytes=MAX_MANIFEST_BYTES)
    try:
        text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise EvaluationIOError("REFERENCE_MANIFEST_ENCODING") from None
    rows: list[_ReferenceRow] = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = _strict_json(line.encode("utf-8"), code="REFERENCE_MANIFEST_JSON")
        except EvaluationIOError:
            raise EvaluationIOError("REFERENCE_MANIFEST_JSON", record_index=index) from None
        if not isinstance(row, dict):
            raise EvaluationIOError("REFERENCE_MANIFEST_ROW", record_index=index)
        # The intake gate checked this exact schema.  Keep this check here too,
        # because the manifest can change between validation and loading.
        required = {
            "audio_id", "audio", "audio_sha256", "session_id", "speaker_count",
            "rttm", "uem", "rttm_sha256", "uem_sha256", "split", "gender_pair",
            "sample_rate_hz", "conditions", "speaker_group_ids", "reference_status",
            "uem_policy", "conversion_evidence_sha256",
        }
        if not required.issubset(row):
            raise EvaluationIOError("REFERENCE_MANIFEST_SCHEMA", record_index=index)
        audio_id = row.get("audio_id")
        if not _opaque_id(audio_id):
            raise EvaluationIOError("REFERENCE_ID_SCHEMA", record_index=index, field="audio_id")
        recording_id = row.get("recording_id", audio_id)
        session_id = row.get("session_id")
        source_recording_id = row.get("source_recording_id")
        augmentation_group = row.get("augmentation_group")
        if (not _opaque_id(recording_id) or not _opaque_id(session_id)
                or (source_recording_id is not None and not _opaque_id(source_recording_id))
                or (augmentation_group is not None and not _opaque_id(augmentation_group))):
            raise EvaluationIOError("REFERENCE_GROUP_ID_SCHEMA", record_index=index)
        rttm_path = _safe_relative_path(root, row.get("rttm"), field="rttm")
        uem_path = _safe_relative_path(root, row.get("uem"), field="uem")
        rttm_bytes = _read_bytes(rttm_path, field="rttm")
        uem_bytes = _read_bytes(uem_path, field="uem")
        for field_name, payload in (("rttm_sha256", rttm_bytes), ("uem_sha256", uem_bytes)):
            digest = row.get(field_name)
            if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
                raise EvaluationIOError("REFERENCE_HASH_SCHEMA", record_index=index, field=field_name)
            if _sha256(payload).lower() != digest.lower():
                raise EvaluationIOError("REFERENCE_HASH_MISMATCH", record_index=index, field=field_name)
        split = row.get("split")
        gender = row.get("gender_pair")
        rate = row.get("sample_rate_hz")
        speaker_count = row.get("speaker_count")
        conditions = row.get("conditions")
        speaker_groups = row.get("speaker_group_ids")
        reference_status = row.get("reference_status")
        uem_policy = row.get("uem_policy")
        conversion_evidence_sha256 = row.get("conversion_evidence_sha256")
        normalized_conditions = (
            [condition.lower() for condition in conditions]
            if isinstance(conditions, list) and all(isinstance(condition, str) for condition in conditions)
            else []
        )
        if (not isinstance(split, str) or split.upper() not in _SPLITS
                or not isinstance(gender, str) or gender.upper() not in _GENDER_PAIRS
                or type(rate) is not int or rate not in _SAMPLE_RATES
                or type(speaker_count) is not int or speaker_count not in {1, 2}
                or not normalized_conditions
                or any(condition not in CONDITION_ALLOWLIST for condition in normalized_conditions)
                or len(normalized_conditions) != len(set(normalized_conditions))
                or not isinstance(speaker_groups, list) or not speaker_groups
                or len(speaker_groups) != speaker_count
                or len(speaker_groups) != len(set(speaker_groups))
                or any(not _opaque_id(group) for group in speaker_groups)
                or not isinstance(reference_status, str)
                or reference_status not in {"CONVERTED_PROVISIONAL", "GOLD_APPROVED", "SILVER", "CHALLENGE"}
                or not isinstance(uem_policy, str)
                or uem_policy not in {"ANNOTATED_EXTENT_PROVISIONAL", "FULL_AUDIO", "AUDITED_EXCLUSIONS"}
                or not isinstance(conversion_evidence_sha256, str)
                or _HEX64.fullmatch(conversion_evidence_sha256) is None
                or (reference_status == "GOLD_APPROVED"
                    and uem_policy == "ANNOTATED_EXTENT_PROVISIONAL")):
            raise EvaluationIOError("REFERENCE_MANIFEST_SCHEMA", record_index=index)
        audio_duration_us: int | None = None
        if split.upper() == selected_split:
            audio_digest = row.get("audio_sha256")
            if not isinstance(audio_digest, str) or _HEX64.fullmatch(audio_digest) is None:
                raise EvaluationIOError("REFERENCE_HASH_SCHEMA", record_index=index,
                                        field="audio_sha256")
            audio_path = _safe_relative_path(root, row.get("audio"), field="audio", suffix=".wav")
            layout, observed_audio_digest = _snapshot_wav(audio_path)
            if observed_audio_digest.lower() != audio_digest.lower():
                raise EvaluationIOError("REFERENCE_HASH_MISMATCH", record_index=index,
                                        field="audio_sha256")
            if (layout.channel_count != 1 or layout.sample_width_bytes not in {1, 2, 3, 4}
                    or layout.sample_rate_hz != rate or layout.frame_count <= 0):
                raise EvaluationIOError("REFERENCE_AUDIO_SCHEMA", record_index=index)
            audio_duration_us = round(layout.frame_count * 1_000_000 / layout.sample_rate_hz)
        rows.append(_ReferenceRow(
            index, audio_id, recording_id, session_id, source_recording_id,
            augmentation_group, rttm_path, uem_path, rttm_bytes, uem_bytes,
            split.upper(), gender.upper(), rate, audio_duration_us, speaker_count,
            tuple(sorted(normalized_conditions)),
            tuple(sorted(speaker_groups)),
            reference_status, uem_policy, conversion_evidence_sha256,
        ))
    if not rows:
        raise EvaluationIOError("REFERENCE_MANIFEST_EMPTY")
    return rows, manifest_bytes


def _read_prediction_rows(manifest_path: Path, root: Path) -> tuple[list[_PredictionRow], bytes]:
    manifest_bytes = _read_bytes(manifest_path, field="prediction_manifest", max_bytes=MAX_MANIFEST_BYTES)
    try:
        text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise EvaluationIOError("PREDICTION_MANIFEST_ENCODING") from None
    rows: list[_PredictionRow] = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = _strict_json(line.encode("utf-8"), code="PREDICTION_MANIFEST_JSON")
        except EvaluationIOError:
            raise EvaluationIOError("PREDICTION_MANIFEST_JSON", record_index=index) from None
        if not isinstance(row, dict) or set(row) != _PREDICTION_FIELDS:
            raise EvaluationIOError("PREDICTION_MANIFEST_SCHEMA", record_index=index)
        audio_id = row.get("audio_id")
        quality_status = row.get("quality_status")
        digest = row.get("rttm_sha256")
        if not _opaque_id(audio_id):
            raise EvaluationIOError("PREDICTION_ID_SCHEMA", record_index=index, field="audio_id")
        if not isinstance(quality_status, str) or quality_status not in _QUALITY_STATUSES:
            raise EvaluationIOError("PREDICTION_QUALITY_SCHEMA", record_index=index,
                                    field="quality_status")
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise EvaluationIOError("PREDICTION_HASH_SCHEMA", record_index=index,
                                    field="rttm_sha256")
        rttm_path = _safe_relative_path(root, row.get("rttm"), field="rttm")
        rttm_bytes = _read_bytes(rttm_path, field="rttm")
        if _sha256(rttm_bytes).lower() != digest.lower():
            raise EvaluationIOError("PREDICTION_HASH_MISMATCH", record_index=index,
                                    field="rttm_sha256")
        rows.append(_PredictionRow(index, audio_id, rttm_path, rttm_bytes, quality_status))
    if not rows:
        raise EvaluationIOError("PREDICTION_MANIFEST_EMPTY")
    return rows, manifest_bytes


def _has_reference_overlap(reference: tuple[Any, ...], uem: tuple[Any, ...]) -> bool:
    """Detect actual two-speaker overlap, constrained to scored UEM."""
    for left_index, left in enumerate(reference):
        for right in reference[left_index + 1:]:
            if left.channel != right.channel or left.speaker_id == right.speaker_id:
                continue
            start = max(left.start_us, right.start_us)
            end = min(left.end_us, right.end_us)
            if end <= start:
                continue
            if any(interval.file_id == left.file_id and interval.channel == left.channel
                   and interval.start_us <= start and end <= interval.end_us
                   for interval in uem):
                return True
    return False


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _redacted_result(report: Any) -> dict[str, Any]:
    """Drop per-recording scores from the scorer's otherwise useful report."""
    reference_us = sum(item.reference_duration_us for score in report.recordings for item in score.speakers)
    assigned_us = sum(item.assigned_duration_us for score in report.recordings for item in score.speakers)
    correct_us = sum(item.correct_duration_us for score in report.recordings for item in score.speakers)
    per_record_coverage: list[float] = []
    per_record_accuracy: list[float] = []
    for score in report.recordings:
        ref = sum(item.reference_duration_us for item in score.speakers)
        assigned = sum(item.assigned_duration_us for item in score.speakers)
        correct = sum(item.correct_duration_us for item in score.speakers)
        per_record_coverage.append(assigned / ref if ref else 0.0)
        per_record_accuracy.append(correct / assigned if assigned else 0.0)
    count = len(report.recordings)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "overall": _json_value(report.overall),
        "metric_views": {
            "recording_count": count,
            "reference_speaker_us": reference_us,
            "assigned_speaker_us": assigned_us,
            "correct_speaker_us": correct_us,
            "der_duration_micro": report.overall.diarization_all.der,
            "der_recording_macro": sum(score.diarization_all.der for score in report.recordings) / count,
            "der_nonoverlap_duration_micro": report.overall.diarization_nonoverlap.der,
            "der_nonoverlap_recording_macro": sum(
                score.diarization_nonoverlap.der for score in report.recordings
            ) / count,
            "jer_recording_macro": report.overall.jer,
            "nonoverlap_speech_coverage_duration_micro": assigned_us / reference_us if reference_us else 0.0,
            "nonoverlap_speech_coverage_recording_macro": sum(per_record_coverage) / count,
            "assigned_accuracy_duration_micro": correct_us / assigned_us if assigned_us else 0.0,
            "assigned_accuracy_recording_macro": sum(per_record_accuracy) / count,
        },
        "subgroups": [
            {"subgroup": subgroup.subgroup,
             "aggregate": _json_value(subgroup.aggregate),
             "bootstrap": [_json_value(item) for item in subgroup.bootstrap]}
            for subgroup in report.subgroups
        ],
        "bootstrap": [_json_value(item) for item in report.bootstrap],
        "run_manifest": report.run_manifest.as_dict(),
    }


def evaluate_korean_benchmark(
    reference_manifest: str | Path,
    prediction_manifest: str | Path,
    *,
    corpus_lock: str | Path,
    split: str,
    dataset_root: str | Path | None = None,
    prediction_root: str | Path | None = None,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    """Evaluate a prediction JSONL against an independently validated corpus.

    ``reference_manifest`` uses the existing annotation-intake schema.  The
    prediction manifest uses exactly ``audio_id``, ``rttm``,
    ``rttm_sha256``, and ``quality_status``.  Both manifests are local files;
    no network, audio decoder, model, or service client is used here (the
    intake validator performs its required WAV structural check).
    """
    if split not in _SPLITS:
        raise EvaluationIOError("SPLIT_INVALID", field="split")
    if (isinstance(bootstrap_iterations, bool) or not isinstance(bootstrap_iterations, int)
            or not 0 <= bootstrap_iterations <= MAX_BOOTSTRAP_ITERATIONS):
        raise EvaluationIOError("SCORING_CONFIG_INVALID", field="bootstrap_iterations")
    reference_manifest_path = Path(reference_manifest)
    prediction_manifest_path = Path(prediction_manifest)
    corpus_lock_path = Path(corpus_lock)
    # Bind validation to a pre-validation manifest snapshot.  The strict typed
    # loader below must observe the exact same bytes or the run fails closed.
    reference_manifest_before_validation = _read_bytes(
        reference_manifest_path, field="reference_manifest", max_bytes=MAX_MANIFEST_BYTES,
    )
    # This call is deliberately before prediction loading and before reference
    # annotation parsing: invalid reference data must never be scored.
    validation = validate_annotation_dataset(reference_manifest_path, dataset_root=dataset_root)
    if not validation.ok:
        raise EvaluationIOError("REFERENCE_VALIDATION_FAILED")
    try:
        config = ScoringConfig(
            der_collar_us=250_000,
            scd_collar_us=500_000,
            bootstrap_iterations=bootstrap_iterations,
        )
        strict_config = ScoringConfig(
            der_collar_us=0,
            scd_collar_us=500_000,
            bootstrap_iterations=0,
        )
    except (EvaluationError, TypeError, ValueError):
        raise EvaluationIOError("SCORING_CONFIG_INVALID", field="bootstrap_iterations") from None
    reference_root = _safe_root(dataset_root, reference_manifest_path.parent)
    prediction_root_path = _safe_root(prediction_root, prediction_manifest_path.parent)
    lock_bytes = _read_bytes(corpus_lock_path, field="corpus_lock", max_bytes=MAX_MANIFEST_BYTES)
    try:
        lock_value = _strict_json(lock_bytes, code="CORPUS_LOCK_INVALID")
        lock = parse_korean_corpus_lock(lock_value)
    except EvaluationIOError:
        raise
    except (KoreanBenchmarkError, TypeError, ValueError):
        raise EvaluationIOError("CORPUS_LOCK_INVALID") from None
    references, reference_manifest_bytes = _read_reference_rows(
        reference_manifest_path, reference_root, selected_split=split,
    )
    if reference_manifest_bytes != reference_manifest_before_validation:
        raise EvaluationIOError("REFERENCE_MANIFEST_CHANGED")
    predictions, prediction_manifest_bytes = _read_prediction_rows(prediction_manifest_path, prediction_root_path)
    reference_manifest_digest = _sha256(reference_manifest_bytes)
    if (lock.annotation_manifest_sha256 != reference_manifest_digest
            or lock.split_lock_sha256 != reference_manifest_digest):
        raise EvaluationIOError("CORPUS_LOCK_MANIFEST_MISMATCH")
    ref_ids = [row.audio_id for row in references]
    pred_ids = [row.audio_id for row in predictions]
    if len(ref_ids) != len(set(ref_ids)):
        raise EvaluationIOError("REFERENCE_DUPLICATE_ID")
    if len(pred_ids) != len(set(pred_ids)):
        raise EvaluationIOError("PREDICTION_DUPLICATE_ID")
    try:
        validate_recording_session_splits(RecordingManifest(
            recording_id=row.recording_id,
            session_id=row.session_id,
            split=row.split,
            speaker_ids=row.speaker_group_ids,
            source_recording_id=row.source_recording_id,
            augmentation_group=row.augmentation_group,
        ) for row in references)
    except SplitLeakageError:
        raise EvaluationIOError("REFERENCE_SPLIT_LEAKAGE") from None
    if set(ref_ids) != set(pred_ids):
        raise EvaluationIOError("PREDICTION_ID_SET_MISMATCH")
    prediction_by_id = {row.audio_id: row for row in predictions}
    selected = [row for row in references if row.split == split]
    if not selected:
        raise EvaluationIOError("SPLIT_EMPTY", field="split")
    subgroup_labels = {
        ("split", row.split) for row in selected
    } | {
        ("gender_pair", row.gender_pair) for row in selected
    } | {
        ("sample_rate_hz", str(row.sample_rate_hz)) for row in selected
    } | {
        ("condition", condition) for row in selected for condition in row.conditions
    }
    bootstrap_work = (
        len(selected) * bootstrap_iterations * (1 + len(subgroup_labels))
        * _BOOTSTRAP_METRIC_FACTOR
    )
    if bootstrap_work > MAX_BOOTSTRAP_WORK:
        raise EvaluationIOError("BOOTSTRAP_WORK_LIMIT", field="bootstrap_iterations")
    if lock.authority_role == "SILVER" and any(
        row.reference_status not in {"SILVER", "CONVERTED_PROVISIONAL"} for row in selected
    ):
        raise EvaluationIOError("REFERENCE_ROLE_MISMATCH")
    if lock.authority_role == "CHALLENGE" and any(
        row.reference_status != "CHALLENGE" for row in selected
    ):
        raise EvaluationIOError("REFERENCE_ROLE_MISMATCH")
    recordings: list[EvaluationRecording] = []
    manifest_inputs: dict[str, bytes] = {
        "reference_manifest": reference_manifest_bytes,
        "prediction_manifest": prediction_manifest_bytes,
        "corpus_lock": lock_bytes,
    }
    for ordinal, reference in enumerate(selected):
        prediction = prediction_by_id[reference.audio_id]
        try:
            rttm_text = reference.rttm_bytes.decode("utf-8")
            uem_text = reference.uem_bytes.decode("utf-8")
            hypothesis_text = prediction.rttm_bytes.decode("utf-8")
            if not _strict_rttm_shape(rttm_text) or not _strict_rttm_shape(hypothesis_text):
                raise EvaluationError("strict RTTM shape is required")
            reference_rttm = parse_rttm(rttm_text)
            reference_uem = parse_uem(uem_text)
            hypothesis = parse_rttm(hypothesis_text)
        except (UnicodeDecodeError, EvaluationError, ValueError):
            raise EvaluationIOError("ANNOTATION_PARSE_FAILED", record_index=reference.index) from None
        if (any(record.file_id != reference.audio_id for record in (*reference_rttm, *reference_uem))
                or any(record.file_id != prediction.audio_id for record in hypothesis)):
            raise EvaluationIOError("FILE_ID_MISMATCH", record_index=reference.index)
        if any(record.channel != "1" for record in reference_uem):
            raise EvaluationIOError("CHANNEL_MISMATCH", record_index=reference.index)
        if reference.audio_duration_us is None or any(
            record.end_us > reference.audio_duration_us
            for record in (*reference_rttm, *reference_uem, *hypothesis)
        ):
            raise EvaluationIOError("ANNOTATION_OUTSIDE_AUDIO", record_index=reference.index)
        if not _uem_is_sorted_nonoverlapping(reference_uem):
            raise EvaluationIOError("REFERENCE_UEM_SCHEMA", record_index=reference.index)
        if reference.uem_policy == "FULL_AUDIO" and not (
            len(reference_uem) == 1
            and reference_uem[0].start_us == 0
            and reference_uem[0].end_us == reference.audio_duration_us
        ):
            raise EvaluationIOError("REFERENCE_UEM_POLICY_MISMATCH", record_index=reference.index)
        if any(not _inside_reference_uem(record, reference_uem) for record in reference_rttm):
            raise EvaluationIOError("REFERENCE_RTTM_OUTSIDE_UEM", record_index=reference.index)
        reference_speakers = {record.speaker_id for record in reference_rttm}
        if (len(reference_speakers) != reference.speaker_count
                or any(not _opaque_id(label) for label in reference_speakers)):
            raise EvaluationIOError("REFERENCE_SPEAKER_SCHEMA", record_index=reference.index)
        if any(
            not _opaque_id(record.speaker_id)
            and record.speaker_id.upper() not in _RESERVED_HYPOTHESIS_LABELS
            for record in hypothesis
        ):
            raise EvaluationIOError("PREDICTION_SPEAKER_SCHEMA", record_index=reference.index)
        subgroups: list[tuple[str, str]] = [
            ("split", reference.split),
            ("gender_pair", reference.gender_pair),
            ("sample_rate_hz", str(reference.sample_rate_hz)),
        ]
        subgroups.extend(("condition", condition) for condition in reference.conditions)
        overlap_available = lock.reference_capabilities.overlap
        recordings.append(EvaluationRecording(
            recording_id=reference.audio_id,
            reference=reference_rttm,
            hypothesis=hypothesis,
            uem=reference_uem,
            quality_status=prediction.quality_status,
            subgroups=tuple(subgroups),
            overlap_reference_available=overlap_available,
        ))
        manifest_inputs[f"reference-rttm-{ordinal}"] = reference.rttm_bytes
        manifest_inputs[f"reference-uem-{ordinal}"] = reference.uem_bytes
        manifest_inputs[f"prediction-rttm-{ordinal}"] = prediction.rttm_bytes
    try:
        report = score_corpus(recordings, config=config, manifest_inputs=manifest_inputs)
        strict_report = score_corpus(recordings, config=strict_config, manifest_inputs=manifest_inputs)
    except EvaluationError:
        raise EvaluationIOError("SCORING_FAILED") from None
    redacted = _redacted_result(report)
    redacted["der_collar_us"] = 250_000
    strict_redacted = _redacted_result(strict_report)
    redacted["strict_0ms"] = {
        "der_collar_us": 0,
        "overall": strict_redacted["overall"],
        "metric_views": strict_redacted["metric_views"],
        "run_manifest": strict_redacted["run_manifest"],
    }
    eligibility = evaluate_benchmark_eligibility(lock, split=split)
    gate = evaluate_korean_release_gate(
        report, lock=lock, split=split, policy=KoreanReleasePolicy(),
    )
    eligibility_value = eligibility.as_dict()
    gate_value = gate.as_dict()
    gold_rows_approved = all(row.reference_status == "GOLD_APPROVED" for row in selected)
    if lock.authority_role == "GOLD" and not gold_rows_approved:
        if gate_value["status"] == "METRIC_GATES_PASS_REVIEW_REQUIRED":
            gate_value["status"] = "REVIEW_REQUIRED"
        gate_value["reason_codes"] = sorted(set(
            gate_value["reason_codes"] + ["REFERENCE_ROWS_NOT_GOLD_APPROVED"]
        ))
        eligibility_value["status"] = "REVIEW_REQUIRED"
        eligibility_value["eligible_for_metric_gating"] = False
        eligibility_value["eligible_for_release_scoring"] = False
        eligibility_value["reason_codes"] = tuple(sorted(set(
            tuple(eligibility_value["reason_codes"]) + ("REFERENCE_ROWS_NOT_GOLD_APPROVED",)
        )))
    redacted.update({
        "authority_role": lock.authority_role,
        "split": split,
        "corpus_lock_sha256": lock.lock_sha256,
        "reference_capabilities": asdict(lock.reference_capabilities),
        "reference_authority": {
            "gold_approved_rows": sum(row.reference_status == "GOLD_APPROVED" for row in selected),
            "provisional_rows": sum(row.reference_status == "CONVERTED_PROVISIONAL" for row in selected),
            "full_audio_uem_rows": sum(row.uem_policy == "FULL_AUDIO" for row in selected),
            "audited_exclusion_uem_rows": sum(row.uem_policy == "AUDITED_EXCLUSIONS" for row in selected),
            "all_selected_rows_gold_approved": all(
                row.reference_status == "GOLD_APPROVED" for row in selected
            ),
        },
        "eligibility": eligibility_value,
        "release_gate": gate_value,
    })
    return redacted


# Descriptive aliases for callers that use runner terminology.
run_korean_benchmark = evaluate_korean_benchmark
evaluate_prediction_manifest = evaluate_korean_benchmark


__all__ = [
    "EvaluationIOError",
    "RESULT_SCHEMA_VERSION",
    "MAX_ANNOTATION_BYTES",
    "MAX_AUDIO_BYTES",
    "MAX_BOOTSTRAP_ITERATIONS",
    "MAX_BOOTSTRAP_WORK",
    "MAX_MANIFEST_BYTES",
    "evaluate_korean_benchmark",
    "evaluate_prediction_manifest",
    "run_korean_benchmark",
]
