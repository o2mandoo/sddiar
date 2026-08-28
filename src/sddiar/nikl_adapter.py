"""Small, strict adapter for the time annotations in the NIKL corpus.

The corpus contains useful linguistic metadata (including transcripts and the
original speaker identifiers), but neither is needed by a diarization scorer.
This module deliberately projects a document onto the generic ``RTTMRecord``
and ``UEMInterval`` types and only exposes opaque, hash-derived recording
identity in its evidence.

No audio or corpus files are opened here.  Callers must provide the exact
audio duration in source microseconds and the annotation timestamp unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException, InvalidOperation
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from .evaluation import RTTMRecord, UEMInterval
from .annotation_intake import CONDITION_ALLOWLIST


# These are intentionally finite bounds: this adapter is an intake boundary,
# not a general JSON parser.  The limits can be raised by a future version
# after measuring the corpus, without changing the output contract.
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 100_000
MAX_UTTERANCES = 100_000
MAX_STRING_LENGTH = 1_000_000
MAX_AUDIO_DURATION_US = 86_400_000_000
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RELATIVE_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,239}$")


class NIKLAdapterError(ValueError):
    """Base error for malformed or unsafe NIKL annotation input."""


class NIKLSchemaError(NIKLAdapterError):
    """The input does not match the supported NIKL document shape."""


class NIKLTimeError(NIKLAdapterError):
    """A duration or annotation timestamp is invalid or out of bounds."""


@dataclass(frozen=True)
class ReferenceEvidence:
    """Content-free evidence emitted alongside generic reference records."""

    recording_id: str
    canonical_payload_sha256: str
    duration_us: int
    speaker_count: int
    utterance_count: int
    uem_start_us: int
    uem_end_us: int
    overlap_duration_us: int
    time_unit: str
    uem_policy: str = "ANNOTATED_EXTENT_PROVISIONAL"
    quality_status: str = "REVIEW_REQUIRED"
    release_authority: str = "none"
    schema_version: str = "sddiar-nikl-reference-v1"

    def __post_init__(self) -> None:
        expected_recording_id = "recording-1-" + self.canonical_payload_sha256[:22]
        if self.recording_id != expected_recording_id:
            raise NIKLAdapterError("recording identity must be opaque")
        if len(self.canonical_payload_sha256) != 64:
            raise NIKLAdapterError("canonical payload SHA-256 must be hexadecimal")
        try:
            int(self.canonical_payload_sha256, 16)
        except ValueError as exc:
            raise NIKLAdapterError("source JSON SHA-256 must be hexadecimal") from exc
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.duration_us,
                self.speaker_count,
                self.utterance_count,
                self.uem_start_us,
                self.uem_end_us,
                self.overlap_duration_us,
            )
        ) or self.duration_us <= 0 or self.speaker_count not in (1, 2):
            raise NIKLAdapterError("invalid reference evidence counts or duration")
        if self.utterance_count <= 0 or self.uem_end_us <= self.uem_start_us:
            raise NIKLAdapterError("invalid UEM extent in reference evidence")
        if self.uem_end_us > self.duration_us:
            raise NIKLAdapterError("UEM extent exceeds audio duration")
        if self.time_unit not in {"milliseconds", "seconds"}:
            raise NIKLAdapterError("invalid evidence time unit")
        if (self.uem_policy != "ANNOTATED_EXTENT_PROVISIONAL"
                or self.quality_status != "REVIEW_REQUIRED" or self.release_authority != "none"):
            raise NIKLAdapterError("NIKL conversion cannot claim release authority")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy containing no source text or speaker IDs."""
        return {
            "schema_version": self.schema_version,
            "recording_id": self.recording_id,
            "canonical_payload_sha256": self.canonical_payload_sha256,
            "duration_us": self.duration_us,
            "speaker_count": self.speaker_count,
            "utterance_count": self.utterance_count,
            "uem_start_us": self.uem_start_us,
            "uem_end_us": self.uem_end_us,
            "overlap_duration_us": self.overlap_duration_us,
            "time_unit": self.time_unit,
            "uem_policy": self.uem_policy,
            "quality_status": self.quality_status,
            "release_authority": self.release_authority,
        }

    @property
    def evidence_sha256(self) -> str:
        payload = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True)
class NIKLReference:
    """Immutable generic reference records, UEM, and redacted evidence."""

    recording_id: str
    records: tuple[RTTMRecord, ...]
    uem: tuple[UEMInterval, ...]
    evidence: ReferenceEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple) or not isinstance(self.uem, tuple):
            raise NIKLAdapterError("reference records and UEM must be immutable tuples")
        if self.evidence.recording_id != self.recording_id:
            raise NIKLAdapterError("reference/evidence recording identity mismatch")
        if any(record.file_id != self.recording_id for record in self.records):
            raise NIKLAdapterError("reference record file identity mismatch")
        if any(interval.file_id != self.recording_id for interval in self.uem):
            raise NIKLAdapterError("UEM file identity mismatch")

    @property
    def reference_records(self) -> tuple[RTTMRecord, ...]:
        """Descriptive alias used by evaluation callers."""
        return self.records

    @property
    def uem_intervals(self) -> tuple[UEMInterval, ...]:
        return self.uem

    def public_evidence(self) -> dict[str, Any]:
        """Return a JSON-safe, content-free evidence mapping.

        The returned copy is intentionally mutable for normal JSON encoding;
        the canonical evidence held by this frozen result remains immutable.
        """
        return self.evidence.as_dict()


def _check_json_bounds(value: Any, *, depth: int = 0, state: list[int] | None = None) -> None:
    if state is None:
        state = [0]
    if depth > MAX_JSON_DEPTH:
        raise NIKLSchemaError("JSON exceeds maximum nesting depth")
    state[0] += 1
    if state[0] > MAX_JSON_NODES:
        raise NIKLSchemaError("JSON exceeds maximum node count")
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise NIKLSchemaError("JSON string exceeds maximum length")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise NIKLSchemaError("JSON object keys must be strings")
            _check_json_bounds(key, depth=depth + 1, state=state)
            _check_json_bounds(child, depth=depth + 1, state=state)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_json_bounds(child, depth=depth + 1, state=state)


def _as_decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise NIKLTimeError(f"{field} must be a finite decimal")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise NIKLTimeError(f"{field} must be a finite decimal") from exc
    if not number.is_finite():
        raise NIKLTimeError(f"{field} must be a finite decimal")
    return number


def _to_us(value: Any, *, time_unit: str, field: str) -> int:
    number = _as_decimal(value, field)
    if number < 0:
        raise NIKLTimeError(f"{field} must not be negative")
    multiplier = {"milliseconds": Decimal(1_000), "seconds": Decimal(1_000_000)}.get(time_unit)
    if multiplier is None:
        raise NIKLTimeError("time_unit must be exactly 'milliseconds' or 'seconds'")
    try:
        scaled = number * multiplier
        if not scaled.is_finite() or scaled > MAX_AUDIO_DURATION_US:
            raise NIKLTimeError(f"{field} exceeds the supported duration")
        integral = scaled.to_integral_value()
        if scaled != integral:
            raise NIKLTimeError(f"{field} does not have microsecond precision")
        return int(integral)
    except (DecimalException, OverflowError, ValueError) as exc:
        raise NIKLTimeError(f"{field} is outside the supported range") from exc


def _document(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(payload) not in ({"document"}, {"id", "metadata", "document"}):
        raise NIKLSchemaError("top level does not match the supported NIKL profile")
    if "id" in payload and (not isinstance(payload["id"], str) or not payload["id"]):
        raise NIKLSchemaError("top-level id must be a non-empty string")
    if "metadata" in payload and not isinstance(payload["metadata"], Mapping):
        raise NIKLSchemaError("top-level metadata must be an object")
    documents = payload["document"]
    if not isinstance(documents, list) or len(documents) != 1:
        raise NIKLSchemaError("document must be a list containing exactly one document")
    document = documents[0]
    if not isinstance(document, Mapping):
        raise NIKLSchemaError("document entry must be an object")
    if not {"metadata", "utterance"}.issubset(document) or set(document) - {"id", "metadata", "utterance"}:
        raise NIKLSchemaError("document does not match the supported NIKL profile")
    return document


def _speaker_ids(document: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping) or "speaker" not in metadata:
        raise NIKLSchemaError("metadata.speaker list is required")
    speakers = metadata["speaker"]
    if not isinstance(speakers, list) or not (1 <= len(speakers) <= 2):
        raise NIKLSchemaError("metadata.speaker must contain one or two speakers")
    result: list[str] = []
    for speaker in speakers:
        # Official NIKL uses objects with an ``id`` field.  Accepting a plain
        # string is useful for small exports that preserve the same contract.
        if isinstance(speaker, str):
            identifier = speaker
        elif isinstance(speaker, Mapping) and set(speaker) >= {"id"}:
            identifier = speaker["id"]
        else:
            raise NIKLSchemaError("each metadata speaker must have an id")
        if not isinstance(identifier, str) or not identifier:
            raise NIKLSchemaError("speaker ids must be non-empty strings")
        if identifier in result:
            raise NIKLSchemaError("metadata speaker ids must be unique")
        result.append(identifier)
    return tuple(result)


def parse_nikl_reference(
    payload: Mapping[str, Any],
    audio_duration_us: int,
    *,
    time_unit: str | None,
) -> NIKLReference:
    """Convert one NIKL document to immutable generic evaluation records.

    ``audio_duration_us`` is always an exact positive integer microsecond
    duration.  Only annotation ``start``/``end`` values use ``time_unit``.
    """
    if not isinstance(payload, Mapping):
        raise NIKLSchemaError("payload must be a JSON object")
    if time_unit not in ("milliseconds", "seconds"):
        raise NIKLTimeError("time_unit must be exactly 'milliseconds' or 'seconds'")
    _check_json_bounds(payload)
    if (isinstance(audio_duration_us, bool) or not isinstance(audio_duration_us, int)
            or not 0 < audio_duration_us <= MAX_AUDIO_DURATION_US):
        raise NIKLTimeError("audio_duration_us must be a positive integer")
    duration_us = audio_duration_us

    document = _document(payload)
    source_speakers = _speaker_ids(document)
    speaker_map = {source_id: f"REF_{index:02d}" for index, source_id in enumerate(source_speakers)}
    utterances = document.get("utterance")
    if not isinstance(utterances, list) or not utterances:
        raise NIKLSchemaError("utterance must be a non-empty list")
    if len(utterances) > MAX_UTTERANCES:
        raise NIKLSchemaError("utterance count exceeds maximum")

    # Hash the source object canonically.  The digest is the only source
    # identity retained in the public result; text and original IDs do not
    # cross this adapter boundary.
    try:
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NIKLSchemaError("payload must be JSON serializable") from exc
    payload_digest = sha256(canonical).hexdigest()
    recording_id = "recording-1-" + payload_digest[:22]

    records: list[RTTMRecord] = []
    for index, utterance in enumerate(utterances):
        if not isinstance(utterance, Mapping):
            raise NIKLSchemaError(f"utterance {index} must be an object")
        if not all(key in utterance for key in ("speaker_id", "start", "end")):
            raise NIKLSchemaError(f"utterance {index} requires speaker_id, start and end")
        if set(utterance) - {
            "id", "form", "original_form", "speaker_id", "start", "end", "note", "file_id"
        }:
            raise NIKLSchemaError(f"utterance {index} contains an unsupported field")
        source_id = utterance["speaker_id"]
        if not isinstance(source_id, str) or source_id not in speaker_map:
            raise NIKLSchemaError(f"utterance {index} references an unknown speaker")
        start_us = _to_us(utterance["start"], time_unit=time_unit, field=f"utterance {index} start")
        end_us = _to_us(utterance["end"], time_unit=time_unit, field=f"utterance {index} end")
        if end_us <= start_us:
            raise NIKLTimeError(f"utterance {index} end must be after start")
        if end_us > duration_us:
            raise NIKLTimeError(f"utterance {index} is outside audio duration")
        records.append(RTTMRecord(recording_id, speaker_map[source_id], start_us, end_us))

    # Merge only same-speaker adjacency/overlap.  Cross-speaker overlap remains
    # as two simultaneous RTTM records and is therefore visible to DER/OSD.
    merged: list[RTTMRecord] = []
    for speaker_id in sorted({record.speaker_id for record in records}):
        current: list[RTTMRecord] = []
        for record in sorted((item for item in records if item.speaker_id == speaker_id),
                             key=lambda item: (item.start_us, item.end_us)):
            if current and record.start_us <= current[-1].end_us:
                previous = current[-1]
                current[-1] = RTTMRecord(recording_id, speaker_id, previous.start_us,
                                         max(previous.end_us, record.end_us))
            else:
                current.append(record)
        merged.extend(current)
    records = sorted(merged, key=lambda item: (item.start_us, item.end_us, item.speaker_id))
    uem_start_us = min(record.start_us for record in records)
    uem_end_us = max(record.end_us for record in records)
    boundaries = sorted({point for record in records for point in (record.start_us, record.end_us)})
    overlap_duration_us = sum(
        right - left for left, right in zip(boundaries, boundaries[1:])
        if len({record.speaker_id for record in records
                if record.start_us < right and record.end_us > left}) >= 2
    )
    uem = (UEMInterval(recording_id, uem_start_us, uem_end_us),)
    evidence = ReferenceEvidence(
        recording_id=recording_id,
        canonical_payload_sha256=payload_digest,
        duration_us=duration_us,
        speaker_count=len(source_speakers),
        utterance_count=len(records),
        uem_start_us=uem_start_us,
        uem_end_us=uem_end_us,
        overlap_duration_us=overlap_duration_us,
        time_unit=time_unit,
    )
    return NIKLReference(recording_id, tuple(records), uem, evidence)


# Descriptive aliases keep callers independent of the adapter's historical
# name while retaining one implementation and one validation policy.
adapt_nikl_reference = parse_nikl_reference
build_nikl_reference = parse_nikl_reference


def _seconds_text(value_us: int) -> str:
    return f"{value_us // 1_000_000}.{value_us % 1_000_000:06d}"


def format_nikl_rttm(reference: NIKLReference) -> str:
    if not isinstance(reference, NIKLReference):
        raise NIKLAdapterError("reference must be NIKLReference")
    return "".join(
        f"SPEAKER {record.file_id} 1 {_seconds_text(record.start_us)} "
        f"{_seconds_text(record.end_us - record.start_us)} <NA> <NA> "
        f"{record.speaker_id} <NA> <NA>\n"
        for record in reference.records
    )


def format_nikl_uem(reference: NIKLReference) -> str:
    if not isinstance(reference, NIKLReference):
        raise NIKLAdapterError("reference must be NIKLReference")
    return "".join(
        f"{interval.file_id} 1 {_seconds_text(interval.start_us)} {_seconds_text(interval.end_us)}\n"
        for interval in reference.uem
    )


def build_nikl_manifest_row(
    reference: NIKLReference,
    *,
    audio_sha256: str,
    sample_rate_hz: int,
    split: str,
    audio_path: str,
    rttm_path: str,
    uem_path: str,
    speaker_group_ids: tuple[str, ...],
    gender_pair: str = "UNKNOWN",
    conditions: tuple[str, ...] = ("regional-interview",),
) -> dict[str, Any]:
    """Build one normalized annotation-intake row without source identities."""
    if (not isinstance(reference, NIKLReference) or not isinstance(audio_sha256, str)
            or _HEX64.fullmatch(audio_sha256) is None):
        raise NIKLAdapterError("invalid normalized manifest identity")
    if type(sample_rate_hz) is not int or sample_rate_hz not in {8000, 16000}:
        raise NIKLAdapterError("normalized manifest sample rate must be 8 or 16 kHz")
    if split not in {"CALIBRATION", "DEVELOPMENT_HOLDOUT", "RELEASE_HOLDOUT"}:
        raise NIKLAdapterError("invalid normalized manifest split")
    if not isinstance(gender_pair, str) or gender_pair not in {"MM", "FF", "MF", "M", "F", "UNKNOWN"}:
        raise NIKLAdapterError("invalid normalized manifest gender pair")
    for value in (audio_path, rttm_path, uem_path):
        if (not isinstance(value, str) or _RELATIVE_PATH.fullmatch(value) is None
                or value.startswith("/") or "\\" in value
                or any(part in {"", ".", ".."} for part in value.split("/"))):
            raise NIKLAdapterError("normalized manifest paths must be canonical relative paths")
    if (not isinstance(conditions, tuple) or not conditions
            or any(not isinstance(item, str) or item not in CONDITION_ALLOWLIST for item in conditions)):
        raise NIKLAdapterError("invalid normalized manifest conditions")
    if (not isinstance(speaker_group_ids, tuple)
            or len(speaker_group_ids) != reference.evidence.speaker_count
            or any(not isinstance(item, str)
                   or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", item)
                   or not any(character.isdigit() for character in item)
                   for item in speaker_group_ids)
            or len(speaker_group_ids) != len(set(speaker_group_ids))):
        raise NIKLAdapterError("invalid normalized speaker group identifiers")
    rttm = format_nikl_rttm(reference).encode("utf-8")
    uem = format_nikl_uem(reference).encode("utf-8")
    return {
        "audio_id": reference.recording_id,
        "recording_id": reference.recording_id,
        "source_recording_id": reference.recording_id,
        "audio_sha256": audio_sha256,
        "session_id": "session-1-" + reference.evidence.canonical_payload_sha256[:22],
        "speaker_group_ids": list(sorted(speaker_group_ids)),
        "reference_status": "CONVERTED_PROVISIONAL",
        "uem_policy": reference.evidence.uem_policy,
        "conversion_evidence_sha256": reference.evidence.evidence_sha256,
        "split": split,
        "sample_rate_hz": sample_rate_hz,
        "speaker_count": reference.evidence.speaker_count,
        "gender_pair": gender_pair,
        "conditions": list(conditions),
        "audio": audio_path,
        "rttm": rttm_path,
        "rttm_sha256": sha256(rttm).hexdigest(),
        "uem": uem_path,
        "uem_sha256": sha256(uem).hexdigest(),
    }


__all__ = [
    "MAX_JSON_DEPTH",
    "MAX_JSON_NODES",
    "MAX_STRING_LENGTH",
    "MAX_UTTERANCES",
    "MAX_AUDIO_DURATION_US",
    "NIKLAdapterError",
    "NIKLSchemaError",
    "NIKLTimeError",
    "ReferenceEvidence",
    "NIKLReference",
    "parse_nikl_reference",
    "adapt_nikl_reference",
    "build_nikl_reference",
    "format_nikl_rttm",
    "format_nikl_uem",
    "build_nikl_manifest_row",
]
