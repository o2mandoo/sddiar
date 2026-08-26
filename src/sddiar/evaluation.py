"""Deterministic, content-free evaluation primitives for the SDDiar P0 scorer.

The evaluator intentionally accepts annotations and already-produced decisions;
it does not load audio, models, or service SDKs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from hashlib import sha256
from itertools import permutations
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCORER_SCHEMA_VERSION = "sddiar-offline-evaluation-v1"
_REFERENCE_NON_SPEAKERS = frozenset({
    "REF_OTHER", "UNKNOWN", "OVERLAP", "OTHER", "NON_SPEECH", "SIL", "SILENCE",
})
_HYPOTHESIS_NON_SPEAKERS = frozenset({
    "REF_OTHER", "UNKNOWN", "UNKNOWN_SHORT", "OVERLAP", "OTHER", "NON_SPEECH", "SIL", "SILENCE",
})
_PASS_SPEAKER_AWARE = frozenset({"PASS_HIGH", "PASS_STANDARD"})
_ALL_PASS = frozenset({"PASS_HIGH", "PASS_STANDARD", "PASS_WITH_UNATTRIBUTED"})


class EvaluationError(ValueError):
    pass


class SplitLeakageError(EvaluationError):
    pass


class CalibrationGuardError(EvaluationError):
    pass


@dataclass(frozen=True)
class RecordingManifest:
    recording_id: str
    session_id: str
    split: str
    speaker_ids: tuple[str, ...] = ()
    source_recording_id: str | None = None
    augmentation_group: str | None = None


def validate_recording_session_splits(records: Iterable[RecordingManifest]) -> None:
    """Reject recording/session/edit/augmentation groups crossing a split."""
    groups: dict[str, set[str]] = defaultdict(set)
    seen: set[str] = set()
    for record in records:
        if record.recording_id in seen:
            raise SplitLeakageError(f"duplicate recording_id: {record.recording_id}")
        seen.add(record.recording_id)
        if not record.recording_id or not record.session_id or not record.split:
            raise SplitLeakageError("recording_id, session_id and split are required")
        keys = ("session:" + record.session_id,
                "recording:" + record.recording_id,
                "source:" + (record.source_recording_id or record.recording_id))
        if record.augmentation_group:
            keys += ("augmentation:" + record.augmentation_group,)
        for key in keys:
            groups[key].add(record.split)
    leaked = sorted(key for key, splits in groups.items() if len(splits) > 1)
    if leaked:
        raise SplitLeakageError("split leakage: " + ", ".join(leaked))


@dataclass(frozen=True)
class UEMInterval:
    file_id: str
    start_us: int
    end_us: int
    channel: str = "1"

    def __post_init__(self) -> None:
        if (not self.file_id or not self.channel or isinstance(self.start_us, bool)
                or isinstance(self.end_us, bool) or not isinstance(self.start_us, int)
                or not isinstance(self.end_us, int) or self.start_us < 0 or self.end_us <= self.start_us):
            raise EvaluationError("invalid UEM interval")


@dataclass(frozen=True)
class RTTMRecord:
    file_id: str
    speaker_id: str
    start_us: int
    end_us: int
    channel: str = "1"

    def __post_init__(self) -> None:
        if (not self.file_id or not self.speaker_id or not self.channel or isinstance(self.start_us, bool)
                or isinstance(self.end_us, bool) or not isinstance(self.start_us, int)
                or not isinstance(self.end_us, int) or self.start_us < 0 or self.end_us <= self.start_us):
            raise EvaluationError("invalid RTTM interval")


@dataclass(frozen=True)
class WordAnnotation:
    word_id: str
    start_us: int
    end_us: int
    text: str
    ref_speaker_id: str | None
    attributable: bool = True
    overlap_flag: bool = False
    boundary_crossing_flag: bool = False

    def __post_init__(self) -> None:
        if (not self.word_id or isinstance(self.start_us, bool) or isinstance(self.end_us, bool)
                or not isinstance(self.start_us, int) or not isinstance(self.end_us, int)
                or self.start_us < 0 or self.end_us <= self.start_us or not isinstance(self.text, str)
                or (self.ref_speaker_id is not None and not isinstance(self.ref_speaker_id, str))
                or any(not isinstance(value, bool) for value in (
                    self.attributable, self.overlap_flag, self.boundary_crossing_flag))):
            raise EvaluationError("invalid word annotation")


@dataclass(frozen=True)
class WordDecision:
    word_id: str
    speaker_id: str | None

    def __post_init__(self) -> None:
        if not self.word_id or (self.speaker_id is not None and not isinstance(self.speaker_id, str)):
            raise EvaluationError("invalid word decision")


@dataclass(frozen=True)
class MicroDecision:
    decision_id: str
    reference_speaker_id: str | None
    predicted_speaker_id: str | None
    eligible: bool = True

    def __post_init__(self) -> None:
        if (not self.decision_id or not isinstance(self.eligible, bool)
                or (self.reference_speaker_id is not None and not isinstance(self.reference_speaker_id, str))
                or (self.predicted_speaker_id is not None and not isinstance(self.predicted_speaker_id, str))):
            raise EvaluationError("invalid MICRO decision")


@dataclass(frozen=True)
class ScoringConfig:
    """Frozen policy for exact interval scoring.

    Duration floors are deliberately explicit.  A topology decision must be
    supported by material speech; merely emitting a second label for one tick
    cannot turn a complete merge into H2.
    """

    reference_speaker_duration_floor_us: int = 250_000
    hypothesis_speaker_duration_floor_us: int = 250_000
    der_collar_us: int = 0
    scd_collar_us: int = 500_000
    bootstrap_iterations: int = 2_000
    bootstrap_seed: int = 17_029
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        integers = (
            self.reference_speaker_duration_floor_us,
            self.hypothesis_speaker_duration_floor_us,
            self.der_collar_us,
            self.scd_collar_us,
            self.bootstrap_iterations,
            self.bootstrap_seed,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integers):
            raise EvaluationError("scoring durations and iterations must be non-negative integers")
        if self.reference_speaker_duration_floor_us <= 0 or self.hypothesis_speaker_duration_floor_us <= 0:
            raise EvaluationError("speaker duration floors must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise EvaluationError("confidence_level must be between zero and one")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class SpeakerMapping:
    hypothesis_speaker_id: str
    reference_speaker_id: str
    cooccurrence_us: int


@dataclass(frozen=True)
class ErrorComponents:
    """DER accounting in speaker-microseconds (except scored UEM time)."""

    scored_uem_us: int
    reference_speaker_us: int
    miss_us: int
    false_alarm_us: int
    confusion_us: int
    der: float


@dataclass(frozen=True)
class SpeakerScore:
    reference_speaker_id: str
    reference_duration_us: int
    assigned_duration_us: int
    correct_duration_us: int
    coverage: float
    assigned_accuracy: float


@dataclass(frozen=True)
class EventMetrics:
    evaluated: bool
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class OverlapMetrics:
    evaluated: bool
    reference_overlap_us: int
    predicted_overlap_us: int
    true_positive_us: int
    false_positive_us: int
    false_negative_us: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class WordMetrics:
    eligible_words: int
    assigned_words: int
    correct_words: int
    precision: float
    coverage: float
    strict_error: float
    unsafe_error: float
    eligible_characters: int
    character_weighted_error: float
    excluded_overlap_or_boundary_words: int
    forced_overlap_or_boundary_assignments: int


@dataclass(frozen=True)
class MicroMetrics:
    eligible_micros: int
    assigned_micros: int
    correct_micros: int
    precision: float
    coverage: float
    unknown_short_rate: float


@dataclass(frozen=True)
class RecordingScore:
    recording_id: str
    mapping: tuple[SpeakerMapping, ...]
    diarization_all: ErrorComponents
    diarization_nonoverlap: ErrorComponents
    jer: float
    reference_speaker_count: int
    hypothesis_speaker_count: int
    eligible_reference_h1: bool
    eligible_reference_h2: bool
    acoustic_complete_merge: bool
    unsafe_complete_merge: bool
    false_h2: bool
    false_h2_secondary_duration_us: int
    false_h2_duration_ratio: float
    speaker_count_correct: bool
    speakers: tuple[SpeakerScore, ...]
    worst_speaker_coverage: float | None
    worst_speaker_assigned_accuracy: float | None
    scd: EventMetrics
    overlap: OverlapMetrics
    words: WordMetrics | None
    micros: MicroMetrics | None
    quality_status: str


@dataclass(frozen=True)
class BootstrapCI:
    metric: str
    confidence_level: float
    point: float
    lower: float
    upper: float
    iterations: int
    seed: int


@dataclass(frozen=True)
class AggregateScore:
    recording_count: int
    diarization_all: ErrorComponents
    diarization_nonoverlap: ErrorComponents
    jer: float
    eligible_h2_files: int
    complete_merge_files: int
    acoustic_complete_merge_rate: float
    unsafe_complete_merge_rate: float
    eligible_h1_files: int
    false_h2_files: int
    false_h2_rate: float
    false_h2_secondary_duration_us: int
    false_h2_duration_ratio: float
    speaker_count_accuracy: float
    worst_speaker_coverage: float | None
    worst_speaker_assigned_accuracy: float | None
    scd: EventMetrics
    overlap: OverlapMetrics
    words: WordMetrics | None
    micros: MicroMetrics | None


@dataclass(frozen=True)
class SubgroupScore:
    subgroup: str
    aggregate: AggregateScore
    bootstrap: tuple[BootstrapCI, ...] = ()


@dataclass(frozen=True)
class EvaluationRunManifest:
    """Content-addressed, redacted evidence for one scorer invocation."""

    schema_version: str
    input_count: int
    input_hashes: tuple[str, ...]
    input_set_sha256: str
    config_sha256: str
    scorer_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_hashes", tuple(self.input_hashes))
        digests = (*self.input_hashes, self.input_set_sha256, self.config_sha256,
                   self.scorer_sha256, self.manifest_sha256)
        if self.schema_version != SCORER_SCHEMA_VERSION or self.input_count != len(self.input_hashes):
            raise EvaluationError("invalid evaluation run manifest")
        if tuple(sorted(self.input_hashes)) != self.input_hashes or any(
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            raise EvaluationError("invalid evaluation run manifest digest")
        if sha256(_canonical_json_bytes(self.input_hashes)).hexdigest() != self.input_set_sha256:
            raise EvaluationError("evaluation input-set digest mismatch")
        unsigned = {
            "schema_version": self.schema_version,
            "input_count": self.input_count,
            "input_hashes": self.input_hashes,
            "input_set_sha256": self.input_set_sha256,
            "config_sha256": self.config_sha256,
            "scorer_sha256": self.scorer_sha256,
        }
        if sha256(_canonical_json_bytes(unsigned)).hexdigest() != self.manifest_sha256:
            raise EvaluationError("evaluation manifest digest mismatch")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_count": self.input_count,
            "input_hashes": list(self.input_hashes),
            "input_set_sha256": self.input_set_sha256,
            "config_sha256": self.config_sha256,
            "scorer_sha256": self.scorer_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class EvaluationReport:
    recordings: tuple[RecordingScore, ...]
    overall: AggregateScore
    subgroups: tuple[SubgroupScore, ...]
    bootstrap: tuple[BootstrapCI, ...]
    run_manifest: EvaluationRunManifest

    def as_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class EvaluationRecording:
    recording_id: str
    reference: tuple[RTTMRecord, ...]
    hypothesis: tuple[RTTMRecord, ...]
    uem: tuple[UEMInterval, ...]
    quality_status: str = ""
    subgroups: tuple[tuple[str, str], ...] = ()
    reference_scd_us: tuple[int, ...] | None = None
    predicted_scd_us: tuple[int, ...] | None = None
    overlap_reference_available: bool | None = None
    words: tuple[WordAnnotation, ...] | None = None
    word_decisions: tuple[WordDecision, ...] | None = None
    micro_decisions: tuple[MicroDecision, ...] | None = None

    def __post_init__(self) -> None:
        if not self.recording_id or not isinstance(self.quality_status, str):
            raise EvaluationError("recording_id is required")
        if self.overlap_reference_available is not None and not isinstance(self.overlap_reference_available, bool):
            raise EvaluationError("overlap_reference_available must be bool or None")
        object.__setattr__(self, "reference", tuple(self.reference))
        object.__setattr__(self, "hypothesis", tuple(self.hypothesis))
        object.__setattr__(self, "uem", tuple(self.uem))
        if isinstance(self.subgroups, Mapping):
            normalized = tuple(sorted((str(key), str(value)) for key, value in self.subgroups.items()))
        else:
            normalized = tuple(sorted({(str(key), str(value)) for key, value in self.subgroups}))
        object.__setattr__(self, "subgroups", normalized)
        for name in ("reference_scd_us", "predicted_scd_us", "words", "word_decisions", "micro_decisions"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))


@dataclass(frozen=True)
class SafetyMetrics:
    reference_files: int
    eligible_two_speaker_files: int
    acoustic_complete_merge_rate: float
    unsafe_complete_merge_rate: float
    word_attribution_precision: float
    word_attribution_coverage: float
    micro_precision: float
    micro_coverage: float
    quality_false_pass_rate: float
    eligible_one_speaker_files: int = 0
    false_h2_rate: float = 0.0
    speaker_count_accuracy: float = 0.0


def _parsed_time_us(raw: str, *, unit: str) -> int:
    if unit not in {"seconds", "microseconds", "us"}:
        raise EvaluationError("unit must be seconds or microseconds")
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError) as exc:
        raise EvaluationError("invalid annotation time") from exc
    if not value.is_finite():
        raise EvaluationError("invalid annotation time")
    scale = Decimal(1_000_000) if unit == "seconds" else Decimal(1)
    return int((value * scale).to_integral_value(rounding=ROUND_HALF_EVEN))


def parse_rttm(text: str, *, unit: str = "seconds") -> tuple[RTTMRecord, ...]:
    """Parse SPEAKER RTTM; IDs are kept opaque and times become integer us."""
    out: list[RTTMRecord] = []
    if unit not in {"seconds", "microseconds", "us"}:
        raise EvaluationError("unit must be seconds or microseconds")
    for line_no, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        # Evaluation accepts the common compact RTTM form (8+ fields); the
        # annotation intake gate separately enforces the repository's strict
        # 10-field archival schema.
        if len(fields) < 8 or fields[0].upper() != "SPEAKER":
            raise EvaluationError(f"invalid RTTM line {line_no}")
        try:
            start = _parsed_time_us(fields[3], unit=unit)
            end = start + _parsed_time_us(fields[4], unit=unit)
        except EvaluationError as exc:
            raise EvaluationError(f"invalid RTTM time line {line_no}") from exc
        out.append(RTTMRecord(fields[1], fields[7], start, end, fields[2]))
    return tuple(out)


def parse_uem(text: str, *, unit: str = "seconds") -> tuple[UEMInterval, ...]:
    out: list[UEMInterval] = []
    if unit not in {"seconds", "microseconds", "us"}:
        raise EvaluationError("unit must be seconds or microseconds")
    for line_no, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 4:
            raise EvaluationError(f"invalid UEM line {line_no}")
        try:
            start = _parsed_time_us(fields[2], unit=unit)
            end = _parsed_time_us(fields[3], unit=unit)
        except EvaluationError as exc:
            raise EvaluationError(f"invalid UEM time line {line_no}") from exc
        out.append(UEMInterval(fields[0], start, end, fields[1]))
    return tuple(out)


def parse_words_jsonl(text: str) -> tuple[WordAnnotation, ...]:
    out = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("word row")
            required = {"word_id", "start_us", "end_us"}
            allowed = required | {"text", "ref_speaker_id", "attributable", "overlap_flag", "boundary_crossing_flag"}
            if required - set(value) or set(value) - allowed:
                raise ValueError("word schema")
            flags = tuple(value.get(name, default) for name, default in (
                ("attributable", True), ("overlap_flag", False), ("boundary_crossing_flag", False)))
            if not isinstance(value["word_id"], str) or not isinstance(value.get("text", ""), str) \
                    or any(not isinstance(flag, bool) for flag in flags):
                raise TypeError("word schema")
            out.append(WordAnnotation(word_id=value["word_id"], start_us=value["start_us"],
                end_us=value["end_us"], text=value.get("text", ""),
                ref_speaker_id=value.get("ref_speaker_id"), attributable=flags[0],
                overlap_flag=flags[1], boundary_crossing_flag=flags[2]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"invalid words JSONL line {line_no}") from exc
    return tuple(out)


@dataclass(frozen=True)
class _Cell:
    start_us: int
    end_us: int
    reference_speakers: frozenset[str]
    hypothesis_speakers: frozenset[str]
    reference_overlap_label: bool
    hypothesis_overlap_label: bool

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us


def _is_reference_speaker(label: str) -> bool:
    return label.upper() not in _REFERENCE_NON_SPEAKERS


def _is_hypothesis_speaker(label: str) -> bool:
    return label.upper() not in _HYPOTHESIS_NON_SPEAKERS


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if start < 0 or end <= start:
            raise EvaluationError("invalid scoring interval")
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def _subtract_intervals(base: Sequence[tuple[int, int]], excluded: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cuts = _merge_intervals(excluded)
    out: list[tuple[int, int]] = []
    for start, end in base:
        cursor = start
        for left, right in cuts:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                out.append((cursor, min(left, end)))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return tuple(out)


def _uem_by_channel(uem: Sequence[UEMInterval]) -> dict[str, tuple[tuple[int, int], ...]]:
    if not uem:
        raise EvaluationError("at least one UEM interval is required")
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for interval in uem:
        grouped[interval.channel].append((interval.start_us, interval.end_us))
    return {channel: _merge_intervals(intervals) for channel, intervals in sorted(grouped.items())}


def _apply_collar(uem: Mapping[str, Sequence[tuple[int, int]]], boundaries: Sequence[int], collar_us: int) -> dict[str, tuple[tuple[int, int], ...]]:
    if collar_us <= 0 or not boundaries:
        return {channel: tuple(intervals) for channel, intervals in uem.items()}
    exclusions = tuple((max(0, point - collar_us), point + collar_us) for point in sorted(set(boundaries)))
    return {channel: _subtract_intervals(intervals, exclusions) for channel, intervals in uem.items()}


def _validate_recording_ids(recording_id: str, reference: Sequence[RTTMRecord], hypothesis: Sequence[RTTMRecord], uem: Sequence[UEMInterval]) -> None:
    ids = {item.file_id for item in (*reference, *hypothesis, *uem)}
    if ids != {recording_id}:
        raise EvaluationError("all RTTM/UEM file IDs must equal recording_id")
    channels = {item.channel for item in uem}
    if len(channels) != 1:
        raise EvaluationError("V1 scorer requires one UEM channel")
    if any(item.channel not in channels for item in (*reference, *hypothesis)):
        raise EvaluationError("RTTM channel has no UEM")


def _timeline_cells(reference: Sequence[RTTMRecord], hypothesis: Sequence[RTTMRecord],
                    scoring_intervals: Mapping[str, Sequence[tuple[int, int]]]) -> tuple[_Cell, ...]:
    cells: list[_Cell] = []
    for channel, intervals in sorted(scoring_intervals.items()):
        refs = tuple(item for item in reference if item.channel == channel)
        hyps = tuple(item for item in hypothesis if item.channel == channel)
        for score_start, score_end in intervals:
            boundaries = {score_start, score_end}
            for item in (*refs, *hyps):
                left, right = max(score_start, item.start_us), min(score_end, item.end_us)
                if left < right:
                    boundaries.update((left, right))
            ordered = sorted(boundaries)
            for left, right in zip(ordered, ordered[1:]):
                if left >= right:
                    continue
                active_refs = {item.speaker_id for item in refs if item.start_us < right and item.end_us > left}
                active_hyps = {item.speaker_id for item in hyps if item.start_us < right and item.end_us > left}
                cells.append(_Cell(
                    left,
                    right,
                    frozenset(label for label in active_refs if _is_reference_speaker(label)),
                    frozenset(label for label in active_hyps if _is_hypothesis_speaker(label)),
                    any(label.upper() == "OVERLAP" for label in active_refs),
                    any(label.upper() == "OVERLAP" for label in active_hyps),
                ))
    return tuple(cells)


def _speaker_durations(cells: Sequence[_Cell], *, reference: bool) -> dict[str, int]:
    durations: dict[str, int] = defaultdict(int)
    for cell in cells:
        speakers = cell.reference_speakers if reference else cell.hypothesis_speakers
        for speaker in speakers:
            durations[speaker] += cell.duration_us
    return dict(durations)


def optimal_speaker_mapping(reference: Sequence[RTTMRecord], hypothesis: Sequence[RTTMRecord],
                            uem: Sequence[UEMInterval]) -> tuple[SpeakerMapping, ...]:
    """Return the deterministic max-cooccurrence hypothesis-to-reference map.

    V1 supports at most two speaker labels on either side.  Exhaustive
    enumeration is therefore smaller, clearer and more reproducible than a
    numerical Hungarian-algorithm dependency.
    """
    if not uem:
        raise EvaluationError("at least one UEM interval is required")
    file_ids = {item.file_id for item in (*reference, *hypothesis, *uem)}
    if len(file_ids) != 1:
        raise EvaluationError("mapping input must contain exactly one file_id")
    _validate_recording_ids(next(iter(file_ids)), reference, hypothesis, uem)
    cells = _timeline_cells(reference, hypothesis, _uem_by_channel(uem))
    refs = sorted({speaker for cell in cells for speaker in cell.reference_speakers})
    hyps = sorted({speaker for cell in cells for speaker in cell.hypothesis_speakers})
    if len(refs) > 2 or len(hyps) > 2:
        raise EvaluationError("V1 scorer supports at most two speaker labels per side")
    if not refs or not hyps:
        return ()
    cooccurrence = {
        (hyp, ref): sum(cell.duration_us for cell in cells
                        if hyp in cell.hypothesis_speakers and ref in cell.reference_speakers)
        for hyp in hyps for ref in refs
    }
    candidates: list[tuple[tuple[str, str], ...]] = []
    if len(hyps) <= len(refs):
        for chosen_refs in permutations(refs, len(hyps)):
            candidates.append(tuple(zip(hyps, chosen_refs)))
    else:
        for chosen_hyps in permutations(hyps, len(refs)):
            candidates.append(tuple(sorted(zip(chosen_hyps, refs))))
    best = min(
        candidates,
        key=lambda pairs: (-sum(cooccurrence[pair] for pair in pairs), pairs),
    )
    return tuple(SpeakerMapping(hyp, ref, cooccurrence[(hyp, ref)]) for hyp, ref in best)


def _mapping_from_cells(cells: Sequence[_Cell]) -> tuple[SpeakerMapping, ...]:
    refs = sorted({speaker for cell in cells for speaker in cell.reference_speakers})
    hyps = sorted({speaker for cell in cells for speaker in cell.hypothesis_speakers})
    if len(refs) > 2 or len(hyps) > 2:
        raise EvaluationError("V1 scorer supports at most two speaker labels per side")
    if not refs or not hyps:
        return ()
    cooccurrence = {(hyp, ref): sum(
        cell.duration_us for cell in cells
        if hyp in cell.hypothesis_speakers and ref in cell.reference_speakers
    ) for hyp in hyps for ref in refs}
    if len(hyps) <= len(refs):
        candidates = [tuple(zip(hyps, selected)) for selected in permutations(refs, len(hyps))]
    else:
        candidates = [tuple(sorted(zip(selected, refs))) for selected in permutations(hyps, len(refs))]
    best = min(candidates, key=lambda pairs: (-sum(cooccurrence[pair] for pair in pairs), pairs))
    return tuple(SpeakerMapping(hyp, ref, cooccurrence[(hyp, ref)]) for hyp, ref in best)


def _error_components(cells: Sequence[_Cell], mapping: Mapping[str, str], *, nonoverlap_only: bool = False) -> ErrorComponents:
    scored = reference_total = miss = false_alarm = confusion = 0
    for cell in cells:
        refs = cell.reference_speakers
        hyps = cell.hypothesis_speakers
        if nonoverlap_only and len(refs) > 1:
            continue
        duration = cell.duration_us
        scored += duration
        reference_count, hypothesis_count = len(refs), len(hyps)
        reference_total += reference_count * duration
        correct = sum(mapping.get(hypothesis) in refs for hypothesis in hyps)
        miss += max(0, reference_count - hypothesis_count) * duration
        false_alarm += max(0, hypothesis_count - reference_count) * duration
        confusion += (min(reference_count, hypothesis_count) - correct) * duration
    errors = miss + false_alarm + confusion
    return ErrorComponents(scored, reference_total, miss, false_alarm, confusion, _ratio(errors, reference_total))


def _jer(cells: Sequence[_Cell], mapping: Sequence[SpeakerMapping]) -> float:
    refs = sorted({speaker for cell in cells for speaker in cell.reference_speakers})
    if not refs:
        return 0.0
    inverse = {item.reference_speaker_id: item.hypothesis_speaker_id for item in mapping}
    errors: list[float] = []
    for ref in refs:
        hypothesis = inverse.get(ref)
        ref_duration = sum(cell.duration_us for cell in cells if ref in cell.reference_speakers)
        if hypothesis is None:
            errors.append(1.0)
            continue
        hyp_duration = sum(cell.duration_us for cell in cells if hypothesis in cell.hypothesis_speakers)
        intersection = sum(cell.duration_us for cell in cells
                           if ref in cell.reference_speakers and hypothesis in cell.hypothesis_speakers)
        union = ref_duration + hyp_duration - intersection
        errors.append(1.0 - _ratio(intersection, union))
    return sum(errors) / len(errors)


def _speaker_scores(cells: Sequence[_Cell], mapping: Mapping[str, str],
                    *, included_speakers: set[str] | None = None) -> tuple[SpeakerScore, ...]:
    refs = sorted(included_speakers if included_speakers is not None else
                  {speaker for cell in cells for speaker in cell.reference_speakers})
    out: list[SpeakerScore] = []
    for ref in refs:
        relevant = [cell for cell in cells if cell.reference_speakers == {ref}]
        reference_duration = sum(cell.duration_us for cell in relevant)
        assigned = sum(cell.duration_us for cell in relevant if cell.hypothesis_speakers)
        correct = sum(cell.duration_us for cell in relevant
                      if any(mapping.get(hypothesis) == ref for hypothesis in cell.hypothesis_speakers))
        out.append(SpeakerScore(ref, reference_duration, assigned, correct,
                                _ratio(assigned, reference_duration), _ratio(correct, assigned)))
    return tuple(out)


def _inside_uem(point_us: int, uem: Mapping[str, Sequence[tuple[int, int]]]) -> bool:
    return any(start <= point_us < end for intervals in uem.values() for start, end in intervals)


def _derive_change_boundaries(records: Sequence[RTTMRecord], uem: Mapping[str, Sequence[tuple[int, int]]],
                              *, reference: bool) -> tuple[int, ...]:
    is_speaker = _is_reference_speaker if reference else _is_hypothesis_speaker
    events: set[int] = set()
    for channel in sorted(uem):
        ordered = sorted((record for record in records
                          if record.channel == channel and is_speaker(record.speaker_id)),
                         key=lambda record: (record.start_us, record.end_us, record.speaker_id))
        starts = sorted({record.start_us for record in ordered})
        for point in starts:
            new_speakers = {record.speaker_id for record in ordered if record.start_us == point}
            prior = {record.speaker_id for record in ordered
                     if record.start_us < point and record.end_us >= point}
            if not prior:
                previous_end = max((record.end_us for record in ordered if record.end_us <= point), default=None)
                if previous_end is not None:
                    prior = {record.speaker_id for record in ordered if record.end_us == previous_end}
            if prior and any(speaker not in prior for speaker in new_speakers) and _inside_uem(point, {channel: uem[channel]}):
                events.add(point)
    return tuple(sorted(events))


def _der_collar_boundaries(reference: Sequence[RTTMRecord],
                           uem: Mapping[str, Sequence[tuple[int, int]]]) -> tuple[int, ...]:
    boundaries = {
        point
        for record in reference if _is_reference_speaker(record.speaker_id)
        for point in (record.start_us, record.end_us)
        if _inside_uem(point, {record.channel: uem.get(record.channel, ())})
    }
    return tuple(sorted(boundaries))


def _normalize_boundaries(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise EvaluationError(f"{name} times must be non-negative integer microseconds")
    return tuple(sorted(set(values)))


def score_boundaries(reference_us: Sequence[int], predicted_us: Sequence[int], *, collar_us: int) -> EventMetrics:
    if collar_us < 0:
        raise EvaluationError("boundary collar must be non-negative")
    refs = _normalize_boundaries(reference_us, name="reference boundary")
    preds = _normalize_boundaries(predicted_us, name="predicted boundary")
    i = j = true_positives = 0
    while i < len(refs) and j < len(preds):
        if preds[j] < refs[i] - collar_us:
            j += 1
        elif preds[j] > refs[i] + collar_us:
            i += 1
        else:
            true_positives += 1
            i += 1
            j += 1
    false_positives = len(preds) - true_positives
    false_negatives = len(refs) - true_positives
    precision, recall, f1 = _precision_recall_f1(true_positives, false_positives, false_negatives)
    return EventMetrics(True, true_positives, false_positives, false_negatives, precision, recall, f1)


def _precision_recall_f1(true_positives: int, false_positives: int, false_negatives: int) -> tuple[float, float, float]:
    if true_positives == false_positives == false_negatives == 0:
        return 1.0, 1.0, 1.0
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _ratio(2 * true_positives, 2 * true_positives + false_positives + false_negatives)
    return precision, recall, f1


def _overlap_metrics(cells: Sequence[_Cell], *, evaluated: bool) -> OverlapMetrics:
    if not evaluated:
        return OverlapMetrics(False, 0, 0, 0, 0, 0, None, None, None)
    reference_overlap = predicted_overlap = true_positive = false_positive = false_negative = 0
    for cell in cells:
        ref_positive = len(cell.reference_speakers) >= 2 or cell.reference_overlap_label
        pred_positive = len(cell.hypothesis_speakers) >= 2 or cell.hypothesis_overlap_label
        duration = cell.duration_us
        reference_overlap += ref_positive * duration
        predicted_overlap += pred_positive * duration
        true_positive += (ref_positive and pred_positive) * duration
        false_positive += (not ref_positive and pred_positive) * duration
        false_negative += (ref_positive and not pred_positive) * duration
    precision, recall, f1 = _precision_recall_f1(true_positive, false_positive, false_negative)
    return OverlapMetrics(True, reference_overlap, predicted_overlap, true_positive,
                          false_positive, false_negative, precision, recall, f1)


def _mapped_prediction(label: str | None, mapping: Mapping[str, str]) -> str | None:
    if label is None or not _is_hypothesis_speaker(label):
        return None
    return mapping.get(label, label)


def score_words(reference: Sequence[WordAnnotation], decisions: Sequence[WordDecision],
                *, mapping: Mapping[str, str] | None = None) -> WordMetrics:
    word_ids = [word.word_id for word in reference]
    decision_ids = [decision.word_id for decision in decisions]
    if len(set(word_ids)) != len(word_ids) or len(set(decision_ids)) != len(decision_ids):
        raise EvaluationError("duplicate word_id")
    if set(decision_ids) - set(word_ids):
        raise EvaluationError("word decision has no reference annotation")
    decision_by_id = {decision.word_id: decision.speaker_id for decision in decisions}
    speaker_mapping = mapping or {}
    eligible = [word for word in reference if word.attributable and not word.overlap_flag
                and not word.boundary_crossing_flag and word.ref_speaker_id is not None
                and _is_reference_speaker(word.ref_speaker_id)]
    assigned: list[tuple[WordAnnotation, str]] = []
    for word in eligible:
        predicted = _mapped_prediction(decision_by_id.get(word.word_id), speaker_mapping)
        if predicted is not None:
            assigned.append((word, predicted))
    correct = sum(predicted == word.ref_speaker_id for word, predicted in assigned)
    excluded = [word for word in reference if word.overlap_flag or word.boundary_crossing_flag]
    forced = sum(_mapped_prediction(decision_by_id.get(word.word_id), speaker_mapping) is not None for word in excluded)
    character_total = 0
    character_errors = 0
    assigned_by_id = {word.word_id: predicted for word, predicted in assigned}
    for word in eligible:
        weight = sum(not character.isspace() for character in word.text) or 1
        character_total += weight
        if assigned_by_id.get(word.word_id) != word.ref_speaker_id:
            character_errors += weight
    return WordMetrics(
        len(eligible), len(assigned), correct,
        _ratio(correct, len(assigned)),
        _ratio(len(assigned), len(eligible)),
        _ratio(len(eligible) - correct, len(eligible)),
        _ratio(len(assigned) - correct, len(assigned)),
        character_total,
        _ratio(character_errors, character_total),
        len(excluded), forced,
    )


def score_micros(decisions: Sequence[MicroDecision], *, mapping: Mapping[str, str] | None = None) -> MicroMetrics:
    ids = [decision.decision_id for decision in decisions]
    if len(set(ids)) != len(ids):
        raise EvaluationError("duplicate MICRO decision_id")
    speaker_mapping = mapping or {}
    eligible = [decision for decision in decisions
                if decision.eligible and decision.reference_speaker_id is not None]
    assigned = [(decision, _mapped_prediction(decision.predicted_speaker_id, speaker_mapping))
                for decision in eligible]
    assigned = [(decision, predicted) for decision, predicted in assigned if predicted is not None]
    correct = sum(predicted == decision.reference_speaker_id for decision, predicted in assigned)
    unknown = sum(decision.predicted_speaker_id is None or
                  str(decision.predicted_speaker_id).upper() in {"UNKNOWN", "UNKNOWN_SHORT"}
                  for decision in eligible)
    return MicroMetrics(len(eligible), len(assigned), correct, _ratio(correct, len(assigned)),
                        _ratio(len(assigned), len(eligible)), _ratio(unknown, len(eligible)))


def score_recording(recording: EvaluationRecording | None = None, *,
                    recording_id: str | None = None,
                    reference: Sequence[RTTMRecord] = (),
                    hypothesis: Sequence[RTTMRecord] = (),
                    uem: Sequence[UEMInterval] = (),
                    quality_status: str = "",
                    reference_scd_us: Sequence[int] | None = None,
                    predicted_scd_us: Sequence[int] | None = None,
                    overlap_reference_available: bool | None = None,
                    words: Sequence[WordAnnotation] | None = None,
                    word_decisions: Sequence[WordDecision] | None = None,
                    micro_decisions: Sequence[MicroDecision] | None = None,
                    config: ScoringConfig | None = None) -> RecordingScore:
    """Score one recording exactly on its UEM union."""
    cfg = ScoringConfig() if config is None else config
    if not isinstance(cfg, ScoringConfig):
        raise EvaluationError("config must be ScoringConfig")
    if recording is not None:
        if (recording_id is not None or reference or hypothesis or uem or quality_status
                or reference_scd_us is not None or predicted_scd_us is not None
                or overlap_reference_available is not None or words is not None
                or word_decisions is not None or micro_decisions is not None):
            raise EvaluationError("pass either recording or explicit recording fields")
        recording_id = recording.recording_id
        reference = recording.reference
        hypothesis = recording.hypothesis
        uem = recording.uem
        quality_status = recording.quality_status
        reference_scd_us = recording.reference_scd_us
        predicted_scd_us = recording.predicted_scd_us
        overlap_reference_available = recording.overlap_reference_available
        words = recording.words
        word_decisions = recording.word_decisions
        micro_decisions = recording.micro_decisions
    reference, hypothesis, uem = tuple(reference), tuple(hypothesis), tuple(uem)
    if recording_id is None:
        file_ids = {item.file_id for item in (*reference, *hypothesis, *uem)}
        if len(file_ids) != 1:
            raise EvaluationError("recording_id cannot be inferred")
        recording_id = next(iter(file_ids))
    _validate_recording_ids(recording_id, reference, hypothesis, uem)
    if not isinstance(quality_status, str):
        raise EvaluationError("quality_status must be a string")
    if words is None and word_decisions is not None:
        raise EvaluationError("word decisions require word annotations")
    uem_by_channel = _uem_by_channel(uem)
    derived_reference_scd = _derive_change_boundaries(reference, uem_by_channel, reference=True)
    ref_scd = _normalize_boundaries(
        derived_reference_scd if reference_scd_us is None else reference_scd_us,
        name="reference SCD",
    )
    derived_predicted_scd = _derive_change_boundaries(hypothesis, uem_by_channel, reference=False)
    pred_scd = _normalize_boundaries(
        derived_predicted_scd if predicted_scd_us is None else predicted_scd_us,
        name="predicted SCD",
    )
    if any(not _inside_uem(point, uem_by_channel) for point in (*ref_scd, *pred_scd)):
        raise EvaluationError("SCD boundary outside UEM")
    if words is not None and any(not any(start <= word.start_us and word.end_us <= end
                                         for intervals in uem_by_channel.values() for start, end in intervals)
                                 for word in words):
        raise EvaluationError("word interval outside UEM")

    cells = _timeline_cells(reference, hypothesis, uem_by_channel)
    reference_durations = _speaker_durations(cells, reference=True)
    hypothesis_durations = _speaker_durations(cells, reference=False)
    if len(reference_durations) > 2 or len(hypothesis_durations) > 2:
        raise EvaluationError("V1 scorer supports at most two speaker labels per side")
    der_cells = _timeline_cells(reference, hypothesis, _apply_collar(
        uem_by_channel, _der_collar_boundaries(reference, uem_by_channel), cfg.der_collar_us,
    ))
    mapping_items = _mapping_from_cells(der_cells)
    mapping = {item.hypothesis_speaker_id: item.reference_speaker_id for item in mapping_items}
    material_refs = {speaker for speaker, duration in reference_durations.items()
                     if duration >= cfg.reference_speaker_duration_floor_us}
    material_hyps = {speaker for speaker, duration in hypothesis_durations.items()
                     if duration >= cfg.hypothesis_speaker_duration_floor_us}
    reference_count, hypothesis_count = len(material_refs), len(material_hyps)
    eligible_h1, eligible_h2 = reference_count == 1, reference_count == 2
    complete_merge = eligible_h2 and hypothesis_count == 1
    false_h2 = eligible_h1 and hypothesis_count == 2
    ordered_hypothesis_durations = sorted(hypothesis_durations.values(), reverse=True)
    false_h2_secondary_duration = (
        ordered_hypothesis_durations[1] if eligible_h1 and len(ordered_hypothesis_durations) >= 2 else 0
    )
    h1_reference_duration = sum(reference_durations[speaker] for speaker in material_refs) if eligible_h1 else 0
    speaker_scores = _speaker_scores(cells, mapping, included_speakers=material_refs)
    word_score = None if words is None else score_words(words, word_decisions or (), mapping=mapping)
    micro_score = None if micro_decisions is None else score_micros(micro_decisions, mapping=mapping)
    return RecordingScore(
        recording_id=recording_id,
        mapping=mapping_items,
        diarization_all=_error_components(der_cells, mapping),
        diarization_nonoverlap=_error_components(der_cells, mapping, nonoverlap_only=True),
        jer=_jer(der_cells, mapping_items),
        reference_speaker_count=reference_count,
        hypothesis_speaker_count=hypothesis_count,
        eligible_reference_h1=eligible_h1,
        eligible_reference_h2=eligible_h2,
        acoustic_complete_merge=complete_merge,
        unsafe_complete_merge=complete_merge and quality_status in _PASS_SPEAKER_AWARE,
        false_h2=false_h2,
        false_h2_secondary_duration_us=false_h2_secondary_duration,
        false_h2_duration_ratio=_ratio(false_h2_secondary_duration, h1_reference_duration),
        speaker_count_correct=(eligible_h1 or eligible_h2) and reference_count == hypothesis_count,
        speakers=speaker_scores,
        worst_speaker_coverage=min((speaker.coverage for speaker in speaker_scores), default=None),
        worst_speaker_assigned_accuracy=min((speaker.assigned_accuracy for speaker in speaker_scores), default=None),
        scd=score_boundaries(ref_scd, pred_scd, collar_us=cfg.scd_collar_us),
        overlap=_overlap_metrics(cells, evaluated=(overlap_reference_available is True or any(
            len(cell.reference_speakers) >= 2 or cell.reference_overlap_label for cell in cells
        ))),
        words=word_score,
        micros=micro_score,
        quality_status=quality_status,
    )


def _sum_error_components(scores: Sequence[RecordingScore], attribute: str) -> ErrorComponents:
    components = [getattr(score, attribute) for score in scores]
    scored = sum(item.scored_uem_us for item in components)
    reference = sum(item.reference_speaker_us for item in components)
    miss = sum(item.miss_us for item in components)
    false_alarm = sum(item.false_alarm_us for item in components)
    confusion = sum(item.confusion_us for item in components)
    return ErrorComponents(scored, reference, miss, false_alarm, confusion,
                           _ratio(miss + false_alarm + confusion, reference))


def _aggregate_events(events: Sequence[EventMetrics]) -> EventMetrics:
    evaluated = [event for event in events if event.evaluated]
    if not evaluated:
        return EventMetrics(False, 0, 0, 0, None, None, None)
    true_positives = sum(event.true_positives for event in evaluated)
    false_positives = sum(event.false_positives for event in evaluated)
    false_negatives = sum(event.false_negatives for event in evaluated)
    precision, recall, f1 = _precision_recall_f1(true_positives, false_positives, false_negatives)
    return EventMetrics(True, true_positives, false_positives, false_negatives, precision, recall, f1)


def _aggregate_overlap(metrics: Sequence[OverlapMetrics]) -> OverlapMetrics:
    evaluated = [metric for metric in metrics if metric.evaluated]
    if not evaluated:
        return OverlapMetrics(False, 0, 0, 0, 0, 0, None, None, None)
    reference = sum(metric.reference_overlap_us for metric in evaluated)
    predicted = sum(metric.predicted_overlap_us for metric in evaluated)
    true_positive = sum(metric.true_positive_us for metric in evaluated)
    false_positive = sum(metric.false_positive_us for metric in evaluated)
    false_negative = sum(metric.false_negative_us for metric in evaluated)
    precision, recall, f1 = _precision_recall_f1(true_positive, false_positive, false_negative)
    return OverlapMetrics(True, reference, predicted, true_positive, false_positive,
                          false_negative, precision, recall, f1)


def _aggregate_words(metrics: Sequence[WordMetrics | None]) -> WordMetrics | None:
    present = [metric for metric in metrics if metric is not None]
    if not present:
        return None
    eligible = sum(metric.eligible_words for metric in present)
    assigned = sum(metric.assigned_words for metric in present)
    correct = sum(metric.correct_words for metric in present)
    characters = sum(metric.eligible_characters for metric in present)
    # Recover exact erroneous character mass from each file before pooling.
    character_errors = sum(round(metric.character_weighted_error * metric.eligible_characters) for metric in present)
    excluded = sum(metric.excluded_overlap_or_boundary_words for metric in present)
    forced = sum(metric.forced_overlap_or_boundary_assignments for metric in present)
    return WordMetrics(eligible, assigned, correct, _ratio(correct, assigned), _ratio(assigned, eligible),
                       _ratio(eligible - correct, eligible), _ratio(assigned - correct, assigned),
                       characters, _ratio(character_errors, characters), excluded, forced)


def _aggregate_micros(metrics: Sequence[MicroMetrics | None]) -> MicroMetrics | None:
    present = [metric for metric in metrics if metric is not None]
    if not present:
        return None
    eligible = sum(metric.eligible_micros for metric in present)
    assigned = sum(metric.assigned_micros for metric in present)
    correct = sum(metric.correct_micros for metric in present)
    unknown = sum(round(metric.unknown_short_rate * metric.eligible_micros) for metric in present)
    return MicroMetrics(eligible, assigned, correct, _ratio(correct, assigned),
                        _ratio(assigned, eligible), _ratio(unknown, eligible))


def aggregate_recording_scores(scores: Sequence[RecordingScore]) -> AggregateScore:
    items = tuple(scores)
    eligible_h2 = [score for score in items if score.eligible_reference_h2]
    eligible_h1 = [score for score in items if score.eligible_reference_h1]
    count_eligible = len(eligible_h1) + len(eligible_h2)
    all_speakers = [speaker for score in items for speaker in score.speakers]
    false_h2_duration = sum(score.false_h2_secondary_duration_us for score in eligible_h1)
    h1_reference_duration = sum(speaker.reference_duration_us for score in eligible_h1 for speaker in score.speakers)
    return AggregateScore(
        recording_count=len(items),
        diarization_all=_sum_error_components(items, "diarization_all"),
        diarization_nonoverlap=_sum_error_components(items, "diarization_nonoverlap"),
        jer=sum(score.jer for score in items) / len(items) if items else 0.0,
        eligible_h2_files=len(eligible_h2),
        complete_merge_files=sum(score.acoustic_complete_merge for score in eligible_h2),
        acoustic_complete_merge_rate=_ratio(sum(score.acoustic_complete_merge for score in eligible_h2), len(eligible_h2)),
        unsafe_complete_merge_rate=_ratio(sum(score.unsafe_complete_merge for score in eligible_h2), len(eligible_h2)),
        eligible_h1_files=len(eligible_h1),
        false_h2_files=sum(score.false_h2 for score in eligible_h1),
        false_h2_rate=_ratio(sum(score.false_h2 for score in eligible_h1), len(eligible_h1)),
        false_h2_secondary_duration_us=false_h2_duration,
        false_h2_duration_ratio=_ratio(false_h2_duration, h1_reference_duration),
        speaker_count_accuracy=_ratio(sum(score.speaker_count_correct for score in items), count_eligible),
        worst_speaker_coverage=min((speaker.coverage for speaker in all_speakers), default=None),
        worst_speaker_assigned_accuracy=min((speaker.assigned_accuracy for speaker in all_speakers), default=None),
        scd=_aggregate_events([score.scd for score in items]),
        overlap=_aggregate_overlap([score.overlap for score in items]),
        words=_aggregate_words([score.words for score in items]),
        micros=_aggregate_micros([score.micros for score in items]),
    )


def _hash_sample_index(seed: int, iteration: int, draw: int, population: int) -> int:
    material = f"{seed}:{iteration}:{draw}:{population}".encode("ascii")
    return int.from_bytes(sha256(material).digest()[:8], "big") % population


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise EvaluationError("cannot take quantile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_recording_ci(scores: Sequence[RecordingScore], *, metric: str,
                           statistic: Callable[[Sequence[RecordingScore]], float | None],
                           iterations: int = 2_000, seed: int = 17_029,
                           confidence_level: float = 0.95) -> BootstrapCI | None:
    """Deterministic percentile CI with recordings as the resampling unit."""
    if (not metric or isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0
            or isinstance(seed, bool) or not isinstance(seed, int)
            or not 0.0 < confidence_level < 1.0):
        raise EvaluationError("invalid bootstrap configuration")
    items = tuple(sorted(scores, key=lambda score: score.recording_id))
    if not items or iterations <= 0:
        return None
    point = statistic(items)
    if point is None:
        return None
    if not math.isfinite(point):
        raise EvaluationError("bootstrap statistic must be finite")
    samples: list[float] = []
    for iteration in range(iterations):
        sample = tuple(items[_hash_sample_index(seed, iteration, draw, len(items))]
                       for draw in range(len(items)))
        value = statistic(sample)
        if value is not None:
            if not math.isfinite(value):
                raise EvaluationError("bootstrap statistic must be finite")
            samples.append(value)
    if not samples:
        return None
    tail = (1.0 - confidence_level) / 2.0
    return BootstrapCI(metric, confidence_level, point, _quantile(samples, tail),
                       _quantile(samples, 1.0 - tail), len(samples), seed)


def _ratio_statistic(numerator: Callable[[RecordingScore], int | bool],
                     eligible: Callable[[RecordingScore], bool]) -> Callable[[Sequence[RecordingScore]], float | None]:
    def calculate(scores: Sequence[RecordingScore]) -> float | None:
        selected = [score for score in scores if eligible(score)]
        return (sum(int(numerator(score)) for score in selected) / len(selected)) if selected else None
    return calculate


def _bootstrap_suite(scores: Sequence[RecordingScore], config: ScoringConfig) -> tuple[BootstrapCI, ...]:
    def der(items: Sequence[RecordingScore]) -> float | None:
        components = _sum_error_components(items, "diarization_all")
        if not components.reference_speaker_us:
            return None
        return components.der

    def jer(items: Sequence[RecordingScore]) -> float | None:
        return sum(score.jer for score in items) / len(items) if items else None

    def worst_coverage(items: Sequence[RecordingScore]) -> float | None:
        values = [speaker.coverage for score in items for speaker in score.speakers]
        return min(values) if values else None

    def worst_accuracy(items: Sequence[RecordingScore]) -> float | None:
        values = [speaker.assigned_accuracy for score in items for speaker in score.speakers]
        return min(values) if values else None

    def word_precision(items: Sequence[RecordingScore]) -> float | None:
        aggregate = _aggregate_words([score.words for score in items])
        return aggregate.precision if aggregate is not None and aggregate.assigned_words else None

    def word_coverage(items: Sequence[RecordingScore]) -> float | None:
        aggregate = _aggregate_words([score.words for score in items])
        return aggregate.coverage if aggregate is not None and aggregate.eligible_words else None

    def micro_precision(items: Sequence[RecordingScore]) -> float | None:
        aggregate = _aggregate_micros([score.micros for score in items])
        return aggregate.precision if aggregate is not None and aggregate.assigned_micros else None

    def micro_coverage(items: Sequence[RecordingScore]) -> float | None:
        aggregate = _aggregate_micros([score.micros for score in items])
        return aggregate.coverage if aggregate is not None and aggregate.eligible_micros else None

    def scd_f1(items: Sequence[RecordingScore]) -> float | None:
        aggregate = _aggregate_events([score.scd for score in items])
        return aggregate.f1 if aggregate.evaluated else None

    def overlap_precision(items: Sequence[RecordingScore]) -> float | None:
        aggregate = _aggregate_overlap([score.overlap for score in items])
        return aggregate.precision if aggregate.evaluated else None

    def overlap_recall(items: Sequence[RecordingScore]) -> float | None:
        aggregate = _aggregate_overlap([score.overlap for score in items])
        return aggregate.recall if aggregate.evaluated else None

    def false_h2_duration(items: Sequence[RecordingScore]) -> float | None:
        aggregate = aggregate_recording_scores(items)
        return aggregate.false_h2_duration_ratio if aggregate.eligible_h1_files else None

    definitions: tuple[tuple[str, Callable[[Sequence[RecordingScore]], float | None]], ...] = (
        ("der_all", der),
        ("der_nonoverlap", lambda items: (
            _sum_error_components(items, "diarization_nonoverlap").der
            if _sum_error_components(items, "diarization_nonoverlap").reference_speaker_us else None)),
        ("jer", jer),
        ("acoustic_complete_merge_rate", _ratio_statistic(
            lambda score: score.acoustic_complete_merge, lambda score: score.eligible_reference_h2)),
        ("unsafe_complete_merge_rate", _ratio_statistic(
            lambda score: score.unsafe_complete_merge, lambda score: score.eligible_reference_h2)),
        ("false_h2_rate", _ratio_statistic(lambda score: score.false_h2, lambda score: score.eligible_reference_h1)),
        ("false_h2_duration_ratio", false_h2_duration),
        ("worst_speaker_coverage", worst_coverage),
        ("worst_speaker_assigned_accuracy", worst_accuracy),
        ("scd_f1", scd_f1),
        ("overlap_precision", overlap_precision),
        ("overlap_recall", overlap_recall),
        ("word_attribution_precision", word_precision),
        ("word_attribution_coverage", word_coverage),
        ("micro_precision", micro_precision),
        ("micro_coverage", micro_coverage),
    )
    results = [bootstrap_recording_ci(
        scores, metric=name, statistic=statistic,
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
    ) for name, statistic in definitions]
    return tuple(result for result in results if result is not None)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    raise EvaluationError(f"cannot canonicalize {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationError("value is not canonical JSON") from exc


def _input_bytes(value: bytes | bytearray | memoryview | str | Path | Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Path):
        try:
            return value.read_bytes()
        except OSError as exc:
            raise EvaluationError("unable to read evaluation input") from exc
    if isinstance(value, str):
        return value.encode("utf-8")
    return _canonical_json_bytes(value)


def build_run_manifest(*, inputs: Iterable[Any], config: ScoringConfig | Mapping[str, Any],
                       scorer_path: str | Path | None = None) -> EvaluationRunManifest:
    """Hash inputs without retaining paths, IDs, transcript, or configuration values."""
    if isinstance(inputs, Mapping):
        # Bind semantic roles to bytes (reference/hypothesis/UEM) while only
        # retaining the resulting digest in the redacted manifest.
        material = (
            _canonical_json_bytes({
                "role": str(role),
                "payload_sha256": sha256(_input_bytes(value)).hexdigest(),
            })
            for role, value in inputs.items()
        )
    else:
        material = (_input_bytes(value) for value in inputs)
    input_hashes = tuple(sorted(sha256(value).hexdigest() for value in material))
    input_set_sha256 = sha256(_canonical_json_bytes(input_hashes)).hexdigest()
    config_sha256 = sha256(_canonical_json_bytes(config)).hexdigest()
    source = Path(__file__) if scorer_path is None else Path(scorer_path)
    try:
        scorer_sha256 = sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvaluationError("unable to hash scorer source") from exc
    unsigned = {
        "schema_version": SCORER_SCHEMA_VERSION,
        "input_count": len(input_hashes),
        "input_hashes": input_hashes,
        "input_set_sha256": input_set_sha256,
        "config_sha256": config_sha256,
        "scorer_sha256": scorer_sha256,
    }
    manifest_sha256 = sha256(_canonical_json_bytes(unsigned)).hexdigest()
    return EvaluationRunManifest(SCORER_SCHEMA_VERSION, len(input_hashes), input_hashes,
                                 input_set_sha256, config_sha256, scorer_sha256, manifest_sha256)


def write_run_manifest(manifest: EvaluationRunManifest, path: str | Path) -> None:
    """Create a read-only manifest and refuse to overwrite prior evidence."""
    target = Path(path)
    payload = json.dumps(manifest.as_dict(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False) + "\n"
    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        target.chmod(0o444)
    except FileExistsError as exc:
        raise EvaluationError("run manifest already exists") from exc
    except OSError as exc:
        raise EvaluationError("unable to write run manifest") from exc


def score_corpus(recordings: Sequence[EvaluationRecording], *,
                 config: ScoringConfig | None = None,
                 manifest_inputs: Iterable[Any] | Mapping[str, Any] | None = None) -> EvaluationReport:
    """Score, aggregate, bootstrap, and content-address one offline corpus."""
    cfg = ScoringConfig() if config is None else config
    if not isinstance(cfg, ScoringConfig):
        raise EvaluationError("config must be ScoringConfig")
    inputs = tuple(recordings)
    if not inputs:
        raise EvaluationError("at least one evaluation recording is required")
    if any(not isinstance(recording, EvaluationRecording) for recording in inputs):
        raise EvaluationError("recordings must contain EvaluationRecording values")
    ids = [recording.recording_id for recording in inputs]
    if len(ids) != len(set(ids)):
        raise EvaluationError("duplicate recording_id")
    ordered = tuple(sorted(inputs, key=lambda recording: recording.recording_id))
    scores = tuple(score_recording(recording, config=cfg) for recording in ordered)
    subgroup_members: dict[str, list[RecordingScore]] = defaultdict(list)
    score_by_id = {score.recording_id: score for score in scores}
    for recording in ordered:
        for key, value in recording.subgroups:
            subgroup_members[f"{key}={value}"].append(score_by_id[recording.recording_id])
    subgroups = tuple(SubgroupScore(label, aggregate_recording_scores(subgroup_members[label]),
                                    _bootstrap_suite(subgroup_members[label], cfg))
                      for label in sorted(subgroup_members))
    manifest = build_run_manifest(inputs=ordered if manifest_inputs is None else manifest_inputs, config=cfg)
    return EvaluationReport(scores, aggregate_recording_scores(scores), subgroups,
                            _bootstrap_suite(scores, cfg), manifest)


# Document and caller terminology aliases.
score_dataset = score_corpus
evaluate_recordings = score_corpus


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def word_attribution_metrics(reference: Sequence[WordAnnotation], decisions: Sequence[WordDecision]) -> tuple[float, float]:
    eligible = {w.word_id: w for w in reference if w.attributable and not w.overlap_flag
                and not w.boundary_crossing_flag and w.ref_speaker_id is not None}
    predicted = {d.word_id: d.speaker_id for d in decisions}
    assigned = [(w, predicted[w.word_id]) for w in eligible.values()
                if w.word_id in predicted and predicted[w.word_id] is not None]
    correct = sum(p == w.ref_speaker_id for w, p in assigned)
    return _ratio(correct, len(assigned)), _ratio(len(assigned), len(eligible))


def micro_metrics(decisions: Sequence[MicroDecision]) -> tuple[float, float]:
    eligible = [d for d in decisions if d.eligible and d.reference_speaker_id is not None]
    assigned = [d for d in eligible if d.predicted_speaker_id is not None
                and _is_hypothesis_speaker(d.predicted_speaker_id)]
    correct = sum(d.predicted_speaker_id == d.reference_speaker_id for d in assigned)
    return _ratio(correct, len(assigned)), _ratio(len(assigned), len(eligible))


def score_safety(*, reference_by_file: Mapping[str, Sequence[RTTMRecord]], predicted_by_file: Mapping[str, Sequence[RTTMRecord]],
                 quality_status_by_file: Mapping[str, str] | None = None,
                 words_by_file: Mapping[str, Sequence[WordAnnotation]] | None = None,
                 word_decisions_by_file: Mapping[str, Sequence[WordDecision]] | None = None,
                 micro_decisions: Sequence[MicroDecision] = (),
                 speaker_duration_floor_us: int = 2) -> SafetyMetrics:
    """Compatibility safety summary with duration-aware topology counting.

    The comprehensive scorer uses :class:`ScoringConfig`'s calibrated 250 ms
    default.  This legacy summary keeps a two-microsecond default so historical
    synthetic fixtures expressed in tiny integer microseconds remain valid,
    while a one-microsecond fake label still cannot hide a merge.
    """
    if (isinstance(speaker_duration_floor_us, bool) or not isinstance(speaker_duration_floor_us, int)
            or speaker_duration_floor_us <= 0):
        raise EvaluationError("speaker_duration_floor_us must be a positive integer")

    def material_speakers(records: Sequence[RTTMRecord], *, reference: bool) -> set[str]:
        predicate = _is_reference_speaker if reference else _is_hypothesis_speaker
        intervals: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for record in records:
            if predicate(record.speaker_id):
                intervals[(record.channel, record.speaker_id)].append((record.start_us, record.end_us))
        duration_by_speaker: dict[str, int] = defaultdict(int)
        for (_, speaker), spans in intervals.items():
            duration_by_speaker[speaker] += sum(end - start for start, end in _merge_intervals(spans))
        return {speaker for speaker, duration in duration_by_speaker.items()
                if duration >= speaker_duration_floor_us}

    quality_status_by_file = quality_status_by_file or {}
    eligible_h2 = eligible_h1 = 0
    complete = unsafe = false_h2 = false_pass = count_correct = 0
    for file_id, refs in reference_by_file.items():
        ref_speakers = material_speakers(refs, reference=True)
        pred_speakers = material_speakers(predicted_by_file.get(file_id, ()), reference=False)
        if len(ref_speakers) == 2:
            eligible_h2 += 1
            merged = len(pred_speakers) == 1
            complete += merged
            passed = quality_status_by_file.get(file_id, "") in _PASS_SPEAKER_AWARE
            unsafe += merged and passed
        elif len(ref_speakers) == 1:
            eligible_h1 += 1
            false_h2 += len(pred_speakers) == 2
        if len(ref_speakers) in {1, 2}:
            count_correct += len(ref_speakers) == len(pred_speakers)
        risk = len(ref_speakers) == 2 and len(pred_speakers) == 1
        if risk and quality_status_by_file.get(file_id, "") in _ALL_PASS:
            false_pass += 1
    all_words = []; all_decisions = []
    for file_id, words in (words_by_file or {}).items():
        all_words.extend(words); all_decisions.extend((word_decisions_by_file or {}).get(file_id, ()))
    wp, wc = word_attribution_metrics(all_words, all_decisions) if all_words else (0.0, 0.0)
    mp, mc = micro_metrics(micro_decisions)
    return SafetyMetrics(
        reference_files=len(reference_by_file),
        eligible_two_speaker_files=eligible_h2,
        acoustic_complete_merge_rate=_ratio(complete, eligible_h2),
        unsafe_complete_merge_rate=_ratio(unsafe, eligible_h2),
        word_attribution_precision=wp,
        word_attribution_coverage=wc,
        micro_precision=mp,
        micro_coverage=mc,
        quality_false_pass_rate=_ratio(false_pass, len(reference_by_file)),
        eligible_one_speaker_files=eligible_h1,
        false_h2_rate=_ratio(false_h2, eligible_h1),
        speaker_count_accuracy=_ratio(count_correct, eligible_h1 + eligible_h2),
    )


def validate_calibration_split(calibration: Iterable[RecordingManifest], holdout: Iterable[RecordingManifest],
                               *, profile_id: str | None = None, profile_split: str = "CALIBRATION") -> None:
    """Guard calibration profile creation/use against holdout contamination."""
    cal = tuple(calibration); ho = tuple(holdout)
    validate_recording_session_splits(cal + ho)
    if any(r.split != profile_split for r in cal):
        raise CalibrationGuardError("calibration records have wrong split")
    if any(r.split != "RELEASE_HOLDOUT" for r in ho):
        raise CalibrationGuardError("holdout records have wrong split")
    if profile_id is not None and not profile_id.strip():
        raise CalibrationGuardError("empty calibration profile id")
