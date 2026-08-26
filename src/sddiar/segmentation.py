"""Rule-based VAD/SCD/OSD evidence assembly for the offline reference path."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from .contracts import SpeechRegion
from .diarization import cosine_distance
from .errors import ContractValidationError
from .vad import VadFrame


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    probe_id: str
    start_us: int
    end_us: int
    clean_speech_us: int
    vector: tuple[float, ...]
    quality: float


_ENFORCEMENT_TOKEN = object()
_MAX_EVENT_ID_LENGTH = 256


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_EVENT_ID_LENGTH:
        raise ContractValidationError(f"{name} must be a bounded non-empty string")
    if any(ord(char) < 32 for char in value):
        raise ContractValidationError(f"{name} contains a control character")
    return value


def _event_time(value: Any, name: str = "time_us") -> int:
    if type(value) is not int or value < 0:
        raise ContractValidationError(f"{name} must be a non-negative integer")
    return value


def _evidence(value: Any, name: str, high: float = 2.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= high:
        raise ContractValidationError(f"{name} must be finite in [0,{high:g}]")
    return result


@dataclass(frozen=True, slots=True)
class SpeakerChangeEvent:
    """Diagnostic-only SCD evidence.

    ``approved=True`` is intentionally invalid.  Authority is represented by
    a private sealed type created only by :func:`authorize_scd_event`.
    """

    time_us: int
    evidence: float
    evidence_id: str
    approved: bool = False

    def __post_init__(self) -> None:
        _event_time(self.time_us)
        _evidence(self.evidence, "evidence")
        _identifier(self.evidence_id, "evidence_id")
        if type(self.approved) is not bool or self.approved:
            raise ContractValidationError("diagnostic SCD evidence cannot be approved")


@dataclass(frozen=True, slots=True, init=False)
class _EnforceableSpeakerChangeEvent:
    """Sealed SCD event; no public constructor or root-package export."""

    time_us: int
    evidence: float
    evidence_id: str
    source_id: str
    calibration_profile_id: str
    binding_sha256: str
    approved: bool = field(default=True, init=False)

    def __init__(self, *, _token: object, time_us: int, evidence: float, evidence_id: str,
                 source_id: str, calibration_profile_id: str, binding_sha256: str) -> None:
        if _token is not _ENFORCEMENT_TOKEN:
            raise TypeError("enforceable SCD event requires release authorization")
        _event_time(time_us)
        _evidence(evidence, "evidence")
        _identifier(evidence_id, "evidence_id")
        _identifier(source_id, "source_id")
        _identifier(calibration_profile_id, "calibration_profile_id")
        if not isinstance(binding_sha256, str) or len(binding_sha256) != 64 or any(c not in "0123456789abcdef" for c in binding_sha256):
            raise ContractValidationError("binding_sha256 must be lowercase SHA-256")
        object.__setattr__(self, "time_us", time_us)
        object.__setattr__(self, "evidence", float(evidence))
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "calibration_profile_id", calibration_profile_id)
        object.__setattr__(self, "binding_sha256", binding_sha256)
        object.__setattr__(self, "approved", True)


@dataclass(frozen=True, slots=True)
class OverlapEvent:
    """Diagnostic-only OSD evidence that cannot protect an interval."""

    start_us: int
    end_us: int
    overlap_evidence: float
    evidence_ids: tuple[str, ...] = ()
    is_high: bool = False

    def __post_init__(self) -> None:
        _event_time(self.start_us, "start_us")
        _event_time(self.end_us, "end_us")
        if self.end_us <= self.start_us:
            raise ContractValidationError("overlap interval must be non-empty")
        _evidence(self.overlap_evidence, "overlap_evidence")
        if type(self.evidence_ids) is not tuple:
            raise ContractValidationError("evidence_ids must be bounded strings")
        for value in self.evidence_ids:
            _identifier(value, "evidence_id")
        if type(self.is_high) is not bool or self.is_high:
            raise ContractValidationError("diagnostic OSD evidence cannot be high")


@dataclass(frozen=True, slots=True, init=False)
class _EnforceableOverlapEvent:
    """Sealed OSD event; no public constructor or root-package export."""

    start_us: int
    end_us: int
    overlap_evidence: float
    evidence_ids: tuple[str, ...]
    source_id: str
    calibration_profile_id: str
    binding_sha256: str
    is_high: bool = field(default=True, init=False)

    def __init__(
        self,
        *,
        _token: object,
        start_us: int,
        end_us: int,
        overlap_evidence: float,
        evidence_ids: tuple[str, ...],
        source_id: str,
        calibration_profile_id: str,
        binding_sha256: str,
    ) -> None:
        if _token is not _ENFORCEMENT_TOKEN:
            raise TypeError("enforceable OSD event requires release authorization")
        _event_time(start_us, "start_us")
        _event_time(end_us, "end_us")
        if end_us <= start_us:
            raise ContractValidationError("overlap interval must be non-empty")
        _evidence(overlap_evidence, "overlap_evidence")
        if type(evidence_ids) is not tuple or not evidence_ids:
            raise ContractValidationError("enforceable OSD evidence_ids are required")
        for value in evidence_ids:
            _identifier(value, "evidence_id")
        _identifier(source_id, "source_id")
        _identifier(calibration_profile_id, "calibration_profile_id")
        if not isinstance(binding_sha256, str) or len(binding_sha256) != 64 or any(c not in "0123456789abcdef" for c in binding_sha256):
            raise ContractValidationError("binding_sha256 must be lowercase SHA-256")
        object.__setattr__(self, "start_us", start_us)
        object.__setattr__(self, "end_us", end_us)
        object.__setattr__(self, "overlap_evidence", float(overlap_evidence))
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "calibration_profile_id", calibration_profile_id)
        object.__setattr__(self, "binding_sha256", binding_sha256)
        object.__setattr__(self, "is_high", True)


def _require_release_binding(binding: Any) -> None:
    from .calibration import VerifiedCalibrationBinding

    if type(binding) is not VerifiedCalibrationBinding or not binding.release_authorized:
        raise ContractValidationError("release-authorized VerifiedCalibrationBinding is required")


def _threshold(value: Any, name: str, high: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= high:
        raise ContractValidationError(f"{name} must be finite in [0,{high:g}]")
    return result


def _authorized_threshold(binding: Any, value: Any, name: str, high: float) -> float:
    supplied = _threshold(value, name, high=high)
    configured = binding.thresholds.get(name)
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        raise ContractValidationError(f"release calibration does not contain {name}")
    configured_value = float(configured)
    if not math.isfinite(configured_value) or not 0.0 <= configured_value <= high:
        raise ContractValidationError(f"release calibration {name} is outside its range")
    if supplied != configured_value:
        raise ContractValidationError(f"{name} does not match the release calibration")
    return supplied


def authorize_scd_event(
    binding: Any,
    *,
    time_us: int,
    evidence: float,
    evidence_id: str,
    source_id: str,
    scd_evidence_min: float,
) -> Any:
    """Reject until a signed per-job boundary-evidence attestation exists.

    A release calibration profile alone authorizes thresholds, not arbitrary
    caller-supplied event bytes or source times.  Shadow evidence is available
    through :mod:`sddiar.boundary_evidence`; enforcement remains unavailable.
    """

    _require_release_binding(binding)
    raise ContractValidationError(
        "release binding alone cannot authorize SCD; signed per-job evidence is required"
    )


def authorize_overlap_event(
    binding: Any,
    *,
    start_us: int,
    end_us: int,
    overlap_evidence: float,
    evidence_ids: tuple[str, ...],
    source_id: str,
    osd_evidence_min: float,
) -> Any:
    """Reject until a signed per-job boundary-evidence attestation exists."""

    _require_release_binding(binding)
    raise ContractValidationError(
        "release binding alone cannot authorize OSD; signed per-job evidence is required"
    )


# Explicit aliases make the capability boundary discoverable without exposing
# the sealed concrete classes.
authorize_speaker_change_event = authorize_scd_event
authorize_osd_event = authorize_overlap_event
authorize_speaker_change = authorize_scd_event
authorize_overlap = authorize_overlap_event

# Historical diagnostic names remain module-level aliases.  They are aliases
# of the public diagnostic dataclasses, never of the sealed capability types.
DiagnosticSpeakerChangeEvent = SpeakerChangeEvent
DiagnosticOverlapEvent = OverlapEvent


@dataclass(frozen=True, slots=True)
class SegmentationEvidence:
    speech_regions: tuple[SpeechRegion, ...]
    scd_events: tuple[_EnforceableSpeakerChangeEvent, ...]
    overlap_regions: tuple[_EnforceableOverlapEvent, ...]
    probe_discontinuities: tuple[SpeakerChangeEvent, ...]


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    vad_merge_gap_us: int = 200_000
    probe_discontinuity_min: float = 0.2


class RuleEvidenceSegmentation:
    """Assemble conservative evidence without inferring an unobserved change.

    Probe discontinuity is retained for diagnostics but cannot by itself create
    an SCD split inside continuous speech. Approved SCD/OSD backends may inject
    their evidence through the explicit arguments.
    """

    def __init__(self, config: SegmentationConfig | None = None):
        self.config = config or SegmentationConfig()

    def build(
        self,
        *,
        view_id: str,
        vad_frames: Sequence[VadFrame],
        probes: Sequence[ProbeEvidence] = (),
        approved_scd_events: Sequence[Any] = (),
        approved_overlap_regions: Sequence[Any] = (),
    ) -> SegmentationEvidence:
        speech_regions = self._speech_regions(view_id, vad_frames)
        discontinuities = self._probe_discontinuities(probes)
        # Do not silently promote mappings or diagnostic evidence.  A caller
        # using this explicitly approved seam must provide the capability type.
        for event in approved_scd_events:
            if type(event) is not _EnforceableSpeakerChangeEvent:
                raise ContractValidationError("approved_scd_events requires enforceable SCD events")
        for event in approved_overlap_regions:
            if type(event) is not _EnforceableOverlapEvent:
                raise ContractValidationError("approved_overlap_regions requires enforceable OSD events")
        scd_events = tuple(sorted(approved_scd_events, key=lambda event: (event.time_us, event.evidence_id)))
        overlaps = tuple(sorted(approved_overlap_regions, key=lambda event: (event.start_us, event.end_us)))
        return SegmentationEvidence(speech_regions, scd_events, overlaps, discontinuities)

    def _speech_regions(self, view_id: str, frames: Sequence[VadFrame]) -> tuple[SpeechRegion, ...]:
        speech = sorted((frame for frame in frames if frame.is_speech), key=lambda frame: (frame.start_us, frame.end_us))
        regions: list[SpeechRegion] = []
        start = end = None
        evidence: list[float] = []
        for frame in speech:
            if start is None:
                start, end, evidence = frame.start_us, frame.end_us, [frame.speech_evidence or 0.0]
                continue
            if frame.start_us - end <= self.config.vad_merge_gap_us:
                end = max(end, frame.end_us)
                evidence.append(frame.speech_evidence or 0.0)
                continue
            regions.append(self._region(view_id, len(regions), start, end, evidence))
            start, end, evidence = frame.start_us, frame.end_us, [frame.speech_evidence or 0.0]
        if start is not None and end is not None:
            regions.append(self._region(view_id, len(regions), start, end, evidence))
        return tuple(regions)

    @staticmethod
    def _region(view_id: str, ordinal: int, start_us: int, end_us: int, evidence: Sequence[float]) -> SpeechRegion:
        return SpeechRegion(f"speech-{view_id}-{ordinal}", view_id, start_us, end_us, sum(evidence) / len(evidence))

    def _probe_discontinuities(self, probes: Sequence[ProbeEvidence]) -> tuple[SpeakerChangeEvent, ...]:
        ordered = sorted(probes, key=lambda probe: (probe.start_us, probe.end_us, probe.probe_id))
        events: list[SpeakerChangeEvent] = []
        for left, right in zip(ordered, ordered[1:]):
            if len(left.vector) != len(right.vector):
                continue
            distance = cosine_distance(left.vector, right.vector)
            if distance >= self.config.probe_discontinuity_min:
                events.append(SpeakerChangeEvent(right.start_us, distance, f"probe-{left.probe_id}-{right.probe_id}"))
        return tuple(events)


def _interval(value: Any) -> tuple[int, int] | None:
    """Read a bounded interval from a pair or an interval-like object."""

    if isinstance(value, (tuple, list)):
        if len(value) < 2:
            return None
        start, end = value[0], value[1]
    elif isinstance(value, dict):
        start = value.get("start_us", value.get("start"))
        end = value.get("end_us", value.get("end"))
    else:
        start = getattr(value, "start_us", getattr(value, "start", None))
        end = getattr(value, "end_us", getattr(value, "end", None))
    if type(start) is not int or type(end) is not int:
        return None
    return start, end


def clip_and_coalesce_speech_mask(
    source_duration_us: int,
    speech_mask: Sequence[Any],
    *,
    merge_gap_us: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Clip source-time speech intervals and coalesce deterministic runs.

    Intervals outside the decoded source are clipped, invalid/empty intervals
    are discarded, and sorted intervals separated by at most ``merge_gap_us``
    are merged.  This helper intentionally returns plain scalar tuples so a
    model or mask object cannot smuggle capability into the segmentation seam.
    """

    if type(source_duration_us) is not int or source_duration_us < 0:
        raise ValueError("source_duration_us must be a non-negative integer")
    if type(merge_gap_us) is not int or merge_gap_us < 0:
        raise ValueError("merge_gap_us must be a non-negative integer")
    if len(speech_mask) > 1_000_000:
        raise ValueError("speech mask exceeds resource bound")
    clipped: list[tuple[int, int]] = []
    for raw in speech_mask:
        interval = _interval(raw)
        if interval is None:
            continue
        start, end = interval
        start, end = max(0, start), min(source_duration_us, end)
        if end > start:
            clipped.append((start, end))
    clipped.sort()
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1] + merge_gap_us:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


# Short aliases kept intentionally boring for callers that use "speech
# regions" rather than "mask" terminology.
clip_and_coalesce_speech_regions = clip_and_coalesce_speech_mask
clip_speech_mask = clip_and_coalesce_speech_mask
clip_speech_regions = clip_and_coalesce_speech_mask
