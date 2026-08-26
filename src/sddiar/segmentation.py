"""Rule-based VAD/SCD/OSD evidence assembly for the offline reference path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import SpeechRegion
from .diarization import cosine_distance
from .vad import VadFrame


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    probe_id: str
    start_us: int
    end_us: int
    clean_speech_us: int
    vector: tuple[float, ...]
    quality: float


@dataclass(frozen=True, slots=True)
class SpeakerChangeEvent:
    time_us: int
    evidence: float
    evidence_id: str
    approved: bool = False


@dataclass(frozen=True, slots=True)
class OverlapEvent:
    start_us: int
    end_us: int
    overlap_evidence: float
    evidence_ids: tuple[str, ...] = ()
    is_high: bool = False


@dataclass(frozen=True, slots=True)
class SegmentationEvidence:
    speech_regions: tuple[SpeechRegion, ...]
    scd_events: tuple[SpeakerChangeEvent, ...]
    overlap_regions: tuple[OverlapEvent, ...]
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
        approved_scd_events: Sequence[SpeakerChangeEvent] = (),
        approved_overlap_regions: Sequence[OverlapEvent] = (),
    ) -> SegmentationEvidence:
        speech_regions = self._speech_regions(view_id, vad_frames)
        discontinuities = self._probe_discontinuities(probes)
        # Only an explicitly approved SCD source can create an intra-region cut.
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
                events.append(SpeakerChangeEvent(right.start_us, distance, f"probe-{left.probe_id}-{right.probe_id}", approved=False))
        return tuple(events)
