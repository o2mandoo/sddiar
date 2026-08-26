"""Session-scoped human correction for UNKNOWN diarization spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .contracts import DiarizationSpan
from .errors import ContractValidationError


class CorrectionConflictError(ContractValidationError):
    """A correction would overwrite an already attributed or overlap span."""


@dataclass(frozen=True, slots=True)
class HumanSpeakerCorrection:
    correction_id: str
    source_start_us: int
    source_end_us: int
    confirmed_speaker_id: str
    reviewer_evidence_id: str

    def __post_init__(self) -> None:
        if not self.correction_id or not self.reviewer_evidence_id:
            raise ContractValidationError("correction and reviewer evidence IDs are required")
        if self.confirmed_speaker_id not in {"SPEAKER_00", "SPEAKER_01"}:
            raise ContractValidationError("human correction must target a session speaker")
        if type(self.source_start_us) is not int or type(self.source_end_us) is not int or self.source_start_us >= self.source_end_us:
            raise ContractValidationError("invalid correction source range")


UnknownReevaluation = Callable[[tuple[DiarizationSpan, ...]], Sequence[DiarizationSpan]]


def _overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def apply_session_corrections(
    spans: Sequence[DiarizationSpan],
    corrections: Sequence[HumanSpeakerCorrection],
    *,
    reevaluate_unknowns: UnknownReevaluation | None = None,
) -> tuple[DiarizationSpan, ...]:
    """Apply human-confirmed labels only to UNKNOWN source intervals.

    A caller may provide `reevaluate_unknowns` to run the remaining UNKNOWN
    spans through the same *session-local* acoustic pipeline. The callback is
    not allowed to change any originally non-UNKNOWN span or create a third
    session speaker.
    """

    current = tuple(sorted(spans, key=lambda span: (span.start_us, span.end_us, span.span_id)))
    for correction in sorted(corrections, key=lambda item: (item.source_start_us, item.source_end_us, item.correction_id)):
        affected = [span for span in current if _overlap(span.start_us, span.end_us, correction.source_start_us, correction.source_end_us)]
        if not affected:
            raise CorrectionConflictError("correction does not intersect an UNKNOWN span")
        if any(span.speaker_id != "UNKNOWN" for span in affected):
            raise CorrectionConflictError("correction cannot overwrite attributed, overlap, or non-speech span")
        replacement: list[DiarizationSpan] = []
        for span in current:
            if not _overlap(span.start_us, span.end_us, correction.source_start_us, correction.source_end_us):
                replacement.append(span)
                continue
            start = max(span.start_us, correction.source_start_us)
            end = min(span.end_us, correction.source_end_us)
            if span.start_us < start:
                replacement.append(DiarizationSpan(f"{span.span_id}:before", span.start_us, start, "UNKNOWN", span.attribution_status, span.evidence_ids, span.reason_codes))
            replacement.append(DiarizationSpan(
                f"{span.span_id}:human:{correction.correction_id}", start, end,
                correction.confirmed_speaker_id, "HUMAN_CONFIRMED",
                span.evidence_ids + (correction.reviewer_evidence_id,),
                span.reason_codes + ("HUMAN_CONFIRMED_SEGMENT",),
            ))
            if end < span.end_us:
                replacement.append(DiarizationSpan(f"{span.span_id}:after", end, span.end_us, "UNKNOWN", span.attribution_status, span.evidence_ids, span.reason_codes))
        current = tuple(sorted(replacement, key=lambda span: (span.start_us, span.end_us, span.span_id)))
    if reevaluate_unknowns is None:
        return current
    reevaluated = tuple(reevaluate_unknowns(current))
    original_non_unknown = {(span.start_us, span.end_us, span.speaker_id) for span in current if span.speaker_id != "UNKNOWN"}
    output_non_unknown = {(span.start_us, span.end_us, span.speaker_id) for span in reevaluated if span.speaker_id != "UNKNOWN"}
    if not original_non_unknown.issubset(output_non_unknown):
        raise CorrectionConflictError("reevaluation changed a fixed session span")
    if any(span.speaker_id not in {"SPEAKER_00", "SPEAKER_01", "UNKNOWN", "OVERLAP", "OTHER", "NON_SPEECH"} for span in reevaluated):
        raise CorrectionConflictError("reevaluation created an unsupported speaker label")
    return reevaluated
