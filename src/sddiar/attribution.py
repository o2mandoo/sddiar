"""Conservative source-time word to speaker attribution.

This module intentionally contains no model/runtime dependency.  It accepts the
small immutable objects from :mod:`sddiar.contracts` (and mappings in tests) and
uses interval evidence, never a word midpoint, to make an assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class WordMappingConfig:
    word_material_speaker_coverage: float = 0.20
    word_min_dominant_coverage: float = 0.80
    word_max_unknown_coverage: float = 0.20
    boundary_guard_us: int = 0


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _overlap(a: int, b: int, c: int, d: int) -> int:
    return max(0, min(b, d) - max(a, c))


def _config(config: Any) -> Any:
    return config if config is not None else WordMappingConfig()


def _build_attributed(word: Any, *, speaker_id: str, status: str,
                      evidence: Sequence[Any], coverage: float | None,
                      competing: float | None, reasons: Sequence[str]) -> Any:
    fields = {
        "word_id": _get(word, "word_id"), "start_us": _get(word, "start_us"),
        "end_us": _get(word, "end_us"), "text": _get(word, "text"),
        "confidence": _get(word, "confidence"),
        "source_chunk_id": _get(word, "source_chunk_id"),
        "speaker_id": speaker_id, "attribution_status": status,
        "supporting_span_ids": tuple(_get(s, "span_id", "") for s in evidence
                                      if _get(s, "span_id", None) is not None),
        "speaker_coverage_ratio": coverage,
        "competing_speaker_coverage_ratio": competing,
        "reason_codes": tuple(reasons),
    }
    # Import late: contracts is deliberately owned by the contract lane.
    try:
        from .contracts import AttributedWord
        return AttributedWord(**fields)
    except ImportError:  # permits isolated unit testing before contracts land
        from dataclasses import make_dataclass
        return make_dataclass("AttributedWord", [(k, type(v)) for k, v in fields.items()])(**fields)


def map_word(word: Any, spans: Sequence[Any], provenance: Any,
             config: Any = None) -> Any:
    """Map one word using whole-interval evidence and safety-first tie rules."""
    cfg = _config(config)
    start, end = _get(word, "start_us"), _get(word, "end_us")
    malformed = not isinstance(start, int) or not isinstance(end, int) or end <= start
    evidence = [s for s in spans if _overlap(start, end, _get(s, "start_us", 0),
                                               _get(s, "end_us", 0)) > 0] if not malformed else []
    if malformed or bool(_get(provenance, "crosses_timewarp_boundary", False)):
        return _build_attributed(word, speaker_id="UNKNOWN", status="UNKNOWN_TIMEWARP_BOUNDARY",
                                 evidence=evidence, coverage=None, competing=None,
                                 reasons=("UNKNOWN_TIMEWARP_BOUNDARY",))
    duration = end - start
    overlap_us = 0
    for span in evidence:
        label = _get(span, "speaker_id")
        status = _get(span, "attribution_status", "")
        if label == "OVERLAP" or status in {"OVERLAP", "OVERLAP_UNATTRIBUTED"} or _get(span, "protected_overlap", False):
            overlap_us = max(overlap_us, _overlap(start, end, _get(span, "start_us"), _get(span, "end_us")))
    if overlap_us:
        return _build_attributed(word, speaker_id="OVERLAP", status="OVERLAP_UNATTRIBUTED",
                                 evidence=evidence, coverage=None, competing=None,
                                 reasons=("OVERLAP_UNATTRIBUTED",))
    coverage: dict[str, int] = {}
    for span in evidence:
        label = _get(span, "speaker_id")
        if label:
            coverage[label] = coverage.get(label, 0) + _overlap(start, end, _get(span, "start_us"), _get(span, "end_us"))
    ratios = {k: v / duration for k, v in coverage.items()}
    speakers = {k: v for k, v in ratios.items() if k in {"SPEAKER_00", "SPEAKER_01"}}
    material = [k for k, v in speakers.items() if v >= float(_get(cfg, "word_material_speaker_coverage", .2))]
    if len(material) >= 2:
        return _build_attributed(word, speaker_id="UNKNOWN", status="UNKNOWN_BOUNDARY", evidence=evidence,
                                 coverage=max(speakers.values()), competing=min(speakers.values()),
                                 reasons=("UNKNOWN_BOUNDARY",))
    if speakers:
        best = max(speakers, key=lambda k: (speakers[k], "SPEAKER_00" == k))
        best_ratio = speakers[best]
        competing = max((v for k, v in speakers.items() if k != best), default=0.0)
        unknown_ratio = ratios.get("UNKNOWN", 0.0) + ratios.get("NON_SPEECH", 0.0)
        if best_ratio >= float(_get(cfg, "word_min_dominant_coverage", .8)) and unknown_ratio <= float(_get(cfg, "word_max_unknown_coverage", .2)):
            return _build_attributed(word, speaker_id=best, status="ASSIGNED", evidence=evidence,
                                     coverage=best_ratio, competing=competing, reasons=("ASSIGNED",))
    if ratios.get("OTHER", 0.0) > 0 and not speakers:
        return _build_attributed(word, speaker_id="OTHER", status="OTHER", evidence=evidence,
                                 coverage=ratios["OTHER"], competing=None, reasons=("OTHER",))
    return _build_attributed(word, speaker_id="UNKNOWN", status="UNKNOWN_INSUFFICIENT_EVIDENCE",
                             evidence=evidence, coverage=max(speakers.values(), default=0.0),
                             competing=None, reasons=("UNKNOWN_INSUFFICIENT_EVIDENCE",))


def map_words(timeline: Any, spans: Sequence[Any], config: Any = None) -> tuple[Any, ...]:
    words = _get(timeline, "words", ())
    prov = _get(timeline, "provenance_by_word_id", {})
    result = []
    for word in words:
        p = prov.get(_get(word, "word_id")) if hasattr(prov, "get") else None
        result.append(map_word(word, spans, p, config))
    return tuple(result)


class WordSpeakerMapper:
    def __init__(self, config: Any = None): self.config = _config(config)
    def map_word(self, word: Any, spans: Sequence[Any], provenance: Any) -> Any:
        return map_word(word, spans, provenance, self.config)
    def map_words(self, timeline: Any, spans: Sequence[Any]) -> tuple[Any, ...]:
        return map_words(timeline, spans, self.config)


# Descriptive aliases kept for adapter callers that prefer verb-first names.
attribute_word = map_word
attribute_words = map_words
map_word_timeline = map_words
