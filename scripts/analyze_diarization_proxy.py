#!/usr/bin/env python3
"""Evaluate a redacted two-speaker diarization result against Clova timings.

The evaluator consumes timing only.  It deliberately ignores transcript text,
input paths, display names, and all other source metadata.  Clova's export is
not a speech activity annotation: a turn's end is the next turn's start, and
therefore the interval includes any silence between turns.  The timeline
intersection reported here is consequently a proxy coverage statistic, not
VAD recall, DER, or a speaker-attribution ground truth score.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


UNKNOWN_LABELS = frozenset({"UNKNOWN", "REF_UNKNOWN"})
OVERLAP_LABELS = frozenset({"OVERLAP"})
ASSIGNED_LABELS = frozenset()
_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("under_1s", 0, 1_000_000),
    ("1s_to_3s", 1_000_000, 3_000_000),
    ("3s_to_10s", 3_000_000, 10_000_000),
    ("10s_plus", 10_000_000, None),
)


class ProxyAnalysisError(ValueError):
    """Raised for malformed or non-two-speaker proxy inputs."""


def _as_us(row: Mapping[str, Any], begin: str = "start", end: str = "end") -> tuple[int, int]:
    """Read ``*_us`` first, then seconds, without retaining the original row."""
    start_key, end_key = f"{begin}_us", f"{end}_us"
    if start_key in row or end_key in row:
        if start_key not in row or end_key not in row:
            raise ProxyAnalysisError("each interval needs both start and end")
        start, finish = int(row[start_key]), int(row[end_key])
    else:
        start_value = row.get(begin + "_sec", row.get(begin))
        end_value = row.get(end + "_sec", row.get(end))
        if start_value is None or end_value is None:
            raise ProxyAnalysisError("each interval needs both start and end")
        start, finish = round(float(start_value) * 1_000_000), round(float(end_value) * 1_000_000)
    if start < 0 or finish <= start:
        raise ProxyAnalysisError("intervals must have 0 <= start < end")
    return start, finish


def _rows(value: Any, *keys: str) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        rows = None
        for key in keys:
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        if rows is None:
            raise ProxyAnalysisError("JSON object has no interval array")
    else:
        raise ProxyAnalysisError("expected an interval array or object")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ProxyAnalysisError("interval array contains a non-object")
    return list(rows)


def parse_reference(value: Any) -> tuple[dict[str, Any], ...]:
    """Parse pseudonymous Clova timing JSON into timing-only rows."""
    rows = _rows(value, "turns", "segments")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        speaker = row.get("speaker_id", row.get("speaker"))
        if speaker is None:
            raise ProxyAnalysisError("reference turn has no pseudonymous speaker_id")
        start, end = _as_us(row)
        parsed.append({"speaker_id": str(speaker), "start_us": start, "end_us": end})
    parsed.sort(key=lambda item: (item["start_us"], item["end_us"], item["speaker_id"]))
    speakers = {item["speaker_id"] for item in parsed if item["speaker_id"] not in UNKNOWN_LABELS}
    if len(speakers) != 2:
        raise ProxyAnalysisError(f"expected exactly two reference speakers, got {len(speakers)}")
    return tuple(parsed)


def parse_spans(value: Any) -> tuple[dict[str, Any], ...]:
    """Parse redacted system spans, dropping all fields except timing/label."""
    rows = _rows(value, "spans", "segments", "diarization")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        speaker = row.get("speaker_id", row.get("speaker"))
        if speaker is None:
            raise ProxyAnalysisError("system span has no speaker_id")
        start, end = _as_us(row)
        parsed.append({"speaker_id": str(speaker), "start_us": start, "end_us": end})
    parsed.sort(key=lambda item: (item["start_us"], item["end_us"], item["speaker_id"]))
    return tuple(parsed)


def _overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _interval_union(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    total = 0
    current: tuple[int, int] | None = None
    for start, end in ordered:
        if current is None:
            current = (start, end)
        elif start <= current[1]:
            current = (current[0], max(current[1], end))
        else:
            total += current[1] - current[0]
            current = (start, end)
    if current is not None:
        total += current[1] - current[0]
    return total


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _system_labels(spans: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    labels = {str(row["speaker_id"]) for row in spans}
    return tuple(sorted(labels - UNKNOWN_LABELS - OVERLAP_LABELS))


def _reference_labels(reference: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(row["speaker_id"]) for row in reference if str(row["speaker_id"]) not in UNKNOWN_LABELS}))


def _mapping(reference: Sequence[Mapping[str, Any]], spans: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], int, dict[str, dict[str, int]]]:
    ref_labels = _reference_labels(reference)
    sys_labels = _system_labels(spans)
    matrix: dict[str, dict[str, int]] = {system: {ref: 0 for ref in ref_labels} for system in sys_labels}
    for span in spans:
        system = str(span["speaker_id"])
        if system not in matrix:
            continue
        for turn in reference:
            ref = str(turn["speaker_id"])
            if ref in ref_labels:
                matrix[system][ref] += _overlap(int(span["start_us"]), int(span["end_us"]), int(turn["start_us"]), int(turn["end_us"]))
    # A merged/partly deferred result can expose only one assigned label.  An
    # extra label is rejected: this report is explicitly a two-speaker proxy.
    if len(sys_labels) > 2:
        raise ProxyAnalysisError("expected at most two assigned system speakers")
    best: dict[str, str] = {}
    best_score = -1
    for candidate_refs in itertools.permutations(ref_labels, len(sys_labels)):
        candidate = dict(zip(sys_labels, candidate_refs))
        score = sum(matrix[system][ref] for system, ref in candidate.items())
        if score > best_score or (score == best_score and tuple(candidate.items()) < tuple(best.items())):
            best, best_score = candidate, score
    return best, max(0, best_score), matrix


def _classification_durations(turn: Mapping[str, Any], spans: Sequence[Mapping[str, Any]], mapping: Mapping[str, str]) -> dict[str, int]:
    """Classify a reference turn by atomic intervals, avoiding double counts."""
    start, end = int(turn["start_us"]), int(turn["end_us"])
    boundaries = {start, end}
    clipped: list[tuple[int, int, str]] = []
    for span in spans:
        span_start, span_end = int(span["start_us"]), int(span["end_us"])
        if _overlap(start, end, span_start, span_end) <= 0:
            continue
        left, right = max(start, span_start), min(end, span_end)
        boundaries.update((left, right))
        clipped.append((left, right, str(span["speaker_id"])))
    points = sorted(boundaries)
    out = defaultdict(int)
    for left, right in zip(points, points[1:]):
        if right <= left:
            continue
        active = [label for span_start, span_end, label in clipped if span_start <= left and span_end >= right]
        # Redacted diarization spans are expected to be disjoint.  If malformed
        # overlapping labels occur, count the cell as OVERLAP instead of giving
        # the interval twice to a speaker.
        label = active[0] if len(active) == 1 else ("OVERLAP" if len(active) > 1 else None)
        length = right - left
        if label is None:
            out["uncovered_duration_us"] += length
        elif label in UNKNOWN_LABELS:
            out["unknown_duration_us"] += length
        elif label in OVERLAP_LABELS:
            out["overlap_duration_us"] += length
        else:
            out["assigned_duration_us"] += length
            if mapping.get(label) == str(turn["speaker_id"]):
                out["correct_duration_us"] += length
            else:
                out["wrong_duration_us"] += length
    return dict(out)


def _turn_status(turn: Mapping[str, Any], spans: Sequence[Mapping[str, Any]], mapping: Mapping[str, str]) -> str:
    votes: dict[str, int] = defaultdict(int)
    for span in spans:
        label = str(span["speaker_id"])
        if label in UNKNOWN_LABELS or label in OVERLAP_LABELS or label not in mapping:
            continue
        votes[label] += _overlap(int(turn["start_us"]), int(turn["end_us"]), int(span["start_us"]), int(span["end_us"]))
    if not votes or max(votes.values()) <= 0:
        return "uncovered"
    # Stable tie break keeps synthetic and production reports reproducible.
    predicted = max(sorted(votes), key=lambda label: votes[label])
    return "correct" if mapping.get(predicted) == str(turn["speaker_id"]) else "wrong"


def _metric_row(duration: Mapping[str, int]) -> dict[str, Any]:
    result = {key: int(duration.get(key, 0)) for key in (
        "reference_duration_us", "assigned_duration_us", "correct_duration_us",
        "wrong_duration_us", "unknown_duration_us", "overlap_duration_us",
        "uncovered_duration_us",
    )}
    result["assigned_accuracy"] = _ratio(result["correct_duration_us"], result["assigned_duration_us"])
    result["assigned_rate"] = _ratio(result["assigned_duration_us"], result["reference_duration_us"])
    result["unknown_rate"] = _ratio(result["unknown_duration_us"], result["assigned_duration_us"] + result["unknown_duration_us"])
    return result


def _bucket_report(reference: Sequence[Mapping[str, Any]], spans: Sequence[Mapping[str, Any]], mapping: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for label, lower, upper in _BUCKETS:
        rows = [turn for turn in reference if lower <= int(turn["end_us"]) - int(turn["start_us"]) < (upper if upper is not None else 2**63)]
        duration = defaultdict(int)
        for turn in rows:
            turn_length = int(turn["end_us"]) - int(turn["start_us"])
            duration["reference_duration_us"] += turn_length
            for key, value in _classification_durations(turn, spans, mapping).items():
                duration[key] += value
        report = _metric_row(duration)
        report["turn_count"] = len(rows)
        buckets[label] = report
    return buckets


def analyze(reference: Any, spans: Any) -> dict[str, Any]:
    """Return a JSON-safe redacted proxy analysis."""
    ref = parse_reference(reference) if not (isinstance(reference, tuple) and reference and "speaker_id" in reference[0]) else reference
    system = parse_spans(spans) if not (isinstance(spans, tuple) and (not spans or "speaker_id" in spans[0])) else spans
    ref_labels = _reference_labels(ref)
    mapping, mapping_score, matrix = _mapping(ref, system)
    ref_duration = sum(int(turn["end_us"]) - int(turn["start_us"]) for turn in ref)
    totals = defaultdict(int)
    per_speaker: dict[str, dict[str, Any]] = {}
    turn_counts = {"correct": 0, "wrong": 0, "uncovered": 0}
    per_turn: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(ref):
        turn_length = int(turn["end_us"]) - int(turn["start_us"])
        totals["reference_duration_us"] += turn_length
        classified = _classification_durations(turn, system, mapping)
        for key, value in classified.items():
            totals[key] += value
        speaker = str(turn["speaker_id"])
        speaker_totals = per_speaker.setdefault(speaker, defaultdict(int))
        speaker_totals["reference_duration_us"] += turn_length
        for key, value in classified.items():
            speaker_totals[key] += value
        status = _turn_status(turn, system, mapping)
        turn_counts[status] += 1
        per_turn.append({"turn_index": turn_index, "reference_speaker_id": speaker, "status": status})
    overall = _metric_row(totals)
    detected_intervals = [(int(span["start_us"]), int(span["end_us"])) for span in system]
    reference_intervals = [(int(turn["start_us"]), int(turn["end_us"])) for turn in ref]
    intersection = _interval_union((max(s, rs), min(e, re)) for s, e in detected_intervals for rs, re in reference_intervals)
    detected_in_reference = _interval_union((max(s, rs), min(e, re)) for s, e in detected_intervals for rs, re in reference_intervals)
    detected_duration = _interval_union(detected_intervals)
    unknown_intervals = [(int(span["start_us"]), int(span["end_us"])) for span in system if str(span["speaker_id"]) in UNKNOWN_LABELS]
    unknown_in_reference = _interval_union((max(s, rs), min(e, re)) for s, e in unknown_intervals for rs, re in reference_intervals)
    overlap_intervals = [(int(span["start_us"]), int(span["end_us"])) for span in system if str(span["speaker_id"]) in OVERLAP_LABELS]
    overlap_in_reference = _interval_union((max(s, rs), min(e, re)) for s, e in overlap_intervals for rs, re in reference_intervals)
    overall["unknown_rate_within_system_detected_speech"] = _ratio(unknown_in_reference, detected_in_reference)
    overall["overlap_rate_within_system_detected_speech"] = _ratio(overlap_in_reference, detected_in_reference)
    overall["system_detected_speech_duration_us"] = detected_duration
    overall["system_detected_speech_in_reference_us"] = detected_in_reference
    fairness_sources = {speaker: _metric_row(values) for speaker, values in per_speaker.items()}
    fairness = {}
    for key in ("assigned_accuracy", "assigned_rate", "unknown_rate"):
        values = [float(row[key]) for row in fairness_sources.values()]
        fairness[f"{key}_max_minus_min"] = (max(values) - min(values)) if values else 0.0
    return {
        "schema": "diarization_proxy_analysis_v1",
        "redaction": {"source_names": "omitted", "transcript_text": "omitted", "audio": "omitted"},
        "warnings": [
            "Clova turn end equals the next turn start; each reference interval can include silence.",
            "Timeline intersection is a Clova-timing proxy and is not VAD recall or DER.",
            "UNKNOWN means an explicit UNKNOWN system span; uncovered is reported separately.",
        ],
        "reference_speakers": list(ref_labels),
        "mapping": mapping,
        "mapping_intersection_duration_us": mapping_score,
        "confusion_matrix_duration_us": matrix,
        "overall": overall,
        "per_reference_speaker": {speaker: _metric_row(values) for speaker, values in sorted(per_speaker.items())},
        "timeline_intersection": {
            "duration_us": intersection,
            "ratio_of_reference_timeline": _ratio(intersection, ref_duration),
            "reference_timeline_duration_us": ref_duration,
            "system_detected_speech_in_reference_us": detected_in_reference,
        },
        "turns": {**turn_counts, "total": len(ref), "coverage": _ratio(turn_counts["correct"] + turn_counts["wrong"], len(ref)), "details": per_turn},
        "duration_buckets": _bucket_report(ref, system, mapping),
        "fairness_gaps": fairness,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze redacted Clova timing and diarization spans")
    parser.add_argument("reference", type=Path, help="pseudonymous Clova timing JSON")
    parser.add_argument("spans", type=Path, help="redacted diarization span JSON")
    parser.add_argument("--indent", type=int, default=None, help="pretty-print JSON with this indentation")
    args = parser.parse_args(argv)
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        spans = json.loads(args.spans.read_text(encoding="utf-8"))
        result = analyze(reference, spans)
    except (OSError, UnicodeError, json.JSONDecodeError, ProxyAnalysisError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
