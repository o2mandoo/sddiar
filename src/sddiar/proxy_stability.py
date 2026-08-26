"""Privacy-safe stability measurements for label-free/weak-label proxies.

This module consumes already redacted span aggregates.  It never returns the
input IDs, paths, transcript fields, or embedding vectors.  Speaker labels are
used internally only to compute permutation-invariant metrics and are exposed
as one-way digests when an input fingerprint is useful.

The report is deliberately evidence for review, not a release gate.  In
particular, a Clova score (when supplied) is a separate observation and cannot
select a candidate or authorize a release.
"""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "proxy_stability_report_v1"
UNKNOWN = frozenset({"UNKNOWN", "REF_UNKNOWN", "UNKNOWN_SHORT", "UNASSIGNED", "NONE", ""})
OVERLAP = frozenset({"OVERLAP"})
MAX_SPANS = 10_000
MAX_VARIANTS = 32
MAX_TOTAL_SPANS = 50_000
MAX_INPUT_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 64 * 1024 * 1024
MAX_ASSIGNED_LABELS = 2


class ProxyStabilityError(ValueError):
    """Malformed or unsupported redacted proxy input."""


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _hash(value: Any) -> str:
    # Hash only a canonical scalar representation; never place the value in a
    # report or error string.
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8", "surrogatepass")).hexdigest()


def _time(row: Mapping[str, Any], name: str) -> int:
    key = f"{name}_us"
    scale = 1.0
    if key in row:
        value = row[key]
    elif f"{name}_sec" in row:
        value = row[f"{name}_sec"]
        scale = 1_000_000.0
    elif name in row:
        value = row[name]
        # A bare start/end is conventionally seconds in JSON proxy exports.
        if isinstance(value, float):
            scale = 1_000_000.0
        elif not isinstance(value, bool):
            try:
                if abs(float(value)) < 100_000:
                    scale = 1_000_000.0
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProxyStabilityError("span timing is invalid") from exc
    else:
        raise ProxyStabilityError("span timing is incomplete")
    if isinstance(value, bool):
        raise ProxyStabilityError("span timing is invalid")
    try:
        numeric = float(value) * scale
        if not math.isfinite(numeric):
            raise ProxyStabilityError("span timing is invalid")
        result = int(round(numeric))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProxyStabilityError("span timing is invalid") from exc
    return result


def _row_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, tuple):
        rows = list(value)
    elif isinstance(value, Mapping):
        rows = None
        # The first-level names cover our canonical format and common evidence
        # exports.  We intentionally do not recursively walk arbitrary input,
        # which could accidentally ingest source metadata.
        for key in ("spans", "segments", "diarization", "assignments", "chunks", "tracklets", "items", "embedding_aggregate"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            return []
    else:
        raise ProxyStabilityError("spans must be an array")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ProxyStabilityError("span array contains a non-object")
    return list(rows)


def _extract_spans(value: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(value, Mapping):
        # Accept an aggregate wrapper without retaining its other fields.
        for key in ("spans", "segments", "diarization", "assignments", "chunks", "tracklets", "items", "embedding_aggregate"):
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        else:
            rows = []
    else:
        rows = _row_list(value)
    if len(rows) > MAX_SPANS:
        raise ProxyStabilityError("span resource bound exceeded")
    parsed: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for index, row in enumerate(rows):
        try:
            start, end = _time(row, "start"), _time(row, "end")
        except ProxyStabilityError:
            # Existing evidence JSON may contain only aggregate metrics.  It is
            # valid input, but has no span-level metrics to score.
            if not any(key in row for key in ("start", "start_us", "start_sec")):
                continue
            raise
        if start < 0 or end <= start:
            raise ProxyStabilityError("span intervals must satisfy 0 <= start < end")
        label = row.get("speaker_id", row.get("speaker", row.get("label", row.get("attribution"))))
        if label is None:
            label = "UNKNOWN"
        identity = row.get("span_id", row.get("tracklet_id", row.get("chunk_id", row.get("id", index))))
        identity = str(identity)
        if identity in seen_keys:
            raise ProxyStabilityError("duplicate span identity")
        seen_keys.add(identity)
        central = row.get("central", row.get("is_central", False))
        if type(central) is not bool:
            raise ProxyStabilityError("span central flag must be boolean")
        # Only these scalar fields survive parsing.  In particular, text,
        # paths, model metadata, and vectors are dropped immediately.
        parsed.append({
            "key": identity, "start_us": start, "end_us": end,
            "label": str(label), "central": central,
        })
    parsed.sort(key=lambda item: (item["start_us"], item["end_us"], item["key"]))
    for previous, current in zip(parsed, parsed[1:]):
        if current["start_us"] < previous["end_us"]:
            raise ProxyStabilityError("overlapping spans are not supported")
    assigned_labels = {row["label"] for row in parsed if _label_kind(str(row["label"])) == "assigned"}
    if len(assigned_labels) > MAX_ASSIGNED_LABELS:
        raise ProxyStabilityError("assigned speaker label resource bound exceeded")
    return tuple(parsed)


def _normalize_state(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    aliases = {
        "H1": "H1", "H1_CONFIRMED": "H1",
        "H2": "H2", "H2_CONFIRMED": "H2",
        "UNCERTAIN": "UNCERTAIN_1_OR_2",
        "UNCERTAIN_1_OR_2": "UNCERTAIN_1_OR_2",
    }
    return aliases.get(value)


def _extract_state(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("state", "decision", "hypothesis_state", "selected_state"):
        candidate = value.get(key)
        if isinstance(candidate, str):
            return _normalize_state(candidate)
    selected = value.get("selected_hypothesis")
    if isinstance(selected, Mapping) and selected.get("k") in (1, 2):
        return f"H{selected['k']}"
    return None


def _unwrap_variants(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        for key in ("variants", "arms", "candidates", "runs", "outputs"):
            if isinstance(value.get(key), list):
                return tuple(value[key])
        return (value,)
    if isinstance(value, (list, tuple)):
        # A direct list of span rows denotes one variant; a list of wrappers
        # denotes several variants.
        if value and all(isinstance(row, Mapping) for row in value) and any(
            key in value[0] for key in ("start", "start_us", "start_sec")
        ):
            return ({"spans": list(value)},)
        return tuple(value)
    raise ProxyStabilityError("variants must be an array or object")


def _label_kind(label: str) -> str:
    upper = label.upper()
    if upper in UNKNOWN:
        return "unknown"
    if upper in OVERLAP:
        return "overlap"
    return "assigned"


def _union(intervals: Iterable[tuple[int, int]]) -> int:
    current: tuple[int, int] | None = None
    total = 0
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if current is None:
            current = (start, end)
        elif start <= current[1]:
            current = (current[0], max(current[1], end))
        else:
            total += current[1] - current[0]
            current = (start, end)
    return total + (current[1] - current[0] if current else 0)


def _speech_intersection(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        current_left, current_right = left[left_index], right[right_index]
        total += max(0, min(current_left["end_us"], current_right["end_us"]) -
                     max(current_left["start_us"], current_right["start_us"]))
        if current_left["end_us"] <= current_right["end_us"]:
            left_index += 1
        if current_right["end_us"] <= current_left["end_us"]:
            right_index += 1
    return total


def _central_retention(
    canonical: Sequence[Mapping[str, Any]], variant: Sequence[Mapping[str, Any]], mapping: Mapping[str, str],
) -> float | None:
    left_index = right_index = 0
    total = sum(
        row["end_us"] - row["start_us"]
        for row in canonical
        if (row.get("central") or str(row["key"]).startswith("chunk")) and _assigned(row)
    )
    retained = 0
    while left_index < len(canonical) and right_index < len(variant):
        left, right = canonical[left_index], variant[right_index]
        start, end = max(left["start_us"], right["start_us"]), min(left["end_us"], right["end_us"])
        if end > start and (left.get("central") or str(left["key"]).startswith("chunk")) and _assigned(left):
            duration = end - start
            if _assigned(right) and mapping.get(str(right["label"])) == str(left["label"]):
                retained += duration
        if left["end_us"] <= right["end_us"]:
            left_index += 1
        if right["end_us"] <= left["end_us"]:
            right_index += 1
    return _ratio(retained, total) if total else None


def _pair_durations(
    canonical: Sequence[Mapping[str, Any]], variant: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    """Sweep two disjoint timelines once and return label-pair durations."""
    pairs: dict[tuple[str, str], int] = {}
    left_index = right_index = 0
    while left_index < len(canonical) and right_index < len(variant):
        left, right = canonical[left_index], variant[right_index]
        start = max(left["start_us"], right["start_us"])
        end = min(left["end_us"], right["end_us"])
        if end > start:
            key = (str(left["label"]), str(right["label"]))
            pairs[key] = pairs.get(key, 0) + end - start
        if left["end_us"] <= right["end_us"]:
            left_index += 1
        if right["end_us"] <= left["end_us"]:
            right_index += 1
    return pairs


def _assigned(row: Mapping[str, Any]) -> bool:
    return _label_kind(str(row["label"])) == "assigned"


def _permutation(pairs: Mapping[tuple[str, str], int]) -> dict[str, str]:
    """Choose the best 1- or 2-speaker mapping without factorial search."""
    matrix = {(variant, canonical): duration for (canonical, variant), duration in pairs.items()
              if _label_kind(canonical) == "assigned" and _label_kind(variant) == "assigned"}
    variants = sorted({variant for variant, _ in matrix})
    references = sorted({canonical for _, canonical in matrix})
    if len(variants) > MAX_ASSIGNED_LABELS or len(references) > MAX_ASSIGNED_LABELS:
        raise ProxyStabilityError("assigned speaker label resource bound exceeded")
    if not variants or not references:
        return {}
    candidates: list[dict[str, str]] = []
    if len(variants) == 1:
        candidates = [{variants[0]: reference} for reference in references]
    elif len(variants) == 2 and len(references) == 2:
        candidates = [
            {variants[0]: references[0], variants[1]: references[1]},
            {variants[0]: references[1], variants[1]: references[0]},
        ]
    else:
        # A one-reference merge is the only remaining valid 1-2-speaker case.
        candidates = [{variant: references[0] for variant in variants}]
    def value(candidate: Mapping[str, str]) -> int:
        return sum(duration for (variant, reference), duration in matrix.items()
                   if candidate.get(variant) == reference)
    best_value = max(value(candidate) for candidate in candidates)
    return min((candidate for candidate in candidates if value(candidate) == best_value),
               key=lambda candidate: tuple(sorted(candidate.items())))


def _coassigned(pairs: Mapping[tuple[str, str], int], mapping: Mapping[str, str]) -> float | None:
    matched = sum(duration for (canonical, variant), duration in pairs.items()
                  if _assigned({"label": canonical}) and _assigned({"label": variant})
                  and mapping.get(variant) == canonical)
    total = sum(duration for (canonical, variant), duration in pairs.items()
                if _assigned({"label": canonical}) and _assigned({"label": variant}))
    return _ratio(matched, total) if total else None


def _flip_rate(
    pairs: Mapping[tuple[str, str], int],
    mapping: Mapping[str, str],
    canonical_assigned_duration: int,
) -> float | None:
    """Source-time mismatch after permutation; missing variant time is a flip."""
    retained = sum(duration for (canonical, variant), duration in pairs.items()
                   if _assigned({"label": canonical}) and mapping.get(variant) == canonical)
    return (_ratio(canonical_assigned_duration - retained, canonical_assigned_duration)
            if canonical_assigned_duration else None)


def _unknown_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    total = sum(row["end_us"] - row["start_us"] for row in rows)
    unknown = sum(row["end_us"] - row["start_us"] for row in rows if _label_kind(str(row["label"])) == "unknown")
    return _ratio(unknown, total)


def _boundaries(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    if not rows:
        return []
    points = [row["start_us"] for row in rows[1:]] + [row["end_us"] for row in rows[:-1]]
    return sorted(set(points))


def _boundary_f1(canonical: Sequence[Mapping[str, Any]], variant: Sequence[Mapping[str, Any]], tolerance_us: int) -> float:
    reference = _boundaries(canonical)
    predicted = _boundaries(variant)
    if not reference and not predicted:
        return 1.0
    tp = 0
    reference_index = predicted_index = 0
    while reference_index < len(reference) and predicted_index < len(predicted):
        target, point = reference[reference_index], predicted[predicted_index]
        if abs(point - target) <= tolerance_us:
            tp += 1
            reference_index += 1
            predicted_index += 1
        elif point < target - tolerance_us:
            predicted_index += 1
        else:
            reference_index += 1
    return _ratio(2 * tp, len(reference) + len(predicted))


def _metric(name: str, values: Sequence[float | None]) -> dict[str, Any]:
    evaluated = [value for value in values if value is not None]
    return {
        "mean": _ratio(sum(evaluated), len(evaluated)) if evaluated else None,
        "min": min(evaluated) if evaluated else None,
        "max": max(evaluated) if evaluated else None,
        "per_variant": list(values),
        "evaluated_count": len(evaluated),
        "variant_count": len(values),
        "name": name,
    }


def audit(canonical: Any, variants: Any = None, *, clova: Any = None, boundary_tolerance_us: int = 250_000) -> dict[str, Any]:
    """Return a deterministic, redacted stability report.

    ``canonical`` and each variant may be a span array or a wrapper containing
    one.  An evidence object containing ``canonical``/``variants`` is accepted
    as a convenience.  Existing metric-only evidence produces a valid empty
    REVIEW_REQUIRED report rather than being mistaken for span evidence.
    """
    if boundary_tolerance_us < 0:
        raise ProxyStabilityError("boundary tolerance must be non-negative")
    if isinstance(canonical, Mapping) and ("canonical" in canonical or "canonical_spans" in canonical):
        if variants is None:
            variants = canonical.get("variants", canonical.get("arms", canonical.get("candidates", canonical.get("variant_spans", []))))
        clova = canonical.get("clova", clova)
        canonical_value = canonical.get("canonical", {"spans": canonical.get("canonical_spans", [])})
    else:
        canonical_value = canonical
    base = _extract_spans(canonical_value)
    variant_values = _unwrap_variants(variants)
    if len(variant_values) > MAX_VARIANTS:
        raise ProxyStabilityError("variant resource bound exceeded")
    variant_spans = []
    total_span_count = len(base)
    for value in variant_values:
        parsed = _extract_spans(value)
        total_span_count += len(parsed)
        if total_span_count > MAX_TOTAL_SPANS:
            raise ProxyStabilityError("total span resource bound exceeded")
        variant_spans.append(parsed)
    permutation_rows: list[dict[str, Any]] = []
    coassigned: list[float | None] = []
    flips: list[float | None] = []
    retention: list[float | None] = []
    unknown_deltas: list[float | None] = []
    speech_ious: list[float | None] = []
    boundaries: list[float | None] = []
    central: list[float | None] = []
    state_changes: list[bool | None] = []
    base_state = _extract_state(canonical_value)
    for value, rows in zip(variant_values, variant_spans):
        variant_state = _extract_state(value)
        state_changes.append(
            None if base_state is None or variant_state is None else base_state != variant_state
        )
        if not base or not rows:
            permutation_rows.append({
                "mapped_label_count": 0,
                "mapping_digest": _hash([]),
                "evaluation_status": "NO_CANONICAL_SPANS" if not base else "NO_VARIANT_SPANS",
            })
            coassigned.append(None)
            flips.append(None)
            retention.append(None)
            unknown_deltas.append(None)
            speech_ious.append(None)
            boundaries.append(None)
            central.append(None)
            continue
        pairs = _pair_durations(base, rows)
        mapping = _permutation(pairs)
        canonical_assigned_duration = sum(
            row["end_us"] - row["start_us"] for row in base if _assigned(row)
        )
        joint_assigned_duration = sum(
            duration for (canonical_label, variant_label), duration in pairs.items()
            if _label_kind(canonical_label) == "assigned" and _label_kind(variant_label) == "assigned"
        )
        if canonical_assigned_duration == 0:
            evaluation_status = "NO_ASSIGNED_EVIDENCE"
        elif joint_assigned_duration == 0:
            evaluation_status = "NO_JOINT_ASSIGNED_EVIDENCE"
        else:
            evaluation_status = "EVALUATED"
        # Mapping is represented only by hashed pseudonymous IDs.
        permutation_rows.append({"mapped_label_count": len(mapping), "mapping_digest": _hash(sorted((key, val) for key, val in mapping.items())), "evaluation_status": evaluation_status})
        coassigned.append(_coassigned(pairs, mapping))
        flips.append(_flip_rate(pairs, mapping, canonical_assigned_duration))
        retained_duration = sum(duration for (canonical_label, variant_label), duration in pairs.items()
                                if _label_kind(canonical_label) == "assigned" and mapping.get(variant_label) == canonical_label)
        retention.append(_ratio(retained_duration, canonical_assigned_duration) if canonical_assigned_duration else None)
        unknown_deltas.append(_unknown_rate(rows) - _unknown_rate(base))
        base_speech = _union((r["start_us"], r["end_us"]) for r in base if _label_kind(str(r["label"])) != "unknown")
        variant_speech = _union((r["start_us"], r["end_us"]) for r in rows if _label_kind(str(r["label"])) != "unknown")
        base_speech_rows = [r for r in base if _label_kind(str(r["label"])) != "unknown"]
        variant_speech_rows = [r for r in rows if _label_kind(str(r["label"])) != "unknown"]
        intersection = _speech_intersection(base_speech_rows, variant_speech_rows)
        speech_union = base_speech + variant_speech - intersection
        speech_ious.append(_ratio(intersection, speech_union) if speech_union else None)
        boundaries.append(_boundary_f1(base, rows, boundary_tolerance_us))
        central.append(_central_retention(base, rows, mapping))
    evaluated_state_changes = [value for value in state_changes if value is not None]
    metrics = {
        "co_assigned_agreement": _metric("co_assigned_agreement", coassigned),
        "speaker_flip_rate": _metric("speaker_flip_rate", flips),
        "canonical_attribution_retention": _metric("canonical_attribution_retention", retention),
        "unknown_delta": _metric("unknown_delta", unknown_deltas),
        "speech_iou": _metric("speech_iou", speech_ious),
        "boundary_f1": _metric("boundary_f1", boundaries),
        "h1_h2_state_change": {
            "changed_count": sum(value is True for value in evaluated_state_changes),
            "evaluated_count": len(evaluated_state_changes),
            "variant_count": len(state_changes),
            "rate": (_ratio(sum(value is True for value in evaluated_state_changes), len(evaluated_state_changes))
                     if evaluated_state_changes else None),
            "per_variant": state_changes,
            "canonical_state": base_state,
        },
        "chunk_central_agreement": _metric("chunk_central_agreement", central),
    }
    summary = {name: value["mean"] for name, value in metrics.items() if "mean" in value}
    summary["h1_h2_state_change"] = metrics["h1_h2_state_change"]["rate"]
    result: dict[str, Any] = {
        "schema": SCHEMA, "release_authority": "none", "quality_status": "REVIEW_REQUIRED",
        "scope": "label_free_weak_label_proxy_stability", "canonical": {
            "span_count": len(base), "speaker_count": len({r["label"] for r in base if _assigned(r)}),
            "speech_duration_us": _union((r["start_us"], r["end_us"]) for r in base if _label_kind(str(r["label"])) != "unknown"),
            "unknown_rate": _unknown_rate(base), "label_set_digest": _hash(sorted(str(r["label"]) for r in base)),
            "canonical_digest": _hash({"state": base_state, "spans": base}),
        }, "variant_count": len(variant_spans), "metrics": metrics, "summary": summary, "permutation": permutation_rows,
        "privacy": {"raw_audio": "omitted", "transcript_text": "omitted", "paths": "omitted", "embedding_vectors": "omitted", "speaker_labels": "hashed_only"},
        "authority": {"candidate_selection": "not_allowed", "release_decision": "none"},
        "warnings": ["Proxy stability is label-free/weak-label evidence and is not ground-truth quality.", "quality_status is fixed to REVIEW_REQUIRED; human review and independent evidence remain required."],
    }
    if clova is not None:
        result["clova"] = {"provided": True, "score_digest": _hash(clova), "selection_authority": "none", "candidate_selection_allowed": False, "note": "optional separate observation; no candidate selection authority"}
    return result


analyze = audit


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute redacted proxy stability aggregates")
    parser.add_argument("inputs", nargs="+", help="one evidence JSON, or canonical followed by variant JSON files")
    parser.add_argument("--clova", help="optional Clova aggregate JSON")
    args = parser.parse_args(argv)
    try:
        if len(args.inputs) > MAX_VARIANTS + 1:
            raise ProxyStabilityError("variant resource bound exceeded")
        paths = [Path(path) for path in args.inputs]
        if args.clova:
            paths.append(Path(args.clova))
        sizes = []
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ProxyStabilityError("input must be a regular local file")
            size = path.stat().st_size
            if size > MAX_INPUT_FILE_BYTES:
                raise ProxyStabilityError("input file resource bound exceeded")
            sizes.append(size)
        if sum(sizes) > MAX_TOTAL_INPUT_BYTES:
            raise ProxyStabilityError("total input resource bound exceeded")
        payloads = [json.loads(path.read_bytes()) for path in paths[:len(args.inputs)]]
        canonical = payloads[0]
        variants = payloads[1:]
        clova = json.loads(paths[-1].read_bytes()) if args.clova else None
        print(json.dumps(audit(canonical, variants or None, clova=clova), ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, ProxyStabilityError) as exc:
        # Do not echo file paths, malformed source text, or arbitrary metadata.
        parser.error("unable to audit redacted proxy input")
        return 2


__all__ = [
    "ProxyStabilityError", "SCHEMA", "MAX_SPANS", "MAX_VARIANTS",
    "MAX_TOTAL_SPANS", "MAX_INPUT_FILE_BYTES", "MAX_TOTAL_INPUT_BYTES",
    "audit", "analyze",
]


if __name__ == "__main__":
    raise SystemExit(_main())
