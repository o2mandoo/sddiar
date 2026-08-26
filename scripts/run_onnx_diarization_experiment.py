#!/usr/bin/env python3
"""Run development-only Silero + WeSpeaker CPU diarization and redacted scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sddiar.audio_gain import (  # noqa: E402
    DEFAULT_GLOBAL_GAIN_POLICY,
    GainScaledWavPcmAccessor,
    analyze_pcm16_global_gain,
    disabled_gain_metadata,
    scale_decoded_chunks,
)
from sddiar.benchmark import peak_rss_bytes  # noqa: E402
from sddiar.contracts import EmbeddingRegion, EmbeddingResult, SpeechRegion as ContractSpeechRegion  # noqa: E402
from sddiar.diarization import (  # noqa: E402
    DiarizationConfig, build_tracklets, decode_sequence, evaluate_hypotheses, finalize_sequence,
    refine_recent_states, select_anchor_evidence, speaker_states_from_decision,
)
from sddiar.graph_rescue_experimental import (  # noqa: E402
    GraphRescueConfig,
    build_redacted_receipt,
    rescue_unknowns,
)
from sddiar.media import DecodedAudioChunk, WavPcmAccessor, WavPcmDecoder  # noqa: E402
from sddiar.model_pack import VerifiedArtifact  # noqa: E402
from sddiar.ort_cpu import create_ort_session  # noqa: E402
from sddiar.quality import RuleBasedQualityGate  # noqa: E402
from sddiar.segmentation import RuleEvidenceSegmentation, SegmentationConfig, SegmentationEvidence  # noqa: E402
from sddiar.silero_runtime import SileroOnnxRuntime  # noqa: E402
from sddiar.silero_temporal import SileroTemporalPostprocessor  # noqa: E402
from sddiar.wespeaker_runtime import (  # noqa: E402
    DEVELOPMENT_APPROXIMATION_TAG,
    STRICT_NATIVE_FBANK_TAG,
    WeSpeakerCpuEmbeddingBackend,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ort_session(path: Path, *, threads: int):
    return create_ort_session(path, threads=threads)


def _artifact(path: Path, file_id: str, role: str) -> VerifiedArtifact:
    return VerifiedArtifact(file_id, role, path, _sha256(path), path.stat().st_size)


def _reference(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("turns", value.get("segments", value))
    result = []
    for row in rows:
        start_value = row.get("start_us")
        end_value = row.get("end_us")
        if start_value is None:
            start_value = round(float(row.get("start_sec", row.get("start", 0))) * 1_000_000)
        if end_value is None:
            end_value = round(float(row.get("end_sec", row.get("end", 0))) * 1_000_000)
        start, end = int(start_value), int(end_value)
        if end > start:
            result.append({"start_us": start, "end_us": end, "speaker": str(row.get("speaker_id", row.get("speaker", "REF_UNKNOWN")))})
    return result


def _score(spans: Sequence[Any], reference: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    reference_speakers = sorted({str(row["speaker"]) for row in reference if str(row["speaker"]) != "UNKNOWN"})
    assigned = [span for span in spans if span.speaker_id in {"SPEAKER_00", "SPEAKER_01"}]
    reference_us = sum(int(row["end_us"]) - int(row["start_us"]) for row in reference)
    unknown_spans = [span for span in spans if span.speaker_id == "UNKNOWN"]
    matrix: dict[tuple[str, str], int] = {}
    for span in assigned:
        for row in reference:
            overlap = max(0, min(span.end_us, int(row["end_us"])) - max(span.start_us, int(row["start_us"])))
            matrix[(span.speaker_id, str(row["speaker"]))] = matrix.get((span.speaker_id, str(row["speaker"])), 0) + overlap
    mappings = [dict(zip(("SPEAKER_00", "SPEAKER_01"), reference_speakers)), dict(zip(("SPEAKER_00", "SPEAKER_01"), reversed(reference_speakers)))] if len(reference_speakers) >= 2 else [{}]
    mapping = max(mappings, key=lambda candidate: sum(matrix.get((predicted, actual), 0) for predicted, actual in candidate.items()))
    correct = sum(matrix.get((predicted, actual), 0) for predicted, actual in mapping.items())
    assigned_us = sum(matrix.values())
    per_speaker_accuracy = []
    per_speaker_coverage = []
    per_speaker_end_to_end = []
    for actual in reference_speakers:
        actual_assigned = sum(matrix.get((predicted, actual), 0) for predicted in ("SPEAKER_00", "SPEAKER_01"))
        actual_correct = sum(
            matrix.get((predicted, actual), 0)
            for predicted, mapped in mapping.items()
            if mapped == actual
        )
        actual_reference = sum(
            int(row["end_us"]) - int(row["start_us"])
            for row in reference
            if str(row["speaker"]) == actual
        )
        per_speaker_accuracy.append(actual_correct / max(1, actual_assigned))
        per_speaker_coverage.append(actual_assigned / max(1, actual_reference))
        per_speaker_end_to_end.append(actual_correct / max(1, actual_reference))
    unknown_us = sum(
        max(0, min(span.end_us, int(row["end_us"])) - max(span.start_us, int(row["start_us"])))
        for span in unknown_spans for row in reference
    )
    coverage = assigned_us / max(1, reference_us)
    covered_turns = 0
    correct_turns = 0
    for row in reference:
        votes = {speaker: 0 for speaker in ("SPEAKER_00", "SPEAKER_01")}
        for span in assigned:
            overlap = max(0, min(span.end_us, int(row["end_us"])) - max(span.start_us, int(row["start_us"])))
            votes[span.speaker_id] += overlap
        predicted = max(votes, key=votes.get)
        if votes[predicted] > 0:
            covered_turns += 1
            correct_turns += int(mapping.get(predicted) == str(row["speaker"]))
    return {
        "reference_timeline_coverage": min(1.0, coverage),
        "speaker_mapped_time_accuracy_end_to_end": correct / max(1, reference_us),
        "speaker_accuracy_given_assigned": correct / max(1, assigned_us),
        "unknown_rate_on_output_speech": unknown_us / max(1, unknown_us + assigned_us),
        "turn_coverage": covered_turns / max(1, len(reference)),
        "turn_accuracy_given_covered": correct_turns / max(1, covered_turns),
        "complete_merge": float(len({span.speaker_id for span in assigned}) < 2),
        "worst_speaker_accuracy_given_assigned": min(per_speaker_accuracy, default=0.0),
        "worst_speaker_reference_timeline_coverage": min(per_speaker_coverage, default=0.0),
        "worst_speaker_end_to_end_time_accuracy": min(per_speaker_end_to_end, default=0.0),
        "speaker_accuracy_gap": max(per_speaker_accuracy, default=0.0) - min(per_speaker_accuracy, default=0.0),
        "speaker_coverage_gap": max(per_speaker_coverage, default=0.0) - min(per_speaker_coverage, default=0.0),
    }


def _clip_reference(reference: Sequence[Mapping[str, Any]], start_us: int, end_us: int) -> list[dict[str, Any]]:
    clipped = []
    for row in reference:
        start = max(start_us, int(row["start_us"]))
        end = min(end_us, int(row["end_us"]))
        if end > start:
            clipped.append({"start_us": start, "end_us": end, "speaker": row["speaker"]})
    return clipped


def _process_cpu_seconds() -> float:
    """Return CPU time consumed by this process (user + system)."""
    return time.process_time()


def _elapsed_wall_seconds(started: float) -> float:
    """Return a JSON-safe, non-negative monotonic wall-clock duration."""
    return max(0.0, time.perf_counter() - started)


def _span_timeline_sha256(spans: Sequence[Any]) -> str:
    """Hash only the redacted public timeline fields, deterministically."""

    timeline = [
        [int(span.start_us), int(span.end_us), str(span.speaker_id), str(span.attribution_status)]
        for span in sorted(spans, key=lambda item: (item.start_us, item.end_us, item.speaker_id, item.attribution_status))
    ]
    encoded = json.dumps(timeline, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _embedding_selector_parity(
    tracklets: Sequence[Any],
    baseline: Sequence[EmbeddingResult],
    selected: Sequence[EmbeddingResult],
) -> dict[str, Any]:
    """Aggregate exact/non-exact embedding parity without retaining vectors."""

    baseline_by_tracklet = {item.tracklet_id: item for item in baseline}
    selected_by_tracklet = {item.tracklet_id: item for item in selected}
    groups: dict[str, dict[str, Any]] = {}
    for tracklet in tracklets:
        before = baseline_by_tracklet[tracklet.tracklet_id]
        after = selected_by_tracklet[tracklet.tracklet_id]
        group = groups.setdefault(str(tracklet.kind), {
            "count": 0,
            "exact_result_count": 0,
            "baseline_valid_count": 0,
            "selected_valid_count": 0,
            "validity_changed_count": 0,
            "comparable_vector_count": 0,
            "cosine_sum": 0.0,
            "cosine_min": 1.0,
            "max_abs_delta": 0.0,
        })
        group["count"] += 1
        group["exact_result_count"] += int(before == after)
        group["baseline_valid_count"] += int(before.is_valid)
        group["selected_valid_count"] += int(after.is_valid)
        group["validity_changed_count"] += int(before.is_valid != after.is_valid)
        if before.is_valid and after.is_valid and before.vector is not None and after.vector is not None:
            cosine = sum(left * right for left, right in zip(before.vector, after.vector))
            max_abs = max(abs(left - right) for left, right in zip(before.vector, after.vector))
            group["comparable_vector_count"] += 1
            group["cosine_sum"] += cosine
            group["cosine_min"] = min(group["cosine_min"], cosine)
            group["max_abs_delta"] = max(group["max_abs_delta"], max_abs)
    rendered = []
    for kind, values in sorted(groups.items()):
        comparable = values.pop("comparable_vector_count")
        cosine_sum = values.pop("cosine_sum")
        rendered.append({
            "kind": kind,
            **values,
            "comparable_vector_count": comparable,
            "cosine_mean": cosine_sum / comparable if comparable else None,
            "cosine_min": values["cosine_min"] if comparable else None,
        })
    non_micro = [item for item in rendered if item["kind"] != "MICRO"]
    non_micro_exact = all(item["exact_result_count"] == item["count"] for item in non_micro)
    return {
        "groups": rendered,
        "anchor_support_exact": non_micro_exact,
        "anchor_support_count": sum(item["count"] for item in non_micro),
        "anchor_support_exact_result_count": sum(item["exact_result_count"] for item in non_micro),
    }


def _decoder_candidate_specs(config: DiarizationConfig) -> tuple[dict[str, Any], ...]:
    """Return the bounded A/B/C/D decoder grid in conservative order."""

    micro_costs = (0.10, 0.20, 0.30, 0.35)
    specs: list[dict[str, Any]] = [{
        "candidate_id": "A_BASELINE",
        "family": "A_BASELINE",
        "unknown_micro_cost": config.unknown_micro_cost,
        "soft_speaker_emissions": False,
        "switch_base": config.switch_base,
    }]
    specs.extend({
        "candidate_id": f"B_HARD_U{round(cost * 100):02d}",
        "family": "B_UNKNOWN_MICRO_COST",
        "unknown_micro_cost": cost,
        "soft_speaker_emissions": False,
        "switch_base": config.switch_base,
    } for cost in micro_costs)
    specs.extend({
        "candidate_id": f"C_SOFT_U{round(cost * 100):02d}",
        "family": "C_STRICT_CEILING_SOFT_EMISSIONS",
        "unknown_micro_cost": cost,
        "soft_speaker_emissions": True,
        "switch_base": config.switch_base,
    } for cost in micro_costs)
    specs.extend({
        "candidate_id": f"D_SOFT_U{round(cost * 100):02d}_S{round(switch * 100):02d}",
        "family": "D_SOFT_SWITCH_ABLATION",
        "unknown_micro_cost": cost,
        "soft_speaker_emissions": True,
        "switch_base": switch,
    } for cost in micro_costs for switch in (0.08, 0.15, 0.25))
    return tuple(specs)


def _local_failure_code(tracklet: Any, assignment: Any, cfg: DiarizationConfig) -> str:
    reasons = set(assignment.reason_codes)
    if assignment.speaker_id in {"SPEAKER_00", "SPEAKER_01"}:
        return "LOCAL_CANDIDATE"
    if "LOCAL_GATE_FAILED" not in reasons:
        return "+".join(sorted(reasons)) or assignment.attribution_status
    prefix = str(tracklet.kind).lower()
    failures = []
    stable_limit = float(getattr(cfg, f"{prefix}_stable_distance_ceiling"))
    absolute_limit = float(getattr(cfg, f"{prefix}_absolute_distance_max"))
    margin_min = float(getattr(cfg, f"{prefix}_margin_min"))
    if assignment.stable_distance is not None and assignment.stable_distance > stable_limit:
        failures.append("STABLE_DISTANCE")
    if assignment.effective_distance is not None and assignment.effective_distance > absolute_limit:
        failures.append("ABSOLUTE_DISTANCE")
    if assignment.margin is not None and assignment.margin < margin_min:
        failures.append("MARGIN")
    return "LOCAL_GATE_" + ("+".join(failures) if failures else "UNCLASSIFIED")


def _aggregate_decoder_diagnostics(
    tracklets: Sequence[Any],
    embeddings: Sequence[EmbeddingResult],
    trace: Any,
    cfg: DiarizationConfig,
) -> dict[str, Any]:
    """Aggregate scalar decoder evidence without IDs, vectors, or centroids."""

    by_tracklet = {embedding.tracklet_id: embedding for embedding in embeddings}
    local_groups: dict[tuple[str, bool, str], list[int]] = {}
    overrides: dict[tuple[str, str, str], list[int]] = {}
    for tracklet, assignment, final_label in zip(tracklets, trace.local_assignments, trace.labels):
        embedding = by_tracklet.get(tracklet.tracklet_id)
        valid = bool(embedding is not None and embedding.is_valid and embedding.vector is not None)
        outcome = _local_failure_code(tracklet, assignment, cfg)
        duration_us = int(tracklet.end_us - tracklet.start_us)
        key = (str(tracklet.kind), valid, outcome)
        bucket = local_groups.setdefault(key, [0, 0, 0])
        bucket[0] += 1
        bucket[1] += duration_us
        bucket[2] += int(tracklet.clean_speech_us)
        if assignment.speaker_id != final_label:
            override_key = (str(tracklet.kind), str(assignment.speaker_id), str(final_label))
            override = overrides.setdefault(override_key, [0, 0])
            override[0] += 1
            override[1] += duration_us
    return {
        "local_groups": [
            {
                "kind": kind,
                "embedding_valid": valid,
                "local_outcome": outcome,
                "count": values[0],
                "duration_us": values[1],
                "clean_speech_us": values[2],
            }
            for (kind, valid, outcome), values in sorted(local_groups.items())
        ],
        "viterbi_overrides": [
            {
                "kind": kind,
                "local_label": local_label,
                "final_label": final_label,
                "count": values[0],
                "duration_us": values[1],
            }
            for (kind, local_label, final_label), values in sorted(overrides.items())
        ],
    }


def _changed_existing_assigned(
    tracklets: Sequence[Any], baseline: Any, candidate: Any, source_duration_us: int
) -> dict[str, int | float]:
    changed_us = sum(
        int(tracklet.end_us - tracklet.start_us)
        for tracklet, old, new in zip(tracklets, baseline.labels, candidate.labels)
        if old in {"SPEAKER_00", "SPEAKER_01"} and new != old
    )
    baseline_assigned_us = sum(
        int(tracklet.end_us - tracklet.start_us)
        for tracklet, label in zip(tracklets, baseline.labels)
        if label in {"SPEAKER_00", "SPEAKER_01"}
    )
    return {
        "changed_existing_assigned_us": changed_us,
        "changed_existing_assigned_ratio_of_audio": changed_us / max(1, source_duration_us),
        "changed_existing_assigned_ratio_of_baseline_assigned": changed_us / max(1, baseline_assigned_us),
    }


def _span_counts(spans: Sequence[Any]) -> dict[str, int]:
    return {
        label: sum(span.speaker_id == label for span in spans)
        for label in ("SPEAKER_00", "SPEAKER_01", "UNKNOWN", "OVERLAP")
    }


def _complete_merge(spans: Sequence[Any]) -> float:
    assigned = {span.speaker_id for span in spans if span.speaker_id in {"SPEAKER_00", "SPEAKER_01"}}
    return float(len(assigned) < 2)


def _graph_existing_label_parity(
    tracklets: Sequence[Any], baseline_labels: Sequence[str], candidate_labels: Sequence[str],
    protected_overlap_spans: Sequence[Any],
) -> dict[str, Any]:
    """Check exact source-time parity for baseline assigned/overlap material.

    Graph rescue is allowed to materialize only a baseline UNKNOWN tracklet.
    Comparing tracklet source intervals, rather than merged output spans,
    avoids treating a newly rescued adjacent UNKNOWN as a change to an
    already-assigned span while still catching any reassignment or overlap
    drift.
    """

    if len(tracklets) != len(baseline_labels) or len(tracklets) != len(candidate_labels):
        raise RuntimeError("graph rescue label/tracklet length mismatch")
    protected = tuple(
        sorted((int(span.start_us), int(span.end_us), "OVERLAP") for span in protected_overlap_spans)
    )
    baseline_existing = tuple(
        (int(tracklet.start_us), int(tracklet.end_us), str(old), str(new))
        for tracklet, old, new in zip(tracklets, baseline_labels, candidate_labels)
        if old in {"SPEAKER_00", "SPEAKER_01", "OVERLAP"}
    )
    changed = tuple(row for row in baseline_existing if row[2] != row[3])
    # The protected overlap spans are materialized outside graph labels and
    # must remain byte-for-byte source-time identical as well.
    parity = {
        "passed": not changed,
        "existing_assigned_or_overlap_count": len(baseline_existing),
        "changed_existing_assigned_or_overlap_count": len(changed),
        "protected_overlap_count": len(protected),
    }
    if changed:
        raise RuntimeError("graph rescue violated existing assigned/OVERLAP source-time parity")
    return parity


def _graph_rescue_experimental_config() -> GraphRescueConfig:
    """Return the fixed, label-independent policy for the opt-in arm."""

    # Keep this policy in code so a run cannot tune graph topology against the
    # reference labels.  ANCHOR_ONLY is the module default: SUPPORT examples
    # are never silently promoted to graph seeds.
    return GraphRescueConfig(
        enabled=True,
        adjacency_mode="bounded_knn",
        k_neighbors=8,
        propagation_steps=1,
    )


def _graph_rescue_report(
    *,
    tracklets: Sequence[Any],
    baseline_trace: Any,
    embeddings: Sequence[EmbeddingResult],
    decision: Any,
    protected_overlap_spans: Sequence[Any],
    source_duration_us: int,
    reference: Sequence[Mapping[str, Any]] | None,
    score: Any,
    materialize_config: DiarizationConfig,
) -> tuple[dict[str, Any], Any, Any]:
    """Run the graph arm and return redacted report, spans, and result."""

    config = _graph_rescue_experimental_config()
    embedding_by_tracklet = {item.tracklet_id: item for item in embeddings}
    baseline_labels = tuple(baseline_trace.labels)
    graph_result = rescue_unknowns(
        tracklets,
        baseline_labels,
        embedding_by_tracklet,
        decision=decision,
        config=config,
        # Intentionally omit seed_tracklet_ids: the fixed policy is
        # ANCHOR_ONLY and must not be selected from labels or references.
    )
    parity = _graph_existing_label_parity(
        tracklets, baseline_labels, graph_result.labels, protected_overlap_spans,
    )
    rescued_duration_us = sum(
        int(tracklet.end_us - tracklet.start_us)
        for tracklet, old, new in zip(tracklets, baseline_labels, graph_result.labels)
        if old == "UNKNOWN" and new in {"SPEAKER_00", "SPEAKER_01"}
    )
    # Importing the library's materializer keeps span status/merge semantics
    # identical to baseline while making no changes to production code.
    from sddiar.diarization import _materialize  # noqa: PLC0415
    candidate_spans = _materialize(
        graph_result.labels, tracklets, protected_overlap_spans, source_duration_us,
        materialize_config,
    )
    # Compare protected and pre-existing assigned time after materialization;
    # this catches accidental changes in the materializer seam too.
    if not all(
        any(
            span.speaker_id == old
            and int(span.start_us) <= int(tracklet.start_us)
            and int(span.end_us) >= int(tracklet.end_us)
            for span in candidate_spans
        )
        for tracklet, old in zip(tracklets, baseline_labels)
        if old in {"SPEAKER_00", "SPEAKER_01"}
    ) or tuple(sorted(
        (int(span.start_us), int(span.end_us))
        for span in candidate_spans if span.speaker_id == "OVERLAP"
    )) != tuple(sorted(
        (int(span.start_us), int(span.end_us))
        for span in baseline_trace.spans if span.speaker_id == "OVERLAP"
    )):
        raise RuntimeError("graph rescue violated materialized source-time parity")
    changed_existing_assigned_us = sum(
        int(tracklet.end_us - tracklet.start_us)
        for tracklet, old, new in zip(tracklets, baseline_labels, graph_result.labels)
        if old in {"SPEAKER_00", "SPEAKER_01"} and old != new
    )
    baseline_metrics = score(baseline_trace.spans, reference) if reference is not None else None
    candidate_metrics = score(candidate_spans, reference) if reference is not None else None
    def thaw(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): thaw(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [thaw(item) for item in value]
        return value

    receipt = thaw(build_redacted_receipt(graph_result))
    receipt["skip_reason"] = graph_result.diagnostics.get("skip_reason")
    receipt["adjacency_mode"] = graph_result.diagnostics.get("adjacency_mode")
    receipt["seed_mode"] = graph_result.diagnostics.get("seed_eligibility", {}).get("mode", "ANCHOR_ONLY")
    receipt["parity_passed"] = parity["passed"]
    policy = {
        "adjacency_mode": config.adjacency_mode,
        "k_neighbors": config.k_neighbors,
        "propagation_steps": config.propagation_steps,
        "seed_mode": "ANCHOR_ONLY",
        "min_anchor_blocks": config.min_anchor_blocks,
        "min_posterior": config.min_posterior,
        "posterior_margin_min": config.posterior_margin_min,
        "leave_block_margin_min": config.leave_block_margin_min,
        "max_edge_distance": config.max_edge_distance,
    }
    policy_sha256 = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()
    report = {
        "kind": "GRAPH_RESCUE_EXPERIMENTAL_V1",
        "experimental": True,
        "default_enabled": False,
        "production_approved": False,
        "quality_status": "REVIEW_REQUIRED",
        "release_authority": "none",
        "policy": policy,
        "policy_sha256": policy_sha256,
        "candidate_count": len(graph_result.candidates),
        "rescued_duration_us": rescued_duration_us,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "changed_existing_assigned_us": changed_existing_assigned_us,
        "graph_diagnostics_redacted": receipt,
        "limitations": (
            "single_recording_clova_timing_proxy_is_not_release_ground_truth",
            "unknown_only_rescue_requires_independent_rttm_uem_validation",
        ),
    }
    return report, candidate_spans, graph_result


def _run_decoder_calibration(
    *,
    tracklets: Sequence[Any],
    protected_overlap_spans: Sequence[Any],
    states: Mapping[str, Any],
    decision: Any,
    source_duration_us: int,
    config: DiarizationConfig,
    embeddings: Sequence[EmbeddingResult],
    reference: Sequence[Mapping[str, Any]],
    score: Any,
) -> tuple[dict[str, Any], Any, Mapping[str, float]]:
    """Evaluate A/B/C/D on calibration only, then score one holdout choice."""

    split_us = round(source_duration_us * 0.60)
    calibration_reference = _clip_reference(reference, 0, split_us)
    holdout_reference = _clip_reference(reference, split_us, source_duration_us)
    baseline_trace = decode_sequence(
        tracklets,
        protected_overlap_spans,
        states,
        decision,
        source_duration_us,
        config,
        embeddings,
    )
    evaluated: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    configs: dict[str, DiarizationConfig] = {}
    for spec in _decoder_candidate_specs(config):
        candidate_config = replace(
            config,
            unknown_micro_cost=float(spec["unknown_micro_cost"]),
            switch_base=float(spec["switch_base"]),
        )
        if spec["candidate_id"] == "A_BASELINE":
            trace = baseline_trace
        else:
            trace = decode_sequence(
                tracklets,
                protected_overlap_spans,
                states,
                decision,
                source_duration_us,
                candidate_config,
                embeddings,
                soft_speaker_emissions=bool(spec["soft_speaker_emissions"]),
            )
        calibration_metrics = score(trace.spans, calibration_reference)
        candidate_id = str(spec["candidate_id"])
        traces[candidate_id] = trace
        configs[candidate_id] = candidate_config
        evaluated.append({
            **spec,
            "calibration_metrics": calibration_metrics,
            "span_counts": _span_counts(trace.spans),
            "span_timeline_sha256": _span_timeline_sha256(trace.spans),
        })
    eligible = [
        item for item in evaluated
        if item["calibration_metrics"]["speaker_accuracy_given_assigned"] >= 0.985
        and item["calibration_metrics"]["turn_accuracy_given_covered"] >= 0.90
    ]
    selected = max(
        eligible,
        key=lambda item: (
            item["calibration_metrics"]["speaker_mapped_time_accuracy_end_to_end"],
            item["calibration_metrics"]["reference_timeline_coverage"],
            item["calibration_metrics"]["speaker_accuracy_given_assigned"],
        ),
    ) if eligible else evaluated[0]
    selected_id = str(selected["candidate_id"])
    selected_trace = traces[selected_id]
    selected_config = configs[selected_id]

    # The holdout is touched exactly once, after calibration-only selection.
    holdout_metrics = score(selected_trace.spans, holdout_reference)
    full_metrics = score(selected_trace.spans, reference)
    changed = _changed_existing_assigned(tracklets, baseline_trace, selected_trace, source_duration_us)
    baseline_complete_merge = _complete_merge(baseline_trace.spans)
    gates = {
        "calibration_candidate_eligible": bool(eligible),
        "full_coverage_gte_0_43": full_metrics["reference_timeline_coverage"] >= 0.43,
        "holdout_coverage_gte_0_42": holdout_metrics["reference_timeline_coverage"] >= 0.42,
        "full_accuracy_gte_0_985": full_metrics["speaker_accuracy_given_assigned"] >= 0.985,
        "holdout_accuracy_gte_0_99": holdout_metrics["speaker_accuracy_given_assigned"] >= 0.99,
        "complete_merge_unchanged": full_metrics["complete_merge"] == baseline_complete_merge,
        # H1/H2 and states are frozen inputs to every decoder arm.
        "h2_unchanged": True,
        "assigned_label_change_lte_0_005": changed["changed_existing_assigned_ratio_of_audio"] <= 0.005,
    }
    accepted = all(gates.values())
    output_trace = selected_trace if accepted else baseline_trace
    output_full_metrics = (
        full_metrics
        if accepted or selected_id == "A_BASELINE"
        else score(baseline_trace.spans, reference)
    )
    rejection_reasons = [name for name, passed in gates.items() if not passed]
    baseline_diagnostics = _aggregate_decoder_diagnostics(tracklets, embeddings, baseline_trace, config)
    selected_diagnostics = _aggregate_decoder_diagnostics(tracklets, embeddings, selected_trace, selected_config)
    report = {
        "kind": "FROZEN_HYPOTHESIS_DECODER_CALIBRATION_V1",
        "split_ratio": 0.60,
        "selection_constraints": {
            "calibration_assigned_accuracy_min": 0.985,
            "calibration_turn_accuracy_min": 0.90,
            "objective_order": [
                "speaker_mapped_time_accuracy_end_to_end",
                "reference_timeline_coverage",
                "speaker_accuracy_given_assigned",
            ],
        },
        "frozen_hypothesis_state": decision.state,
        "candidate_count": len(evaluated),
        "candidates": evaluated,
        "selected_candidate": selected_id,
        "selected_configuration": {
            "family": selected["family"],
            "unknown_micro_cost": selected["unknown_micro_cost"],
            "soft_speaker_emissions": selected["soft_speaker_emissions"],
            "switch_base": selected["switch_base"],
        },
        "selected_calibration_metrics": selected["calibration_metrics"],
        "selected_holdout_metrics": holdout_metrics,
        "selected_full_metrics": full_metrics,
        "baseline_span_timeline_sha256": _span_timeline_sha256(baseline_trace.spans),
        "selected_span_timeline_sha256": _span_timeline_sha256(selected_trace.spans),
        "emitted_span_timeline_sha256": _span_timeline_sha256(output_trace.spans),
        "baseline_span_counts": _span_counts(baseline_trace.spans),
        "selected_span_counts": _span_counts(selected_trace.spans),
        "emitted_span_counts": _span_counts(output_trace.spans),
        "changed_existing_assigned": changed,
        "aggregate_diagnostics": {
            "local_groups": selected_diagnostics["local_groups"],
            "baseline_viterbi_overrides": baseline_diagnostics["viterbi_overrides"],
            "selected_viterbi_overrides": selected_diagnostics["viterbi_overrides"],
        },
        "gates": {**gates, "all_passed": accepted},
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "emitted_configuration": selected_id if accepted else "A_BASELINE_FAIL_CLOSED",
    }
    return report, output_trace, output_full_metrics


def run_experiment(
    audio_path: str | Path,
    silero_model: str | Path,
    wespeaker_model: str | Path,
    reference_path: str | Path | None = None,
    *,
    threads: int = 1,
    silero_threshold: float = 0.5,
    silero_temporal_postprocess: bool = False,
    auto_gain_normalization: bool = False,
    max_duration_sec: float | None = None,
    h2_complexity_penalty: float = 0.0,
    h2_min_cost_gain: float = 0.02,
    max_tracklet_sec: float = 3.0,
    subsegment_windows: bool = False,
    micro_subsegment_windows: bool = False,
    embedding_batch_regions: int = 1,
    exact_length_batching: bool = False,
    batch_buffer_regions: int = 32,
    calibrate_assignment: bool = False,
    assignment_calibration_worst_speaker_accuracy_min: float | None = None,
    calibrate_decoder: bool = False,
    fast_fp32_baseline_rtf: float | None = None,
    graph_rescue_experimental: bool = False,
    silero_runtime: Any | None = None,
    embedding_backend: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = _process_cpu_seconds()
    stage_timings: dict[str, float] = {}
    evaluation_wall_sec = 0.0

    def score_timed(spans: Sequence[Any], reference_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        nonlocal evaluation_wall_sec
        score_started = time.perf_counter()
        scored = _score(spans, reference_rows)
        evaluation_wall_sec += _elapsed_wall_seconds(score_started)
        return scored

    audio = Path(audio_path)
    silero_path, wespeaker_path = Path(silero_model), Path(wespeaker_model)
    accessor = WavPcmAccessor(audio)
    if accessor.layout.sample_rate_hz != 16000 or accessor.layout.channel_count != 1 or accessor.layout.sample_width_bytes != 2:
        raise ValueError("experiment requires PCM16 mono 16 kHz WAV")
    if type(auto_gain_normalization) is not bool:
        raise ValueError("auto_gain_normalization must be boolean")
    if type(graph_rescue_experimental) is not bool:
        raise ValueError("graph_rescue_experimental must be boolean")
    if max_duration_sec is not None and max_duration_sec <= 0:
        raise ValueError("max_duration_sec must be positive")
    if max_tracklet_sec <= 0:
        raise ValueError("max_tracklet_sec must be positive")
    if embedding_batch_regions <= 0 or batch_buffer_regions <= 0:
        raise ValueError("embedding batch sizes must be positive")
    if exact_length_batching and (subsegment_windows or micro_subsegment_windows):
        raise ValueError("exact-length batching is incompatible with subsegment windows")
    if calibrate_decoder and reference_path is None:
        raise ValueError("decoder calibration requires a timing reference")
    if (
        assignment_calibration_worst_speaker_accuracy_min is not None
        and not 0.0 <= assignment_calibration_worst_speaker_accuracy_min <= 1.0
    ):
        raise ValueError("assignment_calibration_worst_speaker_accuracy_min must be in [0,1]")
    if micro_subsegment_windows and subsegment_windows:
        raise ValueError("global and MICRO-only subsegment windows are mutually exclusive")
    if micro_subsegment_windows and not calibrate_decoder:
        raise ValueError("MICRO-only subsegment windows require fail-closed decoder calibration")
    if micro_subsegment_windows and (
        fast_fp32_baseline_rtf is None
        or not math.isfinite(fast_fp32_baseline_rtf)
        or fast_fp32_baseline_rtf <= 0.0
    ):
        raise ValueError("MICRO-only subsegment windows require a positive fast FP32 baseline RTF")
    audio_digest = _sha256(audio)
    if auto_gain_normalization:
        gain_started = time.perf_counter()
        gain_profile = analyze_pcm16_global_gain(audio, source_sha256=audio_digest)
        gain_metadata = gain_profile.to_dict()
        inference_gain = gain_profile.applied_gain
        inference_accessor: WavPcmAccessor | GainScaledWavPcmAccessor = GainScaledWavPcmAccessor(
            accessor, inference_gain
        )
        stage_timings["audio_gain_profile_wall_sec"] = _elapsed_wall_seconds(gain_started)
    else:
        gain_metadata = disabled_gain_metadata()
        inference_gain = 1.0
        inference_accessor = accessor
    frame_limit = accessor.layout.frame_count if max_duration_sec is None else min(accessor.layout.frame_count, round(max_duration_sec * accessor.layout.sample_rate_hz))
    def decoded_chunks():
        decoder = WavPcmDecoder()
        fast_decode = getattr(decoder, "iter_decode_chunks_numpy", None)
        try:
            import numpy  # type: ignore  # noqa: F401
            numpy_available = True
        except ImportError:
            numpy_available = False
        iterator = (
            fast_decode(audio, frames_per_chunk=240_000)
            if callable(fast_decode) and numpy_available
            else decoder.iter_decode_chunks(audio, frames_per_chunk=240_000)
        )
        for chunk in iterator:
            if chunk.source_start_sample >= frame_limit:
                break
            if chunk.source_end_sample <= frame_limit:
                yield chunk
                continue
            keep = frame_limit - chunk.source_start_sample
            yield DecodedAudioChunk(
                chunk.samples[:keep], chunk.source_start_sample, frame_limit,
                chunk.sample_rate_hz, chunk.channel_count, chunk.metadata,
            )
            break
    vad_started = time.perf_counter()
    if silero_runtime is None:
        silero_runtime = SileroOnnxRuntime(silero_path, session=_ort_session(silero_path, threads=threads), threshold=silero_threshold)
    vad_input = (
        scale_decoded_chunks(decoded_chunks(), inference_gain)
        if auto_gain_normalization
        else decoded_chunks()
    )
    vad_frames = silero_runtime.infer_chunk_stream(vad_input)
    stage_timings["vad_decode_wall_sec"] = _elapsed_wall_seconds(vad_started)

    segmentation_started = time.perf_counter()
    temporal_vad = None
    if silero_temporal_postprocess:
        temporal_vad = SileroTemporalPostprocessor().process(vad_frames)
        temporal_regions = tuple(
            ContractSpeechRegion(
                f"speech-temporal-{index}",
                "audio",
                region.start_us,
                region.end_us,
                reason_codes=("SILERO_TEMPORAL_PADDED",),
            )
            for index, region in enumerate(temporal_vad.regions)
        )
        segmentation = SegmentationEvidence(temporal_regions, (), (), ())
    else:
        segmentation = RuleEvidenceSegmentation(SegmentationConfig(vad_merge_gap_us=200_000)).build(view_id="audio", vad_frames=vad_frames)
    config = DiarizationConfig(
        max_tracklet_us=round(max_tracklet_sec * 1_000_000),
        lambda_k2=h2_complexity_penalty,
        h2_min_cost_gain=h2_min_cost_gain,
    )
    built = build_tracklets(segmentation.speech_regions, cfg=config, audio_id=audio_digest[:16])
    regions = tuple(
        EmbeddingRegion(
            f"embedding-{index:06d}", tracklet.tracklet_id,
            tracklet.start_us, tracklet.end_us, tracklet.clean_speech_us,
            min(1.0, tracklet.clean_speech_us / max(1, tracklet.end_us - tracklet.start_us)),
        )
        for index, tracklet in enumerate(built.tracklets)
    )
    micro_region_ids = tuple(
        region.embedding_region_id
        for tracklet, region in zip(built.tracklets, regions)
        if tracklet.kind == "MICRO"
    )
    stage_timings["segmentation_tracklets_wall_sec"] = _elapsed_wall_seconds(segmentation_started)

    embeddings_started = time.perf_counter()
    if embedding_backend is None:
        embedding_session = _ort_session(wespeaker_path, threads=threads)
        fast_reader = getattr(inference_accessor, "read_mono_samples_numpy", None)
        embedding_backend = WeSpeakerCpuEmbeddingBackend(
            _artifact(wespeaker_path, "wespeaker-resnet34", "speaker_embedding"),
            session=embedding_session, max_batch_regions=embedding_batch_regions,
            audio_provider=(
                (lambda region: fast_reader(region.start_us, region.end_us))
                if callable(fast_reader)
                else (lambda region: inference_accessor.read_mono_samples(region.start_us, region.end_us))
            ),
            subsegment_windows=subsegment_windows,
            subsegment_region_ids=micro_region_ids if micro_subsegment_windows else (),
            exact_length_batching=exact_length_batching,
            batch_buffer_regions=batch_buffer_regions,
        )
    elif micro_subsegment_windows:
        configured = frozenset(getattr(embedding_backend, "subsegment_region_ids", ()))
        if configured != frozenset(micro_region_ids):
            raise ValueError(
                "injected embedding backend must be configured for the exact MICRO region selector"
            )
    embeddings = tuple(embedding_backend.embed(regions)) if regions else ()
    baseline_embeddings = embeddings
    embedding_selector_parity = None
    if micro_subsegment_windows:
        retained = getattr(embedding_backend, "last_selector_baseline_results", None)
        if retained is None or len(retained) != len(embeddings):
            raise ValueError("MICRO selector backend did not retain exact baseline embeddings")
        baseline_embeddings = tuple(retained)
        embedding_selector_parity = _embedding_selector_parity(
            built.tracklets, baseline_embeddings, embeddings
        )
    stage_timings["embeddings_wall_sec"] = _elapsed_wall_seconds(embeddings_started)

    hypothesis_started = time.perf_counter()
    anchors, support, deferred = select_anchor_evidence(built.tracklets, baseline_embeddings, config)
    hypothesis_evaluation = evaluate_hypotheses(anchors, config)
    decision = hypothesis_evaluation.decision
    h2_diagnostics = hypothesis_evaluation.h2_diagnostics
    if micro_subsegment_windows:
        selected_anchors, selected_support, selected_deferred = select_anchor_evidence(
            built.tracklets, embeddings, config
        )
        selected_hypothesis_evaluation = evaluate_hypotheses(selected_anchors, config)
        assert embedding_selector_parity is not None
        embedding_selector_parity.update({
            "anchor_evidence_exact": selected_anchors == anchors,
            "support_evidence_exact": selected_support == support,
            "anchor_count_baseline": len(anchors),
            "anchor_count_selected": len(selected_anchors),
            "support_count_baseline": len(support),
            "support_count_selected": len(selected_support),
            "deferred_count_baseline": len(deferred),
            "deferred_count_selected": len(selected_deferred),
            "hypothesis_evaluation_exact": selected_hypothesis_evaluation == hypothesis_evaluation,
            "decision_exact": selected_hypothesis_evaluation.decision == decision,
            "h2_diagnostics_exact": selected_hypothesis_evaluation.h2_diagnostics == h2_diagnostics,
        })
    states = speaker_states_from_decision(decision, anchors)
    states = refine_recent_states(built.tracklets, baseline_embeddings, states, decision, config)
    stage_timings["hypothesis_state_wall_sec"] = _elapsed_wall_seconds(hypothesis_started)
    duration_us = round(frame_limit * 1_000_000 / accessor.layout.sample_rate_hz)
    calibration_report = None
    decoder_report = None
    decoder_metrics = None
    decoder_trace = None
    micro_subsegment_report = None
    strict_baseline_trace = None
    reference = _reference(Path(reference_path)) if reference_path else None
    selected_config = config
    finalization_started = time.perf_counter()
    if calibrate_assignment and reference:
        split_us = round(duration_us * 0.60)
        calibration_reference = _clip_reference(reference, 0, split_us)
        holdout_reference = _clip_reference(reference, split_us, duration_us)
        candidates = []
        for distance_limit in (0.30, 0.35, 0.40, 0.45, 0.50):
            candidate_config = replace(
                config,
                anchor_stable_distance_ceiling=distance_limit,
                anchor_absolute_distance_max=distance_limit,
                support_stable_distance_ceiling=distance_limit,
                support_absolute_distance_max=distance_limit,
                micro_stable_distance_ceiling=min(distance_limit, 0.35),
                micro_absolute_distance_max=min(distance_limit, 0.35),
                unknown_cost=distance_limit + 0.05,
            )
            candidate_spans = finalize_sequence(
                built.tracklets, built.protected_overlap_spans, states, decision,
                duration_us, candidate_config, baseline_embeddings,
            )
            candidate_metrics = score_timed(candidate_spans, calibration_reference)
            candidates.append((distance_limit, candidate_config, candidate_spans, candidate_metrics))
        eligible = [
            item for item in candidates
            if item[3]["speaker_accuracy_given_assigned"] >= 0.95
            and item[3]["turn_accuracy_given_covered"] >= 0.90
            and (
                assignment_calibration_worst_speaker_accuracy_min is None
                or item[3]["worst_speaker_accuracy_given_assigned"]
                >= assignment_calibration_worst_speaker_accuracy_min
            )
        ]
        if eligible:
            selected = max(eligible, key=lambda item: (item[3]["reference_timeline_coverage"], -item[0]))
            distance_limit, selected_config, spans, calibration_metrics = selected
            selection_status = "ELIGIBLE_SELECTED"
        else:
            # A calibration constraint is a safety gate, not a ranking hint.
            # If no candidate passes, preserve the uncalibrated conservative
            # library configuration instead of selecting the highest-coverage
            # failed candidate.
            distance_limit = None
            selected_config = config
            spans = finalize_sequence(
                built.tracklets, built.protected_overlap_spans, states, decision,
                duration_us, config, embeddings,
            )
            calibration_metrics = _score(spans, calibration_reference)
            selection_status = "NO_ELIGIBLE_FAIL_CLOSED"
        calibration_report = {
            "kind": "CLOVA_PROXY_SINGLE_RECORDING_SPLIT",
            "split_ratio": 0.60,
            "selection_status": selection_status,
            "selected_distance_limit": distance_limit,
            "worst_speaker_accuracy_min": assignment_calibration_worst_speaker_accuracy_min,
            "calibration_metrics": calibration_metrics,
            "candidates": [
                {"distance_limit": item[0], "metrics": item[3]} for item in candidates
            ],
        }
        if calibrate_decoder:
            # The decoder experiment owns the single post-selection holdout
            # evaluation.  Distance calibration remains calibration-only.
            calibration_report["holdout_evaluation"] = "DEFERRED_TO_DECODER_CALIBRATION"
        else:
            calibration_report["holdout_metrics"] = score_timed(spans, holdout_reference)
    else:
        spans = finalize_sequence(built.tracklets, built.protected_overlap_spans, states, decision, duration_us, config, baseline_embeddings)
    if micro_subsegment_windows:
        strict_baseline_trace = decode_sequence(
            built.tracklets,
            built.protected_overlap_spans,
            states,
            decision,
            duration_us,
            selected_config,
            baseline_embeddings,
        )
    if calibrate_decoder and reference:
        decoder_report, decoder_trace, decoder_metrics = _run_decoder_calibration(
            tracklets=built.tracklets,
            protected_overlap_spans=built.protected_overlap_spans,
            states=states,
            decision=decision,
            source_duration_us=duration_us,
            config=selected_config,
            embeddings=embeddings,
            reference=reference,
            score=score_timed,
        )
        spans = decoder_trace.spans
    if micro_subsegment_windows and reference:
        assert strict_baseline_trace is not None
        assert embedding_selector_parity is not None
        assert decoder_report is not None
        parity_gates = {
            "anchor_support_embeddings_exact": embedding_selector_parity["anchor_support_exact"],
            "anchor_evidence_exact": embedding_selector_parity["anchor_evidence_exact"],
            "support_evidence_exact": embedding_selector_parity["support_evidence_exact"],
            "decision_exact": embedding_selector_parity["decision_exact"],
            "h2_diagnostics_exact": embedding_selector_parity["h2_diagnostics_exact"],
        }
        selected_metrics = decoder_report["selected_full_metrics"]
        selected_holdout = decoder_report["selected_holdout_metrics"]
        strict_change = _changed_existing_assigned(
            built.tracklets, strict_baseline_trace, decoder_trace, duration_us
        )
        quality_gates = {
            "decoder_calibration_accepted": decoder_report["accepted"],
            "full_coverage_gte_0_43": selected_metrics["reference_timeline_coverage"] >= 0.43,
            "holdout_coverage_gte_0_42": selected_holdout["reference_timeline_coverage"] >= 0.42,
            "full_accuracy_gte_0_985": selected_metrics["speaker_accuracy_given_assigned"] >= 0.985,
            "holdout_accuracy_gte_0_99": selected_holdout["speaker_accuracy_given_assigned"] >= 0.99,
            "assigned_label_change_lte_0_005": strict_change["changed_existing_assigned_ratio_of_audio"] <= 0.005,
        }
        pre_rtf_accepted = all((*parity_gates.values(), *quality_gates.values()))
        if not pre_rtf_accepted:
            spans = strict_baseline_trace.spans
            decoder_metrics = score_timed(spans, reference)
        micro_subsegment_report = {
            "kind": "SELECTIVE_MICRO_FIXED_WINDOW_EMBEDDING_V1",
            "selector": {
                "tracklet_kind": "MICRO",
                "selected_region_count": len(micro_region_ids),
                "window_frames": 150,
                "period_frames": 75,
            },
            "embedding_parity": embedding_selector_parity,
            "baseline_valid_embedding_count": sum(item.is_valid for item in baseline_embeddings),
            "selected_valid_embedding_count": sum(item.is_valid for item in embeddings),
            "strict_baseline_span_timeline_sha256": _span_timeline_sha256(strict_baseline_trace.spans),
            "decoder_emitted_before_e2_gate_sha256": _span_timeline_sha256(decoder_trace.spans),
            "changed_existing_assigned": strict_change,
            "selected_full_metrics": selected_metrics,
            "selected_holdout_metrics": selected_holdout,
            "parity_gates": parity_gates,
            "quality_gates": quality_gates,
            "pre_rtf_accepted": pre_rtf_accepted,
            "fast_fp32_baseline_rtf": fast_fp32_baseline_rtf,
        }
    stage_timings["finalization_proxy_calibration_wall_sec"] = _elapsed_wall_seconds(finalization_started)
    metrics = decoder_metrics if decoder_metrics is not None else (score_timed(spans, reference) if reference else None)
    stage_timings["evaluation_scoring_wall_sec"] = evaluation_wall_sec
    quality = RuleBasedQualityGate().evaluate({"metrics": {}, "hypothesis_uncertain": decision.state == "UNCERTAIN_1_OR_2"}, None)
    elapsed = _elapsed_wall_seconds(started)
    if micro_subsegment_report is not None:
        assert strict_baseline_trace is not None
        measured_candidate_rtf = elapsed / max(1e-9, duration_us / 1_000_000)
        rtf_limit = 1.5 * float(fast_fp32_baseline_rtf)
        rtf_gate = measured_candidate_rtf <= rtf_limit
        final_accepted = bool(micro_subsegment_report["pre_rtf_accepted"] and rtf_gate)
        if not final_accepted and micro_subsegment_report["pre_rtf_accepted"]:
            spans = strict_baseline_trace.spans
            metrics = score_timed(spans, reference) if reference else None
            stage_timings["evaluation_scoring_wall_sec"] = evaluation_wall_sec
            elapsed = _elapsed_wall_seconds(started)
        all_gates = {
            **micro_subsegment_report["parity_gates"],
            **micro_subsegment_report["quality_gates"],
            "rtf_lte_1_5x_fast_fp32": rtf_gate,
        }
        micro_subsegment_report.update({
            "measured_candidate_rtf": measured_candidate_rtf,
            "rtf_limit": rtf_limit,
            "gates": {**all_gates, "all_passed": final_accepted},
            "accepted": final_accepted,
            "rejection_reasons": [name for name, passed in all_gates.items() if not passed],
            "emitted_configuration": (
                decoder_report["emitted_configuration"]
                if final_accepted and decoder_report is not None
                else "STRICT_BASELINE_FAIL_CLOSED"
            ),
            "emitted_span_timeline_sha256": _span_timeline_sha256(spans),
            "emitted_span_counts": _span_counts(spans),
        })
    graph_report = None
    if graph_rescue_experimental:
        graph_started = time.perf_counter()
        # Follow the exact trace that produced the currently emitted baseline.
        # Selective MICRO can restore its strict baseline after the RTF gate.
        current_timeline = _span_timeline_sha256(spans)
        if decoder_trace is not None and current_timeline == _span_timeline_sha256(decoder_trace.spans):
            graph_baseline_trace = decoder_trace
            graph_embeddings = embeddings
        elif strict_baseline_trace is not None and current_timeline == _span_timeline_sha256(strict_baseline_trace.spans):
            graph_baseline_trace = strict_baseline_trace
            graph_embeddings = baseline_embeddings
        else:
            graph_baseline_trace = decode_sequence(
                built.tracklets, built.protected_overlap_spans, states, decision,
                duration_us, selected_config, baseline_embeddings,
            )
            graph_embeddings = baseline_embeddings
            if current_timeline != _span_timeline_sha256(graph_baseline_trace.spans):
                raise RuntimeError("graph rescue baseline trace does not match emitted baseline")
        graph_report, candidate_spans, _graph_result = _graph_rescue_report(
            tracklets=built.tracklets,
            baseline_trace=graph_baseline_trace,
            embeddings=graph_embeddings,
            decision=decision,
            protected_overlap_spans=built.protected_overlap_spans,
            source_duration_us=duration_us,
            reference=reference,
            score=score_timed,
            materialize_config=selected_config,
        )
        spans = candidate_spans
        metrics = graph_report["candidate_metrics"]
        stage_timings["graph_rescue_wall_sec"] = _elapsed_wall_seconds(graph_started)
        stage_timings["evaluation_scoring_wall_sec"] = evaluation_wall_sec
        elapsed = _elapsed_wall_seconds(started)
    process_cpu_sec = max(0.0, _process_cpu_seconds() - cpu_started)
    peak_rss = peak_rss_bytes()
    feature_mode = getattr(embedding_backend, "feature_mode", "injected")
    feature_runtime_version = None
    if feature_mode == "kaldi_native":
        import kaldi_native_fbank  # type: ignore
        feature_runtime_version = kaldi_native_fbank.__version__
    result: dict[str, Any] = {
        "schema": "onnx_diarization_experiment_v2",
        "result_kind": "DEVELOPMENT_ONNX_CPU_DIARIZATION",
        "audio_sha256_prefix": audio_digest[:12],
        "duration_sec": round(duration_us / 1_000_000, 3),
        "sample_rate_hz": accessor.layout.sample_rate_hz,
        "runtime_config": {
            "threads": threads,
            "silero_temporal_postprocess": silero_temporal_postprocess,
            "auto_gain_normalization": auto_gain_normalization,
            "audio_gain_policy": DEFAULT_GLOBAL_GAIN_POLICY.to_dict(),
            "audio_gain_policy_sha256": DEFAULT_GLOBAL_GAIN_POLICY.sha256,
            "max_tracklet_sec": max_tracklet_sec,
            "embedding_batch_regions": embedding_batch_regions,
            "exact_length_batching": exact_length_batching,
            "batch_buffer_regions": batch_buffer_regions,
            "subsegment_windows": subsegment_windows,
            "h2_complexity_penalty": h2_complexity_penalty,
            "h2_min_cost_gain": h2_min_cost_gain,
            "graph_rescue_experimental": graph_rescue_experimental,
            "feature_mode": feature_mode,
            "feature_runtime_version": feature_runtime_version,
            "feature_frontend_tag": (
                STRICT_NATIVE_FBANK_TAG
                if feature_mode == "kaldi_native"
                else DEVELOPMENT_APPROXIMATION_TAG
                if feature_mode == "development_approximation"
                else "injected"
            ),
        },
        "vad_frames": len(vad_frames),
        "speech_regions": len(segmentation.speech_regions),
        "vad_speech_ratio": sum(region.end_us - region.start_us for region in segmentation.speech_regions) / max(1, duration_us),
        "temporal_vad": ({
            "core_duration_us": temporal_vad.core_duration_us,
            "halo_duration_us": temporal_vad.halo_duration_us,
            "region_count": len(temporal_vad.regions),
        } if temporal_vad is not None else None),
        "tracklets": len(built.tracklets),
        "anchors": len(anchors),
        "support": len(support),
        "deferred": len(deferred),
        "valid_embeddings": sum(embedding.is_valid for embedding in embeddings),
        "decision": decision.state,
        "decision_reasons": list(decision.reason_codes),
        "selected_hypothesis": ({
            "k": decision.hypothesis.k,
            "robust_cost": decision.hypothesis.robust_cost,
            "total_cost": decision.hypothesis.total_cost,
            "outlier_ratio": decision.hypothesis.outlier_ratio,
            "cluster_dispersion": list(decision.hypothesis.cluster_dispersion),
        } if decision.hypothesis else None),
        "h2_diagnostics": {
            "is_valid": h2_diagnostics.is_valid,
            "valid_constraints": h2_diagnostics.valid_constraints,
            "robust_cost": h2_diagnostics.robust_cost,
            "total_cost": h2_diagnostics.total_cost,
            "separation": h2_diagnostics.separation,
            "outlier_ratio": h2_diagnostics.outlier_ratio,
            "cluster_dispersion": list(h2_diagnostics.cluster_dispersion),
            "independent_anchor_count": list(h2_diagnostics.independent_anchor_count),
            "label_stability": h2_diagnostics.label_stability,
            "centroid_stability": h2_diagnostics.centroid_stability,
            "temporal_interleaving": h2_diagnostics.temporal_interleaving,
            "continuous_speech_conflict": h2_diagnostics.continuous_speech_conflict,
            "third_speaker_risk": h2_diagnostics.third_speaker_risk,
            "reason_codes": list(h2_diagnostics.reason_codes),
        },
        "quality_status": quality.status,
        "span_counts": _span_counts(spans),
        "span_timeline_sha256": _span_timeline_sha256(spans),
        "elapsed_wall_sec": round(elapsed, 4),
        "rtf": round(elapsed / max(1e-9, duration_us / 1_000_000), 6),
        "stage_timings": {name: round(value, 6) for name, value in stage_timings.items()},
        "process_cpu_sec": round(process_cpu_sec, 6),
        "cpu_seconds_per_wall_second": round(process_cpu_sec / max(1e-9, elapsed), 6),
        "peak_rss_mb": round(peak_rss / (1024 * 1024), 2) if peak_rss is not None else None,
        "silero_model_sha256": _sha256(silero_path) if silero_path.is_file() else None,
        "wespeaker_model_sha256": _sha256(wespeaker_path) if wespeaker_path.is_file() else None,
        "audio_gain_normalization": gain_metadata,
    }
    if metrics is not None:
        result["metrics"] = metrics
    if calibration_report:
        result["proxy_calibration"] = calibration_report
    if decoder_report:
        result["decoder_calibration"] = decoder_report
    if micro_subsegment_report:
        result["runtime_config"]["micro_subsegment_windows"] = True
        result["micro_subsegment_experiment"] = micro_subsegment_report
    if graph_report is not None:
        result["graph_rescue_experimental"] = graph_report
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local CPU Silero + WeSpeaker diarization experiment")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--silero", required=True, type=Path)
    parser.add_argument("--wespeaker", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--silero-threshold", type=float, default=0.5)
    parser.add_argument("--silero-temporal-postprocess", action="store_true")
    parser.add_argument("--auto-gain-normalization", action="store_true")
    parser.add_argument("--max-duration-sec", type=float)
    parser.add_argument("--h2-complexity-penalty", type=float, default=0.0)
    parser.add_argument("--h2-min-cost-gain", type=float, default=0.02)
    parser.add_argument("--max-tracklet-sec", type=float, default=3.0)
    parser.add_argument("--subsegment-windows", action="store_true")
    parser.add_argument("--micro-subsegment-windows", action="store_true")
    parser.add_argument("--embedding-batch-regions", type=int, default=1)
    parser.add_argument("--exact-length-batching", action="store_true")
    parser.add_argument("--batch-buffer-regions", type=int, default=32)
    parser.add_argument("--calibrate-assignment", action="store_true")
    parser.add_argument("--assignment-calibration-worst-speaker-accuracy-min", type=float)
    parser.add_argument("--calibrate-decoder", action="store_true")
    parser.add_argument("--fast-fp32-baseline-rtf", type=float)
    parser.add_argument(
        "--graph-rescue-experimental",
        action="store_true",
        help="opt in to the fixed UNKNOWN-only graph rescue experiment",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run_experiment(
        args.audio, args.silero, args.wespeaker, args.reference,
        threads=args.threads, silero_threshold=args.silero_threshold,
        silero_temporal_postprocess=args.silero_temporal_postprocess,
        auto_gain_normalization=args.auto_gain_normalization,
        max_duration_sec=args.max_duration_sec,
        h2_complexity_penalty=args.h2_complexity_penalty,
        h2_min_cost_gain=args.h2_min_cost_gain,
        max_tracklet_sec=args.max_tracklet_sec,
        subsegment_windows=args.subsegment_windows,
        micro_subsegment_windows=args.micro_subsegment_windows,
        embedding_batch_regions=args.embedding_batch_regions,
        exact_length_batching=args.exact_length_batching,
        batch_buffer_regions=args.batch_buffer_regions,
        calibrate_assignment=args.calibrate_assignment,
        assignment_calibration_worst_speaker_accuracy_min=args.assignment_calibration_worst_speaker_accuracy_min,
        calibrate_decoder=args.calibrate_decoder,
        fast_fp32_baseline_rtf=args.fast_fp32_baseline_rtf,
        graph_rescue_experimental=args.graph_rescue_experimental,
    ), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
