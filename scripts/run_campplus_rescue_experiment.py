#!/usr/bin/env python3
"""ResNet-authority plus CAM++ UNKNOWN-only rescue experiment.

The ResNet H1/H2 decision, speaker IDs, centroids, and already-assigned spans
are immutable authority.  CAM++ receives the same strict FBank regions, but it
may only propose labels for baseline UNKNOWN tracklets when its own strict
distance/margin gates agree with the unbounded nearest ResNet centroid.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_diarization_proxy import analyze as analyze_proxy  # noqa: E402
from run_onnx_diarization_experiment import (  # noqa: E402
    _artifact,
    _clip_reference,
    _ort_session,
    _reference,
    _score,
    _sha256,
    _span_timeline_sha256,
)
from sddiar.contracts import EmbeddingRegion, EmbeddingResult  # noqa: E402
from sddiar.diarization import (  # noqa: E402
    DiarizationConfig,
    build_tracklets,
    cosine_distance,
    decode_sequence,
    evaluate_hypotheses,
    select_anchor_evidence,
    speaker_states_from_decision,
)
from sddiar.media import DecodedAudioChunk, WavPcmAccessor, WavPcmDecoder  # noqa: E402
from sddiar.segmentation import RuleEvidenceSegmentation, SegmentationConfig  # noqa: E402
from sddiar.service import atomic_publish  # noqa: E402
from sddiar.silero_runtime import SileroOnnxRuntime  # noqa: E402
from sddiar.wespeaker_runtime import WeSpeakerCpuEmbeddingBackend  # noqa: E402


SPEAKERS = ("SPEAKER_00", "SPEAKER_01")
DISTANCE_GRID = (0.25, 0.30, 0.35, 0.40, 0.45)
MARGIN_GRID = (0.03, 0.05, 0.08)


@dataclass(frozen=True, slots=True)
class RescueSpan:
    start_us: int
    end_us: int
    speaker_id: str
    attribution_status: str


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    value = raw if sys.platform == "darwin" else raw * 1024
    return round(value / (1024 * 1024), 2)


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("challenger centroid is zero or non-finite")
    return tuple(float(value) / norm for value in values)


def build_challenger_centroids(
    anchors: Sequence[Any],
    authority_states: Mapping[str, Any],
    challenger_embeddings: Sequence[EmbeddingResult],
) -> tuple[dict[str, tuple[float, ...]], dict[str, Any]]:
    """Pool CAM++ anchor vectors using only ResNet labels and weights."""

    anchor_by_id = {anchor.tracklet_id: anchor for anchor in anchors}
    challenger_by_id = {item.tracklet_id: item for item in challenger_embeddings}
    centroids: dict[str, tuple[float, ...]] = {}
    diagnostics: dict[str, Any] = {}
    for speaker_id in sorted(authority_states):
        state = authority_states[speaker_id]
        rows = []
        invalid = 0
        for tracklet_id in state.stable_anchor_ids:
            anchor = anchor_by_id[tracklet_id]
            embedding = challenger_by_id.get(tracklet_id)
            if embedding is None or not embedding.is_valid or embedding.vector is None:
                invalid += 1
                continue
            rows.append((embedding.vector, float(anchor.weight)))
        if invalid or not rows:
            raise ValueError("CAM++ challenger lacks a valid ResNet authority anchor")
        dimension = len(rows[0][0])
        sums = [0.0] * dimension
        total_weight = 0.0
        for vector, weight in rows:
            if len(vector) != dimension:
                raise ValueError("CAM++ challenger embedding dimensions differ")
            total_weight += weight
            for index, value in enumerate(vector):
                sums[index] += weight * float(value)
        centroids[speaker_id] = _normalize(sums)
        diagnostics[speaker_id] = {
            "authority_anchor_count": len(rows),
            "authority_anchor_weight": total_weight,
            "invalid_authority_anchor_count": invalid,
        }
    if tuple(sorted(centroids)) != SPEAKERS:
        raise ValueError("challenger requires exactly two ResNet authority speaker states")
    return centroids, diagnostics


def _nearest(vector: Sequence[float], centroids: Mapping[str, Sequence[float]]) -> tuple[str, float, float]:
    scored = sorted(
        ((speaker_id, cosine_distance(vector, centroid)) for speaker_id, centroid in centroids.items()),
        key=lambda item: (item[1], item[0]),
    )
    margin = scored[1][1] - scored[0][1] if len(scored) > 1 else math.inf
    return scored[0][0], scored[0][1], margin


def _neighbor_consensus(
    index: int,
    candidate: str,
    tracklets: Sequence[Any],
    baseline_labels: Sequence[str],
    max_gap_us: int,
) -> bool:
    left = next((position for position in range(index - 1, -1, -1) if baseline_labels[position] in SPEAKERS), None)
    right = next((position for position in range(index + 1, len(tracklets)) if baseline_labels[position] in SPEAKERS), None)
    if left is None or right is None:
        return False
    return (
        baseline_labels[left] == candidate
        and baseline_labels[right] == candidate
        and tracklets[index].start_us - tracklets[left].end_us <= max_gap_us
        and tracklets[right].start_us - tracklets[index].end_us <= max_gap_us
    )


def rescue_labels(
    *,
    tracklets: Sequence[Any],
    baseline_labels: Sequence[str],
    resnet_embeddings: Sequence[EmbeddingResult],
    resnet_states: Mapping[str, Any],
    challenger_embeddings: Sequence[EmbeddingResult],
    challenger_centroids: Mapping[str, Sequence[float]],
    distance_limit: float,
    margin_min: float,
    require_neighbors: bool,
    max_gap_us: int,
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Rescue baseline UNKNOWN only; never mutate authority assignments."""

    resnet_by_id = {item.tracklet_id: item for item in resnet_embeddings}
    challenger_by_id = {item.tracklet_id: item for item in challenger_embeddings}
    resnet_centroids = {
        speaker_id: state.stable_anchor_centroid for speaker_id, state in resnet_states.items()
    }
    labels = list(baseline_labels)
    counters = {
        "baseline_unknown": 0,
        "challenger_invalid": 0,
        "resnet_invalid": 0,
        "challenger_distance_failed": 0,
        "challenger_margin_failed": 0,
        "authority_disagreement": 0,
        "neighbor_failed": 0,
        "rescued": 0,
        "rescued_duration_us": 0,
    }
    for index, (tracklet, baseline_label) in enumerate(zip(tracklets, baseline_labels)):
        if baseline_label != "UNKNOWN":
            continue
        counters["baseline_unknown"] += 1
        challenger = challenger_by_id.get(tracklet.tracklet_id)
        if challenger is None or not challenger.is_valid or challenger.vector is None:
            counters["challenger_invalid"] += 1
            continue
        authority = resnet_by_id.get(tracklet.tracklet_id)
        if authority is None or not authority.is_valid or authority.vector is None:
            counters["resnet_invalid"] += 1
            continue
        challenger_speaker, challenger_distance, challenger_margin = _nearest(
            challenger.vector, challenger_centroids
        )
        if challenger_distance > distance_limit:
            counters["challenger_distance_failed"] += 1
            continue
        if challenger_margin < margin_min:
            counters["challenger_margin_failed"] += 1
            continue
        authority_speaker, _, _ = _nearest(authority.vector, resnet_centroids)
        if authority_speaker != challenger_speaker:
            counters["authority_disagreement"] += 1
            continue
        if require_neighbors and not _neighbor_consensus(
            index, challenger_speaker, tracklets, baseline_labels, max_gap_us
        ):
            counters["neighbor_failed"] += 1
            continue
        labels[index] = challenger_speaker
        counters["rescued"] += 1
        counters["rescued_duration_us"] += int(tracklet.end_us - tracklet.start_us)
    return tuple(labels), counters


def materialize_labels(labels: Sequence[str], tracklets: Sequence[Any]) -> tuple[RescueSpan, ...]:
    spans: list[RescueSpan] = []
    for label, tracklet in zip(labels, tracklets):
        status = (
            "ASSIGNED" if label in SPEAKERS
            else "UNKNOWN_SHORT" if tracklet.kind == "MICRO"
            else "UNKNOWN_INSUFFICIENT_EVIDENCE"
        )
        span = RescueSpan(int(tracklet.start_us), int(tracklet.end_us), str(label), status)
        if spans and spans[-1].end_us == span.start_us and spans[-1].speaker_id == span.speaker_id and spans[-1].attribution_status == span.attribution_status:
            previous = spans[-1]
            spans[-1] = RescueSpan(previous.start_us, span.end_us, span.speaker_id, span.attribution_status)
        else:
            spans.append(span)
    return tuple(spans)


def _proxy_score(spans: Sequence[Any], reference: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    base = _score(spans, reference)
    ref_rows = tuple({
        "speaker_id": str(row["speaker"]),
        "start_us": int(row["start_us"]),
        "end_us": int(row["end_us"]),
    } for row in reference)
    span_rows = tuple({
        "speaker_id": str(span.speaker_id),
        "start_us": int(span.start_us),
        "end_us": int(span.end_us),
    } for span in spans)
    analysis = analyze_proxy(ref_rows, span_rows)
    accuracies = [
        float(row["assigned_accuracy"])
        for row in analysis["per_reference_speaker"].values()
    ]
    return {**base, "worst_speaker_accuracy_given_assigned": min(accuracies) if accuracies else 0.0}


def _iter_chunks(audio: Path, frame_limit: int):
    decoder = WavPcmDecoder()
    fast = getattr(decoder, "iter_decode_chunks_numpy", None)
    iterator = fast(audio, frames_per_chunk=240_000) if callable(fast) else decoder.iter_decode_chunks(audio, frames_per_chunk=240_000)
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


def run_experiment(
    audio_path: str | Path,
    silero_model: str | Path,
    resnet_model: str | Path,
    campplus_model: str | Path,
    reference_path: str | Path,
    *,
    threads: int = 1,
    c1_rtf: float = 0.010144,
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    stage: dict[str, float] = {}
    audio = Path(audio_path)
    silero_path, resnet_path, campplus_path = map(Path, (silero_model, resnet_model, campplus_model))
    reference = _reference(Path(reference_path))
    accessor = WavPcmAccessor(audio)
    if accessor.layout.sample_rate_hz != 16_000 or accessor.layout.channel_count != 1 or accessor.layout.sample_width_bytes != 2:
        raise ValueError("E3 requires PCM16 mono 16 kHz WAV")
    if not math.isfinite(c1_rtf) or c1_rtf <= 0:
        raise ValueError("c1_rtf must be positive and finite")
    frame_limit = accessor.layout.frame_count
    duration_us = round(frame_limit * 1_000_000 / accessor.layout.sample_rate_hz)

    tick = time.perf_counter()
    silero = SileroOnnxRuntime(
        silero_path, session=_ort_session(silero_path, threads=threads), threshold=0.5
    )
    vad_frames = silero.infer_chunk_stream(_iter_chunks(audio, frame_limit))
    segmentation = RuleEvidenceSegmentation(
        SegmentationConfig(vad_merge_gap_us=200_000)
    ).build(view_id="audio", vad_frames=vad_frames)
    config = DiarizationConfig(max_tracklet_us=3_000_000, lambda_k2=0.0, h2_min_cost_gain=0.02)
    audio_hash = _sha256(audio)
    built = build_tracklets(segmentation.speech_regions, cfg=config, audio_id=audio_hash[:16])
    regions = tuple(
        EmbeddingRegion(
            f"embedding-{index:06d}", tracklet.tracklet_id,
            tracklet.start_us, tracklet.end_us, tracklet.clean_speech_us,
            min(1.0, tracklet.clean_speech_us / max(1, tracklet.end_us - tracklet.start_us)),
        )
        for index, tracklet in enumerate(built.tracklets)
    )
    stage["vad_segmentation_wall_sec"] = time.perf_counter() - tick

    fast_reader = getattr(accessor, "read_mono_samples_numpy", None)
    provider = (
        (lambda region: fast_reader(region.start_us, region.end_us))
        if callable(fast_reader)
        else (lambda region: accessor.read_mono_samples(region.start_us, region.end_us))
    )
    tick = time.perf_counter()
    resnet = WeSpeakerCpuEmbeddingBackend(
        _artifact(resnet_path, "wespeaker-resnet34", "speaker_embedding"),
        session=_ort_session(resnet_path, threads=threads),
        max_batch_regions=1,
        audio_provider=provider,
    )
    resnet_embeddings = tuple(resnet.embed(regions))
    anchors, support, deferred = select_anchor_evidence(built.tracklets, resnet_embeddings, config)
    evaluation = evaluate_hypotheses(anchors, config)
    decision = evaluation.decision
    if decision.state != "H2_CONFIRMED":
        raise ValueError("E3 challenger requires frozen ResNet H2 authority")
    states = speaker_states_from_decision(decision, anchors)
    stage["resnet_authority_wall_sec"] = time.perf_counter() - tick

    split_us = round(duration_us * 0.60)
    calibration_reference = _clip_reference(reference, 0, split_us)
    holdout_reference = _clip_reference(reference, split_us, duration_us)
    distance_candidates = []
    for limit in (0.30, 0.35, 0.40, 0.45, 0.50):
        candidate_config = replace(
            config,
            anchor_stable_distance_ceiling=limit,
            anchor_absolute_distance_max=limit,
            support_stable_distance_ceiling=limit,
            support_absolute_distance_max=limit,
            micro_stable_distance_ceiling=min(limit, 0.35),
            micro_absolute_distance_max=min(limit, 0.35),
            unknown_cost=limit + 0.05,
        )
        trace = decode_sequence(
            built.tracklets, built.protected_overlap_spans, states, decision,
            duration_us, candidate_config, resnet_embeddings,
        )
        metrics = _proxy_score(trace.spans, calibration_reference)
        distance_candidates.append((limit, candidate_config, trace, metrics))
    eligible_distance = [
        item for item in distance_candidates
        if item[3]["speaker_accuracy_given_assigned"] >= 0.95
        and item[3]["turn_accuracy_given_covered"] >= 0.90
    ]
    _, selected_config, baseline_trace, baseline_calibration = max(
        eligible_distance or distance_candidates,
        key=lambda item: (item[3]["reference_timeline_coverage"], -item[0]),
    )
    baseline_spans = materialize_labels(baseline_trace.labels, built.tracklets)
    if _span_timeline_sha256(baseline_spans) != _span_timeline_sha256(baseline_trace.spans):
        raise ValueError("E3 materializer does not match canonical baseline timeline")

    tick = time.perf_counter()
    campplus = WeSpeakerCpuEmbeddingBackend(
        _artifact(campplus_path, "wespeaker-campplus-challenger", "speaker_embedding"),
        session=_ort_session(campplus_path, threads=threads),
        max_batch_regions=1,
        audio_provider=provider,
    )
    challenger_embeddings = tuple(campplus.embed(regions))
    challenger_centroids, centroid_diagnostics = build_challenger_centroids(
        anchors, states, challenger_embeddings
    )
    stage["campplus_challenger_wall_sec"] = time.perf_counter() - tick

    tick = time.perf_counter()
    candidates = []
    candidate_spans: dict[str, tuple[RescueSpan, ...]] = {}
    candidate_labels: dict[str, tuple[str, ...]] = {}
    candidate_diagnostics: dict[str, dict[str, int]] = {}
    for distance_limit in DISTANCE_GRID:
        for margin_min in MARGIN_GRID:
            for require_neighbors in (False, True):
                candidate_id = (
                    f"D{round(distance_limit * 100):02d}_M{round(margin_min * 100):02d}_"
                    f"N{int(require_neighbors)}"
                )
                labels, diagnostics = rescue_labels(
                    tracklets=built.tracklets,
                    baseline_labels=baseline_trace.labels,
                    resnet_embeddings=resnet_embeddings,
                    resnet_states=states,
                    challenger_embeddings=challenger_embeddings,
                    challenger_centroids=challenger_centroids,
                    distance_limit=distance_limit,
                    margin_min=margin_min,
                    require_neighbors=require_neighbors,
                    max_gap_us=selected_config.long_gap_reset_us,
                )
                spans = materialize_labels(labels, built.tracklets)
                metrics = _proxy_score(spans, calibration_reference)
                candidates.append({
                    "candidate_id": candidate_id,
                    "distance_limit": distance_limit,
                    "margin_min": margin_min,
                    "require_neighbors": require_neighbors,
                    "calibration_metrics": metrics,
                    "rescued_count": diagnostics["rescued"],
                    "rescued_duration_us": diagnostics["rescued_duration_us"],
                    "span_timeline_sha256": _span_timeline_sha256(spans),
                })
                candidate_spans[candidate_id] = spans
                candidate_labels[candidate_id] = labels
                candidate_diagnostics[candidate_id] = diagnostics
    eligible = [
        item for item in candidates
        if item["calibration_metrics"]["speaker_accuracy_given_assigned"] >= 0.985
        and item["calibration_metrics"]["worst_speaker_accuracy_given_assigned"] >= 0.95
        and item["calibration_metrics"]["turn_accuracy_given_covered"] >= 0.90
    ]
    best_overall = max(
        candidates,
        key=lambda item: (
            item["calibration_metrics"]["speaker_mapped_time_accuracy_end_to_end"],
            item["calibration_metrics"]["reference_timeline_coverage"],
            item["calibration_metrics"]["speaker_accuracy_given_assigned"],
            item["require_neighbors"],
            -item["distance_limit"],
            item["margin_min"],
        ),
    )
    selected = max(
        eligible,
        key=lambda item: (
            item["calibration_metrics"]["speaker_mapped_time_accuracy_end_to_end"],
            item["calibration_metrics"]["reference_timeline_coverage"],
            item["calibration_metrics"]["speaker_accuracy_given_assigned"],
            item["require_neighbors"],
            -item["distance_limit"],
            item["margin_min"],
        ),
    ) if eligible else None
    selected_for_evaluation = selected or best_overall
    selected_id = str(selected_for_evaluation["candidate_id"])
    selected_spans = candidate_spans[selected_id]
    selected_labels = candidate_labels[selected_id]
    # Holdout is evaluated exactly once after calibration-only selection/fallback.
    holdout_metrics = _proxy_score(selected_spans, holdout_reference)
    full_metrics = _proxy_score(selected_spans, reference)
    existing_assigned_changes_us = sum(
        int(tracklet.end_us - tracklet.start_us)
        for tracklet, before, after in zip(built.tracklets, baseline_trace.labels, selected_labels)
        if before in SPEAKERS and after != before
    )
    baseline_full_metrics = _proxy_score(baseline_spans, reference)
    stage["calibration_scoring_wall_sec"] = time.perf_counter() - tick

    elapsed = time.perf_counter() - started
    actual_rtf = elapsed / max(1e-9, duration_us / 1_000_000)
    rtf_limit = 1.5 * c1_rtf
    baseline_complete_merge = baseline_full_metrics["complete_merge"]
    gates = {
        "calibration_candidate_eligible": selected is not None,
        "full_coverage_gte_0_43": full_metrics["reference_timeline_coverage"] >= 0.43,
        "holdout_coverage_gte_0_42": holdout_metrics["reference_timeline_coverage"] >= 0.42,
        "full_accuracy_gte_0_985": full_metrics["speaker_accuracy_given_assigned"] >= 0.985,
        "holdout_accuracy_gte_0_99": holdout_metrics["speaker_accuracy_given_assigned"] >= 0.99,
        "worst_speaker_full_gte_0_95": full_metrics["worst_speaker_accuracy_given_assigned"] >= 0.95,
        "complete_merge_unchanged": full_metrics["complete_merge"] == baseline_complete_merge,
        "h2_unchanged": True,
        "existing_assigned_changes_eq_0": existing_assigned_changes_us == 0,
        "rtf_lte_1_5x_c1": actual_rtf <= rtf_limit,
    }
    accepted = all(gates.values())
    emitted_spans = selected_spans if accepted else baseline_spans
    emitted_metrics = full_metrics if accepted else baseline_full_metrics
    process_cpu = time.process_time() - cpu_started
    result = {
        "schema": "sddiar.campplus_rescue_experiment_v1",
        "result_kind": "DEVELOPMENT_RESNET_AUTHORITY_CAMPLUS_UNKNOWN_RESCUE",
        "audio_sha256_prefix": audio_hash[:12],
        "duration_us": duration_us,
        "authority": {
            "model": "wespeaker-voxceleb-resnet34",
            "sha256": _sha256(resnet_path),
            "decision": decision.state,
            "h2_separation": evaluation.h2_diagnostics.separation,
            "h2_label_stability": evaluation.h2_diagnostics.label_stability,
            "h2_centroid_stability": evaluation.h2_diagnostics.centroid_stability,
            "anchor_count": len(anchors),
            "support_count": len(support),
            "deferred_count": len(deferred),
            "assigned_labels_immutable": True,
        },
        "challenger": {
            "model": "wespeaker-voxceleb-campplus",
            "sha256": _sha256(campplus_path),
            "valid_embedding_count": sum(item.is_valid for item in challenger_embeddings),
            "centroid_authority": "RESNET_H2_ANCHOR_LABELS_AND_WEIGHTS_ONLY",
            "centroid_diagnostics": centroid_diagnostics,
        },
        "calibration": {
            "split_ratio": 0.60,
            "candidate_count": len(candidates),
            "constraints": {
                "overall_accuracy_min": 0.985,
                "worst_speaker_accuracy_min": 0.95,
                "turn_accuracy_min": 0.90,
            },
            "baseline_metrics": baseline_calibration,
            "candidates": candidates,
            "best_overall_candidate": best_overall["candidate_id"],
            "selected_candidate": selected["candidate_id"] if selected else None,
            "evaluated_candidate": selected_id,
        },
        "selected_configuration": {
            "candidate_id": selected_id,
            "distance_limit": selected_for_evaluation["distance_limit"],
            "margin_min": selected_for_evaluation["margin_min"],
            "require_neighbors": selected_for_evaluation["require_neighbors"],
        },
        "selected_aggregate_diagnostics": candidate_diagnostics[selected_id],
        "selected_calibration_metrics": selected_for_evaluation["calibration_metrics"],
        "selected_holdout_metrics": holdout_metrics,
        "selected_full_metrics": full_metrics,
        "existing_assigned_changes_us": existing_assigned_changes_us,
        "baseline_span_timeline_sha256": _span_timeline_sha256(baseline_spans),
        "selected_span_timeline_sha256": _span_timeline_sha256(selected_spans),
        "emitted_span_timeline_sha256": _span_timeline_sha256(emitted_spans),
        "emitted_span_counts": {
            label: sum(span.speaker_id == label for span in emitted_spans)
            for label in (*SPEAKERS, "UNKNOWN", "OVERLAP")
        },
        "emitted_metrics": emitted_metrics,
        "gates": {**gates, "all_passed": accepted},
        "accepted": accepted,
        "rejection_reasons": [name for name, passed in gates.items() if not passed],
        "emitted_configuration": selected_id if accepted else "RESNET_STRICT_BASELINE_FAIL_CLOSED",
        "runtime": {
            "threads": threads,
            "elapsed_wall_sec": round(elapsed, 4),
            "process_cpu_sec": round(process_cpu, 4),
            "cpu_seconds_per_wall_second": process_cpu / max(1e-9, elapsed),
            "rtf": elapsed / max(1e-9, duration_us / 1_000_000),
            "c1_rtf": c1_rtf,
            "rtf_limit": rtf_limit,
            "peak_rss_mb": _rss_mb(),
            "stage_timings": {name: round(value, 6) for name, value in stage.items()},
        },
        "redaction": {
            "source_path": "omitted",
            "transcript": "omitted",
            "audio_samples": "omitted",
            "embedding_vectors": "omitted",
            "centroids": "omitted",
            "candidate_tracklet_ids": "omitted",
        },
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ResNet-authority CAM++ UNKNOWN rescue")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--silero", required=True, type=Path)
    parser.add_argument("--resnet", required=True, type=Path)
    parser.add_argument("--campplus", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--c1-rtf", type=float, default=0.010144)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(
        args.audio, args.silero, args.resnet, args.campplus, args.reference,
        threads=args.threads, c1_rtf=args.c1_rtf,
    )
    if args.output:
        atomic_publish(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
