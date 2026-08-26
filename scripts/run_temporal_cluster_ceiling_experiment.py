#!/usr/bin/env python3
"""Final same-file E4: temporal-VAD ResNet authority with cluster ceilings."""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_campplus_rescue_experiment import (  # noqa: E402
    SPEAKERS,
    _proxy_score,
    materialize_labels,
)
from run_onnx_diarization_experiment import (  # noqa: E402
    _artifact,
    _clip_reference,
    _ort_session,
    _reference,
    _sha256,
    _span_timeline_sha256,
)
from sddiar.benchmark import peak_rss_bytes  # noqa: E402
from sddiar.contracts import EmbeddingRegion, EmbeddingResult, SpeechRegion as ContractSpeechRegion  # noqa: E402
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
from sddiar.segmentation import SegmentationEvidence  # noqa: E402
from sddiar.service import atomic_publish  # noqa: E402
from sddiar.silero_runtime import SileroOnnxRuntime  # noqa: E402
from sddiar.silero_temporal import SileroTemporalPostprocessor  # noqa: E402
from sddiar.wespeaker_runtime import WeSpeakerCpuEmbeddingBackend  # noqa: E402


CEILINGS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)


def _rss_mb() -> float | None:
    value = peak_rss_bytes()
    return round(value / (1024 * 1024), 2) if value is not None else None


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


def rescue_with_cluster_ceilings(
    *,
    tracklets: Sequence[Any],
    baseline_labels: Sequence[str],
    baseline_local_assignments: Sequence[Any],
    embeddings: Sequence[EmbeddingResult],
    states: Mapping[str, Any],
    config: DiarizationConfig,
    ceilings: Mapping[str, float],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Only rescue local-distance-gated UNKNOWN into its same nearest state."""

    by_tracklet = {item.tracklet_id: item for item in embeddings}
    centroids = {speaker_id: state.stable_anchor_centroid for speaker_id, state in states.items()}
    labels = list(baseline_labels)
    counters: dict[str, Any] = {
        "baseline_unknown": 0,
        "micro_locked": 0,
        "viterbi_unknown_locked": 0,
        "invalid_or_non_distance_unknown": 0,
        "distance_failed": 0,
        "margin_failed": 0,
        "rescued": 0,
        "rescued_duration_us": 0,
        "rescued_by_speaker": {speaker_id: {"count": 0, "duration_us": 0} for speaker_id in SPEAKERS},
    }
    for index, (tracklet, baseline_label, local) in enumerate(
        zip(tracklets, baseline_labels, baseline_local_assignments)
    ):
        if baseline_label != "UNKNOWN":
            continue
        counters["baseline_unknown"] += 1
        if tracklet.kind == "MICRO":
            counters["micro_locked"] += 1
            continue
        if local.speaker_id in SPEAKERS:
            counters["viterbi_unknown_locked"] += 1
            continue
        if "LOCAL_GATE_FAILED" not in local.reason_codes:
            counters["invalid_or_non_distance_unknown"] += 1
            continue
        embedding = by_tracklet.get(tracklet.tracklet_id)
        if embedding is None or not embedding.is_valid or embedding.vector is None:
            counters["invalid_or_non_distance_unknown"] += 1
            continue
        scored = sorted(
            ((speaker_id, cosine_distance(embedding.vector, centroid)) for speaker_id, centroid in centroids.items()),
            key=lambda item: (item[1], item[0]),
        )
        speaker_id, distance = scored[0]
        margin = scored[1][1] - distance
        ceiling = float(ceilings[speaker_id])
        if distance > ceiling:
            counters["distance_failed"] += 1
            continue
        margin_min = float(getattr(config, f"{str(tracklet.kind).lower()}_margin_min"))
        if margin < margin_min:
            counters["margin_failed"] += 1
            continue
        labels[index] = speaker_id
        duration_us = int(tracklet.end_us - tracklet.start_us)
        counters["rescued"] += 1
        counters["rescued_duration_us"] += duration_us
        counters["rescued_by_speaker"][speaker_id]["count"] += 1
        counters["rescued_by_speaker"][speaker_id]["duration_us"] += duration_us
    return tuple(labels), counters


def run_experiment(
    audio_path: str | Path,
    silero_model: str | Path,
    resnet_model: str | Path,
    reference_path: str | Path,
    *,
    threads: int = 1,
    temporal_baseline_rtf: float = 0.010161,
) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    stage: dict[str, float] = {}
    audio = Path(audio_path)
    silero_path, resnet_path = Path(silero_model), Path(resnet_model)
    reference = _reference(Path(reference_path))
    accessor = WavPcmAccessor(audio)
    if accessor.layout.sample_rate_hz != 16_000 or accessor.layout.channel_count != 1 or accessor.layout.sample_width_bytes != 2:
        raise ValueError("E4 requires PCM16 mono 16 kHz WAV")
    if not math.isfinite(temporal_baseline_rtf) or temporal_baseline_rtf <= 0:
        raise ValueError("temporal_baseline_rtf must be positive and finite")
    frame_limit = accessor.layout.frame_count
    duration_us = round(frame_limit * 1_000_000 / accessor.layout.sample_rate_hz)
    audio_hash = _sha256(audio)

    tick = time.perf_counter()
    silero = SileroOnnxRuntime(
        silero_path, session=_ort_session(silero_path, threads=threads), threshold=0.5
    )
    vad_frames = silero.infer_chunk_stream(_iter_chunks(audio, frame_limit))
    temporal = SileroTemporalPostprocessor().process(vad_frames)
    temporal_regions = tuple(
        ContractSpeechRegion(
            f"speech-temporal-{index}", "audio", region.start_us, region.end_us,
            reason_codes=("SILERO_TEMPORAL_PADDED",),
        )
        for index, region in enumerate(temporal.regions)
    )
    segmentation = SegmentationEvidence(temporal_regions, (), (), ())
    config = DiarizationConfig(max_tracklet_us=3_000_000, lambda_k2=0.0, h2_min_cost_gain=0.02)
    built = build_tracklets(segmentation.speech_regions, cfg=config, audio_id=audio_hash[:16])
    regions = tuple(
        EmbeddingRegion(
            f"embedding-{index:06d}", tracklet.tracklet_id,
            tracklet.start_us, tracklet.end_us, tracklet.clean_speech_us,
            min(1.0, tracklet.clean_speech_us / max(1, tracklet.end_us - tracklet.start_us)),
        )
        for index, tracklet in enumerate(built.tracklets)
    )
    stage["temporal_vad_segmentation_wall_sec"] = time.perf_counter() - tick

    tick = time.perf_counter()
    fast_reader = getattr(accessor, "read_mono_samples_numpy", None)
    provider = (
        (lambda region: fast_reader(region.start_us, region.end_us))
        if callable(fast_reader)
        else (lambda region: accessor.read_mono_samples(region.start_us, region.end_us))
    )
    backend = WeSpeakerCpuEmbeddingBackend(
        _artifact(resnet_path, "wespeaker-resnet34", "speaker_embedding"),
        session=_ort_session(resnet_path, threads=threads),
        max_batch_regions=1,
        audio_provider=provider,
    )
    embeddings = tuple(backend.embed(regions))
    stage["resnet_embeddings_wall_sec"] = time.perf_counter() - tick

    tick = time.perf_counter()
    anchors, support, deferred = select_anchor_evidence(built.tracklets, embeddings, config)
    evaluation = evaluate_hypotheses(anchors, config)
    decision = evaluation.decision
    if decision.state != "H2_CONFIRMED":
        raise ValueError("E4 requires fixed temporal ResNet H2 authority")
    states = speaker_states_from_decision(decision, anchors)
    baseline_config = replace(
        config,
        anchor_stable_distance_ceiling=0.50,
        anchor_absolute_distance_max=0.50,
        support_stable_distance_ceiling=0.50,
        support_absolute_distance_max=0.50,
        micro_stable_distance_ceiling=0.35,
        micro_absolute_distance_max=0.35,
        unknown_cost=0.55,
    )
    baseline_trace = decode_sequence(
        built.tracklets, built.protected_overlap_spans, states, decision,
        duration_us, baseline_config, embeddings,
    )
    baseline_spans = materialize_labels(baseline_trace.labels, built.tracklets)
    if _span_timeline_sha256(baseline_spans) != _span_timeline_sha256(baseline_trace.spans):
        raise ValueError("E4 materializer does not match temporal baseline timeline")
    stage["hypothesis_baseline_decode_wall_sec"] = time.perf_counter() - tick

    split_us = round(duration_us * 0.60)
    calibration_reference = _clip_reference(reference, 0, split_us)
    holdout_reference = _clip_reference(reference, split_us, duration_us)
    baseline_calibration = _proxy_score(baseline_spans, calibration_reference)
    baseline_full = _proxy_score(baseline_spans, reference)

    tick = time.perf_counter()
    candidates = []
    labels_by_id: dict[str, tuple[str, ...]] = {}
    spans_by_id: dict[str, tuple[Any, ...]] = {}
    diagnostics_by_id: dict[str, dict[str, Any]] = {}
    single_apply_durations = []
    for speaker_00_limit in CEILINGS:
        for speaker_01_limit in CEILINGS:
            apply_started = time.perf_counter()
            candidate_id = f"S00_{round(speaker_00_limit * 100):02d}_S01_{round(speaker_01_limit * 100):02d}"
            labels, diagnostics = rescue_with_cluster_ceilings(
                tracklets=built.tracklets,
                baseline_labels=baseline_trace.labels,
                baseline_local_assignments=baseline_trace.local_assignments,
                embeddings=embeddings,
                states=states,
                config=baseline_config,
                ceilings={"SPEAKER_00": speaker_00_limit, "SPEAKER_01": speaker_01_limit},
            )
            spans = materialize_labels(labels, built.tracklets)
            single_apply_durations.append(time.perf_counter() - apply_started)
            metrics = _proxy_score(spans, calibration_reference)
            candidates.append({
                "candidate_id": candidate_id,
                "speaker_00_ceiling": speaker_00_limit,
                "speaker_01_ceiling": speaker_01_limit,
                "calibration_metrics": metrics,
                "rescued_count": diagnostics["rescued"],
                "rescued_duration_us": diagnostics["rescued_duration_us"],
                "span_timeline_sha256": _span_timeline_sha256(spans),
            })
            labels_by_id[candidate_id] = labels
            spans_by_id[candidate_id] = spans
            diagnostics_by_id[candidate_id] = diagnostics
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
            -item["speaker_00_ceiling"],
            -item["speaker_01_ceiling"],
        ),
    )
    selected = max(
        eligible,
        key=lambda item: (
            item["calibration_metrics"]["speaker_mapped_time_accuracy_end_to_end"],
            item["calibration_metrics"]["reference_timeline_coverage"],
            item["calibration_metrics"]["speaker_accuracy_given_assigned"],
            -item["speaker_00_ceiling"],
            -item["speaker_01_ceiling"],
        ),
    ) if eligible else None
    evaluated = selected or best_overall
    evaluated_id = str(evaluated["candidate_id"])
    evaluated_spans = spans_by_id[evaluated_id]
    evaluated_labels = labels_by_id[evaluated_id]
    # Holdout is touched once, only after calibration selection/fallback.
    holdout_metrics = _proxy_score(evaluated_spans, holdout_reference)
    full_metrics = _proxy_score(evaluated_spans, reference)
    stage["calibration_grid_wall_sec"] = time.perf_counter() - tick

    existing_assigned_flips_us = sum(
        int(tracklet.end_us - tracklet.start_us)
        for tracklet, before, after in zip(built.tracklets, baseline_trace.labels, evaluated_labels)
        if before in SPEAKERS and after != before
    )
    projected_operational_wall = (
        stage["temporal_vad_segmentation_wall_sec"]
        + stage["resnet_embeddings_wall_sec"]
        + stage["hypothesis_baseline_decode_wall_sec"]
        + max(single_apply_durations, default=0.0)
    )
    projected_operational_rtf = projected_operational_wall / max(1e-9, duration_us / 1_000_000)
    rtf_limit = 1.05 * temporal_baseline_rtf
    gates = {
        "calibration_candidate_eligible": selected is not None,
        "full_coverage_gte_0_43": full_metrics["reference_timeline_coverage"] >= 0.43,
        "holdout_coverage_gte_0_42": holdout_metrics["reference_timeline_coverage"] >= 0.42,
        "full_accuracy_gte_0_985": full_metrics["speaker_accuracy_given_assigned"] >= 0.985,
        "holdout_accuracy_gte_0_99": holdout_metrics["speaker_accuracy_given_assigned"] >= 0.99,
        "worst_speaker_full_gte_0_95": full_metrics["worst_speaker_accuracy_given_assigned"] >= 0.95,
        "worst_speaker_holdout_gte_0_95": holdout_metrics["worst_speaker_accuracy_given_assigned"] >= 0.95,
        "complete_merge_unchanged": full_metrics["complete_merge"] == baseline_full["complete_merge"],
        "h2_unchanged": True,
        "existing_assigned_speaker_flips_eq_0": existing_assigned_flips_us == 0,
        "projected_rtf_lte_1_05x_temporal": projected_operational_rtf <= rtf_limit,
    }
    accepted = all(gates.values())
    emitted_spans = evaluated_spans if accepted else baseline_spans
    emitted_metrics = full_metrics if accepted else baseline_full
    elapsed = time.perf_counter() - started
    process_cpu = time.process_time() - cpu_started
    same_file_status = (
        "ACCEPTED"
        if accepted
        else "STOP_SAME_FILE_THRESHOLD_TUNING_DATA_BLOCKER"
        if selected is None
        else "REJECTED_BY_FINAL_GATES"
    )
    return {
        "schema": "sddiar.temporal_cluster_ceiling_experiment_v1",
        "result_kind": "DEVELOPMENT_TEMPORAL_RESNET_PER_CLUSTER_CEILING",
        "audio_sha256_prefix": audio_hash[:12],
        "duration_us": duration_us,
        "temporal_vad": {
            "core_duration_us": temporal.core_duration_us,
            "halo_duration_us": temporal.halo_duration_us,
            "region_count": len(temporal.regions),
        },
        "authority": {
            "model_sha256": _sha256(resnet_path),
            "decision": decision.state,
            "h2_separation": evaluation.h2_diagnostics.separation,
            "h2_label_stability": evaluation.h2_diagnostics.label_stability,
            "h2_centroid_stability": evaluation.h2_diagnostics.centroid_stability,
            "anchor_count": len(anchors),
            "support_count": len(support),
            "deferred_count": len(deferred),
            "assigned_labels_immutable": True,
            "micro_ceiling": 0.35,
            "margins_unchanged": True,
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
            "evaluated_candidate": evaluated_id,
        },
        "evaluated_configuration": {
            "candidate_id": evaluated_id,
            "speaker_00_ceiling": evaluated["speaker_00_ceiling"],
            "speaker_01_ceiling": evaluated["speaker_01_ceiling"],
        },
        "evaluated_aggregate_diagnostics": diagnostics_by_id[evaluated_id],
        "evaluated_calibration_metrics": evaluated["calibration_metrics"],
        "evaluated_holdout_metrics": holdout_metrics,
        "evaluated_full_metrics": full_metrics,
        "existing_assigned_speaker_flips_us": existing_assigned_flips_us,
        "baseline_span_timeline_sha256": _span_timeline_sha256(baseline_spans),
        "evaluated_span_timeline_sha256": _span_timeline_sha256(evaluated_spans),
        "emitted_span_timeline_sha256": _span_timeline_sha256(emitted_spans),
        "emitted_metrics": emitted_metrics,
        "gates": {**gates, "all_passed": accepted},
        "accepted": accepted,
        "rejection_reasons": [name for name, passed in gates.items() if not passed],
        "same_file_tuning_status": same_file_status,
        "emitted_configuration": evaluated_id if accepted else "TEMPORAL_RESNET_BASELINE_FAIL_CLOSED",
        "runtime": {
            "threads": threads,
            "elapsed_experiment_wall_sec": elapsed,
            "actual_experiment_rtf": elapsed / max(1e-9, duration_us / 1_000_000),
            "process_cpu_sec": process_cpu,
            "cpu_seconds_per_wall_second": process_cpu / max(1e-9, elapsed),
            "projected_operational_wall_sec": projected_operational_wall,
            "projected_operational_rtf": projected_operational_rtf,
            "temporal_baseline_rtf": temporal_baseline_rtf,
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
            "tracklet_ids": "omitted",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run temporal ResNet per-cluster ceiling E4")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--silero", required=True, type=Path)
    parser.add_argument("--resnet", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--temporal-baseline-rtf", type=float, default=0.010161)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(
        args.audio, args.silero, args.resnet, args.reference,
        threads=args.threads, temporal_baseline_rtf=args.temporal_baseline_rtf,
    )
    if args.output:
        atomic_publish(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
