#!/usr/bin/env python3
"""Redacted Silero threshold sweep against pseudonymous turn timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from run_onnx_diarization_experiment import _ort_session  # noqa: E402
from sddiar.media import WavPcmAccessor, WavPcmDecoder  # noqa: E402
from sddiar.segmentation import RuleEvidenceSegmentation, SegmentationConfig  # noqa: E402
from sddiar.service import atomic_publish  # noqa: E402
from sddiar.silero_runtime import SileroOnnxRuntime  # noqa: E402
from sddiar.silero_temporal import SileroTemporalPostprocessor  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _reference(path: Path) -> list[dict[str, object]]:
    turns = json.loads(path.read_text(encoding="utf-8"))["turns"]
    return [
        {
            "speaker_id": str(row["speaker_id"]),
            "start_us": round(float(row["start_sec"]) * 1_000_000),
            "end_us": round(float(row["end_sec"]) * 1_000_000),
        }
        for row in turns
    ]


def _overlap_us(regions, rows, speaker: str) -> int:
    return sum(
        max(0, min(region.end_us, int(row["end_us"])) - max(region.start_us, int(row["start_us"])))
        for region in regions
        for row in rows
        if row["speaker_id"] == speaker
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--model-sha256", required=True)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    if args.threads != 1:
        raise ValueError("1-CPU sweep requires one ORT thread")
    if _sha256(args.model) != args.model_sha256.lower():
        raise ValueError("Silero SHA-256 mismatch")

    accessor = WavPcmAccessor(args.audio)
    decoder = WavPcmDecoder()
    started = time.perf_counter()
    runtime = SileroOnnxRuntime(
        args.model,
        session=_ort_session(args.model, threads=1),
        threshold=0.5,
    )
    fast_decode = getattr(decoder, "iter_decode_chunks_numpy", None)
    chunks = (
        fast_decode(args.audio, frames_per_chunk=240_000)
        if callable(fast_decode)
        else decoder.iter_decode_chunks(args.audio, frames_per_chunk=240_000)
    )
    frames = runtime.infer_chunk_stream(chunks)
    vad_wall = time.perf_counter() - started
    rows = _reference(args.reference)
    speakers = sorted({str(row["speaker_id"]) for row in rows})

    frame_probabilities = {speaker: [] for speaker in speakers}
    cursor = 0
    for frame in frames:
        middle = (frame.start_us + frame.end_us) // 2
        while cursor + 1 < len(rows) and middle >= int(rows[cursor]["end_us"]):
            cursor += 1
        frame_probabilities[str(rows[cursor]["speaker_id"])].append(float(frame.speech_evidence or 0.0))

    thresholds = []
    duration_us = round(accessor.layout.frame_count * 1_000_000 / accessor.layout.sample_rate_hz)
    for threshold in (0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
        marked = tuple(replace(frame, is_speech=float(frame.speech_evidence or 0.0) >= threshold) for frame in frames)
        evidence = RuleEvidenceSegmentation(SegmentationConfig(vad_merge_gap_us=200_000)).build(
            view_id="audio", vad_frames=marked
        )
        speech_us = sum(region.end_us - region.start_us for region in evidence.speech_regions)
        per_speaker = {}
        for speaker in speakers:
            denominator = sum(
                int(row["end_us"]) - int(row["start_us"])
                for row in rows if row["speaker_id"] == speaker
            )
            per_speaker[speaker] = _overlap_us(evidence.speech_regions, rows, speaker) / max(1, denominator)
        thresholds.append({
            "threshold": threshold,
            "speech_region_count": len(evidence.speech_regions),
            "speech_ratio": speech_us / max(1, duration_us),
            "reference_timeline_intersection_by_speaker": per_speaker,
        })

    probability_summary = {}
    for speaker, values in frame_probabilities.items():
        array = np.asarray(values, dtype=np.float32)
        probability_summary[speaker] = {
            "frame_count": int(array.size),
            "p10_p25_p50_p75_p90": [float(np.percentile(array, q)) for q in (10, 25, 50, 75, 90)],
            "positive_rate_at_0_5": float(np.mean(array >= 0.5)),
        }
    temporal = SileroTemporalPostprocessor().process(frames)
    temporal_speech_us = sum(region.end_us - region.start_us for region in temporal.regions)
    temporal_by_speaker = {}
    for speaker in speakers:
        denominator = sum(
            int(row["end_us"]) - int(row["start_us"])
            for row in rows if row["speaker_id"] == speaker
        )
        temporal_by_speaker[speaker] = _overlap_us(temporal.regions, rows, speaker) / max(1, denominator)
    result = {
        "schema": "sddiar_vad_threshold_sweep_v1",
        "audio_sha256_prefix": _sha256(args.audio)[:12],
        "silero_sha256": _sha256(args.model),
        "duration_sec": duration_us / 1_000_000,
        "vad_wall_sec": vad_wall,
        "frame_count": len(frames),
        "probability_summary_by_pseudonymous_speaker_timeline": probability_summary,
        "upstream_temporal_profile": {
            "region_count": len(temporal.regions),
            "speech_ratio": temporal_speech_us / max(1, duration_us),
            "core_duration_us": temporal.core_duration_us,
            "halo_duration_us": temporal.halo_duration_us,
            "reference_timeline_intersection_by_speaker": temporal_by_speaker,
        },
        "thresholds": thresholds,
        "scope_limit": "Clova turns label inter-turn silence; intersection is not ground-truth VAD recall.",
    }
    atomic_publish(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
