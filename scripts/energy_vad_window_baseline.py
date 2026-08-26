#!/usr/bin/env python3
"""A deliberately non-oracle, dependency-free CPU diarization baseline.

The reference file is used *only* by :func:`score_predictions`; it is never
read by :func:`predict`.  This makes the resulting number useful as a small
sanity baseline against an external transcript/annotation (rather than an
oracle replay of its labels).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sddiar.media import WavPcmDecoder  # noqa: E402


def _feature(samples: Sequence[float]) -> tuple[float, float, float]:
    if not samples:
        return (0.0, 0.0, 0.0)
    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    zcr = sum((a < 0) != (b < 0) for a, b in zip(samples, samples[1:])) / max(1, len(samples) - 1)
    roughness = sum(abs(b - a) for a, b in zip(samples, samples[1:])) / max(1, len(samples) - 1)
    return (rms, zcr, roughness)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    # Scale dimensions so RMS does not dominate the bounded shape features.
    return sum(((x - y) / scale) ** 2 for x, y, scale in zip(a, b, (0.1, 0.5, 0.1)))


def _cluster(features: Sequence[tuple[float, float, float]]) -> list[int]:
    if not features:
        return []
    # Deterministic two-means, seeded by the two most distant observations.
    first = min(range(len(features)), key=lambda i: features[i])
    second = max(range(len(features)), key=lambda i: _distance(features[i], features[first]))
    centers = [features[first], features[second]]
    labels = [0] * len(features)
    for _ in range(8):
        labels = [0 if _distance(f, centers[0]) <= _distance(f, centers[1]) else 1 for f in features]
        new = []
        for k, old in enumerate(centers):
            group = [f for f, label in zip(features, labels) if label == k]
            new.append(tuple(sum(f[j] for f in group) / len(group) for j in range(3)) if group else old)
        if new == centers:
            break
        centers = new
    return labels


def predict(audio_path: str | Path, *, window_ms: int = 500, hop_ms: int = 500,
            energy_threshold: float = 0.005) -> dict[str, Any]:
    """Produce fixed-window predictions from audio only."""
    if window_ms <= 0 or hop_ms <= 0 or energy_threshold < 0:
        raise ValueError("window_ms, hop_ms and energy_threshold must be positive/non-negative")
    decoder = WavPcmDecoder()
    rows: list[tuple[int, int, tuple[float, float, float]]] = []
    # Streaming decode keeps memory bounded by the decoder chunk, while the
    # small pending buffer handles windows crossing chunk boundaries.
    pending: list[float] = []
    pending_start = 0
    rate: int | None = None
    next_start = 0
    for chunk in decoder.iter_decode_chunks(audio_path):
        rate = chunk.sample_rate_hz
        # Mix down this chunk only; pending is already mono from prior chunks.
        pending.extend(sum(frame) / len(frame) for frame in chunk.samples)
        frame = max(1, round(rate * window_ms / 1000))
        hop = max(1, round(rate * hop_ms / 1000))
        while next_start + frame <= pending_start + len(pending):
            local = next_start - pending_start
            samples = pending[local:local + frame]
            rows.append((next_start, next_start + frame, _feature(samples)))
            next_start += hop
        trim = max(0, next_start - pending_start)
        if trim:
            pending = pending[trim:]
            pending_start += trim
    if rate is None:
        raise ValueError("audio contains no frames")
    if pending:
        end = pending_start + len(pending)
        if end > next_start:
            rows.append((next_start, end, _feature(pending[next_start - pending_start:])))
    speech = [i for i, (_, _, f) in enumerate(rows) if f[0] >= energy_threshold]
    labels = _cluster([rows[i][2] for i in speech])
    # Canonicalize cluster IDs by centroid order, avoiding arbitrary seed IDs.
    cluster_to_speaker = {k: f"H{k + 1}" for k in range(2)}
    predictions = []
    for i, (start, end, feat) in enumerate(rows):
        if feat[0] < energy_threshold:
            speaker = "UNKNOWN"
        else:
            speaker = cluster_to_speaker[labels[speech.index(i)]]
        predictions.append({"start_us": round(start * 1_000_000 / rate),
                            "end_us": round(end * 1_000_000 / rate), "speaker": speaker,
                            "energy": round(feat[0], 7)})
    speech_us = sum(p["end_us"] - p["start_us"] for p in predictions if p["speaker"] != "UNKNOWN")
    unknown_us = sum(p["end_us"] - p["start_us"] for p in predictions if p["speaker"] == "UNKNOWN")
    return {"baseline": "NON_ORACLE_ENERGY_BASELINE", "sample_rate_hz": rate,
            "window_ms": window_ms, "hop_ms": hop_ms, "predictions": predictions,
            "coverage": speech_us / max(1, speech_us + unknown_us),
            "unknown_rate": unknown_us / max(1, speech_us + unknown_us)}


def _reference(value: Any) -> list[dict[str, Any]]:
    entries = value.get("segments", value.get("turns", value)) if isinstance(value, dict) else value
    out = []
    for item in entries:
        start = item.get("start_us")
        end = item.get("end_us")
        if start is None:
            start_seconds = item.get("start_sec", item.get("start"))
            if start_seconds is None:
                raise ValueError("reference entry is missing start time")
            start = round(float(start_seconds) * 1_000_000)
        if end is None:
            end_seconds = item.get("end_sec", item.get("end"))
            if end_seconds is None:
                raise ValueError("reference entry is missing end time")
            end = round(float(end_seconds) * 1_000_000)
        out.append({"start_us": int(start), "end_us": int(end),
                    "speaker": str(item.get("speaker", item.get("speaker_id", "UNKNOWN")))})
    return [x for x in out if x["end_us"] > x["start_us"]]


def score_predictions(result: dict[str, Any], reference: Sequence[dict[str, Any]]) -> dict[str, float | str]:
    pred = result["predictions"]
    ref_duration = sum(r["end_us"] - r["start_us"] for r in reference if r["speaker"] != "UNKNOWN")
    overlap = correct = predicted_speech = unknown = 0
    pair = {("H1", r["speaker"]): 0 for r in reference} | {("H2", r["speaker"]): 0 for r in reference}
    for p in pred:
        dur = p["end_us"] - p["start_us"]
        if p["speaker"] == "UNKNOWN": unknown += dur; continue
        predicted_speech += dur
        for r in reference:
            ov = max(0, min(p["end_us"], r["end_us"]) - max(p["start_us"], r["start_us"]))
            if ov: pair[(p["speaker"], r["speaker"])] = pair.get((p["speaker"], r["speaker"]), 0) + ov
    speakers = sorted({r["speaker"] for r in reference if r["speaker"] != "UNKNOWN"})
    mappings = [({"H1": speakers[0], "H2": speakers[1]} if len(speakers) > 1 else {"H1": speakers[0] if speakers else "UNKNOWN", "H2": "__none__"}),
                ({"H1": speakers[1], "H2": speakers[0]} if len(speakers) > 1 else {})]
    mapping = max(mappings, key=lambda m: sum(pair.get((h, s), 0) for h, s in m.items()))
    for p in pred:
        if p["speaker"] == "UNKNOWN": continue
        mapped = mapping.get(p["speaker"])
        for r in reference:
            ov = max(0, min(p["end_us"], r["end_us"]) - max(p["start_us"], r["start_us"]))
            overlap += ov
            if ov and mapped == r["speaker"]: correct += ov
    return {"baseline": "NON_ORACLE_ENERGY_BASELINE", "coverage": overlap / max(1, ref_duration),
            "speaker_mapped_frame_accuracy": correct / max(1, ref_duration),
            "unknown_rate": unknown / max(1, predicted_speech + unknown), "mapping": mapping}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--window-ms", type=int, default=500)
    ap.add_argument("--hop-ms", type=int, default=500)
    ap.add_argument("--energy-threshold", type=float, default=0.005)
    args = ap.parse_args(argv)
    result = predict(args.audio, window_ms=args.window_ms, hop_ms=args.hop_ms,
                     energy_threshold=args.energy_threshold)
    if args.reference:
        result["score"] = score_predictions(result, _reference(json.loads(args.reference.read_text())))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
