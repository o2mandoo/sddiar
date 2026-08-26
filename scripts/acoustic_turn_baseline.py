#!/usr/bin/env python3
"""Development-only turn acoustic clustering proxy.

This is deliberately *not* a diarizer: turn boundaries are supplied by an
oracle reference file, while speaker labels are consumed only after acoustic
clustering has produced its labels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sddiar.media import WavPcmDecoder  # noqa: E402


RESULT_KIND = "ORACLE_BOUNDARY_PROXY"


def load_turns(path: str | Path) -> list[tuple[int, int]]:
    """Load frame boundaries from ``{turns:[{start_frame,end_frame}]}``."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = obj.get("turns", obj) if isinstance(obj, dict) else obj
    turns: list[tuple[int, int]] = []
    for row in rows:
        start = row.get("start_frame", row.get("start")) if isinstance(row, dict) else row[0]
        end = row.get("end_frame", row.get("end")) if isinstance(row, dict) else row[1]
        if start is None or end is None or int(end) <= int(start):
            raise ValueError("turn boundaries must have positive start/end")
        turns.append((int(start), int(end)))
    if not turns:
        raise ValueError("reference turn file contains no turns")
    if turns != sorted(turns) or any(right[0] < left[1] for left, right in zip(turns, turns[1:])):
        raise ValueError("turn boundaries must be ordered and non-overlapping")
    return turns


@dataclass
class _Stats:
    count: int = 0
    energy: float = 0.0
    delta: float = 0.0
    crossings: int = 0
    peak: float = 0.0
    previous: float | None = None

    def add(self, samples: Sequence[tuple[float, ...]]) -> None:
        for frame in samples:
            sample = frame[0]
            self.count += 1
            self.energy += sample * sample
            self.peak = max(self.peak, abs(sample))
            if self.previous is not None:
                self.delta += abs(sample - self.previous)
                if (self.previous < 0 <= sample) or (self.previous >= 0 > sample):
                    self.crossings += 1
            self.previous = sample

    def feature(self, rate: int) -> list[float]:
        if self.count < 2:
            raise ValueError("turn is too short for acoustic features")
        rms = math.sqrt(self.energy / self.count)
        return [rms, self.crossings / (self.count / rate), self.delta / (self.count - 1), self.peak / (rms or 1.0)]


def turn_features(path: str | Path, turns: Sequence[tuple[int, int]], chunk_frames: int = 240_000) -> list[list[float]]:
    """One bounded sequential pass over PCM16 WAV using the project decoder."""
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    stats = [_Stats() for _ in turns]
    turn_index = 0
    rate: int | None = None
    total_frames = 0
    for chunk in WavPcmDecoder().iter_decode_chunks(path, frames_per_chunk=chunk_frames):
        rate = chunk.sample_rate_hz
        total_frames = chunk.source_end_sample
        while turn_index < len(turns) and turns[turn_index][1] <= chunk.source_start_sample:
            turn_index += 1
        index = turn_index
        while index < len(turns) and turns[index][0] < chunk.source_end_sample:
            start, end = turns[index]
            local_start = max(start, chunk.source_start_sample) - chunk.source_start_sample
            local_end = min(end, chunk.source_end_sample) - chunk.source_start_sample
            if local_end > local_start:
                stats[index].add(chunk.samples[local_start:local_end])
            if end <= chunk.source_end_sample:
                index += 1
            else:
                break
        turn_index = index
    if rate is None:
        raise ValueError("audio contains no frames")
    if any(end > total_frames for _, end in turns):
        raise ValueError("turn boundary outside WAV")
    return [stat.feature(rate) for stat in stats]


def _unit(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def kmeans(features: Sequence[Sequence[float]], k: int = 2, iterations: int = 32) -> list[int]:
    """Deterministic normalized-feature kmeans; labels are intentionally unused."""
    if not features or k not in (1, 2) or k > len(features):
        raise ValueError("k must be 1 or 2 and no greater than turn count")
    vectors = [_unit(feature) for feature in features]
    if k == 1:
        return [0] * len(vectors)
    first, second, best = 0, 1, -1.0
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            distance = sum((vectors[left][index] - vectors[right][index]) ** 2 for index in range(len(vectors[left])))
            if distance > best:
                first, second, best = left, right, distance
    centroids = [vectors[first][:], vectors[second][:]]
    assignments = [-1] * len(vectors)
    for _ in range(iterations):
        updated = [0 if sum((value - center) ** 2 for value, center in zip(vector, centroids[0])) <= sum((value - center) ** 2 for value, center in zip(vector, centroids[1])) else 1 for vector in vectors]
        if updated == assignments:
            break
        assignments = updated
        for cluster in range(2):
            members = [vector for vector, label in zip(vectors, assignments) if label == cluster]
            if members:
                centroids[cluster] = _unit([sum(vector[index] for vector in members) for index in range(len(vectors[0]))])
    return assignments


def predict(wav: str | Path, boundaries: str | Path, k: int = 2, chunk_frames: int = 240_000) -> list[int]:
    """Cluster audio from oracle boundaries without reading any reference labels."""
    return kmeans(turn_features(wav, load_turns(boundaries), chunk_frames), k)


def score(predicted: Sequence[int], labels: Sequence[int]) -> dict[str, float | int | str]:
    """Read labels only after clustering, using permutation-invariant metrics."""
    if len(predicted) != len(labels) or not labels:
        raise ValueError("predicted and labels must have equal nonzero length")
    mappings = ((0, 1), (1, 0))
    accuracy = max(sum(mapping[predicted_label] == label for predicted_label, label in zip(predicted, labels)) / len(labels) for mapping in mappings)
    pairs = [(left, right) for left in range(len(labels)) for right in range(left + 1, len(labels))]
    true_positive = sum(predicted[left] == predicted[right] and labels[left] == labels[right] for left, right in pairs)
    predicted_same = sum(predicted[left] == predicted[right] for left, right in pairs)
    reference_same = sum(labels[left] == labels[right] for left, right in pairs)
    return {
        "result_kind": RESULT_KIND,
        "turns": len(labels),
        "cluster_count": len(set(predicted)),
        "cluster_accuracy": accuracy,
        "pairwise_precision": true_positive / predicted_same if predicted_same else 0.0,
        "pairwise_recall": true_positive / reference_same if reference_same else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-only ORACLE_BOUNDARY_PROXY acoustic experiment")
    parser.add_argument("wav")
    parser.add_argument("boundaries", help="turn-boundaries JSON; no speaker labels")
    parser.add_argument("--labels", required=True, help="held-out labels JSON, read after clustering")
    parser.add_argument("--k", type=int, choices=(1, 2), default=2)
    parser.add_argument("--chunk-frames", type=int, default=240_000)
    args = parser.parse_args()
    predicted = predict(args.wav, args.boundaries, args.k, args.chunk_frames)
    label_object = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    labels = label_object.get("labels", label_object) if isinstance(label_object, dict) else label_object
    print(json.dumps(score(predicted, [int(value) for value in labels]), sort_keys=True))


if __name__ == "__main__":
    main()
