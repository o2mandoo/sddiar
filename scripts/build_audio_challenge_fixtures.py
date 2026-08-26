#!/usr/bin/env python3
"""Build deterministic, offline audio challenge fixtures.

The fixture set is deliberately timing-only.  A reference file is consumed to
record an opaque source-time contract, but no speaker label (or other field)
from that file participates in an audio transform.  The implementation uses
the project's strict RIFF parser plus standard-library bounded PCM writes.

Examples::

    PYTHONPATH=src python scripts/build_audio_challenge_fixtures.py \
        input.wav --timing-reference timing.json --output-dir fixtures
    PYTHONPATH=src python scripts/build_audio_challenge_fixtures.py \
        input.wav --timing-reference timing.json --output-dir fixtures --plan-only
    python scripts/build_audio_challenge_fixtures.py --evaluate fixtures/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from sddiar.media import MediaError, WavPcmAccessor


SCHEMA_VERSION = "audio-challenge-fixture/v1"
TARGET_RATE_HZ = 16_000
INPUT_RATE_HZ = 16_000
PCM16_MIN = -32_768
PCM16_MAX = 32_767
DEFAULT_CHUNK_FRAMES = 16_384
MAX_CHUNK_FRAMES = 1 << 20
DEFAULT_NOISE_SEED = 20_260_826
DEFAULT_NOISE_SNR_DB = 20.0
GAIN_DB = -12.0
_MASK64 = (1 << 64) - 1
_PRNG_NONZERO = 0x9E3779B97F4A7C15
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15


class FixtureError(ValueError):
    """Raised for an invalid source, reference, or output contract."""


@dataclass(frozen=True)
class WavInfo:
    sha256: str
    sample_rate_hz: int
    channel_count: int
    sample_width_bytes: int
    frame_count: int
    duration_us: int


@dataclass(frozen=True)
class TimingInfo:
    sha256: str
    timing_digest_sha256: str
    interval_count: int
    max_end_us: int


@dataclass(frozen=True)
class ArtifactPlan:
    variant: str
    artifact_id: str
    filename: str
    relationship_id: str
    transform: Mapping[str, Any]


@dataclass(frozen=True)
class FixturePlan:
    source: WavInfo
    timing: TimingInfo
    relationship_id: str
    chunk_frames: int
    noise_seed: int
    noise_snr_db: float
    artifacts: tuple[ArtifactPlan, ...]


def _sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def inspect_wav(path: str | os.PathLike[str]) -> WavInfo:
    """Validate the supported input contract without loading its PCM payload."""
    source = Path(path)
    if not source.is_file():
        raise FixtureError(f"source WAV does not exist: {source}")
    try:
        layout = WavPcmAccessor(source).layout
        rate = layout.sample_rate_hz
        channels = layout.channel_count
        width = layout.sample_width_bytes
        frames = layout.frame_count
    except (EOFError, OSError, MediaError) as exc:
        raise FixtureError(f"invalid WAV source: {source}") from exc
    if (rate, channels, width) != (INPUT_RATE_HZ, 1, 2):
        raise FixtureError("source must be uncompressed PCM16 mono 16 kHz WAV")
    if frames <= 0:
        raise FixtureError("source WAV has no frames")
    return WavInfo(
        sha256=_sha256_file(source),
        sample_rate_hz=rate,
        channel_count=channels,
        sample_width_bytes=width,
        frame_count=frames,
        duration_us=(frames * 1_000_000) // rate,
    )


def _number(value: Any, *, unit: str) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value_f = float(value)
    if not math.isfinite(value_f) or value_f < 0:
        return None
    if unit == "us":
        return int(round(value_f))
    if unit == "ms":
        return int(round(value_f * 1_000.0))
    if unit == "s":
        return int(round(value_f * 1_000_000.0))
    raise AssertionError(unit)


def _timing_pair(value: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return a timing pair only; this function intentionally never reads labels."""
    aliases = (
        (("start_us", "end_us"), "us"),
        (("start_ms", "end_ms"), "ms"),
        (("start_seconds", "end_seconds"), "s"),
        (("start_sec", "end_sec"), "s"),
        (("start_time_us", "end_time_us"), "us"),
        (("start_time_ms", "end_time_ms"), "ms"),
        (("start_time", "end_time"), "s"),
        (("start", "end"), "s"),
        (("startTime", "endTime"), "s"),
    )
    for (start_key, end_key), unit in aliases:
        if start_key in value and end_key in value:
            start = _number(value[start_key], unit=unit)
            end = _number(value[end_key], unit=unit)
            if start is not None and end is not None and end > start:
                return start, end
    return None


def _collect_timing(value: Any, intervals: list[tuple[int, int]]) -> None:
    if isinstance(value, Mapping):
        pair = _timing_pair(value)
        if pair is not None:
            intervals.append(pair)
        for child in value.values():
            if isinstance(child, (Mapping, list, tuple)):
                _collect_timing(child, intervals)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_timing(child, intervals)


def _parse_timing_bytes(raw: bytes) -> list[tuple[int, int]]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # A small RTTM-like timing-only fallback.  The speaker token is never
        # retained; only columns 4 and 5 (start seconds, duration seconds) are
        # parsed.  This keeps intake useful for opaque timing exports.
        intervals: list[tuple[int, int]] = []
        for line in raw.decode("utf-8", errors="strict").splitlines():
            fields = line.split()
            if len(fields) >= 5 and fields[0] == "SPEAKER":
                try:
                    start = _number(float(fields[3]), unit="s")
                    length = _number(float(fields[4]), unit="s")
                except ValueError:
                    continue
                if start is not None and length is not None and length > 0:
                    intervals.append((start, start + length))
        return intervals
    intervals = []
    _collect_timing(value, intervals)
    return intervals


def inspect_timing_reference(path: str | os.PathLike[str], *, source_duration_us: int) -> TimingInfo:
    reference = Path(path)
    if not reference.is_file():
        raise FixtureError(f"timing reference does not exist: {reference}")
    if reference.stat().st_size > 64 * 1024 * 1024:
        raise FixtureError("timing reference exceeds the bounded 64 MiB intake limit")
    raw = reference.read_bytes()
    intervals = _parse_timing_bytes(raw)
    if not intervals:
        raise FixtureError("timing reference contains no valid timing intervals")
    intervals = sorted(set(intervals))
    if any(end > source_duration_us + 1 for _, end in intervals):
        raise FixtureError("timing reference extends beyond source duration")
    timing_payload = [[start, end] for start, end in intervals]
    timing_digest = hashlib.sha256(_canonical_json(timing_payload)).hexdigest()
    return TimingInfo(
        sha256=hashlib.sha256(raw).hexdigest(),
        timing_digest_sha256=timing_digest,
        interval_count=len(intervals),
        max_end_us=max(end for _, end in intervals),
    )


def _relationship_id(source: WavInfo, timing: TimingInfo) -> str:
    payload = f"{source.sha256}:{timing.timing_digest_sha256}".encode("ascii")
    return "rel-" + hashlib.sha256(payload).hexdigest()[:24]


def _make_artifacts(relationship_id: str, *, noise_seed: int, noise_snr_db: float) -> tuple[ArtifactPlan, ...]:
    transforms: tuple[tuple[str, Mapping[str, Any]], ...] = (
        ("resample_8k_to_16k", {"operation": "downsample_16k_to_8k_then_linear_upsample_to_16k"}),
        ("gain_minus12db", {"operation": "global_gain", "gain_db": GAIN_DB}),
        ("noise_snr20db", {"operation": "add_seeded_broadband_noise", "snr_db": noise_snr_db, "seed": noise_seed}),
        ("resample_8k_gain", {"operation": "downsample_16k_to_8k_then_linear_upsample_to_16k_then_global_gain", "gain_db": GAIN_DB}),
    )
    result = []
    for variant, transform in transforms:
        token = hashlib.sha256(_canonical_json({"relationship_id": relationship_id, "variant": variant, "transform": transform})).hexdigest()
        artifact_id = "artifact-" + token[:24]
        result.append(ArtifactPlan(variant, artifact_id, artifact_id + ".wav", relationship_id, transform))
    return tuple(result)


def build_plan(
    source_path: str | os.PathLike[str],
    timing_reference_path: str | os.PathLike[str],
    *,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    noise_seed: int = DEFAULT_NOISE_SEED,
    noise_snr_db: float = DEFAULT_NOISE_SNR_DB,
) -> FixturePlan:
    if chunk_frames <= 0 or chunk_frames > MAX_CHUNK_FRAMES:
        raise FixtureError(f"chunk_frames must be in [1, {MAX_CHUNK_FRAMES}]")
    if not math.isfinite(noise_snr_db) or noise_snr_db <= 0:
        raise FixtureError("noise_snr_db must be finite and positive")
    source = inspect_wav(source_path)
    timing = inspect_timing_reference(timing_reference_path, source_duration_us=source.duration_us)
    relationship_id = _relationship_id(source, timing)
    return FixturePlan(
        source=source,
        timing=timing,
        relationship_id=relationship_id,
        chunk_frames=chunk_frames,
        noise_seed=int(noise_seed),
        noise_snr_db=float(noise_snr_db),
        artifacts=_make_artifacts(relationship_id, noise_seed=int(noise_seed), noise_snr_db=float(noise_snr_db)),
    )


def _iter_pcm16(path: Path, chunk_frames: int) -> Iterator[tuple[int, ...]]:
    layout = WavPcmAccessor(path).layout
    if (layout.sample_rate_hz, layout.channel_count, layout.sample_width_bytes) != (INPUT_RATE_HZ, 1, 2):
        raise FixtureError("source must be uncompressed PCM16 mono 16 kHz WAV")
    remaining = layout.data_bytes
    with path.open("rb") as handle:
        handle.seek(layout.data_offset)
        while remaining:
            read_bytes = min(remaining, chunk_frames * 2)
            payload = handle.read(read_bytes)
            if len(payload) != read_bytes or len(payload) % 2:
                raise FixtureError("source PCM payload is truncated or not frame aligned")
            yield struct.unpack("<%dh" % (len(payload) // 2), payload)
            remaining -= read_bytes


def _avg(a: int, b: int) -> int:
    total = a + b
    # Round half away from zero, independent of floating-point details.
    return (total + 1) // 2 if total >= 0 else (total - 1) // 2


def _iter_downsampled(path: Path, chunk_frames: int) -> Iterator[int]:
    pending: int | None = None
    for values in _iter_pcm16(path, chunk_frames):
        index = 0
        if pending is not None:
            if not values:
                continue
            yield _avg(pending, values[0])
            pending = None
            index = 1
        while index + 1 < len(values):
            yield _avg(values[index], values[index + 1])
            index += 2
        if index < len(values):
            pending = values[index]
    if pending is not None:
        yield pending


def _iter_resampled(path: Path, frame_count: int, chunk_frames: int) -> Iterator[int]:
    previous: int | None = None
    emitted = 0
    for current in _iter_downsampled(path, chunk_frames):
        if previous is None:
            previous = current
            continue
        if emitted < frame_count:
            yield previous
            emitted += 1
        if emitted < frame_count:
            yield _avg(previous, current)
            emitted += 1
        previous = current
    if previous is not None:
        while emitted < frame_count:
            yield previous
            emitted += 1


def _gain_values(values: Iterable[int], factor: float) -> Iterator[int]:
    for value in values:
        yield _quantize(value * factor)


def _quantize(value: float) -> int:
    if not math.isfinite(value):
        raise FixtureError("non-finite generated PCM value")
    # Every generated transform is headroom-safe; this is a guard, not a
    # clipping policy.  Saturating here would hide a bug in a transform.
    rounded = int(round(value))
    if rounded < PCM16_MIN or rounded > PCM16_MAX:
        raise FixtureError("generated PCM would clip")
    return rounded


def _prng_values(seed: int, count: int, *, offset: int = 0) -> Iterator[float]:
    # SplitMix64 indexed by absolute sample offset: chunk boundaries therefore
    # cannot change the sequence, and seeking to a later chunk remains O(1).
    base = int(seed) & _MASK64
    for index in range(max(0, int(offset)), max(0, int(offset)) + count):
        value = (base + (index + 1) * _SPLITMIX_INCREMENT) & _MASK64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        value ^= value >> 31
        value &= _MASK64
        # Open interval-ish uniform white noise in [-1, 1].
        yield ((value >> 11) / float(1 << 53)) * 2.0 - 1.0


def _noise_parameters(path: Path, chunk_frames: int, seed: int, snr_db: float) -> tuple[float, float, float]:
    count = 0
    signal_sq = 0.0
    noise_sum = 0.0
    noise_sq = 0.0
    offset = 0
    for values in _iter_pcm16(path, chunk_frames):
        count += len(values)
        signal_sq += sum(float(value) * float(value) for value in values)
        for noise in _prng_values(seed, len(values), offset=offset):
            noise_sum += noise
            noise_sq += noise * noise
        offset += len(values)
    if count <= 0:
        raise FixtureError("source WAV has no samples")
    signal_rms = math.sqrt(signal_sq / count)
    noise_mean = noise_sum / count
    centered_sq = max(0.0, noise_sq - 2.0 * noise_mean * noise_sum + count * noise_mean * noise_mean)
    base_rms = math.sqrt(centered_sq / count)
    target_rms = signal_rms / (10.0 ** (snr_db / 20.0)) if signal_rms else 0.0
    return signal_rms, target_rms, (target_rms / base_rms if base_rms else 0.0)


def _write_wav(path: Path, values: Iterable[int], *, frame_count: int) -> str:
    digest = hashlib.sha256()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_RATE_HZ)
        handle.setcomptype("NONE", "not compressed")
        pending: list[int] = []
        written = 0
        for value in values:
            if value < PCM16_MIN or value > PCM16_MAX:
                raise FixtureError("generated PCM value outside int16 range")
            pending.append(int(value))
            if len(pending) >= DEFAULT_CHUNK_FRAMES:
                payload = struct.pack("<%dh" % len(pending), *pending)
                handle.writeframesraw(payload)
                digest.update(payload)
                written += len(pending)
                pending.clear()
        if pending:
            payload = struct.pack("<%dh" % len(pending), *pending)
            handle.writeframesraw(payload)
            digest.update(payload)
            written += len(pending)
        if written != frame_count:
            raise FixtureError(f"generated frame count {written} does not match {frame_count}")
    return _sha256_file(path)


def _render_variant(source_path: Path, plan: FixturePlan, artifact: ArtifactPlan, destination: Path) -> dict[str, Any]:
    source_factor = 10.0 ** (GAIN_DB / 20.0)
    if artifact.variant == "resample_8k_to_16k":
        values: Iterable[int] = _iter_resampled(source_path, plan.source.frame_count, plan.chunk_frames)
        anti_clip_scale = 1.0
    elif artifact.variant == "gain_minus12db":
        values = _gain_values((v for chunk in _iter_pcm16(source_path, plan.chunk_frames) for v in chunk), source_factor)
        anti_clip_scale = 1.0
    elif artifact.variant == "resample_8k_gain":
        values = _gain_values(_iter_resampled(source_path, plan.source.frame_count, plan.chunk_frames), source_factor)
        anti_clip_scale = 1.0
    elif artifact.variant == "noise_snr20db":
        _signal_rms, _target_rms, noise_factor = _noise_parameters(source_path, plan.chunk_frames, plan.noise_seed, plan.noise_snr_db)
        # Uniform noise is centered deterministically in the parameter pass;
        # account for its small mean exactly while keeping bounded reads.
        count = plan.source.frame_count
        noise_sum = sum(_prng_values(plan.noise_seed, count))
        noise_mean = noise_sum / count
        max_abs = 0.0
        offset = 0
        for values_chunk in _iter_pcm16(source_path, plan.chunk_frames):
            for value, noise in zip(values_chunk, _prng_values(plan.noise_seed, len(values_chunk), offset=offset)):
                max_abs = max(max_abs, abs(float(value) + (noise - noise_mean) * noise_factor))
            offset += len(values_chunk)
        anti_clip_scale = min(1.0, PCM16_MAX / max_abs) if max_abs else 1.0
        # Use an explicit offset stream to ensure chunk boundaries do not alter
        # the sequence.  Re-centering uses the exact global mean above.
        def noisy() -> Iterator[int]:
            offset = 0
            for values_chunk in _iter_pcm16(source_path, plan.chunk_frames):
                for value, noise in zip(values_chunk, _prng_values(plan.noise_seed, len(values_chunk), offset=offset)):
                    yield _quantize((float(value) + (noise - noise_mean) * noise_factor) * anti_clip_scale)
                offset += len(values_chunk)
        values = noisy()
    else:  # pragma: no cover - plan construction is closed above
        raise FixtureError(f"unknown artifact variant: {artifact.variant}")
    sha256 = _write_wav(destination, values, frame_count=plan.source.frame_count)
    return {
        "artifact_id": artifact.artifact_id,
        "variant": artifact.variant,
        "path": artifact.filename,
        "sha256": sha256,
        "relationship_id": artifact.relationship_id,
        "sample_rate_hz": TARGET_RATE_HZ,
        "channel_count": 1,
        "sample_width_bytes": 2,
        "frame_count": plan.source.frame_count,
        "duration_us": plan.source.duration_us,
        "source_time_start_us": 0,
        "source_time_end_us": plan.source.duration_us,
        "transform": dict(artifact.transform),
        "anti_clip_scale": anti_clip_scale,
        "clipped_sample_count": 0,
    }


def _manifest_payload(plan: FixturePlan, artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "challenge_relationship_id": plan.relationship_id,
        "source": {
            "sha256": plan.source.sha256,
            "sample_rate_hz": plan.source.sample_rate_hz,
            "channel_count": plan.source.channel_count,
            "sample_width_bytes": plan.source.sample_width_bytes,
            "frame_count": plan.source.frame_count,
            "duration_us": plan.source.duration_us,
        },
        "timing_reference": {
            "sha256": plan.timing.sha256,
            "timing_digest_sha256": plan.timing.timing_digest_sha256,
            "interval_count": plan.timing.interval_count,
            "max_end_us": plan.timing.max_end_us,
            "speaker_labels_used": False,
        },
        "policy": {
            "offline": True,
            "chunk_frames": plan.chunk_frames,
            "gain_db": GAIN_DB,
            "noise_seed": plan.noise_seed,
            "noise_snr_db": plan.noise_snr_db,
            "input_contract": "PCM16 mono 16 kHz WAV",
            "output_contract": "PCM16 mono 16 kHz WAV",
            "speaker_label_dependent": False,
        },
        "artifacts": list(artifacts),
    }


def _preflight_outputs(output_dir: Path, plan: FixturePlan, *, overwrite: bool) -> None:
    paths = [output_dir / artifact.filename for artifact in plan.artifacts] + [output_dir / "manifest.json", output_dir / "manifest.sha256"]
    existing = [path for path in paths if path.exists()]
    if output_dir.is_dir() and not overwrite:
        existing.extend(path for path in output_dir.rglob("*") if path.is_file() and path not in existing)
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FixtureError(f"output files already exist; pass --overwrite: {names}")


def build_fixtures(
    source_path: str | os.PathLike[str],
    timing_reference_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    noise_seed: int = DEFAULT_NOISE_SEED,
    noise_snr_db: float = DEFAULT_NOISE_SNR_DB,
) -> dict[str, Any]:
    plan = build_plan(source_path, timing_reference_path, chunk_frames=chunk_frames, noise_seed=noise_seed, noise_snr_db=noise_snr_db)
    destination = Path(output_dir)
    _preflight_outputs(destination, plan, overwrite=overwrite)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []
    source = Path(source_path)
    for artifact in plan.artifacts:
        artifacts.append(_render_variant(source, plan, artifact, destination / artifact.filename))
    manifest = _manifest_payload(plan, artifacts)
    manifest_path = destination / "manifest.json"
    manifest_bytes = _canonical_json(manifest) + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    (destination / "manifest.sha256").write_text(manifest_hash + "  manifest.json\n", encoding="ascii")
    report = dict(manifest)
    report["manifest_path"] = "manifest.json"
    report["manifest_sha256"] = manifest_hash
    return report


def evaluate_manifest(manifest_path: str | os.PathLike[str], *, source_path: str | os.PathLike[str] | None = None, timing_reference_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Verify manifest, relationship IDs, hashes, and every generated WAV."""
    path = Path(manifest_path)
    issues: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"ok": False, "issues": [f"cannot read manifest: {exc}"]}
    root = path.parent.resolve()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        issues.append("unsupported schema_version")
    relationship = manifest.get("challenge_relationship_id")
    artifacts = manifest.get("artifacts")
    if not isinstance(relationship, str) or not isinstance(artifacts, list) or not artifacts:
        issues.append("manifest lacks relationship or artifacts")
        artifacts = []
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            issues.append("artifact entry is not an object")
            continue
        relative = artifact.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative or relative.startswith("."):
            issues.append("artifact path is not an opaque basename")
            continue
        if relative in seen_paths:
            issues.append(f"duplicate artifact path: {relative}")
        seen_paths.add(relative)
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or artifact_id in seen_ids:
            issues.append(f"duplicate or invalid artifact id: {relative}")
        if isinstance(artifact_id, str):
            seen_ids.add(artifact_id)
        artifact_path = (root / relative).resolve()
        if artifact_path.parent != root:
            issues.append(f"artifact escapes output directory: {relative}")
            continue
        if artifact.get("relationship_id") != relationship:
            issues.append(f"relationship mismatch: {relative}")
        expected_hash = artifact.get("sha256")
        if not isinstance(expected_hash, str) or not artifact_path.is_file():
            issues.append(f"missing artifact: {relative}")
            continue
        if _sha256_file(artifact_path) != expected_hash:
            issues.append(f"hash mismatch: {relative}")
        try:
            with wave.open(str(artifact_path), "rb") as handle:
                contract = (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getcomptype(), handle.getnframes())
        except (EOFError, OSError, wave.Error):
            issues.append(f"invalid WAV artifact: {relative}")
            continue
        expected_contract = (TARGET_RATE_HZ, 1, 2, "NONE", artifact.get("frame_count"))
        if contract != expected_contract:
            issues.append(f"WAV contract mismatch: {relative}")
        if artifact.get("source_time_start_us") != 0 or artifact.get("source_time_end_us") != artifact.get("duration_us"):
            issues.append(f"source-time mapping mismatch: {relative}")
        if artifact.get("clipped_sample_count") != 0:
            issues.append(f"clipping reported: {relative}")
    policy = manifest.get("policy", {})
    if policy.get("offline") is not True or policy.get("speaker_label_dependent") is not False:
        issues.append("manifest policy is not offline and label-independent")
    sidecar = root / "manifest.sha256"
    if not sidecar.is_file():
        issues.append("manifest.sha256 is missing")
    else:
        token = sidecar.read_text(encoding="ascii").strip().split()[0] if sidecar.read_text(encoding="ascii").strip() else ""
        if token != _sha256_file(path):
            issues.append("manifest hash mismatch")
    if source_path is not None:
        try:
            source = inspect_wav(source_path)
            if source.sha256 != manifest.get("source", {}).get("sha256"):
                issues.append("source hash mismatch")
        except FixtureError as exc:
            issues.append(str(exc))
    if timing_reference_path is not None:
        try:
            source_duration = int(manifest.get("source", {}).get("duration_us"))
            timing = inspect_timing_reference(timing_reference_path, source_duration_us=source_duration)
            expected = manifest.get("timing_reference", {})
            if timing.sha256 != expected.get("sha256") or timing.timing_digest_sha256 != expected.get("timing_digest_sha256"):
                issues.append("timing reference hash/digest mismatch")
        except (FixtureError, TypeError, ValueError) as exc:
            issues.append(str(exc))
    return {"ok": not issues, "manifest_path": "manifest.json", "relationship_id": relationship, "verified_artifacts": len(artifacts), "issues": issues}


def _plan_json(plan: FixturePlan, output_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_only": True,
        "output_dir": str(output_dir),
        "challenge_relationship_id": plan.relationship_id,
        "source": {"sha256": plan.source.sha256, "frame_count": plan.source.frame_count, "duration_us": plan.source.duration_us},
        "timing_reference": {"sha256": plan.timing.sha256, "timing_digest_sha256": plan.timing.timing_digest_sha256, "interval_count": plan.timing.interval_count},
        "policy": {"offline": True, "chunk_frames": plan.chunk_frames, "noise_seed": plan.noise_seed, "noise_snr_db": plan.noise_snr_db, "gain_db": GAIN_DB},
        "artifacts": [{"artifact_id": a.artifact_id, "variant": a.variant, "path": a.filename, "relationship_id": a.relationship_id} for a in plan.artifacts],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, help="local PCM16 mono 16 kHz WAV")
    parser.add_argument("--timing-reference", type=Path, help="opaque timing-only JSON/RTTM reference")
    parser.add_argument("--output-dir", type=Path, help="explicit fixture output directory")
    parser.add_argument("--plan-only", action="store_true", help="validate and print a plan without writing files")
    parser.add_argument("--overwrite", action="store_true", help="replace existing generated files")
    parser.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES)
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_NOISE_SEED)
    parser.add_argument("--noise-snr-db", type=float, default=DEFAULT_NOISE_SNR_DB)
    parser.add_argument("--evaluate", type=Path, help="verify an existing manifest.json")
    parser.add_argument("--source-for-evaluation", type=Path)
    parser.add_argument("--timing-reference-for-evaluation", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.evaluate is not None:
            report = evaluate_manifest(args.evaluate, source_path=args.source_for_evaluation, timing_reference_path=args.timing_reference_for_evaluation)
        else:
            if args.source is None or args.timing_reference is None or args.output_dir is None:
                parser.error("source, --timing-reference, and --output-dir are required unless --evaluate is used")
            plan = build_plan(args.source, args.timing_reference, chunk_frames=args.chunk_frames, noise_seed=args.noise_seed, noise_snr_db=args.noise_snr_db)
            if args.plan_only:
                report = _plan_json(plan, args.output_dir)
            else:
                report = build_fixtures(args.source, args.timing_reference, args.output_dir, overwrite=args.overwrite, chunk_frames=args.chunk_frames, noise_seed=args.noise_seed, noise_snr_db=args.noise_snr_db)
    except FixtureError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
