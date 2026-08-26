"""Deterministic, bounded global gain for development-only PCM16 trials.

The policy in this module is deliberately label independent: it inspects only
whole-file PCM amplitudes and never consumes diarization/reference data.  The
input WAV is scanned in bounded chunks, so a long recording is not retained in
memory.  Both the stdlib and optional NumPy scanners accumulate the exact same
integer PCM16 sum of squares and peak before deriving one file-wide gain.

This is an opt-in challenger, not an adaptive production default.  It applies
at most +12.04 dB (4x) when the whole-file RMS is below -40 dBFS, limits any
boost so the predicted peak is at most 0.99, and treats candidate gains below
1.25x as an exact no-op.  The activation deadband avoids perturbing model
features for a numerically insignificant boost.  The policy never attenuates
and therefore cannot introduce clipping.
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .media import AudioDecodeError, DecodedAudioChunk, UnsupportedCodecError, WavPcmAccessor


PCM16_SCALE = 32768.0
_SHA256_HEX_LENGTH = 64
_MAX_ANALYSIS_CHUNK_BYTES = 8 * 1024 * 1024


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return (
        len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class GlobalGainPolicy:
    """Predeclared low-RMS boost policy used by the development challenger."""

    rms_trigger: float = 0.01
    max_gain: float = 4.0
    peak_ceiling: float = 0.99
    min_activation_gain: float = 1.25
    analysis_chunk_frames: int = 240_000

    def __post_init__(self) -> None:
        for name in ("rms_trigger", "max_gain", "peak_ceiling", "min_activation_gain"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < float(self.rms_trigger) < 1.0:
            raise ValueError("rms_trigger must be in (0, 1)")
        if float(self.max_gain) < 1.0:
            raise ValueError("max_gain must be at least 1.0")
        if not 0.0 < float(self.peak_ceiling) < 1.0:
            raise ValueError("peak_ceiling must be in (0, 1)")
        if not 1.0 <= float(self.min_activation_gain) <= float(self.max_gain):
            raise ValueError("min_activation_gain must be in [1.0, max_gain]")
        if type(self.analysis_chunk_frames) is not int or self.analysis_chunk_frames <= 0:
            raise ValueError("analysis_chunk_frames must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sddiar.global_gain_policy_v2",
            "rms_trigger": float(self.rms_trigger),
            "max_gain": float(self.max_gain),
            "peak_ceiling": float(self.peak_ceiling),
            "min_activation_gain": float(self.min_activation_gain),
            "analysis_chunk_frames": self.analysis_chunk_frames,
            "boost_only_below_rms_trigger": True,
            "never_attenuate": True,
            "label_independent": True,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha256(self.to_dict())


DEFAULT_GLOBAL_GAIN_POLICY = GlobalGainPolicy()


@dataclass(frozen=True, slots=True)
class GlobalGainProfile:
    """Exact PCM statistics and the deterministic gain chosen for one file."""

    source_sha256: str
    policy_sha256: str
    sample_count: int
    sum_squares_pcm16: int
    peak_abs_pcm16: int
    rms: float
    peak: float
    applied_gain: float
    predicted_peak: float
    reason: str

    def __post_init__(self) -> None:
        if not _valid_sha256(self.source_sha256) or not _valid_sha256(self.policy_sha256):
            raise ValueError("source and policy SHA-256 values must be lowercase hexadecimal")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if type(self.sum_squares_pcm16) is not int or self.sum_squares_pcm16 < 0:
            raise ValueError("sum_squares_pcm16 must be non-negative")
        if type(self.peak_abs_pcm16) is not int or not 0 <= self.peak_abs_pcm16 <= 32768:
            raise ValueError("peak_abs_pcm16 must be in [0, 32768]")
        for name in ("rms", "peak", "applied_gain", "predicted_peak"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.applied_gain < 1.0:
            raise ValueError("global gain must never attenuate")

    @property
    def applied(self) -> bool:
        return self.applied_gain > 1.0

    def _hash_payload(self) -> dict[str, Any]:
        # Hash exact integer statistics as well as the decision.  Derived
        # display floats are intentionally excluded so the proof is stable
        # across equivalent stdlib/NumPy analysis implementations.
        return {
            "schema": "sddiar.global_gain_profile_hash_v1",
            "source_sha256": self.source_sha256,
            "policy_sha256": self.policy_sha256,
            "sample_count": self.sample_count,
            "sum_squares_pcm16": self.sum_squares_pcm16,
            "peak_abs_pcm16": self.peak_abs_pcm16,
            "applied_gain": self.applied_gain,
            "reason": self.reason,
        }

    @property
    def profile_sha256(self) -> str:
        return _mapping_sha256(self._hash_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sddiar.global_gain_profile_v1",
            "enabled": True,
            "source_sha256": self.source_sha256,
            "policy_sha256": self.policy_sha256,
            "profile_sha256": self.profile_sha256,
            "sample_count": self.sample_count,
            "sum_squares_pcm16": self.sum_squares_pcm16,
            "peak_abs_pcm16": self.peak_abs_pcm16,
            "rms": self.rms,
            "peak": self.peak,
            "applied_gain": self.applied_gain,
            "applied_gain_db": 20.0 * math.log10(self.applied_gain),
            "predicted_peak": self.predicted_peak,
            "applied": self.applied,
            "reason": self.reason,
            "label_independent": True,
            "no_new_clipping": self.applied_gain == 1.0 or self.predicted_peak < 1.0,
        }


def disabled_gain_metadata(policy: GlobalGainPolicy = DEFAULT_GLOBAL_GAIN_POLICY) -> dict[str, Any]:
    """Return deterministic result metadata without touching source audio."""

    return {
        "schema": "sddiar.global_gain_profile_v1",
        "enabled": False,
        "policy": policy.to_dict(),
        "policy_sha256": policy.sha256,
        "label_independent": True,
    }


def _load_numpy_optional() -> Any | None:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None
    return np


def _stdlib_pcm16_stats(payload: bytes) -> tuple[int, int, int]:
    if len(payload) % 2:
        raise AudioDecodeError("PCM16 payload is not sample aligned")
    values = array.array("h")
    values.frombytes(payload)
    if sys.byteorder != "little":  # pragma: no cover - supported portability path
        values.byteswap()
    sum_squares = 0
    peak = 0
    for value in values:
        integer = int(value)
        absolute = abs(integer)
        if absolute > peak:
            peak = absolute
        sum_squares += integer * integer
    return len(values), sum_squares, peak


def _numpy_pcm16_stats(payload: bytes, np: Any) -> tuple[int, int, int]:
    if len(payload) % 2:
        raise AudioDecodeError("PCM16 payload is not sample aligned")
    values = np.frombuffer(payload, dtype="<i2").astype(np.int64)
    if int(values.size) == 0:
        return 0, 0, 0
    # A bounded 240k-frame mono chunk cannot overflow int64.  For customized
    # policies, split the vector further so exact accumulation remains true.
    safe_samples = 4_000_000_000
    sum_squares = 0
    peak = 0
    for start in range(0, int(values.size), safe_samples):
        block = values[start:start + safe_samples]
        sum_squares += int(np.sum(block * block, dtype=np.int64))
        peak = max(peak, int(np.max(np.abs(block))))
    return int(values.size), sum_squares, peak


def analyze_pcm16_global_gain(
    source: str | Path,
    *,
    policy: GlobalGainPolicy = DEFAULT_GLOBAL_GAIN_POLICY,
    source_sha256: str | None = None,
    prefer_numpy: bool = True,
) -> GlobalGainProfile:
    """Scan one integer PCM16 WAV and choose one bounded global gain.

    Valid classic PCM and WAVE_FORMAT_EXTENSIBLE PCM files are accepted via
    the package's strict RIFF parser.  Statistics cover every interleaved PCM
    sample exactly and are accumulated with bounded memory.
    """

    path = Path(source)
    if not path.is_file() or path.is_symlink():
        raise AudioDecodeError("audio file does not exist or is a symlink")
    accessor = WavPcmAccessor(path)
    layout = accessor.layout
    if layout.sample_width_bytes != 2:
        raise UnsupportedCodecError("global gain analysis requires integer PCM16 WAV")
    digest = source_sha256 if source_sha256 is not None else _sha256_path(path)
    if not isinstance(digest, str) or not _valid_sha256(digest):
        raise ValueError("source_sha256 must be exactly 64 lowercase hexadecimal characters")

    np = _load_numpy_optional() if prefer_numpy else None
    frame_bytes = layout.sample_width_bytes * layout.channel_count
    bounded_frames = min(
        policy.analysis_chunk_frames,
        max(1, _MAX_ANALYSIS_CHUNK_BYTES // frame_bytes),
    )
    read_limit = bounded_frames * frame_bytes
    sample_count = 0
    sum_squares = 0
    peak_abs = 0
    remaining = layout.data_bytes
    try:
        with path.open("rb") as handle:
            handle.seek(layout.data_offset)
            while remaining:
                read_bytes = min(remaining, read_limit)
                payload = handle.read(read_bytes)
                if len(payload) != read_bytes:
                    raise AudioDecodeError("WAV gain scan ended before declared data bytes")
                count, chunk_squares, chunk_peak = (
                    _numpy_pcm16_stats(payload, np)
                    if np is not None
                    else _stdlib_pcm16_stats(payload)
                )
                sample_count += count
                sum_squares += chunk_squares
                peak_abs = max(peak_abs, chunk_peak)
                remaining -= read_bytes
    except (AudioDecodeError, UnsupportedCodecError):
        raise
    except OSError as exc:
        raise AudioDecodeError(f"cannot scan PCM16 WAV: {exc}") from exc
    expected_samples = layout.frame_count * layout.channel_count
    if sample_count != expected_samples or sample_count <= 0:
        raise AudioDecodeError("PCM16 gain scan sample-count invariant failed")

    rms = math.sqrt(sum_squares / sample_count) / PCM16_SCALE
    peak = peak_abs / PCM16_SCALE
    gain = 1.0
    reason = "RMS_AT_OR_ABOVE_TRIGGER"
    if rms < policy.rms_trigger:
        peak_limited_gain = math.inf if peak == 0.0 else policy.peak_ceiling / peak
        candidate_gain = max(1.0, min(float(policy.max_gain), peak_limited_gain))
        if candidate_gain == 1.0:
            gain = 1.0
            reason = "LOW_RMS_PEAK_PREVENTS_BOOST"
        elif candidate_gain < policy.min_activation_gain:
            gain = 1.0
            reason = "LOW_RMS_GAIN_BELOW_ACTIVATION_FLOOR"
        elif candidate_gain < policy.max_gain:
            gain = candidate_gain
            reason = "LOW_RMS_PEAK_LIMITED_BOOST"
        else:
            gain = candidate_gain
            reason = "LOW_RMS_MAX_BOOST"
    predicted_peak = peak * gain
    if gain > 1.0 and predicted_peak > policy.peak_ceiling + 1e-15:
        raise AudioDecodeError("global gain peak-ceiling invariant failed")
    if predicted_peak > 1.0 + 1e-15:
        raise AudioDecodeError("global gain would introduce clipping")
    return GlobalGainProfile(
        source_sha256=digest,
        policy_sha256=policy.sha256,
        sample_count=sample_count,
        sum_squares_pcm16=sum_squares,
        peak_abs_pcm16=peak_abs,
        rms=rms,
        peak=peak,
        applied_gain=gain,
        predicted_peak=predicted_peak,
        reason=reason,
    )


def scale_normalized_samples(samples: Any, gain: float) -> Any:
    """Scale decoded normalized samples while preserving their container form."""

    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(float(gain)):
        raise ValueError("gain must be finite")
    gain = float(gain)
    if gain < 1.0:
        raise ValueError("gain scaling must never attenuate")
    if gain == 1.0:
        return samples
    np = _load_numpy_optional()
    if np is not None and isinstance(samples, np.ndarray):
        # Both chunk and region readers produce float32.  Keeping float32 here
        # makes the two inference paths bit-equivalent for the same PCM sample.
        return np.multiply(samples, np.float32(gain), dtype=np.float32)
    if isinstance(samples, tuple):
        if samples and isinstance(samples[0], tuple):
            return tuple(tuple(float(value) * gain for value in frame) for frame in samples)
        return tuple(float(value) * gain for value in samples)
    if isinstance(samples, list):
        if samples and isinstance(samples[0], (tuple, list)):
            return [type(frame)(float(value) * gain for value in frame) for frame in samples]
        return [float(value) * gain for value in samples]
    return tuple(float(value) * gain for value in samples)


def scale_decoded_chunks(
    chunks: Iterable[DecodedAudioChunk],
    gain: float,
) -> Iterator[DecodedAudioChunk]:
    """Apply exactly one gain to a bounded decoded-chunk stream."""

    for chunk in chunks:
        scaled = scale_normalized_samples(chunk.samples, gain)
        yield chunk if scaled is chunk.samples else replace(chunk, samples=scaled)


class GainScaledWavPcmAccessor:
    """Random-access region reader applying the same gain as the VAD stream."""

    def __init__(self, accessor: WavPcmAccessor, gain: float):
        if not isinstance(accessor, WavPcmAccessor):
            raise TypeError("accessor must be a WavPcmAccessor")
        # Validate once even when the gain is exactly one.
        scale_normalized_samples((), gain)
        self._accessor = accessor
        self.gain = float(gain)
        self.layout = accessor.layout
        self.path = accessor.path

    def read_mono_samples(self, start_us: int, end_us: int) -> tuple[float, ...]:
        return scale_normalized_samples(
            self._accessor.read_mono_samples(start_us, end_us), self.gain
        )

    def read_mono_samples_numpy(self, start_us: int, end_us: int) -> Any:
        return scale_normalized_samples(
            self._accessor.read_mono_samples_numpy(start_us, end_us), self.gain
        )


__all__ = [
    "DEFAULT_GLOBAL_GAIN_POLICY",
    "GainScaledWavPcmAccessor",
    "GlobalGainPolicy",
    "GlobalGainProfile",
    "analyze_pcm16_global_gain",
    "disabled_gain_metadata",
    "scale_decoded_chunks",
    "scale_normalized_samples",
]
