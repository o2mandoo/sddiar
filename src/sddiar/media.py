"""Offline, stdlib-first audio frontend.

The initial development decoder intentionally supports only little-endian PCM
WAV.  In particular, this module never shells out to FFmpeg or attempts a
network/model lookup.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Protocol

from .contracts import AudioSourceMetadata, AudioView, TimeWarpSegment


class MediaError(Exception):
    code = "MEDIA_ERROR"


class UnsupportedContainerError(MediaError):
    code = "UNSUPPORTED_CONTAINER"


class UnsupportedCodecError(MediaError):
    code = "UNSUPPORTED_CODEC"


class AudioDecodeError(MediaError):
    code = "AUDIO_DECODE_FAILED"


@dataclass(frozen=True, slots=True)
class WavPcmFormat:
    sample_width_bytes: int
    channel_count: int
    sample_rate_hz: int
    frame_count: int
    data_offset: int
    data_bytes: int


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    samples: tuple[tuple[float, ...], ...]  # frame-major, normalized [-1, 1]
    sample_rate_hz: int
    channel_count: int
    metadata: AudioSourceMetadata

    @property
    def total_samples(self) -> int:
        return len(self.samples)


@dataclass(frozen=True, slots=True)
class DecodedAudioChunk:
    samples: tuple[tuple[float, ...], ...]
    source_start_sample: int
    source_end_sample: int
    sample_rate_hz: int
    channel_count: int
    metadata: AudioSourceMetadata


@dataclass(frozen=True, slots=True)
class NormalizedAudioChunk:
    samples: tuple[float, ...]
    view_start_sample: int
    view_end_sample: int
    time_warp: TimeWarpSegment
    metadata: AudioSourceMetadata


@dataclass(frozen=True, slots=True)
class NormalizationPolicy:
    target_sample_rate_hz: int = 16_000
    chunk_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.target_sample_rate_hz <= 0 or self.chunk_seconds <= 0 or not math.isfinite(self.chunk_seconds):
            raise ValueError("invalid normalization policy")


@dataclass(frozen=True, slots=True)
class NormalizedAudio:
    samples: tuple[float, ...]
    view: AudioView
    metadata: AudioSourceMetadata

    def iter_chunks(self, chunk_samples: int | None = None) -> Iterator["AudioChunk"]:
        size = chunk_samples or max(1, round(self.view.sample_rate_hz * 15.0))
        if size <= 0:
            raise ValueError("chunk_samples must be positive")
        for start in range(0, len(self.samples), size):
            end = min(len(self.samples), start + size)
            yield AudioChunk(self.samples[start:end], start, end, self.view)


@dataclass(frozen=True, slots=True)
class AudioChunk:
    samples: tuple[float, ...]
    view_start_sample: int
    view_end_sample: int
    view: AudioView

    @property
    def start_us(self) -> int:
        return self.view.time_warp[0].source_us(self.view_start_sample)


class MediaDecoder(Protocol):
    def decode(self, source: str | os.PathLike[str] | bytes) -> DecodedAudio: ...


class WavPcmDecoder:
    """Strict local RIFF/WAVE PCM decoder, including WAVE_FORMAT_EXTENSIBLE PCM.

    Apple ``afconvert`` commonly emits PCM as WAVE_FORMAT_EXTENSIBLE (0xFFFE).
    CPython 3.11's :mod:`wave` cannot read it, so this parser validates the
    standard PCM subtype directly instead of treating a valid WAV as corrupt.
    """

    def decode(self, source: str | os.PathLike[str] | bytes) -> DecodedAudio:
        try:
            raw = source if isinstance(source, bytes) else Path(source).read_bytes()
            if not isinstance(raw, bytes):
                raise AudioDecodeError("audio source is not bytes")
            layout = _parse_wav_layout(io.BytesIO(raw), len(raw))
        except FileNotFoundError as exc:
            raise AudioDecodeError(f"audio file does not exist: {source}") from exc
        payload = raw[layout.data_offset:layout.data_offset + layout.data_bytes]
        frames = _decode_pcm(payload, layout.sample_width_bytes, layout.channel_count)
        if not frames:
            raise AudioDecodeError("empty WAV has no PCM frames")
        digest = hashlib.sha256(raw).hexdigest()
        metadata = _metadata(digest, layout)
        return DecodedAudio(tuple(frames), layout.sample_rate_hz, layout.channel_count, metadata)

    def iter_decode_chunks(
        self, source: str | os.PathLike[str], *, frames_per_chunk: int = 240_000
    ) -> Iterator[DecodedAudioChunk]:
        """Stream local PCM WAV frames without loading PCM into memory.

        The input file is hashed in a first streaming pass for deterministic
        metadata; decode then proceeds in bounded `frames_per_chunk` reads.
        """

        if frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be positive")
        path = Path(source)
        if not path.is_file():
            raise AudioDecodeError(f"audio file does not exist: {path}")
        digest = _sha256_path(path)
        try:
            with path.open("rb") as handle:
                layout = _parse_wav_layout(handle, path.stat().st_size)
                metadata = _metadata(digest, layout)
                handle.seek(layout.data_offset)
                source_start = 0
                remaining = layout.data_bytes
                frame_bytes = layout.sample_width_bytes * layout.channel_count
                while remaining:
                    read_bytes = min(remaining, frames_per_chunk * frame_bytes)
                    payload = handle.read(read_bytes)
                    if len(payload) != read_bytes:
                        raise AudioDecodeError("WAV decode ended before declared data bytes")
                    frames = _decode_pcm(payload, layout.sample_width_bytes, layout.channel_count)
                    source_end = source_start + len(frames)
                    yield DecodedAudioChunk(tuple(frames), source_start, source_end, layout.sample_rate_hz, layout.channel_count, metadata)
                    source_start, remaining = source_end, remaining - read_bytes
                if source_start != layout.frame_count:
                    raise AudioDecodeError("WAV frame count invariant failed")
        except (UnsupportedContainerError, UnsupportedCodecError, AudioDecodeError):
            raise
        except (EOFError, OSError, ValueError) as exc:
            raise AudioDecodeError(f"invalid WAV: {exc}") from exc

    def iter_decode_chunks_numpy(
        self, source: str | os.PathLike[str], *, frames_per_chunk: int = 240_000
    ) -> Iterator[DecodedAudioChunk]:
        """Stream PCM16-mono chunks as normalized NumPy ``float32`` arrays.

        NumPy is intentionally imported only when this opt-in method is used.
        The file is still read and hashed in bounded operations; in particular,
        this method never materializes the complete PCM payload.  The returned
        chunk samples are one-dimensional for the mono fast path.  Callers that
        need the historical frame-major tuples should use
        :meth:`iter_decode_chunks` instead.
        """

        np = _load_numpy_optional()
        if np is None:
            raise UnsupportedCodecError("NumPy is unavailable for PCM16 fast decoding")
        if frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be positive")
        path = Path(source)
        if not path.is_file():
            raise AudioDecodeError(f"audio file does not exist: {path}")
        digest = _sha256_path(path)
        try:
            with path.open("rb") as handle:
                layout = _parse_wav_layout(handle, path.stat().st_size)
                _require_numpy_pcm16_mono(layout)
                metadata = _metadata(digest, layout)
                handle.seek(layout.data_offset)
                source_start = 0
                remaining = layout.data_bytes
                while remaining:
                    read_bytes = min(remaining, frames_per_chunk * 2)
                    payload = handle.read(read_bytes)
                    if len(payload) != read_bytes:
                        raise AudioDecodeError("WAV decode ended before declared data bytes")
                    # ``astype`` makes the result independent of the temporary
                    # bytes object while keeping conversion entirely vectorized.
                    values = np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768.0)
                    source_end = source_start + int(values.size)
                    yield DecodedAudioChunk(
                        values,
                        source_start,
                        source_end,
                        layout.sample_rate_hz,
                        layout.channel_count,
                        metadata,
                    )
                    source_start, remaining = source_end, remaining - read_bytes
                if source_start != layout.frame_count:
                    raise AudioDecodeError("WAV frame count invariant failed")
        except (UnsupportedContainerError, UnsupportedCodecError, AudioDecodeError):
            raise
        except (EOFError, OSError, ValueError) as exc:
            raise AudioDecodeError(f"invalid WAV: {exc}") from exc


class WavPcmAccessor:
    """Bounded random-access reader for local integer PCM WAV regions."""

    def __init__(self, source: str | os.PathLike[str]):
        self.path = Path(source)
        if not self.path.is_file():
            raise AudioDecodeError(f"audio file does not exist: {self.path}")
        with self.path.open("rb") as handle:
            self.layout = _parse_wav_layout(handle, self.path.stat().st_size)

    def read_mono_samples(self, start_us: int, end_us: int) -> tuple[float, ...]:
        if type(start_us) is not int or type(end_us) is not int or start_us < 0 or end_us <= start_us:
            raise AudioDecodeError("invalid source-time region")
        start_frame = max(0, (start_us * self.layout.sample_rate_hz) // 1_000_000)
        end_frame = min(
            self.layout.frame_count,
            (end_us * self.layout.sample_rate_hz + 999_999) // 1_000_000,
        )
        if end_frame <= start_frame:
            raise AudioDecodeError("source-time region has no PCM frames")
        frame_bytes = self.layout.sample_width_bytes * self.layout.channel_count
        with self.path.open("rb") as handle:
            handle.seek(self.layout.data_offset + start_frame * frame_bytes)
            payload = handle.read((end_frame - start_frame) * frame_bytes)
        frames = _decode_pcm(payload, self.layout.sample_width_bytes, self.layout.channel_count)
        return tuple(sum(frame) / len(frame) for frame in frames)

    def read_mono_samples_numpy(self, start_us: int, end_us: int) -> Any:
        """Read one bounded PCM16-mono region as normalized NumPy ``float32``.

        Time-to-frame rounding and clipping intentionally match
        :meth:`read_mono_samples` exactly.  The returned array is one-
        dimensional and owns its converted values; no file mapping or full-file
        PCM buffer is retained after the call.
        """

        np = _load_numpy_optional()
        if np is None:
            raise UnsupportedCodecError("NumPy is unavailable for PCM16 fast access")
        if type(start_us) is not int or type(end_us) is not int or start_us < 0 or end_us <= start_us:
            raise AudioDecodeError("invalid source-time region")
        _require_numpy_pcm16_mono(self.layout)
        start_frame = max(0, (start_us * self.layout.sample_rate_hz) // 1_000_000)
        end_frame = min(
            self.layout.frame_count,
            (end_us * self.layout.sample_rate_hz + 999_999) // 1_000_000,
        )
        if end_frame <= start_frame:
            raise AudioDecodeError("source-time region has no PCM frames")
        with self.path.open("rb") as handle:
            handle.seek(self.layout.data_offset + start_frame * 2)
            payload = handle.read((end_frame - start_frame) * 2)
        expected_bytes = (end_frame - start_frame) * 2
        if len(payload) != expected_bytes:
            raise AudioDecodeError("WAV decode ended before declared data bytes")
        return np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768.0)


def _decode_pcm(payload: bytes, width: int, channels: int) -> list[tuple[float, ...]]:
    stride = width * channels
    if stride <= 0 or len(payload) % stride:
        raise AudioDecodeError("truncated PCM frame")
    out = []
    for pos in range(0, len(payload), stride):
        row = []
        for c in range(channels):
            b = payload[pos + c * width:pos + (c + 1) * width]
            if width == 1:
                value = (b[0] - 128) / 128.0
            else:
                value = int.from_bytes(b + (b'\xff' if width == 3 and b[2] & 0x80 else b'\x00') if width == 3 else b, "little", signed=True) / float(1 << (width * 8 - 1))
            row.append(max(-1.0, min(1.0, value)))
        out.append(tuple(row))
    return out


def _load_numpy_optional() -> Any | None:
    """Load optional NumPy lazily without changing stdlib-only operation."""

    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None
    return np


def _require_numpy_pcm16_mono(layout: WavPcmFormat) -> None:
    if layout.sample_width_bytes != 2 or layout.channel_count != 1:
        raise UnsupportedCodecError("NumPy fast decoding requires PCM16 mono WAV")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_wav_layout(handle: BinaryIO, file_size: int) -> WavPcmFormat:
    handle.seek(0)
    header = handle.read(12)
    if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise UnsupportedContainerError("only RIFF/WAVE is supported")
    fmt: bytes | None = None
    data_offset: int | None = None
    data_bytes: int | None = None
    while handle.tell() + 8 <= file_size:
        chunk_header = handle.read(8)
        if len(chunk_header) != 8:
            break
        chunk_type = chunk_header[:4]
        chunk_size = int.from_bytes(chunk_header[4:], "little")
        chunk_start = handle.tell()
        chunk_end = chunk_start + chunk_size
        if chunk_end > file_size:
            raise AudioDecodeError("WAV chunk exceeds file size")
        if chunk_type == b"fmt ":
            fmt = handle.read(chunk_size)
        elif chunk_type == b"data":
            data_offset, data_bytes = chunk_start, chunk_size
            handle.seek(chunk_size, io.SEEK_CUR)
        else:
            handle.seek(chunk_size, io.SEEK_CUR)
        if chunk_size & 1:
            handle.seek(1, io.SEEK_CUR)
        if fmt is not None and data_offset is not None:
            break
    if fmt is None or data_offset is None or data_bytes is None:
        raise AudioDecodeError("WAV fmt or data chunk is missing")
    if len(fmt) < 16:
        raise AudioDecodeError("WAV fmt chunk is truncated")
    format_tag = int.from_bytes(fmt[0:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    rate = int.from_bytes(fmt[4:8], "little")
    block_align = int.from_bytes(fmt[12:14], "little")
    bits_per_sample = int.from_bytes(fmt[14:16], "little")
    if format_tag == 0xFFFE:
        # WAVEFORMATEXTENSIBLE: valid bits/channel mask/subformat follow cbSize.
        if len(fmt) < 40 or int.from_bytes(fmt[16:18], "little") < 22:
            raise UnsupportedCodecError("truncated WAVE_FORMAT_EXTENSIBLE")
        # The first word of the subformat GUID is the canonical wave format tag.
        format_tag = int.from_bytes(fmt[24:26], "little")
    if format_tag != 1:
        raise UnsupportedCodecError("only integer PCM WAV is supported")
    if channels <= 0 or rate <= 0 or bits_per_sample not in (8, 16, 24, 32):
        raise UnsupportedCodecError("only PCM widths 8/16/24/32 are supported")
    width = bits_per_sample // 8
    if block_align != channels * width or data_bytes % block_align:
        raise AudioDecodeError("invalid PCM block alignment")
    return WavPcmFormat(width, channels, rate, data_bytes // block_align, data_offset, data_bytes)


def _metadata(digest: str, layout: WavPcmFormat) -> AudioSourceMetadata:
    from .contracts import Timebase

    duration_us = (layout.frame_count * 1_000_000 + layout.sample_rate_hz // 2) // layout.sample_rate_hz
    return AudioSourceMetadata(
        digest,
        "wav",
        f"pcm_s{layout.sample_width_bytes * 8}le",
        layout.sample_rate_hz,
        layout.channel_count,
        duration_us,
        Timebase(f"source:{digest[:26]}", source_sample_rate_hz=layout.sample_rate_hz, duration_us=duration_us),
    )


def _mix_resample(decoded: DecodedAudio, target_rate: int) -> tuple[float, ...]:
    mono = tuple(sum(frame) / len(frame) for frame in decoded.samples)
    if decoded.sample_rate_hz == target_rate:
        return mono
    target_count = max(1, (len(mono) * target_rate + decoded.sample_rate_hz // 2) // decoded.sample_rate_hz)
    result = []
    for i in range(target_count):
        source_pos = i * decoded.sample_rate_hz / target_rate
        left = min(int(source_pos), len(mono) - 1)
        right = min(left + 1, len(mono) - 1)
        frac = source_pos - left
        result.append(mono[left] * (1 - frac) + mono[right] * frac)
    return tuple(result)


class AudioFrontend:
    def __init__(self, decoder: MediaDecoder | None = None, policy: NormalizationPolicy | None = None):
        self.decoder = decoder or WavPcmDecoder()
        self.policy = policy or NormalizationPolicy()

    def decode(self, source: str | os.PathLike[str] | bytes) -> DecodedAudio:
        return self.decoder.decode(source)

    def normalize(self, decoded: DecodedAudio) -> NormalizedAudio:
        samples = _mix_resample(decoded, self.policy.target_sample_rate_hz)
        view_id = f"mixdown16k:{decoded.metadata.audio_sha256[:26]}"
        source_end = decoded.metadata.duration_us
        segment = TimeWarpSegment(view_id, view_id, 0, len(samples), 0, max(1, source_end))
        view = AudioView(view_id, "MIXDOWN_MONO", self.policy.target_sample_rate_hz, len(samples), (segment,))
        return NormalizedAudio(samples, view, decoded.metadata)

    def open(self, source: str | os.PathLike[str] | bytes) -> NormalizedAudio:
        return self.normalize(self.decode(source))

    def iter_open_chunks(
        self, source: str | os.PathLike[str], *, source_frames_per_chunk: int = 240_000
    ) -> Iterator[NormalizedAudioChunk]:
        """Bounded WAV decode/resample path for long files.

        It uses deterministic nearest-sample resampling at chunk boundaries;
        production model calibration must treat this as a separately tested
        frontend profile until a higher-fidelity approved adapter is supplied.
        """

        iterator = getattr(self.decoder, "iter_decode_chunks", None)
        if not callable(iterator):
            raise UnsupportedContainerError("configured decoder does not support bounded chunk streaming")
        for chunk in iterator(source, frames_per_chunk=source_frames_per_chunk):
            target_start = (chunk.source_start_sample * self.policy.target_sample_rate_hz + chunk.sample_rate_hz // 2) // chunk.sample_rate_hz
            target_end = (chunk.source_end_sample * self.policy.target_sample_rate_hz + chunk.sample_rate_hz // 2) // chunk.sample_rate_hz
            mixed = tuple(sum(frame) / len(frame) for frame in chunk.samples)
            output: list[float] = []
            for target_index in range(target_start, target_end):
                source_index = min(chunk.source_end_sample - 1, (target_index * chunk.sample_rate_hz) // self.policy.target_sample_rate_hz)
                output.append(mixed[source_index - chunk.source_start_sample])
            source_start_us = (chunk.source_start_sample * 1_000_000 + chunk.sample_rate_hz // 2) // chunk.sample_rate_hz
            source_end_us = (chunk.source_end_sample * 1_000_000 + chunk.sample_rate_hz // 2) // chunk.sample_rate_hz
            segment = TimeWarpSegment(
                f"stream:{chunk.metadata.audio_sha256[:26]}:{target_start}",
                f"mixdown16k:{chunk.metadata.audio_sha256[:26]}",
                target_start,
                max(target_start + 1, target_end),
                source_start_us,
                max(source_start_us + 1, source_end_us),
            )
            yield NormalizedAudioChunk(tuple(output), target_start, target_end, segment, chunk.metadata)
