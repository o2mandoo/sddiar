"""Deterministic temporal decoding for Silero probability frames.

This module is deliberately a postprocessor only.  It does not import an audio
decoder, a model runtime, or a network client.  ``VadFrame`` objects are already
in source time when they arrive here; all output coordinates therefore remain
integer microseconds in that same source clock.

The decoder follows the useful part of Silero's upstream state machine: a
frame at/above the onset threshold starts speech, while a frame below the
negative threshold only ends speech after a sustained silence.  Padding is
represented as *halo*, never silently promoted to clean/core speech.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .vad import VadFrame


class TemporalVadError(ValueError):
    """Malformed or non-contiguous temporal VAD evidence."""


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    """One decoded region and its explicit clean/core interval.

    ``start_us``/``end_us`` are the public padded (core + halo) bounds.
    ``core_start_us``/``core_end_us`` are the unpadded interval supported by
    the trigger state machine.  A consumer that needs clean anchors must use
    the core fields, not the padded bounds.
    """

    start_us: int
    end_us: int
    core_start_us: int
    core_end_us: int

    def __post_init__(self) -> None:
        values = (self.start_us, self.end_us, self.core_start_us, self.core_end_us)
        if any(type(value) is not int for value in values):
            raise TemporalVadError("speech region coordinates must be integers")
        if not (self.start_us <= self.core_start_us < self.core_end_us <= self.end_us):
            raise TemporalVadError("invalid speech region/core bounds")

    @property
    def core_duration_us(self) -> int:
        return self.core_end_us - self.core_start_us

    @property
    def halo_before_us(self) -> int:
        return self.core_start_us - self.start_us

    @property
    def halo_after_us(self) -> int:
        return self.end_us - self.core_end_us

    @property
    def halo_duration_us(self) -> int:
        return self.halo_before_us + self.halo_after_us

    # Friendly aliases for callers that use ``start``/``end`` regions.
    @property
    def start(self) -> int:
        return self.start_us

    @property
    def end(self) -> int:
        return self.end_us


@dataclass(frozen=True, slots=True)
class MarkedVadFrame:
    """A source-time piece of a VAD frame marked as core or halo.

    Frames are split at region/core boundaries when needed.  Consequently a
    low-probability frame that overlaps only padding is represented as
    ``is_halo=True, is_core=False, is_speech=False`` rather than as clean
    speech.  ``frame`` retains the original evidence object for provenance.
    """

    start_us: int
    end_us: int
    speech_evidence: float
    region_index: int
    frame: VadFrame
    is_core: bool
    is_halo: bool

    def __post_init__(self) -> None:
        if type(self.start_us) is not int or type(self.end_us) is not int or self.start_us >= self.end_us:
            raise TemporalVadError("invalid marked VAD frame range")
        if not math.isfinite(self.speech_evidence) or not 0 <= self.speech_evidence <= 1:
            raise TemporalVadError("invalid marked VAD evidence")
        if self.is_core == self.is_halo:
            raise TemporalVadError("a marked frame must be exactly core or halo")

    @property
    def is_speech(self) -> bool:
        """Core speech only; halo is intentionally not clean speech."""

        return self.is_core

    @property
    def kind(self) -> str:
        return "core" if self.is_core else "halo"


@dataclass(frozen=True, slots=True)
class TemporalVadResult:
    """Regions, optional marks, and auditable core/halo duration totals."""

    regions: tuple[SpeechRegion, ...]
    marked_frames: tuple[MarkedVadFrame, ...] = ()

    @property
    def core_duration_us(self) -> int:
        return sum(region.core_duration_us for region in self.regions)

    @property
    def halo_duration_us(self) -> int:
        return sum(region.halo_duration_us for region in self.regions)

    @property
    def speech_duration_us(self) -> int:
        return self.core_duration_us + self.halo_duration_us

    @property
    def core_us(self) -> int:
        return self.core_duration_us

    @property
    def halo_us(self) -> int:
        return self.halo_duration_us


@dataclass(frozen=True, slots=True)
class TemporalVadConfig:
    """Integer source-time policy for :class:`SileroTemporalPostprocessor`."""

    onset_threshold: float = 0.5
    negative_threshold: float = 0.35
    min_silence_duration_us: int = 100_000
    speech_pad_us: int = 30_000
    max_speech_duration_us: int | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.onset_threshold) and 0 < self.onset_threshold <= 1):
            raise ValueError("onset threshold must be in (0,1]")
        if not (math.isfinite(self.negative_threshold) and 0 <= self.negative_threshold < self.onset_threshold):
            raise ValueError("negative threshold must be below onset threshold")
        if type(self.min_silence_duration_us) is not int or self.min_silence_duration_us < 0:
            raise ValueError("minimum silence must be a non-negative integer number of microseconds")
        if type(self.speech_pad_us) is not int or self.speech_pad_us < 0:
            raise ValueError("speech pad must be a non-negative integer number of microseconds")
        if self.max_speech_duration_us is not None and (
            type(self.max_speech_duration_us) is not int or self.max_speech_duration_us <= 0
        ):
            raise ValueError("maximum speech duration must be a positive integer number of microseconds")


def _validated_frames(frames: Iterable[VadFrame]) -> tuple[VadFrame, ...]:
    try:
        values = tuple(frames)
    except TypeError as exc:
        raise TemporalVadError("VAD frames must be iterable") from exc
    if not values:
        return ()
    previous_end: int | None = None
    for frame in values:
        if not isinstance(frame, VadFrame):
            raise TemporalVadError("temporal decoder requires VadFrame evidence")
        evidence = frame.speech_evidence
        if evidence is None or not math.isfinite(evidence):
            raise TemporalVadError("temporal decoder requires finite probability evidence")
        if previous_end is not None and frame.start_us != previous_end:
            raise TemporalVadError("VAD frames are not source-time contiguous")
        previous_end = frame.end_us
    return values


def _core_regions(frames: Sequence[VadFrame], cfg: TemporalVadConfig) -> list[tuple[int, int]]:
    """Decode core intervals, retaining source boundaries before padding."""

    regions: list[tuple[int, int]] = []
    active = False
    core_start = core_end = 0
    silence_start: int | None = None
    max_end: int | None = None

    def emit(end: int) -> None:
        nonlocal active, core_start, core_end, silence_start, max_end
        if active and end > core_start:
            regions.append((core_start, end))
        active = False
        core_start = core_end = 0
        silence_start = None
        max_end = None

    for frame in frames:
        probability = frame.speech_evidence
        assert probability is not None  # checked by _validated_frames

        if not active:
            if probability >= cfg.onset_threshold:
                active = True
                core_start = frame.start_us
                core_end = frame.end_us
                max_end = (
                    core_start + cfg.max_speech_duration_us
                    if cfg.max_speech_duration_us is not None
                    else None
                )
            continue

        # A configured maximum is a deterministic hard boundary.  If one
        # frame crosses it, split that frame at the exact source timestamp and
        # continue only when its evidence still supports the active state.
        if max_end is not None and frame.end_us > max_end:
            if probability >= cfg.negative_threshold:
                continuation_start = max_end
                emit(max_end)
                active = True
                core_start = continuation_start
                core_end = frame.end_us
                # ``max_end`` was configured, so the duration cannot be None
                # on this branch (the local spelling keeps type checkers happy
                # without weakening the public config validation).
                assert cfg.max_speech_duration_us is not None
                max_end = core_start + cfg.max_speech_duration_us
                silence_start = None
                continue
            # Low evidence should close at its observed start; do not invent a
            # continuation merely because a frame crossed max_end.

        if probability < cfg.negative_threshold:
            if silence_start is None:
                silence_start = frame.start_us
            if frame.start_us - silence_start >= cfg.min_silence_duration_us:
                emit(silence_start)
            # Low-probability frames are tolerated as silence, but never become
            # part of the clean/core interval.
            continue

        # Hysteresis band [negative_threshold, onset_threshold) keeps the
        # current speech state alive and is therefore core evidence.
        silence_start = None
        core_end = frame.end_us

    if active:
        # At EOF only the last non-silence evidence is core.  Any trailing
        # low-probability tolerance is represented, at most, by the pad halo.
        emit(core_end)
    return regions


def _padded_regions(
    core_regions: Sequence[tuple[int, int]],
    timeline_start: int,
    timeline_end: int,
    pad_us: int,
) -> tuple[SpeechRegion, ...]:
    if not core_regions:
        return ()
    starts = [max(timeline_start, start - pad_us) for start, _ in core_regions]
    ends = [min(timeline_end, end + pad_us) for _, end in core_regions]
    for index in range(len(core_regions) - 1):
        left_end = core_regions[index][1]
        right_start = core_regions[index + 1][0]
        gap = max(0, right_start - left_end)
        if gap < 2 * pad_us:
            # Share a small gap between adjacent pads, preserving non-overlap
            # and exact integer coverage (including odd-microsecond gaps).
            left_extra = gap // 2
            starts[index + 1] = max(timeline_start, right_start - (gap - left_extra))
            ends[index] = min(timeline_end, left_end + left_extra)
    return tuple(
        SpeechRegion(
            start_us=starts[index],
            end_us=ends[index],
            core_start_us=start,
            core_end_us=end,
        )
        for index, (start, end) in enumerate(core_regions)
    )


def mark_vad_frames(
    frames: Iterable[VadFrame],
    regions: Sequence[SpeechRegion],
) -> tuple[MarkedVadFrame, ...]:
    """Mark source frame pieces covered by regions as core or halo."""

    values = _validated_frames(frames)
    marks: list[MarkedVadFrame] = []
    for frame in values:
        evidence = frame.speech_evidence
        assert evidence is not None
        for region_index, region in enumerate(regions):
            overlap_start = max(frame.start_us, region.start_us)
            overlap_end = min(frame.end_us, region.end_us)
            if overlap_start >= overlap_end:
                continue
            # Use only the four relevant clipped coordinates.  In particular,
            # never expand a source-time interval into one item per microsecond.
            boundaries = sorted(
                {overlap_start, overlap_end}
                | {
                    boundary
                    for boundary in (region.core_start_us, region.core_end_us)
                    if overlap_start < boundary < overlap_end
                }
            )
            for start, end in zip(boundaries, boundaries[1:]):
                is_core = start >= region.core_start_us and end <= region.core_end_us
                marks.append(
                    MarkedVadFrame(
                        start_us=start,
                        end_us=end,
                        speech_evidence=evidence,
                        region_index=region_index,
                        frame=frame,
                        is_core=is_core,
                        is_halo=not is_core,
                    )
                )
    return tuple(marks)


class SileroTemporalPostprocessor:
    """Default-off, dependency-free Silero temporal decoder."""

    def __init__(self, config: TemporalVadConfig | None = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise TypeError("pass either config or temporal policy keywords")
        # Accept the names used by Silero's upstream helper as a small,
        # dependency-free migration seam.  Canonical config fields remain
        # explicit source-time names in this module.
        aliases = {
            "threshold": "onset_threshold",
            "neg_threshold": "negative_threshold",
            "min_silence_duration_ms": "_min_silence_duration_ms",
            "speech_pad_ms": "_speech_pad_ms",
            "max_speech_duration_s": "_max_speech_duration_s",
        }
        normalized = dict(kwargs)
        for old, new in aliases.items():
            if old in normalized:
                if new in normalized:
                    raise TypeError(f"pass only one of {old!r} and its canonical alias")
                normalized[new] = normalized.pop(old)
        if "_min_silence_duration_ms" in normalized:
            normalized["min_silence_duration_us"] = round(
                float(normalized.pop("_min_silence_duration_ms")) * 1_000
            )
        if "_speech_pad_ms" in normalized:
            normalized["speech_pad_us"] = round(float(normalized.pop("_speech_pad_ms")) * 1_000)
        if "_max_speech_duration_s" in normalized:
            value = normalized.pop("_max_speech_duration_s")
            normalized["max_speech_duration_us"] = None if value is None else round(float(value) * 1_000_000)
        kwargs = normalized
        self.config = config or TemporalVadConfig(**kwargs)

    def regions(self, frames: Iterable[VadFrame]) -> tuple[SpeechRegion, ...]:
        values = _validated_frames(frames)
        if not values:
            return ()
        return _padded_regions(
            _core_regions(values, self.config),
            values[0].start_us,
            values[-1].end_us,
            self.config.speech_pad_us,
        )

    def mark(
        self,
        frames: Iterable[VadFrame],
        regions: Sequence[SpeechRegion] | None = None,
    ) -> tuple[MarkedVadFrame, ...]:
        values = _validated_frames(frames)
        decoded = self.regions(values) if regions is None else tuple(regions)
        return mark_vad_frames(values, decoded)

    # ``mark_frames`` reads naturally at call sites that do not need the
    # stateful-sounding ``mark`` spelling.
    mark_frames = mark

    def process(self, frames: Iterable[VadFrame], *, mark: bool = False) -> TemporalVadResult:
        values = _validated_frames(frames)
        decoded = self.regions(values)
        marked = mark_vad_frames(values, decoded) if mark else ()
        return TemporalVadResult(decoded, marked)


def postprocess_vad(
    frames: Iterable[VadFrame],
    *,
    onset_threshold: float = 0.5,
    negative_threshold: float = 0.35,
    min_silence_duration_us: int = 100_000,
    speech_pad_us: int = 30_000,
    max_speech_duration_us: int | None = None,
) -> tuple[SpeechRegion, ...]:
    """Convenience function returning deterministic padded speech regions."""

    return SileroTemporalPostprocessor(
        onset_threshold=onset_threshold,
        negative_threshold=negative_threshold,
        min_silence_duration_us=min_silence_duration_us,
        speech_pad_us=speech_pad_us,
        max_speech_duration_us=max_speech_duration_us,
    ).regions(frames)


# Names useful to callers migrating from a generic temporal decoder API.
silero_temporal_postprocess = postprocess_vad
apply_silero_temporal = postprocess_vad


__all__ = [
    "MarkedVadFrame",
    "SpeechRegion",
    "SileroTemporalPostprocessor",
    "TemporalVadConfig",
    "TemporalVadError",
    "TemporalVadResult",
    "apply_silero_temporal",
    "mark_vad_frames",
    "postprocess_vad",
    "silero_temporal_postprocess",
]
