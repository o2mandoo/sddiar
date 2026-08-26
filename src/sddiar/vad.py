"""Offline VAD contracts and deterministic development backends."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .media import AudioChunk, NormalizedAudio
from .ort_cpu import create_ort_session


class VadError(Exception):
    code = "VAD_INFERENCE_FAILED"


class VadUnavailableError(VadError):
    code = "VAD_BACKEND_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class VadFrame:
    start_us: int
    end_us: int
    speech_evidence: float | None
    is_speech: bool

    def __post_init__(self) -> None:
        if type(self.start_us) is not int or type(self.end_us) is not int or not 0 <= self.start_us < self.end_us:
            raise ValueError("invalid VAD frame range")
        if self.speech_evidence is not None and not 0 <= self.speech_evidence <= 1:
            raise ValueError("speech evidence must be in [0,1]")


class VadBackend(Protocol):
    def infer(self, audio: NormalizedAudio) -> tuple[VadFrame, ...]: ...


class EnergyVadBackend:
    """Simple deterministic RMS VAD, intended only for fixtures and probes."""
    def __init__(self, frame_ms: int = 30, hop_ms: int = 10, threshold: float = 0.02):
        if frame_ms <= 0 or hop_ms <= 0 or threshold < 0:
            raise ValueError("invalid EnergyVad configuration")
        self.frame_ms, self.hop_ms, self.threshold = frame_ms, hop_ms, threshold

    def infer(self, audio: NormalizedAudio) -> tuple[VadFrame, ...]:
        rate = audio.view.sample_rate_hz
        frame = max(1, round(rate * self.frame_ms / 1000))
        hop = max(1, round(rate * self.hop_ms / 1000))
        result = []
        for start in range(0, len(audio.samples), hop):
            end = min(len(audio.samples), start + frame)
            if end <= start:
                continue
            rms = math.sqrt(sum(x * x for x in audio.samples[start:end]) / (end - start))
            evidence = min(1.0, rms / max(self.threshold, 1e-12))
            # Mapping uses the single affine source segment and integer us.
            seg = audio.view.time_warp[0]
            start_us, end_us = seg.source_us(start), seg.source_us(end)
            if end_us <= start_us:
                continue
            result.append(VadFrame(start_us, end_us, evidence, rms >= self.threshold))
        return tuple(result)


class SileroOnnxVadBackend:
    """Capability boundary for a pre-bundled Silero ONNX model.

    This class deliberately does not download, locate, or auto-install anything.
    The actual model input contract is supplied by the approved model pack.
    """
    def __init__(self, model_path: str, ort_session=None):
        if not model_path:
            raise VadUnavailableError("Silero model path is required")
        self.model_path = model_path
        self._session = ort_session

    def infer(self, audio: NormalizedAudio) -> tuple[VadFrame, ...]:
        if self._session is None:
            if not self.model_path:
                raise VadUnavailableError("Silero model is unavailable")
            try:
                self._session = create_ort_session(self.model_path)
            except Exception as exc:
                raise VadUnavailableError(f"cannot open local Silero model: {exc}") from exc
        # Model-specific tensor/state contracts are pack metadata, not guessed here.
        raise VadError("Silero ONNX input contract is not configured")
