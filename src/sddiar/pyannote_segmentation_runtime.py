"""Development-only pyannote segmentation-3.0 ONNX evidence adapter.

This module is deliberately not a diarization or clustering implementation.
It converts a hash-verified local segmentation model into redacted frame-level
speech, overlap, and conservative within-window change evidence.  No event is
approved for tracklet splitting and no threshold is claimed to be calibrated.

Audio is read in bounded 10 second PCM windows with a one second shift.  Local
speaker axes are aligned across overlapping windows only so the diagnostic
change evidence is permutation-stable; local identities and raw logits are
never exposed by the public result.
"""
from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from .media import AudioDecodeError, WavPcmAccessor
from .ort_cpu import create_ort_session


# Keep the package importable on Windows, where POSIX ``resource`` is absent.
# This adapter reports RSS for diagnostics only; an unavailable measurement is
# represented as ``None`` and never changes an inference decision.
try:  # pragma: no cover - platform dependent
    import resource as _resource
except ImportError:  # pragma: no cover - platform dependent
    _resource = None


EXPECTED_METADATA = {
    "sample_rate": "16000",
    "window_size": "160000",
    "receptive_field_size": "991",
    "receptive_field_shift": "270",
    "num_speakers": "3",
    "powerset_max_classes": "2",
    "num_classes": "7",
    "model_type": "pyannote-segmentation-3.0",
    "version": "1",
    "model_author": "pyannote",
    "maintainer": "k2-fsa",
}
EXPECTED_LICENSE_URL = "https://huggingface.co/pyannote/segmentation-3.0/blob/main/LICENSE"
POWERSET_MAPPING = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
)
PERMUTATIONS = tuple(itertools.permutations(range(3)))


class PyannoteSegmentationError(RuntimeError):
    """Base error for a rejected artifact, runtime, or evidence contract."""


class PyannoteSegmentationHashError(PyannoteSegmentationError):
    """The local model is missing, unsafe, or does not match its expected hash."""


class PyannoteSegmentationMetadataError(PyannoteSegmentationError):
    """The ONNX metadata or I/O schema does not match segmentation-3.0."""


@dataclass(frozen=True, slots=True)
class PyannoteSegmentationConfig:
    window_samples: int = 160_000
    window_shift_samples: int = 16_000
    receptive_field_samples: int = 991
    receptive_field_shift_samples: int = 270
    change_context_frames: int = 15
    change_min_clean_confidence: float = 0.55
    change_min_purity: float = 0.65
    diagnostic_event_threshold: float = 0.20
    diagnostic_event_min_separation_us: int = 250_000

    def __post_init__(self) -> None:
        exact = {
            "window_samples": 160_000,
            "window_shift_samples": 16_000,
            "receptive_field_samples": 991,
            "receptive_field_shift_samples": 270,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must remain {expected} for the inspected model")
        if type(self.change_context_frames) is not int or self.change_context_frames <= 0:
            raise ValueError("change_context_frames must be positive")
        for name in (
            "change_min_clean_confidence",
            "change_min_purity",
            "diagnostic_event_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1)")
        if (
            type(self.diagnostic_event_min_separation_us) is not int
            or self.diagnostic_event_min_separation_us <= 0
        ):
            raise ValueError("diagnostic_event_min_separation_us must be positive")


@dataclass(frozen=True, slots=True)
class PyannoteFrameEvidence:
    frame_index: int
    start_us: int
    end_us: int
    center_us: int
    speech_probability: float
    overlap_probability: float
    speaker_change_evidence: float
    window_support: int
    padded_window_support: int

    def __post_init__(self) -> None:
        if self.frame_index < 0 or not 0 <= self.start_us < self.end_us:
            raise PyannoteSegmentationError("invalid source-time evidence frame")
        if not self.start_us <= self.center_us <= self.end_us:
            raise PyannoteSegmentationError("evidence center is outside its source-time frame")
        for name in (
            "speech_probability",
            "overlap_probability",
            "speaker_change_evidence",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise PyannoteSegmentationError(f"{name} must be finite and in [0, 1]")
        if self.window_support <= 0 or not 0 <= self.padded_window_support <= self.window_support:
            raise PyannoteSegmentationError("invalid evidence window support")


@dataclass(frozen=True, slots=True)
class PyannoteChangeEvent:
    frame_index: int
    time_us: int
    evidence: float
    reason_code: str = "UNCALIBRATED_WITHIN_WINDOW_CHANGE"

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.time_us < 0:
            raise PyannoteSegmentationError("invalid diagnostic event time")
        if not math.isfinite(self.evidence) or not 0.0 <= self.evidence <= 1.0:
            raise PyannoteSegmentationError("invalid diagnostic event evidence")


@dataclass(frozen=True, slots=True)
class PyannoteSegmentationResult:
    schema: str
    result_kind: str
    source_audio_sha256_prefix: str
    segment_start_us: int
    segment_end_us: int
    model_sha256_prefix: str
    model_variant: str
    frames: tuple[PyannoteFrameEvidence, ...]
    diagnostic_change_events: tuple[PyannoteChangeEvent, ...]
    window_count: int
    padded_window_count: int
    elapsed_wall_sec: float
    model_inference_wall_sec: float
    peak_rss_mb: float | None

    def to_dict(self, *, include_frames: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "result_kind": self.result_kind,
            "source_audio_sha256_prefix": self.source_audio_sha256_prefix,
            "segment_start_us": self.segment_start_us,
            "segment_end_us": self.segment_end_us,
            "model_sha256_prefix": self.model_sha256_prefix,
            "model_variant": self.model_variant,
            "window_count": self.window_count,
            "padded_window_count": self.padded_window_count,
            "frame_count": len(self.frames),
            "diagnostic_change_events": [
                {
                    "frame_index": event.frame_index,
                    "time_us": event.time_us,
                    "evidence": event.evidence,
                    "reason_code": event.reason_code,
                }
                for event in self.diagnostic_change_events
            ],
            "elapsed_wall_sec": self.elapsed_wall_sec,
            "model_inference_wall_sec": self.model_inference_wall_sec,
            "peak_rss_mb": self.peak_rss_mb,
            "calibration_status": "UNCALIBRATED_EVIDENCE_ONLY",
            "approved_for_tracklet_split": False,
            "approved_for_overlap_assignment": False,
            "redaction": {
                "source_path": "omitted",
                "audio_samples": "omitted",
                "raw_logits": "omitted",
                "local_speaker_probabilities": "omitted",
            },
        }
        if include_frames:
            value["frames"] = [
                {
                    "frame_index": frame.frame_index,
                    "start_us": frame.start_us,
                    "end_us": frame.end_us,
                    "center_us": frame.center_us,
                    "speech_probability": frame.speech_probability,
                    "overlap_probability": frame.overlap_probability,
                    "speaker_change_evidence": frame.speaker_change_evidence,
                    "window_support": frame.window_support,
                    "padded_window_support": frame.padded_window_support,
                }
                for frame in self.frames
            ]
        return value

    def to_json(self, *, include_frames: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_frames=include_frames),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class PyannoteParityTrace:
    """Development trace used only for FP32/INT8 parity, never public output."""

    window_argmax: tuple[int, ...]
    window_count: int
    frames_per_window: int
    permutation_sha256: str


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifact(path: Path, expected_sha256: str) -> str:
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise PyannoteSegmentationHashError("expected model SHA-256 is invalid")
    if not path.is_file() or path.is_symlink() or path.suffix.lower() != ".onnx":
        raise PyannoteSegmentationHashError("model must be a regular non-symlink ONNX file")
    digest = _sha256_path(path)
    if digest != expected_sha256:
        raise PyannoteSegmentationHashError("model SHA-256 mismatch")
    return digest


def _load_numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise PyannoteSegmentationError("numpy is required for pyannote segmentation") from exc


def _shape_rank(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _validate_session(session: Any) -> Mapping[str, str]:
    providers = tuple(getattr(session, "get_providers", lambda: ())())
    if providers != ("CPUExecutionProvider",):
        raise PyannoteSegmentationMetadataError("segmentation session must be CPU-only")
    metadata = dict(getattr(session.get_modelmeta(), "custom_metadata_map", {}))
    for key, expected in EXPECTED_METADATA.items():
        if metadata.get(key) != expected:
            raise PyannoteSegmentationMetadataError(
                f"segmentation metadata {key} mismatch: {metadata.get(key)!r}"
            )
    if metadata.get("license") != EXPECTED_LICENSE_URL:
        raise PyannoteSegmentationMetadataError("segmentation license metadata mismatch")
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise PyannoteSegmentationMetadataError("segmentation graph must have one input and one output")
    inp, out = inputs[0], outputs[0]
    if (
        inp.name != "x"
        or inp.type != "tensor(float)"
        or _shape_rank(inp.shape) != 3
        or inp.shape[1] not in (1, "1")
    ):
        raise PyannoteSegmentationMetadataError("input must be float x[batch,1,samples]")
    if (
        out.name != "y"
        or out.type != "tensor(float)"
        or _shape_rank(out.shape) != 3
        or out.shape[-1] not in (7, "7")
    ):
        raise PyannoteSegmentationMetadataError("output must be float y[batch,frames,7]")
    return metadata


def _sample_to_us(sample: int, sample_rate: int = 16_000) -> int:
    return (sample * 1_000_000 + sample_rate // 2) // sample_rate


def _segment_samples(layout: Any, start_us: int, end_us: int | None) -> tuple[int, int]:
    if type(start_us) is not int or start_us < 0:
        raise PyannoteSegmentationError("segment_start_us must be a non-negative integer")
    start = (start_us * layout.sample_rate_hz) // 1_000_000
    if end_us is None:
        end = layout.frame_count
    else:
        if type(end_us) is not int or end_us <= start_us:
            raise PyannoteSegmentationError("segment_end_us must be greater than segment_start_us")
        end = min(
            layout.frame_count,
            (end_us * layout.sample_rate_hz + 999_999) // 1_000_000,
        )
    start = min(start, layout.frame_count)
    if end <= start:
        raise PyannoteSegmentationError("selected source-time segment is empty")
    return start, end


def bounded_window_starts(
    segment_start_sample: int,
    segment_end_sample: int,
    *,
    window_samples: int = 160_000,
    shift_samples: int = 16_000,
) -> tuple[int, ...]:
    """Return full windows plus at most one deterministically padded tail."""
    length = segment_end_sample - segment_start_sample
    if length <= 0:
        raise PyannoteSegmentationError("window segment is empty")
    if length <= window_samples:
        return (segment_start_sample,)
    last_full = segment_end_sample - window_samples
    starts = list(range(segment_start_sample, last_full + 1, shift_samples))
    next_start = starts[-1] + shift_samples
    if next_start < segment_end_sample and starts[-1] != last_full:
        starts.append(next_start)
    return tuple(starts)


def _read_pcm16_window(
    accessor: WavPcmAccessor,
    start_sample: int,
    segment_end_sample: int,
    window_samples: int,
    *,
    np: Any,
) -> tuple[Any, int]:
    layout = accessor.layout
    if (layout.sample_width_bytes, layout.channel_count, layout.sample_rate_hz) != (2, 1, 16_000):
        raise PyannoteSegmentationError("pyannote adapter requires PCM16 mono 16 kHz WAV")
    valid = min(window_samples, segment_end_sample - start_sample)
    if valid <= 0:
        raise PyannoteSegmentationError("window has no source samples")
    with accessor.path.open("rb") as handle:
        handle.seek(layout.data_offset + start_sample * 2)
        payload = handle.read(valid * 2)
    if len(payload) != valid * 2:
        raise AudioDecodeError("WAV ended before a segmentation window")
    values = np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    if valid < window_samples:
        values = np.pad(values, (0, window_samples - valid), mode="constant")
    return np.ascontiguousarray(values, dtype=np.float32), valid


def _softmax_log_scores(log_scores: Any, *, np: Any) -> Any:
    values = np.asarray(log_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise PyannoteSegmentationError("segmentation output must be finite [frames,7]")
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    denominator = exponent.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0.0):
        raise PyannoteSegmentationError("segmentation probability normalization failed")
    return exponent / denominator


def align_local_speakers(
    current_activity: Any,
    current_speech: Any,
    current_overlap: Any,
    reference_activity: Any,
    reference_speech: Any,
    reference_overlap: Any,
    weights: Any,
    *,
    np: Any,
) -> tuple[int, int, int]:
    """Choose a deterministic local-speaker permutation on reliable overlap."""
    arrays = (
        current_activity,
        reference_activity,
    )
    if any(array.ndim != 2 or array.shape[1] != 3 for array in arrays):
        raise PyannoteSegmentationError("speaker alignment requires [frames,3] activity")
    if current_activity.shape != reference_activity.shape:
        raise PyannoteSegmentationError("speaker alignment shapes differ")
    reliability = (
        np.asarray(weights, dtype=np.float64)
        * np.minimum(current_speech, reference_speech)
        * np.maximum(0.0, 1.0 - current_overlap)
        * np.maximum(0.0, 1.0 - reference_overlap)
    )
    reliable = reliability >= 0.05
    if int(reliable.sum()) < 10 or float(reliability[reliable].sum()) < 1.0:
        return (0, 1, 2)
    scores: list[tuple[float, tuple[int, int, int]]] = []
    denominator = float(reliability[reliable].sum())
    for permutation in PERMUTATIONS:
        aligned = current_activity[:, permutation]
        squared = np.sum((aligned - reference_activity) ** 2, axis=1)
        score = float(np.sum(squared[reliable] * reliability[reliable]) / denominator)
        scores.append((round(score, 15), permutation))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def conservative_within_window_change(
    activity: Any,
    speech: Any,
    overlap: Any,
    *,
    context_frames: int,
    min_clean_confidence: float,
    min_purity: float,
    np: Any,
) -> Any:
    """Return conservative, uncalibrated change evidence within one window."""
    activity = np.asarray(activity, dtype=np.float64)
    speech = np.asarray(speech, dtype=np.float64)
    overlap = np.asarray(overlap, dtype=np.float64)
    frame_count = activity.shape[0]
    result = np.zeros(frame_count, dtype=np.float64)
    if activity.ndim != 2 or activity.shape[1] != 3 or frame_count < 2 * context_frames + 1:
        return result
    clean = np.clip(speech * (1.0 - overlap), 0.0, 1.0)
    weighted_activity = activity * clean[:, np.newaxis]
    cumulative_weight = np.concatenate(([0.0], np.cumsum(clean)))
    cumulative_activity = np.vstack((np.zeros((1, 3)), np.cumsum(weighted_activity, axis=0)))
    centers = np.arange(context_frames, frame_count - context_frames)
    left_start, left_end = centers - context_frames, centers
    right_start, right_end = centers, centers + context_frames
    left_weight = cumulative_weight[left_end] - cumulative_weight[left_start]
    right_weight = cumulative_weight[right_end] - cumulative_weight[right_start]
    left_sum = cumulative_activity[left_end] - cumulative_activity[left_start]
    right_sum = cumulative_activity[right_end] - cumulative_activity[right_start]
    left_distribution = left_sum / np.maximum(left_sum.sum(axis=1, keepdims=True), 1e-12)
    right_distribution = right_sum / np.maximum(right_sum.sum(axis=1, keepdims=True), 1e-12)
    left_clean = left_weight / context_frames
    right_clean = right_weight / context_frames
    left_purity = left_distribution.max(axis=1)
    right_purity = right_distribution.max(axis=1)
    different = left_distribution.argmax(axis=1) != right_distribution.argmax(axis=1)
    total_variation = 0.5 * np.abs(left_distribution - right_distribution).sum(axis=1)
    valid = (
        different
        & (left_clean >= min_clean_confidence)
        & (right_clean >= min_clean_confidence)
        & (left_purity >= min_purity)
        & (right_purity >= min_purity)
    )
    evidence = (
        total_variation
        * np.minimum(left_clean, right_clean)
        * np.minimum(left_purity, right_purity)
    )
    result[centers[valid]] = np.clip(evidence[valid], 0.0, 1.0)
    return result


def diagnostic_change_events(
    frames: Sequence[PyannoteFrameEvidence],
    *,
    threshold: float,
    min_separation_us: int,
) -> tuple[PyannoteChangeEvent, ...]:
    """Extract uncalibrated local maxima solely for model-parity comparison."""
    candidates: list[PyannoteFrameEvidence] = []
    for index, frame in enumerate(frames):
        if frame.speaker_change_evidence < threshold:
            continue
        left = frames[index - 1].speaker_change_evidence if index else -1.0
        right = frames[index + 1].speaker_change_evidence if index + 1 < len(frames) else -1.0
        if frame.speaker_change_evidence >= left and frame.speaker_change_evidence >= right:
            candidates.append(frame)
    selected: list[PyannoteFrameEvidence] = []
    for frame in sorted(
        candidates,
        key=lambda item: (-item.speaker_change_evidence, item.center_us, item.frame_index),
    ):
        if all(abs(frame.center_us - prior.center_us) >= min_separation_us for prior in selected):
            selected.append(frame)
    return tuple(
        PyannoteChangeEvent(frame.frame_index, frame.center_us, frame.speaker_change_evidence)
        for frame in sorted(selected, key=lambda item: (item.center_us, item.frame_index))
    )


class PyannoteSegmentationOnnxRuntime:
    """Strict CPU-only segmentation-3.0 evidence runtime."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        expected_sha256: str,
        session: Any | None = None,
        config: PyannoteSegmentationConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_sha256 = _verify_artifact(self.model_path, expected_sha256)
        self.config = config or PyannoteSegmentationConfig()
        self._np = _load_numpy()
        self._session = session or create_ort_session(self.model_path, threads=1)
        self.metadata = _validate_session(self._session)
        self.input_name = self._session.get_inputs()[0].name
        self.output_name = self._session.get_outputs()[0].name
        self.model_variant = (
            "dynamic_uint8"
            if "onnx.infer" in self.metadata
            else "fp32"
            if "onnx.quant.pre_process" in self.metadata
            else "inspected_unknown_precision"
        )
        self._mapping = self._np.asarray(POWERSET_MAPPING, dtype=self._np.float64)
        self._window_weights: Any | None = None

    @staticmethod
    def _rss_mb() -> float | None:
        if _resource is not None:
            try:
                raw = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
                value = raw if sys.platform == "darwin" else raw * 1024
                if value >= 0:
                    return round(value / (1024 * 1024), 2)
            except (AttributeError, OSError, TypeError, ValueError):
                pass
        try:
            import psutil  # optional compatibility fallback

            value = int(psutil.Process().memory_info().rss)
            if value >= 0:
                return round(value / (1024 * 1024), 2)
        except (ImportError, OSError, AttributeError, TypeError, ValueError):
            pass
        return None

    def _infer(self, samples: Any) -> tuple[Any, Any, Any, Any, Any]:
        np = self._np
        tensor = np.asarray(samples, dtype=np.float32)
        if tensor.shape != (self.config.window_samples,) or not np.isfinite(tensor).all():
            raise PyannoteSegmentationError("one segmentation window must be finite float32[160000]")
        raw = self._session.run(
            [self.output_name],
            {self.input_name: np.ascontiguousarray(tensor[np.newaxis, np.newaxis, :])},
        )[0]
        raw = np.asarray(raw)
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 7:
            raise PyannoteSegmentationError("segmentation graph returned an invalid shape")
        probabilities = _softmax_log_scores(raw[0], np=np)
        speech = np.clip(1.0 - probabilities[:, 0], 0.0, 1.0)
        overlap = np.clip(probabilities[:, 4:7].sum(axis=1), 0.0, 1.0)
        activity = np.clip(probabilities @ self._mapping, 0.0, 1.0)
        argmax = raw[0].argmax(axis=1).astype(np.int8)
        return probabilities, speech, overlap, activity, argmax

    def process_wav(
        self,
        audio_path: str | Path,
        *,
        segment_start_us: int = 0,
        segment_end_us: int | None = None,
    ) -> PyannoteSegmentationResult:
        result, _ = self.process_wav_with_trace(
            audio_path,
            segment_start_us=segment_start_us,
            segment_end_us=segment_end_us,
        )
        return result

    def process_wav_with_trace(
        self,
        audio_path: str | Path,
        *,
        segment_start_us: int = 0,
        segment_end_us: int | None = None,
    ) -> tuple[PyannoteSegmentationResult, PyannoteParityTrace]:
        """Run bounded evidence inference and return an internal parity trace."""
        started = time.perf_counter()
        np = self._np
        accessor = WavPcmAccessor(audio_path)
        layout = accessor.layout
        if (layout.sample_width_bytes, layout.channel_count, layout.sample_rate_hz) != (2, 1, 16_000):
            raise PyannoteSegmentationError("pyannote adapter requires PCM16 mono 16 kHz WAV")
        segment_start, segment_end = _segment_samples(layout, segment_start_us, segment_end_us)
        segment_length = segment_end - segment_start
        starts = bounded_window_starts(
            segment_start,
            segment_end,
            window_samples=self.config.window_samples,
            shift_samples=self.config.window_shift_samples,
        )
        max_frames = (
            (segment_length + self.config.receptive_field_samples + self.config.receptive_field_shift_samples - 1)
            // self.config.receptive_field_shift_samples
            + 8
        )
        speech_sum = np.zeros(max_frames, dtype=np.float64)
        overlap_sum = np.zeros(max_frames, dtype=np.float64)
        change_sum = np.zeros(max_frames, dtype=np.float64)
        total_weight = np.zeros(max_frames, dtype=np.float64)
        speaker_sum = np.zeros((max_frames, 3), dtype=np.float64)
        support = np.zeros(max_frames, dtype=np.int32)
        padded_support = np.zeros(max_frames, dtype=np.int32)
        argmax_trace: list[Any] = []
        permutations: list[tuple[int, int, int]] = []
        padded_windows = 0
        inference_wall = 0.0
        for window_start in starts:
            samples, valid_samples = _read_pcm16_window(
                accessor,
                window_start,
                segment_end,
                self.config.window_samples,
                np=np,
            )
            padded = valid_samples < self.config.window_samples
            padded_windows += int(padded)
            infer_started = time.perf_counter()
            _, speech, overlap, activity, argmax = self._infer(samples)
            inference_wall += time.perf_counter() - infer_started
            argmax_trace.append(argmax)
            frame_count = int(speech.shape[0])
            if self._window_weights is None or len(self._window_weights) != frame_count:
                self._window_weights = np.hamming(frame_count).astype(np.float64)
            local_starts = np.arange(frame_count, dtype=np.int64) * self.config.receptive_field_shift_samples
            valid_mask = local_starts < valid_samples
            local_indices = np.nonzero(valid_mask)[0]
            window_offset = window_start - segment_start
            global_start = (
                window_offset + self.config.receptive_field_shift_samples // 2
            ) // self.config.receptive_field_shift_samples
            global_indices = global_start + local_indices
            if global_indices.size == 0 or int(global_indices[-1]) >= max_frames:
                raise PyannoteSegmentationError("global frame accounting exceeded its bound")
            weights = self._window_weights[local_indices]
            existing = total_weight[global_indices] > 0.0
            permutation = (0, 1, 2)
            if bool(existing.any()):
                reference_activity = np.zeros((len(global_indices), 3), dtype=np.float64)
                reference_speech = np.zeros(len(global_indices), dtype=np.float64)
                reference_overlap = np.zeros(len(global_indices), dtype=np.float64)
                ref_weight = total_weight[global_indices]
                reference_activity[existing] = (
                    speaker_sum[global_indices[existing]] / ref_weight[existing, np.newaxis]
                )
                reference_speech[existing] = speech_sum[global_indices[existing]] / ref_weight[existing]
                reference_overlap[existing] = overlap_sum[global_indices[existing]] / ref_weight[existing]
                permutation = align_local_speakers(
                    activity[local_indices],
                    speech[local_indices],
                    overlap[local_indices],
                    reference_activity,
                    reference_speech,
                    reference_overlap,
                    weights * existing,
                    np=np,
                )
            permutations.append(permutation)
            aligned_activity = activity[:, permutation]
            change = conservative_within_window_change(
                aligned_activity,
                speech,
                overlap,
                context_frames=self.config.change_context_frames,
                min_clean_confidence=self.config.change_min_clean_confidence,
                min_purity=self.config.change_min_purity,
                np=np,
            )
            speech_sum[global_indices] += speech[local_indices] * weights
            overlap_sum[global_indices] += overlap[local_indices] * weights
            change_sum[global_indices] += change[local_indices] * weights
            speaker_sum[global_indices] += aligned_activity[local_indices] * weights[:, np.newaxis]
            total_weight[global_indices] += weights
            support[global_indices] += 1
            if padded:
                padded_support[global_indices] += 1
        frames: list[PyannoteFrameEvidence] = []
        valid_global = np.nonzero(total_weight > 0.0)[0]
        for global_index in valid_global.tolist():
            frame_start_sample = segment_start + global_index * self.config.receptive_field_shift_samples
            if frame_start_sample >= segment_end:
                continue
            frame_end_sample = min(
                segment_end,
                frame_start_sample + self.config.receptive_field_samples,
            )
            start_time = _sample_to_us(frame_start_sample)
            end_time = _sample_to_us(frame_end_sample)
            center_time = _sample_to_us((frame_start_sample + frame_end_sample) // 2)
            weight = total_weight[global_index]
            frames.append(
                PyannoteFrameEvidence(
                    frame_index=global_index,
                    start_us=start_time,
                    end_us=end_time,
                    center_us=center_time,
                    speech_probability=float(np.clip(speech_sum[global_index] / weight, 0.0, 1.0)),
                    overlap_probability=float(np.clip(overlap_sum[global_index] / weight, 0.0, 1.0)),
                    speaker_change_evidence=float(np.clip(change_sum[global_index] / weight, 0.0, 1.0)),
                    window_support=int(support[global_index]),
                    padded_window_support=int(padded_support[global_index]),
                )
            )
        events = diagnostic_change_events(
            frames,
            threshold=self.config.diagnostic_event_threshold,
            min_separation_us=self.config.diagnostic_event_min_separation_us,
        )
        permutation_payload = json.dumps(permutations, separators=(",", ":")).encode("ascii")
        trace = PyannoteParityTrace(
            window_argmax=tuple(int(value) for block in argmax_trace for value in block.tolist()),
            window_count=len(starts),
            frames_per_window=(len(argmax_trace[0]) if argmax_trace else 0),
            permutation_sha256=hashlib.sha256(permutation_payload).hexdigest(),
        )
        audio_digest = _sha256_path(Path(audio_path))
        elapsed = time.perf_counter() - started
        result = PyannoteSegmentationResult(
            schema="sddiar_pyannote_segmentation_evidence_v1",
            result_kind="DEVELOPMENT_UNCALIBRATED_SCD_OSD_EVIDENCE",
            source_audio_sha256_prefix=audio_digest[:12],
            segment_start_us=_sample_to_us(segment_start),
            segment_end_us=_sample_to_us(segment_end),
            model_sha256_prefix=self.model_sha256[:12],
            model_variant=self.model_variant,
            frames=tuple(frames),
            diagnostic_change_events=events,
            window_count=len(starts),
            padded_window_count=padded_windows,
            elapsed_wall_sec=round(elapsed, 6),
            model_inference_wall_sec=round(inference_wall, 6),
            peak_rss_mb=self._rss_mb(),
        )
        return result, trace


__all__ = [
    "EXPECTED_METADATA",
    "EXPECTED_LICENSE_URL",
    "POWERSET_MAPPING",
    "PyannoteSegmentationError",
    "PyannoteSegmentationHashError",
    "PyannoteSegmentationMetadataError",
    "PyannoteSegmentationConfig",
    "PyannoteFrameEvidence",
    "PyannoteChangeEvent",
    "PyannoteSegmentationResult",
    "PyannoteParityTrace",
    "bounded_window_starts",
    "align_local_speakers",
    "conservative_within_window_change",
    "diagnostic_change_events",
    "PyannoteSegmentationOnnxRuntime",
]
