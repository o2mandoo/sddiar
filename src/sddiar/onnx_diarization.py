"""Development-only high-level local ONNX diarization API.

This module is the package seam for the inspected local Silero + WeSpeaker
CPU path.  It deliberately does not discover, download, install, or fall back
to another model.  Callers must supply a local model file and its exact
SHA-256 (or a :class:`~sddiar.model_pack.VerifiedArtifact`).

The public result is intentionally redacted: it contains time spans and
aggregate metrics, but never the source path, transcript text, audio samples,
embedding vectors, or centroids.  This is a development integration path and
therefore always reports ``REVIEW_REQUIRED`` until a signed calibration
profile and the release runtime gates are supplied by a higher-level service.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .audio_gain import (
    DEFAULT_GLOBAL_GAIN_POLICY,
    GainScaledWavPcmAccessor,
    analyze_pcm16_global_gain,
    disabled_gain_metadata,
    scale_decoded_chunks,
)
from .contracts import DiarizationSpan, EmbeddingRegion, SpeechRegion as ContractSpeechRegion
from .diarization import (
    DiarizationConfig,
    build_tracklets,
    evaluate_hypotheses,
    finalize_sequence,
    refine_recent_states,
    select_anchor_evidence,
    speaker_states_from_decision,
)
from .errors import ContractValidationError, ModelHashMismatch, ModelNotFound, ModelRuntimeIncompatible
from .media import WavPcmAccessor, WavPcmDecoder
from .model_pack import VerifiedArtifact
from .offline import reject_url
from .ort_cpu import OrtCpuError, create_ort_session
from .quality import RuleBasedQualityGate
from .segmentation import RuleEvidenceSegmentation, SegmentationConfig, SegmentationEvidence
from .silero_runtime import SileroOnnxRuntime
from .silero_temporal import SileroTemporalPostprocessor
from .wespeaker_runtime import WeSpeakerCpuEmbeddingBackend


# ``resource`` is POSIX-only and is not present in a normal Windows Python
# installation.  Keep it optional so importing the package is portable; the
# RSS value is only diagnostic and must never be used as a runtime gate.
try:  # pragma: no cover - the fallback is exercised by simulated-Windows tests
    import resource as _resource
except ImportError:  # pragma: no cover - platform dependent
    _resource = None


_SHA256_LENGTH = 64
_SPEAKERS = ("SPEAKER_00", "SPEAKER_01")


@dataclass(frozen=True, slots=True)
class LocalOnnxDiarizationConfig:
    """Bounded settings for the development local ONNX path."""

    silero_threshold: float = 0.5
    silero_window_samples: int = 512
    silero_context_samples: int = 64
    decode_frames_per_chunk: int = 240_000
    vad_merge_gap_us: int = 200_000
    max_tracklet_us: int = 3_000_000
    h2_complexity_penalty: float = 0.0
    h2_min_cost_gain: float = 0.02
    assignment_distance_limit: float | None = None
    silero_temporal_postprocess: bool = False
    auto_gain_normalization: bool = False
    include_non_speech: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.silero_threshold < 1.0:
            raise ContractValidationError("silero_threshold must be in (0, 1)")
        if type(self.silero_window_samples) is not int or self.silero_window_samples <= 0:
            raise ContractValidationError("silero_window_samples must be positive")
        if type(self.silero_context_samples) is not int or self.silero_context_samples < 0:
            raise ContractValidationError("silero_context_samples must be non-negative")
        if type(self.decode_frames_per_chunk) is not int or self.decode_frames_per_chunk <= 0:
            raise ContractValidationError("decode_frames_per_chunk must be positive")
        if type(self.vad_merge_gap_us) is not int or self.vad_merge_gap_us < 0:
            raise ContractValidationError("vad_merge_gap_us must be non-negative")
        if type(self.max_tracklet_us) is not int or self.max_tracklet_us <= 0:
            raise ContractValidationError("max_tracklet_us must be positive")
        if self.h2_complexity_penalty < 0.0 or self.h2_min_cost_gain < 0.0:
            raise ContractValidationError("H2 penalties/gain must be non-negative")
        if self.assignment_distance_limit is not None and not 0.0 < self.assignment_distance_limit <= 1.0:
            raise ContractValidationError("assignment_distance_limit must be in (0, 1]")
        if type(self.silero_temporal_postprocess) is not bool:
            raise ContractValidationError("silero_temporal_postprocess must be boolean")
        if type(self.auto_gain_normalization) is not bool:
            raise ContractValidationError("auto_gain_normalization must be boolean")
        for name in ("silero_threshold", "h2_complexity_penalty", "h2_min_cost_gain"):
            if not math.isfinite(float(getattr(self, name))):
                raise ContractValidationError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class LocalOnnxDiarizationResult:
    """Redacted, JSON-safe output from :class:`LocalOnnxDiarizer`."""

    schema: str
    result_kind: str
    audio_sha256_prefix: str
    duration_us: int
    sample_rate_hz: int
    decision: str
    decision_reasons: tuple[str, ...]
    quality_status: str
    quality_reason_codes: tuple[str, ...]
    spans: tuple[DiarizationSpan, ...]
    metrics: Mapping[str, int | float | None]
    model_sha256_prefixes: Mapping[str, str]
    runtime_config: Mapping[str, Any]
    audio_gain_normalization: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a redacted mapping suitable for JSON serialization."""

        return {
            "schema": self.schema,
            "result_kind": self.result_kind,
            "audio_sha256_prefix": self.audio_sha256_prefix,
            "duration_us": self.duration_us,
            "sample_rate_hz": self.sample_rate_hz,
            "decision": self.decision,
            "decision_reasons": list(self.decision_reasons),
            "quality_status": self.quality_status,
            "quality_reason_codes": list(self.quality_reason_codes),
            "spans": [
                {
                    "span_id": span.span_id,
                    "start_us": span.start_us,
                    "end_us": span.end_us,
                    "speaker_id": span.speaker_id,
                    "attribution_status": span.attribution_status,
                    "evidence_ids": list(span.evidence_ids),
                    "reason_codes": list(span.reason_codes),
                }
                for span in self.spans
            ],
            "metrics": dict(self.metrics),
            "model_sha256_prefixes": dict(self.model_sha256_prefixes),
            "runtime_config": dict(self.runtime_config),
            "audio_gain_normalization": dict(self.audio_gain_normalization),
            "redaction": {
                "source_path": "omitted",
                "transcript": "omitted",
                "audio_samples": "omitted",
                "embedding_vectors": "omitted",
                "speaker_centroids": "omitted",
            },
        }

    def to_json(self) -> str:
        """Return deterministic JSON without paths or internal vectors."""

        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_artifact(spec: str | Path | VerifiedArtifact, expected_sha256: str | None, *, role: str, file_id: str) -> VerifiedArtifact:
    """Verify one local artifact, requiring an exact digest at this seam."""

    if isinstance(spec, VerifiedArtifact):
        artifact = spec
        path = Path(artifact.path)
        expected = artifact.sha256
        if expected_sha256 is not None and expected_sha256.lower() != expected.lower():
            raise ModelHashMismatch("explicit model hash disagrees with VerifiedArtifact")
    else:
        reject_url(spec)
        if expected_sha256 is None:
            raise ModelHashMismatch("an exact SHA-256 is required for every local model")
        expected = expected_sha256.lower()
        path = Path(spec)
        artifact = None
    if len(expected) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in expected):
        raise ModelHashMismatch("model SHA-256 must be exactly 64 lowercase hexadecimal characters")
    if not path.is_file() or path.is_symlink():
        raise ModelNotFound(f"local {role} model is missing or is a symlink")
    if path.suffix.lower() != ".onnx":
        raise ModelHashMismatch(f"local {role} model must use the .onnx extension")
    digest = _sha256(path)
    if digest != expected:
        raise ModelHashMismatch(f"local {role} model SHA-256 mismatch")
    accepted_roles = {
        "vad": {"vad", "voice_activity", "model"},
        "speaker_embedding": {"speaker_embedding", "embedding", "model"},
    }
    if artifact is not None and artifact.role not in accepted_roles[role]:
        raise ModelHashMismatch("VerifiedArtifact role does not match the requested model")
    return VerifiedArtifact(file_id, role, path.resolve(), digest, path.stat().st_size)


def _ort_session(path: Path, *, threads: int):
    """Open a cgroup-bounded, explicitly CPU-only ORT session."""

    if type(threads) is not int or threads <= 0:
        raise ContractValidationError("threads must be positive")
    try:
        return create_ort_session(path, threads=threads)
    except ValueError as exc:
        raise ContractValidationError(str(exc)) from exc
    except OrtCpuError as exc:
        raise ModelRuntimeIncompatible(f"cannot open local ONNX Runtime CPU session: {exc}") from exc


class LocalOnnxDiarizer:
    """Run local Silero VAD + WeSpeaker embeddings + deterministic diarization.

    ``silero_runtime`` and ``embedding_backend`` are injectable test seams. If
    omitted, they are constructed only from the hash-verified local model
    artifacts supplied to this constructor. No URL, download, package install,
    GPU provider, or implicit model fallback is permitted.
    """

    def __init__(
        self,
        silero_model: str | Path | VerifiedArtifact,
        wespeaker_model: str | Path | VerifiedArtifact,
        *,
        silero_sha256: str | None = None,
        wespeaker_sha256: str | None = None,
        config: LocalOnnxDiarizationConfig | None = None,
        threads: int = 1,
        silero_runtime: Any | None = None,
        embedding_backend: Any | None = None,
    ) -> None:
        self.config = config or LocalOnnxDiarizationConfig()
        if type(threads) is not int or threads <= 0:
            raise ContractValidationError("threads must be positive")
        self.threads = threads
        self.silero_artifact = _verified_artifact(silero_model, silero_sha256, role="vad", file_id="silero-vad")
        self.wespeaker_artifact = _verified_artifact(wespeaker_model, wespeaker_sha256, role="speaker_embedding", file_id="wespeaker-speaker-embedding")
        self._silero_runtime = silero_runtime or SileroOnnxRuntime(
            self.silero_artifact.path,
            session=_ort_session(self.silero_artifact.path, threads=threads),
            threshold=self.config.silero_threshold,
            window_samples=self.config.silero_window_samples,
            context_samples=self.config.silero_context_samples,
        )
        self._embedding_backend = embedding_backend
        self._owns_embedding_backend = embedding_backend is None
        if self._owns_embedding_backend:
            # Verify and open the embedding graph once per diarizer.  The
            # process-local lock below keeps this single ORT session and its
            # per-call audio binding isolated when callers share a diarizer.
            self._embedding_backend = WeSpeakerCpuEmbeddingBackend(
                self.wespeaker_artifact,
                session=_ort_session(self.wespeaker_artifact.path, threads=self.threads),
                max_batch_regions=1,
            )
        self._process_lock = threading.Lock()

    @staticmethod
    def _duration_us(frame_count: int, sample_rate_hz: int) -> int:
        return (frame_count * 1_000_000 + sample_rate_hz // 2) // sample_rate_hz

    @staticmethod
    def _rss_mb() -> float | None:
        """Return best-effort process RSS, or ``None`` when unavailable.

        POSIX ``resource`` reports peak RSS (bytes on macOS and KiB on Linux),
        whereas the Windows API reports current working-set bytes.  Both are
        useful diagnostics but are not interchangeable measurements, so the
        platform-specific source is kept explicit.  ``psutil`` is an optional
        last resort for other Python platforms; it is intentionally not a
        package dependency.  Missing RSS is represented as ``None`` rather
        than fabricated as zero.
        """

        if _resource is not None:
            try:
                raw = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
                # macOS reports bytes; Linux and other POSIX systems report KiB.
                value = raw if sys.platform == "darwin" else raw * 1024
                if value >= 0:
                    return round(value / (1024 * 1024), 2)
            except (AttributeError, OSError, TypeError, ValueError):
                pass

        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                class _ProcessMemoryCounters(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = _ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                process = ctypes.windll.kernel32.GetCurrentProcess()
                get_info = ctypes.windll.psapi.GetProcessMemoryInfo
                if get_info(process, ctypes.byref(counters), counters.cb):
                    return round(counters.WorkingSetSize / (1024 * 1024), 2)
            except (AttributeError, ImportError, OSError, TypeError, ValueError):
                pass

        try:
            import psutil  # optional compatibility fallback

            value = int(psutil.Process().memory_info().rss)
            if value >= 0:
                return round(value / (1024 * 1024), 2)
        except (ImportError, OSError, AttributeError, TypeError, ValueError):
            pass
        return None

    def _chunks(self, audio_path: Path):
        decoder = WavPcmDecoder()
        # ONNX inference already has NumPy available in the normal runtime, but
        # keep the stdlib decoder as a real compatibility path for constrained
        # installs and injected test runtimes.
        fast = getattr(decoder, "iter_decode_chunks_numpy", None)
        try:
            importlib.import_module("numpy")
        except ImportError:
            fast = None
        if callable(fast):
            return fast(audio_path, frames_per_chunk=self.config.decode_frames_per_chunk)
        return decoder.iter_decode_chunks(audio_path, frames_per_chunk=self.config.decode_frames_per_chunk)

    def process(self, audio_path: str | Path) -> LocalOnnxDiarizationResult:
        """Serialize one local audio job through this diarizer instance."""

        with self._process_lock:
            return self._process_unlocked(audio_path)

    def _process_unlocked(self, audio_path: str | Path) -> LocalOnnxDiarizationResult:
        """Process one local PCM16 mono 16 kHz WAV and return redacted output."""

        reject_url(audio_path)
        path = Path(audio_path)
        if not path.is_file() or path.is_symlink():
            raise ModelNotFound("local audio file is missing or is a symlink")
        accessor = WavPcmAccessor(path)
        layout = accessor.layout
        if layout.sample_rate_hz != 16_000 or layout.channel_count != 1 or layout.sample_width_bytes != 2:
            raise ContractValidationError("local ONNX API requires PCM16 mono 16 kHz WAV")
        started = time.perf_counter()
        audio_digest = _sha256(path)

        if self.config.auto_gain_normalization:
            gain_profile = analyze_pcm16_global_gain(path, source_sha256=audio_digest)
            gain_metadata = gain_profile.to_dict()
            inference_gain = gain_profile.applied_gain
            inference_accessor: WavPcmAccessor | GainScaledWavPcmAccessor = GainScaledWavPcmAccessor(
                accessor, inference_gain
            )
            vad_chunks = scale_decoded_chunks(self._chunks(path), inference_gain)
        else:
            gain_metadata = disabled_gain_metadata()
            inference_accessor = accessor
            vad_chunks = self._chunks(path)

        vad_frames = self._silero_runtime.infer_chunk_stream(vad_chunks)
        temporal_vad = None
        if self.config.silero_temporal_postprocess:
            temporal_vad = SileroTemporalPostprocessor().process(vad_frames)
            speech_regions = tuple(
                ContractSpeechRegion(
                    f"speech-temporal-{index}",
                    "audio",
                    region.start_us,
                    region.end_us,
                    reason_codes=("SILERO_TEMPORAL_PADDED",),
                )
                for index, region in enumerate(temporal_vad.regions)
            )
            segmentation = SegmentationEvidence(speech_regions, (), (), ())
        else:
            segmentation = RuleEvidenceSegmentation(
                SegmentationConfig(vad_merge_gap_us=self.config.vad_merge_gap_us)
            ).build(view_id="audio", vad_frames=vad_frames)
        duration_us = self._duration_us(layout.frame_count, layout.sample_rate_hz)
        diar_kwargs: dict[str, Any] = dict(
            max_tracklet_us=self.config.max_tracklet_us,
            lambda_k2=self.config.h2_complexity_penalty,
            h2_min_cost_gain=self.config.h2_min_cost_gain,
            include_non_speech=self.config.include_non_speech,
        )
        limit = self.config.assignment_distance_limit
        if limit is not None:
            diar_kwargs.update(
                anchor_stable_distance_ceiling=limit,
                anchor_absolute_distance_max=limit,
                support_stable_distance_ceiling=limit,
                support_absolute_distance_max=limit,
                micro_stable_distance_ceiling=min(limit, 0.35),
                micro_absolute_distance_max=min(limit, 0.35),
                unknown_cost=limit + 0.05,
            )
        diar_config = DiarizationConfig(**diar_kwargs)
        built = build_tracklets(segmentation.speech_regions, cfg=diar_config, audio_id=audio_digest[:16])
        regions = tuple(
            EmbeddingRegion(
                f"embedding-{index:06d}",
                tracklet.tracklet_id,
                tracklet.start_us,
                tracklet.end_us,
                tracklet.clean_speech_us,
                min(1.0, tracklet.clean_speech_us / max(1, tracklet.end_us - tracklet.start_us)),
            )
            for index, tracklet in enumerate(built.tracklets)
        )
        backend = self._embedding_backend
        if backend is None:  # pragma: no cover - invariant established in __init__
            raise ModelRuntimeIncompatible("embedding backend is not initialized")
        if regions and self._owns_embedding_backend:
            fast_reader = getattr(inference_accessor, "read_mono_samples_numpy", None)
            try:
                importlib.import_module("numpy")
            except ImportError:
                fast_reader = None
            if not callable(fast_reader):
                fast_reader = inference_accessor.read_mono_samples
            # Keep the accessor bound to this job only.  The backend session
            # is persistent, but no audio provider is retained on the backend.
            audio_provider = lambda region: fast_reader(region.start_us, region.end_us)
            # Strict snip-edges FBank rejects sub-frame MICRO regions.  The
            # backend converts that rejection to an invalid embedding, so
            # finalization keeps it UNKNOWN instead of inventing audio.
            embeddings = tuple(backend.embed(regions, audio_provider=audio_provider))
        elif regions:
            # Preserve narrow injected test seams (including fakes that accept
            # only the historical regions argument) and their result identity.
            embeddings = tuple(backend.embed(regions))
        else:
            embeddings = ()
        anchors, support, deferred = select_anchor_evidence(built.tracklets, embeddings, diar_config)
        hypothesis_evaluation = evaluate_hypotheses(anchors, diar_config)
        decision = hypothesis_evaluation.decision
        h2 = hypothesis_evaluation.h2_diagnostics
        states = speaker_states_from_decision(decision, anchors)
        states = refine_recent_states(built.tracklets, embeddings, states, decision, diar_config)
        spans = finalize_sequence(
            built.tracklets,
            built.protected_overlap_spans,
            states,
            decision,
            duration_us,
            diar_config,
            embeddings,
        )

        speech_duration_us = sum(region.end_us - region.start_us for region in segmentation.speech_regions)
        durations = {
            label: sum(span.end_us - span.start_us for span in spans if span.speaker_id == label)
            for label in (*_SPEAKERS, "UNKNOWN", "OVERLAP")
        }
        elapsed = time.perf_counter() - started
        unknown_and_assigned = durations["UNKNOWN"] + sum(durations[label] for label in _SPEAKERS)
        quality = RuleBasedQualityGate().evaluate(
            {
                "metrics": {
                    "unknown_ratio": durations["UNKNOWN"] / max(1, unknown_and_assigned),
                    "overlap_ratio": durations["OVERLAP"] / max(1, duration_us),
                },
                "speaker_count_status": "CONFIDENT_2" if decision.state == "H2_CONFIRMED" else "UNCERTAIN_1_OR_2",
                "hypothesis_uncertain": decision.state == "UNCERTAIN_1_OR_2",
                "review_reasons": ("DEV_ONLY_NO_SIGNED_CALIBRATION",),
            },
            None,
        )
        metrics: dict[str, int | float | None] = {
            "duration_us": duration_us,
            "speech_duration_us": speech_duration_us,
            "speech_ratio": speech_duration_us / max(1, duration_us),
            "tracklet_count": len(built.tracklets),
            "anchor_count": len(anchors),
            "support_count": len(support),
            "deferred_count": len(deferred),
            "valid_embedding_count": sum(int(item.is_valid) for item in embeddings),
            "span_count": len(spans),
            "assigned_duration_us": sum(durations[label] for label in _SPEAKERS),
            "unknown_duration_us": durations["UNKNOWN"],
            "overlap_duration_us": durations["OVERLAP"],
            "unknown_ratio": durations["UNKNOWN"] / max(1, unknown_and_assigned),
            "overlap_ratio": durations["OVERLAP"] / max(1, duration_us),
            "elapsed_wall_sec": round(elapsed, 4),
            "rtf": round(elapsed / max(1e-9, duration_us / 1_000_000), 6),
            "peak_rss_mb": self._rss_mb(),
            "h2_is_valid": int(h2.is_valid),
        }
        if temporal_vad is not None:
            metrics["temporal_vad_core_duration_us"] = temporal_vad.core_duration_us
            metrics["temporal_vad_halo_duration_us"] = temporal_vad.halo_duration_us
        return LocalOnnxDiarizationResult(
            schema="sddiar.local_onnx_result_v1",
            result_kind="DEVELOPMENT_LOCAL_ONNX_CPU_DIARIZATION",
            audio_sha256_prefix=audio_digest[:12],
            duration_us=duration_us,
            sample_rate_hz=layout.sample_rate_hz,
            decision=decision.state,
            decision_reasons=decision.reason_codes,
            quality_status=quality.status,
            quality_reason_codes=quality.reason_codes,
            spans=spans,
            metrics=metrics,
            model_sha256_prefixes={
                "silero_vad": self.silero_artifact.sha256[:12],
                "wespeaker_embedding": self.wespeaker_artifact.sha256[:12],
            },
            runtime_config={
                "threads": self.threads,
                "max_tracklet_us": self.config.max_tracklet_us,
                "h2_complexity_penalty": self.config.h2_complexity_penalty,
                "h2_min_cost_gain": self.config.h2_min_cost_gain,
                "assignment_distance_limit": self.config.assignment_distance_limit,
                "silero_temporal_postprocess": self.config.silero_temporal_postprocess,
                "auto_gain_normalization": self.config.auto_gain_normalization,
                "audio_gain_policy": DEFAULT_GLOBAL_GAIN_POLICY.to_dict(),
                "audio_gain_policy_sha256": DEFAULT_GLOBAL_GAIN_POLICY.sha256,
                "feature_mode": getattr(backend, "feature_mode", "injected"),
                "embedding_batch_regions": 1,
            },
            audio_gain_normalization=gain_metadata,
        )

    diarize = process


__all__ = [
    "LocalOnnxDiarizationConfig",
    "LocalOnnxDiarizationResult",
    "LocalOnnxDiarizer",
]
