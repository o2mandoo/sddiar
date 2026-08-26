"""Conservative offline production orchestration.

This module closes the narrow seam between a release-verified model pack, the
local diarizer, caller-supplied source-time words, the calibrated quality gate,
and canonical public-result serialization.  It deliberately does not discover
models, contact a transcript service, infer participant roles, or enable an
experimental enhancement by default.

Raw transcript text is accepted only as explicit result payload.  It is never
copied into quality diagnostics, run metadata, extensions, logs, or exception
messages.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import stat
import struct
import threading
import wave
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from .attribution import WordMappingConfig, WordSpeakerMapper
from .calibration import VerifiedCalibrationBinding
from .contracts import (
    AttributedWord,
    AudioRequest,
    AudioSourceMetadata,
    DiarizationSpan,
    FileQualityReport,
    ParticipantBinding,
    PipelineResult,
    PipelineRunMetadata,
    SpeakerTurn,
    TimeWarpSegment,
    Timebase,
    Word,
    WordProvenance,
    WordTimeline,
    deterministic_id,
)
from .errors import (
    ContractValidationError,
    ManifestSignatureInvalid,
    ModelHashMismatch,
    OfflinePolicyViolation,
)
from .media import WavPcmAccessor
from .model_pack import VerifiedArtifact, VerifiedModelPack
from .offline import reject_url
from .onnx_diarization import LocalOnnxDiarizationConfig, LocalOnnxDiarizer
from .quality import QualityConfig, RuleBasedQualityGate
from .serialization import ResultSerializer


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_REASON_RE = re.compile(r"^[A-Z][A-Z0-9_:-]{0,127}$")
_METRIC_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_SPEAKER_IDS = frozenset({"SPEAKER_00", "SPEAKER_01"})
_PUBLIC_LABELS = _SPEAKER_IDS | {"UNKNOWN", "OVERLAP", "OTHER", "NON_SPEECH"}
_SPAN_STATUSES = {
    "SPEAKER_00": frozenset({"ASSIGNED", "HUMAN_CONFIRMED"}),
    "SPEAKER_01": frozenset({"ASSIGNED", "HUMAN_CONFIRMED"}),
    "UNKNOWN": frozenset({
        "UNKNOWN_INSUFFICIENT_EVIDENCE", "UNKNOWN_SHORT",
        "UNKNOWN_BOUNDARY", "UNKNOWN_TIMEWARP_BOUNDARY",
    }),
    "OVERLAP": frozenset({"OVERLAP"}),
    "OTHER": frozenset({"OTHER", "UNKNOWN_INSUFFICIENT_EVIDENCE"}),
    "NON_SPEECH": frozenset({"NON_SPEECH"}),
}
_BINDING_METHODS = frozenset({
    "TRUSTED_CHANNEL_METADATA",
    "PREREGISTERED_VOICE",
    "HUMAN_CONFIRMED_SEGMENT",
    "EXTERNAL_AUTHORITATIVE_METADATA",
})


class ProductionOrchestrationError(ContractValidationError):
    """A production seam cannot preserve its safety contract."""


class TranscriptBackend(Protocol):
    """Transcript acquisition boundary.

    This release implements only :class:`SuppliedWordsBackend`.  The protocol is
    explicit so a future internal adapter cannot silently bypass source-time and
    offline review.
    """

    def transcribe(
        self,
        request: AudioRequest,
        source: AudioSourceMetadata,
        canonical_audio: "CanonicalAudio",
    ) -> "SuppliedTranscriptPayload | LocalSttTranscriptPayload": ...


class WordTimelineProvider(Protocol):
    """Normalize a transcript payload into original-source microseconds."""

    def words(
        self,
        transcript: "SuppliedTranscriptPayload | LocalSttTranscriptPayload",
        source: AudioSourceMetadata,
        time_warp: Sequence[TimeWarpSegment],
    ) -> WordTimeline: ...


class DiarizerBackend(Protocol):
    def process(self, audio_path: str | os.PathLike[str]) -> Any: ...


class LocalSttEngine(Protocol):
    """Injected local engine contract.

    Implementations must use the supplied canonical local path directly. They
    must not access a network, invoke a shell, download a model, consult a user
    cache, or fall back to another engine/model. Returned words are already in
    original-source integer microseconds.
    """

    def transcribe(
        self,
        canonical_audio_path: Path,
        source: AudioSourceMetadata,
    ) -> WordTimeline: ...


class EnhancementHook(Protocol):
    """Optional post-diarization evidence hook; disabled by default."""

    release_authorized: bool

    def enhance(
        self, canonical_audio: "CanonicalAudio", result: "DiarizationEnvelope"
    ) -> "DiarizationEnvelope": ...


@dataclass(frozen=True, slots=True)
class SuppliedTranscriptPayload:
    """Request-local wrapper around an already source-timed word timeline."""

    timeline: WordTimeline


@dataclass(frozen=True, slots=True)
class LocalSttTranscriptPayload:
    """Hash-bound local STT payload with source-time words only."""

    timeline: WordTimeline
    backend_id: str
    backend_version: str
    engine_sha256: str
    model_sha256: str
    timebase: str = "SOURCE_MICROSECONDS"


_LOCAL_STT_IDENTITY_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedLocalSttIdentity:
    """Sealed identity for one local engine artifact and one local model."""

    backend_id: str
    backend_version: str
    engine_artifact: VerifiedArtifact
    model_artifact: VerifiedArtifact
    _token: object = field(repr=False, compare=False)

    def __init__(
        self,
        backend_id: str,
        backend_version: str,
        engine_artifact: VerifiedArtifact,
        model_artifact: VerifiedArtifact,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _LOCAL_STT_IDENTITY_TOKEN:
            raise TypeError("VerifiedLocalSttIdentity must be created by verify_local_stt_identity")
        object.__setattr__(self, "backend_id", backend_id)
        object.__setattr__(self, "backend_version", backend_version)
        object.__setattr__(self, "engine_artifact", engine_artifact)
        object.__setattr__(self, "model_artifact", model_artifact)
        object.__setattr__(self, "_token", _token)

    def assert_artifacts_unchanged(self) -> None:
        try:
            for artifact in (self.engine_artifact, self.model_artifact):
                path = artifact.path
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != artifact.bytes
                ):
                    raise ModelHashMismatch(
                        f"local STT {artifact.role} artifact changed"
                    )
                if _sha256_file(path) != artifact.sha256:
                    raise ModelHashMismatch(
                        f"local STT {artifact.role} artifact changed"
                    )
        except ModelHashMismatch:
            raise
        except Exception:
            raise ModelHashMismatch("local STT artifact integrity check failed") from None


def _verify_local_artifact(
    path_value: str | os.PathLike[str], expected_sha256: str, *, file_id: str, role: str
) -> VerifiedArtifact:
    try:
        reject_url(path_value)
        path = Path(path_value)
        expected = str(expected_sha256).lower()
        if not _SHA256_RE.fullmatch(expected):
            raise ModelHashMismatch(f"local STT {role} SHA-256 is invalid")
        if not path.is_file() or path.is_symlink():
            raise ModelHashMismatch(
                f"local STT {role} must be a regular non-symlink file"
            )
        digest = _sha256_file(path)
        if digest != expected:
            raise ModelHashMismatch(f"local STT {role} SHA-256 mismatch")
        return VerifiedArtifact(
            file_id, role, path.resolve(), digest, path.stat().st_size
        )
    except (OfflinePolicyViolation, ModelHashMismatch):
        raise
    except Exception:
        raise ModelHashMismatch(f"local STT {role} artifact unreadable") from None


def verify_local_stt_identity(
    *,
    backend_id: str,
    backend_version: str,
    engine_path: str | os.PathLike[str],
    engine_sha256: str,
    model_path: str | os.PathLike[str],
    model_sha256: str,
) -> VerifiedLocalSttIdentity:
    """Hash and seal local STT identity without loading or executing it."""

    if not _OPAQUE_TOKEN_RE.fullmatch(backend_id) or not _OPAQUE_TOKEN_RE.fullmatch(backend_version):
        raise ContractValidationError("local STT backend identity must be opaque")
    engine = _verify_local_artifact(
        engine_path, engine_sha256, file_id=f"stt-engine:{backend_id}", role="stt_engine"
    )
    model = _verify_local_artifact(
        model_path, model_sha256, file_id=f"stt-model:{backend_id}", role="stt_model"
    )
    return VerifiedLocalSttIdentity(
        backend_id,
        backend_version,
        engine,
        model,
        _token=_LOCAL_STT_IDENTITY_TOKEN,
    )


class HashVerifiedLocalTranscriptBackend:
    """Offline wrapper around an injected engine with immutable local identity."""

    __slots__ = ("_identity", "_implementation")

    def __init__(self, identity: VerifiedLocalSttIdentity, implementation: LocalSttEngine):
        if type(identity) is not VerifiedLocalSttIdentity:
            raise ContractValidationError("local STT identity must be verifier-created")
        if implementation is None or not callable(getattr(implementation, "transcribe", None)):
            raise ContractValidationError("local STT implementation is invalid")
        implementation_identity = getattr(implementation, "identity", None)
        if implementation_identity is not None and implementation_identity != identity:
            raise ContractValidationError("local STT implementation identity differs")
        identity.assert_artifacts_unchanged()
        self._identity = identity
        self._implementation = implementation

    @property
    def identity(self) -> VerifiedLocalSttIdentity:
        return self._identity

    @property
    def decoder_policy(self) -> Mapping[str, Any] | None:
        """Expose scalar decoder identity for a redacted production receipt."""
        config = getattr(self._implementation, "config", None)
        public_identity = getattr(config, "public_identity", None)
        if not callable(public_identity):
            return None
        value = public_identity()
        if not isinstance(value, Mapping):
            raise ContractValidationError("local STT decoder identity is invalid")
        return dict(value)

    def transcribe(
        self,
        request: AudioRequest,
        source: AudioSourceMetadata,
        canonical_audio: "CanonicalAudio",
    ) -> LocalSttTranscriptPayload:
        self._identity.assert_artifacts_unchanged()
        try:
            implementation_identity = getattr(self._implementation, "identity", None)
            if implementation_identity is not None and implementation_identity != self._identity:
                raise ProductionOrchestrationError("local STT implementation identity differs")
            timeline = self._implementation.transcribe(canonical_audio.canonical_path, source)
        except MemoryError:
            raise
        except Exception:
            raise ProductionOrchestrationError("local STT backend failed") from None
        self._identity.assert_artifacts_unchanged()
        if not isinstance(timeline, WordTimeline):
            raise ContractValidationError("local STT must return a source-time WordTimeline")
        return LocalSttTranscriptPayload(
            timeline,
            self._identity.backend_id,
            self._identity.backend_version,
            self._identity.engine_artifact.sha256,
            self._identity.model_artifact.sha256,
        )


class SuppliedWordsBackend:
    """Return caller-supplied words without alignment, network, or fallback."""

    def __init__(self, timeline: WordTimeline):
        if not isinstance(timeline, WordTimeline):
            raise ContractValidationError("supplied transcript must be a WordTimeline")
        self._timeline = timeline

    def transcribe(
        self,
        request: AudioRequest,
        source: AudioSourceMetadata,
        canonical_audio: "CanonicalAudio",
    ) -> SuppliedTranscriptPayload:
        if tuple(request.supplied_words) != tuple(self._timeline.words):
            raise ContractValidationError("request supplied words differ from the bound timeline")
        return SuppliedTranscriptPayload(self._timeline)


class SuppliedWordTimelineProvider:
    """Validate exact source-time words; never rescale or guess a boundary."""

    def words(
        self,
        transcript: SuppliedTranscriptPayload | LocalSttTranscriptPayload,
        source: AudioSourceMetadata,
        time_warp: Sequence[TimeWarpSegment],
    ) -> WordTimeline:
        if not isinstance(transcript, (SuppliedTranscriptPayload, LocalSttTranscriptPayload)):
            raise ContractValidationError("unsupported supplied transcript payload")
        if isinstance(transcript, LocalSttTranscriptPayload):
            if transcript.timebase != "SOURCE_MICROSECONDS":
                raise ContractValidationError("local STT words are not in source microseconds")
            if not all(_SHA256_RE.fullmatch(value) for value in (
                transcript.engine_sha256, transcript.model_sha256
            )):
                raise ContractValidationError("local STT payload identity is invalid")
        timeline = transcript.timeline
        if not isinstance(timeline.words, tuple) or not isinstance(
            timeline.provenance_by_word_id, Mapping
        ):
            raise ContractValidationError("word timeline collections have invalid types")
        ordered = tuple(sorted(
            timeline.words, key=lambda word: (word.start_us, word.end_us, word.word_id)
        ))
        if ordered != timeline.words:
            raise ContractValidationError("supplied words must be source-time ordered")
        if any(right.start_us < left.end_us for left, right in zip(ordered, ordered[1:])):
            raise ContractValidationError("supplied source-time words overlap")
        if any(word.end_us > source.duration_us for word in timeline.words):
            raise ContractValidationError("supplied word exceeds source duration")
        if any(
            isinstance(word.confidence, bool)
            or (
                word.confidence is not None
                and (
                    not isinstance(word.confidence, (int, float))
                    or not math.isfinite(float(word.confidence))
                )
            )
            for word in timeline.words
        ):
            raise ContractValidationError("word confidence is invalid")
        if any(
            not _OPAQUE_TOKEN_RE.fullmatch(word.word_id)
            or (
                word.source_chunk_id is not None
                and not _OPAQUE_TOKEN_RE.fullmatch(word.source_chunk_id)
            )
            for word in timeline.words
        ):
            raise ContractValidationError("word and source-chunk IDs must be opaque")
        if any(
            not isinstance(provenance, WordProvenance)
            for provenance in timeline.provenance_by_word_id.values()
        ):
            raise ContractValidationError("supplied word provenance is invalid")
        if any(
            not isinstance(chunk_id, str) or not _OPAQUE_TOKEN_RE.fullmatch(chunk_id)
            for provenance in timeline.provenance_by_word_id.values()
            if isinstance(provenance.source_chunk_ids, tuple)
            for chunk_id in provenance.source_chunk_ids
        ):
            raise ContractValidationError("word provenance chunk IDs must be opaque")
        if any(
            not isinstance(provenance.source_chunk_ids, tuple)
            for provenance in timeline.provenance_by_word_id.values()
        ):
            raise ContractValidationError("word provenance chunk IDs must be tuples")
        # Supplied words are already in source time.  ``time_warp`` is accepted
        # only to make that contract explicit; applying it again would be a bug.
        for segment in time_warp:
            if segment.source_start_us < 0 or segment.source_end_us > source.duration_us:
                raise ContractValidationError("canonical time warp exceeds source duration")
        return timeline


@dataclass(frozen=True, slots=True)
class ProductionQualityEvidence:
    """Content-free candidate evidence; not release authority by itself."""

    metrics: Mapping[str, float] = field(default_factory=dict)
    threshold_relations: Mapping[str, str] = field(default_factory=dict)
    all_required_metrics_evaluated: bool = False
    all_high_rules_pass: bool = False
    osd_coverage: str = "NOT_EVALUATED"
    review_reasons: tuple[str, ...] = ()
    unattributed_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, Mapping) or not isinstance(
            self.threshold_relations, Mapping
        ):
            raise ContractValidationError("quality metrics and relations must be mappings")
        metrics = dict(self.metrics)
        relations = dict(self.threshold_relations)
        if type(self.all_required_metrics_evaluated) is not bool or type(
            self.all_high_rules_pass
        ) is not bool:
            raise ContractValidationError("quality evidence flags must be boolean")
        if not isinstance(self.review_reasons, tuple) or not isinstance(
            self.unattributed_reasons, tuple
        ):
            raise ContractValidationError("quality reason collections must be tuples")
        if any(
            not isinstance(key, str)
            or not _METRIC_RE.fullmatch(key)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for key, value in metrics.items()
        ):
            raise ContractValidationError("quality metrics must be finite numeric aggregates")
        if any(
            not isinstance(key, str)
            or not _METRIC_RE.fullmatch(key)
            or value not in {"PASS", "WARN", "FAIL", "NOT_EVALUATED"}
            for key, value in relations.items()
        ):
            raise ContractValidationError("invalid quality threshold relation")
        if self.osd_coverage not in {"EVALUATED", "PARTIAL", "NOT_EVALUATED"}:
            raise ContractValidationError("invalid OSD coverage state")
        for reasons in (self.review_reasons, self.unattributed_reasons):
            if any(not isinstance(reason, str) or not _REASON_RE.fullmatch(reason) for reason in reasons):
                raise ContractValidationError("quality reason codes must be opaque codes")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "threshold_relations", MappingProxyType(relations))


@dataclass(frozen=True, slots=True)
class ProductionOrchestratorConfig:
    pipeline_version: str = "production-orchestrator-v1"
    config_hash: str | None = None
    threads: int = 1
    resample_chunk_frames: int = 65_536
    enable_enhancement: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_version, str) or not _OPAQUE_TOKEN_RE.fullmatch(
            self.pipeline_version
        ):
            raise ContractValidationError("pipeline_version must be a nonempty opaque token")
        if self.config_hash is not None and (
            not isinstance(self.config_hash, str)
            or not _SHA256_RE.fullmatch(self.config_hash)
        ):
            raise ContractValidationError("config_hash must be a lowercase SHA-256")
        if type(self.threads) is not int or self.threads != 1:
            raise ContractValidationError("production orchestrator requires exactly one thread")
        if type(self.resample_chunk_frames) is not int or not 1 <= self.resample_chunk_frames <= 1_048_576:
            raise ContractValidationError("resample_chunk_frames is outside the bounded range")
        if type(self.enable_enhancement) is not bool:
            raise ContractValidationError("enable_enhancement must be boolean")


@dataclass(frozen=True, slots=True)
class CanonicalAudio:
    source: AudioSourceMetadata
    canonical_path: Path
    canonical_sample_rate_hz: int
    canonical_frame_count: int
    canonical_sha256: str
    time_warp: tuple[TimeWarpSegment, ...]
    resampled: bool


@dataclass(frozen=True, slots=True)
class DiarizationEnvelope:
    spans: tuple[DiarizationSpan, ...]
    duration_us: int
    sample_rate_hz: int
    decision: str
    decision_reasons: tuple[str, ...]
    rtf: float | None = None
    peak_rss_mb: float | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("execution receipt is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _word_timeline_sha256(timeline: WordTimeline) -> str:
    return _json_sha256({
        "words": [
            {
                "word_id": word.word_id,
                "start_us": word.start_us,
                "end_us": word.end_us,
                "text": word.text,
                "confidence": word.confidence,
                "source_chunk_id": word.source_chunk_id,
            }
            for word in timeline.words
        ],
        "provenance": {
            word_id: {
                "crosses_timewarp_boundary": provenance.crosses_timewarp_boundary,
                "source_chunk_ids": list(provenance.source_chunk_ids),
                "duplicate_suspect": provenance.duplicate_suspect,
            }
            for word_id, provenance in sorted(timeline.provenance_by_word_id.items())
        },
    })


def canonical_production_config_hash(
    config: ProductionOrchestratorConfig | None = None,
    *,
    backend_kind: str = "LOCAL_ONNX_DEFAULT",
) -> str:
    """Hash every built-in algorithm/runtime setting used by this seam."""

    selected = config or ProductionOrchestratorConfig()
    if backend_kind not in {"LOCAL_ONNX_DEFAULT", "INJECTED_DIARIZER"}:
        raise ContractValidationError("unknown production diarizer backend kind")
    diarizer_config = (
        asdict(LocalOnnxDiarizationConfig())
        if backend_kind == "LOCAL_ONNX_DEFAULT"
        else {"authority": "INJECTED_UNVERIFIED"}
    )
    return _json_sha256({
        "schema": "sddiar.production_runtime_config_v1",
        "pipeline_version": selected.pipeline_version,
        "threads": selected.threads,
        "resample_chunk_frames": selected.resample_chunk_frames,
        "canonical_audio": {
            "sample_rate_hz": 16_000,
            "accepted_source_rates_hz": [8_000, 16_000],
            "resampler": "INTEGER_LINEAR_2X_V1",
        },
        "word_mapper": asdict(WordMappingConfig()),
        "quality_gate": asdict(QualityConfig()),
        "transcript_contract": "SOURCE_TIME_WORDS_V1",
        "participant_binding_contract": "CALLER_EXPLICIT_UNVERIFIED_V1",
        "enhancement_enabled": selected.enable_enhancement,
        "diarizer_backend_kind": backend_kind,
        "diarizer_config": diarizer_config,
    })


def _duration_us(frame_count: int, sample_rate_hz: int) -> int:
    return (frame_count * 1_000_000 + sample_rate_hz // 2) // sample_rate_hz


def _integer_average(left: int, right: int) -> int:
    total = left + right
    return (total + 1) // 2 if total >= 0 else (total - 1) // 2


class Pcm16CanonicalAdapter:
    """Bounded deterministic PCM16 mono 8/16 kHz to canonical 16 kHz adapter."""

    def __init__(self, *, chunk_frames: int = 65_536):
        if type(chunk_frames) is not int or not 1 <= chunk_frames <= 1_048_576:
            raise ContractValidationError("chunk_frames is outside the bounded range")
        self.chunk_frames = chunk_frames

    @contextmanager
    def prepare(self, source_path: str | os.PathLike[str]) -> Iterator[CanonicalAudio]:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="sddiar-canonical-") as directory:
            root = Path(directory)
            try:
                root.chmod(0o700)
            except OSError:
                pass
            source_snapshot = root / "source.wav"
            try:
                reject_url(source_path)
                digest = self._snapshot_source(Path(source_path), source_snapshot)
                source_snapshot.chmod(0o400)
                accessor = WavPcmAccessor(source_snapshot)
                layout = accessor.layout
                if layout.channel_count != 1 or layout.sample_width_bytes != 2:
                    raise ContractValidationError(
                        "production audio must be mono PCM16 WAV"
                    )
                if layout.sample_rate_hz not in {8_000, 16_000}:
                    raise ContractValidationError(
                        "production audio sample rate must be 8 or 16 kHz"
                    )
                if layout.frame_count <= 0:
                    raise ContractValidationError("production audio is empty")
                if _sha256_file(source_snapshot) != digest:
                    raise ContractValidationError("source snapshot integrity failed")
                duration = _duration_us(layout.frame_count, layout.sample_rate_hz)
                source = AudioSourceMetadata(
                    digest,
                    "wav",
                    "pcm_s16le",
                    layout.sample_rate_hz,
                    1,
                    duration,
                    Timebase(
                        f"source:{digest[:26]}",
                        source_sample_rate_hz=layout.sample_rate_hz,
                        duration_us=duration,
                    ),
                )
                canonical_frames = (
                    layout.frame_count
                    if layout.sample_rate_hz == 16_000
                    else layout.frame_count * 2
                )
                segment = TimeWarpSegment(
                    f"canonical:{digest[:26]}",
                    f"canonical16k:{digest[:26]}",
                    0,
                    canonical_frames,
                    0,
                    duration,
                )
            except (OfflinePolicyViolation, ContractValidationError):
                raise
            except Exception:
                raise ProductionOrchestrationError("local audio intake failed") from None

            canonical_path = root / "canonical.wav"
            try:
                if layout.sample_rate_hz == 16_000:
                    self._copy_16k(source_snapshot, accessor, canonical_path)
                else:
                    self._resample_8k(source_snapshot, accessor, canonical_path)
                if _sha256_file(source_snapshot) != digest:
                    raise ContractValidationError("source snapshot changed")
            except ContractValidationError:
                raise
            except Exception:
                raise ProductionOrchestrationError(
                    "local audio canonicalization failed"
                ) from None
            try:
                canonical_path.chmod(0o400)
            except OSError:
                pass
            try:
                canonical_layout = WavPcmAccessor(canonical_path).layout
                if (
                    canonical_layout.sample_rate_hz != 16_000
                    or canonical_layout.channel_count != 1
                    or canonical_layout.sample_width_bytes != 2
                    or canonical_layout.frame_count != canonical_frames
                    or _duration_us(canonical_layout.frame_count, 16_000) != duration
                ):
                    raise ContractValidationError(
                        "canonical audio conversion violated timebase"
                    )
                canonical_digest = _sha256_file(canonical_path)
            except ContractValidationError:
                raise
            except Exception:
                raise ProductionOrchestrationError(
                    "canonical audio validation failed"
                ) from None
            yield CanonicalAudio(
                source,
                canonical_path,
                16_000,
                canonical_frames,
                canonical_digest,
                (segment,),
                layout.sample_rate_hz == 8_000,
            )

    def _snapshot_source(self, source_path: Path, destination: Path) -> str:
        before = os.lstat(source_path)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        file_attributes = int(getattr(before, "st_file_attributes", 0))
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (reparse_flag and file_attributes & reparse_flag)
        ):
            raise ContractValidationError(
                "audio must be a local regular non-symlink file"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_BINARY", 0)
        )
        descriptor: int | None = None
        digest = hashlib.sha256()
        try:
            descriptor = os.open(os.fspath(source_path), flags)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or (before.st_dev, before.st_ino) != (metadata.st_dev, metadata.st_ino)
            ):
                raise ContractValidationError(
                    "audio must be a local regular non-symlink file"
                )
            with os.fdopen(descriptor, "rb", closefd=True) as source, destination.open(
                "xb"
            ) as snapshot:
                descriptor = None
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    snapshot.write(block)
                after = os.fstat(source.fileno())
                if (
                    (metadata.st_dev, metadata.st_ino, metadata.st_size)
                    != (after.st_dev, after.st_ino, after.st_size)
                    or getattr(metadata, "st_mtime_ns", None)
                    != getattr(after, "st_mtime_ns", None)
                ):
                    raise ContractValidationError("source audio changed during snapshot")
                snapshot.flush()
                os.fsync(snapshot.fileno())
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return digest.hexdigest()

    def _copy_16k(
        self, source_path: Path, accessor: WavPcmAccessor, destination: Path
    ) -> None:
        layout = accessor.layout
        copied = 0
        with source_path.open("rb") as source, wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.setcomptype("NONE", "not compressed")
            source.seek(layout.data_offset)
            remaining = layout.frame_count
            while remaining:
                count = min(remaining, self.chunk_frames)
                payload = source.read(count * 2)
                if len(payload) != count * 2:
                    raise ContractValidationError(
                        "16 kHz PCM ended before declared frame count"
                    )
                output.writeframesraw(payload)
                copied += count
                remaining -= count
        if copied != layout.frame_count:
            raise ContractValidationError("16 kHz canonical frame accounting failed")

    def _resample_8k(
        self, source_path: Path, accessor: WavPcmAccessor, destination: Path
    ) -> None:
        layout = accessor.layout
        previous: int | None = None
        source_frames = 0
        output_frames = 0
        with source_path.open("rb") as source, wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.setcomptype("NONE", "not compressed")
            source.seek(layout.data_offset)
            remaining = layout.frame_count
            while remaining:
                count = min(remaining, self.chunk_frames)
                payload = source.read(count * 2)
                if len(payload) != count * 2:
                    raise ContractValidationError("8 kHz PCM ended before declared frame count")
                values = struct.unpack(f"<{count}h", payload)
                expanded: list[int] = []
                for current in values:
                    source_frames += 1
                    if previous is not None:
                        expanded.extend((previous, _integer_average(previous, current)))
                    previous = current
                if expanded:
                    output.writeframesraw(struct.pack(f"<{len(expanded)}h", *expanded))
                    output_frames += len(expanded)
                remaining -= count
            if previous is not None:
                tail = struct.pack("<2h", previous, previous)
                output.writeframesraw(tail)
                output_frames += 2
        if source_frames != layout.frame_count or output_frames != layout.frame_count * 2:
            raise ContractValidationError("8 kHz conversion frame accounting failed")


def _safe_numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def _assert_canonical_unchanged(canonical: CanonicalAudio) -> None:
    try:
        path = canonical.canonical_path
        if path.is_symlink() or not path.is_file():
            raise ContractValidationError("canonical audio is missing or unsafe")
        if _sha256_file(path) != canonical.canonical_sha256:
            raise ContractValidationError("canonical audio changed during processing")
    except ContractValidationError:
        raise
    except Exception:
        raise ContractValidationError("canonical audio integrity check failed") from None


def _normalize_diarization_result(result: Any, source_duration_us: int) -> DiarizationEnvelope:
    raw_spans = getattr(result, "spans", ())
    if not isinstance(raw_spans, (tuple, list)):
        raise ContractValidationError("diarizer spans must be a sequence")
    spans = tuple(raw_spans)
    duration = getattr(result, "duration_us", None)
    sample_rate = getattr(result, "sample_rate_hz", None)
    decision = str(getattr(result, "decision", "UNCERTAIN_1_OR_2"))
    raw_decision_reasons = getattr(result, "decision_reasons", ()) or ()
    if not isinstance(raw_decision_reasons, (tuple, list)) or any(
        not isinstance(reason, str) for reason in raw_decision_reasons
    ):
        raise ContractValidationError("diarizer decision reasons must be a sequence of codes")
    decision_reasons = tuple(raw_decision_reasons)
    metrics = getattr(result, "metrics", {})
    if not isinstance(duration, int) or isinstance(duration, bool) or duration != source_duration_us:
        raise ContractValidationError("diarizer duration does not match original source timebase")
    if sample_rate != 16_000:
        raise ContractValidationError("diarizer must consume canonical 16 kHz audio")
    if decision not in {"H1_CONFIRMED", "H2_CONFIRMED", "UNCERTAIN_1_OR_2"}:
        raise ContractValidationError("invalid diarizer speaker-count decision")
    if any(not _REASON_RE.fullmatch(reason) for reason in decision_reasons):
        raise ContractValidationError("diarizer decision reasons must be opaque codes")
    previous_end = 0
    normalized: list[DiarizationSpan] = []
    for span in spans:
        if not isinstance(span, DiarizationSpan):
            raise ContractValidationError("diarizer spans must use the public span contract")
        if span.speaker_id not in _PUBLIC_LABELS:
            raise ContractValidationError("diarizer emitted an unsupported speaker label")
        if not _REASON_RE.fullmatch(span.attribution_status):
            raise ContractValidationError("span attribution status must be an opaque code")
        if span.attribution_status not in _SPAN_STATUSES[span.speaker_id]:
            raise ContractValidationError("speaker label and attribution status conflict")
        if span.start_us < previous_end or span.end_us > source_duration_us:
            raise ContractValidationError("diarizer spans overlap or exceed source time")
        if not _OPAQUE_TOKEN_RE.fullmatch(span.span_id):
            raise ContractValidationError("span IDs must be opaque tokens")
        if not isinstance(span.evidence_ids, tuple) or any(
            not isinstance(item, str) or not _OPAQUE_TOKEN_RE.fullmatch(item)
            for item in span.evidence_ids
        ):
            raise ContractValidationError("span evidence IDs must be opaque tokens")
        if not isinstance(span.reason_codes, tuple) or any(
            not isinstance(item, str) or not _REASON_RE.fullmatch(item)
            for item in span.reason_codes
        ):
            raise ContractValidationError("span reasons must be opaque codes")
        previous_end = span.end_us
        normalized.append(span)
    safe_metrics = metrics if isinstance(metrics, Mapping) else {}
    return DiarizationEnvelope(
        tuple(normalized),
        duration,
        sample_rate,
        decision,
        decision_reasons,
        _safe_numeric(getattr(result, "rtf", safe_metrics.get("rtf"))),
        _safe_numeric(getattr(
            result,
            "peak_rss_mb",
            safe_metrics.get("peak_process_tree_rss_mb", safe_metrics.get("peak_rss_mb")),
        )),
    )


def _validate_participant_bindings(
    bindings: Sequence[ParticipantBinding], spans: Sequence[DiarizationSpan]
) -> tuple[ParticipantBinding, ...]:
    present = {span.speaker_id for span in spans if span.speaker_id in _SPEAKER_IDS}
    speakers: set[str] = set()
    external_ids: set[str] = set()
    roles: set[str] = set()
    result: list[ParticipantBinding] = []
    for binding in bindings:
        if not isinstance(binding, ParticipantBinding):
            raise ContractValidationError("participant mappings must use ParticipantBinding")
        if binding.speaker_id not in _SPEAKER_IDS or binding.speaker_id not in present:
            raise ContractValidationError("participant mapping targets UNKNOWN or an absent speaker")
        if binding.speaker_id in speakers:
            raise ContractValidationError("participant mapping collides on speaker ID")
        if binding.method not in _BINDING_METHODS:
            raise ContractValidationError("participant mapping method is not authoritative")
        if (
            isinstance(binding.confidence, bool)
            or not isinstance(binding.confidence, (int, float))
            or not math.isfinite(float(binding.confidence))
            or not 0.0 <= float(binding.confidence) <= 1.0
        ):
            raise ContractValidationError("participant mapping confidence is invalid")
        if not isinstance(binding.evidence_ids, tuple) or any(
            not isinstance(item, str) or not _OPAQUE_TOKEN_RE.fullmatch(item)
            for item in binding.evidence_ids
        ):
            raise ContractValidationError("participant mapping evidence IDs must be opaque")
        for value, seen, label in (
            (binding.external_participant_id, external_ids, "external participant"),
            (binding.role_label, roles, "participant role"),
        ):
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip() or len(value) > 128 or any(
                ord(char) < 32 for char in value
            ):
                raise ContractValidationError(f"{label} is invalid")
            key = value.strip().casefold()
            if key in seen:
                raise ContractValidationError(f"{label} collision")
            seen.add(key)
        speakers.add(binding.speaker_id)
        result.append(binding)
    return tuple(sorted(result, key=lambda item: item.speaker_id))


def _build_turns(
    words: Sequence[AttributedWord], audio_sha256: str, id_namespace: str
) -> tuple[SpeakerTurn, ...]:
    if not words:
        return ()
    groups: list[list[AttributedWord]] = []
    for word in words:
        if (
            groups
            and groups[-1][-1].speaker_id == word.speaker_id
            and groups[-1][-1].attribution_status == word.attribution_status
            and word.start_us - groups[-1][-1].end_us <= 1_000_000
            and word.supporting_span_ids == groups[-1][-1].supporting_span_ids
        ):
            groups[-1].append(word)
        else:
            groups.append([word])
    turns = []
    for ordinal, group in enumerate(groups):
        start, end = group[0].start_us, max(word.end_us for word in group)
        turns.append(SpeakerTurn(
            deterministic_id(
                audio_sha256,
                "speaker_turn",
                start,
                end,
                ordinal,
                pipeline_version=id_namespace,
            ),
            start,
            end,
            group[0].speaker_id,
            tuple(word.word_id for word in group),
            " ".join(word.text for word in group),
            group[0].attribution_status,
            tuple(dict.fromkeys(
                evidence for word in group for evidence in word.supporting_span_ids
            )),
        ))
    return tuple(turns)


def _neutral_words(words: Sequence[AttributedWord]) -> tuple[AttributedWord, ...]:
    result = []
    for word in words:
        if word.speaker_id in {"UNKNOWN", "OVERLAP", "NON_SPEECH"}:
            result.append(word)
        else:
            result.append(replace(
                word,
                speaker_id="UNKNOWN",
                attribution_status="SPEAKER_NEUTRAL",
                supporting_span_ids=(),
                speaker_coverage_ratio=None,
                competing_speaker_coverage_ratio=None,
                reason_codes=tuple(dict.fromkeys((*word.reason_codes, "SPEAKER_NEUTRAL_PROJECTION"))),
            ))
    return tuple(result)


def _diarization_failed_words(timeline: WordTimeline) -> tuple[AttributedWord, ...]:
    """Preserve STT text while removing every speaker claim after failure."""

    return tuple(
        AttributedWord(
            word.word_id,
            word.start_us,
            word.end_us,
            word.text,
            word.confidence,
            word.source_chunk_id,
            "UNKNOWN",
            "UNKNOWN_INSUFFICIENT_EVIDENCE",
            (),
            None,
            None,
            ("DIARIZATION_FAILED",),
        )
        for word in timeline.words
    )


def _artifact_for_role(pack: VerifiedModelPack, roles: set[str], label: str) -> VerifiedArtifact:
    matches = [artifact for artifact in pack.artifacts if artifact.role.lower() in roles]
    if len(matches) != 1:
        raise ModelHashMismatch(f"release pack requires exactly one {label} artifact")
    return matches[0]


def _assert_model_pack_unchanged(pack: VerifiedModelPack) -> None:
    try:
        pack.assert_artifacts_unchanged()
    except ModelHashMismatch:
        raise
    except Exception:
        raise ModelHashMismatch("release model artifact integrity check failed") from None


def _execution_receipt_sha256(
    *,
    source: AudioSourceMetadata,
    config: ProductionOrchestratorConfig,
    runtime_config_hash: str,
    model_pack: VerifiedModelPack | None,
    model_hashes: Mapping[str, str],
    calibration: VerifiedCalibrationBinding | None,
    transcript_receipt: Mapping[str, Any],
    source_timeline: WordTimeline,
    spans: Sequence[DiarizationSpan],
    attributed: Sequence[AttributedWord],
    participant_bindings: Sequence[ParticipantBinding],
    quality: FileQualityReport,
) -> str:
    return _json_sha256({
        "source_audio_sha256": source.audio_sha256,
        "source_sample_rate_hz": source.native_sample_rate_hz,
        "source_duration_us": source.duration_us,
        "pipeline_version": config.pipeline_version,
        "config_hash": runtime_config_hash,
        "threads": config.threads,
        "enhancement_enabled": config.enable_enhancement,
        "model_pack": {
            "pack_id": model_pack.pack_id if model_pack is not None else None,
            "pack_version": model_pack.pack_version if model_pack is not None else None,
            "manifest_sha256": model_pack.manifest_sha256 if model_pack is not None else None,
            "model_hashes": dict(model_hashes),
        },
        "calibration": {
            "profile_id": calibration.profile_id if calibration is not None else None,
            "profile_payload_sha256": (
                calibration.profile_payload_sha256 if calibration is not None else None
            ),
            "config_hash": calibration.config_hash if calibration is not None else None,
        },
        "stt_backend": dict(transcript_receipt),
        "word_timeline_sha256": _word_timeline_sha256(source_timeline),
        "spans": [
            {
                "start_us": span.start_us,
                "end_us": span.end_us,
                "speaker_id": span.speaker_id,
                "attribution_status": span.attribution_status,
                "evidence_ids": list(span.evidence_ids),
                "reason_codes": list(span.reason_codes),
            }
            for span in spans
        ],
        "attributed_words": [
            {
                "word_id": word.word_id,
                "speaker_id": word.speaker_id,
                "attribution_status": word.attribution_status,
                "supporting_span_ids": list(word.supporting_span_ids),
            }
            for word in attributed
        ],
        "participant_bindings": [
            {
                "speaker_id": binding.speaker_id,
                "external_participant_id": binding.external_participant_id,
                "role_label": binding.role_label,
                "method": binding.method,
                "confidence": binding.confidence,
                "evidence_ids": list(binding.evidence_ids),
            }
            for binding in participant_bindings
        ],
        "quality": {
            "status": quality.status,
            "summary_mode": quality.summary_mode,
            "reason_codes": list(quality.reason_codes),
            "metrics": dict(quality.metrics),
            "threshold_relations": dict(quality.threshold_relations),
            "calibration_profile_id": quality.calibration_profile_id,
        },
    })


class ProductionOrchestrator:
    """One safe offline path from canonical audio to validated public JSON."""

    __slots__ = (
        "_model_pack", "_calibration", "_config", "_enhancement_hook",
        "_transcript_backend", "_word_timeline_provider", "_audio_adapter",
        "_pack_bound_backend", "_pack_bound_instance", "_model_hashes",
        "_diarizer", "_diarizer_lock", "_diarizer_builder", "_process_lock",
        "_runtime_config_hash",
    )

    def __init__(
        self,
        *,
        model_pack: VerifiedModelPack | None = None,
        diarizer: DiarizerBackend | None = None,
        diarizer_factory: Callable[[VerifiedArtifact, VerifiedArtifact], DiarizerBackend] | None = None,
        calibration: VerifiedCalibrationBinding | None = None,
        config: ProductionOrchestratorConfig | None = None,
        enhancement_hook: EnhancementHook | None = None,
        transcript_backend: HashVerifiedLocalTranscriptBackend | None = None,
    ) -> None:
        selected_config = config or ProductionOrchestratorConfig()
        if model_pack is not None and type(model_pack) is not VerifiedModelPack:
            raise ManifestSignatureInvalid("model_pack must be verifier-created")
        if calibration is not None and type(calibration) is not VerifiedCalibrationBinding:
            raise ContractValidationError("calibration must be a VerifiedCalibrationBinding")
        if diarizer is not None and diarizer_factory is not None:
            raise ContractValidationError("provide either diarizer or diarizer_factory, not both")
        if transcript_backend is not None and type(transcript_backend) is not HashVerifiedLocalTranscriptBackend:
            raise ContractValidationError(
                "local STT must use HashVerifiedLocalTranscriptBackend"
            )
        backend_kind = (
            "LOCAL_ONNX_DEFAULT"
            if diarizer is None and diarizer_factory is None
            else "INJECTED_DIARIZER"
        )
        runtime_config_hash = canonical_production_config_hash(
            selected_config, backend_kind=backend_kind
        )
        if (
            selected_config.config_hash is not None
            and selected_config.config_hash != runtime_config_hash
        ):
            raise ContractValidationError(
                "declared config_hash does not match canonical runtime config"
            )
        self._model_pack = model_pack
        self._calibration = calibration
        self._config = selected_config
        self._enhancement_hook = enhancement_hook
        self._transcript_backend = transcript_backend
        self._word_timeline_provider = SuppliedWordTimelineProvider()
        self._runtime_config_hash = runtime_config_hash
        self._audio_adapter = Pcm16CanonicalAdapter(
            chunk_frames=selected_config.resample_chunk_frames
        )
        self._pack_bound_backend = False
        self._pack_bound_instance: DiarizerBackend | None = None
        self._model_hashes: dict[str, str] = {}
        self._diarizer_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._diarizer_builder: Callable[[], DiarizerBackend] | None = None

        if diarizer is not None:
            self._diarizer = diarizer
        else:
            if model_pack is None or not model_pack.release_trusted:
                raise ManifestSignatureInvalid(
                    "constructing a local production diarizer requires a release-trusted model pack"
                )
            _assert_model_pack_unchanged(model_pack)
            vad = _artifact_for_role(model_pack, {"vad", "voice_activity"}, "VAD")
            embedding = _artifact_for_role(
                model_pack, {"speaker_embedding", "embedding"}, "speaker embedding"
            )
            factory = diarizer_factory or LocalOnnxDiarizer
            self._diarizer = None
            if diarizer_factory is None:
                self._diarizer_builder = lambda: factory(
                    vad, embedding, threads=selected_config.threads
                )
            else:
                self._diarizer_builder = lambda: factory(vad, embedding)
            # Only the built-in constructor binds the runtime to the verified
            # artifacts.  An injected factory remains a useful test/adapter seam
            # but cannot manufacture release authority.
            self._pack_bound_backend = diarizer_factory is None
            if self._pack_bound_backend:
                self._model_hashes = {
                    vad.file_id: vad.sha256,
                    embedding.file_id: embedding.sha256,
                }

    @property
    def model_pack(self) -> VerifiedModelPack | None:
        return self._model_pack

    @property
    def calibration(self) -> VerifiedCalibrationBinding | None:
        return self._calibration

    @property
    def config(self) -> ProductionOrchestratorConfig:
        return self._config

    @property
    def serializer(self) -> ResultSerializer:
        return ResultSerializer()

    @property
    def runtime_config_hash(self) -> str:
        return self._runtime_config_hash

    @property
    def diarizer(self) -> DiarizerBackend | None:
        return self._diarizer

    def _has_release_model_authority(self) -> bool:
        return bool(
            self._model_pack is not None
            and self._model_pack.release_trusted
            and self._pack_bound_backend
            and self._pack_bound_instance is not None
            and self._diarizer is self._pack_bound_instance
        )

    def process(
        self,
        audio_path: str | os.PathLike[str],
        *,
        supplied_word_timeline: WordTimeline | None = None,
        participant_bindings: Sequence[ParticipantBinding] = (),
        quality_evidence: ProductionQualityEvidence | None = None,
    ) -> PipelineResult:
        """Serialize one STT+diarization job through this orchestrator."""

        with self._process_lock:
            return self._process_unlocked(
                audio_path,
                supplied_word_timeline=supplied_word_timeline,
                participant_bindings=participant_bindings,
                quality_evidence=quality_evidence,
            )

    def _process_unlocked(
        self,
        audio_path: str | os.PathLike[str],
        *,
        supplied_word_timeline: WordTimeline | None = None,
        participant_bindings: Sequence[ParticipantBinding] = (),
        quality_evidence: ProductionQualityEvidence | None = None,
    ) -> PipelineResult:
        if supplied_word_timeline is not None and not isinstance(
            supplied_word_timeline, WordTimeline
        ):
            raise ContractValidationError("supplied_word_timeline must be a WordTimeline")
        if quality_evidence is not None and type(
            quality_evidence
        ) is not ProductionQualityEvidence:
            raise ContractValidationError(
                "quality_evidence must use ProductionQualityEvidence"
            )
        if not isinstance(participant_bindings, (tuple, list)):
            raise ContractValidationError("participant_bindings must be a sequence")
        if self.model_pack is not None:
            _assert_model_pack_unchanged(self.model_pack)
        with self._audio_adapter.prepare(audio_path) as canonical:
            source_timeline, transcript_receipt = self._transcript(
                canonical, supplied_word_timeline
            )
            _assert_canonical_unchanged(canonical)
            diarization_failed = False
            try:
                raw_diarization = self._get_diarizer().process(
                    canonical.canonical_path
                )
                _assert_canonical_unchanged(canonical)
                if self.model_pack is not None:
                    _assert_model_pack_unchanged(self.model_pack)
                base = _normalize_diarization_result(
                    raw_diarization, canonical.source.duration_us
                )
            except (
                ModelHashMismatch,
                ManifestSignatureInvalid,
                OfflinePolicyViolation,
                ContractValidationError,
                MemoryError,
            ):
                raise
            except Exception:
                _assert_canonical_unchanged(canonical)
                if self.model_pack is not None:
                    _assert_model_pack_unchanged(self.model_pack)
                # Preserve a successful local transcript without exposing the
                # exception text (which may contain a path or input content).
                diarization_failed = True
                base = DiarizationEnvelope(
                    (),
                    canonical.source.duration_us,
                    16_000,
                    "UNCERTAIN_1_OR_2",
                    ("DIARIZATION_FAILED",),
                )
            envelope = base
            enhancement_review = False
            if self._config.enable_enhancement and not diarization_failed:
                if self._enhancement_hook is None:
                    raise ProductionOrchestrationError(
                        "enhancement is enabled but no local hook was supplied"
                    )
                try:
                    enhanced = self._enhancement_hook.enhance(canonical, base)
                except MemoryError:
                    raise
                except Exception:
                    raise ProductionOrchestrationError("enhancement hook failed") from None
                if not isinstance(enhanced, DiarizationEnvelope):
                    raise ProductionOrchestrationError(
                        "enhancement must return a DiarizationEnvelope"
                    )
                envelope = _normalize_diarization_result(
                    enhanced, canonical.source.duration_us
                )
                _assert_canonical_unchanged(canonical)
                # No sealed enhancement authority exists in V1.  Even a hook
                # that self-describes as approved cannot grant a release PASS.
                enhancement_review = True

            attributed = (
                _diarization_failed_words(source_timeline)
                if diarization_failed
                else tuple(WordSpeakerMapper().map_words(source_timeline, envelope.spans))
            )
            explicit_bindings = (
                () if diarization_failed
                else _validate_participant_bindings(participant_bindings, envelope.spans)
            )

            quality = self._quality(
                canonical,
                envelope,
                attributed,
                quality_evidence,
                enhancement_review=enhancement_review,
                additional_review_reasons=(
                    ("Q_DIARIZATION_FAILED", "Q_PARTICIPANT_BINDING_SKIPPED")
                    if diarization_failed else ()
                ),
            )
            model_hashes = (
                dict(self._model_hashes) if self._has_release_model_authority() else {}
            )
            execution_receipt = _execution_receipt_sha256(
                source=canonical.source,
                config=self.config,
                runtime_config_hash=self.runtime_config_hash,
                model_pack=self.model_pack,
                model_hashes=model_hashes,
                calibration=self.calibration,
                transcript_receipt=transcript_receipt,
                source_timeline=source_timeline,
                spans=envelope.spans,
                attributed=attributed,
                participant_bindings=explicit_bindings,
                quality=quality,
            )
            id_namespace = f"{self.config.pipeline_version}:{execution_receipt}"
            turns = _build_turns(
                attributed, canonical.source.audio_sha256, id_namespace
            )
            neutral = _neutral_words(attributed)
            aware = turns if quality.summary_mode == "SPEAKER_AWARE" else ()
            calibration_id = (
                self.calibration.profile_id
                if self.calibration is not None
                and self.calibration.release_authorized
                and self.model_pack is not None
                and self.calibration.profile.provenance.get("model_pack_id")
                == self.model_pack.pack_id
                and self.calibration.profile.provenance.get("pipeline_version")
                == self.config.pipeline_version
                and not self.calibration.mismatch_reason_codes({
                    "calibration_profile_id": self.calibration.profile_id,
                    "model_hashes": model_hashes,
                    "source_sample_rate_hz": canonical.source.native_sample_rate_hz,
                    "config_hash": self.runtime_config_hash,
                })
                else None
            )
            stage_rtf = {"diarization": envelope.rtf} if envelope.rtf is not None else {}
            run = PipelineRunMetadata(
                deterministic_id(
                    canonical.source.audio_sha256,
                    "pipeline_run",
                    0,
                    canonical.source.duration_us,
                    pipeline_version=id_namespace,
                ),
                self.config.pipeline_version,
                (
                    self.model_pack.pack_id
                    if self.model_pack is not None and self._has_release_model_authority()
                    else "UNVERIFIED_INJECTED"
                ),
                model_hashes,
                calibration_id,
                "CPUExecutionProvider" if self._has_release_model_authority() else "INJECTED_LOCAL_BACKEND",
                {
                    "platform": platform.system().lower(),
                    "architecture": platform.machine().lower(),
                },
                stage_rtf,
                envelope.peak_rss_mb,
            )
            result = PipelineResult(
                deterministic_id(
                    canonical.source.audio_sha256,
                    "pipeline_result",
                    0,
                    canonical.source.duration_us,
                    pipeline_version=id_namespace,
                ),
                canonical.source,
                run,
                envelope.spans,
                attributed,
                turns,
                explicit_bindings,
                quality,
                aware,
                neutral,
                {
                    "orchestrator": {
                        "schema": "sddiar.production_orchestrator_v1",
                        "runtime_config_sha256": self.runtime_config_hash,
                        "release_model_authority": bool(
                            self._has_release_model_authority()
                        ),
                        "release_calibration_authority": bool(
                            calibration_id is not None
                        ),
                        "enhancement_enabled": self._config.enable_enhancement,
                        "diarization_status": (
                            "FAILED_NEUTRAL_FALLBACK" if diarization_failed else "COMPLETED"
                        ),
                        "source_resampled_to_16k": canonical.resampled,
                        "stt_backend": transcript_receipt,
                        "transcript_word_count": len(attributed),
                        "participant_binding_count": len(explicit_bindings),
                        "participant_binding_authority": (
                            "CALLER_EXPLICIT_UNVERIFIED"
                            if explicit_bindings else "NONE"
                        ),
                        "raw_transcript_in_diagnostics": False,
                    }
                },
            )
            # Serialization is part of the transaction.  Invalid/non-finite or
            # internal state never escapes as a partially published result.
            _assert_canonical_unchanged(canonical)
            if self.model_pack is not None:
                _assert_model_pack_unchanged(self.model_pack)
            ResultSerializer().serialize(result)
            return result

    def process_json(self, *args: Any, **kwargs: Any) -> bytes:
        return ResultSerializer().serialize(self.process(*args, **kwargs))

    def _get_diarizer(self) -> DiarizerBackend:
        if self._diarizer is not None:
            return self._diarizer
        with self._diarizer_lock:
            if self._diarizer is None:
                if self._diarizer_builder is None:
                    raise ProductionOrchestrationError("diarizer is not configured")
                self._diarizer = self._diarizer_builder()
                if self._pack_bound_backend:
                    self._pack_bound_instance = self._diarizer
        return self._diarizer

    def _transcript(
        self,
        canonical: CanonicalAudio,
        supplied_word_timeline: WordTimeline | None,
    ) -> tuple[WordTimeline, dict[str, Any]]:
        timeline = supplied_word_timeline or WordTimeline((), {})
        request_receipt = _json_sha256({
            "pipeline_version": self.config.pipeline_version,
            "runtime_config_hash": self.runtime_config_hash,
            "supplied_word_timeline_sha256": (
                _word_timeline_sha256(timeline)
                if supplied_word_timeline is not None else None
            ),
            "local_stt": (
                {
                    "backend_id": self._transcript_backend.identity.backend_id,
                    "backend_version": self._transcript_backend.identity.backend_version,
                    "engine_sha256": self._transcript_backend.identity.engine_artifact.sha256,
                    "model_sha256": self._transcript_backend.identity.model_artifact.sha256,
                }
                if self._transcript_backend is not None else None
            ),
        })
        request = AudioRequest(
            deterministic_id(
                canonical.source.audio_sha256,
                "audio_request",
                0,
                canonical.source.duration_us,
                pipeline_version=f"{self.config.pipeline_version}:{request_receipt}",
            ),
            "LOCAL_CANONICAL_AUDIO",
            self.calibration.profile_id if self.calibration is not None else "UNVERIFIED",
            timeline.words,
        )
        if supplied_word_timeline is not None:
            backend: TranscriptBackend = SuppliedWordsBackend(timeline)
            provider: WordTimelineProvider = SuppliedWordTimelineProvider()
            payload = backend.transcribe(request, canonical.source, canonical)
            source_timeline = provider.words(
                payload, canonical.source, canonical.time_warp
            )
            return source_timeline, {
                "kind": "CALLER_SUPPLIED_SOURCE_TIME_WORDS",
                "backend_id": None,
                "backend_version": None,
                "engine_sha256": None,
                "model_sha256": None,
                "decoder_policy": None,
            }
        if self._transcript_backend is None:
            return timeline, {
                "kind": "NOT_CONFIGURED",
                "backend_id": None,
                "backend_version": None,
                "engine_sha256": None,
                "model_sha256": None,
                "decoder_policy": None,
            }
        payload = self._transcript_backend.transcribe(
            request, canonical.source, canonical
        )
        source_timeline = self._word_timeline_provider.words(
            payload, canonical.source, canonical.time_warp
        )
        # The final source-time invariant is always enforced by the built-in
        # validator before speaker mapping.
        source_timeline = SuppliedWordTimelineProvider().words(
            SuppliedTranscriptPayload(source_timeline),
            canonical.source,
            canonical.time_warp,
        )
        return source_timeline, {
            "kind": "HASH_BOUND_INJECTED_LOCAL_STT",
            "backend_id": payload.backend_id,
            "backend_version": payload.backend_version,
            "engine_sha256": payload.engine_sha256,
            "model_sha256": payload.model_sha256,
            "implementation_binding": "INJECTED_CONTRACT_REQUIRES_RELEASE_AUDIT",
            "decoder_policy": self._transcript_backend.decoder_policy,
        }

    def _quality(
        self,
        canonical: CanonicalAudio,
        envelope: DiarizationEnvelope,
        attributed: Sequence[AttributedWord],
        evidence: ProductionQualityEvidence | None,
        *,
        enhancement_review: bool,
        additional_review_reasons: Sequence[str] = (),
    ) -> FileQualityReport:
        durations = {
            label: sum(
                span.end_us - span.start_us
                for span in envelope.spans
                if span.speaker_id == label
            )
            for label in (*sorted(_SPEAKER_IDS), "UNKNOWN", "OVERLAP")
        }
        assigned = durations["SPEAKER_00"] + durations["SPEAKER_01"]
        speech = assigned + durations["UNKNOWN"] + durations["OVERLAP"]
        word_eligible = len(attributed)
        word_assigned = sum(word.speaker_id in _SPEAKER_IDS for word in attributed)
        assigned_labels = {
            span.speaker_id for span in envelope.spans if span.speaker_id in _SPEAKER_IDS
        }
        metrics: dict[str, float] = dict(evidence.metrics) if evidence is not None else {}
        metrics.update({
            "unknown_ratio": durations["UNKNOWN"] / max(1, assigned + durations["UNKNOWN"]),
            "overlap_ratio": durations["OVERLAP"] / max(1, speech),
            "word_attribution_coverage": word_assigned / max(1, word_eligible),
        })
        model_hashes = (
            dict(self._model_hashes) if self._has_release_model_authority() else {}
        )
        review_reasons: list[str] = []
        if not (
            self._has_release_model_authority()
        ):
            review_reasons.append("Q_RELEASE_MODEL_AUTHORITY_MISSING")
        if evidence is None or not evidence.all_required_metrics_evaluated:
            review_reasons.append("Q_RELEASE_EVIDENCE_MISSING")
        elif evidence is not None:
            # This public data object is useful for diagnostics and regression
            # reports, but has no signature/seal tying it to the release scorer.
            review_reasons.append("Q_RELEASE_EVIDENCE_UNVERIFIED")
        if evidence is None or evidence.osd_coverage != "EVALUATED":
            review_reasons.append("Q_OSD_NOT_EVALUATED")
        if not attributed:
            review_reasons.append("Q_WORD_TIMELINE_MISSING")
        elif word_assigned == 0:
            review_reasons.append("Q_NO_ASSIGNED_WORDS")
        if assigned == 0:
            review_reasons.append("Q_NO_ASSIGNED_SPEECH")
        if (
            (envelope.decision == "H2_CONFIRMED" and len(assigned_labels) != 2)
            or (envelope.decision == "H1_CONFIRMED" and len(assigned_labels) != 1)
        ):
            review_reasons.append("Q_SPEAKER_TOPOLOGY_OUTPUT_MISMATCH")
        if enhancement_review:
            review_reasons.append("Q_ENHANCEMENT_UNVERIFIED")
        if (
            self.calibration is not None
            and self.model_pack is not None
            and self.calibration.profile.provenance.get("model_pack_id")
            != self.model_pack.pack_id
        ):
            review_reasons.append("Q_CALIBRATION_MODEL_PACK_MISMATCH")
        if (
            self.calibration is not None
            and self.calibration.profile.provenance.get("pipeline_version")
            != self.config.pipeline_version
        ):
            review_reasons.append("Q_CALIBRATION_PIPELINE_VERSION_MISMATCH")
        relations = dict(evidence.threshold_relations) if evidence is not None else {}
        if any(relation != "PASS" for relation in relations.values()):
            review_reasons.append("Q_RELEASE_THRESHOLD_FAILED")
        if self.calibration is not None:
            required = set(self.calibration.thresholds)
            if any(relations.get(name) != "PASS" for name in required):
                review_reasons.append("Q_RELEASE_THRESHOLD_EVIDENCE_INCOMPLETE")
        if evidence is not None:
            review_reasons.extend(evidence.review_reasons)
        review_reasons.extend(additional_review_reasons)
        diagnostics = {
            "metrics": metrics,
            "threshold_relations": relations,
            "speaker_count_status": (
                "CONFIDENT_2" if envelope.decision == "H2_CONFIRMED"
                else "CONFIDENT_1" if envelope.decision == "H1_CONFIRMED"
                else "UNCERTAIN_1_OR_2"
            ),
            "hypothesis_uncertain": envelope.decision == "UNCERTAIN_1_OR_2",
            "review_reasons": tuple(dict.fromkeys(review_reasons)),
            "unattributed_reasons": (
                evidence.unattributed_reasons if evidence is not None else ()
            ),
            "all_high_rules_pass": bool(
                evidence is not None and evidence.all_high_rules_pass
            ),
            "osd_coverage": evidence.osd_coverage if evidence is not None else "NOT_EVALUATED",
            "calibration_profile_id": (
                self.calibration.profile_id if self.calibration is not None else None
            ),
            "model_hashes": model_hashes,
            "source_sample_rate_hz": canonical.source.native_sample_rate_hz,
            "config_hash": self.runtime_config_hash,
        }
        quality = RuleBasedQualityGate().evaluate(diagnostics, self.calibration)
        if quality.status == "REVIEW_REQUIRED" and quality.summary_mode != "SPEAKER_NEUTRAL":
            quality = replace(quality, summary_mode="SPEAKER_NEUTRAL")
        return quality


__all__ = [
    "CanonicalAudio",
    "DiarizationEnvelope",
    "DiarizerBackend",
    "EnhancementHook",
    "HashVerifiedLocalTranscriptBackend",
    "LocalSttEngine",
    "LocalSttTranscriptPayload",
    "Pcm16CanonicalAdapter",
    "ProductionOrchestrationError",
    "ProductionOrchestrator",
    "ProductionOrchestratorConfig",
    "ProductionQualityEvidence",
    "canonical_production_config_hash",
    "SuppliedTranscriptPayload",
    "SuppliedWordTimelineProvider",
    "SuppliedWordsBackend",
    "TranscriptBackend",
    "VerifiedLocalSttIdentity",
    "WordTimelineProvider",
    "verify_local_stt_identity",
]
