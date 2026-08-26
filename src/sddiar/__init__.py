"""Offline CPU speaker-diarization contracts (reference, zero dependencies)."""
from .contracts import *
from .timebase import TimeWarp
from .errors import *
from .pipeline import EvidencePipeline, DiarizationRun
from .serialization import ResultSerializer
from .segmentation import RuleEvidenceSegmentation, SegmentationEvidence
from .correction import HumanSpeakerCorrection, apply_session_corrections
from .onnx_diarization import LocalOnnxDiarizationConfig, LocalOnnxDiarizationResult, LocalOnnxDiarizer
from .ort_cpu import OrtCpuBudget, OrtCpuBudgetExceededError, OrtCpuConfig, create_ort_session
from .runtime_env import RuntimeCpuDelta, RuntimeCpuSnapshot, delta_cpu_snapshots, read_cpu_snapshot
from .silero_temporal import SileroTemporalPostprocessor, TemporalVadConfig, TemporalVadResult
from .pyannote_segmentation_runtime import (
    PyannoteChangeEvent,
    PyannoteFrameEvidence,
    PyannoteSegmentationConfig,
    PyannoteSegmentationOnnxRuntime,
    PyannoteSegmentationResult,
)
from .annotation_intake import (
    AnnotationValidationReport,
    IntakeIssue,
    WordsArtifact,
    load_words_artifact,
    load_words_jsonl_artifact,
    read_words_artifact,
    validate_annotation_dataset,
    validate_annotation_manifest,
)
from .production_orchestrator import (
    CanonicalAudio,
    DiarizationEnvelope,
    HashVerifiedLocalTranscriptBackend,
    LocalSttEngine,
    LocalSttTranscriptPayload,
    Pcm16CanonicalAdapter,
    ProductionOrchestrationError,
    ProductionOrchestrator,
    ProductionOrchestratorConfig,
    ProductionQualityEvidence,
    SuppliedTranscriptPayload,
    SuppliedWordTimelineProvider,
    SuppliedWordsBackend,
    TranscriptBackend,
    VerifiedLocalSttIdentity,
    WordTimelineProvider,
    canonical_production_config_hash,
    verify_local_stt_identity,
)
from .whispercpp_backend import (
    WHISPER_CPP_BACKEND_ID,
    WHISPER_CPP_BACKEND_VERSION,
    WHISPER_CPP_COMMIT,
    WhisperCppBackend,
    WhisperCppConfig,
)

__all__ = [
    "TimeWarp", "Timebase", "TimeWarpSegment", "AudioRequest", "AudioSourceMetadata",
    "AudioView", "SpeechRegion", "Word", "WordTimeline", "WordProvenance",
    "AttributedWord", "DiarizationSpan", "Tracklet", "TrackletBuildResult",
    "ProtectedOverlapSpan", "EmbeddingRegion", "EmbeddingResult", "AnchorEvidence",
    "SpeakerHypothesis", "HypothesisDecision", "SpeakerState", "SpeakerAssignment",
    "SpeakerTurn", "ParticipantBinding", "FileQualityReport", "PipelineRunMetadata",
    "PipelineResult", "deterministic_id",
    "EvidencePipeline", "DiarizationRun",
    "ResultSerializer",
    "RuleEvidenceSegmentation", "SegmentationEvidence",
    "HumanSpeakerCorrection", "apply_session_corrections",
    "LocalOnnxDiarizationConfig", "LocalOnnxDiarizationResult", "LocalOnnxDiarizer",
    "PyannoteSegmentationConfig", "PyannoteFrameEvidence", "PyannoteChangeEvent",
    "PyannoteSegmentationResult", "PyannoteSegmentationOnnxRuntime",
    "OrtCpuBudget", "OrtCpuConfig", "OrtCpuBudgetExceededError", "create_ort_session",
    "RuntimeCpuSnapshot", "RuntimeCpuDelta", "read_cpu_snapshot", "delta_cpu_snapshots",
    "TemporalVadConfig", "TemporalVadResult", "SileroTemporalPostprocessor",
    "AnnotationValidationReport", "IntakeIssue", "validate_annotation_dataset",
    "validate_annotation_manifest", "WordsArtifact", "load_words_artifact",
    "load_words_jsonl_artifact", "read_words_artifact",
    "CanonicalAudio", "DiarizationEnvelope", "Pcm16CanonicalAdapter",
    "ProductionOrchestrationError", "ProductionOrchestrator",
    "ProductionOrchestratorConfig", "ProductionQualityEvidence",
    "TranscriptBackend", "WordTimelineProvider", "SuppliedTranscriptPayload",
    "SuppliedWordsBackend", "SuppliedWordTimelineProvider",
    "LocalSttEngine", "LocalSttTranscriptPayload", "VerifiedLocalSttIdentity",
    "HashVerifiedLocalTranscriptBackend", "verify_local_stt_identity",
    "WhisperCppBackend", "WhisperCppConfig", "WHISPER_CPP_BACKEND_ID",
    "WHISPER_CPP_BACKEND_VERSION", "WHISPER_CPP_COMMIT",
    "canonical_production_config_hash",
]
