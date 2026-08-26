"""Zero-dependency immutable contracts used at package seams.

The production boundary may use Pydantic v2; this module intentionally keeps
the reference implementation runnable with only the Python standard library.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Literal
import hashlib, base64, math, re
from .errors import ContractValidationError, ProtectedOverlapError

MAX_TIME_US = 9_223_372_036_854_775_807
SpeakerId = Literal["SPEAKER_00", "SPEAKER_01", "UNKNOWN", "OVERLAP", "OTHER", "NON_SPEECH"]
_SHA = re.compile(r"^[0-9a-f]{64}$")

def _time(v: int, name: str = "time_us") -> None:
    if type(v) is not int or not 0 <= v <= MAX_TIME_US:
        raise ContractValidationError(f"{name} must be a non-negative int64 microsecond value")
def _range(start: int, end: int) -> None:
    _time(start, "start_us"); _time(end, "end_us")
    if start >= end: raise ContractValidationError("start_us must be less than end_us")
def _finite(v: float, name: str) -> None:
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ContractValidationError(f"{name} must be finite")
def deterministic_id(audio_sha256: str, object_kind: str, start_us: int = 0,
                     end_us: int = 0, ordinal: int = 0,
                     schema_version: str = "1.0", pipeline_version: str = "1") -> str:
    """Return a stable lower-case base32 prefix ID (no padding)."""
    if not _SHA.fullmatch(audio_sha256): raise ContractValidationError("audio_sha256 must be lowercase SHA-256")
    if not object_kind: raise ContractValidationError("object_kind is required")
    payload = f"{audio_sha256}|{object_kind}|{start_us}|{end_us}|{ordinal}|{schema_version}|{pipeline_version}".encode()
    return base64.b32encode(hashlib.sha256(payload).digest()).decode("ascii").rstrip("=").lower()[:26]

@dataclass(frozen=True, slots=True)
class Timebase:
    timebase_id: str; unit: str = "microseconds"; source_sample_rate_hz: Optional[int] = None
    duration_us: int = 0; origin: str = "decoded_source_start"; schema_version: str = "1.0"
    def __post_init__(self):
        _time(self.duration_us, "duration_us")
        if self.unit != "microseconds" or self.origin != "decoded_source_start": raise ContractValidationError("unsupported timebase")
        if self.source_sample_rate_hz is not None and (type(self.source_sample_rate_hz) is not int or self.source_sample_rate_hz <= 0): raise ContractValidationError("invalid sample rate")

@dataclass(frozen=True, slots=True)
class AudioRequest:
    request_id: str; source_ref: str; profile_id: str; supplied_words: tuple[Word, ...] = ()
    include_non_speech: bool = False; options: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        if not self.request_id or not self.source_ref or not self.profile_id: raise ContractValidationError("request/source/profile are required")
        if "://" in self.source_ref or self.source_ref.startswith(("file:", "http:", "https:", "pipe:", "device:")): raise ContractValidationError("network/device URI is not an allowed source")

@dataclass(frozen=True, slots=True)
class AudioSourceMetadata:
    audio_sha256: str; container: str; codec: str; native_sample_rate_hz: int; channel_count: int; duration_us: int; timebase: Timebase
    channel_metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        if not _SHA.fullmatch(self.audio_sha256): raise ContractValidationError("audio_sha256 must be lowercase SHA-256")
        if self.native_sample_rate_hz <= 0 or self.channel_count <= 0: raise ContractValidationError("invalid audio metadata")
        _time(self.duration_us, "duration_us")

@dataclass(frozen=True, slots=True)
class AudioView:
    view_id: str; kind: Literal["MIXDOWN_MONO", "SOURCE_CHANNEL"]; sample_rate_hz: int; total_samples: int; time_warp: tuple[TimeWarpSegment, ...]; channel_count: int = 1
    def __post_init__(self):
        if self.kind not in ("MIXDOWN_MONO", "SOURCE_CHANNEL") or self.channel_count != 1 or self.sample_rate_hz <= 0 or self.total_samples <= 0: raise ContractValidationError("invalid audio view")

@dataclass(frozen=True, slots=True)
class SpeechRegion:
    region_id: str; view_id: str; start_us: int; end_us: int; speech_evidence: Optional[float] = None; reason_codes: tuple[str, ...] = ()
    def __post_init__(self):
        _range(self.start_us, self.end_us)
        if self.speech_evidence is not None: _finite(self.speech_evidence, "speech_evidence")

@dataclass(frozen=True, slots=True)
class TimeWarpSegment:
    segment_id: str; view_id: str; view_start_sample: int; view_end_sample: int
    source_start_us: int; source_end_us: int; mapping_kind: Literal["AFFINE_RESAMPLE", "DECODE_PTS"] = "AFFINE_RESAMPLE"
    def __post_init__(self):
        if type(self.view_start_sample) is not int or type(self.view_end_sample) is not int or self.view_start_sample < 0 or self.view_start_sample >= self.view_end_sample: raise ContractValidationError("invalid view sample range")
        _range(self.source_start_us, self.source_end_us)
        if self.mapping_kind not in ("AFFINE_RESAMPLE", "DECODE_PTS"): raise ContractValidationError("invalid mapping kind")
    def source_us(self, view_sample: int) -> int:
        if not self.view_start_sample <= view_sample <= self.view_end_sample: raise ContractValidationError("view sample outside segment")
        numerator = (view_sample - self.view_start_sample) * (self.source_end_us - self.source_start_us)
        denominator = self.view_end_sample - self.view_start_sample
        # Integer half-up rounding prevents float drift on long recordings.
        return self.source_start_us + (numerator + denominator // 2) // denominator

@dataclass(frozen=True, slots=True)
class Word:
    word_id: str; start_us: int; end_us: int; text: str; confidence: Optional[float] = None; source_chunk_id: Optional[str] = None
    def __post_init__(self):
        _range(self.start_us, self.end_us)
        if not isinstance(self.text, str) or not self.text.strip(): raise ContractValidationError("word text cannot be empty")
        if self.confidence is not None: _finite(self.confidence, "confidence")

@dataclass(frozen=True, slots=True)
class ProtectedOverlapSpan:
    span_id: str; start_us: int; end_us: int; overlap_evidence: float; evidence_ids: tuple[str, ...] = ()
    def __post_init__(self):
        _range(self.start_us, self.end_us); _finite(self.overlap_evidence, "overlap_evidence")

@dataclass(frozen=True, slots=True)
class WordProvenance:
    word_id: str; crosses_timewarp_boundary: bool = False; source_chunk_ids: tuple[str, ...] = (); duplicate_suspect: bool = False

@dataclass(frozen=True, slots=True)
class WordTimeline:
    words: tuple[Word, ...]; provenance_by_word_id: Mapping[str, WordProvenance]
    def __post_init__(self):
        ids = [w.word_id for w in self.words]
        if len(ids) != len(set(ids)): raise ContractValidationError("word IDs must be unique")
        if set(ids) != set(self.provenance_by_word_id): raise ContractValidationError("one provenance entry is required per word")
        if any(p.word_id != wid for wid, p in self.provenance_by_word_id.items()): raise ContractValidationError("provenance key mismatch")

@dataclass(frozen=True, slots=True)
class Tracklet:
    tracklet_id: str; speech_region_id: str; continuity_group_id: str; start_us: int; end_us: int; clean_speech_us: int
    kind: Literal["ANCHOR", "SUPPORT", "MICRO"]; boundary_evidence_ids: tuple[str, ...] = (); scd_evidence_before: Optional[float] = None; scd_evidence_after: Optional[float] = None; protected_overlap: bool = False; mixed_tracklet_suspect: bool = False
    def __post_init__(self):
        _range(self.start_us, self.end_us)
        if type(self.clean_speech_us) is not int or not 0 <= self.clean_speech_us <= self.end_us-self.start_us: raise ContractValidationError("invalid clean speech duration")
        if self.kind not in ("ANCHOR", "SUPPORT", "MICRO"): raise ContractValidationError("invalid tracklet kind")
        if self.protected_overlap and self.kind == "ANCHOR": raise ProtectedOverlapError("protected overlap cannot be an anchor")

@dataclass(frozen=True, slots=True)
class EmbeddingRegion:
    embedding_region_id: str; tracklet_id: str; start_us: int; end_us: int; clean_speech_us: int; speech_coverage_ratio: float
    def __post_init__(self):
        _range(self.start_us, self.end_us); _finite(self.speech_coverage_ratio, "speech_coverage_ratio")
        if not 0 <= self.speech_coverage_ratio <= 1: raise ContractValidationError("coverage ratio must be in [0,1]")

@dataclass(frozen=True, slots=True)
class SpeakerAssignment:
    tracklet_id: str; speaker_id: SpeakerId; attribution_status: str; stable_distance: Optional[float] = None; effective_distance: Optional[float] = None; margin: Optional[float] = None; evidence_ids: tuple[str, ...] = (); reason_codes: tuple[str, ...] = ()
    def __post_init__(self):
        if self.speaker_id not in ("SPEAKER_00", "SPEAKER_01", "UNKNOWN", "OVERLAP", "OTHER", "NON_SPEECH"): raise ContractValidationError("invalid speaker ID")
        for n, v in (("stable_distance", self.stable_distance), ("effective_distance", self.effective_distance), ("margin", self.margin)):
            if v is not None: _finite(v, n)

@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    embedding_region_id: str; tracklet_id: str; is_valid: bool; vector: Any = None; failure_reason: Optional[str] = None; dimension: int = 0; valid_window_count: int = 0; clean_window_coverage: float = 0.0; intra_window_consistency: float = 0.0; quality: float = 0.0; model_pack_id: str = ""; model_hash: str = ""
    def __post_init__(self):
        if self.is_valid and self.vector is None: raise ContractValidationError("valid embedding requires vector")
        if not self.is_valid and self.vector is not None: raise ContractValidationError("invalid embedding cannot carry vector")
        if self.is_valid and self.dimension <= 0: raise ContractValidationError("valid embedding requires dimension")
        if not self.is_valid and not self.failure_reason: raise ContractValidationError("invalid embedding requires failure reason")

@dataclass(frozen=True, slots=True)
class DiarizationSpan:
    span_id: str; start_us: int; end_us: int; speaker_id: SpeakerId; attribution_status: str; evidence_ids: tuple[str, ...] = (); reason_codes: tuple[str, ...] = ()
    def __post_init__(self):
        _range(self.start_us, self.end_us)
        if self.speaker_id not in ("SPEAKER_00", "SPEAKER_01", "UNKNOWN", "OVERLAP", "OTHER", "NON_SPEECH"): raise ContractValidationError("invalid speaker ID")
        if self.speaker_id == "OVERLAP" and self.attribution_status == "ASSIGNED": raise ProtectedOverlapError("overlap cannot be assigned")

@dataclass(frozen=True, slots=True)
class AttributedWord(Word):
    speaker_id: SpeakerId = "UNKNOWN"; attribution_status: str = "UNKNOWN_INSUFFICIENT_EVIDENCE"; supporting_span_ids: tuple[str, ...] = (); speaker_coverage_ratio: Optional[float] = None; competing_speaker_coverage_ratio: Optional[float] = None; reason_codes: tuple[str, ...] = ()
    def __post_init__(self):
        # Explicit base call is compatible with slotted dataclass inheritance
        # on both the CPython 3.11 release target and newer interpreters.
        Word.__post_init__(self)
        if self.speaker_id not in ("SPEAKER_00", "SPEAKER_01", "UNKNOWN", "OVERLAP", "OTHER", "NON_SPEECH"): raise ContractValidationError("invalid speaker ID")
        if self.speaker_id == "OVERLAP" and self.attribution_status != "OVERLAP_UNATTRIBUTED": raise ProtectedOverlapError("overlap word must remain unattributed")

@dataclass(frozen=True, slots=True)
class TrackletBuildResult:
    tracklets: tuple[Tracklet, ...]; protected_overlap_spans: tuple[ProtectedOverlapSpan, ...]; boundary_evidence_ids: tuple[str, ...] = ()


def _unit_vector(vector: tuple[float, ...], name: str = "vector") -> None:
    if not vector:
        raise ContractValidationError(f"{name} cannot be empty")
    for value in vector:
        _finite(value, name)
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ContractValidationError(f"{name} must be L2-normalized")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    embedding_region_id: str
    tracklet_id: str
    is_valid: bool
    vector: tuple[float, ...] | None = None
    failure_reason: Optional[str] = None
    dimension: int = 0
    valid_window_count: int = 0
    clean_window_coverage: float = 0.0
    intra_window_consistency: float = 0.0
    quality: float = 0.0
    model_pack_id: str = ""
    model_hash: str = ""

    def __post_init__(self):
        if self.is_valid:
            if self.vector is None or self.failure_reason is not None:
                raise ContractValidationError("valid embedding requires vector and no failure reason")
            _unit_vector(self.vector)
            if self.dimension != len(self.vector) or self.dimension <= 0:
                raise ContractValidationError("embedding dimension does not match vector")
        elif self.vector is not None or not self.failure_reason:
            raise ContractValidationError("invalid embedding requires reason and no vector")
        for name, value in (("clean_window_coverage", self.clean_window_coverage),
                            ("intra_window_consistency", self.intra_window_consistency),
                            ("quality", self.quality)):
            _finite(value, name)


@dataclass(frozen=True, slots=True)
class AnchorEvidence:
    tracklet_id: str
    vector: tuple[float, ...]
    weight: float
    clean_speech_us: int
    independent_block_id: str
    continuity_group_id: str
    start_us: int
    end_us: int
    scd_evidence_before: Optional[float] = None

    def __post_init__(self):
        _unit_vector(self.vector)
        _finite(self.weight, "weight")
        if self.weight <= 0:
            raise ContractValidationError("anchor weight must be positive")
        _range(self.start_us, self.end_us)
        if not self.independent_block_id or not self.continuity_group_id:
            raise ContractValidationError("anchor evidence IDs are required")
        if self.scd_evidence_before is not None:
            _finite(self.scd_evidence_before, "scd_evidence_before")


@dataclass(frozen=True, slots=True)
class SpeakerHypothesis:
    k: Literal[1, 2]
    centers: tuple[tuple[float, ...], ...]
    anchor_labels: Mapping[str, int | None]
    is_valid: bool
    valid_constraints: bool
    robust_cost: float
    total_cost: float
    cost_components: Mapping[str, float] = field(default_factory=dict)
    outlier_ratio: float = 0.0
    cluster_dispersion: tuple[float, ...] = ()
    clean_duration_us: tuple[int, ...] = ()
    independent_anchor_count: tuple[int, ...] = ()
    cluster_support_ok: bool = False
    dispersion_ok: bool = False
    outlier_ratio_ok: bool = False
    third_speaker_risk: bool = False
    separation: Optional[float] = None
    label_stability: Optional[float] = None
    centroid_stability: Optional[float] = None
    temporal_interleaving: Optional[bool] = None
    continuous_speech_conflict: Optional[bool] = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.k not in (1, 2):
            raise ContractValidationError("hypothesis k must be 1 or 2")
        if self.is_valid:
            if len(self.centers) != self.k:
                raise ContractValidationError("valid hypothesis needs one center per cluster")
            for center in self.centers:
                _unit_vector(center, "centroid")
        for name, value in (("robust_cost", self.robust_cost), ("total_cost", self.total_cost),
                            ("outlier_ratio", self.outlier_ratio)):
            _finite(value, name)


@dataclass(frozen=True, slots=True)
class HypothesisDecision:
    state: Literal["H1_CONFIRMED", "H2_CONFIRMED", "UNCERTAIN_1_OR_2"]
    hypothesis: Optional[SpeakerHypothesis]
    reason_codes: tuple[str, ...] = ()

    @property
    def selected(self) -> Optional[SpeakerHypothesis]:
        return self.hypothesis


@dataclass(frozen=True, slots=True)
class SpeakerState:
    speaker_id: Literal["SPEAKER_00", "SPEAKER_01"]
    stable_anchor_centroid: tuple[float, ...]
    stable_anchor_ids: tuple[str, ...]
    stable_dispersion: float
    recent_centroid: Optional[tuple[float, ...]] = None
    recent_mass: float = 0.0
    recent_last_us: Optional[int] = None
    recent_frozen: bool = False
    drift_flags: frozenset[str] = frozenset()

    def __post_init__(self):
        _unit_vector(self.stable_anchor_centroid, "stable_anchor_centroid")
        if self.recent_centroid is not None:
            _unit_vector(self.recent_centroid, "recent_centroid")
        _finite(self.stable_dispersion, "stable_dispersion")
        _finite(self.recent_mass, "recent_mass")


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    turn_id: str
    start_us: int
    end_us: int
    speaker_id: SpeakerId
    attributed_word_ids: tuple[str, ...]
    text: str
    attribution_status: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _range(self.start_us, self.end_us)
        if not isinstance(self.text, str):
            raise ContractValidationError("turn text must be a string")


@dataclass(frozen=True, slots=True)
class ParticipantBinding:
    speaker_id: Literal["SPEAKER_00", "SPEAKER_01"]
    external_participant_id: Optional[str]
    role_label: Optional[str]
    method: Literal["TRUSTED_CHANNEL_METADATA", "PREREGISTERED_VOICE", "HUMAN_CONFIRMED_SEGMENT", "EXTERNAL_AUTHORITATIVE_METADATA"]
    confidence: float
    evidence_ids: tuple[str, ...]

    def __post_init__(self):
        _finite(self.confidence, "binding confidence")
        if not 0.0 <= self.confidence <= 1.0 or not self.evidence_ids:
            raise ContractValidationError("binding requires evidence and a [0, 1] confidence")


@dataclass(frozen=True, slots=True)
class FileQualityReport:
    status: Literal["PASS_HIGH", "PASS_STANDARD", "PASS_WITH_UNATTRIBUTED", "REVIEW_REQUIRED", "UNSUPPORTED"]
    speaker_count_status: Literal["CONFIDENT_1", "CONFIDENT_2", "UNCERTAIN_1_OR_2", "OUT_OF_PROFILE"]
    summary_mode: Literal["SPEAKER_AWARE", "SPEAKER_NEUTRAL", "MANUAL_REVIEW"]
    reason_codes: tuple[str, ...]
    metrics: Mapping[str, float]
    threshold_relations: Mapping[str, Literal["PASS", "WARN", "FAIL", "NOT_EVALUATED"]] = field(default_factory=dict)
    calibration_profile_id: Optional[str] = None

    def __post_init__(self):
        if self.status.startswith("PASS") and not self.calibration_profile_id:
            raise ContractValidationError("PASS requires a calibration profile")
        for name, value in self.metrics.items():
            _finite(value, f"metric {name}")


@dataclass(frozen=True, slots=True)
class PipelineRunMetadata:
    run_id: str
    pipeline_version: str
    model_pack_id: str
    model_hashes: Mapping[str, str]
    calibration_profile_id: Optional[str]
    execution_provider: str
    hardware_fingerprint: Mapping[str, str]
    stage_rtf: Mapping[str, float]
    peak_process_tree_rss_mb: float

    def __post_init__(self):
        _finite(self.peak_process_tree_rss_mb, "peak_process_tree_rss_mb")
        for name, value in self.stage_rtf.items():
            _finite(value, f"stage rtf {name}")


@dataclass(frozen=True, slots=True)
class PipelineResult:
    result_id: str
    source: AudioSourceMetadata
    run: PipelineRunMetadata
    diarization_spans: tuple[DiarizationSpan, ...]
    attributed_words: tuple[AttributedWord, ...]
    speaker_turns: tuple[SpeakerTurn, ...]
    participant_bindings: tuple[ParticipantBinding, ...]
    quality: FileQualityReport
    speaker_aware_transcript: tuple[SpeakerTurn, ...]
    speaker_neutral_transcript: tuple[AttributedWord, ...]
    extensions: Mapping[str, Any] = field(default_factory=dict)
