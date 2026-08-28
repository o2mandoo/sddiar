"""Authority and release-gate contracts for Korean-only diarization evaluation.

This module does not open audio or annotation files.  It binds a corpus lock
to the already-implemented scorer and keeps GOLD, SILVER, and CHALLENGE lanes
separate.  V1 intentionally emits metric-gate evidence only: no verifier or
policy object supplied by a library caller can create product-release authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Mapping, Protocol

from .evaluation import EvaluationReport


CORPUS_LOCK_SCHEMA = "sddiar-korean-corpus-lock/v1"
RELEASE_POLICY_SCHEMA = "sddiar-korean-release-policy/v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_ROLES = frozenset({"GOLD", "SILVER", "CHALLENGE"})
_ORIGINS = frozenset({"HUMAN", "PUBLISHER_HUMAN", "MACHINE", "HYBRID", "SYNTHETIC"})
_LICENSE = frozenset({"APPROVED_INTERNAL_EVALUATION", "RESEARCH_ONLY", "UNVERIFIED", "PROHIBITED"})
_TIMELINE = frozenset({"VERIFIED", "PROVISIONAL", "ABSENT"})
_AUDIT = frozenset({"VERIFIED", "PROVISIONAL", "NOT_RUN"})
_INDEPENDENCE = frozenset({"VERIFIED", "UNVERIFIED"})
_SPLITS = frozenset({"CALIBRATION", "DEVELOPMENT_HOLDOUT", "RELEASE_HOLDOUT"})
_ELIGIBILITY_STATUSES = frozenset({
    "REVIEW_REQUIRED", "DATA_INELIGIBLE", "RESEARCH_ONLY", "DEV_ONLY", "CHALLENGE_ONLY",
})
_METRIC_GATE_STATUSES = _ELIGIBILITY_STATUSES | frozenset({
    "METRIC_GATES_FAIL", "METRIC_GATES_INCOMPLETE_REVIEW_REQUIRED",
    "METRIC_GATES_PASS_REVIEW_REQUIRED",
})
_SUITE_GATE_STATUSES = frozenset({
    "REVIEW_REQUIRED", "SUITE_METRICS_FAIL", "SUITE_METRICS_PASS_REVIEW_REQUIRED",
})
_EXTERNAL_AUTHORITY_REASON = "EXTERNAL_RELEASE_AUTHORITY_REQUIRED"


class KoreanBenchmarkError(ValueError):
    """Invalid corpus authority, policy, or evaluation input."""


class CorpusLockSignatureVerifier(Protocol):
    trust_level: str

    def verify(self, payload: bytes, signature: str, signer_key_id: str) -> bool: ...


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise KoreanBenchmarkError(f"invalid {label} digest")
    return value


@dataclass(frozen=True, slots=True)
class ReferenceCapabilities:
    diarization: bool
    overlap: bool
    scd: bool
    words: bool

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in asdict(self).values()):
            raise KoreanBenchmarkError("reference capabilities must be boolean")


@dataclass(frozen=True, slots=True)
class KoreanCorpusLock:
    corpus_id: str
    corpus_version: str
    authority_role: str
    annotation_origin: str
    license_status: str
    continuous_timeline: str
    audit_status: str
    speaker_independence: str
    reference_capabilities: ReferenceCapabilities
    source_archive_sha256: tuple[str, ...]
    annotation_manifest_sha256: str
    split_lock_sha256: str
    license_text_sha256: str
    audit_sha256: str | None
    release_holdout_locked: bool
    language: str = "ko"
    schema_version: str = CORPUS_LOCK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_LOCK_SCHEMA:
            raise KoreanBenchmarkError("unsupported Korean corpus lock schema")
        if self.language != "ko":
            raise KoreanBenchmarkError("Korean benchmark language must be ko")
        if not isinstance(self.corpus_id, str) or _OPAQUE.fullmatch(self.corpus_id) is None:
            raise KoreanBenchmarkError("corpus_id must be an opaque lower-case identifier")
        if (not isinstance(self.corpus_version, str) or not self.corpus_version
                or len(self.corpus_version) > 64 or any(ord(char) < 32 for char in self.corpus_version)):
            raise KoreanBenchmarkError("invalid corpus version")
        if self.authority_role not in _ROLES or self.annotation_origin not in _ORIGINS:
            raise KoreanBenchmarkError("invalid corpus authority role or annotation origin")
        if self.license_status not in _LICENSE or self.continuous_timeline not in _TIMELINE:
            raise KoreanBenchmarkError("invalid corpus license or timeline status")
        if self.audit_status not in _AUDIT or self.speaker_independence not in _INDEPENDENCE:
            raise KoreanBenchmarkError("invalid corpus audit or speaker-independence status")
        if type(self.release_holdout_locked) is not bool:
            raise KoreanBenchmarkError("release_holdout_locked must be boolean")
        archives = tuple(self.source_archive_sha256)
        if not archives or tuple(sorted(set(archives))) != archives:
            raise KoreanBenchmarkError("source archive digests must be unique and sorted")
        for value in archives:
            _digest(value, "source archive")
        object.__setattr__(self, "source_archive_sha256", archives)
        _digest(self.annotation_manifest_sha256, "annotation manifest")
        _digest(self.split_lock_sha256, "split lock")
        _digest(self.license_text_sha256, "license text")
        _digest(self.audit_sha256, "audit", optional=True)

    @property
    def lock_sha256(self) -> str:
        return sha256(_canonical_bytes(self.as_dict(include_digest=False))).hexdigest()

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "language": self.language,
            "authority_role": self.authority_role,
            "annotation_origin": self.annotation_origin,
            "license_status": self.license_status,
            "continuous_timeline": self.continuous_timeline,
            "audit_status": self.audit_status,
            "speaker_independence": self.speaker_independence,
            "reference_capabilities": asdict(self.reference_capabilities),
            "source_archive_sha256": list(self.source_archive_sha256),
            "annotation_manifest_sha256": self.annotation_manifest_sha256,
            "split_lock_sha256": self.split_lock_sha256,
            "license_text_sha256": self.license_text_sha256,
            "audit_sha256": self.audit_sha256,
            "release_holdout_locked": self.release_holdout_locked,
        }
        if include_digest:
            value["lock_sha256"] = self.lock_sha256
        return value


_VERIFIED_LOCK_SEAL = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedKoreanCorpusLock:
    """Locally signature-verified lock evidence.

    This type proves only that the caller-provided verifier accepted the lock.
    It never creates product-release authority; an external trust root and
    operational approval remain outside this v1 library boundary.
    """

    lock: KoreanCorpusLock
    signer_key_id: str
    payload_sha256: str
    signature_sha256: str

    def __init__(self, lock: KoreanCorpusLock, signer_key_id: str, payload_sha256: str,
                 signature_sha256: str, *, _seal: object | None = None) -> None:
        if _seal is not _VERIFIED_LOCK_SEAL:
            raise TypeError("VerifiedKoreanCorpusLock must be created by verify_korean_corpus_lock")
        object.__setattr__(self, "lock", lock)
        object.__setattr__(self, "signer_key_id", signer_key_id)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "signature_sha256", signature_sha256)


def verify_korean_corpus_lock(
    lock: KoreanCorpusLock,
    *,
    signature: str,
    signer_key_id: str,
    verifier: CorpusLockSignatureVerifier,
) -> VerifiedKoreanCorpusLock:
    if type(lock) is not KoreanCorpusLock:
        raise KoreanBenchmarkError("corpus lock verification requires the exact lock type")
    if (not isinstance(signature, str) or not signature or len(signature) > 16_384
            or not isinstance(signer_key_id, str) or _OPAQUE.fullmatch(signer_key_id) is None):
        raise KoreanBenchmarkError("invalid corpus lock signature identity")
    if getattr(verifier, "trust_level", None) != "RELEASE" or not callable(getattr(verifier, "verify", None)):
        raise KoreanBenchmarkError("corpus lock verifier is not release-trusted")
    payload = _canonical_bytes(lock.as_dict(include_digest=False))
    try:
        valid = verifier.verify(payload, signature, signer_key_id)
    except Exception:
        raise KoreanBenchmarkError("corpus lock signature verification failed") from None
    if valid is not True:
        raise KoreanBenchmarkError("corpus lock signature verification failed")
    return VerifiedKoreanCorpusLock(
        lock, signer_key_id, sha256(payload).hexdigest(),
        sha256(signature.encode("utf-8")).hexdigest(), _seal=_VERIFIED_LOCK_SEAL,
    )


_LOCK_KEYS = frozenset({
    "schema_version", "corpus_id", "corpus_version", "language", "authority_role",
    "annotation_origin", "license_status", "continuous_timeline", "audit_status",
    "speaker_independence", "reference_capabilities", "source_archive_sha256",
    "annotation_manifest_sha256", "split_lock_sha256", "license_text_sha256",
    "audit_sha256", "release_holdout_locked",
})
_CAPABILITY_KEYS = frozenset({"diarization", "overlap", "scd", "words"})


def parse_korean_corpus_lock(value: Mapping[str, Any]) -> KoreanCorpusLock:
    if not isinstance(value, Mapping) or frozenset(value) not in (_LOCK_KEYS, _LOCK_KEYS | {"lock_sha256"}):
        raise KoreanBenchmarkError("Korean corpus lock schema mismatch")
    declared_digest = value.get("lock_sha256")
    payload = {key: item for key, item in value.items() if key != "lock_sha256"}
    if declared_digest is not None:
        _digest(declared_digest, "corpus lock")
    capabilities = payload["reference_capabilities"]
    if not isinstance(capabilities, Mapping) or frozenset(capabilities) != _CAPABILITY_KEYS:
        raise KoreanBenchmarkError("reference capability schema mismatch")
    archives = payload["source_archive_sha256"]
    if not isinstance(archives, list):
        raise KoreanBenchmarkError("source archive digest list is required")
    try:
        capability_value = ReferenceCapabilities(**dict(capabilities))
        result = KoreanCorpusLock(
            corpus_id=payload["corpus_id"], corpus_version=payload["corpus_version"],
            authority_role=payload["authority_role"], annotation_origin=payload["annotation_origin"],
            license_status=payload["license_status"], continuous_timeline=payload["continuous_timeline"],
            audit_status=payload["audit_status"], speaker_independence=payload["speaker_independence"],
            reference_capabilities=capability_value, source_archive_sha256=tuple(archives),
            annotation_manifest_sha256=payload["annotation_manifest_sha256"],
            split_lock_sha256=payload["split_lock_sha256"], license_text_sha256=payload["license_text_sha256"],
            audit_sha256=payload["audit_sha256"], release_holdout_locked=payload["release_holdout_locked"],
            language=payload["language"], schema_version=payload["schema_version"],
        )
        if declared_digest is not None and declared_digest != result.lock_sha256:
            raise KoreanBenchmarkError("Korean corpus lock digest mismatch")
        return result
    except (KeyError, TypeError) as exc:
        raise KoreanBenchmarkError("Korean corpus lock value schema mismatch") from exc


@dataclass(frozen=True, slots=True)
class BenchmarkEligibility:
    status: str
    split: str
    eligible_for_metric_gating: bool
    eligible_for_release_scoring: bool
    reason_codes: tuple[str, ...]
    release_authority: str = "none"

    def __post_init__(self) -> None:
        if self.status not in _ELIGIBILITY_STATUSES or self.split not in _SPLITS:
            raise KoreanBenchmarkError("invalid benchmark eligibility status")
        if (type(self.eligible_for_metric_gating) is not bool
                or self.eligible_for_release_scoring is not False
                or self.release_authority != "none"):
            raise KoreanBenchmarkError("benchmark eligibility cannot grant release authority")
        if (not isinstance(self.reason_codes, tuple)
                or _EXTERNAL_AUTHORITY_REASON not in self.reason_codes):
            raise KoreanBenchmarkError("external release authority reason is required")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unwrapped_lock(lock: KoreanCorpusLock | VerifiedKoreanCorpusLock) -> KoreanCorpusLock:
    if type(lock) is VerifiedKoreanCorpusLock:
        return lock.lock
    if type(lock) is KoreanCorpusLock:
        return lock
    raise KoreanBenchmarkError("invalid Korean corpus lock capability")


def evaluate_benchmark_eligibility(
    lock: KoreanCorpusLock | VerifiedKoreanCorpusLock, *, split: str
) -> BenchmarkEligibility:
    if split not in _SPLITS:
        raise KoreanBenchmarkError("invalid benchmark eligibility input")
    verified = type(lock) is VerifiedKoreanCorpusLock
    item = _unwrapped_lock(lock)
    reasons: list[str] = [_EXTERNAL_AUTHORITY_REASON]
    blockers: list[str] = []
    if not verified:
        reasons.append("CORPUS_LOCK_SIGNATURE_UNVERIFIED")
    if item.license_status == "PROHIBITED":
        blockers.append("LICENSE_PROHIBITED")
    elif item.license_status == "UNVERIFIED":
        blockers.append("LICENSE_UNVERIFIED")
    elif item.license_status == "RESEARCH_ONLY":
        blockers.append("LICENSE_RESEARCH_ONLY")
    if item.authority_role == "SILVER":
        blockers.append("SILVER_HAS_NO_RELEASE_AUTHORITY")
    elif item.authority_role == "CHALLENGE":
        blockers.append("CHALLENGE_CANNOT_APPROVE_NATURAL_SPEECH_RELEASE")
    if item.annotation_origin not in {"HUMAN", "PUBLISHER_HUMAN"}:
        blockers.append("REFERENCE_NOT_HUMAN_GOLD")
    if item.continuous_timeline != "VERIFIED":
        blockers.append("CONTINUOUS_TIMELINE_UNVERIFIED")
    if item.audit_status != "VERIFIED" or item.audit_sha256 is None:
        blockers.append("REFERENCE_AUDIT_UNVERIFIED")
    if item.speaker_independence != "VERIFIED":
        blockers.append("SPEAKER_SPLIT_INDEPENDENCE_UNVERIFIED")
    if not item.reference_capabilities.diarization:
        blockers.append("DIARIZATION_REFERENCE_UNAVAILABLE")
    if not item.reference_capabilities.overlap:
        blockers.append("OVERLAP_REFERENCE_UNAVAILABLE")
    if not item.reference_capabilities.scd:
        blockers.append("SCD_REFERENCE_UNAVAILABLE")
    if split != "RELEASE_HOLDOUT":
        blockers.append("NOT_RELEASE_HOLDOUT")
    if not item.release_holdout_locked:
        blockers.append("RELEASE_HOLDOUT_NOT_LOCKED")
    reasons.extend(blockers)
    metric_eligible = not blockers and item.authority_role == "GOLD"
    if metric_eligible:
        status = "REVIEW_REQUIRED"
    elif item.license_status in {"PROHIBITED", "UNVERIFIED"} or item.continuous_timeline == "ABSENT":
        status = "DATA_INELIGIBLE"
    elif item.license_status == "RESEARCH_ONLY":
        status = "RESEARCH_ONLY"
    elif item.authority_role == "SILVER":
        status = "DEV_ONLY"
    elif item.authority_role == "CHALLENGE":
        status = "CHALLENGE_ONLY"
    else:
        status = "REVIEW_REQUIRED"
    return BenchmarkEligibility(
        status=status,
        split=split,
        eligible_for_metric_gating=metric_eligible,
        eligible_for_release_scoring=False,
        reason_codes=tuple(sorted(set(reasons))),
    )


@dataclass(frozen=True, slots=True)
class KoreanReleasePolicy:
    der_max: float = 0.15
    jer_max: float = 0.25
    nonoverlap_speech_coverage_min: float = 0.85
    assigned_accuracy_min: float = 0.95
    false_h2_duration_ratio_max: float = 0.01
    scd_f1_min: float = 0.75
    osd_precision_min: float = 0.75
    osd_recall_min: float = 0.60
    subgroup_der_gap_max: float = 0.05
    minimum_recordings: int = 20
    subgroup_minimum_recordings: int = 2
    minimum_h1_files: int = 2
    minimum_h2_files: int = 8
    minimum_scd_reference_events: int = 5
    minimum_overlap_reference_us: int = 1_000_000
    schema_version: str = RELEASE_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_POLICY_SCHEMA:
            raise KoreanBenchmarkError("unsupported Korean release policy schema")
        floats = (
            self.der_max, self.jer_max, self.nonoverlap_speech_coverage_min, self.assigned_accuracy_min,
            self.false_h2_duration_ratio_max, self.scd_f1_min, self.osd_precision_min,
            self.osd_recall_min, self.subgroup_der_gap_max,
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in floats):
            raise KoreanBenchmarkError("release policy ratios must be finite values in [0, 1]")
        if any(type(value) is not int or value <= 0 for value in (
            self.minimum_recordings, self.subgroup_minimum_recordings,
            self.minimum_h1_files, self.minimum_h2_files,
            self.minimum_scd_reference_events, self.minimum_overlap_reference_us,
        )):
            raise KoreanBenchmarkError("release policy sample floors must be positive integers")

    @property
    def policy_sha256(self) -> str:
        return sha256(_canonical_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class KoreanReleaseGateReport:
    status: str
    release_authority: str
    corpus_lock_sha256: str
    policy_sha256: str
    recording_count: int
    metrics: Mapping[str, float | int | None]
    gate_results: Mapping[str, bool | None]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in _METRIC_GATE_STATUSES or self.release_authority != "none":
            raise KoreanBenchmarkError("Korean metric gate cannot grant release authority")
        if type(self.recording_count) is not int or self.recording_count < 0:
            raise KoreanBenchmarkError("invalid Korean metric gate recording count")
        _digest(self.corpus_lock_sha256, "corpus lock")
        _digest(self.policy_sha256, "metric policy")
        if (not isinstance(self.reason_codes, tuple)
                or _EXTERNAL_AUTHORITY_REASON not in self.reason_codes):
            raise KoreanBenchmarkError("external release authority reason is required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "release_authority": self.release_authority,
            "corpus_lock_sha256": self.corpus_lock_sha256,
            "policy_sha256": self.policy_sha256,
            "recording_count": self.recording_count,
            "metrics": dict(self.metrics),
            "gate_results": dict(self.gate_results),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class KoreanSuiteGateReport:
    status: str
    release_authority: str
    gold_status: str
    required_challenges: tuple[str, ...]
    challenge_results: Mapping[str, bool]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in _SUITE_GATE_STATUSES or self.release_authority != "none":
            raise KoreanBenchmarkError("Korean suite gate cannot grant release authority")
        if self.gold_status not in _METRIC_GATE_STATUSES:
            raise KoreanBenchmarkError("invalid Korean suite gold status")
        if (not isinstance(self.reason_codes, tuple)
                or _EXTERNAL_AUTHORITY_REASON not in self.reason_codes):
            raise KoreanBenchmarkError("external release authority reason is required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "release_authority": self.release_authority,
            "gold_status": self.gold_status,
            "required_challenges": list(self.required_challenges),
            "challenge_results": dict(self.challenge_results),
            "reason_codes": list(self.reason_codes),
        }


def evaluate_korean_release_suite(
    gold_gate: KoreanReleaseGateReport,
    *,
    challenge_results: Mapping[str, bool],
    required_challenges: tuple[str, ...],
) -> KoreanSuiteGateReport:
    if not isinstance(gold_gate, KoreanReleaseGateReport):
        raise KoreanBenchmarkError("suite gate requires a Korean gold gate report")
    required = tuple(sorted(set(required_challenges)))
    if (not required or len(required) != len(required_challenges)
            or any(_OPAQUE.fullmatch(item) is None for item in required)):
        raise KoreanBenchmarkError("required challenge identifiers are invalid")
    if not isinstance(challenge_results, Mapping) or any(
        not isinstance(key, str) or _OPAQUE.fullmatch(key) is None or type(value) is not bool
        for key, value in challenge_results.items()
    ):
        raise KoreanBenchmarkError("challenge results are invalid")
    reasons: list[str] = [_EXTERNAL_AUTHORITY_REASON]
    missing = [name for name in required if name not in challenge_results]
    failed = [name for name in required if challenge_results.get(name) is False]
    gold_passed = gold_gate.status == "METRIC_GATES_PASS_REVIEW_REQUIRED"
    if not gold_passed:
        reasons.append("GOLD_METRIC_GATES_NOT_PASSED")
    if missing:
        reasons.append("REQUIRED_CHALLENGE_RESULTS_MISSING")
    if failed:
        reasons.append("REQUIRED_CHALLENGE_FAILED")
    if missing:
        status = "REVIEW_REQUIRED"
    elif not gold_passed or failed:
        status = "SUITE_METRICS_FAIL"
    else:
        status = "SUITE_METRICS_PASS_REVIEW_REQUIRED"
    selected = {name: challenge_results[name] for name in required if name in challenge_results}
    return KoreanSuiteGateReport(
        status, "none", gold_gate.status, required, selected, tuple(sorted(set(reasons)))
    )


def evaluate_korean_release_gate(
    report: EvaluationReport,
    *,
    lock: KoreanCorpusLock | VerifiedKoreanCorpusLock,
    split: str,
    policy: KoreanReleasePolicy | None = None,
) -> KoreanReleaseGateReport:
    if not isinstance(report, EvaluationReport):
        raise KoreanBenchmarkError("invalid Korean release gate input")
    item = _unwrapped_lock(lock)
    cfg = policy or KoreanReleasePolicy()
    eligibility = evaluate_benchmark_eligibility(lock, split=split)
    scores = report.recordings
    reference_us = sum(item.reference_duration_us for score in scores for item in score.speakers)
    assigned_us = sum(item.assigned_duration_us for score in scores for item in score.speakers)
    correct_us = sum(item.correct_duration_us for score in scores for item in score.speakers)
    coverage = assigned_us / reference_us if reference_us else 0.0
    assigned_accuracy = correct_us / assigned_us if assigned_us else 0.0
    scd_f1 = (report.overall.scd.f1
              if item.reference_capabilities.scd and report.overall.scd.evaluated else None)
    osd_precision = (report.overall.overlap.precision
                     if item.reference_capabilities.overlap and report.overall.overlap.evaluated else None)
    osd_recall = (report.overall.overlap.recall
                  if item.reference_capabilities.overlap and report.overall.overlap.evaluated else None)
    subgroup_gap = max((subgroup.aggregate.diarization_all.der - report.overall.diarization_all.der
                        for subgroup in report.subgroups
                        if subgroup.aggregate.recording_count >= cfg.subgroup_minimum_recordings), default=0.0)
    metrics: dict[str, float | int | None] = {
        "der": report.overall.diarization_all.der,
        "jer": report.overall.jer,
        "nonoverlap_speech_coverage": coverage,
        "assigned_accuracy": assigned_accuracy,
        "complete_merge_files": report.overall.complete_merge_files,
        "false_h2_duration_ratio": report.overall.false_h2_duration_ratio,
        "scd_f1": scd_f1,
        "osd_precision": osd_precision,
        "osd_recall": osd_recall,
        "worst_speaker_coverage": report.overall.worst_speaker_coverage,
        "worst_speaker_assigned_accuracy": report.overall.worst_speaker_assigned_accuracy,
        "worst_subgroup_der_gap": subgroup_gap,
        "eligible_h1_files": report.overall.eligible_h1_files,
        "eligible_h2_files": report.overall.eligible_h2_files,
        "scd_reference_events": report.overall.scd.true_positives + report.overall.scd.false_negatives,
        "reference_overlap_us": report.overall.overlap.reference_overlap_us,
    }
    gates: dict[str, bool | None] = {
        "minimum_recordings": len(scores) >= cfg.minimum_recordings,
        "minimum_h1_files": report.overall.eligible_h1_files >= cfg.minimum_h1_files,
        "minimum_h2_files": report.overall.eligible_h2_files >= cfg.minimum_h2_files,
        "minimum_scd_reference_events": (
            report.overall.scd.true_positives + report.overall.scd.false_negatives
        ) >= cfg.minimum_scd_reference_events,
        "minimum_overlap_reference_us": (
            report.overall.overlap.reference_overlap_us >= cfg.minimum_overlap_reference_us
        ),
        "der": report.overall.diarization_all.der <= cfg.der_max,
        "jer": report.overall.jer <= cfg.jer_max,
        "nonoverlap_speech_coverage": coverage >= cfg.nonoverlap_speech_coverage_min,
        "assigned_accuracy": assigned_accuracy >= cfg.assigned_accuracy_min,
        "complete_merge_zero": report.overall.complete_merge_files == 0,
        "false_h2_duration_ratio": report.overall.false_h2_duration_ratio <= cfg.false_h2_duration_ratio_max,
        "scd_f1": None if scd_f1 is None else scd_f1 >= cfg.scd_f1_min,
        "osd_precision": None if osd_precision is None else osd_precision >= cfg.osd_precision_min,
        "osd_recall": None if osd_recall is None else osd_recall >= cfg.osd_recall_min,
        "subgroup_der_gap": subgroup_gap <= cfg.subgroup_der_gap_max,
    }
    reasons = list(eligibility.reason_codes)
    if not gates["minimum_recordings"]:
        reasons.append("RECORDING_SAMPLE_FLOOR_NOT_MET")
    for name, value in gates.items():
        if value is False:
            reasons.append("GATE_FAILED_" + name.upper())
        elif value is None:
            reasons.append("GATE_NOT_EVALUATED_" + name.upper())
    if not eligibility.eligible_for_metric_gating:
        status = eligibility.status
    elif any(value is False for value in gates.values()):
        status = "METRIC_GATES_FAIL"
    elif any(value is None for value in gates.values()):
        status = "METRIC_GATES_INCOMPLETE_REVIEW_REQUIRED"
    else:
        status = "METRIC_GATES_PASS_REVIEW_REQUIRED"
    return KoreanReleaseGateReport(
        status=status, release_authority="none", corpus_lock_sha256=item.lock_sha256,
        policy_sha256=cfg.policy_sha256, recording_count=len(scores), metrics=metrics,
        gate_results=gates, reason_codes=tuple(sorted(set(reasons))),
    )


__all__ = [
    "CORPUS_LOCK_SCHEMA", "RELEASE_POLICY_SCHEMA", "BenchmarkEligibility",
    "KoreanBenchmarkError", "KoreanCorpusLock", "KoreanReleaseGateReport",
    "KoreanReleasePolicy", "KoreanSuiteGateReport", "ReferenceCapabilities",
    "evaluate_benchmark_eligibility", "evaluate_korean_release_suite",
    "evaluate_korean_release_gate", "parse_korean_corpus_lock", "CorpusLockSignatureVerifier",
    "VerifiedKoreanCorpusLock", "verify_korean_corpus_lock",
]
