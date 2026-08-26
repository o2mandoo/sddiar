"""Bounded, lineage-aware, shadow-only SCD boundary evidence.

This module deliberately stops one capability short of segmentation.  A model
candidate and independent acoustic probes can produce a scalar receipt and a
``SHADOW_APPROVE_CANDIDATE`` decision, but no event that can alter tracklets.
Enforcement belongs to the release-authorized factory in ``segmentation``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

from .errors import ContractValidationError


_MAX_ID_LENGTH = 256
_MAX_RECEIPT_TEXT = 128


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > _MAX_ID_LENGTH:
        raise ContractValidationError(f"{name} must be a bounded non-empty string")
    if any(ord(char) < 32 for char in value):
        raise ContractValidationError(f"{name} contains a control character")
    return value


def _time(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractValidationError(f"{name} must be a non-negative integer")
    return value


def _finite_range(value: Any, name: str, low: float = 0.0, high: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < low or result > high:
        raise ContractValidationError(f"{name} must be finite in [{low},{high}]")
    return result


@dataclass(frozen=True, slots=True)
class BoundaryGateConfig:
    """Conservative bounds and thresholds for the opt-in shadow experiment."""

    enabled: bool = False
    model_min_score: float = 0.5
    probe_min_quality: float = 0.6
    probe_min_clean_speech_us: int = 200_000
    min_independent_subprobes: int = 2
    max_probes_per_side: int = 8
    embedding_dimension: int = 256
    max_boundary_gap_us: int = 1_000_000
    within_radius: float = 0.25
    cross_distance_min: float = 0.35
    separation_margin_min: float = 0.05
    config_identity: str = "boundary-gate-v1"
    config_hash: str | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ContractValidationError("enabled must be bool")
        _finite_range(self.model_min_score, "model_min_score")
        _finite_range(self.probe_min_quality, "probe_min_quality")
        _finite_range(self.within_radius, "within_radius", 0.0, 2.0)
        _finite_range(self.cross_distance_min, "cross_distance_min", 0.0, 2.0)
        _finite_range(self.separation_margin_min, "separation_margin_min", 0.0, 2.0)
        if type(self.probe_min_clean_speech_us) is not int or self.probe_min_clean_speech_us <= 0:
            raise ContractValidationError("probe_min_clean_speech_us must be positive")
        if type(self.min_independent_subprobes) is not int or self.min_independent_subprobes < 2:
            raise ContractValidationError("at least two independent subprobes are required")
        if type(self.max_probes_per_side) is not int or self.max_probes_per_side < self.min_independent_subprobes:
            raise ContractValidationError("max_probes_per_side is too small")
        if type(self.embedding_dimension) is not int or self.embedding_dimension <= 0 or self.embedding_dimension > 4096:
            raise ContractValidationError("embedding_dimension is outside the resource bound")
        if type(self.max_boundary_gap_us) is not int or self.max_boundary_gap_us < 0:
            raise ContractValidationError("max_boundary_gap_us must be non-negative")
        if self.config_hash is not None:
            _identifier(self.config_hash, "config_hash")
            if self.config_identity != "boundary-gate-v1" and self.config_identity != self.config_hash:
                raise ContractValidationError("config_identity and config_hash disagree")
            object.__setattr__(self, "config_identity", self.config_hash)
        _identifier(self.config_identity, "config_identity")


@dataclass(frozen=True, slots=True)
class BoundaryModelCandidate:
    """Typed model SCD candidate with immutable source lineage."""

    time_us: int
    score: float
    candidate_id: str
    source_id: str = ""
    view_id: str = ""
    model_identity: str = ""
    config_identity: str = ""
    overlap_veto: bool = False
    model_id: str | None = None
    config_hash: str | None = None

    def __post_init__(self) -> None:
        _time(self.time_us, "time_us")
        _finite_range(self.score, "score")
        if self.model_id is not None:
            _identifier(self.model_id, "model_id")
            if self.model_identity and self.model_identity != self.model_id:
                raise ContractValidationError("model_identity and model_id disagree")
            object.__setattr__(self, "model_identity", self.model_id)
        if self.config_hash is not None:
            _identifier(self.config_hash, "config_hash")
            if self.config_identity and self.config_identity != self.config_hash:
                raise ContractValidationError("config_identity and config_hash disagree")
            object.__setattr__(self, "config_identity", self.config_hash)
        for value, name in (
            (self.candidate_id, "candidate_id"),
            (self.source_id, "source_id"),
            (self.view_id, "view_id"),
            (self.model_identity, "model_identity"),
            (self.config_identity, "config_identity"),
        ):
            _identifier(value, name)
        if self.model_id is None:
            object.__setattr__(self, "model_id", self.model_identity)
        if type(self.overlap_veto) is not bool:
            raise ContractValidationError("overlap_veto must be bool")


@dataclass(frozen=True, slots=True)
class BoundaryProbe:
    """Typed embedding probe; independent block and time lineage are required."""

    probe_id: str
    vector: tuple[float, ...]
    quality: float
    clean_speech_us: int
    source_id: str = ""
    view_id: str = ""
    embedding_identity: str = ""
    independent_block_id: str = ""
    start_us: int = 0
    end_us: int = 0
    config_identity: str = ""
    overlap_veto: bool = False
    embedding_id: str | None = None
    config_hash: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.probe_id, "probe_id")
        if self.embedding_id is not None:
            _identifier(self.embedding_id, "embedding_id")
            if self.embedding_identity and self.embedding_identity != self.embedding_id:
                raise ContractValidationError("embedding_identity and embedding_id disagree")
            object.__setattr__(self, "embedding_identity", self.embedding_id)
        if self.config_hash is not None:
            _identifier(self.config_hash, "config_hash")
            if self.config_identity and self.config_identity != self.config_hash:
                raise ContractValidationError("config_identity and config_hash disagree")
            object.__setattr__(self, "config_identity", self.config_hash)
        if type(self.vector) is not tuple or not self.vector or len(self.vector) > 4096:
            raise ContractValidationError("vector must be a bounded tuple")
        for value in self.vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ContractValidationError("vector values must be finite numbers")
        _finite_range(self.quality, "quality")
        if type(self.clean_speech_us) is not int or self.clean_speech_us < 0:
            raise ContractValidationError("clean_speech_us must be a non-negative integer")
        _time(self.start_us, "start_us")
        _time(self.end_us, "end_us")
        if self.end_us <= self.start_us:
            raise ContractValidationError("probe interval must be non-empty")
        if self.clean_speech_us > self.end_us - self.start_us:
            raise ContractValidationError("clean_speech_us cannot exceed probe duration")
        for value, name in (
            (self.source_id, "source_id"),
            (self.view_id, "view_id"),
            (self.embedding_identity, "embedding_identity"),
            (self.independent_block_id, "independent_block_id"),
            (self.config_identity, "config_identity"),
        ):
            _identifier(value, name)
        if self.embedding_id is None:
            object.__setattr__(self, "embedding_id", self.embedding_identity)
        if type(self.overlap_veto) is not bool:
            raise ContractValidationError("overlap_veto must be bool")

    @property
    def embedding_identity_value(self) -> str:
        return self.embedding_identity


@dataclass(frozen=True, slots=True)
class BoundaryOverlapVeto:
    """Typed overlap interval supplied as a boundary veto."""

    overlap_id: str
    source_id: str
    view_id: str
    config_identity: str
    start_us: int
    end_us: int
    overlap_evidence: float = 1.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.overlap_id, "overlap_id"),
            (self.source_id, "source_id"),
            (self.view_id, "view_id"),
            (self.config_identity, "config_identity"),
        ):
            _identifier(value, name)
        _time(self.start_us, "start_us")
        _time(self.end_us, "end_us")
        if self.end_us <= self.start_us:
            raise ContractValidationError("overlap interval must be non-empty")
        _finite_range(self.overlap_evidence, "overlap_evidence")


@dataclass(frozen=True, slots=True)
class BoundaryReceipt:
    """Bounded scalar-only receipt; all lineage values are truncated hashes."""

    candidate_key: str
    source_key: str
    view_key: str
    config_key: str
    model_key: str
    embedding_key: str
    left_probe_count: int
    right_probe_count: int
    left_independent_count: int
    right_independent_count: int
    model_score: float | None
    left_within_radius: float | None
    right_within_radius: float | None
    cross_distance: float | None
    separation_margin_min: float


@dataclass(frozen=True, slots=True)
class BoundaryGateResult:
    decision: str
    reason_codes: tuple[str, ...]
    candidate_time_us: int | None
    receipt: BoundaryReceipt
    approved_event: None = None

    @property
    def shadow_approved(self) -> bool:
        return self.decision == "SHADOW_APPROVE_CANDIDATE"

    @property
    def reason(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""

    @property
    def event(self) -> None:
        return None


def _scalar_hash(value: object) -> str:
    raw = str(value).encode("utf-8", "replace")[:_MAX_RECEIPT_TEXT]
    return hashlib.sha256(raw).hexdigest()[:16]


def _receipt(
    candidate: BoundaryModelCandidate,
    *,
    probes: Sequence[BoundaryProbe] = (),
    model_score: float | None = None,
    left_radius: float | None = None,
    right_radius: float | None = None,
    cross: float | None = None,
    margin: float = 0.0,
) -> BoundaryReceipt:
    identities = {probe.embedding_identity for probe in probes}
    embedding = next(iter(identities), "") if len(identities) == 1 else "mixed"
    return BoundaryReceipt(
        candidate_key=_scalar_hash(candidate.candidate_id),
        source_key=_scalar_hash(candidate.source_id),
        view_key=_scalar_hash(candidate.view_id),
        config_key=_scalar_hash(candidate.config_identity),
        model_key=_scalar_hash(candidate.model_identity),
        embedding_key=_scalar_hash(embedding),
        left_probe_count=0,
        right_probe_count=0,
        left_independent_count=0,
        right_independent_count=0,
        model_score=model_score,
        left_within_radius=left_radius,
        right_within_radius=right_radius,
        cross_distance=cross,
        separation_margin_min=margin,
    )


def _reject(candidate: Any, reason: str, *, time_us: int | None = None) -> BoundaryGateResult:
    if type(candidate) is BoundaryModelCandidate:
        receipt = _receipt(candidate)
    else:
        placeholder = _scalar_hash("invalid")
        receipt = BoundaryReceipt(placeholder, placeholder, placeholder, placeholder, placeholder, placeholder, 0, 0, 0, 0, None, None, None, None, 0.0)
    return BoundaryGateResult("REJECT", (reason,), time_us, receipt)


def _bounded(values: Any, limit: int) -> tuple[Any, ...] | None:
    if isinstance(values, (str, bytes, Mapping)):
        return None
    try:
        count = len(values)
    except (TypeError, AttributeError):
        return None
    if count > limit:
        return None
    return tuple(values)


def _unit(vector: tuple[float, ...], dimension: int) -> tuple[float, ...] | None:
    if len(vector) != dimension:
        return None
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return tuple(float(value) / norm for value in vector)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    value = 1.0 - sum(a * b for a, b in zip(left, right))
    return max(0.0, min(2.0, value))


def _centroid(vectors: Sequence[tuple[float, ...]]) -> tuple[float, ...] | None:
    if not vectors:
        return None
    dimension = len(vectors[0])
    sums = [sum(vector[index] for vector in vectors) for index in range(dimension)]
    norm = math.sqrt(sum(value * value for value in sums))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return tuple(value / norm for value in sums)


class DualEvidenceBoundaryGate:
    """Evaluate model+embedding evidence without ever enforcing a split."""

    def __init__(self, config: BoundaryGateConfig | None = None) -> None:
        self.config = config or BoundaryGateConfig()

    def evaluate(
        self,
        model_candidate: Any,
        left_probes: Sequence[Any] = (),
        right_probes: Sequence[Any] = (),
        *,
        overlap_veto: bool = False,
        overlap_regions: Sequence[Any] = (),
    ) -> BoundaryGateResult:
        cfg = self.config
        if not cfg.enabled:
            return _reject(model_candidate, "GATE_DISABLED")
        if type(model_candidate) is not BoundaryModelCandidate:
            return _reject(model_candidate, "INVALID_TYPED_MODEL_CANDIDATE")
        if type(overlap_veto) is not bool:
            return _reject(model_candidate, "MALFORMED_OVERLAP_VETO", time_us=model_candidate.time_us)
        if model_candidate.overlap_veto or overlap_veto:
            return _reject(model_candidate, "OVERLAP_VETO", time_us=model_candidate.time_us)
        if model_candidate.config_identity != cfg.config_identity:
            return _reject(model_candidate, "CONFIG_LINEAGE_MISMATCH", time_us=model_candidate.time_us)

        left = _bounded(left_probes, cfg.max_probes_per_side)
        right = _bounded(right_probes, cfg.max_probes_per_side)
        if left is None or right is None:
            return _reject(model_candidate, "RESOURCE_BOUND_EXCEEDED", time_us=model_candidate.time_us)
        if len(left) < cfg.min_independent_subprobes or len(right) < cfg.min_independent_subprobes:
            return _reject(model_candidate, "INSUFFICIENT_INDEPENDENT_SUBPROBES", time_us=model_candidate.time_us)
        if any(type(probe) is not BoundaryProbe for probe in (*left, *right)):
            return _reject(model_candidate, "INVALID_TYPED_PROBE", time_us=model_candidate.time_us)

        typed_left = tuple(left)
        typed_right = tuple(right)
        all_probes = typed_left + typed_right
        if any(
            probe.source_id != model_candidate.source_id
            or probe.view_id != model_candidate.view_id
            or probe.config_identity != model_candidate.config_identity
            for probe in all_probes
        ):
            return _reject(model_candidate, "SOURCE_VIEW_CONFIG_LINEAGE_MISMATCH", time_us=model_candidate.time_us)
        embeddings = {probe.embedding_identity for probe in all_probes}
        if len(embeddings) != 1:
            return _reject(model_candidate, "EMBEDDING_LINEAGE_MISMATCH", time_us=model_candidate.time_us)
        if model_candidate.score < cfg.model_min_score:
            return _reject(model_candidate, "MODEL_EVIDENCE_BELOW_THRESHOLD", time_us=model_candidate.time_us)

        vetoes = _bounded(overlap_regions, cfg.max_probes_per_side * 2)
        if vetoes is None or any(type(item) is not BoundaryOverlapVeto for item in vetoes):
            return _reject(model_candidate, "MALFORMED_OVERLAP_VETO", time_us=model_candidate.time_us)
        for veto in vetoes:
            if (
                veto.source_id != model_candidate.source_id
                or veto.view_id != model_candidate.view_id
                or veto.config_identity != model_candidate.config_identity
            ):
                return _reject(model_candidate, "OVERLAP_LINEAGE_MISMATCH", time_us=model_candidate.time_us)
            if veto.start_us <= model_candidate.time_us < veto.end_us:
                return _reject(model_candidate, "OVERLAP_VETO", time_us=model_candidate.time_us)

        left_blocks = {probe.independent_block_id for probe in typed_left}
        right_blocks = {probe.independent_block_id for probe in typed_right}
        if len(left_blocks) < cfg.min_independent_subprobes or len(right_blocks) < cfg.min_independent_subprobes:
            return _reject(model_candidate, "INSUFFICIENT_INDEPENDENT_SUBPROBES", time_us=model_candidate.time_us)
        if left_blocks & right_blocks:
            return _reject(model_candidate, "INDEPENDENT_BLOCKS_NOT_DISJOINT", time_us=model_candidate.time_us)
        if any(probe.overlap_veto for probe in all_probes):
            return _reject(model_candidate, "OVERLAP_VETO", time_us=model_candidate.time_us)
        if any(
            probe.quality < cfg.probe_min_quality
            or probe.clean_speech_us < cfg.probe_min_clean_speech_us
            for probe in all_probes
        ):
            return _reject(model_candidate, "LOW_QUALITY_OR_CLEAN_DURATION", time_us=model_candidate.time_us)

        if any(probe.end_us > model_candidate.time_us for probe in typed_left) or any(
            probe.start_us < model_candidate.time_us for probe in typed_right
        ):
            return _reject(model_candidate, "PROBE_NOT_ON_EXPECTED_SIDE", time_us=model_candidate.time_us)
        left_gap = model_candidate.time_us - max(probe.end_us for probe in typed_left)
        right_gap = min(probe.start_us for probe in typed_right) - model_candidate.time_us
        if left_gap > cfg.max_boundary_gap_us or right_gap > cfg.max_boundary_gap_us:
            return _reject(model_candidate, "BOUNDARY_GAP_EXCEEDED", time_us=model_candidate.time_us)

        vectors_left: list[tuple[float, ...]] = []
        vectors_right: list[tuple[float, ...]] = []
        for probe in typed_left:
            vector = _unit(probe.vector, cfg.embedding_dimension)
            if vector is None:
                return _reject(model_candidate, "VECTOR_DIMENSION_OR_NORM_INVALID", time_us=model_candidate.time_us)
            vectors_left.append(vector)
        for probe in typed_right:
            vector = _unit(probe.vector, cfg.embedding_dimension)
            if vector is None:
                return _reject(model_candidate, "VECTOR_DIMENSION_OR_NORM_INVALID", time_us=model_candidate.time_us)
            vectors_right.append(vector)
        left_center, right_center = _centroid(vectors_left), _centroid(vectors_right)
        if left_center is None or right_center is None:
            return _reject(model_candidate, "INVALID_PROBE_GEOMETRY", time_us=model_candidate.time_us)
        left_radius = max(_distance(vector, left_center) for vector in vectors_left)
        right_radius = max(_distance(vector, right_center) for vector in vectors_right)
        cross = _distance(left_center, right_center)
        base = _receipt(
            model_candidate,
            probes=all_probes,
            model_score=round(model_candidate.score, 6),
            left_radius=round(left_radius, 6),
            right_radius=round(right_radius, 6),
            cross=round(cross, 6),
            margin=round(cfg.separation_margin_min, 6),
        )
        receipt = BoundaryReceipt(
            base.candidate_key, base.source_key, base.view_key, base.config_key,
            base.model_key, base.embedding_key, len(typed_left), len(typed_right),
            len(left_blocks), len(right_blocks), base.model_score,
            base.left_within_radius, base.right_within_radius, base.cross_distance,
            base.separation_margin_min,
        )
        if left_radius > cfg.within_radius or right_radius > cfg.within_radius:
            return BoundaryGateResult("REJECT", ("PROBES_NOT_WITHIN_RADIUS",), model_candidate.time_us, receipt)
        if cross < cfg.cross_distance_min:
            return BoundaryGateResult("REJECT", ("CROSS_DISTANCE_BELOW_THRESHOLD",), model_candidate.time_us, receipt)
        if not cross > left_radius + right_radius + cfg.separation_margin_min:
            return BoundaryGateResult("REJECT", ("SEPARATION_MARGIN_NOT_MET",), model_candidate.time_us, receipt)
        return BoundaryGateResult("SHADOW_APPROVE_CANDIDATE", (), model_candidate.time_us, receipt)

    decide = evaluate
    gate = evaluate

    def build_enforceable_event(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("release-trusted segmentation factory is required before SCD enforcement")


DualEvidenceBoundaryGateConfig = BoundaryGateConfig
ModelSCDCandidate = BoundaryModelCandidate
OverlapVetoInterval = BoundaryOverlapVeto
