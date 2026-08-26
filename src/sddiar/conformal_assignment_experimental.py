"""Default-off Block-Mondrian Relative Conformal Margin (BM-RCM) v2.

The module is an intentionally isolated experiment.  It is not imported by
the production diarizer and cannot mutate a speaker label.  It accepts
independent anchor *blocks* and one or more candidate blocks, then emits only
scalar scores and hashes.

The v2 core has four properties which are easy to lose in a small prototype:

* every vector is put on a deterministic fixed-point unit sphere before it is
  used; centroids are spherical too and scoring uses relative cosine distance;
* fragments with the same ``(speaker_id, block_id)`` are aggregated into one
  spherical prototype and one duration/quality-derived capped weight;
* class totals are prepared once, so every leave-one-out centroid is a
  subtraction from a total (O(MD), rather than rebuilding M centroids); and
* malformed, mixed, overlap, under-supported, or resource-heavy input
  abstains.  ``enabled=False`` remains the default.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "bm-rcm-v2"
ALGORITHM_VERSION = "bm-rcm-v2-fixedpoint-spherical-relative-cosine"
DECISIONS = frozenset({"DISABLED", "SHADOW_SINGLETON", "OOD", "AMBIGUOUS", "FAIL_CLOSED"})

# These are deliberately module constants rather than user-tunable knobs.
# They are the resource ceiling of this experimental core.
MAX_ANCHOR_BLOCKS = 1_024
MAX_DIMENSION = 512
MAX_OVERLAP_FLAGS = 32
MAX_BATCH_COORDINATE_WORK = 1_048_576
# Compatibility alias. This is a per-call batch ceiling, not a whole-job
# budget; integration layers must declare and enforce their own cumulative cap.
MAX_TOTAL_COORDINATE_WORK = MAX_BATCH_COORDINATE_WORK
MAX_BATCH_CANDIDATES = 1_024
MAX_TOTAL_SPEAKER_DURATION_US = 4 * 60 * 60 * 1_000_000
MAX_BLOCK_DURATION_US = MAX_TOTAL_SPEAKER_DURATION_US
DEFAULT_DURATION_CAP_US = 10 * 1_000_000
DEFAULT_MIN_CLEAN_DURATION_US = 1_500_000

_VECTOR_SCALE = 1_000_000
_QUALITY_SCALE = 1_000_000
_MAX_FRACTION_DENOMINATOR = 1_000_000_000


class ConformalAssignmentError(ValueError):
    """Base error for malformed BM-RCM policy or input."""


class ConformalAssignmentResourceError(ConformalAssignmentError):
    """A hard block, dimension, flag, duration, or coordinate bound failed."""


@dataclass(frozen=True, slots=True)
class AnchorBlock:
    """One anchor fragment.

    ``valid``, ``quality``, ``overlap``, ``mixed``, and
    ``clean_duration_us`` intentionally default to ``None`` only so a missing
    field can be represented and fail closed.  A valid production fixture must
    provide all of them explicitly; arbitrary caller-supplied weights are not
    accepted in v2.
    """

    vector: Sequence[float]
    block_id: str
    speaker_id: str
    valid: bool | None = None
    quality: float | int | Fraction | None = None
    overlap: bool | None = None
    mixed: bool | None = None
    clean_duration_us: int | None = None
    overlap_flags: Sequence[Any] | None = None


@dataclass(frozen=True, slots=True)
class CandidateBlock:
    """One unlabelled candidate block; it has the same safety contract."""

    vector: Sequence[float]
    block_id: str
    valid: bool | None = None
    quality: float | int | Fraction | None = None
    overlap: bool | None = None
    mixed: bool | None = None
    clean_duration_us: int | None = None
    overlap_flags: Sequence[Any] | None = None


@dataclass(frozen=True, slots=True)
class ConformalAssignmentConfig:
    """Bounded policy.  Resource ceilings cannot be raised by configuration."""

    enabled: bool = False
    epsilon: float | int | Fraction = Fraction(1, 10)
    min_blocks: int | None = None
    min_quality: float | int | Fraction = Fraction(1, 2)
    min_clean_duration_us: int = DEFAULT_MIN_CLEAN_DURATION_US
    duration_cap_us: int = DEFAULT_DURATION_CAP_US

    # Kept as compatibility-visible fields, but they may only narrow the
    # hard, non-configurable ceilings above.
    max_anchor_blocks: int = MAX_ANCHOR_BLOCKS
    max_dimension: int = MAX_DIMENSION
    vector_scale: int = _VECTOR_SCALE

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ConformalAssignmentError("enabled must be a boolean")
        if type(self.max_anchor_blocks) is not int or not 1 <= self.max_anchor_blocks <= MAX_ANCHOR_BLOCKS:
            raise ConformalAssignmentError("max_anchor_blocks exceeds the hard ceiling")
        if type(self.max_dimension) is not int or not 1 <= self.max_dimension <= MAX_DIMENSION:
            raise ConformalAssignmentError("max_dimension exceeds the hard ceiling")
        if type(self.vector_scale) is not int or self.vector_scale != _VECTOR_SCALE:
            raise ConformalAssignmentError("vector_scale is fixed for cross-backend determinism")
        if isinstance(self.min_clean_duration_us, bool) or not isinstance(self.min_clean_duration_us, int) or self.min_clean_duration_us <= 0:
            raise ConformalAssignmentError("min_clean_duration_us must be a positive integer")
        if isinstance(self.duration_cap_us, bool) or not isinstance(self.duration_cap_us, int) or self.duration_cap_us <= 0:
            raise ConformalAssignmentError("duration_cap_us must be a positive integer")
        if self.duration_cap_us > MAX_BLOCK_DURATION_US:
            raise ConformalAssignmentError("duration_cap_us exceeds the hard ceiling")
        epsilon = _finite_fraction(self.epsilon, "epsilon")
        if not 0 < epsilon < 1:
            raise ConformalAssignmentError("epsilon must be finite and in (0, 1)")
        quality = _finite_fraction(self.min_quality, "min_quality")
        if not 0 < quality <= 1:
            raise ConformalAssignmentError("min_quality must be finite and in (0, 1]")
        required = (epsilon.denominator + epsilon.numerator - 1) // epsilon.numerator - 1
        required = max(1, required)
        if self.min_blocks is not None:
            if type(self.min_blocks) is not int or self.min_blocks < required:
                raise ConformalAssignmentError("min_blocks must be at least ceil(1/epsilon)-1")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "min_quality", quality)
        object.__setattr__(self, "min_blocks", required if self.min_blocks is None else self.min_blocks)


@dataclass(frozen=True, slots=True)
class HypothesisScore:
    """Scalar-only result for one speaker hypothesis."""

    speaker_id: str
    p_value: Fraction
    candidate_nonconformity: Fraction
    calibration_count: int
    eligible: bool


@dataclass(frozen=True, slots=True)
class ConformalAssignmentResult:
    """Immutable, non-authoritative result; no label mutation is possible."""

    decision: str
    gamma: tuple[str, ...]
    hypotheses: tuple[HypothesisScore, ...] = ()
    input_digest: str = ""
    receipt_hash: str = ""
    reason: str = ""

    @property
    def status(self) -> str:
        return self.decision

    @property
    def p_values(self) -> Mapping[str, Fraction]:
        return MappingProxyType({item.speaker_id: item.p_value for item in self.hypotheses})

    def receipt(self) -> Mapping[str, Any]:
        """Return only scalar fields and hashes; never vectors or raw IDs."""

        return MappingProxyType(
            {
                "schema": SCHEMA,
                "algorithm": ALGORITHM_VERSION,
                "decision": self.decision,
                "gamma_size": len(self.gamma),
                "hypothesis_count": len(self.hypotheses),
                "eligible_count": sum(item.eligible for item in self.hypotheses),
                "input_digest": self.input_digest,
                "receipt_hash": self.receipt_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class _PreparedBlock:
    vector: tuple[int, ...]
    weight: int
    block_id: str
    speaker_id: str
    duration_us: int


@dataclass(frozen=True, slots=True)
class _PreparedClass:
    speaker_id: str
    blocks: tuple[_PreparedBlock, ...]
    total_vector: tuple[int, ...]
    total_weight: int
    total_duration_us: int


@dataclass(frozen=True, slots=True)
class PreparedBMRCM:
    """Prepared calibration state for amortized candidate/batch scoring."""

    config: ConformalAssignmentConfig
    speakers: tuple[str, str]
    classes: tuple[_PreparedClass, _PreparedClass]
    calibration_digest: str
    coordinate_work: int

    def evaluate(self, candidate: Any) -> ConformalAssignmentResult:
        if not self.config.enabled:
            return _disabled_result()
        try:
            parsed = _parse_candidate(candidate)
            return _evaluate_prepared_candidate(self, parsed)
        except (ConformalAssignmentError, TypeError, ValueError, OverflowError):
            return _fail_closed("invalid_input")

    def evaluate_batch(self, candidates: Iterable[Any]) -> tuple[ConformalAssignmentResult, ...]:
        if not self.config.enabled:
            # Disabled mode does not inspect candidate fields, but preserves
            # the batch cardinality so callers can safely zip results.
            try:
                raw = tuple(islice(iter(candidates), MAX_BATCH_CANDIDATES + 1))
            except TypeError:
                return (_fail_closed("invalid_input"),)
            if len(raw) > MAX_BATCH_CANDIDATES:
                return tuple(_fail_closed("resource_bound") for _ in raw[:-1])
            return tuple(_disabled_result() for _ in raw)
        try:
            iterator = iter(candidates)
            raw = tuple(islice(iterator, MAX_BATCH_CANDIDATES + 1))
            if len(raw) > MAX_BATCH_CANDIDATES:
                raise ConformalAssignmentResourceError("candidate batch exceeds hard ceiling")
            block_count = sum(len(item.blocks) for item in self.classes)
            dimension = len(self.classes[0].blocks[0].vector)
            if self.coordinate_work + len(raw) * block_count * dimension > MAX_BATCH_COORDINATE_WORK:
                raise ConformalAssignmentResourceError("total coordinate work exceeds hard ceiling")
            return tuple(self.evaluate(item) for item in raw)
        except (ConformalAssignmentError, TypeError, ValueError, OverflowError):
            # A batch-level resource failure is fail-closed for every item.
            return tuple(_fail_closed("resource_bound") for _ in raw) if "raw" in locals() else ()


def _finite_fraction(value: Any, name: str) -> Fraction:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ConformalAssignmentError(f"{name} must be finite and rational")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, int):
        result = Fraction(value, 1)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ConformalAssignmentError(f"{name} must be finite and rational")
        result = Fraction(str(value)).limit_denominator(_MAX_FRACTION_DENOMINATOR)
    else:
        try:
            result = Fraction(str(value)).limit_denominator(_MAX_FRACTION_DENOMINATOR)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ConformalAssignmentError(f"{name} must be finite and rational") from exc
    if result.denominator > _MAX_FRACTION_DENOMINATOR:
        raise ConformalAssignmentError(f"{name} denominator exceeds bound")
    return result


def _fraction_number(value: Any, name: str) -> Fraction:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ConformalAssignmentError(f"{name} must be finite")
    try:
        if isinstance(value, float) and not math.isfinite(value):
            raise ConformalAssignmentError(f"{name} must be finite")
        result = Fraction(str(value)).limit_denominator(_MAX_FRACTION_DENOMINATOR)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ConformalAssignmentError(f"{name} must be finite") from exc
    return result


def _round_fraction(value: Fraction) -> int:
    if value.denominator == 1:
        return value.numerator
    if value >= 0:
        return (value.numerator + value.denominator // 2) // value.denominator
    return -((-value.numerator + value.denominator // 2) // value.denominator)


def _round_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ConformalAssignmentError("non-positive fixed-point denominator")
    return _round_fraction(Fraction(numerator, denominator))


def _mapping_get(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return default


def _parse_anchor(value: Any) -> AnchorBlock:
    if isinstance(value, AnchorBlock):
        return value
    if not isinstance(value, Mapping) and not hasattr(value, "vector"):
        raise ConformalAssignmentError("anchor block must be an object")
    return AnchorBlock(
        vector=_mapping_get(value, "vector", "prototype"),
        block_id=_mapping_get(value, "block_id", "block", "id"),
        speaker_id=_mapping_get(value, "speaker_id", "speaker"),
        valid=_mapping_get(value, "valid", "is_valid"),
        quality=_mapping_get(value, "quality", "quality_score"),
        overlap=_mapping_get(value, "overlap", "is_overlap"),
        mixed=_mapping_get(value, "mixed", "is_mixed"),
        clean_duration_us=_mapping_get(value, "clean_duration_us", "clean_duration", "duration_us"),
        overlap_flags=_mapping_get(value, "overlap_flags", "flags"),
    )


def _parse_candidate(value: Any) -> CandidateBlock:
    if isinstance(value, CandidateBlock):
        return value
    if not isinstance(value, Mapping) and not hasattr(value, "vector"):
        raise ConformalAssignmentError("candidate block must be an object")
    return CandidateBlock(
        vector=_mapping_get(value, "vector", "prototype"),
        block_id=_mapping_get(value, "block_id", "block", "id"),
        valid=_mapping_get(value, "valid", "is_valid"),
        quality=_mapping_get(value, "quality", "quality_score"),
        overlap=_mapping_get(value, "overlap", "is_overlap"),
        mixed=_mapping_get(value, "mixed", "is_mixed"),
        clean_duration_us=_mapping_get(value, "clean_duration_us", "clean_duration", "duration_us"),
        overlap_flags=_mapping_get(value, "overlap_flags", "flags"),
    )


def _validate_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ConformalAssignmentError(f"{name} must be a non-empty bounded string")
    return value


def _parse_flags(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise ConformalAssignmentError("overlap_flags must be a bounded sequence")
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise ConformalAssignmentError("overlap_flags must be a bounded sequence") from exc
    result = tuple(islice(iterator, MAX_OVERLAP_FLAGS + 1))
    if len(result) > MAX_OVERLAP_FLAGS:
        raise ConformalAssignmentResourceError("overlap_flags exceeds hard ceiling")
    return result


def _validate_safety(
    valid: Any, quality: Any, overlap: Any, mixed: Any, flags: Any,
    duration_us: Any, config: ConformalAssignmentConfig,
) -> tuple[Fraction, int]:
    if type(valid) is not bool or not valid:
        raise ConformalAssignmentError("block valid must be explicitly true")
    q = _finite_fraction(quality, "quality")
    if not config.min_quality <= q <= 1:
        raise ConformalAssignmentError("quality is below min_quality")
    if type(overlap) is not bool or overlap:
        raise ConformalAssignmentError("overlap block is not assignable")
    if type(mixed) is not bool or mixed:
        raise ConformalAssignmentError("mixed block is not assignable")
    parsed_flags = _parse_flags(flags)
    if any(bool(flag) for flag in parsed_flags):
        raise ConformalAssignmentError("overlap block is not assignable")
    if isinstance(duration_us, bool) or not isinstance(duration_us, int) or duration_us < config.min_clean_duration_us:
        raise ConformalAssignmentError("clean duration is below min_clean_duration_us")
    if duration_us > MAX_BLOCK_DURATION_US:
        raise ConformalAssignmentResourceError("block duration exceeds hard ceiling")
    return q, duration_us


def _quantize_unit_vector(value: Any, config: ConformalAssignmentConfig) -> tuple[int, ...]:
    """Ratio-normalize then L2-normalize one input on the fixed-point sphere."""

    if isinstance(value, (str, bytes, bytearray)) or value is None:
        raise ConformalAssignmentError("vector must be a finite numeric sequence")
    try:
        raw = tuple(islice(iter(value), config.max_dimension + 1))
    except TypeError as exc:
        raise ConformalAssignmentError("vector must be a finite numeric sequence") from exc
    if not raw:
        raise ConformalAssignmentError("vector must not be empty")
    if len(raw) > config.max_dimension:
        raise ConformalAssignmentResourceError("vector dimension exceeds hard ceiling")
    fractions: list[Fraction] = []
    for item in raw:
        number = _fraction_number(item, "vector value")
        fractions.append(number)
    max_abs = max(abs(item) for item in fractions)
    if max_abs == 0:
        raise ConformalAssignmentError("zero vector is invalid")
    # Dividing by max_abs before fixed-point quantization makes positive
    # scalar multiples land on the same deterministic lattice.
    ratio = [_round_fraction(item / max_abs * _VECTOR_SCALE) for item in fractions]
    if not any(ratio):
        raise ConformalAssignmentError("vector quantizes to zero")
    return _normalize_integer_vector(tuple(ratio))


def _normalize_integer_vector(vector: tuple[int, ...]) -> tuple[int, ...]:
    if not vector or not any(vector):
        raise ConformalAssignmentError("zero or cancelling centroid")
    max_abs = max(abs(value) for value in vector)
    if max_abs == 0:
        raise ConformalAssignmentError("zero or cancelling centroid")
    reduced = tuple(_round_ratio(value * _VECTOR_SCALE, max_abs) for value in vector)
    norm2 = sum(value * value for value in reduced)
    if norm2 <= 0:
        raise ConformalAssignmentError("zero or cancelling centroid")
    norm = math.isqrt(norm2)
    if norm <= 0:
        raise ConformalAssignmentError("zero or cancelling centroid")
    unit = tuple(_round_ratio(value * _VECTOR_SCALE, norm) for value in reduced)
    if not any(unit):
        raise ConformalAssignmentError("zero or cancelling centroid")
    return unit


def _effective_weight(quality: Fraction, duration_us: int, duration_cap_us: int) -> int:
    capped = min(duration_us, duration_cap_us)
    result = _round_fraction(quality * capped)
    if result <= 0:
        raise ConformalAssignmentError("quality-duration weight quantizes to zero")
    return result


def _aggregate_fragments(
    values: Sequence[tuple[tuple[int, ...], Fraction, int]], config: ConformalAssignmentConfig,
) -> tuple[tuple[int, ...], int, int]:
    """Spherical aggregate of fragments, with one capped block weight."""

    if not values:
        raise ConformalAssignmentError("empty block fragment")
    dimension = len(values[0][0])
    total_duration = sum(item[2] for item in values)
    if total_duration > MAX_BLOCK_DURATION_US:
        raise ConformalAssignmentResourceError("aggregated block duration exceeds hard ceiling")
    # q * duration controls the spherical prototype.  For equal-quality
    # fragments, splitting or merging a block is algebraically identical.
    weighted_quality_duration = sum(item[1] * item[2] for item in values)
    # Keep fragment contributions as exact Fractions and quantize only once
    # after the complete block has been assembled.  Rounding per fragment
    # would make a block change when one fragment is split into two.
    sums = [
        _round_fraction(
            sum(item[0][index] * item[1] * item[2] for item in values) * _QUALITY_SCALE
        )
        for index in range(dimension)
    ]
    vector = _normalize_integer_vector(tuple(sums))
    mean_quality = weighted_quality_duration / total_duration
    weight = _effective_weight(mean_quality, total_duration, config.duration_cap_us)
    return vector, weight, total_duration


def _iter_anchors(anchors: Iterable[Any], config: ConformalAssignmentConfig) -> tuple[Any, ...]:
    if isinstance(anchors, (str, bytes, bytearray)):
        raise ConformalAssignmentError("anchors must be a finite sequence")
    try:
        iterator = iter(anchors)
    except TypeError as exc:
        raise ConformalAssignmentError("anchors must be a finite sequence") from exc
    result = tuple(islice(iterator, config.max_anchor_blocks + 1))
    if len(result) > config.max_anchor_blocks:
        raise ConformalAssignmentResourceError("anchor block count exceeds hard ceiling")
    return result


def _prepare(anchors: Iterable[Any], config: ConformalAssignmentConfig) -> PreparedBMRCM:
    raw_anchors = _iter_anchors(anchors, config)
    grouped: dict[tuple[str, str], list[tuple[tuple[int, ...], Fraction, int]]] = {}
    grouped_dimensions: int | None = None
    total_coordinate_work = 0
    for raw in raw_anchors:
        block = _parse_anchor(raw)
        speaker = _validate_id(block.speaker_id, "speaker_id")
        block_id = _validate_id(block.block_id, "block_id")
        quality, duration = _validate_safety(
            block.valid, block.quality, block.overlap, block.mixed, block.overlap_flags,
            block.clean_duration_us, config,
        )
        vector = _quantize_unit_vector(block.vector, config)
        if grouped_dimensions is None:
            grouped_dimensions = len(vector)
        if len(vector) != grouped_dimensions:
            raise ConformalAssignmentError("all anchor vectors must share one dimension")
        total_coordinate_work += len(vector)
        if total_coordinate_work > MAX_TOTAL_COORDINATE_WORK:
            raise ConformalAssignmentResourceError("total coordinate work exceeds hard ceiling")
        grouped.setdefault((speaker, block_id), []).append((vector, quality, duration))
    speakers = tuple(sorted({speaker for speaker, _ in grouped}))
    if len(speakers) != 2:
        raise ConformalAssignmentError("exactly two speaker classes are required")
    ids_to_speakers: dict[str, set[str]] = {}
    for speaker, block_id in grouped:
        ids_to_speakers.setdefault(block_id, set()).add(speaker)
    if any(len(values) > 1 for values in ids_to_speakers.values()):
        raise ConformalAssignmentError("block_id cannot identify multiple speaker classes")
    prepared_by_speaker: dict[str, list[_PreparedBlock]] = {speaker: [] for speaker in speakers}
    for (speaker, block_id), fragments in sorted(grouped.items(), key=lambda item: item[0]):
        vector, weight, duration = _aggregate_fragments(fragments, config)
        prepared_by_speaker[speaker].append(_PreparedBlock(vector, weight, block_id, speaker, duration))
    if any(len(prepared_by_speaker[speaker]) < config.min_blocks for speaker in speakers):
        raise ConformalAssignmentError("insufficient independent anchor blocks")
    classes: list[_PreparedClass] = []
    for speaker in speakers:
        blocks = tuple(prepared_by_speaker[speaker])
        total_duration = sum(item.duration_us for item in blocks)
        if total_duration > MAX_TOTAL_SPEAKER_DURATION_US:
            raise ConformalAssignmentResourceError("total speaker duration exceeds hard ceiling")
        total_weight = sum(item.weight for item in blocks)
        total_vector = tuple(sum(item.vector[index] * item.weight for item in blocks) for index in range(grouped_dimensions or 0))
        # Fail at preparation time so a cancelling centroid cannot surface as
        # a candidate-dependent result.
        _normalize_integer_vector(total_vector)
        classes.append(_PreparedClass(speaker, blocks, total_vector, total_weight, total_duration))
    calibration_identity = {
        "schema": SCHEMA,
        "algorithm": ALGORITHM_VERSION,
        "epsilon": str(config.epsilon),
        "min_quality": str(config.min_quality),
        "min_clean_duration_us": config.min_clean_duration_us,
        "duration_cap_us": config.duration_cap_us,
        "speakers": speakers,
        "classes": tuple(
            (item.speaker_id, tuple((block.block_id, block.vector, block.weight, block.duration_us) for block in item.blocks))
            for item in classes
        ),
    }
    digest = _canonical_digest(calibration_identity)
    return PreparedBMRCM(config, speakers, (classes[0], classes[1]), digest, total_coordinate_work)


def _centroid_from_totals(total_vector: tuple[int, ...], total_weight: int) -> tuple[int, ...]:
    if total_weight <= 0:
        raise ConformalAssignmentError("centroid has no weight")
    return _normalize_integer_vector(total_vector)


def _cosine_distance_ratio(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, int]:
    if len(left) != len(right):
        raise ConformalAssignmentError("cosine vectors have different dimensions")
    left_norm2 = sum(item * item for item in left)
    right_norm2 = sum(item * item for item in right)
    if left_norm2 <= 0 or right_norm2 <= 0:
        raise ConformalAssignmentError("zero vector in cosine distance")
    denominator = math.isqrt(left_norm2 * right_norm2)
    if denominator <= 0:
        raise ConformalAssignmentError("zero vector in cosine distance")
    dot = sum(a * b for a, b in zip(left, right))
    return max(0, denominator - dot), denominator


def _relative_margin_ratio(
    vector: tuple[int, ...], own_centroid: tuple[int, ...], rival_centroid: tuple[int, ...]
) -> tuple[int, int]:
    own_num, own_den = _cosine_distance_ratio(vector, own_centroid)
    rival_num, rival_den = _cosine_distance_ratio(vector, rival_centroid)
    return own_num * rival_den - rival_num * own_den, own_den * rival_den


def _ratio_ge(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] * right[1] >= right[0] * left[1]


def _score_class(
    candidate_vector: tuple[int, ...], candidate_weight: int,
    own: _PreparedClass, rival: _PreparedClass,
) -> tuple[Fraction, tuple[tuple[int, int], ...], tuple[int, int]]:
    rival_centroid = _centroid_from_totals(rival.total_vector, rival.total_weight)
    # Candidate score: candidate is LOO from the augmented own total, hence
    # this is the original own class centroid.
    own_centroid = _centroid_from_totals(own.total_vector, own.total_weight)
    candidate_ratio = _relative_margin_ratio(candidate_vector, own_centroid, rival_centroid)
    augmented_vector = tuple(
        own.total_vector[index] + candidate_vector[index] * candidate_weight
        for index in range(len(candidate_vector))
    )
    augmented_weight = own.total_weight + candidate_weight
    calibration: list[tuple[int, int]] = []
    for block in own.blocks:
        loo_vector = tuple(
            augmented_vector[index] - block.vector[index] * block.weight
            for index in range(len(candidate_vector))
        )
        loo_centroid = _centroid_from_totals(loo_vector, augmented_weight - block.weight)
        calibration.append(_relative_margin_ratio(block.vector, loo_centroid, rival_centroid))
    return Fraction(candidate_ratio[0], candidate_ratio[1]), tuple(calibration), candidate_ratio


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fail_closed(reason: str) -> ConformalAssignmentResult:
    digest = _canonical_digest({"schema": SCHEMA, "algorithm": ALGORITHM_VERSION, "failure": reason})
    return ConformalAssignmentResult(
        "FAIL_CLOSED", (), (), digest,
        _canonical_digest({"decision": "FAIL_CLOSED", "reason": reason, "input_digest": digest}), reason,
    )


def _disabled_result() -> ConformalAssignmentResult:
    identity = {"schema": SCHEMA, "algorithm": ALGORITHM_VERSION, "enabled": False}
    digest = _canonical_digest(identity)
    return ConformalAssignmentResult(
        "DISABLED", (), (), digest,
        _canonical_digest({**identity, "decision": "DISABLED", "input_digest": digest}), "disabled",
    )


def _evaluate_prepared_candidate(prepared: PreparedBMRCM, candidate: CandidateBlock) -> ConformalAssignmentResult:
    config = prepared.config
    candidate_id = _validate_id(candidate.block_id, "candidate block_id")
    if any(candidate_id == block.block_id for item in prepared.classes for block in item.blocks):
        raise ConformalAssignmentError("candidate block_id duplicates an anchor block")
    quality, duration = _validate_safety(
        candidate.valid, candidate.quality, candidate.overlap, candidate.mixed,
        candidate.overlap_flags, candidate.clean_duration_us, config,
    )
    candidate_vector = _quantize_unit_vector(candidate.vector, config)
    dimension = len(prepared.classes[0].blocks[0].vector)
    if len(candidate_vector) != dimension:
        raise ConformalAssignmentError("candidate vector dimension differs from anchors")
    candidate_weight = _effective_weight(quality, duration, config.duration_cap_us)
    scores: list[HypothesisScore] = []
    for index, speaker in enumerate(prepared.speakers):
        own = prepared.classes[index]
        rival = prepared.classes[1 - index]
        candidate_a, calibration, candidate_ratio = _score_class(candidate_vector, candidate_weight, own, rival)
        rank = 1 + sum(_ratio_ge(value, candidate_ratio) for value in calibration)
        p_value = Fraction(rank, len(calibration) + 1)
        # BM-RCM v2 uses strict p > epsilon.  Equality is deliberately OOD.
        scores.append(HypothesisScore(speaker, p_value, candidate_a, len(calibration), p_value > config.epsilon))
    gamma = tuple(item.speaker_id for item in scores if item.eligible)
    decision = "SHADOW_SINGLETON" if len(gamma) == 1 else "OOD" if not gamma else "AMBIGUOUS"
    input_digest = _canonical_digest(
        {
            "calibration_digest": prepared.calibration_digest,
            "candidate": (candidate_id, candidate_vector, candidate_weight, duration),
        }
    )
    receipt_hash = _canonical_digest(
        {
            "schema": SCHEMA,
            "algorithm": ALGORITHM_VERSION,
            "decision": decision,
            "gamma_size": len(gamma),
            "p_values": tuple((item.speaker_id, item.p_value.numerator, item.p_value.denominator) for item in scores),
            "input_digest": input_digest,
        }
    )
    return ConformalAssignmentResult(decision, gamma, tuple(scores), input_digest, receipt_hash, "epsilon_gamma")


def prepare_bm_rcm(anchors: Iterable[Any], config: ConformalAssignmentConfig | None = None) -> PreparedBMRCM:
    """Prepare bounded calibration once for one or many candidate blocks."""

    config = config or ConformalAssignmentConfig()
    if not isinstance(config, ConformalAssignmentConfig):
        raise ConformalAssignmentError("config must be ConformalAssignmentConfig")
    if not config.enabled:
        # A real calibration is intentionally not consumed while disabled.
        return PreparedBMRCM(config, ("", ""), (), "", 0)  # type: ignore[arg-type]
    return _prepare(anchors, config)


def evaluate_bm_rcm(
    anchors: Iterable[Any], candidate: Any, config: ConformalAssignmentConfig | None = None,
) -> ConformalAssignmentResult:
    """Score one candidate; malformed/unsafe experimental input abstains."""

    config = config or ConformalAssignmentConfig()
    if not isinstance(config, ConformalAssignmentConfig):
        raise ConformalAssignmentError("config must be ConformalAssignmentConfig")
    if not config.enabled:
        return _disabled_result()
    try:
        prepared = _prepare(anchors, config)
        return prepared.evaluate(candidate)
    except (ConformalAssignmentError, TypeError, ValueError, OverflowError):
        return _fail_closed("invalid_input")


def evaluate_bm_rcm_batch(
    anchors: Iterable[Any], candidates: Iterable[Any], config: ConformalAssignmentConfig | None = None,
) -> tuple[ConformalAssignmentResult, ...]:
    """Prepare once and score a bounded candidate batch."""

    config = config or ConformalAssignmentConfig()
    if not isinstance(config, ConformalAssignmentConfig):
        raise ConformalAssignmentError("config must be ConformalAssignmentConfig")
    if not config.enabled:
        try:
            raw = tuple(islice(iter(candidates), MAX_BATCH_CANDIDATES + 1))
        except TypeError:
            return (_fail_closed("invalid_input"),)
        if len(raw) > MAX_BATCH_CANDIDATES:
            return tuple(_fail_closed("resource_bound") for _ in raw[:-1])
        return tuple(_disabled_result() for _ in raw)
    try:
        prepared = _prepare(anchors, config)
        return prepared.evaluate_batch(candidates)
    except (ConformalAssignmentError, TypeError, ValueError, OverflowError):
        return (_fail_closed("invalid_input"),)


def build_redacted_receipt(result: ConformalAssignmentResult) -> Mapping[str, Any]:
    if not isinstance(result, ConformalAssignmentResult):
        raise ConformalAssignmentError("result must be ConformalAssignmentResult")
    return result.receipt()


def assign_candidate(anchors: Iterable[Any], candidate: Any, config: ConformalAssignmentConfig | None = None) -> ConformalAssignmentResult:
    return evaluate_bm_rcm(anchors, candidate, config)


def score_candidate(anchors: Iterable[Any], candidate: Any, config: ConformalAssignmentConfig | None = None) -> ConformalAssignmentResult:
    return evaluate_bm_rcm(anchors, candidate, config)


# Notebook-friendly aliases retained for the isolated experiment.
BMRCMConfig = ConformalAssignmentConfig
BMRCMResult = ConformalAssignmentResult
AnchorPrototype = AnchorBlock
CandidatePrototype = CandidateBlock
evaluate = evaluate_bm_rcm
run_bm_rcm = evaluate_bm_rcm


__all__ = [
    "SCHEMA", "ALGORITHM_VERSION", "MAX_ANCHOR_BLOCKS", "MAX_DIMENSION",
    "MAX_OVERLAP_FLAGS", "MAX_BATCH_COORDINATE_WORK", "MAX_TOTAL_COORDINATE_WORK", "AnchorBlock",
    "CandidateBlock", "ConformalAssignmentConfig", "ConformalAssignmentError",
    "ConformalAssignmentResourceError", "HypothesisScore",
    "ConformalAssignmentResult", "PreparedBMRCM", "prepare_bm_rcm",
    "evaluate_bm_rcm", "evaluate_bm_rcm_batch", "assign_candidate",
    "score_candidate", "build_redacted_receipt", "BMRCMConfig", "BMRCMResult",
    "AnchorPrototype", "CandidatePrototype", "evaluate", "run_bm_rcm",
]
