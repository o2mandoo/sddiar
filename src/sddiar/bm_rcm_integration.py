"""Development-only BM-RCM v2 integration adapter.

The conformal implementation is intentionally kept separate from this
adapter.  This module translates the diarizer's baseline H2 state and
tracklet/embedding evidence into the strict ``AnchorBlock``/
``CandidateBlock`` contract and applies only singleton decisions to baseline
``UNKNOWN`` labels.  It never changes an assigned or protected-overlap label.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .conformal_assignment_experimental import (
    ALGORITHM_VERSION,
    MAX_BATCH_COORDINATE_WORK,
    AnchorBlock,
    CandidateBlock,
    ConformalAssignmentConfig,
    ConformalAssignmentResult,
    build_redacted_receipt,
    prepare_bm_rcm,
)


SPEAKER_IDS = ("SPEAKER_00", "SPEAKER_01")
MAX_INTEGRATION_CANDIDATES = 64
MAX_INTEGRATION_COORDINATE_WORK = 8_388_608
BM_RCM_INTEGRATION_VERSION = "bm-rcm-v2-baseline-h2-adapter-v1"
BM_RCM_ALGORITHM_VERSION = ALGORITHM_VERSION


@dataclass(frozen=True, slots=True)
class BMRCMConfig:
    """Fixed experiment policy derived from the existing diarizer config."""

    epsilon: Fraction
    min_quality: Fraction
    min_clean_duration_us: int
    duration_cap_us: int


@dataclass(frozen=True, slots=True)
class BMRCMRun:
    """A contiguous, valid UNKNOWN run eligible for BM-RCM scoring."""

    ordinal: int
    tracklet_indexes: tuple[int, ...]
    tracklet_ids: tuple[str, ...]
    continuity_group_id: str
    clean_duration_us: int
    quality: float
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BMRCMIntegrationResult:
    """Scalar diagnostics plus labels; vectors/IDs stay out of diagnostics."""

    labels: tuple[str, ...]
    runs: tuple[BMRCMRun, ...]
    decisions: tuple[ConformalAssignmentResult, ...]
    candidate_run_count: int
    candidate_count: int
    rescued_duration_us: int
    singleton_count: int
    ood_count: int
    ambiguous_count: int
    fail_closed_count: int
    unchanged_existing_assigned_us: int
    unchanged_overlap_count: int
    decision_state: str
    preparation_failed: bool
    preparation_reason: str | None
    coordinate_work: int
    batch_count: int
    calibration_receipt: Mapping[str, Any]
    decision_receipts: tuple[Mapping[str, Any], ...]

    def redacted_diagnostics(self) -> Mapping[str, Any]:
        """Return bounded scalar-only diagnostics suitable for a report."""

        return MappingProxyType({
            "integration_version": BM_RCM_INTEGRATION_VERSION,
            "algorithm": ALGORITHM_VERSION,
            "decision_state": self.decision_state,
            "candidate_run_count": self.candidate_run_count,
            "candidate_count": self.candidate_count,
            "rescued_duration_us": self.rescued_duration_us,
            "singleton_count": self.singleton_count,
            "ood_count": self.ood_count,
            "ambiguous_count": self.ambiguous_count,
            "fail_closed_count": self.fail_closed_count,
            "preparation_failed": self.preparation_failed,
            "preparation_reason": self.preparation_reason,
            "coordinate_work": self.coordinate_work,
            "coordinate_work_cap": MAX_INTEGRATION_COORDINATE_WORK,
            "batch_count": self.batch_count,
            "calibration_receipt": dict(self.calibration_receipt),
        })


def _effective_min_quality(value: Any) -> Fraction:
    """Convert the existing threshold to BM's strict positive quality domain.

    The reference diarizer permits a zero threshold.  BM-RCM requires a
    positive minimum, so zero is represented by the smallest bounded decimal
    floor; this excludes only an embedding whose quality is exactly zero.
    """

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = 0.0
    if not isfinite(number) or number < 0.0:
        raise ValueError("anchor_quality_min must be finite and non-negative")
    return Fraction(str(max(number, 1e-6)))


def fixed_bm_rcm_config(diarization_config: Any) -> BMRCMConfig:
    """Build the pre-fixed BM policy from existing anchor safety thresholds."""

    anchor_min = int(diarization_config.anchor_min_clean_us)
    if anchor_min <= 0:
        raise ValueError("anchor_min_clean_us must be positive")
    # Keep epsilon and duration cap explicit and non-tunable for this arm.
    return BMRCMConfig(
        epsilon=Fraction(1, 10),
        min_quality=_effective_min_quality(diarization_config.anchor_quality_min),
        min_clean_duration_us=anchor_min,
        duration_cap_us=10_000_000,
    )


def _valid_embedding(embedding: Any) -> bool:
    return bool(
        embedding is not None
        and getattr(embedding, "is_valid", False)
        and getattr(embedding, "vector", None) is not None
    )


def _quality(embedding: Any) -> float:
    value = float(getattr(embedding, "quality", 0.0))
    if not isfinite(value):
        return 0.0
    return value


def _spherical_duration_weighted(
    vectors: Sequence[Sequence[float]], durations: Sequence[int],
) -> tuple[float, ...]:
    if not vectors or len(vectors) != len(durations):
        raise ValueError("candidate vector fragments are empty or misaligned")
    dimension = len(vectors[0])
    sums = [0.0] * dimension
    for vector, duration in zip(vectors, durations):
        if len(vector) != dimension or duration <= 0:
            raise ValueError("candidate vector fragments have invalid dimensions/duration")
        for index, value in enumerate(vector):
            number = float(value)
            if not isfinite(number):
                raise ValueError("candidate vector contains a non-finite value")
            sums[index] += number * duration
    norm = sqrt(sum(value * value for value in sums))
    if not isfinite(norm) or norm <= 1e-12:
        raise ValueError("candidate spherical vector is degenerate")
    return tuple(value / norm for value in sums)


def _anchor_blocks(
    tracklets: Sequence[Any], embeddings: Mapping[str, Any], states: Mapping[str, Any],
    anchor_evidence: Sequence[Any] = (),
) -> tuple[AnchorBlock, ...]:
    """Use only H2 state's stable anchor IDs; preserve independent block IDs."""

    by_id = {item.tracklet_id: item for item in tracklets}
    blocks: list[AnchorBlock] = []
    for speaker_id in SPEAKER_IDS:
        state = states.get(speaker_id)
        if state is None:
            continue
        # One state anchor ID may have multiple fragments sharing its
        # continuity/independent block.  Pass all such evidence fragments and
        # let BM-RCM aggregate them by block_id.
        stable_ids = frozenset(str(item) for item in getattr(state, "stable_anchor_ids", ()))
        if not stable_ids:
            continue
        # Prefer the already-selected AnchorEvidence so the integration does
        # not reconstruct independent_block_id from a potentially different
        # grouping policy.  The tracklet fallback keeps this adapter useful
        # for small synthetic callers that only provide states.
        selected = (
            tuple(item for item in anchor_evidence if item.tracklet_id in stable_ids)
            if anchor_evidence else
            tuple(item for item in tracklets if item.tracklet_id in stable_ids)
        )
        for evidence in selected:
            tracklet = by_id.get(evidence.tracklet_id)
            if tracklet is None:
                continue
            embedding = embeddings.get(tracklet.tracklet_id)
            if not _valid_embedding(embedding):
                continue
            if tracklet.protected_overlap or tracklet.mixed_tracklet_suspect:
                continue
            evidence_block_id = str(
                getattr(evidence, "independent_block_id", tracklet.continuity_group_id)
            )
            evidence_duration = int(
                getattr(evidence, "clean_speech_us", tracklet.clean_speech_us)
            )
            blocks.append(AnchorBlock(
                vector=tuple(embedding.vector),
                block_id=evidence_block_id,
                speaker_id=speaker_id,
                valid=True,
                quality=_quality(embedding),
                overlap=False,
                mixed=False,
                clean_duration_us=evidence_duration,
                overlap_flags=(),
            ))
    return tuple(blocks)


def _candidate_runs(
    tracklets: Sequence[Any], baseline_labels: Sequence[str], embeddings: Mapping[str, Any],
    min_clean_duration_us: int,
) -> tuple[BMRCMRun, ...]:
    """Collect source-contiguous valid UNKNOWN runs within one continuity group."""

    if len(tracklets) != len(baseline_labels):
        raise ValueError("tracklet and baseline label counts differ")
    runs: list[BMRCMRun] = []
    current: list[int] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        selected = [tracklets[index] for index in current]
        duration = sum(int(item.clean_speech_us) for item in selected)
        if duration >= min_clean_duration_us:
            vectors = [tuple(embeddings[item.tracklet_id].vector) for item in selected]
            durations = [int(item.clean_speech_us) for item in selected]
            quality = min(_quality(embeddings[item.tracklet_id]) for item in selected)
            try:
                vector = _spherical_duration_weighted(vectors, durations)
            except (TypeError, ValueError, OverflowError):
                current = []
                return
            runs.append(BMRCMRun(
                ordinal=len(runs),
                tracklet_indexes=tuple(current),
                tracklet_ids=tuple(item.tracklet_id for item in selected),
                continuity_group_id=str(selected[0].continuity_group_id),
                clean_duration_us=duration,
                quality=quality,
                vector=vector,
            ))
        current = []

    previous: Any | None = None
    for index, (tracklet, label) in enumerate(zip(tracklets, baseline_labels)):
        embedding = embeddings.get(tracklet.tracklet_id)
        eligible = bool(
            label == "UNKNOWN"
            and _valid_embedding(embedding)
            and not tracklet.protected_overlap
            and not tracklet.mixed_tracklet_suspect
            and int(tracklet.clean_speech_us) > 0
        )
        contiguous = bool(
            previous is not None
            and previous.continuity_group_id == tracklet.continuity_group_id
            and int(previous.end_us) == int(tracklet.start_us)
        )
        if not eligible or (current and not contiguous):
            flush()
        if eligible:
            current.append(index)
        previous = tracklet if eligible else None
    flush()
    return tuple(runs)


def integrate_bm_rcm(
    *,
    tracklets: Sequence[Any],
    baseline_labels: Sequence[str],
    embeddings: Sequence[Any],
    anchor_evidence: Sequence[Any] = (),
    states: Mapping[str, Any],
    decision: Any,
    diarization_config: Any,
) -> BMRCMIntegrationResult:
    """Run BM-RCM v2 as an UNKNOWN-only, singleton-only development arm."""

    labels = tuple(str(item) for item in baseline_labels)
    if len(tracklets) != len(labels):
        raise ValueError("tracklet and baseline label counts differ")
    embedding_by_id = {item.tracklet_id: item for item in embeddings}
    decision_state = str(getattr(decision, "state", decision))
    policy = fixed_bm_rcm_config(diarization_config)
    runs = _candidate_runs(tracklets, labels, embedding_by_id, policy.min_clean_duration_us)
    blocks = _anchor_blocks(tracklets, embedding_by_id, states, anchor_evidence)
    config = ConformalAssignmentConfig(
        enabled=True,
        epsilon=policy.epsilon,
        min_quality=policy.min_quality,
        min_clean_duration_us=policy.min_clean_duration_us,
        duration_cap_us=policy.duration_cap_us,
    )
    decisions: list[ConformalAssignmentResult] = []
    updated = list(labels)
    preparation_failed = False
    preparation_reason: str | None = None
    coordinate_work = 0
    batch_count = 0
    prepared = None
    try:
        if decision_state != "H2_CONFIRMED":
            raise ValueError("BM-RCM requires H2_CONFIRMED baseline")
        prepared = prepare_bm_rcm(blocks, config)
    except Exception as exc:  # conformal module intentionally reports fail-closed input
        preparation_failed = True
        preparation_reason = type(exc).__name__

    if prepared is not None:
        candidates = tuple(
            CandidateBlock(
                vector=run.vector,
                block_id=f"bm-candidate-{run.ordinal}",
                valid=True,
                quality=run.quality,
                overlap=False,
                mixed=False,
                clean_duration_us=run.clean_duration_us,
                overlap_flags=(),
            )
            for run in runs
        )
        block_count = sum(len(item.blocks) for item in prepared.classes)
        dimension = len(prepared.classes[0].blocks[0].vector)
        coordinate_work = prepared.coordinate_work + len(candidates) * block_count * dimension
        if len(candidates) > MAX_INTEGRATION_CANDIDATES or coordinate_work > MAX_INTEGRATION_COORDINATE_WORK:
            decisions.extend(
                ConformalAssignmentResult(
                    "FAIL_CLOSED", (), (), "", "", "integration_candidate_bound"
                )
                for _ in candidates
            )
        else:
            per_candidate_work = block_count * dimension
            remaining_work = MAX_BATCH_COORDINATE_WORK - prepared.coordinate_work
            batch_size = max(1, remaining_work // max(1, per_candidate_work))
            for start in range(0, len(candidates), batch_size):
                batch_count += 1
                decisions.extend(prepared.evaluate_batch(candidates[start:start + batch_size]))
        for run, result in zip(runs, decisions):
            # Only a strict singleton is authoritative for this arm.  The
            # baseline UNKNOWN guard is repeated at application time.
            if result.decision == "SHADOW_SINGLETON" and len(result.gamma) == 1:
                for index in run.tracklet_indexes:
                    if updated[index] == "UNKNOWN":
                        updated[index] = result.gamma[0]

    singleton = sum(item.decision == "SHADOW_SINGLETON" and len(item.gamma) == 1 for item in decisions)
    ood = sum(item.decision == "OOD" for item in decisions)
    ambiguous = sum(item.decision == "AMBIGUOUS" for item in decisions)
    failed = sum(item.decision == "FAIL_CLOSED" for item in decisions) + int(preparation_failed)
    rescued_duration = sum(
        int(tracklets[index].end_us - tracklets[index].start_us)
        for index, (old, new) in enumerate(zip(labels, updated))
        if old == "UNKNOWN" and new in SPEAKER_IDS
    )
    unchanged_assigned = sum(
        int(tracklet.end_us - tracklet.start_us)
        for tracklet, old, new in zip(tracklets, labels, updated)
        if old in SPEAKER_IDS and new != old
    )
    unchanged_overlap = sum(
        old == "OVERLAP" and new == "OVERLAP"
        for old, new in zip(labels, updated)
    )
    if unchanged_assigned:
        raise RuntimeError("BM-RCM changed an existing assigned label")
    if any(old == "OVERLAP" and new != old for old, new in zip(labels, updated)):
        raise RuntimeError("BM-RCM changed an OVERLAP label")
    receipts = tuple(build_redacted_receipt(item) for item in decisions)
    calibration_receipt: Mapping[str, Any]
    if prepared is not None:
        calibration_receipt = MappingProxyType({
            "schema": "bm-rcm-v2",
            "algorithm": ALGORITHM_VERSION,
            "decision": "PREPARED",
            "gamma_size": 0,
            "hypothesis_count": 0,
            "eligible_count": 0,
            "input_digest": prepared.calibration_digest,
            "receipt_hash": prepared.calibration_digest,
        })
    else:
        calibration_receipt = MappingProxyType({
            "schema": "bm-rcm-v2",
            "algorithm": ALGORITHM_VERSION,
            "decision": "FAIL_CLOSED",
            "gamma_size": 0,
            "hypothesis_count": 0,
            "eligible_count": 0,
            "input_digest": "",
            "receipt_hash": "",
        })
    return BMRCMIntegrationResult(
        labels=tuple(updated),
        runs=runs,
        decisions=tuple(decisions),
        candidate_run_count=len(runs),
        candidate_count=len(decisions),
        rescued_duration_us=rescued_duration,
        singleton_count=singleton,
        ood_count=ood,
        ambiguous_count=ambiguous,
        fail_closed_count=failed,
        unchanged_existing_assigned_us=unchanged_assigned,
        unchanged_overlap_count=unchanged_overlap,
        decision_state=decision_state,
        preparation_failed=preparation_failed,
        preparation_reason=preparation_reason,
        coordinate_work=coordinate_work,
        batch_count=batch_count,
        calibration_receipt=calibration_receipt,
        decision_receipts=receipts,
    )


__all__ = [
    "BMRCMConfig", "BMRCMRun", "BMRCMIntegrationResult",
    "BM_RCM_INTEGRATION_VERSION", "BM_RCM_ALGORITHM_VERSION", "MAX_INTEGRATION_CANDIDATES",
    "MAX_INTEGRATION_COORDINATE_WORK",
    "fixed_bm_rcm_config", "integrate_bm_rcm",
]
