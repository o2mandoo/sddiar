"""Deterministic, conservative 1--2 speaker diarization reference core.

This module deliberately has no audio decoder, VAD, ONNX Runtime, or network
dependency. It consumes source-time speech evidence and already-normalized
speaker embeddings. The production embedding backend belongs behind the
model-pack boundary; this core preserves uncertainty when its evidence is not
strong enough.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf, sqrt
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    AnchorEvidence,
    DiarizationSpan,
    EmbeddingResult,
    HypothesisDecision,
    ProtectedOverlapSpan,
    SpeakerAssignment,
    SpeakerHypothesis,
    SpeakerState,
    Tracklet,
    TrackletBuildResult,
)
from .errors import ContractValidationError


INF = inf
SPEAKER_IDS = ("SPEAKER_00", "SPEAKER_01")


@dataclass(frozen=True, slots=True)
class DiarizationConfig:
    """Development defaults only; a release must use a signed calibration profile."""

    anchor_min_clean_us: int = 1_500_000
    support_min_clean_us: int = 700_000
    max_tracklet_us: int = 30_000_000
    min_split_side_us: int = 150_000
    scd_split_min: float = 0.5
    scd_support_min: float = 0.5
    overlap_protect_min: float = 0.5
    anchor_quality_min: float = 0.0
    anchor_consistency_min: float = -1.0
    anchor_weight_cap_us: int = 5_000_000
    anchor_outlier_distance_max: float = 0.35
    cluster_max_iter: int = 30
    centroid_convergence: float = 1e-6
    bounded_seed_count: int = 6
    max_stability_blocks: int = 8
    h1_max_outlier_ratio: float = 0.25
    h1_max_dispersion: float = 0.35
    h2_min_independent_anchor_count: int = 2
    h2_min_clean_anchor_us: int = 1_500_000
    h2_min_separation: float = 0.15
    h2_max_cluster_dispersion: float = 0.35
    h2_max_outlier_ratio: float = 0.25
    h2_min_cost_gain: float = 0.02
    h2_min_label_stability: float = 0.5
    h2_min_centroid_stability: float = 0.5
    lambda_k2: float = 0.05
    lambda_stability: float = 0.05
    lambda_condition: float = 0.05
    third_risk_residual_distance_min: float = 0.45
    third_risk_max_dispersion: float = 0.2
    third_risk_min_separation: float = 0.15
    anchor_stable_distance_ceiling: float = 0.35
    anchor_absolute_distance_max: float = 0.4
    anchor_margin_min: float = 0.03
    support_stable_distance_ceiling: float = 0.35
    support_absolute_distance_max: float = 0.4
    support_margin_min: float = 0.03
    micro_stable_distance_ceiling: float = 0.25
    micro_absolute_distance_max: float = 0.3
    micro_margin_min: float = 0.08
    enable_recent_centroid: bool = False
    recent_blend: float = 0.2
    recent_max_benefit: float = 0.08
    recent_update_margin_min: float = 0.08
    recent_update_quality_min: float = 0.8
    recent_update_stable_fit_max: float = 0.25
    recent_radius_max: float = 0.2
    recent_opponent_margin_min: float = 0.05
    recent_decay: float = 0.8
    switch_base: float = 0.35
    scd_relief: float = 0.25
    gap_relief: float = 0.25
    long_gap_reset_us: int = 1_500_000
    uncertainty_transition: float = 0.02
    # Above ANCHOR/SUPPORT hard-distance ceilings so Viterbi does not undo a
    # locally valid assignment; MICRO retains its much cheaper abstention cost.
    unknown_cost: float = 0.40
    unknown_micro_cost: float = 0.02
    other_cost: float = 0.4
    isolated_run_max_us: int = 350_000
    include_non_speech: bool = False


@dataclass(frozen=True, slots=True)
class HypothesisEvaluation:
    """The single-pass H1/H2 evaluation result.

    ``h2`` is retained as the diagnostics object used by the existing local
    experiment output.  The aliases make the relationship explicit for new
    callers while keeping the diagnostic payload as the existing immutable
    :class:`SpeakerHypothesis` value.
    """

    decision: HypothesisDecision
    h1: SpeakerHypothesis
    h2: SpeakerHypothesis

    @property
    def diagnostics(self) -> SpeakerHypothesis:
        return self.h2

    @property
    def h1_diagnostics(self) -> SpeakerHypothesis:
        return self.h1

    @property
    def h2_diagnostics(self) -> SpeakerHypothesis:
        return self.h2


@dataclass(frozen=True, slots=True)
class SequenceDecodeResult:
    """Internal, scalar-only trace from one sequence-decoder pass.

    ``labels`` and ``local_assignments`` are aligned with the input tracklets.
    They intentionally contain no embedding vectors or centroids.  The public
    ``finalize_sequence`` compatibility seam still returns spans only; this
    richer result exists for bounded, opt-in decoder ablations and aggregate
    diagnostics.
    """

    spans: tuple[DiarizationSpan, ...]
    labels: tuple[str, ...]
    local_assignments: tuple[SpeakerAssignment, ...]


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _source_start(value: Any) -> int:
    return int(_get(value, "start_us", _get(value, "start", 0)))


def _source_end(value: Any) -> int:
    return int(_get(value, "end_us", _get(value, "end", 0)))


def _vector(value: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(item) for item in value)
    if not vector:
        raise ContractValidationError("speaker embedding cannot be empty")
    norm = sqrt(sum(item * item for item in vector))
    if norm <= 1e-12:
        raise ContractValidationError("speaker embedding cannot be zero")
    return tuple(item / norm for item in vector)


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ContractValidationError("embedding dimensions do not match")
    return 1.0 - sum(float(x) * float(y) for x, y in zip(left, right))


def _weighted_centroid(items: Sequence[tuple[Sequence[float], float]]) -> tuple[float, ...]:
    if not items:
        raise ContractValidationError("cannot compute an empty centroid")
    dimension = len(items[0][0])
    sums = [0.0] * dimension
    for vector, weight in items:
        if len(vector) != dimension:
            raise ContractValidationError("embedding dimensions do not match")
        for index, item in enumerate(vector):
            sums[index] += float(weight) * float(item)
    # A one-speaker fit over perfectly opposed evidence can have a zero
    # resultant vector. Keep the fit deterministic so H2 can still evaluate
    # rather than crashing before its stronger evidence is considered.
    try:
        return _vector(sums)
    except ContractValidationError:
        return _vector(items[0][0])


def _stable_id(namespace: str, *parts: object) -> str:
    import hashlib

    raw = "|".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    return f"{namespace}_{hashlib.sha256(raw).hexdigest()[:20]}"


def _duration_class(clean_speech_us: int, cfg: DiarizationConfig) -> str:
    if clean_speech_us >= cfg.anchor_min_clean_us:
        return "ANCHOR"
    if clean_speech_us >= cfg.support_min_clean_us:
        return "SUPPORT"
    return "MICRO"


def _merge_protected_overlap(
    overlap_regions: Iterable[Any], cfg: DiarizationConfig, audio_id: str
) -> tuple[ProtectedOverlapSpan, ...]:
    candidates: list[tuple[int, int, float, tuple[str, ...]]] = []
    for index, region in enumerate(overlap_regions):
        start, end = _source_start(region), _source_end(region)
        if end <= start:
            continue
        evidence = float(_get(region, "overlap_evidence", _get(region, "evidence", 1.0)))
        if _get(region, "is_high", None) is False or evidence < cfg.overlap_protect_min:
            continue
        evidence_ids = tuple(_get(region, "evidence_ids", ()))
        if not evidence_ids:
            evidence_ids = (str(_get(region, "evidence_id", _stable_id("overlap-evidence", audio_id, index))),)
        candidates.append((start, end, evidence, evidence_ids))
    candidates.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int, float, tuple[str, ...]]] = []
    for start, end, evidence, evidence_ids in candidates:
        if merged and start <= merged[-1][1]:
            old_start, old_end, old_evidence, old_ids = merged[-1]
            merged[-1] = (old_start, max(old_end, end), max(old_evidence, evidence), tuple(dict.fromkeys(old_ids + evidence_ids)))
        else:
            merged.append((start, end, evidence, evidence_ids))
    return tuple(
        ProtectedOverlapSpan(
            span_id=_stable_id("protected-overlap", audio_id, start, end, ordinal),
            start_us=start,
            end_us=end,
            overlap_evidence=evidence,
            evidence_ids=evidence_ids,
        )
        for ordinal, (start, end, evidence, evidence_ids) in enumerate(merged)
    )


def _interval_overlap(start: int, end: int, other_start: int, other_end: int) -> int:
    return max(0, min(end, other_end) - max(start, other_start))


def _scd_evidence_at(time_us: int, events: Sequence[Any]) -> float | None:
    exact = [float(_get(event, "evidence", _get(event, "score", 0.0))) for event in events if int(_get(event, "time_us", _get(event, "time", -1))) == time_us]
    return max(exact) if exact else None


def build_tracklets(
    vad_regions: Sequence[Any],
    scd_events: Sequence[Any] = (),
    overlap_regions: Sequence[Any] = (),
    cfg: DiarizationConfig | None = None,
    audio_id: str = "audio",
) -> TrackletBuildResult:
    """Create non-overlap tracklets and preserve high-overlap spans separately."""

    cfg = cfg or DiarizationConfig()
    # Capability boundary: only the concrete event values emitted by an
    # approved segmentation gate may alter tracklets.  In particular, do not
    # accept mappings, duck-typed objects, or diagnostic evidence with an
    # ``approved``/``is_high`` flag: those are observational by contract.
    from .segmentation import _EnforceableOverlapEvent, _EnforceableSpeakerChangeEvent

    for event in scd_events:
        if type(event) is not _EnforceableSpeakerChangeEvent:
            raise ContractValidationError(
                "scd_events must contain exact sealed enforceable SCD values"
            )
        if event.source_id != audio_id:
            raise ContractValidationError("SCD event source does not match audio_id")
    for region in overlap_regions:
        if type(region) is not _EnforceableOverlapEvent:
            raise ContractValidationError(
                "overlap_regions must contain exact sealed enforceable OSD values"
            )
        if region.source_id != audio_id:
            raise ContractValidationError("OSD event source does not match audio_id")
    protected = _merge_protected_overlap(overlap_regions, cfg, audio_id)
    tracklets: list[Tracklet] = []
    boundary_ids: list[str] = []
    for region_index, region in enumerate(vad_regions):
        region_start, region_end = _source_start(region), _source_end(region)
        if region_end <= region_start:
            continue
        cuts = {region_start, region_end}
        # Keep boundary evidence scoped to this source speech region.  The
        # previous accumulator leaked IDs from an earlier region into every
        # later tracklet, which made receipts appear to support unrelated cuts.
        evidence_ids_at_cut: dict[int, list[str]] = {}
        for overlap in protected:
            if region_start < overlap.start_us < region_end:
                cuts.add(overlap.start_us)
            if region_start < overlap.end_us < region_end:
                cuts.add(overlap.end_us)
        for event_index, event in enumerate(scd_events):
            event_time = int(_get(event, "time_us", _get(event, "time", -1)))
            score = float(_get(event, "evidence", _get(event, "score", 0.0)))
            if not (region_start < event_time < region_end and score >= cfg.scd_split_min):
                continue
            if event_time - region_start < cfg.min_split_side_us or region_end - event_time < cfg.min_split_side_us:
                continue
            cuts.add(event_time)
            evidence_id = str(_get(event, "evidence_id", _stable_id("scd", audio_id, event_time, event_index)))
            boundary_ids.append(evidence_id)
            evidence_ids_at_cut.setdefault(event_time, []).append(evidence_id)
        if cfg.max_tracklet_us > 0:
            artificial = region_start + cfg.max_tracklet_us
            while artificial < region_end:
                cuts.add(artificial)
                artificial += cfg.max_tracklet_us
        sorted_cuts = sorted(cuts)
        continuity_group_id = _stable_id("continuity", audio_id, region_start, region_end)
        speech_region_id = str(_get(region, "region_id", _stable_id("speech-region", audio_id, region_index, region_start, region_end)))
        for ordinal, (start, end) in enumerate(zip(sorted_cuts, sorted_cuts[1:])):
            protected_us = sum(_interval_overlap(start, end, overlap.start_us, overlap.end_us) for overlap in protected)
            clean_speech_us = max(0, end - start - protected_us)
            if clean_speech_us == 0:
                # The protected span is returned independently and later materialized as OVERLAP.
                continue
            local_boundary_ids = tuple(dict.fromkeys(
                tuple(evidence_ids_at_cut.get(start, ())) + tuple(evidence_ids_at_cut.get(end, ()))
            ))
            tracklets.append(
                Tracklet(
                    tracklet_id=_stable_id("tracklet", audio_id, start, end, ordinal),
                    speech_region_id=speech_region_id,
                    continuity_group_id=continuity_group_id,
                    start_us=start,
                    end_us=end,
                    clean_speech_us=clean_speech_us,
                    kind=_duration_class(clean_speech_us, cfg),
                    boundary_evidence_ids=local_boundary_ids,
                    scd_evidence_before=_scd_evidence_at(start, scd_events),
                    scd_evidence_after=_scd_evidence_at(end, scd_events),
                    protected_overlap=protected_us > 0,
                    mixed_tracklet_suspect=False,
                )
            )
    return TrackletBuildResult(
        tracklets=tuple(sorted(tracklets, key=lambda item: (item.start_us, item.end_us, item.tracklet_id))),
        protected_overlap_spans=protected,
        boundary_evidence_ids=tuple(dict.fromkeys(boundary_ids)),
    )


def select_anchor_evidence(
    tracklets: Sequence[Tracklet], embeddings: Sequence[EmbeddingResult], cfg: DiarizationConfig | None = None
) -> tuple[tuple[AnchorEvidence, ...], tuple[tuple[Tracklet, EmbeddingResult], ...], tuple[tuple[Tracklet, str], ...]]:
    cfg = cfg or DiarizationConfig()
    by_tracklet = {embedding.tracklet_id: embedding for embedding in embeddings}
    anchors: list[AnchorEvidence] = []
    support: list[tuple[Tracklet, EmbeddingResult]] = []
    deferred: list[tuple[Tracklet, str]] = []
    for tracklet in tracklets:
        embedding = by_tracklet.get(tracklet.tracklet_id)
        if embedding is None or not embedding.is_valid or embedding.vector is None:
            deferred.append((tracklet, "INVALID_EMBEDDING"))
            continue
        anchor_eligible = (
            tracklet.kind == "ANCHOR"
            and not tracklet.protected_overlap
            and not tracklet.mixed_tracklet_suspect
            and embedding.quality >= cfg.anchor_quality_min
            and embedding.intra_window_consistency >= cfg.anchor_consistency_min
        )
        if anchor_eligible:
            anchors.append(
                AnchorEvidence(
                    tracklet_id=tracklet.tracklet_id,
                    vector=embedding.vector,
                    weight=max(1e-6, embedding.quality) * min(tracklet.clean_speech_us, cfg.anchor_weight_cap_us),
                    clean_speech_us=tracklet.clean_speech_us,
                    independent_block_id=tracklet.continuity_group_id,
                    continuity_group_id=tracklet.continuity_group_id,
                    start_us=tracklet.start_us,
                    end_us=tracklet.end_us,
                    scd_evidence_before=tracklet.scd_evidence_before,
                )
            )
        elif tracklet.kind == "SUPPORT" and not tracklet.protected_overlap and not tracklet.mixed_tracklet_suspect:
            support.append((tracklet, embedding))
        else:
            deferred.append((tracklet, "DEFERRED_MICRO_OR_LOW_QUALITY"))
    return tuple(anchors), tuple(support), tuple(deferred)


def _assign(anchors: Sequence[AnchorEvidence], centers: Sequence[tuple[float, ...]], threshold: float) -> tuple[dict[str, int | None], list[list[AnchorEvidence]], list[float]]:
    labels: dict[str, int | None] = {}
    inliers: list[list[AnchorEvidence]] = [[] for _ in centers]
    distances: list[float] = []
    for anchor in anchors:
        values = [cosine_distance(anchor.vector, center) for center in centers]
        label = min(range(len(values)), key=lambda index: (values[index], index))
        distance = values[label]
        distances.append(distance)
        if distance <= threshold:
            labels[anchor.tracklet_id] = label
            inliers[label].append(anchor)
        else:
            labels[anchor.tracklet_id] = None
    return labels, inliers, distances


def _invalid_hypothesis(k: int, reason: str) -> SpeakerHypothesis:
    # SpeakerHypothesis deliberately requires finite costs.  Keep this
    # sentinel far outside any practical cost while allowing invalid fits to
    # be returned as diagnostics (including the empty-input H2 path).
    invalid_cost = 1e30
    return SpeakerHypothesis(
        k=k,
        centers=(),
        anchor_labels={},
        is_valid=False,
        valid_constraints=False,
        robust_cost=invalid_cost,
        total_cost=invalid_cost,
        reason_codes=(reason,),
    )


def robust_spherical_fit(
    anchors: Sequence[AnchorEvidence], k: int, seed_centers: Sequence[Sequence[float]], cfg: DiarizationConfig | None = None
) -> SpeakerHypothesis:
    cfg = cfg or DiarizationConfig()
    if k not in (1, 2) or len(seed_centers) != k or len(anchors) < k:
        return _invalid_hypothesis(k, "INSUFFICIENT_ANCHORS")
    centers = tuple(_vector(seed) for seed in seed_centers)
    for _ in range(cfg.cluster_max_iter):
        # Move centers with all nearest anchors first. Applying the outlier
        # threshold during initialization can trap H2 in a singleton cluster.
        groups: list[list[AnchorEvidence]] = [[] for _ in centers]
        for anchor in anchors:
            distances = [cosine_distance(anchor.vector, center) for center in centers]
            label = min(range(len(distances)), key=lambda index: (distances[index], index))
            groups[label].append(anchor)
        if any(not group for group in groups):
            return _invalid_hypothesis(k, "EMPTY_INLIER_CLUSTER")
        updated = tuple(_weighted_centroid([(anchor.vector, anchor.weight) for anchor in group]) for group in groups)
        if all(cosine_distance(old, new) <= cfg.centroid_convergence for old, new in zip(centers, updated)):
            centers = updated
            break
        centers = updated
    labels, inliers, distances = _assign(anchors, centers, cfg.anchor_outlier_distance_max)
    if any(not group for group in inliers):
        return _invalid_hypothesis(k, "EMPTY_INLIER_CLUSTER")
    total_weight = sum(anchor.weight for anchor in anchors)
    robust_cost = sum(anchor.weight * min(distance, cfg.anchor_outlier_distance_max) for anchor, distance in zip(anchors, distances)) / total_weight
    outlier_ratio = sum(anchor.weight for anchor, distance in zip(anchors, distances) if distance > cfg.anchor_outlier_distance_max) / total_weight
    dispersion = tuple(sum(anchor.weight * cosine_distance(anchor.vector, center) for anchor in group) / sum(anchor.weight for anchor in group) for center, group in zip(centers, inliers))
    duration = tuple(sum(anchor.clean_speech_us for anchor in group) for group in inliers)
    independent_count = tuple(len({anchor.independent_block_id for anchor in group}) for group in inliers)
    separation = cosine_distance(centers[0], centers[1]) if k == 2 else None
    if k == 1:
        cluster_support_ok = True
        dispersion_ok = dispersion[0] <= cfg.h1_max_dispersion
        outlier_ratio_ok = outlier_ratio <= cfg.h1_max_outlier_ratio
    else:
        cluster_support_ok = all(count >= cfg.h2_min_independent_anchor_count and speech >= cfg.h2_min_clean_anchor_us for count, speech in zip(independent_count, duration))
        dispersion_ok = all(value <= cfg.h2_max_cluster_dispersion for value in dispersion)
        outlier_ratio_ok = outlier_ratio <= cfg.h2_max_outlier_ratio
    return SpeakerHypothesis(
        k=k,
        centers=centers,
        anchor_labels=labels,
        is_valid=True,
        valid_constraints=cluster_support_ok and dispersion_ok and outlier_ratio_ok,
        robust_cost=robust_cost,
        total_cost=robust_cost,
        cost_components={"robust_cost": robust_cost, "outlier_ratio": outlier_ratio},
        outlier_ratio=outlier_ratio,
        cluster_dispersion=dispersion,
        clean_duration_us=duration,
        independent_anchor_count=independent_count,
        cluster_support_ok=cluster_support_ok,
        dispersion_ok=dispersion_ok,
        outlier_ratio_ok=outlier_ratio_ok,
        separation=separation,
    )


def _farthest(anchor: AnchorEvidence, anchors: Sequence[AnchorEvidence]) -> AnchorEvidence:
    return max(anchors, key=lambda other: (cosine_distance(anchor.vector, other.vector), other.tracklet_id))


def fit_h2_once(anchors: Sequence[AnchorEvidence], cfg: DiarizationConfig | None = None) -> SpeakerHypothesis:
    """Run bounded k=2 fits only. Stability evaluation never calls itself recursively."""

    cfg = cfg or DiarizationConfig()
    if len(anchors) < 2:
        return _invalid_hypothesis(2, "INSUFFICIENT_ANCHORS")
    h1_seed = _weighted_centroid([(anchor.vector, anchor.weight) for anchor in anchors])
    h1 = robust_spherical_fit(anchors, 1, (h1_seed,), cfg)
    if not h1.is_valid:
        return _invalid_hypothesis(2, "H1_SEED_FAILURE")
    highest_weight = max(anchors, key=lambda anchor: (anchor.weight, anchor.tracklet_id))
    residual_anchor = max(anchors, key=lambda anchor: (cosine_distance(anchor.vector, h1.centers[0]), anchor.tracklet_id))
    ordered = sorted(anchors, key=lambda anchor: (anchor.start_us, anchor.tracklet_id))
    starters = [
        highest_weight,
        residual_anchor,
        ordered[0],
        ordered[len(ordered) // 4],
        ordered[len(ordered) // 2],
        ordered[(3 * len(ordered)) // 4],
    ]
    pairs = tuple((starter, _farthest(starter, anchors)) for starter in starters)
    fits: list[SpeakerHypothesis] = []
    seen: set[tuple[str, str]] = set()
    for first, second in pairs[: cfg.bounded_seed_count]:
        key = tuple(sorted((first.tracklet_id, second.tracklet_id)))
        if first.tracklet_id == second.tracklet_id or key in seen:
            continue
        seen.add(key)
        fits.append(robust_spherical_fit(anchors, 2, (first.vector, second.vector), cfg))
    valid = [fit for fit in fits if fit.is_valid]
    structurally_valid = [fit for fit in valid if fit.valid_constraints]
    candidates = structurally_valid or valid
    return min(candidates, key=lambda fit: (fit.robust_cost, fit.centers)) if candidates else _invalid_hypothesis(2, "EMPTY_INLIER_CLUSTER")


def _permutation(base: SpeakerHypothesis, candidate: SpeakerHypothesis) -> tuple[int, int]:
    identity = cosine_distance(base.centers[0], candidate.centers[0]) + cosine_distance(base.centers[1], candidate.centers[1])
    swapped = cosine_distance(base.centers[0], candidate.centers[1]) + cosine_distance(base.centers[1], candidate.centers[0])
    return (0, 1) if identity <= swapped else (1, 0)


def _stability(base: SpeakerHypothesis, anchors: Sequence[AnchorEvidence], cfg: DiarizationConfig) -> tuple[float, float]:
    blocks = sorted({anchor.independent_block_id for anchor in anchors})[: cfg.max_stability_blocks]
    agreements: list[float] = []
    similarities: list[float] = []
    for block in blocks:
        subset = tuple(anchor for anchor in anchors if anchor.independent_block_id != block)
        if len(subset) < 2:
            continue
        refit = fit_h2_once(subset, cfg)
        if not refit.is_valid:
            agreements.append(0.0)
            similarities.append(0.0)
            continue
        mapping = _permutation(base, refit)
        common = [anchor for anchor in subset if base.anchor_labels.get(anchor.tracklet_id) is not None and refit.anchor_labels.get(anchor.tracklet_id) is not None]
        if not common:
            agreements.append(0.0)
        else:
            matched_weight = sum(anchor.weight for anchor in common if base.anchor_labels[anchor.tracklet_id] == mapping[refit.anchor_labels[anchor.tracklet_id]])
            agreements.append(matched_weight / sum(anchor.weight for anchor in common))
        similarities.append(min(1.0 - cosine_distance(base.centers[index], refit.centers[mapping[index]]) for index in (0, 1)))
    return (median(agreements) if agreements else 0.0, median(similarities) if similarities else 0.0)


def _temporal_evidence(base: SpeakerHypothesis, anchors: Sequence[AnchorEvidence], cfg: DiarizationConfig) -> tuple[bool, bool]:
    ordered = sorted((anchor for anchor in anchors if base.anchor_labels.get(anchor.tracklet_id) is not None), key=lambda anchor: (anchor.start_us, anchor.end_us, anchor.tracklet_id))
    labels = [base.anchor_labels[anchor.tracklet_id] for anchor in ordered]
    interleaving = any(left == 0 and right == 1 for left, right in zip(labels, labels[1:])) and any(left == 1 and right == 0 for left, right in zip(labels, labels[1:]))
    conflict = any(left.continuity_group_id == right.continuity_group_id and base.anchor_labels[left.tracklet_id] != base.anchor_labels[right.tracklet_id] and (right.scd_evidence_before or 0.0) < cfg.scd_support_min for left, right in zip(ordered, ordered[1:]))
    return interleaving, conflict


def _third_speaker_risk(base: SpeakerHypothesis, anchors: Sequence[AnchorEvidence], cfg: DiarizationConfig) -> bool:
    residuals = [anchor for anchor in anchors if min(cosine_distance(anchor.vector, center) for center in base.centers) > cfg.third_risk_residual_distance_min]
    if len({anchor.independent_block_id for anchor in residuals}) < 2:
        return False
    center = _weighted_centroid([(anchor.vector, anchor.weight) for anchor in residuals])
    dispersion = sum(anchor.weight * cosine_distance(anchor.vector, center) for anchor in residuals) / sum(anchor.weight for anchor in residuals)
    return dispersion <= cfg.third_risk_max_dispersion and all(cosine_distance(center, speaker) >= cfg.third_risk_min_separation for speaker in base.centers)


def evaluate_h2(anchors: Sequence[AnchorEvidence], cfg: DiarizationConfig | None = None) -> SpeakerHypothesis:
    cfg = cfg or DiarizationConfig()
    base = fit_h2_once(anchors, cfg)
    if not base.is_valid:
        return base
    label_stability, centroid_stability = _stability(base, anchors, cfg)
    interleaving, continuous_conflict = _temporal_evidence(base, anchors, cfg)
    third_risk = _third_speaker_risk(base, anchors, cfg)
    condition_penalty = float(not interleaving) + float(continuous_conflict)
    total_cost = base.robust_cost + cfg.lambda_k2 + cfg.lambda_stability * (1.0 - label_stability) + cfg.lambda_condition * condition_penalty
    reasons: list[str] = []
    if label_stability < cfg.h2_min_label_stability or centroid_stability < cfg.h2_min_centroid_stability:
        reasons.append("H2_UNSTABLE")
    if not interleaving:
        reasons.append("H2_NO_INTERLEAVING_EVIDENCE")
    if continuous_conflict:
        reasons.append("H2_CONTINUOUS_SPEECH_CONFLICT")
    if third_risk:
        reasons.append("THIRD_SPEAKER_RISK")
    return replace(base, total_cost=total_cost, cost_components={**base.cost_components, "stability": label_stability, "condition_penalty": condition_penalty}, label_stability=label_stability, centroid_stability=centroid_stability, temporal_interleaving=interleaving, continuous_speech_conflict=continuous_conflict, third_speaker_risk=third_risk, reason_codes=tuple(reasons))


def evaluate_hypotheses(anchors: Sequence[AnchorEvidence], cfg: DiarizationConfig | None = None) -> HypothesisEvaluation:
    """Evaluate H1 and H2 once and return the decision with H2 diagnostics.

    This is the preferred seam for callers that need both the selected
    decision and the H2 diagnostic payload.  It deliberately delegates to
    the existing fit/evidence functions without changing any thresholds or
    scoring rules.  ``choose_hypothesis`` remains the compatibility wrapper
    for callers that only need the decision.
    """

    cfg = cfg or DiarizationConfig()
    if anchors:
        h1 = robust_spherical_fit(anchors, 1, (_weighted_centroid([(anchor.vector, anchor.weight) for anchor in anchors]),), cfg)
    else:
        # Keep the empty-input decision and its reason code identical to the
        # historical choose_hypothesis path while still returning complete
        # diagnostics from this API.
        h1 = _invalid_hypothesis(1, "INSUFFICIENT_ANCHORS")
    h2 = evaluate_h2(anchors, cfg)
    if not anchors:
        decision = HypothesisDecision("UNCERTAIN_1_OR_2", None, ("INSUFFICIENT_CLEAN_ANCHORS",))
        return HypothesisEvaluation(decision=decision, h1=h1, h2=h2)
    h1_confident = h1.is_valid and h1.valid_constraints
    h2_acoustic_candidate = h2.is_valid and h2.separation is not None and h2.separation >= cfg.h2_min_separation and h1.robust_cost - h2.total_cost >= cfg.h2_min_cost_gain
    h2_confirmed = h2_acoustic_candidate and all((h2.cluster_support_ok, h2.dispersion_ok, h2.outlier_ratio_ok, (h2.label_stability or 0.0) >= cfg.h2_min_label_stability, (h2.centroid_stability or 0.0) >= cfg.h2_min_centroid_stability, h2.temporal_interleaving is True, not h2.continuous_speech_conflict, not h2.third_speaker_risk))
    if h2_confirmed:
        decision = HypothesisDecision("H2_CONFIRMED", h2, h2.reason_codes)
    elif h2_acoustic_candidate:
        decision = HypothesisDecision("UNCERTAIN_1_OR_2", None, h2.reason_codes)
    elif h1_confident:
        decision = HypothesisDecision("H1_CONFIRMED", h1, h1.reason_codes)
    else:
        decision = HypothesisDecision("UNCERTAIN_1_OR_2", None, tuple(dict.fromkeys(h1.reason_codes + h2.reason_codes)))
    return HypothesisEvaluation(decision=decision, h1=h1, h2=h2)


def choose_hypothesis(anchors: Sequence[AnchorEvidence], cfg: DiarizationConfig | None = None) -> HypothesisDecision:
    """Return the historical decision-only API.

    The implementation is intentionally a thin wrapper around
    :func:`evaluate_hypotheses`, so both public seams use exactly the same
    deterministic H1/H2 evaluation.
    """

    return evaluate_hypotheses(anchors, cfg).decision


def speaker_states_from_decision(decision: HypothesisDecision, anchors: Sequence[AnchorEvidence]) -> Mapping[str, SpeakerState]:
    if decision.hypothesis is None:
        return {}
    hypothesis = decision.hypothesis
    earliest: dict[int, int] = {}
    for anchor in anchors:
        label = hypothesis.anchor_labels.get(anchor.tracklet_id)
        if label is not None:
            earliest[label] = min(earliest.get(label, anchor.start_us), anchor.start_us)
    order = sorted(range(hypothesis.k), key=lambda label: (earliest.get(label, INF), label))
    states: dict[str, SpeakerState] = {}
    for output_index, cluster_index in enumerate(order):
        selected = [anchor for anchor in anchors if hypothesis.anchor_labels.get(anchor.tracklet_id) == cluster_index]
        states[SPEAKER_IDS[output_index]] = SpeakerState(SPEAKER_IDS[output_index], hypothesis.centers[cluster_index], tuple(anchor.tracklet_id for anchor in selected), hypothesis.cluster_dispersion[cluster_index])
    return states


def _role_thresholds(role: str, cfg: DiarizationConfig) -> tuple[float, float, float]:
    prefix = role.lower()
    return (float(getattr(cfg, f"{prefix}_stable_distance_ceiling")), float(getattr(cfg, f"{prefix}_absolute_distance_max")), float(getattr(cfg, f"{prefix}_margin_min")))


def _state_distances(vector: Sequence[float], state: SpeakerState, cfg: DiarizationConfig) -> tuple[float, float]:
    """Return stable and bounded effective distance without replacing stable evidence."""

    stable = cosine_distance(vector, state.stable_anchor_centroid)
    if not cfg.enable_recent_centroid or state.recent_centroid is None:
        return stable, stable
    recent = cosine_distance(vector, state.recent_centroid)
    benefit = min(cfg.recent_max_benefit, max(0.0, stable - recent))
    return stable, stable - cfg.recent_blend * benefit


def _assignment(tracklet: Tracklet, speaker_id: str, status: str, *, stable_distance: float | None = None, effective_distance: float | None = None, margin: float | None = None, reason_codes: Sequence[str] = ()) -> SpeakerAssignment:
    return SpeakerAssignment(tracklet.tracklet_id, speaker_id, status, stable_distance, effective_distance, margin, tracklet.boundary_evidence_ids, tuple(reason_codes))  # type: ignore[arg-type]


def local_assignment(tracklet: Tracklet, embedding: EmbeddingResult | None, states: Mapping[str, SpeakerState], decision: HypothesisDecision, cfg: DiarizationConfig | None = None) -> SpeakerAssignment:
    cfg = cfg or DiarizationConfig()
    if tracklet.protected_overlap:
        return _assignment(tracklet, "OVERLAP", "OVERLAP", reason_codes=("PROTECTED_OVERLAP",))
    if decision.state not in {"H1_CONFIRMED", "H2_CONFIRMED"}:
        return _assignment(tracklet, "UNKNOWN", "UNKNOWN_INSUFFICIENT_EVIDENCE", reason_codes=("HYPOTHESIS_UNCONFIRMED",))
    if tracklet.mixed_tracklet_suspect or embedding is None or not embedding.is_valid or embedding.vector is None:
        return _assignment(tracklet, "UNKNOWN", "UNKNOWN_INSUFFICIENT_EVIDENCE", reason_codes=("MIXED_OR_INVALID_EMBEDDING",))
    ordered_states = tuple(sorted(states.items()))
    if not ordered_states:
        return _assignment(tracklet, "UNKNOWN", "UNKNOWN_INSUFFICIENT_EVIDENCE", reason_codes=("NO_SPEAKER_STATE",))
    scores = sorted(
        ((speaker_id, *_state_distances(embedding.vector, state, cfg)) for speaker_id, state in ordered_states),
        key=lambda item: (item[2], item[0]),
    )
    speaker_id, stable_distance, effective_distance = scores[0]
    margin = scores[1][2] - effective_distance if len(scores) > 1 else None
    stable_ceiling, absolute_limit, margin_min = _role_thresholds(tracklet.kind, cfg)
    if stable_distance > stable_ceiling or effective_distance > absolute_limit or (margin is not None and margin < margin_min):
        return _assignment(tracklet, "UNKNOWN", "UNKNOWN_INSUFFICIENT_EVIDENCE", stable_distance=stable_distance, effective_distance=effective_distance, margin=margin, reason_codes=("LOCAL_GATE_FAILED",))
    return _assignment(tracklet, speaker_id, "LOCAL_CANDIDATE", stable_distance=stable_distance, effective_distance=effective_distance, margin=margin)


def maybe_update_recent(
    state: SpeakerState,
    tracklet: Tracklet,
    embedding: EmbeddingResult | None,
    assignment: SpeakerAssignment,
    opponent: SpeakerState | None,
    cfg: DiarizationConfig | None = None,
) -> SpeakerState:
    """Apply the P2 bounded recent-centroid rule or leave state unchanged.

    This function never mutates the stable centroid and never accepts MICRO,
    overlap, mixed, invalid, low-margin, or poor-stable-fit evidence.
    """

    cfg = cfg or DiarizationConfig()
    if not cfg.enable_recent_centroid or state.recent_frozen:
        return state
    if (
        tracklet.kind == "MICRO" or tracklet.protected_overlap or tracklet.mixed_tracklet_suspect
        or embedding is None or not embedding.is_valid or embedding.vector is None
        or assignment.speaker_id != state.speaker_id
        or assignment.margin is None or assignment.margin < cfg.recent_update_margin_min
        or assignment.stable_distance is None or assignment.stable_distance > cfg.recent_update_stable_fit_max
        or embedding.quality < cfg.recent_update_quality_min
    ):
        return state
    old = state.recent_centroid or state.stable_anchor_centroid
    proposal = _weighted_centroid(((old, cfg.recent_decay), (embedding.vector, 1.0)))
    if cosine_distance(proposal, state.stable_anchor_centroid) > cfg.recent_radius_max:
        return replace(state, recent_frozen=True, drift_flags=state.drift_flags | frozenset({"RECENT_UPDATE_FROZEN"}))
    if opponent is not None:
        own = cosine_distance(proposal, state.stable_anchor_centroid)
        other = cosine_distance(proposal, opponent.stable_anchor_centroid)
        if other - own < cfg.recent_opponent_margin_min:
            return state
    return replace(
        state,
        recent_centroid=proposal,
        recent_mass=state.recent_mass * cfg.recent_decay + 1.0,
        recent_last_us=tracklet.end_us,
    )


def refine_recent_states(
    tracklets: Sequence[Tracklet],
    embeddings: Sequence[EmbeddingResult],
    states: Mapping[str, SpeakerState],
    decision: HypothesisDecision,
    cfg: DiarizationConfig | None = None,
) -> Mapping[str, SpeakerState]:
    """Optionally perform a bounded P2 recent-centroid pass before final Viterbi."""

    cfg = cfg or DiarizationConfig()
    if not cfg.enable_recent_centroid:
        return dict(states)
    current = dict(states)
    by_tracklet = {embedding.tracklet_id: embedding for embedding in embeddings}
    for tracklet in sorted(tracklets, key=lambda item: (item.start_us, item.end_us, item.tracklet_id)):
        assignment = local_assignment(tracklet, by_tracklet.get(tracklet.tracklet_id), current, decision, cfg)
        if assignment.speaker_id not in current:
            continue
        opponent = next((state for speaker, state in current.items() if speaker != assignment.speaker_id), None)
        current[assignment.speaker_id] = maybe_update_recent(
            current[assignment.speaker_id], tracklet, by_tracklet.get(tracklet.tracklet_id), assignment, opponent, cfg
        )
    return current


def re_evaluate_micro(tracklet: Tracklet, embedding: EmbeddingResult | None, states: Mapping[str, SpeakerState], decision: HypothesisDecision, cfg: DiarizationConfig | None = None) -> SpeakerAssignment:
    assignment = local_assignment(tracklet, embedding, states, decision, cfg)
    if assignment.speaker_id == "UNKNOWN":
        return replace(assignment, attribution_status="UNKNOWN_SHORT")
    if assignment.speaker_id == "OVERLAP":
        return assignment
    return replace(assignment, attribution_status="CANDIDATE_SPEAKER")


def _emission(tracklet: Tracklet, candidate: str, assignment: SpeakerAssignment, decision: HypothesisDecision, cfg: DiarizationConfig) -> float:
    if candidate in SPEAKER_IDS:
        return assignment.effective_distance if assignment.speaker_id == candidate and assignment.effective_distance is not None else INF
    if candidate == "UNKNOWN":
        return cfg.unknown_micro_cost if tracklet.kind == "MICRO" else cfg.unknown_cost
    if candidate == "OVERLAP":
        return 0.0 if tracklet.protected_overlap else INF
    if candidate == "OTHER":
        return cfg.other_cost if decision.hypothesis and decision.hypothesis.third_speaker_risk else INF
    return INF


def _soft_speaker_emissions(
    tracklet: Tracklet,
    embedding: EmbeddingResult | None,
    states: Mapping[str, SpeakerState],
    decision: HypothesisDecision,
    cfg: DiarizationConfig,
) -> Mapping[str, float]:
    """Return bounded finite emissions for every eligible speaker state.

    This is an opt-in experiment seam.  It never expands a role's stable or
    absolute distance ceiling and never makes an invalid, mixed, overlap, or
    unconfirmed tracklet assignable.  Unlike the historical hard emission it
    does not discard the second speaker solely because the local margin gate
    selected another label (or abstained); sequence evidence may compare both
    strictly bounded distances against UNKNOWN.
    """

    result = {speaker_id: INF for speaker_id in states}
    if (
        tracklet.protected_overlap
        or tracklet.mixed_tracklet_suspect
        or decision.state not in {"H1_CONFIRMED", "H2_CONFIRMED"}
        or embedding is None
        or not embedding.is_valid
        or embedding.vector is None
    ):
        return result
    stable_ceiling, absolute_limit, _ = _role_thresholds(tracklet.kind, cfg)
    for speaker_id, state in sorted(states.items()):
        stable_distance, effective_distance = _state_distances(embedding.vector, state, cfg)
        if stable_distance <= stable_ceiling and effective_distance <= absolute_limit:
            result[speaker_id] = effective_distance
    return result


def _transition(previous: str, current: str, previous_tracklet: Tracklet, current_tracklet: Tracklet, cfg: DiarizationConfig) -> float:
    gap_us = max(0, current_tracklet.start_us - previous_tracklet.end_us)
    if gap_us >= cfg.long_gap_reset_us:
        return 0.0
    if previous == current:
        return 0.0
    if previous in SPEAKER_IDS and current in SPEAKER_IDS:
        scd = current_tracklet.scd_evidence_before or 0.0
        relief = cfg.scd_relief * scd + cfg.gap_relief * min(1.0, gap_us / max(1, cfg.long_gap_reset_us))
        return max(0.0, cfg.switch_base - relief)
    return cfg.uncertainty_transition


def _materialize(labels: Sequence[str], tracklets: Sequence[Tracklet], protected_overlap_spans: Sequence[ProtectedOverlapSpan], source_duration_us: int, cfg: DiarizationConfig) -> tuple[DiarizationSpan, ...]:
    spans: list[DiarizationSpan] = []
    for ordinal, (label, tracklet) in enumerate(zip(labels, tracklets)):
        status = "UNKNOWN_SHORT" if label == "UNKNOWN" and tracklet.kind == "MICRO" else ("OVERLAP" if label == "OVERLAP" else "ASSIGNED" if label in SPEAKER_IDS else "UNKNOWN_INSUFFICIENT_EVIDENCE")
        spans.append(DiarizationSpan(_stable_id("span", tracklet.start_us, tracklet.end_us, label, ordinal), tracklet.start_us, tracklet.end_us, label, status, tracklet.boundary_evidence_ids))  # type: ignore[arg-type]
    for ordinal, overlap in enumerate(protected_overlap_spans):
        spans.append(DiarizationSpan(_stable_id("span", overlap.start_us, overlap.end_us, "OVERLAP", ordinal), overlap.start_us, overlap.end_us, "OVERLAP", "OVERLAP", overlap.evidence_ids, ("PROTECTED_OVERLAP",)))
    spans.sort(key=lambda span: (span.start_us, span.end_us, span.speaker_id))
    for left, right in zip(spans, spans[1:]):
        if right.start_us < left.end_us:
            raise ContractValidationError("final diarization spans overlap")
    if cfg.include_non_speech:
        with_gaps: list[DiarizationSpan] = []
        cursor = 0
        for span in spans:
            if cursor < span.start_us:
                with_gaps.append(DiarizationSpan(_stable_id("nonspeech", cursor, span.start_us), cursor, span.start_us, "NON_SPEECH", "NON_SPEECH"))
            with_gaps.append(span)
            cursor = span.end_us
        if cursor < source_duration_us:
            with_gaps.append(DiarizationSpan(_stable_id("nonspeech", cursor, source_duration_us), cursor, source_duration_us, "NON_SPEECH", "NON_SPEECH"))
        spans = with_gaps
    merged: list[DiarizationSpan] = []
    for span in spans:
        if merged and merged[-1].end_us == span.start_us and merged[-1].speaker_id == span.speaker_id and merged[-1].attribution_status == span.attribution_status:
            previous = merged[-1]
            merged[-1] = DiarizationSpan(previous.span_id, previous.start_us, span.end_us, previous.speaker_id, previous.attribution_status, previous.evidence_ids + span.evidence_ids, previous.reason_codes + span.reason_codes)
        else:
            merged.append(span)
    return tuple(merged)


def decode_sequence(
    tracklets: Sequence[Tracklet],
    protected_overlap_spans: Sequence[ProtectedOverlapSpan],
    states: Mapping[str, SpeakerState],
    decision: HypothesisDecision,
    source_duration_us: int,
    cfg: DiarizationConfig | None = None,
    embeddings: Sequence[EmbeddingResult] = (),
    *,
    soft_speaker_emissions: bool = False,
) -> SequenceDecodeResult:
    """Decode once and retain a scalar-only trace for opt-in ablations.

    With ``soft_speaker_emissions=False`` this is the historical decoder byte
    for byte: only the locally selected speaker has a finite emission.  The
    experimental mode gives all strictly in-ceiling speaker states their finite
    distance while keeping the same UNKNOWN, transition, overlap, and isolated
    run rules.
    """

    cfg = cfg or DiarizationConfig()
    if not tracklets:
        spans = _materialize((), (), protected_overlap_spans, source_duration_us, cfg)
        return SequenceDecodeResult(spans, (), ())
    by_tracklet = {embedding.tracklet_id: embedding for embedding in embeddings}
    local_assignments = tuple(
        re_evaluate_micro(tracklet, by_tracklet.get(tracklet.tracklet_id), states, decision, cfg)
        if tracklet.kind == "MICRO"
        else local_assignment(tracklet, by_tracklet.get(tracklet.tracklet_id), states, decision, cfg)
        for tracklet in tracklets
    )
    local = {
        tracklet.tracklet_id: assignment
        for tracklet, assignment in zip(tracklets, local_assignments)
    }
    soft = (
        {
            tracklet.tracklet_id: _soft_speaker_emissions(
                tracklet, by_tracklet.get(tracklet.tracklet_id), states, decision, cfg
            )
            for tracklet in tracklets
        }
        if soft_speaker_emissions
        else {}
    )
    candidates = tuple(sorted(states)) + ("UNKNOWN", "OVERLAP", "OTHER")
    dp: list[dict[str, float]] = []
    back: list[dict[str, str | None]] = []
    for index, tracklet in enumerate(tracklets):
        current: dict[str, float] = {}
        pointers: dict[str, str | None] = {}
        for candidate in candidates:
            if soft_speaker_emissions and candidate in states:
                emission = soft[tracklet.tracklet_id][candidate]
            else:
                emission = _emission(tracklet, candidate, local[tracklet.tracklet_id], decision, cfg)
            if index == 0:
                current[candidate] = emission
                pointers[candidate] = None
            else:
                options = [(dp[index - 1][previous] + _transition(previous, candidate, tracklets[index - 1], tracklet, cfg) + emission, previous) for previous in candidates]
                current[candidate], pointers[candidate] = min(options, key=lambda item: (item[0], 0 if item[1] == "UNKNOWN" else 1, item[1]))
        dp.append(current)
        back.append(pointers)
    last = min(candidates, key=lambda candidate: (dp[-1][candidate], 0 if candidate == "UNKNOWN" else 1, candidate))
    labels = [last]
    for index in range(len(tracklets) - 1, 0, -1):
        previous = back[index][labels[-1]]
        labels.append(previous if previous is not None else "UNKNOWN")
    labels.reverse()
    for index in range(1, len(labels) - 1):
        if labels[index] in SPEAKER_IDS and labels[index - 1] == labels[index + 1] and labels[index] != labels[index - 1] and tracklets[index].end_us - tracklets[index].start_us <= cfg.isolated_run_max_us:
            labels[index] = "UNKNOWN"
    final_labels = tuple(labels)
    spans = _materialize(final_labels, tracklets, protected_overlap_spans, source_duration_us, cfg)
    return SequenceDecodeResult(spans, final_labels, local_assignments)


def finalize_sequence(tracklets: Sequence[Tracklet], protected_overlap_spans: Sequence[ProtectedOverlapSpan], states: Mapping[str, SpeakerState], decision: HypothesisDecision, source_duration_us: int, cfg: DiarizationConfig | None = None, embeddings: Sequence[EmbeddingResult] = ()) -> tuple[DiarizationSpan, ...]:
    """Finalize with the historical hard-emission decoder."""

    return decode_sequence(
        tracklets,
        protected_overlap_spans,
        states,
        decision,
        source_duration_us,
        cfg,
        embeddings,
    ).spans


__all__ = ["DiarizationConfig", "HypothesisEvaluation", "SequenceDecodeResult", "build_tracklets", "select_anchor_evidence", "robust_spherical_fit", "fit_h2_once", "evaluate_h2", "evaluate_hypotheses", "choose_hypothesis", "speaker_states_from_decision", "local_assignment", "maybe_update_recent", "refine_recent_states", "re_evaluate_micro", "decode_sequence", "finalize_sequence", "cosine_distance"]
