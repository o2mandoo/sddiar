"""Opt-in UNKNOWN-only graph rescue experiment.

This module is intentionally isolated from :mod:`sddiar.diarization`.  It is
an experiment for measuring whether a small, sparse embedding graph can
reduce UNKNOWN output after the existing H2 decision.  The graph is
*anchor-clamped*: labels which already have a speaker assignment are never
changed, and only baseline ``UNKNOWN`` tracklets can receive a candidate.

The module has no SciPy (or runtime) dependency.  When optional NumPy is
available, chunked matrix multiplication accelerates distance discovery
without allocating an NxN matrix; otherwise a deterministic bounded
pure-Python fallback is used for small fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from math import exp, isfinite, sqrt
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SPEAKER_IDS = ("SPEAKER_00", "SPEAKER_01")
GRAPH_RESCUE_ALGORITHM_VERSION = "anchor-clamped-fixedpoint-v2"
_VECTOR_SCALE = 1_000_000
_DOT_SCALE = _VECTOR_SCALE * _VECTOR_SCALE
ALLOWED_LABELS = frozenset((*SPEAKER_IDS, "UNKNOWN", "OVERLAP", "OTHER", "NON_SPEECH"))
DECISION_STATES = frozenset(("H1_CONFIRMED", "H2_CONFIRMED", "UNCERTAIN_1_OR_2"))


class GraphRescueError(ValueError):
    """Base error for invalid graph-rescue inputs or policy."""


class GraphRescueResourceError(GraphRescueError):
    """The configured graph resource budget would be exceeded."""


@dataclass(frozen=True, slots=True)
class GraphRescueConfig:
    """Bounded policy for the experiment; omission means disabled."""

    enabled: bool = False
    k_neighbors: int = 4
    adjacency_mode: str = "mutual_knn"
    propagation_steps: int = 1
    # These are deliberately explicit experiment bounds.  The production
    # path never calls this module, but real review fixtures contain roughly
    # 800--1,000 tracklets.
    max_nodes: int = 1_200
    max_edges: int = 12_000
    max_dimension: int = 512
    max_distance_evaluations: int = 1_500_000
    distance_chunk_nodes: int = 128
    max_edge_distance: float = 0.75
    posterior_temperature: float = 0.15
    min_posterior: float = 0.60
    posterior_margin_min: float = 0.15
    leave_block_margin_min: float = 0.05
    min_anchor_blocks: int = 2

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise GraphRescueError("enabled must be a boolean")
        for name in (
            "k_neighbors", "propagation_steps", "max_nodes", "max_edges",
            "max_dimension", "max_distance_evaluations", "min_anchor_blocks",
            "distance_chunk_nodes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise GraphRescueError(f"{name} must be a positive integer")
        if self.adjacency_mode not in {"mutual_knn", "bounded_knn"}:
            raise GraphRescueError("adjacency_mode must be mutual_knn or bounded_knn")
        for name in (
            "max_edge_distance", "posterior_temperature", "min_posterior",
            "posterior_margin_min", "leave_block_margin_min",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise GraphRescueError(f"{name} must be finite")
        if self.max_edge_distance < 0:
            raise GraphRescueError("max_edge_distance must be non-negative")
        if self.posterior_temperature <= 0:
            raise GraphRescueError("posterior_temperature must be positive")
        if not 0 <= self.min_posterior <= 1:
            raise GraphRescueError("min_posterior must be in [0, 1]")
        if not 0 <= self.posterior_margin_min <= 1:
            raise GraphRescueError("posterior_margin_min must be in [0, 1]")
        if not 0 <= self.leave_block_margin_min <= 1:
            raise GraphRescueError("leave_block_margin_min must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class GraphRescueCandidate:
    """An accepted UNKNOWN replacement, with all safety gates observable."""

    tracklet_id: str
    speaker_id: str
    posterior: float
    margin: float
    supporting_anchor_blocks: tuple[str, ...]
    leave_block_margins: tuple[tuple[str, float], ...]
    leave_block_stable: bool
    neighbor_count: int
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphRescueResult:
    """Immutable output; ``labels`` is a copy and never aliases caller data."""

    labels: tuple[str, ...]
    candidates: tuple[GraphRescueCandidate, ...]
    applied_count: int
    diagnostics: Mapping[str, Any]


def _freeze(value: Any) -> Any:
    """Recursively freeze diagnostics before exposing them to callers."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _tracklet_id(value: Any) -> str:
    result = _get(value, "tracklet_id", None)
    if not isinstance(result, str) or not result:
        raise GraphRescueError("every tracklet needs a non-empty tracklet_id")
    return result


def _normalise(vector: Any, max_dimension: int) -> tuple[float, ...] | None:
    if vector is None:
        return None
    if isinstance(vector, Mapping):
        vector = vector.get("vector")
    if isinstance(vector, (str, bytes, bytearray)):
        return None
    try:
        raw_values = tuple(islice(iter(vector), max_dimension + 1))
    except TypeError:
        return None
    if len(raw_values) > max_dimension:
        raise GraphRescueResourceError("embedding dimension exceeds max_dimension")
    try:
        values = tuple(float(item) for item in raw_values)
    except (TypeError, ValueError, OverflowError):
        return None
    if not values or not all(isfinite(item) for item in values):
        return None
    norm = sqrt(sum(item * item for item in values))
    if norm <= 1e-12:
        return None
    return tuple(item / norm for item in values)


def _embedding_vector(value: Any) -> Any:
    # EmbeddingResult and simple mapping fixtures both work.  Invalid
    # EmbeddingResult values are deliberately ignored, as they cannot support
    # a rescue candidate.
    if isinstance(value, Mapping):
        if value.get("is_valid", True) is False:
            return None
        return value.get("vector")
    if hasattr(value, "is_valid") and not bool(getattr(value, "is_valid")):
        return None
    return getattr(value, "vector", value)


def _decision_state(decision: Any, explicit: Any) -> str | None:
    value = explicit if explicit is not None else decision
    if value is None:
        return None
    state = getattr(value, "state", value)
    return str(state)


def _resolve_decision(
    decision: Any,
    explicit_state: str | None,
    h2_confirmed: bool | None,
) -> tuple[str | None, bool]:
    """Resolve one decision authority and reject contradictory hints."""

    if h2_confirmed is not None and type(h2_confirmed) is not bool:
        raise GraphRescueError("h2_confirmed must be a boolean when provided")
    decision_value = _decision_state(decision, None)
    explicit_value = _decision_state(None, explicit_state)
    if decision_value is not None and decision_value not in DECISION_STATES:
        raise GraphRescueError("unknown hypothesis decision state")
    if explicit_value is not None and explicit_value not in DECISION_STATES:
        raise GraphRescueError("unknown hypothesis decision state")
    if decision_value is not None and explicit_value is not None and decision_value != explicit_value:
        raise GraphRescueError("decision and decision_state conflict")
    state = decision_value or explicit_value
    if h2_confirmed is not None and state is not None and h2_confirmed != (state == "H2_CONFIRMED"):
        raise GraphRescueError("h2_confirmed conflicts with hypothesis decision")
    return state, (h2_confirmed if h2_confirmed is not None else state == "H2_CONFIRMED")


def _quantize_vector(vector: Sequence[float]) -> tuple[int, ...]:
    """Convert a normalized vector to a cross-backend fixed-point form."""

    return tuple(int(round(float(value) * _VECTOR_SCALE)) for value in vector)


def _distance_score(left: Sequence[int], right: Sequence[int]) -> int:
    """Return deterministic fixed-point cosine-distance numerator."""

    return max(0, _DOT_SCALE - sum(a * b for a, b in zip(left, right)))


def _numpy_module() -> Any:
    """Load NumPy only when installed; SciPy is intentionally not used."""

    try:
        import numpy  # type: ignore[import-not-found]
    except ImportError:
        return None
    return numpy


def _posterior(scores: Mapping[str, float]) -> tuple[str | None, float, float]:
    total = sum(max(0.0, value) for value in scores.values())
    if total <= 1e-12:
        return None, 0.0, 0.0
    ordered = sorted(((speaker, max(0.0, score) / total) for speaker, score in scores.items()), key=lambda item: (-item[1], item[0]))
    top, probability = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    return top, probability, probability - second


def _speaker_classes(labels: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(sorted(label for label in set(labels.values()) if label in SPEAKER_IDS))


def _anchor_scores(
    node: str,
    adjacency: Mapping[str, Sequence[tuple[str, float]]],
    labels: Mapping[str, str],
    blocks: Mapping[str, str],
    removed_block: str | None = None,
) -> dict[str, float]:
    """One-step raw scores from clamped anchors only."""

    scores = {speaker: 0.0 for speaker in _speaker_classes(labels)}
    for neighbor, weight in adjacency.get(node, ()):
        speaker = labels.get(neighbor)
        if speaker not in scores:
            continue
        if removed_block is not None and blocks.get(neighbor) == removed_block:
            continue
        scores[speaker] += weight
    return scores


def _normalised_scores(scores: Mapping[str, float]) -> Mapping[str, float]:
    total = sum(max(0.0, value) for value in scores.values())
    if total <= 1e-12:
        return {}
    return {speaker: max(0.0, value) / total for speaker, value in scores.items()}


def _build_adjacency(
    node_ids: Sequence[str],
    vectors: Mapping[str, tuple[float, ...]],
    cfg: GraphRescueConfig,
    *,
    use_numpy: bool | None = None,
) -> tuple[Mapping[str, tuple[tuple[str, float], ...]], int, int]:
    n = len(node_ids)
    evaluations = n * max(0, n - 1)
    if evaluations > cfg.max_distance_evaluations:
        raise GraphRescueResourceError("distance evaluation budget exceeded")
    numpy = _numpy_module() if use_numpy is not False else None
    quantized = {node: _quantize_vector(vectors[node]) for node in node_ids}
    edge_distance_limit = int(round(cfg.max_edge_distance * _DOT_SCALE))
    directed: dict[str, tuple[str, ...]] = {}
    if numpy is not None:
        # Keep the NxN distance matrix out of memory.  A float64 chunk for
        # 128x1200 is about 1.2 MiB; only the chunk and the transposed vector
        # matrix are live at once.  Sorting remains Python-side so ties have
        # exactly the same stable (distance, tracklet_id) order as fallback.
        matrix = numpy.asarray([quantized[node] for node in node_ids], dtype=numpy.int64)
        for start in range(0, n, cfg.distance_chunk_nodes):
            stop = min(n, start + cfg.distance_chunk_nodes)
            similarities = matrix[start:stop].dot(matrix.T)
            for offset, left in enumerate(node_ids[start:stop]):
                nearest = [
                    (max(0, _DOT_SCALE - int(similarities[offset, index])), node_ids[index])
                    for index in range(n)
                    if index != start + offset
                    and max(0, _DOT_SCALE - int(similarities[offset, index])) <= edge_distance_limit
                ]
                nearest.sort(key=lambda item: (item[0], item[1]))
                directed[left] = tuple(right for _, right in nearest[: cfg.k_neighbors])
        del matrix
    else:
        for left in node_ids:
            nearest: list[tuple[float, str]] = []
            for right in node_ids:
                if left == right:
                    continue
                distance_score = _distance_score(quantized[left], quantized[right])
                if distance_score <= edge_distance_limit:
                    nearest.append((distance_score, right))
            nearest.sort(key=lambda item: (item[0], item[1]))
            directed[left] = tuple(right for _, right in nearest[: cfg.k_neighbors])

    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in node_ids}
    for left in node_ids:
        for right in directed[left]:
            if cfg.adjacency_mode == "mutual_knn" and left not in directed[right]:
                continue
            distance_score = _distance_score(quantized[left], quantized[right])
            distance = distance_score / _DOT_SCALE
            weight = exp(-max(0.0, distance) / cfg.posterior_temperature)
            adjacency[left].append((right, weight))
    edge_count = sum(len(items) for items in adjacency.values())
    if edge_count > cfg.max_edges:
        raise GraphRescueResourceError("sparse graph edge budget exceeded")
    return MappingProxyType({key: tuple(value) for key, value in adjacency.items()}), edge_count, evaluations


def _run_propagation(
    node_ids: Sequence[str],
    adjacency: Mapping[str, Sequence[tuple[str, float]]],
    labels: Mapping[str, str],
    removed_block: str | None,
    blocks: Mapping[str, str],
    steps: int,
) -> Mapping[str, Mapping[str, float]]:
    # UNKNOWN is a graph node state, never a propagated class.  Including it
    # here would let an unknown prior absorb its own probability and defeat
    # the anchor-clamped rescue gate.
    speakers = _speaker_classes(labels)
    probabilities: dict[str, dict[str, float]] = {}
    for node in node_ids:
        label = labels.get(node)
        if label in speakers:
            if removed_block is not None and blocks.get(node) == removed_block:
                probabilities[node] = {}
            else:
                probabilities[node] = {speaker: float(speaker == label) for speaker in speakers}
        else:
            probabilities[node] = {}
    for _ in range(steps):
        updated = {node: dict(value) for node, value in probabilities.items()}
        for node in node_ids:
            if labels.get(node) != "UNKNOWN":
                continue
            scores = {speaker: 0.0 for speaker in speakers}
            for neighbor, weight in adjacency.get(node, ()):
                if not probabilities.get(neighbor):
                    continue
                for speaker in speakers:
                    scores[speaker] += weight * probabilities[neighbor].get(speaker, 0.0)
            total = sum(scores.values())
            updated[node] = ({speaker: score / total for speaker, score in scores.items()} if total > 1e-12 else {})
        probabilities = updated
    return MappingProxyType({node: MappingProxyType(dict(value)) for node, value in probabilities.items()})


def rescue_unknowns(
    tracklets: Sequence[Any],
    baseline_labels: Sequence[str],
    embeddings: Mapping[str, Any] | Sequence[Any],
    *,
    decision: Any = None,
    decision_state: str | None = None,
    h2_confirmed: bool | None = None,
    anchor_block_ids: Mapping[str, str] | None = None,
    seed_tracklet_ids: Sequence[str] | None = None,
    config: GraphRescueConfig | None = None,
) -> GraphRescueResult:
    """Return an opt-in rescue result without changing caller-owned objects.

    Candidate computation is deliberately skipped unless both ``enabled`` and
    ``H2_CONFIRMED`` are true.  Existing speaker and ``OVERLAP`` labels are
    anchor-clamped; a candidate can only replace a baseline ``UNKNOWN``.
    """

    cfg = config or GraphRescueConfig()
    if len(tracklets) > cfg.max_nodes or len(baseline_labels) > cfg.max_nodes:
        raise GraphRescueResourceError("tracklet or label count exceeds max_nodes")
    original_labels = tuple(baseline_labels)
    if any(not isinstance(label, str) or label not in ALLOWED_LABELS for label in original_labels):
        raise GraphRescueError("baseline_labels contains an unsupported label")
    ids = tuple(_tracklet_id(item) for item in tracklets)
    if len(ids) != len(original_labels):
        raise GraphRescueError("tracklets and baseline_labels must have equal length")
    if len(set(ids)) != len(ids):
        raise GraphRescueError("tracklet IDs must be unique")
    state, h2 = _resolve_decision(decision, decision_state, h2_confirmed)
    explicit_seed_ids: frozenset[str] | None = None
    if seed_tracklet_ids is not None:
        if len(seed_tracklet_ids) > cfg.max_nodes:
            raise GraphRescueResourceError("seed tracklet count exceeds max_nodes")
        requested = tuple(seed_tracklet_ids)
        if any(not isinstance(item, str) or not item for item in requested):
            raise GraphRescueError("seed_tracklet_ids must contain non-empty strings")
        if len(set(requested)) != len(requested):
            raise GraphRescueError("seed_tracklet_ids must be unique")
        missing = set(requested).difference(ids)
        if missing:
            raise GraphRescueError("seed_tracklet_ids contains an unknown tracklet")
        explicit_seed_ids = frozenset(requested)
    base_diagnostics = {
        "enabled": cfg.enabled,
        "decision_state": state,
        "h2_confirmed": bool(h2),
        "candidate_computation": False,
        "node_count": 0,
        "edge_count": 0,
        "distance_evaluations": 0,
        "resource_bounds": {
            "max_nodes": cfg.max_nodes,
            "max_edges": cfg.max_edges,
            "max_dimension": cfg.max_dimension,
            "max_distance_evaluations": cfg.max_distance_evaluations,
            "distance_chunk_nodes": cfg.distance_chunk_nodes,
        },
        "seed_eligibility": {
            "mode": "EXPLICIT_IDS" if explicit_seed_ids is not None else "ANCHOR_ONLY",
            "eligible_seed_count": 0,
            "anchor_seed_count": 0,
            "support_seed_count": 0,
            "excluded_assigned_count": 0,
            "explicit_requested_count": len(explicit_seed_ids or ()),
        },
    }
    if not cfg.enabled or not h2:
        base_diagnostics["skip_reason"] = "DISABLED" if not cfg.enabled else "H2_REQUIRED"
        return GraphRescueResult(original_labels, (), 0, _freeze(base_diagnostics))
    if isinstance(embeddings, Mapping):
        by_id = embeddings
    else:
        if len(embeddings) > cfg.max_nodes:
            raise GraphRescueResourceError("embedding count exceeds max_nodes")
        by_id = {}
        for value in embeddings:
            item_id = _get(value, "tracklet_id", None)
            if item_id is not None:
                by_id[str(item_id)] = value
    vectors: dict[str, tuple[float, ...]] = {}
    embedding_dimension: int | None = None
    for item_id in ids:
        raw = _embedding_vector(by_id.get(item_id))
        vector = _normalise(raw, cfg.max_dimension)
        if vector is not None:
            if embedding_dimension is None:
                embedding_dimension = len(vector)
            elif len(vector) != embedding_dimension:
                raise GraphRescueError("all valid embeddings must have the same dimension")
            vectors[item_id] = vector
    speaker_labels = {label for label in original_labels if label in SPEAKER_IDS}
    unknown_ids = {item_id for item_id, label in zip(ids, original_labels) if label == "UNKNOWN"}
    kind_by_id = {_tracklet_id(item): _get(item, "kind", None) for item in tracklets}
    seed_ids: set[str] = set()
    excluded_assigned_count = 0
    anchor_seed_count = 0
    support_seed_count = 0
    for item_id, label in zip(ids, original_labels):
        if label not in speaker_labels or item_id not in vectors:
            continue
        kind = kind_by_id[item_id]
        explicitly_requested = explicit_seed_ids is not None and item_id in explicit_seed_ids
        eligible = kind == "ANCHOR" or (explicitly_requested and kind == "SUPPORT")
        if eligible:
            seed_ids.add(item_id)
            if kind == "ANCHOR":
                anchor_seed_count += 1
            else:
                support_seed_count += 1
        else:
            excluded_assigned_count += 1
    anchor_ids = seed_ids
    node_ids = tuple(item_id for item_id in ids if item_id in vectors and (item_id in unknown_ids or item_id in anchor_ids))
    base_diagnostics["seed_eligibility"] = {
        "mode": "EXPLICIT_IDS" if explicit_seed_ids is not None else "ANCHOR_ONLY",
        "eligible_seed_count": len(seed_ids),
        "anchor_seed_count": anchor_seed_count,
        "support_seed_count": support_seed_count,
        "excluded_assigned_count": excluded_assigned_count,
        "explicit_requested_count": len(explicit_seed_ids or ()),
    }
    if speaker_labels != set(SPEAKER_IDS) or any(
        not any(label == speaker and item_id in seed_ids for item_id, label in zip(ids, original_labels))
        for speaker in SPEAKER_IDS
    ):
        base_diagnostics["skip_reason"] = "TWO_SEED_CLASSES_REQUIRED"
        return GraphRescueResult(original_labels, (), 0, _freeze(base_diagnostics))
    if not unknown_ids or not anchor_ids or not speaker_labels:
        base_diagnostics["skip_reason"] = "NO_VALID_UNKNOWN_OR_ANCHOR"
        return GraphRescueResult(original_labels, (), 0, _freeze(base_diagnostics))

    block_map: dict[str, str] = {}
    block_owner: dict[str, str] = {}
    for item, label in zip(tracklets, original_labels):
        item_id = _tracklet_id(item)
        if item_id not in anchor_ids:
            continue
        explicit_block = anchor_block_ids.get(item_id) if anchor_block_ids is not None else None
        block = explicit_block or _get(item, "continuity_group_id", item_id)
        if not isinstance(block, str) or not block:
            raise GraphRescueError(f"anchor {item_id} has no independent block ID")
        owner = block_owner.setdefault(block, label)
        if owner != label:
            raise GraphRescueError("an independent anchor block cannot contain two speakers")
        block_map[item_id] = block
    adjacency, edge_count, evaluations = _build_adjacency(node_ids, vectors, cfg)
    labels_by_id = {item_id: label for item_id, label in zip(ids, original_labels)}
    # For the default one-step propagation, retain raw anchor contributions so
    # leave-block posteriors can be obtained by subtracting the removed block's
    # incident weights.  This is mathematically identical to rerunning the
    # clamped one-step update, but changes O(blocks * nodes * edges) work to
    # O(nodes * edges + candidate_support_blocks * candidate_degree).
    if cfg.propagation_steps == 1:
        full_raw = {
            item_id: _anchor_scores(item_id, adjacency, labels_by_id, block_map)
            for item_id in unknown_ids
        }
        full = {
            item_id: _normalised_scores(scores)
            for item_id, scores in full_raw.items()
        }
    else:
        full_raw = {}
        full = _run_propagation(node_ids, adjacency, labels_by_id, None, block_map, cfg.propagation_steps)
    candidates: list[GraphRescueCandidate] = []
    rejected_count = 0
    for item_id in ids:
        if item_id not in unknown_ids or item_id not in vectors:
            continue
        support_by_speaker: dict[str, set[str]] = {speaker: set() for speaker in speaker_labels}
        for neighbor, _weight in adjacency.get(item_id, ()):
            label = labels_by_id.get(neighbor)
            if label in speaker_labels and neighbor in block_map:
                support_by_speaker[label].add(block_map[neighbor])
        top, posterior, margin = _posterior(full.get(item_id, {}))
        if top is None:
            continue
        support = tuple(sorted(support_by_speaker.get(top, set())))
        if len(support) < cfg.min_anchor_blocks:
            rejected_count += 1
            continue
        leave_margins: list[tuple[str, float]] = []
        leave_stable = True
        # Only blocks that actually support the winning speaker can threaten
        # its leave-block stability.  Other-speaker block removal is irrelevant
        # to this conservative gate and is intentionally not recomputed.
        for block in support:
            if len(support) - 1 < cfg.min_anchor_blocks:
                leave_top, leave_margin = None, 0.0
            elif cfg.propagation_steps == 1:
                leave_scores = dict(full_raw[item_id])
                for neighbor, weight in adjacency.get(item_id, ()):
                    if labels_by_id.get(neighbor) == top and block_map.get(neighbor) == block:
                        leave_scores[top] -= weight
                leave_top, _leave_posterior, leave_margin = _posterior(leave_scores)
            else:
                leave = _run_propagation(node_ids, adjacency, labels_by_id, block, block_map, cfg.propagation_steps)
                leave_top, _leave_posterior, leave_margin = _posterior(leave.get(item_id, {}))
            leave_margins.append((block, leave_margin))
            if leave_top != top or leave_margin < cfg.leave_block_margin_min:
                leave_stable = False
        if posterior < cfg.min_posterior or margin < cfg.posterior_margin_min or not leave_stable:
            rejected_count += 1
            continue
        candidates.append(GraphRescueCandidate(
            item_id, top, posterior, margin, support, tuple(leave_margins),
            leave_stable, len(adjacency.get(item_id, ())),
            ("ANCHOR_CLAMPED", "TWO_INDEPENDENT_BLOCKS", "POSTERIOR_MARGIN", "LEAVE_BLOCK_STABLE"),
        ))
    rescued = dict(zip(ids, original_labels))
    for candidate in candidates:
        rescued[candidate.tracklet_id] = candidate.speaker_id
    base_diagnostics.update({
        "candidate_computation": True,
        "node_count": len(node_ids),
        "edge_count": edge_count,
        "distance_evaluations": evaluations,
        "anchor_count": len(anchor_ids),
        "unknown_count": len(unknown_ids),
        "candidate_count": len(candidates),
        "rejected_candidate_count": rejected_count,
        "adjacency_mode": cfg.adjacency_mode,
        "skip_reason": None,
    })
    return GraphRescueResult(tuple(rescued[item_id] for item_id in ids), tuple(candidates), len(candidates), _freeze(base_diagnostics))


def build_redacted_receipt(result: GraphRescueResult) -> Mapping[str, Any]:
    """Return aggregate/hash-only diagnostics suitable for persisted review.

    Candidate and block identifiers remain available on the in-memory
    experimental result, but never appear in this receipt.  The digest lets a
    reviewer correlate two receipts without retaining those identifiers.
    """

    identity_lines = [
        "|".join((candidate.tracklet_id, candidate.speaker_id, *candidate.supporting_anchor_blocks))
        for candidate in result.candidates
    ]
    digest = sha256("\n".join(sorted(identity_lines)).encode("utf-8")).hexdigest()
    speaker_counts: dict[str, int] = {}
    support_block_total = 0
    for candidate in result.candidates:
        speaker_counts[candidate.speaker_id] = speaker_counts.get(candidate.speaker_id, 0) + 1
        support_block_total += len(candidate.supporting_anchor_blocks)
    diagnostics = result.diagnostics
    receipt = {
        "schema": "graph_rescue_redacted_receipt_v1",
        "algorithm_version": GRAPH_RESCUE_ALGORITHM_VERSION,
        "candidate_count": len(result.candidates),
        "applied_count": result.applied_count,
        "candidate_speaker_counts": speaker_counts,
        "support_block_total": support_block_total,
        "candidate_identity_digest": digest,
        "node_count": diagnostics.get("node_count", 0),
        "edge_count": diagnostics.get("edge_count", 0),
        "distance_evaluations": diagnostics.get("distance_evaluations", 0),
        "candidate_computation": diagnostics.get("candidate_computation", False),
        "h2_confirmed": diagnostics.get("h2_confirmed", False),
    }
    return _freeze(receipt)


# Descriptive aliases keep the experiment easy to discover without changing
# the package's production exports.
propagate_unknowns = rescue_unknowns
graph_rescue = rescue_unknowns
redacted_receipt = build_redacted_receipt


__all__ = [
    "GRAPH_RESCUE_ALGORITHM_VERSION", "GraphRescueConfig", "GraphRescueCandidate", "GraphRescueResult",
    "GraphRescueError", "GraphRescueResourceError", "rescue_unknowns",
    "propagate_unknowns", "graph_rescue", "build_redacted_receipt",
    "redacted_receipt",
]
