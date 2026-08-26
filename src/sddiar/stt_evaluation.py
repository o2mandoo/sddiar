"""Dependency-free, privacy-safe speech-to-text evaluation.

This module deliberately operates on values supplied by the caller.  It never
opens a transcript, audio file, or path.  Text is used while scoring, but
structured results contain counts, rates, and digests only; they do not retain
text, tokens, or paths.

Speaker-attributed metrics use a *global one-to-one mapping* (the usual cpWER
style convention): each hypothesis speaker is assigned to at most one
reference speaker and the assignment minimizing total edit distance is chosen.
The same mapping is used for SA-CER and SA-WER.  One or two speaker labels per
side are supported, including a missing speaker on either side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import random
import re
import unicodedata
from itertools import permutations
from typing import Any, Iterable, Mapping, Sequence


class STTEvaluationError(ValueError):
    """Invalid evaluation input or unsupported privacy-sensitive input."""


@dataclass(frozen=True)
class TextNormalizationConfig:
    """Explicit Korean-friendly normalization policy.

    ``punctuation_policy`` is one of ``preserve``, ``remove`` (delete
    punctuation), or ``space`` (replace it with one separator).  Whitespace is
    then either preserved, collapsed to one ASCII space, or removed.  NFC is
    the default and is configurable for reproducible experiments.
    """

    unicode_form: str = "NFC"
    punctuation_policy: str = "preserve"
    whitespace_policy: str = "collapse"
    trim: bool = True

    def __post_init__(self) -> None:
        if self.unicode_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise STTEvaluationError("unicode_form must be NFC, NFD, NFKC, or NFKD")
        if self.punctuation_policy not in {"preserve", "remove", "space"}:
            raise STTEvaluationError("invalid punctuation_policy")
        if self.whitespace_policy not in {"preserve", "collapse", "remove"}:
            raise STTEvaluationError("invalid whitespace_policy")
        if not isinstance(self.trim, bool):
            raise STTEvaluationError("trim must be boolean")


def normalize_text(text: str, config: TextNormalizationConfig | None = None) -> str:
    """Return normalized text without retaining it in any result object."""
    if not isinstance(text, str):
        raise STTEvaluationError("text must be a string")
    policy = config or TextNormalizationConfig()
    value = unicodedata.normalize(policy.unicode_form, text)
    if policy.punctuation_policy != "preserve":
        value = "".join(
            (" " if policy.punctuation_policy == "space" else "")
            if unicodedata.category(char).startswith("P") else char
            for char in value
        )
    if policy.whitespace_policy == "collapse":
        value = " ".join(value.split())
    elif policy.whitespace_policy == "remove":
        value = "".join(value.split())
    if policy.trim and policy.whitespace_policy == "preserve":
        value = value.strip()
    return value


normalize_korean_text = normalize_text


@dataclass(frozen=True)
class EditMetrics:
    """Levenshtein accounting.  ``error_rate`` is finite for empty inputs."""

    reference_units: int
    hypothesis_units: int
    substitutions: int
    deletions: int
    insertions: int
    distance: int
    error_rate: float


def _levenshtein(reference: Sequence[str], hypothesis: Sequence[str]) -> EditMetrics:
    n, m = len(reference), len(hypothesis)
    # Rows contain only integer edit costs; this is dependency-free and avoids
    # retaining either input in a result.
    # Store operation counts in each cell so substitutions/deletions/insertions
    # are deterministic even when several shortest paths exist.
    rows: list[list[tuple[int, int, int, int]]] = [[(i, 0, 0, i) for i in range(m + 1)]]
    for i in range(1, n + 1):
        row: list[tuple[int, int, int, int]] = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                candidates = [(rows[i - 1][j - 1], 0)]
            else:
                s = rows[i - 1][j - 1]
                candidates = [((s[0] + 1, s[1] + 1, s[2], s[3]), 0)]
            d = rows[i - 1][j]
            candidates.append(((d[0] + 1, d[1], d[2] + 1, d[3]), 1))
            ins = row[j - 1]
            candidates.append(((ins[0] + 1, ins[1], ins[2], ins[3] + 1), 2))
            # Prefer match/substitution, then deletion, then insertion on ties.
            best = min(candidates, key=lambda item: (item[0][0], item[1]))[0]
            row.append(best)
        rows.append(row)
    distance, substitutions, deletions, insertions = rows[n][m]
    # An empty reference has no conventional WER/CER denominator.  Keep the
    # public rate finite and bounded (empty-vs-empty is exact; any output for
    # an empty reference is a complete error), while preserving insertion
    # counts in the structured result.
    rate = 0.0 if n == 0 and m == 0 else 1.0 if n == 0 else distance / n
    return EditMetrics(n, m, substitutions, deletions, insertions, distance, rate)


def cer_metrics(reference: str, hypothesis: str,
                config: TextNormalizationConfig | None = None) -> EditMetrics:
    """Character error rate after configured Unicode/text normalization."""
    ref = list(normalize_text(reference, config))
    hyp = list(normalize_text(hypothesis, config))
    return _levenshtein(ref, hyp)


def wer_metrics(reference: str, hypothesis: str,
                config: TextNormalizationConfig | None = None) -> EditMetrics:
    """Whitespace-token WER after configured Unicode/text normalization."""
    ref = normalize_text(reference, config).split()
    hyp = normalize_text(hypothesis, config).split()
    return _levenshtein(ref, hyp)


def character_error_rate(reference: str, hypothesis: str,
                         config: TextNormalizationConfig | None = None) -> float:
    return cer_metrics(reference, hypothesis, config).error_rate


def word_error_rate(reference: str, hypothesis: str,
                    config: TextNormalizationConfig | None = None) -> float:
    return wer_metrics(reference, hypothesis, config).error_rate


# Short aliases are useful to callers building a small evaluation script.
cer = character_error_rate
wer = word_error_rate


@dataclass(frozen=True)
class SpeakerText:
    """Text for one pseudonymous speaker; the text is never serialized."""

    speaker_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.speaker_id or not isinstance(self.speaker_id, str) or not isinstance(self.text, str):
            raise STTEvaluationError("invalid SpeakerText")


def _speaker_texts(value: Mapping[str, str | Sequence[str]] | Sequence[SpeakerText]) -> dict[str, str]:
    if isinstance(value, Mapping):
        result: dict[str, str] = {}
        for key, text in value.items():
            if not isinstance(key, str) or not key:
                raise STTEvaluationError("speaker IDs must be non-empty strings")
            if isinstance(text, str):
                result[key] = text
            elif isinstance(text, Sequence) and not isinstance(text, (str, bytes, bytearray)):
                if not all(isinstance(part, str) for part in text):
                    raise STTEvaluationError("speaker text chunks must be strings")
                result[key] = " ".join(text)
            else:
                raise STTEvaluationError("speaker values must be strings or string sequences")
        return result
    result = {}
    for item in value:
        if not isinstance(item, SpeakerText):
            raise STTEvaluationError("speaker sequences must contain SpeakerText")
        if item.speaker_id in result:
            result[item.speaker_id] += " " + item.text
        else:
            result[item.speaker_id] = item.text
    return result


@dataclass(frozen=True)
class SpeakerMapping:
    hypothesis_speaker_id: str
    reference_speaker_id: str | None


@dataclass(frozen=True)
class SpeakerAttributedMetrics:
    """SA-CER/SA-WER (global cp-style speaker mapping) with no transcript data."""

    metric: str
    error_rate: float
    distance: int
    reference_units: int
    mapping: tuple[SpeakerMapping, ...]
    unmatched_reference_speakers: int
    unmatched_hypothesis_speakers: int


def _mapping_candidates(refs: Sequence[str], hyps: Sequence[str]) -> Iterable[tuple[tuple[str, str | None], ...]]:
    # Enumerate all one-to-one assignments, including an unmatched hypothesis.
    # Input cardinalities are explicitly bounded by the public API.
    if len(refs) > 2 or len(hyps) > 2:
        raise STTEvaluationError("speaker-attributed metrics support at most two speakers")
    count = min(len(refs), len(hyps))
    # Permute both sides: with one reference and two hypotheses, either
    # hypothesis may be the matched one.  Sorting only one side would make the
    # answer depend on input order and could miss the optimum.
    for ref_order in permutations(sorted(refs), count):
        for hyp_order in permutations(sorted(hyps), count):
            pairs = tuple(zip(hyp_order, ref_order))
            if len(hyps) > len(refs):
                unmatched = tuple((speaker, None) for speaker in sorted(hyps) if speaker not in hyp_order)
                yield pairs + unmatched
            else:
                yield pairs
    if not refs and hyps:
        yield tuple((speaker, None) for speaker in sorted(hyps))


def _speaker_attributed(reference: Mapping[str, str | Sequence[str]] | Sequence[SpeakerText],
                        hypothesis: Mapping[str, str | Sequence[str]] | Sequence[SpeakerText],
                        metric: str, config: TextNormalizationConfig | None) -> SpeakerAttributedMetrics:
    refs, hyps = _speaker_texts(reference), _speaker_texts(hypothesis)
    if len(refs) > 2 or len(hyps) > 2:
        raise STTEvaluationError("speaker-attributed metrics support at most two speakers")
    metric_fn = cer_metrics if metric == "SA-CER" else wer_metrics
    candidates = list(_mapping_candidates(sorted(refs), sorted(hyps)))
    if not candidates:
        # Both sides empty.
        return SpeakerAttributedMetrics(metric, 0.0, 0, 0, (), 0, 0)
    scored: list[tuple[tuple[int, tuple[tuple[str, str | None], ...]], int, int, tuple[SpeakerMapping, ...]]] = []
    for candidate in candidates:
        by_ref = {ref: hyp for hyp, ref in candidate if ref is not None}
        total = 0
        ref_units = 0
        for ref in sorted(refs):
            result = metric_fn(refs[ref], hyps[by_ref[ref]] if ref in by_ref else "", config)
            total += result.distance
            ref_units += result.reference_units
        for hyp, ref in candidate:
            if ref is None:
                total += metric_fn("", hyps[hyp], config).distance
        pairs = tuple(SpeakerMapping(hyp, ref) for hyp, ref in candidate)
        key = tuple((item.hypothesis_speaker_id, item.reference_speaker_id or "") for item in pairs)
        scored.append(((total, key), total, ref_units, pairs))
    _, total, ref_units, mapping = min(scored, key=lambda item: item[0])
    mapping = tuple(sorted(mapping, key=lambda item: item.hypothesis_speaker_id))
    # Include an unmatched reference implicitly (it has no hypothesis mapping).
    mapped_refs = {item.reference_speaker_id for item in mapping if item.reference_speaker_id is not None}
    return SpeakerAttributedMetrics(
        metric, total / max(ref_units, 1), total, ref_units, mapping,
        len(set(refs) - mapped_refs), sum(item.reference_speaker_id is None for item in mapping),
    )


def speaker_attributed_cer(reference, hypothesis, config=None) -> SpeakerAttributedMetrics:
    return _speaker_attributed(reference, hypothesis, "SA-CER", config)


def speaker_attributed_wer(reference, hypothesis, config=None) -> SpeakerAttributedMetrics:
    return _speaker_attributed(reference, hypothesis, "SA-WER", config)


@dataclass(frozen=True)
class TimedWord:
    word_id: str
    text: str
    start_us: int
    end_us: int
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.word_id, str) or not self.word_id or not isinstance(self.text, str)
                or isinstance(self.start_us, bool) or isinstance(self.end_us, bool)
                or not isinstance(self.start_us, int) or not isinstance(self.end_us, int)
                or self.start_us < 0 or self.end_us <= self.start_us):
            raise STTEvaluationError("invalid TimedWord")


@dataclass(frozen=True)
class TimestampWordMetrics:
    reference_words: int
    hypothesis_words: int
    matched_words: int
    coverage: float
    median_boundary_error_us: float | None
    p95_boundary_error_us: float | None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def timestamp_word_metrics(reference: Sequence[TimedWord], hypothesis: Sequence[TimedWord],
                           config: TextNormalizationConfig | None = None) -> TimestampWordMetrics:
    """Align word streams and report timing error for exact aligned words.

    Boundary error is ``max(abs(start_delta), abs(end_delta))`` in microseconds;
    this definition makes the metric interpretable for both early/late starts
    and duration drift.  If no reference timings are supplied, coverage and
    percentiles are explicitly zero/``None`` rather than inferred.
    """
    refs = list(reference)
    hyps = list(hypothesis)
    ref_tokens = [normalize_text(item.text, config) for item in refs]
    hyp_tokens = [normalize_text(item.text, config) for item in hyps]
    n, m = len(refs), len(hyps)
    # DP stores cost; traceback chooses diagonal exact matches first.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (ref_tokens[i - 1] != hyp_tokens[j - 1]))
    i, j, errors = n, m, []
    while i or j:
        if i and j and ref_tokens[i - 1] == hyp_tokens[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ref_word, hyp_word = refs[i - 1], hyps[j - 1]
            errors.append(float(max(abs(ref_word.start_us - hyp_word.start_us),
                                    abs(ref_word.end_us - hyp_word.end_us))))
            i -= 1
            j -= 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    errors.reverse()
    return TimestampWordMetrics(n, m, len(errors), len(errors) / max(n, 1),
                                _percentile(errors, 0.5), _percentile(errors, 0.95))


@dataclass(frozen=True)
class RecordingSTTInput:
    """Caller-owned input; not serialized by this module."""

    recording_id: str
    reference_text: str
    hypothesis_text: str
    subgroup: Mapping[str, str] = field(default_factory=dict)
    reference_speakers: Mapping[str, str | Sequence[str]] | Sequence[SpeakerText] | None = None
    hypothesis_speakers: Mapping[str, str | Sequence[str]] | Sequence[SpeakerText] | None = None
    reference_words: Sequence[TimedWord] = ()
    hypothesis_words: Sequence[TimedWord] = ()

    def __post_init__(self) -> None:
        if not self.recording_id or not isinstance(self.recording_id, str):
            raise STTEvaluationError("recording_id must be a non-empty string")
        if not isinstance(self.reference_text, str) or not isinstance(self.hypothesis_text, str):
            raise STTEvaluationError("recording text must be strings")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.subgroup.items()):
            raise STTEvaluationError("subgroup keys and values must be strings")


@dataclass(frozen=True)
class RecordingSTTResult:
    recording_key: str
    cer: float
    wer: float
    reference_characters: int
    reference_words: int
    timestamp: TimestampWordMetrics | None
    sa_cer: float | None
    sa_wer: float | None
    subgroup: tuple[tuple[str, str], ...]


def _digest(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    return sha256(payload).hexdigest()


def evaluate_recording(item: RecordingSTTInput, config: TextNormalizationConfig | None = None) -> RecordingSTTResult:
    """Score one recording; ``recording_id`` is immediately replaced by a digest."""
    cm, wm = cer_metrics(item.reference_text, item.hypothesis_text, config), wer_metrics(item.reference_text, item.hypothesis_text, config)
    sa_cer = sa_wer = None
    if item.reference_speakers is not None and item.hypothesis_speakers is not None:
        sa_cer = speaker_attributed_cer(item.reference_speakers, item.hypothesis_speakers, config).error_rate
        sa_wer = speaker_attributed_wer(item.reference_speakers, item.hypothesis_speakers, config).error_rate
    timestamp = None
    if item.reference_words:
        timestamp = timestamp_word_metrics(item.reference_words, item.hypothesis_words, config)
    return RecordingSTTResult(_digest(item.recording_id), cm.error_rate, wm.error_rate,
                              cm.reference_units, wm.reference_units, timestamp, sa_cer, sa_wer,
                              tuple(sorted(item.subgroup.items())))


@dataclass(frozen=True)
class BootstrapCI:
    metric: str
    estimate: float
    lower: float
    upper: float
    iterations: int
    seed: int


@dataclass(frozen=True)
class SubgroupSTTResult:
    subgroup: tuple[tuple[str, str], ...]
    count: int
    cer: float
    wer: float
    bootstrap: tuple[BootstrapCI, ...]


@dataclass(frozen=True)
class AggregateSTTResult:
    count: int
    cer: float
    wer: float
    bootstrap: tuple[BootstrapCI, ...]
    subgroups: tuple[SubgroupSTTResult, ...]
    run_manifest: "STTRunManifest"


def _bootstrap(values: Sequence[float], metric: str, iterations: int, seed: int,
               confidence_level: float) -> BootstrapCI:
    if not values:
        return BootstrapCI(metric, 0.0, 0.0, 0.0, iterations, seed)
    estimate = sum(values) / len(values)
    rng = random.Random(seed)
    samples = [sum(values[rng.randrange(len(values))] for _ in values) / len(values)
               for _ in range(iterations)]
    alpha = (1.0 - confidence_level) / 2.0
    return BootstrapCI(metric, estimate, _percentile(samples, alpha) or 0.0,
                       _percentile(samples, 1.0 - alpha) or 0.0, iterations, seed)


def aggregate_recordings(items: Sequence[RecordingSTTInput] | Sequence[RecordingSTTResult],
                         *, config: TextNormalizationConfig | None = None,
                         bootstrap_iterations: int = 2000, bootstrap_seed: int = 17029,
                         confidence_level: float = 0.95) -> AggregateSTTResult:
    """Aggregate recording means and deterministic per-recording bootstrap CIs."""
    if isinstance(bootstrap_iterations, bool) or bootstrap_iterations < 1:
        raise STTEvaluationError("bootstrap_iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise STTEvaluationError("confidence_level must be between zero and one")
    results = [item if isinstance(item, RecordingSTTResult) else evaluate_recording(item, config) for item in items]
    results.sort(key=lambda item: item.recording_key)
    cer_values = [item.cer for item in results]
    wer_values = [item.wer for item in results]
    overall = tuple((_bootstrap(cer_values, "cer", bootstrap_iterations, bootstrap_seed, confidence_level),
                     _bootstrap(wer_values, "wer", bootstrap_iterations, bootstrap_seed + 1, confidence_level)))
    groups: dict[tuple[tuple[str, str], ...], list[RecordingSTTResult]] = {}
    for result in results:
        if result.subgroup:
            groups.setdefault(result.subgroup, []).append(result)
    subgroup_results = []
    for subgroup in sorted(groups):
        members = groups[subgroup]
        subgroup_results.append(SubgroupSTTResult(
            subgroup, len(members), sum(item.cer for item in members) / len(members),
            sum(item.wer for item in members) / len(members),
            (_bootstrap([item.cer for item in members], "cer", bootstrap_iterations, bootstrap_seed, confidence_level),
             _bootstrap([item.wer for item in members], "wer", bootstrap_iterations, bootstrap_seed + 1, confidence_level)),
        ))
    manifest = build_run_manifest(results, config={"normalization": (config or TextNormalizationConfig()).__dict__,
                                                   "bootstrap_iterations": bootstrap_iterations,
                                                   "bootstrap_seed": bootstrap_seed,
                                                   "confidence_level": confidence_level})
    return AggregateSTTResult(len(results), sum(cer_values) / len(results) if results else 0.0,
                               sum(wer_values) / len(results) if results else 0.0, overall,
                               tuple(subgroup_results), manifest)


@dataclass(frozen=True)
class STTRunManifest:
    schema_version: str
    input_count: int
    input_sha256: str
    config_sha256: str
    run_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "input_count": self.input_count,
                "input_sha256": self.input_sha256, "config_sha256": self.config_sha256,
                "run_sha256": self.run_sha256}


def build_run_manifest(inputs: Iterable[Any], config: Mapping[str, Any] | None = None) -> STTRunManifest:
    """Build a content-addressed manifest containing hashes and counts only."""
    values = list(inputs)
    # Recording results are already redacted; arbitrary caller values are
    # hashed but never copied to the manifest.
    input_hash = _digest([_digest(value.as_dict() if hasattr(value, "as_dict") else value) for value in values])
    safe_config = dict(config or {})
    config_hash = _digest(safe_config)
    run_hash = _digest({"schema_version": "sddiar-stt-evaluation-v1", "input_count": len(values),
                        "input_sha256": input_hash, "config_sha256": config_hash})
    return STTRunManifest("sddiar-stt-evaluation-v1", len(values), input_hash, config_hash, run_hash)


def run_identity(reference: str, hypothesis: str,
                 config: TextNormalizationConfig | None = None) -> str:
    """Return a hash-only identity for one text comparison."""
    policy = (config or TextNormalizationConfig()).__dict__
    return _digest({"reference": _digest(reference), "hypothesis": _digest(hypothesis), "config": policy})


_SAFE_ID = re.compile(r"^(?:ref|hyp|spk|speaker|anon|record|rec|word|u)[_-]?[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$", re.IGNORECASE)


@dataclass(frozen=True)
class PseudonymizedReferenceRecord:
    record_id: str
    text: str
    speaker_id: str
    start_us: int | None = None
    end_us: int | None = None


def parse_pseudonymized_reference_records(records: Iterable[Mapping[str, Any]]) -> tuple[PseudonymizedReferenceRecord, ...]:
    """Parse in-memory, already-pseudonymized records; never accepts paths.

    The adapter rejects names/paths as identifiers and requires ``speaker_id``
    values such as ``ref_01`` or ``spk_a``.  It does not open or discover any
    user transcript; callers must pass records explicitly.
    """
    output = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or any(key in record for key in ("path", "file", "filename")):
            raise STTEvaluationError("reference adapter accepts mappings, not paths")
        try:
            speaker_id = record["speaker_id"]
            text = record.get("text", record.get("word"))
        except (KeyError, TypeError) as exc:
            raise STTEvaluationError("speaker_id and text are required") from exc
        if not isinstance(speaker_id, str) or not _SAFE_ID.fullmatch(speaker_id):
            raise STTEvaluationError("speaker_id must already be pseudonymized")
        if not isinstance(text, str):
            raise STTEvaluationError("text must be a string")
        record_id = record.get("record_id", record.get("word_id", f"rec_{index}"))
        if not isinstance(record_id, str) or not record_id or not _SAFE_ID.fullmatch(record_id):
            raise STTEvaluationError("record_id must already be pseudonymized")
        start_us, end_us = record.get("start_us"), record.get("end_us")
        if (start_us is None) != (end_us is None):
            raise STTEvaluationError("both start_us and end_us are required for timing")
        if start_us is not None and (isinstance(start_us, bool) or isinstance(end_us, bool)
                                     or not isinstance(start_us, int) or not isinstance(end_us, int)
                                     or start_us < 0 or end_us <= start_us):
            raise STTEvaluationError("invalid reference timing")
        output.append(PseudonymizedReferenceRecord(record_id, text, speaker_id, start_us, end_us))
    return tuple(output)


def timed_words_from_reference_records(records: Sequence[PseudonymizedReferenceRecord]) -> tuple[TimedWord, ...]:
    """Convert parsed records to timing inputs; records without timings fail closed."""
    if any(record.start_us is None or record.end_us is None for record in records):
        raise STTEvaluationError("reference timings are required")
    return tuple(TimedWord(record.record_id, record.text, record.start_us, record.end_us, record.speaker_id)
                 for record in records)


__all__ = [
    "STTEvaluationError", "TextNormalizationConfig", "normalize_text", "normalize_korean_text",
    "EditMetrics", "cer_metrics", "wer_metrics", "character_error_rate", "word_error_rate", "cer", "wer",
    "SpeakerText", "SpeakerMapping", "SpeakerAttributedMetrics", "speaker_attributed_cer", "speaker_attributed_wer",
    "TimedWord", "TimestampWordMetrics", "timestamp_word_metrics", "RecordingSTTInput", "RecordingSTTResult",
    "BootstrapCI", "SubgroupSTTResult", "AggregateSTTResult", "aggregate_recordings", "STTRunManifest",
    "build_run_manifest", "run_identity", "PseudonymizedReferenceRecord", "parse_pseudonymized_reference_records",
    "timed_words_from_reference_records",
]
