"""A small offline orchestration seam for the P0/P1 reference library.

The class accepts already-collected VAD/SCD/OSD evidence and an injected local
embedding provider. It intentionally does not decode media, load models, call
STT, or open a network connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .attribution import WordSpeakerMapper
from .contracts import (
    AnchorEvidence,
    AttributedWord,
    EmbeddingResult,
    FileQualityReport,
    HypothesisDecision,
    SpeakerState,
    Tracklet,
    TrackletBuildResult,
    WordTimeline,
)
from .diarization import (
    DiarizationConfig,
    build_tracklets,
    choose_hypothesis,
    finalize_sequence,
    refine_recent_states,
    select_anchor_evidence,
    speaker_states_from_decision,
)
from .errors import ContractValidationError
from .quality import RuleBasedQualityGate
from .segmentation import SegmentationEvidence


EmbeddingProvider = Callable[[Sequence[Tracklet]], Sequence[EmbeddingResult]]


@dataclass(frozen=True, slots=True)
class DiarizationRun:
    build: TrackletBuildResult
    embeddings: tuple[EmbeddingResult, ...]
    anchors: tuple[AnchorEvidence, ...]
    support_count: int
    deferred_count: int
    decision: HypothesisDecision
    speaker_states: Mapping[str, SpeakerState]
    spans: tuple[Any, ...]
    attributed_words: tuple[AttributedWord, ...] = ()
    quality: FileQualityReport | None = None


class EvidencePipeline:
    """Deterministic, whole-file finalization orchestrator.

    The provider must be a local, prevalidated implementation. Model-pack
    verification occurs outside this class before the provider is created.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        config: DiarizationConfig | None = None,
        word_mapper: WordSpeakerMapper | None = None,
        quality_gate: RuleBasedQualityGate | None = None,
    ):
        self._embedding_provider = embedding_provider
        self._config = config or DiarizationConfig()
        self._word_mapper = word_mapper or WordSpeakerMapper()
        self._quality_gate = quality_gate or RuleBasedQualityGate()

    def run(
        self,
        *,
        audio_id: str,
        source_duration_us: int,
        vad_regions: Sequence[Any],
        scd_events: Sequence[Any] = (),
        overlap_regions: Sequence[Any] = (),
        word_timeline: WordTimeline | None = None,
        quality_diagnostics: Any | None = None,
        calibration: Any | None = None,
    ) -> DiarizationRun:
        if source_duration_us < 0:
            raise ContractValidationError("source_duration_us must be non-negative")
        build = build_tracklets(
            vad_regions,
            scd_events=scd_events,
            overlap_regions=overlap_regions,
            cfg=self._config,
            audio_id=audio_id,
        )
        embeddings = tuple(self._embedding_provider(build.tracklets))
        tracklet_ids = {tracklet.tracklet_id for tracklet in build.tracklets}
        if any(embedding.tracklet_id not in tracklet_ids for embedding in embeddings):
            raise ContractValidationError("embedding provider returned an unknown tracklet ID")
        if len({embedding.tracklet_id for embedding in embeddings}) != len(embeddings):
            raise ContractValidationError("embedding provider returned duplicate tracklet IDs")
        anchors, support, deferred = select_anchor_evidence(build.tracklets, embeddings, self._config)
        decision = choose_hypothesis(anchors, self._config)
        speaker_states = speaker_states_from_decision(decision, anchors)
        speaker_states = refine_recent_states(build.tracklets, embeddings, speaker_states, decision, self._config)
        spans = finalize_sequence(
            build.tracklets,
            build.protected_overlap_spans,
            speaker_states,
            decision,
            source_duration_us,
            self._config,
            embeddings,
        )
        attributed_words = self._word_mapper.map_words(word_timeline, spans) if word_timeline is not None else ()
        quality = self._quality_gate.evaluate(quality_diagnostics, calibration) if quality_diagnostics is not None else None
        return DiarizationRun(
            build=build,
            embeddings=embeddings,
            anchors=anchors,
            support_count=len(support),
            deferred_count=len(deferred),
            decision=decision,
            speaker_states=speaker_states,
            spans=spans,
            attributed_words=tuple(attributed_words),
            quality=quality,
        )

    def run_segmentation(
        self,
        *,
        audio_id: str,
        source_duration_us: int,
        segmentation: SegmentationEvidence,
        word_timeline: WordTimeline | None = None,
        quality_diagnostics: Any | None = None,
        calibration: Any | None = None,
    ) -> DiarizationRun:
        return self.run(
            audio_id=audio_id,
            source_duration_us=source_duration_us,
            vad_regions=segmentation.speech_regions,
            scd_events=segmentation.scd_events,
            overlap_regions=segmentation.overlap_regions,
            word_timeline=word_timeline,
            quality_diagnostics=quality_diagnostics,
            calibration=calibration,
        )
