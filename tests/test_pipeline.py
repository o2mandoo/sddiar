import unittest

from sddiar.contracts import EmbeddingResult, Word, WordProvenance, WordTimeline
from sddiar.diarization import DiarizationConfig
from sddiar.pipeline import EvidencePipeline


class EvidencePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = DiarizationConfig(
            anchor_outlier_distance_max=0.8,
            h2_min_independent_anchor_count=2,
            h2_min_clean_anchor_us=1_000_000,
            h2_min_separation=0.1,
            h2_min_cost_gain=-0.5,
            h2_min_label_stability=0.0,
            h2_min_centroid_stability=0.0,
            lambda_k2=0.0,
            lambda_stability=0.0,
            lambda_condition=0.0,
            h1_max_outlier_ratio=1.0,
            h1_max_dispersion=1.0,
        )

    def provider(self, tracklets):
        vectors = ((1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0))
        return tuple(
            EmbeddingResult(
                embedding_region_id=f"emb-{tracklet.tracklet_id}",
                tracklet_id=tracklet.tracklet_id,
                is_valid=True,
                vector=vectors[index],
                dimension=2,
                valid_window_count=1,
                clean_window_coverage=1.0,
                intra_window_consistency=1.0,
                quality=1.0,
            )
            for index, tracklet in enumerate(tracklets)
        )

    def test_runs_whole_file_finalization_and_word_mapping(self) -> None:
        pipeline = EvidencePipeline(self.provider, config=self.cfg)
        timeline = WordTimeline((Word("w", 0, 1_000_000, "hello"),), {"w": WordProvenance("w")})
        run = pipeline.run(
            audio_id="test",
            source_duration_us=11_000_000,
            vad_regions=(
                {"start_us": 0, "end_us": 2_000_000},
                {"start_us": 3_000_000, "end_us": 5_000_000},
                {"start_us": 6_000_000, "end_us": 8_000_000},
                {"start_us": 9_000_000, "end_us": 11_000_000},
            ),
            word_timeline=timeline,
        )
        self.assertEqual(run.decision.state, "H2_CONFIRMED")
        self.assertEqual(len(run.spans), 4)
        self.assertEqual(run.attributed_words[0].speaker_id, "SPEAKER_00")

    def test_never_requires_embedding_for_pure_overlap(self) -> None:
        pipeline = EvidencePipeline(lambda tracklets: (), config=self.cfg)
        run = pipeline.run(
            audio_id="overlap",
            source_duration_us=1_000,
            vad_regions=(),
            overlap_regions=({"start_us": 100, "end_us": 900, "overlap_evidence": 1.0},),
        )
        self.assertEqual(run.decision.state, "UNCERTAIN_1_OR_2")
        self.assertEqual([(span.speaker_id, span.start_us, span.end_us) for span in run.spans], [("OVERLAP", 100, 900)])


if __name__ == "__main__":
    unittest.main()
