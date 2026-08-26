import base64
import hashlib
import unittest
from dataclasses import replace
from math import sqrt

from sddiar.calibration import CalibrationProfileVerifier, DigestCalibrationSignatureVerifier, canonical_calibration_bytes
from sddiar.contracts import AnchorEvidence, EmbeddingResult, HypothesisDecision, ProtectedOverlapSpan, SpeakerAssignment, SpeakerState, Tracklet
from sddiar.diarization import (
    DiarizationConfig,
    build_tracklets,
    choose_hypothesis,
    decode_sequence,
    evaluate_h2,
    evaluate_hypotheses,
    finalize_sequence,
    maybe_update_recent,
    re_evaluate_micro,
    speaker_states_from_decision,
)
from boundary_test_helpers import sealed_osd, sealed_scd


class _ReleaseVerifier(DigestCalibrationSignatureVerifier):
    trust_level = "RELEASE"


def _release_binding():
    key = b"diarization-test-key"
    profile = {
        "schema_version": "1", "profile_id": "p", "calibration_version": "v1",
        "model_hashes": {"m": "a" * 64}, "source_sample_rates": [8000], "thresholds": {"scd_evidence_min": 0.5, "osd_evidence_min": 0.5},
        "dataset_manifest_hash": "b" * 64, "scorer_hash": "c" * 64, "config_hash": "d" * 64,
        "approver": "a", "provenance": {"annotation_schema_version": "1", "created_at": "now", "model_pack_id": "m", "pipeline_version": "p", "safety_constraints": ["safe"], "selection_objective": "safe"}, "signer_key_id": "k",
    }
    profile["signature"] = base64.b64encode(hashlib.sha256(key + canonical_calibration_bytes(profile)).digest()).decode()
    return CalibrationProfileVerifier(_ReleaseVerifier(key)).verify(profile, model_hashes={"m": "a" * 64}, source_sample_rate=8000, config_hash="d" * 64, profile_id="p")


def anchor(tracklet_id: str, vector: tuple[float, ...], start_us: int, group: str, scd: float | None = 1.0) -> AnchorEvidence:
    return AnchorEvidence(
        tracklet_id=tracklet_id,
        vector=vector,
        weight=1.0,
        clean_speech_us=2_000_000,
        independent_block_id=group,
        continuity_group_id=group,
        start_us=start_us,
        end_us=start_us + 2_000_000,
        scd_evidence_before=scd,
    )


def embedding(tracklet_id: str, vector: tuple[float, ...]) -> EmbeddingResult:
    return EmbeddingResult(
        embedding_region_id=f"emb-{tracklet_id}",
        tracklet_id=tracklet_id,
        is_valid=True,
        vector=vector,
        dimension=len(vector),
        valid_window_count=1,
        clean_window_coverage=1.0,
        intra_window_consistency=1.0,
        quality=1.0,
    )


class DiarizationTests(unittest.TestCase):
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

    def test_single_pass_matches_legacy_decision_and_h2_diagnostics(self) -> None:
        fixtures = (
            # H1: both anchors support the same centroid.
            (
                "H1_CONFIRMED",
                (
                    anchor("a0", (1.0, 0.0), 0, "a0"),
                    anchor("a1", (1.0, 0.0), 3_000_000, "a1"),
                ),
            ),
            # H2: repeated, temporally interleaved evidence.
            (
                "H2_CONFIRMED",
                (
                    anchor("a0", (1.0, 0.0), 0, "a0"),
                    anchor("b0", (-1.0, 0.0), 3_000_000, "b0"),
                    anchor("a1", (1.0, 0.0), 6_000_000, "a1"),
                    anchor("b1", (-1.0, 0.0), 9_000_000, "b1"),
                ),
            ),
            # Uncertain: two acoustic groups without bidirectional evidence.
            (
                "UNCERTAIN_1_OR_2",
                (
                    anchor("a0", (1.0, 0.0), 0, "same", 0.0),
                    anchor("a1", (1.0, 0.0), 3_000_000, "same", 0.0),
                    anchor("b0", (-1.0, 0.0), 6_000_000, "same", 0.0),
                    anchor("b1", (-1.0, 0.0), 9_000_000, "same", 0.0),
                ),
            ),
        )
        for expected_state, anchors in fixtures:
            with self.subTest(expected_state=expected_state):
                # This is the pre-single-pass call shape used by the runtime:
                # choose first, then evaluate_h2 again for diagnostics.
                legacy_decision = choose_hypothesis(anchors, self.cfg)
                legacy_diagnostics = evaluate_h2(anchors, self.cfg)
                single_pass = evaluate_hypotheses(anchors, self.cfg)
                self.assertEqual(legacy_decision.state, expected_state)
                self.assertEqual(single_pass.decision, legacy_decision)
                self.assertEqual(single_pass.h2_diagnostics, legacy_diagnostics)

    def test_repeated_interleaving_confirms_h2_deterministically(self) -> None:
        anchors = (
            anchor("a0", (1.0, 0.0), 0, "a0"),
            anchor("b0", (-1.0, 0.0), 3_000_000, "b0"),
            anchor("a1", (1.0, 0.0), 6_000_000, "a1"),
            anchor("b1", (-1.0, 0.0), 9_000_000, "b1"),
        )
        first = choose_hypothesis(anchors, self.cfg)
        second = choose_hypothesis(anchors, self.cfg)
        self.assertEqual(first.state, "H2_CONFIRMED")
        self.assertEqual(first, second)

    def test_one_way_condition_shift_stays_uncertain(self) -> None:
        anchors = (
            anchor("a0", (1.0, 0.0), 0, "same", 0.0),
            anchor("a1", (1.0, 0.0), 3_000_000, "same", 0.0),
            anchor("b0", (-1.0, 0.0), 6_000_000, "same", 0.0),
            anchor("b1", (-1.0, 0.0), 9_000_000, "same", 0.0),
        )
        self.assertEqual(choose_hypothesis(anchors, self.cfg).state, "UNCERTAIN_1_OR_2")

    def test_pure_protected_overlap_is_preserved_without_tracklets(self) -> None:
        overlap = ProtectedOverlapSpan("ov", 100, 400, 1.0)
        decision = choose_hypothesis((), self.cfg)
        spans = finalize_sequence((), (overlap,), {}, decision, 1_000, self.cfg)
        self.assertEqual([(span.speaker_id, span.start_us, span.end_us) for span in spans], [("OVERLAP", 100, 400)])

    def test_empty_audio_can_materialize_non_speech(self) -> None:
        decision = choose_hypothesis((), self.cfg)
        cfg = DiarizationConfig(include_non_speech=True)
        spans = finalize_sequence((), (), {}, decision, 100, cfg)
        self.assertEqual([(span.speaker_id, span.start_us, span.end_us) for span in spans], [("NON_SPEECH", 0, 100)])

    def test_micro_is_not_assigned_without_strict_margin(self) -> None:
        anchors = (
            anchor("a0", (1.0, 0.0), 0, "a0"),
            anchor("b0", (-1.0, 0.0), 3_000_000, "b0"),
            anchor("a1", (1.0, 0.0), 6_000_000, "a1"),
            anchor("b1", (-1.0, 0.0), 9_000_000, "b1"),
        )
        decision = choose_hypothesis(anchors, self.cfg)
        states = speaker_states_from_decision(decision, anchors)
        micro = Tracklet("micro", "r", "m", 12_000_000, 12_200_000, 200_000, "MICRO")
        result = re_evaluate_micro(micro, embedding("micro", (0.0, 1.0)), states, decision, self.cfg)
        self.assertEqual((result.speaker_id, result.attribution_status), ("UNKNOWN", "UNKNOWN_SHORT"))

    def test_opt_in_micro_cost_rescues_valid_micro_without_changing_default(self) -> None:
        tracklet = Tracklet("micro-cost", "r", "g", 0, 200_000, 200_000, "MICRO")
        vector = (0.95, sqrt(1.0 - 0.95 * 0.95))
        states = {
            "SPEAKER_00": SpeakerState("SPEAKER_00", (1.0, 0.0), ("a",), 0.0),
            "SPEAKER_01": SpeakerState("SPEAKER_01", (0.0, 1.0), ("b",), 0.0),
        }
        decision = HypothesisDecision("H2_CONFIRMED", None)
        cfg = DiarizationConfig(
            micro_stable_distance_ceiling=0.35,
            micro_absolute_distance_max=0.35,
            micro_margin_min=0.08,
            unknown_micro_cost=0.02,
        )
        baseline = decode_sequence((tracklet,), (), states, decision, 200_000, cfg, (embedding(tracklet.tracklet_id, vector),))
        candidate = decode_sequence(
            (tracklet,), (), states, decision, 200_000,
            replace(cfg, unknown_micro_cost=0.10),
            (embedding(tracklet.tracklet_id, vector),),
        )
        self.assertEqual(baseline.labels, ("UNKNOWN",))
        self.assertEqual(candidate.labels, ("SPEAKER_00",))
        self.assertEqual(
            finalize_sequence((tracklet,), (), states, decision, 200_000, cfg, (embedding(tracklet.tracklet_id, vector),)),
            baseline.spans,
        )

    def test_opt_in_soft_emission_compares_both_strictly_in_ceiling_speakers(self) -> None:
        tracklet = Tracklet("micro-soft", "r", "g", 0, 200_000, 200_000, "MICRO")
        vector = (0.73, sqrt(1.0 - 0.73 * 0.73))
        states = {
            "SPEAKER_00": SpeakerState("SPEAKER_00", (1.0, 0.0), ("a",), 0.0),
            "SPEAKER_01": SpeakerState("SPEAKER_01", (0.0, 1.0), ("b",), 0.0),
        }
        decision = HypothesisDecision("H2_CONFIRMED", None)
        cfg = DiarizationConfig(
            micro_stable_distance_ceiling=0.35,
            micro_absolute_distance_max=0.35,
            micro_margin_min=0.08,
            unknown_micro_cost=0.35,
        )
        hard = decode_sequence((tracklet,), (), states, decision, 200_000, cfg, (embedding(tracklet.tracklet_id, vector),))
        soft = decode_sequence(
            (tracklet,), (), states, decision, 200_000, cfg,
            (embedding(tracklet.tracklet_id, vector),),
            soft_speaker_emissions=True,
        )
        self.assertEqual(hard.labels, ("UNKNOWN",))
        self.assertEqual(soft.labels, ("SPEAKER_00",))

    def test_tracklet_builder_keeps_overlap_separate_from_clean_tracklets(self) -> None:
        built = build_tracklets(
            ({"region_id": "r", "start_us": 0, "end_us": 2_000_000},),
            overlap_regions=(sealed_osd(500_000, 1_500_000, 1.0, ("osd",), source_id="test"),),
            cfg=DiarizationConfig(anchor_min_clean_us=500_000, support_min_clean_us=200_000),
            audio_id="test",
        )
        self.assertEqual([(span.start_us, span.end_us) for span in built.protected_overlap_spans], [(500_000, 1_500_000)])
        self.assertTrue(all(not tracklet.protected_overlap for tracklet in built.tracklets))

    def test_boundary_evidence_ids_are_local_to_the_cut_region_and_tracklet(self) -> None:
        first = sealed_scd(1_000_000, 0.9, "first")
        second = sealed_scd(4_000_000, 0.9, "second")
        built = build_tracklets(
            (
                {"region_id": "r1", "start_us": 0, "end_us": 2_000_000},
                {"region_id": "r2", "start_us": 3_000_000, "end_us": 5_000_000},
            ),
            scd_events=(first, second),
            cfg=DiarizationConfig(min_split_side_us=100_000),
        )
        self.assertEqual([tracklet.boundary_evidence_ids for tracklet in built.tracklets], [("first",), ("first",), ("second",), ("second",)])

    def test_recent_centroid_never_updates_micro_but_updates_trusted_support(self) -> None:
        state = SpeakerState("SPEAKER_00", (1.0, 0.0), ("a",), 0.0)
        support = Tracklet("support", "r", "g", 0, 1_000_000, 1_000_000, "SUPPORT")
        trusted = SpeakerAssignment("support", "SPEAKER_00", "LOCAL_CANDIDATE", 0.01, 0.01, 0.2)
        cfg = DiarizationConfig(enable_recent_centroid=True, recent_update_quality_min=0.5, recent_update_stable_fit_max=0.1)
        updated = maybe_update_recent(state, support, embedding("support", (0.995, 0.0998749218)), trusted, None, cfg)
        self.assertIsNotNone(updated.recent_centroid)
        micro = Tracklet("micro-2", "r", "g", 1_000_000, 1_200_000, 200_000, "MICRO")
        unchanged = maybe_update_recent(updated, micro, embedding("micro-2", (0.995, 0.0998749218)), trusted, None, cfg)
        self.assertEqual(unchanged, updated)


if __name__ == "__main__":
    unittest.main()
