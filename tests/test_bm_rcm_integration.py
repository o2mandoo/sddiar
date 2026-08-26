from __future__ import annotations

import unittest
from types import SimpleNamespace

from sddiar.bm_rcm_integration import integrate_bm_rcm
from sddiar.contracts import AnchorEvidence, EmbeddingResult, Tracklet
from sddiar.diarization import DiarizationConfig


def _fixture(*, decision_state="H2_CONFIRMED", enough_anchors=True):
    tracklets = []
    embeddings = []
    labels = []
    states = {}
    anchor_count = 9 if enough_anchors else 2
    for speaker_id, vector in (("SPEAKER_00", (1.0, 0.0)), ("SPEAKER_01", (-1.0, 0.0))):
        anchor_ids = []
        for index in range(anchor_count):
            ordinal = len(tracklets)
            start = ordinal * 2_000_000
            tracklet = Tracklet(
                f"{speaker_id}-anchor-{index}", f"region-{ordinal}",
                f"{speaker_id}-block-{index}", start, start + 1_500_000,
                1_500_000, "ANCHOR",
            )
            tracklets.append(tracklet)
            labels.append(speaker_id)
            anchor_ids.append(tracklet.tracklet_id)
            embeddings.append(EmbeddingResult(
                f"embedding-{ordinal}", tracklet.tracklet_id, True, vector,
                dimension=2, quality=0.9, intra_window_consistency=1.0,
            ))
        states[speaker_id] = SimpleNamespace(stable_anchor_ids=tuple(anchor_ids))
    # Two source-contiguous UNKNOWN fragments in one continuity group exercise
    # candidate spherical duration aggregation.
    for index in range(2):
        ordinal = len(tracklets)
        start = tracklets[-1].end_us if index else ordinal * 2_000_000
        tracklet = Tracklet(
            f"unknown-{index}", f"region-u-{index}", "unknown-group",
            start, start + 1_000_000, 1_000_000, "MICRO",
        )
        tracklets.append(tracklet)
        labels.append("UNKNOWN")
        embeddings.append(EmbeddingResult(
            f"embedding-u-{index}", tracklet.tracklet_id, True, (1.0, 0.0),
            dimension=2, quality=0.9,
        ))
    return (
        tuple(tracklets), tuple(labels), tuple(embeddings), states,
        SimpleNamespace(state=decision_state),
    )


class BMRCMIntegrationTests(unittest.TestCase):
    def test_h2_singleton_rescues_only_unknown_contiguous_run(self):
        tracklets, labels, embeddings, states, decision = _fixture()
        anchor_evidence = tuple(
            AnchorEvidence(
                item.tracklet_id, tuple(embedding.vector), 1.0,
                item.clean_speech_us, item.continuity_group_id,
                item.continuity_group_id, item.start_us, item.end_us,
            )
            for item, embedding in zip(tracklets, embeddings)
            if item.kind == "ANCHOR"
        )
        result = integrate_bm_rcm(
            tracklets=tracklets, baseline_labels=labels, embeddings=embeddings,
            anchor_evidence=anchor_evidence, states=states, decision=decision,
            diarization_config=DiarizationConfig(anchor_quality_min=0.5),
        )
        self.assertEqual(result.candidate_run_count, 1)
        self.assertEqual(result.singleton_count, 1)
        self.assertEqual(result.labels[-2:], ("SPEAKER_00", "SPEAKER_00"))
        self.assertEqual(result.rescued_duration_us, 2_000_000)
        self.assertEqual(result.fail_closed_count, 0)
        self.assertGreater(result.coordinate_work, 0)
        self.assertGreater(result.batch_count, 0)
        self.assertNotIn("vector", str(result.redacted_diagnostics()).lower())
        self.assertNotIn("unknown-", str(result.redacted_diagnostics()))

    def test_h1_and_insufficient_anchor_preparation_fail_closed_without_mutation(self):
        for decision_state, enough_anchors in (("H1_CONFIRMED", True), ("H2_CONFIRMED", False)):
            tracklets, labels, embeddings, states, decision = _fixture(
                decision_state=decision_state, enough_anchors=enough_anchors,
            )
            result = integrate_bm_rcm(
                tracklets=tracklets, baseline_labels=labels, embeddings=embeddings,
                states=states, decision=decision,
                diarization_config=DiarizationConfig(anchor_quality_min=0.5),
            )
            self.assertEqual(result.labels, labels)
            self.assertEqual(result.singleton_count, 0)
            self.assertGreaterEqual(result.fail_closed_count, 1)
            self.assertTrue(result.preparation_failed)

    def test_existing_assigned_and_overlap_labels_are_exactly_preserved(self):
        tracklets, labels, embeddings, states, decision = _fixture()
        labels = labels[:-1] + ("OVERLAP",)
        result = integrate_bm_rcm(
            tracklets=tracklets, baseline_labels=labels, embeddings=embeddings,
            states=states, decision=decision,
            diarization_config=DiarizationConfig(anchor_quality_min=0.5),
        )
        self.assertEqual(result.labels[-1], "OVERLAP")
        self.assertEqual(result.unchanged_existing_assigned_us, 0)


if __name__ == "__main__":
    unittest.main()
