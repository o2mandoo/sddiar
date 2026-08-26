from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from sddiar.contracts import EmbeddingResult, SpeakerState, Tracklet


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "run_campplus_rescue_experiment.py"
    spec = importlib.util.spec_from_file_location("campplus_rescue_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def embedding(tracklet_id: str, vector: tuple[float, ...], *, valid: bool = True) -> EmbeddingResult:
    return EmbeddingResult(
        f"e-{tracklet_id}", tracklet_id, valid,
        vector if valid else None,
        dimension=len(vector) if valid else None,
        valid_window_count=1 if valid else 0,
        clean_window_coverage=1.0 if valid else 0.0,
        intra_window_consistency=1.0 if valid else 0.0,
        quality=1.0 if valid else 0.0,
        failure_reason=None if valid else "invalid",
    )


class CampPlusRescueTests(unittest.TestCase):
    def test_challenger_centroids_use_only_authority_anchor_ids_and_weights(self):
        module = load_module()
        anchors = (
            SimpleNamespace(tracklet_id="a0", weight=3.0),
            SimpleNamespace(tracklet_id="a1", weight=1.0),
            SimpleNamespace(tracklet_id="b0", weight=2.0),
        )
        states = {
            "SPEAKER_00": SimpleNamespace(stable_anchor_ids=("a0", "a1")),
            "SPEAKER_01": SimpleNamespace(stable_anchor_ids=("b0",)),
        }
        challenger = (
            embedding("a0", (1.0, 0.0)),
            embedding("a1", (0.0, 1.0)),
            embedding("b0", (-1.0, 0.0)),
        )
        centroids, diagnostics = module.build_challenger_centroids(anchors, states, challenger)
        expected = (3.0 / math.sqrt(10.0), 1.0 / math.sqrt(10.0))
        self.assertAlmostEqual(centroids["SPEAKER_00"][0], expected[0])
        self.assertAlmostEqual(centroids["SPEAKER_00"][1], expected[1])
        self.assertEqual(centroids["SPEAKER_01"], (-1.0, 0.0))
        self.assertEqual(diagnostics["SPEAKER_00"]["authority_anchor_count"], 2)

    def test_rescue_changes_unknown_only_and_requires_authority_agreement_and_neighbors(self):
        module = load_module()
        tracklets = (
            Tracklet("a", "r", "g0", 0, 1_000_000, 1_000_000, "SUPPORT"),
            Tracklet("u", "r", "g1", 1_000_000, 1_400_000, 400_000, "MICRO"),
            Tracklet("b", "r", "g2", 1_400_000, 2_400_000, 1_000_000, "SUPPORT"),
            Tracklet("fixed", "r", "g3", 2_400_000, 3_400_000, 1_000_000, "SUPPORT"),
        )
        baseline = ("SPEAKER_00", "UNKNOWN", "SPEAKER_00", "SPEAKER_01")
        states = {
            "SPEAKER_00": SpeakerState("SPEAKER_00", (1.0, 0.0), ("a",), 0.0),
            "SPEAKER_01": SpeakerState("SPEAKER_01", (0.0, 1.0), ("fixed",), 0.0),
        }
        resnet = tuple(embedding(tracklet.tracklet_id, (1.0, 0.0)) for tracklet in tracklets)
        challenger = tuple(embedding(tracklet.tracklet_id, (1.0, 0.0)) for tracklet in tracklets)
        labels, diagnostics = module.rescue_labels(
            tracklets=tracklets,
            baseline_labels=baseline,
            resnet_embeddings=resnet,
            resnet_states=states,
            challenger_embeddings=challenger,
            challenger_centroids={"SPEAKER_00": (1.0, 0.0), "SPEAKER_01": (0.0, 1.0)},
            distance_limit=0.25,
            margin_min=0.08,
            require_neighbors=True,
            max_gap_us=1_500_000,
        )
        self.assertEqual(labels, ("SPEAKER_00", "SPEAKER_00", "SPEAKER_00", "SPEAKER_01"))
        self.assertEqual(diagnostics["rescued"], 1)
        self.assertEqual(diagnostics["rescued_duration_us"], 400_000)

        disagreeing = list(challenger)
        disagreeing[1] = embedding("u", (0.0, 1.0))
        labels, diagnostics = module.rescue_labels(
            tracklets=tracklets,
            baseline_labels=baseline,
            resnet_embeddings=resnet,
            resnet_states=states,
            challenger_embeddings=tuple(disagreeing),
            challenger_centroids={"SPEAKER_00": (1.0, 0.0), "SPEAKER_01": (0.0, 1.0)},
            distance_limit=0.25,
            margin_min=0.08,
            require_neighbors=False,
            max_gap_us=1_500_000,
        )
        self.assertEqual(labels, baseline)
        self.assertEqual(diagnostics["authority_disagreement"], 1)


if __name__ == "__main__":
    unittest.main()
