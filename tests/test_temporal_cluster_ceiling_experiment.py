from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from sddiar.contracts import EmbeddingResult, SpeakerState, Tracklet
from sddiar.diarization import DiarizationConfig


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "run_temporal_cluster_ceiling_experiment.py"
    spec = importlib.util.spec_from_file_location("temporal_cluster_ceiling_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def embedding(tracklet_id: str, vector: tuple[float, ...]) -> EmbeddingResult:
    return EmbeddingResult(
        f"e-{tracklet_id}", tracklet_id, True, vector,
        dimension=len(vector), valid_window_count=1,
        clean_window_coverage=1.0, intra_window_consistency=1.0, quality=1.0,
    )


class TemporalClusterCeilingTests(unittest.TestCase):
    def test_only_distance_failed_non_micro_unknown_can_be_rescued(self):
        module = load_module()
        vector = (0.48, math.sqrt(1.0 - 0.48 * 0.48))
        tracklets = (
            Tracklet("fixed", "r", "g0", 0, 1_000_000, 1_000_000, "SUPPORT"),
            Tracklet("distance", "r", "g1", 1_000_000, 2_000_000, 1_000_000, "SUPPORT"),
            Tracklet("micro", "r", "g2", 2_000_000, 2_400_000, 400_000, "MICRO"),
            Tracklet("viterbi", "r", "g3", 2_400_000, 3_400_000, 1_000_000, "SUPPORT"),
        )
        baseline = ("SPEAKER_01", "UNKNOWN", "UNKNOWN", "UNKNOWN")
        local = (
            SimpleNamespace(speaker_id="SPEAKER_01", reason_codes=()),
            SimpleNamespace(speaker_id="UNKNOWN", reason_codes=("LOCAL_GATE_FAILED",)),
            SimpleNamespace(speaker_id="UNKNOWN", reason_codes=("LOCAL_GATE_FAILED",)),
            SimpleNamespace(speaker_id="SPEAKER_00", reason_codes=()),
        )
        embeddings = tuple(embedding(tracklet.tracklet_id, vector) for tracklet in tracklets)
        states = {
            "SPEAKER_00": SpeakerState("SPEAKER_00", (1.0, 0.0), ("a",), 0.0),
            "SPEAKER_01": SpeakerState("SPEAKER_01", (-1.0, 0.0), ("b",), 0.0),
        }
        cfg = DiarizationConfig(
            support_margin_min=0.03,
            micro_stable_distance_ceiling=0.35,
            micro_absolute_distance_max=0.35,
        )
        labels, diagnostics = module.rescue_with_cluster_ceilings(
            tracklets=tracklets,
            baseline_labels=baseline,
            baseline_local_assignments=local,
            embeddings=embeddings,
            states=states,
            config=cfg,
            ceilings={"SPEAKER_00": 0.55, "SPEAKER_01": 0.30},
        )
        self.assertEqual(labels, ("SPEAKER_01", "SPEAKER_00", "UNKNOWN", "UNKNOWN"))
        self.assertEqual(diagnostics["rescued"], 1)
        self.assertEqual(diagnostics["micro_locked"], 1)
        self.assertEqual(diagnostics["viterbi_unknown_locked"], 1)

        labels, _ = module.rescue_with_cluster_ceilings(
            tracklets=tracklets,
            baseline_labels=baseline,
            baseline_local_assignments=local,
            embeddings=embeddings,
            states=states,
            config=cfg,
            ceilings={"SPEAKER_00": 0.50, "SPEAKER_01": 0.30},
        )
        self.assertEqual(labels, baseline)


if __name__ == "__main__":
    unittest.main()
