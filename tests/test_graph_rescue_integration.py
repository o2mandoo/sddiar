from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sddiar.contracts import EmbeddingResult, Tracklet
from sddiar.diarization import DiarizationConfig, _materialize


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "run_onnx_diarization_experiment.py"
    spec = importlib.util.spec_from_file_location("onnx_graph_integration_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tracklet(tracklet_id: str, block: str, start_us: int, kind: str) -> Tracklet:
    return Tracklet(
        tracklet_id, f"region-{tracklet_id}", block, start_us, start_us + 1_000_000,
        1_000_000, kind,
    )


class GraphRescueIntegrationTests(unittest.TestCase):
    def test_same_baseline_trace_is_rescued_and_receipt_is_redacted(self):
        module = load_module()
        tracklets = tuple(
            tracklet(f"a{index}", f"block-a{index}", index * 1_000_000, "ANCHOR")
            for index in range(3)
        ) + tuple(
            tracklet(f"b{index}", f"block-b{index}", (index + 3) * 1_000_000, "ANCHOR")
            for index in range(3)
        ) + (tracklet("u", "block-u", 6_000_000, "MICRO"),)
        labels = (
            "SPEAKER_00", "SPEAKER_00", "SPEAKER_00",
            "SPEAKER_01", "SPEAKER_01", "SPEAKER_01", "UNKNOWN",
        )
        vectors = {
            **{f"a{index}": (1.0, 0.0, 0.0) for index in range(3)},
            **{f"b{index}": (-1.0, 0.0, 0.0) for index in range(3)},
            "u": (1.0, 0.0, 0.0),
        }
        embeddings = tuple(
            EmbeddingResult(
                f"embedding-{item.tracklet_id}", item.tracklet_id, True,
                vectors[item.tracklet_id], dimension=3, valid_window_count=1,
                clean_window_coverage=1.0, intra_window_consistency=1.0, quality=1.0,
            )
            for item in tracklets
        )
        baseline_spans = _materialize(labels, tracklets, (), 7_000_000, DiarizationConfig())
        baseline_trace = SimpleNamespace(labels=labels, spans=baseline_spans)
        reference = tuple(
            {"start_us": index * 1_000_000, "end_us": (index + 1) * 1_000_000,
             "speaker": "REF_00" if index < 3 or index == 6 else "REF_01"}
            for index in range(7)
        )
        report, candidate_spans, graph_result = module._graph_rescue_report(
            tracklets=tracklets,
            baseline_trace=baseline_trace,
            embeddings=embeddings,
            decision=SimpleNamespace(state="H2_CONFIRMED"),
            protected_overlap_spans=(),
            source_duration_us=7_000_000,
            reference=reference,
            score=module._score,
            materialize_config=DiarizationConfig(),
        )
        self.assertEqual(graph_result.applied_count, 1)
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["rescued_duration_us"], 1_000_000)
        self.assertEqual(report["changed_existing_assigned_us"], 0)
        self.assertIn("algorithm_version", report["policy"])
        self.assertTrue(report["graph_diagnostics_redacted"]["parity_passed"])
        self.assertNotIn("tracklet_id", str(report))
        self.assertNotIn("block-a", str(report))
        self.assertNotIn("vector", str(report).lower())
        self.assertTrue(any(span.speaker_id == "SPEAKER_00" and span.start_us == 6_000_000 for span in candidate_spans))

    def test_cli_flag_is_present_but_default_off(self):
        module = load_module()
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

        with patch.object(module, "run_experiment", side_effect=fake_run):
            self.assertEqual(module.main(["fixture.wav", "--silero", "s.onnx", "--wespeaker", "w.onnx"]), 0)
        self.assertFalse(captured["graph_rescue_experimental"])


if __name__ == "__main__":
    unittest.main()
