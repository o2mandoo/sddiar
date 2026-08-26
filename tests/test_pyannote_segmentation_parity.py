import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_pyannote_segmentation_parity.py"
SPEC = importlib.util.spec_from_file_location("evaluate_pyannote_segmentation_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
parity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parity
SPEC.loader.exec_module(parity)


def payload(*, speech_delta=0.0, events=None, argmax=None):
    events = events if events is not None else [[1_000_000, 0.5]]
    argmax = argmax if argmax is not None else [0, 1, 1, 2]
    return {
        "frames": [
            [0, 100_000, 0.8 + speech_delta, 0.1, 0.0],
            [1, 200_000, 0.9 + speech_delta, 0.2, 0.4],
        ],
        "events": events,
        "window_argmax": argmax,
    }


@unittest.skipIf(numpy is None, "numpy is an optional runtime dependency")
class PyannoteSegmentationParityTests(unittest.TestCase):
    def test_identical_trace_passes_all_gates(self):
        result = parity._segment_metrics(payload(), payload())
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["metrics"]["diagnostic_events"]["f1"], 1.0)

    def test_argmax_or_probability_drift_fails_closed(self):
        changed = payload(speech_delta=0.05, argmax=[0, 2, 2, 2])
        result = parity._segment_metrics(payload(), changed)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["argmax_agreement"])
        self.assertFalse(result["checks"]["speech_mae"])

    def test_event_matching_is_deterministic_with_collar(self):
        result = parity._match_events(
            [[1_000_000, 0.4], [2_000_000, 0.5]],
            [[1_100_000, 0.4], [2_300_001, 0.5]],
            collar_us=250_000,
        )
        self.assertEqual(result["matched_event_count"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)

    def test_failed_int8_parity_keeps_only_fp32_reference_candidate(self):
        selection = parity.select_evidence_candidate(
            int8_parity_passed=False,
            int8_not_slower=False,
        )
        self.assertEqual(selection["selected_artifact"], "fp32")
        self.assertEqual(selection["status"], "FP32_EVIDENCE_CANDIDATE_INT8_REJECTED")
        self.assertTrue(selection["eligible_for_later_annotated_scd_osd_calibration"])


if __name__ == "__main__":
    unittest.main()
