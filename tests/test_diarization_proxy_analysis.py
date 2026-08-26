import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "analyze_diarization_proxy.py"
    spec = importlib.util.spec_from_file_location("diarization_proxy_analysis_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class DiarizationProxyAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.reference = {
            "schema": "clova_reference_timing_v2",
            "turns": [
                {"speaker_id": "REF_A", "start_sec": 0, "end_sec": 10},
                {"speaker_id": "REF_B", "start_sec": 10, "end_sec": 20},
                {"speaker_id": "REF_A", "start_sec": 20, "end_sec": 30},
            ],
        }
        self.spans = {
            "schema": "redacted_diarization_spans_v1",
            "spans": [
                {"speaker_id": "SYS_X", "start_us": 0, "end_us": 4_000_000, "text": "must not appear"},
                {"speaker_id": "UNKNOWN", "start_us": 4_000_000, "end_us": 6_000_000},
                {"speaker_id": "SYS_Y", "start_us": 6_000_000, "end_us": 10_000_000},
                {"speaker_id": "SYS_Y", "start_us": 10_000_000, "end_us": 15_000_000},
                {"speaker_id": "SYS_X", "start_us": 15_000_000, "end_us": 18_000_000},
                {"speaker_id": "SYS_Y", "start_us": 20_000_000, "end_us": 25_000_000},
            ],
        }

    def test_hand_calculated_optimal_mapping_and_durations(self) -> None:
        result = self.module.analyze(self.reference, self.spans)

        self.assertEqual(result["mapping"], {"SYS_X": "REF_B", "SYS_Y": "REF_A"})
        self.assertEqual(result["mapping_intersection_duration_us"], 12_000_000)
        overall = result["overall"]
        self.assertEqual(overall["reference_duration_us"], 30_000_000)
        self.assertEqual(overall["assigned_duration_us"], 21_000_000)
        self.assertEqual(overall["correct_duration_us"], 12_000_000)
        self.assertEqual(overall["wrong_duration_us"], 9_000_000)
        self.assertEqual(overall["unknown_duration_us"], 2_000_000)
        self.assertEqual(overall["uncovered_duration_us"], 7_000_000)
        self.assertAlmostEqual(overall["assigned_accuracy"], 12 / 21)
        self.assertEqual(result["timeline_intersection"]["duration_us"], 23_000_000)
        self.assertAlmostEqual(overall["unknown_rate_within_system_detected_speech"], 2 / 23)

        by_speaker = result["per_reference_speaker"]
        self.assertEqual(by_speaker["REF_A"]["assigned_duration_us"], 13_000_000)
        self.assertEqual(by_speaker["REF_A"]["correct_duration_us"], 9_000_000)
        self.assertEqual(by_speaker["REF_A"]["wrong_duration_us"], 4_000_000)
        self.assertEqual(by_speaker["REF_A"]["unknown_duration_us"], 2_000_000)
        self.assertEqual(by_speaker["REF_A"]["uncovered_duration_us"], 5_000_000)
        self.assertEqual(by_speaker["REF_B"]["assigned_duration_us"], 8_000_000)
        self.assertEqual(by_speaker["REF_B"]["correct_duration_us"], 3_000_000)
        self.assertEqual(by_speaker["REF_B"]["wrong_duration_us"], 5_000_000)
        self.assertEqual(by_speaker["REF_B"]["uncovered_duration_us"], 2_000_000)

        self.assertEqual({key: result["turns"][key] for key in ("correct", "wrong", "uncovered")},
                         {"correct": 1, "wrong": 2, "uncovered": 0})
        self.assertAlmostEqual(result["fairness_gaps"]["assigned_accuracy_max_minus_min"], 9 / 13 - 3 / 8)
        self.assertEqual(result["duration_buckets"]["10s_plus"]["turn_count"], 3)

    def test_redaction_and_clova_warning_are_explicit(self) -> None:
        result = self.module.analyze(self.reference, self.spans)
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("must not appear", encoded)
        self.assertTrue(any(warning.startswith("Clova turn end equals the next turn start") for warning in result["warnings"]))
        self.assertIn("not VAD recall or DER", " ".join(result["warnings"]))
        self.assertEqual(result["redaction"]["transcript_text"], "omitted")

    def test_cli_emits_json_only_and_no_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "person_name_reference.json"
            spans_path = root / "meeting_transcript_spans.json"
            reference_path.write_text(json.dumps(self.reference), encoding="utf-8")
            spans_path.write_text(json.dumps(self.spans), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(self.module._main([str(reference_path), str(spans_path)]), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["schema"], "diarization_proxy_analysis_v1")
            self.assertNotIn("person_name_reference", stdout.getvalue())
            self.assertNotIn("meeting_transcript", stdout.getvalue())

    def test_rejects_non_two_speaker_reference(self) -> None:
        with self.assertRaisesRegex(self.module.ProxyAnalysisError, "exactly two"):
            self.module.analyze({"turns": [{"speaker_id": "REF_A", "start_sec": 0, "end_sec": 1}]}, {"spans": []})


if __name__ == "__main__":
    unittest.main()
