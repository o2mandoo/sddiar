import importlib.util
import io
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("energy_baseline", ROOT / "scripts/energy_vad_window_baseline.py")
baseline = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(baseline)


def make_wav(values, rate=1000):
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes(b"".join(struct.pack("<h", x) for x in values))
    return out.getvalue()


class NonOracleBaselineTests(unittest.TestCase):
    def test_predict_uses_audio_only_and_is_deterministic(self):
        # Two acoustically distinct speech halves, separated by silence.
        values = [0] * 500 + [8000] * 1000 + [0] * 500 + [-8000] * 1000
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.wav"; path.write_bytes(make_wav(values))
            first = baseline.predict(path, window_ms=200, hop_ms=100, energy_threshold=.02)
            second = baseline.predict(path, window_ms=200, hop_ms=100, energy_threshold=.02)
        self.assertEqual(first, second)
        self.assertEqual(first["baseline"], "NON_ORACLE_ENERGY_BASELINE")
        self.assertTrue(any(p["speaker"] == "UNKNOWN" for p in first["predictions"]))
        self.assertTrue(any(p["speaker"] in {"H1", "H2"} for p in first["predictions"]))

    def test_score_maps_anonymous_speakers_against_separate_reference(self):
        values = [10000] * 1000 + [10000 if i % 2 else -10000 for i in range(1000)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.wav"; path.write_bytes(make_wav(values))
            result = baseline.predict(path, window_ms=200, hop_ms=200, energy_threshold=.02)
        score = baseline.score_predictions(result, [
            {"start_us": 0, "end_us": 1_000_000, "speaker": "clova-A"},
            {"start_us": 1_000_000, "end_us": 2_000_000, "speaker": "clova-B"},
        ])
        self.assertEqual(score["baseline"], "NON_ORACLE_ENERGY_BASELINE")
        self.assertGreaterEqual(score["coverage"], 0.9)
        self.assertGreaterEqual(score["speaker_mapped_frame_accuracy"], 0.9)

    def test_reference_parser_accepts_seconds_without_being_prediction_input(self):
        parsed = baseline._reference({"segments": [{"start": 0, "end": 1.5, "speaker_id": "A"}]})
        self.assertEqual(parsed, [{"start_us": 0, "end_us": 1_500_000, "speaker": "A"}])


if __name__ == "__main__":
    unittest.main()
