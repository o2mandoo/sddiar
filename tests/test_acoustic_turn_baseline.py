import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "acoustic_turn_baseline.py"
    spec = importlib.util.spec_from_file_location("acoustic_turn_baseline_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AcousticTurnBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_synthetic_two_speaker_proxy(self) -> None:
        rate = 8000
        turns = [(0, 800), (800, 1600), (1600, 2400), (2400, 3200)]
        frequencies = [220, 880, 220, 880]
        with tempfile.TemporaryDirectory() as temp:
            wav_path = Path(temp) / "synthetic.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
                samples = []
                for frequency in frequencies:
                    samples.extend(struct.pack("<h", int(12000 * math.sin(2 * math.pi * frequency * index / rate))) for index in range(800))
                wav.writeframes(b"".join(samples))
            boundary_path = Path(temp) / "bounds.json"
            boundary_path.write_text(json.dumps({"turns": [{"start_frame": start, "end_frame": end} for start, end in turns]}))
            predicted = self.module.predict(wav_path, boundary_path, chunk_frames=37)
        result = self.module.score(predicted, [0, 1, 0, 1])
        self.assertEqual(result["result_kind"], "ORACLE_BOUNDARY_PROXY")
        self.assertEqual(result["turns"], 4)
        self.assertEqual(result["cluster_accuracy"], 1.0)

    def test_boundaries_have_no_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bounds.json"
            path.write_text('{"turns":[{"start":0,"end":2}]}')
            self.assertEqual(self.module.load_turns(path), [(0, 2)])


if __name__ == "__main__":
    unittest.main()
