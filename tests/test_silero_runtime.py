import unittest
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np = None

from sddiar.silero_runtime import SileroOnnxRuntime
from sddiar.vad import VadError, VadUnavailableError


class Meta:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class FakeSession:
    def __init__(self):
        self.calls = []

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [Meta("input", [1, "frames"]), Meta("state", [2, None, 128]), Meta("sr", [])]

    def get_outputs(self):
        return [Meta("output", [1, 1]), Meta("stateN", [2, 1, 128])]

    def run(self, names, feeds):
        self.calls.append(feeds["state"].copy())
        value = float(np.mean(np.abs(feeds["input"])))
        return [np.asarray([[value]], dtype=np.float32), feeds["state"] + 1]


@unittest.skipIf(np is None, "numpy is optional")
class SileroRuntimeTests(unittest.TestCase):
    def test_fake_runtime_dynamic_contract_and_state_reset(self):
        session = FakeSession()
        runtime = SileroOnnxRuntime(None, session=session, threshold=0.2, window_samples=4, context_samples=0)
        frames = runtime.infer_samples([0.0, 0.0, 1.0, 1.0, 0.0])
        self.assertEqual(len(frames), 2)
        self.assertTrue(frames[0].is_speech)
        self.assertEqual(session.calls[0].max(), 0)
        self.assertEqual(session.calls[1].max(), 1)
        runtime.infer_samples([0.0] * 4)
        self.assertEqual(session.calls[2].max(), 0)

    def test_runtime_rejects_non_cpu_session(self):
        session = FakeSession(); session.get_providers = lambda: ["CUDAExecutionProvider"]
        with self.assertRaises(VadUnavailableError):
            SileroOnnxRuntime(None, session=session)

    def test_runtime_rejects_non_16k(self):
        runtime = SileroOnnxRuntime(None, session=FakeSession())
        with self.assertRaises(VadError):
            runtime.infer_samples([0.0], sample_rate_hz=8000)

    def test_stream_accepts_one_dimensional_numpy_chunks(self):
        runtime = SileroOnnxRuntime(None, session=FakeSession(), threshold=0.2, window_samples=4, context_samples=0)
        chunks = [
            SimpleNamespace(samples=np.asarray([0.0, 0.0], dtype=np.float32), source_start_sample=0,
                            source_end_sample=2, sample_rate_hz=16000, channel_count=1),
            SimpleNamespace(samples=np.asarray([1.0, 1.0, 0.0], dtype=np.float32), source_start_sample=2,
                            source_end_sample=5, sample_rate_hz=16000, channel_count=1),
        ]
        frames = runtime.infer_chunk_stream(chunks)
        self.assertEqual(len(frames), 2)
        self.assertTrue(frames[0].is_speech)


if __name__ == "__main__":
    unittest.main()
