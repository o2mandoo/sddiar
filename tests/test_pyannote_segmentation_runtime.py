import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import wave

from sddiar.pyannote_segmentation_runtime import (
    EXPECTED_LICENSE_URL,
    EXPECTED_METADATA,
    PyannoteSegmentationHashError,
    PyannoteSegmentationMetadataError,
    PyannoteSegmentationOnnxRuntime,
    align_local_speakers,
    bounded_window_starts,
    conservative_within_window_change,
)


class _Value:
    def __init__(self, name, shape, value_type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = value_type


class _Meta:
    def __init__(self, values):
        self.custom_metadata_map = values


class _FakeSession:
    def __init__(self, np, *, metadata=None, variant="fp32"):
        self.np = np
        self.metadata = dict(EXPECTED_METADATA)
        self.metadata["license"] = EXPECTED_LICENSE_URL
        self.metadata["onnx.quant.pre_process" if variant == "fp32" else "onnx.infer"] = "onnxruntime.quant"
        if metadata:
            self.metadata.update(metadata)
        self.calls = []

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_modelmeta(self):
        return _Meta(self.metadata)

    def get_inputs(self):
        return [_Value("x", ["N", 1, "T"])]

    def get_outputs(self):
        return [_Value("y", ["N", "T", 7])]

    def run(self, _outputs, inputs):
        tensor = self.np.asarray(inputs["x"], dtype=self.np.float32)
        self.calls.append(tensor.copy())
        frames = 589
        probabilities = self.np.full((frames, 7), 0.001, dtype=self.np.float64)
        probabilities[:, 0] = 0.003
        probabilities[: frames // 2, 1] = 0.991
        probabilities[frames // 2 :, 2] = 0.991
        probabilities[frames // 2 - 4 : frames // 2 + 4, 4] = 0.994
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return [self.np.log(probabilities)[self.np.newaxis].astype(self.np.float32)]


def _write_wav(path: Path, seconds: float, *, sample_rate: int = 16_000) -> None:
    count = round(seconds * sample_rate)
    values = [
        round(3000 * math.sin(2.0 * math.pi * 220.0 * index / sample_rate))
        for index in range(count)
    ]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(value.to_bytes(2, "little", signed=True) for value in values))


class PyannoteSegmentationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("numpy is optional")
        cls.np = np

    def _runtime(self, directory: str, *, session=None):
        model = Path(directory) / "model.onnx"
        model.write_bytes(b"inspected-pyannote-fixture")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        return PyannoteSegmentationOnnxRuntime(
            model,
            expected_sha256=digest,
            session=session or _FakeSession(self.np),
        )

    def test_hash_and_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"model")
            with self.assertRaises(PyannoteSegmentationHashError):
                PyannoteSegmentationOnnxRuntime(
                    model,
                    expected_sha256="0" * 64,
                    session=_FakeSession(self.np),
                )
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            bad = _FakeSession(self.np, metadata={"window_size": "80000"})
            with self.assertRaisesRegex(PyannoteSegmentationMetadataError, "window_size"):
                PyannoteSegmentationOnnxRuntime(model, expected_sha256=digest, session=bad)
            non_cpu = _FakeSession(self.np)
            non_cpu.get_providers = lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            with self.assertRaisesRegex(PyannoteSegmentationMetadataError, "CPU-only"):
                PyannoteSegmentationOnnxRuntime(model, expected_sha256=digest, session=non_cpu)

    def test_default_session_uses_central_factory_with_one_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"model")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            fake = _FakeSession(self.np)
            with mock.patch(
                "sddiar.pyannote_segmentation_runtime.create_ort_session",
                return_value=fake,
            ) as factory:
                PyannoteSegmentationOnnxRuntime(model, expected_sha256=digest)
            factory.assert_called_once_with(model, threads=1)

    def test_bounded_window_accounting_and_edge_padding_are_deterministic(self):
        self.assertEqual(bounded_window_starts(0, 160_000), (0,))
        self.assertEqual(bounded_window_starts(0, 168_000), (0, 16_000))
        self.assertEqual(bounded_window_starts(0, 176_000), (0, 16_000))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            _write_wav(path, 10.5)
            session = _FakeSession(self.np)
            runtime = self._runtime(directory, session=session)
            first, trace_first = runtime.process_wav_with_trace(path)
            second, trace_second = runtime.process_wav_with_trace(path)
            self.assertEqual(first.window_count, 2)
            self.assertEqual(first.padded_window_count, 1)
            self.assertEqual(trace_first.window_argmax, trace_second.window_argmax)
            self.assertEqual(trace_first.permutation_sha256, trace_second.permutation_sha256)
            self.assertTrue(all(call.shape == (1, 1, 160_000) for call in session.calls))
            self.assertTrue(self.np.all(session.calls[1][0, 0, 152_000:] == 0.0))
            self.assertLessEqual(first.frames[-1].end_us, 10_500_000)
            serialized = first.to_json()
            self.assertNotIn(str(path), serialized)
            self.assertNotIn("local_speaker_probabilities\":[", serialized)
            self.assertIn('"approved_for_tracklet_split":false', serialized)

    def test_permutation_alignment_recovers_reference_axis_order(self):
        rng = self.np.random.default_rng(3)
        reference = rng.uniform(size=(60, 3))
        reference /= reference.sum(axis=1, keepdims=True)
        shuffled = reference[:, (2, 0, 1)]
        ones = self.np.ones(60)
        zeros = self.np.zeros(60)
        permutation = align_local_speakers(
            shuffled,
            ones,
            zeros,
            reference,
            ones,
            zeros,
            ones,
            np=self.np,
        )
        self.assertTrue(self.np.allclose(shuffled[:, permutation], reference))

    def test_change_evidence_requires_clean_different_dominant_speakers(self):
        activity = self.np.zeros((100, 3), dtype=self.np.float64)
        activity[:50, 0] = 0.98
        activity[50:, 1] = 0.98
        speech = self.np.full(100, 0.99)
        overlap = self.np.zeros(100)
        change = conservative_within_window_change(
            activity,
            speech,
            overlap,
            context_frames=10,
            min_clean_confidence=0.55,
            min_purity=0.65,
            np=self.np,
        )
        self.assertGreater(float(change.max()), 0.8)
        suppressed = conservative_within_window_change(
            activity,
            speech,
            self.np.ones(100),
            context_frames=10,
            min_clean_confidence=0.55,
            min_purity=0.65,
            np=self.np,
        )
        self.assertEqual(float(suppressed.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
