import hashlib
import tempfile
import unittest
from pathlib import Path

from sddiar.contracts import EmbeddingRegion
from sddiar.embedding import (
    DeterministicFixtureEmbeddingBackend,
    EmbeddingInputContract,
    OnnxRuntimeSpeakerEmbeddingBackend,
    SpeakerEmbeddingBackend,
)
from sddiar.errors import ContractValidationError, ModelHashMismatch, ModelRuntimeIncompatible
from sddiar.model_pack import VerifiedArtifact


def region(i="r1"):
    return EmbeddingRegion(i, "t-" + i, 0, 1_000_000, 800_000, 0.8)


class EmbeddingTests(unittest.TestCase):
    def test_fixture_is_deterministic_and_contract_compatible(self):
        backend = DeterministicFixtureEmbeddingBackend(4)
        self.assertIsInstance(backend, SpeakerEmbeddingBackend)
        first = backend.embed([region(), region("r2")])
        second = backend.embed([region(), region("r2")])
        self.assertEqual(first, second)
        self.assertEqual(first[0].dimension, 4)
        self.assertAlmostEqual(sum(x * x for x in first[0].vector), 1.0, places=6)

    def test_fixture_rejects_invalid_region_input(self):
        with self.assertRaises(ContractValidationError):
            DeterministicFixtureEmbeddingBackend().embed([object()])

    def test_onnx_requires_verified_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.onnx"
            p.write_bytes(b"not-an-onnx-model")
            artifact = VerifiedArtifact("x", "embedding", p, hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size)
            # Missing optional runtime is itself a typed fail-closed result; a
            # present runtime must still reject this invalid graph.
            with self.assertRaises(ModelRuntimeIncompatible):
                OnnxRuntimeSpeakerEmbeddingBackend(artifact, EmbeddingInputContract("x", "y"))

    def test_onnx_rejects_changed_verified_artifact_before_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.onnx"
            p.write_bytes(b"model")
            artifact = VerifiedArtifact("x", "embedding", p, "0" * 64, 5)
            with self.assertRaises(ModelHashMismatch):
                OnnxRuntimeSpeakerEmbeddingBackend(artifact, EmbeddingInputContract("x", "y"))

    def test_onnx_rejects_unverified_or_wrong_role(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.onnx"
            p.write_bytes(b"model")
            artifact = VerifiedArtifact("x", "vad", p, hashlib.sha256(p.read_bytes()).hexdigest(), 5)
            with self.assertRaises(ModelRuntimeIncompatible):
                OnnxRuntimeSpeakerEmbeddingBackend(artifact, EmbeddingInputContract("x", "y"))


if __name__ == "__main__":
    unittest.main()
