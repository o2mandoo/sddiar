import base64
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from sddiar.model_pack import (
    DigestSignatureVerifier, ModelPackVerifier, VerifiedModelPack, canonical_manifest_bytes,
)
from sddiar.offline import OfflinePolicyViolation, local_path, reject_url


class ModelPackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "models").mkdir()
        self.model = self.root / "models" / "m.onnx"
        self.model.write_bytes(b"immutable model")

    def tearDown(self): self.tmp.cleanup()

    def manifest(self):
        raw = self.model.read_bytes()
        return {"schema_version": "1.0", "pack_id": "p", "pack_version": "1",
                "production_approved": False, "integrity": {},
                "runtime_compatibility": {"onnxruntime": {"exact_build_id": "b", "exact_version": "1"},
                    "allowed_execution_providers": ["CPUExecutionProvider"],
                    "target_matrix": [{"os": "linux", "arch": "x86_64", "python_abi": "cp311"}]},
                "files": [{"file_id": "m", "role": "model", "relative_path": "models/m.onnx",
                           "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]}

    def verifier(self, manifest):
        return ModelPackVerifier(self.root, runtime={"exact_build_id": "b", "exact_version": "1",
            "execution_provider": "CPUExecutionProvider", "os": "linux", "arch": "x86_64", "python_abi": "cp311"},
            signature_verifier=DigestSignatureVerifier(b"test-key"))

    def sign(self, m):
        m["integrity"] = {"signer_key_id": "k"}
        digest = hashlib.sha256(b"test-key" + canonical_manifest_bytes(m)).digest()
        m["integrity"]["signature"] = base64.b64encode(digest).decode()
        return m

    def signed(self):
        return self.sign(self.manifest())

    def test_valid_signed_pack(self):
        pack = self.verifier(self.signed()).verify(self.signed())
        self.assertEqual(pack.pack_id, "p")
        self.assertEqual(pack.trust_level, "DEVELOPMENT")
        self.assertFalse(pack.release_trusted)

    def test_missing_artifact_fails_closed(self):
        m = self.signed(); self.model.unlink()
        with self.assertRaises(Exception) as ctx: self.verifier(m).verify(m)
        self.assertEqual(getattr(ctx.exception, "code", None), "MODEL_NOT_FOUND")

    def test_modified_artifact_fails_closed(self):
        m = self.signed(); self.model.write_bytes(b"modified")
        with self.assertRaises(Exception) as ctx: self.verifier(m).verify(m)
        self.assertEqual(getattr(ctx.exception, "code", None), "MODEL_HASH_MISMATCH")

    def test_path_traversal_is_rejected(self):
        m = self.signed(); m["files"][0]["relative_path"] = "../outside"; self.sign(m)
        with self.assertRaises(Exception): self.verifier(m).verify(m)

    def test_symlink_is_rejected_even_when_it_resolves_inside_pack(self):
        link = self.root / "models" / "link.onnx"
        link.symlink_to(self.model.name)
        m = self.signed(); m["files"][0]["relative_path"] = "models/link.onnx"; self.sign(m)
        with self.assertRaises(Exception) as ctx:
            self.verifier(m).verify(m)
        self.assertEqual(getattr(ctx.exception, "code", None), "MODEL_HASH_MISMATCH")

    def test_url_is_rejected_without_fallback(self):
        m = self.signed(); m["files"][0]["relative_path"] = "https://example/model.onnx"; self.sign(m)
        with self.assertRaises((OfflinePolicyViolation, Exception)): self.verifier(m).verify(m)

    def test_runtime_mismatch_is_rejected(self):
        m = self.signed()
        with self.assertRaises(Exception) as ctx:
            ModelPackVerifier(self.root, runtime={"exact_build_id": "other", "exact_version": "1",
                "execution_provider": "CPUExecutionProvider", "os": "linux", "arch": "x86_64", "python_abi": "cp311"},
                signature_verifier=DigestSignatureVerifier(b"test-key")).verify(m)
        self.assertEqual(getattr(ctx.exception, "code", None), "MODEL_RUNTIME_INCOMPATIBLE")

    def test_local_path_rejects_file_and_http_urls(self):
        for value in ("file:///tmp/m", "http://host/m"):
            with self.assertRaises(OfflinePolicyViolation): local_path(value)

    def test_windows_drive_path_is_not_mistaken_for_a_url(self):
        reject_url(r"C:\offline\models\model.onnx")

    def test_empty_or_identity_free_manifest_is_rejected(self):
        for manifest in ({}, {"schema_version": "1", "pack_id": "p", "pack_version": "1",
                              "production_approved": False, "runtime_compatibility": {}, "files": []}):
            with self.subTest(manifest=manifest):
                with self.assertRaises(Exception):
                    ModelPackVerifier(self.root, development_mode=True).verify(manifest)

    def test_verified_pack_is_sealed_and_deeply_immutable(self):
        with self.assertRaises(TypeError):
            VerifiedModelPack("p", "1", {}, (), {}, "0" * 64, "RELEASE", True, True)
        pack = self.verifier(self.signed()).verify(self.signed())
        with self.assertRaises(TypeError):
            pack.manifest["pack_id"] = "forged"
        with self.assertRaises(TypeError):
            pack.runtime_compatibility["onnxruntime"]["exact_version"] = "other"

    def test_release_trust_requires_approved_manifest_and_release_verifier(self):
        class ReleaseDigestVerifier(DigestSignatureVerifier):
            trust_level = "RELEASE"

        manifest = self.manifest()
        manifest["production_approved"] = True
        self.sign(manifest)
        runtime = {"exact_build_id": "b", "exact_version": "1",
                   "execution_provider": "CPUExecutionProvider", "os": "linux",
                   "arch": "x86_64", "python_abi": "cp311"}
        pack = ModelPackVerifier(
            self.root, runtime=runtime, signature_verifier=ReleaseDigestVerifier(b"test-key"),
            require_release_trust=True,
        ).verify(manifest)
        self.assertTrue(pack.release_trusted)
        with self.assertRaises(Exception):
            ModelPackVerifier(
                self.root, runtime=runtime, signature_verifier=DigestSignatureVerifier(b"test-key"),
                require_release_trust=True,
            ).verify(manifest)

    def test_duplicate_calibration_and_artifact_mutation_fail_closed(self):
        manifest = self.manifest()
        first = manifest["files"][0]
        first["role"] = "calibration_profile"
        duplicate = dict(first)
        duplicate["file_id"] = "cal-2"
        duplicate["relative_path"] = "models/cal-2.json"
        second = self.root / duplicate["relative_path"]
        second.write_bytes(self.model.read_bytes())
        duplicate["bytes"] = second.stat().st_size
        duplicate["sha256"] = hashlib.sha256(second.read_bytes()).hexdigest()
        manifest["files"].append(duplicate)
        self.sign(manifest)
        with self.assertRaises(Exception):
            self.verifier(manifest).verify(manifest)

        pack = self.verifier(self.signed()).verify(self.signed())
        self.model.write_bytes(b"changed after verification")
        with self.assertRaises(Exception) as ctx:
            pack.assert_artifacts_unchanged()
        self.assertEqual(getattr(ctx.exception, "code", None), "MODEL_HASH_MISMATCH")


if __name__ == "__main__": unittest.main()
