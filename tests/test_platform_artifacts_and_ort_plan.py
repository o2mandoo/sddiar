import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "build_ort_no_telemetry.py"
SPEC = importlib.util.spec_from_file_location("build_ort_no_telemetry_test", SCRIPT)
assert SPEC and SPEC.loader
ort_plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ort_plan)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PlatformArtifactTests(unittest.TestCase):
    def test_windows_manifest_is_complete_and_hashes_are_local(self):
        root = ROOT / "artifacts" / "dev-windows-x86_64"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["production_approved"])
        self.assertEqual(manifest["target"], "windows-x86_64-cp311")
        self.assertEqual(_sha256(root / "requirements.lock"), manifest["offline_lock"]["sha256"])
        self.assertEqual(len(manifest["wheels"]), 7)
        for record in manifest["wheels"]:
            artifact = root / record["path"]
            self.assertTrue(artifact.is_file(), artifact)
            self.assertEqual(_sha256(artifact), record["sha256"], artifact)
        lock = (root / "requirements.lock").read_text(encoding="utf-8")
        self.assertEqual(lock.count("--hash=sha256:"), 7)

    def test_macos_intel_manifest_does_not_hide_missing_runtime_wheels(self):
        root = ROOT / "artifacts" / "dev-macos-x86_64"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["production_approved"])
        self.assertFalse(manifest["installable"])
        self.assertEqual(manifest["required_runtime"]["onnxruntime"]["status"], "missing_official_wheel")
        self.assertEqual(manifest["required_runtime"]["kaldi-native-fbank"]["status"], "missing_official_wheel")
        self.assertFalse((root / "wheels" / "onnxruntime-1.29.0-cp311-cp311-macosx_10_9_x86_64.whl").exists())
        self.assertEqual(
            _sha256(root / "available-components.lock"),
            manifest["available_components_lock"]["sha256"],
        )
        for record in manifest["available_wheels"]:
            artifact = root / record["path"]
            self.assertTrue(artifact.is_file(), artifact)
            self.assertEqual(_sha256(artifact), record["sha256"], artifact)


class OrtBuildPlanTests(unittest.TestCase):
    def test_plan_covers_all_native_targets_with_explicit_telemetry_off(self):
        plan = ort_plan.build_plan()
        self.assertEqual(plan["source"]["commit"], ort_plan.SOURCE_COMMIT)
        self.assertFalse(plan["production_approved"])
        self.assertEqual(set(plan["targets"]), set(ort_plan.TARGETS))
        for target, record in plan["targets"].items():
            self.assertIn("onnxruntime_USE_TELEMETRY=OFF", record["command"])
            self.assertIn("--no_telemetry", record["command"])
            self.assertIn("--build_dir", record["command"])
            build_dir_index = record["command"].index("--build_dir")
            self.assertTrue(record["command"][build_dir_index + 1])
            self.assertIn("--skip_submodule_sync", record["command"])
            self.assertIn("--skip_pip_install", record["command"])
            self.assertIn("--use_vcpkg", record["command"])
            self.assertEqual(record["command"][-1].startswith("CMAKE_OSX_ARCHITECTURES="), target.startswith("macos-"))
            self.assertEqual(record["command_shell"].count("onnxruntime_USE_TELEMETRY=OFF"), 1)

    def test_attestation_validator_rejects_telemetry_free_claims(self):
        payload = {
            "schema_version": "1.0",
            "production_approved": False,
            "source": {"expected_commit": ort_plan.SOURCE_COMMIT, "commit": ort_plan.SOURCE_COMMIT},
            "target": {"id": "linux-x86_64"},
            "artifact": {"path": "missing.whl", "sha256": "0" * 64},
            "toolchain": {},
            "configuration": {
                "onnxruntime_version": ort_plan.ORT_VERSION,
                "cmake_definitions": {"onnxruntime_USE_TELEMETRY": "OFF"},
            },
            "telemetry": {
                "build_flag": "onnxruntime_USE_TELEMETRY=OFF",
                "status": "telemetry_free",
                "evidence": [],
            },
        }
        issues = ort_plan.validate_attestation(payload, check_files=False)
        self.assertIn("telemetry_status_must_remain_not_verified", issues)


if __name__ == "__main__":
    unittest.main()
