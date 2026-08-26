import copy
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "build_rnnoise_offline.py"
SPEC = importlib.util.spec_from_file_location("build_rnnoise_offline_test", SCRIPT)
assert SPEC and SPEC.loader
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


def _descriptor(name, digest="1" * 64, size=1):
    return {"logical_name": name, "sha256": digest, "bytes": size}


def _valid_attestation(target="linux-x86_64"):
    records = []
    spec = plan.TARGETS[target]
    payload = {
        "schema_version": "1.0",
        "kind": "rnnoise-offline-build-attestation",
        "experimental": True,
        "default_enabled": False,
        "production_approved": False,
        "source": {
            "repository": plan.SOURCE_URL,
            "expected_commit": plan.SOURCE_COMMIT,
            "checkout_commit": plan.SOURCE_COMMIT,
            "source_archive": _descriptor("rnnoise-source.tar", "2" * 64),
            "tracked_tree_sha256": "3" * 64,
            "license_spdx": plan.LICENSE_SPDX,
        },
        "submodules": {
            "records": records,
            "count": 0,
            "canonical_sha256": plan._canonical_sha256(records),
            "explicit_zero_allowed": True,
        },
        "model": {
            "archive": _descriptor(plan.MODEL_TAR_NAME, plan.MODEL_TAR_SHA256),
            "expected_sha256": plan.MODEL_TAR_SHA256,
            "model_version": plan.MODEL_TAR_SHA256,
            "imported_offline": True,
            "staged_files": [
                {"relative_path": "src/rnnoise_data.c", "sha256": "9" * 64, "bytes": 1},
                {"relative_path": "src/rnnoise_data.h", "sha256": "a" * 64, "bytes": 1},
            ],
        },
        "target": {"id": target, "arch": spec["arch"], "endianness": "little"},
        "host": {"system": spec["system"], "machine": spec["machines"][0], "matches_target": True},
        "toolchain": {
            "manifest": _descriptor("toolchain.json", "4" * 64),
            "manifest_payload_sha256": "5" * 64,
            "required_roles": list(spec["toolchain_roles"]),
        },
        "configuration": {
            "commands": plan._build_commands(target),
            "autogen_executed": False,
            "download_model_executed": False,
            "x86_rtcd": False,
            "compile_time_vectorization": "target_compiler_default_not_scalar_claim",
            "jobs": 1,
            "build_network_required_state": "disabled",
        },
        "native_binary": _descriptor(spec["binary_name"], "6" * 64),
        "build_log": _descriptor("build.log", "7" * 64),
        "dependency_report": _descriptor("dependencies.json", "8" * 64),
        "validation": {
            "single_target_native_build_recorded": True,
            "binary_functional_smoke": "not_run",
            "four_platform_native_validation": "not_run",
            "xeon_validation": "not_run",
            "independent_review": "not_run",
            "release_authority": "none",
        },
        "redaction": {"local_paths": "omitted", "environment": "omitted"},
    }
    plan._attach_integrity(payload)
    return payload


class RNNoiseBuildPlanTests(unittest.TestCase):
    def test_plan_covers_five_targets_and_never_calls_autogen_or_download(self):
        payload = plan.build_plan()
        self.assertFalse(payload["production_approved"])
        self.assertFalse(payload["default_enabled"])
        self.assertEqual(set(payload["targets"]), set(plan.TARGETS))
        self.assertEqual(payload["source"]["commit"], plan.SOURCE_COMMIT)
        self.assertEqual(payload["model"]["archive_sha256"], plan.MODEL_TAR_SHA256)
        self.assertEqual(payload["source"]["license_spdx"], "BSD-3-Clause")
        self.assertEqual(payload["approval_boundary"]["four_platform_native_validation"], "not_run")
        self.assertEqual(payload["approval_boundary"]["xeon_validation"], "not_run")
        for target, record in payload["targets"].items():
            self.assertTrue(record["target_native_required"], target)
            self.assertIsInstance(record["commands"], list)
            self.assertTrue(all(isinstance(command, list) for command in record["commands"]))
            flattened = " ".join(token for command in record["commands"] for token in command).lower()
            for forbidden in ("autogen.sh", "download_model.sh", "wget", "curl"):
                self.assertNotIn(forbidden, flattened)
            self.assertIn("autoreconf", record["commands"][0])
            self.assertIn("--disable-x86-rtcd", record["commands"][1])
            self.assertIn("--disable-shared", record["commands"][1])
            self.assertIn("--enable-static", record["commands"][1])
            self.assertEqual(record["commands"][2][1], "-j1")

    def test_attestation_validator_accepts_only_nonproduction_pins(self):
        payload = _valid_attestation()
        self.assertEqual(plan.validate_attestation(payload), [])

        mutations = {
            "production": lambda value: value.__setitem__("production_approved", True),
            "model": lambda value: value["model"].__setitem__("model_version", "0" * 64),
            "commit": lambda value: value["source"].__setitem__("checkout_commit", "0" * 40),
            "autogen": lambda value: value["configuration"].__setitem__("autogen_executed", True),
            "download": lambda value: value["configuration"].__setitem__("download_model_executed", True),
            "network": lambda value: value["configuration"].__setitem__(
                "build_network_required_state", "enabled"
            ),
            "four_platform_claim": lambda value: value["validation"].__setitem__(
                "four_platform_native_validation", "passed"
            ),
            "xeon_claim": lambda value: value["validation"].__setitem__("xeon_validation", "passed"),
            "host": lambda value: value["host"].__setitem__("machine", "aarch64"),
            "binary_path": lambda value: value["native_binary"].__setitem__(
                "logical_name", "/private/build/rnnoise_demo"
            ),
            "functional_smoke_claim": lambda value: value["validation"].__setitem__(
                "binary_functional_smoke", "passed"
            ),
            "native_record_removed": lambda value: value["validation"].__setitem__(
                "single_target_native_build_recorded", False
            ),
            "output_hash_tamper": lambda value: value["native_binary"].__setitem__(
                "sha256", "f" * 64
            ),
        }
        for label, mutate in mutations.items():
            candidate = copy.deepcopy(payload)
            mutate(candidate)
            self.assertTrue(plan.validate_attestation(candidate), label)

    def test_attestation_validator_binds_submodule_records_and_commands(self):
        payload = _valid_attestation()
        payload["submodules"] = {
            "records": [{"state": " ", "commit": "a" * 40, "path": "vendor/one"}],
            "count": 1,
            "canonical_sha256": "0" * 64,
            "explicit_zero_allowed": True,
        }
        self.assertIn("submodule_canonical_sha256_mismatch", plan.validate_attestation(payload))
        payload = _valid_attestation()
        payload["configuration"]["commands"].insert(0, ["./autogen.sh"])
        issues = plan.validate_attestation(payload)
        self.assertIn("build_commands_mismatch", issues)
        self.assertIn("forbidden_download_or_autogen_command", issues)

    def test_cli_requires_outer_attestation_hash_and_reports_structure_only_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attestation.json"
            path.write_text(json.dumps(_valid_attestation(), sort_keys=True), encoding="utf-8")
            expected = plan._sha256(path)
            output = io.StringIO()
            with redirect_stdout(output):
                status = plan.main(
                    ["verify-attestation", str(path), "--expected-sha256", expected]
                )
            self.assertEqual(status, 0)
            report = json.loads(output.getvalue())
            self.assertTrue(report["ok"])
            self.assertEqual(report["verification_scope"], "caller_hash_bound_structure_only")
            self.assertEqual(report["cryptographic_authenticity"], "not_verified")

            with redirect_stdout(io.StringIO()):
                bad = plan.main(
                    ["verify-attestation", str(path), "--expected-sha256", "0" * 64]
                )
            self.assertEqual(bad, 1)

    def test_toolchain_manifest_requires_target_specific_hashed_roles(self):
        target = "macos-arm64"
        components = [
            {"role": role, "version": "fixture", "sha256": "a" * 64}
            for role in plan.TARGETS[target]["toolchain_roles"]
        ]
        payload = {
            "schema_version": "1.0",
            "target": target,
            "production_approved": False,
            "components": components,
        }
        self.assertEqual(plan._validate_toolchain_manifest(payload, target), [])
        payload["components"].pop()
        self.assertTrue(
            any(issue.startswith("toolchain_required_roles_missing") for issue in plan._validate_toolchain_manifest(payload, target))
        )

    def test_import_verifier_rejects_symlink_even_when_target_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "actual.bin"
            target.write_bytes(b"approved")
            link = root / "linked.bin"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                plan._verified_input(
                    link,
                    plan._sha256(target),
                    label="fixture import",
                )

    def test_safe_model_staging_rejects_traversal_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "src").mkdir(parents=True)
            (source / "model_version").write_text(plan.MODEL_TAR_SHA256, encoding="utf-8")
            model = root / plan.MODEL_TAR_NAME
            with tarfile.open(model, "w:gz") as archive:
                for name, value in (
                    ("src/rnnoise_data.c", b"model-c"),
                    ("src/rnnoise_data.h", b"model-h"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
            original_sha = plan._sha256
            with mock.patch.object(
                plan,
                "_sha256",
                side_effect=lambda path: plan.MODEL_TAR_SHA256
                if Path(path).name == plan.MODEL_TAR_NAME
                else original_sha(Path(path)),
            ):
                result = plan.stage_model(model, source)
            self.assertTrue(result["ok"])
            self.assertEqual(len(result["staged_files"]), 2)
            with mock.patch.object(plan, "_sha256", return_value=plan.MODEL_TAR_SHA256):
                with self.assertRaises(FileExistsError):
                    plan.stage_model(model, source)

            unsafe_source = root / "unsafe-source"
            (unsafe_source / "src").mkdir(parents=True)
            (unsafe_source / "model_version").write_text(plan.MODEL_TAR_SHA256, encoding="utf-8")
            unsafe = root / plan.MODEL_TAR_NAME
            # Recreate at the same pinned logical name after the safe fixture is no longer needed.
            unsafe.unlink()
            with tarfile.open(unsafe, "w:gz") as archive:
                value = b"escape"
                info = tarfile.TarInfo("../escape")
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
            with mock.patch.object(plan, "_sha256", return_value=plan.MODEL_TAR_SHA256):
                with self.assertRaisesRegex(ValueError, "unsafe member"):
                    plan.stage_model(unsafe, unsafe_source)


if __name__ == "__main__":
    unittest.main()
