import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

from sddiar.cli import main as cli_main
from sddiar.release import TARGETS, clean_install_plan, scan_zero_network_policy, validate_release_layout


def make_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    targets = {}
    for target in TARGETS:
        target_root = root / "targets" / target
        for directory in ("wheels", "models", "native"):
            (target_root / directory).mkdir(parents=True, exist_ok=True)
        (target_root / "wheels" / "sddiar.whl").write_bytes(b"wheel")
        wheel_hash = hashlib.sha256(b"wheel").hexdigest()
        (target_root / "requirements.lock").write_text(
            f"sddiar==0.3.0 --hash=sha256:{wheel_hash}\n"
        )
        (target_root / "models" / "model.onnx").write_bytes(b"model")
        (target_root / "native" / "ffmpeg.bin").write_bytes(b"native")
        (target_root / "sbom").mkdir()
        (target_root / "sbom" / "sbom.spdx.json").write_text("{}")
        (target_root / "notices").mkdir()
        (target_root / "notices" / "NOTICE").write_text("notice")
        entries = [
            {"path": str(path.relative_to(target_root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in target_root.rglob("*")
            if path.is_file()
        ]
        targets[target] = {"files": entries}
    root.mkdir(exist_ok=True)
    (root / "release-manifest.json").write_text(json.dumps({"targets": targets}))
    return root


def make_production_release(tmp_path: Path, target: str = TARGETS[0]) -> Path:
    """Build a small signed-evidence fixture for the explicit production gate."""
    root = make_release(tmp_path)
    target_root = root / "targets" / target
    model_manifest = {
        "schema_version": "1.0",
        "pack_id": "model-pack-prod",
        "production_approved": True,
        "integrity": {
            "signature": "model-signature",
            "signer_key_id": "model-key-1",
            "verification": {"verified": True, "trusted": True, "evidence_id": "model-verification-1"},
        },
        "files": [{"relative_path": "model.onnx", "sha256": hashlib.sha256(b"model").hexdigest(), "bytes": 5}],
    }
    model_path = target_root / "models" / "manifest.json"
    model_path.write_text(json.dumps(model_manifest), encoding="utf-8")
    (target_root / "sbom" / "sbom.spdx.json").write_text(
        json.dumps({
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "components": [
                {
                    "type": "library",
                    "name": path.name,
                    "version": "fixture",
                    "hashes": [{"alg": "SHA-256", "content": hashlib.sha256(path.read_bytes()).hexdigest()}],
                }
                for path in target_root.rglob("*")
                if path.is_file() and path.parent.name in {"wheels", "models", "native"}
            ],
        }),
        encoding="utf-8",
    )
    evidence = target_root / "evidence"
    evidence.mkdir()
    (evidence / "ort.json").write_text(json.dumps({
        "target": target,
        "production_approved": True,
        "telemetry": {"status": "independently_verified", "build_flag": "onnxruntime_USE_TELEMETRY=OFF"},
        "verification": {
            "verified": True,
            "trusted": True,
            "independent": True,
            "evidence_id": "ort-verification-1",
        },
    }), encoding="utf-8")
    (evidence / "golden.json").write_text(json.dumps({
        "target": target,
        "status": "verified",
        "verified": True,
        "digests": [hashlib.sha256(b"golden-output").hexdigest()],
    }), encoding="utf-8")
    (evidence / "abi.json").write_text(json.dumps({
        "target": target,
        "status": "verified",
        "verified": True,
        "python_abi": "cp311",
        "libraries": [{"name": "onnxruntime", "version": "1.29.0"}],
    }), encoding="utf-8")
    catalog_path = root / "release-manifest.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    section = catalog["targets"][target]
    section["production_approved"] = True
    lock = target_root / "requirements.lock"
    section["offline_lock"] = {"path": "requirements.lock", "sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "target": target}
    section["model_pack"] = {
        "path": "models/manifest.json",
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    for name in ("ort", "golden", "abi"):
        path = evidence / f"{name}.json"
        section[f"{ {'ort':'ort_telemetry_attestation','golden':'golden_digest_report','abi':'target_native_abi_report'}[name] }"] = {
            "path": f"evidence/{name}.json",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    catalog["production_approved"] = True
    catalog["integrity"] = {
        "signature": "catalog-signature",
        "signer_key_id": "catalog-key-1",
        "verification": {"verified": True, "trusted": True, "evidence_id": "catalog-verification-1"},
    }
    section["files"] = [
        {"path": str(path.relative_to(target_root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in target_root.rglob("*")
        if path.is_file()
    ]
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return root


class ReleaseTests(unittest.TestCase):
    def test_layout_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            self.assertTrue(validate_release_layout(make_release(Path(temp))).ok)

    def test_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            (root / "targets" / TARGETS[0] / "models" / "model.onnx").write_bytes(b"tampered")
            report = validate_release_layout(root)
            self.assertFalse(report.ok)
            self.assertTrue(any(issue.code == "HASH_MISMATCH" for issue in report.issues))

    def test_network_scan_flags_request_but_not_urllib_parse(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_file = root / "request.py"
            request_file.write_text("import urllib.request\nurllib.request.urlopen('https://example.invalid')\n")
            parse_file = root / "parse.py"
            parse_file.write_text("from urllib.parse import urlparse\nurlparse('file:///local')\n")
            self.assertTrue(any(issue.code == "NETWORK_IMPORT" for issue in scan_zero_network_policy([request_file])))
            self.assertEqual(scan_zero_network_policy([parse_file]), [])

    def test_project_static_scan_has_no_network_issue(self) -> None:
        source_root = Path(__file__).parents[1] / "src" / "sddiar"
        self.assertEqual(scan_zero_network_policy([source_root]), [])

    def test_clean_plan_is_offline_and_uses_alias_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            plan = clean_install_plan(TARGETS[0], root)
            self.assertIn("--no-index", plan[2])
            self.assertIn("--require-hashes", plan[2])
            self.assertIn("wheels", plan[2])
            self.assertIn("-r", plan[2])
            self.assertIn("requirements.lock", plan[2])

    def test_windows_clean_plan_uses_python_launcher_and_windows_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            plan = clean_install_plan("windows-x64-cp311", root)
            self.assertIn("py -3.11 -m pip", plan[2])
            self.assertIn("Scripts\\activate", plan[1])
            self.assertIn("--no-index", plan[2])
            self.assertIn("--require-hashes", plan[2])

    def test_production_mode_rejects_development_placeholders_and_empty_sbom(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            report = validate_release_layout(root, targets=(TARGETS[0],), mode="production")
            codes = {issue.code for issue in report.issues}
            self.assertFalse(report.ok)
            self.assertIn("PRODUCTION_APPROVAL_REQUIRED", codes)
            self.assertIn("CATALOG_SIGNATURE_MISSING", codes)
            self.assertIn("SBOM_NOT_CYCLONEDX", codes)
            self.assertIn("MODEL_PACK_SIGNATURE_MISSING", codes)
            self.assertIn("ORT_ATTESTATION_MISSING", codes)
            self.assertIn("GOLDEN_DIGEST_REPORT_MISSING", codes)
            self.assertIn("ABI_REPORT_MISSING", codes)

    def test_development_mode_never_accepts_a_catalog_claiming_production(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            catalog_path = root / "release-manifest.json"
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog["production_approved"] = True
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            report = validate_release_layout(root)
            self.assertFalse(report.ok)
            self.assertIn("DEVELOPMENT_APPROVAL_MISMATCH", {issue.code for issue in report.issues})

    def test_production_mode_accepts_only_complete_signed_evidence_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_production_release(Path(temp))
            report = validate_release_layout(root, targets=(TARGETS[0],), mode="production")
            self.assertTrue(report.ok, [issue.__dict__ for issue in report.issues])
            self.assertEqual(report.mode, "production")
            alias_report = validate_release_layout(root, targets=(TARGETS[0],), production=True)
            self.assertTrue(alias_report.ok, [issue.__dict__ for issue in alias_report.issues])
            output = StringIO()
            with redirect_stdout(output):
                status = cli_main(["verify-release", str(root), "--target", TARGETS[0], "--mode", "production"])
            self.assertEqual(status, 0)
            self.assertIn('"mode": "production"', output.getvalue())

    def test_production_mode_rejects_unverified_ort_and_lock_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_production_release(Path(temp))
            target_root = root / "targets" / TARGETS[0]
            ort = json.loads((target_root / "evidence" / "ort.json").read_text(encoding="utf-8"))
            ort["telemetry"]["status"] = "not_verified"
            ort_path = target_root / "evidence" / "ort.json"
            ort_path.write_text(json.dumps(ort), encoding="utf-8")
            lock_path = target_root / "requirements.lock"
            lock_path.write_text(lock_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            report = validate_release_layout(root, targets=(TARGETS[0],), mode="production")
            codes = {issue.code for issue in report.issues}
            self.assertIn("ORT_ATTESTATION_NOT_INDEPENDENTLY_VERIFIED", codes)
            self.assertIn("TARGET_LOCK_HASH_MISMATCH", codes)

    def test_production_mode_rejects_unmanifested_and_symlink_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_production_release(Path(temp))
            target_root = root / "targets" / TARGETS[0]
            (target_root / "models" / "unlisted.onnx").write_bytes(b"extra")
            (target_root / "models" / "link.onnx").symlink_to("model.onnx")
            report = validate_release_layout(root, targets=(TARGETS[0],), mode="production")
            codes = {issue.code for issue in report.issues}
            self.assertIn("UNMANIFESTED_ARTIFACT", codes)
            self.assertIn("RELEASE_SYMLINK_PROHIBITED", codes)

    def test_missing_or_unpinned_lock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            target_root = root / "targets" / TARGETS[0]
            (target_root / "requirements.lock").unlink()
            report = validate_release_layout(root)
            self.assertTrue(any(issue.code == "LOCKFILE_MISSING" for issue in report.issues))

        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            target_root = root / "targets" / TARGETS[0]
            (target_root / "requirements.lock").write_text("sddiar>=0.3\n")
            report = validate_release_layout(root)
            self.assertTrue(any(issue.code == "LOCKFILE_UNPINNED" for issue in report.issues))

    def test_lock_rejects_network_or_recursive_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            target_root = root / "targets" / TARGETS[0]
            (target_root / "requirements.lock").write_text(
                "--index-url https://example.invalid/simple\n-r other.txt\n"
            )
            report = validate_release_layout(root)
            self.assertTrue(any(issue.code == "LOCKFILE_NETWORK_OR_INCLUDE" for issue in report.issues))

    def test_unmanifested_file_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            target_root = root / "targets" / TARGETS[0]
            (target_root / "models" / "unlisted.onnx").write_bytes(b"extra")
            report = validate_release_layout(root)
            self.assertTrue(any(issue.code == "UNMANIFESTED_ARTIFACT" for issue in report.issues))

        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            target_root = root / "targets" / TARGETS[0]
            (target_root / "models" / "link.onnx").symlink_to("model.onnx")
            report = validate_release_layout(root)
            self.assertTrue(any(issue.code == "RELEASE_SYMLINK_PROHIBITED" for issue in report.issues))

    def test_cli_verifies_release_and_source_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = make_release(Path(temp))
            output = StringIO()
            with redirect_stdout(output):
                status = cli_main(["verify-release", str(root), "--scan-source", str(Path(__file__).parents[1] / "src" / "sddiar")])
            self.assertEqual(status, 0)
            self.assertIn('"ok": true', output.getvalue())


if __name__ == "__main__":
    unittest.main()
