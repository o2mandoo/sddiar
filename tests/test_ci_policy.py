from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


media = _load("verify_git_media_policy_test", ROOT / "scripts" / "verify_git_media_policy.py")
wheel = _load("ci_verify_wheel_test", ROOT / "scripts" / "ci_verify_wheel.py")


class GitMediaPolicyTests(unittest.TestCase):
    def test_forbidden_paths_cover_audio_models_and_private_trees(self):
        for path in (
            "sample.WAV", "models/model.onnx", "recordings/session.data",
            "nested/audio/input.unknown", "root-transcript.txt", "runtime.dll",
        ):
            self.assertTrue(media.forbidden_path(path), path)
        for path in ("src/sddiar/audio_gain.py", "docs/transcript-policy.md", "manifest.json"):
            self.assertFalse(media.forbidden_path(path), path)

    def test_media_magic_detects_common_containers(self):
        fixtures = (
            b"RIFF\0\0\0\0WAVE", b"FORM\0\0\0\0AIFF", b"fLaCdata",
            b"OggSdata", b"ID3data", b"\0\0\0\x18ftypM4A ", b"\x1aE\xdf\xa3",
        )
        self.assertTrue(all(media.media_magic(payload) for payload in fixtures))
        self.assertFalse(media.media_magic(b"#!/usr/bin/env python3\n"))

    def test_current_repository_history_has_no_media(self):
        report = media.audit()
        self.assertEqual(report["forbidden_current_paths"], [])
        self.assertEqual(report["forbidden_history_paths"], [])
        self.assertEqual(report["media_magic_history_paths"], [])


class CiWheelVerifierTests(unittest.TestCase):
    def test_architecture_aliases_are_explicit(self):
        with mock.patch.object(wheel.platform, "machine", return_value="AMD64"):
            self.assertTrue(wheel._architecture_matches("x86_64"))
            self.assertFalse(wheel._architecture_matches("arm64"))
        with mock.patch.object(wheel.platform, "machine", return_value="aarch64"):
            self.assertTrue(wheel._architecture_matches("arm64"))


if __name__ == "__main__":
    unittest.main()
