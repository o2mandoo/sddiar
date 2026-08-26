from __future__ import annotations

import hashlib
import json
import os
import threading
import sys
import subprocess
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from sddiar.contracts import EmbeddingResult
from sddiar.errors import ModelHashMismatch
from sddiar.model_pack import VerifiedArtifact
from sddiar.onnx_diarization import LocalOnnxDiarizationConfig, LocalOnnxDiarizer
from sddiar.vad import VadFrame


class FakeSilero:
    def __init__(self):
        self.chunk_count = 0

    def infer_chunk_stream(self, chunks):
        self.chunk_count = sum(1 for _ in chunks)
        return tuple(
            VadFrame(start, end, 0.99, True)
            for start, end in ((0, 2_000_000), (3_000_000, 5_000_000), (6_000_000, 8_000_000), (9_000_000, 11_000_000))
        )


class CapturingSilero(FakeSilero):
    def __init__(self):
        super().__init__()
        self.first_sample = None

    def infer_chunk_stream(self, chunks):
        materialized = tuple(chunks)
        self.chunk_count = len(materialized)
        first = materialized[0].samples[0]
        self.first_sample = float(first[0] if isinstance(first, tuple) else first)
        return tuple(
            VadFrame(start, end, 0.99, True)
            for start, end in ((0, 2_000_000), (3_000_000, 5_000_000), (6_000_000, 8_000_000), (9_000_000, 11_000_000))
        )


class FakeEmbedding:
    def embed(self, regions):
        return tuple(
            EmbeddingResult(
                region.embedding_region_id,
                region.tracklet_id,
                True,
                (1.0, 0.0) if index % 2 == 0 else (0.0, 1.0),
                dimension=2,
                valid_window_count=1,
                clean_window_coverage=1.0,
                intra_window_consistency=1.0,
                quality=1.0,
            )
            for index, region in enumerate(regions)
        )


class PersistentEmbedding:
    instances = []
    provider_samples = []

    def __init__(self, artifact, *, session, max_batch_regions):
        self.artifact = artifact
        self.session = session
        self.max_batch_regions = max_batch_regions
        type(self).instances.append(self)

    def embed(self, regions, audio_provider=None):
        self.provider_samples.append(
            tuple(float(audio_provider(region)[0]) for region in regions)
        )
        return FakeEmbedding().embed(regions)


class LocalOnnxDiarizerTests(unittest.TestCase):
    def test_module_import_and_rss_probe_survive_missing_resource_on_windows(self):
        # ``resource`` is absent on Windows.  Exercise a fresh interpreter so
        # this catches import-time regressions rather than only the optional
        # helper branch in the already-loaded test process.
        code = """
import builtins
import sys

real_import = builtins.__import__
def reject_resource(name, *args, **kwargs):
    if name == 'resource':
        raise ImportError('simulated Windows: resource is unavailable')
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_resource
sys.platform = 'win32'
import sddiar.onnx_diarization as module
assert module._resource is None
value = module.LocalOnnxDiarizer._rss_mb()
assert value is None or (isinstance(value, float) and value >= 0.0)
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.audio = root / "fixture.wav"
        with wave.open(str(self.audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x10" * (16_000 * 12))
        self.silero = root / "silero.onnx"
        self.silero.write_bytes(b"local silero development artifact")
        self.wespeaker = root / "wespeaker.onnx"
        self.wespeaker.write_bytes(b"local wespeaker development artifact")
        self.audio_two = root / "fixture-two.wav"
        with wave.open(str(self.audio_two), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x20" * (16_000 * 12))

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def diarizer(self, **kwargs):
        return LocalOnnxDiarizer(
            self.silero,
            self.wespeaker,
            silero_sha256=self.digest(self.silero),
            wespeaker_sha256=self.digest(self.wespeaker),
            silero_runtime=FakeSilero(),
            embedding_backend=FakeEmbedding(),
            **kwargs,
        )

    def test_local_fake_pipeline_returns_redacted_json_result(self):
        diarizer = self.diarizer()
        result = diarizer.process(self.audio)
        payload = json.loads(result.to_json())
        self.assertEqual(payload["schema"], "sddiar.local_onnx_result_v1")
        self.assertEqual(payload["sample_rate_hz"], 16_000)
        self.assertEqual(payload["duration_us"], 12_000_000)
        self.assertGreater(payload["metrics"]["tracklet_count"], 0)
        self.assertEqual(payload["metrics"]["valid_embedding_count"], payload["metrics"]["tracklet_count"])
        self.assertEqual(payload["quality_status"], "REVIEW_REQUIRED")
        self.assertIn("Q_CALIBRATION_MISSING", payload["quality_reason_codes"])
        self.assertEqual(payload["runtime_config"]["feature_mode"], "injected")
        self.assertEqual(payload["redaction"]["source_path"], "omitted")
        self.assertNotIn(str(self.audio), result.to_json())
        self.assertNotIn("transcript", payload)
        self.assertNotIn("audio_samples", payload)
        self.assertNotIn("embedding_vectors", payload)
        self.assertTrue(all("start_us" in span and "speaker_id" in span for span in payload["spans"]))

    def test_fake_runtime_consumes_bounded_decoder_stream(self):
        runtime = FakeSilero()
        result = LocalOnnxDiarizer(
            self.silero,
            self.wespeaker,
            silero_sha256=self.digest(self.silero),
            wespeaker_sha256=self.digest(self.wespeaker),
            silero_runtime=runtime,
            embedding_backend=FakeEmbedding(),
        ).diarize(self.audio)
        self.assertEqual(runtime.chunk_count, 1)
        self.assertEqual(result.duration_us, 12_000_000)

    def test_auto_gain_is_opt_in_and_default_processing_has_exact_span_parity(self):
        implicit = self.diarizer().process(self.audio)
        explicit = self.diarizer(
            config=LocalOnnxDiarizationConfig(auto_gain_normalization=False)
        ).process(self.audio)
        self.assertEqual(implicit.spans, explicit.spans)
        self.assertEqual(implicit.decision, explicit.decision)
        self.assertEqual(implicit.decision_reasons, explicit.decision_reasons)
        self.assertFalse(implicit.runtime_config["auto_gain_normalization"])
        self.assertFalse(implicit.audio_gain_normalization["enabled"])
        self.assertEqual(
            implicit.runtime_config["audio_gain_policy_sha256"],
            explicit.runtime_config["audio_gain_policy_sha256"],
        )

    def test_auto_gain_applies_one_identical_gain_to_vad_and_embedding_audio(self):
        quiet = Path(self.tmp.name) / "quiet.wav"
        with wave.open(str(quiet), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes((100).to_bytes(2, "little", signed=True) * (16_000 * 12))
        runtime = CapturingSilero()
        PersistentEmbedding.instances = []
        PersistentEmbedding.provider_samples = []
        with mock.patch("sddiar.onnx_diarization.WeSpeakerCpuEmbeddingBackend", PersistentEmbedding), \
             mock.patch("sddiar.onnx_diarization._ort_session", return_value=object()):
            result = LocalOnnxDiarizer(
                self.silero,
                self.wespeaker,
                silero_sha256=self.digest(self.silero),
                wespeaker_sha256=self.digest(self.wespeaker),
                silero_runtime=runtime,
                config=LocalOnnxDiarizationConfig(auto_gain_normalization=True),
            ).process(quiet)

        profile = result.audio_gain_normalization
        self.assertTrue(profile["enabled"])
        self.assertEqual(profile["applied_gain"], 4.0)
        expected = 100.0 / 32768.0 * 4.0
        self.assertAlmostEqual(runtime.first_sample, expected, places=8)
        self.assertTrue(PersistentEmbedding.provider_samples)
        self.assertAlmostEqual(PersistentEmbedding.provider_samples[0][0], expected, places=8)
        self.assertEqual(profile["source_sha256"][:12], result.audio_sha256_prefix)
        self.assertEqual(
            profile["policy_sha256"], result.runtime_config["audio_gain_policy_sha256"]
        )

    def test_path_model_requires_exact_hash(self):
        with self.assertRaises(ModelHashMismatch):
            LocalOnnxDiarizer(self.silero, self.wespeaker, wespeaker_sha256=self.digest(self.wespeaker), silero_runtime=FakeSilero(), embedding_backend=FakeEmbedding())

    def test_modified_model_fails_closed(self):
        original = self.digest(self.silero)
        self.silero.write_bytes(b"modified")
        with self.assertRaises(ModelHashMismatch):
            LocalOnnxDiarizer(self.silero, self.wespeaker, silero_sha256=original, wespeaker_sha256=self.digest(self.wespeaker), silero_runtime=FakeSilero(), embedding_backend=FakeEmbedding())

    def test_accepts_preverified_model_pack_artifacts(self):
        silero = VerifiedArtifact("silero", "vad", self.silero, self.digest(self.silero), self.silero.stat().st_size)
        wespeaker = VerifiedArtifact("wespeaker", "speaker_embedding", self.wespeaker, self.digest(self.wespeaker), self.wespeaker.stat().st_size)
        result = LocalOnnxDiarizer(
            silero,
            wespeaker,
            silero_runtime=FakeSilero(),
            embedding_backend=FakeEmbedding(),
        ).process(self.audio)
        self.assertEqual(result.quality_status, "REVIEW_REQUIRED")

    def test_default_embedding_backend_is_opened_once_and_gets_each_job_audio(self):
        PersistentEmbedding.instances = []
        PersistentEmbedding.provider_samples = []
        with mock.patch("sddiar.onnx_diarization.WeSpeakerCpuEmbeddingBackend", PersistentEmbedding), \
             mock.patch("sddiar.onnx_diarization._ort_session", return_value=object()):
            diarizer = LocalOnnxDiarizer(
                self.silero,
                self.wespeaker,
                silero_sha256=self.digest(self.silero),
                wespeaker_sha256=self.digest(self.wespeaker),
                silero_runtime=FakeSilero(),
            )
            first = diarizer.process(self.audio)
            second = diarizer.process(self.audio_two)
            repeat = diarizer.process(self.audio)

        self.assertEqual(len(PersistentEmbedding.instances), 1)
        self.assertEqual(len(PersistentEmbedding.provider_samples), 3)
        self.assertNotEqual(
            PersistentEmbedding.provider_samples[0][0],
            PersistentEmbedding.provider_samples[1][0],
        )
        self.assertEqual(first.spans, repeat.spans)
        self.assertEqual(first.decision, repeat.decision)
        self.assertNotEqual(first.audio_sha256_prefix, second.audio_sha256_prefix)

    def test_process_lock_serializes_jobs_and_releases_after_exception(self):
        diarizer = self.diarizer()
        active = 0
        maximum = 0
        guard = threading.Lock()

        def fake_process(path):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return path

        with mock.patch.object(diarizer, "_process_unlocked", side_effect=fake_process):
            threads = [threading.Thread(target=diarizer.process, args=(self.audio,)) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(maximum, 1)

        with mock.patch.object(diarizer, "_process_unlocked", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                diarizer.process(self.audio)
        with mock.patch.object(diarizer, "_process_unlocked", side_effect=fake_process):
            self.assertEqual(diarizer.process(self.audio), self.audio)


if __name__ == "__main__":
    unittest.main()
