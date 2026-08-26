from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from sddiar.contracts import AudioRequest, AudioSourceMetadata, TimeWarpSegment, Timebase
from sddiar.errors import ContractValidationError, ModelHashMismatch
from sddiar.production_orchestrator import (
    CanonicalAudio,
    HashVerifiedLocalTranscriptBackend,
    verify_local_stt_identity,
)
from sddiar.whispercpp_backend import (
    WHISPER_CPP_BACKEND_ID,
    WHISPER_CPP_BACKEND_VERSION,
    WhisperCppBackend,
    WhisperCppConfig,
    WhisperCppOutputError,
    WhisperCppRunResult,
    WhisperCppTimeoutError,
    WhisperCppTimestampError,
    WhisperCppJsonError,
    WhisperCppConfigurationError,
    SubprocessArgvRunner,
)
import sddiar.whispercpp_backend as whispercpp_module


class _Runner:
    def __init__(self, payload, *, mutate_model: bool = False, delay: float = 0.0):
        self.payload = payload
        self.mutate_model = mutate_model
        self.delay = delay
        self.invocations = []

    def run(self, invocation):
        self.invocations.append(invocation)
        if self.delay:
            time.sleep(self.delay)
        if self.mutate_model:
            model_index = invocation.argv.index("-m") + 1
            Path(invocation.argv[model_index]).write_bytes(b"tampered")
        invocation.output_path.write_bytes(json.dumps(self.payload).encode("utf-8"))
        return WhisperCppRunResult(0)


def _payload(tokens):
    return {"transcription": [{"tokens": [
        {"text": text, "offsets": {"from": start, "to": end}}
        for text, start, end in tokens
    ]}]}


class WhisperCppBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.engine = root / "whisper-cli"
        self.model = root / "ko-model.bin"
        self.audio = root / "canonical.wav"
        self.engine.write_bytes(b"pinned executable")
        self.model.write_bytes(b"caller selected model")
        self.audio.write_bytes(b"canonical pcm")
        self.identity = verify_local_stt_identity(
            backend_id=WHISPER_CPP_BACKEND_ID,
            backend_version=WHISPER_CPP_BACKEND_VERSION,
            engine_path=self.engine,
            engine_sha256=hashlib.sha256(self.engine.read_bytes()).hexdigest(),
            model_path=self.model,
            model_sha256=hashlib.sha256(self.model.read_bytes()).hexdigest(),
        )
        self.source = AudioSourceMetadata(
            audio_sha256="a" * 64,
            container="wav",
            codec="pcm_s16le",
            native_sample_rate_hz=16_000,
            channel_count=1,
            duration_us=3_000_000,
            timebase=Timebase("source"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_korean_eojeol_grouping_and_punctuation(self):
        runner = _Runner(_payload([
            (" 안", 0, 200), ("녕", 200, 400), (" 하", 500, 700),
            ("세", 700, 900), ("요", 900, 1000), (".", 1000, 1050),
        ]))
        timeline = WhisperCppBackend(self.identity, runner=runner).transcribe(self.audio, self.source)
        self.assertEqual([word.text for word in timeline.words], ["안녕", "하세요."])
        self.assertEqual([(word.start_us, word.end_us) for word in timeline.words], [(0, 400_000), (500_000, 1_050_000)])
        self.assertTrue(all(timeline.provenance_by_word_id[word.word_id].source_chunk_ids for word in timeline.words))

    def test_real_whisper_markers_and_zero_duration_timewarp_word(self):
        runner = _Runner(_payload([
            ("[_BEG_]", 0, 0), ("[_TT_0]", 0, 0),
            (" 안", 0, 0), ("녕", 0, 0), (" [_TT_1]", 0, 0),
        ]))
        timeline = WhisperCppBackend(self.identity, runner=runner).transcribe(self.audio, self.source)
        self.assertEqual([word.text for word in timeline.words], ["안녕"])
        word = timeline.words[0]
        self.assertEqual((word.start_us, word.end_us), (0, 1))
        self.assertTrue(timeline.provenance_by_word_id[word.word_id].crosses_timewarp_boundary)

        bad = _Runner(_payload([("[_END_]", 0, 0)]))
        with self.assertRaises(WhisperCppJsonError):
            WhisperCppBackend(self.identity, runner=bad).transcribe(self.audio, self.source)

    def test_allowlisted_special_tokens_are_filtered(self):
        runner = _Runner(_payload([
            ("<|startoftranscript|>", 0, 0), (" 안녕", 0, 500), ("<|endoftext|>", 500, 500),
        ]))
        timeline = WhisperCppBackend(self.identity, runner=runner).transcribe(self.audio, self.source)
        self.assertEqual([word.text for word in timeline.words], ["안녕"])

    def test_invalid_special_token_and_timestamp_fail(self):
        with self.assertRaises(WhisperCppJsonError):
            WhisperCppBackend(self.identity, runner=_Runner(_payload([("<|secret|>", 0, 1)]))).transcribe(self.audio, self.source)
        with self.assertRaises(WhisperCppTimestampError):
            WhisperCppBackend(self.identity, runner=_Runner(_payload([(" 안", 500, 600), (" 녕", 400, 700)]))).transcribe(self.audio, self.source)

    def test_strict_utf8_and_output_bound(self):
        runner = _Runner(None)
        runner.run = lambda invocation: (invocation.output_path.write_bytes(b"{\xff"), WhisperCppRunResult(0))[1]
        with self.assertRaises(WhisperCppJsonError):
            WhisperCppBackend(self.identity, runner=runner).transcribe(self.audio, self.source)

        oversized = _Runner(_payload([(" hi", 0, 1)]))
        oversized.run = lambda invocation: (invocation.output_path.write_bytes(b"x" * 100), WhisperCppRunResult(0))[1]
        with self.assertRaises(WhisperCppOutputError):
            WhisperCppBackend(self.identity, config=WhisperCppConfig(max_output_bytes=10), runner=oversized).transcribe(self.audio, self.source)

    def test_timeout_and_job_cleanup(self):
        class TimeoutRunner:
            def run(self, invocation):
                raise WhisperCppTimeoutError("timeout")
        with self.assertRaises(WhisperCppTimeoutError):
            WhisperCppBackend(self.identity, runner=TimeoutRunner()).transcribe(self.audio, self.source)

        runner = _Runner(_payload([(" hi", 0, 1)]))
        WhisperCppBackend(self.identity, runner=runner).transcribe(self.audio, self.source)
        self.assertFalse(runner.invocations[0].cwd.exists())

    def test_tamper_is_rejected_and_text_is_not_in_errors(self):
        runner = _Runner(_payload([(" PRIVATE TRANSCRIPT", 0, 1)]), mutate_model=True)
        with self.assertRaises(WhisperCppConfigurationError) as context:
            WhisperCppBackend(self.identity, runner=runner).transcribe(self.audio, self.source)
        self.assertNotIn("PRIVATE TRANSCRIPT", str(context.exception))
        self.assertFalse(runner.invocations[0].cwd.exists())

    def test_policy_and_default_are_explicit(self):
        argv = _Runner(_payload([(" hi", 0, 1)]))
        backend = WhisperCppBackend(self.identity, config=WhisperCppConfig(dtw_preset="small"), runner=argv)
        self.assertFalse(backend.default_selected)
        self.assertFalse(backend.experimental is False)
        backend.transcribe(self.audio, self.source)
        command = argv.invocations[0].argv
        for flag in ("-t", "-p", "-ng", "-l", "-ojf", "-bs", "-bo", "-tp", "-nf"):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("-l") + 1], "ko")
        self.assertEqual(command[command.index("-t") + 1], "1")
        self.assertEqual(command[command.index("-p") + 1], "1")
        self.assertEqual(command[command.index("-dtw") + 1], "small")
        self.assertEqual(backend.config.public_identity()["dtw_preset"], "small")
        ojf = command.index("-ojf")
        self.assertEqual(command[ojf + 1], "-of")
        self.assertEqual(Path(command[ojf + 2]).with_suffix(".json"), argv.invocations[0].output_path)

    def test_windows_runner_environment_is_minimal_and_has_temp_contract(self):
        cwd = Path(self.tmp.name) / "job"
        cwd.mkdir()
        # Keep the check isolated from the host platform while exercising the
        # exact environment branch used by Windows subprocesses.
        with mock.patch.object(whispercpp_module.os, "name", "nt"), mock.patch.dict(
            whispercpp_module.os.environ,
            {"SYSTEMROOT": "C:\\Windows", "WINDIR": "C:\\Windows"},
            clear=True,
        ):
            environment = SubprocessArgvRunner._environment(cwd)
        self.assertEqual(environment["TEMP"], str(cwd))
        self.assertEqual(environment["TMP"], str(cwd))
        self.assertEqual(environment["SYSTEMROOT"], "C:\\Windows")
        self.assertNotIn("PATH", environment)
        self.assertNotIn("HOME", environment)

    def test_hash_verified_wrapper_uses_bound_identity_and_two_argument_protocol(self):
        runner = _Runner(_payload([(" 안녕", 0, 500)]))
        engine = WhisperCppBackend(self.identity, runner=runner)
        wrapper = HashVerifiedLocalTranscriptBackend(self.identity, engine)
        canonical = CanonicalAudio(
            self.source,
            self.audio,
            16_000,
            48_000,
            self.source.audio_sha256,
            (),
            False,
        )
        payload = wrapper.transcribe(AudioRequest("request", "local", "profile"), self.source, canonical)
        self.assertEqual([word.text for word in payload.timeline.words], ["안녕"])
        self.assertIs(engine.identity, wrapper.identity)
        self.assertEqual(wrapper.decoder_policy["dtw_preset"], "base")

        other_model = Path(self.tmp.name) / "other-model.bin"
        other_model.write_bytes(b"different model")
        other = verify_local_stt_identity(
            backend_id=WHISPER_CPP_BACKEND_ID,
            backend_version=WHISPER_CPP_BACKEND_VERSION,
            engine_path=self.engine,
            engine_sha256=hashlib.sha256(self.engine.read_bytes()).hexdigest(),
            model_path=other_model,
            model_sha256=hashlib.sha256(other_model.read_bytes()).hexdigest(),
        )
        with self.assertRaises(ContractValidationError):
            HashVerifiedLocalTranscriptBackend(self.identity, WhisperCppBackend(other, runner=runner))


if __name__ == "__main__":
    unittest.main()
