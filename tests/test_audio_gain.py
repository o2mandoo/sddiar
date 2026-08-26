from __future__ import annotations

import hashlib
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from sddiar.audio_gain import (
    DEFAULT_GLOBAL_GAIN_POLICY,
    GainScaledWavPcmAccessor,
    GlobalGainPolicy,
    analyze_pcm16_global_gain,
    scale_decoded_chunks,
)
from sddiar.media import WavPcmAccessor, WavPcmDecoder


def _pcm16(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}h", *values)


def _write_pcm16(path: Path, values: list[int]) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(_pcm16(values))


def _write_extensible_pcm16(path: Path, values: list[int]) -> None:
    payload = _pcm16(values)
    pcm_subformat = bytes.fromhex("0100000000001000800000aa00389b71")
    fmt = struct.pack(
        "<HHIIHHHHI16s",
        0xFFFE,
        1,
        16_000,
        32_000,
        2,
        16,
        22,
        16,
        0,
        pcm_subformat,
    )
    riff_size = 4 + (8 + len(fmt)) + (8 + len(payload))
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )


class AudioGainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_default_policy_is_frozen_and_label_independent(self) -> None:
        self.assertEqual(
            DEFAULT_GLOBAL_GAIN_POLICY.to_dict(),
            {
                "schema": "sddiar.global_gain_policy_v2",
                "rms_trigger": 0.01,
                "max_gain": 4.0,
                "peak_ceiling": 0.99,
                "min_activation_gain": 1.25,
                "analysis_chunk_frames": 240_000,
                "boost_only_below_rms_trigger": True,
                "never_attenuate": True,
                "label_independent": True,
            },
        )
        self.assertEqual(
            DEFAULT_GLOBAL_GAIN_POLICY.sha256,
            "0f9cac01b09df993dec7af523a69f803bd64bfbff44f75650da8725641c271f2",
        )

    def test_stdlib_numpy_and_wave_extensible_have_exact_pcm_statistics(self) -> None:
        values = [-32768, -1000, -1, 0, 1, 999, 32767] * 31
        classic = self.root / "classic.wav"
        extensible = self.root / "extensible.wav"
        _write_pcm16(classic, values)
        _write_extensible_pcm16(extensible, values)

        stdlib_profile = analyze_pcm16_global_gain(classic, prefer_numpy=False)
        optional_profile = analyze_pcm16_global_gain(classic, prefer_numpy=True)
        extensible_profile = analyze_pcm16_global_gain(extensible, prefer_numpy=False)
        extensible_optional_profile = analyze_pcm16_global_gain(extensible, prefer_numpy=True)

        self.assertEqual(stdlib_profile, optional_profile)
        self.assertEqual(extensible_profile, extensible_optional_profile)
        self.assertEqual(stdlib_profile.sample_count, len(values))
        self.assertEqual(stdlib_profile.sum_squares_pcm16, sum(value * value for value in values))
        self.assertEqual(stdlib_profile.peak_abs_pcm16, 32768)
        self.assertEqual(
            (stdlib_profile.sample_count, stdlib_profile.sum_squares_pcm16, stdlib_profile.peak_abs_pcm16),
            (extensible_profile.sample_count, extensible_profile.sum_squares_pcm16, extensible_profile.peak_abs_pcm16),
        )
        self.assertEqual(stdlib_profile.policy_sha256, extensible_profile.policy_sha256)

    def test_low_rms_boost_is_deterministic_and_never_clips(self) -> None:
        quiet = self.root / "quiet.wav"
        _write_pcm16(quiet, [100, -100] * 16_000)
        first = analyze_pcm16_global_gain(quiet)
        second = analyze_pcm16_global_gain(quiet)
        self.assertEqual(first, second)
        self.assertEqual(first.applied_gain, 4.0)
        self.assertLessEqual(first.predicted_peak, 0.99)
        self.assertTrue(first.to_dict()["no_new_clipping"])
        self.assertEqual(len(first.profile_sha256), 64)
        self.assertNotIn(str(quiet), str(first.to_dict()))

        impulse = self.root / "impulse.wav"
        _write_pcm16(impulse, [32767] + [0] * 199_999)
        constrained = analyze_pcm16_global_gain(impulse)
        self.assertLess(constrained.rms, 0.01)
        self.assertEqual(constrained.applied_gain, 1.0)
        self.assertEqual(constrained.reason, "LOW_RMS_PEAK_PREVENTS_BOOST")
        self.assertLessEqual(constrained.predicted_peak, 1.0)

        near_noop = self.root / "near-noop.wav"
        _write_pcm16(near_noop, [32_303] + [0] * 199_999)
        deadband = analyze_pcm16_global_gain(near_noop)
        self.assertLess(deadband.rms, 0.01)
        self.assertGreater(0.99 / deadband.peak, 1.0)
        self.assertLess(0.99 / deadband.peak, 1.25)
        self.assertEqual(deadband.applied_gain, 1.0)
        self.assertEqual(deadband.reason, "LOW_RMS_GAIN_BELOW_ACTIVATION_FLOOR")

        peak_limited = self.root / "peak-limited.wav"
        _write_pcm16(peak_limited, [19_660] + [0] * 199_999)
        profile = analyze_pcm16_global_gain(peak_limited)
        self.assertGreater(profile.applied_gain, 1.0)
        self.assertLess(profile.applied_gain, 4.0)
        self.assertAlmostEqual(profile.predicted_peak, 0.99, places=12)

    def test_chunk_and_region_wrappers_apply_the_identical_gain(self) -> None:
        audio = self.root / "source.wav"
        values = [index - 200 for index in range(401)]
        _write_pcm16(audio, values)
        profile = analyze_pcm16_global_gain(audio)
        self.assertEqual(profile.applied_gain, 4.0)

        decoder = WavPcmDecoder()
        scaled_chunks = tuple(
            scale_decoded_chunks(
                decoder.iter_decode_chunks(audio, frames_per_chunk=37),
                profile.applied_gain,
            )
        )
        chunk_values = tuple(value for chunk in scaled_chunks for frame in chunk.samples for value in frame)
        reader = GainScaledWavPcmAccessor(WavPcmAccessor(audio), profile.applied_gain)
        end_us = (len(values) * 1_000_000 + 15_999) // 16_000
        region_values = reader.read_mono_samples(0, end_us)
        self.assertEqual(chunk_values, region_values)

        try:
            import numpy as np
        except ImportError:
            return
        numpy_chunks = tuple(
            scale_decoded_chunks(
                decoder.iter_decode_chunks_numpy(audio, frames_per_chunk=37),
                profile.applied_gain,
            )
        )
        numpy_values = np.concatenate([chunk.samples for chunk in numpy_chunks])
        self.assertTrue(np.array_equal(numpy_values, reader.read_mono_samples_numpy(0, end_us)))
        self.assertLessEqual(float(np.max(np.abs(numpy_values))), profile.predicted_peak + 1e-7)

    def test_policy_and_profile_hashes_are_chunk_size_independent_where_expected(self) -> None:
        audio = self.root / "hash.wav"
        _write_pcm16(audio, [50, -50, 100, -100] * 1000)
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        small = analyze_pcm16_global_gain(
            audio,
            source_sha256=digest,
            policy=GlobalGainPolicy(analysis_chunk_frames=7),
            prefer_numpy=False,
        )
        repeat = analyze_pcm16_global_gain(
            audio,
            source_sha256=digest,
            policy=GlobalGainPolicy(analysis_chunk_frames=7),
            prefer_numpy=True,
        )
        different_chunk_policy = analyze_pcm16_global_gain(
            audio,
            source_sha256=digest,
            policy=GlobalGainPolicy(analysis_chunk_frames=31),
            prefer_numpy=False,
        )
        self.assertEqual(small, repeat)
        self.assertEqual(small.sum_squares_pcm16, different_chunk_policy.sum_squares_pcm16)
        self.assertEqual(small.applied_gain, different_chunk_policy.applied_gain)
        self.assertNotEqual(small.policy_sha256, different_chunk_policy.policy_sha256)
        self.assertNotEqual(small.profile_sha256, different_chunk_policy.profile_sha256)

    def test_numpy_is_optional(self) -> None:
        audio = self.root / "stdlib.wav"
        _write_pcm16(audio, [10, -10] * 100)
        with mock.patch("sddiar.audio_gain._load_numpy_optional", return_value=None):
            profile = analyze_pcm16_global_gain(audio, prefer_numpy=True)
        self.assertEqual(profile.sum_squares_pcm16, 20_000)

    def test_gain_policy_is_part_of_the_public_library_api(self) -> None:
        import sddiar

        self.assertIs(sddiar.GlobalGainPolicy, GlobalGainPolicy)
        self.assertIs(sddiar.DEFAULT_GLOBAL_GAIN_POLICY, DEFAULT_GLOBAL_GAIN_POLICY)
        self.assertIs(sddiar.analyze_pcm16_global_gain, analyze_pcm16_global_gain)


if __name__ == "__main__":
    unittest.main()
