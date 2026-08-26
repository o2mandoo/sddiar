import io
import struct
import tempfile
import wave
from pathlib import Path
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from sddiar.media import (
    AudioDecodeError, AudioFrontend, UnsupportedCodecError,
    UnsupportedContainerError, WavPcmAccessor, WavPcmDecoder,
)
from sddiar.vad import EnergyVadBackend, SileroOnnxVadBackend, VadUnavailableError


def wav(rate=8000, channels=1, values=(0, 1000, 1000, 0)):
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(channels); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes(b"".join(struct.pack("<h", v) for v in values for _ in range(channels)))
    return out.getvalue()


def extensible_pcm_wav(values=(1000, -1000), rate=16000):
    data = b"".join(struct.pack("<h", value) for value in values)
    pcm_subformat_guid = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    fmt = struct.pack("<HHIIHHH", 0xFFFE, 1, rate, rate * 2, 2, 16, 22)
    fmt += struct.pack("<HI", 16, 0) + pcm_subformat_guid
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


@unittest.skipIf(np is None, "numpy is optional")
class NumpyPcmFastPathTests(unittest.TestCase):
    def test_bounded_chunks_match_stdlib_pcm16_values_and_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fast.wav"
            values = tuple((index * 997) % 65536 - 32768 for index in range(23))
            path.write_bytes(wav(rate=16000, values=values))
            expected = WavPcmDecoder().decode(path)
            chunks = list(WavPcmDecoder().iter_decode_chunks_numpy(path, frames_per_chunk=7))
        actual = np.concatenate([chunk.samples for chunk in chunks])
        reference = np.asarray([frame[0] for frame in expected.samples], dtype=np.float32)
        np.testing.assert_array_equal(actual, reference)
        self.assertEqual([(c.source_start_sample, c.source_end_sample) for c in chunks], [(0, 7), (7, 14), (14, 21), (21, 23)])
        self.assertTrue(all(isinstance(chunk.samples, np.ndarray) and chunk.samples.ndim == 1 for chunk in chunks))

    def test_random_access_matches_stdlib_time_rounding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "access.wav"
            values = (0, 1000, -1000, 32767, -32768) * 100
            path.write_bytes(wav(rate=16000, values=values))
            accessor = WavPcmAccessor(path)
            expected = accessor.read_mono_samples(5_000, 20_000)
            actual = accessor.read_mono_samples_numpy(5_000, 20_000)
        np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float32))

    def test_extensible_pcm16_is_supported_by_fast_path(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "extensible.wav"
            path.write_bytes(extensible_pcm_wav())
            chunks = list(WavPcmDecoder().iter_decode_chunks_numpy(path, frames_per_chunk=1))
        np.testing.assert_array_equal(np.concatenate([c.samples for c in chunks]), np.asarray([1000, -1000], dtype=np.float32) / 32768.0)


class MediaVadTests(unittest.TestCase):
    def test_rejects_non_wav(self):
        with self.assertRaises(UnsupportedContainerError): WavPcmDecoder().decode(b"not audio")

    def test_decodes_wave_format_extensible_pcm(self):
        decoded = WavPcmDecoder().decode(extensible_pcm_wav())
        self.assertEqual((decoded.sample_rate_hz, decoded.channel_count, decoded.total_samples), (16000, 1, 2))
        self.assertAlmostEqual(decoded.samples[0][0], 1000 / 32768)

    def test_decodes_mixdown_and_resamples_with_source_time(self):
        d = WavPcmDecoder().decode(wav(values=(1000, -1000) * 40, channels=2))
        self.assertEqual(d.channel_count, 2); self.assertEqual(d.samples[0], (1000 / 32768, 1000 / 32768))
        n = AudioFrontend().normalize(d)
        self.assertEqual(n.view.sample_rate_hz, 16000)
        self.assertEqual(len(n.samples), 160)
        self.assertEqual(n.view.time_warp[0].source_us(160), 10000)

    def test_chunking_is_bounded_and_repeatable(self):
        n = AudioFrontend().open(wav(rate=16000, values=(1000,) * 1000))
        chunks = list(n.iter_chunks(101))
        self.assertEqual([len(c.samples) for c in chunks], [101] * 9 + [91])
        self.assertEqual(chunks[-1].view_end_sample, len(n.samples))

    def test_energy_vad_produces_integer_source_timestamps(self):
        n = AudioFrontend().open(wav(rate=16000, values=(2000,) * 800))
        frames = EnergyVadBackend(threshold=.01).infer(n)
        self.assertTrue(frames and all(type(f.start_us) is int for f in frames))
        self.assertTrue(any(f.is_speech for f in frames))

    def test_silero_fails_closed_without_runtime_contract(self):
        n = AudioFrontend().open(wav(rate=16000, values=(0,) * 100))
        backend = SileroOnnxVadBackend("/definitely/not/a/model.onnx")
        with self.assertRaises(VadUnavailableError): backend.infer(n)

    def test_streaming_wav_path_is_bounded_and_keeps_source_time(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "long.wav"
            path.write_bytes(wav(rate=8000, values=(1000,) * 40_000))
            chunks = list(AudioFrontend().iter_open_chunks(path, source_frames_per_chunk=10_000))
        self.assertEqual(len(chunks), 4)
        self.assertEqual(sum(len(chunk.samples) for chunk in chunks), 80_000)
        self.assertEqual((chunks[0].time_warp.source_start_us, chunks[-1].time_warp.source_end_us), (0, 5_000_000))

    def test_random_access_pcm_region_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.wav"
            path.write_bytes(wav(rate=16000, values=(0, 1000, -1000, 0) * 4000))
            samples = WavPcmAccessor(path).read_mono_samples(250_000, 500_000)
        self.assertEqual(len(samples), 4000)


if __name__ == "__main__": unittest.main()
