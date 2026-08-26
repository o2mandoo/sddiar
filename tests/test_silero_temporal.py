import unittest

from sddiar.silero_temporal import (
    SileroTemporalPostprocessor,
    SpeechRegion,
    TemporalVadConfig,
    TemporalVadError,
    mark_vad_frames,
    postprocess_vad,
)
from sddiar.vad import VadFrame


def frames(probabilities, duration_us=50_000):
    return tuple(
        VadFrame(i * duration_us, (i + 1) * duration_us, probability, probability >= 0.5)
        for i, probability in enumerate(probabilities)
    )


class SileroTemporalTests(unittest.TestCase):
    def test_upstream_style_hysteresis_and_explicit_halo(self):
        # 0--50 is below onset, 50--150 is core (the 0.40 frame is held by
        # hysteresis), and silence begins at 150.  Silence reaches 100 ms at
        # the 250 ms frame boundary, so the core closes at 150 ms.  The 30 ms
        # pad is halo, not clean speech.
        source = frames((0.1, 0.6, 0.4, 0.2, 0.2, 0.2))
        region = postprocess_vad(source)[0]
        self.assertEqual((region.start_us, region.end_us), (20_000, 180_000))
        self.assertEqual((region.core_start_us, region.core_end_us), (50_000, 150_000))
        self.assertEqual(region.core_duration_us, 100_000)
        self.assertEqual(region.halo_duration_us, 60_000)

        marked = mark_vad_frames(source, (region,))
        self.assertEqual([(item.kind, item.start_us, item.end_us) for item in marked], [
            ("halo", 20_000, 50_000),
            ("core", 50_000, 100_000),
            ("core", 100_000, 150_000),
            ("halo", 150_000, 180_000),
        ])
        self.assertFalse(marked[0].is_speech)
        self.assertTrue(marked[1].is_speech)

    def test_final_speech_uses_last_core_evidence_and_bounds_pad(self):
        source = frames((0.1, 0.6, 0.4, 0.2))
        region = postprocess_vad(source)[0]
        # The final low frame is not silently converted to core at EOF; only
        # the configured 30 ms halo extends into it.
        self.assertEqual(
            (region.start_us, region.end_us, region.core_start_us, region.core_end_us),
            (20_000, 180_000, 50_000, 150_000),
        )

    def test_adjacent_padding_shares_small_gap(self):
        source = frames((0.6, 0.6, 0.1, 0.1, 0.1, 0.6, 0.6), duration_us=20_000)
        # The observed core gap is 60 ms, enough to split under this
        # deliberately small test policy, and less than two 40 ms pads.
        config = TemporalVadConfig(min_silence_duration_us=40_000, speech_pad_us=40_000)
        regions = SileroTemporalPostprocessor(config).regions(source)
        self.assertEqual(len(regions), 2)
        self.assertEqual((regions[0].core_start_us, regions[0].core_end_us), (0, 40_000))
        self.assertEqual((regions[1].core_start_us, regions[1].core_end_us), (100_000, 140_000))
        self.assertEqual(regions[0].end_us, regions[1].start_us)
        self.assertEqual(regions[0].halo_after_us, 30_000)
        self.assertEqual(regions[1].halo_before_us, 30_000)

    def test_max_duration_is_off_by_default_and_exact_when_enabled(self):
        source = frames((0.6,) * 5)
        self.assertEqual(len(postprocess_vad(source)), 1)
        config = TemporalVadConfig(max_speech_duration_us=120_000)
        regions = SileroTemporalPostprocessor(config).regions(source)
        self.assertEqual([(r.core_start_us, r.core_end_us) for r in regions], [
            (0, 120_000),
            (120_000, 240_000),
            (240_000, 250_000),
        ])

    def test_empty_input_and_malformed_timeline_fail_closed(self):
        self.assertEqual(postprocess_vad(()), ())
        with self.assertRaises(TemporalVadError):
            postprocess_vad((VadFrame(0, 10, None, False),))
        with self.assertRaises(TemporalVadError):
            postprocess_vad((VadFrame(0, 10, 0.6, True), VadFrame(11, 20, 0.6, True)))

    def test_result_aggregates_core_and_halo_separately(self):
        source = frames((0.6, 0.6, 0.1, 0.1, 0.1, 0.1))
        result = SileroTemporalPostprocessor().process(source, mark=True)
        self.assertEqual(result.core_duration_us, 100_000)
        self.assertEqual(result.halo_duration_us, 30_000)
        self.assertEqual(result.speech_duration_us, 130_000)
        self.assertEqual(sum(item.is_core for item in result.marked_frames), 2)


if __name__ == "__main__":
    unittest.main()
