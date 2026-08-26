import unittest

from sddiar.contracts import DiarizationSpan
from sddiar.correction import CorrectionConflictError, HumanSpeakerCorrection, apply_session_corrections


class CorrectionTests(unittest.TestCase):
    def test_human_correction_only_relabels_unknown_interval(self) -> None:
        spans = (DiarizationSpan("u", 0, 100, "UNKNOWN", "UNKNOWN_INSUFFICIENT_EVIDENCE"),)
        correction = HumanSpeakerCorrection("c", 20, 80, "SPEAKER_00", "review-1")
        output = apply_session_corrections(spans, (correction,))
        self.assertEqual(
            [(span.start_us, span.end_us, span.speaker_id) for span in output],
            [(0, 20, "UNKNOWN"), (20, 80, "SPEAKER_00"), (80, 100, "UNKNOWN")],
        )
        self.assertIn("HUMAN_CONFIRMED_SEGMENT", output[1].reason_codes)

    def test_correction_cannot_overwrite_assigned_or_overlap_span(self) -> None:
        correction = HumanSpeakerCorrection("c", 0, 50, "SPEAKER_00", "review-1")
        for speaker in ("SPEAKER_01", "OVERLAP"):
            with self.subTest(speaker=speaker):
                spans = (DiarizationSpan("x", 0, 100, speaker, "ASSIGNED" if speaker != "OVERLAP" else "OVERLAP"),)
                with self.assertRaises(CorrectionConflictError):
                    apply_session_corrections(spans, (correction,))


if __name__ == "__main__":
    unittest.main()
