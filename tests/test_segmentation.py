import unittest

from sddiar.contracts import ContractValidationError
from sddiar.segmentation import (
    OverlapEvent,
    ProbeEvidence,
    RuleEvidenceSegmentation,
    SegmentationConfig,
    SpeakerChangeEvent,
)
from sddiar.vad import VadFrame


class SegmentationTests(unittest.TestCase):
    def test_vad_frames_merge_into_source_time_regions(self) -> None:
        segmentation = RuleEvidenceSegmentation(SegmentationConfig(vad_merge_gap_us=50))
        evidence = segmentation.build(
            view_id="v",
            vad_frames=(VadFrame(0, 100, 0.9, True), VadFrame(120, 200, 0.8, True), VadFrame(400, 500, 0.8, True)),
        )
        self.assertEqual([(region.start_us, region.end_us) for region in evidence.speech_regions], [(0, 200), (400, 500)])

    def test_probe_discontinuity_is_diagnostic_not_unapproved_scd_cut(self) -> None:
        segmentation = RuleEvidenceSegmentation(SegmentationConfig(probe_discontinuity_min=0.2))
        evidence = segmentation.build(
            view_id="v",
            vad_frames=(VadFrame(0, 1_000, 1.0, True),),
            probes=(
                ProbeEvidence("a", 0, 400, 400, (1.0, 0.0), 1.0),
                ProbeEvidence("b", 400, 800, 400, (-1.0, 0.0), 1.0),
            ),
        )
        self.assertEqual(evidence.scd_events, ())
        self.assertEqual(len(evidence.probe_discontinuities), 1)

    def test_boundary_enforcement_inputs_are_rejected(self) -> None:
        segmentation = RuleEvidenceSegmentation()
        with self.assertRaises(ContractValidationError):
            segmentation.build(
                view_id="v",
                vad_frames=(),
                approved_scd_events=(SpeakerChangeEvent(500, 0.9, "diagnostic"),),
            )
        with self.assertRaises(ContractValidationError):
            segmentation.build(
                view_id="v",
                vad_frames=(),
                approved_overlap_regions=(OverlapEvent(600, 800, 0.9),),
            )


if __name__ == "__main__":
    unittest.main()
