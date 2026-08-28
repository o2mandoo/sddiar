from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import unittest

from sddiar.evaluation import RTTMRecord, UEMInterval
from sddiar.nikl_adapter import (
    NIKLAdapterError,
    NIKLReference,
    NIKLSchemaError,
    NIKLTimeError,
    build_nikl_manifest_row,
    format_nikl_rttm,
    format_nikl_uem,
    parse_nikl_reference,
)


def _payload() -> dict:
    return {
        "document": [
            {
                "metadata": {"speaker": [{"id": "원본-A"}, {"id": "원본-B"}]},
                "utterance": [
                    {"speaker_id": "원본-A", "start": "0.0", "end": "1.25", "form": "비공개 문장"},
                    {"speaker_id": "원본-B", "start": "1.0", "end": "2.0", "form": "또 다른 비공개 문장"},
                ],
            }
        ]
    }


class NIKLAdapterTests(unittest.TestCase):
    def test_converts_seconds_to_immutable_generic_records_and_preserves_overlap(self) -> None:
        result = parse_nikl_reference(_payload(), 2_500_000, time_unit="seconds")

        self.assertIsInstance(result, NIKLReference)
        self.assertEqual(result.records, (
            RTTMRecord(result.recording_id, "REF_00", 0, 1_250_000),
            RTTMRecord(result.recording_id, "REF_01", 1_000_000, 2_000_000),
        ))
        self.assertEqual(result.uem, (UEMInterval(result.recording_id, 0, 2_000_000),))
        self.assertEqual(result.evidence.duration_us, 2_500_000)
        self.assertEqual(result.evidence.speaker_count, 2)
        self.assertEqual(result.evidence.overlap_duration_us, 250_000)
        self.assertEqual(result.evidence.quality_status, "REVIEW_REQUIRED")
        self.assertEqual(result.public_evidence()["recording_id"], result.recording_id)
        with self.assertRaises(FrozenInstanceError):
            result.records = ()  # type: ignore[misc]

    def test_recording_id_is_canonical_source_hash_and_public_output_is_redacted(self) -> None:
        payload = _payload()
        expected = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        result = parse_nikl_reference(payload, 2_500_000, time_unit="milliseconds")
        self.assertEqual(result.evidence.canonical_payload_sha256, expected)
        self.assertTrue(result.recording_id.startswith("recording-"))
        public = json.dumps(result.public_evidence(), ensure_ascii=False)
        self.assertNotIn("원본-A", public)
        self.assertNotIn("원본-B", public)
        self.assertNotIn("비공개", public)
        self.assertNotIn("form", public)

    def test_rejects_ambiguous_unit_unknown_speaker_three_speakers_and_bad_bounds(self) -> None:
        with self.assertRaises(NIKLTimeError):
            parse_nikl_reference(_payload(), 2_000_000, time_unit=None)

        unknown = _payload()
        unknown["document"][0]["utterance"][0]["speaker_id"] = "not-listed"
        with self.assertRaises(NIKLSchemaError):
            parse_nikl_reference(unknown, 3_000_000, time_unit="seconds")

        three = _payload()
        three["document"][0]["metadata"]["speaker"].append({"id": "원본-C"})
        with self.assertRaises(NIKLSchemaError):
            parse_nikl_reference(three, 3_000_000, time_unit="seconds")

        outside = _payload()
        outside["document"][0]["utterance"][0]["end"] = "3.0"
        with self.assertRaises(NIKLTimeError):
            parse_nikl_reference(outside, 2_500_000, time_unit="seconds")

    def test_rejects_wrong_document_cardinality_missing_fields_and_nonfinite_times(self) -> None:
        with self.assertRaises(NIKLSchemaError):
            parse_nikl_reference({"document": []}, 1_000_000, time_unit="seconds")
        missing = _payload()
        del missing["document"][0]["utterance"][0]["start"]
        with self.assertRaises(NIKLSchemaError):
            parse_nikl_reference(missing, 3_000_000, time_unit="seconds")
        nonfinite = _payload()
        nonfinite["document"][0]["utterance"][0]["start"] = "NaN"
        with self.assertRaises(NIKLTimeError):
            parse_nikl_reference(nonfinite, 3_000_000, time_unit="seconds")

    def test_accepts_official_top_level_profile_and_merges_only_same_speaker(self) -> None:
        payload = _payload()
        payload.update({"id": "private-source-id", "metadata": {"category": "spoken"}})
        payload["document"][0]["id"] = "private-document-id"
        payload["document"][0]["utterance"].insert(
            1, {"speaker_id": "원본-A", "start": "1.0", "end": "1.5"},
        )
        result = parse_nikl_reference(payload, 2_500_000, time_unit="seconds")
        self.assertEqual(result.records[0].speaker_id, "REF_00")
        self.assertEqual((result.records[0].start_us, result.records[0].end_us), (0, 1_500_000))
        self.assertEqual(len(result.records), 2)

    def test_formats_hash_bound_normalized_manifest_without_source_identity(self) -> None:
        result = parse_nikl_reference(_payload(), 2_500_000, time_unit="seconds")
        rttm = format_nikl_rttm(result)
        uem = format_nikl_uem(result)
        row = build_nikl_manifest_row(
            result, audio_sha256="a" * 64, sample_rate_hz=16000,
            split="DEVELOPMENT_HOLDOUT", audio_path=f"audio/{result.recording_id}.wav",
            rttm_path=f"rttm/{result.recording_id}.rttm",
            uem_path=f"uem/{result.recording_id}.uem",
            speaker_group_ids=("person-hmac-001", "person-hmac-002"),
        )
        self.assertEqual(row["rttm_sha256"], hashlib.sha256(rttm.encode()).hexdigest())
        self.assertEqual(row["uem_sha256"], hashlib.sha256(uem.encode()).hexdigest())
        self.assertEqual(row["reference_status"], "CONVERTED_PROVISIONAL")
        self.assertEqual(row["uem_policy"], "ANNOTATED_EXTENT_PROVISIONAL")
        serialized = json.dumps(row, ensure_ascii=False)
        self.assertNotIn("원본-A", serialized)
        self.assertNotIn("비공개", serialized)

    def test_manifest_speaker_groups_must_match_reference_cardinality(self) -> None:
        result = parse_nikl_reference(_payload(), 2_500_000, time_unit="seconds")
        with self.assertRaisesRegex(NIKLAdapterError, "speaker group"):
            build_nikl_manifest_row(
                result, audio_sha256="a" * 64, sample_rate_hz=16000,
                split="CALIBRATION", audio_path="audio/opaque.wav",
                rttm_path="rttm/opaque.rttm", uem_path="uem/opaque.uem",
                speaker_group_ids=("person-hmac-001",),
            )


if __name__ == "__main__":
    unittest.main()
