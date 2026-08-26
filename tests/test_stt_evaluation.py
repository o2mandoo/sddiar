import json
import unittest

from sddiar.stt_evaluation import (
    RecordingSTTInput,
    STTEvaluationError,
    SpeakerText,
    TextNormalizationConfig,
    TimedWord,
    aggregate_recordings,
    build_run_manifest,
    cer_metrics,
    character_error_rate,
    evaluate_recording,
    normalize_text,
    parse_pseudonymized_reference_records,
    run_identity,
    speaker_attributed_cer,
    speaker_attributed_wer,
    timestamp_word_metrics,
    wer_metrics,
)


class STTEvaluationTests(unittest.TestCase):
    def test_nfc_and_explicit_punctuation_space_policies(self):
        decomposed = "각,  나다!"
        self.assertEqual(normalize_text(decomposed), "각, 나다!")
        policy = TextNormalizationConfig(punctuation_policy="remove", whitespace_policy="collapse")
        self.assertEqual(normalize_text(decomposed, policy), "각 나다")
        space_policy = TextNormalizationConfig(punctuation_policy="space", whitespace_policy="collapse")
        self.assertEqual(normalize_text("가,나다", space_policy), "가 나다")

    def test_cer_wer_and_empty_cases(self):
        self.assertEqual(cer_metrics("안녕 세계", "안녕 세게").distance, 1)
        self.assertEqual(wer_metrics("안녕 세계", "안녕").error_rate, 0.5)
        self.assertEqual(character_error_rate("", ""), 0.0)
        self.assertEqual(character_error_rate("", "가"), 1.0)
        self.assertEqual(character_error_rate("가", ""), 1.0)
        self.assertEqual(wer_metrics("", "가 나").insertions, 2)
        self.assertEqual(wer_metrics("", "가 나").error_rate, 1.0)

    def test_speaker_mapping_is_optimal_deterministic_and_supports_korean(self):
        ref = (SpeakerText("ref_a", "안녕 세계"), SpeakerText("ref_b", "좋은 아침"))
        hyp = (SpeakerText("hyp_x", "좋은 아침"), SpeakerText("hyp_y", "안녕 세계"))
        score = speaker_attributed_cer(ref, hyp)
        self.assertEqual(score.error_rate, 0.0)
        self.assertEqual(tuple((x.hypothesis_speaker_id, x.reference_speaker_id) for x in score.mapping),
                         (("hyp_x", "ref_b"), ("hyp_y", "ref_a")))
        self.assertEqual(speaker_attributed_wer(ref, hyp).error_rate, 0.0)
        one_ref = speaker_attributed_cer({"ref_a": "정답"}, {"hyp_a": "오답", "hyp_b": "정답"})
        self.assertEqual(tuple((x.hypothesis_speaker_id, x.reference_speaker_id) for x in one_ref.mapping),
                         (("hyp_a", None), ("hyp_b", "ref_a")))
        self.assertEqual(one_ref.error_rate, 1.0)
        self.assertEqual(speaker_attributed_cer({"ref_a": "가"}, {}).unmatched_reference_speakers, 1)
        with self.assertRaises(STTEvaluationError):
            speaker_attributed_cer({"a": "가", "b": "나", "c": "다"}, {})

    def test_timestamp_coverage_median_p95_and_no_reference(self):
        ref = (TimedWord("ref_1", "안녕", 0, 100), TimedWord("ref_2", "세계", 100, 200))
        hyp = (TimedWord("hyp_1", "안녕", 10, 110), TimedWord("hyp_2", "틀림", 120, 250))
        score = timestamp_word_metrics(ref, hyp)
        self.assertEqual(score.coverage, 0.5)
        self.assertEqual(score.matched_words, 1)
        self.assertEqual(score.median_boundary_error_us, 10.0)
        self.assertEqual(score.p95_boundary_error_us, 10.0)
        empty = timestamp_word_metrics((), ())
        self.assertEqual((empty.coverage, empty.median_boundary_error_us), (0.0, None))

    def test_aggregate_subgroups_bootstrap_order_and_hash_only_manifest(self):
        items = (
            RecordingSTTInput("record-a", "안녕 세계", "안녕 세계", subgroup={"kind": "clean"}),
            RecordingSTTInput("record-b", "좋은 아침", "좋은", subgroup={"kind": "short"}),
        )
        first = aggregate_recordings(items, bootstrap_iterations=32, bootstrap_seed=9)
        second = aggregate_recordings(tuple(reversed(items)), bootstrap_iterations=32, bootstrap_seed=9)
        self.assertEqual(first, second)
        self.assertEqual(first.count, 2)
        self.assertEqual([x.subgroup for x in first.subgroups], [(('kind', 'clean'),), (('kind', 'short'),)])
        serialized = json.dumps(first.run_manifest.as_dict(), sort_keys=True)
        self.assertNotIn("안녕", serialized)
        self.assertNotIn("record-a", serialized)
        self.assertNotIn("/", serialized)
        self.assertEqual(len(run_identity("비공개 원문", "비공개 결과")), 64)
        manifest = build_run_manifest(["비공개 원문", "/private/transcript.txt"])
        self.assertNotIn("비공개", json.dumps(manifest.as_dict(), ensure_ascii=False))
        self.assertNotIn("transcript", json.dumps(manifest.as_dict()))

    def test_parser_requires_already_pseudonymized_in_memory_records(self):
        records = parse_pseudonymized_reference_records([
            {"record_id": "rec_1", "speaker_id": "ref_01", "text": "안녕", "start_us": 0, "end_us": 50},
        ])
        self.assertEqual(records[0].text, "안녕")
        with self.assertRaises(STTEvaluationError):
            parse_pseudonymized_reference_records([{"speaker_id": "홍길동", "text": "안녕"}])
        with self.assertRaises(STTEvaluationError):
            parse_pseudonymized_reference_records([{"speaker_id": "ref_01", "text": "안녕", "path": "/tmp/x"}])


if __name__ == "__main__":
    unittest.main()
