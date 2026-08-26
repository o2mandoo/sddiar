import unittest
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile

from sddiar.evaluation import (
    CalibrationGuardError, EvaluationError, EvaluationRecording, MicroDecision,
    RecordingManifest, RTTMRecord, ScoringConfig, SplitLeakageError, UEMInterval,
    WordAnnotation, WordDecision, build_run_manifest, parse_rttm, parse_uem,
    parse_words_jsonl, score_boundaries, score_corpus, score_recording,
    score_safety, validate_calibration_split, word_attribution_metrics,
    write_run_manifest,
)


class EvaluationTests(unittest.TestCase):
  def test_rttm_uem_and_words_are_source_microseconds(self):
    assert parse_rttm("SPEAKER f 1 1.5 0.5 <NA> <NA> REF_00 <NA>")[0].end_us == 2_000_000
    assert parse_uem("f 1 1.5 2.0")[0].end_us == 2_000_000
    word = parse_words_jsonl('{"word_id":"w1","start_us":1,"end_us":2,"text":"x","ref_speaker_id":"REF_00"}')[0]
    assert word.word_id == "w1"


  def test_complete_merge_is_unsafe_when_quality_passes(self):
    refs = {"f": (RTTMRecord("f", "REF_00", 0, 10), RTTMRecord("f", "REF_01", 10, 20))}
    pred = {"f": (RTTMRecord("f", "SPEAKER_00", 0, 20),)}
    m = score_safety(reference_by_file=refs, predicted_by_file=pred, quality_status_by_file={"f": "PASS_STANDARD"})
    assert m.acoustic_complete_merge_rate == 1.0
    assert m.unsafe_complete_merge_rate == 1.0
    assert m.quality_false_pass_rate == 1.0


  def test_word_and_micro_precision_coverage_exclude_unsafe_reference_words(self):
    words = [WordAnnotation("ok", 0, 1, "a", "REF_00"), WordAnnotation("ov", 1, 2, "b", "REF_00", overlap_flag=True)]
    p, c = word_attribution_metrics(words, [WordDecision("ok", "REF_00"), WordDecision("ov", "REF_01")])
    assert (p, c) == (1.0, 1.0)
    m = score_safety(reference_by_file={}, predicted_by_file={}, micro_decisions=(
        MicroDecision("1", "A", "A"), MicroDecision("2", "B", None)))
    assert m.micro_precision == 1.0 and m.micro_coverage == 0.5


  def test_split_guard_rejects_session_leakage_and_wrong_calibration_split(self):
    cal = [RecordingManifest("a", "session-1", "CALIBRATION")]
    holdout = [RecordingManifest("b", "session-1", "RELEASE_HOLDOUT")]
    with self.assertRaises(SplitLeakageError):
        validate_calibration_split(cal, holdout)
    with self.assertRaises(CalibrationGuardError):
        validate_calibration_split([RecordingManifest("a", "s", "DEV")], [])


  def test_uem_exact_duration_weighted_der_and_deterministic_mapping(self):
    second = 1_000_000
    refs = (
        RTTMRecord("f", "REF_00", 0, 4 * second),
        RTTMRecord("f", "REF_01", 4 * second, 8 * second),
    )
    predictions = (
        RTTMRecord("f", "X", 0, 3 * second),
        RTTMRecord("f", "Y", 3 * second, 5 * second),
        RTTMRecord("f", "X", 8 * second, 10 * second),
    )
    score = score_recording(
        recording_id="f", reference=refs, hypothesis=predictions,
        uem=(UEMInterval("f", second, 9 * second),),
    )
    assert tuple((item.hypothesis_speaker_id, item.reference_speaker_id) for item in score.mapping) == (
        ("X", "REF_00"), ("Y", "REF_01"),
    )
    assert score.diarization_all.scored_uem_us == 8 * second
    assert score.diarization_all.reference_speaker_us == 7 * second
    assert score.diarization_all.miss_us == 3 * second
    assert score.diarization_all.false_alarm_us == second
    assert score.diarization_all.confusion_us == second
    assert score.diarization_all.der == 5 / 7
    tied = score_recording(
        recording_id="f", reference=refs,
        hypothesis=(RTTMRecord("f", "ONLY", 0, 8 * second),),
        uem=(UEMInterval("f", 0, 8 * second),),
    )
    assert tied.mapping[0].reference_speaker_id == "REF_00"


  def test_one_microsecond_fake_second_speaker_cannot_defeat_complete_merge(self):
    second = 1_000_000
    refs = (
        RTTMRecord("f", "REF_00", 0, 5 * second),
        RTTMRecord("f", "REF_01", 5 * second, 10 * second),
    )
    predictions = (
        RTTMRecord("f", "SPEAKER_00", 0, 10 * second),
        RTTMRecord("f", "SPEAKER_01", 9 * second, 9 * second + 1),
    )
    score = score_recording(
        recording_id="f", reference=refs, hypothesis=predictions,
        uem=(UEMInterval("f", 0, 10 * second),),
    )
    assert score.reference_speaker_count == 2
    assert score.hypothesis_speaker_count == 1
    assert score.acoustic_complete_merge
    assert score.jer > 0.74
    safety = score_safety(reference_by_file={"f": refs}, predicted_by_file={"f": predictions})
    assert safety.acoustic_complete_merge_rate == 1.0


  def test_duration_floor_also_prevents_false_h2_from_one_tick_label(self):
    second = 1_000_000
    refs = (RTTMRecord("f", "REF_00", 0, 10 * second),)
    predictions = (
        RTTMRecord("f", "SPEAKER_00", 0, 10 * second),
        RTTMRecord("f", "SPEAKER_01", 3 * second, 3 * second + 1),
    )
    score = score_recording(
        recording_id="f", reference=refs, hypothesis=predictions,
        uem=(UEMInterval("f", 0, 10 * second),),
    )
    assert score.eligible_reference_h1
    assert not score.false_h2
    assert score.speaker_count_correct
    assert score.false_h2_secondary_duration_us == 1
    assert score.false_h2_duration_ratio == 1 / (10 * second)


  def test_overlap_duration_and_scd_collar_are_scored_separately(self):
    second = 1_000_000
    refs = (
        RTTMRecord("f", "REF_00", 0, 6 * second),
        RTTMRecord("f", "REF_01", 4 * second, 8 * second),
    )
    predictions = (
        RTTMRecord("f", "SPEAKER_00", 0, 4 * second),
        RTTMRecord("f", "OVERLAP", 4 * second, 5 * second),
        RTTMRecord("f", "SPEAKER_01", 5 * second, 8 * second),
    )
    score = score_recording(
        recording_id="f", reference=refs, hypothesis=predictions,
        uem=(UEMInterval("f", 0, 8 * second),),
        reference_scd_us=(4 * second,), predicted_scd_us=(4 * second + 400_000,),
    )
    assert score.overlap.evaluated
    assert score.overlap.reference_overlap_us == 2 * second
    assert score.overlap.predicted_overlap_us == second
    assert score.overlap.precision == 1.0
    assert score.overlap.recall == 0.5
    assert score.scd.f1 == 1.0
    missed = score_boundaries((second, 3 * second),
                              (second + 400_000, 2 * second, 3 * second + 600_000),
                              collar_us=500_000)
    assert (missed.true_positives, missed.false_positives, missed.false_negatives) == (1, 2, 1)


  def test_word_and_micro_metrics_use_optimal_speaker_mapping(self):
    second = 1_000_000
    refs = (
        RTTMRecord("f", "REF_00", 0, 5 * second),
        RTTMRecord("f", "REF_01", 5 * second, 10 * second),
    )
    predictions = (
        RTTMRecord("f", "B", 0, 5 * second),
        RTTMRecord("f", "A", 5 * second, 10 * second),
    )
    words = (
        WordAnnotation("w0", 0, second, "가", "REF_00"),
        WordAnnotation("w1", 5 * second, 6 * second, "나다", "REF_01"),
        WordAnnotation("boundary", 4 * second, 6 * second, "경계", "REF_00", boundary_crossing_flag=True),
    )
    score = score_recording(
        recording_id="f", reference=refs, hypothesis=predictions,
        uem=(UEMInterval("f", 0, 10 * second),), words=words,
        word_decisions=(WordDecision("w0", "B"), WordDecision("w1", None), WordDecision("boundary", "B")),
        micro_decisions=(MicroDecision("m0", "REF_00", "B"), MicroDecision("m1", "REF_01", "UNKNOWN_SHORT")),
    )
    assert score.words is not None
    assert (score.words.precision, score.words.coverage, score.words.strict_error) == (1.0, 0.5, 0.5)
    assert score.words.character_weighted_error == 2 / 3
    assert score.words.forced_overlap_or_boundary_assignments == 1
    assert score.micros is not None
    assert (score.micros.precision, score.micros.coverage, score.micros.unknown_short_rate) == (1.0, 0.5, 0.5)


  def test_subgroups_and_recording_bootstrap_are_order_invariant(self):
    second = 1_000_000

    def recording(file_id, wrong, pair):
        refs = (RTTMRecord(file_id, "REF_00", 0, 2 * second),)
        predicted = () if wrong else (RTTMRecord(file_id, "S0", 0, 2 * second),)
        return EvaluationRecording(
            file_id, refs, predicted, (UEMInterval(file_id, 0, 2 * second),),
            subgroups=(("gender_pair", pair), ("sample_rate_hz", "16000")),
        )

    first, second_recording = recording("opaque-001", False, "MF"), recording("opaque-002", True, "FF")
    config = ScoringConfig(bootstrap_iterations=64, bootstrap_seed=99)
    forward = score_corpus((first, second_recording), config=config)
    reverse = score_corpus((second_recording, first), config=config)
    assert forward == reverse
    assert forward.overall.diarization_all.der == 0.5
    labels = [subgroup.subgroup for subgroup in forward.subgroups]
    assert labels == ["gender_pair=FF", "gender_pair=MF", "sample_rate_hz=16000"]
    assert all(subgroup.bootstrap for subgroup in forward.subgroups)
    assert {interval.metric for interval in forward.bootstrap} >= {"der_all", "jer"}


  def test_run_manifest_is_redacted_content_addressed_and_no_clobber(self):
    config = ScoringConfig(bootstrap_iterations=7)
    manifest = build_run_manifest(inputs=(b"secret transcript", b"audio bytes"), config=config)
    serialized = json.dumps(manifest.as_dict(), sort_keys=True)
    assert "secret transcript" not in serialized
    assert "audio bytes" not in serialized
    assert "path" not in serialized
    assert len(manifest.config_sha256) == len(manifest.scorer_sha256) == 64
    assert manifest != build_run_manifest(inputs=(b"different",), config=config)
    role_bound = build_run_manifest(inputs={"reference": b"same", "hypothesis": b"other"}, config=config)
    changed_value = build_run_manifest(inputs={"reference": b"changed", "hypothesis": b"other"}, config=config)
    assert role_bound != changed_value
    with self.assertRaises(FrozenInstanceError):
        manifest.input_count = 99
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "run-manifest.json"
        write_run_manifest(manifest, target)
        assert json.loads(target.read_text(encoding="utf-8"))["manifest_sha256"] == manifest.manifest_sha256
        with self.assertRaises(EvaluationError):
            write_run_manifest(manifest, target)


  def test_scorer_rejects_more_than_two_speaker_labels(self):
    second = 1_000_000
    refs = tuple(RTTMRecord("f", f"REF_0{index}", index * second, (index + 1) * second)
                 for index in range(3))
    with self.assertRaises(EvaluationError):
        score_recording(recording_id="f", reference=refs, hypothesis=(),
                        uem=(UEMInterval("f", 0, 3 * second),))
    with self.assertRaises(EvaluationError):
        score_corpus(())


if __name__ == "__main__":
  unittest.main()
