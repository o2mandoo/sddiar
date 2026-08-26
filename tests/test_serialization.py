import json
import unittest

from sddiar.contracts import (
    AudioSourceMetadata,
    FileQualityReport,
    PipelineResult,
    PipelineRunMetadata,
    Timebase,
)
from sddiar.errors import ResultSchemaValidationFailed
from sddiar.serialization import ResultSerializer, canonical_json
from sddiar.contracts import EmbeddingResult


class SerializationTests(unittest.TestCase):
    def result(self) -> PipelineResult:
        timebase = Timebase("tb", source_sample_rate_hz=16000, duration_us=100)
        source = AudioSourceMetadata("a" * 64, "wav", "pcm_s16le", 16000, 1, 100, timebase)
        run = PipelineRunMetadata("run", "0.1", "pack", {}, None, "CPUExecutionProvider", {}, {}, 1.0)
        quality = FileQualityReport("REVIEW_REQUIRED", "UNCERTAIN_1_OR_2", "MANUAL_REVIEW", ("Q_CALIBRATION_MISSING",), {})
        return PipelineResult("result", source, run, (), (), (), (), quality, (), ())

    def test_pipeline_result_is_canonical_json(self) -> None:
        payload = ResultSerializer().serialize(self.result())
        self.assertEqual(payload, ResultSerializer().serialize(self.result()))
        self.assertEqual(json.loads(payload)["result_id"], "result")

    def test_internal_embedding_is_rejected(self) -> None:
        embedding = EmbeddingResult("e", "t", True, (1.0, 0.0), dimension=2)
        with self.assertRaises(ResultSchemaValidationFailed):
            canonical_json(embedding)


if __name__ == "__main__":
    unittest.main()
