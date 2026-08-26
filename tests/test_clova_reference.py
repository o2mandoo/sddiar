import importlib.util
import json
import sys
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "parse_clova_reference.py"
    spec = importlib.util.spec_from_file_location("parse_clova_reference_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ClovaReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_parse_pseudonymous_timing_and_excludes_text(self) -> None:
        source = """회의 제목
2026.08.24 (월) 00:05 / 01:10
참석자 2명


홍길동 00:10
첫 번째 발화입니다.

김서준 00:25
다음 발화, punctuation!

홍길동 00:40
마지막 발화
"""
        result = self.module.parse_clova_text(source, source_name="원본.txt")
        self.assertEqual(result["speaker_ids"], ["REF_00", "REF_01"])
        self.assertEqual(
            [(turn["speaker_id"], turn["start_sec"], turn["end_sec"]) for turn in result["turns"]],
            [("REF_00", 10.0, 25.0), ("REF_01", 25.0, 40.0), ("REF_00", 40.0, 70.0)],
        )
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("첫 번째", encoded)
        self.assertNotIn("홍길동", encoded)
        self.assertNotIn("원본.txt", encoded)
        self.assertRegex(result["source_id"], r"^[0-9a-f]{16}$")

    def test_rejects_preamble_as_header_and_supports_hour_timestamps(self) -> None:
        source = "제목\n2026-08-24 녹음 1시간 20분 0초\n\nA팀 01:02:03\n내용\nB팀 01:03:04\n내용"
        result = self.module.parse_clova_text(source)
        self.assertEqual(result["stats"]["duration_sec"], 4800.0)
        self.assertEqual(result["turns"][-1]["end_sec"], 4800.0)

    def test_no_headers_is_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "headers"):
            self.module.parse_clova_text("제목\n본문만 있음")


if __name__ == "__main__":
    unittest.main()
