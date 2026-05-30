from __future__ import annotations

import unittest

from smzdm_notice.llm.json_utils import extract_json_object


class JsonUtilsTests(unittest.TestCase):
    def test_extracts_first_valid_json_object_when_multiple_blocks_exist(self) -> None:
        content = 'before {"first": true} middle {"second": true}'

        self.assertEqual(extract_json_object(content), {"first": True})

    def test_skips_invalid_brace_before_valid_json_object(self) -> None:
        content = 'not json { broken } then {"ok": true, "count": 2}'

        self.assertEqual(extract_json_object(content), {"ok": True, "count": 2})


if __name__ == "__main__":
    unittest.main()
