from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from smzdm_notice.core.json_store import read_json_file, write_json_file
from smzdm_notice.llm.json_utils import extract_json_object


class JsonUtilsTests(unittest.TestCase):
    def test_extracts_first_valid_json_object_when_multiple_blocks_exist(self) -> None:
        content = 'before {"first": true} middle {"second": true}'

        self.assertEqual(extract_json_object(content), {"first": True})

    def test_skips_invalid_brace_before_valid_json_object(self) -> None:
        content = 'not json { broken } then {"ok": true, "count": 2}'

        self.assertEqual(extract_json_object(content), {"ok": True, "count": 2})

    def test_json_store_writes_parent_directory_and_reads_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "store.json"
            data = {"title": "好价", "count": 2}

            write_json_file(path, data)

            self.assertEqual(read_json_file(path), data)


if __name__ == "__main__":
    unittest.main()
