from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from smzdm_notice.core.dedup import DedupManager


class DedupManagerTests(unittest.TestCase):
    def test_is_new_treats_expired_url_as_new_without_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = DedupManager(str(Path(tmp) / "dedup.json"), expire_hours=1)
            manager._cache["https://example.com/deal/1001"] = time.time() - 3601

            self.assertTrue(manager.is_new("https://example.com/deal/1001"))

    def test_cleanup_removes_expired_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = DedupManager(str(Path(tmp) / "dedup.json"), expire_hours=1)
            manager._cache["expired"] = time.time() - 3601
            manager._cache["fresh"] = time.time()

            manager.cleanup()

            self.assertNotIn("expired", manager._cache)
            self.assertIn("fresh", manager._cache)


if __name__ == "__main__":
    unittest.main()
