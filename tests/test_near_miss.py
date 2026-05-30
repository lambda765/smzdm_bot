from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smzdm_notice.core.near_miss import NearMissManager


class NearMissManagerTests(unittest.TestCase):
    def test_remove_batch_saves_once_when_entries_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = NearMissManager(str(Path(tmp) / "near_misses.json"))
            manager._store = {"a": {"article_id": "a"}, "b": {"article_id": "b"}, "c": {"article_id": "c"}}

            with patch.object(manager, "_save") as save:
                manager.remove_batch(["a", "b", "missing"])

            self.assertEqual(manager._store, {"c": {"article_id": "c"}})
            save.assert_called_once_with()

    def test_remove_batch_skips_save_when_no_entries_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = NearMissManager(str(Path(tmp) / "near_misses.json"))
            manager._store = {"a": {"article_id": "a"}}

            with patch.object(manager, "_save") as save:
                manager.remove_batch(["missing"])

            save.assert_not_called()

    def test_clear_and_set_digest_date_saves_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = NearMissManager(str(Path(tmp) / "near_misses.json"))
            manager._store = {"a": {"article_id": "a"}}

            with patch.object(manager, "_save") as save:
                manager.clear_and_set_digest_date("2026-05-29")

            self.assertEqual(manager._store, {})
            self.assertEqual(manager.get_last_digest_date(), "2026-05-29")
            save.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
