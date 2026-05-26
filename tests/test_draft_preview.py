from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smzdm_notice.preferences import preview as draft_preview
from smzdm_notice.preferences.models import ConfigDraft


class DraftPreviewTests(unittest.TestCase):
    def test_change_preview_uses_card_friendly_sections_for_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preference.md").write_text("before\n- 坚果\nafter\n", encoding="utf-8")
            with patch("smzdm_notice.core.config.PROJECT_ROOT", root):
                preview = draft_preview.build_change_preview(
                    "preference.md",
                    "- 坚果",
                    "- 坚果、坚果礼盒、坚果混合装等同类商品",
                )

        self.assertNotIn("```diff", preview)
        self.assertNotIn("@@", preview)
        self.assertIn("位置：`preference.md` 第 2 行", preview)
        self.assertIn("**附近原文**", preview)
        self.assertIn("> before", preview)
        self.assertIn("> after", preview)
        self.assertIn("**删除**", preview)
        self.assertIn("~~- 坚果~~", preview)
        self.assertIn("**新增**", preview)
        self.assertIn("- 坚果、坚果礼盒、坚果混合装等同类商品", preview)

    def test_change_preview_uses_card_friendly_sections_for_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preference.md").write_text("before\n- 坚果\nafter\n", encoding="utf-8")
            with patch("smzdm_notice.core.config.PROJECT_ROOT", root):
                preview = draft_preview.build_change_preview("preference.md", "- 坚果", "")

        self.assertIn("**删除**", preview)
        self.assertIn("~~- 坚果~~", preview)
        self.assertNotIn("**新增**", preview)

    def test_append_preview_uses_card_friendly_sections(self) -> None:
        draft = ConfigDraft(
            draft_id="d1",
            target_file="preference.md",
            title="新增规则",
            summary="新增一条规则",
            append_text="- 不再推荐坚果",
            source="test",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preference.md").write_text("line1\nline2\n", encoding="utf-8")
            with patch("smzdm_notice.core.config.PROJECT_ROOT", root):
                preview = draft_preview.build_draft_change_preview(draft)

        self.assertIn("位置：`preference.md` 文件末尾", preview)
        self.assertIn("> line1", preview)
        self.assertIn("> line2", preview)
        self.assertIn("**新增**", preview)
        self.assertIn("- 不再推荐坚果", preview)

    def test_change_preview_filters_blank_and_separator_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preference.md").write_text(
                "- 板栗仁\n- 宠物相关用品\n- 咖啡机\n- 坚果\n\n---\n\n## 下一节\n",
                encoding="utf-8",
            )
            with patch("smzdm_notice.core.config.PROJECT_ROOT", root):
                preview = draft_preview.build_change_preview("preference.md", "- 坚果", "- 坚果礼盒")

        self.assertIn("> - 板栗仁", preview)
        self.assertIn("> - 宠物相关用品", preview)
        self.assertIn("> - 咖啡机", preview)
        self.assertNotIn("> ---", preview)


if __name__ == "__main__":
    unittest.main()
