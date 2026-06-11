from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import BadRequestError

from smzdm_notice.llm.routing import ResolvedLLMConfig
from smzdm_notice.preferences.builder import (
    _call_llm_for_draft,
    _deal_action_message,
    _draft_with_llm,
    _parse_llm_draft_content,
    build_arbitration_draft,
    build_deal_action_draft,
    build_message_draft,
    build_revision_draft,
)
from smzdm_notice.preferences.models import ConfigDraft
from smzdm_notice.preferences.store import DraftStore


def _openai_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://llm.example.com/v1/chat/completions")
    return httpx.Response(status_code, request=request)


def _draft_llm_config(api_key: str = "draft-key") -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        agent="draft",
        connection="draft",
        connection_label="Draft",
        provider="openai_compatible",
        base_url="https://draft.example.com",
        api_key_env="LLM_DRAFT_TEST_API_KEY",
        api_key=api_key,
        model_id="draft-model",
        timeout_seconds=300.0,
        max_retries=2,
        temperature=0.0,
        response_format={"type": "json_object"},
        extra_body={},
    )


class PreferenceEditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "preference.md").write_text("# preference\n", encoding="utf-8")
        (self.root / "inventory.md").write_text("# inventory\n", encoding="utf-8")
        self.store = DraftStore(
            draft_file=self.root / "drafts.json",
            backup_dir=self.root / "backups",
            audit_file=self.root / "audit.jsonl",
            root=self.root,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_message_draft_uses_llm_data(self) -> None:
        with patch(
            "smzdm_notice.preferences.builder._draft_with_llm",
            return_value={
                "target_file": "inventory.md",
                "title": "更新库存",
                "summary": "记录抽纸库存",
                "append_text": "- 抽纸还剩 3 包",
            },
        ):
            draft = build_message_draft("抽纸还剩 3 包", self.store)

        self.assertIsNotNone(draft)
        self.assertEqual(draft.target_file, "inventory.md")
        self.assertIn("抽纸还剩 3 包", draft.append_text)

    def test_message_draft_returns_none_without_llm_fallback(self) -> None:
        with patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config(api_key="")):
            draft = build_message_draft("拉黑坚果", self.store)

        self.assertIsNone(draft)

    def test_message_draft_llm_uses_draft_model(self) -> None:
        content = (
            '{"target_file":"preference.md","title":"拉黑坚果","summary":"新增排除规则","append_text":"- 不再推荐坚果"}'
        )
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        create_kwargs = {}

        def create_completion(**kwargs):
            create_kwargs.update(kwargs)
            return response

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)))

        with (
            patch(
                "smzdm_notice.preferences.builder.resolve",
                return_value=replace(_draft_llm_config(), temperature=0.2, extra_body={"do_sample": False}),
            ),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client) as get_client,
        ):
            data = _draft_with_llm("拉黑坚果")

        self.assertEqual(data["target_file"], "preference.md")
        get_client.assert_called_once()
        self.assertEqual(create_kwargs["model"], "draft-model")
        self.assertEqual(create_kwargs["temperature"], 0.2)
        self.assertEqual(create_kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(create_kwargs["extra_body"], {"do_sample": False})

    def test_call_llm_for_draft_returns_none_on_sdk_error(self) -> None:
        class FailingCompletions:
            def create(self, **kwargs):
                raise BadRequestError("bad request", response=_openai_response(400), body=None)

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            data = _call_llm_for_draft([{"role": "user", "content": "改配置"}])

        self.assertIsNone(data)

    def test_arbitration_draft_uses_structured_data(self) -> None:
        draft = build_arbitration_draft(
            {
                "target_file": "preference.md",
                "title": "限制黑名单扩展",
                "summary": "避免把坚果扩展到所有生鲜",
                "append_text": "- 黑名单只按用户明确写出的品类精确匹配，不扩展到更大类别。",
            },
            self.store,
            suggestion="不要扩展黑名单",
        )

        self.assertIsNotNone(draft)
        self.assertEqual(draft.target_file, "preference.md")
        self.assertNotIn("仲裁建议：", draft.append_text)
        self.assertNotIn("来源", draft.append_text)
        self.assertNotIn("机器人确认修改", draft.append_text)
        self.assertIn("suggestion_hash", draft.metadata)

    def test_arbitration_draft_rejects_invalid_target(self) -> None:
        draft = build_arbitration_draft(
            {
                "target_file": "inventory.md",
                "title": "bad",
                "summary": "bad",
                "append_text": "- bad",
            },
            self.store,
        )

        self.assertIsNone(draft)

    def test_message_draft_rejects_invalid_llm_data(self) -> None:
        with patch(
            "smzdm_notice.preferences.builder._draft_with_llm",
            return_value={
                "target_file": "main.py",
                "title": "bad",
                "summary": "bad",
                "append_text": "- bad",
            },
        ):
            self.assertIsNone(build_message_draft("改配置", self.store))
        with patch(
            "smzdm_notice.preferences.builder._draft_with_llm",
            return_value={
                "target_file": "preference.md",
                "title": "bad",
                "summary": "bad",
                "append_text": "",
            },
        ):
            self.assertIsNone(build_message_draft("改配置", self.store))

    def test_parse_llm_draft_json_variants(self) -> None:
        plain = '{"target_file":"preference.md","title":"t","summary":"s","append_text":"- 关注婴儿车"}'
        fenced = '```json\n{"target_file":"inventory.md","title":"t","summary":"s","append_text":"- 抽纸 3 包"}\n```'
        wrapped = (
            '好的，配置如下：{"target_file":"preference.md","title":"t","summary":"s","append_text":"- 不再推荐坚果"}'
        )

        self.assertEqual(_parse_llm_draft_content(plain)["target_file"], "preference.md")
        self.assertEqual(_parse_llm_draft_content(fenced)["target_file"], "inventory.md")
        self.assertIn("坚果", _parse_llm_draft_content(wrapped)["append_text"])

    def test_parse_llm_draft_rejects_empty_or_invalid_json(self) -> None:
        with self.assertRaises(ValueError):
            _parse_llm_draft_content("")
        with self.assertRaises(ValueError):
            _parse_llm_draft_content("没有 JSON")

    def test_apply_draft_writes_backup_and_audit(self) -> None:
        draft = ConfigDraft(
            draft_id="d1",
            target_file="preference.md",
            title="新增关注",
            summary="关注婴儿车",
            append_text="- 关注婴儿车",
            source="test",
        )
        self.store.create(draft)

        ok, message = self.store.apply("d1", operator="u1")

        self.assertTrue(ok, message)
        content = (self.root / "preference.md").read_text(encoding="utf-8")
        self.assertIn("关注婴儿车", content)
        self.assertNotIn("机器人确认修改", content)
        self.assertNotIn("来源：", content)
        self.assertEqual(len(list((self.root / "backups").glob("preference.md.*.bak"))), 1)
        audit = (self.root / "audit.jsonl").read_text(encoding="utf-8")
        self.assertIn('"action": "applied"', audit)
        self.assertIn('"source": "test"', audit)
        self.assertIn('"operator": "u1"', audit)
        self.assertIn('"draft_id": "d1"', audit)
        self.assertIn('"backup":', audit)

    def test_duplicate_signature_is_idempotent(self) -> None:
        first = ConfigDraft("d1", "preference.md", "t", "s", "- 不再推荐卷纸", "test")
        second = ConfigDraft("d2", "preference.md", "t", "s", "- 不再推荐卷纸", "test")
        self.store.create(first)
        self.store.create(second)

        self.assertTrue(self.store.apply("d1")[0])
        ok, message = self.store.apply("d2")

        self.assertTrue(ok)
        self.assertIn("已采纳过", message)

    def test_expire_pending_cancels_only_expired_pending_drafts(self) -> None:
        expired = ConfigDraft(
            "expired",
            "preference.md",
            "t",
            "s",
            "- 旧草案",
            "test",
            created_at=100.0,
            preview_message_id="om_expired",
        )
        active = ConfigDraft(
            "active",
            "preference.md",
            "t",
            "s",
            "- 新草案",
            "test",
            created_at=100.0 + 24 * 60 * 60,
        )
        applied = ConfigDraft(
            "applied",
            "preference.md",
            "t",
            "s",
            "- 已应用",
            "test",
            created_at=100.0,
            status="applied",
        )
        self.store.create(expired)
        self.store.create(active)
        self.store.create(applied)

        with patch("smzdm_notice.preferences.models.time.time", return_value=100.0 + 24 * 60 * 60 + 1):
            expired_drafts = self.store.expire_pending()

        self.assertEqual([d.draft_id for d in expired_drafts], ["expired"])
        self.assertEqual(self.store.get("expired").status, "cancelled")
        self.assertEqual(self.store.get("active").status, "pending")
        self.assertEqual(self.store.get("applied").status, "applied")

    def test_compact_removes_old_terminal_drafts_only(self) -> None:
        old_cancelled = ConfigDraft(
            "old-cancelled",
            "preference.md",
            "t",
            "s",
            "- old",
            "test",
            created_at=100.0,
            status="cancelled",
        )
        old_applied = ConfigDraft(
            "old-applied",
            "preference.md",
            "t",
            "s",
            "- old applied",
            "test",
            created_at=100.0,
            status="applied",
        )
        old_pending = ConfigDraft(
            "old-pending",
            "preference.md",
            "t",
            "s",
            "- old pending",
            "test",
            created_at=100.0,
        )
        recent_cancelled = ConfigDraft(
            "recent-cancelled",
            "preference.md",
            "t",
            "s",
            "- recent",
            "test",
            created_at=100.0 + 24 * 60 * 60,
            status="cancelled",
        )
        for draft in (old_cancelled, old_applied, old_pending, recent_cancelled):
            self.store.create(draft)

        with patch("smzdm_notice.preferences.store.time.time", return_value=100.0 + 24 * 60 * 60 + 1):
            removed = self.store.compact()

        self.assertEqual({d.draft_id for d in removed}, {"old-cancelled", "old-applied"})
        self.assertIsNone(self.store.get("old-cancelled"))
        self.assertIsNone(self.store.get("old-applied"))
        self.assertIsNotNone(self.store.get("old-pending"))
        self.assertIsNotNone(self.store.get("recent-cancelled"))

    def test_backup_names_do_not_collide_within_same_second(self) -> None:
        target = self.root / "preference.md"
        with patch("smzdm_notice.preferences.store.datetime") as mock_datetime:
            mock_datetime.now.side_effect = [
                datetime(2026, 5, 18, 13, 30, 1, 1),
                datetime(2026, 5, 18, 13, 30, 1, 2),
            ]
            first = self.store._backup(target)
            second = self.store._backup(target)

        self.assertNotEqual(first.name, second.name)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_rejects_unknown_target_file(self) -> None:
        draft = ConfigDraft("d1", "main.py", "t", "s", "bad", "test")
        with self.assertRaises(ValueError):
            self.store.create(draft)

    def test_deal_action_draft(self) -> None:
        llm_data = {
            "target_file": "preference.md",
            "edit_mode": "replace",
            "title": "关注婴儿推车",
            "summary": "把婴儿推车加入关注品类",
            "search_text": "- **生活家电**：例如电风扇、台灯、加湿器等",
            "replace_text": "- **生活家电**：例如电风扇、台灯、加湿器等\n- **婴儿推车**：有好价可以推荐",
        }
        value = {
            "item_title": "婴儿推车 好价",
            "item_brand": "测试品牌",
            "article_id": "1",
            "item_link": "https://example.com/deal/1",
        }
        with patch("smzdm_notice.preferences.builder._draft_with_llm", return_value=llm_data) as draft_with_llm:
            draft = build_deal_action_draft("deal_follow", value, self.store)

        self.assertIsNotNone(draft)
        self.assertEqual(draft.target_file, "preference.md")
        self.assertEqual(draft.edit_mode, "replace")
        self.assertEqual(draft.source, "商品卡片快捷操作")
        self.assertEqual(draft.metadata["article_id"], "1")
        self.assertIn("婴儿推车", draft.replace_text)
        message = draft_with_llm.call_args.args[0]
        self.assertIn("关注与该商品相关或同类商品", message)
        self.assertIn("婴儿推车 好价", message)
        self.assertIn("测试品牌", message)
        self.assertEqual(draft_with_llm.call_args.kwargs["root"], self.store.root)

    def test_deal_action_messages_describe_each_intent(self) -> None:
        value = {"item_title": "抽纸", "item_brand": "蓝月亮", "article_id": "1001"}

        ignore = _deal_action_message("deal_ignore_category", value)
        stock = _deal_action_message("deal_stock_enough", value)
        follow = _deal_action_message("deal_follow", value)

        self.assertIn("不要推荐与该商品同类或高度相似", ignore)
        self.assertIn("库存充足", stock)
        self.assertIn("关注与该商品相关或同类商品", follow)
        self.assertIn("抽纸", ignore)
        self.assertIn("蓝月亮", stock)
        self.assertIn("1001", follow)

    def test_deal_action_draft_returns_none_when_llm_fails(self) -> None:
        with patch("smzdm_notice.preferences.builder._draft_with_llm", return_value=None):
            draft = build_deal_action_draft(
                "deal_follow",
                {"item_title": "婴儿推车 好价", "article_id": "1"},
                self.store,
            )

        self.assertIsNone(draft)

    def test_revision_of_append_can_remove_part_from_pending_draft(self) -> None:
        original = ConfigDraft(
            draft_id="original",
            target_file="preference.md",
            title="开放推荐规则",
            summary="测试",
            append_text="- 开放推荐硬性门槛\n- 推荐理由真实性",
            source="仲裁建议一键采纳",
        )
        with patch(
            "smzdm_notice.preferences.builder._revision_with_llm",
            return_value={
                "target_file": "preference.md",
                "edit_mode": "append",
                "title": "开放推荐规则",
                "summary": "去掉硬性门槛要求，只保留理由真实性",
                "append_text": "- 推荐理由真实性",
            },
        ):
            draft = build_revision_draft("不要硬性门槛要求", original, self.store)

        self.assertIsNotNone(draft)
        self.assertEqual(draft.edit_mode, "append")
        self.assertEqual(draft.append_text, "- 推荐理由真实性")

    def test_revision_retries_when_delete_targets_unapplied_draft_text(self) -> None:
        original = ConfigDraft(
            draft_id="original",
            target_file="preference.md",
            title="开放推荐规则",
            summary="测试",
            append_text="- 开放推荐硬性门槛\n- 推荐理由真实性",
            source="仲裁建议一键采纳",
        )
        bad_delete = {
            "target_file": "preference.md",
            "edit_mode": "delete",
            "title": "移除硬性门槛",
            "summary": "错误地删除未执行草案文本",
            "search_text": "- 开放推荐硬性门槛",
        }
        fixed_append = {
            "target_file": "preference.md",
            "edit_mode": "append",
            "title": "开放推荐规则",
            "summary": "去掉硬性门槛要求，只保留理由真实性",
            "append_text": "- 推荐理由真实性",
        }
        with patch(
            "smzdm_notice.preferences.builder._revision_with_llm",
            side_effect=[bad_delete, fixed_append],
        ) as revise:
            draft = build_revision_draft("不要硬性门槛要求", original, self.store)

        self.assertEqual(revise.call_count, 2)
        self.assertIsNotNone(draft)
        self.assertEqual(draft.edit_mode, "append")
        self.assertEqual(draft.append_text, "- 推荐理由真实性")

    def test_revision_returns_none_when_retry_still_targets_unapplied_draft_text(self) -> None:
        original = ConfigDraft(
            draft_id="original",
            target_file="preference.md",
            title="开放推荐规则",
            summary="测试",
            append_text="- 开放推荐硬性门槛\n- 推荐理由真实性",
            source="仲裁建议一键采纳",
        )
        bad_delete = {
            "target_file": "preference.md",
            "edit_mode": "delete",
            "title": "移除硬性门槛",
            "summary": "错误地删除未执行草案文本",
            "search_text": "- 开放推荐硬性门槛",
        }
        with patch("smzdm_notice.preferences.builder._revision_with_llm", side_effect=[bad_delete, bad_delete]):
            draft = build_revision_draft("不要硬性门槛要求", original, self.store)

        self.assertIsNone(draft)

    def test_revision_allows_delete_when_search_text_exists_in_real_file(self) -> None:
        (self.root / "preference.md").write_text("# preference\n- 旧规则\n", encoding="utf-8")
        original = ConfigDraft(
            draft_id="original",
            target_file="preference.md",
            title="开放推荐规则",
            summary="测试",
            append_text="- 新规则",
            source="用户对话：新增规则",
        )
        with patch(
            "smzdm_notice.preferences.builder._revision_with_llm",
            return_value={
                "target_file": "preference.md",
                "edit_mode": "delete",
                "title": "删除旧规则",
                "summary": "用户明确要求删除真实文件里的旧规则",
                "search_text": "- 旧规则",
            },
        ):
            draft = build_revision_draft("不要追加了，删除文件里的旧规则", original, self.store)

        self.assertIsNotNone(draft)
        self.assertEqual(draft.edit_mode, "delete")
        self.assertEqual(draft.search_text, "- 旧规则")

    def test_revision_allows_replace_when_search_text_exists_in_real_file(self) -> None:
        (self.root / "preference.md").write_text("# preference\n- 旧规则\n", encoding="utf-8")
        original = ConfigDraft(
            draft_id="original",
            target_file="preference.md",
            title="开放推荐规则",
            summary="测试",
            append_text="- 新规则",
            source="用户对话：新增规则",
        )
        with patch(
            "smzdm_notice.preferences.builder._revision_with_llm",
            return_value={
                "target_file": "preference.md",
                "edit_mode": "replace",
                "title": "替换旧规则",
                "summary": "用户明确要求替换真实文件里的旧规则",
                "search_text": "- 旧规则",
                "replace_text": "- 新规则",
            },
        ):
            draft = build_revision_draft("不要追加了，改文件里的旧规则", original, self.store)

        self.assertIsNotNone(draft)
        self.assertEqual(draft.edit_mode, "replace")
        self.assertEqual(draft.search_text, "- 旧规则")
        self.assertEqual(draft.replace_text, "- 新规则")


if __name__ == "__main__":
    unittest.main()
