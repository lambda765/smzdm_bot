from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from smzdm_notice.feishu.binding import FeishuBindingStore
from smzdm_notice.feishu.bot import (
    BotRuntime,
    FeishuInteractiveBot,
    MessageDeduper,
    _extract_card_value,
    _extract_message_text,
    _is_allowed_message,
    _strip_bot_mention,
)
from smzdm_notice.feishu.commands import COMMAND_SPECS, help_markdown
from smzdm_notice.preferences.models import ConfigDraft
from smzdm_notice.preferences.store import DraftStore


class FakeAction:
    def __init__(self, value):
        self.value = value


class FakeEvent:
    def __init__(self, value):
        self.action = FakeAction(value)


class FakeData:
    def __init__(self, value):
        self.event = FakeEvent(value)


class FakeMessage:
    chat_id = "oc_real_chat_id"
    chat_type = "p2p"
    content = '{"text": "你好"}'
    message_id = "om_real_message_id"


class FakeSenderId:
    open_id = "ou_real_open_id"


class FakeSender:
    sender_id = FakeSenderId()


class FakeMessageEvent:
    message = FakeMessage()
    sender = FakeSender()


class FakeMessageData:
    event = FakeMessageEvent()


class FeishuBotParsingTests(unittest.TestCase):
    def test_extract_message_text_from_feishu_json(self) -> None:
        self.assertEqual(_extract_message_text('{"text": " /status "}'), "/status")

    def test_strip_bot_mention(self) -> None:
        self.assertEqual(_strip_bot_mention("@机器人 /run"), "/run")

    def test_extract_card_value_from_dict(self) -> None:
        self.assertEqual(_extract_card_value(FakeData({"action": "apply_draft"}))["action"], "apply_draft")

    def test_extract_card_value_from_json_string(self) -> None:
        self.assertEqual(_extract_card_value(FakeData('{"action": "cancel_draft"}'))["action"], "cancel_draft")

    def test_allowed_message_matches_bound_private_user(self) -> None:
        store = FeishuBindingStore(filepath="/private/tmp/smzdm_test_binding.json")
        store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
        try:
            self.assertTrue(_is_allowed_message(FakeMessageData(), store))
        finally:
            store.clear()

    def test_allowed_message_filters_unbound_user(self) -> None:
        store = FeishuBindingStore(filepath="/private/tmp/smzdm_test_binding.json")
        store.bind("open_id", "ou_other_open_id", "ou_other_open_id", "p2p")
        try:
            self.assertFalse(_is_allowed_message(FakeMessageData(), store))
        finally:
            store.clear()

    def test_message_deduper_filters_duplicate_message_id(self) -> None:
        deduper = MessageDeduper(ttl_seconds=60)

        self.assertTrue(deduper.claim("om_1"))
        self.assertFalse(deduper.claim("om_1"))
        self.assertTrue(deduper.claim("om_2"))

    def test_message_deduper_allows_expired_message_id(self) -> None:
        now = [100.0]
        deduper = MessageDeduper(ttl_seconds=10, time_func=lambda: now[0])

        self.assertTrue(deduper.claim("om_1"))
        now[0] = 111.0
        self.assertTrue(deduper.claim("om_1"))

    def test_handle_message_reacts_and_starts_worker_once_for_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(
                        draft_file=root / "drafts.json",
                        backup_dir=root / "backups",
                        audit_file=root / "audit.jsonl",
                        root=root,
                    ),
                    binding_store=FeishuBindingStore(root / "binding.json"),
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch.object(bot, "_start_reaction_worker") as start_reaction,
                patch.object(bot, "_start_message_worker") as start_worker,
            ):
                data = FakeMessageData()
                bot._handle_message(data)
                bot._handle_message(data)

        start_reaction.assert_called_once_with("om_real_message_id")
        start_worker.assert_called_once_with("你好", data, "", "om_real_message_id")

    def test_add_get_reaction_uses_feishu_reaction_api(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(),
            )
        )

        CreateRequest = Mock()
        create_builder = Mock()
        create_builder.message_id.return_value = create_builder
        create_builder.request_body.return_value = create_builder
        create_builder.build.return_value = "request"
        CreateRequest.builder.return_value = create_builder

        CreateBody = Mock()
        body_builder = Mock()
        body_builder.reaction_type.return_value = body_builder
        body_builder.build.return_value = "body"
        CreateBody.builder.return_value = body_builder

        Emoji = Mock()
        emoji_builder = Mock()
        emoji_builder.emoji_type.return_value = emoji_builder
        emoji_builder.build.return_value = "emoji"
        Emoji.builder.return_value = emoji_builder

        response = Mock()
        response.success.return_value = True
        client = Mock()
        client.im.v1.message_reaction.create.return_value = response

        with (
            patch(
                "smzdm_notice.feishu.bot.get_message_reaction_models",
                return_value=(CreateRequest, CreateBody, Emoji),
            ),
            patch("smzdm_notice.feishu.bot.get_lark_client", return_value=client),
        ):
            bot._add_get_reaction("om_1")

        emoji_builder.emoji_type.assert_called_once_with("Get")
        create_builder.message_id.assert_called_once_with("om_1")
        client.im.v1.message_reaction.create.assert_called_once_with("request")

    def test_add_get_reaction_failure_is_non_blocking(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(),
            )
        )
        with patch("smzdm_notice.feishu.bot.get_message_reaction_models", side_effect=RuntimeError("no permission")):
            bot._add_get_reaction("om_1")

    def test_first_private_message_binds_and_triggers_poll(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            run_once = Mock(return_value=True)
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=run_once,
                )
            )

            with patch("smzdm_notice.feishu.bot.send_text_to"):
                bot._handle_text_command("你好", FakeMessageData())

            binding = binding_store.get()
            self.assertIsNotNone(binding)
            self.assertEqual(binding.receive_id_type, "open_id")
            self.assertEqual(binding.receive_id, "ou_real_open_id")
            run_once.assert_called_once()

    def test_text_command_sends_failure_when_llm_draft_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(
                        draft_file=root / "drafts.json",
                        backup_dir=root / "backups",
                        audit_file=root / "audit.jsonl",
                        root=root,
                    ),
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.feishu.bot.build_message_draft", return_value=None),
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_text_command("拉黑坚果", FakeMessageData())

            send_text.assert_called_once()
            self.assertIn("草案生成失败", send_text.call_args.args[0])

    def test_help_command_uses_registered_help_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(root / "drafts.json", root / "backups", root / "audit.jsonl", root=root),
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.feishu.bot.send_help", return_value=True) as send_help,
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_text_command("/help", FakeMessageData(), reply_to_message_id="om_help")

            send_help.assert_called_once_with(help_markdown(), reply_to_message_id="om_help")
            send_text.assert_not_called()

    def test_message_draft_uses_processing_card_then_updates_same_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            draft_store = DraftStore(root / "drafts.json", root / "backups", root / "audit.jsonl", root=root)
            generated = draft_store.create(
                ConfigDraft(
                    draft_id="generated-draft",
                    target_file="preference.md",
                    title="拉黑坚果",
                    summary="测试",
                    append_text="- 不买坚果",
                    source="test",
                )
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            def update_preview(message_id, draft):
                draft.preview_message_id = message_id
                return True

            with (
                patch("smzdm_notice.feishu.bot.send_draft_processing", return_value="om_processing") as processing,
                patch("smzdm_notice.feishu.bot.build_message_draft", return_value=generated) as build_draft,
                patch("smzdm_notice.feishu.bot.update_draft_preview", side_effect=update_preview) as update_preview_fn,
                patch("smzdm_notice.feishu.bot.send_draft_preview") as send_preview,
            ):
                bot._handle_text_command("拉黑坚果", FakeMessageData(), reply_to_message_id="om_original")

            processing.assert_called_once_with("正在理解偏好/库存修改", reply_to_message_id="om_original")
            build_draft.assert_called_once_with("拉黑坚果", draft_store)
            update_preview_fn.assert_called_once_with("om_processing", generated)
            send_preview.assert_not_called()
            self.assertEqual(draft_store.get_by_preview_message_id("om_processing").draft_id, "generated-draft")

    def test_message_draft_failure_updates_processing_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(root / "drafts.json", root / "backups", root / "audit.jsonl", root=root),
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.feishu.bot.send_draft_processing", return_value="om_processing"),
                patch("smzdm_notice.feishu.bot.build_message_draft", return_value=None),
                patch("smzdm_notice.feishu.bot.update_card_message", return_value=True) as update_card,
                patch("smzdm_notice.feishu.bot.reply_text") as reply_text_fn,
                patch("smzdm_notice.feishu.bot.send_text") as send_text_fn,
            ):
                bot._handle_text_command("拉黑坚果", FakeMessageData(), reply_to_message_id="om_original")

            update_card.assert_called_once()
            self.assertEqual(update_card.call_args.args[0], "om_processing")
            reply_text_fn.assert_not_called()
            send_text_fn.assert_not_called()

    def test_status_command_replies_to_original_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(root / "drafts.json", root / "backups", root / "audit.jsonl", root=root),
                    binding_store=binding_store,
                    status_provider=lambda: "status text",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.feishu.bot.reply_text", return_value=True) as reply_text,
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_text_command("/status", FakeMessageData(), reply_to_message_id="om_status")

            reply_text.assert_called_once_with("om_status", "status text")
            send_text.assert_not_called()

    def test_help_content_includes_every_registered_command(self) -> None:
        content = help_markdown()

        for spec in COMMAND_SPECS:
            self.assertIn(spec.usage, content)
        self.assertIn("基础命令 / Basic", content)
        self.assertIn("搜索关键词 / Search keywords", content)
        self.assertIn("查看运行状态 / Show runtime status", content)

    def test_search_add_command_preserves_inner_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keyword_file = root / "search_keywords.json"
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(root / "drafts.json", root / "backups", root / "audit.jsonl", root=root),
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.smzdm.keywords.config.SEARCH_KEYWORDS_FILE", str(keyword_file)),
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_text_command("/search add AirPods  Pro 2", FakeMessageData())

            self.assertEqual(
                json.loads(keyword_file.read_text(encoding="utf-8")),
                {"keywords": [{"keyword": "AirPods  Pro 2", "max_price": None}]},
            )
            self.assertIn("AirPods  Pro 2", send_text.call_args.args[0])

    def test_search_add_with_price_and_price_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keyword_file = root / "search_keywords.json"
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(root / "drafts.json", root / "backups", root / "audit.jsonl", root=root),
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.smzdm.keywords.config.SEARCH_KEYWORDS_FILE", str(keyword_file)),
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_text_command("/search add AirPods Pro 2 -price 99.9", FakeMessageData())
                bot._handle_text_command("/search price AirPods Pro 2 88", FakeMessageData())

            self.assertEqual(
                json.loads(keyword_file.read_text(encoding="utf-8")),
                {"keywords": [{"keyword": "AirPods Pro 2", "max_price": 88.0}]},
            )
            self.assertIn("max_price: 88", send_text.call_args.args[0])

    def test_search_add_with_unicode_dash_price_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keyword_file = root / "search_keywords.json"
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(root / "drafts.json", root / "backups", root / "audit.jsonl", root=root),
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.smzdm.keywords.config.SEARCH_KEYWORDS_FILE", str(keyword_file)),
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_text_command("/search add AirPods Pro 2 —price 99.9", FakeMessageData())

            self.assertEqual(
                json.loads(keyword_file.read_text(encoding="utf-8")),
                {"keywords": [{"keyword": "AirPods Pro 2", "max_price": 99.9}]},
            )
            self.assertIn("max_price: 99.9", send_text.call_args.args[0])

    def test_search_list_and_remove_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keyword_file = root / "search_keywords.json"
            keyword_file.write_text(
                json.dumps(
                    {
                        "keywords": [
                            {"keyword": "AirPods Pro 2", "max_price": None},
                            {"keyword": "充电宝", "max_price": None},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(root / "drafts.json", root / "backups", root / "audit.jsonl", root=root),
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.smzdm.keywords.config.SEARCH_KEYWORDS_FILE", str(keyword_file)),
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_text_command("/search list", FakeMessageData())
                bot._handle_text_command("/search remove AirPods Pro 2", FakeMessageData())

            self.assertIn("AirPods Pro 2", send_text.call_args_list[0].args[0])
            self.assertEqual(
                json.loads(keyword_file.read_text(encoding="utf-8")),
                {"keywords": [{"keyword": "充电宝", "max_price": None}]},
            )

    def test_search_delete_is_not_supported_and_does_not_fall_through_to_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(root / "drafts.json", root / "backups", root / "audit.jsonl", root=root),
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.feishu.bot.build_message_draft") as build_message_draft,
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_text_command("/search delete AirPods Pro 2", FakeMessageData())

            build_message_draft.assert_not_called()
            self.assertIn("/search remove <keyword>", send_text.call_args.args[0])

    def test_search_price_card_action_clears_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keyword_file = root / "search_keywords.json"
            keyword_file.write_text(
                json.dumps({"keywords": [{"keyword": "AirPods Pro 2", "max_price": 99.9}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(root / "drafts.json", root / "backups", root / "audit.jsonl", root=root),
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )
            data = Mock()
            data.event.action.value = {
                "action": "search_clear_price",
                "search_keyword": "AirPods Pro 2",
            }
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_search"

            with (
                patch("smzdm_notice.smzdm.keywords.config.SEARCH_KEYWORDS_FILE", str(keyword_file)),
                patch("smzdm_notice.feishu.bot.reply_text", return_value=True),
            ):
                bot._handle_card_action(data)

            self.assertEqual(
                json.loads(keyword_file.read_text(encoding="utf-8")),
                {"keywords": [{"keyword": "AirPods Pro 2", "max_price": None}]},
            )

    def test_reply_to_arbitration_preview_revises_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            original = draft_store.create(
                ConfigDraft(
                    draft_id="arbiter-draft",
                    target_file="preference.md",
                    title="仲裁草案",
                    summary="测试",
                    append_text="- 黑名单只精确匹配",
                    source="仲裁建议一键采纳",
                    preview_message_id="om_arbiter",
                )
            )
            revised = draft_store.create(
                ConfigDraft(
                    draft_id="revised-draft",
                    target_file="preference.md",
                    title="修订草案",
                    summary="测试",
                    append_text="- 黑名单按用户原文精确匹配",
                    source="仲裁建议一键采纳",
                )
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            def send_preview(draft, **_kwargs):
                draft.preview_message_id = "om_revised"
                return True

            with (
                patch("smzdm_notice.feishu.bot.build_revision_draft", return_value=revised) as build_revision,
                patch("smzdm_notice.feishu.bot.send_draft_preview", side_effect=send_preview),
                patch("smzdm_notice.feishu.bot.disable_draft_card") as disable_card,
            ):
                bot._handle_text_command("说得更具体一点", FakeMessageData(), parent_id="om_arbiter")

            build_revision.assert_called_once_with("说得更具体一点", original, draft_store)
            self.assertEqual(draft_store.get("arbiter-draft").status, "cancelled")
            self.assertEqual(draft_store.get_by_preview_message_id("om_revised").draft_id, "revised-draft")
            disable_card.assert_called_once_with("om_arbiter", "已生成新的修改预览", original)

    def test_revision_uses_processing_card_for_new_preview_and_disables_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            original = draft_store.create(
                ConfigDraft(
                    draft_id="original-draft",
                    target_file="preference.md",
                    title="原草案",
                    summary="测试",
                    append_text="- 黑名单",
                    source="test",
                    preview_message_id="om_original_preview",
                )
            )
            revised = draft_store.create(
                ConfigDraft(
                    draft_id="revised-draft",
                    target_file="preference.md",
                    title="修订草案",
                    summary="测试",
                    append_text="- 黑名单按原文匹配",
                    source="test",
                )
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=Mock(),
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            def update_preview(message_id, draft):
                draft.preview_message_id = message_id
                return True

            with (
                patch("smzdm_notice.feishu.bot.send_draft_processing", return_value="om_revision_processing"),
                patch("smzdm_notice.feishu.bot.build_revision_draft", return_value=revised) as build_revision,
                patch("smzdm_notice.feishu.bot.update_draft_preview", side_effect=update_preview) as update_preview_fn,
                patch("smzdm_notice.feishu.bot.disable_draft_card") as disable_card,
            ):
                bot._handle_draft_revision("说得更具体一点", original, FakeMessageData(), "om_reply")

            build_revision.assert_called_once_with("说得更具体一点", original, draft_store)
            update_preview_fn.assert_called_once_with("om_revision_processing", revised)
            self.assertEqual(draft_store.get("original-draft").status, "cancelled")
            self.assertEqual(draft_store.get_by_preview_message_id("om_revision_processing").draft_id, "revised-draft")
            disable_card.assert_called_once_with("om_original_preview", "已生成新的修改预览", original)

    def test_reply_to_cancelled_preview_is_not_treated_as_new_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            cancelled = draft_store.create(
                ConfigDraft(
                    draft_id="cancelled-draft",
                    target_file="preference.md",
                    title="旧草案",
                    summary="测试",
                    append_text="- 旧规则",
                    source="test",
                    status="cancelled",
                    preview_message_id="om_cancelled",
                )
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.feishu.bot.build_message_draft") as build_message_draft,
                patch("smzdm_notice.feishu.bot.disable_draft_card") as disable_card,
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_text_command("继续改一下", FakeMessageData(), parent_id="om_cancelled")

            build_message_draft.assert_not_called()
            disable_card.assert_called_once_with("om_cancelled", "该预览已失效，请以最新预览为准", cancelled)
            self.assertIn("该预览已失效", send_text.call_args.args[0])

    def test_reply_to_unknown_preview_is_not_treated_as_new_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=DraftStore(
                        draft_file=root / "drafts.json",
                        backup_dir=root / "backups",
                        audit_file=root / "audit.jsonl",
                        root=root,
                    ),
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.feishu.bot.build_message_draft") as build_message_draft,
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_text_command("继续改一下", FakeMessageData(), parent_id="om_missing")

            build_message_draft.assert_not_called()
            self.assertIn("预览不存在或已失效", send_text.call_args.args[0])

    def test_deal_action_card_action_starts_background_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )
            data = Mock()
            data.event.action.value = {
                "action": "deal_follow",
                "item_title": "婴儿车",
                "article_id": "1001",
            }
            data.event.operator.open_id = "ou_real_open_id"

            with (
                patch.object(bot, "_start_deal_action_worker") as start_worker,
                patch("smzdm_notice.feishu.bot.build_deal_action_draft") as build_draft,
            ):
                response = bot._handle_card_action(data)

            build_draft.assert_not_called()
            start_worker.assert_called_once_with("deal_follow", data.event.action.value, "")
            self.assertIn("正在生成配置修改预览", response.toast.content)

    def test_deal_action_worker_stores_preview_for_reply_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )
            value = {
                "action": "deal_follow",
                "item_title": "婴儿车",
                "article_id": "1001",
            }
            generated = draft_store.create(
                ConfigDraft(
                    draft_id="deal-draft",
                    target_file="preference.md",
                    title="关注婴儿车",
                    summary="测试",
                    append_text="- 关注婴儿车",
                    source="商品卡片快捷操作",
                )
            )

            def send_preview(draft, **_kwargs):
                draft.preview_message_id = "om_deal"
                return True

            with (
                patch("smzdm_notice.feishu.bot.build_deal_action_draft", return_value=generated) as build_draft,
                patch("smzdm_notice.feishu.bot.send_draft_preview", side_effect=send_preview),
            ):
                bot._run_deal_action("deal_follow", value)

            build_draft.assert_called_once_with("deal_follow", value, draft_store)
            draft = draft_store.get_by_preview_message_id("om_deal")
            self.assertIsNotNone(draft)
            self.assertEqual(draft.status, "pending")
            self.assertIn("婴儿车", draft.append_text)

    def test_deal_action_worker_uses_processing_card_when_reply_message_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )
            value = {
                "action": "deal_follow",
                "item_title": "婴儿车",
                "article_id": "1001",
            }
            generated = draft_store.create(
                ConfigDraft(
                    draft_id="deal-draft",
                    target_file="preference.md",
                    title="关注婴儿车",
                    summary="测试",
                    append_text="- 关注婴儿车",
                    source="商品卡片快捷操作",
                )
            )

            def update_preview(message_id, draft):
                draft.preview_message_id = message_id
                return True

            with (
                patch("smzdm_notice.feishu.bot.send_draft_processing", return_value="om_deal_processing"),
                patch("smzdm_notice.feishu.bot.build_deal_action_draft", return_value=generated) as build_draft,
                patch("smzdm_notice.feishu.bot.update_draft_preview", side_effect=update_preview) as update_preview_fn,
                patch("smzdm_notice.feishu.bot.send_draft_preview") as send_preview,
            ):
                bot._run_deal_action("deal_follow", value, reply_to_message_id="om_action")

            build_draft.assert_called_once_with("deal_follow", value, draft_store)
            update_preview_fn.assert_called_once_with("om_deal_processing", generated)
            send_preview.assert_not_called()
            self.assertEqual(draft_store.get_by_preview_message_id("om_deal_processing").draft_id, "deal-draft")

    def test_deal_action_worker_failure_sends_text_without_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )
            value = {
                "action": "deal_follow",
                "item_title": "婴儿车",
                "article_id": "1001",
            }

            with (
                patch("smzdm_notice.feishu.bot.build_deal_action_draft", return_value=None),
                patch("smzdm_notice.feishu.bot.send_draft_preview") as send_preview,
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._run_deal_action("deal_follow", value)

            send_preview.assert_not_called()
            self.assertIn("无法生成配置修改预览", send_text.call_args.args[0])

    def test_revision_preview_failure_keeps_original_draft_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            original = draft_store.create(
                ConfigDraft(
                    draft_id="original-draft",
                    target_file="preference.md",
                    title="原草案",
                    summary="测试",
                    append_text="- 黑名单只精确匹配",
                    source="仲裁建议一键采纳",
                    preview_message_id="om_original",
                )
            )
            revised = draft_store.create(
                ConfigDraft(
                    draft_id="revised-draft",
                    target_file="preference.md",
                    title="修订草案",
                    summary="测试",
                    append_text="- 黑名单按用户原文精确匹配",
                    source="仲裁建议一键采纳",
                )
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=Mock(),
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )

            with (
                patch("smzdm_notice.feishu.bot.build_revision_draft", return_value=revised),
                patch("smzdm_notice.feishu.bot.send_draft_preview", return_value=False),
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
            ):
                bot._handle_draft_revision("再具体一点", original, FakeMessageData())

            self.assertEqual(draft_store.get("original-draft").status, "pending")
            self.assertEqual(draft_store.get("revised-draft").status, "cancelled")
            self.assertIn("原草案仍保留", send_text.call_args.args[0])

    def test_ignore_arbitration_cancels_pending_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            draft_store.create(
                ConfigDraft(
                    draft_id="arbiter-draft",
                    target_file="preference.md",
                    title="仲裁草案",
                    summary="测试",
                    append_text="- 黑名单只精确匹配",
                    source="仲裁建议一键采纳",
                    metadata={
                        "card_kind": "arbitration",
                        "arbitration_card": {
                            "sent_at": "2026-05-20 10:00",
                            "diff_text": "差异商品 A",
                            "chosen": "B",
                            "reason": "B 更准确",
                            "analysis": "A 过度扩展黑名单。",
                            "suggestion": "黑名单只按字面精确匹配。",
                        },
                    },
                )
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )
            data = Mock()
            data.event.action.value = {
                "action": "ignore_arbitration",
                "draft_id": "arbiter-draft",
                "card_kind": "arbitration",
            }
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_arbiter"

            with (
                patch("smzdm_notice.feishu.bot.reply_text", return_value=True),
                patch("smzdm_notice.feishu.bot.send_text"),
                patch("smzdm_notice.feishu.bot.disable_draft_card") as disable_card,
            ):
                response = bot._handle_card_action(data)

            self.assertEqual(draft_store.get("arbiter-draft").status, "cancelled")
            disable_card.assert_not_called()
            self.assertEqual(response.card.type, "raw")
            self.assertIn("仲裁分析", response.card.data["header"]["title"]["content"])
            elements = response.card.data["elements"]
            self.assertNotIn("action", {element.get("tag") for element in elements})
            markdown = "\n".join(element.get("content", "") for element in elements)
            self.assertIn("已忽略", markdown)
            self.assertIn("A 过度扩展黑名单。", markdown)

    def test_apply_expired_draft_disables_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            draft_store.create(
                ConfigDraft(
                    draft_id="expired-draft",
                    target_file="preference.md",
                    title="过期草案",
                    summary="测试",
                    append_text="- 旧规则",
                    source="test",
                    created_at=100.0,
                    preview_message_id="om_expired",
                )
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )
            data = Mock()
            data.event.action.value = {"action": "apply_draft", "draft_id": "expired-draft"}
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_expired"

            with (
                patch("smzdm_notice.preferences.models.time.time", return_value=100.0 + 24 * 60 * 60 + 1),
                patch("smzdm_notice.feishu.bot.reply_text", return_value=True) as reply_text,
                patch("smzdm_notice.feishu.bot.send_text") as send_text,
                patch("smzdm_notice.feishu.bot.disable_draft_card") as disable_card,
            ):
                response = bot._handle_card_action(data)

            self.assertEqual(draft_store.get("expired-draft").status, "cancelled")
            self.assertIn("该预览已失效", reply_text.call_args.args[1])
            send_text.assert_not_called()
            disable_card.assert_not_called()
            self.assertEqual(response.card.type, "raw")
            elements = response.card.data["elements"]
            self.assertNotIn("action", {element.get("tag") for element in elements})
            self.assertIn("预览已失效", elements[0]["content"])

    def test_apply_expired_arbitration_draft_keeps_arbitration_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding_store = FeishuBindingStore(root / "binding.json")
            binding_store.bind("open_id", "ou_real_open_id", "ou_real_open_id", "p2p")
            draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            draft_store.create(
                ConfigDraft(
                    draft_id="expired-arbiter",
                    target_file="preference.md",
                    title="仲裁草案",
                    summary="测试",
                    append_text="- 旧规则",
                    source="仲裁建议一键采纳",
                    created_at=100.0,
                    preview_message_id="om_expired_arbiter",
                    metadata={
                        "card_kind": "arbitration",
                        "arbitration_card": {
                            "sent_at": "2026-05-20 10:00",
                            "diff_text": "差异商品 A",
                            "chosen": "B",
                            "reason": "B 更准确",
                            "analysis": "A 过度扩展黑名单。",
                            "suggestion": "黑名单只按字面精确匹配。",
                        },
                    },
                )
            )
            bot = FeishuInteractiveBot(
                BotRuntime(
                    draft_store=draft_store,
                    binding_store=binding_store,
                    status_provider=lambda: "status",
                    run_once=Mock(return_value=True),
                )
            )
            data = Mock()
            data.event.action.value = {
                "action": "apply_draft",
                "draft_id": "expired-arbiter",
                "card_kind": "arbitration",
            }
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_expired_arbiter"

            with (
                patch(
                    "smzdm_notice.preferences.models.time.time",
                    return_value=100.0 + 24 * 60 * 60 + 1,
                ),
                patch("smzdm_notice.feishu.bot.reply_text", return_value=True),
                patch("smzdm_notice.feishu.bot.send_text"),
                patch("smzdm_notice.feishu.bot.disable_draft_card") as disable_card,
            ):
                response = bot._handle_card_action(data)

            self.assertEqual(draft_store.get("expired-arbiter").status, "cancelled")
            disable_card.assert_not_called()
            self.assertIn("仲裁分析", response.card.data["header"]["title"]["content"])
            elements = response.card.data["elements"]
            self.assertNotIn("action", {element.get("tag") for element in elements})
            markdown = "\n".join(element.get("content", "") for element in elements)
            self.assertIn("A 过度扩展黑名单。", markdown)
            self.assertIn("预览已失效", markdown)


if __name__ == "__main__":
    unittest.main()
