from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from smzdm_notice.feishu.binding import FeishuBindingStore
from smzdm_notice.feishu.bot import (
    MODEL_CARD_FORM_STATE_LIMIT,
    BotRuntime,
    DraftProcessingMessage,
    FeishuInteractiveBot,
    MessageDeduper,
    _extract_card_value,
    _extract_message_text,
    _is_allowed_message,
    _run_model_test,
    _strip_bot_mention,
)
from smzdm_notice.feishu.commands import COMMAND_SPECS, help_markdown
from smzdm_notice.feishu.notifier import build_model_management_card
from smzdm_notice.llm.routing import ResolvedLLMConfig, RoutingSnapshot
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


def _model_test_config(model_id: str = "deepseek-chat", temperature: float | None = 0.7) -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        agent="filter",
        connection="deepseek",
        connection_label="DeepSeek",
        provider="openai_compatible",
        base_url="https://api.deepseek.com/v1",
        api_key_env="LLM_DEEPSEEK_API_KEY",
        api_key="key",
        model_id=model_id,
        timeout_seconds=300,
        max_retries=2,
        temperature=temperature,
        response_format={"type": "json_object"},
        extra_body={"do_sample": False},
    )


def _model_card_state() -> dict:
    return {
        "version": 1,
        "source": "file",
        "defaults": {"connection": "deepseek", "model_id": "deepseek-chat", "temperature": None},
        "connections": [
            {
                "name": "deepseek",
                "label": "DeepSeek",
                "provider": "openai_compatible",
                "base_url_host": "api.deepseek.com",
                "key_configured": True,
            },
            {
                "name": "glm",
                "label": "GLM",
                "provider": "openai_compatible",
                "base_url_host": "open.bigmodel.cn",
                "key_configured": True,
            },
        ],
        "agents": [
            {
                "name": "filter",
                "connection": "deepseek",
                "connection_label": "DeepSeek",
                "model_id": "deepseek-chat",
                "temperature": 0.3,
                "base_url_host": "api.deepseek.com",
                "inherits_connection": True,
                "inherits_model": True,
            },
            {
                "name": "arbiter",
                "connection": "deepseek",
                "connection_label": "DeepSeek",
                "model_id": "deepseek-chat",
                "temperature": 0.0,
                "base_url_host": "api.deepseek.com",
                "inherits_connection": True,
                "inherits_model": True,
            },
            {
                "name": "draft",
                "connection": "deepseek",
                "connection_label": "DeepSeek",
                "model_id": "deepseek-chat",
                "temperature": 0.0,
                "base_url_host": "api.deepseek.com",
                "inherits_connection": True,
                "inherits_model": True,
            },
        ],
    }


def _routing_snapshot(agent: str = "filter", connection: str = "glm", model_id: str = "glm-4-flash") -> RoutingSnapshot:
    return RoutingSnapshot(
        raw={
            "connections": {
                connection: {
                    "provider": "openai_compatible",
                    "label": "GLM",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "api_key_env": "LLM_GLM_API_KEY",
                }
            },
            "defaults": {
                "connection": connection,
                "model_id": model_id,
                "timeout_seconds": 300,
                "max_retries": 2,
                "request": {"extra_body": {}},
            },
            "agents": {agent: {"connection": connection, "model_id": model_id}},
        },
        version=1,
        path=Path("llm_models.json"),
        source="file",
    )


class FeishuBotParsingTests(unittest.TestCase):
    def test_extract_message_text_from_feishu_json(self) -> None:
        self.assertEqual(_extract_message_text('{"text": " /status "}'), "/status")

    def test_strip_bot_mention(self) -> None:
        self.assertEqual(_strip_bot_mention("@机器人 /run"), "/run")

    def test_extract_card_value_from_dict(self) -> None:
        self.assertEqual(_extract_card_value(FakeData({"action": "apply_draft"}))["action"], "apply_draft")

    def test_extract_card_value_from_json_string(self) -> None:
        self.assertEqual(_extract_card_value(FakeData('{"action": "cancel_draft"}'))["action"], "cancel_draft")

    def test_extract_card_value_merges_form_values(self) -> None:
        data = Mock()
        data.event.action.value = {"action": "model_apply_connection_model"}
        data.event.action.form_value = {
            "target": {"value": "arbiter"},
            "connection": {"value": "glm"},
            "model_id": {"value": "glm-4-flash"},
        }

        value = _extract_card_value(data)

        self.assertEqual(value["action"], "model_apply_connection_model")
        self.assertEqual(value["form_value"]["target"]["value"], "arbiter")
        self.assertEqual(value["target"]["value"], "arbiter")

    def test_extract_card_value_reads_component_callback_fields(self) -> None:
        data = Mock()
        data.event.action.value = {"field": "connection"}
        data.event.action.name = None
        data.event.action.option = "glm"
        data.event.action.tag = "select_static"

        value = _extract_card_value(data)

        self.assertEqual(value["value"]["field"], "connection")
        self.assertEqual(value["option"], "glm")
        self.assertEqual(value["tag"], "select_static")

    def test_extract_card_value_preserves_empty_input_value(self) -> None:
        data = Mock()
        data.event.action.value = {}
        data.event.action.name = "model_id"
        data.event.action.option = None
        data.event.action.input_value = ""
        data.event.action.tag = "input"

        value = _extract_card_value(data)

        self.assertEqual(value["name"], "model_id")
        self.assertIn("input_value", value)
        self.assertEqual(value["input_value"], "")

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

    def test_message_draft_sends_new_preview_when_processing_thread_is_still_running(self) -> None:
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

            with (
                patch.object(
                    bot,
                    "_start_draft_processing",
                    return_value=DraftProcessingMessage(message_id="om_processing"),
                ),
                patch.object(bot, "_stop_draft_processing", return_value=False),
                patch("smzdm_notice.feishu.bot.build_message_draft", return_value=generated),
                patch("smzdm_notice.feishu.bot.update_draft_preview") as update_preview,
                patch("smzdm_notice.feishu.bot.send_draft_preview", return_value=True) as send_preview,
            ):
                bot._handle_text_command("拉黑坚果", FakeMessageData(), reply_to_message_id="om_original")

            update_preview.assert_not_called()
            send_preview.assert_called_once_with(generated, reply_to_message_id="om_original")

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
        self.assertIn("`/model`", content)
        self.assertIn("`/model status`", content)
        self.assertNotIn("/model use", content)

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


class FeishuModelCommandTests(unittest.TestCase):
    def test_model_management_card_defaults_to_default_target(self) -> None:
        card = build_model_management_card(_model_card_state())
        serialized = json.dumps(card, ensure_ascii=False)

        self.assertIn("initial_option", serialized)
        self.assertIn("default_value", serialized)
        self.assertIn("select_static", serialized)
        self.assertIn("input", serialized)
        self.assertIn("切换 model_id", serialized)
        self.assertNotIn("切换 connection + model", serialized)
        # elements: [markdown, hr, hint_markdown, select_action, input_action, ...]
        select_actions = card["elements"][3]["actions"]
        self.assertEqual(select_actions[0]["value"], {"field": "target"})
        self.assertEqual(select_actions[1]["value"], {"field": "connection"})
        self.assertEqual(select_actions[0]["initial_option"], "default")
        self.assertEqual(select_actions[1]["initial_option"], "deepseek")
        self.assertEqual(select_actions[0]["options"][1]["value"], "filter")
        self.assertEqual(select_actions[1]["options"][0]["value"], "deepseek")
        input_actions = card["elements"][4]["actions"]
        self.assertEqual(input_actions[0]["default_value"], "deepseek-chat")

    def test_model_management_card_builds_target_options_from_agents_state(self) -> None:
        state = _model_card_state()
        state["agents"].append(
            {
                "name": "reviewer",
                "connection": "deepseek",
                "connection_label": "DeepSeek",
                "model_id": "deepseek-chat",
                "temperature": 0.0,
                "base_url_host": "api.deepseek.com",
                "inherits_connection": True,
                "inherits_model": True,
            }
        )

        card = build_model_management_card(state)

        target_options = card["elements"][3]["actions"][0]["options"]
        self.assertEqual([option["value"] for option in target_options], ["default", "filter", "arbiter", "draft", "reviewer"])

    def test_model_command_replies_with_management_card_and_ignores_subcommands(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
            patch("smzdm_notice.feishu.bot.reply_card", return_value="om_model_card") as reply_card,
            patch("smzdm_notice.feishu.bot.llm_routing.use_default_model") as use_default,
        ):
            handled = bot._handle_slash_command("/model use deepseek-reasoner", reply_to_message_id="om_model")

        self.assertTrue(handled)
        reply_card.assert_called_once()
        self.assertEqual(reply_card.call_args.args[0], "om_model")
        self.assertEqual(reply_card.call_args.args[1]["header"]["title"]["content"], "LLM 模型路由")
        use_default.assert_not_called()

    def test_model_status_command_replies_with_routing_status(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.format_status", return_value="LLM status") as format_status,
            patch("smzdm_notice.feishu.bot.reply_text", return_value=True) as reply_text,
            patch("smzdm_notice.feishu.bot.reply_card") as reply_card,
        ):
            handled = bot._handle_slash_command("/model status", reply_to_message_id="om_model")

        self.assertTrue(handled)
        format_status.assert_called_once_with()
        reply_text.assert_called_once_with("om_model", "LLM status")
        reply_card.assert_not_called()

    def test_slash_command_key_is_case_insensitive(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status text",
                run_once=Mock(return_value=True),
            )
        )

        with patch("smzdm_notice.feishu.bot.reply_text", return_value=True) as reply_text:
            handled = bot._handle_slash_command("/Status", reply_to_message_id="om_status")

        self.assertTrue(handled)
        reply_text.assert_called_once_with("om_status", "status text")

    def test_model_status_command_key_is_case_insensitive(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.format_status", return_value="LLM status") as format_status,
            patch("smzdm_notice.feishu.bot.reply_text", return_value=True) as reply_text,
        ):
            handled = bot._handle_slash_command("/MODEL status", reply_to_message_id="om_model")

        self.assertTrue(handled)
        format_status.assert_called_once_with()
        reply_text.assert_called_once_with("om_model", "LLM status")

    def test_search_command_preserves_original_argument_case(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with patch.object(bot, "_handle_search_command") as handle_search:
            handled = bot._handle_slash_command("/Search add FooBar -price 12", reply_to_message_id="om_search")

        self.assertTrue(handled)
        handle_search.assert_called_once_with("/Search add FooBar -price 12", "/search add", "om_search")

    def test_model_command_fallback_mentions_card_send_failure(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
            patch("smzdm_notice.feishu.bot.reply_card", return_value=None),
            patch("smzdm_notice.feishu.bot.llm_routing.format_status", return_value="LLM status"),
            patch("smzdm_notice.feishu.bot.reply_text", return_value=True) as reply_text,
            patch("smzdm_notice.feishu.bot.send_text") as send_text,
        ):
            handled = bot._handle_slash_command("/model", reply_to_message_id="om_model")

        self.assertTrue(handled)
        reply_text.assert_called_once()
        self.assertIn("模型管理卡片发送失败", reply_text.call_args.args[1])
        self.assertIn("LLM status", reply_text.call_args.args[1])
        send_text.assert_not_called()

    def test_model_card_form_change_without_action_is_ignored(self) -> None:
        binding_store = Mock()
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )
        data = Mock()
        data.event.action.value = {}
        data.event.action.form_value = {"target": {"value": "filter"}}
        data.event.operator.open_id = "ou_real_open_id"
        data.event.context.open_message_id = "om_card"

        with (
            patch.object(bot, "_dispatch_card_action") as dispatch,
            patch("smzdm_notice.feishu.bot.reply_text") as reply_text,
            patch("smzdm_notice.feishu.bot.send_text") as send_text,
        ):
            result = bot._handle_card_action(data)

        self.assertIsNone(result)
        dispatch.assert_not_called()
        binding_store.is_bound_operator.assert_not_called()
        reply_text.assert_not_called()
        send_text.assert_not_called()

    def test_model_card_form_change_from_unbound_user_does_not_cache(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = False
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )
        data = Mock()
        data.event.action.value = {"field": "connection"}
        data.event.action.name = None
        data.event.action.option = "glm"
        data.event.action.input_value = None
        data.event.action.tag = "select_static"
        data.event.operator.open_id = "ou_real_open_id"
        data.event.context.open_message_id = "om_card"

        with patch.object(bot, "_dispatch_card_action") as dispatch:
            result = bot._handle_card_action(data)

        self.assertIsNone(result)
        dispatch.assert_not_called()
        binding_store.is_bound_operator.assert_called_once_with("ou_real_open_id")
        self.assertEqual(bot._model_card_form_state, {})

    def test_model_card_button_uses_cached_component_values(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        def component_event(name: str, option: str = "", input_value: str = ""):
            data = Mock()
            data.event.action.value = {"field": name} if option else {}
            data.event.action.name = None if option else name
            data.event.action.option = option
            data.event.action.input_value = input_value
            data.event.action.tag = "input" if input_value else "select_static"
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_card"
            return data

        button_data = Mock()
        button_data.event.action.value = {"action": "model_apply_connection_model"}
        button_data.event.action.name = None
        button_data.event.action.option = None
        button_data.event.action.input_value = None
        button_data.event.action.tag = "button"
        button_data.event.operator.open_id = "ou_real_open_id"
        button_data.event.context.open_message_id = "om_card"

        resolved_arbiter = _model_test_config()

        with (
            patch(
                "smzdm_notice.feishu.bot.llm_routing.use_agent_model",
                return_value=_routing_snapshot(agent="arbiter", connection="glm", model_id="glm-4-flash"),
            ) as use_agent,
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
            patch("smzdm_notice.feishu.bot.llm_routing.resolve", return_value=resolved_arbiter),
            patch.dict("os.environ", {"LLM_GLM_API_KEY": "key"}),
        ):
            # Target change returns a card (auto-populate), not None
            target_result = bot._handle_card_action(component_event("target", option="arbiter"))
            self.assertIsNotNone(target_result)
            connection_result = bot._handle_card_action(component_event("connection", option="glm"))
            self.assertIsNotNone(connection_result)
            self.assertIsNone(bot._handle_card_action(component_event("model_id", input_value="glm-4-flash")))
            bot._handle_card_action(button_data)
            bot._handle_card_action(button_data)

        self.assertEqual(use_agent.call_count, 2)
        use_agent.assert_called_with("arbiter", "glm-4-flash", connection="glm")
        self.assertIn("om_card:ou_real_open_id", bot._model_card_form_state)

    def test_model_card_empty_input_clears_cached_value_before_apply(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        def input_event(value: str):
            data = Mock()
            data.event.action.value = {}
            data.event.action.name = "model_id"
            data.event.action.option = None
            data.event.action.input_value = value
            data.event.action.tag = "input"
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_card"
            return data

        button_data = Mock()
        button_data.event.action.value = {"action": "model_apply_model", "target": "filter"}
        button_data.event.action.name = None
        button_data.event.action.option = None
        button_data.event.action.input_value = None
        button_data.event.action.tag = "button"
        button_data.event.operator.open_id = "ou_real_open_id"
        button_data.event.context.open_message_id = "om_card"

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.use_agent_model") as use_agent,
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            self.assertIsNone(bot._handle_card_action(input_event("old-model")))
            self.assertIsNone(bot._handle_card_action(input_event("")))
            bot._handle_card_action(button_data)

        use_agent.assert_not_called()
        self.assertEqual(bot._model_card_form_state["om_card:ou_real_open_id"]["model_id"], "")
        self.assertNotIn("model_id_manual", bot._model_card_form_state["om_card:ou_real_open_id"])

    def test_model_card_empty_input_clears_manual_flag_before_target_change(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        def input_event(name: str, value: str):
            data = Mock()
            data.event.action.value = {}
            data.event.action.name = name
            data.event.action.option = None
            data.event.action.input_value = value
            data.event.action.tag = "input"
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_card"
            return data

        target_data = Mock()
        target_data.event.action.value = {"field": "target"}
        target_data.event.action.name = None
        target_data.event.action.option = "filter"
        target_data.event.action.input_value = None
        target_data.event.action.tag = "select_static"
        target_data.event.operator.open_id = "ou_real_open_id"
        target_data.event.context.open_message_id = "om_card"

        latest_filter = _model_test_config(model_id="latest-filter-model")

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.resolve", return_value=latest_filter),
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            self.assertIsNone(bot._handle_card_action(input_event("model_id", "custom-model")))
            self.assertIsNone(bot._handle_card_action(input_event("model_id", "")))
            result = bot._handle_card_action(target_data)

        self.assertIsNotNone(result)
        card_json = json.dumps(result.card.data, ensure_ascii=False)
        self.assertIn("latest-filter-model", card_json)
        self.assertNotIn("custom-model", card_json)
        self.assertNotIn("model_id_manual", bot._model_card_form_state["om_card:ou_real_open_id"])

    def test_model_card_empty_temperature_clears_manual_flag_before_target_change(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        def input_event(value: str):
            data = Mock()
            data.event.action.value = {}
            data.event.action.name = "temperature"
            data.event.action.option = None
            data.event.action.input_value = value
            data.event.action.tag = "input"
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_card"
            return data

        target_data = Mock()
        target_data.event.action.value = {"field": "target"}
        target_data.event.action.name = None
        target_data.event.action.option = "filter"
        target_data.event.action.input_value = None
        target_data.event.action.tag = "select_static"
        target_data.event.operator.open_id = "ou_real_open_id"
        target_data.event.context.open_message_id = "om_card"

        latest_filter = _model_test_config(temperature=0.42)

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.resolve", return_value=latest_filter),
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            self.assertIsNone(bot._handle_card_action(input_event("0.9")))
            self.assertIsNone(bot._handle_card_action(input_event("")))
            result = bot._handle_card_action(target_data)

        self.assertIsNotNone(result)
        card_json = json.dumps(result.card.data, ensure_ascii=False)
        self.assertIn("0.42", card_json)
        self.assertNotIn("temperature_manual", bot._model_card_form_state["om_card:ou_real_open_id"])

    def test_model_card_form_state_is_bounded(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        def input_event(index: int):
            data = Mock()
            data.event.action.value = {}
            data.event.action.name = "model_id"
            data.event.action.option = None
            data.event.action.input_value = f"model-{index}"
            data.event.action.tag = "input"
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = f"om_card_{index}"
            return data

        for index in range(MODEL_CARD_FORM_STATE_LIMIT + 1):
            self.assertIsNone(bot._handle_card_action(input_event(index)))

        self.assertEqual(len(bot._model_card_form_state), MODEL_CARD_FORM_STATE_LIMIT)
        self.assertNotIn("om_card_0:ou_real_open_id", bot._model_card_form_state)
        self.assertIn(f"om_card_{MODEL_CARD_FORM_STATE_LIMIT}:ou_real_open_id", bot._model_card_form_state)

    def test_model_refresh_clears_cached_state_and_prefills_default(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        def component_event(name: str, option: str = "", input_value: str = ""):
            data = Mock()
            data.event.action.value = {"field": name} if option else {}
            data.event.action.name = None if option else name
            data.event.action.option = option
            data.event.action.input_value = input_value
            data.event.action.tag = "input" if input_value else "select_static"
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_card"
            return data

        refresh_data = Mock()
        refresh_data.event.action.value = {"action": "model_refresh"}
        refresh_data.event.action.name = None
        refresh_data.event.action.option = None
        refresh_data.event.action.input_value = None
        refresh_data.event.action.tag = "button"
        refresh_data.event.operator.open_id = "ou_real_open_id"
        refresh_data.event.context.open_message_id = "om_card"

        with patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()):
            self.assertIsNotNone(bot._handle_card_action(component_event("connection", option="glm")))
            self.assertIsNone(bot._handle_card_action(component_event("model_id", input_value="glm-4-flash")))
            result = bot._handle_card_action(refresh_data)

        self.assertNotIn("om_card:ou_real_open_id", bot._model_card_form_state)
        self.assertIsNotNone(result)
        card_data = result.card.data
        select_actions = card_data["elements"][3]["actions"]
        input_actions = card_data["elements"][4]["actions"]
        self.assertEqual(select_actions[0]["initial_option"], "default")
        self.assertEqual(select_actions[1]["initial_option"], "deepseek")
        self.assertEqual(input_actions[0]["default_value"], "deepseek-chat")

    def test_target_change_after_refresh_uses_latest_agent_config(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        def component_event(name: str, option: str = "", input_value: str = ""):
            data = Mock()
            data.event.action.value = {"field": name} if option else {}
            data.event.action.name = None if option else name
            data.event.action.option = option
            data.event.action.input_value = input_value
            data.event.action.tag = "input" if input_value else "select_static"
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_card"
            return data

        refresh_data = Mock()
        refresh_data.event.action.value = {"action": "model_refresh"}
        refresh_data.event.action.name = None
        refresh_data.event.action.option = None
        refresh_data.event.action.input_value = None
        refresh_data.event.action.tag = "button"
        refresh_data.event.operator.open_id = "ou_real_open_id"
        refresh_data.event.context.open_message_id = "om_card"

        latest_filter = _model_test_config(model_id="deepseek-reasoner")

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
            patch("smzdm_notice.feishu.bot.llm_routing.resolve", return_value=latest_filter),
        ):
            self.assertIsNone(bot._handle_card_action(component_event("model_id", input_value="stale-model")))
            bot._handle_card_action(refresh_data)
            result = bot._handle_card_action(component_event("target", option="filter"))

        self.assertIsNotNone(result)
        card_json = json.dumps(result.card.data, ensure_ascii=False)
        self.assertIn("deepseek-reasoner", card_json)
        self.assertNotIn("stale-model", card_json)

    def test_model_card_action_applies_agent_connection_and_model(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with (
            patch(
                "smzdm_notice.feishu.bot.llm_routing.use_agent_model",
                return_value=_routing_snapshot(agent="arbiter", connection="glm", model_id="glm-4-flash"),
            ) as use_agent,
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
            patch.dict("os.environ", {"LLM_GLM_API_KEY": "key"}),
        ):
            result = bot._dispatch_card_action(
                "model_apply_connection_model",
                {
                    "target": "arbiter",
                    "connection": "glm",
                    "model_id": "glm-4-flash",
                },
                "ou_real_open_id",
                "om_card",
            )

        use_agent.assert_called_once_with("arbiter", "glm-4-flash", connection="glm")
        self.assertIn("已更新", result.message)
        self.assertIn("arbiter: glm/glm-4-flash", result.message)
        self.assertEqual(result.response_card["header"]["title"]["content"], "LLM 模型路由")

    def test_model_card_reset_clears_cached_manual_state(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )
        cache_key = "om_card:ou_real_open_id"
        bot._model_card_form_state[cache_key] = {
            "target": "filter",
            "connection": "glm",
            "connection_manual": True,
            "model_id": "stale-model",
            "model_id_manual": True,
            "temperature": "0.9",
            "temperature_manual": True,
        }
        data = Mock()
        data.event.action.value = {"action": "model_reset_agent", "target": "filter"}
        data.event.action.name = None
        data.event.action.option = None
        data.event.action.input_value = None
        data.event.action.tag = "button"
        data.event.operator.open_id = "ou_real_open_id"
        data.event.context.open_message_id = "om_card"

        with (
            patch(
                "smzdm_notice.feishu.bot.llm_routing.reset_agent",
                return_value=_routing_snapshot(agent="filter", connection="deepseek", model_id="deepseek-chat"),
            ) as reset_agent,
            patch.dict("os.environ", {"LLM_GLM_API_KEY": "key"}),
        ):
            result = bot._handle_card_action(data)

        reset_agent.assert_called_once_with("filter")
        self.assertNotIn(cache_key, bot._model_card_form_state)
        self.assertIsNotNone(result)
        card_json = json.dumps(result.card.data, ensure_ascii=False)
        self.assertIn("deepseek-chat", card_json)
        self.assertNotIn("stale-model", card_json)

    def test_model_card_unexpected_error_keeps_management_card(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with (
            patch("smzdm_notice.feishu.bot._apply_model_card_action", side_effect=RuntimeError("boom")),
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            result = bot._dispatch_card_action(
                "model_apply_model",
                {"target": "filter", "model_id": "deepseek-reasoner"},
                "ou_real_open_id",
                "om_card",
            )

        self.assertEqual(result.message, "处理消息时遇到内部错误，请稍后重试。")
        self.assertIsNotNone(result.response_card)
        self.assertEqual(result.response_card["header"]["title"]["content"], "LLM 模型路由")

    def test_model_card_action_reads_form_value_from_real_card_payload(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )
        data = Mock()
        data.event.action.value = {"action": "model_apply_connection_model"}
        data.event.action.form_value = {
            "target": {"value": "arbiter"},
            "connection": {"value": "glm"},
            "model_id": {"value": "glm-4-flash"},
        }
        data.event.operator.open_id = "ou_real_open_id"
        data.event.context.open_message_id = "om_card"

        with (
            patch(
                "smzdm_notice.feishu.bot.llm_routing.use_agent_model",
                return_value=_routing_snapshot(agent="arbiter", connection="glm", model_id="glm-4-flash"),
            ) as use_agent,
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
            patch.dict("os.environ", {"LLM_GLM_API_KEY": "key"}),
        ):
            bot._handle_card_action(data)

        use_agent.assert_called_once_with("arbiter", "glm-4-flash", connection="glm")

    def test_model_card_action_sets_default_temperature(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.set_default_temperature", return_value=object()) as set_temp,
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            result = bot._dispatch_card_action(
                "model_set_temperature",
                {"target": "default", "temperature": "0.6"},
                "ou_real_open_id",
                "om_card",
            )

        set_temp.assert_called_once_with(0.6)
        self.assertIn("已更新", result.message)
        self.assertIn("default temperature=0.6", result.message)

    def test_model_card_apply_connection_model_requires_model_id(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.use_agent_model") as use_agent,
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            result = bot._dispatch_card_action(
                "model_apply_connection_model",
                {"target": "filter", "connection": "glm"},
                "ou_real_open_id",
                "om_card",
            )

        use_agent.assert_not_called()
        self.assertIn("请输入 model_id 后再应用", result.message)

    def test_model_card_apply_connection_model_requires_connection(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.use_agent_model") as use_agent,
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            result = bot._dispatch_card_action(
                "model_apply_connection_model",
                {"target": "filter", "model_id": "deepseek-reasoner"},
                "ou_real_open_id",
                "om_card",
            )

        use_agent.assert_not_called()
        self.assertIn("请选择 connection 后再应用", result.message)

    def test_model_card_test_uses_selected_connection_model_without_writing(self) -> None:
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=Mock(),
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        with (
            patch(
                "smzdm_notice.feishu.bot.llm_routing.test_config_for_connection",
                return_value=_model_test_config(),
            ) as test_config,
            patch("smzdm_notice.feishu.bot._run_model_test", return_value="OK") as run_test,
            patch("smzdm_notice.feishu.bot.llm_routing.use_agent_model") as use_agent,
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            result = bot._dispatch_card_action(
                "model_test",
                {"target": "filter", "connection": "glm", "model_id": "glm-4-flash"},
                "ou_real_open_id",
                "om_card",
            )

        test_config.assert_called_once_with("glm", "glm-4-flash")
        run_test.assert_called_once()
        use_agent.assert_not_called()
        self.assertEqual(result.message, "OK")

    def test_model_test_uses_resolved_request_options(self) -> None:
        captured = {}
        response = Mock()
        response.choices = [Mock(message=Mock(content='{"ok": true}'))]
        fake_client = Mock()
        fake_client.chat.completions.create.return_value = response

        def create_completion(**kwargs):
            captured.update(kwargs)
            return response

        fake_client.chat.completions.create.side_effect = create_completion

        with patch("smzdm_notice.feishu.bot.get_client_for_config", return_value=fake_client):
            message = _run_model_test(_model_test_config())

        self.assertIn("测试成功", message)
        self.assertEqual(captured["model"], "deepseek-chat")
        self.assertEqual(captured["temperature"], 0.7)
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(captured["extra_body"], {"do_sample": False})
        self.assertEqual(captured["timeout"], 30)

    def test_model_management_card_pre_populates_from_form_state(self) -> None:
        form_state = {
            "target": "filter",
            "connection": "deepseek",
            "model_id": "deepseek-chat",
            "temperature": "0.3",
        }
        card = build_model_management_card(_model_card_state(), form_state=form_state)
        serialized = json.dumps(card, ensure_ascii=False)

        self.assertIn("initial_option", serialized)
        self.assertIn("default_value", serialized)
        self.assertIn("切换 model_id", serialized)
        self.assertNotIn("切换 connection + model", serialized)
        self.assertNotIn("仅切换模型", serialized)
        self.assertNotIn("应用全部设置", serialized)

        # elements: [markdown, hr, hint_markdown, select_action, input_action, ...]
        select_actions = card["elements"][3]["actions"]
        self.assertEqual(select_actions[0]["initial_option"], "filter")
        self.assertEqual(select_actions[1]["initial_option"], "deepseek")

        input_actions = card["elements"][4]["actions"]
        self.assertEqual(input_actions[0]["default_value"], "deepseek-chat")
        self.assertEqual(input_actions[1]["default_value"], "0.3")

    def test_model_management_card_preserves_empty_input_defaults(self) -> None:
        form_state = {
            "target": "filter",
            "connection": "deepseek",
            "model_id": "",
            "temperature": "",
        }
        card = build_model_management_card(_model_card_state(), form_state=form_state)

        input_actions = card["elements"][4]["actions"]
        self.assertEqual(input_actions[0]["default_value"], "")
        self.assertEqual(input_actions[1]["default_value"], "")

    def test_model_management_card_shows_connection_model_button_when_connection_changes(self) -> None:
        form_state = {
            "target": "filter",
            "connection": "glm",
            "model_id": "glm-4-flash",
        }
        card = build_model_management_card(_model_card_state(), form_state=form_state)
        serialized = json.dumps(card, ensure_ascii=False)

        self.assertIn("切换 connection + model", serialized)
        self.assertNotIn("切换 model_id", serialized)
        route_actions = card["elements"][5]["actions"]
        self.assertEqual(route_actions[0]["value"]["action"], "model_apply_connection_model")

    def test_model_management_card_ignores_invalid_initial_option(self) -> None:
        form_state = {
            "target": "filter",
            "connection": "nonexistent_connection",
            "model_id": "some-model",
        }
        card = build_model_management_card(_model_card_state(), form_state=form_state)

        select_actions = card["elements"][3]["actions"]
        self.assertEqual(select_actions[0]["initial_option"], "filter")
        # invalid connection should be skipped
        self.assertNotIn("initial_option", select_actions[1])

    def test_target_change_auto_populates_agent_fields(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        data = Mock()
        data.event.action.value = {"field": "target"}
        data.event.action.name = None
        data.event.action.option = "filter"
        data.event.action.input_value = None
        data.event.action.tag = "select_static"
        data.event.operator.open_id = "ou_real_open_id"
        data.event.context.open_message_id = "om_card"

        resolved_filter = _model_test_config()  # filter agent config

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.resolve", return_value=resolved_filter) as resolve,
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            result = bot._handle_card_action(data)

        resolve.assert_called_once_with("filter")
        self.assertIsNotNone(result)
        # The response card should have pre-populated fields
        card_data = result.card.data
        card_json = json.dumps(card_data, ensure_ascii=False)
        self.assertIn("deepseek", card_json)
        self.assertIn("deepseek-chat", card_json)

    def test_connection_change_redraws_model_card_with_connection_model_button(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        def component_event(name: str, option: str = "", input_value: str = ""):
            data = Mock()
            data.event.action.value = {"field": name} if option else {}
            data.event.action.name = None if option else name
            data.event.action.option = option
            data.event.action.input_value = input_value
            data.event.action.tag = "input" if input_value else "select_static"
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_card"
            return data

        with patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()):
            self.assertIsNone(bot._handle_card_action(component_event("model_id", input_value="glm-4-flash")))
            result = bot._handle_card_action(component_event("connection", option="glm"))

        self.assertIsNotNone(result)
        card_json = json.dumps(result.card.data, ensure_ascii=False)
        self.assertIn("切换 connection + model", card_json)
        self.assertNotIn("切换 model_id", card_json)
        self.assertEqual(bot._model_card_form_state["om_card:ou_real_open_id"]["connection"], "glm")
        self.assertEqual(bot._model_card_form_state["om_card:ou_real_open_id"]["model_id"], "glm-4-flash")

    def test_target_change_to_default_auto_populates(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        data = Mock()
        data.event.action.value = {"field": "target"}
        data.event.action.name = None
        data.event.action.option = "default"
        data.event.action.input_value = None
        data.event.action.tag = "select_static"
        data.event.operator.open_id = "ou_real_open_id"
        data.event.context.open_message_id = "om_card"

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            result = bot._handle_card_action(data)

        self.assertIsNotNone(result)
        card_data = result.card.data
        card_json = json.dumps(card_data, ensure_ascii=False)
        # defaults from _model_card_state: connection=deepseek, model_id=deepseek-chat
        self.assertIn("deepseek", card_json)

    def test_target_change_preserves_manually_set_fields(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        def component_event(name: str, option: str = "", input_value: str = ""):
            data = Mock()
            data.event.action.value = {"field": name} if option else {}
            data.event.action.name = None if option else name
            data.event.action.option = option
            data.event.action.input_value = input_value
            data.event.action.tag = "input" if input_value else "select_static"
            data.event.operator.open_id = "ou_real_open_id"
            data.event.context.open_message_id = "om_card"
            return data

        resolved_filter = _model_test_config()

        with (
            patch("smzdm_notice.feishu.bot.llm_routing.resolve", return_value=resolved_filter),
            patch("smzdm_notice.feishu.bot.llm_routing.model_card_state", return_value=_model_card_state()),
        ):
            # First manually set route fields.
            connection_result = bot._handle_card_action(component_event("connection", option="glm"))
            self.assertIsNotNone(connection_result)
            self.assertIsNone(bot._handle_card_action(component_event("model_id", input_value="custom-model")))
            self.assertIsNone(bot._handle_card_action(component_event("temperature", input_value="0.9")))
            # Then change target. User-selected route fields must survive the target switch.
            result = bot._handle_card_action(component_event("target", option="filter"))

        self.assertIsNotNone(result)
        card_data = result.card.data
        card_json = json.dumps(card_data, ensure_ascii=False)
        # User's manual connection/model/temperature should be preserved.
        self.assertIn("glm", card_json)
        self.assertIn("custom-model", card_json)
        self.assertIn("0.9", card_json)

    def test_input_form_change_still_returns_none(self) -> None:
        binding_store = Mock()
        binding_store.is_bound_operator.return_value = True
        bot = FeishuInteractiveBot(
            BotRuntime(
                draft_store=Mock(),
                binding_store=binding_store,
                status_provider=lambda: "status",
                run_once=Mock(return_value=True),
            )
        )

        data = Mock()
        data.event.action.value = {}
        data.event.action.name = "model_id"
        data.event.action.option = None
        data.event.action.input_value = "deepseek-reasoner"
        data.event.action.tag = "input"
        data.event.operator.open_id = "ou_real_open_id"
        data.event.context.open_message_id = "om_card"

        with (
            patch.object(bot, "_dispatch_card_action") as dispatch,
        ):
            result = bot._handle_card_action(data)

        self.assertIsNone(result)
        dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
