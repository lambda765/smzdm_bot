from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from smzdm_notice.feishu import notifier
from smzdm_notice.feishu.binding import FeishuBinding, FeishuBindingStore
from smzdm_notice.llm.models import ArbiterInfo, FilterResult, Recommendation
from smzdm_notice.preferences.models import ConfigDraft
from smzdm_notice.smzdm.ranking import RankingItem


def _item(pic: str = "https://img.example.com/a.jpg") -> RankingItem:
    return RankingItem(
        rank=1,
        title="测试商品",
        article_id="1001",
        price="¥9.9",
        worthy=100,
        unworthy=1,
        comments=20,
        favorites=30,
        mall="测试商城",
        brand="测试品牌",
        tab_name="综合",
        link="https://example.com/deal",
        pic=pic,
    )


def _search_bypass_item() -> RankingItem:
    item = _item()
    item.source_type = "search"
    item.search_keyword = "AirPods Pro 2"
    item.search_max_price = 99.9
    return item


def _digest_entry(index: int) -> dict:
    return {
        "article_id": f"10{index:02d}",
        "title": f"完整测试商品 {index}",
        "price": f"¥{index}.9",
        "mall": f"测试商城 {index}",
        "brand": f"测试品牌 {index}",
        "worthy": index,
        "unworthy": index // 2,
        "comments": index * 3,
        "favorites": index * 4,
        "tags": ["好价", f"标签{index}"],
        "link": f"https://example.com/deal/{index}",
        "tab_name": "综合",
        "rank": index,
        "skip_reason": f"跳过原因：完整跳过原因 {index}",
    }


class NotifierBindingTests(unittest.TestCase):
    def test_binding_file_is_written_with_owner_only_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "binding.json"
            store = FeishuBindingStore(path)

            binding = store.bind("open_id", "ou_1", "ou_operator", "private")

            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["receive_id"], binding.receive_id)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_binding_file_permissions_are_fixed_when_replacing_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "binding.json"
            path.write_text("{}", encoding="utf-8")
            os.chmod(path, 0o644)
            store = FeishuBindingStore(path)

            store.bind("open_id", "ou_1", "ou_operator", "private")

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_send_card_success_returns_true_when_message_id_exists(self) -> None:
        with patch("smzdm_notice.feishu.notifier._send_card_message_id", return_value="om_1"):
            self.assertTrue(notifier._send_card_success({"elements": []}))

    def test_send_card_success_returns_false_when_message_id_missing(self) -> None:
        with patch("smzdm_notice.feishu.notifier._send_card_message_id", return_value=None):
            self.assertFalse(notifier._send_card_success({"elements": []}))

    def test_send_text_without_binding_skips_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeishuBindingStore(Path(tmp) / "binding.json")
            with ExitStack() as stack:
                stack.enter_context(patch("smzdm_notice.feishu.notifier._BINDING_STORE", store))
                stack.enter_context(patch("smzdm_notice.feishu.notifier.logger.warning"))
                get_message_models = stack.enter_context(patch("smzdm_notice.feishu.notifier.get_message_models"))
                self.assertFalse(notifier.send_text("hello"))
                get_message_models.assert_not_called()

    def test_send_deals_embeds_uploaded_image(self) -> None:
        sent_cards = []
        with (
            patch(
                "smzdm_notice.feishu.notifier.get_feishu_image_key",
                return_value="img_test_key",
            ) as get_image_key,
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_deal",
            ),
        ):
            self.assertTrue(notifier.send_deals([(_item(), "值得买")]))

        get_image_key.assert_called_once_with("https://img.example.com/a.jpg")
        elements = sent_cards[0]["elements"]
        image_elements = [element for element in elements if element.get("tag") == "img"]
        self.assertEqual(image_elements[0]["img_key"], "img_test_key")
        self.assertEqual(image_elements[0]["alt"]["content"], "测试商品")
        markdown = "\n".join(element.get("content", "") for element in elements if element.get("tag") == "markdown")
        self.assertNotIn("查看商品图片", markdown)

    def test_send_deals_falls_back_to_image_link_when_upload_fails(self) -> None:
        sent_cards = []
        with (
            patch("smzdm_notice.feishu.notifier.get_feishu_image_key", return_value=""),
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_deal",
            ),
        ):
            self.assertTrue(notifier.send_deals([(_item(), "值得买")]))

        elements = sent_cards[0]["elements"]
        self.assertFalse([element for element in elements if element.get("tag") == "img"])
        markdown = "\n".join(element.get("content", "") for element in elements if element.get("tag") == "markdown")
        self.assertIn("查看商品图片", markdown)
        self.assertIn("https://img.example.com/a.jpg", markdown)

    def test_send_deals_without_pic_does_not_process_image(self) -> None:
        with (
            patch("smzdm_notice.feishu.notifier.get_feishu_image_key") as get_image_key,
            patch("smzdm_notice.feishu.notifier._send_card_message_id", return_value="om_deal"),
        ):
            self.assertTrue(notifier.send_deals([(_item(pic=""), "值得买")]))

        get_image_key.assert_not_called()

    def test_send_deals_uses_search_price_bypass_buttons(self) -> None:
        sent_cards = []
        with (
            patch("smzdm_notice.feishu.notifier.get_feishu_image_key", return_value=""),
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_deal",
            ),
        ):
            item = _search_bypass_item()
            self.assertTrue(notifier.send_deals([(item, "价格直推")], price_bypass_article_ids={item.article_id}))

        actions = [element for element in sent_cards[0]["elements"] if element.get("tag") == "action"][0]["actions"]
        action_values = [action.get("value", {}).get("action") for action in actions if action.get("value")]
        self.assertIn("search_remove_keyword", action_values)
        self.assertIn("search_clear_price", action_values)
        self.assertNotIn("deal_ignore_category", action_values)

    def test_send_deals_keeps_normal_buttons_for_llm_search_match_with_price_config(self) -> None:
        sent_cards = []
        with (
            patch("smzdm_notice.feishu.notifier.get_feishu_image_key", return_value=""),
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_deal",
            ),
        ):
            self.assertTrue(notifier.send_deals([(_search_bypass_item(), "LLM 推荐")]))

        actions = [element for element in sent_cards[0]["elements"] if element.get("tag") == "action"][0]["actions"]
        action_values = [action.get("value", {}).get("action") for action in actions if action.get("value")]
        self.assertIn("deal_ignore_category", action_values)
        self.assertIn("deal_stock_enough", action_values)
        self.assertIn("deal_follow", action_values)
        self.assertNotIn("search_remove_keyword", action_values)
        self.assertNotIn("search_clear_price", action_values)

    def test_send_digest_without_overflow_sends_only_card(self) -> None:
        sent_cards = []
        entries = [_digest_entry(index) for index in range(1, 21)]
        with (
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_digest",
            ),
            patch("smzdm_notice.feishu.notifier._upload_file") as upload_file,
        ):
            self.assertTrue(notifier.send_digest(entries, "2026-05-30"))

        upload_file.assert_not_called()
        markdown = "\n".join(element.get("content", "") for element in sent_cards[0]["elements"])
        self.assertIn("完整测试商品 20", markdown)
        self.assertNotIn("完整测试商品 21", markdown)
        self.assertNotIn("完整内容见附件", markdown)

    def test_send_digest_with_overflow_sends_full_markdown_attachment(self) -> None:
        sent_cards = []
        sent_files = []
        calls = []
        entries = [_digest_entry(index) for index in range(1, 26)]
        binding = FeishuBinding("chat_id", "oc_digest", "2026-05-30T22:00:00", "ou_user", "test")
        with (
            patch("smzdm_notice.feishu.notifier._current_binding", return_value=binding),
            patch(
                "smzdm_notice.feishu.notifier._upload_file",
                side_effect=lambda file_name, content: (
                    calls.append("upload") or sent_files.append((file_name, content)) or "file_digest"
                ),
            ),
            patch(
                "smzdm_notice.feishu.notifier._send_card_to_message_id",
                side_effect=lambda _receive_id_type, _receive_id, card: (
                    calls.append("card") or sent_cards.append(card) or "om_digest"
                ),
            ),
            patch(
                "smzdm_notice.feishu.notifier._send_file_to",
                side_effect=lambda _receive_id_type, _receive_id, _file_key: calls.append("file") or True,
            ),
        ):
            self.assertTrue(notifier.send_digest(entries, "2026-05-30"))

        self.assertEqual(calls, ["upload", "card", "file"])
        card_markdown = "\n".join(element.get("content", "") for element in sent_cards[0]["elements"])
        self.assertIn("完整测试商品 20", card_markdown)
        self.assertNotIn("完整测试商品 21", card_markdown)
        self.assertIn("完整内容见附件", card_markdown)

        self.assertEqual(sent_files[0][0], "smzdm_digest_2026-05-30.md")
        attachment = sent_files[0][1].decode("utf-8")
        self.assertIn("# 什么值得买夜间汇总 (2026-05-30)", attachment)
        self.assertIn("## 1. 完整测试商品 1", attachment)
        self.assertIn("## 25. 完整测试商品 25", attachment)
        self.assertIn("- 链接: https://example.com/deal/25", attachment)
        self.assertIn("- 跳过原因: 完整跳过原因 25", attachment)

    def test_send_digest_returns_false_when_attachment_fails(self) -> None:
        entries = [_digest_entry(index) for index in range(1, 22)]
        binding = FeishuBinding("chat_id", "oc_digest", "2026-05-30T22:00:00", "ou_user", "test")
        with (
            patch("smzdm_notice.feishu.notifier._current_binding", return_value=binding),
            patch("smzdm_notice.feishu.notifier._upload_file", side_effect=RuntimeError("upload failed")),
            patch("smzdm_notice.feishu.notifier._send_card_to_message_id") as send_card,
        ):
            self.assertFalse(notifier.send_digest(entries, "2026-05-30"))
        send_card.assert_not_called()

    def test_upload_file_returns_file_key(self) -> None:
        CreateFileRequest = Mock()
        request_builder = Mock()
        request_builder.request_body.return_value = request_builder
        request_builder.build.return_value = "request"
        CreateFileRequest.builder.return_value = request_builder

        CreateFileRequestBody = Mock()
        body_builder = Mock()
        body_builder.file_type.return_value = body_builder
        body_builder.file_name.return_value = body_builder
        body_builder.file.return_value = body_builder
        body_builder.build.return_value = "body"
        CreateFileRequestBody.builder.return_value = body_builder

        response = Mock()
        response.success.return_value = True
        response.data.file_key = "file_key"
        client = Mock()
        client.im.v1.file.create.return_value = response

        with (
            patch(
                "smzdm_notice.feishu.notifier.get_file_models",
                return_value=(CreateFileRequest, CreateFileRequestBody),
            ),
            patch("smzdm_notice.feishu.notifier.get_lark_client", return_value=client),
        ):
            self.assertEqual(notifier._upload_file("digest.md", b"content"), "file_key")

        body_builder.file_type.assert_called_once_with("stream")
        body_builder.file_name.assert_called_once_with("digest.md")
        body_builder.file.assert_called_once()
        client.im.v1.file.create.assert_called_once_with("request")

    def test_send_poll_failure_warning_sanitizes_detail(self) -> None:
        sent_cards = []
        detail = "usage limit exceeded api_key=sk-secret123456789 token=abc"
        with patch(
            "smzdm_notice.feishu.notifier._send_card_message_id",
            side_effect=lambda card: sent_cards.append(card) or "om_warn",
        ):
            self.assertTrue(notifier.send_poll_failure_warning(3, "llm_failed", detail))

        self.assertEqual(sent_cards[0]["header"]["template"], "red")
        markdown = sent_cards[0]["elements"][0]["content"]
        self.assertIn("连续 **3** 次轮询失败", markdown)
        self.assertIn("LLM 调用失败", markdown)
        self.assertIn("<redacted>", markdown)
        self.assertNotIn("sk-secret123456789", markdown)
        self.assertNotIn("token=abc", markdown)

    def test_build_help_card_uses_markdown_content(self) -> None:
        card = notifier.build_help_card("help content")

        self.assertEqual(card["header"]["template"], "blue")
        self.assertEqual(card["elements"][0]["content"], "help content")

    def test_reply_text_uses_feishu_reply_api(self) -> None:
        ReplyRequest = Mock()
        request_builder = Mock()
        request_builder.message_id.return_value = request_builder
        request_builder.request_body.return_value = request_builder
        request_builder.build.return_value = "request"
        ReplyRequest.builder.return_value = request_builder

        ReplyBody = Mock()
        body_builder = Mock()
        body_builder.msg_type.return_value = body_builder
        body_builder.content.return_value = body_builder
        body_builder.build.return_value = "body"
        ReplyBody.builder.return_value = body_builder

        response = Mock()
        response.success.return_value = True
        response.data.message_id = "om_reply"
        client = Mock()
        client.im.v1.message.reply.return_value = response

        with (
            patch(
                "smzdm_notice.feishu.notifier.get_reply_message_models",
                return_value=(ReplyRequest, ReplyBody),
            ),
            patch("smzdm_notice.feishu.notifier.get_lark_client", return_value=client),
        ):
            self.assertTrue(notifier.reply_text("om_original", "hello"))

        request_builder.message_id.assert_called_once_with("om_original")
        body_builder.msg_type.assert_called_once_with("text")
        self.assertEqual(json.loads(body_builder.content.call_args.args[0]), {"text": "hello"})
        client.im.v1.message.reply.assert_called_once_with("request")

    def test_send_help_replies_when_reply_target_exists(self) -> None:
        with (
            patch("smzdm_notice.feishu.notifier.reply_card", return_value="om_reply") as reply_card,
            patch("smzdm_notice.feishu.notifier._send_card_message_id") as send_card,
        ):
            self.assertTrue(notifier.send_help("help content", reply_to_message_id="om_original"))

        reply_card.assert_called_once()
        self.assertEqual(reply_card.call_args.args[0], "om_original")
        send_card.assert_not_called()

    def test_send_help_falls_back_to_regular_card_when_reply_fails(self) -> None:
        with (
            patch("smzdm_notice.feishu.notifier.reply_card", return_value=None),
            patch("smzdm_notice.feishu.notifier._send_card_message_id", return_value="om_regular") as send_card,
        ):
            self.assertTrue(notifier.send_help("help content", reply_to_message_id="om_original"))

        send_card.assert_called_once()

    def test_send_draft_preview_delegates_preview_rendering_and_keeps_buttons(self) -> None:
        draft = ConfigDraft(
            draft_id="preview-only",
            target_file="preference.md",
            title="样式预览",
            summary="测试",
            append_text="- 新规则",
            source="test",
        )
        sent_cards = []
        with (
            patch(
                "smzdm_notice.feishu.notifier.build_draft_preview_content",
                return_value="preview content",
            ) as build_content,
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_preview",
            ),
        ):
            self.assertTrue(notifier.send_draft_preview(draft))

        build_content.assert_called_once_with(draft)
        self.assertEqual(draft.preview_message_id, "om_preview")
        self.assertEqual(sent_cards[0]["elements"][0]["content"], "preview content")
        actions = sent_cards[0]["elements"][1]["actions"]
        self.assertEqual(actions[0]["value"]["draft_id"], "preview-only")
        self.assertEqual(actions[1]["value"]["action"], "cancel_draft")

    def test_draft_status_cards_do_not_include_actions(self) -> None:
        processing = notifier.build_draft_processing_card("正在理解偏好/库存修改", elapsed_seconds=15)
        failure = notifier.build_draft_failure_card("草案生成失败")

        self.assertNotIn("action", {element.get("tag") for element in processing["elements"]})
        self.assertNotIn("action", {element.get("tag") for element in failure["elements"]})
        self.assertIn("15 秒", processing["elements"][0]["content"])
        self.assertIn("草案生成失败", failure["elements"][0]["content"])

    def test_build_draft_preview_card_includes_apply_and_cancel_actions(self) -> None:
        draft = ConfigDraft(
            draft_id="preview-card",
            target_file="preference.md",
            title="样式预览",
            summary="测试",
            append_text="- 新规则",
            source="test",
        )
        with patch("smzdm_notice.feishu.notifier.build_draft_preview_content", return_value="preview content"):
            card = notifier.build_draft_preview_card(draft)

        actions = card["elements"][1]["actions"]
        self.assertEqual(actions[0]["value"], {"action": "apply_draft", "draft_id": "preview-card"})
        self.assertEqual(actions[1]["value"], {"action": "cancel_draft", "draft_id": "preview-card"})

    def test_send_draft_preview_replies_and_stores_reply_message_id(self) -> None:
        draft = ConfigDraft(
            draft_id="preview-reply",
            target_file="preference.md",
            title="样式预览",
            summary="测试",
            append_text="- 新规则",
            source="test",
        )
        with (
            patch("smzdm_notice.feishu.notifier.build_draft_preview_content", return_value="preview content"),
            patch("smzdm_notice.feishu.notifier.reply_card", return_value="om_reply") as reply_card,
            patch("smzdm_notice.feishu.notifier._send_card_message_id") as send_card,
        ):
            self.assertTrue(notifier.send_draft_preview(draft, reply_to_message_id="om_original"))

        reply_card.assert_called_once()
        self.assertEqual(reply_card.call_args.args[0], "om_original")
        send_card.assert_not_called()
        self.assertEqual(draft.preview_message_id, "om_reply")

    def test_update_card_message_patches_message_content(self) -> None:
        PatchRequest = Mock()
        request_builder = Mock()
        request_builder.message_id.return_value = request_builder
        request_builder.request_body.return_value = request_builder
        request_builder.build.return_value = "request"
        PatchRequest.builder.return_value = request_builder

        PatchBody = Mock()
        body_builder = Mock()
        content_holder = {}

        def capture_content(content):
            content_holder["content"] = content
            return body_builder

        body_builder.content.side_effect = capture_content
        body_builder.build.return_value = "body"
        PatchBody.builder.return_value = body_builder

        response = Mock()
        response.success.return_value = True
        client = Mock()
        client.im.v1.message.patch.return_value = response

        with (
            patch("smzdm_notice.feishu.notifier.get_message_update_models", return_value=(PatchRequest, PatchBody)),
            patch("smzdm_notice.feishu.notifier.get_lark_client", return_value=client),
        ):
            self.assertTrue(notifier.update_card_message("om_update", {"config": {"update_multi": True}}))

        request_builder.message_id.assert_called_once_with("om_update")
        client.im.v1.message.patch.assert_called_once_with("request")
        self.assertEqual(json.loads(content_holder["content"]), {"config": {"update_multi": True}})

    def test_send_arbitration_embeds_draft_preview_and_apply_button(self) -> None:
        draft = ConfigDraft(
            draft_id="arbiter-draft",
            target_file="preference.md",
            title="限制黑名单扩展",
            summary="避免误判",
            append_text="- 黑名单只按字面精确匹配",
            source="仲裁建议一键采纳",
        )
        info = ArbiterInfo(
            chosen="B",
            reason="B 更准确",
            analysis="A 过度扩展黑名单。",
            suggestion="黑名单只按字面精确匹配。",
            result_a=FilterResult(recommendations=[Recommendation(id="1", reason="A")]),
            result_b=FilterResult(recommendations=[Recommendation(id="2", reason="B")]),
            items={},
        )
        sent_cards = []
        with (
            patch(
                "smzdm_notice.feishu.notifier.build_draft_preview_content",
                return_value="preview content",
            ) as build_content,
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_arbiter",
            ),
        ):
            self.assertTrue(notifier.send_arbitration(info, draft))

        build_content.assert_called_once_with(draft)
        self.assertEqual(draft.preview_message_id, "om_arbiter")
        markdown = "\n".join(
            element.get("content", "") for element in sent_cards[0]["elements"] if element.get("tag") == "markdown"
        )
        self.assertIn("preview content", markdown)
        actions = sent_cards[0]["elements"][-1]["actions"]
        self.assertEqual(actions[0]["text"]["content"], "采纳并更新")
        self.assertEqual(
            actions[0]["value"],
            {"action": "apply_draft", "draft_id": "arbiter-draft", "card_kind": "arbitration"},
        )
        self.assertEqual(
            actions[1]["value"],
            {"action": "ignore_arbitration", "draft_id": "arbiter-draft", "card_kind": "arbitration"},
        )
        self.assertEqual(draft.metadata["card_kind"], "arbitration")
        self.assertEqual(draft.metadata["arbitration_card"]["analysis"], "A 过度扩展黑名单。")

    def test_send_arbitration_without_draft_has_no_adopt_button(self) -> None:
        info = ArbiterInfo(
            chosen="A",
            reason="A 更准确",
            analysis="差异不足以形成规则。",
            suggestion="无需修改。",
            result_a=FilterResult(),
            result_b=FilterResult(),
            items={},
        )
        sent_cards = []
        with patch(
            "smzdm_notice.feishu.notifier._send_card_message_id",
            side_effect=lambda card: sent_cards.append(card) or "om_arbiter",
        ):
            self.assertTrue(notifier.send_arbitration(info))

        actions = sent_cards[0]["elements"][-1]["actions"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["value"]["action"], "ignore_arbitration")
        markdown = "\n".join(
            element.get("content", "") for element in sent_cards[0]["elements"] if element.get("tag") == "markdown"
        )
        self.assertIn("未生成可直接采纳", markdown)

    def test_disable_draft_card_patches_message_without_buttons(self) -> None:
        draft = ConfigDraft(
            draft_id="draft-1",
            target_file="preference.md",
            title="样式预览",
            summary="测试",
            append_text="- 新规则",
            source="test",
        )
        card = notifier.build_disabled_draft_card("已生成新的修改预览", draft)

        self.assertEqual(card["header"]["template"], "grey")
        self.assertNotIn("action", {element.get("tag") for element in card["elements"]})
        self.assertIn("新规则", card["elements"][0]["content"])
        self.assertIn("已生成新的修改预览", card["elements"][0]["content"])

    def test_disabled_arbitration_card_preserves_analysis_without_buttons(self) -> None:
        draft = ConfigDraft(
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

        card = notifier.build_disabled_arbitration_card("已忽略", draft)

        self.assertEqual(card["header"]["template"], "grey")
        self.assertIn("仲裁分析", card["header"]["title"]["content"])
        self.assertNotIn("action", {element.get("tag") for element in card["elements"]})
        markdown = "\n".join(element.get("content", "") for element in card["elements"])
        self.assertIn("A 过度扩展黑名单。", markdown)
        self.assertIn("黑名单只按字面精确匹配。", markdown)
        self.assertIn("已忽略", markdown)

    def test_disable_draft_card_patches_message(self) -> None:
        draft = ConfigDraft(
            draft_id="draft-1",
            target_file="preference.md",
            title="样式预览",
            summary="测试",
            append_text="- 新规则",
            source="test",
        )
        PatchRequest = Mock()
        request_builder = Mock()
        request_builder.message_id.return_value = request_builder
        request_builder.request_body.return_value = request_builder
        request_builder.build.return_value = "request"
        PatchRequest.builder.return_value = request_builder

        PatchBody = Mock()
        body_builder = Mock()
        content_holder = {}

        def capture_content(content):
            content_holder["content"] = content
            return body_builder

        body_builder.content.side_effect = capture_content
        body_builder.build.return_value = "body"
        PatchBody.builder.return_value = body_builder

        response = Mock()
        response.success.return_value = True
        client = Mock()
        client.im.v1.message.patch.return_value = response

        with (
            patch(
                "smzdm_notice.feishu.notifier.get_message_update_models",
                return_value=(PatchRequest, PatchBody),
            ),
            patch("smzdm_notice.feishu.notifier.get_lark_client", return_value=client),
            patch(
                "smzdm_notice.feishu.notifier.build_draft_preview_content",
                return_value="preview content",
            ),
        ):
            self.assertTrue(notifier.disable_draft_card("om_1", "已生成新的修改预览", draft))

        request_builder.message_id.assert_called_once_with("om_1")
        client.im.v1.message.patch.assert_called_once_with("request")
        card = json.loads(content_holder["content"])
        self.assertEqual(card["header"]["template"], "grey")
        self.assertNotIn("action", {element.get("tag") for element in card["elements"]})
        self.assertIn("preview content", card["elements"][0]["content"])
        self.assertIn("已生成新的修改预览", card["elements"][0]["content"])


if __name__ == "__main__":
    unittest.main()
