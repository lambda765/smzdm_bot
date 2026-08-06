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
from smzdm_notice.feishu.card_v2 import card_component_count, card_json_size_bytes
from smzdm_notice.feishu.model_cards import build_model_management_card
from smzdm_notice.llm.models import ArbiterInfo, FilterResult, Recommendation
from smzdm_notice.preferences.models import ConfigDraft
from smzdm_notice.smzdm.ranking import RankingItem


def _elements(card: dict) -> list[dict]:
    return card["body"]["elements"]


def _components(card: dict) -> list[dict]:
    result: list[dict] = []

    def visit(component: dict) -> None:
        result.append(component)
        for key in ("elements", "columns"):
            for child in component.get(key, []):
                visit(child)

    for element in _elements(card):
        visit(element)
    return result


def _callback_value(component: dict) -> dict:
    for behavior in component.get("behaviors", []):
        if behavior.get("type") == "callback":
            return behavior.get("value", {})
    return {}


def _actions(card: dict) -> list[dict]:
    return [_callback_value(component) for component in _components(card) if _callback_value(component).get("action")]


def _send_single_deals_card(
    items: list[tuple[RankingItem, str]],
    price_bypass_article_ids: set[str] | None = None,
    notification_names_by_article_id: dict[str, str] | None = None,
) -> dict:
    """通过真实发送入口捕获单张商品卡，避免为测试保留生产构造包装器。"""
    sent_cards: list[dict] = []
    with (
        patch("smzdm_notice.feishu.notifier.get_feishu_image_key", return_value=""),
        patch(
            "smzdm_notice.feishu.notifier._send_card_message_id",
            side_effect=lambda card: sent_cards.append(card) or "om_deal",
        ),
    ):
        result = notifier.send_deals(
            items,
            price_bypass_article_ids=price_bypass_article_ids,
            notification_names_by_article_id=notification_names_by_article_id,
        )
    if len(sent_cards) != 1 or result.failed_article_ids:
        raise AssertionError("测试商品应生成且成功发送一张卡片")
    return sent_cards[0]


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
    def assert_card_v2(self, card: dict) -> None:
        self.assertEqual(card.get("schema"), "2.0")
        self.assertNotIn("elements", card)
        self.assertIn("summary", card.get("config", {}))
        self.assertLessEqual(len(card["config"]["summary"]["content"]), 80)
        interactive_ids: list[str] = []
        for component in _components(card):
            self.assertNotEqual(component.get("tag"), "action")
            self.assertNotIn("form_action_type", component)
            if component.get("tag") == "button":
                self.assertNotIn("value", component)
                self.assertNotIn("url", component)
                self.assertIn("behaviors", component)
            if component.get("tag") in {"button", "input", "select_static"}:
                interactive_ids.append(component.get("element_id", ""))
            if "confirm" in component:
                self.assertIn("text", component["confirm"])
                self.assertNotIn("content", component["confirm"])
        self.assertTrue(all(interactive_ids))
        self.assertEqual(len(interactive_ids), len(set(interactive_ids)))

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
        elements = _elements(sent_cards[0])
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

        elements = _elements(sent_cards[0])
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

    def test_deals_summary_groups_names_and_preserves_item_order(self) -> None:
        first = _item(pic="")
        second = _item(pic="")
        third = _item(pic="")
        first.article_id = "1"
        second.article_id = "2"
        third.article_id = "3"

        card = _send_single_deals_card(
            [(first, "A"), (second, "B"), (third, "C")],
            notification_names_by_article_id={"1": "纸尿裤", "2": "纸尿裤", "3": "蓝莓"},
        )

        self.assertEqual(card["config"]["summary"]["content"], "好价：纸尿裤×2、蓝莓")
        markdown = "\n".join(element.get("content", "") for element in _elements(card))
        self.assertLess(markdown.index("**测试商品**"), markdown.rindex("**测试商品**"))

    def test_deals_summary_skips_missing_names_but_keeps_all_card_items(self) -> None:
        first = _item(pic="")
        second = _item(pic="")
        third = _item(pic="")
        first.article_id, first.title = "1", "蓝莓商品正文"
        second.article_id, second.title = "2", "缺名商品正文一"
        third.article_id, third.title = "3", "缺名商品正文二"

        card = _send_single_deals_card(
            [(first, "A"), (second, "B"), (third, "C")],
            notification_names_by_article_id={"1": "蓝莓"},
        )

        self.assertEqual(card["config"]["summary"]["content"], "好价：蓝莓等 3 件")
        markdown = "\n".join(element.get("content", "") for element in _elements(card))
        self.assertIn("蓝莓商品正文", markdown)
        self.assertIn("缺名商品正文一", markdown)
        self.assertIn("缺名商品正文二", markdown)

    def test_deals_summary_uses_count_when_all_names_are_missing(self) -> None:
        items = [(_item(pic=""), "A"), (_item(pic=""), "B")]
        items[0][0].article_id = "1"
        items[1][0].article_id = "2"

        card = _send_single_deals_card(items)

        self.assertEqual(card["config"]["summary"]["content"], "推荐了 2 个商品")

    def test_deals_summary_uses_search_keyword_and_stays_within_80_characters(self) -> None:
        items: list[tuple[RankingItem, str]] = []
        names: dict[str, str] = {}
        for index in range(8):
            item = _item(pic="")
            item.article_id = str(index)
            item.title = f"原标题{index}"
            items.append((item, "reason"))
            names[item.article_id] = f"精简商品名称{index}号超长文本"
        search_item = _search_bypass_item()
        search_item.article_id = "search"
        items.insert(0, (search_item, "bypass"))
        names[search_item.article_id] = search_item.search_keyword

        summary = notifier._build_deals_summary(items, names)

        self.assertTrue(summary.startswith("好价：AirPods Pro 2"))
        self.assertTrue(summary.endswith("等 9 件"))
        self.assertLessEqual(len(summary), 80)

    def test_representative_cards_use_json_2_without_legacy_actions(self) -> None:
        draft = ConfigDraft(
            draft_id="v2-check",
            target_file="preference.md",
            title="test",
            summary="test",
            append_text="- rule",
            source="test",
        )
        model_state = {
            "defaults": {"connection": "test", "model_id": "model"},
            "agents": [],
            "connections": [{"name": "test", "label": "Test", "key_configured": True}],
        }
        cards = [
            _send_single_deals_card([(_item(pic=""), "reason")]),
            notifier.build_help_card("help"),
            notifier.build_draft_preview_card(draft),
            notifier.build_draft_processing_card(),
            notifier.build_draft_failure_card("failed"),
            notifier.build_draft_handoff_card(),
            notifier.build_disabled_draft_card("disabled", draft),
            build_model_management_card(model_state),
        ]

        for card in cards:
            self.assert_card_v2(card)

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

        all_action_values = [value["action"] for value in _actions(sent_cards[0])]
        self.assertNotIn("deal_good", all_action_values)
        self.assertNotIn("deal_not_worth", all_action_values)
        self.assertIn("search_remove_keyword", all_action_values)
        self.assertIn("search_clear_price", all_action_values)
        self.assertNotIn("deal_ignore_category", all_action_values)

    def test_send_deals_keeps_normal_buttons_for_llm_search_match_with_price_config(self) -> None:
        sent_cards = []
        with (
            patch("smzdm_notice.feishu.notifier.get_feishu_image_key", return_value=""),
            patch("smzdm_notice.feishu.notifier.config.DEAL_MEMORY_ENABLED", True),
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_deal",
            ),
        ):
            self.assertTrue(notifier.send_deals([(_search_bypass_item(), "LLM 推荐")]))

        all_action_values = [value["action"] for value in _actions(sent_cards[0])]
        self.assertIn("deal_good", all_action_values)
        self.assertIn("deal_not_worth", all_action_values)
        self.assertIn("deal_ignore_category", all_action_values)
        self.assertIn("deal_stock_enough", all_action_values)
        self.assertIn("deal_follow", all_action_values)
        self.assertNotIn("search_remove_keyword", all_action_values)
        self.assertNotIn("search_clear_price", all_action_values)
        self.assertFalse(
            any(
                component.get("tag") == "input"
                for component in _components(sent_cards[0])
            )
        )

    def test_send_deals_hides_memory_buttons_when_disabled(self) -> None:
        sent_cards = []
        with (
            patch("smzdm_notice.feishu.notifier.get_feishu_image_key", return_value=""),
            patch("smzdm_notice.feishu.notifier.config.DEAL_MEMORY_ENABLED", False),
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_deal",
            ),
        ):
            self.assertTrue(notifier.send_deals([(_item(), "LLM 推荐")]))

        all_action_values = [value["action"] for value in _actions(sent_cards[0])]
        self.assertNotIn("deal_good", all_action_values)
        self.assertNotIn("deal_not_worth", all_action_values)
        self.assertIn("deal_ignore_category", all_action_values)

    def test_update_deal_card_feedback_state_toggles_cached_button_row(self) -> None:
        sent_cards = []
        updated_cards = []
        with (
            patch("smzdm_notice.feishu.notifier.get_feishu_image_key", return_value=""),
            patch("smzdm_notice.feishu.notifier.config.DEAL_MEMORY_ENABLED", True),
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_deal_toggle",
            ),
            patch(
                "smzdm_notice.feishu.notifier.update_card_message",
                side_effect=lambda _message_id, card: updated_cards.append(card) or True,
            ) as update_card,
        ):
            item = _item()
            self.assertTrue(notifier.send_deals([(item, "LLM 推荐")]))
            result = notifier.update_deal_card_feedback_state("om_deal_toggle", item.article_id, selected="deal_good")
            self.assertIsNotNone(result)

        update_card.assert_called_once()
        components = _components(updated_cards[0])
        memory_labels = [
            component["text"]["content"]
            for component in components
            if _callback_value(component).get("action") in {"deal_good", "deal_not_worth"}
        ]
        self.assertEqual(memory_labels, ["✅ 好价"])

    def test_update_deal_card_feedback_state_shows_optional_not_worth_reason_input(self) -> None:
        sent_cards = []
        updated_cards = []
        with (
            patch("smzdm_notice.feishu.notifier.get_feishu_image_key", return_value=""),
            patch("smzdm_notice.feishu.notifier.config.DEAL_MEMORY_ENABLED", True),
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or "om_deal_reason",
            ),
            patch(
                "smzdm_notice.feishu.notifier.update_card_message",
                side_effect=lambda _message_id, card: updated_cards.append(card) or True,
            ),
        ):
            item = _item()
            self.assertTrue(notifier.send_deals([(item, "LLM 推荐")]))
            result = notifier.update_deal_card_feedback_state(
                "om_deal_reason",
                item.article_id,
                selected="deal_not_worth",
                reason="价格一般\n非刚需",
            )
            self.assertIsNotNone(result)

        self.assertFalse(any(component.get("tag") == "form" for component in _components(sent_cards[0])))
        components = _components(updated_cards[0])
        forms = [component for component in components if component.get("tag") == "form"]
        self.assertEqual(len(forms), 1)
        reason_inputs = [component for component in components if component.get("tag") == "input"]
        self.assertEqual(len(reason_inputs), 1)
        self.assertEqual(reason_inputs[0]["default_value"], "价格一般 非刚需")
        self.assertEqual(_callback_value(reason_inputs[0])["article_id"], item.article_id)
        self.assertEqual(_callback_value(reason_inputs[0])["field"], notifier.NOT_WORTH_REASON_FIELD)
        self.assertTrue(
            any(value.get("action") == "deal_not_worth_reason" for value in _actions(updated_cards[0]))
        )
        save_buttons = [
            component
            for component in components
            if _callback_value(component).get("action") == "deal_not_worth_reason"
        ]
        self.assertEqual(save_buttons[0]["action_type"], "form_submit")
        self.assertNotIn("form_action_type", save_buttons[0])
        markdown = "\n".join(element.get("content", "") for element in _elements(updated_cards[0]))
        self.assertIn("价格一般 非刚需", markdown)

    def test_send_deals_splits_large_result_within_card_budgets_and_preserves_order(self) -> None:
        items: list[tuple[RankingItem, str]] = []
        for index in range(20):
            item = _item(pic=f"https://img.example.com/{index}.jpg")
            item.article_id = str(index)
            item.link = f"https://example.com/deal/{index}"
            item.title = f"测试商品 {index}"
            items.append((item, f"推荐理由 {index}"))

        sent_cards: list[dict] = []
        with (
            patch("smzdm_notice.feishu.notifier.config.DEAL_MEMORY_ENABLED", True),
            patch(
                "smzdm_notice.feishu.notifier.get_feishu_image_key",
                side_effect=lambda url: f"img_{url.rsplit('/', 1)[-1]}",
            ) as get_image_key,
            patch(
                "smzdm_notice.feishu.notifier._send_card_message_id",
                side_effect=lambda card: sent_cards.append(card) or f"om_{len(sent_cards)}",
            ),
        ):
            result = notifier.send_deals(items)

        self.assertEqual(result.delivered_article_ids, tuple(str(index) for index in range(20)))
        self.assertFalse(result.failed_article_ids)
        self.assertGreater(len(sent_cards), 1)
        self.assertEqual(get_image_key.call_count, len(items))
        sent_order = [
            value["article_id"]
            for card in sent_cards
            for value in _actions(card)
            if value.get("action") == "deal_ignore_category"
        ]
        self.assertEqual(sent_order, [str(index) for index in range(20)])
        for card in sent_cards:
            self.assertLessEqual(card_component_count(card), notifier.DEAL_CARD_COMPONENT_BUDGET)
            self.assertLessEqual(card_json_size_bytes(card), notifier.DEAL_CARD_JSON_BUDGET_BYTES)

    def test_send_deals_reports_only_articles_from_successful_chunks(self) -> None:
        items: list[tuple[RankingItem, str]] = []
        for index in range(20):
            item = _item(pic=f"https://img.example.com/{index}.jpg")
            item.article_id = str(index)
            item.link = f"https://example.com/deal/{index}"
            items.append((item, "推荐理由"))

        sent_cards: list[dict] = []

        def send_card(card: dict) -> str | None:
            sent_cards.append(card)
            return None if len(sent_cards) == 2 else f"om_{len(sent_cards)}"

        with (
            patch("smzdm_notice.feishu.notifier.config.DEAL_MEMORY_ENABLED", True),
            patch("smzdm_notice.feishu.notifier.get_feishu_image_key", return_value="img_key"),
            patch("smzdm_notice.feishu.notifier._send_card_message_id", side_effect=send_card),
        ):
            result = notifier.send_deals(items)

        chunk_article_ids = [
            [
                value["article_id"]
                for value in _actions(card)
                if value.get("action") == "deal_ignore_category"
            ]
            for card in sent_cards
        ]
        self.assertEqual(result.failed_article_ids, tuple(chunk_article_ids[1]))
        self.assertEqual(
            result.delivered_article_ids,
            tuple([*chunk_article_ids[0], *chunk_article_ids[2]]),
        )

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
        markdown = "\n".join(element.get("content", "") for element in _elements(sent_cards[0]))
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
        card_markdown = "\n".join(element.get("content", "") for element in _elements(sent_cards[0]))
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
        markdown = _elements(sent_cards[0])[0]["content"]
        self.assertIn("连续 **3** 次轮询失败", markdown)
        self.assertIn("LLM 调用失败", markdown)
        self.assertIn("<redacted>", markdown)
        self.assertNotIn("sk-secret123456789", markdown)
        self.assertNotIn("token=abc", markdown)

    def test_build_help_card_uses_markdown_content(self) -> None:
        card = notifier.build_help_card("help content")

        self.assertEqual(card["header"]["template"], "blue")
        self.assertEqual(_elements(card)[0]["content"], "help content")

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
        self.assertEqual(_elements(sent_cards[0])[0]["content"], "preview content")
        actions = _actions(sent_cards[0])
        self.assertEqual(actions[0]["draft_id"], "preview-only")
        self.assertEqual(actions[1]["action"], "cancel_draft")

    def test_draft_status_cards_do_not_include_actions(self) -> None:
        processing = notifier.build_draft_processing_card("正在理解偏好/库存修改")
        failure = notifier.build_draft_failure_card("草案生成失败")

        self.assertNotIn("action", {element.get("tag") for element in _elements(processing)})
        self.assertNotIn("action", {element.get("tag") for element in _elements(failure)})
        self.assertIn("正在理解偏好/库存修改", _elements(processing)[0]["content"])
        self.assertNotIn("已等待", _elements(processing)[0]["content"])
        self.assertIn("草案生成失败", _elements(failure)[0]["content"])

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

        actions = _actions(card)
        self.assertEqual(actions[0], {"action": "apply_draft", "draft_id": "preview-card"})
        self.assertEqual(actions[1], {"action": "cancel_draft", "draft_id": "preview-card"})

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

    def test_streaming_content_update_uses_unique_uuid_and_requested_sequence(self) -> None:
        ContentRequest = Mock()
        request_builder = Mock()
        request_builder.card_id.return_value = request_builder
        request_builder.element_id.return_value = request_builder
        request_builder.request_body.return_value = request_builder
        request_builder.build.return_value = "request"
        ContentRequest.builder.return_value = request_builder

        ContentBody = Mock()
        body_builder = Mock()
        body_builder.uuid.return_value = body_builder
        body_builder.content.return_value = body_builder
        body_builder.sequence.return_value = body_builder
        body_builder.build.return_value = "body"
        ContentBody.builder.return_value = body_builder

        response = Mock()
        response.success.return_value = True
        client = Mock()
        client.cardkit.v1.card_element.content.return_value = response

        with (
            patch(
                "smzdm_notice.feishu.notifier.get_cardkit_content_models",
                return_value=(ContentRequest, ContentBody),
            ),
            patch("smzdm_notice.feishu.notifier.get_lark_client", return_value=client),
        ):
            self.assertTrue(notifier.update_streaming_draft_content("card-1", "stage 1", 3))
            self.assertTrue(notifier.update_streaming_draft_content("card-1", "stage 2", 4))

        uuids = [call.args[0] for call in body_builder.uuid.call_args_list]
        self.assertEqual(len(set(uuids)), 2)
        self.assertEqual([call.args[0] for call in body_builder.sequence.call_args_list], [3, 4])
        request_builder.element_id.assert_called_with(notifier.DRAFT_STREAMING_ELEMENT_ID)

    def test_finish_streaming_card_closes_mode_before_final_update(self) -> None:
        calls: list[tuple] = []
        with (
            patch(
                "smzdm_notice.feishu.notifier._close_cardkit_streaming_mode",
                side_effect=lambda card_id, sequence: calls.append(("settings", card_id, sequence)) or True,
            ),
            patch(
                "smzdm_notice.feishu.notifier._update_cardkit_card",
                side_effect=lambda card_id, card, sequence: calls.append(("update", card_id, card, sequence)) or True,
            ),
        ):
            card = notifier.build_draft_failure_card("failed")
            self.assertTrue(notifier.finish_streaming_draft_card("card-1", card, 5, 6))

        self.assertEqual(calls[0], ("settings", "card-1", 5))
        self.assertEqual(calls[1], ("update", "card-1", card, 6))

    def test_streaming_settings_wraps_streaming_mode_in_config(self) -> None:
        SettingsRequest = Mock()
        request_builder = Mock()
        request_builder.card_id.return_value = request_builder
        request_builder.request_body.return_value = request_builder
        request_builder.build.return_value = "request"
        SettingsRequest.builder.return_value = request_builder

        SettingsBody = Mock()
        body_builder = Mock()
        body_builder.settings.return_value = body_builder
        body_builder.uuid.return_value = body_builder
        body_builder.sequence.return_value = body_builder
        body_builder.build.return_value = "body"
        SettingsBody.builder.return_value = body_builder

        response = Mock()
        response.success.return_value = True
        client = Mock()
        client.cardkit.v1.card.settings.return_value = response

        with (
            patch(
                "smzdm_notice.feishu.notifier.get_cardkit_settings_models",
                return_value=(SettingsRequest, SettingsBody),
            ),
            patch("smzdm_notice.feishu.notifier.get_lark_client", return_value=client),
        ):
            self.assertTrue(notifier._close_cardkit_streaming_mode("card-1", 5))

        self.assertEqual(
            json.loads(body_builder.settings.call_args.args[0]),
            {"config": {"streaming_mode": False}},
        )
        body_builder.sequence.assert_called_once_with(5)

    def test_streaming_start_falls_back_when_create_or_send_fails(self) -> None:
        with (
            patch("smzdm_notice.feishu.notifier._create_cardkit_card", return_value=""),
            patch("smzdm_notice.feishu.notifier.reply_card_entity") as reply_entity,
        ):
            self.assertIsNone(notifier.start_streaming_draft_processing("stage", "om-parent"))
            reply_entity.assert_not_called()

        with (
            patch("smzdm_notice.feishu.notifier._create_cardkit_card", return_value="card-1"),
            patch("smzdm_notice.feishu.notifier.reply_card_entity", return_value=None),
        ):
            self.assertIsNone(notifier.start_streaming_draft_processing("stage", "om-parent"))

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
            element.get("content", "") for element in _elements(sent_cards[0]) if element.get("tag") == "markdown"
        )
        self.assertIn("preview content", markdown)
        buttons = [component for component in _components(sent_cards[0]) if component.get("tag") == "button"]
        actions = _actions(sent_cards[0])
        self.assertEqual(buttons[0]["text"]["content"], "采纳并更新")
        self.assertEqual(
            actions[0],
            {"action": "apply_draft", "draft_id": "arbiter-draft", "card_kind": "arbitration"},
        )
        self.assertEqual(
            actions[1],
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

        actions = _actions(sent_cards[0])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "ignore_arbitration")
        markdown = "\n".join(
            element.get("content", "") for element in _elements(sent_cards[0]) if element.get("tag") == "markdown"
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
        self.assertNotIn("action", {element.get("tag") for element in _elements(card)})
        self.assertIn("新规则", _elements(card)[0]["content"])
        self.assertIn("已生成新的修改预览", _elements(card)[0]["content"])

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
        self.assertNotIn("action", {element.get("tag") for element in _elements(card)})
        markdown = "\n".join(element.get("content", "") for element in _elements(card))
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
        self.assertNotIn("action", {element.get("tag") for element in _elements(card)})
        self.assertIn("preview content", _elements(card)[0]["content"])
        self.assertIn("已生成新的修改预览", _elements(card)[0]["content"])


if __name__ == "__main__":
    unittest.main()
