"""飞书通知模块。"""

from __future__ import annotations

import json
import re
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any

from loguru import logger

from smzdm_notice.core import config
from smzdm_notice.feishu.binding import FeishuBinding, FeishuBindingStore
from smzdm_notice.feishu.card_v2 import (
    Card,
    body_elements,
    build_card,
    button,
    callback_value,
    card_component_count,
    card_json_size_bytes,
    column_set,
    element_id,
    form,
    input_box,
    iter_components,
)
from smzdm_notice.feishu.media import get_feishu_image_key
from smzdm_notice.feishu.sdk import (
    get_cardkit_content_models,
    get_cardkit_create_models,
    get_cardkit_settings_models,
    get_cardkit_update_models,
    get_file_models,
    get_lark_client,
    get_message_models,
    get_message_update_models,
    get_reply_message_models,
)
from smzdm_notice.llm.models import ArbiterInfo
from smzdm_notice.preferences.models import DraftBuildOutcome
from smzdm_notice.preferences.preview import build_draft_preview_content
from smzdm_notice.smzdm.ranking import RankingItem

_BINDING_STORE = FeishuBindingStore()
ARBITRATION_CARD_KIND = "arbitration"
ARBITRATION_CARD_METADATA_KEY = "arbitration_card"
_DIGEST_PREVIEW_LIMIT = 20
_DIGEST_ATTACHMENT_FORMAT = "markdown"
_DIGEST_ATTACHMENT_EXTENSIONS = {"markdown": "md"}
MessageId = str
_DEAL_CARD_CACHE_MAX = 100
_DEAL_CARD_CACHE: OrderedDict[str, Card] = OrderedDict()
NOT_WORTH_REASON_FIELD = "not_worth_reason"
NOT_WORTH_REASON_PLACEHOLDER = "可选：价格一般 / 已有类似 / 非刚需 / 品类不合适"
DRAFT_STREAMING_ELEMENT_ID = "draft_progress_text"
DEAL_CARD_COMPONENT_BUDGET = 180
DEAL_CARD_JSON_BUDGET_BYTES = 28_000


@dataclass(frozen=True)
class DealSendResult:
    """一轮商品卡片发送结果，用于精确持久化已送达商品。"""

    delivered_article_ids: tuple[str, ...] = ()
    failed_article_ids: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.delivered_article_ids)


def _current_binding() -> FeishuBinding | None:
    binding = _BINDING_STORE.get()
    if not binding:
        logger.warning("飞书通知目标尚未绑定，请先私聊机器人发送 /bind")
    return binding


def _send_card_message_id(card: Card) -> MessageId | None:
    """发送卡片到当前绑定目标，成功返回 message_id。"""
    binding = _current_binding()
    if not binding:
        return None
    return _send_card_to_message_id(binding.receive_id_type, binding.receive_id, card)


def _send_card_success(card: Card) -> bool:
    """发送卡片到当前绑定目标，只返回是否成功。"""
    return _send_card_message_id(card) is not None


def _do_reply_message(message_id: str, msg_type: str, content: str) -> MessageId | None:
    """回复指定消息，成功返回回复 message_id。"""
    if not message_id:
        return None
    try:
        ReplyMessageRequest, ReplyMessageRequestBody = get_reply_message_models()
        request = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(ReplyMessageRequestBody.builder().msg_type(msg_type).content(content).build())
            .build()
        )
        response = get_lark_client().im.v1.message.reply(request)
        if response.success():
            msg_id = str(getattr(response.data, "message_id", "") or "")
            logger.info(f"飞书回复消息发送成功: {msg_id}")
            return msg_id or None
        logger.error(f"飞书回复消息发送失败: code={response.code}, msg={response.msg}")
        return None
    except Exception as e:
        logger.error(f"飞书回复消息发送异常: {e}")
        return None


def reply_text(message_id: str, text: str) -> bool:
    """回复一条文本消息。"""
    content = json.dumps({"text": text}, ensure_ascii=False)
    return _do_reply_message(message_id, "text", content) is not None


def reply_card(message_id: str, card: Card) -> MessageId | None:
    """回复一张交互卡片，成功返回回复 message_id。"""
    return _do_reply_message(message_id, "interactive", json.dumps(card, ensure_ascii=False))


def reply_card_entity(message_id: str, card_id: str) -> MessageId | None:
    """在线程中回复一个 CardKit 卡片实体。"""
    content = json.dumps({"type": "card", "data": {"card_id": card_id}}, ensure_ascii=False)
    return _do_reply_message(message_id, "interactive", content)


def update_card_message(message_id: str, card: Card) -> bool:
    """更新一条已发送的交互卡片消息。"""
    if not message_id:
        return False
    try:
        PatchMessageRequest, PatchMessageRequestBody = get_message_update_models()
        request = (
            PatchMessageRequest.builder()
            .message_id(message_id)
            .request_body(PatchMessageRequestBody.builder().content(json.dumps(card, ensure_ascii=False)).build())
            .build()
        )
        response = get_lark_client().im.v1.message.patch(request)
        if response.success():
            logger.info(f"飞书卡片已更新: {message_id}")
            return True
        logger.warning(f"飞书卡片更新失败: code={response.code}, msg={response.msg}")
        return False
    except Exception as e:
        logger.warning(f"飞书卡片更新异常: {e}")
        return False


def _create_message(receive_id_type: str, receive_id: str, msg_type: str, content: str):
    """统一构造并发送飞书消息，调用方负责解释不同消息类型的返回语义。"""
    CreateMessageRequest, CreateMessageRequestBody = get_message_models()
    request = (
        CreateMessageRequest.builder()
        .receive_id_type(receive_id_type)
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(receive_id)
            .msg_type(msg_type)
            .content(content)
            .build()
        )
        .build()
    )
    return get_lark_client().im.v1.message.create(request)


def send_text_to(receive_id_type: str, receive_id: str, text: str) -> bool:
    try:
        response = _create_message(
            receive_id_type,
            receive_id,
            "text",
            json.dumps({"text": text}, ensure_ascii=False),
        )
        if response.success():
            return True
        logger.error(f"飞书文本消息发送失败: code={response.code}, msg={response.msg}")
        return False
    except Exception as e:
        logger.error(f"飞书文本消息发送异常: {e}")
        return False


def _upload_file(file_name: str, content: bytes, file_type: str = "stream") -> str:
    """上传文件到飞书，成功返回 file_key。"""
    if not content:
        raise ValueError("文件内容为空")
    CreateFileRequest, CreateFileRequestBody = get_file_models()
    request = (
        CreateFileRequest.builder()
        .request_body(
            CreateFileRequestBody.builder().file_type(file_type).file_name(file_name).file(BytesIO(content)).build()
        )
        .build()
    )
    response = get_lark_client().im.v1.file.create(request)
    if response.success():
        file_key = str(getattr(response.data, "file_key", "") or "")
        if file_key:
            return file_key
        raise ValueError("飞书文件上传成功但未返回 file_key")
    raise RuntimeError(f"飞书文件上传失败: code={response.code}, msg={response.msg}")


def _send_file_to(receive_id_type: str, receive_id: str, file_key: str) -> bool:
    try:
        response = _create_message(
            receive_id_type,
            receive_id,
            "file",
            json.dumps({"file_key": file_key}, ensure_ascii=False),
        )
        if response.success():
            logger.info(f"飞书文件消息发送成功: {getattr(response.data, 'message_id', '')}")
            return True
        logger.error(f"飞书文件消息发送失败: code={response.code}, msg={response.msg}")
        return False
    except Exception as e:
        logger.error(f"飞书文件消息发送异常: {e}")
        return False


def _send_card_to_message_id(receive_id_type: str, receive_id: str, card: Card) -> MessageId | None:
    """底层发送卡片消息，成功返回 message_id，失败返回 None。"""
    try:
        response = _create_message(
            receive_id_type,
            receive_id,
            "interactive",
            json.dumps(card, ensure_ascii=False),
        )
        if response.success():
            msg_id = str(getattr(response.data, "message_id", "") or "")
            logger.info(f"飞书应用消息发送成功: {msg_id}")
            return msg_id or None
        logger.error(f"飞书应用消息发送失败: code={response.code}, msg={response.msg}")
        return None
    except Exception as e:
        logger.error(f"飞书应用消息发送异常: {e}")
        return None


def send_deals(
    items: list[tuple[RankingItem, str]],
    price_bypass_article_ids: set[str] | None = None,
    notification_names_by_article_id: dict[str, str] | None = None,
) -> DealSendResult:
    """推送匹配到的好价商品。"""
    if not items:
        return DealSendResult()
    price_bypass_article_ids = price_bypass_article_ids or set()
    notification_names = notification_names_by_article_id or {}
    prepared = _prepare_deal_items(items, price_bypass_article_ids)
    chunks = _pack_prepared_deals(prepared, notification_names)
    delivered: list[str] = []
    failed: list[str] = []
    sent_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    for chunk in chunks:
        chunk_items = [match for match, _ in chunk]
        card = _build_prepared_deals_card(chunk, notification_names, sent_at=sent_at)
        article_ids = [item.article_id for item, _ in chunk_items]
        if not _deal_card_within_budget(card):
            logger.error(f"单件商品卡片仍超过飞书限制，跳过发送: article_ids={article_ids}")
            failed.extend(article_ids)
            continue
        message_id = _send_card_message_id(card)
        if not message_id:
            failed.extend(article_ids)
            continue
        delivered.extend(article_ids)
        _cache_deal_card_snapshot(message_id, card)
    if failed:
        logger.warning(f"好价卡片部分发送失败: delivered={len(delivered)}, failed={len(failed)}")
    return DealSendResult(tuple(delivered), tuple(failed))


def _prepare_deal_items(
    items: list[tuple[RankingItem, str]],
    price_bypass_article_ids: set[str],
) -> list[tuple[tuple[RankingItem, str], list[Card]]]:
    """预构建每件商品的元素，拆卡计算时不重复上传图片。"""
    return [
        (
            (item, reason),
            _build_deal_item_elements(item, reason, item.article_id in price_bypass_article_ids),
        )
        for item, reason in items
    ]


def _pack_prepared_deals(
    prepared: list[tuple[tuple[RankingItem, str], list[Card]]],
    notification_names_by_article_id: dict[str, str],
) -> list[list[tuple[tuple[RankingItem, str], list[Card]]]]:
    """按飞书组件和体积限制顺序拆分商品卡片。"""
    chunks: list[list[tuple[tuple[RankingItem, str], list[Card]]]] = []
    current: list[tuple[tuple[RankingItem, str], list[Card]]] = []
    for entry in prepared:
        candidate = [*current, entry]
        candidate_card = _build_prepared_deals_card(candidate, notification_names_by_article_id)
        if current and not _deal_card_within_budget(candidate_card):
            chunks.append(current)
            current = [entry]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _deal_card_within_budget(card: Card) -> bool:
    return (
        card_component_count(card) <= DEAL_CARD_COMPONENT_BUDGET
        and card_json_size_bytes(card) <= DEAL_CARD_JSON_BUDGET_BYTES
    )


def _build_prepared_deals_card(
    prepared: list[tuple[tuple[RankingItem, str], list[Card]]],
    notification_names_by_article_id: dict[str, str],
    *,
    sent_at: str | None = None,
) -> Card:
    items = [match for match, _ in prepared]
    elements = [
        {
            "tag": "markdown",
            "content": f"🔥 发现 **{len(items)}** 件好价商品！\n📅 {sent_at or datetime.now().strftime('%Y-%m-%d %H:%M')}",
        },
        {"tag": "hr"},
    ]
    for _, item_elements in prepared:
        elements.extend(item_elements)
    return build_card(
        "🛒 什么值得买 · 好价推荐",
        "red",
        elements,
        summary=_build_deals_summary(items, notification_names_by_article_id),
    )


def _build_deals_summary(
    items: list[tuple[RankingItem, str]],
    notification_names_by_article_id: dict[str, str],
) -> str:
    item_count = len(items)
    counts: OrderedDict[str, int] = OrderedDict()
    for item, _ in items:
        name = _notification_summary_name(notification_names_by_article_id.get(item.article_id))
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1

    labels = [f"{name}×{count}" if count > 1 else name for name, count in counts.items()]
    if not labels:
        return f"推荐了 {item_count} 个商品"

    prefix = "好价："
    full = prefix + "、".join(labels)
    named_item_count = sum(counts.values())
    if named_item_count == item_count and len(full) <= 80:
        return full

    suffix = f"等 {item_count} 件"
    included: list[str] = []
    for label in labels:
        candidate = prefix + "、".join([*included, label]) + suffix
        if len(candidate) > 80:
            break
        included.append(label)
    if included:
        return prefix + "、".join(included) + suffix
    return f"推荐了 {item_count} 个商品"


def _notification_summary_name(value: object) -> str:
    """清理显式通知名称；价格直推搜索词允许截短到摘要名称上限。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip(" 、，,|-")
    return _compact_table_text(text, 16)


def _build_deal_item_elements(item: RankingItem, reason: str, is_price_bypass: bool) -> list[Card]:
    """构造单个好价商品行，包含长期偏好反馈和当前偏好操作。"""
    image_key = get_feishu_image_key(item.pic) if item.pic else ""
    elements: list[Card] = [{"tag": "markdown", "content": "\n".join(_deal_markdown_lines(item, reason, image_key))}]
    if image_key:
        elements.append(_deal_image_element(item, image_key))
    # 第一行：偏好反馈（好价/不值）+ 查看详情
    memory_enabled = config.DEAL_MEMORY_ENABLED and not is_price_bypass
    memory_container = _memory_action_container(item, enabled=memory_enabled)
    if memory_container:
        elements.append(memory_container)
    # 第二行：当前需求操作
    elements.append(column_set(_config_action_buttons(item, is_price_bypass), component_id=element_id("config", item.article_id)))
    elements.append({"tag": "hr"})
    return elements


def _memory_action_container(item: RankingItem, enabled: bool = True) -> Card | None:
    """长期偏好反馈按钮。"""
    value = _button_value(item)
    return _memory_action_container_from_value(value, item.link, enabled=enabled)


def _memory_action_container_from_value(
    base_value: Card,
    item_link: str = "",
    enabled: bool = True,
    selected: str = "",
    reason: str = "",
) -> Card | None:
    elements = _memory_action_buttons_from_value(
        base_value,
        item_link,
        enabled=enabled,
        selected=selected,
        reason=reason,
    )
    if not elements:
        return None

    article_id = str(base_value.get("article_id") or "item")
    component_id = element_id("memory", article_id)
    if enabled and selected == "deal_not_worth":
        return form(
            elements,
            name=element_id("memory_form", article_id),
            component_id=component_id,
        )

    container = elements[0]
    container["element_id"] = component_id
    return container


def _memory_action_buttons_from_value(
    base_value: Card,
    item_link: str = "",
    enabled: bool = True,
    selected: str = "",
    reason: str = "",
) -> list[Card]:
    """长期偏好反馈按钮。选中后只显示当前选中项，再次点击取消恢复两个。"""
    buttons: list[Card] = []
    fields: list[Card] = []
    reason_field = element_id("reason", str(base_value.get("article_id") or "item"))
    if enabled:
        if selected == "deal_good":
            buttons.append(_button_from_value("✅ 好价", "deal_good", base_value, "primary"))
        elif selected == "deal_not_worth":
            buttons.append(_button_from_value("❌ 不值", "deal_not_worth", base_value, "danger"))
            fields.append(
                input_box(
                    reason_field,
                    NOT_WORTH_REASON_PLACEHOLDER,
                    default_value=reason,
                    value={**base_value, "field": NOT_WORTH_REASON_FIELD, "reason_field": reason_field},
                )
            )
            save_value = {**base_value, "reason_field": reason_field}
            buttons.append(
                _button_from_value(
                    "保存理由",
                    "deal_not_worth_reason",
                    save_value,
                    "default",
                    action_type="form_submit",
                )
            )
        else:
            buttons.append(_button_from_value("好价👍", "deal_good", base_value, "primary"))
            buttons.append(_button_from_value("不值👎", "deal_not_worth", base_value, "danger"))
    link = item_link or str(base_value.get("item_link") or "")
    if link:
        buttons.append(button("查看详情", url=link))
    if buttons:
        fields.append(column_set(buttons))
    return fields


def update_deal_card_feedback_state(
    message_id: str,
    article_id: str,
    selected: str = "",
    reason: str = "",
) -> Card | None:
    """用 PATCH 更新已缓存的好价卡片快照，反映用户的 Deal Memory 反馈。

    该缓存刻意保持为进程内状态：它只用于 PATCH 当前进程早先发出的卡片，
    持久反馈仍由 DealMemory 保存。
    """
    if not message_id or not article_id:
        return None
    cached = _DEAL_CARD_CACHE.get(message_id)
    if not cached:
        return None

    updated = deepcopy(cached)
    if not _apply_deal_feedback_state_to_card(updated, article_id, selected, reason=reason):
        return None
    update_card_message(message_id, updated)  # 尽力 PATCH，失败不影响已记录的反馈
    _cache_deal_card_snapshot(message_id, updated)
    return updated


def _apply_deal_feedback_state_to_card(card: Card, article_id: str, selected: str, reason: str = "") -> bool:
    """原地重写匹配商品的反馈容器和相邻状态文本。"""
    elements = body_elements(card)
    for idx, element in enumerate(elements):
        base_value = _memory_row_value(element, article_id)
        if base_value is None:
            continue
        replacement = _memory_action_container_from_value(
            base_value,
            enabled=True,
            selected=selected,
            reason=_clean_feedback_reason(reason),
        )
        if replacement is None:
            return False
        elements[idx] = replacement
        _update_preceding_markdown(elements, idx, _feedback_status_text(selected, reason=reason))
        return True
    return False


_FEEDBACK_STATUS_RE = re.compile(
    r"\n\n> (✅ 你标记为 \*\*好价\*\*|❌ 你标记为 \*\*不值\*\*)[^\n]*$"
)


def _feedback_status_text(selected: str, reason: str = "") -> str:
    """生成反馈状态行文本。"""
    if selected == "deal_good":
        return "✅ 你标记为 **好价**（再次点击可取消）"
    if selected == "deal_not_worth":
        reason_text = _clean_feedback_reason(reason)
        if reason_text:
            return f"❌ 你标记为 **不值**：{reason_text}（再次点击可取消）"
        return "❌ 你标记为 **不值**（再次点击可取消）"
    return ""


def _clean_feedback_reason(reason: str) -> str:
    text = re.sub(r"\s+", " ", str(reason or "")).strip()
    if len(text) > 60:
        return text[:57].rstrip() + "..."
    return text


def _strip_feedback_status_line(content: str) -> str:
    """移除已有的反馈状态行，仅匹配我们注入的 blockquote。"""
    return _FEEDBACK_STATUS_RE.sub("", content)


def _update_preceding_markdown(elements: list[Card], action_idx: int, feedback_status: str) -> None:
    """更新 action 元素前方的 markdown 元素，注入或清除反馈状态行。"""
    for i in range(action_idx - 1, -1, -1):
        el = elements[i]
        if el.get("tag") != "markdown":
            continue
        content = str(el.get("content", ""))
        content = _strip_feedback_status_line(content)
        if feedback_status:
            content = content.rstrip("\n") + f"\n\n> {feedback_status}"
        el["content"] = content
        return


def _memory_row_value(container: Card, article_id: str) -> Card | None:
    for action in iter_components(container):
        value = callback_value(action)
        if value is None:
            continue
        if str(value.get("article_id") or "") != article_id:
            continue
        if value.get("action") in {"deal_good", "deal_not_worth", "deal_not_worth_reason"}:
            base = dict(value)
            base.pop("action", None)
            return base
    return None


def _cache_deal_card_snapshot(message_id: str, card: Card) -> None:
    """记住最近发送的卡片正文，便于后续反馈操作 PATCH。"""
    _DEAL_CARD_CACHE[message_id] = deepcopy(card)
    _DEAL_CARD_CACHE.move_to_end(message_id)
    while len(_DEAL_CARD_CACHE) > _DEAL_CARD_CACHE_MAX:
        _DEAL_CARD_CACHE.popitem(last=False)


def _button_value(item: RankingItem) -> Card:
    value: Card = {
        "item_title": item.title,
        "item_brand": item.brand,
        "item_link": item.link,
        "article_id": item.article_id,
    }
    if item.search_keyword:
        value["search_keyword"] = item.search_keyword
    if item.search_max_price is not None:
        value["search_max_price"] = item.search_max_price
    return value


def _button_from_value(
    label: str,
    action: str,
    base_value: Card,
    button_type: str,
    *,
    action_type: str = "",
) -> dict:
    value = dict(base_value)
    value["action"] = action
    return button(label, button_type=button_type, value=value, action_type=action_type)


def _config_action_buttons(item: RankingItem, is_price_bypass: bool) -> list[Card]:
    """当前需求操作按钮。"""
    if is_price_bypass:
        return [
            _button("移除搜索词", "search_remove_keyword", item, "danger"),
            _button("清除价格阈值", "search_clear_price", item, "default"),
        ]
    return [
        _button("不再推荐", "deal_ignore_category", item, "danger"),
        _button("库存足够", "deal_stock_enough", item, "default"),
        _button("关注", "deal_follow", item, "default"),
    ]


def _deal_markdown_lines(item: RankingItem, reason: str, image_key: str) -> list[str]:
    tags_str = " ".join(f"`{tag}`" for tag in item.tags) if item.tags else ""
    lines = [
        f"**{item.title}**",
        f"💰 **{item.price}** | 🏪 {item.mall} | 🏷️ {item.brand}",
        f"👍 值 {item.worthy} / 👎 不值 {item.unworthy} | 💬 {item.comments} | ⭐ {item.favorites}",
    ]
    if tags_str:
        lines.append(f"🏅 {tags_str}")
    lines.append(f"📋 **推荐理由**: {reason}")
    lines.append(f"📊 来源: [{item.tab_name}榜 #{item.rank}]")
    if item.pic and not image_key:
        lines.append(f"🖼️ [查看商品图片]({item.pic})")
    return lines


def _deal_image_element(item: RankingItem, image_key: str) -> Card:
    return {
        "tag": "img",
        "img_key": image_key,
        "alt": {"tag": "plain_text", "content": _compact_table_text(item.title, 60)},
    }


def send_heartbeat(hours: int) -> bool:
    """发送心跳消息。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return _send_card_success(
        build_card(
            "🤖 好价监控 · 心跳",
            "blue",
            [
                {
                    "tag": "markdown",
                    "content": f"✅ 好价监控运行正常\n⏰ {now}\n📢 最近 **{hours} 小时**未发现匹配商品\n💡 机器人将持续监控，发现好价立即推送",
                }
            ],
        )
    )


def send_shutdown(reason: str = "") -> bool:
    """发送停止通知。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    reason_text = f"\n📌 原因: {reason}" if reason else ""
    return _send_card_success(
        build_card(
            "⛔ 好价监控 · 已停止",
            "red",
            [{"tag": "markdown", "content": f"📅 {now}{reason_text}\n\n如需恢复，请重新启动程序"}],
        )
    )


def send_startup(config_summary: str) -> bool:
    """发送启动通知。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return _send_card_success(
        build_card(
            "🚀 好价监控 · 启动",
            "green",
            [{"tag": "markdown", "content": f"📅 {now}\n\n{config_summary}"}],
        )
    )


def send_config_warning(message: str) -> bool:
    """发送运行时配置读取告警。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return _send_card_success(
        build_card(
            "⚠️ 好价监控 · 配置读取告警",
            "orange",
            [
                {
                    "tag": "markdown",
                    "content": f"📅 {now}\n\n{message}\n\n将继续使用上一次成功读取的内容。",
                }
            ],
        )
    )


def build_help_card(help_content: str) -> Card:
    """构造快捷命令帮助卡片。"""
    return build_card(
        "好价监控 · 快捷命令",
        "blue",
        [{"tag": "markdown", "content": help_content}],
    )


def send_help(help_content: str, reply_to_message_id: str = "") -> bool:
    """以卡片形式发送快捷命令帮助。"""
    card = build_help_card(help_content)
    if reply_to_message_id:
        msg_id = reply_card(reply_to_message_id, card)
        if msg_id:
            return True
    return _send_card_success(card)


def send_poll_failure_warning(count: int, reason: str, detail: str | None = None) -> bool:
    """发送连续轮询失败告警。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    reason_text = {
        "ranking_fetch_failed": "榜单抓取失败",
        "llm_failed": "LLM 调用失败",
    }.get(reason, reason or "未知失败")
    detail_text = _sanitize_warning_detail(detail) if detail else "无详细错误信息"
    return _send_card_success(
        build_card(
            "⚠️ 好价监控 · 轮询失败告警",
            "red",
            [
                {
                    "tag": "markdown",
                    "content": (
                        f"📅 {now}\n\n"
                        f"连续 **{count}** 次轮询失败。\n\n"
                        f"- 最近失败类型：{reason_text}\n"
                        f"- 最近错误：`{detail_text}`\n\n"
                        "请检查榜单网络访问、LLM 额度或 API 配置。"
                    ),
                }
            ],
        )
    )


def send_arbitration(
    arbiter_info: ArbiterInfo,
    draft: Any | None = None,
    draft_outcome: DraftBuildOutcome | None = None,
) -> bool:
    """推送仲裁分析结果，附带一键采纳按钮。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    snapshot = _build_arbitration_card_snapshot(arbiter_info, now)
    if draft_outcome is not None:
        snapshot["draft_outcome_status"] = str(getattr(draft_outcome, "status", "") or "")
        snapshot["draft_outcome_message"] = str(getattr(draft_outcome, "message", "") or "")
    if draft and hasattr(draft, "metadata"):
        draft.metadata[ARBITRATION_CARD_METADATA_KEY] = snapshot
        draft.metadata["card_kind"] = ARBITRATION_CARD_KIND
    msg_id = _send_card_message_id(build_arbitration_card(snapshot, draft))
    if msg_id and draft and hasattr(draft, "preview_message_id"):
        draft.preview_message_id = msg_id
    return msg_id is not None


def _build_arbitration_card_snapshot(arbiter_info: ArbiterInfo, sent_at: str) -> dict:
    """提取仲裁卡片展示所需的稳定快照。"""
    recs_a = {r.id: r for r in arbiter_info.result_a.recommendations}
    recs_b = {r.id: r for r in arbiter_info.result_b.recommendations}
    diff_text = _format_arbitration_diffs(
        only_a=set(recs_a) - set(recs_b),
        only_b=set(recs_b) - set(recs_a),
        recs_a=recs_a,
        recs_b=recs_b,
        items=arbiter_info.items,
    )
    return {
        "sent_at": sent_at,
        "diff_text": diff_text,
        "chosen": arbiter_info.chosen,
        "reason": arbiter_info.reason,
        "analysis": arbiter_info.analysis,
        "suggestion": arbiter_info.suggestion,
        "change_assessment": arbiter_info.change_assessment,
    }


def _build_arbitration_content(snapshot: dict) -> str:
    assessment = snapshot.get("change_assessment") or {}
    assessment_block = ""
    if assessment:
        cause_labels = {
            "preference_gap": "存在真实偏好缺口",
            "filter_prompt_gap": "筛选 Prompt 说明不足",
            "model_execution_error": "模型单次执行错误",
            "soft_judgment": "软信号判断差异",
        }
        assessment_text = cause_labels.get(str(assessment.get("cause") or ""), "未分类")
        preference_note = (
            "建议修改 preference.md"
            if assessment.get("should_change_preference") is True
            else "本次不建议修改 preference.md"
        )
        assessment_block = f"**差异归因：** {assessment_text}；{preference_note}\n\n"
    return (
        f"📅 {snapshot.get('sent_at', '')}\n\n"
        f"**两次判断不一致，已仲裁**\n\n"
        f"**差异商品：**\n{snapshot.get('diff_text', '')}\n\n"
        f"**仲裁选择：** 判断 {snapshot.get('chosen', '')}\n"
        f"**原因：** {snapshot.get('reason', '')}\n\n"
        f"**不一致分析：**\n{snapshot.get('analysis', '')}\n\n"
        f"{assessment_block}"
        f"**Prompt 优化建议：**\n{snapshot.get('suggestion', '')}"
    )


def build_arbitration_card(
    snapshot: dict,
    draft: Any | None = None,
    disabled_reason: str = "",
) -> Card:
    """构造仲裁分析卡片；disabled_reason 非空时移除按钮并显示失效原因。"""
    elements = _arbitration_elements(snapshot, draft, disabled_reason)
    return build_card(
        "⚖️ 好价监控 · 仲裁分析" + ("（已失效）" if disabled_reason else ""),
        "grey" if disabled_reason else "purple",
        elements,
    )


def _arbitration_elements(snapshot: dict, draft: Any | None, disabled_reason: str) -> list[Card]:
    elements: list[Card] = [{"tag": "markdown", "content": _build_arbitration_content(snapshot)}]
    actions = _arbitration_actions(draft, disabled_reason)
    if draft:
        elements.extend([{"tag": "hr"}, {"tag": "markdown", "content": build_draft_preview_content(draft)}])
    elif not disabled_reason:
        assessment = snapshot.get("change_assessment") or {}
        outcome_status = str(snapshot.get("draft_outcome_status") or "")
        if outcome_status == "noop":
            outcome_message = str(snapshot.get("draft_outcome_message") or "").strip()
            no_change_text = f"ℹ️ {outcome_message or '当前 preference.md 已覆盖该候选规则，无需修改。'}"
        elif not assessment:
            no_change_text = "⚠️ 本次未生成可直接采纳的配置修改，请按需手动调整偏好文件。"
        elif assessment.get("should_change_preference") is not True:
            no_change_text = "ℹ️ 本次差异不代表缺少用户偏好，因此不生成 preference.md 修改。"
        else:
            no_change_text = "⚠️ 已识别偏好缺口，但未能生成通过安全校验的修改草案。"
        elements.append(
            {
                "tag": "markdown",
                "content": no_change_text,
            }
        )
    if disabled_reason:
        elements.extend([{"tag": "hr"}, {"tag": "markdown", "content": f"**状态：已失效**\n\n原因：{disabled_reason}"}])
    elif actions:
        elements.append(column_set(actions, component_id=element_id("arbiter_actions", str(getattr(draft, "draft_id", "")))))
    return elements


def _arbitration_actions(draft: Any | None, disabled_reason: str) -> list[Card]:
    if disabled_reason:
        return []
    actions: list[Card] = []
    if draft:
        actions.append(
            button(
                "采纳并更新",
                button_type="primary",
                value={
                    "action": "apply_draft",
                    "draft_id": draft.draft_id,
                    "card_kind": ARBITRATION_CARD_KIND,
                },
            )
        )
    actions.append(
        button(
            "忽略",
            value={
                "action": "ignore_arbitration",
                "draft_id": getattr(draft, "draft_id", ""),
                "card_kind": ARBITRATION_CARD_KIND,
            },
        )
    )
    return actions


def send_digest(entries: list[dict], digest_date: str) -> bool:
    """发送夜间汇总消息。"""
    if not entries:
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    count = len(entries)
    elements = [
        {"tag": "markdown", "content": f"📋 本日共 **{count}** 件 near-miss 商品\n📅 {now}"},
    ]

    table_rows = ["| 商品 | 价格 | 热度 | 跳过原因 |", "| --- | ---: | --- | --- |"]
    for entry in entries[:_DIGEST_PREVIEW_LIMIT]:
        title = _compact_table_text(entry.get("title", "未知商品"), 28)
        if entry.get("link"):
            title = f"[{title}]({entry['link']})"
        price = _compact_table_text(entry.get("price", "-"), 10)
        heat = _compact_table_text(
            f"{entry.get('worthy', 0)}/{entry.get('unworthy', 0)} · {entry.get('comments', 0)}评",
            18,
        )
        reason = _compact_table_text(_strip_skip_reason_prefix(entry.get("skip_reason", "")), 36)
        table_rows.append(f"| {title} | {price} | {heat} | {reason} |")

    elements.append({"tag": "markdown", "content": "\n".join(table_rows)})
    needs_attachment = count > _DIGEST_PREVIEW_LIMIT
    if needs_attachment:
        elements.append(
            {
                "tag": "markdown",
                "content": f"...以及其他 {count - _DIGEST_PREVIEW_LIMIT} 件商品（已省略，完整内容见附件）",
            }
        )

    card = build_card(
        f"🌙 好价监控 · 夜间汇总 ({digest_date})",
        "orange",
        elements,
    )

    if not needs_attachment:
        return _send_card_success(card)

    binding = _current_binding()
    if not binding:
        return False
    try:
        file_name, content = _build_digest_attachment(entries, digest_date, _DIGEST_ATTACHMENT_FORMAT)
        file_key = _upload_file(file_name, content)
    except Exception as e:
        logger.error(f"夜间汇总附件上传失败: {e}")
        return False

    card_sent = _send_card_to_message_id(binding.receive_id_type, binding.receive_id, card) is not None
    if not card_sent:
        return False
    return _send_file_to(binding.receive_id_type, binding.receive_id, file_key)


def _build_digest_attachment(entries: list[dict], digest_date: str, format: str = "markdown") -> tuple[str, bytes]:
    extension = _DIGEST_ATTACHMENT_EXTENSIONS.get(format)
    if not extension:
        raise ValueError(f"不支持的夜间汇总附件格式: {format}")
    if format == "markdown":
        content = _format_digest_markdown(entries, digest_date)
    else:
        raise ValueError(f"不支持的夜间汇总附件格式: {format}")
    return f"smzdm_digest_{digest_date}.{extension}", content.encode("utf-8")


def _format_digest_markdown(entries: list[dict], digest_date: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        f"# 什么值得买夜间汇总 ({digest_date})",
        "",
        f"- 生成时间: {now}",
        f"- near-miss 商品数: {len(entries)}",
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        title = str(entry.get("title") or "未知商品").strip()
        link = str(entry.get("link") or "").strip()
        tags = entry.get("tags") or []
        tag_text = " ".join(str(tag) for tag in tags if str(tag).strip()) if isinstance(tags, list) else str(tags)
        reason = _strip_skip_reason_prefix(entry.get("skip_reason", ""))

        parts.extend(
            [
                f"## {index}. {title}",
                "",
                f"- 链接: {link or '-'}",
                f"- 价格: {entry.get('price') or '-'}",
                f"- 商城: {entry.get('mall') or '-'}",
                f"- 品牌: {entry.get('brand') or '-'}",
                (
                    f"- 热度: 值 {entry.get('worthy', 0)} / 不值 {entry.get('unworthy', 0)}"
                    f" / 评论 {entry.get('comments', 0)} / 收藏 {entry.get('favorites', 0)}"
                ),
                f"- 榜单: {entry.get('tab_name') or '-'} #{entry.get('rank') or '-'}",
                f"- 标签: {tag_text or '-'}",
                f"- 跳过原因: {reason or '-'}",
                "",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def build_draft_preview_card(draft: Any) -> Card:
    """构造带确认/取消按钮的偏好/库存修改预览卡片。"""
    return build_card(
        "📝 配置修改预览",
        "blue",
        [
            {"tag": "markdown", "content": build_draft_preview_content(draft)},
            column_set(
                [
                    button(
                        "确认应用",
                        button_type="primary",
                        value={"action": "apply_draft", "draft_id": draft.draft_id},
                    ),
                    button(
                        "取消",
                        button_type="danger",
                        value={"action": "cancel_draft", "draft_id": draft.draft_id},
                    ),
                ],
                component_id=element_id("draft_actions", str(draft.draft_id)),
            ),
        ],
    )


def build_draft_processing_card(stage: str = "正在生成配置修改预览") -> Card:
    """构造无按钮的草案生成处理中卡片。"""
    return build_card(
        "📝 配置修改预览生成中",
        "blue",
        [
            {
                "tag": "markdown",
                "content": f"正在生成配置修改预览，请稍候。\n\n当前阶段：{stage}",
            },
        ],
        summary="正在生成配置修改预览",
    )


def build_streaming_draft_processing_card(stage: str) -> Card:
    """构造启用 CardKit 流式模式、只有固定 Markdown 元素的进度卡片。"""
    return build_card(
        "📝 配置修改预览生成中",
        "blue",
        [{"tag": "markdown", "element_id": DRAFT_STREAMING_ELEMENT_ID, "content": stage}],
        summary="正在生成配置修改预览",
        streaming_mode=True,
    )


def start_streaming_draft_processing(stage: str, reply_to_message_id: str) -> tuple[str, str] | None:
    """创建并回复 CardKit 流式卡片，返回 (message_id, card_id)。"""
    if not reply_to_message_id:
        return None
    card_id = _create_cardkit_card(build_streaming_draft_processing_card(stage))
    if not card_id:
        return None
    message_id = reply_card_entity(reply_to_message_id, card_id)
    if not message_id:
        logger.warning(f"CardKit 卡片实体已创建但发送失败，将回退普通卡片: {card_id}")
        return None
    return message_id, card_id


def _create_cardkit_card(card: Card) -> str:
    try:
        CreateCardRequest, CreateCardRequestBody = get_cardkit_create_models()
        request = (
            CreateCardRequest.builder()
            .request_body(
                CreateCardRequestBody.builder()
                .type("card_json")
                .data(json.dumps(card, ensure_ascii=False))
                .build()
            )
            .build()
        )
        response = get_lark_client().cardkit.v1.card.create(request)
        if response.success():
            card_id = str(getattr(response.data, "card_id", "") or "")
            if card_id:
                return card_id
        logger.warning(f"CardKit 卡片实体创建失败: code={response.code}, msg={response.msg}")
    except Exception as e:
        logger.warning(f"CardKit 卡片实体创建异常，将回退普通卡片: {e}")
    return ""


def update_streaming_draft_content(card_id: str, content: str, sequence: int) -> bool:
    """按 sequence 流式更新固定 Markdown 元素。"""
    try:
        ContentRequest, ContentRequestBody = get_cardkit_content_models()
        request = (
            ContentRequest.builder()
            .card_id(card_id)
            .element_id(DRAFT_STREAMING_ELEMENT_ID)
            .request_body(
                ContentRequestBody.builder()
                .uuid(str(uuid.uuid4()))
                .content(content)
                .sequence(sequence)
                .build()
            )
            .build()
        )
        response = get_lark_client().cardkit.v1.card_element.content(request)
        if response.success():
            return True
        logger.warning(f"CardKit 流式文本更新失败: code={response.code}, msg={response.msg}")
    except Exception as e:
        logger.warning(f"CardKit 流式文本更新异常: {e}")
    return False


def finish_streaming_draft_card(card_id: str, card: Card, close_sequence: int, update_sequence: int) -> bool:
    """先关闭流式模式，再整体替换为最终交互卡片。"""
    if not _close_cardkit_streaming_mode(card_id, close_sequence):
        return False
    return _update_cardkit_card(card_id, card, update_sequence)


def _close_cardkit_streaming_mode(card_id: str, sequence: int) -> bool:
    try:
        SettingsRequest, SettingsRequestBody = get_cardkit_settings_models()
        request = (
            SettingsRequest.builder()
            .card_id(card_id)
            .request_body(
                SettingsRequestBody.builder()
                .settings(json.dumps({"config": {"streaming_mode": False}}))
                .uuid(str(uuid.uuid4()))
                .sequence(sequence)
                .build()
            )
            .build()
        )
        response = get_lark_client().cardkit.v1.card.settings(request)
        if response.success():
            return True
        logger.warning(f"CardKit 流式模式关闭失败: code={response.code}, msg={response.msg}")
    except Exception as e:
        logger.warning(f"CardKit 流式模式关闭异常: {e}")
    return False


def _update_cardkit_card(card_id: str, card: Card, sequence: int) -> bool:
    try:
        CardModel, UpdateRequest, UpdateRequestBody = get_cardkit_update_models()
        card_model = CardModel.builder().type("card_json").data(json.dumps(card, ensure_ascii=False)).build()
        request = (
            UpdateRequest.builder()
            .card_id(card_id)
            .request_body(
                UpdateRequestBody.builder()
                .card(card_model)
                .uuid(str(uuid.uuid4()))
                .sequence(sequence)
                .build()
            )
            .build()
        )
        response = get_lark_client().cardkit.v1.card.update(request)
        if response.success():
            return True
        logger.warning(f"CardKit 最终卡片更新失败: code={response.code}, msg={response.msg}")
    except Exception as e:
        logger.warning(f"CardKit 最终卡片更新异常: {e}")
    return False


def build_draft_failure_card(reason: str) -> Card:
    """构造无按钮的草案生成失败卡片。"""
    return build_card(
        "📝 配置修改预览生成失败",
        "red",
        [
            {
                "tag": "markdown",
                "content": f"{reason}\n\n请换一种更明确的说法重试，或稍后再试。",
            },
        ],
    )


def build_draft_noop_card(message: str) -> Card:
    """构造无需修改的无按钮信息卡片。"""
    return build_card(
        "📝 当前配置无需修改",
        "blue",
        [{"tag": "markdown", "content": message}],
        summary="当前配置无需修改",
    )


def build_draft_handoff_card() -> Card:
    """流式卡片已由新消息承接时，替换旧卡片的生成中状态。"""
    return build_card(
        "📝 配置修改预览已生成",
        "green",
        [{"tag": "markdown", "content": "已生成新的配置修改预览，请查看后续消息。"}],
        summary="配置修改预览已生成",
    )


def send_draft_processing(stage: str, reply_to_message_id: str = "") -> MessageId | None:
    """发送草案生成处理中卡片，成功返回 message_id。"""
    if not reply_to_message_id:
        return None
    return reply_card(reply_to_message_id, build_draft_processing_card(stage))


def send_draft_preview(draft: Any, reply_to_message_id: str = "") -> bool:
    """发送偏好/库存修改预览卡片，成功时将 message_id 写入 draft.preview_message_id。"""
    card = build_draft_preview_card(draft)
    msg_id = reply_card(reply_to_message_id, card) if reply_to_message_id else None
    if not msg_id:
        msg_id = _send_card_message_id(card)
    if msg_id and hasattr(draft, "preview_message_id"):
        draft.preview_message_id = msg_id
    return msg_id is not None


def update_draft_preview(message_id: str, draft: Any) -> bool:
    """将处理中卡片更新为最终草案预览卡片，成功时写入 draft.preview_message_id。"""
    if update_card_message(message_id, build_draft_preview_card(draft)):
        if hasattr(draft, "preview_message_id"):
            draft.preview_message_id = message_id
        return True
    return False


def update_rebased_draft_preview(message_id: str, draft: Any) -> bool:
    """在原消息上展示冲突刷新后的草案，并保留仲裁卡片上下文。"""
    if _is_arbitration_draft(draft):
        snapshot = draft.metadata.get(ARBITRATION_CARD_METADATA_KEY) or {}
        card = build_arbitration_card(snapshot, draft)
    else:
        card = build_draft_preview_card(draft)
    if not update_card_message(message_id, card):
        return False
    if hasattr(draft, "preview_message_id"):
        draft.preview_message_id = message_id
    return True


def build_disabled_draft_card(reason: str, draft: Any | None = None) -> Card:
    """构造无按钮的失效预览卡片。"""
    content = (
        f"{build_draft_preview_content(draft)}\n\n**状态：已失效**\n\n原因：{reason}"
        if draft
        else f"~~此预览已失效~~\n原因：{reason}"
    )
    return build_card(
        "📝 配置修改预览（已失效）",
        "grey",
        [{"tag": "markdown", "content": content}],
    )


def build_disabled_arbitration_card(reason: str, draft: Any | None = None) -> Card:
    """构造无按钮的失效仲裁卡片，优先保留原仲裁正文。"""
    snapshot = {}
    if draft and hasattr(draft, "metadata"):
        snapshot = draft.metadata.get(ARBITRATION_CARD_METADATA_KEY) or {}
    if snapshot:
        return build_arbitration_card(snapshot, draft, disabled_reason=reason)
    content = f"~~本次仲裁卡片已失效~~\n\n原因：{reason}"
    if draft:
        content = f"{content}\n\n**原配置修改预览：**\n\n{build_draft_preview_content(draft)}"
    return build_card(
        "⚖️ 好价监控 · 仲裁分析（已失效）",
        "grey",
        [{"tag": "markdown", "content": content}],
    )


def disable_draft_card(message_id: str, reason: str, draft: Any | None = None) -> bool:
    """将已发送的预览卡片更新为已失效状态（移除按钮，显示失效原因）。"""
    if _is_arbitration_draft(draft):
        card = build_disabled_arbitration_card(reason, draft)
    else:
        card = build_disabled_draft_card(reason, draft)
    return update_card_message(message_id, card)


def _is_arbitration_draft(draft: Any | None) -> bool:
    if not draft or not hasattr(draft, "metadata"):
        return False
    return draft.metadata.get("card_kind") == ARBITRATION_CARD_KIND or bool(
        draft.metadata.get(ARBITRATION_CARD_METADATA_KEY)
    )


def send_text(text: str) -> bool:
    binding = _current_binding()
    if not binding:
        return False
    return send_text_to(binding.receive_id_type, binding.receive_id, text)


def _button(label: str, action: str, item: RankingItem, button_type: str) -> dict:
    return _button_from_value(label, action, _button_value(item), button_type)


def _format_arbitration_diffs(
    only_a: set[str],
    only_b: set[str],
    recs_a: Mapping[str, object],
    recs_b: Mapping[str, object],
    items: dict[str, dict],
) -> str:
    parts = [
        "仅 A 推荐：",
        _format_arbitration_group(only_a, recs_a, items),
        "",
        "仅 B 推荐：",
        _format_arbitration_group(only_b, recs_b, items),
    ]
    return "\n".join(parts)


def _format_arbitration_group(
    ids: set[str],
    recs: Mapping[str, object],
    items: dict[str, dict],
) -> str:
    if not ids:
        return "- 无"
    return "\n".join(_format_arbitration_item(aid, recs.get(aid), items.get(aid)) for aid in sorted(ids))


def _format_arbitration_item(article_id: str, rec: object | None, item: dict | None) -> str:
    reason = getattr(rec, "reason", "") or "未提供理由"
    if not item:
        return f"- {article_id}\n  理由：{reason}"

    brand = _compact_table_text(item.get("brand"), 16)
    title = _compact_table_text(item.get("title"), 54)
    title_text = f"{brand} {title}".strip() if brand else title
    link = str(item.get("link") or "").strip()
    title_part = f"[{title_text}]({link})" if link else title_text
    price = _compact_table_text(item.get("price"), 18)
    worthy = item.get("worthy", 0)
    unworthy = item.get("unworthy", 0)
    comments = item.get("comments", 0)

    return f"- {article_id}｜{title_part}｜{price}｜值{worthy}/不值{unworthy}｜评{comments}\n  理由：{reason}"


def _compact_table_text(value: object, max_len: int) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "/").strip()
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1]}…"


def _sanitize_warning_detail(detail: object, max_len: int = 500) -> str:
    text = str(detail or "").replace("\n", " ").strip()
    text = re.sub(r"(?i)(api[_-]?key|authorization|bearer|token|secret)[=: ]+\S+", r"\1=<redacted>", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-<redacted>", text)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1]}…"


def _strip_skip_reason_prefix(reason: object) -> str:
    text = str(reason or "").strip()
    for prefix in ("跳过原因：", "跳过原因:"):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text
