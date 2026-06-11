"""飞书通知模块。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from io import BytesIO
from typing import Any

from loguru import logger

from smzdm_notice.feishu.binding import FeishuBinding, FeishuBindingStore
from smzdm_notice.feishu.media import get_feishu_image_key
from smzdm_notice.feishu.sdk import (
    get_file_models,
    get_lark_client,
    get_message_models,
    get_message_update_models,
    get_reply_message_models,
)
from smzdm_notice.llm.models import ArbiterInfo
from smzdm_notice.preferences.preview import build_draft_preview_content
from smzdm_notice.smzdm.ranking import RankingItem

_BINDING_STORE = FeishuBindingStore()
ARBITRATION_CARD_KIND = "arbitration"
ARBITRATION_CARD_METADATA_KEY = "arbitration_card"
_DIGEST_PREVIEW_LIMIT = 20
_DIGEST_ATTACHMENT_FORMAT = "markdown"
_DIGEST_ATTACHMENT_EXTENSIONS = {"markdown": "md"}
Card = dict[str, Any]
MessageId = str


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


def _send_text(text: str) -> bool:
    binding = _current_binding()
    if not binding:
        return False
    return send_text_to(binding.receive_id_type, binding.receive_id, text)


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


def send_text_to(receive_id_type: str, receive_id: str, text: str) -> bool:
    try:
        CreateMessageRequest, CreateMessageRequestBody = get_message_models()
        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type("text")
                .content(json.dumps({"text": text}, ensure_ascii=False))
                .build()
            )
            .build()
        )
        response = get_lark_client().im.v1.message.create(request)
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
        CreateMessageRequest, CreateMessageRequestBody = get_message_models()
        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type("file")
                .content(json.dumps({"file_key": file_key}, ensure_ascii=False))
                .build()
            )
            .build()
        )
        response = get_lark_client().im.v1.message.create(request)
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
        CreateMessageRequest, CreateMessageRequestBody = get_message_models()
        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type("interactive")
                .content(json.dumps(card, ensure_ascii=False))
                .build()
            )
            .build()
        )
        response = get_lark_client().im.v1.message.create(request)
        if response.success():
            msg_id = str(getattr(response.data, "message_id", "") or "")
            logger.info(f"飞书应用消息发送成功: {msg_id}")
            return msg_id or None
        logger.error(f"飞书应用消息发送失败: code={response.code}, msg={response.msg}")
        return None
    except Exception as e:
        logger.error(f"飞书应用消息发送异常: {e}")
        return None


def send_card_to(receive_id_type: str, receive_id: str, card: Card) -> bool:
    return _send_card_to_message_id(receive_id_type, receive_id, card) is not None


def send_deals(
    items: list[tuple[RankingItem, str]],
    price_bypass_article_ids: set[str] | None = None,
) -> bool:
    """推送匹配到的好价商品。"""
    if not items:
        return False
    price_bypass_article_ids = price_bypass_article_ids or set()
    return _send_card_success(_build_deals_card(items, price_bypass_article_ids))


def _build_deals_card(
    items: list[tuple[RankingItem, str]],
    price_bypass_article_ids: set[str],
) -> Card:
    elements = [
        {
            "tag": "markdown",
            "content": f"🔥 发现 **{len(items)}** 件好价商品！\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        },
        {"tag": "hr"},
    ]
    for item, reason in items:
        elements.extend(_deal_item_elements(item, reason, item.article_id in price_bypass_article_ids))
    return {
        "header": {
            "title": {"tag": "plain_text", "content": "🛒 什么值得买 · 好价推荐"},
            "template": "red",
        },
        "elements": elements,
    }


def _deal_item_elements(item: RankingItem, reason: str, is_price_bypass: bool) -> list[Card]:
    image_key = get_feishu_image_key(item.pic) if item.pic else ""
    elements: list[Card] = [{"tag": "markdown", "content": "\n".join(_deal_markdown_lines(item, reason, image_key))}]
    if image_key:
        elements.append(_deal_image_element(item, image_key))
    elements.append({"tag": "action", "actions": _deal_actions(item, is_price_bypass)})
    elements.append({"tag": "hr"})
    return elements


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


def _deal_actions(item: RankingItem, is_price_bypass: bool) -> list[Card]:
    actions = []
    if item.link:
        actions.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "查看详情"},
                "url": item.link,
                "type": "primary",
            }
        )
    actions.extend(_item_action_buttons(item, is_price_bypass))
    return actions


def send_heartbeat(hours: int) -> bool:
    """发送心跳消息。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return _send_card_success(
        {
            "header": {"title": {"tag": "plain_text", "content": "🤖 好价监控 · 心跳"}, "template": "blue"},
            "elements": [
                {
                    "tag": "markdown",
                    "content": f"✅ 好价监控运行正常\n⏰ {now}\n📢 最近 **{hours} 小时**未发现匹配商品\n💡 机器人将持续监控，发现好价立即推送",
                }
            ],
        }
    )


def send_shutdown(reason: str = "") -> bool:
    """发送停止通知。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    reason_text = f"\n📌 原因: {reason}" if reason else ""
    return _send_card_success(
        {
            "header": {"title": {"tag": "plain_text", "content": "⛔ 好价监控 · 已停止"}, "template": "red"},
            "elements": [{"tag": "markdown", "content": f"📅 {now}{reason_text}\n\n如需恢复，请重新启动程序"}],
        }
    )


def send_startup(config_summary: str) -> bool:
    """发送启动通知。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return _send_card_success(
        {
            "header": {"title": {"tag": "plain_text", "content": "🚀 好价监控 · 启动"}, "template": "green"},
            "elements": [{"tag": "markdown", "content": f"📅 {now}\n\n{config_summary}"}],
        }
    )


def send_config_warning(message: str) -> bool:
    """发送运行时配置读取告警。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return _send_card_success(
        {
            "header": {
                "title": {"tag": "plain_text", "content": "⚠️ 好价监控 · 配置读取告警"},
                "template": "orange",
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": f"📅 {now}\n\n{message}\n\n将继续使用上一次成功读取的内容。",
                }
            ],
        }
    )


def build_help_card(help_content: str) -> Card:
    """Build the shortcut help card."""
    return {
        "header": {
            "title": {"tag": "plain_text", "content": "好价监控 · 快捷命令"},
            "template": "blue",
        },
        "elements": [{"tag": "markdown", "content": help_content}],
    }


def build_model_management_card(state: Mapping[str, Any], form_state: Mapping[str, str] | None = None) -> Card:
    """Build the interactive LLM routing management card."""
    form_state = form_state or default_model_form_state(state)
    target_options = _model_target_options(state)
    connection_options = [
        _select_option(
            f"{conn.get('name')} ({conn.get('label')}, {'key ok' if conn.get('key_configured') else 'key missing'})",
            str(conn.get("name") or ""),
        )
        for conn in state.get("connections", [])
    ]
    target_initial = _valid_initial_option(form_state, "target", target_options)
    connection_initial = _valid_initial_option(form_state, "connection", connection_options)
    primary_route_button = _model_primary_route_button(state, form_state)
    return {
        "config": {"update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": "LLM 模型路由"},
            "template": "blue",
        },
        "elements": [
            {"tag": "markdown", "content": _model_management_markdown(state)},
            {"tag": "hr"},
            {
                "tag": "markdown",
                "content": "💡 先选择「作用范围」，再填写参数，最后点击按钮执行操作。",
            },
            {
                "tag": "action",
                "actions": [
                    _select_static("target", "选择作用范围", target_options, {"field": "target"}, initial_option=target_initial),
                    _select_static("connection", "选择连接", connection_options, {"field": "connection"}, initial_option=connection_initial),
                ],
            },
            {
                "tag": "action",
                "actions": [
                    _input("model_id", "model_id，例如 deepseek-chat", default_value=form_state.get("model_id")),
                    _input("temperature", "temperature，0 到 5", default_value=form_state.get("temperature")),
                ],
            },
            {
                "tag": "action",
                "actions": [
                    primary_route_button,
                    _card_button("设置温度", "model_set_temperature", "default"),
                ],
            },
            {
                "tag": "action",
                "actions": [
                    _card_button(
                        "恢复默认",
                        "model_reset_agent",
                        "danger",
                        confirm={
                            "title": {"tag": "plain_text", "content": "确认恢复默认？"},
                            "content": {"tag": "plain_text", "content": "将清除该 agent 的自定义设置，恢复为继承默认配置。"},
                        },
                    ),
                    _card_button("发送测试", "model_test", "default"),
                    _card_button("刷新状态", "model_refresh", "default"),
                ],
            },
        ],
    }


def _model_primary_route_button(state: Mapping[str, Any], form_state: Mapping[str, str]) -> dict:
    current_connection = _current_model_connection(state, form_state)
    selected_connection = str(form_state.get("connection") or _default_model_connection(state)).strip()
    if selected_connection and selected_connection != current_connection:
        return _card_button("切换 connection + model", "model_apply_connection_model", "primary")
    return _card_button("切换 model_id", "model_apply_model", "primary")


def _model_target_options(state: Mapping[str, Any]) -> list[dict]:
    options = [_select_option("默认配置", "default")]
    for agent in state.get("agents", []):
        if not isinstance(agent, Mapping):
            continue
        name = str(agent.get("name") or "").strip()
        if name:
            options.append(_select_option(name, name))
    return options


def default_model_form_state(state: Mapping[str, Any]) -> dict[str, str]:
    defaults = state.get("defaults", {})
    if not isinstance(defaults, Mapping):
        return {"target": "default"}
    result = {
        "target": "default",
        "connection": str(defaults.get("connection") or ""),
        "model_id": str(defaults.get("model_id") or ""),
    }
    temperature = defaults.get("temperature")
    if temperature is not None and temperature != "":
        result["temperature"] = str(temperature)
    return result


def _current_model_connection(state: Mapping[str, Any], form_state: Mapping[str, str]) -> str:
    target = str(form_state.get("target") or "default").strip()
    if target == "default":
        return _default_model_connection(state)
    for agent in state.get("agents", []):
        if isinstance(agent, Mapping) and agent.get("name") == target:
            return str(agent.get("connection") or "").strip()
    return _default_model_connection(state)


def _default_model_connection(state: Mapping[str, Any]) -> str:
    defaults = state.get("defaults", {})
    if not isinstance(defaults, Mapping):
        return ""
    return str(defaults.get("connection") or "").strip()


def _model_management_markdown(state: Mapping[str, Any]) -> str:
    defaults = state.get("defaults", {})
    if not isinstance(defaults, Mapping):
        defaults = {}
    lines = [
        f"**默认配置**：`{defaults.get('connection')}/{defaults.get('model_id')}`"
        + _temperature_text(defaults.get("temperature")),
        "",
        "**Agents**",
    ]
    for agent in state.get("agents", []):
        if not isinstance(agent, Mapping):
            continue
        inherited = []
        if agent.get("inherits_connection"):
            inherited.append("connection")
        if agent.get("inherits_model"):
            inherited.append("model")
        suffix = f"（继承 {'/'.join(inherited)}）" if inherited else ""
        lines.append(
            f"- `{agent.get('name')}`: `{agent.get('connection')}/{agent.get('model_id')}`"
            f"{_temperature_text(agent.get('temperature'))} {suffix}".rstrip()
        )
    lines.extend(["", "**Connections**"])
    for conn in state.get("connections", []):
        if not isinstance(conn, Mapping):
            continue
        key_status = "ok" if conn.get("key_configured") else "missing key"
        lines.append(
            f"- `{conn.get('name')}`: {conn.get('label')}，{conn.get('provider')}，{conn.get('base_url_host')}，{key_status}"
        )
    return "\n".join(lines)


def _temperature_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"，temperature `{value}`"


def _plain_text(content: str) -> dict:
    return {"tag": "plain_text", "content": content}


def _select_option(label: str, value: str) -> dict:
    return {"text": _plain_text(label), "value": value}


def _select_static(
    name: str, placeholder: str, options: list[dict], value: dict | None = None, initial_option: str | None = None,
) -> dict:
    payload: dict = {
        "tag": "select_static",
        "name": name,
        "placeholder": _plain_text(placeholder),
        "options": options,
    }
    if value:
        payload["value"] = value
    if initial_option:
        payload["initial_option"] = initial_option
    return payload


def _input(name: str, placeholder: str, default_value: str | None = None) -> dict:
    payload: dict = {
        "tag": "input",
        "name": name,
        "placeholder": _plain_text(placeholder),
    }
    if default_value is not None:
        payload["default_value"] = default_value
    return payload


def _valid_initial_option(form_state: Mapping[str, str], key: str, options: list[dict]) -> str | None:
    """Return the form_state value for *key* only if it matches an existing option value."""
    value = form_state.get(key)
    if not value:
        return None
    valid_values = {opt.get("value") for opt in options}
    return value if value in valid_values else None


def _card_button(label: str, action: str, button_type: str, confirm: dict | None = None) -> dict:
    btn: dict = {
        "tag": "button",
        "text": _plain_text(label),
        "type": button_type,
        "value": {"action": action},
    }
    if confirm:
        btn["confirm"] = confirm
    return btn


def send_help(help_content: str, reply_to_message_id: str = "") -> bool:
    """Send shortcut help as a card."""
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
        {
            "header": {
                "title": {"tag": "plain_text", "content": "⚠️ 好价监控 · 轮询失败告警"},
                "template": "red",
            },
            "elements": [
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
        }
    )


def send_arbitration(arbiter_info: ArbiterInfo, draft: Any | None = None) -> bool:
    """推送仲裁分析结果，附带一键采纳按钮。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    snapshot = _build_arbitration_card_snapshot(arbiter_info, now)
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
    }


def _build_arbitration_content(snapshot: dict) -> str:
    return (
        f"📅 {snapshot.get('sent_at', '')}\n\n"
        f"**两次判断不一致，已仲裁**\n\n"
        f"**差异商品：**\n{snapshot.get('diff_text', '')}\n\n"
        f"**仲裁选择：** 判断 {snapshot.get('chosen', '')}\n"
        f"**原因：** {snapshot.get('reason', '')}\n\n"
        f"**不一致分析：**\n{snapshot.get('analysis', '')}\n\n"
        f"**Prompt 优化建议：**\n{snapshot.get('suggestion', '')}"
    )


def build_arbitration_card(
    snapshot: dict,
    draft: Any | None = None,
    disabled_reason: str = "",
) -> Card:
    """构造仲裁分析卡片；disabled_reason 非空时移除按钮并显示失效原因。"""
    elements = _arbitration_elements(snapshot, draft, disabled_reason)
    return {
        "config": {"update_multi": True},
        "header": {
            "title": {
                "tag": "plain_text",
                "content": "⚖️ 好价监控 · 仲裁分析" + ("（已失效）" if disabled_reason else ""),
            },
            "template": "grey" if disabled_reason else "purple",
        },
        "elements": elements,
    }


def _arbitration_elements(snapshot: dict, draft: Any | None, disabled_reason: str) -> list[Card]:
    elements: list[Card] = [{"tag": "markdown", "content": _build_arbitration_content(snapshot)}]
    actions = _arbitration_actions(draft, disabled_reason)
    if draft:
        elements.extend([{"tag": "hr"}, {"tag": "markdown", "content": build_draft_preview_content(draft)}])
    elif not disabled_reason:
        elements.append(
            {
                "tag": "markdown",
                "content": "⚠️ 本次未生成可直接采纳的配置修改，请按需手动调整偏好文件。",
            }
        )
    if disabled_reason:
        elements.extend([{"tag": "hr"}, {"tag": "markdown", "content": f"**状态：已失效**\n\n原因：{disabled_reason}"}])
    elif actions:
        elements.append({"tag": "action", "actions": actions})
    return elements


def _arbitration_actions(draft: Any | None, disabled_reason: str) -> list[Card]:
    if disabled_reason:
        return []
    actions: list[Card] = []
    if draft:
        actions.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "采纳并更新"},
                "type": "primary",
                "value": {
                    "action": "apply_draft",
                    "draft_id": draft.draft_id,
                    "card_kind": ARBITRATION_CARD_KIND,
                },
            }
        )
    actions.append(
        {
            "tag": "button",
            "text": {"tag": "plain_text", "content": "忽略"},
            "type": "default",
            "value": {
                "action": "ignore_arbitration",
                "draft_id": getattr(draft, "draft_id", ""),
                "card_kind": ARBITRATION_CARD_KIND,
            },
        }
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

    card = {
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"🌙 好价监控 · 夜间汇总 ({digest_date})",
            },
            "template": "orange",
        },
        "elements": elements,
    }

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
    return {
        "config": {"update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": "📝 配置修改预览"},
            "template": "blue",
        },
        "elements": [
            {"tag": "markdown", "content": build_draft_preview_content(draft)},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "确认应用"},
                        "type": "primary",
                        "value": {"action": "apply_draft", "draft_id": draft.draft_id},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "取消"},
                        "type": "danger",
                        "value": {"action": "cancel_draft", "draft_id": draft.draft_id},
                    },
                ],
            },
        ],
    }


def build_draft_processing_card(stage: str = "正在生成配置修改预览", elapsed_seconds: int = 0) -> Card:
    """构造无按钮的草案生成处理中卡片。"""
    elapsed_text = f"\n\n已等待：{elapsed_seconds} 秒" if elapsed_seconds > 0 else ""
    return {
        "config": {"update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": "📝 配置修改预览生成中"},
            "template": "blue",
        },
        "elements": [
            {
                "tag": "markdown",
                "content": f"正在生成配置修改预览，请稍候。\n\n当前阶段：{stage}{elapsed_text}",
            },
        ],
    }


def build_draft_failure_card(reason: str) -> Card:
    """构造无按钮的草案生成失败卡片。"""
    return {
        "config": {"update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": "📝 配置修改预览生成失败"},
            "template": "red",
        },
        "elements": [
            {
                "tag": "markdown",
                "content": f"{reason}\n\n请换一种更明确的说法重试，或稍后再试。",
            },
        ],
    }


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


def build_disabled_draft_card(reason: str, draft: Any | None = None) -> Card:
    """构造无按钮的失效预览卡片。"""
    content = (
        f"{build_draft_preview_content(draft)}\n\n**状态：已失效**\n\n原因：{reason}"
        if draft
        else f"~~此预览已失效~~\n原因：{reason}"
    )
    return {
        "config": {"update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": "📝 配置修改预览（已失效）"},
            "template": "grey",
        },
        "elements": [
            {"tag": "markdown", "content": content},
        ],
    }


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
    return {
        "config": {"update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": "⚖️ 好价监控 · 仲裁分析（已失效）"},
            "template": "grey",
        },
        "elements": [{"tag": "markdown", "content": content}],
    }


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
    return _send_text(text)


def _button(label: str, action: str, item: RankingItem, button_type: str) -> dict:
    value: Card = {
        "action": action,
        "item_title": item.title,
        "item_brand": item.brand,
        "item_link": item.link,
        "article_id": item.article_id,
    }
    if item.search_keyword:
        value["search_keyword"] = item.search_keyword
    if item.search_max_price is not None:
        value["search_max_price"] = item.search_max_price
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": button_type,
        "value": value,
    }


def _item_action_buttons(item: RankingItem, is_price_bypass: bool = False) -> list[dict]:
    if is_price_bypass:
        return [
            _button("移除搜索词", "search_remove_keyword", item, "danger"),
            _button("清除价格阈值", "search_clear_price", item, "default"),
        ]
    return [
        _button("不再推荐此类", "deal_ignore_category", item, "danger"),
        _button("库存充足", "deal_stock_enough", item, "default"),
        _button("加入关注", "deal_follow", item, "default"),
    ]


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
