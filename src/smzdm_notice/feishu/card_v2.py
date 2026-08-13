"""飞书 JSON 2.0 卡片的共享构造与遍历工具。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

Card = dict[str, Any]
SUMMARY_MAX_LENGTH = 80


def element_id(prefix: str, key: str = "") -> str:
    """生成只含合法字符、长度稳定且在动态列表中唯一的 element_id。"""
    safe_prefix = re.sub(r"[^A-Za-z0-9_]", "_", prefix).strip("_") or "el"
    if not safe_prefix[0].isalpha():
        safe_prefix = f"el_{safe_prefix}"
    if not key:
        return safe_prefix[:20]
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{safe_prefix[:11]}_{digest}"


def plain_text(content: str) -> Card:
    return {"tag": "plain_text", "content": str(content)}


def build_card(
    title: str,
    template: str,
    elements: list[Card],
    *,
    summary: str | None = None,
    streaming_mode: bool = False,
) -> Card:
    summary_text = str(summary if summary is not None else title)[:SUMMARY_MAX_LENGTH]
    return {
        "schema": "2.0",
        "config": {
            "update_multi": True,
            "streaming_mode": streaming_mode,
            "summary": {"content": summary_text},
        },
        "header": {"title": plain_text(title), "template": template},
        "body": {"elements": elements},
    }


def callback_behavior(value: Card) -> Card:
    return {"type": "callback", "value": value}


def open_url_behavior(url: str) -> Card:
    return {"type": "open_url", "default_url": url}


def button(
    label: str,
    *,
    button_type: str = "default",
    value: Card | None = None,
    url: str = "",
    component_id: str = "",
    confirm: Card | None = None,
    form_action_type: str = "",
) -> Card:
    component: Card = {
        "tag": "button",
        "text": plain_text(label),
        "type": button_type,
    }
    identity = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) if value is not None else url
    component["element_id"] = component_id or element_id("button", f"{label}:{identity}")
    if value is not None:
        component["behaviors"] = [callback_behavior(value)]
    elif url:
        component["behaviors"] = [open_url_behavior(url)]
    if confirm:
        component["confirm"] = confirm
    if form_action_type:
        component["form_action_type"] = form_action_type
        # JSON 2.0 表单通过 name 识别 submit/reset 按钮。复用已经满足
        # 全卡唯一约束的 element_id，避免维护第二套组件标识。
        component["name"] = component["element_id"]
    return component


def input_box(
    name: str,
    placeholder: str,
    *,
    default_value: str | None = None,
    value: Card | None = None,
    component_id: str = "",
    label: str = "",
    width: str = "",
) -> Card:
    component: Card = {
        "tag": "input",
        "name": name,
        "placeholder": plain_text(placeholder),
    }
    component["element_id"] = component_id or element_id("input", name)
    if label:
        component["label"] = plain_text(label)
    if width:
        component["width"] = width
    if default_value is not None:
        component["default_value"] = default_value
    if value is not None:
        component["behaviors"] = [callback_behavior(value)]
    return component


def select_static(
    name: str,
    placeholder: str,
    options: list[Card],
    *,
    value: Card | None = None,
    component_id: str = "",
    initial_option: str = "",
) -> Card:
    component: Card = {
        "tag": "select_static",
        "name": name,
        "placeholder": plain_text(placeholder),
        "options": options,
    }
    component["element_id"] = component_id or element_id("select", name)
    if initial_option:
        component["initial_option"] = initial_option
    if value is not None:
        component["behaviors"] = [callback_behavior(value)]
    return component


def column_set(components: Iterable[Card], *, component_id: str = "") -> Card:
    columns = [
        {
            "tag": "column",
            "width": "auto",
            "vertical_align": "center",
            "elements": [component],
        }
        for component in components
    ]
    result: Card = {
        "tag": "column_set",
        "flex_mode": "flow",
        "horizontal_spacing": "small",
        "columns": columns,
    }
    if component_id:
        result["element_id"] = component_id
    return result


def form(elements: list[Card], *, name: str, component_id: str) -> Card:
    return {"tag": "form", "name": name, "element_id": component_id, "elements": elements}


def body_elements(card: Card) -> list[Card]:
    body = card.get("body")
    if not isinstance(body, dict):
        return []
    elements = body.get("elements")
    return elements if isinstance(elements, list) else []


def iter_components(component: Card) -> Iterable[Card]:
    """深度遍历 column_set、column 和 form 等容器中的组件。"""
    yield component
    for key in ("elements", "columns"):
        children = component.get(key)
        if not isinstance(children, list):
            continue
        for child in children:
            if isinstance(child, dict):
                yield from iter_components(child)


def callback_value(component: Card) -> Card | None:
    behaviors = component.get("behaviors")
    if not isinstance(behaviors, list):
        return None
    for behavior in behaviors:
        if not isinstance(behavior, dict) or behavior.get("type") != "callback":
            continue
        value = behavior.get("value")
        if isinstance(value, dict):
            return value
    return None


def card_component_count(value: object) -> int:
    """统计卡片 JSON 中所有带 tag 的元素和组件。"""
    if isinstance(value, dict):
        return int("tag" in value) + sum(card_component_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(card_component_count(child) for child in value)
    return 0


def card_json_size_bytes(card: Card) -> int:
    """按实际发送时的 JSON 编码计算 UTF-8 字节数。"""
    return len(json.dumps(card, ensure_ascii=False).encode("utf-8"))
