"""用于解析 Feishu 消息与卡片 payload 的工具。"""

from __future__ import annotations

import json
from typing import Any


def extract_message_text(content: str) -> str:
    """从 Feishu 消息内容中提取文本，解析失败时回退为原始字符串。"""
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return str(data.get("text") or "").strip()
    except (TypeError, json.JSONDecodeError):
        pass
    return str(content or "").strip()


def strip_bot_mention(text: str) -> str:
    """移除开头的机器人 mention，让命令解析只看到用户意图。"""
    clean = text.strip()
    while clean.startswith("@"):
        parts = clean.split(maxsplit=1)
        if len(parts) == 1:
            return ""
        clean = parts[1].strip()
    return clean


def extract_card_action_value(data) -> dict:
    """将 Feishu 卡片 callback 归一化为扁平的 action/form payload。

    不同 Feishu SDK 版本和组件类型会把提交值放在不同属性中。这里保留
    原始分组，方便特殊场景读取；同时扁平化字段，让 action handler
    读取稳定结构。
    """
    action = getattr(data.event, "action", None)
    result = _parse_card_dict(getattr(action, "value", None))
    if result.get("field"):
        result["value"] = dict(result)
    name = _clean_card_scalar(getattr(action, "name", None))
    option = _clean_card_scalar(getattr(action, "option", None))
    raw_input_value = getattr(action, "input_value", None)
    input_value = _clean_card_scalar(raw_input_value)
    tag = _clean_card_scalar(getattr(action, "tag", None))
    if name:
        result["name"] = name
    if option:
        result["option"] = option
    if _is_card_scalar(raw_input_value):
        result["input_value"] = input_value
    if tag:
        result["tag"] = tag
    for attr in ("form_value", "form_values", "form", "input_values"):
        form_value = _parse_card_dict(getattr(action, attr, None))
        if not form_value:
            continue
        result[attr] = form_value
        for key, value in form_value.items():
            result.setdefault(key, value)
    return result


def _parse_card_dict(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _clean_card_scalar(value) -> str:
    if value is None:
        return ""
    if not _is_card_scalar(value):
        return ""
    return str(value).strip()


def _is_card_scalar(value) -> bool:
    return isinstance(value, (str, int, float, bool))


def model_card_form_cache_key(message_id: str, operator: str) -> str:
    """为某个操作者编辑某张模型卡片构造进程内缓存键。"""
    if not message_id or not operator:
        return ""
    return f"{message_id}:{operator}"


def trim_model_card_form_cache(state: dict[str, dict[str, Any]], limit: int) -> None:
    """通过丢弃最早插入的卡片/操作者状态限制表单缓存大小。"""
    while len(state) > limit:
        oldest = next(iter(state), None)
        if oldest is None:
            return
        state.pop(oldest, None)


def replace_model_form_cache_values(state: dict[str, Any], form_state: dict[str, str]) -> None:
    """在 target/connection 重绘后替换可见模型表单字段。"""
    for key in ("target", "connection", "model_id", "temperature"):
        state.pop(key, None)
        if key not in form_state:
            state.pop(f"{key}_manual", None)
    for key, value in form_state.items():
        state[key] = value


def apply_model_form_cache_patch(state: dict[str, Any], patch: dict[str, Any]) -> None:
    """将单次组件 callback 变更应用到模型表单缓存。"""
    for key, value in patch.items():
        if key.endswith("_manual") and value is False:
            state.pop(key, None)
        else:
            state[key] = value


def build_model_card_form_patch(value: dict) -> dict[str, Any]:
    """从组件 callback 中提取发生变化的模型表单字段。"""
    field = model_card_field_from_callback(value)
    if field not in {"target", "connection", "model_id", "temperature"}:
        return {}
    if "option" in value:
        option = str(value.get("option") or "").strip()
        return {field: option, f"{field}_manual": bool(option)}
    if "input_value" in value:
        input_value = str(value.get("input_value") or "").strip()
        return {field: input_value, f"{field}_manual": bool(input_value)}
    return {}


def model_card_field_from_callback(value: dict) -> str:
    """无论 callback payload 形态如何，都返回模型表单字段名。"""
    raw_value = value.get("value")
    if isinstance(raw_value, dict):
        field = str(raw_value.get("field") or "").strip()
        if field:
            return field
    return str(value.get("name") or "").strip()


def optional_card_field_value(value: dict, key: str) -> str:
    """从扁平或分组的 Feishu 卡片 payload 中读取可选字段。"""
    raw = _model_card_raw_value(value, key)
    if raw is None:
        return ""
    return str(raw).strip()


def optional_model_card_field_value(value: dict, key: str) -> str:
    """读取归一化后的模型卡片字段，并返回去除首尾空白的字符串。"""
    raw = _model_card_raw_value(value, key)
    if raw is None:
        return ""
    return str(raw).strip()


def _model_card_raw_value(value: dict, key: str):
    if key in value:
        return _normalize_card_form_value(value.get(key))
    for group_key in ("form_values", "form_value", "form", "input_values"):
        group = value.get(group_key)
        if isinstance(group, dict) and key in group:
            return _normalize_card_form_value(group.get(key))
    return None


def _normalize_card_form_value(raw):
    if isinstance(raw, dict):
        for key in ("value", "text", "content"):
            if raw.get(key) not in (None, ""):
                return raw.get(key)
        if raw.get("option") is not None:
            return _normalize_card_form_value(raw.get("option"))
        if any(key in raw for key in ("value", "text", "content")):
            return ""
    if isinstance(raw, list):
        if not raw:
            return ""
        return _normalize_card_form_value(raw[0])
    return raw
