"""用于 Feishu LLM 模型管理卡片的构造器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from smzdm_notice.feishu.card_v2 import (
    Card,
    build_card,
    button,
    column_set,
    element_id,
    input_box,
    plain_text,
    select_static,
)


def build_model_management_card(state: Mapping[str, Any], form_state: Mapping[str, str] | None = None) -> Card:
    """构造可交互的 LLM 路由管理卡片。"""
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
    return build_card(
        "LLM 模型路由",
        "blue",
        [
            {"tag": "markdown", "content": _model_management_markdown(state)},
            {"tag": "hr"},
            {
                "tag": "markdown",
                "content": "💡 先选择「作用范围」，再填写参数，最后点击按钮执行操作。",
            },
            column_set(
                [
                    _select_static(
                        "target", "选择作用范围", target_options, {"field": "target"}, initial_option=target_initial
                    ),
                    _select_static(
                        "connection",
                        "选择连接",
                        connection_options,
                        {"field": "connection"},
                        initial_option=connection_initial,
                    ),
                ],
                component_id="model_route_selects",
            ),
            _input(
                "model_id",
                "例如 deepseek-chat",
                label="模型 ID",
                default_value=form_state.get("model_id"),
            ),
            _input(
                "temperature",
                "例如 0.7",
                label="Temperature（0 到 5）",
                default_value=form_state.get("temperature"),
            ),
            column_set(
                [
                    primary_route_button,
                    _card_button("设置温度", "model_set_temperature", "default"),
                ],
                component_id="model_route_apply",
            ),
            column_set(
                [
                    _card_button(
                        "恢复默认",
                        "model_reset_agent",
                        "danger",
                        confirm={
                            "title": {"tag": "plain_text", "content": "确认恢复默认？"},
                            "text": {"tag": "plain_text", "content": "将清除该 agent 的自定义设置，恢复为继承默认配置。"},
                        },
                    ),
                    _card_button("发送测试", "model_test", "default"),
                    _card_button("刷新状态", "model_refresh", "default"),
                ],
                component_id="model_route_tools",
            ),
        ],
    )


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
            f"- `{agent.get('name')}`: `{agent.get('connection')}/{agent.get('model_id')}` "
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


def _select_option(label: str, value: str) -> dict:
    return {"text": plain_text(label), "value": value}


def _select_static(
    name: str, placeholder: str, options: list[dict], value: dict | None = None, initial_option: str | None = None,
) -> dict:
    return select_static(
        name,
        placeholder,
        options,
        value=value,
        component_id=element_id("model_select", name),
        initial_option=initial_option or "",
    )


def _input(name: str, placeholder: str, label: str, default_value: str | None = None) -> dict:
    return input_box(
        name,
        placeholder,
        default_value=default_value,
        value={"field": name},
        component_id=element_id("model_input", name),
        label=label,
        width="fill",
    )


def _valid_initial_option(form_state: Mapping[str, str], key: str, options: list[dict]) -> str | None:
    """仅当 form_state 中的值存在于选项列表时，返回该初始选项值。"""
    value = form_state.get(key)
    if not value:
        return None
    valid_values = {opt.get("value") for opt in options}
    return value if value in valid_values else None


def _card_button(label: str, action: str, button_type: str, confirm: dict | None = None) -> dict:
    return button(label, button_type=button_type, value={"action": action}, confirm=confirm)
