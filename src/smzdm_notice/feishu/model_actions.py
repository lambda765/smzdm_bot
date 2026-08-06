"""Feishu 模型管理卡片的路由修改与字段解析。"""

from __future__ import annotations

from dataclasses import dataclass

from smzdm_notice.feishu.card_payload import optional_model_card_field_value
from smzdm_notice.llm import routing as llm_routing
from smzdm_notice.llm.routing import AGENTS, ResolvedLLMConfig


@dataclass
class ModelRouteUpdateResult:
    """模型卡片修改路由后的快照和提示文案。"""

    snapshot: llm_routing.RoutingSnapshot
    message: str


def apply_model_route_card_action(action: str, value: dict) -> ModelRouteUpdateResult:
    """持久化一次模型管理卡片路由操作。"""
    target = model_card_target(value)
    handlers = {
        "model_apply_connection_model": _apply_connection_model_route,
        "model_apply_model": _apply_model_route,
        "model_set_temperature": _apply_model_temperature,
        "model_reset_agent": _reset_agent_route,
    }
    handler = handlers.get(action)
    if handler is None:
        raise ValueError(f"未知操作：{action}")
    return handler(value, target)


def resolve_model_test_config(value: dict) -> ResolvedLLMConfig:
    """从卡片字段解析模型测试目标，不修改路由配置。"""
    connection = optional_model_card_field_value(value, "connection")
    model_id = optional_model_card_field_value(value, "model_id")
    if connection and model_id:
        return llm_routing.test_config_for_connection(connection, model_id)
    target = model_card_target(value)
    if target != "default":
        return llm_routing.test_config_for_agent(target)
    defaults = llm_routing.model_card_state().get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("默认 LLM 配置不可用")
    return llm_routing.test_config_for_connection(
        str(defaults.get("connection") or ""),
        str(defaults.get("model_id") or ""),
    )


def model_card_target(value: dict) -> str:
    target = optional_model_card_field_value(value, "target") or "default"
    if target == "default" or target in AGENTS:
        return target
    raise ValueError("作用范围必须是 default/filter/arbiter/draft")


def _apply_connection_model_route(value: dict, target: str) -> ModelRouteUpdateResult:
    connection = optional_model_card_field_value(value, "connection")
    model_id = optional_model_card_field_value(value, "model_id")
    if not connection:
        raise ValueError("请选择 connection 后再应用")
    if not model_id:
        raise ValueError("请输入 model_id 后再应用")
    if target == "default":
        snapshot = llm_routing.use_default_connection_model(connection, model_id)
    else:
        snapshot = llm_routing.use_agent_model(target, model_id, connection=connection)
    return ModelRouteUpdateResult(snapshot, f"已更新 {_model_route_label(snapshot, target)}，下一次 LLM 调用生效。")


def _apply_model_route(value: dict, target: str) -> ModelRouteUpdateResult:
    model_id = _required_field(value, "model_id", "请输入 model_id 后再应用")
    if target == "default":
        snapshot = llm_routing.use_default_model(model_id)
    else:
        snapshot = llm_routing.use_agent_model(target, model_id)
    return ModelRouteUpdateResult(snapshot, f"已更新 {_model_route_label(snapshot, target)}，下一次 LLM 调用生效。")


def _apply_model_temperature(value: dict, target: str) -> ModelRouteUpdateResult:
    temperature = _temperature(value)
    if target == "default":
        snapshot = llm_routing.set_default_temperature(temperature)
    else:
        snapshot = llm_routing.set_agent_temperature(target, temperature)
    label = f"{target} temperature={temperature:g}"
    return ModelRouteUpdateResult(snapshot, f"已更新 {label}，下一次 LLM 调用生效。")


def _reset_agent_route(_value: dict, target: str) -> ModelRouteUpdateResult:
    if target == "default":
        raise ValueError("默认配置不能 reset，请直接应用新的 connection/model_id")
    snapshot = llm_routing.reset_agent(target)
    return ModelRouteUpdateResult(snapshot, f"已重置 {_model_route_label(snapshot, target)}，下一次 LLM 调用生效。")


def _model_route_label(snapshot: llm_routing.RoutingSnapshot, target: str) -> str:
    if target == "default":
        defaults = snapshot.raw.get("defaults", {})
        if not isinstance(defaults, dict):
            return "default"
        return f"default: {defaults.get('connection')}/{defaults.get('model_id')}"
    resolved = snapshot.resolve(target)
    return f"{target}: {resolved.connection}/{resolved.model_id}"


def _temperature(value: dict) -> float:
    raw = _required_field(value, "temperature", "请输入 temperature")
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError("temperature 必须是数字") from e


def _required_field(value: dict, key: str, message: str) -> str:
    clean = optional_model_card_field_value(value, key)
    if not clean:
        raise ValueError(message)
    return clean
