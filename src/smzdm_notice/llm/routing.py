"""Runtime LLM routing for model connections and agent-level overrides."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlparse

from smzdm_notice.core import config

AGENTS = ("filter", "arbiter", "draft")
DEFAULT_ROUTING_FILENAME = "llm_models.json"
DEFAULT_REQUEST: dict[str, Any] = {"response_format": {"type": "json_object"}, "extra_body": {}}
SUPPORTED_PROVIDERS = {"openai_compatible"}


class LLMRoutingError(ValueError):
    """Raised when LLM routing configuration or updates are invalid."""


@dataclass(frozen=True)
class ResolvedLLMConfig:
    agent: str
    connection: str
    connection_label: str
    provider: str
    base_url: str
    api_key_env: str
    api_key: str
    model_id: str
    timeout_seconds: float
    max_retries: int
    temperature: float | None
    response_format: dict[str, Any] | None
    extra_body: dict[str, Any]

    @property
    def base_url_host(self) -> str:
        return _extract_host(self.base_url)


@dataclass(frozen=True)
class RoutingSnapshot:
    raw: dict[str, Any]
    version: int
    path: Path
    source: str

    def resolve(self, agent: str) -> ResolvedLLMConfig:
        return _resolve_agent(self.raw, agent)


@dataclass
class _RoutingState:
    raw: dict[str, Any]
    path: Path
    source: str
    version: int = 1

    def snapshot(self) -> RoutingSnapshot:
        return RoutingSnapshot(raw=deepcopy(self.raw), version=self.version, path=self.path, source=self.source)


_LOCK = RLock()
_UPDATE_LOCK = RLock()
_STATE: _RoutingState | None = None


def initialize(force: bool = False) -> RoutingSnapshot:
    """Load routing state from llm_models.json."""
    with _UPDATE_LOCK, _LOCK:
        return _initialize_locked(force=force)


def _initialize_locked(force: bool = False) -> RoutingSnapshot:
    """Load routing state while _LOCK is already held."""
    global _STATE
    if _STATE is not None and not force:
        return _STATE.snapshot()
    path = _routing_path()
    if not path.exists():
        raise LLMRoutingError(f"{path.name} missing; run smzdm-notice migrate-llm-config or smzdm-notice setup")
    raw = _load_json(path)
    _validate_raw(raw)
    _STATE = _RoutingState(raw=raw, path=path, source="file")
    return _STATE.snapshot()


def get_snapshot() -> RoutingSnapshot:
    with _LOCK:
        return _initialize_locked() if _STATE is None else _STATE.snapshot()


def _clear_routing_state() -> None:
    """Clear cached routing state for tests."""
    global _STATE
    with _LOCK:
        _STATE = None


def resolve(agent: str, snapshot: RoutingSnapshot | None = None) -> ResolvedLLMConfig:
    return (snapshot or get_snapshot()).resolve(agent)


def build_chat_completion_kwargs(llm_config: ResolvedLLMConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build OpenAI chat completion kwargs from a resolved LLM config."""
    options: dict[str, Any] = {"model": llm_config.model_id, "messages": messages}
    if llm_config.temperature is not None:
        options["temperature"] = llm_config.temperature
    if llm_config.response_format is not None:
        options["response_format"] = deepcopy(llm_config.response_format)
    if llm_config.extra_body:
        options["extra_body"] = deepcopy(llm_config.extra_body)
    return options


def validate_raw(raw: dict[str, Any], env: Mapping[str, str] | None = None) -> None:
    """Validate routing data with the same rules used at runtime."""
    _validate_raw(raw, env=env)


def format_status(snapshot: RoutingSnapshot | None = None) -> str:
    snap = snapshot or get_snapshot()
    raw = snap.raw
    default_connection = str(raw.get("defaults", {}).get("connection") or "")
    default_model = str(raw.get("defaults", {}).get("model_id") or "")
    lines = [
        "**LLM 路由**",
        f"- 配置来源: {snap.source}",
        f"- 默认: {default_connection}/{default_model}",
    ]
    for agent in AGENTS:
        resolved = snap.resolve(agent)
        inherited = _agent_inheritance_text(raw, agent)
        lines.append(
            f"- {agent}: {resolved.connection}/{resolved.model_id} "
            f"({resolved.connection_label}, {resolved.base_url_host}){inherited}"
        )
    return "\n".join(lines)


def model_card_state(snapshot: RoutingSnapshot | None = None) -> dict[str, Any]:
    """Return safe routing data for the Feishu model management card."""
    snap = snapshot or get_snapshot()
    raw = snap.raw
    defaults = _object(raw.get("defaults"), "defaults")
    default_request = _object(defaults.get("request", {}), "defaults.request")
    connections = _object(raw.get("connections"), "connections")
    agents = _object(raw.get("agents", {}), "agents")
    return {
        "version": snap.version,
        "source": snap.source,
        "defaults": {
            "connection": str(defaults.get("connection") or ""),
            "model_id": str(defaults.get("model_id") or ""),
            "temperature": _optional_float(default_request.get("temperature")),
        },
        "connections": [
            _connection_card_info(name, _object(connections[name], f"connections.{name}"))
            for name in sorted(connections)
        ],
        "agents": [
            _agent_card_state(snap, agents, agent)
            for agent in AGENTS
        ],
    }


def _connection_card_info(name: str, conn: dict[str, Any]) -> dict[str, Any]:
    """Build safe connection info dict for the model management card."""
    base_url = str(conn.get("base_url") or "")
    api_key_env = str(conn.get("api_key_env") or "")
    return {
        "name": name,
        "label": str(conn.get("label") or name),
        "provider": str(conn.get("provider") or ""),
        "base_url_host": _extract_host(base_url),
        "key_configured": bool(os.getenv(api_key_env)),
    }


def _agent_card_state(snap: RoutingSnapshot, agents: dict[str, Any], agent: str) -> dict[str, Any]:
    cfg = _object(agents.get(agent, {}), f"agents.{agent}")
    resolved = snap.resolve(agent)
    inherits_connection, inherits_model = _agent_inheritance_flags(cfg)
    return {
        "name": agent,
        "connection": resolved.connection,
        "connection_label": resolved.connection_label,
        "model_id": resolved.model_id,
        "temperature": resolved.temperature,
        "base_url_host": resolved.base_url_host,
        "inherits_connection": inherits_connection,
        "inherits_model": inherits_model,
    }


def use_default_connection_model(connection: str, model_id: str) -> RoutingSnapshot:
    """Set the default connection and model_id, persisting to llm_models.json."""
    connection = _clean_required(connection, "connection")
    model_id = _clean_required(model_id, "model_id")

    def mutate(raw: dict[str, Any]) -> None:
        defaults = _object(raw.setdefault("defaults", {}), "defaults")
        defaults["connection"] = connection
        defaults["model_id"] = model_id

    return _update_routing(mutate)


def use_default_model(model_id: str) -> RoutingSnapshot:
    """Set the default model_id without changing the default connection."""
    model_id = _clean_required(model_id, "model_id")

    def mutate(raw: dict[str, Any]) -> None:
        raw.setdefault("defaults", {})["model_id"] = model_id

    return _update_routing(mutate)


def use_agent_model(agent: str, model_id: str, connection: str | None = None) -> RoutingSnapshot:
    """Override model_id (and optionally connection) for a specific agent."""
    agent = _validate_agent(agent)
    model_id = _clean_required(model_id, "model_id")
    connection = _clean_optional(connection)

    def mutate(raw: dict[str, Any]) -> None:
        value = _agent_config_for_mutation(raw, agent)
        value["model_id"] = model_id
        if connection:
            value["connection"] = connection

    return _update_routing(mutate)


def reset_agent(agent: str) -> RoutingSnapshot:
    """Remove agent-level overrides so it falls back to defaults."""
    agent = _validate_agent(agent)

    def mutate(raw: dict[str, Any]) -> None:
        value = _agent_config_for_mutation(raw, agent)
        value.pop("connection", None)
        value.pop("model_id", None)

    return _update_routing(mutate)


def set_default_temperature(temperature: float) -> RoutingSnapshot:
    """Set the default temperature in defaults.request."""
    _validate_temperature(temperature)

    def mutate(raw: dict[str, Any]) -> None:
        defaults = _object(raw.setdefault("defaults", {}), "defaults")
        request = _object(defaults.setdefault("request", {}), "defaults.request")
        request["temperature"] = temperature

    return _update_routing(mutate)


def set_agent_temperature(agent: str, temperature: float) -> RoutingSnapshot:
    """Override temperature for a specific agent's request settings."""
    agent = _validate_agent(agent)
    _validate_temperature(temperature)

    def mutate(raw: dict[str, Any]) -> None:
        request = _agent_request_for_mutation(raw, agent)
        request["temperature"] = temperature

    return _update_routing(mutate)


def test_config_for_agent(agent: str) -> ResolvedLLMConfig:
    return resolve(_validate_agent(agent))


def test_config_for_connection(connection: str, model_id: str) -> ResolvedLLMConfig:
    connection = _clean_required(connection, "connection")
    model_id = _clean_required(model_id, "model_id")
    snap = get_snapshot()
    raw = deepcopy(snap.raw)
    raw.setdefault("agents", {})["_test"] = {"connection": connection, "model_id": model_id}
    return _resolve_agent(raw, "_test", allow_test_agent=True)


def _update_routing(mutator) -> RoutingSnapshot:
    """Apply a mutation to routing config, validate, persist, and return updated snapshot.

    Uses two locks to avoid holding _LOCK during disk I/O:
    - _UPDATE_LOCK serialises concurrent writes (only one mutation at a time).
    - _LOCK protects in-memory _STATE reads/writes (released before I/O).
    """
    global _STATE
    with _UPDATE_LOCK:
        # Read current state under _LOCK, then release it before disk I/O
        with _LOCK:
            if _STATE is None:
                _initialize_locked()
            if _STATE is None:
                raise LLMRoutingError("LLM routing state unavailable")
            raw = deepcopy(_STATE.raw)
            path = _STATE.path
            source = _STATE.source
            version = _STATE.version + 1
        # Mutation and validation run outside _LOCK so readers aren't blocked
        mutator(raw)
        _validate_raw(raw)
        _write_json_atomic(path, raw)
        # Re-acquire _LOCK to commit the new state
        with _LOCK:
            _STATE = _RoutingState(raw=raw, path=path, source=source, version=version)
            return _STATE.snapshot()


def _resolve_agent(
    raw: dict[str, Any],
    agent: str,
    allow_test_agent: bool = False,
    env: Mapping[str, str] | None = None,
) -> ResolvedLLMConfig:
    """Resolve the effective LLM config for an agent.

    Inheritance priority (agent-level overrides take precedence):
      connection/model_id: agent_cfg → defaults
      request (temperature, response_format, extra_body): merged via _merge_request
      timeout_seconds/max_retries: agent_cfg → connection → defaults (via _first_config_value)
    """
    if not allow_test_agent:
        agent = _validate_agent(agent)
    defaults = _object(raw.get("defaults"), "defaults")
    agents = _object(raw.get("agents", {}), "agents")
    agent_cfg = _object(agents.get(agent, {}), f"agents.{agent}")

    # Inherit: agent connection overrides default connection
    connection_name = str(agent_cfg.get("connection") or defaults.get("connection") or "").strip()
    if not connection_name:
        raise LLMRoutingError("defaults.connection 未配置")
    connections = _object(raw.get("connections"), "connections")
    if connection_name not in connections:
        raise LLMRoutingError(f"未知 LLM connection: {connection_name}")
    conn = _object(connections[connection_name], f"connections.{connection_name}")

    # Inherit: agent model_id overrides default model_id
    model_id = str(agent_cfg.get("model_id") or defaults.get("model_id") or "").strip()
    if not model_id:
        raise LLMRoutingError(f"{agent} 未解析到 model_id")

    # Merge request-level settings: defaults.request ← agent.request
    request = _merge_request(defaults.get("request", {}), agent_cfg.get("request", {}))
    provider = str(conn.get("provider") or "").strip()
    base_url = str(conn.get("base_url") or "").strip()
    api_key_env = str(conn.get("api_key_env") or "").strip()
    # env param allows tests to inject keys without touching os.environ
    api_key = str((env.get(api_key_env) if env is not None else os.getenv(api_key_env)) or "").strip()

    response_format = _optional_object(request.get("response_format"), "response_format")
    extra_body = _object(request.get("extra_body", {}), "extra_body")

    return ResolvedLLMConfig(
        agent=agent,
        connection=connection_name,
        connection_label=str(conn.get("label") or connection_name),
        provider=provider,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=api_key,
        model_id=model_id,
        timeout_seconds=_float_value(
            _first_config_value(agent_cfg, conn, defaults, key="timeout_seconds", default=300.0)
        ),
        max_retries=_int_value(_first_config_value(agent_cfg, conn, defaults, key="max_retries", default=2)),
        temperature=_optional_float(request.get("temperature")),
        response_format=deepcopy(response_format) if response_format is not None else None,
        extra_body=deepcopy(extra_body),
    )


def _validate_raw(raw: dict[str, Any], env: Mapping[str, str] | None = None) -> None:
    _object(raw, "llm routing")
    connections = _object(raw.get("connections"), "connections")
    if not connections:
        raise LLMRoutingError("connections 至少需要一个连接")
    for name, value in connections.items():
        if not isinstance(name, str) or not name.strip():
            raise LLMRoutingError("connection 名称不能为空")
        conn = _object(value, f"connections.{name}")
        provider = str(conn.get("provider") or "").strip()
        if provider not in SUPPORTED_PROVIDERS:
            raise LLMRoutingError(f"connections.{name}.provider 仅支持 openai_compatible")
        for key in ("base_url", "api_key_env"):
            if not str(conn.get(key) or "").strip():
                raise LLMRoutingError(f"connections.{name}.{key} 未配置")

    defaults = _object(raw.get("defaults"), "defaults")
    if not str(defaults.get("connection") or "").strip():
        raise LLMRoutingError("defaults.connection 未配置")
    if not str(defaults.get("model_id") or "").strip():
        raise LLMRoutingError("defaults.model_id 未配置")

    agents = _object(raw.get("agents", {}), "agents")
    unknown_agents = sorted(set(agents) - set(AGENTS))
    if unknown_agents:
        raise LLMRoutingError("未知 LLM agent: " + ", ".join(unknown_agents))

    for agent in AGENTS:
        resolved = _resolve_agent(raw, agent, env=env)
        if not resolved.api_key:
            raise LLMRoutingError(f"{agent} 使用的密钥环境变量未配置: {resolved.api_key_env}")
        if not resolved.base_url:
            raise LLMRoutingError(f"{agent} 使用的 base_url 未配置")
        if resolved.provider != "openai_compatible":
            raise LLMRoutingError(f"{agent} provider 不支持: {resolved.provider}")
        if resolved.temperature is not None:
            _validate_temperature(resolved.temperature)


def _merge_request(default_request: Any, agent_request: Any) -> dict[str, Any]:
    """Merge defaults.request with agent-specific request overrides.

    Strategy:
    - Start from the built-in default request as the base.
    - Overlay defaults.request and then agent.request, skipping None values
      (None means "keep inherited default").
    - extra_body is deep-merged: agent keys override default keys.
    """
    default_obj = _object(default_request, "defaults.request")
    agent_obj = _object(agent_request, "agent.request")
    merged = deepcopy(DEFAULT_REQUEST)
    merged.update({key: value for key, value in default_obj.items() if value is not None})
    merged.update({key: value for key, value in agent_obj.items() if value is not None})
    builtin_extra = _object(DEFAULT_REQUEST.get("extra_body", {}), "built-in request.extra_body")
    default_extra = _object(default_obj.get("extra_body", {}), "defaults.request.extra_body")
    agent_extra = _object(agent_obj.get("extra_body", {}), "agent.request.extra_body")
    merged["extra_body"] = {**builtin_extra, **default_extra, **agent_extra}
    return merged


def _first_config_value(*objects: dict[str, Any], key: str, default: Any) -> Any:
    """Return the first non-None/non-empty value for *key* across *objects* (left-to-right priority)."""
    for obj in objects:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return default


def _routing_path() -> Path:
    raw = getattr(config, "LLM_MODELS_FILE", DEFAULT_ROUTING_FILENAME)
    path = Path(raw)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise LLMRoutingError(f"{path.name} JSON 解析失败: {e}") from e
    except OSError as e:
        raise LLMRoutingError(f"无法读取 {path.name}: {e}") from e
    return _object(data, path.name)


def _write_json_atomic(path: Path, raw: dict[str, Any]) -> None:
    """Write JSON to a temp file then atomically replace the target (os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _extract_host(url: str) -> str:
    """Extract hostname from a URL, falling back to the raw string."""
    return urlparse(url).netloc or url


def _object(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LLMRoutingError(f"{label} 必须是 JSON object")
    return value


def _optional_object(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _object(value, label)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return _float_value(value)


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise LLMRoutingError(f"数值配置无效: {value!r}") from e


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise LLMRoutingError(f"整数配置无效: {value!r}") from e


def _validate_temperature(temperature: float) -> None:
    if not math.isfinite(temperature) or temperature < 0 or temperature > 5:
        raise LLMRoutingError("temperature 必须在 0 到 5 之间")


def _validate_agent(agent: str) -> str:
    agent = str(agent or "").strip().lower()
    if agent not in AGENTS:
        raise LLMRoutingError(f"未知 LLM agent: {agent or '<empty>'}")
    return agent


def _clean_required(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise LLMRoutingError(f"{label} 不能为空")
    return clean


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    clean = str(value or "").strip()
    return clean or None


def _agent_config_for_mutation(raw: dict[str, Any], agent: str) -> dict[str, Any]:
    if raw.get("agents") is None:
        raw["agents"] = {}
    agents = _object(raw["agents"], "agents")
    return _object(agents.setdefault(agent, {}), f"agents.{agent}")


def _agent_request_for_mutation(raw: dict[str, Any], agent: str) -> dict[str, Any]:
    agent_cfg = _agent_config_for_mutation(raw, agent)
    if agent_cfg.get("request") is None:
        agent_cfg["request"] = {}
    return _object(agent_cfg["request"], f"agents.{agent}.request")


def _agent_inheritance_flags(cfg: dict[str, Any]) -> tuple[bool, bool]:
    return "connection" not in cfg, "model_id" not in cfg


def _agent_inheritance_text(raw: dict[str, Any], agent: str) -> str:
    cfg = _object(_object(raw.get("agents", {}), "agents").get(agent, {}), f"agents.{agent}")
    inherits_connection, inherits_model = _agent_inheritance_flags(cfg)
    parts = []
    if inherits_connection:
        parts.append("connection")
    if inherits_model:
        parts.append("model")
    if not parts:
        return ""
    return "，继承 " + "/".join(parts)
