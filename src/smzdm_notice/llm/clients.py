"""OpenAI SDK client reuse by LLM usage scenario."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

from loguru import logger
from openai import OpenAI

from smzdm_notice.llm.routing import ResolvedLLMConfig


@dataclass(frozen=True)
class ClientSpec:
    connection: str
    api_key: str
    base_url: str
    timeout: float
    max_retries: int


CLIENT_CACHE_MAX_SIZE = 16
_CLIENT_LOCK = Lock()
_CLIENT_SLOTS: OrderedDict[ClientSpec, OpenAI] = OrderedDict()


def get_client_for_config(llm_config: ResolvedLLMConfig) -> OpenAI:
    """Return a cached OpenAI-compatible client for a resolved LLM config."""
    return _get_client(
        ClientSpec(
            connection=llm_config.connection,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            timeout=llm_config.timeout_seconds,
            max_retries=llm_config.max_retries,
        )
    )


def _get_client(spec: ClientSpec) -> OpenAI:
    """Return a cached OpenAI client, creating one only on first use for this spec.

    Uses ClientSpec (frozen dataclass) as dict key so agents that share the
    same connection parameters (api_key, base_url, timeout, etc.) reuse a
    single OpenAI instance rather than creating duplicate clients.
    """
    with _CLIENT_LOCK:
        if spec in _CLIENT_SLOTS:
            _CLIENT_SLOTS.move_to_end(spec)
            return _CLIENT_SLOTS[spec]
        client = OpenAI(
            api_key=spec.api_key,
            base_url=spec.base_url,
            timeout=spec.timeout,
            max_retries=spec.max_retries,
        )
        _CLIENT_SLOTS[spec] = client
        if len(_CLIENT_SLOTS) > CLIENT_CACHE_MAX_SIZE:
            _, old_client = _CLIENT_SLOTS.popitem(last=False)
            _close_client(old_client)
        return client


def _clear_client_cache() -> None:
    """Clear cached SDK clients for tests."""
    with _CLIENT_LOCK:
        for client in _CLIENT_SLOTS.values():
            _close_client(client)
        _CLIENT_SLOTS.clear()


def _close_client(client: OpenAI) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as e:
        logger.debug(f"关闭 OpenAI client 失败: {e}")
