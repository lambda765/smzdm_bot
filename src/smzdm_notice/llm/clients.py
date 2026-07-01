"""按 LLM 使用场景复用 OpenAI SDK client。"""

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
    """为已解析的 LLM 配置返回缓存的 OpenAI-compatible client。"""
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
    """返回缓存的 OpenAI client，仅在首次遇到该规格时创建。

    使用 frozen dataclass ClientSpec 作为 dict key，让共享相同连接参数
    （api_key、base_url、timeout 等）的 agent 复用同一个 OpenAI 实例。
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
    """清空测试中使用的 SDK client 缓存。"""
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
