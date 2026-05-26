"""OpenAI SDK client reuse by LLM usage scenario."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from openai import OpenAI

from smzdm_notice.core import config


@dataclass(frozen=True)
class ClientSpec:
    api_key: str
    base_url: str
    timeout: float
    max_retries: int


@dataclass
class ClientSlot:
    spec: ClientSpec | None = None
    client: OpenAI | None = None


_CLIENT_LOCK = Lock()
_CLIENT_SLOTS: dict[str, ClientSlot] = {}


def get_filter_client() -> OpenAI:
    """Return the shared client for primary item filtering."""
    return _get_scene_client(
        "filter",
        ClientSpec(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=config.LLM_MAX_RETRIES,
        ),
    )


def get_arbiter_client() -> OpenAI:
    """Return the shared client for arbitration calls."""
    return _get_scene_client(
        "arbiter",
        ClientSpec(
            api_key=config.LLM_ARBITER_API_KEY,
            base_url=config.LLM_ARBITER_BASE_URL,
            timeout=config.LLM_ARBITER_TIMEOUT_SECONDS,
            max_retries=config.LLM_MAX_RETRIES,
        ),
    )


def get_draft_client() -> OpenAI:
    """Return the shared client for preference/inventory draft generation."""
    return _get_scene_client(
        "draft",
        ClientSpec(
            api_key=config.LLM_DRAFT_API_KEY,
            base_url=config.LLM_DRAFT_BASE_URL,
            timeout=config.LLM_DRAFT_TIMEOUT_SECONDS,
            max_retries=config.LLM_MAX_RETRIES,
        ),
    )


def _get_scene_client(scene: str, spec: ClientSpec) -> OpenAI:
    with _CLIENT_LOCK:
        slot = _CLIENT_SLOTS.setdefault(scene, ClientSlot())
        if slot.client is None or slot.spec != spec:
            slot.spec = spec
            slot.client = OpenAI(
                api_key=spec.api_key,
                base_url=spec.base_url,
                timeout=spec.timeout,
                max_retries=spec.max_retries,
            )
        return slot.client


def _clear_client_cache() -> None:
    """Clear cached SDK clients for tests."""
    with _CLIENT_LOCK:
        _CLIENT_SLOTS.clear()
