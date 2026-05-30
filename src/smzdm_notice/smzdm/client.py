"""SMZDM API shared request, signing, and parsing helpers."""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Callable

import httpx

from smzdm_notice.core import config

TIMEOUT = 30.0
_CLIENT_LOCK = threading.Lock()
_SHARED_CLIENT: httpx.Client | None = None

# 需要过滤掉的 cell_type（广告/推广位等）
AD_CELL_TYPES = {"21017"}

ValueNormalizer = Callable[[str], str]


def compact_sign_value(value: str) -> str:
    """Normalize search API values before signing."""
    return value.replace(" ", "").replace("\t", "").replace("\n", "")


def _require_smzdm_sign_key() -> str:
    if not config.SMZDM_SIGN_KEY:
        raise RuntimeError("SMZDM_SIGN_KEY 未配置，请在 .env 中填写什么值得买签名 key")
    return config.SMZDM_SIGN_KEY


def _require_smzdm_user_agent() -> str:
    if not config.SMZDM_USER_AGENT:
        raise RuntimeError("SMZDM_USER_AGENT 未配置，请在 .env 中填写什么值得买 User-Agent")
    return config.SMZDM_USER_AGENT


def get_client() -> httpx.Client:
    """Return the shared SMZDM HTTP client."""
    global _SHARED_CLIENT
    with _CLIENT_LOCK:
        if _SHARED_CLIENT is None:
            _SHARED_CLIENT = httpx.Client(timeout=TIMEOUT)
        return _SHARED_CLIENT


def close_client() -> None:
    """Close and clear the shared SMZDM HTTP client."""
    global _SHARED_CLIENT
    with _CLIENT_LOCK:
        if _SHARED_CLIENT is not None:
            _SHARED_CLIENT.close()
            _SHARED_CLIENT = None


def build_signed_params(
    params: dict,
    *,
    value_normalizer: ValueNormalizer | None = None,
    now_ms: int | None = None,
) -> dict:
    """Return request params with a SMZDM app timestamp and MD5 sign."""
    signed = dict(params)
    if now_ms is not None or "time" not in signed:
        signed["time"] = now_ms if now_ms is not None else int(round(time.time() * 1000))

    normalize = value_normalizer or (lambda value: value)
    parts = []
    for key, value in sorted(signed.items()):
        if key == "sign":
            continue
        normalized = normalize("" if value is None else str(value))
        if normalized:
            parts.append(f"{key}={normalized}")
    sign_str = "&".join(parts) + f"&key={_require_smzdm_sign_key()}"
    signed["sign"] = hashlib.md5(sign_str.encode()).hexdigest().upper()
    return signed


def get_json(
    base_url: str,
    endpoint: str,
    params: dict,
    *,
    value_normalizer: ValueNormalizer | None = None,
) -> dict:
    """Send a signed SMZDM GET request and return decoded JSON."""
    signed = build_signed_params(params, value_normalizer=value_normalizer)
    headers = {
        "accept-encoding": "gzip",
        "User-Agent": _require_smzdm_user_agent(),
    }
    resp = get_client().get(f"{base_url}{endpoint}", params=signed, headers=headers)
    resp.raise_for_status()

    data = resp.json()
    code = data.get("error_code")
    if code is not None and int(code) != 0:
        raise RuntimeError(f"API 错误: {data.get('error_msg', '未知')} (code={code})")
    return data


def extract_nested_title(raw) -> str:
    """Extract article_title from a string or a single-item dict list."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict):
            return first.get("article_title", "")
    return ""


def extract_article_tags(*sources) -> list[str]:
    """Extract and dedupe user-facing tag strings from SMZDM tag shapes."""
    tags: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        tag = value.strip()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)

    for source in sources:
        if isinstance(source, str):
            add(source)
            continue
        if not isinstance(source, list):
            continue
        for item in source:
            if isinstance(item, str):
                add(item)
            elif isinstance(item, dict):
                add(str(item.get("article_title") or item.get("title") or item.get("name") or ""))
    return tags


def parse_num(val) -> int:
    """Parse SMZDM counters, including '18k' and '1.2w' abbreviations."""
    if isinstance(val, (int, float)):
        return int(val)
    if not isinstance(val, str) or not val:
        return 0
    val = val.strip().lower()
    try:
        if val.endswith("w"):
            return int(float(val[:-1]) * 10000)
        if val.endswith("k"):
            return int(float(val[:-1]) * 1000)
        return int(float(val))
    except (ValueError, TypeError):
        return 0
