"""筛选专用的有界重试和跨调用共享限流冷却。"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
from random import uniform
from threading import Lock
from time import monotonic, time
from typing import TypeVar

import httpx
from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError

from smzdm_notice.llm.routing import ResolvedLLMConfig

T = TypeVar("T")
MAX_COOLDOWN_WAIT_SECONDS = 60.0


class FilterCooldownError(RuntimeError):
    """等待限流冷却会超出本次筛选的等待预算。"""


class ConnectionCooldown:
    def __init__(self) -> None:
        self._lock = Lock()
        self._deadlines: dict[tuple[str, str, str], float] = {}

    @staticmethod
    def key(config: ResolvedLLMConfig) -> tuple[str, str, str]:
        # 密钥轮换、连接切换后不继承旧账户的冷却；不保存或输出明文密钥。
        return config.connection, config.base_url, hashlib.sha256(config.api_key.encode()).hexdigest()

    def extend(self, key: tuple[str, str, str], seconds: float) -> None:
        now = monotonic()
        with self._lock:
            self._deadlines = {key: deadline for key, deadline in self._deadlines.items() if deadline > now}
            self._deadlines[key] = max(self._deadlines.get(key, 0), now + seconds)

    async def wait(self, key: tuple[str, str, str], budget: float, context: str) -> float:
        started = monotonic()
        while True:
            with self._lock:
                remaining = max(0.0, self._deadlines.get(key, 0) - monotonic())
            elapsed = monotonic() - started
            if remaining <= 0:
                return elapsed
            # 醒来后重新检查其他线程是否延长了冷却，但绝不无限续等。
            if remaining > max(0.0, budget - elapsed):
                logger.warning(f"{context} error=rate_limit_cooldown remaining={remaining:.3f}s wait={elapsed:.3f}s")
                raise FilterCooldownError(
                    f"限流冷却剩余 {remaining:.1f} 秒，超过本次剩余等待预算 {max(0, budget - elapsed):.1f} 秒"
                )
            logger.info(f"{context} cooldown_wait={remaining:.3f}s")
            await asyncio.sleep(remaining)


_COOLDOWN = ConnectionCooldown()


def _retry_after_seconds(error: APIStatusError) -> float:
    headers = error.response.headers
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) * scale
        except ValueError:
            if name != "retry-after":
                continue
            try:
                seconds = parsedate_to_datetime(value).timestamp() - time()
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return MAX_COOLDOWN_WAIT_SECONDS


def _error_kind(error: Exception) -> str:
    if isinstance(error, APITimeoutError):
        return "timeout"
    if isinstance(error, APIConnectionError):
        return "connection"
    if isinstance(error, APIStatusError):
        return "rate_limit" if error.status_code == 429 else f"http_{error.status_code}"
    return type(error).__name__


def _retryable(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, httpx.TransportError)):
        return True
    return isinstance(error, APIStatusError) and (
        error.status_code in {408, 409, 429} or 500 <= error.status_code < 600
    )


async def call_filter_with_retry(
    request: Callable[[str], Awaitable[T]],
    config: ResolvedLLMConfig,
    label: str,
    *,
    round_id: str,
    deadline: float,
    retryable_errors: tuple[type[Exception], ...] = (),
) -> T:
    """每路最多重试一次；SDK 自动重试必须由调用方关闭。"""
    attempts = 1 + min(1, max(0, config.max_retries))
    key = _COOLDOWN.key(config)
    cooldown_waited = 0.0
    retry_jitter = 0.0
    for attempt in range(1, attempts + 1):
        context = f"LLM round={round_id} role={label} connection={config.connection} model={config.model_id} attempt={attempt}/{attempts}"
        cooldown_waited += await _COOLDOWN.wait(key, MAX_COOLDOWN_WAIT_SECONDS - cooldown_waited, context)
        if retry_jitter:
            logger.info(f"{context} retry_wait={retry_jitter:.3f}s")
            await asyncio.sleep(retry_jitter)
            # 退避期间另一条路可能收到 429，发请求前再检查一次。
            cooldown_waited += await _COOLDOWN.wait(key, MAX_COOLDOWN_WAIT_SECONDS - cooldown_waited, context)
        started = monotonic()
        if started >= deadline:
            raise asyncio.TimeoutError("LLM 阶段总期限耗尽")
        logger.info(
            f"{context} started remaining={deadline - started:.3f}s timeout={config.timeout_seconds:g}s sdk_retries=0"
        )
        try:
            response = await request(context)
        except Exception as error:
            elapsed = monotonic() - started
            request_id = getattr(error, "request_id", None) or "-"
            retry = (
                attempt < attempts
                and monotonic() < deadline
                and (_retryable(error) or isinstance(error, retryable_errors))
            )
            if isinstance(error, APIStatusError) and error.status_code == 429:
                cooldown = _retry_after_seconds(error)
                # 即使本路没有重试额度，也必须让另一路和下一轮看到限流。
                _COOLDOWN.extend(key, cooldown)
                retry = retry and cooldown <= MAX_COOLDOWN_WAIT_SECONDS
                retry_jitter = uniform(0, 3) if retry else 0
                logger.warning(f"{context} rate_limit_cooldown={cooldown:.3f}s")
            else:
                retry_jitter = uniform(1, 3) if retry else 0
            logger.warning(
                f"{context} error={_error_kind(error)} elapsed={elapsed:.3f}s "
                f"cause={type(error.__cause__).__name__ if error.__cause__ else '-'} "
                f"request_id={request_id} retry={retry}"
            )
            if not retry:
                raise
        else:
            logger.info(f"{context} success elapsed={monotonic() - started:.3f}s validated=true")
            return response
    raise AssertionError("筛选重试循环未产生结果")
