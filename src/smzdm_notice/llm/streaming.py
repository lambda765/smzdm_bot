"""有期限的流式收集；客户端仅在当前事件循环内共享。"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from time import monotonic
from typing import Any, TypeVar

from loguru import logger
from openai import AsyncOpenAI

from smzdm_notice.llm.filter_retry import call_filter_with_retry
from smzdm_notice.llm.routing import ResolvedLLMConfig, build_chat_completion_kwargs

T = TypeVar("T")


class StreamEOFError(ValueError):
    """未收到正常结束标记。"""


class InvalidJSONError(ValueError):
    """响应不是完整的 JSON 对象。"""


class InvalidResultError(ValueError):
    """JSON 不符合业务结构。"""


class StreamFinishError(ValueError):
    """截断、拒绝或不支持的结束方式，不能用残缺结果继续。"""


class DecisionTimeoutError(TimeoutError):
    """整个阶段期限耗尽。"""


class StreamIdleTimeoutError(TimeoutError):
    """连续没有响应数据。"""


class AsyncClientPool(AsyncExitStack):
    def __init__(self) -> None:
        super().__init__()
        self._clients: dict[tuple, AsyncOpenAI] = {}

    def get(self, config: ResolvedLLMConfig) -> AsyncOpenAI:
        key = (config.connection, config.base_url, config.api_key, config.timeout_seconds)
        if key not in self._clients:
            client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=0,
            )
            self._clients[key] = client
            self.push_async_callback(client.close)
        return self._clients[key]


def total_timeout(config: ResolvedLLMConfig) -> float:
    return (
        config.total_timeout_seconds
        if config.total_timeout_seconds is not None
        else (600.0 if config.agent == "filter" else 300.0)
    )


def complete_json_object(content: str) -> dict:
    # 只接受完整对象或完整代码块，不扫描残缺响应里的内层对象。
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as error:
        raise InvalidJSONError("响应不是完整 JSON") from error
    if not isinstance(data, dict):
        raise InvalidResultError("响应必须是 JSON 对象")
    return data


@dataclass
class StreamStats:
    started: float
    chunks: int = 0
    content_chars: int = 0
    reasoning_chars: int = 0
    first_chunk: float | None = None
    first_content: float | None = None
    request_id: str = "-"
    completion_id: str = "-"
    finish_reason: str = "-"
    usage: dict | None = None

    def summary(self) -> str:
        return (
            f"elapsed={monotonic() - self.started:.3f}s chunks={self.chunks} "
            f"content_chars={self.content_chars} reasoning_chars={self.reasoning_chars} "
            f"first_chunk={self.first_chunk} first_content={self.first_content} "
            f"finish_reason={self.finish_reason} request_id={self.request_id} "
            f"completion_id={self.completion_id} usage={json.dumps(self.usage, ensure_ascii=False)}"
        )


async def _report_progress(stats: StreamStats, context: str) -> None:
    while True:
        await asyncio.sleep(30)
        logger.info(f"{context} progress {stats.summary()}")


async def _collect(
    client: AsyncOpenAI,
    config: ResolvedLLMConfig,
    messages: list[dict[str, Any]],
    context: str,
    deadline: float,
) -> str:
    stats = StreamStats(started=monotonic())
    progress = asyncio.create_task(_report_progress(stats, context))
    stream = None
    parts: list[str] = []
    end_reason = "failed"
    try:
        kwargs = build_chat_completion_kwargs(config, messages)
        kwargs.update(stream=True, stream_options={"include_usage": True})
        try:
            stream = await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=config.timeout_seconds)
        except asyncio.TimeoutError as error:
            raise StreamIdleTimeoutError(f"连续 {config.timeout_seconds:g} 秒无响应数据") from error
        # HTTP 请求 ID 和 completion ID 分开记录，避免混淆。
        headers = stream.response.headers
        stats.request_id = headers.get("x-request-id") or headers.get("request-id") or "-"
        iterator = stream.__aiter__()
        while True:
            # Python 3.9 的嵌套 wait_for 在数据与取消同时就绪时可能吞掉取消。
            # 每个读取边界再核对绝对期限，不能让持续生成越过总时限。
            if monotonic() >= deadline:
                raise DecisionTimeoutError("LLM 阶段总期限耗尽")
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=config.timeout_seconds)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as error:
                raise StreamIdleTimeoutError(f"连续 {config.timeout_seconds:g} 秒无响应数据") from error
            if monotonic() >= deadline:
                raise DecisionTimeoutError("LLM 阶段总期限耗尽")
            elapsed = monotonic() - stats.started
            stats.chunks += 1
            if stats.first_chunk is None:
                stats.first_chunk = elapsed
                logger.info(f"{context} first_chunk={elapsed:.3f}s request_id={stats.request_id}")
            stats.completion_id = getattr(chunk, "id", None) or stats.completion_id
            if chunk.usage is not None:
                stats.usage = chunk.usage.model_dump(exclude_none=True)
            for choice in chunk.choices:
                if choice.index != 0:
                    continue
                delta = choice.delta
                if getattr(delta, "refusal", None):
                    raise StreamFinishError("模型拒绝生成结果")
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if isinstance(reasoning, str):
                    stats.reasoning_chars += len(reasoning)
                if delta.content:
                    if stats.first_content is None:
                        stats.first_content = elapsed
                        logger.info(f"{context} first_content={elapsed:.3f}s")
                    parts.append(delta.content)
                    stats.content_chars += len(delta.content)
                if choice.finish_reason is not None:
                    stats.finish_reason = choice.finish_reason
                    if choice.finish_reason != "stop":
                        raise StreamFinishError(f"异常结束: {choice.finish_reason}")
        if stats.finish_reason != "stop":
            raise StreamEOFError("流结束但没有 stop 标记")
        end_reason = "stop"
        return "".join(parts)
    except BaseException as error:
        end_reason = type(error).__name__
        raise
    finally:
        progress.cancel()
        with suppress(asyncio.CancelledError):
            await progress
        try:
            if stream is not None:
                await stream.close()
        finally:
            logger.info(f"{context} stream_end={end_reason} {stats.summary()}")


async def stream_result(
    client: AsyncOpenAI,
    config: ResolvedLLMConfig,
    messages: list[dict[str, Any]],
    parse: Callable[[str], T],
    *,
    label: str,
    round_id: str,
    deadline: float,
) -> tuple[T, str]:
    """同一个绝对期限覆盖所有请求、流读取、解析、冷却和重试。"""

    async def request(context: str) -> tuple[T, str]:
        content = await _collect(client, config, messages, context, deadline)
        result = parse(content)
        if monotonic() >= deadline:
            raise DecisionTimeoutError("LLM 阶段总期限耗尽")
        return result, content

    remaining = deadline - monotonic()
    if remaining <= 0:
        raise DecisionTimeoutError("LLM 阶段总期限耗尽")
    try:
        return await asyncio.wait_for(
            call_filter_with_retry(
                request,
                config,
                label,
                round_id=round_id,
                deadline=deadline,
                retryable_errors=(StreamEOFError, InvalidJSONError, StreamIdleTimeoutError),
            ),
            timeout=remaining,
        )
    except asyncio.TimeoutError as error:
        if monotonic() < deadline:
            raise
        raise DecisionTimeoutError("LLM 阶段总期限耗尽") from error
