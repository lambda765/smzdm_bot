"""流式测试替身和无需等待墙钟的事件循环。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from openai.types.chat import ChatCompletionChunk


def chunk(content=None, *, reasoning=None, finish=None, usage=None, empty=False, refusal=None):
    return ChatCompletionChunk(
        id="completion-test",
        object="chat.completion.chunk",
        created=0,
        model="test",
        choices=[]
        if empty
        else [
            {
                "index": 0,
                "delta": {"content": content, "reasoning_content": reasoning, "refusal": refusal},
                "finish_reason": finish,
            }
        ],
        usage=usage,
    )


class FakeStream:
    def __init__(self, events):
        self.events = iter(events)
        self.closed = False
        self.cancelled = False
        self.response = SimpleNamespace(headers=httpx.Headers({"x-request-id": "request-test"}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = next(self.events, None)
        if event is None:
            raise StopAsyncIteration
        try:
            if isinstance(event, tuple):
                delay, event = event
                await asyncio.sleep(delay)
            if isinstance(event, Exception):
                raise event
            return event
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def close(self):
        self.closed = True


def text_stream(content):
    return FakeStream([chunk(content), chunk(finish="stop")])


def fake_client(*streams):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=list(streams)))),
        close=AsyncMock(),
    )


class VirtualClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def run(self, coroutine):
        loop = asyncio.new_event_loop()
        loop.time = self.monotonic
        selector = loop._selector
        original_select = selector.select

        def select(timeout=None):
            self.now += max(0, timeout or 0)
            return original_select(0)

        selector.select = select
        try:
            return loop.run_until_complete(coroutine)
        finally:
            pending = asyncio.all_tasks(loop)
            self.pending_count = len(pending)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
