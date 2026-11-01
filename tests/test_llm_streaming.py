from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI
from stream_fakes import FakeStream, VirtualClock, chunk, fake_client, text_stream
from test_filter_retry import config as retry_config
from test_filter_retry import status_error
from test_llm_filter import _item

from smzdm_notice.llm import arbitration, filter_retry, streaming
from smzdm_notice.llm import filter as llm_filter
from smzdm_notice.llm.clients import _clear_client_cache, get_client_for_config
from smzdm_notice.llm.filter import _parse_response, _single_llm_call
from smzdm_notice.llm.models import FilterResult
from smzdm_notice.llm.streaming import AsyncClientPool, InvalidJSONError, InvalidResultError

config = retry_config

EMPTY = '{"recommendations": [], "near_misses": []}'
REC = '{"recommendations": [{"id":"1001","reason":"good"}], "near_misses": []}'


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    clock = VirtualClock()
    for module in (streaming, filter_retry, llm_filter, arbitration):
        monkeypatch.setattr(module, "monotonic", clock.monotonic)
    monkeypatch.setattr(filter_retry, "uniform", lambda low, high: (low + high) / 2)
    monkeypatch.setattr(filter_retry, "_COOLDOWN", filter_retry.ConnectionCooldown())
    return clock


def test_interleaved_reasoning_empty_and_usage_chunks_keep_only_complete_json(config, clock):
    stream = FakeStream(
        [
            chunk(empty=True),
            chunk(reasoning="secret reasoning"),
            chunk(REC[:15]),
            chunk(REC[15:40], reasoning="more private reasoning"),
            chunk(REC[40:]),
            chunk(finish="stop"),
            chunk(empty=True, usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}),
        ]
    )
    client = fake_client(stream)
    logs = []
    sink = streaming.logger.add(lambda message: logs.append(str(message)))
    try:
        outcome = clock.run(_single_llm_call(client, config, "prompt"))
    finally:
        streaming.logger.remove(sink)
    assert outcome.result.raw_content == REC
    assert outcome.result.result.recommendations[0].id == "1001"
    assert stream.closed
    assert "secret reasoning" not in "".join(logs)
    assert '"total_tokens": 120' in "".join(logs)
    assert "request_id=request-test" in "".join(logs)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs


def test_data_keeps_read_alive_past_300_seconds(config, clock):
    stream = FakeStream([(200, chunk(reasoning="a")), (200, chunk(EMPTY)), chunk(finish="stop")])
    outcome = clock.run(_single_llm_call(fake_client(stream), config, "prompt"))
    assert outcome.result.result == FilterResult()
    assert stream.closed
    assert clock.now == 1400


@pytest.mark.parametrize("scenario,elapsed,attempts", [("idle", 600, 2), ("idle_no_retry", 300, 1), ("busy", 600, 1)])
def test_deadline_cancels_reader_and_closes_streams(config, clock, scenario, elapsed, attempts):
    if scenario == "busy":
        streams = [FakeStream([(100, chunk(reasoning="thinking")) for _ in range(10)])]
    else:
        streams = [FakeStream([(1000, chunk(EMPTY))]) for _ in range(2)]
    client = fake_client(*streams)
    if scenario == "idle_no_retry":
        config = replace(config, max_retries=0)
    outcome = clock.run(_single_llm_call(client, config, "prompt"))
    assert outcome.result is None
    assert clock.now == 1000 + elapsed
    assert client.chat.completions.create.call_count == attempts
    assert all(stream.closed for stream in streams[:attempts])
    if scenario != "busy":
        assert all(stream.cancelled for stream in streams[:attempts])
    assert ("总期限耗尽" in outcome.error_summary) == (scenario != "idle_no_retry")


def test_no_headers_cancels_create_and_retries_within_total_budget(config, clock):
    cancelled = []

    async def create(**kwargs):
        try:
            await asyncio.sleep(1000)
        finally:
            cancelled.append(True)

    client = fake_client()
    client.chat.completions.create = AsyncMock(side_effect=create)
    outcome = clock.run(_single_llm_call(client, config, "prompt"))
    assert outcome.result is None and len(cancelled) == 2
    assert clock.now == 1600


@pytest.mark.parametrize(
    "first_events",
    [
        [chunk('{"old":'), httpx.ReadError("disconnected")],
        [chunk(REC)],  # 缺少正常结束标记，不能接受内容恰巧完整的 JSON。
        [chunk('{"recommendations": [{"id":"wrong"}'), chunk(finish="stop")],
    ],
)
def test_interrupted_or_invalid_json_retries_without_old_fragments(config, clock, first_events):
    first = FakeStream(first_events)
    second = text_stream(EMPTY)
    client = fake_client(first, second)
    outcome = clock.run(_single_llm_call(client, config, "prompt"))
    assert outcome.result.raw_content == EMPTY
    assert outcome.result.result.recommendations == []
    assert client.chat.completions.create.call_count == 2
    assert first.closed and second.closed


@pytest.mark.parametrize(
    "events",
    [
        [chunk(REC), chunk(finish="length")],
        [chunk(REC), chunk(finish="content_filter")],
        [chunk(refusal="declined")],
        [chunk("{}"), chunk(finish="stop")],
        [chunk('{"recommendations": []}'), chunk(finish="stop")],
        [chunk('{"recommendations": null, "near_misses": []}'), chunk(finish="stop")],
    ],
)
def test_truncated_refused_or_missing_fields_are_failures_without_retry(config, clock, events):
    stream = FakeStream(events)
    client = fake_client(stream)
    outcome = clock.run(_single_llm_call(client, config, "prompt"))
    assert outcome.result is None
    assert client.chat.completions.create.call_count == 1
    assert stream.closed


@pytest.mark.parametrize("text", ["", "[]", "{}", '{"recommendations":[{}', EMPTY + " trailing", EMPTY + EMPTY])
def test_partial_or_wrong_json_is_not_empty_success(text):
    with pytest.raises((InvalidJSONError, InvalidResultError)):
        _parse_response(text)


def test_sdk_stream_does_not_retry_implicitly_and_closes_http_client(config, clock):
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "unavailable"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        async with AsyncOpenAI(
            api_key=config.api_key, base_url=config.base_url, max_retries=0, http_client=http_client
        ) as client:
            client._platform = "MacOS"  # 虚拟时钟不推进 SDK 的平台检测工作线程。
            return await _single_llm_call(client, config, "test")

    assert clock.run(run()).result is None
    assert len(requests) == 2
    assert all(r.headers["x-stainless-retry-count"] == "0" for r in requests)
    assert http_client.is_closed


def test_real_sdk_decodes_stream_and_closes_on_deadline(config, clock):
    class Body(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield ("data: " + chunk(reasoning="hidden").model_dump_json() + "\n\n").encode()
            await asyncio.sleep(1000)

        async def aclose(self):
            self.closed = True

    bodies = []

    async def handler(request):
        body = Body()
        bodies.append(body)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    async def run():
        async with AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            client._platform = "MacOS"
            return await _single_llm_call(client, config, "prompt")

    assert clock.run(run()).result is None
    assert len(bodies) == 2 and all(body.closed for body in bodies)
    assert clock.now == 1600


def test_async_pool_reuses_only_within_loop_and_other_clients_keep_sdk_retries(config, clock, monkeypatch):
    made = []

    def factory(**kwargs):
        client = fake_client()
        made.append((kwargs, client))
        return client

    monkeypatch.setattr(streaming, "AsyncOpenAI", factory)

    async def run():
        async with AsyncClientPool() as pool:
            assert pool.get(config) is pool.get(replace(config, agent="arbiter"))

    clock.run(run())
    clock.run(run())
    assert len(made) == 2
    assert all(kwargs["max_retries"] == 0 and client.close.call_count == 1 for kwargs, client in made)
    _clear_client_cache()
    try:
        client = get_client_for_config(replace(config, agent="draft", max_retries=4))
        assert client.max_retries == 4
    finally:
        _clear_client_cache()


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_dual_requests_start_together_and_preserve_successful_lane(config, clock, monkeypatch, failures):
    starts = []
    streams = []

    async def run():
        ready = asyncio.Event()

        async def create(**kwargs):
            index = len(starts)
            starts.append(clock.now)
            if len(starts) == 2:
                ready.set()
            await ready.wait()
            if index < failures:
                raise status_error(401)
            stream = text_stream(REC)
            streams.append(stream)
            return stream

        client = fake_client()
        client.chat.completions.create = AsyncMock(side_effect=create)
        monkeypatch.setattr(AsyncClientPool, "get", lambda *_: client)
        return await llm_filter._filter_async(
            config, llm_filter._build_prompt_context([_item()], "pref", "inv"), None, True
        )

    result = clock.run(run())
    assert starts == [1000, 1000]
    assert result.diagnostics.llm_failed == (failures == 2)
    assert len(result.matched) == (0 if failures == 2 else 1)
    assert all(s.closed for s in streams)


@pytest.mark.parametrize("timeout", [False, True])
def test_arbiter_uses_same_loop_new_300s_budget_and_intersection_on_timeout(config, clock, monkeypatch, timeout):
    recs_a = json.loads(REC)
    recs_a["recommendations"].append({"id": "1002", "reason": "second"})
    # 每 100 秒有数据，首个正文直到 500 秒才返回。
    a = FakeStream(
        [(100, chunk(reasoning="x")) for _ in range(4)] + [(100, chunk(json.dumps(recs_a))), chunk(finish="stop")]
    )
    b = text_stream(REC)
    payload = json.dumps(
        {"chosen": "A", "change_assessment": {"cause": "soft_judgment", "should_change_preference": False}}
    )
    arb = FakeStream([(100, chunk(reasoning="x")) for _ in range(4)]) if timeout else text_stream(payload)
    client = fake_client(a, b, arb)
    monkeypatch.setattr(AsyncClientPool, "get", lambda *_: client)
    monkeypatch.setattr(arbitration, "resolve", lambda *_: replace(config, agent="arbiter", total_timeout_seconds=300))
    monkeypatch.setattr("smzdm_notice.core.config.LLM_ARBITER_ENABLED", True)
    result = clock.run(
        llm_filter._filter_async(
            config,
            llm_filter._build_prompt_context([_item(), _item("1002")], "pref", "inv"),
            None,
            True,
        )
    )
    assert clock.now == (1800 if timeout else 1500)
    assert len(result.matched) == (1 if timeout else 2)
    assert (result.arbiter_info is None) == timeout
    assert client.chat.completions.create.call_count == 3
    assert a.closed and b.closed and arb.closed


def test_cooling_lane_does_not_block_other_connection(config, clock):
    filter_retry._COOLDOWN.extend(filter_retry.ConnectionCooldown.key(config), 60)
    starts = []

    async def run():
        async def call(cfg):
            client = fake_client()

            async def create(**kwargs):
                starts.append((cfg.connection, clock.now))
                return text_stream(EMPTY)

            client.chat.completions.create = create
            return await _single_llm_call(client, cfg, "prompt")

        return await asyncio.gather(call(config), call(replace(config, connection="other")))

    outcomes = clock.run(run())
    assert all(o.result is not None for o in outcomes)
    assert starts == [("other", 1000), ("test", 1060)]


def test_deadline_closes_pooled_client_and_leaves_no_tasks(config, clock, monkeypatch):
    streams = [FakeStream([(1000, chunk(EMPTY))]) for _ in range(4)]
    client = fake_client(*streams)
    monkeypatch.setattr(streaming, "AsyncOpenAI", lambda **_: client)
    outcome = clock.run(llm_filter._filter_async(
        config, llm_filter._build_prompt_context([_item()], "pref", "inv"), None, True,
    ))
    assert outcome.diagnostics.llm_failed
    assert clock.now == 1600
    assert client.chat.completions.create.call_count == 4
    client.close.assert_awaited_once()
    assert all(s.closed for s in streams)
    assert clock.pending_count == 0


def test_unsupported_stream_is_explicit_failure_without_nonstream_fallback(config, clock):
    client = fake_client(status_error(400))
    outcome = clock.run(_single_llm_call(client, config, "prompt"))
    assert outcome.result is None
    client.chat.completions.create.assert_awaited_once()
    assert client.chat.completions.create.call_args.kwargs["stream"] is True
