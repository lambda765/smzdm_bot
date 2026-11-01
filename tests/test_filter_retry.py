from __future__ import annotations

import asyncio
from dataclasses import replace
from email.utils import formatdate
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError
from stream_fakes import VirtualClock

from smzdm_notice.llm import filter_retry
from smzdm_notice.llm.filter_retry import ConnectionCooldown, FilterCooldownError, call_filter_with_retry
from smzdm_notice.llm.routing import ResolvedLLMConfig


@pytest.fixture
def config():
    return ResolvedLLMConfig(
        agent="filter",
        connection="test",
        connection_label="Test",
        provider="openai_compatible",
        base_url="https://llm.example.com/v1",
        api_key_env="TEST_KEY",
        api_key="secret",
        model_id="test-model",
        timeout_seconds=300,
        max_retries=1,
        temperature=0,
        response_format={"type": "json_object"},
        extra_body={},
        total_timeout_seconds=600,
    )


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    clock = VirtualClock()
    monkeypatch.setattr(filter_retry, "monotonic", clock.monotonic)
    monkeypatch.setattr(filter_retry, "time", lambda: 1800000000.0)
    monkeypatch.setattr(filter_retry, "uniform", lambda low, high: (low + high) / 2)
    monkeypatch.setattr(filter_retry, "_COOLDOWN", ConnectionCooldown())
    return clock


def status_error(code, headers=None):
    response = httpx.Response(
        code,
        headers=headers or {},
        request=httpx.Request("POST", "https://llm.example.com/v1/chat/completions"),
    )
    return APIStatusError("test failure", response=response, body=None)


def run(clock, config, request, *, budget=600):
    return clock.run(
        asyncio.wait_for(
            call_filter_with_retry(request, config, "A", round_id="test", deadline=clock.now + budget),
            budget,
        )
    )


@pytest.mark.parametrize("code", [408, 409, 500, 502, 503, 599])
def test_transient_status_retries_once(config, clock, code):
    request = AsyncMock(side_effect=[status_error(code), "ok"])
    assert run(clock, config, request) == "ok"
    assert request.call_count == 2
    assert clock.now == 1002


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 600])
def test_other_status_does_not_retry(config, clock, code):
    request = AsyncMock(side_effect=status_error(code))
    with pytest.raises(APIStatusError):
        run(clock, config, request)
    assert request.call_count == 1
    assert clock.now == 1000


@pytest.mark.parametrize("retries,expected", [(0, 1), (1, 2), (2, 2), (5, 2)])
def test_timeout_attempts_are_capped_and_deadline_does_not_reset(config, clock, retries, expected):
    async def timeout(_context):
        await asyncio.sleep(300)
        raise APITimeoutError(httpx.Request("POST", config.base_url))

    request = AsyncMock(side_effect=timeout)
    with pytest.raises((APITimeoutError, asyncio.TimeoutError)):
        run(clock, replace(config, max_retries=retries), request)
    assert request.call_count == expected
    assert clock.now == (1600 if expected == 2 else 1300)


@pytest.mark.parametrize("error_type", [APIConnectionError, httpx.ReadError, httpx.RemoteProtocolError])
def test_connection_error_then_success(config, clock, error_type):
    error = (
        error_type(request=httpx.Request("POST", config.base_url))
        if error_type is APIConnectionError
        else error_type("disconnect")
    )
    request = AsyncMock(side_effect=[error, "ok"])
    assert run(clock, config, request) == "ok"
    assert request.call_count == 2


@pytest.mark.parametrize(
    "headers,wait",
    [
        ({}, 60),
        ({"retry-after-ms": "2500", "retry-after": "30"}, 2.5),
        ({"retry-after-ms": "bad", "retry-after": "2"}, 2),
        ({"retry-after": formatdate(1800000010, usegmt=True)}, 10),
        ({"retry-after": "0"}, 0),
        ({"retry-after": "-1"}, 60),
        ({"retry-after": "nan"}, 60),
        ({"retry-after": "inf"}, 60),
        ({"retry-after": "invalid"}, 60),
    ],
)
def test_rate_limit_header_and_jitter(config, clock, headers, wait):
    request = AsyncMock(side_effect=[status_error(429, headers), "ok"])
    assert run(clock, config, request) == "ok"
    assert request.call_count == 2
    assert clock.now == 1000 + wait + 1.5


def test_long_cooldown_fails_both_lanes_without_sleep(config, clock):
    a = AsyncMock(side_effect=status_error(429, {"retry-after": "120"}))
    with pytest.raises(APIStatusError):
        run(clock, config, a)
    b = AsyncMock(return_value="ok")
    with pytest.raises(FilterCooldownError, match="限流冷却"):
        run(clock, config, b)
    assert a.call_count == 1
    b.assert_not_called()
    assert clock.now == 1000
    clock.now += 61
    assert run(clock, config, b) == "ok"
    assert clock.now == 1120


def test_no_retry_still_publishes_cooldown_to_other_lane(config, clock):
    config = replace(config, max_retries=0)
    with pytest.raises(APIStatusError):
        run(clock, config, AsyncMock(side_effect=status_error(429, {"retry-after": "12"})))
    assert run(clock, config, AsyncMock(return_value="ok")) == "ok"
    assert clock.now == 1012


def test_connection_or_key_change_does_not_inherit_cooldown(config, clock):
    filter_retry._COOLDOWN.extend(ConnectionCooldown.key(config), 120)
    for other in (replace(config, connection="other"), replace(config, api_key="new-key")):
        assert run(clock, other, AsyncMock(return_value="ok")) == "ok"
    assert clock.now == 1000


def test_cooldown_extension_cannot_prolong_wait_indefinitely(config, clock):
    cooldown = filter_retry._COOLDOWN
    key = cooldown.key(config)
    cooldown.extend(key, 30)
    request = AsyncMock()

    async def work():
        async def extend():
            await asyncio.sleep(20)
            cooldown.extend(key, 50)

        await asyncio.gather(extend(), call_filter_with_retry(request, config, "B", round_id="test", deadline=1600))

    with pytest.raises(FilterCooldownError):
        clock.run(work())
    request.assert_not_called()
    assert clock.now == 1030


def test_simultaneous_429s_share_longest_cooldown_without_blocking(config, clock):
    starts = []
    calls = {"A": 0, "B": 0}

    async def work():
        ready = asyncio.Event()

        async def request(label, _context):
            calls[label] += 1
            starts.append((label, clock.now))
            if calls[label] == 1:
                if len(starts) == 2:
                    ready.set()
                await ready.wait()
                raise status_error(429, {"retry-after": "10" if label == "A" else "20"})
            return label

        return await asyncio.gather(
            *[
                call_filter_with_retry(
                    lambda ctx, label=label: request(label, ctx), config, label, round_id="test", deadline=1600
                )
                for label in ("A", "B")
            ]
        )

    assert clock.run(work()) == ["A", "B"]
    assert starts[:2] == [("A", 1000), ("B", 1000)]
    assert all(t >= 1020 for _, t in starts[2:])
    assert calls == {"A": 2, "B": 2}


@pytest.mark.parametrize("budget", [1, 30])
def test_retry_wait_and_cooldown_respect_remaining_deadline(config, clock, budget):
    code = 500 if budget == 1 else 429
    request = AsyncMock(side_effect=status_error(code))
    with pytest.raises(asyncio.TimeoutError):
        run(clock, config, request, budget=budget)
    assert request.call_count == 1
    assert clock.now == 1000 + budget
