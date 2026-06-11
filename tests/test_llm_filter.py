from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import APITimeoutError, BadRequestError, RateLimitError

from smzdm_notice.llm import clients as llm_clients
from smzdm_notice.llm.arbitration import ArbitrationRequest, arbitrate
from smzdm_notice.llm.clients import _clear_client_cache, get_client_for_config
from smzdm_notice.llm.filter import _build_prompt_context, _single_llm_call, filter_items
from smzdm_notice.llm.models import FilterResult, LLMCallOutcome, LLMCallResult, Recommendation
from smzdm_notice.llm.routing import ResolvedLLMConfig, RoutingSnapshot
from smzdm_notice.smzdm.ranking import RankingItem


def _item(
    article_id: str = "1001",
    worthy: int = 100,
    unworthy: int = 1,
    comments: int = 10,
    favorites: int = 20,
) -> RankingItem:
    return RankingItem(
        rank=1,
        title="测试商品",
        article_id=article_id,
        price="9.9元",
        worthy=worthy,
        unworthy=unworthy,
        comments=comments,
        favorites=favorites,
        mall="测试商城",
        brand="测试品牌",
        link=f"https://example.com/deal/{article_id}",
    )


def _openai_request() -> httpx.Request:
    return httpx.Request("POST", "https://llm.example.com/v1/chat/completions")


def _openai_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_openai_request())


def _failing_client(error: Exception) -> SimpleNamespace:
    class FailingCompletions:
        def create(self, **kwargs):
            raise error

    return SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))


def _llm_config(
    agent: str = "filter",
    connection: str = "test",
    api_key: str = "key",
    base_url: str = "https://llm.example.com",
    timeout: float = 300.0,
    max_retries: int = 2,
    model_id: str = "model",
    temperature: float = 0.3,
) -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        agent=agent,
        connection=connection,
        connection_label=connection,
        provider="openai_compatible",
        base_url=base_url,
        api_key_env="LLM_TEST_API_KEY",
        api_key=api_key,
        model_id=model_id,
        timeout_seconds=timeout,
        max_retries=max_retries,
        temperature=temperature,
        response_format={"type": "json_object"},
        extra_body={},
    )


def _routing_snapshot() -> RoutingSnapshot:
    raw = {
        "connections": {
            "test": {
                "provider": "openai_compatible",
                "label": "Test",
                "base_url": "https://llm.example.com",
                "api_key_env": "LLM_TEST_API_KEY",
            }
        },
        "defaults": {"connection": "test", "model_id": "model"},
        "agents": {
            "filter": {"request": {"temperature": 0.3}},
            "arbiter": {"request": {"temperature": 0.0}},
            "draft": {"request": {"temperature": 0.0}},
        },
    }
    return RoutingSnapshot(raw=raw, version=1, path=Path("llm_models.json"), source="test")


def _filter_items_with_mocked_call(*outcomes: LLMCallOutcome, dual_filter: bool = False):
    with (
        patch("smzdm_notice.core.config.LLM_DUAL_FILTER", dual_filter),
        patch("smzdm_notice.llm.filter.get_client_for_config", return_value=object()),
        patch("smzdm_notice.llm.filter._single_llm_call", side_effect=list(outcomes)),
        patch("smzdm_notice.core.config.PREFILTER_ENABLED", False),
        patch.dict("os.environ", {"LLM_TEST_API_KEY": "key"}, clear=False),
    ):
        return filter_items(
            items=[_item()],
            user_prompt="用户偏好",
            inventory_data="库存",
            routing_snapshot=_routing_snapshot(),
        )


class LlmClientReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_client_cache()

    def tearDown(self) -> None:
        _clear_client_cache()

    def test_client_reuses_same_connection_slot(self) -> None:
        created = []

        def openai_factory(**kwargs):
            created.append(kwargs)
            return SimpleNamespace(name=f"client-{len(created)}")

        with patch("smzdm_notice.llm.clients.OpenAI", side_effect=openai_factory):
            first = get_client_for_config(_llm_config(timeout=123.0, max_retries=4))
            second = get_client_for_config(_llm_config(timeout=123.0, max_retries=4))

        self.assertIs(first, second)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["timeout"], 123.0)
        self.assertEqual(created[0]["max_retries"], 4)

    def test_client_connections_do_not_share_even_when_config_matches(self) -> None:
        created = []

        def openai_factory(**kwargs):
            created.append(kwargs)
            return SimpleNamespace(name=f"client-{len(created)}")

        with patch("smzdm_notice.llm.clients.OpenAI", side_effect=openai_factory):
            filter_client = get_client_for_config(_llm_config(connection="filter"))
            arbiter_client = get_client_for_config(_llm_config(connection="arbiter"))
            draft_client = get_client_for_config(_llm_config(connection="draft"))

        self.assertIsNot(filter_client, arbiter_client)
        self.assertIsNot(arbiter_client, draft_client)
        self.assertEqual(len(created), 3)
        self.assertEqual([kwargs["timeout"] for kwargs in created], [300.0, 300.0, 300.0])

    def test_clients_use_connection_specific_timeout(self) -> None:
        created = []

        def openai_factory(**kwargs):
            created.append(kwargs)
            return SimpleNamespace(name=f"client-{len(created)}")

        with patch("smzdm_notice.llm.clients.OpenAI", side_effect=openai_factory):
            get_client_for_config(
                _llm_config(connection="filter", api_key="filter-key", base_url="https://filter.example.com", timeout=111.0)
            )
            get_client_for_config(
                _llm_config(connection="arbiter", api_key="arbiter-key", base_url="https://arbiter.example.com", timeout=222.0)
            )
            get_client_for_config(
                _llm_config(connection="draft", api_key="draft-key", base_url="https://draft.example.com", timeout=333.0)
            )

        self.assertEqual([kwargs["timeout"] for kwargs in created], [111.0, 222.0, 333.0])
        self.assertEqual(created[2]["api_key"], "draft-key")
        self.assertEqual(created[2]["base_url"], "https://draft.example.com")

    def test_client_cache_keeps_distinct_specs_without_thrashing(self) -> None:
        created = []

        def openai_factory(**kwargs):
            created.append(kwargs)
            return SimpleNamespace(name=f"client-{len(created)}")

        with patch("smzdm_notice.llm.clients.OpenAI", side_effect=openai_factory):
            first = get_client_for_config(_llm_config())
            second = get_client_for_config(_llm_config(base_url="https://other.example.com"))
            third = get_client_for_config(_llm_config())

        self.assertIsNot(first, second)
        self.assertIs(first, third)
        self.assertEqual(len(created), 2)
        self.assertEqual(created[1]["base_url"], "https://other.example.com")

    def test_client_cache_keeps_same_connection_with_different_timeouts(self) -> None:
        created = []

        def openai_factory(**kwargs):
            created.append(kwargs)
            return SimpleNamespace(name=f"client-{len(created)}")

        with patch("smzdm_notice.llm.clients.OpenAI", side_effect=openai_factory):
            first = get_client_for_config(_llm_config(timeout=300.0))
            second = get_client_for_config(_llm_config(timeout=120.0))
            third = get_client_for_config(_llm_config(timeout=300.0))

        self.assertIsNot(first, second)
        self.assertIs(first, third)
        self.assertEqual(len(created), 2)
        self.assertEqual(created[1]["timeout"], 120.0)

    def test_client_cache_evicts_least_recently_used_spec(self) -> None:
        created = []

        def openai_factory(**kwargs):
            client = SimpleNamespace(name=f"client-{len(created) + 1}", close=Mock())
            created.append((kwargs, client))
            return client

        with (
            patch("smzdm_notice.llm.clients.OpenAI", side_effect=openai_factory),
            patch.object(llm_clients, "CLIENT_CACHE_MAX_SIZE", 2),
        ):
            first = get_client_for_config(_llm_config(base_url="https://one.example.com"))
            second = get_client_for_config(_llm_config(base_url="https://two.example.com"))
            first_again = get_client_for_config(_llm_config(base_url="https://one.example.com"))
            third = get_client_for_config(_llm_config(base_url="https://three.example.com"))
            second_again = get_client_for_config(_llm_config(base_url="https://two.example.com"))

        self.assertIs(first, first_again)
        self.assertIsNot(second, second_again)
        self.assertIsNot(third, second_again)
        self.assertEqual(len(created), 4)
        self.assertEqual(created[-1][0]["base_url"], "https://two.example.com")
        second.close.assert_called_once_with()
        first.close.assert_called_once_with()
        third.close.assert_not_called()
        second_again.close.assert_not_called()

    def test_clear_client_cache_closes_cached_clients(self) -> None:
        created = []

        def openai_factory(**kwargs):
            client = SimpleNamespace(name=f"client-{len(created) + 1}", close=Mock())
            created.append(client)
            return client

        with patch("smzdm_notice.llm.clients.OpenAI", side_effect=openai_factory):
            get_client_for_config(_llm_config(base_url="https://one.example.com"))
            get_client_for_config(_llm_config(base_url="https://two.example.com"))
            _clear_client_cache()

        for client in created:
            client.close.assert_called_once_with()

    def test_client_close_failure_does_not_interrupt_eviction(self) -> None:
        first = SimpleNamespace(close=Mock(side_effect=RuntimeError("close failed")))
        second = SimpleNamespace(close=Mock())
        third = SimpleNamespace(close=Mock())

        with (
            patch("smzdm_notice.llm.clients.OpenAI", side_effect=[first, second, third]),
            patch.object(llm_clients, "CLIENT_CACHE_MAX_SIZE", 2),
        ):
            get_client_for_config(_llm_config(base_url="https://one.example.com"))
            get_client_for_config(_llm_config(base_url="https://two.example.com"))
            get_client_for_config(_llm_config(base_url="https://three.example.com"))

        first.close.assert_called_once_with()


class LlmPromptContextTests(unittest.TestCase):
    def test_build_prompt_context_reuses_item_summary_without_losing_arbiter_fields(self) -> None:
        item = _item()
        item.rank = 7
        item.link = "https://example.com/deal/1001"

        with patch.object(item, "to_llm_summary", wraps=item.to_llm_summary) as to_llm_summary:
            context = _build_prompt_context([item], "用户偏好", "库存")

        to_llm_summary.assert_called_once_with()
        self.assertNotIn("rank", context.items_summary[0])
        self.assertEqual(context.arbiter_items[item.article_id]["rank"], 7)
        self.assertEqual(context.arbiter_items[item.article_id]["link"], "https://example.com/deal/1001")


class LlmFilterDiagnosticsTests(unittest.TestCase):
    def test_low_quality_items_still_enter_llm_request(self) -> None:
        captured = {}

        def fake_call(client, model, user_message):
            captured["user_message"] = user_message
            return LLMCallOutcome(result=LLMCallResult(result=FilterResult()))

        user_prompt = '## 质量门槛\n- **"值"票数 ≥ 20**\n- **"值" / "不值" 比值 ≥ 3:2**'
        with (
            patch("smzdm_notice.core.config.LLM_DUAL_FILTER", False),
            patch(
                "smzdm_notice.core.config.PREFILTER_ENABLED",
                False,
            ),
            patch("smzdm_notice.core.config.PREFILTER_MIN_WORTHY", 999),
            patch(
                "smzdm_notice.core.config.PREFILTER_MIN_COMMENTS",
                999,
            ),
            patch("smzdm_notice.llm.filter.get_client_for_config", return_value=object()),
            patch("smzdm_notice.llm.filter._single_llm_call", side_effect=fake_call),
        ):
            filter_items(
                items=[_item(article_id="low-worthy", worthy=1, unworthy=100)],
                user_prompt=user_prompt,
                inventory_data="库存",
                routing_snapshot=_routing_snapshot(),
            )

        self.assertIn("low-worthy", captured["user_message"])
        self.assertIn('"worthy": 1', captured["user_message"])
        self.assertIn('"unworthy": 100', captured["user_message"])

    def test_prefilter_requires_all_regular_metrics_when_enabled(self) -> None:
        captured = {}

        def fake_call(client, model, user_message):
            captured["user_message"] = user_message
            return LLMCallOutcome(result=LLMCallResult(result=FilterResult()))

        items = [
            _item(article_id="pass", worthy=30, unworthy=10, comments=20, favorites=5),
            _item(article_id="low-comments", worthy=30, unworthy=10, comments=1, favorites=5),
            _item(article_id="low-rate", worthy=30, unworthy=90, comments=20, favorites=5),
        ]
        with (
            patch("smzdm_notice.core.config.LLM_DUAL_FILTER", False),
            patch(
                "smzdm_notice.core.config.PREFILTER_ENABLED",
                True,
            ),
            patch(
                "smzdm_notice.core.config.PREFILTER_BYPASS_ENABLED",
                False,
            ),
            patch("smzdm_notice.core.config.PREFILTER_MIN_WORTHY", 20),
            patch("smzdm_notice.core.config.PREFILTER_MIN_WORTHY_RATE", 0.5),
            patch("smzdm_notice.core.config.PREFILTER_MIN_COMMENTS", 10),
            patch("smzdm_notice.core.config.PREFILTER_MIN_FAVORITES", 5),
            patch("smzdm_notice.llm.filter.get_client_for_config", return_value=object()),
            patch("smzdm_notice.llm.filter._single_llm_call", side_effect=fake_call),
        ):
            filter_items(items, "用户偏好", "库存", routing_snapshot=_routing_snapshot())

        self.assertIn("pass", captured["user_message"])
        self.assertNotIn("low-comments", captured["user_message"])
        self.assertNotIn("low-rate", captured["user_message"])

    def test_prefilter_bypass_disabled_does_not_override_regular_metrics(self) -> None:
        with (
            patch("smzdm_notice.core.config.LLM_DUAL_FILTER", False),
            patch(
                "smzdm_notice.core.config.PREFILTER_ENABLED",
                True,
            ),
            patch(
                "smzdm_notice.core.config.PREFILTER_BYPASS_ENABLED",
                False,
            ),
            patch("smzdm_notice.core.config.PREFILTER_MIN_WORTHY", 100),
            patch("smzdm_notice.core.config.PREFILTER_MIN_WORTHY_RATE", 0.9),
            patch("smzdm_notice.core.config.PREFILTER_MIN_COMMENTS", 100),
            patch("smzdm_notice.core.config.PREFILTER_MIN_FAVORITES", 100),
            patch("smzdm_notice.core.config.PREFILTER_BYPASS_MIN_COMMENTS", 10),
            patch("smzdm_notice.core.config.PREFILTER_BYPASS_MIN_WORTHY", 10),
            patch("smzdm_notice.llm.filter._single_llm_call") as llm_call,
        ):
            result = filter_items(
                [_item(article_id="bypass-would-match", worthy=10, unworthy=1, comments=10)],
                "用户偏好",
                "库存",
                "model",
            )

        llm_call.assert_not_called()
        self.assertEqual(result.matched, [])

    def test_prefilter_bypasses_on_comments_when_enabled(self) -> None:
        captured = {}

        def fake_call(client, model, user_message):
            captured["user_message"] = user_message
            return LLMCallOutcome(result=LLMCallResult(result=FilterResult()))

        with (
            patch("smzdm_notice.core.config.LLM_DUAL_FILTER", False),
            patch(
                "smzdm_notice.core.config.PREFILTER_ENABLED",
                True,
            ),
            patch(
                "smzdm_notice.core.config.PREFILTER_BYPASS_ENABLED",
                True,
            ),
            patch("smzdm_notice.core.config.PREFILTER_MIN_WORTHY", 100),
            patch("smzdm_notice.core.config.PREFILTER_MIN_WORTHY_RATE", 0.9),
            patch("smzdm_notice.core.config.PREFILTER_MIN_COMMENTS", 200),
            patch("smzdm_notice.core.config.PREFILTER_MIN_FAVORITES", 100),
            patch("smzdm_notice.core.config.PREFILTER_BYPASS_MIN_COMMENTS", 100),
            patch("smzdm_notice.core.config.PREFILTER_BYPASS_MIN_WORTHY", 999),
            patch("smzdm_notice.llm.filter.get_client_for_config", return_value=object()),
            patch("smzdm_notice.llm.filter._single_llm_call", side_effect=fake_call),
        ):
            filter_items(
                [_item(article_id="comment-bypass", worthy=1, unworthy=99, comments=100, favorites=0)],
                "用户偏好",
                "库存",
                routing_snapshot=_routing_snapshot(),
            )

        self.assertIn("comment-bypass", captured["user_message"])

    def test_prefilter_bypasses_on_worthy_when_enabled(self) -> None:
        captured = {}

        def fake_call(client, model, user_message):
            captured["user_message"] = user_message
            return LLMCallOutcome(result=LLMCallResult(result=FilterResult()))

        with (
            patch("smzdm_notice.core.config.LLM_DUAL_FILTER", False),
            patch(
                "smzdm_notice.core.config.PREFILTER_ENABLED",
                True,
            ),
            patch(
                "smzdm_notice.core.config.PREFILTER_BYPASS_ENABLED",
                True,
            ),
            patch("smzdm_notice.core.config.PREFILTER_MIN_WORTHY", 200),
            patch("smzdm_notice.core.config.PREFILTER_MIN_WORTHY_RATE", 0.9),
            patch("smzdm_notice.core.config.PREFILTER_MIN_COMMENTS", 100),
            patch("smzdm_notice.core.config.PREFILTER_MIN_FAVORITES", 100),
            patch("smzdm_notice.core.config.PREFILTER_BYPASS_MIN_COMMENTS", 999),
            patch("smzdm_notice.core.config.PREFILTER_BYPASS_MIN_WORTHY", 100),
            patch("smzdm_notice.llm.filter.get_client_for_config", return_value=object()),
            patch("smzdm_notice.llm.filter._single_llm_call", side_effect=fake_call),
        ):
            filter_items(
                [_item(article_id="worthy-bypass", worthy=100, unworthy=100, comments=0, favorites=0)],
                "用户偏好",
                "库存",
                routing_snapshot=_routing_snapshot(),
            )

        self.assertIn("worthy-bypass", captured["user_message"])

    def test_filter_items_model_argument_overrides_routed_model(self) -> None:
        captured = {}

        def fake_call(client, llm_config, user_message):
            captured["model_id"] = llm_config.model_id
            return LLMCallOutcome(result=LLMCallResult(result=FilterResult()))

        with (
            patch("smzdm_notice.core.config.LLM_DUAL_FILTER", False),
            patch("smzdm_notice.core.config.PREFILTER_ENABLED", False),
            patch("smzdm_notice.llm.filter.get_client_for_config", return_value=object()),
            patch("smzdm_notice.llm.filter._single_llm_call", side_effect=fake_call),
        ):
            filter_items(
                [_item()],
                "用户偏好",
                "库存",
                model="override-model",
                routing_snapshot=_routing_snapshot(),
            )

        self.assertEqual(captured["model_id"], "override-model")

    def test_single_call_failure_marks_llm_failed(self) -> None:
        result = _filter_items_with_mocked_call(
            LLMCallOutcome(error_summary="usage limit"),
            dual_filter=False,
        )

        self.assertTrue(result.diagnostics.llm_failed)
        self.assertEqual(result.diagnostics.error_summary, "usage limit")

    def test_dual_call_failures_mark_llm_failed(self) -> None:
        result = _filter_items_with_mocked_call(
            LLMCallOutcome(error_summary="429 a"),
            LLMCallOutcome(error_summary="429 b"),
            dual_filter=True,
        )

        self.assertTrue(result.diagnostics.llm_failed)
        self.assertIn("429 a", result.diagnostics.error_summary)
        self.assertIn("429 b", result.diagnostics.error_summary)

    def test_dual_call_partial_failure_is_not_llm_failed(self) -> None:
        result = _filter_items_with_mocked_call(
            LLMCallOutcome(error_summary="429 a"),
            LLMCallOutcome(result=LLMCallResult(result=FilterResult())),
            dual_filter=True,
        )

        self.assertFalse(result.diagnostics.llm_failed)

    def test_successful_empty_result_is_not_llm_failed(self) -> None:
        result = _filter_items_with_mocked_call(
            LLMCallOutcome(result=LLMCallResult(result=FilterResult())),
            dual_filter=False,
        )

        self.assertFalse(result.diagnostics.llm_failed)

    def test_single_llm_call_classifies_retryable_error(self) -> None:
        outcome = _single_llm_call(
            _failing_client(RateLimitError("rate limited", response=_openai_response(429), body=None)),
            _llm_config(),
            "user message",
        )

        self.assertIsNone(outcome.result)
        self.assertIn("可重试/网络类问题", outcome.error_summary)

    def test_single_llm_call_classifies_timeout_error(self) -> None:
        outcome = _single_llm_call(_failing_client(APITimeoutError(_openai_request())), _llm_config(), "user message")

        self.assertIsNone(outcome.result)
        self.assertIn("可重试/网络类问题", outcome.error_summary)

    def test_single_llm_call_classifies_non_retryable_error(self) -> None:
        outcome = _single_llm_call(
            _failing_client(BadRequestError("bad request", response=_openai_response(400), body=None)),
            _llm_config(),
            "user message",
        )

        self.assertIsNone(outcome.result)
        self.assertIn("配置或请求不可重试问题", outcome.error_summary)

    def test_single_llm_call_uses_resolved_request_options(self) -> None:
        captured = {}
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"recommendations": [], "near_misses": []}')
                )
            ]
        )

        def create_completion(**kwargs):
            captured.update(kwargs)
            return response

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)))
        llm_config = replace(_llm_config(temperature=0.8), extra_body={"do_sample": False})

        outcome = _single_llm_call(client, llm_config, "user message")

        self.assertIsNotNone(outcome.result)
        self.assertEqual(captured["model"], "model")
        self.assertEqual(captured["temperature"], 0.8)
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(captured["extra_body"], {"do_sample": False})


class LlmFilterArbitrationTests(unittest.TestCase):
    def test_arbitration_parses_config_change_draft(self) -> None:
        payload = {
            "chosen": "B",
            "reason": "B 更严格遵守黑名单",
            "inconsistency_analysis": "A 过度扩展了黑名单范围。",
            "prompt_optimization_suggestion": "黑名单只能按字面精确匹配。",
            "config_change_draft": {
                "target_file": "preference.md",
                "edit_mode": "append",
                "title": "限制黑名单扩展",
                "summary": "避免把明确黑名单扩展到更大类目。",
                "append_text": "- 明确不感兴趣的品类只做字面精确匹配，不扩展到上级类目。",
            },
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))

        info = arbitrate(
            ArbitrationRequest(
                result_a=FilterResult(recommendations=[Recommendation(id="1", reason="A")]),
                result_b=FilterResult(recommendations=[Recommendation(id="2", reason="B")]),
                raw_a="{}",
                raw_b="{}",
                items_summary=[],
                items_by_id={},
                user_message="用户偏好",
                client=client,
                llm_config=_llm_config(agent="arbiter", model_id="arbiter-model", temperature=0.0),
            )
        )

        self.assertIsNotNone(info)
        self.assertEqual(info.chosen, "B")
        self.assertEqual(info.config_change_draft["target_file"], "preference.md")
        self.assertIn("字面精确匹配", info.config_change_draft["append_text"])

    def test_arbitration_ignores_non_object_config_change_draft(self) -> None:
        payload = {
            "chosen": "A",
            "reason": "A 更准确",
            "inconsistency_analysis": "差异不足以形成规则。",
            "prompt_optimization_suggestion": "无需修改。",
            "config_change_draft": None,
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))

        info = arbitrate(
            ArbitrationRequest(
                result_a=FilterResult(),
                result_b=FilterResult(),
                raw_a="{}",
                raw_b="{}",
                items_summary=[],
                items_by_id={},
                user_message="用户偏好",
                client=client,
                llm_config=_llm_config(agent="arbiter", model_id="arbiter-model", temperature=0.0),
            )
        )

        self.assertIsNotNone(info)
        self.assertIsNone(info.config_change_draft)

    def test_arbitration_returns_none_on_sdk_error(self) -> None:
        info = arbitrate(
            ArbitrationRequest(
                result_a=FilterResult(),
                result_b=FilterResult(),
                raw_a="{}",
                raw_b="{}",
                items_summary=[],
                items_by_id={},
                user_message="用户偏好",
                client=_failing_client(RateLimitError("rate limited", response=_openai_response(429), body=None)),
                llm_config=_llm_config(agent="arbiter", model_id="arbiter-model", temperature=0.0),
            )
        )

        self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main()
