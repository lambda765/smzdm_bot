"""LLM 商品筛选主流程。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime

from loguru import logger
from openai import OpenAI

from smzdm_notice.llm.arbitration import resolve_dual_result
from smzdm_notice.llm.clients import get_client_for_config
from smzdm_notice.llm.errors import (
    GENERAL_OPENAI_ERRORS,
    NON_RETRYABLE_OPENAI_ERRORS,
    RETRYABLE_OPENAI_ERRORS,
    error_summary,
)
from smzdm_notice.llm.json_utils import extract_json_object
from smzdm_notice.llm.models import (
    FilterDiagnostics,
    FilterItemsResult,
    FilterResult,
    LLMCallOutcome,
    LLMCallResult,
)
from smzdm_notice.llm.prompts import SYSTEM_PROMPT
from smzdm_notice.llm.routing import ResolvedLLMConfig, RoutingSnapshot, build_chat_completion_kwargs, resolve
from smzdm_notice.smzdm.ranking import RankingItem


@dataclass
class FilterPromptContext:
    user_message: str
    item_map: dict[str, RankingItem]
    items_summary: list[dict]
    arbiter_items: dict[str, dict]


def filter_items(
    items: list[RankingItem],
    user_prompt: str,
    inventory_data: str,
    model: str | None = None,
    routing_snapshot: RoutingSnapshot | None = None,
) -> FilterItemsResult:
    """使用 LLM 筛选商品。"""
    if not items:
        logger.info("商品列表为空，跳过 LLM 筛选")
        return FilterItemsResult()

    from smzdm_notice.core import config

    items = _prefilter_items(items, config)
    if not items:
        logger.info("所有新商品均未通过 env 预筛选，无需 LLM 筛选")
        return FilterItemsResult()

    prompt_context = _build_prompt_context(items, user_prompt, inventory_data)
    llm_config = resolve("filter", routing_snapshot)
    if model:
        llm_config = replace(llm_config, model_id=model)
    logger.info(
        f"发送 {len(prompt_context.item_map)} 条商品到 LLM ({llm_config.connection}/{llm_config.model_id}) 进行筛选..."
    )

    client = get_client_for_config(llm_config)
    if not config.LLM_DUAL_FILTER:
        return _filter_with_single_call(client, llm_config, prompt_context)
    return _filter_with_dual_calls(client, llm_config, prompt_context, routing_snapshot)


def _build_prompt_context(
    items: list[RankingItem],
    user_prompt: str,
    inventory_data: str,
) -> FilterPromptContext:
    items_summary: list[dict] = []
    item_map: dict[str, RankingItem] = {}
    arbiter_items: dict[str, dict] = {}

    for item in items:
        item_id = item.article_id
        raw_summary = item.to_llm_summary()

        summary = dict(raw_summary)
        if "rank" in summary:
            del summary["rank"]

        items_summary.append(summary)
        item_map[item_id] = item
        arbiter_summary = dict(raw_summary)
        arbiter_summary["link"] = item.link
        arbiter_items[item_id] = arbiter_summary

    user_message = (
        f"## 当前系统时间\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## 用户购物偏好\n{user_prompt}\n\n"
        f"## 耗材库存记录\n{inventory_data}\n\n"
        f"## 当前好价排行榜商品列表\n"
        f"```json\n{json.dumps(items_summary, ensure_ascii=False, indent=2)}\n```\n\n"
        f"请根据以上信息，严格执行筛选逻辑并输出推荐结果。"
    )
    return FilterPromptContext(user_message, item_map, items_summary, arbiter_items)


def _filter_with_single_call(
    client: OpenAI,
    llm_config: ResolvedLLMConfig,
    prompt_context: FilterPromptContext,
) -> FilterItemsResult:
    call_outcome = _single_llm_call(client, llm_config, prompt_context.user_message)
    if call_outcome.result is None:
        return FilterItemsResult(
            diagnostics=FilterDiagnostics(
                llm_failed=True,
                error_summary=call_outcome.error_summary,
            )
        )
    matched, near_misses = _match_result(call_outcome.result.result, prompt_context.item_map)
    logger.info(f"LLM 筛选完成: {len(matched)}/{len(prompt_context.item_map)} 条推荐, {len(near_misses)} 条 near-miss")
    return FilterItemsResult(matched=matched, near_misses=near_misses)


def _filter_with_dual_calls(
    client: OpenAI,
    llm_config: ResolvedLLMConfig,
    prompt_context: FilterPromptContext,
    routing_snapshot: RoutingSnapshot | None = None,
) -> FilterItemsResult:
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_single_llm_call, client, llm_config, prompt_context.user_message)
        future_b = executor.submit(_single_llm_call, client, llm_config, prompt_context.user_message)

        outcome_a = future_a.result()
        outcome_b = future_b.result()
        call_a = outcome_a.result
        call_b = outcome_b.result

    _log_dual_call_result("A", call_a)
    _log_dual_call_result("B", call_b)
    final_result, arbiter_info = resolve_dual_result(
        call_a,
        call_b,
        prompt_context.items_summary,
        prompt_context.arbiter_items,
        prompt_context.user_message,
        routing_snapshot=routing_snapshot,
    )

    matched, near_misses = _match_result(final_result, prompt_context.item_map)
    logger.info(f"双重判断完成: {len(matched)}/{len(prompt_context.item_map)} 条推荐, {len(near_misses)} 条 near-miss")
    llm_failed = call_a is None and call_b is None
    return FilterItemsResult(
        matched=matched,
        near_misses=near_misses,
        arbiter_info=arbiter_info,
        diagnostics=FilterDiagnostics(
            llm_failed=llm_failed,
            error_summary=_join_error_summaries(outcome_a.error_summary, outcome_b.error_summary)
            if llm_failed
            else None,
        ),
    )


def _log_dual_call_result(label: str, call: LLMCallResult | None) -> None:
    if call is not None:
        logger.info(f"调用 {label}: {len(call.result.recommendations)} 条推荐")
    else:
        logger.warning(f"调用 {label} 失败")


def _prefilter_items(items: list[RankingItem], config) -> list[RankingItem]:
    """根据 env 量化阈值做 LLM 前粗筛。"""
    if not config.PREFILTER_ENABLED:
        return items

    filtered = [item for item in items if _passes_prefilter(item, config)]
    bypass_text = "未启用"
    if config.PREFILTER_BYPASS_ENABLED:
        bypass_text = f"启用(评论>={config.PREFILTER_BYPASS_MIN_COMMENTS}, 值票>={config.PREFILTER_BYPASS_MIN_WORTHY})"
    logger.info(
        "env 预筛选: "
        f"{len(filtered)}/{len(items)} 条进入 LLM；"
        f"普通阈值=值票>={config.PREFILTER_MIN_WORTHY}, "
        f"值率>={config.PREFILTER_MIN_WORTHY_RATE:.2f}, "
        f"评论>={config.PREFILTER_MIN_COMMENTS}, "
        f"收藏>={config.PREFILTER_MIN_FAVORITES}；"
        f"强信号直通={bypass_text}"
    )
    return filtered


def _passes_prefilter(item: RankingItem, config) -> bool:
    if config.PREFILTER_BYPASS_ENABLED and _passes_bypass(item, config):
        return True
    return (
        item.worthy >= config.PREFILTER_MIN_WORTHY
        and _worthy_rate(item) >= config.PREFILTER_MIN_WORTHY_RATE
        and item.comments >= config.PREFILTER_MIN_COMMENTS
        and item.favorites >= config.PREFILTER_MIN_FAVORITES
    )


def _passes_bypass(item: RankingItem, config) -> bool:
    return item.comments >= config.PREFILTER_BYPASS_MIN_COMMENTS or item.worthy >= config.PREFILTER_BYPASS_MIN_WORTHY


def _worthy_rate(item: RankingItem) -> float:
    total = item.worthy + item.unworthy
    if total <= 0:
        return 0.0
    return item.worthy / total


def _single_llm_call(
    client: OpenAI,
    llm_config: ResolvedLLMConfig,
    user_message: str,
) -> LLMCallOutcome:
    """执行一次 LLM 调用并解析结果。"""
    try:
        response = client.chat.completions.create(
            **build_chat_completion_kwargs(
                llm_config,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
        )

        content = response.choices[0].message.content or ""
        logger.debug(f"LLM 原始响应: {content}")

        return LLMCallOutcome(result=LLMCallResult(result=_parse_response(content), raw_content=content))

    except RETRYABLE_OPENAI_ERRORS as e:
        summary = error_summary("可重试/网络类问题", e)
        logger.error(f"LLM 调用失败（{summary}）")
        return LLMCallOutcome(error_summary=summary)
    except NON_RETRYABLE_OPENAI_ERRORS as e:
        summary = error_summary("配置或请求不可重试问题", e)
        logger.error(f"LLM 调用失败（{summary}）")
        return LLMCallOutcome(error_summary=summary)
    except GENERAL_OPENAI_ERRORS as e:
        summary = error_summary("OpenAI SDK/API 错误", e)
        logger.error(f"LLM 调用失败（{summary}）")
        return LLMCallOutcome(error_summary=summary)
    except Exception as e:
        summary = error_summary("非 OpenAI SDK 异常", e)
        logger.error(f"LLM 调用失败（{summary}）")
        return LLMCallOutcome(error_summary=summary)


def _join_error_summaries(*summaries: str) -> str | None:
    parts = [summary for summary in summaries if summary]
    if not parts:
        return None
    return "；".join(parts)


def _match_result(
    result: FilterResult,
    item_map: dict[str, RankingItem],
) -> tuple[list[tuple[RankingItem, str]], list[tuple[RankingItem, str]]]:
    """将 FilterResult 中的 ID 匹配回原始商品。"""
    matched: list[tuple[RankingItem, str]] = []
    for rec in result.recommendations:
        if rec.id in item_map:
            matched.append((item_map[rec.id], rec.reason))
        else:
            logger.warning(f"LLM 返回了无效推荐 ID: {rec.id}")

    near_misses: list[tuple[RankingItem, str]] = []
    for nm in result.near_misses:
        if nm.id in item_map:
            near_misses.append((item_map[nm.id], nm.reason))
        else:
            logger.warning(f"LLM 返回了无效 near_miss ID: {nm.id}")

    return matched, near_misses


def _parse_response(content: str) -> FilterResult:
    """解析 LLM 响应，兼容各种格式。"""
    data = extract_json_object(content)
    if data is not None:
        try:
            return FilterResult(**data)
        except (TypeError, ValueError):
            pass

    logger.warning(f"无法解析 LLM 响应: {(content or '')[:200]}")
    return FilterResult()
