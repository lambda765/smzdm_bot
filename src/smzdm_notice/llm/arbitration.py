"""双重判断仲裁与结果合并。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger
from openai import OpenAI

from smzdm_notice.llm.clients import get_client_for_config
from smzdm_notice.llm.errors import (
    GENERAL_OPENAI_ERRORS,
    NON_RETRYABLE_OPENAI_ERRORS,
    RETRYABLE_OPENAI_ERRORS,
    error_summary,
)
from smzdm_notice.llm.json_utils import extract_json_object
from smzdm_notice.llm.models import ArbiterInfo, FilterResult, LLMCallResult, NearMiss
from smzdm_notice.llm.prompts import ARBITER_SYSTEM_PROMPT
from smzdm_notice.llm.routing import ResolvedLLMConfig, RoutingSnapshot, build_chat_completion_kwargs, resolve

_CHANGE_ASSESSMENT_CAUSES = {
    "preference_gap",
    "filter_prompt_gap",
    "model_execution_error",
    "soft_judgment",
}


@dataclass
class ArbitrationRequest:
    result_a: FilterResult
    result_b: FilterResult
    raw_a: str
    raw_b: str
    items_summary: list[dict]
    items_by_id: dict[str, dict]
    user_message: str
    client: OpenAI
    llm_config: ResolvedLLMConfig


def resolve_dual_result(
    call_a: LLMCallResult | None,
    call_b: LLMCallResult | None,
    items_summary: list[dict],
    items_by_id: dict[str, dict],
    user_message: str,
    routing_snapshot: RoutingSnapshot | None = None,
) -> tuple[FilterResult, ArbiterInfo | None]:
    """根据双次调用结果决定最终推荐。"""
    if call_a is None:
        if call_b is None:
            logger.error("两次 LLM 调用均失败")
            return FilterResult(), None
        logger.warning("调用 A 失败，使用调用 B 结果")
        return call_b.result, None
    if call_b is None:
        logger.warning("调用 B 失败，使用调用 A 结果")
        return call_a.result, None

    result_a = call_a.result
    result_b = call_b.result

    if compare_results(result_a, result_b):
        logger.info("两次判断推荐商品集合一致，无需仲裁，采用 A 的结果与排序")
        return result_a, None

    ids_a = {rec.id for rec in result_a.recommendations}
    ids_b = {rec.id for rec in result_b.recommendations}
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a
    logger.warning(f"两次判断不一致: A 独有 {only_a}, B 独有 {only_b}")

    arbiter_info = _run_arbiter(
        result_a,
        result_b,
        call_a,
        call_b,
        items_summary,
        items_by_id,
        user_message,
        routing_snapshot,
    )
    final_from_arbitration = _result_from_arbitration(result_a, result_b, arbiter_info)
    if final_from_arbitration is not None:
        return final_from_arbitration, arbiter_info

    return _intersect_results(result_a, result_b, ids_a & ids_b), None


def _run_arbiter(
    result_a: FilterResult,
    result_b: FilterResult,
    call_a: LLMCallResult,
    call_b: LLMCallResult,
    items_summary: list[dict],
    items_by_id: dict[str, dict],
    user_message: str,
    routing_snapshot: RoutingSnapshot | None = None,
) -> ArbiterInfo | None:
    from smzdm_notice.core import config

    if not config.LLM_ARBITER_ENABLED:
        return None
    llm_config = resolve("arbiter", routing_snapshot)
    logger.info(f"仲裁 LLM 模型: {llm_config.connection}/{llm_config.model_id}")
    logger.debug(f"仲裁 LLM Base URL: {llm_config.base_url}")
    return arbitrate(
        ArbitrationRequest(
            result_a=result_a,
            result_b=result_b,
            raw_a=call_a.raw_content,
            raw_b=call_b.raw_content,
            items_summary=items_summary,
            items_by_id=items_by_id,
            user_message=user_message,
            client=get_client_for_config(llm_config),
            llm_config=llm_config,
        )
    )


def _result_from_arbitration(
    result_a: FilterResult,
    result_b: FilterResult,
    arbiter_info: ArbiterInfo | None,
) -> FilterResult | None:
    if arbiter_info is None:
        return None
    logger.info(f"仲裁选择: {arbiter_info.chosen}, 原因: {arbiter_info.reason}")
    logger.info(f"不一致分析: {arbiter_info.analysis}")
    logger.info(f"Prompt 优化建议: {arbiter_info.suggestion}")
    return result_b if arbiter_info.chosen == "B" else result_a


def _intersect_results(result_a: FilterResult, result_b: FilterResult, common_ids: set[str]) -> FilterResult:
    logger.warning("仲裁未执行或失败，取两次结果的交集")
    if not common_ids:
        logger.warning("交集为空，返回空推荐，避免误推")
        return FilterResult(recommendations=[], near_misses=merge_near_misses(result_a, result_b))

    common_recs = [rec for rec in result_a.recommendations if rec.id in common_ids]
    return FilterResult(
        recommendations=common_recs,
        near_misses=merge_near_misses(result_a, result_b),
    )


def arbitrate(request: ArbitrationRequest) -> ArbiterInfo | None:
    """调用仲裁 agent 决定哪个结果更准确。"""
    try:
        response = request.client.chat.completions.create(
            **build_chat_completion_kwargs(
                request.llm_config,
                messages=[
                    {"role": "system", "content": ARBITER_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_arbitration_message(request)},
                ],
            )
        )
        return _parse_arbiter_response(response.choices[0].message.content, request)
    except RETRYABLE_OPENAI_ERRORS as e:
        logger.error(f"仲裁调用失败（{error_summary('可重试/网络类问题', e)}）")
        return None
    except NON_RETRYABLE_OPENAI_ERRORS as e:
        logger.error(f"仲裁调用失败（{error_summary('配置或请求不可重试问题', e)}）")
        return None
    except GENERAL_OPENAI_ERRORS as e:
        logger.error(f"仲裁调用失败（{error_summary('OpenAI SDK/API 错误', e)}）")
        return None
    except Exception as e:
        logger.error(f"仲裁调用失败（{error_summary('非 OpenAI SDK 异常', e)}）")
        return None


def _build_arbitration_message(request: ArbitrationRequest) -> str:
    recs_a_text = json.dumps(
        [{"id": r.id, "reason": r.reason} for r in request.result_a.recommendations],
        ensure_ascii=False,
    )
    recs_b_text = json.dumps(
        [{"id": r.id, "reason": r.reason} for r in request.result_b.recommendations],
        ensure_ascii=False,
    )

    return (
        f"{request.user_message}\n\n"
        f"## 判断 A 的推荐结果\n{recs_a_text}\n\n"
        f"## 判断 A 的原始响应（包含可能的 <think> 推理）\n{request.raw_a}\n\n"
        f"## 判断 B 的推荐结果\n{recs_b_text}\n\n"
        f"## 判断 B 的原始响应（包含可能的 <think> 推理）\n{request.raw_b}\n\n"
        f"请根据评判标准，决定哪个判断更准确。"
    )


def _parse_arbiter_response(content: str | None, request: ArbitrationRequest) -> ArbiterInfo | None:
    logger.debug(f"仲裁响应: {content}")
    data = extract_json_object(content)
    if data is None:
        logger.warning(f"仲裁响应无法解析为 JSON: {(content or '')[:500]}")
        return None

    chosen = str(data.get("chosen", "")).upper()
    if chosen not in {"A", "B"}:
        logger.warning(f"仲裁响应 chosen 无效: {data.get('chosen')!r}")
        return None

    raw_assessment = data.get("change_assessment")
    if not isinstance(raw_assessment, dict):
        logger.warning("仲裁响应缺少有效的 change_assessment")
        return None
    cause = str(raw_assessment.get("cause") or "")
    should_change = raw_assessment.get("should_change_preference")
    if cause not in _CHANGE_ASSESSMENT_CAUSES:
        logger.warning(f"仲裁响应 change_assessment.cause 无效: {cause!r}")
        return None
    if not isinstance(should_change, bool):
        logger.warning("仲裁响应 change_assessment.should_change_preference 必须是布尔值")
        return None
    if should_change and cause != "preference_gap":
        logger.warning("仲裁响应只有 preference_gap 才能建议修改 preference.md")
        return None

    raw_preference_change = data.get("preference_change")
    preference_change = raw_preference_change if isinstance(raw_preference_change, dict) else None
    if should_change and (
        preference_change is None or not str(preference_change.get("rule") or "").strip()
    ):
        logger.warning("仲裁响应建议修改 preference.md 时必须提供有效的 preference_change.rule")
        return None
    if preference_change is not None and not should_change:
        logger.warning("仲裁响应不建议修改 preference.md 时 preference_change 必须为空")
        return None

    return ArbiterInfo(
        chosen=chosen,
        reason=data.get("reason", ""),
        analysis=data.get("inconsistency_analysis") or data.get("analysis", ""),
        suggestion=data.get("prompt_optimization_suggestion") or data.get("suggestion", ""),
        result_a=request.result_a,
        result_b=request.result_b,
        items=request.items_by_id,
        change_assessment=raw_assessment,
        preference_change=preference_change,
    )


def compare_results(a: FilterResult, b: FilterResult) -> bool:
    """比较两次结果的推荐商品 ID 集合是否一致（忽略排序）。"""
    ids_a = {rec.id for rec in a.recommendations}
    ids_b = {rec.id for rec in b.recommendations}
    return ids_a == ids_b


def merge_near_misses(a: FilterResult, b: FilterResult) -> list[NearMiss]:
    """按 ID 合并 near-miss，A 的理由优先。"""
    merged: dict[str, NearMiss] = {}
    for nm in a.near_misses:
        merged[nm.id] = nm
    for nm in b.near_misses:
        merged.setdefault(nm.id, nm)
    return list(merged.values())
