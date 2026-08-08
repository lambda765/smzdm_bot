"""配置草案 LLM 生成与修订。"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from smzdm_notice.core import config
from smzdm_notice.llm.clients import get_client_for_config
from smzdm_notice.llm.errors import (
    GENERAL_OPENAI_ERRORS,
    NON_RETRYABLE_OPENAI_ERRORS,
    RETRYABLE_OPENAI_ERRORS,
    error_summary,
)
from smzdm_notice.llm.json_utils import parse_json_object
from smzdm_notice.llm.routing import build_chat_completion_kwargs, resolve
from smzdm_notice.preferences.models import ALLOWED_TARGETS, ConfigDraft, DraftBuildOutcome, DraftStreamEvent
from smzdm_notice.preferences.prompts import (
    draft_rules_prompt,
    file_context_block,
    read_target_content,
    revision_system_prompt,
)
from smzdm_notice.preferences.store import DraftStore
from smzdm_notice.preferences.validation import validate_draft_data


def build_message_draft(message: str, store: DraftStore | None = None) -> ConfigDraft | None:
    """将用户自然语言转换为待确认配置草案。"""
    return build_message_draft_outcome(message, store).draft


DraftStreamCallback = Callable[[DraftStreamEvent], None]
_OPENAI_REQUEST_ERRORS = RETRYABLE_OPENAI_ERRORS + NON_RETRYABLE_OPENAI_ERRORS + GENERAL_OPENAI_ERRORS
_STREAM_UNSUPPORTED_STATUS_CODES = {400, 404, 405, 422, 501}


def build_message_draft_outcome(
    message: str,
    store: DraftStore | None = None,
    *,
    stream_callback: DraftStreamCallback | None = None,
) -> DraftBuildOutcome:
    """生成对话草案并保留 noop/rejected/failed 等结果语义。"""
    return _build_llm_draft_from_message(
        message=message,
        source=f"用户对话：{message.strip()}",
        store=store,
        stream_callback=stream_callback,
    )


def _build_llm_draft_from_message(
    message: str,
    source: str,
    store: DraftStore | None = None,
    metadata: dict | None = None,
    expected_target: str | None = None,
    stream_callback: DraftStreamCallback | None = None,
) -> DraftBuildOutcome:
    """复用同一条 LLM 草案生成管线构造 ConfigDraft。"""
    # 用户直接对话、商品快捷操作都走这里：入口只负责表达意图，
    # 操作 append/replace/delete 的落点由 LLM 根据当前配置文件上下文决定。
    root = store.root if store else None
    data = _draft_with_llm(message, root=root, stream_callback=stream_callback)
    if not isinstance(data, dict):
        return DraftBuildOutcome("failed", message="模型未生成有效的配置修改")
    ok, error = _validate_generated_data(data, root=root, expected_target=expected_target)
    if not ok:
        data = _draft_with_llm(
            message,
            root=root,
            retry_context={"error": error, "data": data},
            stream_callback=stream_callback,
            attempt=2,
        )
        ok, error = _validate_generated_data(data, root=root, expected_target=expected_target)
    if not isinstance(data, dict):
        logger.warning("对话修改 LLM 草案生成失败: 模型未返回 JSON 对象")
        return DraftBuildOutcome("failed", message="模型未生成有效的配置修改")
    if not ok:
        logger.warning(f"对话修改 LLM 草案安全校验失败: {error}")
        return DraftBuildOutcome("rejected", message=error or "草案未通过安全校验")
    if str(data.get("edit_mode") or "") == "noop":
        return DraftBuildOutcome("noop", message=str(data.get("summary") or "当前配置已覆盖该需求"))

    draft = _build_draft(data, source=source)
    draft.revision_history = [{"role": "user", "content": message.strip()}]
    if metadata:
        draft.metadata.update(metadata)
    if store:
        draft = store.create(draft)
    return DraftBuildOutcome("draft", draft=draft)


def build_arbitration_candidate_draft(
    candidate: dict | None,
    store: DraftStore | None = None,
    suggestion: str = "",
) -> ConfigDraft | None:
    """将仲裁识别出的语义偏好缺口交给统一草案管线定位。"""
    return build_arbitration_candidate_draft_outcome(candidate, store, suggestion).draft


def build_arbitration_candidate_draft_outcome(
    candidate: dict | None,
    store: DraftStore | None = None,
    suggestion: str = "",
) -> DraftBuildOutcome:
    """生成仲裁偏好候选草案并保留 noop/rejected/failed 语义。"""
    if not isinstance(candidate, dict):
        return DraftBuildOutcome("rejected", message="仲裁偏好候选结构无效")
    rule = str(candidate.get("rule") or "").strip()
    reason = str(candidate.get("reason") or "").strip()
    if not rule:
        return DraftBuildOutcome("rejected", message="仲裁偏好候选规则为空")
    metadata = {"source_kind": "arbitration"}
    if suggestion:
        from smzdm_notice.llm.json_utils import content_hash as suggestion_content_hash

        metadata["suggestion_hash"] = suggestion_content_hash(suggestion)
    outcome = _build_llm_draft_from_message(
        message=(
            "仲裁确认存在一项真实的长期用户偏好缺口。请检查完整 preference.md，"
            "已有相近规则时使用 replace 合并，没有合适归属时才 append。\n\n"
            f"候选规则：{rule}\n依据：{reason}"
        ),
        source="仲裁建议一键采纳",
        store=store,
        metadata=metadata,
        expected_target="preference.md",
    )
    if outcome.status == "noop":
        return DraftBuildOutcome(
            "noop",
            message="当前 preference.md 已覆盖候选规则，无需修改。",
        )
    return outcome


def build_memory_rule_draft(
    rule: dict,
    analysis_summary: str,
    store: DraftStore | None = None,
) -> ConfigDraft | None:
    """将带证据的 Memory 规则候选交给统一草案管线。"""
    return build_memory_rule_draft_outcome(rule, analysis_summary, store).draft


def build_memory_rule_draft_outcome(
    rule: dict,
    analysis_summary: str,
    store: DraftStore | None = None,
) -> DraftBuildOutcome:
    """生成 Memory 规则草案并保留 noop/rejected/failed 语义。"""
    rule_text = str(rule.get("rule") or "").strip()
    if not rule_text:
        return DraftBuildOutcome("noop", message="候选规则为空")
    from smzdm_notice.llm.json_utils import content_hash as suggestion_content_hash

    raw_evidence_ids = rule.get("evidence_ids")
    evidence_ids: list = raw_evidence_ids if isinstance(raw_evidence_ids, list) else []
    metadata = {
        "source_kind": "memory",
        "suggestion_hash": suggestion_content_hash(rule_text),
        "evidence_ids": [str(item) for item in evidence_ids if str(item).strip()],
        "evidence_scope": str(rule.get("evidence_scope") or ""),
    }
    message = (
        "Deal Memory 发现以下长期偏好候选。请检查完整 preference.md：已有相近规则时使用 replace 合并，"
        "当前规则已覆盖时输出 noop，确无合适归属时才 append。\n\n"
        f"分析摘要：{analysis_summary}\n候选规则：{rule_text}\n"
        f"依据：{str(rule.get('reason') or rule.get('evidence') or '').strip()}"
    )
    return _build_llm_draft_from_message(
        message=message,
        source="Deal Memory 偏好学习建议",
        store=store,
        metadata=metadata,
        expected_target="preference.md",
    )


def build_deal_action_draft(action: str, value: dict, store: DraftStore | None = None) -> ConfigDraft | None:
    """根据商品卡片快捷按钮生成配置草案。"""
    return build_deal_action_draft_outcome(action, value, store).draft


def build_deal_action_draft_outcome(
    action: str,
    value: dict,
    store: DraftStore | None = None,
    *,
    stream_callback: DraftStreamCallback | None = None,
) -> DraftBuildOutcome:
    """生成商品快捷操作草案并保留 noop/rejected/failed 语义。"""
    # 快捷按钮是用户少打字的入口，不在代码里维护固定 append 模板，
    # 否则会绕过“优先融入现有章节”的草案生成规则。
    message = _deal_action_message(action, value)
    return _build_llm_draft_from_message(
        message=message,
        source="商品卡片快捷操作",
        store=store,
        metadata=value,
        stream_callback=stream_callback,
    )


def _deal_action_message(action: str, value: dict) -> str:
    context = _format_deal_context(value)
    if action == "deal_ignore_category":
        intent = "以后不要推荐与该商品同类或高度相似的商品。"
    elif action == "deal_stock_enough":
        today = datetime.now().strftime("%Y年%m月%d日")
        intent = f"该商品或同类耗材库存充足，{today} 起暂时不需要补货。"
    elif action == "deal_follow":
        intent = "关注与该商品相关或同类商品，有好价可以推荐。"
    else:
        raise ValueError(f"未知商品快捷操作: {action}")
    return f"商品卡片快捷操作：{intent}\n\n商品信息：\n{context}"


def _format_deal_context(value: dict) -> str:
    rows = []
    fields = [
        ("标题", value.get("item_title")),
        ("品牌", value.get("item_brand")),
        ("article_id", value.get("article_id")),
        ("链接", value.get("item_link")),
    ]
    for label, raw in fields:
        text = str(raw or "").strip()
        if text:
            rows.append(f"- {label}：{text}")
    if not rows:
        return "- 标题：该商品"
    return "\n".join(rows)


class _DraftStreamCollector:
    """收集 OpenAI-compatible 流，分离可展示思考与最终 JSON。"""

    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"

    def __init__(self, callback: DraftStreamCallback, attempt: int) -> None:
        self.callback: DraftStreamCallback | None = callback
        self.attempt = attempt
        self.content_parts: list[str] = []
        self.reasoning_details_snapshot = ""
        self.structured_reasoning_seen = False
        self.reasoning_content_seen = False
        self.embedded_mode: bool | None = None
        self.embedded_pending = ""

    def consume(self, chunk: Any) -> None:
        choices = _stream_field(chunk, "choices")
        if not choices:
            return
        delta = _stream_field(choices[0], "delta")
        if delta is None:
            return

        reasoning = _stream_field(delta, "reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            self.reasoning_content_seen = True
            self.structured_reasoning_seen = True
            self._emit("reasoning_delta", reasoning)
        elif not self.reasoning_content_seen:
            reasoning_details = _reasoning_details_text(_stream_field(delta, "reasoning_details"))
            if reasoning_details:
                self.structured_reasoning_seen = True
                if reasoning_details.startswith(self.reasoning_details_snapshot):
                    reasoning_details = reasoning_details[len(self.reasoning_details_snapshot) :]
                self.reasoning_details_snapshot = _reasoning_details_text(
                    _stream_field(delta, "reasoning_details")
                )
                self._emit("reasoning_delta", reasoning_details)

        content = _stream_field(delta, "content")
        if isinstance(content, str) and content:
            self._consume_content(content)

    def finish(self) -> str:
        if self.embedded_mode is None:
            self._append_content(self.embedded_pending)
        elif self.embedded_mode:
            self._emit_embedded_reasoning(self.embedded_pending)
        self.embedded_pending = ""
        return "".join(self.content_parts)

    def _consume_content(self, text: str) -> None:
        if self.embedded_mode is False:
            self._append_content(text)
            return

        self.embedded_pending += text
        if self.embedded_mode is None:
            stripped = self.embedded_pending.lstrip()
            if self._OPEN_TAG.startswith(stripped):
                return
            if not stripped.startswith(self._OPEN_TAG):
                pending = self.embedded_pending
                self.embedded_pending = ""
                self.embedded_mode = False
                self._append_content(pending)
                return
            self.embedded_pending = stripped[len(self._OPEN_TAG) :]
            self.embedded_mode = True

        close_index = self.embedded_pending.find(self._CLOSE_TAG)
        if close_index >= 0:
            self._emit_embedded_reasoning(self.embedded_pending[:close_index])
            trailing = self.embedded_pending[close_index + len(self._CLOSE_TAG) :]
            self.embedded_pending = ""
            self.embedded_mode = False
            self._append_content(trailing)
            return

        retained = len(self._CLOSE_TAG) - 1
        if len(self.embedded_pending) > retained:
            self._emit_embedded_reasoning(self.embedded_pending[:-retained])
            self.embedded_pending = self.embedded_pending[-retained:]

    def _emit_embedded_reasoning(self, text: str) -> None:
        if text and not self.structured_reasoning_seen:
            self._emit("reasoning_delta", text)

    def _append_content(self, text: str) -> None:
        if not text:
            return
        self.content_parts.append(text)
        self._emit("content_delta", text)

    def _emit(self, kind: Literal["reasoning_delta", "content_delta"], text: str) -> None:
        if not text or self.callback is None:
            return
        try:
            self.callback(DraftStreamEvent(kind=kind, text=text, attempt=self.attempt))
        except Exception as e:
            logger.warning(f"配置草案流式回调异常，后续不再推送增量: {e}")
            self.callback = None


def _stream_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _reasoning_details_text(details: Any) -> str:
    if not isinstance(details, list):
        return ""
    parts: list[str] = []
    for detail in details:
        text = _stream_field(detail, "text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _emit_attempt_start(callback: DraftStreamCallback, attempt: int) -> None:
    text = "模型请求已提交，正在生成配置修改预览" if attempt == 1 else "首次结果未通过校验，正在修正生成结果"
    try:
        callback(DraftStreamEvent(kind="attempt_start", text=text, attempt=attempt))
    except Exception as e:
        logger.warning(f"配置草案流式回调异常，模型请求继续执行: {e}")


def _is_stream_unsupported_error(error: Exception) -> bool:
    message = str(error).casefold()
    status_code = getattr(error, "status_code", None)
    return status_code in _STREAM_UNSUPPORTED_STATUS_CODES and ("stream" in message or "流式" in message)


def _call_llm_for_draft(
    messages: list,
    validate: bool = True,
    *,
    stream_callback: DraftStreamCallback | None = None,
    attempt: int = 1,
) -> dict | None:
    try:
        llm_config = resolve("draft")
        if not llm_config.api_key:
            return None
        logger.info(f"配置草案 LLM 模型: {llm_config.connection}/{llm_config.model_id}")
        client = get_client_for_config(llm_config)
        kwargs = build_chat_completion_kwargs(llm_config, messages=messages)
        if stream_callback is not None:
            _emit_attempt_start(stream_callback, attempt)
            try:
                response = client.chat.completions.create(**kwargs, stream=True)
            except _OPENAI_REQUEST_ERRORS as e:
                if not _is_stream_unsupported_error(e):
                    raise
                logger.warning(
                    "配置草案流式请求创建失败，降级为非流式请求（"
                    f"{error_summary('OpenAI SDK/API 错误', e)}）"
                )
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
            else:
                collector = _DraftStreamCollector(stream_callback, attempt)
                for chunk in response:
                    collector.consume(chunk)
                content = collector.finish()
        else:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
        data = _parse_llm_draft_content(content)
        if not validate or _is_valid_draft_data(data):
            return data
        logger.warning("对话修改 LLM 草案内容校验失败")
    except RETRYABLE_OPENAI_ERRORS as e:
        logger.warning(f"对话修改 LLM 草案生成失败（{error_summary('可重试/网络类问题', e)}）")
    except NON_RETRYABLE_OPENAI_ERRORS as e:
        logger.warning(f"对话修改 LLM 草案生成失败（{error_summary('配置或请求不可重试问题', e)}）")
    except GENERAL_OPENAI_ERRORS as e:
        logger.warning(f"对话修改 LLM 草案生成失败（{error_summary('OpenAI SDK/API 错误', e)}）")
    except Exception as e:
        logger.warning(f"对话修改 LLM 草案生成失败（{error_summary('非 OpenAI SDK 异常', e)}）")
    return None


def _draft_with_llm(
    message: str,
    root: Path | None = None,
    retry_context: dict | None = None,
    *,
    stream_callback: DraftStreamCallback | None = None,
    attempt: int = 1,
) -> dict | None:
    system_prompt = draft_rules_prompt() + file_context_block(root)
    user_content = f"用户消息：{message}"
    if retry_context:
        user_content += (
            "\n\n上一次草案未通过安全校验，请基于完整文件重新生成。"
            f"\n校验问题：{retry_context.get('error')}"
            "\n上一次输出：" + json.dumps(retry_context.get("data"), ensure_ascii=False)
        )
    return _call_llm_for_draft(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        stream_callback=stream_callback,
        attempt=attempt,
    )


def build_revision_draft(message: str, original: ConfigDraft, store: DraftStore | None = None) -> ConfigDraft | None:
    """根据用户对预览草案的修改意见，重新生成草案。支持多轮修改。"""
    return build_revision_draft_outcome(message, original, store).draft


def build_revision_draft_outcome(
    message: str,
    original: ConfigDraft,
    store: DraftStore | None = None,
    *,
    stream_callback: DraftStreamCallback | None = None,
) -> DraftBuildOutcome:
    """生成草案修订并保留 noop/rejected/failed 语义。"""
    # 字段 revision_history 描述的是“待确认草案”的演化，不代表文件已经被写入；
    # 构造 prompt 和校验都要基于这个前提，避免把未执行草案误当成真实文件去 delete。
    history = list(original.revision_history)
    history.append({"role": "draft", "content": _draft_summary(original)})
    history.append({"role": "user", "content": message.strip()})

    return _build_revision_from_history(history, original, store, stream_callback=stream_callback)


def build_rebase_draft_outcome(
    original_draft: ConfigDraft,
    store: DraftStore,
) -> DraftBuildOutcome:
    """基于最新文件重新定位发生真实冲突的草案，不自动应用结果。"""
    if DraftStore.is_legacy_arbitration_draft(original_draft):
        return DraftBuildOutcome("rejected", message="旧版仲裁草案不能刷新或执行")
    if original_draft.status != "pending":
        return DraftBuildOutcome("rejected", message=f"原草案状态不是 pending: {original_draft.status}")

    history = list(original_draft.revision_history)
    history.append({"role": "draft", "content": _draft_summary(original_draft)})
    history.append(
        {
            "role": "user",
            "content": (
                "应用预演发现当前文件与旧预览存在确定性冲突。请基于当前完整文件重新实现原始意图；"
                "若当前文件已经覆盖该意图，输出 noop。不得更换目标文件。\n"
                f"原始来源：{original_draft.source}\n"
                "原始 metadata："
                + json.dumps(original_draft.metadata, ensure_ascii=False)
            ),
        }
    )
    return _build_revision_from_history(history, original_draft, store, is_rebase=True)


def _build_revision_from_history(
    history: list,
    original: ConfigDraft,
    store: DraftStore | None,
    *,
    is_rebase: bool = False,
    stream_callback: DraftStreamCallback | None = None,
) -> DraftBuildOutcome:
    """执行普通修订和冲突刷新共用的生成、重试与安全校验。"""

    root = store.root if store else config.PROJECT_ROOT
    data = _revision_with_llm(history, root=root, stream_callback=stream_callback)
    ok, error = _validate_revision_data(data, original, root)
    if not ok:
        # 给模型一次带明确校验错误的自修正机会，常见问题是把用户对草案的删改
        # 误输出为针对真实文件的 delete/replace。
        data = _revision_with_llm(
            history,
            root=root,
            retry_context={"error": error, "data": data},
            stream_callback=stream_callback,
            attempt=2,
        )
        ok, error = _validate_revision_data(data, original, root)
    if not ok:
        if not isinstance(data, dict):
            logger.warning("对话修改 LLM 修订草案生成失败: 模型未返回 JSON 对象")
            return DraftBuildOutcome("failed", message="模型未生成有效的修订草案")
        logger.warning(f"对话修改 LLM 修订草案校验失败: {error}")
        return DraftBuildOutcome("rejected", message=error or "修订草案未通过安全校验")
    assert isinstance(data, dict)
    if str(data.get("edit_mode") or "") == "noop":
        message = str(data.get("summary") or "当前配置已覆盖修改意见")
        logger.info(f"草案修订无需修改: {message}")
        return DraftBuildOutcome("noop", message=message)
    draft = _build_draft(data, source=original.source)
    draft.revision_history = history
    draft.metadata = dict(original.metadata)
    draft.metadata["supersedes_draft_id"] = original.draft_id
    if is_rebase:
        draft.metadata["rebase_reason"] = "apply_conflict"
    if store:
        draft = store.create(draft)
    return DraftBuildOutcome("draft", draft=draft)


def _draft_summary(draft: ConfigDraft) -> dict:
    return {
        "target_file": draft.target_file,
        "edit_mode": draft.edit_mode,
        "title": draft.title,
        "summary": draft.summary,
        "append_text": draft.append_text,
        "search_text": draft.search_text,
        "replace_text": draft.replace_text,
    }


def _revision_with_llm(
    history: list,
    root: Path | None = None,
    retry_context: dict | None = None,
    *,
    stream_callback: DraftStreamCallback | None = None,
    attempt: int = 1,
) -> dict | None:
    messages = [{"role": "system", "content": revision_system_prompt(root)}]
    for entry in history:
        role = "assistant" if entry["role"] == "draft" else "user"
        content = json.dumps(entry["content"], ensure_ascii=False) if entry["role"] == "draft" else entry["content"]
        messages.append({"role": role, "content": content})
    if retry_context:
        messages.append(
            {
                "role": "user",
                "content": (
                    "上一次输出的草案未通过校验，请重新输出一个 JSON 对象。\n"
                    f"校验失败原因：{retry_context.get('error')}\n"
                    "上一次输出：" + json.dumps(retry_context.get("data"), ensure_ascii=False)
                ),
            }
        )
    return _call_llm_for_draft(
        messages,
        validate=False,
        stream_callback=stream_callback,
        attempt=attempt,
    )


def _validate_revision_data(
    data: dict | None,
    original: ConfigDraft,
    root: Path | None = None,
) -> tuple[bool, str]:
    if not isinstance(data, dict) or not _is_valid_draft_data(data):
        return False, "草案结构无效"
    if data.get("target_file") != original.target_file:
        return False, "修订草案不能更换目标文件"
    actual_content = read_target_content(str(data.get("target_file")), root)
    validation = validate_draft_data(data, actual_content)
    return validation.ok, validation.error


def _validate_generated_data(
    data: dict | None,
    *,
    root: Path | None,
    expected_target: str | None,
) -> tuple[bool, str]:
    if not isinstance(data, dict) or not _is_valid_draft_data(data):
        return False, "草案结构无效"
    target = str(data.get("target_file") or "").strip()
    if expected_target and target != expected_target:
        return False, f"该建议只能修改 {expected_target}"
    actual = read_target_content(target, root)
    validation = validate_draft_data(data, actual)
    return validation.ok, validation.error


def _is_valid_draft_data(data: dict | None) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("target_file") not in ALLOWED_TARGETS:
        return False
    mode = str(data.get("edit_mode") or "append")
    if mode == "noop":
        return True
    if mode == "append":
        return bool(str(data.get("append_text") or "").strip())
    if mode in ("replace", "delete"):
        return bool(str(data.get("search_text") or "").strip())
    return False


def _parse_llm_draft_content(content: str) -> dict:
    return parse_json_object(content)


def _build_draft(data: dict, source: str) -> ConfigDraft:
    target_file = str(data.get("target_file") or "").strip()
    edit_mode = str(data.get("edit_mode") or "append").strip()
    append_text = str(data.get("append_text") or "").strip()
    search_text = str(data.get("search_text") or "").strip()
    replace_text = str(data.get("replace_text") or "").strip()

    if target_file not in ALLOWED_TARGETS:
        raise ValueError(f"不允许修改 {target_file}")
    if edit_mode == "append" and not append_text:
        raise ValueError("append 模式下 append_text 不能为空")
    if edit_mode in ("replace", "delete") and not search_text:
        raise ValueError(f"{edit_mode} 模式下 search_text 不能为空")

    if edit_mode == "replace":
        # 字段 append_text 在 replace/delete 草案中仅作为预览摘要内容使用；
        # 真正写入仍由 edit_mode + search_text/replace_text 决定。
        append_text = append_text or replace_text
    elif edit_mode == "delete":
        append_text = append_text or f"(删除：{search_text[:50]})"

    now = time.time()
    raw_id = f"{target_file}:{edit_mode}:{search_text}:{append_text}:{now}".encode()
    return ConfigDraft(
        draft_id=hashlib.sha256(raw_id).hexdigest()[:12],
        target_file=target_file,
        title=str(data.get("title") or "配置修改").strip(),
        summary=str(data.get("summary") or "待确认配置修改").strip(),
        append_text=append_text,
        source=source,
        created_at=now,
        edit_mode=edit_mode,
        search_text=search_text,
        replace_text=replace_text,
    )
