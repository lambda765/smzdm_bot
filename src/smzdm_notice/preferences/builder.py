"""配置草案 LLM 生成与修订。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from smzdm_notice.core import config
from smzdm_notice.llm.clients import get_draft_client
from smzdm_notice.llm.errors import (
    GENERAL_OPENAI_ERRORS,
    NON_RETRYABLE_OPENAI_ERRORS,
    RETRYABLE_OPENAI_ERRORS,
    error_summary,
)
from smzdm_notice.llm.json_utils import parse_json_object
from smzdm_notice.preferences.models import ALLOWED_TARGETS, ConfigDraft
from smzdm_notice.preferences.prompts import (
    draft_rules_prompt,
    file_context_block,
    read_target_content,
    revision_system_prompt,
)
from smzdm_notice.preferences.store import DraftStore


def build_message_draft(message: str, store: DraftStore | None = None) -> ConfigDraft | None:
    """将用户自然语言转换为待确认配置草案。"""
    return _build_llm_draft_from_message(
        message=message,
        source=f"用户对话：{message.strip()}",
        store=store,
    )


def _build_llm_draft_from_message(
    message: str,
    source: str,
    store: DraftStore | None = None,
    metadata: dict | None = None,
) -> ConfigDraft | None:
    """复用同一条 LLM 草案生成管线构造 ConfigDraft。"""
    # 用户直接对话、商品快捷操作都走这里：入口只负责表达意图，
    # append/replace/delete 的落点由 LLM 根据当前配置文件上下文决定。
    root = store.root if store else None
    data = _draft_with_llm(message, root=root)
    if not isinstance(data, dict) or not _is_valid_draft_data(data):
        return None
    draft = _build_draft(data, source=source)
    draft.revision_history = [{"role": "user", "content": message.strip()}]
    if metadata:
        draft.metadata.update(metadata)
    if store:
        return store.create(draft)
    return draft


def build_arbitration_draft(
    data: dict | None,
    store: DraftStore | None = None,
    suggestion: str = "",
) -> ConfigDraft | None:
    """将仲裁 LLM 输出的结构化建议转换为偏好文件草案。"""
    # 仲裁只能产出偏好规则调整；库存变更必须来自用户对话或商品快捷操作。
    if not isinstance(data, dict):
        return None
    if data.get("target_file") != "preference.md":
        return None
    if not _is_valid_draft_data(data):
        return None
    draft = _build_draft(data, source="仲裁建议一键采纳")
    if suggestion:
        draft.metadata["suggestion_hash"] = hashlib.sha256(suggestion.strip().encode("utf-8")).hexdigest()[:16]
    if store:
        return store.create(draft)
    return draft


def build_deal_action_draft(action: str, value: dict, store: DraftStore | None = None) -> ConfigDraft | None:
    """根据商品卡片快捷按钮生成配置草案。"""
    # 快捷按钮是用户少打字的入口，不在代码里维护固定 append 模板，
    # 否则会绕过“优先融入现有章节”的草案生成规则。
    message = _deal_action_message(action, value)
    return _build_llm_draft_from_message(
        message=message,
        source="商品卡片快捷操作",
        store=store,
        metadata=value,
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


def _call_llm_for_draft(messages: list, validate: bool = True) -> dict | None:
    if not config.LLM_DRAFT_API_KEY:
        return None
    try:
        logger.info(f"配置草案 LLM 模型: {config.LLM_DRAFT_MODEL}")
        client = get_draft_client()
        response = client.chat.completions.create(
            model=config.LLM_DRAFT_MODEL,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
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


def _draft_with_llm(message: str, root: Path | None = None) -> dict | None:
    system_prompt = draft_rules_prompt() + file_context_block(root)
    user_content = f"用户消息：{message}"
    return _call_llm_for_draft(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    )


def build_revision_draft(message: str, original: ConfigDraft, store: DraftStore | None = None) -> ConfigDraft | None:
    """根据用户对预览草案的修改意见，重新生成草案。支持多轮修改。"""
    # revision_history 描述的是“待确认草案”的演化，不代表文件已经被写入；
    # prompt 和校验都要基于这个前提，避免把未执行草案误当成真实文件去 delete。
    history = list(original.revision_history)
    history.append({"role": "draft", "content": _draft_summary(original)})
    history.append({"role": "user", "content": message.strip()})

    root = store.root if store else config.PROJECT_ROOT
    data = _revision_with_llm(history, root=root)
    ok, error = _validate_revision_data(data, original, root)
    if not ok:
        # 给模型一次带明确校验错误的自修正机会，常见问题是把用户对草案的删改
        # 误输出为针对真实文件的 delete/replace。
        data = _revision_with_llm(history, root=root, retry_context={"error": error, "data": data})
        ok, error = _validate_revision_data(data, original, root)
    if not ok:
        logger.warning(f"对话修改 LLM 修订草案校验失败: {error}")
        return None
    if not isinstance(data, dict):
        logger.warning("对话修改 LLM 修订草案校验失败: 草案结构无效")
        return None
    draft = _build_draft(data, source=original.source)
    draft.revision_history = history
    if store:
        return store.create(draft)
    return draft


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
    return _call_llm_for_draft(messages, validate=False)


def _validate_revision_data(
    data: dict | None,
    original: ConfigDraft,
    root: Path | None = None,
) -> tuple[bool, str]:
    if not isinstance(data, dict) or not _is_valid_draft_data(data):
        return False, "草案结构无效"
    if data.get("target_file") != original.target_file:
        return False, "修订草案不能更换目标文件"
    mode = str(data.get("edit_mode") or "append")
    if mode in ("replace", "delete"):
        search_text = str(data.get("search_text") or "").strip()
        actual_content = read_target_content(str(data.get("target_file")), root)
        # replace/delete 必须命中当前真实文件；如果只命中原草案内容，
        # 说明用户是在调整预览方案，应让模型重新生成完整草案。
        if search_text not in actual_content:
            return False, "replace/delete 的 search_text 不存在于当前真实文件，可能误用了未执行草案文本"
    return True, ""


def _is_valid_draft_data(data: dict | None) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("target_file") not in ALLOWED_TARGETS:
        return False
    mode = str(data.get("edit_mode") or "append")
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
        # append_text 在 replace/delete 草案中仅作为预览摘要内容使用；
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
