"""校准生成与记忆分析模块。

CalibrationGenerator: 从 deal memory 生成校准文本，供 prompt 注入。
MemoryAnalyzer: 用 LLM 分析记忆记录，发现长期偏好模式并生成 preference.md 修改建议。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

from loguru import logger

from smzdm_notice.core.memory import DealMemoryStore
from smzdm_notice.llm.categories import (
    UNCATEGORIZED_CATEGORY,
    candidate_calibration_categories,
    candidate_search_keywords,
)
from smzdm_notice.llm.clients import get_client_for_config
from smzdm_notice.llm.routing import build_chat_completion_kwargs, resolve
from smzdm_notice.smzdm.ranking import RankingItem

_CALIBRATION_SECTION_HEADER = "## 历史决策校准参考（历史个例，仅供校准，不代表稳定规则）"


class CalibrationGenerator:
    """从 DealMemoryStore 动态生成校准文本。

    纯计算，不调 LLM，不落盘。渲染当前候选商品相关的历史个例，供注入 filter prompt。
    """

    def __init__(self, memory_store: DealMemoryStore, max_examples: int = 5, min_category_records: int = 2) -> None:
        self._store = memory_store
        self._max_examples = max_examples
        self._min_category_records = max(1, min_category_records)

    def build_section(self, items: list[RankingItem]) -> str:
        """为当前候选商品生成相关历史校准个例。"""
        if not items:
            return ""
        records = self._store.get_records()
        if not records:
            return ""
        calibration_by_category = self._build_related_calibration_by_category(items, records)
        return self._join_category_texts(calibration_by_category)

    def _build_related_calibration_by_category(self, items: list[RankingItem], records: list[dict]) -> dict[str, dict]:
        grouped: dict[str, list[dict]] = {}
        for record in records:
            category = str(record.get("category_hint") or "").strip()
            if not category or category == UNCATEGORIZED_CATEGORY:
                continue
            grouped.setdefault(category, []).append(record)

        source_categories = candidate_calibration_categories(items)
        search_keywords = candidate_search_keywords(items)
        calibration_by_category: dict[str, dict] = {}
        for category in sorted(grouped):
            category_records = grouped[category]
            if len(category_records) < self._min_category_records:
                continue
            if not self._is_related_category(category, category_records, source_categories, search_keywords):
                continue
            selected_records = category_records[: self._max_examples]
            good_records = [r for r in selected_records if r.get("feedback", {}).get("action") == "deal_good"]
            not_worth_records = [r for r in selected_records if r.get("feedback", {}).get("action") == "deal_not_worth"]
            text = self._build_category_text(category, good_records, not_worth_records)
            if not text:
                continue
            calibration_by_category[category] = {
                "text": text,
                "good_count": len(good_records),
                "not_worth_count": len(not_worth_records),
                "record_count": len(category_records),
                "source_tab_names": sorted(
                    {str(r.get("tab_name") or "").strip() for r in category_records if str(r.get("tab_name") or "").strip()}
                ),
                "search_keywords": sorted(
                    {
                        str(r.get("search_keyword") or "").strip()
                        for r in category_records
                        if str(r.get("search_keyword") or "").strip()
                    }
                ),
            }
        return calibration_by_category

    @staticmethod
    def _is_related_category(
        category: str,
        records: list[dict],
        source_categories: set[str],
        search_keywords: set[str],
    ) -> bool:
        if category in source_categories:
            return True
        if not search_keywords:
            return False
        historical_keywords = {
            str(record.get("search_keyword") or "").strip()
            for record in records
            if str(record.get("search_keyword") or "").strip()
        }
        return bool(historical_keywords.intersection(search_keywords))

    def _build_category_text(self, category: str, good_records: list[dict], not_worth_records: list[dict]) -> str:
        """渲染单个品类的校准文本。"""
        sections: list[str] = [f"### {category}"]

        if good_records:
            lines = ["好价案例："]
            for i, r in enumerate(good_records[: self._max_examples], 1):
                lines.append(_render_record_case(i, r))
            sections.append("\n".join(lines))

        if not_worth_records:
            lines = ["不值案例："]
            for i, r in enumerate(not_worth_records[: self._max_examples], 1):
                lines.append(_render_record_case(i, r))
            sections.append("\n".join(lines))

        if len(sections) == 1:
            return ""

        return "\n".join(sections)

    @staticmethod
    def _join_category_texts(calibration_by_category: dict[str, dict]) -> str:
        sections = [data.get("text", "") for _, data in sorted(calibration_by_category.items()) if data.get("text")]
        if not sections:
            return ""
        return _CALIBRATION_SECTION_HEADER + "\n\n" + "\n\n".join(sections) + "\n\n"


def _render_record_case(index: int, record: dict) -> str:
    """渲染单条案例，包含商品信号、推荐理由和反馈时间。"""
    category = record.get("category_hint", "未知品类")
    title = record.get("title", "未知商品")
    price = record.get("price", "?")
    worthy = record.get("worthy", 0)
    unworthy = record.get("unworthy", 0)
    comments = record.get("comments", 0)
    tags = record.get("tags", [])
    tags_str = ", ".join(tags) if tags else ""

    signal_line = f"{index}. [{category}] {title} | {price}元 | 值{worthy}/不值{unworthy} | 评论{comments}"
    if tags_str:
        signal_line += f" | [{tags_str}]"

    recommendation = record.get("recommendation", {})
    filter_reason = str(recommendation.get("filter_reason") or record.get("context", {}).get("filter_reason") or "").strip()
    if filter_reason:
        signal_line += f" | 推荐理由：{_truncate_context(filter_reason, 60)}"
    decision_context = _render_decision_context(record)
    if decision_context:
        signal_line += f" | 上下文：{decision_context}"
    feedback_reason = str((record.get("feedback") or {}).get("reason") or "").strip()
    if feedback_reason:
        signal_line += f" | 用户反馈理由：{_truncate_context(feedback_reason, 60)}"
    acted_at = str((record.get("feedback") or {}).get("acted_at") or "").strip()
    if acted_at:
        signal_line += f" | 反馈时间：{acted_at}"
    return signal_line


def _render_decision_context(record: dict) -> str:
    context = record.get("context") or {}
    decision_context = context.get("decision_context") or {}
    if not isinstance(decision_context, dict):
        return ""

    need_label = {
        "urgent": "急缺补货",
        "normal": "普通需求",
        "unknown": "",
    }.get(str(decision_context.get("need_state") or ""), "")
    threshold_label = {
        "relaxed_due_to_need": "标准放宽",
        "strict_normal": "标准严格",
        "none": "标准未调整",
        "unknown": "",
    }.get(str(decision_context.get("threshold_adjustment") or ""), "")

    parts: list[str] = []
    if need_label:
        parts.append(need_label)
    inventory_basis = str(decision_context.get("inventory_basis") or "").strip()
    if inventory_basis:
        parts.append(inventory_basis)
    preference_basis = decision_context.get("preference_basis")
    if isinstance(preference_basis, list):
        basis_text = "、".join(str(item).strip() for item in preference_basis if str(item).strip())
        if basis_text:
            parts.append(f"偏好：{basis_text}")
    if threshold_label:
        parts.append(threshold_label)
    context_summary = str(decision_context.get("context_summary") or "").strip()
    if context_summary:
        parts.append(context_summary)
    return "，".join(parts)


def _truncate_context(text: str, max_length: int) -> str:
    """截断上下文文本为一行摘要。"""
    # 取第一行非空内容
    for line in text.splitlines():
        line = line.strip()
        if line:
            if len(line) <= max_length:
                return line
            return line[:max_length] + "…"
    return text[:max_length] + "…" if len(text) > max_length else text


# ── LLM 记忆分析 ──


@dataclass
class MemoryAnalysis:
    """表示 LLM 分析结果。"""

    summary: str = ""
    suggested_rules: list[dict] = field(default_factory=list)
    patterns: list[dict] = field(default_factory=list)


class MemoryAnalyzer:
    """用 LLM 分析 deal memory 记录，发现长期偏好模式。

    在夜间汇总时调用，不频繁。使用 draft client 复用现有 LLM 配置。
    """

    def __init__(self) -> None:
        pass

    def analyze(self, records: list[dict], preference_text: str = "") -> MemoryAnalysis | None:
        """调 LLM 分析 records，返回模式和建议。返回 None 表示分析失败。"""
        try:
            llm_config = resolve("draft")
            if llm_config.temperature is None:
                llm_config = replace(llm_config, temperature=0.1)
            client = get_client_for_config(llm_config)
            messages = self._build_messages(records, preference_text)
            response = client.chat.completions.create(**build_chat_completion_kwargs(llm_config, messages=messages))
            content = response.choices[0].message.content or ""
            return self._parse_response(content)

        except Exception as e:
            from smzdm_notice.llm.errors import (
                GENERAL_OPENAI_ERRORS,
                NON_RETRYABLE_OPENAI_ERRORS,
                RETRYABLE_OPENAI_ERRORS,
            )
            if isinstance(e, (RETRYABLE_OPENAI_ERRORS, NON_RETRYABLE_OPENAI_ERRORS, GENERAL_OPENAI_ERRORS)):
                logger.error(f"Deal Memory LLM 分析失败: {e}")
                return None
            raise

    def _build_messages(self, records: list[dict], preference_text: str = "") -> list[dict]:
        """构建 LLM 分析 prompt。"""
        from smzdm_notice.llm.memory_prompts import MEMORY_ANALYSIS_SYSTEM_PROMPT

        records_text = json.dumps(records, ensure_ascii=False, indent=2)
        user_message = (
            f"以下是当前 preference.md 完整内容。已有规则不得重复新增；需要强化时应明确指出待合并的原规则。\n\n"
            f"```markdown\n{preference_text}\n```\n\n"
            f"以下是用户对推荐商品的好价/不值反馈历史记录，每条记录包含商品信号、"
            f"推荐理由、品类提示、决策上下文和用户评价。\n\n"
            f"```json\n{records_text}\n```\n\n"
            f"请分析这些记录，发现用户的长期偏好模式。"
        )
        return [
            {"role": "system", "content": MEMORY_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def _parse_response(self, content: str) -> MemoryAnalysis | None:
        """解析 LLM 输出的分析结果。"""
        try:
            from smzdm_notice.llm.json_utils import extract_json_object

            data = extract_json_object(content)
            if data is None:
                logger.warning(f"Deal Memory LLM 分析响应无法解析: {(content or '')[:200]}")
                return None

            return MemoryAnalysis(
                summary=str(data.get("summary", "")),
                suggested_rules=_filter_suggested_rules(data.get("suggested_rules", []))[:1],
                patterns=data.get("patterns", []),
            )
        except Exception as e:
            logger.warning(f"Deal Memory LLM 分析响应解析失败: {e}")
            return None


def compute_suggestion_hash(rule_text: str) -> str:
    """计算建议规则的哈希，用于去重。"""
    from smzdm_notice.llm.json_utils import content_hash
    return content_hash(rule_text)


def _filter_suggested_rules(rules: object) -> list[dict]:
    if not isinstance(rules, list):
        return []
    filtered: list[dict] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_text = str(rule.get("rule") or "").strip()
        if not rule_text:
            continue
        if _contains_hard_threshold_rule(rule_text):
            logger.info(f"Deal Memory: 跳过硬阈值建议规则: {rule_text}")
            continue
        if not _has_stable_rule_evidence(rule):
            logger.info(f"Deal Memory: 跳过样本不足或比例不稳定的建议规则: {rule_text}")
            continue
        filtered.append(rule)
    return filtered


_HARD_THRESHOLD_PATTERN = re.compile(
    r"(值票|评论|值率|不值|收藏).{0,12}(>=|≥|<=|≤|大于等于|小于等于|不少于|至少|必须|需|需要).{0,12}\d+"
)


def _contains_hard_threshold_rule(rule_text: str) -> bool:
    return bool(_HARD_THRESHOLD_PATTERN.search(rule_text))


def _has_stable_rule_evidence(rule: dict) -> bool:
    counts = _extract_rule_counts(rule)
    if counts is None:
        return False
    good, not_worth = counts
    total = good + not_worth
    if total < 5:
        return False
    if good > 0 and not_worth > 0:
        ratio = good / not_worth
        if 0.5 <= ratio <= 2:
            return False
    return True


def _extract_rule_counts(rule: dict) -> tuple[int, int] | None:
    good = _coerce_count(rule.get("good_count"))
    not_worth = _coerce_count(rule.get("not_worth_count"))
    if good is not None and not_worth is not None:
        return good, not_worth

    text = f"{rule.get('reason', '')}\n{rule.get('evidence', '')}"
    good = _extract_named_count(text, ("good", "deal_good", "好价", "正样本"))
    not_worth = _extract_named_count(text, ("not_worth", "deal_not_worth", "不值", "反样本"))
    if good is None or not_worth is None:
        return None
    return good, not_worth


def _coerce_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        count = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        count = int(value.strip())
    else:
        return None
    return count if count >= 0 else None


def _extract_named_count(text: str, labels: tuple[str, ...]) -> int | None:
    for label in labels:
        match = re.search(rf"{re.escape(label)}\D{{0,8}}(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(rf"(\d+)\D{{0,4}}{re.escape(label)}", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None
