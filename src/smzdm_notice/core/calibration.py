"""校准生成与记忆分析模块。

CalibrationGenerator: 从 deal memory 生成校准文本，供 prompt 注入。
MemoryAnalyzer: 用 LLM 分析记忆记录，发现长期偏好模式并生成 preference.md 修改建议。
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from loguru import logger

from smzdm_notice.core.memory import DealMemoryStore
from smzdm_notice.llm.categories import UNCATEGORIZED_CATEGORY
from smzdm_notice.llm.clients import get_client_for_config
from smzdm_notice.llm.routing import build_chat_completion_kwargs, resolve

_MIN_RECORDS_FOR_CALIBRATION = 5


@dataclass
class CalibrationProfile:
    """校准配置文件，包含渲染好的校准文本和分析元数据。"""

    calibration_text: str = ""
    calibration_by_category: dict[str, dict] = field(default_factory=dict)
    record_count: int = 0
    generated_at: str = ""


class CalibrationGenerator:
    """从 DealMemoryStore 生成校准文本。

    纯计算，不调 LLM。渲染带上下文的具体案例文本，供注入 filter prompt。
    """

    def __init__(self, memory_store: DealMemoryStore, max_examples: int = 5) -> None:
        self._store = memory_store
        self._max_examples = max_examples

    def generate(self) -> CalibrationProfile:
        """生成校准文本。records 不足时返回空文本。"""
        records = self._store.get_records()
        if len(records) < _MIN_RECORDS_FOR_CALIBRATION:
            logger.debug(f"Deal Memory: records {len(records)} < {_MIN_RECORDS_FOR_CALIBRATION}，跳过校准生成")
            return CalibrationProfile(record_count=len(records))

        calibration_by_category = self._build_calibration_by_category(records)
        calibration_text = self._join_category_texts(calibration_by_category)
        return CalibrationProfile(
            calibration_text=calibration_text,
            calibration_by_category=calibration_by_category,
            record_count=len(records),
            generated_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def _build_calibration_by_category(self, records: list[dict]) -> dict[str, dict]:
        grouped: dict[str, list[dict]] = {}
        for record in records:
            category = str(record.get("category_hint") or "").strip()
            if not category or category == UNCATEGORIZED_CATEGORY:
                continue
            grouped.setdefault(category, []).append(record)

        calibration_by_category: dict[str, dict] = {}
        for category in sorted(grouped):
            category_records = grouped[category]
            good_records = [r for r in category_records if r.get("feedback", {}).get("action") == "deal_good"]
            not_worth_records = [r for r in category_records if r.get("feedback", {}).get("action") == "deal_not_worth"]
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
        return "## 历史决策校准参考（参考信息，不覆盖上述规则）\n\n" + "\n\n".join(sections)


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
    acted_at = str((record.get("feedback") or {}).get("acted_at") or "").strip()
    if acted_at:
        signal_line += f" | 反馈时间：{acted_at}"
    return signal_line


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
    """LLM 分析结果。"""

    summary: str = ""
    suggested_rules: list[dict] = field(default_factory=list)
    patterns: list[dict] = field(default_factory=list)


class MemoryAnalyzer:
    """用 LLM 分析 deal memory 记录，发现长期偏好模式。

    在夜间汇总时调用，不频繁。使用 draft client 复用现有 LLM 配置。
    """

    def __init__(self) -> None:
        pass

    def analyze(self, records: list[dict]) -> MemoryAnalysis | None:
        """调 LLM 分析 records，返回模式和建议。返回 None 表示分析失败。"""
        try:
            llm_config = resolve("draft")
            if llm_config.temperature is None:
                llm_config = replace(llm_config, temperature=0.1)
            client = get_client_for_config(llm_config)
            messages = self._build_messages(records)
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

    def _build_messages(self, records: list[dict]) -> list[dict]:
        """构建 LLM 分析 prompt。"""
        from smzdm_notice.llm.memory_prompts import MEMORY_ANALYSIS_SYSTEM_PROMPT

        records_text = json.dumps(records, ensure_ascii=False, indent=2)
        user_message = (
            f"以下是用户对推荐商品的好价/不值反馈历史记录，每条记录包含商品信号、"
            f"推荐理由、品类提示和用户评价。\n\n"
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
                suggested_rules=_filter_suggested_rules(data.get("suggested_rules", [])),
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


_calibration_lock = threading.Lock()


def save_calibration_profile(profile: CalibrationProfile, filepath: str) -> None:
    """保存校准配置文件到磁盘。"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "__meta__": {
            "generated_at": profile.generated_at,
            "record_count": profile.record_count,
        },
        "calibration_text": profile.calibration_text,
        "calibration_by_category": profile.calibration_by_category,
    }
    with _calibration_lock, open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_calibration_text(filepath: str) -> str:
    """从磁盘加载校准文本。文件不存在或记录不足时返回空字符串。"""
    path = Path(filepath)
    if not path.exists():
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("__meta__", {})
        if meta.get("record_count", 0) < _MIN_RECORDS_FOR_CALIBRATION:
            return ""
        return data.get("calibration_text", "")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"校准配置加载失败: {e}")
        return ""


def load_calibration_profile_data(filepath: str) -> dict:
    """加载新结构校准配置。文件不存在、记录不足或旧格式时返回空分组。"""
    path = Path(filepath)
    if not path.exists():
        return {"calibration_by_category": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("__meta__", {})
        if meta.get("record_count", 0) < _MIN_RECORDS_FOR_CALIBRATION:
            return {"calibration_by_category": {}}
        calibration_by_category = data.get("calibration_by_category")
        if not isinstance(calibration_by_category, dict):
            return {"calibration_by_category": {}}
        return {"calibration_by_category": calibration_by_category}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"校准配置加载失败: {e}")
        return {"calibration_by_category": {}}
