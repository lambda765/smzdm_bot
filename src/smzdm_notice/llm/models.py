"""商品筛选与仲裁使用的 LLM 数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from smzdm_notice.smzdm.ranking import RankingItem


class DecisionContext(BaseModel):
    """推荐决策时的轻量上下文摘要。"""

    need_state: str = ""
    inventory_basis: str = ""
    preference_basis: list[str] = Field(default_factory=list)
    threshold_adjustment: str = ""
    context_summary: str = ""

    @field_validator("need_state", "inventory_basis", "threshold_adjustment", "context_summary", mode="before")
    def _normalize_text_field(cls, value: object) -> str:
        if isinstance(value, str):
            return value
        if value is not None:
            logger.warning(f"LLM 推荐 decision_context 文本字段类型异常，已降级为空: {type(value).__name__}")
        return ""

    @field_validator("preference_basis", mode="before")
    def _normalize_preference_basis(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        return []


class Recommendation(BaseModel):
    """单个推荐商品。"""

    id: str
    reason: str
    category: str = ""
    notification_name: str = ""
    decision_context: DecisionContext = Field(default_factory=DecisionContext)

    @field_validator("category", mode="before")
    def _normalize_category(cls, value: object) -> str:
        if isinstance(value, str):
            return value
        if value is not None:
            logger.warning(f"LLM 推荐 category 类型异常，已降级为未分类: {type(value).__name__}")
        return ""

    @field_validator("notification_name", mode="before")
    def _normalize_notification_name(cls, value: object) -> str:
        if isinstance(value, str):
            return value
        if value is not None:
            logger.warning(f"LLM 推荐 notification_name 类型异常，已从通知摘要略过: {type(value).__name__}")
        return ""

    @field_validator("decision_context", mode="before")
    def _normalize_decision_context(cls, value: object) -> object:
        if value is None:
            logger.debug("LLM 推荐 decision_context=null，已降级为空上下文")
            return {}
        if isinstance(value, (dict, DecisionContext)):
            return value
        logger.warning(f"LLM 推荐 decision_context 类型异常，已降级为空上下文: {type(value).__name__}")
        return {}


class NearMiss(BaseModel):
    """接近推荐但最终未推送的商品。"""

    id: str
    reason: str


class FilterResult(BaseModel):
    """表示 LLM 筛选结果。"""

    recommendations: list[Recommendation] = []
    near_misses: list[NearMiss] = []


class LLMCallResult(BaseModel):
    """单次 LLM 调用结果，保留原始响应供仲裁使用。"""

    result: FilterResult
    raw_content: str = ""


@dataclass
class LLMCallOutcome:
    """单次 LLM 调用 outcome，失败时保留错误摘要。"""

    result: LLMCallResult | None = None
    error_summary: str = ""


@dataclass
class FilterDiagnostics:
    """表示 LLM 筛选诊断信息。"""

    llm_failed: bool = False
    error_summary: str | None = None


@dataclass
class FilterItemsResult:
    """商品筛选结果。"""

    matched: list[tuple[RankingItem, str]] = field(default_factory=list)
    near_misses: list[tuple[RankingItem, str]] = field(default_factory=list)
    categories_by_article_id: dict[str, str] = field(default_factory=dict)
    contexts_by_article_id: dict[str, dict] = field(default_factory=dict)
    notification_names_by_article_id: dict[str, str] = field(default_factory=dict)
    arbiter_info: ArbiterInfo | None = None
    diagnostics: FilterDiagnostics = field(default_factory=FilterDiagnostics)


class ArbiterInfo(BaseModel):
    """仲裁结果信息，供飞书推送使用。"""

    chosen: str
    reason: str
    analysis: str
    suggestion: str
    result_a: FilterResult
    result_b: FilterResult
    items: dict[str, dict] = Field(default_factory=dict)
    change_assessment: dict
    preference_change: Optional[dict] = None  # noqa: UP045
