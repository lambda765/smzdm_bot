"""LLM 商品筛选与仲裁数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from smzdm_notice.smzdm.ranking import RankingItem


class Recommendation(BaseModel):
    """单个推荐商品。"""

    id: str
    reason: str


class NearMiss(BaseModel):
    """接近推荐但最终未推送的商品。"""

    id: str
    reason: str


class FilterResult(BaseModel):
    """LLM 筛选结果。"""

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

    @property
    def succeeded(self) -> bool:
        return self.result is not None


@dataclass
class FilterDiagnostics:
    """LLM 筛选诊断信息。"""

    llm_failed: bool = False
    error_summary: str | None = None


@dataclass
class FilterItemsResult:
    """商品筛选结果。"""

    matched: list[tuple[RankingItem, str]] = field(default_factory=list)
    near_misses: list[tuple[RankingItem, str]] = field(default_factory=list)
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
    config_change_draft: Optional[dict] = None  # noqa: UP045 - Pydantic needs this on Python 3.9.
