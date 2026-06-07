"""Deterministic category handling for Deal Memory."""

from __future__ import annotations

import re

from smzdm_notice.smzdm.ranking import RankingItem

UNCATEGORIZED_CATEGORY = "未分类"
PRESET_CATEGORIES = {
    "电脑数码",
    "食品生鲜",
    "运动户外",
    "家用电器",
    "服饰鞋包",
    "日用百货",
    "母婴用品",
    "家居家装",
    "办公设备",
    "个护化妆",
    "本地生活",
    "医疗健康",
    "图书文娱",
    "玩模乐器",
}

PRESET_CATEGORY_LIST = [
    "电脑数码",
    "食品生鲜",
    "运动户外",
    "家用电器",
    "服饰鞋包",
    "日用百货",
    "母婴用品",
    "家居家装",
    "办公设备",
    "个护化妆",
    "本地生活",
    "医疗健康",
    "图书文娱",
    "玩模乐器",
]

_INVALID_CATEGORY_TERMS = {
    "热卖榜",
    "热评榜",
    "热搜榜",
    "综合榜",
    "搜索",
    "历史低价",
    "好价",
    "神价",
    "券",
    "满减",
    "补贴",
    "到手",
    "自营",
}
_PRICE_OR_SPEC_PATTERN = re.compile(
    r"(¥|￥|\d+(?:\.\d+)?\s*(?:元|块|折|L|ML|ml|G|GB|g|kg|KG|片|包|瓶|支|只|个|件|枚|粒|颗|寸|英寸))",
    re.IGNORECASE,
)
_CUSTOM_CATEGORY_PATTERN = re.compile(r"^[A-Za-z\u4e00-\u9fff]{2,8}$")


def sanitize_category(raw: str, item: RankingItem | None = None) -> str:
    """Normalize and validate a category label returned by the filter LLM.

    This is intentionally deterministic and never calls an LLM. Invalid labels
    are treated as unknown instead of being inferred from source tab names.
    """
    category = _normalize_category(raw)
    if not category:
        return UNCATEGORIZED_CATEGORY
    if category in PRESET_CATEGORIES:
        return category
    if _is_valid_custom_category(category, item):
        return category
    return UNCATEGORIZED_CATEGORY


def category_from_tab_name(tab_name: str) -> str:
    """Return a preset category encoded in a source tab name, if any."""
    text = str(tab_name or "").strip()
    prefix = "综合榜-"
    if not text.startswith(prefix):
        return ""
    candidate = text[len(prefix) :].strip()
    if candidate in PRESET_CATEGORIES:
        return candidate
    return ""


def candidate_calibration_categories(items: list[RankingItem]) -> set[str]:
    """Infer only safe calibration categories from candidate source metadata."""
    categories: set[str] = set()
    for item in items:
        category = category_from_tab_name(getattr(item, "tab_name", ""))
        if category:
            categories.add(category)
    return categories


def candidate_search_keywords(items: list[RankingItem]) -> set[str]:
    keywords: set[str] = set()
    for item in items:
        if getattr(item, "source_type", "") != "search":
            continue
        keyword = str(getattr(item, "search_keyword", "") or "").strip()
        if keyword:
            keywords.add(keyword)
    return keywords


def _normalize_category(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"\s+", "", text)
    return text.strip("，。；;、,.!！?？:：[]【】()（）<>《》\"'`")


def _is_valid_custom_category(category: str, item: RankingItem | None) -> bool:
    if not _CUSTOM_CATEGORY_PATTERN.fullmatch(category):
        return False
    if _PRICE_OR_SPEC_PATTERN.search(category):
        return False
    if any(term in category for term in _INVALID_CATEGORY_TERMS):
        return False
    return item is None or not _contains_item_source_text(category, item)


def _contains_item_source_text(category: str, item: RankingItem) -> bool:
    for value in (getattr(item, "brand", ""), getattr(item, "mall", "")):
        text = str(value or "").strip()
        if text and text in category:
            return True
    return False
