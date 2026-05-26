"""什么值得买关键词搜索抓取模块。"""

from __future__ import annotations

import time
from typing import Callable

from loguru import logger

from smzdm_notice.core import config
from smzdm_notice.smzdm.client import (
    AD_CELL_TYPES,
    compact_sign_value,
    extract_article_tags,
    extract_nested_title,
    get_json,
    parse_num,
)
from smzdm_notice.smzdm.ranking import RankingItem

SEARCH_API = "https://s-api.smzdm.com"
SEARCH_STALE_SECONDS = 30 * 24 * 60 * 60


def _search_params(keyword: str, limit: int) -> dict:
    return {
        "basic_v": 0,
        "category_id": "",
        "category_name": "",
        "f": config.get_smzdm_client_platform(),
        "filter_json_data": "{}",
        "is_biserial": 2,
        "keyword": keyword,
        "limit": limit,
        "offset": 0,
        "order": "time",
        "page": 1,
        "search_from": "输入搜索",
        "search_scenarios": "home",
        "search_scene": 18,
        "search_session_id": "",
        "search_source": 1,
        "search_tab": "good_price",
        "sid": str(int(time.time() * 10000)),
        "subtype": "",
        "tab_source": "",
        "type": "good_price",
        "v": config.SMZDM_APP_VERSION,
        "weixin": "1",
        "zhifa_tag_id": "",
        "zhilv_rate": "",
        "zhuanzai_ab": "b",
    }


def get_search(keyword: str, top_n: int = 20, max_price: float | None = None) -> list[RankingItem]:
    """获取单个关键词的 SMZDM 好价搜索结果。"""
    clean_keyword = keyword.strip()
    if not clean_keyword:
        return []

    logger.info(f"正在搜索 [{clean_keyword}] Top {top_n} ...")
    response = get_json(
        SEARCH_API,
        "/sou/list_v10",
        _search_params(clean_keyword, top_n),
        value_normalizer=compact_sign_value,
    )
    rows = response.get("data", {}).get("rows", [])
    items = _parse_search_rows(rows, clean_keyword, top_n, max_price=max_price)
    logger.info(f"[搜索-{clean_keyword}] 获取到 {len(items)} 条商品")
    return items


def _parse_search_rows(
    rows: list[dict],
    keyword: str,
    top_n: int,
    now: float | None = None,
    max_price: float | None = None,
) -> list[RankingItem]:
    items: list[RankingItem] = []
    rank = 0
    current_time = time.time() if now is None else now
    for row in rows:
        if not _is_search_product_row(row, current_time):
            continue

        rank += 1
        items.append(_search_item_from_row(row, keyword, rank, max_price))
        if rank >= top_n:
            break
    return items


def _is_search_product_row(row: dict, now: float) -> bool:
    if row.get("cell_type") in AD_CELL_TYPES:
        return False
    if not row.get("article_price"):
        return False
    if row.get("article_is_timeout") == "is_timeout":
        return False
    return not _is_stale_search_row(row, now)


def _search_item_from_row(
    row: dict,
    keyword: str,
    rank: int,
    max_price: float | None,
) -> RankingItem:
    article_price = row.get("article_price", "")
    return RankingItem(
        rank=rank,
        title=row.get("article_title", ""),
        article_id=str(row.get("article_id", "")),
        price=article_price,
        worthy=parse_num(row.get("article_worthy", 0)),
        unworthy=parse_num(row.get("article_unworthy", 0)),
        comments=parse_num(row.get("article_comment", 0)),
        favorites=parse_num(row.get("article_collection", 0)),
        mall=extract_nested_title(row.get("article_mall")),
        brand=extract_nested_title(row.get("article_brand")),
        tags=extract_article_tags(row.get("article_tag"), row.get("article_tag_arr"), row.get("article_tag_list")),
        link=row.get("article_url", ""),
        pic=row.get("article_pic", ""),
        tab_id="search",
        tab_name=f"搜索-{keyword}",
        source_type="search",
        search_keyword=keyword,
        search_max_price=max_price,
        numeric_price=_parse_numeric_price(article_price),
    )


def _parse_numeric_price(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _is_stale_search_row(row: dict, now: float) -> bool:
    """搜索结果超过 30 天未发布/更新时过滤；字段缺失或异常时保留。"""
    published_at = row.get("publish_date_lt")
    if published_at in (None, ""):
        return False
    try:
        timestamp = float(published_at)
    except (TypeError, ValueError):
        return False
    return timestamp < now - SEARCH_STALE_SECONDS


def fetch_all_searches(
    keywords: list[str],
    top_n: int = 20,
    interval_seconds: int = 5,
    should_stop: Callable[[], bool] | None = None,
) -> list[RankingItem]:
    """抓取多个关键词搜索结果。"""
    from smzdm_notice.smzdm.sources import fetch_sources

    sources = [(f"搜索-{keyword}", lambda keyword=keyword: get_search(keyword, top_n)) for keyword in keywords]
    return fetch_sources(sources, interval_seconds=interval_seconds, should_stop=should_stop)
