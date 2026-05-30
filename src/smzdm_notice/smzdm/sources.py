"""Unified SMZDM item source aggregation."""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger

from smzdm_notice.core.sleep import interruptible_sleep
from smzdm_notice.smzdm.keywords import SearchKeywordRule
from smzdm_notice.smzdm.ranking import RankingConfig, RankingItem, get_ranking
from smzdm_notice.smzdm.search import get_search

ItemFetcher = Callable[[], list[RankingItem]]


def fetch_sources(
    sources: list[tuple[str, ItemFetcher]],
    *,
    interval_seconds: int = 5,
    should_stop: Callable[[], bool] | None = None,
) -> list[RankingItem]:
    """Fetch item lists from named sources with interruptible spacing."""
    all_items: list[RankingItem] = []
    for i, (name, fetcher) in enumerate(sources):
        if should_stop and should_stop():
            logger.info("收到停止信号，中断商品来源抓取")
            break
        try:
            all_items.extend(fetcher())
        except Exception as e:
            logger.error(f"获取商品来源 [{name}] 失败: {e}")

        if i < len(sources) - 1:
            _sleep_between_sources(interval_seconds, should_stop)
            if should_stop and should_stop():
                break
    return all_items


def fetch_all_sources(
    ranking_configs: list[RankingConfig],
    search_keywords: list[str] | None = None,
    search_rules: list[SearchKeywordRule] | None = None,
    *,
    top_n: int = 20,
    interval_seconds: int = 5,
    should_stop: Callable[[], bool] | None = None,
) -> list[RankingItem]:
    """Fetch configured ranking and keyword-search sources."""
    if search_rules is None:
        search_rules = [SearchKeywordRule(keyword) for keyword in (search_keywords or [])]
    sources: list[tuple[str, ItemFetcher]] = [
        (cfg.name, lambda cfg=cfg: get_ranking(config=cfg, top_n=top_n)) for cfg in ranking_configs
    ]
    sources.extend(
        (
            f"搜索-{rule.keyword}",
            lambda rule=rule: get_search(
                keyword=rule.keyword,
                top_n=top_n,
                max_price=rule.max_price,
            ),
        )
        for rule in search_rules
    )

    all_items = fetch_sources(sources, interval_seconds=interval_seconds, should_stop=should_stop)
    logger.info(
        f"共获取 {len(all_items)} 条商品（来自 {len(ranking_configs)} 个榜单、{len(search_rules)} 个搜索关键词）"
    )
    return all_items


def _sleep_between_sources(
    interval_seconds: int,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    interruptible_sleep(interval_seconds, should_stop)
