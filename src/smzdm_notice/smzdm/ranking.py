"""什么值得买好价排行榜抓取模块。

复用 smzdm-ranking/scripts/get_ranking.py 的核心逻辑，
增加多榜单抓取支持和抓取间隔控制。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from loguru import logger

from smzdm_notice.smzdm.client import (
    AD_CELL_TYPES,
    extract_article_tags,
    extract_nested_title,
    get_json,
    parse_num,
)

HAOJIA_API = "https://haojia-api.smzdm.com"


# ========== 数据模型 ==========


@dataclass
class RankingConfig:
    """单个榜单的 API 参数配置。"""

    name: str
    tab_id: str = ""
    category_ids: str = ""
    tag_ids: str = ""
    sub_tab: int = 0
    order: int | None = 4
    slot: int | None = 4


# ========== 榜单注册表 ==========

RANKINGS: dict[str, RankingConfig] = {
    # 综合榜
    "综合榜-全部": RankingConfig(name="综合榜-全部", tab_id="67"),
    "综合榜-电脑数码": RankingConfig(name="综合榜-电脑数码", tab_id="5", category_ids="163"),
    "综合榜-白菜": RankingConfig(name="综合榜-白菜", tab_id="9", tag_ids="322337"),
    "综合榜-食品生鲜": RankingConfig(name="综合榜-食品生鲜", tab_id="6", category_ids="95"),
    "综合榜-运动户外": RankingConfig(name="综合榜-运动户外", tab_id="4", category_ids="191"),
    "综合榜-家用电器": RankingConfig(name="综合榜-家用电器", tab_id="7", category_ids="27"),
    "综合榜-服饰鞋包": RankingConfig(name="综合榜-服饰鞋包", tab_id="72", category_ids="57"),
    "综合榜-日用百货": RankingConfig(name="综合榜-日用百货", tab_id="73", category_ids="1515"),
    "综合榜-母婴用品": RankingConfig(name="综合榜-母婴用品", tab_id="76", category_ids="75"),
    "综合榜-家居家装": RankingConfig(name="综合榜-家居家装", tab_id="77", category_ids="37"),
    "综合榜-办公设备": RankingConfig(name="综合榜-办公设备", tab_id="78", category_ids="177"),
    "综合榜-个护化妆": RankingConfig(name="综合榜-个护化妆", tab_id="79", category_ids="113"),
    "综合榜-本地生活": RankingConfig(name="综合榜-本地生活", tab_id="80", category_ids="5847", order=None, slot=None),
    "综合榜-医疗健康": RankingConfig(name="综合榜-医疗健康", tab_id="81", category_ids="5686", order=None, slot=None),
    "综合榜-图书文娱": RankingConfig(name="综合榜-图书文娱", tab_id="83", category_ids="7,5375", order=24, slot=24),
    "综合榜-玩模乐器": RankingConfig(name="综合榜-玩模乐器", tab_id="84", category_ids="93", order=None, slot=None),
    # 其他榜单
    "热卖榜": RankingConfig(name="热卖榜", sub_tab=1, order=3, slot=3),
    "热评榜": RankingConfig(name="热评榜", sub_tab=2, order=3, slot=3),
    "热搜榜": RankingConfig(name="热搜榜", sub_tab=3, order=3, slot=3),
}


# ========== 商品数据模型 ==========


@dataclass
class RankingItem:
    """榜单商品条目。"""

    rank: int
    title: str
    article_id: str
    price: str
    worthy: int
    unworthy: int
    comments: int
    favorites: int
    mall: str
    brand: str
    tab_id: str = ""
    tab_name: str = ""
    tags: list[str] = field(default_factory=list)
    link: str = ""
    pic: str = ""
    source_type: str = "ranking"
    search_keyword: str = ""
    search_max_price: float | None = None
    numeric_price: float | None = None

    def to_dict(self) -> dict:
        """转换为字典。"""
        return asdict(self)

    def to_llm_summary(self) -> dict:
        """提取给 LLM 用的精简信息，节省 token。"""
        return {
            "id": self.article_id,
            "rank": self.rank,
            "title": self.title,
            "price": self.price,
            "brand": self.brand,
            "mall": self.mall,
            "worthy": self.worthy,
            "unworthy": self.unworthy,
            "comments": self.comments,
            "favorites": self.favorites,
            "tags": self.tags,
            "tab_name": self.tab_name,
        }

    def to_text(self) -> str:
        """单行文字摘要。"""
        tags_str = f"  [{', '.join(self.tags)}]" if self.tags else ""
        return (
            f"#{self.rank} [{self.brand}] {self.title}  {self.price}  "
            f"值{self.worthy}/不值{self.unworthy}  评论{self.comments}  "
            f"收藏{self.favorites}  来自{self.mall}{tags_str}"
        )


# ========== 核心功能 ==========


def get_ranking(config: RankingConfig, top_n: int = 20) -> list[RankingItem]:
    """获取什么值得买好价榜单前 N 个商品。

    Args:
        config: 榜单配置。
        top_n: 返回前 N 条，默认 20。

    Returns:
        包含核心信息的 RankingItem 列表。
    """
    tab_name = config.name
    logger.info(f"正在获取 [{tab_name}] 榜单 Top {top_n} ...")

    response = get_json(HAOJIA_API, "/ranking_list/articles", _ranking_params(config))
    rows = response.get("data", {}).get("rows", [])
    items = _parse_ranking_rows(rows, config, top_n)
    logger.info(f"[{tab_name}] 获取到 {len(items)} 条商品")
    return items


def _ranking_params(config: RankingConfig) -> dict:
    params: dict = {
        "basic_v": 0,
        "f": "iphone",
        "mall_ids": "",
        "offset": 0,
        "page": 1,
        "sub_tab": config.sub_tab,
        "tab": 1,
        "tab_id": config.tab_id,
        "category_ids": config.category_ids,
        "tag_ids": config.tag_ids,
        "v": "11.1.70",
        "weixin": "1",
        "zhuanzai_ab": "b",
    }
    if config.order is not None:
        params["order"] = config.order
    if config.slot is not None:
        params["slot"] = config.slot
    return params


def _parse_ranking_rows(rows: list[dict], config: RankingConfig, top_n: int) -> list[RankingItem]:
    items: list[RankingItem] = []
    rank = 0
    for row in rows:
        if not _is_ranking_product_row(row):
            continue

        rank += 1
        items.append(_ranking_item_from_row(row, config, rank))
        if rank >= top_n:
            break
    return items


def _is_ranking_product_row(row: dict) -> bool:
    return row.get("cell_type") not in AD_CELL_TYPES and bool(row.get("article_price"))


def _ranking_item_from_row(row: dict, config: RankingConfig, rank: int) -> RankingItem:
    interaction = row.get("article_interaction", {})
    favorites = parse_num(interaction.get("article_collection", 0) or row.get("article_favorite", 0))
    return RankingItem(
        rank=rank,
        title=row.get("article_title", ""),
        article_id=str(row.get("article_id", "")),
        price=row.get("article_price", ""),
        worthy=parse_num(row.get("article_worthy", 0)),
        unworthy=parse_num(row.get("article_unworthy", 0)),
        comments=parse_num(row.get("article_comment", 0)),
        favorites=favorites,
        mall=extract_nested_title(row.get("article_mall")),
        brand=extract_nested_title(row.get("article_brand")),
        tags=extract_article_tags(row.get("article_tag")),
        link=row.get("article_url", ""),
        pic=row.get("article_pic", ""),
        tab_id=config.tab_id,
        tab_name=config.name,
    )


def fetch_all_rankings(
    configs: list[RankingConfig],
    top_n: int = 20,
    interval_seconds: int = 5,
    should_stop: Callable[[], bool] | None = None,
) -> list[RankingItem]:
    """抓取多个榜单，每个榜单之间间隔指定秒数。

    Args:
        configs: 要抓取的榜单配置列表。
        top_n: 每个榜单返回前 N 条。
        interval_seconds: 两次抓取之间的间隔秒数，默认 5 秒。
        should_stop: 可选回调，返回 True 时立即停止抓取。

    Returns:
        所有榜单的商品合并列表。
    """
    all_items: list[RankingItem] = []

    for i, cfg in enumerate(configs):
        if should_stop and should_stop():
            logger.info("收到停止信号，中断榜单抓取")
            break
        try:
            items = get_ranking(config=cfg, top_n=top_n)
            all_items.extend(items)
        except Exception as e:
            logger.error(f"获取榜单 [{cfg.name}] 失败: {e}")

        # 在两个榜单之间等待，最后一个不等
        if i < len(configs) - 1:
            # 分段 sleep，以便及时响应停止信号
            elapsed = 0
            while elapsed < interval_seconds:
                if should_stop and should_stop():
                    logger.info("收到停止信号，中断等待")
                    break
                time.sleep(min(1, interval_seconds - elapsed))
                elapsed += 1

    logger.info(f"共获取 {len(all_items)} 条商品（来自 {len(configs)} 个榜单）")
    return all_items


if __name__ == "__main__":
    """独立运行测试。"""
    import json

    items = get_ranking(config=RANKINGS["综合榜-全部"], top_n=5)
    print(json.dumps([i.to_dict() for i in items], ensure_ascii=False, indent=2))
