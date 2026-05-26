"""Near-miss 商品存储模块。

存储 LLM 认为是好价但因用户偏好而跳过的商品，
供夜间汇总推送。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

from smzdm_notice.smzdm.ranking import RankingItem

_STORE_META_KEY = "__meta__"


class NearMissManager:
    """Near-miss 商品管理器。"""

    def __init__(self, filepath: str, expire_hours: int = 24) -> None:
        self._filepath = Path(filepath)
        self._expire_seconds = expire_hours * 3600
        self._store: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """从文件加载。"""
        if self._filepath.exists():
            try:
                with open(self._filepath, encoding="utf-8") as f:
                    self._store = json.load(f)
                # 分离元数据
                self._meta = self._store.pop(_STORE_META_KEY, {})
                logger.debug(f"加载 near-miss 缓存: {len(self._store)} 条记录")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"near-miss 缓存加载失败，重新创建: {e}")
                self._store = {}
                self._meta = {}
        else:
            self._meta = {}
        self._cleanup()

    def _save(self) -> None:
        """保存到文件。"""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {**self._store, _STORE_META_KEY: self._meta}
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _cleanup(self) -> None:
        """清理过期记录。"""
        now = time.time()
        expired = [aid for aid, entry in self._store.items() if now - entry.get("timestamp", 0) > self._expire_seconds]
        for aid in expired:
            del self._store[aid]
        if expired:
            logger.debug(f"清理 {len(expired)} 条过期 near-miss 记录")
            self._save()

    def add(self, item: RankingItem, skip_reason: str) -> None:
        """添加一条 near-miss（同一 article_id 保留最新记录）。"""
        self._store[item.article_id] = {
            "article_id": item.article_id,
            "title": item.title,
            "price": item.price,
            "mall": item.mall,
            "brand": item.brand,
            "worthy": item.worthy,
            "unworthy": item.unworthy,
            "comments": item.comments,
            "favorites": item.favorites,
            "tags": item.tags,
            "link": item.link,
            "pic": item.pic,
            "tab_name": item.tab_name,
            "rank": item.rank,
            "skip_reason": skip_reason,
            "timestamp": time.time(),
        }
        self._save()

    def add_batch(self, items: list[tuple[RankingItem, str]]) -> None:
        """批量添加 near-miss 条目。"""
        now = time.time()
        for item, reason in items:
            self._store[item.article_id] = {
                "article_id": item.article_id,
                "title": item.title,
                "price": item.price,
                "mall": item.mall,
                "brand": item.brand,
                "worthy": item.worthy,
                "unworthy": item.unworthy,
                "comments": item.comments,
                "favorites": item.favorites,
                "tags": item.tags,
                "link": item.link,
                "pic": item.pic,
                "tab_name": item.tab_name,
                "rank": item.rank,
                "skip_reason": reason,
                "timestamp": now,
            }
        self._save()

    def remove(self, article_id: str) -> None:
        """移除单条记录。"""
        if article_id in self._store:
            del self._store[article_id]
            self._save()

    def get_all_sorted(self) -> list[dict]:
        """获取所有未过期的 near-miss 条目，按时间排序。"""
        self._cleanup()
        entries = list(self._store.values())
        entries.sort(key=lambda e: e.get("timestamp", 0))
        return entries

    def clear(self) -> None:
        """清空所有条目。"""
        self._store.clear()
        self._save()

    def get_last_digest_date(self) -> str:
        """获取上次发送汇总的日期。"""
        return self._meta.get("last_digest_date", "")

    def set_last_digest_date(self, date_str: str) -> None:
        """设置上次发送汇总的日期。"""
        self._meta["last_digest_date"] = date_str
        self._save()

    @property
    def size(self) -> int:
        return len(self._store)
