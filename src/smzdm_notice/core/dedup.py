"""去重管理模块。

基于本地 JSON 文件存储已推送商品 URL，24 小时后自动过期。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger


class DedupManager:
    """商品去重管理器。"""

    def __init__(self, filepath: str, expire_hours: int = 24) -> None:
        self._filepath = Path(filepath)
        self._expire_seconds = expire_hours * 3600
        self._cache: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        """从文件加载缓存。"""
        if self._filepath.exists():
            try:
                with open(self._filepath, encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug(f"加载去重缓存: {len(self._cache)} 条记录")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"去重缓存加载失败，重新创建: {e}")
                self._cache = {}
        self._cleanup()

    def _save(self) -> None:
        """保存缓存到文件。"""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def _cleanup(self) -> None:
        """清理过期记录。"""
        now = time.time()
        expired = [k for k, ts in self._cache.items() if now - ts > self._expire_seconds]
        for k in expired:
            del self._cache[k]
        if expired:
            logger.debug(f"清理 {len(expired)} 条过期去重记录")
            self._save()

    def is_new(self, url: str) -> bool:
        """判断该 URL 是否为新商品（未在缓存中或已过期）。"""
        self._cleanup()
        return url not in self._cache

    def mark_sent(self, url: str) -> None:
        """标记该 URL 已推送。"""
        self._cache[url] = time.time()
        self._save()

    def mark_batch(self, urls: list[str]) -> None:
        """批量标记已推送。"""
        now = time.time()
        for url in urls:
            self._cache[url] = now
        self._save()

    @property
    def size(self) -> int:
        """当前缓存大小。"""
        return len(self._cache)
