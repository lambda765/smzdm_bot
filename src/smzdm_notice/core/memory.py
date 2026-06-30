"""Deal Memory 存储模块。

记录每次推荐的评估数据和用户的显式反馈（好价/不值），
供校准生成和长期偏好学习使用。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from loguru import logger

from smzdm_notice.llm.categories import UNCATEGORIZED_CATEGORY
from smzdm_notice.smzdm.ranking import RankingItem

_STORE_META_KEY = "__meta__"
_RECORDS_KEY = "records"
_PENDING_KEY = "pending"


class DealMemoryStore:
    """商品推荐记忆存储。

    pending 区：推送成功后缓存推荐商品及其上下文。
    records 区：用户点击「好价/不值」后，从 pending 移入 records，绑定 feedback。
    """

    def __init__(self, filepath: str, expire_days: int = 90) -> None:
        self._filepath = Path(filepath)
        self._expire_seconds = expire_days * 86400
        self._meta: dict = {}
        self._records: dict[str, dict] = {}
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    # ── 文件 I/O ──

    def _load(self) -> None:
        """从文件加载。仅在 __init__ 中调用，此时对象未共享，不需要锁。"""
        if self._filepath.exists():
            try:
                with open(self._filepath, encoding="utf-8") as f:
                    data = json.load(f)
                self._meta = data.pop(_STORE_META_KEY, {})
                records_data = data.pop(_RECORDS_KEY, None)
                if isinstance(records_data, dict):
                    self._records = records_data
                else:
                    if records_data is not None:
                        logger.warning(f"Deal Memory: {_RECORDS_KEY} 类型异常({type(records_data).__name__})，重置为空")
                    self._records = {}
                pending_data = data.pop(_PENDING_KEY, None)
                if isinstance(pending_data, dict):
                    self._pending = pending_data
                else:
                    if pending_data is not None:
                        logger.warning(f"Deal Memory: {_PENDING_KEY} 类型异常({type(pending_data).__name__})，重置为空")
                    self._pending = {}
                logger.debug(f"加载 Deal Memory: {len(self._records)} 条 records, {len(self._pending)} 条 pending")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Deal Memory 加载失败，重新创建: {e}")
                self._meta = {}
                self._records = {}
                self._pending = {}
        self._compact_unlocked()

    def _save(self) -> None:
        """保存到文件。"""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            _STORE_META_KEY: self._meta,
            _RECORDS_KEY: self._records,
            _PENDING_KEY: self._pending,
        }
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 写入操作 ──

    def record_to_pending(
        self,
        items: list[tuple[RankingItem, str]],
        categories_by_article_id: dict[str, str] | None = None,
        contexts_by_article_id: dict[str, dict] | None = None,
        arbiter_info: object | None = None,
    ) -> None:
        """推送成功后调用，缓存推荐商品和轻量推荐上下文到 pending。"""
        categories_by_article_id = categories_by_article_id or {}
        contexts_by_article_id = contexts_by_article_id or {}
        now = time.time()
        with self._lock:
            for item, reason in items:
                self._pending[item.article_id] = _serialize_pending_entry(
                    item=item,
                    filter_reason=reason,
                    category_hint=categories_by_article_id.get(item.article_id, UNCATEGORIZED_CATEGORY),
                    decision_context=contexts_by_article_id.get(item.article_id, {}),
                    arbiter_involved=arbiter_info is not None,
                    timestamp=now,
                )
            self._save()
            logger.info(f"Deal Memory: 写入 {len(items)} 条 pending")

    def record_feedback(self, article_id: str, action: str, reason: str | None = None) -> str:
        """用户点好价/不值时调用，支持首次记录、覆盖和取消。

        reason=None 表示普通反馈点击；reason 为字符串时表示补充或更新不值理由。

        Returns: recorded / updated / cancelled / reason_updated / not_found / invalid_action.
        """
        if action not in {"deal_good", "deal_not_worth"}:
            logger.warning(f"未知的 feedback action: {action}")
            return "invalid_action"

        with self._lock:
            entry = self._pending.pop(article_id, None)
            if entry is not None:
                entry["feedback"] = _build_feedback_payload(action, reason)
                self._records[article_id] = entry
                self._save()
                logger.info(f"Deal Memory: {article_id} feedback={action}, pending → records")
                return "recorded"

            existing = self._records.get(article_id)
            if existing is None:
                logger.warning(f"Deal Memory: article_id {article_id} 不在 pending/records 中")
                return "not_found"

            prev = existing.get("feedback") or {}
            prev_action = prev.get("action")
            if prev_action == action and action == "deal_not_worth" and reason is not None:
                existing["feedback"] = _build_feedback_payload(
                    action,
                    reason,
                    acted_at=str(prev.get("acted_at") or ""),
                    previous_action=prev.get("previous_action"),
                )
                self._save()
                logger.info(f"Deal Memory: {article_id} feedback={action} reason updated")
                return "reason_updated"

            if prev_action == action:
                existing["feedback"] = None
                self._pending[article_id] = existing
                del self._records[article_id]
                self._save()
                logger.info(f"Deal Memory: {article_id} feedback={action} 已取消，records → pending")
                return "cancelled"

            existing["feedback"] = _build_feedback_payload(action, reason, previous_action=prev_action)
            self._save()
            logger.info(f"Deal Memory: {article_id} feedback={prev_action} → {action}")
            return "updated"

    # ── 读取操作 ──

    def get_pending(self, article_id: str) -> dict | None:
        """按 article_id 查找 pending 条目。"""
        with self._lock:
            return self._pending.get(article_id)

    def get_records(self, limit: int | None = 100) -> list[dict]:
        """获取最近的 records，按反馈时间倒序。limit=None 时不截断。"""
        with self._lock:
            entries = list(self._records.values())
        entries.sort(key=lambda e: (e.get("feedback", {}).get("acted_at", "")), reverse=True)
        return entries[:limit] if limit else entries

    def get_records_by_category(self, category_hint: str) -> list[dict]:
        """按 category_hint 精确匹配获取 records。"""
        with self._lock:
            return [
                entry
                for entry in self._records.values()
                if str(entry.get("category_hint", "")).strip() == category_hint
            ]

    # ── 清理 ──

    def cleanup_expired_pending(self, expire_days: int = 30) -> int:
        """清理超时 pending，静默删除不做推断。Returns 清理数量。"""
        threshold = time.time() - expire_days * 86400
        with self._lock:
            expired = [
                aid
                for aid, entry in self._pending.items()
                if entry.get("timestamp", 0) < threshold
            ]
            for aid in expired:
                del self._pending[aid]
            if expired:
                self._save()
        if expired:
            logger.info(f"Deal Memory: 清理 {len(expired)} 条过期 pending")
        return len(expired)

    def compact(self) -> int:
        """清除超过 expire_days 的旧 records。Returns 清理数量。"""
        with self._lock:
            return self._compact_unlocked()

    # ── 属性 ──

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def last_analysis_date(self) -> str:
        """获取上次 LLM 分析的日期。"""
        with self._lock:
            return self._meta.get("last_analysis_date", "")

    def set_last_analysis_date(self, date_str: str) -> None:
        """设置上次 LLM 分析的日期。"""
        with self._lock:
            self._meta["last_analysis_date"] = date_str
            self._save()

    @property
    def analysis_failure_count(self) -> int:
        """今天分析连续失败次数。"""
        with self._lock:
            return self._meta.get("analysis_failure_count", 0)

    def record_analysis_failure(self) -> int:
        """记录一次分析失败，返回当前失败次数。"""
        with self._lock:
            count = self._meta.get("analysis_failure_count", 0) + 1
            self._meta["analysis_failure_count"] = count
            self._save()
            return count

    def reset_analysis_state(self) -> None:
        """分析成功后重置失败计数。"""
        with self._lock:
            self._meta["analysis_failure_count"] = 0
            self._save()

    # ── 内部方法 ──

    def _compact_unlocked(self) -> int:
        """清除过期 records（不加锁，调用者负责加锁）。仅在 __init__ 中调用。"""
        threshold = time.time() - self._expire_seconds
        expired = []
        for aid, entry in list(self._records.items()):
            fb = entry.get("feedback")
            if not fb or not fb.get("acted_at"):
                continue  # 跳过无 feedback 的记录，不误删
            if _parse_timestamp(fb["acted_at"]) < threshold:
                expired.append(aid)
        for aid in expired:
            del self._records[aid]
        if expired:
            self._save()
            logger.info(f"Deal Memory: compact 清理 {len(expired)} 条过期 records")
        return len(expired)


def _serialize_pending_entry(
    item: RankingItem,
    filter_reason: str,
    category_hint: str,
    decision_context: dict,
    arbiter_involved: bool,
    timestamp: float,
) -> dict:
    """将推荐商品序列化为 pending 条目。"""
    return {
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
        "tab_name": item.tab_name,
        "source_type": getattr(item, "source_type", ""),
        "search_keyword": getattr(item, "search_keyword", ""),
        "search_max_price": getattr(item, "search_max_price", None),
        "category_hint": category_hint or UNCATEGORIZED_CATEGORY,
        "recommendation": {
            "filter_reason": filter_reason,
            "arbiter_involved": arbiter_involved,
            "recommended_at": _format_timestamp(timestamp),
        },
        "context": {
            "filter_reason": filter_reason,
            "snapshot_time": _format_timestamp(timestamp),
            "decision_context": dict(decision_context or {}),
        },
        "feedback": None,
        "timestamp": timestamp,
    }


def _build_feedback_payload(
    action: str,
    reason: str | None,
    acted_at: str = "",
    previous_action: object = None,
) -> dict:
    payload = {
        "action": action,
        "acted_at": acted_at or _format_timestamp(time.time()),
    }
    if previous_action:
        payload["previous_action"] = previous_action
    clean_reason = str(reason or "").strip()
    if action == "deal_not_worth" and clean_reason:
        payload["reason"] = clean_reason
    return payload


def _format_timestamp(ts: float) -> str:
    """格式化时间戳为 ISO 格式字符串。"""
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def _parse_timestamp(ts_str: str) -> float:
    """解析 ISO 格式时间戳字符串为 float。"""
    if not ts_str:
        return 0.0
    try:
        from datetime import datetime

        dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
        return dt.timestamp()
    except (ValueError, OSError):
        return 0.0
