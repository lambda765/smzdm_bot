from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from smzdm_notice.core.memory import DealMemoryStore


def _make_item(article_id: str = "1001", title: str = "测试商品", price: str = "99.9",
               mall: str = "京东", brand: str = "测试品牌", worthy: int = 100,
               unworthy: int = 5, comments: int = 200, favorites: int = 50,
               tags: list | None = None, tab_name: str = "热卖榜", link: str = "https://example.com") -> dict:
    """构造一个模拟 RankingItem 属性的 dict，用于测试。"""
    return {
        "article_id": article_id, "title": title, "price": price,
        "mall": mall, "brand": brand, "worthy": worthy, "unworthy": unworthy,
        "comments": comments, "favorites": favorites,
        "tags": tags or ["历史低价"], "link": link, "tab_name": tab_name,
    }


class _MockRankingItem:
    """最小化的 RankingItem mock。"""

    def __init__(self, **kwargs):
        defaults = _make_item()
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)
        self.pic = ""
        self.rank = 0
        self.search_keyword = ""
        self.search_max_price = None


class DealMemoryStoreTests(unittest.TestCase):
    def _new_store(self, tmp: str, expire_days: int = 90) -> DealMemoryStore:
        return DealMemoryStore(str(Path(tmp) / "deal_memory.json"), expire_days=expire_days)

    def test_record_to_pending_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp)
            item = _MockRankingItem(article_id="1001", title="测试电饭锅")

            store.record_to_pending(
                [(item, "推荐理由：好价")],
                categories_by_article_id={"1001": "厨房小家电"},
            )

            self.assertEqual(store.pending_count, 1)
            self.assertEqual(store.record_count, 0)

            # 待反馈 pending 中有数据
            pending = store.get_pending("1001")
            self.assertIsNotNone(pending)
            self.assertEqual(pending["title"], "测试电饭锅")
            self.assertEqual(pending["category_hint"], "厨房小家电")
            self.assertIn("context", pending)
            self.assertIsNone(pending["feedback"])

            # 记录反馈
            result = store.record_feedback("1001", "deal_good")
            self.assertEqual(result, "recorded")
            self.assertEqual(store.pending_count, 0)
            self.assertEqual(store.record_count, 1)

            # 已确认 records 中有完整数据
            records = store.get_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["feedback"]["action"], "deal_good")
            self.assertIsNotNone(records[0]["feedback"]["acted_at"])

    def test_record_feedback_not_in_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp)
            result = store.record_feedback("nonexistent", "deal_good")
            self.assertEqual(result, "not_found")

    def test_record_feedback_invalid_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp)
            item = _MockRankingItem(article_id="1002")
            store.record_to_pending([(item, "test")])
            result = store.record_feedback("1002", "invalid_action")
            self.assertEqual(result, "invalid_action")

    def test_deal_not_worth_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp)
            item = _MockRankingItem(article_id="1003")
            store.record_to_pending([(item, "test")])
            result = store.record_feedback("1003", "deal_not_worth")
            self.assertEqual(result, "recorded")

            records = store.get_records()
            self.assertEqual(records[0]["feedback"]["action"], "deal_not_worth")

    def test_deal_not_worth_reason_can_be_added_and_cleared_without_cancelling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp)
            item = _MockRankingItem(article_id="1004")
            store.record_to_pending([(item, "test")])

            self.assertEqual(store.record_feedback("1004", "deal_not_worth"), "recorded")
            self.assertEqual(store.record_feedback("1004", "deal_not_worth", "价格一般"), "reason_updated")
            records = store.get_records()
            self.assertEqual(records[0]["feedback"]["action"], "deal_not_worth")
            self.assertEqual(records[0]["feedback"]["reason"], "价格一般")
            self.assertEqual(store.pending_count, 0)
            self.assertEqual(store.record_count, 1)

            self.assertEqual(store.record_feedback("1004", "deal_not_worth", ""), "reason_updated")
            records = store.get_records()
            self.assertNotIn("reason", records[0]["feedback"])
            self.assertEqual(store.pending_count, 0)
            self.assertEqual(store.record_count, 1)

            self.assertEqual(store.record_feedback("1004", "deal_not_worth"), "cancelled")
            self.assertEqual(store.pending_count, 1)
            self.assertEqual(store.record_count, 0)

    def test_cleanup_expired_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp)

            # 手动写入带旧时间戳的 pending
            store._pending["old1"] = {
                "article_id": "old1",
                "timestamp": time.time() - 31 * 86400,  # 31 天前
            }
            store._save()

            store._pending["fresh1"] = {
                "article_id": "fresh1",
                "timestamp": time.time(),
            }
            store._save()

            cleaned = store.cleanup_expired_pending(expire_days=30)
            self.assertEqual(cleaned, 1)
            self.assertEqual(store.pending_count, 1)
            self.assertIn("fresh1", store._pending)

    def test_compact_removes_old_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp, expire_days=30)

            # 直接写入一条过期的 record
            store._records["old_rec"] = {
                "article_id": "old_rec",
                "feedback": {
                    "action": "deal_good",
                    "acted_at": "2026-04-01T10:00:00",  # 很久以前
                },
            }
            store._records["new_rec"] = {
                "article_id": "new_rec",
                "feedback": {
                    "action": "deal_good",
                    "acted_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                },
            }
            store._save()

            removed = store.compact()
            self.assertEqual(removed, 1)
            self.assertEqual(store.record_count, 1)
            self.assertIn("new_rec", store._records)

    def test_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "deal_memory.json")
            store = DealMemoryStore(filepath)
            item = _MockRankingItem(article_id="1001")
            store.record_to_pending([(item, "test")])
            store.record_feedback("1001", "deal_good")

            # 重新加载
            store2 = DealMemoryStore(filepath)
            self.assertEqual(store2.record_count, 1)
            self.assertEqual(store2.pending_count, 0)

    def test_get_records_by_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp)

            # 直接写入 records
            store._records["a1"] = {"article_id": "a1", "category_hint": "厨房小家电",
                                    "feedback": {"action": "deal_good", "acted_at": ""}}
            store._records["a2"] = {"article_id": "a2", "category_hint": "厨房小家电",
                                    "feedback": {"action": "deal_not_worth", "acted_at": ""}}
            store._records["b1"] = {"article_id": "b1", "category_hint": "零食",
                                    "feedback": {"action": "deal_good", "acted_at": ""}}

            kitchen = store.get_records_by_category("厨房小家电")
            self.assertEqual(len(kitchen), 2)

            snacks = store.get_records_by_category("零食")
            self.assertEqual(len(snacks), 1)

    def test_analysis_date_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp)
            self.assertEqual(store.last_analysis_date, "")

            store.set_last_analysis_date("2026-06-07")
            self.assertEqual(store.last_analysis_date, "2026-06-07")

    def test_feedback_toggle_cancel_and_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp)
            item = _MockRankingItem(article_id="toggle1")
            store.record_to_pending([(item, "test")], categories_by_article_id={"toggle1": "厨房小家电"})

            self.assertEqual(store.record_feedback("toggle1", "deal_good"), "recorded")
            self.assertEqual(store.pending_count, 0)
            self.assertEqual(store.record_count, 1)

            self.assertEqual(store.record_feedback("toggle1", "deal_not_worth"), "updated")
            records = store.get_records()
            self.assertEqual(records[0]["feedback"]["action"], "deal_not_worth")
            self.assertEqual(records[0]["feedback"]["previous_action"], "deal_good")

            self.assertEqual(store.record_feedback("toggle1", "deal_not_worth"), "cancelled")
            self.assertEqual(store.pending_count, 1)
            self.assertEqual(store.record_count, 0)
            self.assertIsNone(store.get_pending("toggle1")["feedback"])

    def test_lightweight_context_stored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._new_store(tmp)
            item = _MockRankingItem(article_id="ctx1")
            store.record_to_pending(
                [(item, "test")],
                categories_by_article_id={"ctx1": "咖啡器具"},
                contexts_by_article_id={
                    "ctx1": {
                        "need_state": "urgent",
                        "inventory_basis": "咖啡豆库存不足",
                        "preference_basis": ["关注咖啡器具"],
                        "threshold_adjustment": "relaxed_due_to_need",
                        "context_summary": "急缺补货，标准放宽",
                    }
                },
            )

            pending = store.get_pending("ctx1")
            self.assertEqual(pending["category_hint"], "咖啡器具")
            self.assertEqual(pending["context"]["filter_reason"], "test")
            self.assertEqual(pending["context"]["decision_context"]["need_state"], "urgent")
            self.assertEqual(pending["context"]["decision_context"]["inventory_basis"], "咖啡豆库存不足")
            self.assertEqual(pending["context"]["decision_context"]["threshold_adjustment"], "relaxed_due_to_need")
            self.assertNotIn("preferences_snapshot", pending["context"])
            self.assertNotIn("inventory_snapshot", pending["context"])


if __name__ == "__main__":
    unittest.main()
