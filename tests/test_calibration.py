from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from smzdm_notice.core.calibration import (
    CalibrationGenerator,
    MemoryAnalyzer,
    compute_suggestion_hash,
)
from smzdm_notice.core.memory import DealMemoryStore
from smzdm_notice.llm.categories import UNCATEGORIZED_CATEGORY
from smzdm_notice.llm.memory_prompts import MEMORY_ANALYSIS_SYSTEM_PROMPT
from smzdm_notice.llm.routing import ResolvedLLMConfig
from smzdm_notice.smzdm.ranking import RankingItem


def _make_record(article_id: str, action: str, category: str = "厨房小家电",
                 title: str = "测试商品", price: str = "99.9",
                 worthy: int = 100, unworthy: int = 5, comments: int = 200,
                 tags: list | None = None, search_keyword: str = "",
                 decision_context: dict | None = None,
                 feedback_reason: str = "") -> dict:
    feedback = {"action": action, "acted_at": "2026-06-01T10:30:00"}
    if feedback_reason:
        feedback["reason"] = feedback_reason
    return {
        "article_id": article_id, "title": title, "price": price,
        "mall": "京东", "brand": "测试品牌",
        "worthy": worthy, "unworthy": unworthy, "comments": comments,
        "favorites": 50, "tags": tags or ["历史低价"], "link": "https://example.com",
        "tab_name": "热卖榜", "source_type": "search" if search_keyword else "ranking",
        "search_keyword": search_keyword, "category_hint": category,
        "recommendation": {"filter_reason": "好价", "arbiter_involved": False, "recommended_at": "2026-06-01T10:00:00"},
        "context": {
            "filter_reason": "好价",
            "snapshot_time": "2026-06-01T10:00:00",
            "decision_context": decision_context or {},
        },
        "feedback": feedback,
    }


def _ranking_item(tab_name: str, source_type: str = "ranking", search_keyword: str = "") -> RankingItem:
    return RankingItem(
        rank=1,
        title="测试商品",
        article_id="item-1",
        price="9.9元",
        worthy=10,
        unworthy=0,
        comments=5,
        favorites=1,
        mall="测试商城",
        brand="测试品牌",
        tab_name=tab_name,
        source_type=source_type,
        search_keyword=search_keyword,
    )


def _draft_llm_config(api_key: str = "draft-key") -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        agent="draft",
        connection="draft",
        connection_label="Draft",
        provider="openai_compatible",
        base_url="https://draft.example.com",
        api_key_env="LLM_DRAFT_TEST_API_KEY",
        api_key=api_key,
        model_id="draft-model",
        timeout_seconds=300.0,
        max_retries=2,
        temperature=0.1,
        response_format={"type": "json_object"},
        extra_body={},
    )


class CalibrationGeneratorTests(unittest.TestCase):
    def test_returns_empty_when_no_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=5)

            text = gen.build_section([_ranking_item("综合榜-家用电器")])

            self.assertEqual(text, "")

    def test_skips_category_below_min_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=5, min_category_records=2)
            store._records["1"] = _make_record("1", "deal_good", category="家用电器")

            text = gen.build_section([_ranking_item("综合榜-家用电器")])

            self.assertEqual(text, "")

    def test_injects_related_category_when_category_meets_min_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=5, min_category_records=2)
            store._records["1"] = _make_record("1", "deal_good", category="家用电器")
            store._records["2"] = _make_record("2", "deal_not_worth", category="家用电器")

            text = gen.build_section([_ranking_item("综合榜-家用电器")])

            self.assertIn("历史决策校准参考", text)
            self.assertIn("历史个例", text)
            self.assertIn("家用电器", text)
            self.assertIn("好价案例", text)
            self.assertIn("不值案例", text)

    def test_omits_snapshot_context_in_calibration_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=5, min_category_records=2)

            for i in range(2):
                store._records[str(i)] = _make_record(str(i), "deal_good", category="家用电器")

            text = gen.build_section([_ranking_item("综合榜-家用电器")])
            self.assertIn("推荐理由", text)
            self.assertNotIn("当时偏好", text)
            self.assertNotIn("当时库存", text)

    def test_includes_decision_context_in_calibration_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=5, min_category_records=2)
            decision_context = {
                "need_state": "urgent",
                "inventory_basis": "咖啡豆库存不足",
                "preference_basis": ["关注咖啡器具"],
                "threshold_adjustment": "relaxed_due_to_need",
                "context_summary": "急缺补货，允许放宽质量信号",
            }

            for i in range(2):
                store._records[str(i)] = _make_record(
                    str(i),
                    "deal_good",
                    category="咖啡器具",
                    search_keyword="咖啡豆",
                    decision_context=decision_context,
                )

            text = gen.build_section([_ranking_item("搜索-咖啡豆", source_type="search", search_keyword="咖啡豆")])

            self.assertIn("上下文", text)
            self.assertIn("急缺补货", text)
            self.assertIn("咖啡豆库存不足", text)
            self.assertIn("标准放宽", text)

    def test_includes_feedback_reason_in_calibration_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=5, min_category_records=2)
            store._records["1"] = _make_record(
                "1",
                "deal_not_worth",
                category="食品生鲜",
                title="京鲜生 陕西大荔冬枣",
                feedback_reason="水果类，值数不够高",
            )
            store._records["2"] = _make_record("2", "deal_not_worth", category="食品生鲜")

            text = gen.build_section([_ranking_item("综合榜-食品生鲜")])

            self.assertIn("用户反馈理由：水果类，值数不够高", text)

    def test_omits_empty_feedback_reason_in_calibration_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=5, min_category_records=2)
            store._records["1"] = _make_record("1", "deal_not_worth", category="食品生鲜")
            store._records["2"] = _make_record("2", "deal_not_worth", category="食品生鲜")

            text = gen.build_section([_ranking_item("综合榜-食品生鲜")])

            self.assertIn("不值案例", text)
            self.assertNotIn("用户反馈理由： |", text)
            self.assertNotIn("用户反馈理由：反馈时间", text)

    def test_respects_max_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=2, min_category_records=2)

            for i in range(10):
                store._records[str(i)] = _make_record(str(i), "deal_good", category="家用电器")

            text = gen.build_section([_ranking_item("综合榜-家用电器")])
            # 好价案例最多 2 条（max_examples=2）
            good_count = text.count(". [")
            self.assertLessEqual(good_count, 2)

    def test_only_good_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, min_category_records=2)

            for i in range(2):
                store._records[str(i)] = _make_record(str(i), "deal_good", category="家用电器")

            text = gen.build_section([_ranking_item("综合榜-家用电器")])
            self.assertIn("好价案例", text)
            self.assertNotIn("不值案例", text)

    def test_search_keyword_selects_related_custom_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, min_category_records=2)

            for i in range(4):
                category = "咖啡器具" if i < 2 else "宠物用品"
                keyword = "咖啡豆" if category == "咖啡器具" else "猫粮"
                store._records[str(i)] = _make_record(str(i), "deal_good", category=category, search_keyword=keyword)

            text = gen.build_section([_ranking_item("搜索-咖啡豆", source_type="search", search_keyword="咖啡豆")])

            self.assertIn("咖啡器具", text)
            self.assertNotIn("宠物用品", text)

    def test_global_records_do_not_override_per_category_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, min_category_records=2)
            categories = ["家用电器", "食品生鲜", "电脑数码", "运动户外", "图书文娱"]
            for i, category in enumerate(categories):
                store._records[str(i)] = _make_record(str(i), "deal_good", category=category)

            text = gen.build_section([_ranking_item("综合榜-家用电器")])

            self.assertEqual(text, "")

    def test_unrelated_and_uncategorized_records_are_not_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, min_category_records=2)
            store._records["1"] = _make_record("1", "deal_good", category="食品生鲜")
            store._records["2"] = _make_record("2", "deal_good", category="食品生鲜")
            store._records["3"] = _make_record("3", "deal_good", category=UNCATEGORIZED_CATEGORY)
            store._records["4"] = _make_record("4", "deal_good", category=UNCATEGORIZED_CATEGORY)

            text = gen.build_section([_ranking_item("综合榜-家用电器")])

            self.assertEqual(text, "")

    def test_min_category_records_is_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, min_category_records=3)
            store._records["1"] = _make_record("1", "deal_good", category="家用电器")
            store._records["2"] = _make_record("2", "deal_good", category="家用电器")

            self.assertEqual(gen.build_section([_ranking_item("综合榜-家用电器")]), "")

            store._records["3"] = _make_record("3", "deal_not_worth", category="家用电器")
            self.assertIn("家用电器", gen.build_section([_ranking_item("综合榜-家用电器")]))


class SuggestionHashTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        h1 = compute_suggestion_hash("零食需要极强信号才推荐")
        h2 = compute_suggestion_hash("零食需要极强信号才推荐")
        self.assertEqual(h1, h2)

    def test_different_for_different_rules(self) -> None:
        h1 = compute_suggestion_hash("零食需要极强信号才推荐")
        h2 = compute_suggestion_hash("厨房小家电接受度高")
        self.assertNotEqual(h1, h2)


class MemoryAnalyzerTests(unittest.TestCase):
    def _parse_suggested_rules(self, rules: list[dict]) -> list[dict]:
        payload = {
            "summary": "测试摘要",
            "patterns": [{"dimension": "品类偏好", "description": "探索性发现", "evidence_count": 3}],
            "suggested_rules": rules,
        }
        analysis = MemoryAnalyzer()._parse_response(json.dumps(payload, ensure_ascii=False))
        self.assertIsNotNone(analysis)
        return analysis.suggested_rules

    def test_build_messages_include_complete_current_preference(self) -> None:
        preference = "# preference\n\n## 水果\n- 水果需要更强社区信号\n"

        messages = MemoryAnalyzer()._build_messages([], preference)

        self.assertIn(preference, messages[1]["content"])

    def test_analyzes_three_records_without_internal_five_record_guard(self) -> None:
        captured = {}
        payload = {
            "summary": "样本足够形成初步偏好。",
            "patterns": [{"dimension": "品类偏好", "description": "厨房小家电接受度高", "evidence_count": 3}],
            "suggested_rules": [],
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
                )
            ]
        )

        def create_completion(**kwargs):
            captured.update(kwargs)
            return response

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)))
        records = [_make_record(str(i), "deal_good") for i in range(3)]

        with (
            patch("smzdm_notice.core.calibration.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.core.calibration.get_client_for_config", return_value=fake_client),
        ):
            analysis = MemoryAnalyzer().analyze(records)

        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.summary, "样本足够形成初步偏好。")
        self.assertEqual(captured["model"], "draft-model")

    def test_suggested_rule_rejects_evidence_count_without_good_not_worth_counts(self) -> None:
        rules = self._parse_suggested_rules(
            [
                {
                    "rule": "咖啡器具可优先推荐",
                    "reason": "样本数量足够",
                    "evidence": "模型只给了总样本数",
                    "evidence_count": 5,
                }
            ]
        )

        self.assertEqual(rules, [])

    def test_suggested_rule_rejects_balanced_good_not_worth_counts(self) -> None:
        rules = self._parse_suggested_rules(
            [
                {
                    "rule": "咖啡器具可优先推荐",
                    "reason": "好价 3，不值 2",
                    "evidence": "正反样本比例接近",
                    "good_count": 3,
                    "not_worth_count": 2,
                }
            ]
        )

        self.assertEqual(rules, [])

    def test_suggested_rule_accepts_one_sided_good_counts(self) -> None:
        rule = {
            "rule": "咖啡器具可优先推荐",
            "reason": "好价 5，不值 0",
            "evidence": "用户连续认可咖啡器具",
            "good_count": 5,
            "not_worth_count": 0,
        }

        self.assertEqual(self._parse_suggested_rules([rule]), [rule])

    def test_suggested_rule_accepts_directional_good_counts(self) -> None:
        rule = {
            "rule": "咖啡器具可优先推荐",
            "reason": "好价 4，不值 1",
            "evidence": "正向样本明显多于反向样本",
            "good_count": 4,
            "not_worth_count": 1,
        }

        self.assertEqual(self._parse_suggested_rules([rule]), [rule])

    def test_suggested_rule_rejects_insufficient_counts(self) -> None:
        rules = self._parse_suggested_rules(
            [
                {
                    "rule": "咖啡器具可优先推荐",
                    "reason": "好价 2，不值 1",
                    "evidence": "总样本不足",
                    "good_count": 2,
                    "not_worth_count": 1,
                }
            ]
        )

        self.assertEqual(rules, [])

    def test_suggested_rule_accepts_string_counts(self) -> None:
        rule = {
            "rule": "咖啡器具可优先推荐",
            "reason": "好价 5，不值 0",
            "evidence": "顶层计数为字符串",
            "good_count": "5",
            "not_worth_count": "0",
        }

        self.assertEqual(self._parse_suggested_rules([rule]), [rule])

    def test_suggested_rule_accepts_counts_from_reason_or_evidence_text(self) -> None:
        rule = {
            "rule": "咖啡器具可优先推荐",
            "reason": "用户反馈集中在咖啡器具",
            "evidence": "好价 5，不值 0",
        }

        self.assertEqual(self._parse_suggested_rules([rule]), [rule])

    def test_suggested_rule_rejects_hard_threshold_even_with_valid_counts(self) -> None:
        rules = self._parse_suggested_rules(
            [
                {
                    "rule": "咖啡器具必须值票 >= 100 才推荐",
                    "reason": "好价 5，不值 0",
                    "evidence": "硬阈值规则",
                    "good_count": 5,
                    "not_worth_count": 0,
                }
            ]
        )

        self.assertEqual(rules, [])

    def test_memory_analysis_prompt_requires_suggested_rule_counts(self) -> None:
        self.assertIn('"good_count": 5', MEMORY_ANALYSIS_SYSTEM_PROMPT)
        self.assertIn('"not_worth_count": 0', MEMORY_ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("suggested_rules 不使用 evidence_count", MEMORY_ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("两者都必须是非负整数", MEMORY_ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("decision_context", MEMORY_ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("relaxed_due_to_need", MEMORY_ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("不得泛化为任何时候都适用", MEMORY_ANALYSIS_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
