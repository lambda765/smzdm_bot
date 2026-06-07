from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from smzdm_notice.core.calibration import (
    _MIN_RECORDS_FOR_CALIBRATION,
    CalibrationGenerator,
    CalibrationProfile,
    MemoryAnalyzer,
    compute_suggestion_hash,
    load_calibration_profile_data,
    load_calibration_text,
    save_calibration_profile,
)
from smzdm_notice.core.memory import DealMemoryStore
from smzdm_notice.llm.memory_prompts import build_calibration_section
from smzdm_notice.llm.routing import ResolvedLLMConfig
from smzdm_notice.smzdm.ranking import RankingItem


def _make_record(article_id: str, action: str, category: str = "厨房小家电",
                 title: str = "测试商品", price: str = "99.9",
                 worthy: int = 100, unworthy: int = 5, comments: int = 200,
                 tags: list | None = None, search_keyword: str = "") -> dict:
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
        },
        "feedback": {"action": action, "acted_at": "2026-06-01T10:30:00"},
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
    def test_returns_empty_when_insufficient_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=5)

            # 直接写入少量 records
            for i in range(_MIN_RECORDS_FOR_CALIBRATION - 1):
                store._records[str(i)] = _make_record(str(i), "deal_good")

            profile = gen.generate()
            self.assertEqual(profile.calibration_text, "")
            self.assertEqual(profile.record_count, _MIN_RECORDS_FOR_CALIBRATION - 1)

    def test_generates_text_with_enough_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=5)

            for i in range(_MIN_RECORDS_FOR_CALIBRATION):
                action = "deal_good" if i < 3 else "deal_not_worth"
                store._records[str(i)] = _make_record(str(i), action)

            profile = gen.generate()
            self.assertTrue(len(profile.calibration_text) > 0)
            self.assertIn("历史决策校准参考", profile.calibration_text)
            self.assertIn("厨房小家电", profile.calibration_text)
            self.assertIn("好价案例", profile.calibration_text)
            self.assertIn("不值案例", profile.calibration_text)
            self.assertIn("厨房小家电", profile.calibration_by_category)

    def test_omits_snapshot_context_in_calibration_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=5)

            for i in range(_MIN_RECORDS_FOR_CALIBRATION):
                store._records[str(i)] = _make_record(str(i), "deal_good")

            profile = gen.generate()
            self.assertIn("推荐理由", profile.calibration_text)
            self.assertNotIn("当时偏好", profile.calibration_text)
            self.assertNotIn("当时库存", profile.calibration_text)

    def test_respects_max_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store, max_examples=2)

            for i in range(10):
                store._records[str(i)] = _make_record(str(i), "deal_good")

            profile = gen.generate()
            # 好价案例最多 2 条（max_examples=2）
            good_count = profile.calibration_text.count(". [")
            self.assertLessEqual(good_count, 2)

    def test_only_good_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store)

            for i in range(_MIN_RECORDS_FOR_CALIBRATION):
                store._records[str(i)] = _make_record(str(i), "deal_good")

            profile = gen.generate()
            self.assertIn("好价案例", profile.calibration_text)
            self.assertNotIn("不值案例", profile.calibration_text)

    def test_groups_by_category_and_records_search_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "memory.json")
            store = DealMemoryStore(filepath)
            gen = CalibrationGenerator(store)

            for i in range(_MIN_RECORDS_FOR_CALIBRATION):
                category = "咖啡器具" if i < 3 else "宠物用品"
                keyword = "咖啡豆" if category == "咖啡器具" else ""
                store._records[str(i)] = _make_record(str(i), "deal_good", category=category, search_keyword=keyword)

            profile = gen.generate()
            self.assertIn("咖啡器具", profile.calibration_by_category)
            self.assertIn("宠物用品", profile.calibration_by_category)
            self.assertEqual(profile.calibration_by_category["咖啡器具"]["search_keywords"], ["咖啡豆"])


class CalibrationProfileIOTests(unittest.TestCase):
    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "calibration.json")

            profile = CalibrationProfile(
                calibration_text="## 校准参考\n测试文本",
                calibration_by_category={
                    "咖啡器具": {
                        "text": "### 咖啡器具\n1. 测试",
                        "good_count": 5,
                        "not_worth_count": 0,
                        "search_keywords": ["咖啡豆"],
                    }
                },
                record_count=10,
                generated_at="2026-06-07T10:00:00",
            )
            save_calibration_profile(profile, filepath)

            text = load_calibration_text(filepath)
            self.assertIn("校准参考", text)
            self.assertIn("测试文本", text)
            data = load_calibration_profile_data(filepath)
            self.assertIn("咖啡器具", data["calibration_by_category"])

    def test_load_returns_empty_when_insufficient_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "calibration.json")

            profile = CalibrationProfile(
                calibration_text="some text",
                record_count=3,  # 低于阈值
                generated_at="2026-06-07T10:00:00",
            )
            save_calibration_profile(profile, filepath)

            text = load_calibration_text(filepath)
            self.assertEqual(text, "")
            data = load_calibration_profile_data(filepath)
            self.assertEqual(data["calibration_by_category"], {})

    def test_load_returns_empty_for_missing_file(self) -> None:
        text = load_calibration_text("/nonexistent/path.json")
        self.assertEqual(text, "")

    def test_build_calibration_section_selects_related_category_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = str(Path(tmp) / "calibration.json")
            profile = CalibrationProfile(
                calibration_text="legacy",
                calibration_by_category={
                    "家用电器": {
                        "text": "### 家用电器\n1. 电饭锅",
                        "good_count": 5,
                        "not_worth_count": 0,
                        "record_count": 5,
                        "search_keywords": [],
                    },
                    "咖啡器具": {
                        "text": "### 咖啡器具\n1. 手冲壶",
                        "good_count": 5,
                        "not_worth_count": 0,
                        "record_count": 5,
                        "search_keywords": ["咖啡豆"],
                    },
                },
                record_count=10,
                generated_at="2026-06-07T10:00:00",
            )
            save_calibration_profile(profile, filepath)

            appliance = _ranking_item("综合榜-家用电器")
            search = _ranking_item("搜索-咖啡豆", source_type="search", search_keyword="咖啡豆")
            hot = _ranking_item("热卖榜")

            with (
                patch("smzdm_notice.llm.memory_prompts.config.DEAL_MEMORY_ENABLED", True),
                patch("smzdm_notice.llm.memory_prompts.config.CALIBRATION_FILE", filepath),
            ):
                appliance_text = build_calibration_section([appliance])
                search_text = build_calibration_section([search])
                hot_text = build_calibration_section([hot])

            self.assertIn("家用电器", appliance_text)
            self.assertNotIn("咖啡器具", appliance_text)
            self.assertIn("咖啡器具", search_text)
            self.assertEqual(hot_text, "")


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


if __name__ == "__main__":
    unittest.main()
