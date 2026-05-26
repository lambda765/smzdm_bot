from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qsl, urlsplit

from smzdm_notice.core import config
from smzdm_notice.smzdm.client import build_signed_params, compact_sign_value, get_json
from smzdm_notice.smzdm.ranking import RankingConfig, RankingItem, _ranking_params
from smzdm_notice.smzdm.search import _parse_search_rows, _search_params
from smzdm_notice.smzdm.sources import fetch_all_sources


class SmzdmSignTests(unittest.TestCase):
    def test_search_sign_uses_configured_key(self) -> None:
        url = (
            "https://s-api.smzdm.com/sou/list_v10?basic_v=0&category_id=&category_name=&f=iphone"
            "&filter_json_data=%7B%7D&is_biserial=2&keyword=AirPods%20Pro%202"
            "&limit=20&offset=0&order=score&page=1&search_from=%E8%BE%93%E5%85%A5%E6%90%9C%E7%B4%A2"
            "&search_scenarios=home&search_scene=18&search_session_id=&search_source=1"
            "&search_tab=good_price&sid=17793475509421&subtype=&tab_source="
            "&time=1779348218000&type=good_price&v=11.1.70&weixin=1"
            "&zhifa_tag_id=&zhilv_rate=&zhuanzai_ab=b"
        )
        params = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
        expected = _expected_sign(params, compact_sign_value, "test-sign-key")

        with patch.object(config, "SMZDM_SIGN_KEY", "test-sign-key"):
            signed = build_signed_params(params, value_normalizer=compact_sign_value)

        self.assertEqual(signed["sign"], expected)

    def test_ranking_sign_keeps_spaces_in_values(self) -> None:
        params = {"keyword": "a b", "time": 1000}

        with patch.object(config, "SMZDM_SIGN_KEY", "test-sign-key"):
            regular = build_signed_params(params)
            compact = build_signed_params(params, value_normalizer=compact_sign_value)

        self.assertNotEqual(regular["sign"], compact["sign"])

    def test_missing_sign_key_fails_fast(self) -> None:
        with patch.object(config, "SMZDM_SIGN_KEY", ""), self.assertRaisesRegex(RuntimeError, "SMZDM_SIGN_KEY"):
            build_signed_params({"keyword": "AirPods", "time": 1000})

    def test_missing_user_agent_fails_fast(self) -> None:
        with (
            patch.object(config, "SMZDM_SIGN_KEY", "test-sign-key"),
            patch.object(config, "SMZDM_USER_AGENT", ""),
            self.assertRaisesRegex(RuntimeError, "SMZDM_USER_AGENT"),
        ):
            get_json("https://example.com", "/api", {"keyword": "AirPods", "time": 1000})


def _expected_sign(params: dict, normalizer, sign_key: str) -> str:
    parts = []
    for key, value in sorted(params.items()):
        normalized = normalizer("" if value is None else str(value))
        if normalized:
            parts.append(f"{key}={normalized}")
    return hashlib.md5(("&".join(parts) + f"&key={sign_key}").encode()).hexdigest().upper()


class SmzdmSearchParsingTests(unittest.TestCase):
    def test_search_params_sort_by_time(self) -> None:
        self.assertEqual(_search_params("AirPods Pro 2", 20)["order"], "time")

    def test_search_params_use_configured_platform_and_version(self) -> None:
        with (
            patch.object(config, "SMZDM_CLIENT_PLATFORM", "android"),
            patch.object(config, "SMZDM_APP_VERSION", "10.8.0"),
        ):
            params = _search_params("AirPods Pro 2", 20)

        self.assertEqual(params["f"], "android")
        self.assertEqual(params["v"], "10.8.0")

    def test_ranking_params_use_configured_platform_and_version(self) -> None:
        ranking = RankingConfig(name="综合榜-全部", tab_id="67")
        with (
            patch.object(config, "SMZDM_CLIENT_PLATFORM", "iphone"),
            patch.object(config, "SMZDM_APP_VERSION", "11.2.0"),
        ):
            params = _ranking_params(ranking)

        self.assertEqual(params["f"], "iphone")
        self.assertEqual(params["v"], "11.2.0")

    def test_invalid_smzdm_platform_fails_fast(self) -> None:
        with patch.object(config, "SMZDM_CLIENT_PLATFORM", "ipad"), self.assertRaisesRegex(RuntimeError, "iphone"):
            _search_params("AirPods Pro 2", 20)

    def test_parse_search_rows_maps_fields_and_skips_empty_price(self) -> None:
        rows = [
            {
                "article_id": "1001",
                "article_title": "AirPods Pro 2纸尿裤",
                "article_price": "17.9",
                "article_mall": "拼多多",
                "article_collection": "58",
                "article_comment": "439",
                "article_worthy": "28",
                "article_unworthy": "5",
                "article_url": "https://www.smzdm.com/p/1001/",
                "article_pic": "https://example.com/1001.jpg",
                "article_tag": "精选",
                "article_tag_arr": ["每日白菜", "精选"],
                "article_tag_list": [{"article_title": "比上次发布低82%"}],
                "cell_type": "38009",
            },
            {
                "article_id": "1002",
                "article_title": "促销活动",
                "article_price": "",
                "cell_type": "38009",
            },
            {
                "article_id": "1003",
                "article_title": "已过期商品",
                "article_price": "1.0",
                "article_is_timeout": "is_timeout",
                "cell_type": "38009",
            },
            {
                "article_id": "1004",
                "article_title": "未过期商品",
                "article_price": "2.0",
                "article_is_timeout": "",
                "cell_type": "38009",
            },
        ]

        items = _parse_search_rows(rows, "AirPods Pro 2", 20, max_price=99.9)

        self.assertEqual(len(items), 2)
        item = items[0]
        self.assertEqual(item.article_id, "1001")
        self.assertEqual(item.mall, "拼多多")
        self.assertEqual(item.favorites, 58)
        self.assertEqual(item.comments, 439)
        self.assertEqual(item.tags, ["精选", "每日白菜", "比上次发布低82%"])
        self.assertEqual(item.tab_id, "search")
        self.assertEqual(item.tab_name, "搜索-AirPods Pro 2")
        self.assertEqual(item.source_type, "search")
        self.assertEqual(item.search_keyword, "AirPods Pro 2")
        self.assertEqual(item.search_max_price, 99.9)
        self.assertEqual(item.numeric_price, 17.9)
        self.assertEqual(items[1].article_id, "1004")
        self.assertEqual(items[1].numeric_price, 2.0)

    def test_parse_search_rows_keeps_invalid_numeric_price_as_none(self) -> None:
        rows = [
            {
                "article_id": "bad-price",
                "article_title": "异常价格商品",
                "article_price": "bad",
                "cell_type": "38009",
            }
        ]

        items = _parse_search_rows(rows, "AirPods Pro 2", 20, max_price=99.9)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].price, "bad")
        self.assertIsNone(items[0].numeric_price)

    def test_parse_search_rows_skips_items_older_than_30_days(self) -> None:
        now = 2_000_000_000.0
        rows = [
            {
                "article_id": "fresh",
                "article_title": "30天内商品",
                "article_price": "10.0",
                "publish_date_lt": str(now - 29 * 24 * 60 * 60),
                "cell_type": "38009",
            },
            {
                "article_id": "stale",
                "article_title": "超过30天商品",
                "article_price": "10.0",
                "publish_date_lt": str(now - 31 * 24 * 60 * 60),
                "cell_type": "38009",
            },
            {
                "article_id": "missing-time",
                "article_title": "缺少时间字段商品",
                "article_price": "10.0",
                "cell_type": "38009",
            },
            {
                "article_id": "bad-time",
                "article_title": "异常时间字段商品",
                "article_price": "10.0",
                "publish_date_lt": "not-a-number",
                "cell_type": "38009",
            },
        ]

        items = _parse_search_rows(rows, "AirPods Pro 2", 20, now=now)

        self.assertEqual([item.article_id for item in items], ["fresh", "missing-time", "bad-time"])


class SearchKeywordConfigTests(unittest.TestCase):
    def test_missing_keyword_file_returns_empty_list(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(config, "SEARCH_KEYWORDS_FILE", str(Path(tmp) / "missing.json")),
        ):
            self.assertEqual(config.get_search_keywords(), [])

    def test_valid_keyword_file_dedupes_and_strips_object_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keywords.json"
            path.write_text(
                json.dumps(
                    {
                        "keywords": [
                            {"keyword": " AirPods Pro 2 ", "max_price": None},
                            {"keyword": "", "max_price": None},
                            {"keyword": "充电宝", "max_price": 88},
                            {"keyword": "充电宝", "max_price": None},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(config, "SEARCH_KEYWORDS_FILE", str(path)):
                self.assertEqual(config.get_search_keywords(), ["AirPods Pro 2", "充电宝"])

    def test_legacy_string_keyword_file_raises_for_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keywords.json"
            path.write_text(json.dumps({"keywords": ["AirPods Pro 2"]}, ensure_ascii=False), encoding="utf-8")
            with patch.object(config, "SEARCH_KEYWORDS_FILE", str(path)), self.assertRaises(ValueError):
                config.get_search_keywords()

    def test_malformed_keyword_file_raises_for_caller_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keywords.json"
            path.write_text("{bad json", encoding="utf-8")
            with patch.object(config, "SEARCH_KEYWORDS_FILE", str(path)), self.assertRaises(json.JSONDecodeError):
                config.get_search_keywords()


class SourceAggregationTests(unittest.TestCase):
    def test_fetch_all_sources_merges_ranking_and_search_failures_are_non_blocking(self) -> None:
        ranking_item = RankingItem(
            rank=1,
            title="榜单商品",
            article_id="r1",
            price="9.9",
            worthy=1,
            unworthy=0,
            comments=0,
            favorites=0,
            mall="京东",
            brand="",
            link="https://example.com/r1",
        )
        search_item = RankingItem(
            rank=1,
            title="搜索商品",
            article_id="s1",
            price="19.9",
            worthy=2,
            unworthy=0,
            comments=0,
            favorites=0,
            mall="天猫",
            brand="",
            link="https://example.com/s1",
        )

        with (
            patch("smzdm_notice.smzdm.sources.get_ranking", return_value=[ranking_item]),
            patch("smzdm_notice.smzdm.sources.get_search", side_effect=[RuntimeError("search down"), [search_item]]),
        ):
            items = fetch_all_sources(
                [RankingConfig(name="综合榜")],
                ["坏关键词", "好关键词"],
                top_n=1,
                interval_seconds=0,
            )

        self.assertEqual([item.article_id for item in items], ["r1", "s1"])


if __name__ == "__main__":
    unittest.main()
