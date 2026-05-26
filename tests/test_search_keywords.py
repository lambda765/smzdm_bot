from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smzdm_notice.smzdm import keywords


class SearchKeywordManagerTests(unittest.TestCase):
    def test_missing_file_returns_empty_list(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(Path(tmp) / "missing.json")),
        ):
            self.assertEqual(keywords.list_keywords(), [])

    def test_add_creates_file_preserves_inner_spaces_and_dedupes_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)):
                first = keywords.add_keyword("  AirPods  Pro 2  ")
                second = keywords.add_keyword("AirPods  Pro 2")

            self.assertTrue(first.success)
            self.assertTrue(second.success)
            self.assertEqual(first.keywords, ["AirPods  Pro 2"])
            self.assertEqual(second.keywords, ["AirPods  Pro 2"])
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"keywords": [{"keyword": "AirPods  Pro 2", "max_price": None}]},
            )

    def test_add_with_price_and_update_price_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)):
                added = keywords.add_keyword("AirPods Pro 2 -price 99.9")
                spaced = keywords.add_keyword("AirPods  充电宝   -price   66")
                updated = keywords.set_keyword_price("AirPods Pro 2 88")
                cleared = keywords.set_keyword_price("AirPods Pro 2 clear")

            self.assertTrue(added.success)
            self.assertEqual(added.rules[0].max_price, 99.9)
            self.assertTrue(spaced.success)
            self.assertEqual(spaced.rules[1].keyword, "AirPods  充电宝")
            self.assertEqual(spaced.rules[1].max_price, 66)
            self.assertTrue(updated.success)
            self.assertEqual(updated.rules[0].max_price, 88)
            self.assertTrue(cleared.success)
            self.assertIsNone(cleared.rules[0].max_price)

    def test_add_with_unicode_single_dash_price_option(self) -> None:
        cases = [
            ("AirPods Pro 2 ‐price 99.9", "AirPods Pro 2", 99.9),
            ("充电宝 ‑price 88", "充电宝", 88.0),
            ("湿巾 ‒price 77", "湿巾", 77.0),
            ("抽纸 –price 66", "抽纸", 66.0),
            ("洗衣液 —price 55", "洗衣液", 55.0),
            ("牙膏 ―price 44", "牙膏", 44.0),
            ("纸巾 −price 33", "纸巾", 33.0),
            ("沐浴露 －price 22", "沐浴露", 22.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)):
                results = [keywords.add_keyword(text) for text, _keyword, _price in cases]

            self.assertTrue(all(result.success for result in results))
            rules = json.loads(path.read_text(encoding="utf-8"))["keywords"]
            self.assertEqual(
                rules,
                [{"keyword": keyword, "max_price": price} for _text, keyword, price in cases],
            )

    def test_add_rejects_double_dash_and_price_equals_forms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)):
                double_ascii = keywords.add_keyword("AirPods Pro 2 --price 99.9")
                double_unicode = keywords.add_keyword("AirPods Pro 2 ——price 99.9")
                double_full_width = keywords.add_keyword("AirPods Pro 2 －－price 99.9")
                equals_form = keywords.add_keyword("AirPods Pro 2 -price=99.9")

            self.assertFalse(double_ascii.success)
            self.assertFalse(double_unicode.success)
            self.assertFalse(double_full_width.success)
            self.assertFalse(equals_form.success)
            self.assertFalse(path.exists())

    def test_add_rejects_missing_or_invalid_price_option_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)):
                missing = keywords.add_keyword("AirPods Pro 2 -price")
                invalid = keywords.add_keyword("AirPods Pro 2 -price nope")
                extra = keywords.add_keyword("AirPods Pro 2 -price 99.9 extra")

            self.assertFalse(missing.success)
            self.assertFalse(invalid.success)
            self.assertFalse(extra.success)
            self.assertFalse(path.exists())

    def test_add_preserves_unicode_dash_keyword_without_price_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)):
                result = keywords.add_keyword("AirPods—Pro 2")

            self.assertTrue(result.success)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"keywords": [{"keyword": "AirPods—Pro 2", "max_price": None}]},
            )

    def test_reads_object_keyword_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            path.write_text(
                json.dumps(
                    {
                        "keywords": [
                            {"keyword": "AirPods Pro 2", "max_price": None},
                            {"keyword": "充电宝", "max_price": 88.5},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)):
                rules = keywords.list_keyword_rules()

            self.assertEqual([rule.keyword for rule in rules], ["AirPods Pro 2", "充电宝"])
            self.assertIsNone(rules[0].max_price)
            self.assertEqual(rules[1].max_price, 88.5)

    def test_rejects_legacy_string_keyword_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            path.write_text(json.dumps({"keywords": ["AirPods Pro 2"]}, ensure_ascii=False), encoding="utf-8")
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)), self.assertRaises(ValueError):
                keywords.list_keyword_rules()

    def test_remove_uses_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            path.write_text(
                json.dumps(
                    {
                        "keywords": [
                            {"keyword": "AirPods Pro 2", "max_price": None},
                            {"keyword": "AirPods  Pro 2", "max_price": None},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)):
                result = keywords.remove_keyword("AirPods Pro 2")

            self.assertTrue(result.success)
            self.assertEqual(result.keywords, ["AirPods  Pro 2"])

    def test_clear_requires_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            path.write_text(
                json.dumps({"keywords": [{"keyword": "a", "max_price": None}, {"keyword": "b", "max_price": None}]}),
                encoding="utf-8",
            )
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)):
                rejected = keywords.clear_keywords("")
                accepted = keywords.clear_keywords("confirm")

            self.assertFalse(rejected.success)
            self.assertEqual(rejected.keywords, ["a", "b"])
            self.assertTrue(accepted.success)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"keywords": []})

    def test_rejects_invalid_price_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            path.write_text(
                json.dumps({"keywords": [{"keyword": "AirPods Pro 2", "max_price": None}]}),
                encoding="utf-8",
            )
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)):
                added = keywords.add_keyword("充电宝 -price nope")
                result = keywords.set_keyword_price("AirPods Pro 2 nope")

            self.assertFalse(added.success)
            self.assertFalse(result.success)
            self.assertIn("Price must be a positive number", result.message)

    def test_malformed_json_raises_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_keywords.json"
            path.write_text("{bad json", encoding="utf-8")
            with patch.object(keywords.config, "SEARCH_KEYWORDS_FILE", str(path)), self.assertRaises(ValueError):
                keywords.add_keyword("AirPods")

            self.assertEqual(path.read_text(encoding="utf-8"), "{bad json")


if __name__ == "__main__":
    unittest.main()
