"""Search keyword JSON file management.

The persisted file only supports object entries:
{"keywords": [{"keyword": "...", "max_price": null}]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from smzdm_notice.core import config
from smzdm_notice.preferences.store import CONFIG_FILE_LOCK

ADD_KEYWORD_USAGE = "Usage: /search add <keyword> [-price <price>]"
PRICE_OPTION_NAME = "price"
# Device input methods may emit visually similar single dash characters.
OPTION_DASH_CHARS = frozenset("-‐‑‒–—―−－")


@dataclass(frozen=True)
class SearchKeywordRule:
    keyword: str
    max_price: float | None = None


@dataclass
class KeywordOperationResult:
    success: bool
    message: str
    keywords: list[str]
    rules: list[SearchKeywordRule] | None = None


def resolve_keywords_path() -> Path:
    """Resolve the keyword JSON path relative to the configured project root."""
    path = Path(config.SEARCH_KEYWORDS_FILE)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    return path


def list_keywords() -> list[str]:
    """Return configured keyword text values only."""
    with CONFIG_FILE_LOCK:
        return [rule.keyword for rule in _read_keyword_rules(resolve_keywords_path())]


def list_keyword_rules() -> list[SearchKeywordRule]:
    """Return configured keyword rules including optional max_price thresholds."""
    with CONFIG_FILE_LOCK:
        return _read_keyword_rules(resolve_keywords_path())


def add_keyword(keyword: str) -> KeywordOperationResult:
    """Add one exact keyword, optionally with a trailing single-dash price option."""
    try:
        clean, max_price = _parse_add_argument(keyword)
    except ValueError as e:
        rules = list_keyword_rules()
        return _result(False, str(e), rules)
    if not clean:
        rules = list_keyword_rules()
        return _result(False, ADD_KEYWORD_USAGE, rules)
    path = resolve_keywords_path()
    with CONFIG_FILE_LOCK:
        rules = _read_keyword_rules(path)
        if any(rule.keyword == clean for rule in rules):
            return _result(True, f"Keyword already exists: {clean}", rules)
        rules.append(SearchKeywordRule(clean, max_price))
        _write_keyword_rules(path, rules)
    suffix = f" (max_price: {max_price:g})" if max_price is not None else ""
    return _result(True, f"Added keyword: {clean}{suffix}", rules)


def remove_keyword(keyword: str) -> KeywordOperationResult:
    """Remove one keyword by exact text match."""
    clean = keyword.strip()
    if not clean:
        rules = list_keyword_rules()
        return _result(False, "Usage: /search remove <keyword>", rules)
    path = resolve_keywords_path()
    with CONFIG_FILE_LOCK:
        rules = _read_keyword_rules(path)
        if not any(rule.keyword == clean for rule in rules):
            return _result(False, f"Keyword not found: {clean}", rules)
        rules = [rule for rule in rules if rule.keyword != clean]
        _write_keyword_rules(path, rules)
    return _result(True, f"Removed keyword: {clean}", rules)


def set_keyword_price(keyword_and_price: str) -> KeywordOperationResult:
    """Set or clear max_price for an existing keyword rule."""
    keyword, price_text = _split_price_command(keyword_and_price)
    if not keyword or not price_text:
        rules = list_keyword_rules()
        return _result(False, "Usage: /search price <keyword> <price|clear>", rules)
    path = resolve_keywords_path()
    with CONFIG_FILE_LOCK:
        rules = _read_keyword_rules(path)
        for i, rule in enumerate(rules):
            if rule.keyword != keyword:
                continue
            if price_text.lower() == "clear":
                rules[i] = SearchKeywordRule(rule.keyword)
                _write_keyword_rules(path, rules)
                return _result(True, f"Cleared price threshold for keyword: {keyword}", rules)
            try:
                max_price = _parse_price_value(price_text)
            except ValueError as e:
                return _result(False, str(e), rules)
            rules[i] = SearchKeywordRule(rule.keyword, max_price)
            _write_keyword_rules(path, rules)
            return _result(True, f"Updated keyword price: {keyword} <= {max_price:g}", rules)
    return _result(False, f"Keyword not found: {keyword}", rules)


def clear_keywords(confirm: str) -> KeywordOperationResult:
    """Clear all keyword rules when the user supplies the confirm token."""
    path = resolve_keywords_path()
    with CONFIG_FILE_LOCK:
        rules = _read_keyword_rules(path)
        if confirm.strip().lower() != "confirm":
            return _result(False, "Usage: /search clear confirm", rules)
        _write_keyword_rules(path, [])
    return _result(True, "Cleared all search keywords.", [])


def _read_keyword_rules(path: Path) -> list[SearchKeywordRule]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Failed to read search keywords: {e}") from e
    raw_keywords = data.get("keywords") if isinstance(data, dict) else None
    if not isinstance(raw_keywords, list):
        return []
    rules: list[SearchKeywordRule] = []
    seen: set[str] = set()
    for item in raw_keywords:
        rule = _coerce_rule(item)
        if rule and rule.keyword not in seen:
            seen.add(rule.keyword)
            rules.append(rule)
    return rules


def _write_keyword_rules(path: Path, rules: list[SearchKeywordRule]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"keywords": [{"keyword": rule.keyword, "max_price": rule.max_price} for rule in rules]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _coerce_rule(item) -> SearchKeywordRule | None:
    if not isinstance(item, dict):
        raise ValueError("search_keywords.json only supports object entries with keyword and max_price fields.")
    keyword = str(item.get("keyword") or "").strip()
    if not keyword:
        return None
    max_price = item.get("max_price")
    if max_price in (None, ""):
        return SearchKeywordRule(keyword)
    try:
        return SearchKeywordRule(keyword, _parse_price_value(str(max_price)))
    except ValueError:
        return SearchKeywordRule(keyword)


def _parse_add_argument(text: str) -> tuple[str, float | None]:
    clean = text.strip()
    if not clean:
        return clean, None

    tokens = clean.split()
    if len(tokens) >= 2 and _is_price_option_like(tokens[-2]):
        if not _is_single_dash_price_option(tokens[-2]):
            raise ValueError(ADD_KEYWORD_USAGE)
        parts = clean.rsplit(maxsplit=2)
        keyword = parts[0].strip() if len(parts) == 3 else ""
        return keyword, _parse_price_value(tokens[-1])

    if any(_is_price_option_like(token) for token in tokens):
        raise ValueError(ADD_KEYWORD_USAGE)
    return clean, None


def _is_price_option_like(token: str) -> bool:
    dash_count, option_name = _split_option_token(token)
    option_name = option_name.lower()
    return dash_count > 0 and (option_name == PRICE_OPTION_NAME or option_name.startswith(f"{PRICE_OPTION_NAME}="))


def _is_single_dash_price_option(token: str) -> bool:
    dash_count, option_name = _split_option_token(token)
    return dash_count == 1 and option_name.lower() == PRICE_OPTION_NAME


def _split_option_token(token: str) -> tuple[int, str]:
    dash_count = 0
    for char in token:
        if char not in OPTION_DASH_CHARS:
            break
        dash_count += 1
    return dash_count, token[dash_count:]


def _split_price_command(text: str) -> tuple[str, str]:
    clean = text.strip()
    if not clean:
        return "", ""
    parts = clean.rsplit(maxsplit=1)
    if len(parts) != 2:
        return "", ""
    return parts[0].strip(), parts[1].strip()


def _parse_price_value(text: str) -> float:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError) as e:
        raise ValueError("Price must be a positive number.") from e
    if value <= 0:
        raise ValueError("Price must be a positive number.")
    return value


def _result(success: bool, message: str, rules: list[SearchKeywordRule]) -> KeywordOperationResult:
    return KeywordOperationResult(success, message, [rule.keyword for rule in rules], list(rules))
