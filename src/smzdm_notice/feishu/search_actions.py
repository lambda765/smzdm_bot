"""用于 Feishu 搜索关键词命令与卡片 action 的处理器。"""

from __future__ import annotations

from smzdm_notice.smzdm import keywords as search_keywords


def handle_search_command(text: str, command: str) -> str:
    if command in {"/search", "/search list"}:
        return format_search_keywords(search_keywords.list_keyword_rules())
    if command == "/search add":
        result = search_keywords.add_keyword(_search_command_argument(text, "/search add"))
        return format_keyword_result(result)
    if command == "/search remove":
        result = search_keywords.remove_keyword(_search_command_argument(text, "/search remove"))
        return format_keyword_result(result)
    if command == "/search price":
        result = search_keywords.set_keyword_price(_search_command_argument(text, "/search price"))
        return format_keyword_result(result)
    if command == "/search clear":
        result = search_keywords.clear_keywords(_search_command_argument(text, "/search clear"))
        return format_keyword_result(result)
    return search_usage_text()


def handle_search_card_action(action: str, value: dict) -> str:
    keyword = str(value.get("search_keyword") or "").strip()
    if not keyword:
        return "搜索关键词信息缺失，无法处理。"
    if action == "search_remove_keyword":
        result = search_keywords.remove_keyword(keyword)
    else:
        result = search_keywords.set_keyword_price(f"{keyword} clear")
    return format_keyword_result(result)


def format_keyword_result(result: search_keywords.KeywordOperationResult) -> str:
    prefix = "OK" if result.success else "WARN"
    return f"{prefix}: {result.message}\n\n{format_search_keywords(result.rules or result.keywords)}"


def format_search_keywords(keywords: list) -> str:
    if not keywords:
        return "Search keywords: none"
    lines = ["Search keywords:"]
    lines.extend(f"{i}. {_format_search_keyword_entry(keyword)}" for i, keyword in enumerate(keywords, 1))
    return "\n".join(lines)


def search_usage_text() -> str:
    return (
        "Search keyword commands:\n"
        "- /search\n"
        "- /search list\n"
        "- /search add <keyword> [-price <price>]\n"
        "- /search remove <keyword>\n"
        "- /search price <keyword> <price|clear>\n"
        "- /search clear confirm"
    )


def _search_command_argument(text: str, prefix: str) -> str:
    return text.strip()[len(prefix) :].strip()


def _format_search_keyword_entry(keyword) -> str:
    if hasattr(keyword, "keyword"):
        max_price = getattr(keyword, "max_price", None)
        if max_price is not None:
            return f"{keyword.keyword} (max_price: {max_price:g})"
        return keyword.keyword
    return str(keyword)
