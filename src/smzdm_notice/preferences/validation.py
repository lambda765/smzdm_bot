"""配置草案的确定性应用预演和 Markdown 结构校验。"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from smzdm_notice.preferences.models import ALLOWED_TARGETS

_AUDIT_NOISE = ("机器人确认修改", "仲裁建议一键采纳", "> 来源：")
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class DraftValidation:
    ok: bool
    error: str = ""
    new_content: str = ""


@dataclass(frozen=True)
class _Heading:
    level: int
    title: str
    path: tuple[str, ...]


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def markdown_outline(content: str) -> str:
    """生成动态标题索引，帮助模型定位而不固化章节模板。"""
    rows = []
    for match in _HEADING_RE.finditer(content):
        level = len(match.group(1))
        line = content.count("\n", 0, match.start()) + 1
        rows.append(f"- L{line} H{level}: {match.group(2).strip()}")
    return "\n".join(rows) if rows else "- （无 Markdown 标题）"


def validate_draft_data(data: dict | None, original: str) -> DraftValidation:
    if not isinstance(data, dict):
        return DraftValidation(False, "草案不是 JSON 对象")
    target = str(data.get("target_file") or "").strip()
    if target not in ALLOWED_TARGETS:
        return DraftValidation(False, "目标文件无效")
    mode = str(data.get("edit_mode") or "append").strip()
    if mode == "noop":
        return DraftValidation(True, new_content=original)
    if mode not in {"append", "replace", "delete"}:
        return DraftValidation(False, f"未知 edit_mode: {mode}")

    if mode == "append":
        added = str(data.get("append_text") or "").strip()
        if not added:
            return DraftValidation(False, "append_text 不能为空")
        if not original:
            new_content = f"{added}\n"
        elif original.endswith("\n\n"):
            new_content = f"{original}{added}\n"
        elif original.endswith("\n"):
            new_content = f"{original}\n{added}\n"
        else:
            new_content = f"{original}\n\n{added}\n"
    else:
        search = str(data.get("search_text") or "").strip()
        if not search:
            return DraftValidation(False, "search_text 不能为空")
        count = original.count(search)
        if count != 1:
            return DraftValidation(False, f"search_text 必须唯一命中，当前命中 {count} 处")
        replacement = str(data.get("replace_text") or "") if mode == "replace" else ""
        new_content = original.replace(search, replacement, 1)

    if new_content == original:
        return DraftValidation(False, "草案应用后文件没有变化")
    if target == "preference.md":
        error = _validate_preference_markdown(original, new_content, data)
        if error:
            return DraftValidation(False, error)
    return DraftValidation(True, new_content=new_content)


def _validate_preference_markdown(original: str, updated: str, data: dict) -> str:
    if any(noise in updated for noise in _AUDIT_NOISE):
        return "偏好正文包含审计或仲裁包装文字"

    original_headings = _parse_headings(original)
    updated_headings = _parse_headings(updated)
    if _issue_increased(_multiple_h1_issues(original_headings), _multiple_h1_issues(updated_headings)):
        return "Markdown 出现多个一级标题"
    if _issue_increased(_heading_jump_issues(original_headings), _heading_jump_issues(updated_headings)):
        return "Markdown 标题层级发生跳跃"
    if _issue_increased(_duplicate_heading_issues(original_headings), _duplicate_heading_issues(updated_headings)):
        return "Markdown 同一层级路径下出现重复标题"

    mode = str(data.get("edit_mode") or "append")
    if mode in {"replace", "delete"}:
        old_headings = [m.group(0) for m in _HEADING_RE.finditer(original)]
        new_headings = [m.group(0) for m in _HEADING_RE.finditer(updated)]
        search = str(data.get("search_text") or "")
        replacement = str(data.get("replace_text") or "")
        touches_heading = bool(_HEADING_RE.search(search) or _HEADING_RE.search(replacement))
        if old_headings != new_headings and not touches_heading:
            return "普通正文修改意外改变了标题结构"
    return ""


def _normalize_heading_title(title: str) -> str:
    # Markdown 允许结尾使用可选的 # 关闭 ATX 标题；它不属于标题语义。
    return re.sub(r"\s+", " ", title.rstrip().rstrip("#").rstrip()).casefold()


def _parse_headings(content: str) -> list[_Heading]:
    headings: list[_Heading] = []
    parents: list[tuple[int, str]] = []
    for match in _HEADING_RE.finditer(content):
        level = len(match.group(1))
        title = _normalize_heading_title(match.group(2))
        while parents and parents[-1][0] >= level:
            parents.pop()
        path = tuple(parent_title for _, parent_title in parents) + (title,)
        headings.append(_Heading(level=level, title=title, path=path))
        parents.append((level, title))
    return headings


def _multiple_h1_issues(headings: list[_Heading]) -> Counter[tuple[str]]:
    # 以“额外 H1 数量”计数，使已有多个 H1 不会阻断无关正文修改，
    # 但继续新增 H1 仍会被确定性拒绝。
    extra = max(0, sum(heading.level == 1 for heading in headings) - 1)
    return Counter({("multiple_h1",): extra}) if extra else Counter()


def _heading_jump_issues(headings: list[_Heading]) -> Counter[tuple]:
    issues: Counter[tuple] = Counter()
    if headings and headings[0].level > 1:
        issues[((), headings[0].path, 0, headings[0].level)] += 1
    for previous, current in zip(headings, headings[1:]):
        if current.level > previous.level + 1:
            issues[(previous.path, current.path, previous.level, current.level)] += 1
    return issues


def _duplicate_heading_issues(headings: list[_Heading]) -> Counter[tuple[str, ...]]:
    counts = Counter(heading.path for heading in headings)
    return Counter({path: count - 1 for path, count in counts.items() if count > 1})


def _issue_increased(original: Counter, updated: Counter) -> bool:
    return any(count > original.get(issue, 0) for issue, count in updated.items())
