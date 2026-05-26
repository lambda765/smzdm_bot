"""配置修改预览卡片渲染。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smzdm_notice.core import config


def build_draft_preview_content(draft: Any) -> str:
    """构造飞书配置修改预览 Markdown。"""
    mode = getattr(draft, "edit_mode", "append") or "append"
    mode_labels = {"append": "追加", "replace": "替换", "delete": "删除"}
    preview_text = build_draft_change_preview(draft)
    return (
        f"**{draft.title}**\n\n"
        f"目标文件：`{draft.target_file}` ｜ 操作：**{mode_labels.get(mode, mode)}**\n\n"
        f"说明：{draft.summary}\n\n"
        f"{preview_text}"
    )


def build_draft_change_preview(draft: Any, context_lines: int = 3) -> str:
    """按草案类型构造适合飞书卡片阅读的变更预览。"""
    mode = getattr(draft, "edit_mode", "append") or "append"
    if mode == "append":
        return build_append_change_preview(draft.target_file, draft.append_text, context_lines)
    if mode == "replace":
        return build_change_preview(draft.target_file, draft.search_text, draft.replace_text, context_lines)
    if mode == "delete":
        return build_change_preview(draft.target_file, draft.search_text, "", context_lines)
    return format_change_preview(
        target_file=draft.target_file,
        location="未知位置",
        context=[],
        removed=[],
        added=str(getattr(draft, "append_text", "")).splitlines(),
        warning=f"未知操作类型：{mode}",
    )


def build_append_change_preview(target_file: str, append_text: str, context_lines: int = 3) -> str:
    file_path = Path(config.PROJECT_ROOT) / target_file
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError:
        content = ""

    lines = content.splitlines()
    ctx_start = max(0, len(lines) - context_lines)
    context = [line for line in lines[ctx_start:] if not is_preview_noise(line)]
    return format_change_preview(
        target_file=target_file,
        location=f"文件末尾（当前 {len(lines)} 行）",
        context=context,
        removed=[],
        added=str(append_text or "").strip().splitlines(),
    )


def build_change_preview(target_file: str, search_text: str, replace_text: str, context_lines: int = 3) -> str:
    file_path = Path(config.PROJECT_ROOT) / target_file
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError:
        return format_fallback_change_preview(target_file, search_text, replace_text, "无法读取文件内容")

    idx = content.find(search_text)
    if idx < 0:
        return format_fallback_change_preview(target_file, search_text, replace_text, "未找到目标文本")

    lines = content.splitlines()
    start_line, end_line = locate_line_range(lines, idx, idx + len(search_text))
    ctx_start = max(0, start_line - context_lines)
    ctx_end = min(len(lines), end_line + context_lines + 1)
    context = [
        lines[i]
        for i in range(ctx_start, ctx_end)
        if (i < start_line or i > end_line) and not is_preview_noise(lines[i])
    ]
    return format_change_preview(
        target_file=target_file,
        location=format_line_location(start_line + 1, end_line + 1),
        context=context,
        removed=str(search_text or "").splitlines(),
        added=str(replace_text or "").splitlines(),
    )


def locate_line_range(lines: list[str], start_char: int, end_char: int) -> tuple[int, int]:
    start_line = 0
    char_pos = 0
    for i, line in enumerate(lines):
        if char_pos + len(line) + 1 > start_char:
            start_line = i
            break
        char_pos += len(line) + 1

    end_line = start_line
    char_pos = 0
    for i, line in enumerate(lines):
        if char_pos + len(line) + 1 >= end_char:
            end_line = i
            break
        char_pos += len(line) + 1
    return start_line, end_line


def format_fallback_change_preview(target_file: str, search_text: str, replace_text: str, reason: str) -> str:
    return format_change_preview(
        target_file=target_file,
        location="未定位",
        context=[],
        removed=str(search_text or "").splitlines(),
        added=str(replace_text or "").splitlines(),
        warning=reason,
    )


def format_change_preview(
    target_file: str,
    location: str,
    context: list[str],
    removed: list[str],
    added: list[str],
    warning: str = "",
    max_section_lines: int = 8,
) -> str:
    parts = ["**变更预览**", f"位置：`{target_file}` {location}"]
    if warning:
        parts.append(f"提示：{warning}")
    if context:
        parts.append("**附近原文**\n" + format_quote_lines(context, max_section_lines))
    if removed:
        parts.append("**删除**\n" + format_removed_lines(removed, max_section_lines))
    if added:
        parts.append("**新增**\n" + format_added_lines(added, max_section_lines))
    if not removed and not added:
        parts.append("没有可展示的变更内容。")
    return "\n\n".join(parts)


def format_quote_lines(lines: list[str], max_lines: int) -> str:
    selected, note = trim_preview_lines(lines, max_lines)
    body = "\n".join(f"> {sanitize_preview_line(line)}" for line in selected)
    return body + note


def format_removed_lines(lines: list[str], max_lines: int) -> str:
    selected, note = trim_preview_lines(lines, max_lines)
    body = "\n".join(f"~~{sanitize_preview_line(line)}~~" for line in selected)
    return body + note


def format_added_lines(lines: list[str], max_lines: int) -> str:
    selected, note = trim_preview_lines(lines, max_lines)
    body = "\n".join(sanitize_preview_line(line) for line in selected)
    return body + note


def trim_preview_lines(lines: list[str], max_lines: int) -> tuple[list[str], str]:
    cleaned = [line for line in lines if str(line).strip()]
    if len(cleaned) <= max_lines:
        return cleaned, ""
    return cleaned[:max_lines], f"\n...已省略 {len(cleaned) - max_lines} 行"


def format_line_location(start_line: int, end_line: int) -> str:
    if start_line == end_line:
        return f"第 {start_line} 行"
    return f"第 {start_line}-{end_line} 行"


def is_preview_noise(line: str) -> bool:
    stripped = str(line or "").strip()
    return not stripped or stripped in {"---", "***", "___"}


def sanitize_preview_line(line: str) -> str:
    return str(line).replace("```", "'''")
