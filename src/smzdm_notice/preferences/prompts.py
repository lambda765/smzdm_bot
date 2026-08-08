"""配置草案 LLM Prompt。"""

from __future__ import annotations

from pathlib import Path

from smzdm_notice.core import config
from smzdm_notice.preferences.validation import markdown_outline


def read_target_content(filename: str, root: Path | None = None) -> str:
    path = (root or config.PROJECT_ROOT) / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def draft_rules_prompt() -> str:
    return """\
<role>
你把用户对购物偏好或耗材库存的自然语言修改，转换成待确认的配置修改指令。只能输出 JSON 对象。
</role>

<rules>
edit_mode 取值：
- append：在文件末尾追加新内容。字段：edit_mode, target_file, title, summary, append_text。仅当没有合适的现有段落可融入时使用。
- replace：找到文件中已有文本并替换为新文本。字段：edit_mode, target_file, title, summary, search_text, replace_text。
- delete：从文件中删除已有文本。字段：edit_mode, target_file, title, summary, search_text。

target_file 只能是 preference.md 或 inventory.md：
- 库存、剩余数量、补货状态 → inventory.md
- 想买、不想买、已有物品、质量信号、专项品类评估要求 → preference.md
</rules>

<constraints>
- 优先使用 replace，把规则自然合并进现有章节或列表项，不堆到文件末尾
- 在现有章节中新增内容时，也使用 replace：用该章节内唯一的相邻规则作为 search_text，replace_text 包含原规则和新增规则
- search_text 必须是文件中已有的、足够精确定位的连续文本（1-3行），不能是模糊描述，应包含足够上下文使其在文件中唯一
- replace_text 是替换后的完整新文本，不是增量差异
- delete 模式将 search_text 完整移除，不需要 replace_text
- 只有当用户意思明确是新增独立条目且文件无合适位置时才用 append
- 如果现有规则已经完整覆盖用户意图，输出 {"edit_mode":"noop","target_file":"preference.md","summary":"现有规则已覆盖"}
- 修改前检查全文，避免同一需求散落在多个章节、与已有物品/黑名单/关注项冲突，或重复已有规则
- 由你负责语义去重：结合完整标题路径和规则含义识别近义内容；同名子标题位于不同父章节时可以合法共存
- append_text/replace_text 使用自然的 Markdown 正文，可直接成为配置文件的一部分
- 不在 append_text/replace_text 中写"机器人确认修改""来源""仲裁建议""一键采纳"等审计或包装文字
- 数字阈值默认写成参考线或质量信号，不要写成硬性门槛；只有用户明确使用"必须""一律""不得低于""绝不推荐"等强约束时，才生成硬规则表述
</constraints>
"""


def file_context_block(root: Path | None = None) -> str:
    pref_content = read_target_content("preference.md", root)
    inv_content = read_target_content("inventory.md", root)
    return (
        f"\n\n--- preference.md 动态结构索引 ---\n{markdown_outline(pref_content)}"
        f"\n\n--- preference.md 当前完整内容 ---\n{pref_content}"
        f"\n\n--- inventory.md 动态结构索引 ---\n{markdown_outline(inv_content)}"
        f"\n\n--- inventory.md 当前完整内容 ---\n{inv_content}"
    )


def revision_system_prompt(root: Path | None = None) -> str:
    return (
        "<role>\n"
        "你是一个配置修改助手。以下是用户和系统之间的对话历史，"
        "包含用户的原始请求、系统生成的草案、以及后续的修改意见。"
        "请根据完整对话历史，生成最终的配置修改指令。\n"
        "</role>\n\n"
        "<critical_rule>\n"
        "历史中的草案是尚未执行的修改方案，不是文件实际内容。"
        "用户回复预览卡片时，通常是在修改这份待确认草案"
        "（如去掉某一条、改弱某个表述、保留另一条），"
        "此时应输出修改后的完整草案，而不是对真实文件生成删除操作。\n\n"
        "只有当用户明确要求修改或删除当前真实文件中已存在的内容时，才输出 replace 或 delete。"
        "replace/delete 的 search_text 必须匹配下方文件当前的实际内容，"
        "不能使用未执行草案中的文本。\n"
        "</critical_rule>\n\n" + draft_rules_prompt() + file_context_block(root)
    )
