"""Feishu slash command metadata and help rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    command: str
    usage: str
    description: str
    group: str = "基础命令 / Basic"


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec("/help", "/help", "查看所有快捷命令 / Show all available shortcuts."),
    CommandSpec("/bind", "/bind", "绑定当前私聊或群聊为通知目标 / Bind this chat as the notification target."),
    CommandSpec("/unbind", "/unbind", "解绑当前通知目标 / Unbind the current notification target."),
    CommandSpec("/status", "/status", "查看运行状态 / Show runtime status."),
    CommandSpec("/run", "/run", "手动触发一次轮询 / Trigger one manual polling run."),
    CommandSpec("/restart", "/restart", "重启程序 / Restart the process when available."),
    CommandSpec(
        "/search", "/search", "查看当前搜索关键词 / List current search keywords.", "搜索关键词 / Search keywords"
    ),
    CommandSpec(
        "/search list",
        "/search list",
        "查看当前搜索关键词 / List current search keywords.",
        "搜索关键词 / Search keywords",
    ),
    CommandSpec(
        "/search add",
        "/search add <keyword> [-price <price>]",
        "添加搜索关键词；手机输入法里的单个长横线也可识别 / Add one keyword; single long dashes work.",
        "搜索关键词 / Search keywords",
    ),
    CommandSpec(
        "/search remove",
        "/search remove <keyword>",
        "删除完全匹配的搜索关键词 / Remove one exact keyword.",
        "搜索关键词 / Search keywords",
    ),
    CommandSpec(
        "/search price",
        "/search price <keyword> <price|clear>",
        "设置或清除关键词直推价格阈值 / Set or clear the direct-push price threshold.",
        "搜索关键词 / Search keywords",
    ),
    CommandSpec(
        "/search clear",
        "/search clear confirm",
        "清空全部搜索关键词 / Clear all search keywords.",
        "搜索关键词 / Search keywords",
    ),
)


def find_command_spec(command: str) -> CommandSpec | None:
    for spec in COMMAND_SPECS:
        if spec.command == command:
            return spec
    return None


def help_markdown() -> str:
    lines: list[str] = []
    groups: dict[str, list[CommandSpec]] = {}
    for spec in COMMAND_SPECS:
        groups.setdefault(spec.group, []).append(spec)

    for group, specs in groups.items():
        lines.append(f"**{group}**")
        for spec in specs:
            lines.append(f"- `{spec.usage}` - {spec.description}")
        lines.append("")
    return "\n".join(lines).strip()
