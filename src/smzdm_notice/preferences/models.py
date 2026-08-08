"""配置草案数据模型。"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Literal

ALLOWED_TARGETS = {"preference.md", "inventory.md"}
DRAFT_TTL_SECONDS = 24 * 60 * 60
TERMINAL_DRAFT_RETENTION_SECONDS = 24 * 60 * 60


@dataclass
class ConfigDraft:
    """一次待用户确认的配置修改。"""

    draft_id: str
    target_file: str
    title: str
    summary: str
    append_text: str
    source: str
    created_at: float = field(default_factory=time.time)
    status: str = "pending"
    metadata: dict = field(default_factory=dict)
    edit_mode: str = "append"
    search_text: str = ""
    replace_text: str = ""
    preview_message_id: str = ""
    revision_history: list = field(default_factory=list)

    @property
    def signature(self) -> str:
        raw = f"{self.target_file}\n{self.edit_mode}\n{self.search_text}\n{self.append_text}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > DRAFT_TTL_SECONDS


@dataclass
class DraftBuildOutcome:
    """配置修改生成结果，区分无修改和真正的生成失败。"""

    status: Literal["draft", "noop", "rejected", "failed"]
    draft: ConfigDraft | None = None
    message: str = ""


@dataclass(frozen=True)
class DraftStreamEvent:
    """配置草案生成期间可供交互层展示的模型增量。"""

    kind: Literal["attempt_start", "reasoning_delta", "content_delta"]
    text: str = ""
    attempt: int = 1


@dataclass
class DraftApplyOutcome:
    """草案应用结果，区分真实冲突和不可重试的拒绝。"""

    status: Literal[
        "applied",
        "already_applied",
        "needs_refresh",
        "expired",
        "rejected",
        "missing",
    ]
    message: str = ""
    draft: ConfigDraft | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"applied", "already_applied"}
