"""配置草案数据模型。"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

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
