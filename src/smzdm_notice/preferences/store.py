"""配置草案存储与文件写入。"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from smzdm_notice.core import config
from smzdm_notice.preferences.models import ALLOWED_TARGETS, TERMINAL_DRAFT_RETENTION_SECONDS, ConfigDraft

# 草案状态、配置文件写入和审计日志需要同一把锁保护，避免飞书按钮、
# 消息回复和轮询清理并发时出现“状态已变但文件未写”的交错。
CONFIG_FILE_LOCK = threading.RLock()


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{4,}", "\n\n\n", text)


def _expand_search_context(original: str, search: str) -> str | None:
    """把短 search_text 扩展到相邻行，帮助唯一定位 replace/delete 目标。"""
    idx = original.find(search)
    if idx < 0:
        return None
    lines = original.splitlines(keepends=True)
    char_pos = 0
    start_line = 0
    for i, line in enumerate(lines):
        if char_pos + len(line) > idx:
            start_line = i
            break
        char_pos += len(line)
    end_line = start_line
    char_pos = 0
    for i, line in enumerate(lines):
        if char_pos + len(line) >= idx + len(search):
            end_line = i
            break
        char_pos += len(line)
    ctx_start = max(0, start_line - 1)
    ctx_end = min(len(lines), end_line + 2)
    return "".join(lines[ctx_start:ctx_end])


class DraftStore:
    """本地草案存储和配置文件写入器。"""

    def __init__(
        self,
        draft_file: str | Path | None = None,
        backup_dir: str | Path | None = None,
        audit_file: str | Path | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.root = Path(root or config.PROJECT_ROOT)
        self.draft_file = self._resolve_path(draft_file or config.CONFIG_DRAFT_FILE)
        self.backup_dir = self._resolve_path(backup_dir or config.CONFIG_BACKUP_DIR)
        self.audit_file = self._resolve_path(audit_file or config.CONFIG_AUDIT_FILE)
        self._lock = CONFIG_FILE_LOCK
        self._drafts: dict[str, ConfigDraft] = {}
        self._load()

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root / path

    def _load(self) -> None:
        if not self.draft_file.exists():
            return
        try:
            data = json.loads(self.draft_file.read_text(encoding="utf-8"))
            self._drafts = {
                draft_id: ConfigDraft(**draft) for draft_id, draft in data.items() if isinstance(draft, dict)
            }
        except (OSError, TypeError, json.JSONDecodeError) as e:
            logger.warning(f"草案文件读取失败，将重新创建: {e}")
            self._drafts = {}

    def _save(self) -> None:
        self.draft_file.parent.mkdir(parents=True, exist_ok=True)
        data = {draft_id: asdict(draft) for draft_id, draft in self._drafts.items()}
        self.draft_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def create(self, draft: ConfigDraft) -> ConfigDraft:
        if draft.target_file not in ALLOWED_TARGETS:
            raise ValueError(f"不允许修改 {draft.target_file}")
        with self._lock:
            self._drafts[draft.draft_id] = draft
            self._save()
        return draft

    def get(self, draft_id: str) -> ConfigDraft | None:
        with self._lock:
            return self._drafts.get(draft_id)

    def update(self, draft: ConfigDraft) -> None:
        with self._lock:
            if draft.draft_id in self._drafts:
                self._drafts[draft.draft_id] = draft
                self._save()

    def get_by_preview_message_id(self, msg_id: str) -> ConfigDraft | None:
        """按卡片消息 ID 查找仍可继续交互的草案。"""
        with self._lock:
            for draft in self._drafts.values():
                if draft.preview_message_id == msg_id and draft.status == "pending":
                    return draft
        return None

    def get_any_by_preview_message_id(self, msg_id: str) -> ConfigDraft | None:
        """按卡片消息 ID 查找任意状态草案，用于识别已失效旧卡片回复。"""
        with self._lock:
            for draft in self._drafts.values():
                if draft.preview_message_id == msg_id:
                    return draft
        return None

    def expire_pending(self) -> list[ConfigDraft]:
        """取消所有过期的 pending 草案，返回被清理的列表。"""
        # 这里只改变草案状态；调用方拿到返回列表后负责禁用对应飞书卡片。
        expired = []
        with self._lock:
            for draft in self._drafts.values():
                if draft.status == "pending" and draft.is_expired:
                    draft.status = "cancelled"
                    expired.append(draft)
            if expired:
                self._save()
        return expired

    def compact(self, retention_seconds: int = TERMINAL_DRAFT_RETENTION_SECONDS) -> list[ConfigDraft]:
        """移除已结束且超过保留期的草案，返回被移除的列表。"""
        # pending_config_changes.json 只保留还有交互价值的草案；
        # 长期历史依赖 audit 日志，避免 pending 文件持续膨胀。
        now = time.time()
        removed = []
        with self._lock:
            for draft_id, draft in list(self._drafts.items()):
                if draft.status in {"applied", "cancelled"} and now - draft.created_at > retention_seconds:
                    removed.append(draft)
                    self._drafts.pop(draft_id, None)
            if removed:
                self._save()
        return removed

    def cancel(self, draft_id: str, operator: str = "") -> ConfigDraft | None:
        with self._lock:
            draft = self._drafts.get(draft_id)
            if not draft:
                return None
            draft.status = "cancelled"
            self._save()
            self._append_audit(draft, "cancelled", operator)
            return draft

    def apply(self, draft_id: str, operator: str = "") -> tuple[bool, str]:
        with self._lock:
            draft = self._drafts.get(draft_id)
            if not draft:
                return False, "草案不存在或已过期"
            if draft.status == "applied":
                return True, "草案已应用过"
            if draft.status != "pending":
                return False, f"草案状态不是 pending: {draft.status}"
            if draft.is_expired:
                draft.status = "cancelled"
                self._save()
                return False, "草案已超过 24 小时自动失效"
            if self._is_signature_applied(draft.signature):
                draft.status = "applied"
                self._save()
                return True, "相同修改已采纳过，已标记为完成"

            target_path = self._target_path(draft.target_file)
            if not target_path.exists():
                return False, f"{draft.target_file} 不存在"

            original = target_path.read_text(encoding="utf-8")
            backup_path = self._backup(target_path)

            mode = draft.edit_mode or "append"
            if mode == "append":
                new_content = self._apply_append(original, draft)
            elif mode == "replace":
                new_content, err = self._apply_replace(original, draft)
                if err:
                    return False, err
            elif mode == "delete":
                new_content, err = self._apply_delete(original, draft)
                if err:
                    return False, err
            else:
                return False, f"未知的 edit_mode: {mode}"

            target_path.write_text(new_content, encoding="utf-8")
            draft.status = "applied"
            self._save()
            self._append_audit(draft, "applied", operator, backup_path)
            return True, f"已写入 {draft.target_file}，备份：{backup_path.name}"

    def _apply_append(self, original: str, draft: ConfigDraft) -> str:
        append_text = draft.append_text.strip()
        if not append_text:
            return original
        if not original.strip():
            return f"{append_text}\n"
        return _collapse_blank_lines(f"{original.rstrip()}\n\n{append_text}\n")

    def _apply_replace(self, original: str, draft: ConfigDraft) -> tuple[str, str | None]:
        search = draft.search_text
        if search not in original:
            return original, f"在 {draft.target_file} 中未找到目标文本，请检查后重试"
        count = original.count(search)
        if count > 1:
            # LLM 有时会给出 1-3 行短 search_text。若短文本多处出现，
            # 尝试带上相邻行后再唯一匹配；仍不唯一就拒绝，防止误改。
            expanded = _expand_search_context(original, search)
            if expanded and original.count(expanded) == 1:
                new_content = original.replace(expanded, draft.replace_text, 1)
                return new_content, None
            return original, f"在 {draft.target_file} 中找到 {count} 处匹配，为安全起见请提供更精确的文本"
        new_content = original.replace(search, draft.replace_text, 1)
        return new_content, None

    def _apply_delete(self, original: str, draft: ConfigDraft) -> tuple[str, str | None]:
        search = draft.search_text
        if search not in original:
            return original, f"在 {draft.target_file} 中未找到目标文本，请检查后重试"
        count = original.count(search)
        if count > 1:
            # delete 和 replace 使用同一套唯一定位策略，宁可让用户补充说明，
            # 也不在多处匹配时猜测删除哪一段。
            expanded = _expand_search_context(original, search)
            if expanded and original.count(expanded) == 1:
                new_content = original.replace(expanded, "", 1)
                new_content = _collapse_blank_lines(new_content)
                return new_content, None
            return original, f"在 {draft.target_file} 中找到 {count} 处匹配，为安全起见请提供更精确的文本"
        new_content = original.replace(search, "", 1)
        new_content = _collapse_blank_lines(new_content)
        return new_content, None

    def _target_path(self, target_file: str) -> Path:
        if target_file not in ALLOWED_TARGETS:
            raise ValueError(f"不允许修改 {target_file}")
        return self.root / target_file

    def _backup(self, target_path: Path) -> Path:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup_path = self.backup_dir / f"{target_path.name}.{stamp}.bak"
        shutil.copy2(target_path, backup_path)
        return backup_path

    def _append_audit(
        self,
        draft: ConfigDraft,
        action: str,
        operator: str,
        backup_path: Path | None = None,
    ) -> None:
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "operator": operator,
            "draft_id": draft.draft_id,
            "target_file": draft.target_file,
            "title": draft.title,
            "source": draft.source,
            "signature": draft.signature,
            "edit_mode": draft.edit_mode,
            "search_text": draft.search_text,
            "replace_text": draft.replace_text,
            "backup": str(backup_path) if backup_path else "",
            "metadata": draft.metadata,
        }
        # source/operator/backup 等审计信息只进 audit，不写回 preference.md/inventory.md，
        # 配置文件保持为纯偏好和库存正文。
        with self.audit_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _is_signature_applied(self, signature: str) -> bool:
        if not self.audit_file.exists():
            return False
        try:
            for line in self.audit_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("action") == "applied" and event.get("signature") == signature:
                    return True
        except (OSError, json.JSONDecodeError):
            return False
        return False
