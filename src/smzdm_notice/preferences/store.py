"""配置草案存储与文件写入。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from smzdm_notice.core import config
from smzdm_notice.preferences.models import (
    ALLOWED_TARGETS,
    TERMINAL_DRAFT_RETENTION_SECONDS,
    ConfigDraft,
    DraftApplyOutcome,
)
from smzdm_notice.preferences.validation import content_hash, validate_draft_data

# 草案状态、配置文件写入和审计日志需要同一把锁保护，避免飞书按钮、
# 消息回复和轮询清理并发时出现“状态已变但文件未写”的交错。
CONFIG_FILE_LOCK = threading.RLock()


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
                draft_id: self._load_draft(draft) for draft_id, draft in data.items() if isinstance(draft, dict)
            }
        except (OSError, TypeError, json.JSONDecodeError) as e:
            logger.warning(f"草案文件读取失败，将重新创建: {e}")
            self._drafts = {}

    @staticmethod
    def _load_draft(data: dict) -> ConfigDraft:
        """读取草案时丢弃已废弃的整文件版本字段。"""
        normalized = dict(data)
        normalized.pop("base_content_hash", None)
        return ConfigDraft(**normalized)

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
        # 调用方拿到返回列表后负责禁用对应飞书卡片；取消原因和审计由存储层
        # 同步落盘，避免定时清理与用户点击应用产生不同的历史语义。
        expired = []
        with self._lock:
            for draft in self._drafts.values():
                if draft.status == "pending" and draft.is_expired:
                    draft.status = "cancelled"
                    draft.metadata["cancel_reason"] = "expired"
                    expired.append(draft)
            if expired:
                self._save()
                for draft in expired:
                    self._append_audit(draft, "cancelled", "system")
        return expired

    def compact(self, retention_seconds: int = TERMINAL_DRAFT_RETENTION_SECONDS) -> list[ConfigDraft]:
        """移除已结束且超过保留期的草案，返回被移除的列表。"""
        # 文件 pending_config_changes.json 只保留还有交互价值的草案；
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

    def cancel(self, draft_id: str, operator: str = "", *, reason: str) -> ConfigDraft | None:
        with self._lock:
            draft = self._drafts.get(draft_id)
            if not draft:
                return None
            draft.status = "cancelled"
            draft.metadata["cancel_reason"] = reason
            self._save()
            self._append_audit(draft, "cancelled", operator)
            return draft

    def apply(self, draft_id: str, operator: str = "") -> DraftApplyOutcome:
        with self._lock:
            draft, early_result = self._prepare_draft_for_apply(draft_id, operator=operator)
            if early_result is not None:
                return early_result
            assert draft is not None

            schema_error = self._draft_schema_error(draft)
            if schema_error:
                return self._reject_draft(draft, operator=operator, reason="invalid_schema", message=schema_error)
            if self.is_legacy_arbitration_draft(draft):
                return self._reject_draft(
                    draft,
                    operator=operator,
                    reason="legacy_arbitration_protocol",
                    message="旧版仲裁草案缺少差异归因，只允许查看历史状态，不能执行",
                )

            target_path = self._target_path(draft.target_file)
            if not target_path.exists():
                return self._reject_draft(
                    draft,
                    operator=operator,
                    reason="target_missing",
                    message=f"{draft.target_file} 不存在",
                )

            original = target_path.read_text(encoding="utf-8")
            if draft.edit_mode == "append" and self._contains_append_block(original, draft.append_text):
                draft.status = "applied"
                draft.metadata["apply_result"] = "content_already_present"
                self._save()
                return DraftApplyOutcome(
                    status="already_applied",
                    message=f"{draft.target_file} 已包含相同内容，未重复写入",
                    draft=draft,
                )
            validation = validate_draft_data(
                {
                    "target_file": draft.target_file,
                    "edit_mode": draft.edit_mode,
                    "append_text": draft.append_text,
                    "search_text": draft.search_text,
                    "replace_text": draft.replace_text,
                },
                original,
            )
            if not validation.ok:
                return DraftApplyOutcome(
                    status="needs_refresh",
                    message=validation.error,
                    draft=draft,
                )
            new_content = validation.new_content
            backup_path = self._backup(target_path)

            self._atomic_write_text(target_path, new_content)
            draft.status = "applied"
            self._save()
            self._append_audit(draft, "applied", operator, backup_path, original, new_content)
            return DraftApplyOutcome(
                status="applied",
                message=f"已写入 {draft.target_file}，备份：{backup_path.name}",
                draft=draft,
            )

    @staticmethod
    def _contains_append_block(original: str, append_text: str) -> bool:
        """按完整行匹配追加块，仅忽略候选块首尾的空行。"""
        candidate_lines = append_text.splitlines()
        while candidate_lines and not candidate_lines[0].strip():
            candidate_lines.pop(0)
        while candidate_lines and not candidate_lines[-1].strip():
            candidate_lines.pop()
        if not candidate_lines:
            return False

        original_lines = original.splitlines()
        block_size = len(candidate_lines)
        return any(
            original_lines[index : index + block_size] == candidate_lines
            for index in range(len(original_lines) - block_size + 1)
        )

    def _prepare_draft_for_apply(
        self,
        draft_id: str,
        *,
        operator: str,
    ) -> tuple[ConfigDraft | None, DraftApplyOutcome | None]:
        """在持锁状态下检查草案状态，并完成幂等状态更新。"""
        draft = self._drafts.get(draft_id)
        if not draft:
            return None, DraftApplyOutcome(status="missing", message="草案不存在或已过期")
        if draft.status == "applied":
            return draft, DraftApplyOutcome(status="already_applied", message="草案已应用过", draft=draft)
        if draft.status != "pending":
            return draft, DraftApplyOutcome(
                status="rejected",
                message=f"草案状态不是 pending: {draft.status}",
                draft=draft,
            )
        if draft.is_expired:
            draft.status = "cancelled"
            draft.metadata["cancel_reason"] = "expired"
            self._save()
            self._append_audit(draft, "cancelled", operator)
            return draft, DraftApplyOutcome(
                status="expired",
                message="草案已超过 24 小时自动失效",
                draft=draft,
            )
        return draft, None

    @staticmethod
    def _draft_schema_error(draft: ConfigDraft) -> str:
        if draft.target_file not in ALLOWED_TARGETS:
            return "目标文件无效"
        if draft.edit_mode not in {"append", "replace", "delete"}:
            return f"未知 edit_mode: {draft.edit_mode}"
        if draft.edit_mode == "append" and not draft.append_text.strip():
            return "append_text 不能为空"
        if draft.edit_mode in {"replace", "delete"} and not draft.search_text.strip():
            return "search_text 不能为空"
        return ""

    @staticmethod
    def is_legacy_arbitration_draft(draft: ConfigDraft) -> bool:
        metadata = draft.metadata if isinstance(draft.metadata, dict) else {}
        snapshot = metadata.get("arbitration_card")
        is_arbitration = metadata.get("card_kind") == "arbitration" or isinstance(snapshot, dict)
        if not is_arbitration:
            return False
        if not isinstance(snapshot, dict):
            return True
        assessment = snapshot.get("change_assessment")
        return not (
            isinstance(assessment, dict)
            and isinstance(assessment.get("cause"), str)
            and bool(assessment["cause"].strip())
            and isinstance(assessment.get("should_change_preference"), bool)
        )

    def _reject_draft(
        self,
        draft: ConfigDraft,
        *,
        operator: str,
        reason: str,
        message: str,
    ) -> DraftApplyOutcome:
        draft.status = "cancelled"
        draft.metadata["cancel_reason"] = reason
        self._save()
        self._append_audit(draft, "cancelled", operator)
        return DraftApplyOutcome(status="rejected", message=message, draft=draft)

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

    @staticmethod
    def _atomic_write_text(target_path: Path, content: str) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        original_mode = target_path.stat().st_mode if target_path.exists() else None
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target_path.name}.", dir=target_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if original_mode is not None:
                os.chmod(tmp_name, original_mode)
            os.replace(tmp_name, target_path)
        except Exception:
            with suppress(OSError):
                os.unlink(tmp_name)
            raise

    def _append_audit(
        self,
        draft: ConfigDraft,
        action: str,
        operator: str,
        backup_path: Path | None = None,
        original_content: str = "",
        new_content: str = "",
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
            "append_text": draft.append_text,
            "search_text": draft.search_text,
            "replace_text": draft.replace_text,
            "before_content_hash": content_hash(original_content) if original_content else "",
            "after_content_hash": content_hash(new_content) if new_content else "",
            "backup": str(backup_path) if backup_path else "",
            "metadata": draft.metadata,
        }
        # 字段 source/operator/backup 等审计信息只进 audit，不写回 preference.md/inventory.md，
        # 配置文件保持为纯偏好和库存正文。
        with self.audit_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def has_recent_suggestion(
        self,
        suggestion_hash: str,
        *,
        cancelled_within_seconds: int = 30 * 86400,
    ) -> bool:
        """检查建议是否已采纳，或近期被用户明确取消。"""
        if not suggestion_hash or not self.audit_file.exists():
            return False
        now = datetime.now().timestamp()
        try:
            for line in reversed(self.audit_file.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                event = json.loads(line)
                metadata = event.get("metadata") or {}
                if metadata.get("suggestion_hash") != suggestion_hash:
                    continue
                if event.get("action") == "applied":
                    return True
                if event.get("action") != "cancelled" or metadata.get("cancel_reason") != "user_cancelled":
                    continue
                event_time = datetime.fromisoformat(str(event.get("time"))).timestamp()
                return now - event_time <= cancelled_within_seconds
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return False
