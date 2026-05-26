"""飞书通知目标绑定存储。"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from smzdm_notice.core import config

_BINDING_LOCK = threading.RLock()


@dataclass
class FeishuBinding:
    """当前唯一通知目标。"""

    receive_id_type: str
    receive_id: str
    bound_at: str
    bound_by_open_id: str
    source: str


class FeishuBindingStore:
    """读写当前飞书通知目标绑定。"""

    def __init__(self, filepath: str | Path | None = None) -> None:
        self.filepath = Path(filepath or config.FEISHU_BINDING_FILE)
        self._lock = _BINDING_LOCK

    def get(self) -> FeishuBinding | None:
        with self._lock:
            if not self.filepath.exists():
                return None
            try:
                data = json.loads(self.filepath.read_text(encoding="utf-8"))
                return FeishuBinding(**data)
            except (OSError, TypeError, json.JSONDecodeError) as e:
                logger.warning(f"飞书绑定文件读取失败: {e}")
                return None

    def bind(self, receive_id_type: str, receive_id: str, operator_open_id: str, source: str) -> FeishuBinding:
        if receive_id_type not in {"open_id", "chat_id"}:
            raise ValueError(f"不支持的绑定类型: {receive_id_type}")
        if not receive_id:
            raise ValueError("绑定目标 ID 为空")
        binding = FeishuBinding(
            receive_id_type=receive_id_type,
            receive_id=receive_id,
            bound_at=datetime.now().isoformat(timespec="seconds"),
            bound_by_open_id=operator_open_id,
            source=source,
        )
        with self._lock:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self.filepath.write_text(json.dumps(asdict(binding), ensure_ascii=False, indent=2), encoding="utf-8")
        return binding

    def clear(self) -> None:
        with self._lock:
            if self.filepath.exists():
                self.filepath.unlink()

    def is_bound_operator(self, operator_open_id: str) -> bool:
        binding = self.get()
        return bool(binding and operator_open_id and binding.bound_by_open_id == operator_open_id)

    def describe(self) -> str:
        binding = self.get()
        if not binding:
            return "未绑定"
        target = "私聊用户" if binding.receive_id_type == "open_id" else "群聊"
        return f"{target} ({binding.receive_id_type})，绑定时间：{binding.bound_at}"
