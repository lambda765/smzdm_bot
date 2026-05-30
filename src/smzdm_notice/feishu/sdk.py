"""飞书 SDK 导入和客户端初始化工具。"""

from __future__ import annotations

import threading
from typing import Any

from smzdm_notice.core import config

_IMPORT_LOCK = threading.RLock()
_LARK_CLIENT: Any | None = None


def get_lark_module() -> Any:
    """串行导入 lark_oapi，避免长连接线程和发送线程并发导入死锁。"""
    with _IMPORT_LOCK:
        import lark_oapi as lark

        return lark


def get_lark_client() -> Any:
    """获取飞书 OpenAPI client。"""
    global _LARK_CLIENT
    with _IMPORT_LOCK:
        if _LARK_CLIENT is not None:
            return _LARK_CLIENT
        lark = get_lark_module()
        _LARK_CLIENT = lark.Client.builder().app_id(config.FEISHU_APP_ID).app_secret(config.FEISHU_APP_SECRET).build()
        return _LARK_CLIENT


def get_message_models() -> tuple[Any, Any]:
    """获取发送消息需要的 SDK model。"""
    with _IMPORT_LOCK:
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        return CreateMessageRequest, CreateMessageRequestBody


def get_reply_message_models() -> tuple[Any, Any]:
    """获取回复消息需要的 SDK model。"""
    with _IMPORT_LOCK:
        from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody

        return ReplyMessageRequest, ReplyMessageRequestBody


def get_message_reaction_models() -> tuple[Any, Any, Any]:
    """获取消息表情回复需要的 SDK model。"""
    with _IMPORT_LOCK:
        from lark_oapi.api.im.v1 import (
            CreateMessageReactionRequest,
            CreateMessageReactionRequestBody,
            Emoji,
        )

        return CreateMessageReactionRequest, CreateMessageReactionRequestBody, Emoji


def get_image_models() -> tuple[Any, Any]:
    """获取上传图片需要的 SDK model。"""
    with _IMPORT_LOCK:
        from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody

        return CreateImageRequest, CreateImageRequestBody


def get_file_models() -> tuple[Any, Any]:
    """获取上传文件需要的 SDK model。"""
    with _IMPORT_LOCK:
        from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody

        return CreateFileRequest, CreateFileRequestBody


def get_message_update_models() -> tuple[Any, Any]:
    """获取更新消息需要的 SDK model（使用 PATCH 方法）。"""
    with _IMPORT_LOCK:
        from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody

        return PatchMessageRequest, PatchMessageRequestBody


def get_card_action_response_model() -> Any:
    """获取卡片回调响应 model。"""
    with _IMPORT_LOCK:
        from lark_oapi.event.callback.model.p2_card_action_trigger import (
            P2CardActionTriggerResponse,
        )

        return P2CardActionTriggerResponse
