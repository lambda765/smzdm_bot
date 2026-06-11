"""飞书自建应用机器人长连接交互层。"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from smzdm_notice.core import config
from smzdm_notice.feishu.binding import FeishuBindingStore
from smzdm_notice.feishu.commands import find_command_spec, help_markdown
from smzdm_notice.feishu.notifier import (
    ARBITRATION_CARD_KIND,
    ARBITRATION_CARD_METADATA_KEY,
    build_disabled_arbitration_card,
    build_disabled_draft_card,
    build_draft_failure_card,
    build_draft_processing_card,
    build_model_management_card,
    default_model_form_state,
    disable_draft_card,
    reply_card,
    reply_text,
    send_draft_preview,
    send_draft_processing,
    send_help,
    send_text,
    send_text_to,
    update_card_message,
    update_draft_preview,
)
from smzdm_notice.feishu.sdk import (
    get_card_action_response_model,
    get_lark_client,
    get_lark_module,
    get_message_reaction_models,
)
from smzdm_notice.llm import routing as llm_routing
from smzdm_notice.llm.clients import get_client_for_config
from smzdm_notice.llm.errors import (
    GENERAL_OPENAI_ERRORS,
    NON_RETRYABLE_OPENAI_ERRORS,
    RETRYABLE_OPENAI_ERRORS,
    error_summary,
)
from smzdm_notice.llm.routing import AGENTS, LLMRoutingError, ResolvedLLMConfig, build_chat_completion_kwargs
from smzdm_notice.preferences.builder import build_deal_action_draft, build_message_draft, build_revision_draft
from smzdm_notice.preferences.models import ConfigDraft
from smzdm_notice.preferences.store import DraftStore
from smzdm_notice.smzdm import keywords as search_keywords

DRAFT_PROGRESS_INTERVAL_SECONDS = 15
INTERNAL_ERROR_MESSAGE = "处理消息时遇到内部错误，请稍后重试。"


class MessageDeduper:
    """进程内消息幂等表，避免飞书重试导致同一 message_id 重复处理。"""

    def __init__(self, ttl_seconds: int = 24 * 60 * 60, time_func: Callable[[], float] | None = None) -> None:
        self.ttl_seconds = ttl_seconds
        self._time = time_func or time.time
        self._seen: dict[str, float] = {}
        self._lock = threading.RLock()

    def claim(self, message_id: str) -> bool:
        if not message_id:
            return True
        now = self._time()
        with self._lock:
            self._prune(now)
            expires_at = self._seen.get(message_id)
            if expires_at and expires_at > now:
                return False
            self._seen[message_id] = now + self.ttl_seconds
            return True

    def _prune(self, now: float) -> None:
        expired = [key for key, expires_at in self._seen.items() if expires_at <= now]
        for key in expired:
            self._seen.pop(key, None)


@dataclass
class BotRuntime:
    """飞书交互层需要调用的运行时能力。"""

    draft_store: DraftStore
    binding_store: FeishuBindingStore
    status_provider: Callable[[], str]
    run_once: Callable[[], bool]
    restart: Callable[[], bool] | None = None


@dataclass
class CardActionResult:
    message: str
    response_card: dict | None = None


@dataclass
class ModelCardUpdateResult:
    snapshot: llm_routing.RoutingSnapshot
    message: str


@dataclass
class DraftProcessingMessage:
    message_id: str = ""
    stop_event: threading.Event | None = None
    thread: threading.Thread | None = None


MODEL_CARD_FORM_STATE_LIMIT = 64


class FeishuInteractiveBot:
    """基于 lark-oapi 长连接接收消息和卡片事件。"""

    def __init__(self, runtime: BotRuntime, deduper: MessageDeduper | None = None) -> None:
        self.runtime = runtime
        self.deduper = deduper or MessageDeduper()
        self._model_card_form_state: dict[str, dict[str, Any]] = {}
        self._model_card_form_lock = threading.RLock()

    def start_blocking(self) -> None:
        if not (config.FEISHU_APP_ID and config.FEISHU_APP_SECRET):
            logger.warning("未配置 FEISHU_APP_ID/FEISHU_APP_SECRET，跳过飞书长连接机器人")
            return
        try:
            lark = get_lark_module()
            event_handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(self._handle_message)
                .register_p2_card_action_trigger(self._handle_card_action)
                .build()
            )
            ws_client = lark.ws.Client(
                app_id=config.FEISHU_APP_ID,
                app_secret=config.FEISHU_APP_SECRET,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO,
                auto_reconnect=True,
            )
            logger.info("飞书长连接机器人启动")
            ws_client.start()
        except ImportError:
            logger.error("缺少 lark-oapi，无法启动飞书长连接机器人")
        except Exception as e:
            logger.error(f"飞书长连接机器人异常退出: {e}", exc_info=True)

    def _handle_message(self, data) -> None:
        message_id = ""
        try:
            message = data.event.message
            text = _extract_message_text(getattr(message, "content", ""))
            if not text:
                return
            message_id = str(getattr(message, "message_id", "") or "")
            if not self.deduper.claim(message_id):
                logger.info(f"忽略重复飞书消息: {message_id}")
                return
            chat_id = str(getattr(message, "chat_id", "") or "")
            chat_type = str(getattr(message, "chat_type", "") or "")
            sender_open_id = _extract_sender_open_id(data)
            logger.info(
                f"收到飞书消息: message_id={message_id}, chat_id={chat_id}, "
                f"chat_type={chat_type}, sender_open_id={sender_open_id}, text={text}"
            )
            self._start_reaction_worker(message_id)
            parent_id = str(getattr(message, "parent_id", "") or "")
            self._start_message_worker(text, data, parent_id, message_id)
        except Exception as e:
            logger.error(f"处理飞书消息失败: {e}", exc_info=True)
            self._reply_text(message_id, INTERNAL_ERROR_MESSAGE)

    def _start_reaction_worker(self, message_id: str) -> None:
        if not message_id:
            return
        thread = threading.Thread(
            target=self._add_get_reaction,
            args=(message_id,),
            name="feishu-reaction-worker",
            daemon=True,
        )
        thread.start()

    def _start_message_worker(self, text: str, data, parent_id: str = "", reply_to_message_id: str = "") -> None:
        thread = threading.Thread(
            target=self._run_text_command,
            args=(text, data, parent_id, reply_to_message_id),
            name="feishu-message-worker",
            daemon=True,
        )
        thread.start()

    def _run_text_command(self, text: str, data, parent_id: str = "", reply_to_message_id: str = "") -> None:
        try:
            self._handle_text_command(text, data, parent_id, reply_to_message_id)
        except Exception as e:
            logger.error(f"处理飞书消息失败: {e}", exc_info=True)
            self._reply_text(reply_to_message_id, INTERNAL_ERROR_MESSAGE)

    def _add_get_reaction(self, message_id: str) -> None:
        if not message_id:
            return
        try:
            CreateMessageReactionRequest, CreateMessageReactionRequestBody, Emoji = get_message_reaction_models()
            request = (
                CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type("Get").build())
                    .build()
                )
                .build()
            )
            response = get_lark_client().im.v1.message_reaction.create(request)
            if not response.success():
                logger.warning(f"飞书消息 Get 表情回复失败: code={response.code}, msg={response.msg}")
        except Exception as e:
            logger.warning(f"飞书消息 Get 表情回复异常: {e}")

    def _handle_text_command(self, text: str, data, parent_id: str = "", reply_to_message_id: str = "") -> None:
        clean = _strip_bot_mention(text)
        if _is_bind_command(clean):
            self._bind_current_conversation(data, reply_to_message_id)
            return
        if not self.runtime.binding_store.get() and not _is_group_message(data):
            self._bind_current_conversation(data, reply_to_message_id)
            return
        if _is_unbind_command(clean):
            self._unbind_current_operator(data, reply_to_message_id)
            return
        if not _is_allowed_message(data, self.runtime.binding_store):
            self._maybe_prompt_bind(data, reply_to_message_id)
            return
        if parent_id:
            original_draft = self.runtime.draft_store.get_any_by_preview_message_id(parent_id)
            if original_draft:
                if original_draft.status != "pending":
                    disable_draft_card(parent_id, "该预览已失效，请以最新预览为准", original_draft)
                    self._reply_text(reply_to_message_id, "该预览已失效，请以最新预览为准。")
                    return
                if original_draft.is_expired:
                    self.runtime.draft_store.cancel(original_draft.draft_id)
                    if original_draft.preview_message_id:
                        disable_draft_card(
                            original_draft.preview_message_id,
                            "草案已超过 24 小时自动失效",
                            original_draft,
                        )
                    self._reply_text(reply_to_message_id, "该预览已超过 24 小时自动失效，请发新消息重新生成。")
                    return
                self._handle_draft_revision(clean, original_draft, data, reply_to_message_id)
                return
            if not clean.startswith("/"):
                self._reply_text(reply_to_message_id, "该回复引用的预览不存在或已失效，请发新消息重新生成。")
                return
        if self._handle_slash_command(clean, reply_to_message_id):
            return
        processing = self._start_draft_processing(reply_to_message_id, "正在理解偏好/库存修改")
        try:
            draft = build_message_draft(clean, self.runtime.draft_store)
        except Exception as e:
            processing_stopped = self._stop_draft_processing(processing)
            logger.error(f"配置草案生成失败: {e}", exc_info=True)
            self._finish_draft_processing_failure(
                processing if processing_stopped else DraftProcessingMessage(),
                "处理消息失败，没能生成配置修改预览。",
                reply_to_message_id,
                INTERNAL_ERROR_MESSAGE,
            )
            return
        processing_stopped = self._stop_draft_processing(processing)
        if not draft:
            self._finish_draft_processing_failure(
                processing if processing_stopped else DraftProcessingMessage(),
                "草案生成失败：没能理解这次偏好/库存修改。",
                reply_to_message_id,
                "草案生成失败：没能理解这次偏好/库存修改，请换一种更明确的说法重试。",
            )
            return
        processing_message_id = processing.message_id if processing_stopped else ""
        self._send_and_store_draft_preview(draft, reply_to_message_id, processing_message_id)

    def _handle_draft_revision(
        self,
        text: str,
        original_draft: ConfigDraft,
        data,
        reply_to_message_id: str = "",
    ) -> None:
        processing = self._start_draft_processing(reply_to_message_id, "正在根据修改意见生成新预览")
        try:
            revised = build_revision_draft(text, original_draft, self.runtime.draft_store)
        except Exception as e:
            processing_stopped = self._stop_draft_processing(processing)
            logger.error(f"配置草案修订失败: {e}", exc_info=True)
            self._finish_draft_processing_failure(
                processing if processing_stopped else DraftProcessingMessage(),
                "处理修改意见失败，没能生成新的配置修改预览。",
                reply_to_message_id,
                INTERNAL_ERROR_MESSAGE,
            )
            return
        processing_stopped = self._stop_draft_processing(processing)
        if not revised:
            self._finish_draft_processing_failure(
                processing if processing_stopped else DraftProcessingMessage(),
                "没能理解修改意见。",
                reply_to_message_id,
                "没能理解修改意见，请换一种说法重试，或发新消息重新生成。",
            )
            return
        processing_message_id = processing.message_id if processing_stopped else ""
        if self._send_and_store_draft_preview(revised, reply_to_message_id, processing_message_id):
            self.runtime.draft_store.cancel(original_draft.draft_id)
            if original_draft.preview_message_id:
                disable_draft_card(original_draft.preview_message_id, "已生成新的修改预览", original_draft)
            return
        self._reply_text(reply_to_message_id, "修改后的预览发送失败，原草案仍保留，可继续回复原预览。")

    def _send_and_store_draft_preview(
        self,
        draft: ConfigDraft,
        reply_to_message_id: str = "",
        processing_message_id: str = "",
    ) -> bool:
        preview_sent = False
        if processing_message_id:
            preview_sent = update_draft_preview(processing_message_id, draft)
            if not preview_sent:
                logger.warning(f"处理中卡片更新为预览失败，回退为发送新预览: {processing_message_id}")
        if not preview_sent:
            preview_sent = send_draft_preview(draft, reply_to_message_id=reply_to_message_id)
        if not preview_sent:
            self.runtime.draft_store.cancel(draft.draft_id)
            return False
        if draft.preview_message_id:
            self.runtime.draft_store.update(draft)
        return True

    def _start_draft_processing(self, reply_to_message_id: str, stage: str) -> DraftProcessingMessage:
        if not reply_to_message_id:
            return DraftProcessingMessage()
        message_id = send_draft_processing(stage, reply_to_message_id=reply_to_message_id)
        if not message_id:
            return DraftProcessingMessage()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_draft_processing_progress,
            args=(message_id, stage, time.monotonic(), stop_event),
            name="feishu-draft-progress-worker",
            daemon=True,
        )
        thread.start()
        return DraftProcessingMessage(message_id, stop_event, thread)

    def _run_draft_processing_progress(
        self,
        message_id: str,
        stage: str,
        started_at: float,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.wait(DRAFT_PROGRESS_INTERVAL_SECONDS):
            elapsed_seconds = int(time.monotonic() - started_at)
            update_card_message(message_id, build_draft_processing_card(stage, elapsed_seconds))

    def _stop_draft_processing(self, processing: DraftProcessingMessage) -> bool:
        if processing.stop_event:
            processing.stop_event.set()
        if processing.thread:
            processing.thread.join(timeout=1)
            return not processing.thread.is_alive()
        return True

    def _finish_draft_processing_failure(
        self,
        processing: DraftProcessingMessage,
        card_reason: str,
        reply_to_message_id: str,
        fallback_text: str,
    ) -> None:
        if processing.message_id and update_card_message(processing.message_id, build_draft_failure_card(card_reason)):
            return
        self._reply_text(reply_to_message_id, fallback_text)

    def _bind_current_conversation(self, data, reply_to_message_id: str = "") -> None:
        operator_open_id = _extract_sender_open_id(data)
        if not operator_open_id:
            self._reply_text(reply_to_message_id, "无法识别当前用户，请确认已开通接收消息权限。")
            return
        binding = self.runtime.binding_store.get()
        if binding and binding.bound_by_open_id != operator_open_id:
            self._send_to_sender(data, "当前已有绑定用户，只有原绑定用户可以切换通知目标。", reply_to_message_id)
            return
        receive_id_type, receive_id, source = _binding_target_from_message(data)
        binding = self.runtime.binding_store.bind(
            receive_id_type=receive_id_type,
            receive_id=receive_id,
            operator_open_id=operator_open_id,
            source=source,
        )
        target = "私聊" if binding.receive_id_type == "open_id" else "群聊"
        self._reply_text_to_target(
            reply_to_message_id,
            binding.receive_id_type,
            binding.receive_id,
            f"已绑定通知目标：{target}。后续好价通知会发送到这里。",
        )
        started = self.runtime.run_once()
        self._reply_text_to_target(
            reply_to_message_id,
            binding.receive_id_type,
            binding.receive_id,
            "已开始首次查询。" if started else "当前已有查询在执行，绑定成功后会继续使用该通知目标。",
        )

    def _unbind_current_operator(self, data, reply_to_message_id: str = "") -> None:
        operator_open_id = _extract_sender_open_id(data)
        binding = self.runtime.binding_store.get()
        if not binding:
            self._send_to_sender(data, "当前没有绑定通知目标。", reply_to_message_id)
            return
        if not operator_open_id or binding.bound_by_open_id != operator_open_id:
            self._send_to_sender(data, "只有当前绑定用户可以解绑。", reply_to_message_id)
            return
        receive_id_type = binding.receive_id_type
        receive_id = binding.receive_id
        self.runtime.binding_store.clear()
        self._reply_text_to_target(
            reply_to_message_id,
            receive_id_type,
            receive_id,
            "已解绑通知目标。重新私聊 /bind 可再次绑定。",
        )

    def _maybe_prompt_bind(self, data, reply_to_message_id: str = "") -> None:
        if not _is_group_message(data):
            self._send_to_sender(data, "请先发送 /bind 完成绑定，后续通知会发到这个私聊。", reply_to_message_id)

    def _handle_card_action(self, data) -> object | None:
        reply_to_message_id = _card_open_message_id(data)
        try:
            value = _extract_card_value(data)
            action = str(value.get("action") or "")
            operator = _extract_operator(data)
            card_token = str(getattr(getattr(data.event, "action", None), "token", "") or "")
            logger.info(f"收到飞书卡片操作: action={action}, operator={operator}, card_token={card_token}")
            if not action:
                field = _model_card_field_from_callback(value)
                if field in {"target", "connection", "model_id", "temperature"}:
                    if not self.runtime.binding_store.is_bound_operator(operator):
                        logger.debug(f"忽略未授权模型卡片表单变更: operator={operator}, field={field}")
                        return None
                    self._remember_model_card_form_value(reply_to_message_id, operator, value)
                if field in {"target", "connection"}:
                    return self._handle_model_form_change(reply_to_message_id, operator)
                logger.debug(f"忽略飞书卡片表单变更回调: operator={operator}, keys={sorted(value.keys())}")
                return None
            if not self.runtime.binding_store.is_bound_operator(operator):
                message = "只有当前绑定用户可以操作卡片"
                self._reply_text(reply_to_message_id, message)
                return _card_response(message)
            if action == "model_refresh":
                self._forget_model_card_form_state(reply_to_message_id, operator)
            if action.startswith("model_"):
                value = self._merge_model_card_form_state(reply_to_message_id, operator, value)

            result = self._dispatch_card_action(action, value, operator, reply_to_message_id)
            if action == "model_reset_agent" and not result.message.startswith("WARN:"):
                self._forget_model_card_form_state(reply_to_message_id, operator)
        except Exception as e:
            message = INTERNAL_ERROR_MESSAGE
            logger.error(f"处理卡片操作失败: {e}", exc_info=True)
            self._reply_text(reply_to_message_id, message)
            result = CardActionResult(message)
        return _card_response(result.message, result.response_card)

    def _dispatch_card_action(
        self,
        action: str,
        value: dict,
        operator: str,
        reply_to_message_id: str,
    ) -> CardActionResult:
        if action == "apply_draft":
            return self._apply_draft_card_action(value, operator, reply_to_message_id)
        if action == "cancel_draft":
            return self._cancel_draft_card_action(value, operator, reply_to_message_id)
        if action == "adopt_arbitration":
            message = "这是旧版仲裁卡片，请等待下一次仲裁后直接在卡片中采纳配置修改。"
            self._reply_text(reply_to_message_id, message)
            return CardActionResult(message, build_disabled_arbitration_card("旧版卡片已失效"))
        if action == "ignore_arbitration":
            return self._ignore_arbitration_card_action(value, operator, reply_to_message_id)
        if action in {"deal_ignore_category", "deal_stock_enough", "deal_follow"}:
            self._start_deal_action_worker(action, dict(value), reply_to_message_id)
            return CardActionResult("正在生成配置修改预览，请稍候。")
        if action in {"search_remove_keyword", "search_clear_price"}:
            message = self._handle_search_card_action(action, value)
            self._reply_text(reply_to_message_id, message)
            return CardActionResult(message)
        if action.startswith("model_"):
            return self._handle_model_card_action(action, value)
        message = f"未知操作：{action}"
        self._reply_text(reply_to_message_id, message)
        return CardActionResult(message)

    def _apply_draft_card_action(
        self,
        value: dict,
        operator: str,
        reply_to_message_id: str,
    ) -> CardActionResult:
        draft_id = str(value.get("draft_id") or "")
        draft = self.runtime.draft_store.get(draft_id)
        ok, message = self.runtime.draft_store.apply(draft_id, operator=operator)
        if not ok and _is_stale_draft(draft):
            message = "该预览已失效，请发新消息重新生成。"
        self._reply_text(reply_to_message_id, ("✅ " if ok else "⚠️ ") + message)
        if ok or _is_stale_draft(draft):
            reason = "已确认应用" if ok else "预览已失效"
            return CardActionResult(message, _build_disabled_card_for_action(reason, draft, value))
        return CardActionResult(message)

    def _cancel_draft_card_action(
        self,
        value: dict,
        operator: str,
        reply_to_message_id: str,
    ) -> CardActionResult:
        draft_id = str(value.get("draft_id") or "")
        draft = self.runtime.draft_store.get(draft_id)
        if draft and draft.status == "pending":
            draft = self.runtime.draft_store.cancel(draft_id, operator=operator)
            message = "已取消草案"
            reason = "已取消"
        else:
            message = "该预览已失效，请发新消息重新生成。"
            reason = "预览已失效"
        self._reply_text(reply_to_message_id, message)
        return CardActionResult(message, _build_disabled_card_for_action(reason, draft, value))

    def _ignore_arbitration_card_action(
        self,
        value: dict,
        operator: str,
        reply_to_message_id: str,
    ) -> CardActionResult:
        draft_id = str(value.get("draft_id") or "")
        draft = self.runtime.draft_store.get(draft_id) if draft_id else None
        if draft and draft.status == "pending":
            self.runtime.draft_store.cancel(draft_id, operator=operator)
            message = "已忽略本次仲裁建议"
            reason = "已忽略"
        else:
            message = "该预览已失效，请发新消息重新生成。"
            reason = "预览已失效"
        self._reply_text(reply_to_message_id, message)
        return CardActionResult(message, _build_disabled_card_for_action(reason, draft, value))

    def _start_deal_action_worker(self, action: str, value: dict, reply_to_message_id: str = "") -> None:
        thread = threading.Thread(
            target=self._run_deal_action,
            args=(action, value, reply_to_message_id),
            name="feishu-deal-action-worker",
            daemon=True,
        )
        thread.start()

    def _run_deal_action(self, action: str, value: dict, reply_to_message_id: str = "") -> None:
        processing = DraftProcessingMessage()
        try:
            processing = self._start_draft_processing(reply_to_message_id, "正在生成商品快捷操作预览")
            try:
                draft = build_deal_action_draft(action, value, self.runtime.draft_store)
            finally:
                processing_stopped = self._stop_draft_processing(processing)
            if not draft:
                self._finish_draft_processing_failure(
                    processing if processing_stopped else DraftProcessingMessage(),
                    "无法生成配置修改预览。",
                    reply_to_message_id,
                    "无法生成配置修改预览，请直接回复说明想怎么改。",
                )
                return
            processing_message_id = processing.message_id if processing_stopped else ""
            if not self._send_and_store_draft_preview(draft, reply_to_message_id, processing_message_id):
                self._reply_text(reply_to_message_id, "商品快捷操作预览发送失败")
        except Exception as e:
            processing_stopped = self._stop_draft_processing(processing)
            logger.error(f"商品快捷操作处理失败: {e}", exc_info=True)
            if processing_stopped and processing.message_id and update_card_message(
                processing.message_id,
                build_draft_failure_card("商品快捷操作处理失败，没能生成配置修改预览。"),
            ):
                return
            self._reply_text(reply_to_message_id, INTERNAL_ERROR_MESSAGE)

    def _handle_slash_command(self, text: str, reply_to_message_id: str = "") -> bool:
        if not text.startswith("/"):
            return False
        command = _command_key(text)
        if not find_command_spec(command):
            if command.startswith("/search"):
                self._reply_text(reply_to_message_id, _search_usage_text())
                return True
            if command.startswith("/model"):
                self._reply_text(reply_to_message_id, _model_usage_text())
                return True
            return False

        if command == "/help":
            content = help_markdown()
            if not send_help(content, reply_to_message_id=reply_to_message_id):
                self._reply_text(reply_to_message_id, content)
            return True
        if command == "/status":
            self._reply_text(reply_to_message_id, self.runtime.status_provider())
            return True
        if command == "/run":
            started = self.runtime.run_once()
            self._reply_text(reply_to_message_id, "已开始手动轮询。" if started else "当前已有轮询在执行，请稍后再试。")
            return True
        if command == "/restart":
            if not self.runtime.restart:
                self._reply_text(reply_to_message_id, "重启功能不可用")
                return True
            started = self.runtime.restart()
            self._reply_text(reply_to_message_id, "正在重启程序..." if started else "已在重启中，请稍候")
            return True
        if command.startswith("/search"):
            self._handle_search_command(text, command, reply_to_message_id)
            return True
        if command.startswith("/model"):
            self._handle_model_command(text, command, reply_to_message_id)
            return True
        return False

    def _handle_search_command(self, text: str, command: str, reply_to_message_id: str = "") -> None:
        try:
            if command in {"/search", "/search list"}:
                self._reply_text(reply_to_message_id, _format_search_keywords(search_keywords.list_keyword_rules()))
                return
            if command == "/search add":
                result = search_keywords.add_keyword(_search_command_argument(text, "/search add"))
                self._reply_text(reply_to_message_id, _format_keyword_result(result))
                return
            if command == "/search remove":
                result = search_keywords.remove_keyword(_search_command_argument(text, "/search remove"))
                self._reply_text(reply_to_message_id, _format_keyword_result(result))
                return
            if command == "/search price":
                result = search_keywords.set_keyword_price(_search_command_argument(text, "/search price"))
                self._reply_text(reply_to_message_id, _format_keyword_result(result))
                return
            if command == "/search clear":
                result = search_keywords.clear_keywords(_search_command_argument(text, "/search clear"))
                self._reply_text(reply_to_message_id, _format_keyword_result(result))
                return
            self._reply_text(reply_to_message_id, _search_usage_text())
        except ValueError as e:
            self._reply_text(reply_to_message_id, f"搜索关键词配置读取失败：{e}")

    def _handle_search_card_action(self, action: str, value: dict) -> str:
        keyword = str(value.get("search_keyword") or "").strip()
        if not keyword:
            return "搜索关键词信息缺失，无法处理。"
        if action == "search_remove_keyword":
            result = search_keywords.remove_keyword(keyword)
        else:
            result = search_keywords.set_keyword_price(f"{keyword} clear")
        return _format_keyword_result(result)

    def _handle_model_command(self, text: str, command: str, reply_to_message_id: str = "") -> None:
        try:
            if command == "/model status":
                self._reply_text(reply_to_message_id, llm_routing.format_status())
                return
            self._reply_model_card(reply_to_message_id)
        except LLMRoutingError as e:
            self._reply_text(reply_to_message_id, f"WARN: {e}")
        except ValueError as e:
            self._reply_text(reply_to_message_id, f"WARN: {e}")

    def _reply_model_card(self, reply_to_message_id: str = "") -> bool:
        state = llm_routing.model_card_state()
        card = build_model_management_card(state, form_state=default_model_form_state(state))
        if reply_to_message_id and reply_card(reply_to_message_id, card):
            return True
        self._reply_text(
            reply_to_message_id,
            "WARN: LLM 模型管理卡片发送失败，已降级显示当前状态。请查看日志中的飞书 code/msg。\n\n"
            + llm_routing.format_status(),
        )
        return False

    def _handle_model_card_action(self, action: str, value: dict) -> CardActionResult:
        try:
            if action == "model_refresh":
                return CardActionResult("已刷新 LLM 路由", _model_management_card())
            if action == "model_test":
                message = _run_model_test(_model_test_config_from_card(value))
                return CardActionResult(message, _model_management_card(form_state=_extract_form_state(value)))
            result = _apply_model_card_action(action, value)
            logger.info(
                "LLM 路由卡片操作成功: "
                f"action={action}, target={_model_card_optional(value, 'target') or 'default'}, "
                f"connection={_model_card_optional(value, 'connection')}, "
                f"model_id={_model_card_optional(value, 'model_id')}"
            )
            form_state = (
                _model_form_state_from_snapshot(result.snapshot, _model_card_target(value))
                if action == "model_reset_agent"
                else _extract_form_state(value)
            )
            return CardActionResult(result.message, _model_management_card(result.snapshot, form_state=form_state))
        except (LLMRoutingError, ValueError) as e:
            return CardActionResult(f"WARN: {e}", _model_management_card(form_state=_extract_form_state(value)))
        except Exception as e:
            logger.error(f"模型卡片操作失败: {e}", exc_info=True)
            return CardActionResult(INTERNAL_ERROR_MESSAGE, _model_management_card(form_state=_extract_form_state(value)))

    def _remember_model_card_form_value(self, message_id: str, operator: str, value: dict) -> None:
        key = _model_card_form_state_key(message_id, operator)
        if not key:
            return
        patch = _model_card_form_patch(value)
        if not patch:
            return
        with self._model_card_form_lock:
            state = self._model_card_form_state.pop(key, {})
            self._model_card_form_state[key] = state
            _apply_model_form_patch(state, patch)
            _trim_model_card_form_state(self._model_card_form_state)
            cached_keys = sorted(state.keys())
        logger.debug(f"缓存模型卡片表单值: operator={operator}, cached_keys={cached_keys}")

    def _merge_model_card_form_state(self, message_id: str, operator: str, value: dict) -> dict:
        key = _model_card_form_state_key(message_id, operator)
        if not key:
            return value
        with self._model_card_form_lock:
            cached = dict(self._model_card_form_state.get(key, {}))
        merged = dict(cached)
        merged.update(value)
        return merged

    def _forget_model_card_form_state(self, message_id: str, operator: str) -> None:
        key = _model_card_form_state_key(message_id, operator)
        if not key:
            return
        with self._model_card_form_lock:
            self._model_card_form_state.pop(key, None)

    def _handle_model_form_change(
        self,
        message_id: str,
        operator: str,
    ) -> object | None:
        """Redraw model card when target/connection dropdown changes."""
        cache_key = _model_card_form_state_key(message_id, operator)
        with self._model_card_form_lock:
            cached = dict(self._model_card_form_state.get(cache_key, {}))
            target = str(cached.get("target") or "default").strip()
            if target not in AGENTS and target != "default":
                return None
            final_form = self._model_form_for_target(target, cached)
            state = self._model_card_form_state.setdefault(cache_key, {})
            _replace_model_form_state_values(state, final_form)

        card = build_model_management_card(
            llm_routing.model_card_state(),
            form_state=final_form,
        )
        return _card_response(f"已切换到 {target}", card)

    def _model_form_for_target(self, target: str, cached: dict[str, Any]) -> dict[str, str]:
        auto_fill: dict[str, str] = {}
        if target == "default":
            state = llm_routing.model_card_state()
            defaults = state.get("defaults", {})
            auto_fill["connection"] = str(defaults.get("connection") or "")
            auto_fill["model_id"] = str(defaults.get("model_id") or "")
            auto_fill["temperature"] = str(defaults.get("temperature") or "") if defaults.get("temperature") is not None else ""
        else:
            try:
                resolved = llm_routing.resolve(target)
            except LLMRoutingError:
                return {"target": target}
            auto_fill["connection"] = resolved.connection
            auto_fill["model_id"] = resolved.model_id
            auto_fill["temperature"] = str(resolved.temperature) if resolved.temperature is not None else ""

        final_form: dict[str, str] = {"target": target}
        for field_key in ("connection", "model_id", "temperature"):
            # Preserve all user-edited route fields across target changes.
            # Users often choose connection/model_id first and then switch the agent;
            # auto-filling the new target here would silently discard their intended route.
            if cached.get(f"{field_key}_manual"):
                final_form[field_key] = str(cached.get(field_key) or "")
            else:
                final_form[field_key] = auto_fill.get(field_key, "")
        return final_form

    def _reply_text(self, reply_to_message_id: str, text: str) -> bool:
        if reply_to_message_id and reply_text(reply_to_message_id, text):
            return True
        return send_text(text)

    def _reply_text_to_target(
        self,
        reply_to_message_id: str,
        receive_id_type: str,
        receive_id: str,
        text: str,
    ) -> bool:
        if reply_to_message_id and reply_text(reply_to_message_id, text):
            return True
        return send_text_to(receive_id_type, receive_id, text)

    def _send_to_sender(self, data, text: str, reply_to_message_id: str = "") -> bool:
        if reply_to_message_id and reply_text(reply_to_message_id, text):
            return True
        return _send_to_sender(data, text)


def start_bot_thread(runtime: BotRuntime) -> threading.Thread | None:
    """启动飞书机器人后台线程。"""
    if not (config.FEISHU_APP_ID and config.FEISHU_APP_SECRET):
        return None
    bot = FeishuInteractiveBot(runtime)
    thread = threading.Thread(target=bot.start_blocking, name="feishu-bot", daemon=True)
    thread.start()
    return thread


def _extract_message_text(content: str) -> str:
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return str(data.get("text") or "").strip()
    except (TypeError, json.JSONDecodeError):
        pass
    return str(content or "").strip()


def _strip_bot_mention(text: str) -> str:
    clean = text.strip()
    while clean.startswith("@"):
        parts = clean.split(maxsplit=1)
        if len(parts) == 1:
            return ""
        clean = parts[1].strip()
    return clean


def _extract_card_value(data) -> dict:
    action = getattr(data.event, "action", None)
    result = _parse_card_dict(getattr(action, "value", None))
    if result.get("field"):
        result["value"] = dict(result)
    name = _clean_card_scalar(getattr(action, "name", None))
    option = _clean_card_scalar(getattr(action, "option", None))
    raw_input_value = getattr(action, "input_value", None)
    input_value = _clean_card_scalar(raw_input_value)
    tag = _clean_card_scalar(getattr(action, "tag", None))
    if name:
        result["name"] = name
    if option:
        result["option"] = option
    if _is_card_scalar(raw_input_value):
        result["input_value"] = input_value
    if tag:
        result["tag"] = tag
    for attr in ("form_value", "form_values", "form", "input_values"):
        form_value = _parse_card_dict(getattr(action, attr, None))
        if not form_value:
            continue
        result[attr] = form_value
        for key, value in form_value.items():
            result.setdefault(key, value)
    return result


def _parse_card_dict(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _clean_card_scalar(value) -> str:
    if value is None:
        return ""
    if not _is_card_scalar(value):
        return ""
    return str(value).strip()


def _is_card_scalar(value) -> bool:
    return isinstance(value, (str, int, float, bool))


def _extract_operator(data) -> str:
    operator = getattr(data.event, "operator", None)
    for attr in ("open_id", "user_id", "union_id"):
        value = getattr(operator, attr, None)
        if value:
            return str(value)
    return ""


def _card_open_message_id(data) -> str:
    context = getattr(getattr(data.event, "context", None), "open_message_id", "")
    if not isinstance(context, (str, int)):
        return ""
    return str(context or "")


def _model_card_form_state_key(message_id: str, operator: str) -> str:
    if not message_id or not operator:
        return ""
    return f"{message_id}:{operator}"


def _trim_model_card_form_state(state: dict[str, dict[str, Any]]) -> None:
    while len(state) > MODEL_CARD_FORM_STATE_LIMIT:
        oldest = next(iter(state), None)
        if oldest is None:
            return
        state.pop(oldest, None)


def _replace_model_form_state_values(state: dict[str, Any], form_state: dict[str, str]) -> None:
    for key in ("target", "connection", "model_id", "temperature"):
        state.pop(key, None)
        if key not in form_state:
            state.pop(f"{key}_manual", None)
    for key, value in form_state.items():
        state[key] = value


def _apply_model_form_patch(state: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if key.endswith("_manual") and value is False:
            state.pop(key, None)
        else:
            state[key] = value


def _model_card_form_patch(value: dict) -> dict[str, Any]:
    field = _model_card_field_from_callback(value)
    if field not in {"target", "connection", "model_id", "temperature"}:
        return {}
    if "option" in value:
        option = str(value.get("option") or "").strip()
        return {field: option, f"{field}_manual": bool(option)}
    if "input_value" in value:
        input_value = str(value.get("input_value") or "").strip()
        return {field: input_value, f"{field}_manual": bool(input_value)}
    return {}


def _model_card_field_from_callback(value: dict) -> str:
    raw_value = value.get("value")
    if isinstance(raw_value, dict):
        field = str(raw_value.get("field") or "").strip()
        if field:
            return field
    return str(value.get("name") or "").strip()


def _extract_sender_open_id(data) -> str:
    sender_id = getattr(getattr(data.event, "sender", None), "sender_id", None)
    return str(getattr(sender_id, "open_id", "") or "")


def _is_stale_draft(draft: ConfigDraft | None) -> bool:
    return draft is None or draft.status != "pending"


def _build_disabled_card_for_action(reason: str, draft: ConfigDraft | None, value: dict) -> dict:
    if _is_arbitration_card_action(value, draft):
        return build_disabled_arbitration_card(reason, draft)
    return build_disabled_draft_card(reason, draft)


def _is_arbitration_card_action(value: dict, draft: ConfigDraft | None) -> bool:
    if value.get("card_kind") == ARBITRATION_CARD_KIND:
        return True
    if not draft:
        return False
    return draft.metadata.get("card_kind") == ARBITRATION_CARD_KIND or bool(
        draft.metadata.get(ARBITRATION_CARD_METADATA_KEY)
    )


def _message_chat_id(data) -> str:
    return str(getattr(data.event.message, "chat_id", "") or "")


def _is_group_message(data) -> bool:
    chat_type = str(getattr(data.event.message, "chat_type", "") or "").lower()
    return chat_type == "group"


def _binding_target_from_message(data) -> tuple[str, str, str]:
    if _is_group_message(data):
        chat_id = _message_chat_id(data)
        if not chat_id:
            raise ValueError("无法识别群聊 chat_id")
        return "chat_id", chat_id, "group"
    open_id = _extract_sender_open_id(data)
    if not open_id:
        raise ValueError("无法识别用户 open_id")
    return "open_id", open_id, "p2p"


def _is_allowed_message(data, binding_store: FeishuBindingStore) -> bool:
    binding = binding_store.get()
    if not binding:
        return False
    operator_open_id = _extract_sender_open_id(data)
    if not operator_open_id or binding.bound_by_open_id != operator_open_id:
        return False
    if binding.receive_id_type == "chat_id":
        return _message_chat_id(data) == binding.receive_id
    return binding.receive_id == operator_open_id


def _maybe_prompt_bind(data) -> None:
    if not _is_group_message(data):
        _send_to_sender(data, "请先发送 /bind 完成绑定，后续通知会发到这个私聊。")


def _send_to_sender(data, text: str) -> bool:
    open_id = _extract_sender_open_id(data)
    if not open_id:
        return False
    return send_text_to("open_id", open_id, text)


def _is_bind_command(text: str) -> bool:
    return text.strip().lower() == "/bind"


def _is_unbind_command(text: str) -> bool:
    return text.strip().lower() == "/unbind"


def _command_key(text: str) -> str:
    clean = text.strip().lower()
    if clean in {"/help", "/status", "/run", "/restart", "/bind", "/unbind"}:
        return clean
    if clean == "/search":
        return "/search"
    if clean == "/model":
        return "/model"
    if clean.startswith("/model "):
        rest = clean[len("/model") :].lstrip()
        action = rest.split(maxsplit=1)[0].lower() if rest else ""
        if action == "status":
            return "/model status"
        return "/model"
    if not clean.startswith("/search "):
        return clean.split(maxsplit=1)[0] if clean else ""
    rest = clean[len("/search") :].lstrip()
    action = rest.split(maxsplit=1)[0].lower() if rest else ""
    if action in {"list", "add", "remove", "price", "clear"}:
        return f"/search {action}"
    return "/search unknown"


def _search_command_argument(text: str, prefix: str) -> str:
    return text.strip()[len(prefix) :].strip()


def _format_keyword_result(result: search_keywords.KeywordOperationResult) -> str:
    prefix = "OK" if result.success else "WARN"
    return f"{prefix}: {result.message}\n\n{_format_search_keywords(result.rules or result.keywords)}"


def _format_search_keywords(keywords: list) -> str:
    if not keywords:
        return "Search keywords: none"
    lines = ["Search keywords:"]
    lines.extend(f"{i}. {_format_search_keyword_entry(keyword)}" for i, keyword in enumerate(keywords, 1))
    return "\n".join(lines)


def _format_search_keyword_entry(keyword) -> str:
    if hasattr(keyword, "keyword"):
        max_price = getattr(keyword, "max_price", None)
        if max_price is not None:
            return f"{keyword.keyword} (max_price: {max_price:g})"
        return keyword.keyword
    return str(keyword)


def _search_usage_text() -> str:
    return (
        "Search keyword commands:\n"
        "- /search\n"
        "- /search list\n"
        "- /search add <keyword> [-price <price>]\n"
        "- /search remove <keyword>\n"
        "- /search price <keyword> <price|clear>\n"
        "- /search clear confirm"
    )


def _model_management_card(snapshot=None, form_state: dict | None = None) -> dict:
    state = llm_routing.model_card_state(snapshot)
    return build_model_management_card(state, form_state=form_state or default_model_form_state(state))


def _extract_form_state(value: dict) -> dict[str, str]:
    """Extract form field values suitable for pre-populating a new card."""
    result: dict[str, str] = {}
    for key in ("target", "connection", "model_id", "temperature"):
        v = _model_card_optional(value, key)
        if v:
            result[key] = v
    return result


def _model_form_state_from_snapshot(snapshot: llm_routing.RoutingSnapshot, target: str) -> dict[str, str]:
    if target == "default":
        return default_model_form_state(llm_routing.model_card_state(snapshot))
    resolved = snapshot.resolve(target)
    result = {
        "target": target,
        "connection": resolved.connection,
        "model_id": resolved.model_id,
    }
    if resolved.temperature is not None:
        result["temperature"] = str(resolved.temperature)
    return result


def _apply_model_card_action(action: str, value: dict) -> ModelCardUpdateResult:
    target = _model_card_target(value)
    if action == "model_apply_connection_model":
        connection = _model_card_optional(value, "connection")
        model_id = _model_card_optional(value, "model_id")
        if not connection:
            raise ValueError("请选择 connection 后再应用")
        if not model_id:
            raise ValueError("请输入 model_id 后再应用")
        if target == "default":
            snapshot = llm_routing.use_default_connection_model(connection, model_id)
        else:
            snapshot = llm_routing.use_agent_model(target, model_id, connection=connection)
        return ModelCardUpdateResult(snapshot, f"已更新 {_model_route_label(snapshot, target)}，下一次 LLM 调用生效。")
    if action == "model_apply_model":
        model_id = _model_card_required(value, "model_id", "请输入 model_id 后再应用")
        if target == "default":
            snapshot = llm_routing.use_default_model(model_id)
        else:
            snapshot = llm_routing.use_agent_model(target, model_id)
        return ModelCardUpdateResult(snapshot, f"已更新 {_model_route_label(snapshot, target)}，下一次 LLM 调用生效。")
    if action == "model_set_temperature":
        temperature = _model_card_temperature(value)
        if target == "default":
            snapshot = llm_routing.set_default_temperature(temperature)
        else:
            snapshot = llm_routing.set_agent_temperature(target, temperature)
        return ModelCardUpdateResult(snapshot, f"已更新 {_model_temperature_label(target, temperature)}，下一次 LLM 调用生效。")
    if action == "model_reset_agent":
        if target == "default":
            raise ValueError("默认配置不能 reset，请直接应用新的 connection/model_id")
        snapshot = llm_routing.reset_agent(target)
        return ModelCardUpdateResult(snapshot, f"已重置 {_model_route_label(snapshot, target)}，下一次 LLM 调用生效。")
    raise ValueError(f"未知操作：{action}")


def _model_route_label(snapshot: llm_routing.RoutingSnapshot, target: str) -> str:
    if target == "default":
        defaults = snapshot.raw.get("defaults", {})
        if not isinstance(defaults, dict):
            return "default"
        return f"default: {defaults.get('connection')}/{defaults.get('model_id')}"
    resolved = snapshot.resolve(target)
    return f"{target}: {resolved.connection}/{resolved.model_id}"


def _model_temperature_label(target: str, temperature: float) -> str:
    return f"{target} temperature={temperature:g}"


def _model_test_config_from_card(value: dict) -> ResolvedLLMConfig:
    connection = _model_card_optional(value, "connection")
    model_id = _model_card_optional(value, "model_id")
    if connection and model_id:
        return llm_routing.test_config_for_connection(connection, model_id)
    target = _model_card_target(value)
    if target != "default":
        return llm_routing.test_config_for_agent(target)
    state = llm_routing.model_card_state()
    defaults = state.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("默认 LLM 配置不可用")
    return llm_routing.test_config_for_connection(str(defaults.get("connection") or ""), str(defaults.get("model_id") or ""))


def _model_card_target(value: dict) -> str:
    target = _model_card_optional(value, "target") or "default"
    if target == "default" or target in AGENTS:
        return target
    raise ValueError("作用范围必须是 default/filter/arbiter/draft")


def _model_card_temperature(value: dict) -> float:
    raw = _model_card_required(value, "temperature", "请输入 temperature")
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError("temperature 必须是数字") from e


def _model_card_required(value: dict, key: str, message: str) -> str:
    clean = _model_card_optional(value, key)
    if not clean:
        raise ValueError(message)
    return clean


def _model_card_optional(value: dict, key: str) -> str:
    raw = _model_card_raw_value(value, key)
    if raw is None:
        return ""
    return str(raw).strip()


def _model_card_raw_value(value: dict, key: str):
    if key in value:
        return _normalize_card_form_value(value.get(key))
    for group_key in ("form_values", "form_value", "form", "input_values"):
        group = value.get(group_key)
        if isinstance(group, dict) and key in group:
            return _normalize_card_form_value(group.get(key))
    return None


def _normalize_card_form_value(raw):
    if isinstance(raw, dict):
        for key in ("value", "text", "content"):
            if raw.get(key) not in (None, ""):
                return raw.get(key)
        if raw.get("option") is not None:
            return _normalize_card_form_value(raw.get("option"))
    if isinstance(raw, list):
        if not raw:
            return ""
        return _normalize_card_form_value(raw[0])
    return raw


def _run_model_test(llm_config: ResolvedLLMConfig) -> str:
    try:
        kwargs = build_chat_completion_kwargs(
            llm_config,
            messages=[
                {"role": "system", "content": "Return a JSON object."},
                {"role": "user", "content": 'Return exactly {"ok": true}.'},
            ],
        )
        kwargs["timeout"] = min(llm_config.timeout_seconds, 30)
        response = get_client_for_config(llm_config).chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        preview = content[:200]
        return f"OK: {llm_config.connection}/{llm_config.model_id} 测试成功。\n{preview}"
    except RETRYABLE_OPENAI_ERRORS as e:
        return f"WARN: 测试失败（{error_summary('可重试/网络类问题', e)}）"
    except NON_RETRYABLE_OPENAI_ERRORS as e:
        return f"WARN: 测试失败（{error_summary('配置或请求不可重试问题', e)}）"
    except GENERAL_OPENAI_ERRORS as e:
        return f"WARN: 测试失败（{error_summary('OpenAI SDK/API 错误', e)}）"
    except Exception as e:
        return f"WARN: 测试失败（{error_summary('非 OpenAI SDK 异常', e)}）"


def _model_usage_text() -> str:
    return "发送 /model 打开 LLM 模型路由管理卡片。"


def _card_response(message: str, card: dict | None = None) -> object | None:
    try:
        P2CardActionTriggerResponse = get_card_action_response_model()
        payload: dict[str, Any] = {"toast": {"type": "success", "content": message}}
        if card:
            payload["card"] = {"type": "raw", "data": card}
        return P2CardActionTriggerResponse(payload)
    except Exception:
        return None
