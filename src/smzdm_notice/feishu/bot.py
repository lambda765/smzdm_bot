"""飞书自建应用机器人长连接交互层。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from smzdm_notice.core import config
from smzdm_notice.feishu.binding import FeishuBindingStore
from smzdm_notice.feishu.card_payload import (
    apply_model_form_cache_patch,
    build_model_card_form_patch,
    extract_card_action_value,
    extract_message_text,
    model_card_field_from_callback,
    model_card_form_cache_key,
    optional_card_field_value,
    optional_model_card_field_value,
    replace_model_form_cache_values,
    strip_bot_mention,
    trim_model_card_form_cache,
)
from smzdm_notice.feishu.commands import find_command_spec, help_markdown
from smzdm_notice.feishu.model_actions import (
    apply_model_route_card_action,
    model_card_target,
    resolve_model_test_config,
)
from smzdm_notice.feishu.model_cards import build_model_management_card, default_model_form_state
from smzdm_notice.feishu.notifier import (
    ARBITRATION_CARD_KIND,
    ARBITRATION_CARD_METADATA_KEY,
    NOT_WORTH_REASON_FIELD,
    build_disabled_arbitration_card,
    build_disabled_draft_card,
    build_draft_failure_card,
    build_draft_handoff_card,
    build_draft_noop_card,
    build_draft_preview_card,
    disable_draft_card,
    finish_streaming_draft_card,
    reply_card,
    reply_text,
    send_draft_preview,
    send_draft_processing,
    send_help,
    send_text,
    send_text_to,
    start_streaming_draft_processing,
    update_card_message,
    update_deal_card_feedback_state,
    update_draft_preview,
    update_rebased_draft_preview,
    update_streaming_draft_content,
)
from smzdm_notice.feishu.sdk import (
    get_card_action_response_model,
    get_lark_client,
    get_lark_module,
    get_message_reaction_models,
)
from smzdm_notice.feishu.search_actions import (
    handle_search_card_action,
    handle_search_command,
)
from smzdm_notice.feishu.search_actions import (
    search_usage_text as _search_usage_text,
)
from smzdm_notice.llm import routing as llm_routing
from smzdm_notice.llm.clients import get_client_for_config
from smzdm_notice.llm.errors import (
    GENERAL_OPENAI_ERRORS,
    NON_RETRYABLE_OPENAI_ERRORS,
    RETRYABLE_OPENAI_ERRORS,
    error_summary,
)
from smzdm_notice.llm.routing import (
    AGENTS,
    LLMRoutingError,
    ResolvedLLMConfig,
    build_chat_completion_kwargs,
)
from smzdm_notice.preferences.builder import (
    build_deal_action_draft_outcome,
    build_message_draft_outcome,
    build_rebase_draft_outcome,
    build_revision_draft_outcome,
)
from smzdm_notice.preferences.models import ConfigDraft, DraftBuildOutcome, DraftStreamEvent
from smzdm_notice.preferences.store import DraftStore

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
    record_memory_feedback: Callable[..., str] | None = None


@dataclass
class CardActionDispatchResult:
    """卡片 action 分发结果，稍后会包装成 Feishu callback 响应。"""

    message: str
    response_card: dict | None = None


@dataclass
class DraftProgressCard:
    """草案生成期间使用的进度卡；仅流式卡片带刷新线程。"""

    message_id: str = ""
    card_id: str = ""
    stop_event: threading.Event | None = None
    thread: threading.Thread | None = None
    sequence: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    stage: str = ""
    wake_event: threading.Event | None = None
    buffer_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    attempt_status: str = ""
    reasoning_text: str = ""
    content_text: str = ""
    reasoning_seen: bool = False
    reasoning_truncated: bool = False
    content_truncated: bool = False

    @property
    def streaming(self) -> bool:
        return bool(self.card_id)


MODEL_CARD_FORM_STATE_LIMIT = 64
DEAL_REASON_FORM_STATE_LIMIT = 256
DRAFT_STREAMING_FINALIZE_LOCK_TIMEOUT_SECONDS = 1.0
DRAFT_STREAMING_UPDATE_INTERVAL_SECONDS = 0.5
DRAFT_STREAMING_DISPLAY_MAX_BYTES = 8 * 1024
DRAFT_STREAMING_OMITTED_TEXT = "较早内容已省略\n\n"


def _append_draft_stream_text(current: str, delta: str) -> tuple[str, bool]:
    clean_delta = "".join(char for char in delta if char in "\n\t" or ord(char) >= 32)
    combined = current + clean_delta
    encoded = combined.encode("utf-8")
    if len(encoded) <= DRAFT_STREAMING_DISPLAY_MAX_BYTES:
        return combined, False
    tail = encoded[-DRAFT_STREAMING_DISPLAY_MAX_BYTES :]
    while tail:
        try:
            return tail.decode("utf-8"), True
        except UnicodeDecodeError:
            tail = tail[1:]
    return "", True


class FeishuInteractiveBot:
    """基于 lark-oapi 长连接接收消息和卡片事件。"""

    def __init__(self, runtime: BotRuntime, deduper: MessageDeduper | None = None) -> None:
        self.runtime = runtime
        self.deduper = deduper or MessageDeduper()
        self.card_action_deduper = MessageDeduper()
        self._model_card_form_state: dict[str, dict[str, Any]] = {}
        self._model_card_form_lock = threading.RLock()
        self._deal_reason_form_state: dict[str, str] = {}
        self._deal_reason_form_lock = threading.RLock()
        self._draft_refreshing: set[str] = set()
        self._draft_refresh_lock = threading.RLock()

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
        """接收 Feishu 消息事件，并把耗时工作交给后台线程。"""
        message_id = ""
        try:
            message = data.event.message
            text = extract_message_text(getattr(message, "content", ""))
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
        clean = strip_bot_mention(text)
        if self._handle_binding_state(clean, data, reply_to_message_id):
            return
        if self._handle_parent_draft(clean, parent_id, reply_to_message_id):
            return
        if self._handle_slash_command(clean, reply_to_message_id):
            return
        self._handle_new_draft(clean, reply_to_message_id)

    def _handle_binding_state(self, text: str, data, reply_to_message_id: str) -> bool:
        command = text.lower()
        if command == "/bind":
            self._bind_current_conversation(data, reply_to_message_id)
            return True
        # 私聊首次消息沿用自动绑定行为，消息本身不再继续生成草案。
        if not self.runtime.binding_store.get() and not _is_group_message(data):
            self._bind_current_conversation(data, reply_to_message_id)
            return True
        if command == "/unbind":
            self._unbind_current_operator(data, reply_to_message_id)
            return True
        if not _is_allowed_message(data, self.runtime.binding_store):
            self._maybe_prompt_bind(data, reply_to_message_id)
            return True
        return False

    def _handle_parent_draft(self, text: str, parent_id: str, reply_to_message_id: str) -> bool:
        if not parent_id:
            return False
        original_draft = self.runtime.draft_store.get_any_by_preview_message_id(parent_id)
        if not original_draft:
            # 未知草案上的斜杠命令仍按普通命令执行。
            if text.startswith("/"):
                return False
            self._reply_text(reply_to_message_id, "该回复引用的预览不存在或已失效，请发新消息重新生成。")
            return True
        if original_draft.status != "pending":
            disable_draft_card(parent_id, "该预览已失效，请以最新预览为准", original_draft)
            self._reply_text(reply_to_message_id, "该预览已失效，请以最新预览为准。")
            return True
        if original_draft.is_expired:
            self.runtime.draft_store.cancel(original_draft.draft_id, reason="expired")
            if original_draft.preview_message_id:
                disable_draft_card(
                    original_draft.preview_message_id,
                    "草案已超过 24 小时自动失效",
                    original_draft,
                )
            self._reply_text(reply_to_message_id, "该预览已超过 24 小时自动失效，请发新消息重新生成。")
            return True
        self._handle_draft_revision(text, original_draft, reply_to_message_id)
        return True

    def _handle_new_draft(self, text: str, reply_to_message_id: str) -> None:
        processing = self._start_draft_processing(reply_to_message_id, "正在理解偏好/库存修改")
        try:
            stream_callback = self._draft_stream_callback(processing)
            if stream_callback:
                outcome = build_message_draft_outcome(
                    text,
                    self.runtime.draft_store,
                    stream_callback=stream_callback,
                )
            else:
                outcome = build_message_draft_outcome(text, self.runtime.draft_store)
            draft = outcome.draft
        except Exception as e:
            self._stop_draft_processing(processing)
            logger.error(f"配置草案生成失败: {e}", exc_info=True)
            self._finish_draft_processing_failure(
                processing,
                "处理消息失败，没能生成配置修改预览。",
                reply_to_message_id,
                INTERNAL_ERROR_MESSAGE,
            )
            return
        self._stop_draft_processing(processing)
        if not draft:
            if isinstance(outcome, DraftBuildOutcome) and outcome.status == "noop":
                message = outcome.message or "当前配置已经覆盖这项需求，无需重复修改。"
                self._finish_draft_processing_noop(processing, message, reply_to_message_id)
                return
            self._finish_draft_processing_failure(
                processing,
                "草案生成失败：没能理解这次偏好/库存修改。",
                reply_to_message_id,
                "草案生成失败：没能理解这次偏好/库存修改，请换一种更明确的说法重试。",
            )
            return
        self._send_and_store_draft_preview(draft, reply_to_message_id, processing)

    def _handle_draft_revision(
        self,
        text: str,
        original_draft: ConfigDraft,
        reply_to_message_id: str = "",
    ) -> None:
        processing = self._start_draft_processing(reply_to_message_id, "正在根据修改意见生成新预览")
        try:
            stream_callback = self._draft_stream_callback(processing)
            if stream_callback:
                outcome = build_revision_draft_outcome(
                    text,
                    original_draft,
                    self.runtime.draft_store,
                    stream_callback=stream_callback,
                )
            else:
                outcome = build_revision_draft_outcome(text, original_draft, self.runtime.draft_store)
            revised = outcome.draft
        except Exception as e:
            self._stop_draft_processing(processing)
            logger.error(f"配置草案修订失败: {e}", exc_info=True)
            self._finish_draft_processing_failure(
                processing,
                "处理修改意见失败，没能生成新的配置修改预览。",
                reply_to_message_id,
                INTERNAL_ERROR_MESSAGE,
            )
            return
        self._stop_draft_processing(processing)
        if not revised:
            if outcome.status == "noop":
                message = outcome.message or "当前草案已经符合修改意见，无需生成新预览。"
                self._finish_draft_processing_noop(processing, message, reply_to_message_id)
                return
            self._finish_draft_processing_failure(
                processing,
                "没能理解修改意见。",
                reply_to_message_id,
                "没能理解修改意见，请换一种说法重试，或发新消息重新生成。",
            )
            return
        if self._send_and_store_draft_preview(
            revised,
            reply_to_message_id,
            processing,
        ):
            self.runtime.draft_store.cancel(original_draft.draft_id, reason="superseded_by_revision")
            if original_draft.preview_message_id:
                disable_draft_card(original_draft.preview_message_id, "已生成新的修改预览", original_draft)
            return
        self._reply_text(reply_to_message_id, "修改后的预览发送失败，原草案仍保留，可继续回复原预览。")

    def _send_and_store_draft_preview(
        self,
        draft: ConfigDraft,
        reply_to_message_id: str = "",
        processing: DraftProgressCard | None = None,
    ) -> bool:
        processing = processing or DraftProgressCard()
        preview_sent = self._replace_processing_card_with_preview(draft, processing)
        fallback_preview_sent = False
        if not preview_sent:
            fallback_preview_sent = send_draft_preview(draft, reply_to_message_id=reply_to_message_id)
            preview_sent = fallback_preview_sent
        if fallback_preview_sent and processing.streaming:
            self._schedule_streaming_cleanup(processing, build_draft_handoff_card())
        elif not preview_sent and processing.streaming:
            preview_sent = self._finish_streaming_preview(draft, processing)
        if not preview_sent:
            self.runtime.draft_store.cancel(draft.draft_id, reason="preview_send_failed")
            return False
        if draft.preview_message_id:
            self.runtime.draft_store.update(draft)
        return True

    def _replace_processing_card_with_preview(
        self,
        draft: ConfigDraft,
        processing: DraftProgressCard,
    ) -> bool:
        if not processing.message_id:
            return False
        if processing.streaming:
            return self._finish_streaming_preview(
                draft,
                processing,
                lock_timeout=DRAFT_STREAMING_FINALIZE_LOCK_TIMEOUT_SECONDS,
            )
        preview_sent = update_draft_preview(processing.message_id, draft)
        if not preview_sent:
            logger.warning(f"处理中卡片更新为预览失败，回退为发送新预览: {processing.message_id}")
        return preview_sent

    def _finish_streaming_preview(
        self,
        draft: ConfigDraft,
        processing: DraftProgressCard,
        *,
        lock_timeout: float | None = None,
    ) -> bool:
        preview_sent = (
            self._finish_streaming_card(
                processing,
                build_draft_preview_card(draft),
                lock_timeout=lock_timeout,
            )
            is True
        )
        if preview_sent:
            draft.preview_message_id = processing.message_id
        return preview_sent

    def _start_draft_processing(self, reply_to_message_id: str, stage: str) -> DraftProgressCard:
        """优先发送流式进度卡；失败时发送不定时刷新的普通进度卡。"""
        if not reply_to_message_id:
            return DraftProgressCard()
        streaming_handle = start_streaming_draft_processing(stage, reply_to_message_id)
        if streaming_handle:
            message_id, card_id = streaming_handle
            progress = DraftProgressCard(
                message_id=message_id,
                card_id=card_id,
                stop_event=threading.Event(),
                stage=stage,
                wake_event=threading.Event(),
            )
            progress.thread = threading.Thread(
                target=self._run_draft_processing_progress,
                args=(progress,),
                name="feishu-draft-progress-worker",
                daemon=True,
            )
            progress.thread.start()
            return progress

        message_id = send_draft_processing(stage, reply_to_message_id=reply_to_message_id)
        if not message_id:
            return DraftProgressCard()
        return DraftProgressCard(message_id=message_id)

    def _run_draft_processing_progress(
        self,
        processing: DraftProgressCard,
    ) -> None:
        """合并模型增量并按固定频率刷新 CardKit 文本。"""
        stop_event = processing.stop_event
        wake_event = processing.wake_event
        if stop_event is None or wake_event is None or not processing.streaming:
            return
        last_content = ""
        last_updated_at = 0.0
        while True:
            wake_event.wait()
            wake_event.clear()
            stopping = stop_event.is_set()
            if not stopping and last_updated_at:
                remaining = DRAFT_STREAMING_UPDATE_INTERVAL_SECONDS - (time.monotonic() - last_updated_at)
                if remaining > 0 and stop_event.wait(remaining):
                    stopping = True
            content = self._render_draft_stream_content(processing)
            if content and content != last_content:
                self._update_streaming_progress(processing, content, allow_stopped=stopping)
                last_content = content
                last_updated_at = time.monotonic()
            if stopping:
                return

    def _draft_stream_callback(self, processing: DraftProgressCard) -> Callable[[DraftStreamEvent], None] | None:
        if not processing.streaming:
            return None

        def callback(event: DraftStreamEvent) -> None:
            self._record_draft_stream_event(processing, event)

        return callback

    def _record_draft_stream_event(self, processing: DraftProgressCard, event: DraftStreamEvent) -> None:
        stop_event = processing.stop_event
        wake_event = processing.wake_event
        if stop_event is None or wake_event is None or stop_event.is_set():
            return
        with processing.buffer_lock:
            if event.kind == "attempt_start":
                processing.attempt_status = event.text
                processing.reasoning_text = ""
                processing.content_text = ""
                processing.reasoning_seen = False
                processing.reasoning_truncated = False
                processing.content_truncated = False
            elif event.kind == "reasoning_delta":
                processing.reasoning_seen = True
                processing.content_text = ""
                processing.content_truncated = False
                processing.reasoning_text, truncated = _append_draft_stream_text(
                    processing.reasoning_text,
                    event.text,
                )
                processing.reasoning_truncated = processing.reasoning_truncated or truncated
            elif event.kind == "content_delta" and not processing.reasoning_seen:
                processing.content_text, truncated = _append_draft_stream_text(
                    processing.content_text,
                    event.text,
                )
                processing.content_truncated = processing.content_truncated or truncated
        wake_event.set()

    def _render_draft_stream_content(self, processing: DraftProgressCard) -> str:
        with processing.buffer_lock:
            lines = [processing.stage]
            if processing.attempt_status:
                lines.append(processing.attempt_status)
            if processing.reasoning_seen and processing.reasoning_text:
                text = processing.reasoning_text
                if processing.reasoning_truncated:
                    text = DRAFT_STREAMING_OMITTED_TEXT + text
                lines.extend(["**模型思考**", text])
            elif processing.content_text:
                text = processing.content_text
                if processing.content_truncated:
                    text = DRAFT_STREAMING_OMITTED_TEXT + text
                lines.extend(["**模型原始输出**", text])
            return "\n\n".join(line for line in lines if line)

    def _update_streaming_progress(
        self,
        processing: DraftProgressCard,
        content: str,
        *,
        allow_stopped: bool = False,
    ) -> bool:
        with processing.lock:
            if not allow_stopped and processing.stop_event and processing.stop_event.is_set():
                return False
            processing.sequence += 1
            return update_streaming_draft_content(processing.card_id, content, processing.sequence)

    def _finish_streaming_card(
        self,
        processing: DraftProgressCard,
        card: dict,
        *,
        lock_timeout: float | None = None,
    ) -> bool | None:
        if lock_timeout is None:
            processing.lock.acquire()
            acquired = True
        else:
            acquired = processing.lock.acquire(timeout=lock_timeout)
        if not acquired:
            logger.warning(f"流式进度更新仍在执行，先回退发送新卡片: {processing.card_id}")
            return None
        try:
            close_sequence = processing.sequence + 1
            update_sequence = close_sequence + 1
            processing.sequence = update_sequence
            return finish_streaming_draft_card(
                processing.card_id,
                card,
                close_sequence,
                update_sequence,
            )
        finally:
            processing.lock.release()

    def _schedule_streaming_cleanup(self, processing: DraftProgressCard, card: dict) -> None:
        """新消息已承接结果后，在后台关闭并替换旧流式卡片。"""
        def cleanup() -> None:
            if self._finish_streaming_card(processing, card) is not True:
                logger.warning(f"旧流式卡片清理失败: {processing.card_id}")

        threading.Thread(
            target=cleanup,
            name="feishu-streaming-card-cleanup",
            daemon=True,
        ).start()

    def _stop_draft_processing(self, processing: DraftProgressCard) -> None:
        """停止流式刷新线程；普通静态进度卡无需停止。"""
        if processing.stop_event:
            processing.stop_event.set()
        if processing.wake_event:
            processing.wake_event.set()
        if processing.thread:
            processing.thread.join(timeout=1)

    def _finish_draft_processing_failure(
        self,
        processing: DraftProgressCard,
        card_reason: str,
        reply_to_message_id: str,
        fallback_text: str,
    ) -> None:
        self._finish_draft_processing_with_card(
            processing,
            build_draft_failure_card(card_reason),
            reply_to_message_id,
            fallback_text,
        )

    def _finish_draft_processing_noop(
        self,
        processing: DraftProgressCard,
        message: str,
        reply_to_message_id: str,
    ) -> None:
        self._finish_draft_processing_with_card(
            processing,
            build_draft_noop_card(message),
            reply_to_message_id,
            message,
        )

    def _finish_draft_processing_with_card(
        self,
        processing: DraftProgressCard,
        card: dict,
        reply_to_message_id: str,
        fallback_text: str,
    ) -> None:
        """将草案进度卡收尾为最终卡片，必要时回退为新消息或文本。"""
        if processing.message_id:
            if processing.streaming:
                finish_result = self._finish_streaming_card(
                    processing,
                    card,
                    lock_timeout=DRAFT_STREAMING_FINALIZE_LOCK_TIMEOUT_SECONDS,
                )
                if finish_result is True:
                    return
                if reply_to_message_id and reply_card(reply_to_message_id, card):
                    self._schedule_streaming_cleanup(processing, card)
                    return
                if self._finish_streaming_card(processing, card) is True:
                    return
            if not processing.streaming and update_card_message(processing.message_id, card):
                return
            if reply_to_message_id and reply_card(reply_to_message_id, card):
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
        """鉴权并分发一次 Feishu 卡片 callback。

        模型表单组件可能发出不含 action 的 callback；这类 callback 只更新
        或重绘进程内表单缓存。
        """
        reply_to_message_id = _card_open_message_id(data)
        try:
            value = extract_card_action_value(data)
            action = str(value.get("action") or "")
            operator = _extract_operator(data)
            card_event_id = _card_event_id(data)
            logger.info(
                f"收到飞书卡片操作: action={action}, operator={operator}, "
                f"event_id={_mask_card_event_id(card_event_id)}"
            )
            if not action:
                return self._handle_card_form_callback(value, operator, reply_to_message_id)
            if not self.runtime.binding_store.is_bound_operator(operator):
                message = "只有当前绑定用户可以操作卡片"
                self._reply_text(reply_to_message_id, message)
                return _card_response(message)
            if card_event_id and not self.card_action_deduper.claim(card_event_id):
                message = "操作已处理，请勿重复提交"
                logger.info(
                    f"忽略重复飞书卡片操作: action={action}, "
                    f"event_id={_mask_card_event_id(card_event_id)}"
                )
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
            result = CardActionDispatchResult(message)
        return _card_response(result.message, result.response_card)

    def _handle_card_form_callback(self, value: dict, operator: str, message_id: str) -> object | None:
        field = model_card_field_from_callback(value)
        if field == NOT_WORTH_REASON_FIELD:
            if not self.runtime.binding_store.is_bound_operator(operator):
                logger.debug(f"忽略未授权不值理由输入变更: operator={operator}")
                return None
            self._remember_deal_reason_form_value(message_id, operator, value)
            return None
        if field not in {"target", "connection", "model_id", "temperature"}:
            logger.debug(f"忽略飞书卡片表单变更回调: operator={operator}, keys={sorted(value.keys())}")
            return None
        if not self.runtime.binding_store.is_bound_operator(operator):
            logger.debug(f"忽略未授权模型卡片表单变更: operator={operator}, field={field}")
            return None
        self._remember_model_card_form_value(message_id, operator, value)
        if field in {"target", "connection"}:
            return self._handle_model_form_change(message_id, operator)
        logger.debug(f"已缓存飞书模型卡片表单变更: operator={operator}, field={field}")
        return None

    def _dispatch_card_action(
        self,
        action: str,
        value: dict,
        operator: str,
        reply_to_message_id: str,
    ) -> CardActionDispatchResult:
        """将归一化后的卡片 action 路由到对应功能处理器。"""
        if action == "apply_draft":
            return self._apply_draft_card_action(value, operator, reply_to_message_id)
        if action == "cancel_draft":
            return self._cancel_draft_card_action(value, operator, reply_to_message_id)
        if action == "ignore_arbitration":
            return self._ignore_arbitration_card_action(value, operator, reply_to_message_id)
        if action in {"deal_good", "deal_not_worth", "deal_not_worth_reason"}:
            return self._handle_memory_feedback(action, value, reply_to_message_id, operator)
        if action in {"deal_ignore_category", "deal_stock_enough", "deal_follow"}:
            self._start_deal_action_worker(action, dict(value), reply_to_message_id)
            return CardActionDispatchResult("正在生成配置修改预览，请稍候。")
        if action in {"search_remove_keyword", "search_clear_price"}:
            message = self._handle_search_card_action(action, value)
            self._reply_text(reply_to_message_id, message)
            return CardActionDispatchResult(message)
        if action.startswith("model_"):
            return self._handle_model_card_action(action, value)
        message = f"未知操作：{action}"
        self._reply_text(reply_to_message_id, message)
        return CardActionDispatchResult(message)

    def _apply_draft_card_action(
        self,
        value: dict,
        operator: str,
        reply_to_message_id: str,
    ) -> CardActionDispatchResult:
        draft_id = str(value.get("draft_id") or "")
        draft = self.runtime.draft_store.get(draft_id)
        outcome = self.runtime.draft_store.apply(draft_id, operator=operator)
        if outcome.status == "needs_refresh" and draft:
            if not self._claim_draft_refresh(draft_id):
                message = "当前配置与预览存在冲突，正在刷新预览，请稍候。"
                return CardActionDispatchResult(message)
            message = "当前配置与预览存在冲突，正在刷新预览。"
            self._reply_text(reply_to_message_id, "🔄 " + message)
            self._start_draft_refresh_worker(draft, operator, reply_to_message_id)
            return CardActionDispatchResult(message)

        if outcome.ok:
            self._reply_text(reply_to_message_id, "✅ " + outcome.message)
            return CardActionDispatchResult(
                outcome.message,
                _build_disabled_card_for_action("已确认应用", outcome.draft or draft, value),
            )

        message = outcome.message
        if outcome.status in {"missing", "expired"} or _is_stale_draft(draft):
            message = "该预览已失效，请发新消息重新生成。"
        self._reply_text(reply_to_message_id, "⚠️ " + message)
        if outcome.status in {"missing", "expired", "rejected"} or _is_stale_draft(draft):
            return CardActionDispatchResult(
                message,
                _build_disabled_card_for_action("预览已失效", outcome.draft or draft, value),
            )
        return CardActionDispatchResult(message)

    def _claim_draft_refresh(self, draft_id: str) -> bool:
        with self._draft_refresh_lock:
            if draft_id in self._draft_refreshing:
                return False
            self._draft_refreshing.add(draft_id)
            return True

    def _release_draft_refresh(self, draft_id: str) -> None:
        with self._draft_refresh_lock:
            self._draft_refreshing.discard(draft_id)

    def _start_draft_refresh_worker(
        self,
        draft: ConfigDraft,
        operator: str,
        reply_to_message_id: str,
    ) -> None:
        try:
            thread = threading.Thread(
                target=self._run_draft_refresh,
                args=(draft, operator, reply_to_message_id),
                name="preference-draft-refresh-worker",
                daemon=True,
            )
            thread.start()
        except Exception:
            self._release_draft_refresh(draft.draft_id)
            raise

    def _run_draft_refresh(
        self,
        original: ConfigDraft,
        operator: str,
        reply_to_message_id: str,
    ) -> None:
        new_draft: ConfigDraft | None = None
        try:
            outcome = build_rebase_draft_outcome(original, self.runtime.draft_store)
            new_draft = outcome.draft

            current = self.runtime.draft_store.get(original.draft_id)
            if not current or current.status != "pending":
                if new_draft:
                    self.runtime.draft_store.cancel(
                        new_draft.draft_id,
                        operator="system",
                        reason="original_draft_not_pending",
                    )
                return

            if outcome.status == "noop":
                reason = outcome.message or "当前配置已覆盖原修改意图"
                if not original.preview_message_id or not disable_draft_card(
                    original.preview_message_id,
                    f"当前配置已覆盖：{reason}",
                    original,
                ):
                    self._reply_text(reply_to_message_id, "⚠️ 预览刷新失败，原草案仍可重试。")
                    return
                self.runtime.draft_store.cancel(
                    original.draft_id,
                    operator=operator,
                    reason="already_covered_after_rebase",
                )
                self._reply_text(reply_to_message_id, f"ℹ️ 当前配置已覆盖：{reason}")
                return

            if not new_draft:
                self._reply_text(
                    reply_to_message_id,
                    f"⚠️ 预览刷新失败，原草案仍可重试：{outcome.message or '模型未生成有效草案'}",
                )
                return

            if not original.preview_message_id or not update_rebased_draft_preview(
                original.preview_message_id,
                new_draft,
            ):
                self.runtime.draft_store.cancel(
                    new_draft.draft_id,
                    operator="system",
                    reason="preview_update_failed",
                )
                self._reply_text(reply_to_message_id, "⚠️ 预览刷新失败，原草案仍可重试。")
                return

            # 用户可能在 LLM 生成或卡片更新期间取消旧草案。更新后再检查一次，
            # 确保这种竞态不会留下一个用户没有确认过的新 pending 草案。
            current = self.runtime.draft_store.get(original.draft_id)
            if not current or current.status != "pending":
                self.runtime.draft_store.cancel(
                    new_draft.draft_id,
                    operator="system",
                    reason="original_draft_not_pending",
                )
                if original.preview_message_id:
                    disable_draft_card(original.preview_message_id, "原草案已取消，刷新结果已丢弃", original)
                return

            self.runtime.draft_store.update(new_draft)
            self.runtime.draft_store.cancel(
                original.draft_id,
                operator=operator,
                reason="superseded_by_rebase",
            )
            self._reply_text(reply_to_message_id, "✅ 预览已刷新，请检查后再次点击采纳。")
        except Exception as e:
            logger.error(f"配置草案冲突刷新失败: {e}", exc_info=True)
            if new_draft and new_draft.status == "pending":
                self.runtime.draft_store.cancel(
                    new_draft.draft_id,
                    operator="system",
                    reason="rebase_failed",
                )
            self._reply_text(reply_to_message_id, "⚠️ 预览刷新失败，原草案仍可重试。")
        finally:
            self._release_draft_refresh(original.draft_id)

    def _cancel_draft_card_action(
        self,
        value: dict,
        operator: str,
        reply_to_message_id: str,
    ) -> CardActionDispatchResult:
        draft_id = str(value.get("draft_id") or "")
        draft = self.runtime.draft_store.get(draft_id)
        if draft and draft.status == "pending":
            draft = self.runtime.draft_store.cancel(draft_id, operator=operator, reason="user_cancelled")
            message = "已取消草案"
            reason = "已取消"
        else:
            message = "该预览已失效，请发新消息重新生成。"
            reason = "预览已失效"
        self._reply_text(reply_to_message_id, message)
        return CardActionDispatchResult(message, _build_disabled_card_for_action(reason, draft, value))

    def _ignore_arbitration_card_action(
        self,
        value: dict,
        operator: str,
        reply_to_message_id: str,
    ) -> CardActionDispatchResult:
        draft_id = str(value.get("draft_id") or "")
        draft = self.runtime.draft_store.get(draft_id) if draft_id else None
        if draft and draft.status == "pending":
            self.runtime.draft_store.cancel(draft_id, operator=operator, reason="user_cancelled")
            message = "已忽略本次仲裁建议"
            reason = "已忽略"
        else:
            message = "该预览已失效，请发新消息重新生成。"
            reason = "预览已失效"
        self._reply_text(reply_to_message_id, message)
        return CardActionDispatchResult(message, _build_disabled_card_for_action(reason, draft, value))

    def _handle_memory_feedback(
        self,
        action: str,
        value: dict,
        reply_to_message_id: str,
        operator: str = "",
    ) -> CardActionDispatchResult:
        """处理好价/不值反馈，记录到 DealMemory。"""
        article_id = str(value.get("article_id") or "")
        if not article_id:
            message = "无法识别商品信息"
            self._reply_text(reply_to_message_id, message)
            return CardActionDispatchResult(message)

        feedback_action = "deal_not_worth" if action == "deal_not_worth_reason" else action
        reason: str | None = None
        if action == "deal_not_worth_reason":
            reason = self._resolve_deal_reason_form_value(reply_to_message_id, operator, article_id, value)
            if not reason:
                message = "未填写不值理由，已保留不值反馈"
                return CardActionDispatchResult(message)
        result = self._record_memory_feedback(article_id, feedback_action, reason)
        return self._memory_feedback_result(
            result,
            article_id,
            feedback_action,
            reason,
            reply_to_message_id,
            operator,
        )

    def _record_memory_feedback(self, article_id: str, action: str, reason: str | None) -> str:
        recorder = self.runtime.record_memory_feedback
        if recorder is None:
            return "not_found"
        if reason is None:
            return recorder(article_id, action)
        return recorder(article_id, action, reason)

    def _memory_feedback_result(
        self,
        result: str,
        article_id: str,
        feedback_action: str,
        reason: str | None,
        message_id: str,
        operator: str,
    ) -> CardActionDispatchResult:
        label = "好价" if feedback_action == "deal_good" else "不值"
        messages = {
            "recorded": f"已标记为{label}，偏好将用于后续推荐",
            "updated": f"已更新为{label}，偏好将用于后续推荐",
            "reason_updated": "已保存不值理由",
            "cancelled": "已取消反馈",
            "invalid_action": "未知反馈操作",
        }
        message = messages.get(result, "反馈记录失败，该商品可能已过期或记忆功能未启用。")
        selected_by_result = {
            "recorded": feedback_action,
            "updated": feedback_action,
            "reason_updated": "deal_not_worth",
            "cancelled": "",
        }
        if result not in selected_by_result:
            return CardActionDispatchResult(message)
        updated_card = update_deal_card_feedback_state(
            message_id,
            article_id,
            selected=selected_by_result[result],
            reason=reason or "",
        )
        if result in {"reason_updated", "cancelled"}:
            self._forget_deal_reason_form_value(message_id, operator, article_id)
        return CardActionDispatchResult(message, updated_card)

    def _remember_deal_reason_form_value(self, message_id: str, operator: str, value: dict) -> None:
        article_id = str(value.get("article_id") or "").strip()
        if not article_id:
            logger.debug("忽略缺少 article_id 的不值理由输入变更")
            return
        found, reason = _deal_reason_from_card_value(value)
        if not found:
            return
        key = _deal_reason_form_state_key(message_id, operator, article_id)
        if not key:
            return
        with self._deal_reason_form_lock:
            self._deal_reason_form_state.pop(key, None)
            if reason:
                self._deal_reason_form_state[key] = reason
                _trim_deal_reason_form_state(self._deal_reason_form_state)
        logger.debug(
            f"缓存不值理由输入: operator={operator}, article_id={article_id}, "
            f"has_reason={bool(reason)}"
        )

    def _resolve_deal_reason_form_value(
        self,
        message_id: str,
        operator: str,
        article_id: str,
        value: dict,
    ) -> str:
        found, reason = _deal_reason_from_card_value(value)
        if found:
            return reason
        key = _deal_reason_form_state_key(message_id, operator, article_id)
        if not key:
            return ""
        with self._deal_reason_form_lock:
            return self._deal_reason_form_state.get(key, "")

    def _forget_deal_reason_form_value(self, message_id: str, operator: str, article_id: str) -> None:
        key = _deal_reason_form_state_key(message_id, operator, article_id)
        if not key:
            return
        with self._deal_reason_form_lock:
            self._deal_reason_form_state.pop(key, None)

    def _start_deal_action_worker(self, action: str, value: dict, reply_to_message_id: str = "") -> None:
        thread = threading.Thread(
            target=self._run_deal_action,
            args=(action, value, reply_to_message_id),
            name="feishu-deal-action-worker",
            daemon=True,
        )
        thread.start()

    def _run_deal_action(self, action: str, value: dict, reply_to_message_id: str = "") -> None:
        processing = DraftProgressCard()
        try:
            processing = self._start_draft_processing(reply_to_message_id, "正在生成商品快捷操作预览")
            try:
                stream_callback = self._draft_stream_callback(processing)
                if stream_callback:
                    outcome = build_deal_action_draft_outcome(
                        action,
                        value,
                        self.runtime.draft_store,
                        stream_callback=stream_callback,
                    )
                else:
                    outcome = build_deal_action_draft_outcome(action, value, self.runtime.draft_store)
                draft = outcome.draft
            finally:
                self._stop_draft_processing(processing)
            if not draft:
                if outcome.status == "noop":
                    message = outcome.message or "当前配置已经覆盖该快捷操作，无需修改。"
                    self._finish_draft_processing_noop(processing, message, reply_to_message_id)
                    return
                self._finish_draft_processing_failure(
                    processing,
                    "无法生成配置修改预览。",
                    reply_to_message_id,
                    "无法生成配置修改预览，请直接回复说明想怎么改。",
                )
                return
            if not self._send_and_store_draft_preview(
                draft,
                reply_to_message_id,
                processing,
            ):
                self._reply_text(reply_to_message_id, "商品快捷操作预览发送失败")
        except Exception as e:
            self._stop_draft_processing(processing)
            logger.error(f"商品快捷操作处理失败: {e}", exc_info=True)
            self._finish_draft_processing_failure(
                processing,
                "商品快捷操作处理失败，没能生成配置修改预览。",
                reply_to_message_id,
                INTERNAL_ERROR_MESSAGE,
            )

    def _handle_slash_command(self, text: str, reply_to_message_id: str = "") -> bool:
        if not text.startswith("/"):
            return False
        command = _command_key(text)
        if not find_command_spec(command):
            return self._handle_unknown_slash_command(command, reply_to_message_id)
        if self._handle_simple_slash_command(command, reply_to_message_id):
            return True
        if command.startswith("/search"):
            self._handle_search_command(text, command, reply_to_message_id)
            return True
        if command.startswith("/model"):
            self._handle_model_command(command, reply_to_message_id)
            return True
        return False

    def _handle_unknown_slash_command(self, command: str, reply_to_message_id: str) -> bool:
        if command.startswith("/search"):
            self._reply_text(reply_to_message_id, _search_usage_text())
            return True
        if command.startswith("/model"):
            self._reply_text(reply_to_message_id, _model_usage_text())
            return True
        return False

    def _handle_simple_slash_command(self, command: str, reply_to_message_id: str) -> bool:
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
        return False

    def _handle_search_command(self, text: str, command: str, reply_to_message_id: str = "") -> None:
        try:
            self._reply_text(reply_to_message_id, handle_search_command(text, command))
        except ValueError as e:
            self._reply_text(reply_to_message_id, f"搜索关键词配置读取失败：{e}")

    def _handle_search_card_action(self, action: str, value: dict) -> str:
        return handle_search_card_action(action, value)

    def _handle_model_command(self, command: str, reply_to_message_id: str = "") -> None:
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

    def _handle_model_card_action(self, action: str, value: dict) -> CardActionDispatchResult:
        """应用模型路由卡片操作，并重绘管理卡片。"""
        try:
            if action == "model_refresh":
                return CardActionDispatchResult("已刷新 LLM 路由", _build_model_management_card_response())
            if action == "model_test":
                message = _run_model_test(resolve_model_test_config(value))
                return CardActionDispatchResult(
                    message,
                    _build_model_management_card_response(form_state=_extract_form_state(value)),
                )
            result = apply_model_route_card_action(action, value)
            logger.info(
                "LLM 路由卡片操作成功: "
                f"action={action}, target={optional_model_card_field_value(value, 'target') or 'default'}, "
                f"connection={optional_model_card_field_value(value, 'connection')}, "
                f"model_id={optional_model_card_field_value(value, 'model_id')}"
            )
            form_state = (
                _model_form_state_from_snapshot(result.snapshot, model_card_target(value))
                if action == "model_reset_agent"
                else _extract_form_state(value)
            )
            return CardActionDispatchResult(
                result.message,
                _build_model_management_card_response(result.snapshot, form_state=form_state),
            )
        except (LLMRoutingError, ValueError) as e:
            return CardActionDispatchResult(
                f"WARN: {e}",
                _build_model_management_card_response(form_state=_extract_form_state(value)),
            )
        except Exception as e:
            logger.error(f"模型卡片操作失败: {e}", exc_info=True)
            return CardActionDispatchResult(
                INTERNAL_ERROR_MESSAGE,
                _build_model_management_card_response(form_state=_extract_form_state(value)),
            )

    def _remember_model_card_form_value(self, message_id: str, operator: str, value: dict) -> None:
        """缓存 Feishu 后续 callback 可能不会携带的模型表单局部编辑。"""
        key = model_card_form_cache_key(message_id, operator)
        if not key:
            return
        patch = build_model_card_form_patch(value)
        if not patch:
            return
        with self._model_card_form_lock:
            state = self._model_card_form_state.pop(key, {})
            self._model_card_form_state[key] = state
            apply_model_form_cache_patch(state, patch)
            trim_model_card_form_cache(self._model_card_form_state, MODEL_CARD_FORM_STATE_LIMIT)
            cached_keys = sorted(state.keys())
        logger.debug(f"缓存模型卡片表单值: operator={operator}, cached_keys={cached_keys}")

    def _merge_model_card_form_state(self, message_id: str, operator: str, value: dict) -> dict:
        """执行 action 前，把缓存的表单编辑合并进 payload。"""
        key = model_card_form_cache_key(message_id, operator)
        if not key:
            return value
        with self._model_card_form_lock:
            cached = dict(self._model_card_form_state.get(key, {}))
        merged = dict(cached)
        merged.update(value)
        return merged

    def _forget_model_card_form_state(self, message_id: str, operator: str) -> None:
        """刷新、重置或状态不再有用时丢弃模型表单缓存。"""
        key = model_card_form_cache_key(message_id, operator)
        if not key:
            return
        with self._model_card_form_lock:
            self._model_card_form_state.pop(key, None)

    def _handle_model_form_change(
        self,
        message_id: str,
        operator: str,
    ) -> object | None:
        """当 target/connection 下拉框变化时重绘模型卡片。

        自动填充值跟随选中的 target，但保留操作者在当前卡片上手动改过的字段。
        """
        cache_key = model_card_form_cache_key(message_id, operator)
        with self._model_card_form_lock:
            cached = dict(self._model_card_form_state.get(cache_key, {}))
            target = str(cached.get("target") or "default").strip()
            if target not in AGENTS and target != "default":
                return None
            final_form = self._model_form_for_target(target, cached)
            state = self._model_card_form_state.setdefault(cache_key, {})
            replace_model_form_cache_values(state, final_form)

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
            # 切换 target 时保留用户手动编辑过的路由字段。
            # 用户常先选 connection/model_id 再切换 agent；如果此处直接自动填充，
            # 会静默丢失用户想应用的路由。
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
        open_id = _extract_sender_open_id(data)
        return bool(open_id) and send_text_to("open_id", open_id, text)


def start_bot_thread(runtime: BotRuntime) -> threading.Thread | None:
    """启动飞书机器人后台线程。"""
    if not (config.FEISHU_APP_ID and config.FEISHU_APP_SECRET):
        return None
    bot = FeishuInteractiveBot(runtime)
    thread = threading.Thread(target=bot.start_blocking, name="feishu-bot", daemon=True)
    thread.start()
    return thread


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


def _card_event_id(data) -> str:
    event_id = getattr(getattr(data, "header", None), "event_id", "")
    if isinstance(event_id, str) and event_id.strip():
        return event_id.strip()
    legacy_uuid = getattr(data, "uuid", "")
    return legacy_uuid.strip() if isinstance(legacy_uuid, str) else ""


def _mask_card_event_id(event_id: str) -> str:
    """只为日志保留事件 ID 的有限首尾信息。"""
    if len(event_id) < 2:
        return "*" * len(event_id)
    if len(event_id) <= 8:
        return f"{event_id[:1]}******{event_id[-1:]}"
    return f"{event_id[:4]}******{event_id[-4:]}"


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


def _build_model_management_card_response(snapshot=None, form_state: dict | None = None) -> dict:
    """根据路由状态和可选表单值构造新的模型管理卡片。"""
    state = llm_routing.model_card_state(snapshot)
    return build_model_management_card(state, form_state=form_state or default_model_form_state(state))


def _extract_form_state(value: dict) -> dict[str, str]:
    """提取可用于预填充新卡片的表单字段值。"""
    result: dict[str, str] = {}
    for key in ("target", "connection", "model_id", "temperature"):
        v = optional_model_card_field_value(value, key)
        if v:
            result[key] = v
    return result


def _deal_reason_from_card_value(value: dict) -> tuple[bool, str]:
    """读取不值理由输入；bool 表示 payload 是否明确包含该字段。"""
    reason_field = str(value.get("reason_field") or NOT_WORTH_REASON_FIELD).strip()
    field_names = {NOT_WORTH_REASON_FIELD, reason_field}
    if str(value.get("name") or "").strip() in field_names and "input_value" in value:
        return True, str(value.get("input_value") or "").strip()
    for field_name in field_names:
        if field_name in value:
            return True, optional_card_field_value(value, field_name)
    for group_key in ("form_values", "form_value", "form", "input_values"):
        group = value.get(group_key)
        if not isinstance(group, dict):
            continue
        for field_name in field_names:
            if field_name in group:
                return True, optional_card_field_value(value, field_name)
    return False, ""


def _deal_reason_form_state_key(message_id: str, operator: str, article_id: str) -> str:
    if not message_id or not operator or not article_id:
        return ""
    return f"{message_id}:{operator}:{article_id}"


def _trim_deal_reason_form_state(state: dict[str, str]) -> None:
    while len(state) > DEAL_REASON_FORM_STATE_LIMIT:
        oldest = next(iter(state), None)
        if oldest is None:
            return
        state.pop(oldest, None)


def _model_form_state_from_snapshot(snapshot: llm_routing.RoutingSnapshot, target: str) -> dict[str, str]:
    """把已提交的路由快照转换回模型卡片表单默认值。"""
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
