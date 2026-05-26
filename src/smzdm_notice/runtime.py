"""SMZDM 好价提醒机器人 — 主入口。

定时轮询什么值得买排行榜，通过 LLM 筛选后推送到飞书。
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal

from loguru import logger

from smzdm_notice.core import config
from smzdm_notice.core.dedup import DedupManager
from smzdm_notice.core.near_miss import NearMissManager
from smzdm_notice.feishu.binding import FeishuBindingStore
from smzdm_notice.feishu.bot import BotRuntime, start_bot_thread
from smzdm_notice.feishu.notifier import (
    disable_draft_card,
    send_arbitration,
    send_config_warning,
    send_deals,
    send_digest,
    send_heartbeat,
    send_poll_failure_warning,
    send_shutdown,
    send_startup,
)
from smzdm_notice.llm.filter import filter_items
from smzdm_notice.llm.models import ArbiterInfo
from smzdm_notice.preferences.builder import build_arbitration_draft
from smzdm_notice.preferences.store import CONFIG_FILE_LOCK, DraftStore
from smzdm_notice.smzdm.keywords import SearchKeywordRule
from smzdm_notice.smzdm.ranking import RANKINGS, RankingItem
from smzdm_notice.smzdm.sources import fetch_all_sources

# ========== 日志配置 ==========

config.ensure_workspace_dirs()
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    config.LOG_FILE_PATTERN,
    rotation="00:00",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
)

# ========== 全局状态 ==========

_stop_event = threading.Event()
_restart_requested = False
_last_push_time: float = time.time()
_ranking_configs: list = []  # 启动时由 _resolve_ranking_configs() 填充
_search_keywords: list[SearchKeywordRule] = []  # 启动时读取，轮询前刷新
_current_user_prompt: str = ""
_current_inventory_data: str = ""
_last_config_warning_times: dict[str, float] = {}
_CONFIG_WARNING_INTERVAL_SECONDS = 3600
_poll_lock = threading.Lock()
_dedup_manager: DedupManager | None = None
_near_miss_manager: NearMissManager | None = None
_draft_store: DraftStore | None = None
_poll_failure_tracker: PollFailureTracker | None = None
_last_poll_started_at: float = 0
_last_poll_finished_at: float = 0
_BINDING_WAIT_SECONDS = 60
_RESTART_RELOAD_DOTENV_ENV = "SMZDM_RESTART_RELOAD_DOTENV"


def _signal_handler(sig, frame) -> None:
    """优雅退出。"""
    logger.info("收到退出信号，正在停止...")
    _stop_event.set()


def _request_restart() -> bool:
    """由飞书 /restart 触发，标记重启并退出主循环。"""
    global _restart_requested
    if _restart_requested:
        return False
    _restart_requested = True
    _stop_event.set()
    return True


def _exec_restart() -> None:
    """重新执行当前进程，并让新进程重新读取 .env 中的最新值。"""
    env = os.environ.copy()
    env[_RESTART_RELOAD_DOTENV_ENV] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ========== 配置校验 ==========


def _resolve_ranking_configs() -> list:
    """根据配置解析榜单列表，配置为空则使用全部榜单。"""
    if config.RANKING_NAMES:
        configs = []
        for name in config.RANKING_NAMES:
            if name not in RANKINGS:
                logger.error(f"未知榜单: {name}")
                sys.exit(1)
            configs.append(RANKINGS[name])
        return configs
    return list(RANKINGS.values())


def _validate_config() -> bool:
    """校验必要配置。"""
    errors = []
    if not config.FEISHU_APP_ID:
        errors.append("FEISHU_APP_ID 未配置")
    if not config.FEISHU_APP_SECRET:
        errors.append("FEISHU_APP_SECRET 未配置")
    if config.FEISHU_APP_ID and _looks_like_placeholder(config.FEISHU_APP_ID):
        errors.append("FEISHU_APP_ID 仍是示例占位符，请填入真实 App ID")
    if config.FEISHU_APP_SECRET and _looks_like_placeholder(config.FEISHU_APP_SECRET):
        errors.append("FEISHU_APP_SECRET 仍是示例占位符，请填入真实 App Secret")
    if not config.LLM_API_KEY:
        errors.append("LLM_API_KEY 未配置")

    if errors:
        for e in errors:
            logger.error(f"配置错误: {e}")
        return False
    return True


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {"cli_xxx", "your-app-secret"} or "xxx" in lowered


def _load_initial_runtime_config() -> bool:
    """启动时严格读取动态文本配置，失败则不允许进入主循环。"""
    global _current_user_prompt, _current_inventory_data
    try:
        with CONFIG_FILE_LOCK:
            _current_user_prompt = config.get_user_prompt()
            _current_inventory_data = config.get_inventory_data()
    except Exception as e:
        logger.error(f"动态配置读取失败: {e}")
        return False
    return True


def _load_search_keywords(previous: list[SearchKeywordRule] | None = None) -> list[SearchKeywordRule]:
    """读取可选搜索关键词；失败时沿用上一次成功配置。"""
    fallback = previous or []
    try:
        from smzdm_notice.smzdm.keywords import list_keyword_rules

        return list_keyword_rules()
    except Exception as e:
        logger.warning(f"搜索关键词配置读取失败，将使用上一次成功读取的内容: {e}")
        _send_config_warning_throttled(
            "search_keywords",
            f"**搜索关键词配置** 读取失败：`{e}`",
        )
        return fallback


def _send_config_warning_throttled(key: str, message: str) -> None:
    """节流发送配置读取告警，避免每轮刷屏。"""
    now = time.time()
    last_sent = _last_config_warning_times.get(key, 0)
    if now - last_sent < _CONFIG_WARNING_INTERVAL_SECONDS:
        return
    if send_config_warning(message):
        _last_config_warning_times[key] = now


def _refresh_runtime_text(
    key: str,
    label: str,
    loader: Callable[[], str],
    previous: str,
) -> str:
    """运行中刷新单个文本配置，失败时使用上一次成功内容。"""
    try:
        with CONFIG_FILE_LOCK:
            return loader()
    except Exception as e:
        logger.warning(f"{label} 读取失败，将使用上一次成功读取的内容: {e}")
        _send_config_warning_throttled(
            key,
            f"**{label}** 读取失败：`{e}`",
        )
        return previous


def _refresh_runtime_config() -> tuple[str, str]:
    """每轮筛选前刷新偏好和库存，失败项用上一次成功内容兜底。"""
    global _current_user_prompt, _current_inventory_data

    _current_user_prompt = _refresh_runtime_text(
        key="preference.md",
        label="preference.md",
        loader=config.get_user_prompt,
        previous=_current_user_prompt,
    )
    _current_inventory_data = _refresh_runtime_text(
        key="inventory.md",
        label="inventory.md",
        loader=config.get_inventory_data,
        previous=_current_inventory_data,
    )
    return _current_user_prompt, _current_inventory_data


def _config_summary() -> str:
    """生成配置摘要。"""
    ranking_names = [cfg.name for cfg in _ranking_configs]
    search_summary = _search_keywords_summary()
    prefilter_summary = _prefilter_config_summary()
    return (
        f"**监控配置**\n"
        f"- 🕐 轮询间隔: {config.POLL_INTERVAL_MINUTES} 分钟\n"
        f"- 💓 心跳间隔: {config.HEARTBEAT_HOURS} 小时\n"
        f"- 🌙 夜间汇总: 每天 {config.DIGEST_HOUR}:00\n"
        f"- 📊 监控榜单: {', '.join(ranking_names)}\n"
        f"- 🔎 搜索关键词: {search_summary}\n"
        f"- 🔢 每榜 Top: {config.TOP_N}\n"
        f"- 🤖 LLM 模型: {config.LLM_MODEL}\n"
        f"- 🔀 双重判断: {'已启用' if config.LLM_DUAL_FILTER else '未启用'}"
        + (" (含仲裁)" if config.LLM_DUAL_FILTER and config.LLM_ARBITER_ENABLED else "")
        + "\n"
        + (
            f"- ⚖️ 仲裁模型: {config.LLM_ARBITER_MODEL}\n"
            if config.LLM_DUAL_FILTER and config.LLM_ARBITER_ENABLED
            else ""
        )
        + prefilter_summary
        + f"- 🎯 用户偏好: {_current_user_prompt[:80]}{'...' if len(_current_user_prompt) > 80 else ''}"
    )


def _search_keywords_summary() -> str:
    if not _search_keywords:
        return "未配置"
    preview = ", ".join(_format_search_rule(rule) for rule in _search_keywords[:8])
    if len(_search_keywords) > 8:
        preview += f" 等 {len(_search_keywords)} 个"
    return preview


def _format_search_rule(rule: SearchKeywordRule | str) -> str:
    if isinstance(rule, str):
        return rule
    if rule.max_price is None:
        return rule.keyword
    return f"{rule.keyword}(≤{rule.max_price:g})"


def _prefilter_config_summary() -> str:
    if not config.PREFILTER_ENABLED:
        return "- 🧮 预筛选: 未启用\n"
    summary = (
        "- 🧮 预筛选: 已启用\n"
        f"  - 普通阈值: 值票≥{config.PREFILTER_MIN_WORTHY}, "
        f"值率≥{config.PREFILTER_MIN_WORTHY_RATE:.2f}, "
        f"评论≥{config.PREFILTER_MIN_COMMENTS}, "
        f"收藏≥{config.PREFILTER_MIN_FAVORITES}\n"
    )
    if config.PREFILTER_BYPASS_ENABLED:
        summary += (
            f"  - 强信号直通: 评论≥{config.PREFILTER_BYPASS_MIN_COMMENTS}, 值票≥{config.PREFILTER_BYPASS_MIN_WORTHY}\n"
        )
    else:
        summary += "  - 强信号直通: 未启用\n"
    return summary


@dataclass
class PollOutcome:
    """单轮轮询结果，用于统一失败追踪。"""

    status: Literal["success", "failure", "skipped"]
    reason: str | None = None
    detail: str | None = None

    @classmethod
    def success(cls) -> PollOutcome:
        return cls(status="success")

    @classmethod
    def failure(cls, reason: str, detail: str | None = None) -> PollOutcome:
        return cls(status="failure", reason=reason, detail=detail)

    @classmethod
    def skipped(cls, reason: str | None = None) -> PollOutcome:
        return cls(status="skipped", reason=reason)


@dataclass
class PollFailureTracker:
    """记录连续轮询失败，并在达到阈值时发送一次告警。"""

    threshold: int = 3
    consecutive_failures: int = 0
    alerted: bool = False

    def record(self, outcome: PollOutcome) -> None:
        if outcome.status == "skipped":
            return
        if outcome.status == "success":
            self.reset()
            return

        self.consecutive_failures += 1
        if self.consecutive_failures < self.threshold or self.alerted:
            return

        if send_poll_failure_warning(
            self.consecutive_failures,
            outcome.reason or "unknown",
            outcome.detail,
        ):
            self.alerted = True

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.alerted = False


@dataclass
class MatchEvaluation:
    """轮询中商品筛选后的待处理结果。"""

    matched: list[tuple[RankingItem, str]]
    near_misses: list[tuple[RankingItem, str]]
    arbiter_info: ArbiterInfo | None = None


_poll_failure_tracker = PollFailureTracker()


# ========== 单次轮询 ==========


def _poll_once(dedup: DedupManager, near_miss_mgr: NearMissManager) -> None:
    """执行一次完整的轮询流程。"""
    global _last_push_time, _last_poll_started_at, _last_poll_finished_at

    logger.info("=" * 50)
    logger.info("开始新一轮轮询...")
    _last_poll_started_at = time.time()
    _maintain_config_drafts("轮询开始清理")
    try:
        outcome = _poll_once_unlocked(dedup, near_miss_mgr)
        if _poll_failure_tracker:
            _poll_failure_tracker.record(outcome)
    finally:
        _last_poll_finished_at = time.time()
        _maintain_config_drafts("轮询结束清理")


def _poll_once_unlocked(dedup: DedupManager, near_miss_mgr: NearMissManager) -> PollOutcome:
    """不负责加锁的单次轮询实现。"""
    all_items, early_outcome = _fetch_poll_items()
    if early_outcome:
        return early_outcome

    new_items, early_outcome = _dedupe_poll_items(all_items, dedup)
    if early_outcome:
        return early_outcome

    bypass_matches, llm_candidates = _split_price_bypass_items(new_items)
    if bypass_matches:
        logger.info(f"搜索价格阈值直推: {len(bypass_matches)} 条；其余 {len(llm_candidates)} 条进入 LLM 流程")

    evaluation = _evaluate_poll_matches(bypass_matches, llm_candidates, dedup, near_miss_mgr)
    if isinstance(evaluation, PollOutcome):
        return evaluation

    _handle_arbitration(evaluation.arbiter_info)
    _record_near_misses(evaluation.matched, evaluation.near_misses, near_miss_mgr)
    _send_or_log_matches(evaluation.matched, bypass_matches, dedup, near_miss_mgr)
    _check_digest(near_miss_mgr)
    if not evaluation.matched:
        _check_heartbeat()
    return PollOutcome.success()


def _fetch_poll_items() -> tuple[list[RankingItem], PollOutcome | None]:
    global _search_keywords

    _search_keywords = _load_search_keywords(_search_keywords)
    try:
        all_items = fetch_all_sources(
            ranking_configs=_ranking_configs,
            search_rules=_search_keywords,
            top_n=config.TOP_N,
            interval_seconds=config.FETCH_INTERVAL_SECONDS,
            should_stop=_stop_event.is_set,
        )
    except Exception as e:
        logger.error(f"商品来源抓取失败: {e}")
        return [], PollOutcome.failure("ranking_fetch_failed", str(e))

    if _stop_event.is_set():
        logger.info("收到停止信号，跳过后续筛选与推送")
        return [], PollOutcome.skipped("stopped")
    if not all_items:
        logger.warning("未获取到任何商品")
        _check_heartbeat()
        return [], PollOutcome.success()
    return all_items, None


def _dedupe_poll_items(
    all_items: list[RankingItem],
    dedup: DedupManager,
) -> tuple[list[RankingItem], PollOutcome | None]:
    new_items = [item for item in all_items if dedup.is_new(item.link)]
    logger.info(f"去重后剩余 {len(new_items)}/{len(all_items)} 条新商品")

    if not new_items:
        logger.info("没有新商品，跳过 LLM 筛选")
        _check_heartbeat()
        return [], PollOutcome.success()
    if _stop_event.is_set():
        logger.info("收到停止信号，跳过 LLM 筛选")
        return [], PollOutcome.skipped("stopped")
    return new_items, None


def _evaluate_poll_matches(
    bypass_matches: list[tuple[RankingItem, str]],
    llm_candidates: list[RankingItem],
    dedup: DedupManager,
    near_miss_mgr: NearMissManager,
) -> MatchEvaluation | PollOutcome:
    if not llm_candidates:
        return MatchEvaluation(matched=bypass_matches, near_misses=[])

    user_prompt, inventory_data = _refresh_runtime_config()
    if _stop_event.is_set():
        logger.info("收到停止信号，跳过 LLM 筛选")
        return PollOutcome.skipped("stopped")

    filter_result = filter_items(
        items=llm_candidates,
        user_prompt=user_prompt,
        inventory_data=inventory_data,
        model=config.LLM_MODEL,
    )
    if filter_result.diagnostics.llm_failed:
        logger.error("LLM 筛选全失败，跳过 LLM 推荐")
        if bypass_matches:
            _send_matches_and_update_state(
                bypass_matches,
                dedup,
                near_miss_mgr,
                price_bypass_article_ids={item.article_id for item, _ in bypass_matches},
            )
        return PollOutcome.failure("llm_failed", filter_result.diagnostics.error_summary)

    return MatchEvaluation(
        matched=bypass_matches + filter_result.matched,
        near_misses=filter_result.near_misses,
        arbiter_info=filter_result.arbiter_info,
    )


def _handle_arbitration(arbiter_info: ArbiterInfo | None) -> None:
    if arbiter_info:
        arbitration_draft = None
        if _draft_store:
            arbitration_draft = build_arbitration_draft(
                arbiter_info.config_change_draft,
                _draft_store,
                suggestion=arbiter_info.suggestion,
            )
            if arbiter_info.config_change_draft and not arbitration_draft:
                logger.warning("仲裁配置草案无效，跳过一键采纳按钮")
        arbitration_sent = send_arbitration(arbiter_info, arbitration_draft)
        if arbitration_sent and arbitration_draft and _draft_store and arbitration_draft.preview_message_id:
            _draft_store.update(arbitration_draft)


def _record_near_misses(
    matched: list[tuple[RankingItem, str]],
    near_misses: list[tuple[RankingItem, str]],
    near_miss_mgr: NearMissManager,
) -> None:
    if near_misses:
        matched_ids = {item.article_id for item, _ in matched}
        filtered_near_misses = [(item, reason) for item, reason in near_misses if item.article_id not in matched_ids]
        if filtered_near_misses:
            near_miss_mgr.add_batch(filtered_near_misses)
            logger.info(f"收集 {len(filtered_near_misses)} 条 near-miss 条目")


def _send_or_log_matches(
    matched: list[tuple[RankingItem, str]],
    bypass_matches: list[tuple[RankingItem, str]],
    dedup: DedupManager,
    near_miss_mgr: NearMissManager,
) -> None:
    if matched:
        _send_matches_and_update_state(
            matched,
            dedup,
            near_miss_mgr,
            price_bypass_article_ids={item.article_id for item, _ in bypass_matches},
        )
    else:
        logger.info("LLM 判断无匹配商品")


def _send_matches_and_update_state(
    matched: list[tuple[RankingItem, str]],
    dedup: DedupManager,
    near_miss_mgr: NearMissManager,
    price_bypass_article_ids: set[str] | None = None,
) -> bool:
    logger.info(f"发现 {len(matched)} 件匹配商品，推送飞书...")
    success = send_deals(matched, price_bypass_article_ids=price_bypass_article_ids)
    if success:
        global _last_push_time
        _last_push_time = time.time()
        dedup.mark_batch([item.link for item, _ in matched])
        for item, _ in matched:
            near_miss_mgr.remove(item.article_id)
        logger.info("推送成功，已更新去重缓存")
    else:
        logger.error("推送失败")
    return success


def _split_price_bypass_items(items: list[RankingItem]) -> tuple[list[tuple[RankingItem, str]], list[RankingItem]]:
    bypass_matches = []
    llm_candidates = []
    for item in items:
        reason = _price_bypass_reason(item)
        if reason:
            bypass_matches.append((item, reason))
        else:
            llm_candidates.append(item)
    return bypass_matches, llm_candidates


def _price_bypass_reason(item) -> str:
    if getattr(item, "source_type", "") != "search":
        return ""
    max_price = getattr(item, "search_max_price", None)
    if max_price is None:
        return ""
    price = getattr(item, "numeric_price", None)
    if price is None or price > max_price:
        return ""
    keyword = getattr(item, "search_keyword", "") or "搜索关键词"
    return f"搜索关键词「{keyword}」价格 {price:g} 小于等于阈值 {max_price:g}，直接推送。"


def _check_heartbeat() -> None:
    """检查是否需要发送心跳。"""
    global _last_push_time

    hours_since_push = (time.time() - _last_push_time) / 3600
    if hours_since_push >= config.HEARTBEAT_HOURS:
        logger.info(f"已 {hours_since_push:.1f} 小时未推送，发送心跳...")
        success = send_heartbeat(config.HEARTBEAT_HOURS)
        if success:
            _last_push_time = time.time()


def _check_digest(near_miss_mgr: NearMissManager) -> None:
    """检查是否需要发送夜间汇总。"""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # 今天已经发过了
    if near_miss_mgr.get_last_digest_date() == today_str:
        return

    # 还没到汇总时间
    if now.hour < config.DIGEST_HOUR:
        return

    # 发送汇总
    entries = near_miss_mgr.get_all_sorted()
    if not entries:
        logger.info("今日无 near-miss 条目，跳过汇总")
        near_miss_mgr.set_last_digest_date(today_str)
        return

    logger.info(f"发送夜间汇总: {len(entries)} 条 near-miss 条目")
    success = send_digest(entries, today_str)
    if success:
        near_miss_mgr.set_last_digest_date(today_str)
        near_miss_mgr.clear()
        logger.info("夜间汇总发送成功，已清空 near-miss 缓存")


def _status_summary() -> str:
    """生成飞书 /status 文本。"""
    ranking_names = [cfg.name for cfg in _ranking_configs]
    binding_status = FeishuBindingStore().describe()
    last_push = datetime.fromtimestamp(_last_push_time).strftime("%Y-%m-%d %H:%M:%S")
    last_started = (
        datetime.fromtimestamp(_last_poll_started_at).strftime("%Y-%m-%d %H:%M:%S")
        if _last_poll_started_at
        else "尚未轮询"
    )
    last_finished = (
        datetime.fromtimestamp(_last_poll_finished_at).strftime("%Y-%m-%d %H:%M:%S")
        if _last_poll_finished_at
        else "尚未完成"
    )
    pref_mtime = _file_mtime("preference.md")
    inv_mtime = _file_mtime("inventory.md")
    return (
        "好价监控状态\n"
        f"- 轮询中：{'是' if _poll_lock.locked() else '否'}\n"
        f"- 上次轮询开始：{last_started}\n"
        f"- 上次轮询完成：{last_finished}\n"
        f"- 上次成功推送：{last_push}\n"
        f"- 飞书通知目标：{binding_status}\n"
        f"- 监控榜单：{', '.join(ranking_names)}\n"
        f"- 搜索关键词：{_search_keywords_summary()}\n"
        f"- preference.md 更新时间：{pref_mtime}\n"
        f"- inventory.md 更新时间：{inv_mtime}"
    )


def _file_mtime(filename: str) -> str:
    path = config.PROJECT_ROOT / filename
    if not path.exists():
        return "不存在"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _trigger_manual_poll() -> bool:
    """由飞书 /run 触发后台轮询。"""
    dedup_manager = _dedup_manager
    near_miss_manager = _near_miss_manager
    if dedup_manager is None or near_miss_manager is None:
        return False
    if not _poll_lock.acquire(blocking=False):
        return False

    def worker() -> None:
        try:
            _poll_once(dedup_manager, near_miss_manager)
        except Exception as e:
            logger.error(f"手动轮询异常: {e}", exc_info=True)
        finally:
            _poll_lock.release()

    threading.Thread(target=worker, name="manual-poll", daemon=True).start()
    return True


def _maintain_config_drafts(reason: str = "草案清理") -> None:
    if not _draft_store:
        return
    for draft in _draft_store.expire_pending():
        if draft.preview_message_id:
            disable_draft_card(draft.preview_message_id, f"{reason}：草案已超过 24 小时自动失效", draft)
    removed = _draft_store.compact()
    if removed:
        logger.info(f"{reason}: 已清理 {len(removed)} 条已结束配置草案")


# ========== 主入口 ==========


def main() -> None:
    """主函数。"""
    logger.info("🚀 SMZDM 好价提醒机器人启动")
    _ensure_startup_ready()
    dedup, near_miss_mgr, binding_store = _initialize_runtime()
    _notify_startup_if_bound(binding_store)
    _run_poll_loop(binding_store, dedup, near_miss_mgr)
    _notify_shutdown_or_restart()


def _ensure_startup_ready() -> None:
    if not _validate_config():
        logger.error("配置校验失败，退出")
        sys.exit(1)

    if not _load_initial_runtime_config():
        logger.error("动态配置校验失败，退出")
        sys.exit(1)


def _initialize_runtime() -> tuple[DedupManager, NearMissManager, FeishuBindingStore]:
    global _dedup_manager, _draft_store, _near_miss_manager, _ranking_configs, _search_keywords

    _ranking_configs = _resolve_ranking_configs()
    _search_keywords = _load_search_keywords()
    logger.info(f"已配置 {len(_ranking_configs)} 个榜单")
    logger.info(f"已配置 {len(_search_keywords)} 个搜索关键词")

    summary = _config_summary()
    logger.info(f"\n{summary}")

    # 初始化去重管理器
    dedup = DedupManager(
        filepath=config.DEDUP_FILE,
        expire_hours=config.DEDUP_EXPIRE_HOURS,
    )
    _dedup_manager = dedup
    logger.info(f"去重缓存已加载: {dedup.size} 条记录")

    # 初始化 near-miss 管理器
    near_miss_mgr = NearMissManager(
        filepath=config.DIGEST_FILE,
        expire_hours=config.DEDUP_EXPIRE_HOURS,
    )
    _near_miss_manager = near_miss_mgr
    logger.info(f"Near-miss 缓存已加载: {near_miss_mgr.size} 条记录")

    draft_store = DraftStore()
    _draft_store = draft_store
    _maintain_config_drafts("重启后清理")
    binding_store = FeishuBindingStore()
    _start_bot(draft_store, binding_store)
    return dedup, near_miss_mgr, binding_store


def _start_bot(draft_store: DraftStore, binding_store: FeishuBindingStore) -> None:
    runtime = BotRuntime(
        draft_store=draft_store,
        binding_store=binding_store,
        status_provider=_status_summary,
        run_once=_trigger_manual_poll,
        restart=_request_restart,
    )
    start_bot_thread(runtime)


def _notify_startup_if_bound(binding_store: FeishuBindingStore) -> None:
    global _last_push_time

    if binding_store.get():
        send_startup(_config_summary())
        _last_push_time = time.time()
        time.sleep(5)  # 避免飞书限流
    else:
        logger.info("飞书通知目标尚未绑定，绑定前不进行榜单轮询")


def _run_poll_loop(
    binding_store: FeishuBindingStore,
    dedup: DedupManager,
    near_miss_mgr: NearMissManager,
) -> None:
    poll_interval = config.POLL_INTERVAL_MINUTES * 60
    logger.info(f"进入主循环，每 {config.POLL_INTERVAL_MINUTES} 分钟轮询一次")

    while not _stop_event.is_set():
        if not binding_store.get():
            logger.info("等待飞书绑定完成，跳过本轮 API 轮询")
            _stop_event.wait(timeout=min(_BINDING_WAIT_SECONDS, poll_interval))
            continue
        _run_scheduled_poll(dedup, near_miss_mgr)
        _wait_until_next_poll(poll_interval)


def _run_scheduled_poll(dedup: DedupManager, near_miss_mgr: NearMissManager) -> None:
    if _poll_lock.acquire(blocking=False):
        try:
            _poll_once(dedup, near_miss_mgr)
        except Exception as e:
            logger.error(f"轮询异常: {e}", exc_info=True)
        finally:
            _poll_lock.release()
    else:
        logger.info("已有轮询在执行，本轮定时触发跳过")


def _wait_until_next_poll(poll_interval: int) -> None:
    if _stop_event.is_set():
        return
    next_time = datetime.now().timestamp() + poll_interval
    next_str = datetime.fromtimestamp(next_time).strftime("%H:%M:%S")
    logger.info(f"下次轮询: {next_str}")
    _stop_event.wait(timeout=poll_interval)


def _notify_shutdown_or_restart() -> None:
    logger.info("👋 好价监控已停止")
    if _restart_requested:
        send_shutdown("正在重启程序...")
        logger.info("🔄 重启程序...")
        _exec_restart()
    else:
        send_shutdown("收到退出信号，正常停止")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序异常退出: {e}", exc_info=True)
        with suppress(Exception):
            send_shutdown(f"程序异常退出: {e}")
