"""配置加载模块。

从 .env 文件和环境变量加载所有配置。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_HOME_ENV = "SMZDM_NOTICE_HOME"


def _resolve_project_root() -> Path:
    configured = os.getenv(PROJECT_HOME_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
WORKSPACE_STATE_DIR = WORKSPACE_DIR / "state"
WORKSPACE_LOG_DIR = WORKSPACE_DIR / "logs"
WORKSPACE_AUDIT_DIR = WORKSPACE_DIR / "audit"
WORKSPACE_BACKUP_DIR = WORKSPACE_DIR / "backups"

# 加载 .env 文件。普通启动保持环境变量优先；飞书 /restart 后需要覆盖继承的旧 .env 值。
load_dotenv(PROJECT_ROOT / ".env", override=os.getenv("SMZDM_RESTART_RELOAD_DOTENV") == "1")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _get_bool(key: str, default: bool = False) -> bool:
    val = _get(key, "true" if default else "false").lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_int(key: str, default: int = 0) -> int:
    val = _get(key, str(default))
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(key: str, default: float = 0.0) -> float:
    val = _get(key, str(default))
    try:
        return float(val)
    except ValueError:
        return default


def _get_fallback(key: str, fallback: str) -> str:
    return _get(key) or fallback


def _get_float_fallback(key: str, fallback: float) -> float:
    val = _get(key)
    if not val:
        return fallback
    try:
        return float(val)
    except ValueError:
        return fallback


def _clamp_rate(value: float) -> float:
    return min(1.0, max(0.0, value))


def _get_str_list(key: str, default: str = "") -> list[str]:
    """解析逗号分隔的字符串列表。"""
    raw = _get(key, default)
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


# ========== 飞书配置 ==========
FEISHU_APP_ID: str = _get("FEISHU_APP_ID")
FEISHU_APP_SECRET: str = _get("FEISHU_APP_SECRET")
FEISHU_BINDING_FILE: str = _get("FEISHU_BINDING_FILE", str(WORKSPACE_STATE_DIR / "feishu_binding.json"))

# ========== SMZDM 配置 ==========
SMZDM_SIGN_KEY: str = _get("SMZDM_SIGN_KEY")
SMZDM_USER_AGENT: str = _get("SMZDM_USER_AGENT")

# ========== LLM 配置 ==========
LLM_API_KEY: str = _get("LLM_API_KEY")
LLM_BASE_URL: str = _get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL: str = _get("LLM_MODEL", "deepseek-chat")
LLM_MAX_RETRIES: int = _get_int("LLM_MAX_RETRIES", 2)
LLM_TIMEOUT_SECONDS: float = _get_float_fallback("LLM_TIMEOUT_SECONDS", 300.0)
LLM_DUAL_FILTER: bool = _get("LLM_DUAL_FILTER", "false").lower() == "true"
LLM_ARBITER_ENABLED: bool = _get("LLM_ARBITER_ENABLED", "true").lower() == "true"
LLM_ARBITER_API_KEY: str = _get_fallback("LLM_ARBITER_API_KEY", LLM_API_KEY)
LLM_ARBITER_BASE_URL: str = _get_fallback("LLM_ARBITER_BASE_URL", LLM_BASE_URL)
LLM_ARBITER_MODEL: str = _get_fallback("LLM_ARBITER_MODEL", LLM_MODEL)
LLM_ARBITER_TIMEOUT_SECONDS: float = _get_float_fallback("LLM_ARBITER_TIMEOUT_SECONDS", LLM_TIMEOUT_SECONDS)
LLM_DRAFT_API_KEY: str = _get_fallback("LLM_DRAFT_API_KEY", LLM_ARBITER_API_KEY)
LLM_DRAFT_BASE_URL: str = _get_fallback("LLM_DRAFT_BASE_URL", LLM_ARBITER_BASE_URL)
LLM_DRAFT_MODEL: str = _get_fallback("LLM_DRAFT_MODEL", LLM_ARBITER_MODEL)
LLM_DRAFT_TIMEOUT_SECONDS: float = _get_float_fallback("LLM_DRAFT_TIMEOUT_SECONDS", LLM_ARBITER_TIMEOUT_SECONDS)

# ========== 预筛选配置 ==========
PREFILTER_ENABLED: bool = _get_bool("PREFILTER_ENABLED", False)
PREFILTER_BYPASS_ENABLED: bool = _get_bool("PREFILTER_BYPASS_ENABLED", False)
PREFILTER_MIN_WORTHY: int = _get_int("PREFILTER_MIN_WORTHY", 0)
PREFILTER_MIN_WORTHY_RATE: float = _clamp_rate(_get_float("PREFILTER_MIN_WORTHY_RATE", 0.0))
PREFILTER_MIN_COMMENTS: int = _get_int("PREFILTER_MIN_COMMENTS", 0)
PREFILTER_MIN_FAVORITES: int = _get_int("PREFILTER_MIN_FAVORITES", 0)
PREFILTER_BYPASS_MIN_COMMENTS: int = _get_int("PREFILTER_BYPASS_MIN_COMMENTS", 0)
PREFILTER_BYPASS_MIN_WORTHY: int = _get_int("PREFILTER_BYPASS_MIN_WORTHY", 0)

# ========== 动态文本配置 ==========
CONFIG_BACKUP_DIR: str = _get("CONFIG_BACKUP_DIR", str(WORKSPACE_BACKUP_DIR))
CONFIG_DRAFT_FILE: str = _get("CONFIG_DRAFT_FILE", str(WORKSPACE_STATE_DIR / "pending_config_changes.json"))
CONFIG_AUDIT_FILE: str = _get("CONFIG_AUDIT_FILE", str(WORKSPACE_AUDIT_DIR / "config_change_audit.jsonl"))


def load_required_text_file(filename: str) -> str:
    """读取必需文本配置，缺失、读取失败或空文件均视为错误。"""
    path = PROJECT_ROOT / filename
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"无法读取 {filename}: {e}") from e
    if not content.strip():
        raise ValueError(f"{filename} 为空")
    return content


def get_user_prompt() -> str:
    """实时读取用户购物偏好。"""
    return load_required_text_file("preference.md")


def get_inventory_data() -> str:
    """实时读取耗材库存记录。"""
    return load_required_text_file("inventory.md")


# ========== 轮询配置 ==========
POLL_INTERVAL_MINUTES: int = _get_int("POLL_INTERVAL_MINUTES", 30)
HEARTBEAT_HOURS: int = _get_int("HEARTBEAT_HOURS", 6)
FETCH_INTERVAL_SECONDS: int = _get_int("FETCH_INTERVAL_SECONDS", 5)

# ========== 排行榜配置 ==========
TOP_N: int = _get_int("TOP_N", 20)
# 可选榜单（逗号分隔，不配置则默认全部）：
#   综合榜-全部 综合榜-电脑数码 综合榜-白菜 综合榜-食品生鲜
#   综合榜-运动户外 综合榜-家用电器 综合榜-服饰鞋包 综合榜-日用百货
#   综合榜-母婴用品 综合榜-家居家装 综合榜-办公设备 综合榜-个护化妆
#   综合榜-本地生活 综合榜-医疗健康 综合榜-图书文娱 综合榜-玩模乐器
#   热卖榜 热评榜 热搜榜
RANKING_NAMES: list[str] = _get_str_list("RANKING_NAMES", "")

# ========== 搜索关键词配置 ==========
SEARCH_KEYWORDS_FILE: str = _get("SEARCH_KEYWORDS_FILE", "search_keywords.json")


def get_search_keywords() -> list[str]:
    """读取可选的 SMZDM 搜索关键词配置，只接受对象格式。"""
    path = Path(SEARCH_KEYWORDS_FILE)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        raise
    keywords = data.get("keywords") if isinstance(data, dict) else None
    if not isinstance(keywords, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in keywords:
        if not isinstance(item, dict):
            raise ValueError("search_keywords.json 只支持对象格式：{'keyword': '...', 'max_price': null}")
        keyword = str(item.get("keyword") or "").strip()
        if keyword and keyword not in seen:
            seen.add(keyword)
            normalized.append(keyword)
    return normalized


# ========== 去重配置 ==========
DEDUP_EXPIRE_HOURS: int = _get_int("DEDUP_EXPIRE_HOURS", 24)
DEDUP_FILE: str = _get("DEDUP_FILE", str(WORKSPACE_STATE_DIR / "dedup_cache.json"))

# ========== 夜间汇总配置 ==========
DIGEST_HOUR: int = _get_int("DIGEST_HOUR", 22)
DIGEST_FILE: str = _get("DIGEST_FILE", str(WORKSPACE_STATE_DIR / "near_misses.json"))
LOG_FILE_PATTERN: str = _get("LOG_FILE_PATTERN", str(WORKSPACE_LOG_DIR / "smzdm_notice_{time:YYYY-MM-DD}.log"))


def ensure_workspace_dirs() -> None:
    """创建运行时 workspace 目录。"""
    for path in (WORKSPACE_STATE_DIR, WORKSPACE_LOG_DIR, WORKSPACE_AUDIT_DIR, WORKSPACE_BACKUP_DIR):
        path.mkdir(parents=True, exist_ok=True)
