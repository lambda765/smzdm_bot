"""商品来源抓取流程共用的 sleep 工具。"""

from __future__ import annotations

import time
from collections.abc import Callable

from loguru import logger


def interruptible_sleep(
    seconds: int,
    should_stop: Callable[[], bool] | None = None,
    stop_message: str = "收到停止信号，中断等待",
) -> bool:
    """按一秒粒度等待；若被停止信号中断则返回 True。"""
    elapsed = 0
    while elapsed < seconds:
        if should_stop and should_stop():
            logger.info(stop_message)
            return True
        time.sleep(min(1, seconds - elapsed))
        elapsed += 1
    return False
