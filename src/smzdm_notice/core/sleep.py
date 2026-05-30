"""Sleep helpers shared by polling source fetchers."""

from __future__ import annotations

import time
from collections.abc import Callable

from loguru import logger


def interruptible_sleep(
    seconds: int,
    should_stop: Callable[[], bool] | None = None,
    stop_message: str = "收到停止信号，中断等待",
) -> bool:
    """Sleep in one-second chunks and return True when interrupted."""
    elapsed = 0
    while elapsed < seconds:
        if should_stop and should_stop():
            logger.info(stop_message)
            return True
        time.sleep(min(1, seconds - elapsed))
        elapsed += 1
    return False
