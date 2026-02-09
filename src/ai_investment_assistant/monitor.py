"""Monitoring and scheduling helpers."""

from __future__ import annotations

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)


def run_monitor(task: Callable[[], None], interval_seconds: int = 3600, cycles: int = 0) -> None:
    """Run a task on a fixed interval. Set cycles=0 for infinite loop."""
    iteration = 0
    while True:
        start = time.time()
        try:
            task()
            logger.info("Monitor task completed in %.2fs", time.time() - start)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Monitor task failed: %s", exc)
        iteration += 1
        if cycles and iteration >= cycles:
            break
        time.sleep(interval_seconds)
