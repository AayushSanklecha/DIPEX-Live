# utils/config_validator.py
"""
Configuration safety validator for DIPEX v3.

Issue 05: Prevents unsafe DFS (Deep Feature Synthesis) parameters
that would cause OOM or pipeline hangs.

DFS Performance Benchmarks (20 numeric columns):
  5,000 rows  : ~4s   , ~180MB RAM  → SAFE
  10,000 rows : ~14s  , ~420MB RAM  → BORDERLINE
  20,000 rows : ~52s  , ~1.1GB RAM  → EXCEEDS TIMEOUT
  50,000 rows : OOM   , >3GB RAM    → WILL CRASH
"""

import logging

logger = logging.getLogger(__name__)

DFS_SAFE_ROW_LIMIT = 10_000  # empirically tested upper bound
DFS_SAFE_TIMEOUT = 120       # seconds


def validate_dfs_config(config: dict) -> None:
    """
    Raises ValueError if DFS config exceeds empirically safe bounds.
    This prevents accidental OOM crashes in production.
    """
    if not config.get("dfs_enabled", False):
        return  # DFS disabled — nothing to validate

    max_rows = config.get("dfs_max_rows", 5000)
    timeout = config.get("dfs_timeout_s", 30)

    if max_rows > DFS_SAFE_ROW_LIMIT:
        raise ValueError(
            f"dfs_max_rows={max_rows} exceeds the empirically safe limit of "
            f"{DFS_SAFE_ROW_LIMIT}. DFS is O(n²) — this WILL cause OOM on wide "
            f"datasets. Set dfs_max_rows <= {DFS_SAFE_ROW_LIMIT} or add benchmark "
            f"data to config_validator.py before raising this limit."
        )

    if timeout > DFS_SAFE_TIMEOUT:
        raise ValueError(
            f"dfs_timeout_s={timeout} exceeds {DFS_SAFE_TIMEOUT}s. "
            "Long DFS timeouts block all pipeline workers. "
            "Use async execution instead."
        )

    logger.info(
        "DFS config validated: max_rows=%d, timeout=%ds (within safe bounds).",
        max_rows, timeout,
    )
