"""
scripts/_demo_setup.py
──────────────────────
Shared setup for all demo scripts.
Configures environment for clean but honest output — errors are handled,
not suppressed. Only suppresses non-actionable pandas deprecation warnings.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def configure_demo_environment() -> None:
    """Set up clean logging for demo — shows important pipeline messages."""
    # Only suppress non-actionable pandas deprecation/future warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Set logging to show INFO for the pipeline, suppress DEBUG spam
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
    )

    # ── Set default DB credentials for Docker ─────────────────────────
    os.environ.setdefault("POSTGRES_USER", "dipex")
    os.environ.setdefault("POSTGRES_PASSWORD", "dipex_demo")
    os.environ.setdefault("DB_USER", "dipex")
    os.environ.setdefault("DB_PASS", "dipex_demo")
