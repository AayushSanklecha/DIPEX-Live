from __future__ import annotations

"""
Structured logger for the reporting service (STEP 10).

Logs:
  - correlation_id
  - dataset_hash (if available)
  - confidence_score
  - qa_passed
  - model_name
  - tokens_used
  - validation_status
  - timestamp

Output is structured JSON lines, typically co-located with the cost tracker log.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import LOG_FILE


def log_event(
    correlation_id: str,
    dataset_hash: Optional[str],
    confidence_score: float,
    qa_passed: bool,
    model_name: str,
    tokens_used: int,
    validation_status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    record: Dict[str, Any] = {
        "correlation_id": correlation_id,
        "dataset_hash": dataset_hash,
        "confidence_score": confidence_score,
        "qa_passed": qa_passed,
        "model_name": model_name,
        "tokens_used": tokens_used,
        "validation_status": validation_status,
        "timestamp": time.time(),
    }
    if extra:
        record.update(extra)

    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

