from __future__ import annotations

"""
Cost & token tracker (STEP 8).

Tracks:
  - request_id
  - timestamp
  - prompt_length
  - response_length
  - total_tokens
  - estimated_cost (simulated, for future API switch)

Appends structured JSON to the configured LOG_FILE.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .config import LOG_FILE


def log_usage(
    prompt: str,
    response_text: str,
    model_name: str,
    request_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    request_id = request_id or str(uuid.uuid4())
    ts = time.time()

    prompt_len = len((prompt or "").split())
    resp_len = len((response_text or "").split())
    total_tokens = prompt_len + resp_len

    # Simulated cost: placeholder until a real pricing model is plugged in.
    cost_per_1k = 0.000001
    estimated_cost = (total_tokens / 1000.0) * cost_per_1k

    record: Dict[str, Any] = {
        "request_id": request_id,
        "timestamp": ts,
        "model_name": model_name,
        "prompt_length": prompt_len,
        "response_length": resp_len,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
    }
    if extra:
        record.update(extra)

    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record

