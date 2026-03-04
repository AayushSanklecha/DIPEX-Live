"""
proposal/proposers/window_proposer.py
-------------------------------------
Suggests window and watermark sizes for streaming analytics.

This proposer inspects timestamp columns (or columns that look like
timestamps) to estimate typical inter-arrival times and then generates
candidate tumbling / sliding windows and watermarks.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import logging

from .base_proposer import BaseProposer

logger = logging.getLogger(__name__)


class WindowProposer(BaseProposer):
    """
    Generates candidate window configurations for streaming workloads.
    """

    def propose(self, df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Returns:
            {
              "window_candidates": [
                {
                  "timestamp_column": str,
                  "median_interarrival_seconds": float,
                  "tumbling_window_seconds": int,
                  "sliding_window_seconds": int,
                  "watermark_seconds": int,
                  "rationale": str,
                },
                ...
              ]
            }
        """
        if df is None or df.empty:
            return {"error": "Empty DataFrame — no window sizes suggested."}

        ts_candidates: List[str] = []

        # Prefer native datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                ts_candidates.append(col)

        # Heuristic: columns whose name suggests time, if not already datetime
        if not ts_candidates:
            for col in df.columns:
                name = str(col).lower()
                if any(token in name for token in ("time", "timestamp", "date")):
                    ts_candidates.append(col)

        if not ts_candidates:
            return {
                "window_candidates": [],
                "status": "NO_CANDIDATES",
                "message": "No timestamp-like columns found.",
            }

        candidates: List[Dict[str, Any]] = []

        for col in ts_candidates:
            series = df[col]
            if not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, errors="coerce", utc=True)

            series = series.dropna().sort_values()
            if len(series) < 3:
                continue

            deltas = series.diff().dropna().dt.total_seconds()
            if deltas.empty:
                continue

            median_delta = float(np.median(deltas))
            # Guard rails for degenerate cases
            if not np.isfinite(median_delta) or median_delta <= 0:
                median_delta = 60.0

            # Propose windows as simple multiples of the median inter-arrival time
            tumbling = int(max(10.0, min(median_delta * 5.0, 3600.0)))
            sliding = int(max(5.0, min(median_delta * 2.0, tumbling)))
            watermark = int(max(tumbling, tumbling * 3.0))

            candidates.append(
                {
                    "timestamp_column": col,
                    "median_interarrival_seconds": round(median_delta, 3),
                    "tumbling_window_seconds": tumbling,
                    "sliding_window_seconds": sliding,
                    "watermark_seconds": watermark,
                    "rationale": (
                        f"Median inter-arrival for '{col}' is ~{median_delta:.1f}s; "
                        f"proposing tumbling window ≈5×, sliding ≈2×, watermark ≈3× tumbling "
                        "to balance latency and stability."
                    ),
                }
            )

        status = "CANDIDATES_COLLECTED" if candidates else "NO_CANDIDATES"
        return {
            "window_candidates": candidates,
            "status": status,
        }

