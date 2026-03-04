"""
proposal/proposers/aggregation_proposer.py
------------------------------------------
Suggests aggregation hypotheses (group-bys and rollups).

This proposer is assistive-only: it does not execute aggregations, it
only recommends likely useful groupings and metrics based on dataset
shape and cardinality.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import logging

from .base_proposer import BaseProposer

logger = logging.getLogger(__name__)


class AggregationProposer(BaseProposer):
    """
    Generates candidate aggregation plans:

      - Low-cardinality categorical columns as group keys
      - Numeric columns as aggregation metrics
      - Standard rollup functions (count, sum, mean, min, max)
    """

    def propose(self, df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Returns:
            {
              "aggregation_candidates": [
                 {
                   "group_keys": [...],
                   "metrics": [...],
                   "aggregations": [...],
                   "rationale": str,
                 },
                 ...
              ],
              "status": "CANDIDATES_COLLECTED" | "NO_CANDIDATES",
            }
        """
        if df is None or df.empty:
            return {"error": "Empty DataFrame — no aggregations suggested."}

        n_rows = len(df)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not num_cols:
            return {
                "aggregation_candidates": [],
                "status": "NO_CANDIDATES",
                "message": "No numeric columns available for aggregation.",
            }

        group_key_candidates: List[str] = []
        for col in cat_cols:
            unique = df[col].nunique(dropna=True)
            if unique == 0:
                continue
            ratio = unique / float(n_rows)
            # Low-cardinality columns are ideal for rollups (status, region, etc.)
            if ratio <= 0.05:
                group_key_candidates.append(col)

        # Fallback: use at most 3 categorical columns as single-key groupings
        if not group_key_candidates:
            group_key_candidates = cat_cols[:3]

        agg_candidates: List[Dict[str, Any]] = []
        for key in group_key_candidates:
            agg_candidates.append(
                {
                    "group_keys": [key],
                    "metrics": num_cols,
                    "aggregations": ["count", "sum", "mean", "min", "max"],
                    "rationale": (
                        f"Column '{key}' has comparatively low cardinality and is "
                        "a strong candidate for group-by rollups over numeric metrics."
                    ),
                }
            )

        status = "CANDIDATES_COLLECTED" if agg_candidates else "NO_CANDIDATES"
        return {
            "aggregation_candidates": agg_candidates,
            "status": status,
        }

