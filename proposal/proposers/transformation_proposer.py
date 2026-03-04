"""
proposal/proposers/transformation_proposer.py
---------------------------------------------
Suggests feature transformation hypotheses (scaling, log transforms, binning).

This proposer is assistive-only and does not modify data.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import logging

from .base_proposer import BaseProposer

logger = logging.getLogger(__name__)


class TransformationProposer(BaseProposer):
    """
    Generates transformation suggestions for numeric features based
    on simple distributional heuristics (skewness, range).
    """

    def propose(self, df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Returns:
            {
              "transformation_candidates": [
                {
                  "column": str,
                  "suggestions": [ "log1p" | "standard_scale" | "robust_scale" | "bucketize", ...],
                  "skewness": float,
                  "min": float,
                  "max": float,
                  "rationale": str,
                },
                ...
              ]
            }
        """
        if df is None or df.empty:
            return {"error": "Empty DataFrame — no transformations suggested."}

        candidates: List[Dict[str, Any]] = []
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in num_cols:
            series = df[col].dropna()
            if series.empty or len(series) < 10:
                continue

            col_min = float(series.min())
            col_max = float(series.max())
            skew = float(series.skew()) if len(series) > 2 else 0.0
            std = float(series.std(ddof=1)) if len(series) > 1 else 0.0

            suggestions: List[str] = []
            rationale_parts: List[str] = []

            # Log-transform suggestion for positive, right-skewed distributions
            if col_min >= 0 and skew > 1.0:
                suggestions.append("log1p")
                rationale_parts.append(
                    "Right-skewed non-negative distribution — log1p can stabilise variance."
                )

            # Standard scaling for wide numeric ranges
            if std > 0 and (col_max - col_min) > 10 * std:
                suggestions.append("standard_scale")
                rationale_parts.append(
                    "Wide numeric range relative to standard deviation — standard scaling recommended."
                )

            # Robust scaling when heavy tails or strong outliers are present
            q1, q3 = np.percentile(series.to_numpy(dtype=float), [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                fence_low = q1 - 1.5 * iqr
                fence_high = q3 + 1.5 * iqr
                outlier_frac = float(
                    ((series < fence_low) | (series > fence_high)).mean()
                )
                if outlier_frac > 0.03:
                    suggestions.append("robust_scale")
                    rationale_parts.append(
                        f"{outlier_frac:.2%} of values lie outside Tukey IQR fences — "
                        "robust (median/IQR-based) scaling recommended."
                    )

            # Binning for quasi-continuous but low-information signals
            unique_ratio = series.nunique() / float(len(series))
            if 0.1 < unique_ratio < 0.5:
                suggestions.append("bucketize")
                rationale_parts.append(
                    "Intermediate cardinality — discretisation into buckets may improve tree-based models."
                )

            if not suggestions:
                continue

            # Deduplicate suggestions while preserving order
            seen = set()
            ordered_suggestions = []
            for s in suggestions:
                if s not in seen:
                    seen.add(s)
                    ordered_suggestions.append(s)

            candidates.append(
                {
                    "column": col,
                    "suggestions": ordered_suggestions,
                    "skewness": round(skew, 6),
                    "min": col_min,
                    "max": col_max,
                    "rationale": " ".join(rationale_parts),
                }
            )

        status = "CANDIDATES_COLLECTED" if candidates else "NO_CANDIDATES"
        return {
            "transformation_candidates": candidates,
            "status": status,
        }

