"""
proposal/proposers/encoding_proposer.py
---------------------------------------
Suggests encoding strategies for categorical features.

This proposer analyses cardinality and recommends:
  - one_hot for very low-cardinality categoricals
  - ordinal / label encoding for medium-cardinality
  - target / frequency encoding for high-cardinality
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import logging

from .base_proposer import BaseProposer

logger = logging.getLogger(__name__)


class EncodingProposer(BaseProposer):
    """
    Generates encoding recommendations for each categorical column.
    """

    def propose(self, df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Returns:
            {
              "encoding_candidates": [
                {
                  "column": str,
                  "cardinality": int,
                  "cardinality_ratio": float,
                  "recommended_encoding": str,
                  "alternatives": [str, ...],
                  "rationale": str,
                },
                ...
              ]
            }
        """
        if df is None or df.empty:
            return {"error": "Empty DataFrame — no encodings suggested."}

        n_rows = len(df)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        candidates: List[Dict[str, Any]] = []

        for col in cat_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            cardinality = int(series.nunique())
            ratio = cardinality / float(n_rows) if n_rows else 0.0

            if cardinality <= 10 and ratio <= 0.1:
                recommended = "one_hot"
                alternatives = ["ordinal"]
                rationale = (
                    "Very low cardinality categorical — one-hot encoding is safe and interpretable."
                )
            elif cardinality <= 100 and ratio <= 0.5:
                recommended = "ordinal"
                alternatives = ["one_hot", "target_encoding"]
                rationale = (
                    "Medium cardinality — ordinal/label encoding keeps dimensionality reasonable; "
                    "target encoding is an option with leakage controls."
                )
            else:
                recommended = "target_encoding"
                alternatives = ["frequency_encoding", "hashing"]
                rationale = (
                    "High-cardinality categorical — target or frequency encoding recommended to "
                    "avoid explosive one-hot dimensionality."
                )

            candidates.append(
                {
                    "column": col,
                    "cardinality": cardinality,
                    "cardinality_ratio": round(ratio, 6),
                    "recommended_encoding": recommended,
                    "alternatives": alternatives,
                    "rationale": rationale,
                }
            )

        status = "CANDIDATES_COLLECTED" if candidates else "NO_CANDIDATES"
        return {
            "encoding_candidates": candidates,
            "status": status,
        }

