"""
validation/shap_explainer.py
----------------------------
SHAP-based column risk explainer for Hard Gate 1 failures.

Purpose
-------
When Hard Gate 1 REJECTs a dataset, this module uses SHAP to identify
*which columns* contributed most to the failure signal — giving analysts
an actionable "fix these columns first" recommendation.

How it works
------------
1. Converts the rejected DataFrame to numeric (encode categoricals)
2. Trains a lightweight IsolationForest to score each row's anomaly level
3. Uses SHAP TreeExplainer to compute per-column SHAP values
4. Returns a ranked list of columns by their mean absolute SHAP impact

This is a post-hoc explainer — it does NOT change the gate decision.
It only adds human-readable insight to the GateResult.

Usage
-----
    from validation.shap_explainer import explain_gate_failure

    shap_result = explain_gate_failure(df, run_id="abc123", top_n=5)
    # shap_result["top_risk_columns"] → [{"column": "age", "shap_impact": 0.43}, ...]
    # shap_result["explanation"] → "Columns most responsible for quality issues: age, salary, ..."
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MAX_ROWS = 2000   # cap for SHAP computation speed


def explain_gate_failure(
    df: pd.DataFrame,
    run_id: str = "N/A",
    top_n: int = 5,
    failures: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute SHAP-based column risk scores for a rejected dataset.

    Parameters
    ----------
    df       : The DataFrame that failed Hard Gate 1
    run_id   : Pipeline run identifier (for logging)
    top_n    : Number of top risk columns to return
    failures : List of gate failure dicts (used to pre-rank known bad columns)

    Returns
    -------
    {
      "run_id": str,
      "method": "shap_tree" | "shap_linear" | "column_null_rank",
      "top_risk_columns": [
          {"column": "col_name", "shap_impact": 0.43, "rank": 1},
          ...
      ],
      "explanation": str,   # ready to embed in gate failure report
      "shap_available": bool,
    }
    """
    logger.info("[SHAP] Explaining gate failure for run_id=%s shape=%s", run_id, df.shape)

    # ── Pre-extract known bad columns from failures ───────────────────────────
    known_bad: set[str] = set()
    if failures:
        for f in failures:
            col = f.get("column") or f.get("field") or f.get("col")
            if col and col in df.columns:
                known_bad.add(col)

    # ── Prepare numeric matrix ────────────────────────────────────────────────
    try:
        X, col_names = _to_numeric(df)
        if X.shape[0] > _MAX_ROWS:
            idx = np.random.default_rng(42).choice(X.shape[0], _MAX_ROWS, replace=False)
            X = X[idx]
        if X.shape[1] == 0:
            return _null_rank_fallback(df, run_id, top_n, known_bad)
    except Exception as exc:
        logger.warning("[SHAP] Numeric prep failed: %s", exc)
        return _null_rank_fallback(df, run_id, top_n, known_bad)

    # ── Try SHAP with IsolationForest ─────────────────────────────────────────
    try:
        import shap
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(n_estimators=80, contamination="auto", random_state=42)
        iso.fit(X)

        explainer  = shap.TreeExplainer(iso)
        shap_vals  = explainer.shap_values(X)          # shape: (n_rows, n_cols)
        mean_impact = np.abs(shap_vals).mean(axis=0)   # per-column mean |SHAP|

        ranked = sorted(
            zip(col_names, mean_impact.tolist()),
            key=lambda x: x[1], reverse=True
        )[:top_n]

        top_cols = [
            {"column": c, "shap_impact": round(v, 4), "rank": i + 1}
            for i, (c, v) in enumerate(ranked)
        ]
        col_str = ", ".join(c["column"] for c in top_cols)
        explanation = (
            f"SHAP analysis identified {len(top_cols)} columns with "
            f"highest anomaly contribution: {col_str}. "
            f"Review and fix these columns to pass Hard Gate 1."
        )
        if known_bad:
            known_str = ", ".join(sorted(known_bad))
            explanation += f" Deterministic failures flagged: {known_str}."

        logger.info("[SHAP] Top risk columns: %s", col_str)

        return {
            "run_id":           run_id,
            "method":           "shap_tree",
            "top_risk_columns": top_cols,
            "explanation":      explanation,
            "shap_available":   True,
        }

    except ImportError:
        logger.warning("[SHAP] shap package not installed — using null-rate fallback")
    except Exception as exc:
        logger.warning("[SHAP] TreeExplainer failed: %s — using null-rate fallback", exc)

    return _null_rank_fallback(df, run_id, top_n, known_bad)


# ── Fallback: rank by null rate + known failures ──────────────────────────────

def _null_rank_fallback(
    df: pd.DataFrame,
    run_id: str,
    top_n: int,
    known_bad: set[str],
) -> Dict[str, Any]:
    """
    Fallback when SHAP is unavailable.
    Ranks columns by null rate + flags known bad columns from gate failures.
    """
    null_rates = df.isnull().mean().sort_values(ascending=False)

    # Boost known_bad columns to top
    def _score(col: str, rate: float) -> float:
        return rate + (0.5 if col in known_bad else 0.0)

    ranked = sorted(
        ((col, _score(col, rate), rate) for col, rate in null_rates.items()),
        key=lambda x: x[1], reverse=True
    )[:top_n]

    top_cols = [
        {"column": col, "shap_impact": round(null_rate, 4), "rank": i + 1}
        for i, (col, _, null_rate) in enumerate(ranked)
    ]
    col_str = ", ".join(c["column"] for c in top_cols)
    return {
        "run_id":           run_id,
        "method":           "column_null_rank",
        "top_risk_columns": top_cols,
        "explanation":      f"Columns with highest failure risk (by null rate): {col_str}.",
        "shap_available":   False,
    }


# ── Numeric conversion ────────────────────────────────────────────────────────

def _to_numeric(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Convert DataFrame to a clean numeric matrix for SHAP."""
    from sklearn.preprocessing import LabelEncoder

    out = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            out[col] = series.fillna(series.median() if not series.isna().all() else 0)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            le = LabelEncoder()
            out[col] = le.fit_transform(series.astype(str).fillna("__NA__"))
        # skip datetime etc.

    if not out:
        return np.empty((len(df), 0)), []

    mat = np.column_stack([out[c].values for c in out])
    return mat.astype(np.float32), list(out.keys())
