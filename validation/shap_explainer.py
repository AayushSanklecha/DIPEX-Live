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


# ─────────────────────────────────────────────────────────────────────────────
# SHAP × Compliance Cross-Reference
# ─────────────────────────────────────────────────────────────────────────────

def explain_compliance_violations(
    df: pd.DataFrame,
    violations: list,
    run_id: str = "N/A",
    top_n: int = 10,
) -> list:
    """
    Cross-references SHAP column-level anomaly scores with regulatory violations.

    For each regulatory violation, enriches the record with:
      - ``shap_impact``: the column's mean-absolute SHAP anomaly score
      - ``risk_rank``:   the column's rank by combined risk signal
                         (SHAP impact × severity weight)

    This tells analysts: "column X breaches the AML rule AND is the #1
    driver of anomaly signal — fix this one first."

    Parameters
    ----------
    df         : The dataset being evaluated
    violations : List of RegulatoryViolation objects (or dicts with .column / .severity)
    run_id     : Run identifier for logging
    top_n      : How many combined-risk columns to return (sorted by risk = SHAP × weight)

    Returns
    -------
    List of dicts, each with:
        {column, severity, rule_name, shap_impact, severity_weight, combined_risk, risk_rank}
    Sorted descending by combined_risk.
    """
    import logging as _logging
    _log = _logging.getLogger("dipex.shap.compliance")

    if df is None or df.empty or not violations:
        return []

    # ── Get SHAP column risk scores ───────────────────────────────────────────
    try:
        shap_result = explain_gate_failure(df, run_id=run_id, top_n=len(df.columns))
        shap_scores = {
            item["column"]: item["shap_impact"]
            for item in shap_result.get("top_risk_columns", [])
        }
    except Exception as exc:  # noqa: BLE001
        _log.debug("SHAP computation failed in explain_compliance_violations: %s", exc)
        shap_scores = {}

    # ── Severity weights for combined risk score ──────────────────────────────
    _SEVERITY_WEIGHT = {"CRITICAL": 3.0, "ERROR": 2.0, "WARNING": 1.0}

    # ── Build per-column risk records ─────────────────────────────────────────
    seen_cols: set = set()
    records = []

    for v in violations:
        # Support both RegulatoryViolation objects and dicts
        if hasattr(v, "column"):
            col, severity, rule_name = v.column, v.severity, v.rule_name
        else:
            col = v.get("column", "N/A")
            severity = v.get("severity", "WARNING")
            rule_name = v.get("rule_name", "unknown")

        if col in ("N/A", "") or col in seen_cols:
            continue
        seen_cols.add(col)

        shap_impact = shap_scores.get(col, 0.0)
        sev_weight  = _SEVERITY_WEIGHT.get(severity, 1.0)
        combined    = shap_impact * sev_weight

        records.append({
            "column":           col,
            "severity":         severity,
            "rule_name":        rule_name,
            "shap_impact":      round(shap_impact, 5),
            "severity_weight":  sev_weight,
            "combined_risk":    round(combined, 5),
            "risk_rank":        None,  # filled below
        })

    # ── Sort by combined risk and assign ranks ────────────────────────────────
    records.sort(key=lambda r: r["combined_risk"], reverse=True)
    for rank, rec in enumerate(records[:top_n], start=1):
        rec["risk_rank"] = rank

    _log.info(
        "[%s] SHAP×Compliance: ranked %d violation columns; top=%s (combined_risk=%.4f)",
        run_id[:8],
        len(records),
        records[0]["column"] if records else "N/A",
        records[0]["combined_risk"] if records else 0.0,
    )

    return records[:top_n]
