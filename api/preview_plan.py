"""
api/preview_plan.py
---------------------
Lightweight Pre-Analysis Approval endpoint.

GET  /api/pipeline/preview-plan  — returns the analysis plan (schema scan only, no ML)
POST /api/pipeline/preview-plan  — submits form data, returns plan in 2-5 seconds

The plan shows exactly what will happen before the user commits to a full run.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("dipex.api.preview_plan")


def generate_preview_plan(
    df: pd.DataFrame,
    domain: str = "generic",
    target_col: Optional[str] = None,
    mode: str = "auto",
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a pre-analysis plan from a DataFrame sample (no ML training).

    Parameters
    ----------
    df          : DataFrame (or sample) to inspect
    domain      : regulatory domain selected by user
    target_col  : target column for supervised ML (optional)
    mode        : analysis mode (auto/classification/regression/unsupervised)
    user_context: analyst context string

    Returns
    -------
    dict with plan sections: data_summary, domain_rules, operations, warnings
    """
    t0 = time.perf_counter()
    n_rows, n_cols = df.shape

    # ── Data Summary ──────────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    high_null_cols = [col for col, cnt in null_counts.items() if cnt / max(n_rows, 1) > 0.90]
    likely_dropped = high_null_cols  # >90% null → auto-drop
    
    # Row quality
    row_null_fracs = df.isnull().mean(axis=1)
    quarantine_est = int((row_null_fracs > 0.80).sum())

    # Duplicate detection (sample)
    dup_count = int(df.duplicated().sum()) if n_rows <= 100_000 else 0
    
    # Null rate
    overall_null_pct = float(df.isnull().values.mean() * 100)

    # Numeric vs categorical
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # ── Domain rules detection ────────────────────────────────────────────────
    detected_domains: List[str] = []
    try:
        from validation.regulatory.auto_domain_detector import detect_domains
        detected_domains = detect_domains(df)
    except Exception:
        pass

    active_domain = domain if domain and domain != "generic" else (
        detected_domains[0] if detected_domains else "generic"
    )

    domain_rules_map = {
        "banking":    ["AML Threshold Check", "Positive Amount Rule", "LTV/Loan Ratio", "BCBS239 Currency Concentration"],
        "healthcare": ["PHI Presence Scan", "Age Range Validation", "Diagnosis Code Format", "HIPAA Encryption Check"],
        "finance":    ["Revenue Recognition Rule", "Capital Adequacy (CAR ≥ 8%)", "Double-Entry Balance"],
        "pci_dss":    ["PAN Detection (Luhn)", "CVV Storage Check", "Cardholder Data Isolation"],
        "gdpr":       ["Data Residency Check", "Consent Required Rule"],
        "esg":        ["Carbon Emissions Validity", "ESG Score Range"],
        "fatf":       ["Structuring Detection", "PEP Screening", "Sanctions Check"],
        "generic":    [],
    }
    active_rules = domain_rules_map.get(active_domain, [])

    # ── Operations to perform ─────────────────────────────────────────────────
    operations = []

    # Imputation
    null_cols_moderate = [col for col, cnt in null_counts.items()
                          if 0 < cnt / max(n_rows, 1) <= 0.90]
    if null_cols_moderate:
        operations.append({
            "op": "null_imputation",
            "label": "Null Imputation",
            "detail": f"KNN (numeric) · Mode (categorical) for {len(null_cols_moderate)} column(s)",
            "status": "planned",
        })

    # Outlier detection
    if numeric_cols:
        operations.append({
            "op": "outlier_detection",
            "label": "Outlier Detection",
            "detail": "IQR winsorize at 1%/99% percentile",
            "status": "planned",
        })

    # PII scan
    operations.append({
        "op": "pii_scan",
        "label": "PII Scan",
        "detail": "Email, SSN, credit card, phone number detection",
        "status": "planned",
    })

    # Regulatory compliance
    if active_domain != "generic":
        operations.append({
            "op": "regulatory_compliance",
            "label": f"{active_domain.upper()} Compliance Check",
            "detail": f"{len(active_rules)} rule(s): {', '.join(active_rules[:3])}{'...' if len(active_rules) > 3 else ''}",
            "status": "planned",
        })

    # ML modeling
    if target_col and target_col in df.columns:
        n_classes = 0
        try:
            n_classes = int(df[target_col].nunique())
        except Exception:
            pass
        ml_type = "classification" if n_classes <= 20 and n_classes >= 2 else "regression"
        operations.append({
            "op": "automl",
            "label": f"AutoML ({ml_type.capitalize()})",
            "detail": "XGBoost + LightGBM + Random Forest · Stratified 5-fold CV",
            "status": "planned",
        })
    else:
        operations.append({
            "op": "unsupervised",
            "label": "Unsupervised Analysis",
            "detail": "Isolation Forest anomaly detection + K-Means clustering",
            "status": "planned",
        })

    # ── Warnings ──────────────────────────────────────────────────────────────
    warnings = []

    if likely_dropped:
        warnings.append({
            "level": "warning",
            "message": f"{len(likely_dropped)} column(s) will be AUTO-DROPPED (>90% null): {', '.join(likely_dropped[:5])}",
        })

    if quarantine_est > 0:
        pct = quarantine_est / max(n_rows, 1) * 100
        warnings.append({
            "level": "warning",
            "message": f"~{quarantine_est} rows may be QUARANTINED (>80% null per row) — {pct:.1f}% of data",
        })

    if dup_count > 0:
        warnings.append({
            "level": "info",
            "message": f"{dup_count} duplicate row(s) detected — will be removed before modeling",
        })

    if overall_null_pct > 20:
        warnings.append({
            "level": "warning",
            "message": f"High overall null rate: {overall_null_pct:.1f}% — consider investigating data collection pipeline",
        })

    if detected_domains and domain == "generic":
        warnings.append({
            "level": "info",
            "message": f"Auto-detected domain(s): {', '.join(detected_domains)} — rules will be applied",
        })

    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "data_summary": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "numeric_cols": len(numeric_cols),
            "categorical_cols": len(categorical_cols),
            "overall_null_pct": round(overall_null_pct, 1),
            "columns_to_drop": likely_dropped,
            "rows_to_quarantine_est": quarantine_est,
            "duplicate_rows": dup_count,
            "target_col": target_col,
        },
        "domain": {
            "selected": domain,
            "detected": detected_domains,
            "active": active_domain,
            "rules_count": len(active_rules),
            "rules": active_rules,
        },
        "operations": operations,
        "warnings": warnings,
        "user_context": user_context or "",
        "plan_elapsed_ms": round(elapsed, 1),
    }
