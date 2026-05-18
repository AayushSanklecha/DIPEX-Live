# tests/test_pii_governance.py
"""
Issue 04: Core test — PII columns MUST be dropped before data reaches the ML layer.
This test is non-legacy and must pass 100% in CI.

Tests the apply_pii_governance() utility function that drops PII-flagged columns
and writes an audit record.
"""

import json
import os
import tempfile

import pandas as pd
import pytest


# ── Helper: PII governance function (importable in pipeline_bridge too) ──────

def _write_pii_audit_entry(
    columns_removed: list[str],
    regulatory_domain: str,
    run_id: str,
    audit_path: str = "audit.jsonl",
) -> None:
    """
    Writes a PII-drop audit record to the append-only JSONL audit trail.
    Every call appends a new timestamped entry — intentional for full audit lineage.
    """
    from datetime import datetime, timezone

    entry = {
        "event": "pii_columns_dropped",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regulatory_domain": regulatory_domain,
        "columns_removed": columns_removed,
        "column_count": len(columns_removed),
    }
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def apply_pii_governance(
    df: pd.DataFrame,
    cols_to_drop: list[str],
    run_id: str,
    regulatory_domain: str = "unknown",
    audit_path: str = "audit.jsonl",
) -> pd.DataFrame:
    """
    Drops PII-flagged columns from the DataFrame and writes an audit entry.
    Returns a copy with PII columns removed.
    """
    import logging

    logger = logging.getLogger("dipex.pii_governance")

    if not cols_to_drop:
        logger.info("PII scan complete — no columns flagged for removal.")
        return df

    # Only drop columns that actually exist in the dataframe
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    if not existing_cols:
        logger.info("PII columns %s not found in dataframe — nothing to drop.", cols_to_drop)
        return df

    logger.info(
        "PII/compliance columns flagged for removal: %s — dropping now.",
        existing_cols,
    )
    df = df.drop(columns=existing_cols)

    # Write to audit trail — every drop is tracked
    _write_pii_audit_entry(
        columns_removed=existing_cols,
        regulatory_domain=regulatory_domain,
        run_id=run_id,
        audit_path=audit_path,
    )

    return df


# ── Tests ────────────────────────────────────────────────────────────────────


def make_pii_dataframe() -> pd.DataFrame:
    """Sample dataframe with known PII columns."""
    return pd.DataFrame({
        "account_id": [1, 2, 3],
        "patient_name": ["Alice", "Bob", "Charlie"],      # PII
        "ssn": ["123-45-6789", "987-65-4321", "111-22-3333"],  # PII
        "age": [34, 45, 29],
        "loan_amount": [5000, 12000, 8500],
        "target_churn": [0, 1, 0],
    })


def test_pii_columns_are_dropped_before_ml():
    """
    Given a dataframe with known PII columns,
    when the governance layer runs,
    then PII columns MUST NOT appear in the output dataframe.
    """
    df = make_pii_dataframe()
    pii_columns = ["patient_name", "ssn"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        audit_path = f.name

    try:
        result_df = apply_pii_governance(
            df, cols_to_drop=pii_columns, run_id="test-001", audit_path=audit_path,
        )

        for col in pii_columns:
            assert col not in result_df.columns, (
                f"CRITICAL: PII column '{col}' still present after governance layer. "
                "This is a data privacy violation."
            )

        # Non-PII columns must be preserved
        assert "account_id" in result_df.columns
        assert "loan_amount" in result_df.columns
        assert "age" in result_df.columns

        # Audit must have been written
        with open(audit_path) as af:
            audit_line = af.readline()
            entry = json.loads(audit_line)
            assert entry["event"] == "pii_columns_dropped"
            assert set(entry["columns_removed"]) == set(pii_columns)
            assert entry["run_id"] == "test-001"
    finally:
        os.unlink(audit_path)


def test_empty_pii_list_does_not_crash():
    """Edge case: if no PII columns are flagged, the dataframe passes through unchanged."""
    df = make_pii_dataframe()
    result_df = apply_pii_governance(df, cols_to_drop=[], run_id="test-002")
    assert result_df.shape == df.shape
