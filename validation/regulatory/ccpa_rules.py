"""
validation/regulatory/ccpa_rules.py
--------------------------------------
CCPA (California Consumer Privacy Act) compliance rules.

Rules
-----
CalifonriaResidencyDisclosureRule : Verifies disclosure columns exist for CA residents
ConsumerRightsDeletionFieldRule   : 'deletion_request' and 'opt_out' fields presence
SaleOfDataFlagRule                : 'data_sale_consent' required if personal info sold
PIIInventoryRule                  : Validates PII columns have privacy labels
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("dipex.validation.regulatory.ccpa")

_CA_REGIONS = {"CA", "California", "US-CA", "california"}
_PII_COLUMNS = {"ssn", "email", "phone", "address", "name", "date_of_birth", "ip_address",
                "geolocation", "browsing_history", "biometric", "social_security"}


class CaliforniaResidencyDisclosureRule:
    """
    If California resident data is present, disclosure/notice columns must exist.
    CCPA § 1798.100: Right to know about personal information.
    """
    name = "CCPA_CA_RESIDENCY_DISCLOSURE"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        region_col = next(
            (c for c in df.columns if any(h in c.lower() for h in {"state", "region", "residency"})),
            None
        )
        if not region_col:
            return violations

        try:
            ca_rows = df[region_col].astype(str).isin(_CA_REGIONS).sum()
        except Exception:  # noqa: BLE001
            return violations

        if ca_rows == 0:
            return violations

        disclosure_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"privacy_notice", "disclosure_sent", "notice_at_collection", "privacy_policy_version"})),
            None
        )
        if not disclosure_col:
            violations.append({
                "rule": self.name,
                "severity": self.severity,
                "column": region_col,
                "message": f"{ca_rows} California resident record(s) found but no privacy disclosure column present. "
                           "CCPA §1798.100 requires notice at collection.",
                "what_it_means": "California residents are entitled to know what data is collected; no notice column found.",
                "why_it_matters": "CCPA non-compliance: $100/violation (unintentional) or $750/consumer/incident (breach).",
                "recommended_action": "Add 'privacy_notice_sent' column and ensure notices are delivered to CA residents.",
                "affected_rows": int(ca_rows),
            })
        return violations


class ConsumerRightsDeletionFieldRule:
    """
    Check for deletion_request and opt_out columns.
    CCPA § 1798.105 (Right to Delete) and § 1798.120 (Right to Opt-Out).
    """
    name = "CCPA_CONSUMER_RIGHTS_FIELDS"
    severity = "WARNING"

    REQUIRED = {
        "deletion_request": ["deletion_request", "right_to_delete", "delete_request_flag"],
        "opt_out": ["opt_out", "do_not_sell", "dsf", "optout", "ccpa_opt_out"],
    }

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        has_consumer_data = any(
            any(h in c.lower() for h in {"customer", "consumer", "user", "member"})
            for c in df.columns
        )
        if not has_consumer_data:
            return violations

        cols_lower = {c.lower() for c in df.columns}
        for right_name, hints in self.REQUIRED.items():
            if not any(any(h in c for c in cols_lower) for h in hints):
                violations.append({
                    "rule": self.name,
                    "severity": self.severity,
                    "column": f"missing:{right_name}",
                    "message": f"Consumer data present but CCPA required field '{right_name}' not found.",
                    "what_it_means": f"Cannot track consumer's '{right_name}' request per CCPA requirements.",
                    "why_it_matters": f"CCPA § 1798.105/120 mandates honoring {right_name} within 45 days.",
                    "recommended_action": f"Add '{right_name}' boolean column and implement a data subject rights workflow.",
                    "affected_rows": len(df),
                })
        return violations


class SaleOfDataFlagRule:
    """
    If data sharing/sale is inferred, 'data_sale_consent' must be present.
    CCPA § 1798.120: Right to opt-out of sale of personal information.
    """
    name = "CCPA_DATA_SALE_FLAG"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        sale_hints = {"partner_id", "third_party", "affiliate_id", "data_buyer"}
        has_sale_cols = any(any(h in c.lower() for h in sale_hints) for c in df.columns)
        if not has_sale_cols:
            return violations

        consent_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"data_sale_consent", "sale_opt_out", "do_not_sell"})),
            None
        )
        if not consent_col:
            violations.append({
                "rule": self.name,
                "severity": self.severity,
                "column": "missing:data_sale_consent",
                "message": "Data sharing/sale columns detected without 'data_sale_consent' tracking. "
                           "CCPA § 1798.120 requires opt-out tracking for sale of PI.",
                "what_it_means": "Personal information may be sold without recording consumer's opt-out status.",
                "why_it_matters": "Selling PI without consent tracking: $7,500/intentional violation under CCPA.",
                "recommended_action": "Implement 'data_sale_consent' field and Global Privacy Control (GPC) support.",
                "affected_rows": len(df),
            })
        return violations


class PIIInventoryRule:
    """
    Validate that PII columns are inventoried and privacy-labeled.
    CCPA requires businesses to maintain a record of categories of personal information collected.
    """
    name = "CCPA_PII_INVENTORY"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        detected_pii = [
            c for c in df.columns
            if any(h in c.lower() for h in _PII_COLUMNS)
        ]
        if not detected_pii:
            return violations

        labeling_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"data_category", "pi_category", "sensitivity_label", "pii_type"})),
            None
        )
        if not labeling_col:
            violations.append({
                "rule": self.name,
                "severity": self.severity,
                "column": ", ".join(detected_pii[:5]),
                "message": f"PII columns detected ({len(detected_pii)} total) but no PII category/inventory label column found. "
                           "CCPA requires businesses to disclose categories of PI collected.",
                "what_it_means": "Cannot generate a CCPA-compliant privacy notice or data map without PII categorization.",
                "why_it_matters": "CCPA mandates disclosure of PI categories; missing inventory = non-compliant privacy policy.",
                "recommended_action": "Add 'pi_category' column and maintain a data inventory mapping each PII column to CCPA category.",
                "affected_rows": len(df),
            })
        return violations
