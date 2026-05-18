"""
validation/regulatory/cyber_rules.py
---------------------------------------
Cybersecurity compliance rules (ISO 27001 / DORA / NIS2).

Rules
-----
SensitiveFieldEncryptionRule  : High-sensitivity columns should have _encrypted flag
AccessLogCompletenessRule     : Audit logs need access_time, user_id, action triad
ICTIncidentFieldRule          : DORA Art. 19 — incidents need severity, rto_minutes, detection_time
DataClassificationLabelRule   : Datasets should have a 'classification' column
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("dipex.validation.regulatory.cyber")

_SENSITIVE_COL_HINTS = {"ssn", "passport", "national_id", "bank_account", "credit_card",
                         "password", "api_key", "secret_key", "private_key", "access_token"}
_VALID_CLASSIFICATIONS = {"public", "internal", "confidential", "secret",
                           "restricted", "highly_confidential", "top_secret"}
_ACCESS_LOG_REQUIRED = {"access_time", "user_id", "action"}
_INCIDENT_REQUIRED = {"severity", "rto_minutes", "detection_time"}


class SensitiveFieldEncryptionRule:
    """
    Verify that high-sensitivity columns have corresponding _encrypted flag.
    ISO 27001 Annex A.8.2.3 / DORA Art. 9.
    """
    name = "CYBER_SENSITIVE_FIELD_ENCRYPTION"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        for col in df.columns:
            col_lower = col.lower()
            if any(h in col_lower for h in _SENSITIVE_COL_HINTS):
                encrypted_col = f"{col}_encrypted"
                if encrypted_col not in df.columns and f"{col_lower}_encrypted" not in [c.lower() for c in df.columns]:
                    violations.append({
                        "rule": self.name,
                        "severity": self.severity,
                        "column": col,
                        "message": f"Sensitive column '{col}' detected but no '{encrypted_col}' flag found. "
                                   "ISO 27001 A.8.2.3 / DORA Art. 9 require encryption indicators.",
                        "what_it_means": f"There is no evidence that '{col}' data is encrypted at rest.",
                        "why_it_matters": "Unencrypted sensitive data violates ISO 27001 and DORA resilience requirements.",
                        "recommended_action": f"Encrypt '{col}' and add '{encrypted_col}' boolean flag to track encryption status.",
                        "affected_rows": df[col].notna().sum(),
                    })
        return violations


class AccessLogCompletenessRule:
    """
    Check that access/audit logs have the required triad: access_time, user_id, action.
    ISO 27001 A.12.4.1 / NIS2 Art. 21.
    """
    name = "CYBER_ACCESS_LOG_COMPLETENESS"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        log_hints = {"log", "audit", "access_log", "event_log", "activity_log"}
        is_log_data = any(
            any(h in c.lower() for h in log_hints)
            for c in df.columns
        )
        if not is_log_data:
            return violations

        cols_lower = {c.lower() for c in df.columns}
        for field in _ACCESS_LOG_REQUIRED:
            if not any(field in c for c in cols_lower):
                violations.append({
                    "rule": self.name,
                    "severity": self.severity,
                    "column": f"missing:{field}",
                    "message": f"Access log data detected but required field '{field}' is missing. "
                               "ISO 27001 A.12.4.1 requires complete audit trails.",
                    "what_it_means": f"Audit logs are incomplete — '{field}' cannot be determined for access events.",
                    "why_it_matters": "Incomplete logs prevent forensic investigation and fail NIS2 Art. 21 requirements.",
                    "recommended_action": f"Ensure all access events include '{field}' field in the logging pipeline.",
                    "affected_rows": len(df),
                })
        return violations


class ICTIncidentFieldRule:
    """
    DORA Art. 19: ICT incident reports must include severity, rto_minutes, detection_time.
    """
    name = "DORA_ICT_INCIDENT_FIELDS"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        incident_hints = {"incident", "outage", "disruption", "breach", "ict_event"}
        is_incident_data = any(
            any(h in c.lower() for h in incident_hints) for c in df.columns
        )
        if not is_incident_data:
            return violations

        cols_lower = {c.lower() for c in df.columns}
        for field in _INCIDENT_REQUIRED:
            if not any(field in c for c in cols_lower):
                violations.append({
                    "rule": self.name,
                    "severity": self.severity,
                    "column": f"missing:{field}",
                    "message": f"ICT incident data detected but DORA-required field '{field}' is missing. "
                               "DORA Art. 19 mandates severity, RTO, and detection time in incident reports.",
                    "what_it_means": f"'{field}' is required in all ICT incident reports under DORA.",
                    "why_it_matters": "Incomplete DORA incident reporting triggers regulatory scrutiny from EBA/EIOPA.",
                    "recommended_action": f"Add '{field}' to all incident records per DORA Art. 19 guidelines.",
                    "affected_rows": len(df),
                })
        return violations


class DataClassificationLabelRule:
    """
    Check that datasets have a 'classification' column with valid labels.
    Required by ISO 27001 A.8.2.1 and most enterprise data governance frameworks.
    """
    name = "CYBER_DATA_CLASSIFICATION_LABEL"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        class_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"classification", "data_class", "sensitivity_level", "data_classification"})),
            None
        )
        if not class_col:
            violations.append({
                "rule": self.name,
                "severity": self.severity,
                "column": "missing:classification",
                "message": "No 'classification' column found. ISO 27001 A.8.2.1 requires data labeling.",
                "what_it_means": "Data assets cannot be classified, preventing appropriate security controls from being applied.",
                "why_it_matters": "Unclassified data may receive insufficient protection or over-protection.",
                "recommended_action": "Add 'classification' column with values: PUBLIC, INTERNAL, CONFIDENTIAL, SECRET.",
                "affected_rows": 0,
            })
        else:
            try:
                values = df[class_col].dropna().astype(str).str.lower().str.strip()
                invalid = values[~values.isin(_VALID_CLASSIFICATIONS)]
                if len(invalid) > 0:
                    violations.append({
                        "rule": self.name,
                        "severity": self.severity,
                        "column": class_col,
                        "message": f"{len(invalid)} row(s) have invalid classification labels: "
                                   f"{list(invalid.unique()[:5])}.",
                        "what_it_means": "Non-standard classification labels prevent automated policy enforcement.",
                        "why_it_matters": "Inconsistent labels break DLP (Data Loss Prevention) systems.",
                        "recommended_action": f"Standardize to: {sorted(_VALID_CLASSIFICATIONS)}.",
                        "affected_rows": len(invalid),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("DataClassificationLabelRule error: %s", exc)
        return violations
