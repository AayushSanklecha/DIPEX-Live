"""
validation/regulatory/cross_domain_rules.py
---------------------------------------------
Cross-framework regulatory rules that span multiple compliance domains.

Rules implemented:
  1. GDPRDataResidencyRule   — GDPR Art. 44: data transfer outside allowed regions
  2. SOXAuditTrailRule       — SOX §404: every record must have audit timestamp + user
  3. HIPAAEncryptionFlagRule — HIPAA §164.312: records containing PHI must be flagged encrypted
  4. GDPRConsentRequiredRule — GDPR Art. 7: explicit consent must be True for rows containing PHI

All rules inherit BaseRegulatoryRule and produce RegulatoryViolation objects.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_rule import BaseRegulatoryRule, RegulatoryViolation

logger = logging.getLogger(__name__)

# ── PHI regex patterns (mirrors llm_provider._PII_PATTERNS) ──────────────────

_PHI_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                          # SSN
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"),  # phone
    re.compile(r"\b(?:NHS|MRN|DOB)[:\s#]?\s*[\d\-A-Za-z]+\b", re.I),  # health ID
]

def _contains_phi(text: str) -> bool:
    return any(p.search(str(text)) for p in _PHI_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GDPR Data Residency Rule
# ─────────────────────────────────────────────────────────────────────────────

class GDPRDataResidencyRule(BaseRegulatoryRule):
    """
    GDPR Article 44 — Transfers of personal data to third countries.

    Checks that a 'data_region' column (or equivalent) only contains
    allowed geographic regions. Any row with a region outside the allowed
    list represents a potential unlawful data transfer.
    """
    name = "gdpr_data_residency"
    domain = "gdpr"

    def __init__(
        self,
        residency_column: str = "data_region",
        allowed_regions: Optional[List[str]] = None,
    ) -> None:
        self.residency_column = residency_column
        self.allowed_regions = {r.upper() for r in (allowed_regions or ["EU", "EEA"])}

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self._col_missing(self.residency_column, df):
            return []

        series = df[self.residency_column].dropna().astype(str).str.upper()
        bad = (~series.isin(self.allowed_regions)).sum()
        if bad == 0:
            return []

        bad_regions = sorted(series[~series.isin(self.allowed_regions)].unique().tolist())
        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="CRITICAL",
            column=self.residency_column,
            offending_count=int(bad),
            message=(
                f"[GDPR Art. 44] {bad} record(s) contain data region(s) outside "
                f"the allowed set {sorted(self.allowed_regions)}. "
                f"Non-compliant regions found: {bad_regions}."
            ),
            remediation=(
                "Ensure personal data is only stored/processed within allowed regions. "
                "For transfers to non-adequate countries, establish Standard Contractual "
                "Clauses (SCCs) or Binding Corporate Rules (BCRs) per GDPR Art. 46."
            ),
        )]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SOX Audit Trail Rule
# ─────────────────────────────────────────────────────────────────────────────

class SOXAuditTrailRule(BaseRegulatoryRule):
    """
    Sarbanes-Oxley §404 — Internal control over financial reporting.

    Every financial record must have:
      - A non-null audit timestamp (when was this record last modified?)
      - A non-null audit user identifier (who modified it?)

    Missing audit fields indicate incomplete internal controls.
    """
    name = "sox_audit_trail"
    domain = "sox"

    def __init__(
        self,
        audit_timestamp_column: str = "modified_at",
        audit_user_column: str = "modified_by",
    ) -> None:
        self.audit_timestamp_column = audit_timestamp_column
        self.audit_user_column = audit_user_column

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        violations: List[RegulatoryViolation] = []

        for col, label in [
            (self.audit_timestamp_column, "audit timestamp"),
            (self.audit_user_column, "audit user"),
        ]:
            if col not in df.columns:
                # Missing column entirely — this is itself a violation
                violations.append(RegulatoryViolation(
                    rule_name=self.name,
                    domain=self.domain,
                    severity="ERROR",
                    column=col,
                    offending_count=len(df),
                    message=(
                        f"[SOX §404] Audit trail column '{col}' ({label}) is absent "
                        f"from the dataset. All financial records must carry a complete "
                        f"audit trail."
                    ),
                    remediation=(
                        f"Add column '{col}' to the data schema. Populate with "
                        f"timestamps/user IDs for every record modification."
                    ),
                ))
                continue

            null_count = int(df[col].isna().sum())
            if null_count > 0:
                violations.append(RegulatoryViolation(
                    rule_name=self.name,
                    domain=self.domain,
                    severity="ERROR",
                    column=col,
                    offending_count=null_count,
                    message=(
                        f"[SOX §404] {null_count} record(s) have a null {label} "
                        f"in column '{col}'. All financial records require a complete "
                        f"audit trail per Sarbanes-Oxley internal control requirements."
                    ),
                    remediation=(
                        f"Backfill null values in '{col}' using ETL audit metadata. "
                        f"Enforce NOT NULL constraint at the source system level."
                    ),
                ))

        return violations


# ─────────────────────────────────────────────────────────────────────────────
# 3. HIPAA Encryption Flag Rule
# ─────────────────────────────────────────────────────────────────────────────

class HIPAAEncryptionFlagRule(BaseRegulatoryRule):
    """
    HIPAA §164.312(a)(2)(iv) — Encryption and decryption (Addressable).
    HIPAA §164.312(e)(2)(ii) — Encryption in transmission (Addressable).

    Records containing PHI (Protected Health Information) must be flagged
    as encrypted. If an 'is_encrypted' column is present, PHI rows with
    is_encrypted != True are flagged as CRITICAL violations.
    """
    name = "hipaa_encryption_flag"
    domain = "hipaa"

    def __init__(
        self,
        phi_columns: Optional[List[str]] = None,
        encryption_flag_column: str = "is_encrypted",
    ) -> None:
        self.phi_columns = phi_columns or ["patient_id", "ssn", "date_of_birth", "full_name"]
        self.encryption_flag_column = encryption_flag_column

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self.encryption_flag_column not in df.columns:
            # No encryption column at all — warn but don't block if no PHI columns present
            phi_cols_present = [c for c in self.phi_columns if c in df.columns]
            if not phi_cols_present:
                return []
            return [RegulatoryViolation(
                rule_name=self.name,
                domain=self.domain,
                severity="CRITICAL",
                column=self.encryption_flag_column,
                offending_count=len(df),
                message=(
                    f"[HIPAA §164.312] Dataset contains PHI columns "
                    f"{phi_cols_present} but encryption flag column "
                    f"'{self.encryption_flag_column}' is absent. Cannot verify "
                    f"encryption compliance."
                ),
                remediation=(
                    f"Add boolean column '{self.encryption_flag_column}' to the dataset. "
                    f"Set True only for records where PHI fields are encrypted at rest."
                ),
            )]

        phi_cols_present = [c for c in self.phi_columns if c in df.columns]
        if not phi_cols_present:
            return []

        # Rows that have at least one PHI column non-null and are NOT encrypted
        has_phi_mask = df[phi_cols_present].notna().any(axis=1)
        not_encrypted = ~df[self.encryption_flag_column].astype(bool)
        bad = (has_phi_mask & not_encrypted).sum()

        if bad == 0:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="CRITICAL",
            column=self.encryption_flag_column,
            offending_count=int(bad),
            message=(
                f"[HIPAA §164.312] {bad} record(s) contain PHI in columns "
                f"{phi_cols_present} but '{self.encryption_flag_column}' is False/null. "
                f"Unencrypted PHI violates HIPAA Technical Safeguards."
            ),
            remediation=(
                "Apply field-level encryption to PHI columns before storage. "
                "Set is_encrypted=True only after verified encryption. "
                "Review BAA agreements with all downstream data processors."
            ),
        )]


# ─────────────────────────────────────────────────────────────────────────────
# 4. GDPR Consent Required Rule
# ─────────────────────────────────────────────────────────────────────────────

class GDPRConsentRequiredRule(BaseRegulatoryRule):
    """
    GDPR Article 7 — Conditions for consent.

    For rows that contain personal data (PHI-like columns), a 'consent_given'
    column must be True. Processing personal data without explicit consent
    is unlawful under GDPR Art. 6 and Art. 7.
    """
    name = "gdpr_consent_required"
    domain = "gdpr"

    def __init__(
        self,
        consent_column: str = "consent_given",
        phi_columns: Optional[List[str]] = None,
    ) -> None:
        self.consent_column = consent_column
        self.phi_columns = phi_columns or ["patient_id", "ssn", "email", "full_name", "date_of_birth"]

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        phi_present = [c for c in self.phi_columns if c in df.columns]
        if not phi_present:
            return []

        if self.consent_column not in df.columns:
            return [RegulatoryViolation(
                rule_name=self.name,
                domain=self.domain,
                severity="ERROR",
                column=self.consent_column,
                offending_count=len(df),
                message=(
                    f"[GDPR Art. 7] Consent column '{self.consent_column}' is absent "
                    f"but dataset contains personal data columns {phi_present}. "
                    f"Cannot verify lawful basis for processing."
                ),
                remediation=(
                    f"Add boolean column '{self.consent_column}' and populate it "
                    f"from your consent management platform. Only process records "
                    f"where consent_given=True (or another valid GDPR Art. 6 basis)."
                ),
            )]

        has_phi_mask = df[phi_present].notna().any(axis=1)
        no_consent = ~df[self.consent_column].astype(bool)
        bad = (has_phi_mask & no_consent).sum()

        if bad == 0:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="ERROR",
            column=self.consent_column,
            offending_count=int(bad),
            message=(
                f"[GDPR Art. 7] {bad} record(s) contain personal data columns "
                f"{phi_present} but '{self.consent_column}' is False/null. "
                f"Processing without consent violates GDPR Art. 6 lawful basis."
            ),
            remediation=(
                "Filter out records without consent before processing, or establish "
                "another GDPR Art. 6 lawful basis (contract, legitimate interest, etc). "
                "Document the legal basis in your Record of Processing Activities (RoPA)."
            ),
        )]
