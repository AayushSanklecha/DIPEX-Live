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
                    "from your consent management platform. Only process records "
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


# ── Phase 4 Extended Cross-Domain Rules (Part 6 — MODIFY cross_domain_rules.py) ──


class GDPRDataMinimizationRule(BaseRegulatoryRule):
    """
    GDPR Article 5(1)(c) — Data Minimisation Principle.

    Flags datasets with a high proportion of optional/non-essential personal
    data columns. The more columns containing PII that are present relative
    to total columns, the higher the minimisation risk.

    Threshold: >30% of columns identified as PII = WARNING.
    """
    name = "gdpr_data_minimization"
    domain = "gdpr"

    _PII_COL_PATTERNS = re.compile(
        r"\b(ssn|passport|dob|date.of.birth|full.name|firstname|lastname|surname|"
        r"address|postcode|zipcode|phone|mobile|fax|email|nic|aadhar|pan|"
        r"ip.address|mac.address|device.id|cookie|biometric|fingerprint|"
        r"location|gps|lat|lon|latitude|longitude)\b",
        re.I,
    )
    MAX_PII_RATIO = 0.30

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if df.empty:
            return []
        pii_cols = [c for c in df.columns if self._PII_COL_PATTERNS.search(c)]
        ratio = len(pii_cols) / max(len(df.columns), 1)
        if ratio <= self.MAX_PII_RATIO:
            return []
        return [RegulatoryViolation(
            rule_name=self.name, domain=self.domain, severity="WARNING",
            column=", ".join(pii_cols[:5]),
            offending_count=len(pii_cols),
            message=(
                f"[GDPR Art. 5(1)(c)] {len(pii_cols)} PII-like columns detected "
                f"({ratio:.1%} of total). GDPR requires collecting only data that is "
                f"'adequate, relevant, and limited to what is necessary.' "
                f"PII columns: {pii_cols[:8]}."
            ),
            remediation=(
                "Audit each PII column against a documented business need. "
                "Drop columns not required for the stated processing purpose. "
                "Update your Privacy Impact Assessment (DPIA) accordingly."
            ),
        )]


class GDPRRightToErasureRule(BaseRegulatoryRule):
    """
    GDPR Article 17 — Right to Erasure ('Right to be Forgotten').

    Flags if a deletion_flag / erasure_requested column exists and records
    with erasure_requested=True are still present in the dataset.
    """
    name = "gdpr_right_to_erasure"
    domain = "gdpr"

    _ERASURE_COL_PATTERNS = re.compile(
        r"\b(deletion.flag|erasure.requested|forget.me|opt.out.erasure|"
        r"right.to.erasure|rtbf|delete.requested)\b", re.I
    )

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        erasure_cols = [c for c in df.columns if self._ERASURE_COL_PATTERNS.search(c)]
        if not erasure_cols:
            return []
        violations = []
        for col in erasure_cols:
            try:
                pending = df[col].astype(bool).sum()
                if pending > 0:
                    violations.append(RegulatoryViolation(
                        rule_name=self.name, domain=self.domain, severity="CRITICAL",
                        column=col,
                        offending_count=int(pending),
                        message=(
                            f"[GDPR Art. 17] {pending} record(s) have erasure requested "
                            f"('{col}'=True) but data is still present in the dataset. "
                            "Data subjects have exercised their Right to be Forgotten."
                        ),
                        remediation=(
                            "Immediately delete or anonymise all records with erasure flags. "
                            "Confirm deletion across all downstream systems, backups, and third parties. "
                            "Log completion of erasure request per GDPR Art. 12 (within 30 days)."
                        ),
                    ))
            except Exception:
                pass
        return violations


class GDPRBreachNotificationRule(BaseRegulatoryRule):
    """
    GDPR Article 33/34 — Data Breach Notification.

    If 'breach_occurred' column is True for any records, GDPR requires
    notification to DPA within 72 hours and to data subjects without undue delay.
    """
    name = "gdpr_breach_notification"
    domain = "gdpr"

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        breach_cols = [c for c in df.columns if re.search(r"breach.occurred|data.breach|security.incident", c, re.I)]
        if not breach_cols:
            return []
        col = breach_cols[0]
        try:
            breached = df[col].astype(bool).sum()
            if breached == 0:
                return []
            return [RegulatoryViolation(
                rule_name=self.name, domain=self.domain, severity="CRITICAL",
                column=col,
                offending_count=int(breached),
                message=(
                    f"[GDPR Art. 33/34] {breached} record(s) indicate a confirmed data breach. "
                    "GDPR mandates DPA notification within 72 hours of becoming aware."
                ),
                remediation=(
                    "Prepare breach notification for DPA (Art. 33). "
                    "Assess whether breach 'likely results in risk to rights and freedoms' (Art. 34). "
                    "If high risk — notify affected data subjects without undue delay. "
                    "Document breach in internal Data Breach Register."
                ),
            )]
        except Exception:
            return []


class HIPAAMinimumNecessaryRule(BaseRegulatoryRule):
    """
    HIPAA 45 CFR §164.502(b) — Minimum Necessary Standard.

    When PHI is used or disclosed, the covered entity must make reasonable
    efforts to limit PHI to the minimum necessary for the purpose.
    Flags if clinical notes / full medical history columns are present
    alongside non-clinical processing columns.
    """
    name = "hipaa_minimum_necessary"
    domain = "hipaa"

    _FULL_RECORD_COLS = re.compile(
        r"\b(full.medical.history|complete.record|all.diagnoses|"
        r"clinical.notes|physician.notes|discharge.summary|"
        r"procedure.history|medication.history)\b", re.I
    )
    _NON_CLINICAL_COLS = re.compile(
        r"\b(marketing|campaign|segment|score|propensity|churn|cltv|clv)\b", re.I
    )

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        full_cols = [c for c in df.columns if self._FULL_RECORD_COLS.search(c)]
        non_clin  = [c for c in df.columns if self._NON_CLINICAL_COLS.search(c)]
        if not full_cols or not non_clin:
            return []
        return [RegulatoryViolation(
            rule_name=self.name, domain=self.domain, severity="ERROR",
            column=", ".join(full_cols[:3]),
            offending_count=len(df),
            message=(
                f"[HIPAA §164.502(b)] Full medical history columns ({full_cols[:3]}) "
                f"co-exist with non-clinical columns ({non_clin[:3]}). "
                "HIPAA Minimum Necessary standard requires limiting PHI to what is "
                "necessary for the processing purpose."
            ),
            remediation=(
                "Apply field-level access controls — expose only required PHI fields. "
                "Create purpose-specific data views. "
                "Document purpose limitation in your HIPAA Privacy Policy."
            ),
        )]


class FCAThanksConductRule(BaseRegulatoryRule):
    """
    FCA Consumer Duty (PS22/9) — Fair Value & Consumer Outcomes.

    Checks that retail financial product datasets include required
    outcome monitoring columns (e.g., customer_complaint_flag, outcome_fair,
    product_value_score). Missing these suggests non-compliance with FCA
    Consumer Duty outcome monitoring requirements (effective July 2023).
    """
    name = "fca_consumer_duty"
    domain = "finance"

    _REQUIRED_COL_PATTERNS = [
        re.compile(r"complaint|complaint.flag|complaint.rate", re.I),
        re.compile(r"outcome|fair.outcome|consumer.outcome", re.I),
        re.compile(r"product.value|value.assessment|price.fairness", re.I),
        re.compile(r"vulnerability.flag|vulnerable.customer", re.I),
    ]
    _RETAIL_SIGNAL_PATTERNS = re.compile(
        r"\b(retail|customer|consumer|adviser|adviser.charge|mortgage|"
        r"savings|pension|insurance|protection|investment|retail.fund)\b", re.I
    )

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        retail_cols = [c for c in df.columns if self._RETAIL_SIGNAL_PATTERNS.search(c)]
        if not retail_cols:
            return []  # Not a retail financial product dataset

        missing_cats = []
        for pat in self._REQUIRED_COL_PATTERNS:
            if not any(pat.search(c) for c in df.columns):
                # Extract the first part of the pattern for description
                missing_cats.append(pat.pattern.split("|")[0].replace("\\b(", "").replace("\\b", ""))

        if not missing_cats:
            return []

        return [RegulatoryViolation(
            rule_name=self.name, domain=self.domain, severity="WARNING",
            column="N/A",
            offending_count=0,
            message=(
                f"[FCA Consumer Duty PS22/9] Dataset appears to contain retail financial "
                f"product data but is missing outcome monitoring columns for: "
                f"{', '.join(missing_cats[:4])}. "
                "FCA Consumer Duty requires firms to monitor and demonstrate good outcomes."
            ),
            remediation=(
                "Add outcome monitoring columns as required by FCA Consumer Duty. "
                "Implement complaints tracking, value assessment, and vulnerability flagging. "
                "Report against outcome metrics in Annual Consumer Duty Board Report."
            ),
        )]


class RBIKYCRule(BaseRegulatoryRule):
    """
    RBI Master Direction on KYC (Jan 2020 / updated 2023).

    Checks that Indian retail banking datasets include required KYC
    columns: customer_id, pan_number or aadhaar_number, kyc_status,
    and risk_category. Absence of KYC columns in customer-level data
    indicates non-compliance with RBI KYC norms.
    """
    name = "rbi_kyc"
    domain = "banking"

    _CUSTOMER_SIGNALS = re.compile(r"\b(customer|account|client|depositor)\b", re.I)
    _REQUIRED_GROUPS = [
        ("customer_id", re.compile(r"customer.id|account.no|cif.number|client.id", re.I)),
        ("kyc_status",  re.compile(r"kyc.status|kyc.completed|kyc.verified|kyc.done", re.I)),
        ("risk_category", re.compile(r"risk.category|risk.rating|risk.profile|aml.risk", re.I)),
        ("id_proof",    re.compile(r"pan|aadhaar|passport.number|voter.id|driving.licence", re.I)),
    ]

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        # Only applies to datasets that appear to be customer-level banking data
        customer_cols = [c for c in df.columns if self._CUSTOMER_SIGNALS.search(c)]
        if not customer_cols:
            return []

        missing_groups = []
        for label, pat in self._REQUIRED_GROUPS:
            if not any(pat.search(c) for c in df.columns):
                missing_groups.append(label)

        if not missing_groups:
            return []

        return [RegulatoryViolation(
            rule_name=self.name, domain=self.domain, severity="ERROR",
            column="N/A",
            offending_count=0,
            message=(
                f"[RBI KYC Master Direction 2020] Customer-level banking dataset "
                f"is missing required KYC fields: {', '.join(missing_groups)}. "
                "RBI requires all Regulated Entities to collect and verify KYC documents."
            ),
            remediation=(
                "Add mandatory KYC columns: customer ID, KYC status (Verified/Pending/Expired), "
                "risk category (Low/Medium/High), and valid identity document reference. "
                "Perform periodic KYC re-verification per RBI Master Direction timelines. "
                "Report non-compliant accounts to RBI within prescribed timeframes."
            ),
        )]
