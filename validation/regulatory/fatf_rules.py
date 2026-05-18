"""
validation/regulatory/fatf_rules.py
--------------------------------------
FATF (Financial Action Task Force) AML/CFT compliance rules.

Rules
-----
StructuringDetectionRule   : Detect cash transactions just below reporting thresholds (smurfing)
PEPScreeningRule           : Verifies pep_flag column exists when customer data present
SanctionsScreeningRule     : Checks for OFAC/EU sanctions list column markers
BeneficialOwnershipRule    : Flags missing beneficial_owner_id in large corporate transactions
RemittanceOriginRule       : Verifies cross-border remittances have origin country + bank fields
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("dipex.validation.regulatory.fatf")

# Common reporting thresholds by currency (USD equivalent)
_STRUCTURING_THRESHOLD = 10_000.0
_STRUCTURING_BUFFER = 500.0      # Flag if amount is within $500 below threshold (smurfing zone)
_LARGE_CORPORATE_THRESHOLD = 100_000.0


class StructuringDetectionRule:
    """
    Detect cash transactions in the 'smurfing zone' — just below reporting thresholds.
    FATF Recommendation 7: Targeted financial sanctions for terrorism financing.
    """
    name = "FATF_STRUCTURING_DETECTION"
    severity = "CRITICAL"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        amount_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"amount", "transaction_amount", "cash_amount", "value"})),
            None
        )
        if not amount_col:
            return violations
        try:
            amounts = pd.to_numeric(df[amount_col], errors="coerce").dropna()
            threshold_low = _STRUCTURING_THRESHOLD - _STRUCTURING_BUFFER
            threshold_high = _STRUCTURING_THRESHOLD
            structured = ((amounts >= threshold_low) & (amounts < threshold_high)).sum()
            if structured > 0:
                violations.append({
                    "rule": self.name,
                    "severity": self.severity,
                    "column": amount_col,
                    "message": f"{structured} transaction(s) fall in the structuring zone "
                               f"(${threshold_low:,.0f}–${threshold_high:,.0f}). Possible smurfing pattern.",
                    "what_it_means": "Multiple transactions just below the cash reporting threshold — a classic money laundering pattern.",
                    "why_it_matters": "Structuring is a federal crime (31 U.S.C. § 5324). FATF compliance requires SAR filing.",
                    "recommended_action": "File a Suspicious Activity Report (SAR) and escalate to AML compliance officer.",
                    "affected_rows": int(structured),
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("StructuringDetectionRule error: %s", exc)
        return violations


class PEPScreeningRule:
    """
    Verify that a PEP flag column exists when customer/person data is present.
    FATF Recommendation 12: Enhanced due diligence for politically exposed persons.
    """
    name = "FATF_PEP_SCREENING"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        customer_hints = {"customer_id", "client_id", "person_id", "user_id", "account_holder"}
        has_customer_data = any(any(h in c.lower() for h in customer_hints) for c in df.columns)
        if not has_customer_data:
            return violations

        pep_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"pep_flag", "politically_exposed", "pep_status", "is_pep"})),
            None
        )
        if not pep_col:
            violations.append({
                "rule": self.name,
                "severity": self.severity,
                "column": "missing:pep_flag",
                "message": "Customer data present but no PEP flag column found. "
                           "FATF Rec. 12 requires PEP identification.",
                "what_it_means": "The dataset cannot identify politically exposed persons, who require enhanced due diligence.",
                "why_it_matters": "Missing PEP screening creates regulatory exposure across all FATF member jurisdictions.",
                "recommended_action": "Add 'pep_flag' (boolean) column and screen against PEP databases (e.g., World-Check).",
                "affected_rows": len(df),
            })
        return violations


class SanctionsScreeningRule:
    """
    Check for OFAC/EU sanctions list screening column markers.
    FATF Recommendation 6: Targeted financial sanctions.
    """
    name = "FATF_SANCTIONS_SCREENING"
    severity = "CRITICAL"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        customer_hints = {"customer", "client", "counterparty", "beneficiary", "name"}
        has_person_data = any(any(h in c.lower() for h in customer_hints) for c in df.columns)
        if not has_person_data:
            return violations

        sanctions_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"sanctions_flag", "ofac_flag", "eu_sanctions", "sanctioned", "watchlist"})),
            None
        )
        if not sanctions_col:
            violations.append({
                "rule": self.name,
                "severity": self.severity,
                "column": "missing:sanctions_flag",
                "message": "Customer/counterparty data present but no sanctions screening column found. "
                           "FATF Rec. 6 + OFAC requirements mandate sanctions checks.",
                "what_it_means": "Transactions may be processed without verifying parties against sanctions lists.",
                "why_it_matters": "Processing sanctioned entities carries criminal liability and fines up to $1M+ per transaction.",
                "recommended_action": "Integrate real-time OFAC/EU/UN sanctions screening and add 'sanctions_flag' column.",
                "affected_rows": len(df),
            })
        return violations


class BeneficialOwnershipRule:
    """
    Flag missing beneficial_owner_id in large corporate transactions.
    FATF Recommendation 24: Transparency and beneficial ownership of legal persons.
    """
    name = "FATF_BENEFICIAL_OWNERSHIP"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        amount_col = next(
            (c for c in df.columns if any(h in c.lower() for h in {"amount", "transaction_amount", "value"})),
            None
        )
        entity_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"company", "corporate", "legal_entity", "organisation", "entity_type"})),
            None
        )
        if not (amount_col and entity_col):
            return violations

        bo_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"beneficial_owner", "bo_id", "ultimate_owner", "ubo"})),
            None
        )
        if not bo_col:
            try:
                large_corporate = (pd.to_numeric(df[amount_col], errors="coerce") > _LARGE_CORPORATE_THRESHOLD).sum()
                if large_corporate > 0:
                    violations.append({
                        "rule": self.name,
                        "severity": self.severity,
                        "column": f"missing:beneficial_owner_id",
                        "message": f"{large_corporate} large corporate transaction(s) > ${_LARGE_CORPORATE_THRESHOLD:,.0f} "
                                   "but no beneficial ownership column found. FATF Rec. 24 requires UBO identification.",
                        "what_it_means": "Large corporate transactions cannot be traced to their ultimate beneficial owner.",
                        "why_it_matters": "UBO transparency is required in all FATF jurisdictions; non-compliance enables shell company abuse.",
                        "recommended_action": "Add 'beneficial_owner_id' and 'ubo_name' columns for corporate transactions.",
                        "affected_rows": int(large_corporate),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("BeneficialOwnershipRule error: %s", exc)
        return violations


class RemittanceOriginRule:
    """
    Verify cross-border remittances have origin_country and correspondent_bank fields.
    FATF Recommendation 16: Wire transfer rules.
    """
    name = "FATF_REMITTANCE_ORIGIN"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        remittance_hints = {"remittance", "wire_transfer", "swift", "cross_border"}
        is_remittance = any(any(h in c.lower() for h in remittance_hints) for c in df.columns)
        if not is_remittance:
            return violations

        required = {
            "origin_country": ["origin_country", "sender_country", "source_country"],
            "correspondent_bank": ["correspondent_bank", "intermediary_bank", "bank_bic", "swift_code"],
        }
        for field, hints in required.items():
            if not any(any(h in c.lower() for h in hints) for c in df.columns):
                violations.append({
                    "rule": self.name,
                    "severity": self.severity,
                    "column": f"missing:{field}",
                    "message": f"Cross-border remittance data detected but '{field}' field is missing. "
                               "FATF Rec. 16 (Wire Transfers) requires originator information.",
                    "what_it_means": f"Wire transfer records lack the required '{field}' field for AML tracing.",
                    "why_it_matters": "Incomplete wire transfer data prevents AML compliance under FATF Rec. 16.",
                    "recommended_action": f"Populate '{field}' for all cross-border transactions.",
                    "affected_rows": len(df),
                })
        return violations
