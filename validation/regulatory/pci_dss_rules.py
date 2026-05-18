"""
validation/regulatory/pci_dss_rules.py
----------------------------------------
PCI-DSS compliance rules for payment card data.

Rules
-----
PANDetectionRule           : Finds unmasked Primary Account Numbers (Luhn check)
CVVStorageRule             : Flags columns likely containing CVV/CVC codes
PANMaskingValidationRule   : Verifies PANs are stored in masked format
CardholderDataIsolationRule: Fails if cardholder data is mixed with non-payment data

Severity: CRITICAL for PANs/CVVs; WARNING for isolation issues.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("dipex.validation.regulatory.pci_dss")

_PAN_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_MASKED_PAN_PATTERN = re.compile(r"X{4}[-\s]?X{4}[-\s]?X{4}[-\s]?\d{4}")
_CVV_SUSPICIOUS_NAMES = {"cvv", "cvc", "cvv2", "cvc2", "card_code", "security_code", "card_security"}


def _luhn_check(number: str) -> bool:
    """Validate a card number using the Luhn algorithm."""
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class PANDetectionRule:
    """
    Detect unmasked Primary Account Numbers (PANs) in any column.

    PCI-DSS Requirement 3.2: PANs must be rendered unreadable anywhere they are stored.
    """
    name = "PAN_DETECTION"
    severity = "CRITICAL"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        for col in df.select_dtypes(include="object").columns:
            try:
                sample = df[col].dropna().astype(str).head(200)
                unmasked_count = 0
                for val in sample:
                    matches = _PAN_PATTERN.findall(val)
                    for m in matches:
                        if _luhn_check(m):
                            unmasked_count += 1
                if unmasked_count > 0:
                    violations.append({
                        "rule": self.name,
                        "severity": self.severity,
                        "column": col,
                        "message": f"Column '{col}' contains {unmasked_count} unmasked PAN(s) that pass Luhn check. "
                                   "PCI-DSS Req 3.2: PANs must be masked or tokenized.",
                        "what_it_means": "Unmasked credit/debit card numbers were found stored in plain text.",
                        "why_it_matters": "PCI-DSS breach can result in fines of $5,000–$100,000/month and card scheme bans.",
                        "recommended_action": "Immediately tokenize or mask these PANs. Use format-preserving encryption (FPE).",
                        "affected_rows": unmasked_count,
                    })
                    logger.warning("[PCI-DSS] %d unmasked PAN(s) in column '%s'", unmasked_count, col)
            except Exception as exc:  # noqa: BLE001
                logger.debug("PAN detection error on col '%s': %s", col, exc)
        return violations


class CVVStorageRule:
    """
    Flag columns that likely contain CVV/CVC security codes.

    PCI-DSS Requirement 3.2.1: CVV MUST NOT be stored after authorization.
    """
    name = "CVV_STORAGE"
    severity = "CRITICAL"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        for col in df.columns:
            col_lower = col.lower().replace(" ", "_")
            if any(s in col_lower for s in _CVV_SUSPICIOUS_NAMES):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    violations.append({
                        "rule": self.name,
                        "severity": self.severity,
                        "column": col,
                        "message": f"Column '{col}' appears to store CVV/CVC codes. "
                                   "PCI-DSS Req 3.2.1: CVV must NEVER be stored after auth.",
                        "what_it_means": "Card verification codes found in persistent storage — strictly prohibited.",
                        "why_it_matters": "Storing CVV post-authorization is a direct PCI-DSS level 1 violation.",
                        "recommended_action": "Drop this column immediately and audit all historical backups.",
                        "affected_rows": len(non_null),
                    })
        return violations


class PANMaskingValidationRule:
    """
    Verify that stored PANs follow the masked format XXXX-XXXX-XXXX-1234.
    """
    name = "PAN_MASKING_VALIDATION"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        pan_col_hints = {"pan", "card_number", "account_number", "card_no", "pan_number"}
        for col in df.columns:
            if any(h in col.lower() for h in pan_col_hints):
                sample = df[col].dropna().astype(str).head(200)
                unmasked = [v for v in sample if _PAN_PATTERN.search(v) and not _MASKED_PAN_PATTERN.search(v)]
                if unmasked:
                    violations.append({
                        "rule": self.name,
                        "severity": self.severity,
                        "column": col,
                        "message": f"Column '{col}' has {len(unmasked)} values that appear unmasked. "
                                   "Format should be XXXX-XXXX-XXXX-1234.",
                        "what_it_means": "PAN storage does not follow the required masking standard.",
                        "why_it_matters": "Exposed PANs create direct fraud risk and PCI audit failure.",
                        "recommended_action": "Apply masking: store only first 6 and last 4 digits.",
                        "affected_rows": len(unmasked),
                    })
        return violations


class CardholderDataIsolationRule:
    """
    Fail if cardholder data exists alongside high-cardinality non-payment columns.
    """
    name = "CARDHOLDER_DATA_ISOLATION"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        payment_hints = {"pan", "card_number", "cardholder", "cvv", "expiry"}
        has_payment_cols = any(
            any(h in c.lower() for h in payment_hints) for c in df.columns
        )
        if not has_payment_cols:
            return violations

        high_card_non_payment = [
            c for c in df.columns
            if not any(h in c.lower() for h in payment_hints)
            and df[c].dtype == object
        ]
        try:
            suspicious = [
                c for c in high_card_non_payment
                if df[c].nunique() > 500
            ]
        except Exception:  # noqa: BLE001
            suspicious = []

        if suspicious:
            violations.append({
                "rule": self.name,
                "severity": self.severity,
                "column": ", ".join(suspicious[:5]),
                "message": f"Cardholder data detected alongside high-cardinality non-payment columns: "
                           f"{suspicious[:5]}. PCI-DSS requires cardholder data environment (CDE) isolation.",
                "what_it_means": "Your cardholder data is stored with unrelated data, widening the PCI scope.",
                "why_it_matters": "Wider PCI scope = more compliance controls required and higher audit costs.",
                "recommended_action": "Separate cardholder data into an isolated CDE schema/database.",
                "affected_rows": 0,
            })
        return violations
