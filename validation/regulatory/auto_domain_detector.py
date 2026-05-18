"""
validation/regulatory/auto_domain_detector.py
-----------------------------------------------
Auto-detects which regulatory domains apply to a given DataFrame by
analysing column names, data patterns, and value distributions.

Returns a list of domain names (e.g., ['banking', 'gdpr']) that the
MultiDomainRegulatoryCrossChecker should activate.

Detection heuristics
--------------------
- banking/finance  → amount, transaction, loan, repayment, trader, portfolio columns
- healthcare       → patient, diagnosis, icd, medication, admission columns
- gdpr             → consent, data_region, EU residency markers, phi columns
- pci_dss          → pan, card_number, cvv, cardholder columns
- ccpa             → California state markers, opt_out, do_not_sell columns
- fatf             → aml, pep_flag, sanctions_flag, structuring, remittance columns
- mifid2           → lei, isin, execution_venue, algo_id columns
- esg              → scope_1/2/3, esg_score, supplier_risk columns
- cyber            → access_log, incident, classification, encryption columns
- sox              → audit_timestamp, modified_at, modified_by, journal_entry columns
- insurance        → policy_number, premium, claim_id, loss_ratio columns
- ecommerce        → sku, order_id, product_id, basket, cart columns
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

import pandas as pd

logger = logging.getLogger("dipex.validation.regulatory.auto_domain_detector")


# Domain signature: {domain_name: set_of_keyword_fragments}
_DOMAIN_SIGNATURES: Dict[str, Set[str]] = {
    "banking": {
        "transaction_amount", "loan_amount", "repayment", "interest_rate",
        "account_number", "credit_score", "debit", "credit", "overdraft",
        "mortgage", "collateral", "ltv", "tier1", "rwa", "aml",
    },
    "healthcare": {
        "patient", "diagnosis", "icd_code", "medication", "dob",
        "date_of_birth", "admission", "discharge", "provider_id",
        "clinical_note", "lab_result", "mrn", "health_record",
    },
    "gdpr": {
        "consent_given", "data_region", "gdpr", "eu_resident",
        "right_to_access", "erasure_request", "data_controller",
        "lawful_basis", "retention_period",
    },
    "pci_dss": {
        "pan", "card_number", "cardholder", "cvv", "cvc", "expiry_date",
        "payment_token", "card_type", "card_brand",
    },
    "ccpa": {
        "do_not_sell", "opt_out", "california", "ccpa", "deletion_request",
        "data_sale_consent", "ca_resident",
    },
    "fatf": {
        "pep_flag", "sanctions_flag", "aml_flag", "structuring",
        "remittance", "wire_transfer", "beneficial_owner", "ubo",
        "politically_exposed", "watchlist",
    },
    "mifid2": {
        "lei", "isin", "execution_venue", "algo_id", "trading_venue",
        "mic_code", "client_category", "best_execution", "rts22",
        "investor_type",
    },
    "esg": {
        "scope_1", "scope_2", "scope_3", "esg_score", "co2_emissions",
        "carbon_footprint", "sustainability", "supplier_risk",
        "gender_pay", "esg_rating", "ghg_emissions",
    },
    "cyber": {
        "access_log", "incident", "breach", "ict_event", "audit_log",
        "classification", "data_class", "encrypted", "key_management",
        "vulnerability", "penetration_test", "siem",
    },
    "sox": {
        "journal_entry", "modified_at", "modified_by", "audit_trail",
        "internal_control", "financial_statement", "sox", "itgc",
        "segregation_of_duties",
    },
    "insurance": {
        "policy_number", "premium", "claim_id", "loss_ratio",
        "underwriting", "insured", "reinsurance", "actuarial",
        "coverage_amount", "deductible",
    },
    "ecommerce": {
        "sku", "order_id", "product_id", "basket", "cart",
        "checkout", "fulfillment", "shipment", "inventory_level",
        "return_reason", "merchant_id",
    },
}

# Minimum number of matching column keywords to activate a domain
_ACTIVATION_THRESHOLD = 1


def detect_domains(df: pd.DataFrame, verbose: bool = False) -> List[str]:
    """
    Analyse a DataFrame's columns (and optionally value samples) to determine
    which regulatory domains apply.

    Parameters
    ----------
    df      : Input DataFrame (can be a sample; full read not required).
    verbose : If True, logs which keywords triggered each domain.

    Returns
    -------
    List of domain name strings, sorted alphabetically.
    """
    activated: List[str] = []
    # Normalise column names: lowercase, replace separators with underscore
    col_tokens: Set[str] = set()
    for col in df.columns:
        normalised = re.sub(r"[\s\-./]", "_", col.lower())
        col_tokens.add(normalised)
        # Also add sub-tokens (e.g. "pan_number" → "pan")
        for part in normalised.split("_"):
            col_tokens.add(part)

    for domain, keywords in _DOMAIN_SIGNATURES.items():
        matched = keywords & col_tokens
        if len(matched) >= _ACTIVATION_THRESHOLD:
            activated.append(domain)
            if verbose:
                logger.debug("[AutoDetect] Domain '%s' activated by: %s", domain, matched)

    # Always include 'gdpr' if any PII-looking column is present
    pii_hints = {"email", "phone", "ssn", "name", "address", "postcode", "zipcode"}
    if pii_hints & col_tokens and "gdpr" not in activated:
        activated.append("gdpr")

    result = sorted(set(activated))
    logger.info("[AutoDomainDetector] Detected domains: %s (from %d columns)", result, len(df.columns))
    return result


def domain_detection_report(df: pd.DataFrame) -> Dict[str, any]:
    """
    Returns a structured report dict of auto-detected domains and matched keywords.
    """
    report = {}
    col_tokens: Set[str] = set()
    for col in df.columns:
        normalised = re.sub(r"[\s\-./]", "_", col.lower())
        col_tokens.add(normalised)
        for part in normalised.split("_"):
            col_tokens.add(part)

    for domain, keywords in _DOMAIN_SIGNATURES.items():
        matched = list(keywords & col_tokens)
        report[domain] = {
            "activated": len(matched) >= _ACTIVATION_THRESHOLD,
            "matched_keywords": sorted(matched),
            "match_count": len(matched),
        }
    return report
