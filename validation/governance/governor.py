"""
validation/governance/governor.py
-----------------------------------
Active policy enforcer for the Data Governance engine.

Applies configurable policies (redact, reject, flag) when PII
is detected in a DataFrame. Should be invoked during ingestion 
before data hits the Silver or Gold layers.
"""

import logging
from typing import Dict, Any, Tuple

import pandas as pd

from .pii_detector import PIIDetector

logger = logging.getLogger("dipex.validation.governance.governor")

class GovernanceError(Exception):
    """Raised when the Governance engine blocks data ingestion due to a policy violation."""
    def __init__(self, message: str, report: Dict = None):
        super().__init__(message)
        self.report = report or {}

DOMAIN_MAP = {
    "gdpr": ["Email", "SSN", "Phone", "IPAddress"],
    "healthcare": ["Email", "SSN", "Phone", "ICD10"],
    "banking": ["Email", "CreditCard", "IBAN", "Swift"],
    "finance": ["Email", "CreditCard"],
    "sox": ["Email"],
    "hipaa": ["Email", "SSN", "Phone", "ICD10"],
    "generic": ["Email", "SSN", "CreditCard", "Phone"]
}

class DataGovernor:
    """
    Enforces governance policies on DataFrames based on the project configuration.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        val_cfg = self.config.get("validation", {})
        gov_cfg = val_cfg.get("governance", {})
        reg_cfg = val_cfg.get("regulatory", {})
        
        # Policy: 'redact', 'reject', 'flag', 'off'
        self.policy = gov_cfg.get("policy", "off").lower()
        self.active = (gov_cfg.get("pii_detection", False) or reg_cfg.get("domains")) and self.policy != "off"
        
        # Build target list from domains
        domains = reg_cfg.get("domains", [])
        if isinstance(domains, str):
            domains = [domains]
            
        targets = set()
        for d in domains:
            targets.update(DOMAIN_MAP.get(d.lower(), []))
            
        # If no domains but global PII is on, use generic
        if not targets and gov_cfg.get("pii_detection"):
            targets.update(DOMAIN_MAP["generic"])
            
        self.detector = PIIDetector(targets=list(targets) if targets else None)
        
        if self.active:
            logger.info("DataGovernor initialized. Mode: ACTIVE, Policy: %s, Domains: %s, Targets: %d", 
                        self.policy.upper(), domains, len(targets))
        else:
            logger.info("DataGovernor initialized. Mode: INACTIVE")

    def enforce(self, df: pd.DataFrame, dataset_id: str = "unknown") -> Tuple[pd.DataFrame, Dict]:
        """
        Runs the configured governance policy on the DataFrame.

        Returns:
            - Modifed DataFrame (if 'redact' policy was triggered)
            - A report dictionary of what was found and action taken
        """
        if not self.active:
            return df, {"status": "skipped", "reason": "governance_inactive"}

        report_summary = {
            "dataset_id": dataset_id,
            "policy_applied": self.policy,
            "status": "passed",
            "pii_hits": {},
            "total_redactions": 0
        }

        # Run PII Scan
        detected = self.detector.detect(df)

        if not detected:
            logger.debug("[Governor] No PII detected in dataset '%s'.", dataset_id)
            return df, report_summary

        # PII Found! Apply Policy
        total_hits = sum(sum(hits.values()) for hits in detected.values())
        report_summary["pii_hits"] = detected

        if self.policy == "reject":
            logger.error("[Governor] REJECTING dataset '%s'. Found %d PII instances across %d columns.", 
                         dataset_id, total_hits, len(detected))
            report_summary["status"] = "rejected"
            raise GovernanceError(
                f"Governance policy REJECT triggered: dataset '{dataset_id}' contains {total_hits} instances of PII "
                f"across columns {list(detected.keys())}.",
                report=report_summary
            )

        elif self.policy == "redact":
            logger.warning("[Governor] REDACTING dataset '%s'. Found %d PII instances. Stripping from DataFrame in-memory.", 
                           dataset_id, total_hits)
            cleansed_df, redact_report = self.detector.redact(df)
            
            # Double check redaction success (Optional sanity check)
            __sanity = self.detector.detect(cleansed_df)
            if __sanity:
                logger.error("[Governor] Redaction failed to clear all PII! Detected post-redaction: %s", __sanity)
                raise GovernanceError(f"Sanity check failed: PII leaked past redaction filter on dataset '{dataset_id}'.")

            report_summary["status"] = "redacted"
            report_summary["total_redactions"] = total_hits
            return cleansed_df, report_summary

        elif self.policy == "flag":
            logger.warning("[Governor] FLAGGING dataset '%s' due to %d PII instances. Proceeding without modification.", 
                           dataset_id, total_hits)
            report_summary["status"] = "flagged"
            return df, report_summary

        else:
            logger.warning("Unknown governance policy '%s'. Defaulting to pass-through.", self.policy)
            return df, report_summary
