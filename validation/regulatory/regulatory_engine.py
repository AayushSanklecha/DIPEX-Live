"""
validation/regulatory/regulatory_engine.py
--------------------------------------------
Orchestrates domain-specific regulatory rule evaluation for Hard Gate 1.

The engine loads domain profiles from ``config.yaml``, supports MULTIPLE
simultaneous domains (e.g. banking + gdpr), runs conflict resolution, and
exposes the violating columns for downstream feature masking.

Supported domains
-----------------
``"banking"``    — PositiveAmountRule, AMLThresholdRule, LoanRatioRule,
                   RepaymentConsistencyRule, SuspiciousTransactionPatternRule,
                   CurrencyConcentrationRule
``"healthcare"`` — AgeRangeRule, VitalSignsRule, DiagnosisCodeFormatRule,
                   PHIPresenceRule, ConsentValidationRule, DeIdentificationRule
``"finance"``    — RevenueRecognitionRule, CapitalAdequacyRule, NetPositionLimitRule,
                   MarginCallThresholdRule, DoubleEntryBalanceRule, FairValueHierarchyRule
``"gdpr"``       — GDPRDataResidencyRule, GDPRConsentRequiredRule
``"sox"``        — SOXAuditTrailRule
``"hipaa"``      — HIPAAEncryptionFlagRule
``"generic"``    — No domain rules applied (safe default for non-regulated data)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import yaml

from .base_rule import BaseRegulatoryRule, RegulatoryViolation
from .banking_rules import (
    AMLThresholdRule,
    CurrencyConcentrationRule,
    LoanRatioRule,
    PositiveAmountRule,
    RepaymentConsistencyRule,
    SuspiciousTransactionPatternRule,
    # Basel III extensions
    BaselIIILeverageRule,
    LCRRule,
    NSFRRule,
    IRRBBRule,
)
from .conflict_resolver import RuleConflictResolver
from .cross_domain_rules import (
    GDPRConsentRequiredRule,
    GDPRDataResidencyRule,
    HIPAAEncryptionFlagRule,
    SOXAuditTrailRule,
    # Phase 4 cross-domain extensions
    GDPRDataMinimizationRule,
    GDPRRightToErasureRule,
    GDPRBreachNotificationRule,
    HIPAAMinimumNecessaryRule,
    FCAThanksConductRule,
    RBIKYCRule,
)
from .finance_rules import (
    CapitalAdequacyRule,
    DoubleEntryBalanceRule,
    FairValueHierarchyRule,
    MarginCallThresholdRule,
    NetPositionLimitRule,
    RevenueRecognitionRule,
    SECFilingBoundsRule,
)
from .healthcare_rules import (
    AgeRangeRule,
    ConsentValidationRule,
    DeIdentificationRule,
    DiagnosisCodeFormatRule,
    PHIPresenceRule,
    VitalSignsRule,
)
# Phase 4 new rule modules
try:
    from .pci_dss_rules import (
        PANPresenceRule, CVVStorageRule, PCICardholderIsolationRule,
    )
    _HAS_PCI = True
except ImportError:
    _HAS_PCI = False

try:
    from .mifid2_rules import (
        BestExecutionEvidenceRule, LEIISINFormatRule, AlgoTradingTagRule,
    )
    _HAS_MIFID2 = True
except ImportError:
    _HAS_MIFID2 = False

try:
    from .fatf_rules import (
        StructuringDetectionRule, PEPScreeningRule, SanctionsCheckRule,
    )
    _HAS_FATF = True
except ImportError:
    _HAS_FATF = False

try:
    from .esg_rules import (
        CarbonEmissionsValidityRule, ESGScoreRangeRule,
    )
    _HAS_ESG = True
except ImportError:
    _HAS_ESG = False

try:
    from .ccpa_rules import (
        CaliforniaResidencyDisclosureRule, ConsumerRightsDeletionRule,
        OptOutOfSaleRule,
    )
    _HAS_CCPA = True
except ImportError:
    _HAS_CCPA = False

try:
    from .cyber_rules import (
        DataClassificationRule, EncryptionAtRestRule,
    )
    _HAS_CYBER = True
except ImportError:
    _HAS_CYBER = False

try:
    from .insurance_rules import (
        SCRCapitalAdequacyRule, CombinedRatioRule,
        ClaimSettlementTimelinessRule, PremiumIntegrityRule,
    )
    _HAS_INSURANCE = True
except ImportError:
    _HAS_INSURANCE = False

try:
    from .ecommerce_rules import (
        SCAExemptionTagRule, ChargebackThresholdRule,
        RefundRightWindowRule, CookieConsentTrackingRule,
    )
    _HAS_ECOMMERCE = True
except ImportError:
    _HAS_ECOMMERCE = False

try:
    from .auto_domain_detector import detect_domains as _auto_detect_domains
    _HAS_AUTO_DETECT = True
except ImportError:
    _HAS_AUTO_DETECT = False

logger = logging.getLogger(__name__)

# Severity sort order for deterministic output ordering
_SEVERITY_ORDER: Dict[str, int] = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}


class RegulatoryEngine:
    """
    Pluggable multi-domain regulatory rule engine.

    Instantiation — from project config (recommended):

        engine = RegulatoryEngine.from_config(config)
        violations = engine.evaluate(df)
        bad_cols   = engine.get_violating_columns()  # for feature masking

    Instantiation — from rules.yaml directly:

        engine = RegulatoryEngine.from_yaml_config("validation/regulatory/rules.yaml")

    Instantiation — manual (for testing or custom domains):

        engine = RegulatoryEngine(domain="banking", rules=[PositiveAmountRule([...])])
        violations = engine.evaluate(df)

    Multi-domain:

        engine = RegulatoryEngine.from_config_multi_domain(config, domains=["banking", "gdpr"])
    """

    def __init__(
        self,
        domain: str = "generic",
        rules: Optional[List[BaseRegulatoryRule]] = None,
        conflict_strategy: str = "strictest_wins",
    ) -> None:
        self.domain = domain.lower()
        self.rules: List[BaseRegulatoryRule] = rules if rules is not None else []
        self.conflict_strategy = conflict_strategy
        self._last_violations: List[RegulatoryViolation] = []
        self._last_conflict_report: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RegulatoryEngine":
        """
        Builds a RegulatoryEngine from the project config dict.
        Reads ``validation.regulatory`` section.
        Supports the new ``domains`` list for multi-domain.
        Falls back to the single ``domain`` key for backward-compatibility.
        """
        reg_cfg = config.get("validation", {}).get("regulatory", {})
        conflict_strategy = str(reg_cfg.get("conflict_resolution", "strictest_wins"))

        # Multi-domain: new ``domains`` list takes precedence over singular ``domain``
        domains_raw = reg_cfg.get("domains") or []
        if domains_raw:
            domains = [str(d).lower() for d in domains_raw]
        else:
            domains = [str(reg_cfg.get("domain", "generic")).lower()]

        all_rules: List[BaseRegulatoryRule] = []

        for domain in domains:
            if domain == "banking":
                rules = cls._build_banking_rules(reg_cfg.get("banking", {}))
            elif domain == "healthcare":
                rules = cls._build_healthcare_rules(reg_cfg.get("healthcare", {}))
            elif domain == "finance":
                rules = cls._build_finance_rules(reg_cfg.get("finance", {}))
            elif domain in ("gdpr",):
                rules = cls._build_gdpr_rules(reg_cfg.get("gdpr", {}))
            elif domain == "sox":
                rules = cls._build_sox_rules(reg_cfg.get("sox", {}))
            elif domain == "hipaa":
                rules = cls._build_hipaa_rules(reg_cfg.get("hipaa", {}))
            elif domain == "pci_dss":
                rules = cls._build_pci_dss_rules(reg_cfg.get("pci_dss", {}))
            elif domain == "mifid2":
                rules = cls._build_mifid2_rules(reg_cfg.get("mifid2", {}))
            elif domain == "fatf":
                rules = cls._build_fatf_rules(reg_cfg.get("fatf", {}))
            elif domain == "esg":
                rules = cls._build_esg_rules(reg_cfg.get("esg", {}))
            elif domain == "ccpa":
                rules = cls._build_ccpa_rules(reg_cfg.get("ccpa", {}))
            elif domain == "cyber":
                rules = cls._build_cyber_rules(reg_cfg.get("cyber", {}))
            elif domain == "insurance":
                rules = cls._build_insurance_rules(reg_cfg.get("insurance", {}))
            elif domain == "ecommerce":
                rules = cls._build_ecommerce_rules(reg_cfg.get("ecommerce", {}))
            else:
                rules = []

            if rules:
                logger.info(
                    "RegulatoryEngine: domain='%s' loaded — %d rule(s).", domain, len(rules)
                )
            all_rules.extend(rules)

        primary_domain = domains[0] if domains else "generic"
        engine = cls(domain=primary_domain, rules=all_rules, conflict_strategy=conflict_strategy)
        logger.info(
            "RegulatoryEngine: %d total rule(s) across domain(s) %s.", len(all_rules), domains
        )
        return engine

    @classmethod
    def from_config_auto(
        cls,
        config: Dict[str, Any],
        df: Optional["pd.DataFrame"] = None,  # type: ignore[name-defined]
    ) -> "RegulatoryEngine":
        """
        Auto-detect domains from the DataFrame columns and build a
        RegulatoryEngine without requiring explicit domain configuration.
        Falls back to from_config() if auto-detection is unavailable.
        """
        if df is not None and _HAS_AUTO_DETECT:
            detected = _auto_detect_domains(df)
            if detected:
                reg_cfg = config.get("validation", {}).get("regulatory", {})
                conflict_strategy = str(reg_cfg.get("conflict_resolution", "strictest_wins"))
                all_rules: List[BaseRegulatoryRule] = []
                for domain in detected:
                    rules = cls._build_rules_for_domain(domain, reg_cfg)
                    all_rules.extend(rules)
                    if rules:
                        logger.info("[AutoEngine] domain='%s' auto-detected — %d rule(s).", domain, len(rules))
                primary = detected[0]
                engine = cls(domain=primary, rules=all_rules, conflict_strategy=conflict_strategy)
                logger.info("[AutoEngine] %d rule(s) loaded from %d auto-detected domains: %s",
                            len(all_rules), len(detected), detected)
                return engine
        return cls.from_config(config)

    @classmethod
    def _build_rules_for_domain(
        cls, domain: str, reg_cfg: Dict[str, Any]
    ) -> List[BaseRegulatoryRule]:
        """Single-domain rule builder — delegates to existing static factories."""
        domain = domain.lower()
        if domain == "banking":       return cls._build_banking_rules(reg_cfg.get("banking", {}))
        if domain == "healthcare":    return cls._build_healthcare_rules(reg_cfg.get("healthcare", {}))
        if domain == "finance":       return cls._build_finance_rules(reg_cfg.get("finance", {}))
        if domain == "gdpr":          return cls._build_gdpr_rules(reg_cfg.get("gdpr", {}))
        if domain == "sox":           return cls._build_sox_rules(reg_cfg.get("sox", {}))
        if domain == "hipaa":         return cls._build_hipaa_rules(reg_cfg.get("hipaa", {}))
        if domain == "pci_dss":       return cls._build_pci_dss_rules(reg_cfg.get("pci_dss", {}))
        if domain == "mifid2":        return cls._build_mifid2_rules(reg_cfg.get("mifid2", {}))
        if domain == "fatf":          return cls._build_fatf_rules(reg_cfg.get("fatf", {}))
        if domain == "esg":           return cls._build_esg_rules(reg_cfg.get("esg", {}))
        if domain == "ccpa":          return cls._build_ccpa_rules(reg_cfg.get("ccpa", {}))
        if domain == "cyber":         return cls._build_cyber_rules(reg_cfg.get("cyber", {}))
        if domain == "insurance":     return cls._build_insurance_rules(reg_cfg.get("insurance", {}))
        if domain == "ecommerce":     return cls._build_ecommerce_rules(reg_cfg.get("ecommerce", {}))
        return []

    @classmethod
    def from_yaml_config(cls, yaml_path: str) -> "RegulatoryEngine":
        """
        Builds a RegulatoryEngine by reading a rules.yaml file directly.
        This enables override of config.yaml parameters for testing.
        """
        if not os.path.exists(yaml_path):
            logger.warning("RegulatoryEngine.from_yaml_config: file not found: %s", yaml_path)
            return cls(domain="generic", rules=[])

        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        rules: List[BaseRegulatoryRule] = []

        # Wrap into a synthetic config dict and call from_config
        synthetic_config = {
            "validation": {
                "regulatory": {
                    "domains": list(raw.keys()),
                    **{k: {} for k in raw.keys()},
                }
            }
        }
        engine = cls.from_config(synthetic_config)
        logger.info(
            "RegulatoryEngine.from_yaml_config: loaded from '%s', %d rule(s).",
            yaml_path,
            len(engine.rules),
        )
        return engine

    # ------------------------------------------------------------------
    # Domain rule builders (static factories)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_banking_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        rules: List[BaseRegulatoryRule] = []

        amount_cols: List[str] = cfg.get("amount_columns", [])
        if amount_cols:
            rules.append(PositiveAmountRule(
                amount_columns=amount_cols,
                allow_zero=cfg.get("allow_zero_amounts", False),
            ))

        aml_col: Optional[str] = cfg.get("aml_amount_column")
        if aml_col:
            rules.append(AMLThresholdRule(
                amount_column=aml_col,
                threshold=float(cfg.get("aml_threshold", 10_000.0)),
                currency_column=cfg.get("currency_column"),
            ))

        ltv_cfg: Dict[str, Any] = cfg.get("loan_ratio", {})
        if ltv_cfg.get("loan_col") and ltv_cfg.get("value_col"):
            rules.append(LoanRatioRule(
                loan_col=str(ltv_cfg["loan_col"]),
                value_col=str(ltv_cfg["value_col"]),
                max_ltv=float(ltv_cfg.get("max_ltv", 0.90)),
                severity=str(ltv_cfg.get("severity", "ERROR")),
            ))

        repay_cfg: Dict[str, Any] = cfg.get("repayment", {})
        if repay_cfg.get("repayment_col") and repay_cfg.get("balance_col"):
            rules.append(RepaymentConsistencyRule(
                repayment_col=str(repay_cfg["repayment_col"]),
                balance_col=str(repay_cfg["balance_col"]),
            ))

        # Velocity spike (FATF Rec. 20)
        velocity_cfg: Dict[str, Any] = cfg.get("velocity", {})
        tx_id_col = velocity_cfg.get("transaction_id_column") or cfg.get("aml_amount_column")
        if tx_id_col or amount_cols:
            rules.append(SuspiciousTransactionPatternRule(
                transaction_id_column=velocity_cfg.get("transaction_id_column", "account_id"),
                timestamp_column=velocity_cfg.get("timestamp_column", "transaction_date"),
                max_transactions_per_day=int(
                    velocity_cfg.get("max_transactions_per_day", 50)
                ),
            ))

        # Currency concentration (BCBS239)
        curr_col = cfg.get("currency_column")
        if curr_col:
            rules.append(CurrencyConcentrationRule(
                currency_column=curr_col,
                max_concentration_pct=float(cfg.get("max_currency_concentration", 0.90)),
            ))

        # ── Basel III extensions ──────────────────────────────────────────────
        leverage_cfg = cfg.get("leverage", {})
        if leverage_cfg.get("tier1_col") and leverage_cfg.get("exposure_col"):
            rules.append(BaselIIILeverageRule(
                tier1_col=leverage_cfg["tier1_col"],
                exposure_col=leverage_cfg["exposure_col"],
                min_leverage=float(leverage_cfg.get("min_leverage", 0.03)),
            ))

        lcr_cfg = cfg.get("lcr", {})
        if lcr_cfg.get("hqla_col") and lcr_cfg.get("net_outflow_col"):
            rules.append(LCRRule(
                hqla_col=lcr_cfg["hqla_col"],
                net_outflow_col=lcr_cfg["net_outflow_col"],
                min_lcr=float(lcr_cfg.get("min_lcr", 1.00)),
            ))

        nsfr_cfg = cfg.get("nsfr", {})
        if nsfr_cfg.get("asf_col") and nsfr_cfg.get("rsf_col"):
            rules.append(NSFRRule(
                asf_col=nsfr_cfg["asf_col"],
                rsf_col=nsfr_cfg["rsf_col"],
                min_nsfr=float(nsfr_cfg.get("min_nsfr", 1.00)),
            ))

        # IRRBB — always add (self-detecting: only fires on rate-like datasets)
        rules.append(IRRBBRule())

        return rules


    @staticmethod
    def _build_finance_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        rules: List[BaseRegulatoryRule] = []

        rev_cols: List[str] = cfg.get("revenue_columns", [])
        if rev_cols:
            rules.append(RevenueRecognitionRule(
                revenue_columns=rev_cols,
                credit_memo_column=cfg.get("credit_memo_column"),
            ))

        car_cfg = cfg.get("capital_adequacy", {})
        if car_cfg.get("tier1_col") and car_cfg.get("rwa_col"):
            rules.append(CapitalAdequacyRule(
                tier1_col=car_cfg["tier1_col"],
                rwa_col=car_cfg["rwa_col"],
                min_car=float(car_cfg.get("min_car", 0.08)),
            ))

        pos_cfg = cfg.get("net_position", {})
        if pos_cfg.get("position_column"):
            rules.append(NetPositionLimitRule(
                position_column=pos_cfg["position_column"],
                max_long=float(pos_cfg.get("max_long", 1_000_000)),
                max_short=float(pos_cfg.get("max_short", 500_000)),
            ))

        margin_cfg = cfg.get("margin", {})
        if margin_cfg.get("balance_col") and margin_cfg.get("maintenance_col"):
            rules.append(MarginCallThresholdRule(
                margin_balance_col=margin_cfg["balance_col"],
                maintenance_margin_col=margin_cfg["maintenance_col"],
            ))

        de_cfg = cfg.get("double_entry", {})
        if de_cfg.get("amount_col") and de_cfg.get("transaction_id_col"):
            rules.append(DoubleEntryBalanceRule(
                amount_col=de_cfg["amount_col"],
                transaction_id_col=de_cfg["transaction_id_col"],
                tolerance=float(de_cfg.get("tolerance", 0.01)),
            ))

        fvh_cfg = cfg.get("fair_value", {})
        if fvh_cfg.get("level3_col") and fvh_cfg.get("total_col"):
            rules.append(FairValueHierarchyRule(
                level3_col=fvh_cfg["level3_col"],
                total_fair_value_col=fvh_cfg["total_col"],
                max_level3_ratio=float(fvh_cfg.get("max_ratio", 0.20)),
            ))

        return rules

    @staticmethod
    def _build_healthcare_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        rules: List[BaseRegulatoryRule] = []

        rules.append(AgeRangeRule(
            age_column=str(cfg.get("age_column", "patient_age")),
            min_age=float(cfg.get("min_age", 0.0)),
            max_age=float(cfg.get("max_age", 125.0)),
        ))
        rules.append(VitalSignsRule(column_bounds=cfg.get("vital_sign_bounds")))

        diag_cols: List[str] = cfg.get("diagnosis_columns", ["diagnosis_code"])
        if diag_cols:
            rules.append(DiagnosisCodeFormatRule(diagnosis_columns=diag_cols))

        rules.append(PHIPresenceRule(
            text_columns=cfg.get("text_columns_for_phi_scan"),
            allowed_phi_columns=cfg.get("allowed_phi_columns", []),
        ))

        # New: consent + de-identification
        phi_cols = cfg.get("phi_columns") or cfg.get("allowed_phi_columns") or None
        rules.append(ConsentValidationRule(
            consent_column=cfg.get("consent_column", "consent_given"),
            phi_columns=phi_cols,
        ))
        rules.append(DeIdentificationRule())

        return rules

    @staticmethod
    def _build_gdpr_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        rules: List[BaseRegulatoryRule] = []

        rules.append(GDPRDataResidencyRule(
            residency_column=cfg.get("residency_column", "data_region"),
            allowed_regions=cfg.get("allowed_regions", ["EU", "EEA"]),
        ))
        rules.append(GDPRConsentRequiredRule(
            consent_column=cfg.get("consent_column", "consent_given"),
            phi_columns=cfg.get("phi_columns"),
        ))
        return rules

    @staticmethod
    def _build_sox_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        return [SOXAuditTrailRule(
            audit_timestamp_column=cfg.get("audit_timestamp_column", "modified_at"),
            audit_user_column=cfg.get("audit_user_column", "modified_by"),
        )]

    @staticmethod
    def _build_hipaa_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        return [HIPAAEncryptionFlagRule(
            phi_columns=cfg.get("phi_columns"),
            encryption_flag_column=cfg.get("encryption_flag_column", "is_encrypted"),
        )]

    # ── Phase 4 new rule module builders ─────────────────────────────────────

    @staticmethod
    def _build_pci_dss_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        if not _HAS_PCI:
            logger.debug("PCI-DSS rules not available — import failed.")
            return []
        rules: List[BaseRegulatoryRule] = []
        for RuleCls in [PANPresenceRule, CVVStorageRule, PCICardholderIsolationRule]:  # type: ignore[name-defined]
            class _Wrapped(BaseRegulatoryRule):  # type: ignore[misc]
                def __init__(self, cls=RuleCls, c=cfg):
                    self._r = cls()
                    self._cfg = c
                    self.name = cls.name
                    self.domain = "pci_dss"
                def evaluate(self, df):  # type: ignore[override]
                    from .base_rule import RegulatoryViolation as RV
                    violations = []
                    for v in self._r.run(df, self._cfg):
                        violations.append(RV(
                            rule_name=v.get("rule", self.name),
                            domain="pci_dss", severity=v.get("severity", "ERROR"),
                            column=v.get("column", "N/A"),
                            offending_count=int(v.get("affected_rows", 0)),
                            message=v.get("message", ""),
                            remediation=v.get("recommended_action", v.get("remediation", "")),
                        ))
                    return violations
            rules.append(_Wrapped())
        return rules

    @staticmethod
    def _build_mifid2_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        if not _HAS_MIFID2:
            return []
        rules: List[BaseRegulatoryRule] = []
        for RuleCls in [BestExecutionEvidenceRule, LEIISINFormatRule, AlgoTradingTagRule]:  # type: ignore[name-defined]
            class _Wrapped(BaseRegulatoryRule):  # type: ignore[misc]
                def __init__(self, cls=RuleCls, c=cfg):
                    self._r = cls()
                    self._cfg = c
                    self.name = cls.name
                    self.domain = "mifid2"
                def evaluate(self, df):  # type: ignore[override]
                    from .base_rule import RegulatoryViolation as RV
                    violations = []
                    for v in self._r.run(df, self._cfg):
                        violations.append(RV(
                            rule_name=v.get("rule", self.name),
                            domain="mifid2", severity=v.get("severity", "WARNING"),
                            column=v.get("column", "N/A"),
                            offending_count=int(v.get("affected_rows", 0)),
                            message=v.get("message", ""),
                            remediation=v.get("recommended_action", v.get("remediation", "")),
                        ))
                    return violations
            rules.append(_Wrapped())
        return rules

    @staticmethod
    def _build_fatf_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        if not _HAS_FATF:
            return []
        rules: List[BaseRegulatoryRule] = []
        for RuleCls in [StructuringDetectionRule, PEPScreeningRule, SanctionsCheckRule]:  # type: ignore[name-defined]
            class _Wrapped(BaseRegulatoryRule):  # type: ignore[misc]
                def __init__(self, cls=RuleCls, c=cfg):
                    self._r = cls()
                    self._cfg = c
                    self.name = cls.name
                    self.domain = "fatf"
                def evaluate(self, df):  # type: ignore[override]
                    from .base_rule import RegulatoryViolation as RV
                    violations = []
                    for v in self._r.run(df, self._cfg):
                        violations.append(RV(
                            rule_name=v.get("rule", self.name),
                            domain="fatf", severity=v.get("severity", "WARNING"),
                            column=v.get("column", "N/A"),
                            offending_count=int(v.get("affected_rows", 0)),
                            message=v.get("message", ""),
                            remediation=v.get("recommended_action", v.get("remediation", "")),
                        ))
                    return violations
            rules.append(_Wrapped())
        return rules

    @staticmethod
    def _build_esg_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        if not _HAS_ESG:
            return []
        rules: List[BaseRegulatoryRule] = []
        for RuleCls in [CarbonEmissionsValidityRule, ESGScoreRangeRule]:  # type: ignore[name-defined]
            class _Wrapped(BaseRegulatoryRule):  # type: ignore[misc]
                def __init__(self, cls=RuleCls, c=cfg):
                    self._r = cls()
                    self._cfg = c
                    self.name = cls.name
                    self.domain = "esg"
                def evaluate(self, df):  # type: ignore[override]
                    from .base_rule import RegulatoryViolation as RV
                    violations = []
                    for v in self._r.run(df, self._cfg):
                        violations.append(RV(
                            rule_name=v.get("rule", self.name),
                            domain="esg", severity=v.get("severity", "WARNING"),
                            column=v.get("column", "N/A"),
                            offending_count=int(v.get("affected_rows", 0)),
                            message=v.get("message", ""),
                            remediation=v.get("recommended_action", v.get("remediation", "")),
                        ))
                    return violations
            rules.append(_Wrapped())
        return rules

    @staticmethod
    def _build_ccpa_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        if not _HAS_CCPA:
            return []
        rules: List[BaseRegulatoryRule] = []
        for RuleCls in [CaliforniaResidencyDisclosureRule, ConsumerRightsDeletionRule, OptOutOfSaleRule]:  # type: ignore[name-defined]
            class _Wrapped(BaseRegulatoryRule):  # type: ignore[misc]
                def __init__(self, cls=RuleCls, c=cfg):
                    self._r = cls()
                    self._cfg = c
                    self.name = cls.name
                    self.domain = "ccpa"
                def evaluate(self, df):  # type: ignore[override]
                    from .base_rule import RegulatoryViolation as RV
                    violations = []
                    for v in self._r.run(df, self._cfg):
                        violations.append(RV(
                            rule_name=v.get("rule", self.name),
                            domain="ccpa", severity=v.get("severity", "WARNING"),
                            column=v.get("column", "N/A"),
                            offending_count=int(v.get("affected_rows", 0)),
                            message=v.get("message", ""),
                            remediation=v.get("recommended_action", v.get("remediation", "")),
                        ))
                    return violations
            rules.append(_Wrapped())
        return rules

    @staticmethod
    def _build_cyber_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        if not _HAS_CYBER:
            return []
        rules: List[BaseRegulatoryRule] = []
        for RuleCls in [EncryptionAtRestRule, DataClassificationRule]:  # type: ignore[name-defined]
            class _Wrapped(BaseRegulatoryRule):  # type: ignore[misc]
                def __init__(self, cls=RuleCls, c=cfg):
                    self._r = cls()
                    self._cfg = c
                    self.name = cls.name
                    self.domain = "cyber"
                def evaluate(self, df):  # type: ignore[override]
                    from .base_rule import RegulatoryViolation as RV
                    violations = []
                    for v in self._r.run(df, self._cfg):
                        violations.append(RV(
                            rule_name=v.get("rule", self.name),
                            domain="cyber", severity=v.get("severity", "ERROR"),
                            column=v.get("column", "N/A"),
                            offending_count=int(v.get("affected_rows", 0)),
                            message=v.get("message", ""),
                            remediation=v.get("recommended_action", v.get("remediation", "")),
                        ))
                    return violations
            rules.append(_Wrapped())
        return rules

    @staticmethod
    def _build_insurance_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        if not _HAS_INSURANCE:
            return []
        rules: List[BaseRegulatoryRule] = []
        # Use duck-typing wrappers since insurance rules have .run() not .evaluate()
        for RuleCls in [SCRCapitalAdequacyRule, CombinedRatioRule,  # type: ignore[name-defined]
                        ClaimSettlementTimelinessRule, PremiumIntegrityRule]:  # type: ignore[name-defined]
            class _Wrapped(BaseRegulatoryRule):  # type: ignore[misc]
                def __init__(self, cls=RuleCls, c=cfg):
                    self._r = cls()
                    self._cfg = c
                    self.name = cls.name
                    self.domain = "insurance"
                def evaluate(self, df):  # type: ignore[override]
                    from .base_rule import RegulatoryViolation as RV
                    violations = []
                    for v in self._r.run(df, self._cfg):
                        violations.append(RV(
                            rule_name=v.get("rule", self.name),
                            domain="insurance", severity=v.get("severity", "WARNING"),
                            column=v.get("column", "N/A"),
                            offending_count=int(v.get("affected_rows", 0)),
                            message=v.get("message", ""),
                            remediation=v.get("recommended_action", ""),
                        ))
                    return violations
            rules.append(_Wrapped())
        return rules

    @staticmethod
    def _build_ecommerce_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        if not _HAS_ECOMMERCE:
            return []
        rules: List[BaseRegulatoryRule] = []
        for RuleCls in [SCAExemptionTagRule, ChargebackThresholdRule,  # type: ignore[name-defined]
                        RefundRightWindowRule, CookieConsentTrackingRule]:  # type: ignore[name-defined]
            class _Wrapped(BaseRegulatoryRule):  # type: ignore[misc]
                def __init__(self, cls=RuleCls, c=cfg):
                    self._r = cls()
                    self._cfg = c
                    self.name = cls.name
                    self.domain = "ecommerce"
                def evaluate(self, df):  # type: ignore[override]
                    from .base_rule import RegulatoryViolation as RV
                    violations = []
                    for v in self._r.run(df, self._cfg):
                        violations.append(RV(
                            rule_name=v.get("rule", self.name),
                            domain="ecommerce", severity=v.get("severity", "WARNING"),
                            column=v.get("column", "N/A"),
                            offending_count=int(v.get("affected_rows", 0)),
                            message=v.get("message", ""),
                            remediation=v.get("recommended_action", ""),
                        ))
                    return violations
            rules.append(_Wrapped())
        return rules

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        """
        Runs all registered rules against ``df``.

        Each rule is evaluated independently inside a try/except block so that
        a misconfigured rule cannot silently suppress other rules.

        After evaluation, RuleConflictResolver is applied to handle any
        column-level contradictions between rules from different domains.

        Returns:
            Flat, severity-sorted list of RegulatoryViolation objects.
        """
        if not self.rules:
            logger.debug("RegulatoryEngine: no rules registered — skipping.")
            self._last_violations = []
            return []

        all_violations: List[RegulatoryViolation] = []

        for rule in self.rules:
            try:
                violations = rule.evaluate(df)
                all_violations.extend(violations)

                if violations:
                    logger.warning(
                        "Regulatory rule '%s' [%s]: %d violation(s) found.",
                        rule.name, rule.domain, len(violations),
                    )
                else:
                    logger.debug(
                        "Regulatory rule '%s' [%s]: passed.", rule.name, rule.domain
                    )

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Regulatory rule '%s' raised an unexpected exception: %s. "
                    "Recording as ERROR and continuing.",
                    rule.name, exc, exc_info=True,
                )
                all_violations.append(RegulatoryViolation(
                    rule_name=rule.name,
                    domain=self.domain,
                    severity="ERROR",
                    column="N/A",
                    offending_count=0,
                    message=(
                        f"Rule '{rule.name}' raised an unexpected exception: {exc}. "
                        "Investigate rule configuration and DataFrame schema."
                    ),
                    remediation=(
                        "Review the rule's column references in config.yaml "
                        "and ensure the DataFrame schema matches expectations."
                    ),
                ))

        # ── Conflict resolution ───────────────────────────────────────────────
        resolver = RuleConflictResolver(
            strategy=self.conflict_strategy,
            primary_domain=self.domain,
        )
        resolved_violations, conflict_report = resolver.resolve(all_violations)
        if conflict_report:
            logger.info(
                "RegulatoryEngine: %s", resolver.summarize(conflict_report)
            )
        self._last_conflict_report = conflict_report

        # Deterministic ordering: CRITICAL → ERROR → WARNING
        resolved_violations.sort(key=lambda v: _SEVERITY_ORDER.get(v.severity, 99))
        self._last_violations = resolved_violations
        return resolved_violations

    def get_violating_columns(self) -> Set[str]:
        """
        Returns the set of column names that triggered violations in the
        last call to evaluate().

        Used by PipelineBridge to mask non-compliant columns before model training.
        Only includes columns from violations with severity CRITICAL or ERROR —
        WARNING violations are flagged but their columns are not masked.
        """
        return {
            v.column
            for v in self._last_violations
            if v.severity in ("CRITICAL", "ERROR") and v.column not in ("N/A", "")
        }

    def get_last_conflict_report(self) -> List[Dict[str, Any]]:
        """Returns the conflict report from the last evaluate() call."""
        return self._last_conflict_report

    def add_rule(self, rule: BaseRegulatoryRule) -> None:
        """
        Dynamically registers an additional rule at runtime.
        The preferred extension point for domain-specific rules not in config.yaml.
        """
        if not isinstance(rule, BaseRegulatoryRule):
            raise TypeError(
                f"Expected a BaseRegulatoryRule subclass, got {type(rule).__name__!r}."
            )
        self.rules.append(rule)
        logger.info(
            "RegulatoryEngine: rule '%s' [%s] registered.", rule.name, rule.domain
        )

    def __repr__(self) -> str:
        return (
            f"RegulatoryEngine(domain={self.domain!r}, "
            f"rules={[r.name for r in self.rules]}, "
            f"conflict_strategy={self.conflict_strategy!r})"
        )
