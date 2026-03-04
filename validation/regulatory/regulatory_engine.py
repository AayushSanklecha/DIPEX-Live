"""
validation/regulatory/regulatory_engine.py
--------------------------------------------
Orchestrates domain-specific regulatory rule evaluation for Hard Gate 1.

The engine loads a domain profile from ``config.yaml``, instantiates the
appropriate rule set, and runs every registered rule against the incoming
DataFrame, returning a unified, severity-sorted list of ``RegulatoryViolation``
objects.

Supported domains
-----------------
``"banking"``    — PositiveAmountRule, AMLThresholdRule, LoanRatioRule,
                   RepaymentConsistencyRule
``"healthcare"`` — AgeRangeRule, VitalSignsRule, DiagnosisCodeFormatRule,
                   PHIPresenceRule
``"generic"``    — No domain rules applied (safe default for non-regulated data)

Extension pattern
-----------------
To add a new domain::

    class MyCustomRule(BaseRegulatoryRule):
        name = "my_rule"
        domain = "custom"
        def evaluate(self, df): ...

    engine = RegulatoryEngine(domain="custom")
    engine.add_rule(MyCustomRule())
    violations = engine.evaluate(df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_rule import BaseRegulatoryRule, RegulatoryViolation
from .banking_rules import (
    AMLThresholdRule,
    LoanRatioRule,
    PositiveAmountRule,
    RepaymentConsistencyRule,
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
    DiagnosisCodeFormatRule,
    PHIPresenceRule,
    VitalSignsRule,
)

logger = logging.getLogger(__name__)

# Severity sort order for deterministic output ordering
_SEVERITY_ORDER: Dict[str, int] = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}


class RegulatoryEngine:
    """
    Pluggable regulatory rule engine.

    Instantiation — from project config (recommended)::

        engine = RegulatoryEngine.from_config(config)
        violations = engine.evaluate(df)

    Instantiation — manual (for testing or custom domains)::

        engine = RegulatoryEngine(domain="banking", rules=[PositiveAmountRule([...])])
        violations = engine.evaluate(df)

    Runtime rule injection::

        engine.add_rule(MyCustomRule())
    """

    def __init__(
        self,
        domain: str = "generic",
        rules: Optional[List[BaseRegulatoryRule]] = None,
    ) -> None:
        self.domain = domain.lower()
        self.rules: List[BaseRegulatoryRule] = rules if rules is not None else []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RegulatoryEngine":
        """
        Builds a fully configured ``RegulatoryEngine`` from the project config dict.

        The ``validation.regulatory`` section drives domain selection and
        parameter loading.  Missing or empty sections are handled gracefully
        — the engine falls back to ``"generic"`` (no rules applied).
        """
        reg_cfg = config.get("validation", {}).get("regulatory", {})
        domain = str(reg_cfg.get("domain", "generic")).lower()
        rules: List[BaseRegulatoryRule] = []

        if domain == "banking":
            rules = cls._build_banking_rules(reg_cfg.get("banking", {}))
            logger.info(
                "RegulatoryEngine: banking domain loaded — %d rule(s).", len(rules)
            )
        elif domain == "healthcare":
            rules = cls._build_healthcare_rules(reg_cfg.get("healthcare", {}))
            logger.info(
                "RegulatoryEngine: healthcare domain loaded — %d rule(s).", len(rules)
            )
        elif domain == "finance":
            rules = cls._build_finance_rules(reg_cfg.get("finance", {}))
            logger.info(
                "RegulatoryEngine: finance domain loaded — %d rule(s).", len(rules)
            )
        else:
            logger.info(
                "RegulatoryEngine: domain='%s' — no domain-specific rules loaded.", domain
            )

        return cls(domain=domain, rules=rules)

    # ------------------------------------------------------------------
    # Domain rule builders (static factories)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_banking_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        """Instantiates all banking-domain rules from the config sub-section."""
        rules: List[BaseRegulatoryRule] = []

        amount_cols: List[str] = cfg.get("amount_columns", [])
        if amount_cols:
            rules.append(
                PositiveAmountRule(
                    amount_columns=amount_cols,
                    allow_zero=cfg.get("allow_zero_amounts", False),
                )
            )

        aml_col: Optional[str] = cfg.get("aml_amount_column")
        if aml_col:
            rules.append(
                AMLThresholdRule(
                    amount_column=aml_col,
                    threshold=float(cfg.get("aml_threshold", 10_000.0)),
                    currency_column=cfg.get("currency_column"),
                )
            )

        ltv_cfg: Dict[str, Any] = cfg.get("loan_ratio", {})
        if ltv_cfg.get("loan_col") and ltv_cfg.get("value_col"):
            rules.append(
                LoanRatioRule(
                    loan_col=str(ltv_cfg["loan_col"]),
                    value_col=str(ltv_cfg["value_col"]),
                    max_ltv=float(ltv_cfg.get("max_ltv", 0.90)),
                    severity=str(ltv_cfg.get("severity", "ERROR")),
                )
            )

        repay_cfg: Dict[str, Any] = cfg.get("repayment", {})
        if repay_cfg.get("repayment_col") and repay_cfg.get("balance_col"):
            rules.append(
                RepaymentConsistencyRule(
                    repayment_col=str(repay_cfg["repayment_col"]),
                    balance_col=str(repay_cfg["balance_col"]),
                )
            )

        return rules

    @staticmethod
    def _build_finance_rules(cfg: Dict[str, Any]) -> List[BaseRegulatoryRule]:
        """Instantiates all finance-domain rules from the config sub-section."""
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
        """Instantiates all healthcare-domain rules from the config sub-section."""
        rules: List[BaseRegulatoryRule] = []

        rules.append(
            AgeRangeRule(
                age_column=str(cfg.get("age_column", "age")),
                min_age=float(cfg.get("min_age", 0.0)),
                max_age=float(cfg.get("max_age", 130.0)),
            )
        )

        vital_bounds: Optional[Dict[str, Dict[str, float]]] = cfg.get(
            "vital_sign_bounds"
        )
        rules.append(VitalSignsRule(column_bounds=vital_bounds))

        diag_cols: List[str] = cfg.get("diagnosis_columns", ["diagnosis_code"])
        if diag_cols:
            rules.append(DiagnosisCodeFormatRule(diagnosis_columns=diag_cols))

        rules.append(
            PHIPresenceRule(
                text_columns=cfg.get("text_columns_for_phi_scan"),
                allowed_phi_columns=cfg.get("allowed_phi_columns", []),
            )
        )

        return rules

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        """
        Runs all registered rules against ``df``.

        Each rule is evaluated independently inside a try/except block so that
        a misconfigured or buggy rule cannot silently suppress evaluation of
        subsequent rules.  Rule exceptions are surfaced as ``ERROR``-severity
        violations.

        Returns:
            Flat list of ``RegulatoryViolation`` objects,
            sorted CRITICAL → ERROR → WARNING.
        """
        if not self.rules:
            logger.debug("RegulatoryEngine: no rules registered — skipping.")
            return []

        all_violations: List[RegulatoryViolation] = []

        for rule in self.rules:
            try:
                violations = rule.evaluate(df)
                all_violations.extend(violations)

                if violations:
                    logger.warning(
                        "Regulatory rule '%s' [%s]: %d violation(s) found.",
                        rule.name,
                        rule.domain,
                        len(violations),
                    )
                else:
                    logger.debug(
                        "Regulatory rule '%s' [%s]: passed.", rule.name, rule.domain
                    )

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Regulatory rule '%s' raised an unexpected exception: %s. "
                    "Recording as ERROR and continuing.",
                    rule.name,
                    exc,
                    exc_info=True,
                )
                all_violations.append(
                    RegulatoryViolation(
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
                    )
                )

        # Deterministic ordering: CRITICAL → ERROR → WARNING
        all_violations.sort(
            key=lambda v: _SEVERITY_ORDER.get(v.severity, 99)
        )
        return all_violations

    def add_rule(self, rule: BaseRegulatoryRule) -> None:
        """
        Dynamically registers an additional rule at runtime.

        This is the preferred extension point for domain-specific rules that
        are not driven by ``config.yaml``.
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
            f"rules={[r.name for r in self.rules]})"
        )
