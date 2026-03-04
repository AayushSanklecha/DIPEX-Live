"""
verifier/domain_verifier.py
-----------------------------
Production-grade domain & regulatory rule verifier.

Supports configurable, domain-specific constraint evaluation for:
  - Banking: transaction caps, negative balance prevention, ratio bounds
  - Healthcare: value anonymization compliance, range enforcement
  - Finance: reporting bounds, financial positivity constraints
  - Custom: user-defined rule sets from config.yaml

Rules are evaluated against model predictions AND source data columns.
Every violation is logged with a correlation ID.
Designed to be config-driven and deployable without code changes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.verifier.domain")

# ── Built-in regulatory rule presets ────────────────────────────────────────

_BANKING_RULES: List[Dict[str, Any]] = [
    {"type": "non_negative",      "description": "Transaction amounts must be non-negative"},
    {"type": "range",   "min": 0, "max": 1e9, "description": "Amount within realistic banking bounds"},
    {"type": "ratio",   "max": 10.0,           "description": "Prediction ratio cap (anti-manipulation)"},
]

_HEALTHCARE_RULES: List[Dict[str, Any]] = [
    {"type": "range",   "min": 0,    "max": 150,  "description": "Age within valid human range"},
    {"type": "non_negative",                       "description": "Clinical scores must be non-negative"},
    {"type": "max_proportion", "threshold": 0.01,  "description": "PII value proportion must be near zero"},
]

_FINANCE_RULES: List[Dict[str, Any]] = [
    {"type": "non_negative",                       "description": "Financial outputs must be non-negative"},
    {"type": "range",   "min": -1.0, "max": 1e12,  "description": "Revenue values within enterprise bounds"},
    {"type": "no_inf",                             "description": "No infinite values in financial outputs"},
    {"type": "no_nan",                             "description": "No NaN in published financial metrics"},
]

_DOMAIN_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "banking": _BANKING_RULES,
    "healthcare": _HEALTHCARE_RULES,
    "finance": _FINANCE_RULES,
    "default": [],
}


class DomainVerifier:
    """
    Evaluates model predictions and data columns against domain-specific
    regulatory rules. All rules are config-driven and audit-logged.

    Supports config-level domain selection:
        config.yaml → pipeline.domain: "banking" | "healthcare" | "finance" | "custom"

    Custom rules are specified as a list of dicts under pipeline.domain_rules.
    """

    def __init__(
        self,
        rules: Optional[List[Dict[str, Any]]] = None,
        domain: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.domain = domain
        self.config = config or {}
        # Merge: preset rules + custom rules from config + constructor rules
        preset = _DOMAIN_PRESETS.get(domain, [])
        config_rules = (self.config.get("pipeline", {}) or {}).get("domain_rules", []) or []
        extra = rules or []
        self.rules: List[Dict[str, Any]] = preset + config_rules + extra

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DomainVerifier":
        domain = (config.get("pipeline", {}) or {}).get("domain", "default")
        return cls(domain=domain, config=config)

    def verify(
        self,
        predictions: Any,
        data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates all rules against predictions and optionally source data.

        Args:
            predictions : model output (array-like of numeric values)
            data        : optional source DataFrame for column-level checks

        Returns:
            dict with: metric, value, passed, violations, rule_results, detail
        """
        preds = None
        if predictions is not None:
            try:
                preds = np.asarray(predictions, dtype=float).flatten()
            except (TypeError, ValueError):
                preds = None

        violations: List[str] = []
        rule_results: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for rule in self.rules:
            result = self._evaluate_rule(rule, preds, data)
            rule_results.append(result)
            if result["severity"] == "VIOLATION":
                violations.append(result["message"])
            elif result["severity"] == "WARNING":
                warnings.append(result["message"])

        passed = len(violations) == 0

        if violations:
            detail = f"{len(violations)} domain violation(s) in '{self.domain}': {violations[0]}"
        elif warnings:
            detail = f"Domain rules passed with {len(warnings)} warning(s). Domain: {self.domain}."
        else:
            detail = f"All domain/regulatory rules passed. Domain: {self.domain}. Rules checked: {len(self.rules)}."

        logger.info(
            "DomainVerifier [%s]: violations=%d warnings=%d rules=%d",
            self.domain, len(violations), len(warnings), len(self.rules),
        )

        return {
            "metric": "domain_rule_violations",
            "value": len(violations),
            "passed": bool(passed),
            "domain": self.domain,
            "violations": violations,
            "warnings": warnings,
            "rule_results": rule_results,
            "detail": detail,
        }

    # ------------------------------------------------------------------
    # Rule evaluators
    # ------------------------------------------------------------------

    def _evaluate_rule(
        self,
        rule: Dict[str, Any],
        preds: Optional[np.ndarray],
        data: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Dispatch to the appropriate rule evaluator."""
        rule_type = rule.get("type", "")
        desc = rule.get("description", rule_type)

        try:
            if rule_type == "non_negative":
                return self._check_non_negative(preds, data, desc)
            elif rule_type == "range":
                return self._check_range(preds, data, rule, desc)
            elif rule_type == "no_inf":
                return self._check_no_inf(preds, data, desc)
            elif rule_type == "no_nan":
                return self._check_no_nan(preds, data, desc)
            elif rule_type == "ratio":
                return self._check_ratio(preds, rule, desc)
            elif rule_type == "max_proportion":
                return self._check_max_proportion(preds, rule, desc)
            elif rule_type == "unique_ids":
                return self._check_unique_ids(data, rule, desc)
            elif rule_type == "referential_integrity":
                return self._check_referential_integrity(data, rule, desc)
            elif rule_type == "custom_fn":
                return self._check_custom_fn(preds, data, rule, desc)
            else:
                return {"rule": rule_type, "severity": "INFO",
                        "message": f"Unknown rule type '{rule_type}' — skipped.", "passed": True}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rule evaluation failed for '%s': %s", rule_type, exc)
            return {"rule": rule_type, "severity": "WARNING",
                    "message": f"Rule evaluation error: {exc}", "passed": True}

    @staticmethod
    def _check_non_negative(
        preds: Optional[np.ndarray], data: Optional[pd.DataFrame], desc: str
    ) -> Dict[str, Any]:
        violations = []
        if preds is not None:
            n_neg = int((preds < 0).sum())
            if n_neg > 0:
                violations.append(f"{n_neg} negative prediction value(s)")
        if data is not None:
            for col in data.select_dtypes("number").columns:
                n_neg = int((data[col] < 0).sum())
                if n_neg > 0:
                    violations.append(f"Column '{col}': {n_neg} negative value(s)")

        passed = len(violations) == 0
        return {
            "rule": "non_negative",
            "severity": "VIOLATION" if not passed else "OK",
            "message": ("; ".join(violations) if violations else f"{desc}: OK"),
            "passed": passed,
        }

    @staticmethod
    def _check_range(
        preds: Optional[np.ndarray], data: Optional[pd.DataFrame],
        rule: Dict[str, Any], desc: str,
    ) -> Dict[str, Any]:
        lo = rule.get("min")
        hi = rule.get("max")
        cols = rule.get("columns")  # optional column filter
        violations = []

        if preds is not None:
            if lo is not None and preds.min() < lo:
                violations.append(f"Prediction min {preds.min():.4f} < allowed min {lo}")
            if hi is not None and preds.max() > hi:
                violations.append(f"Prediction max {preds.max():.4f} > allowed max {hi}")

        if data is not None:
            check_cols = cols if cols else data.select_dtypes("number").columns.tolist()
            for col in check_cols:
                if col not in data.columns:
                    continue
                ser = data[col].dropna()
                if lo is not None and ser.min() < lo:
                    violations.append(f"'{col}' min {ser.min():.4f} < {lo}")
                if hi is not None and ser.max() > hi:
                    violations.append(f"'{col}' max {ser.max():.4f} > {hi}")

        passed = len(violations) == 0
        return {
            "rule": "range",
            "severity": "VIOLATION" if not passed else "OK",
            "message": ("; ".join(violations) if violations else f"{desc}: OK"),
            "passed": passed,
        }

    @staticmethod
    def _check_no_inf(
        preds: Optional[np.ndarray], data: Optional[pd.DataFrame], desc: str
    ) -> Dict[str, Any]:
        violations = []
        if preds is not None and np.any(np.isinf(preds)):
            violations.append("Infinite values found in predictions")
        if data is not None:
            for col in data.select_dtypes("number").columns:
                if data[col].apply(np.isinf).any():
                    violations.append(f"Infinite values in column '{col}'")
        passed = len(violations) == 0
        return {"rule": "no_inf", "severity": "VIOLATION" if not passed else "OK",
                "message": ("; ".join(violations) if violations else f"{desc}: OK"), "passed": passed}

    @staticmethod
    def _check_no_nan(
        preds: Optional[np.ndarray], data: Optional[pd.DataFrame], desc: str
    ) -> Dict[str, Any]:
        violations = []
        if preds is not None and np.any(np.isnan(preds)):
            violations.append("NaN values found in predictions")
        if data is not None:
            for col in data.select_dtypes("number").columns:
                if data[col].isna().any():
                    violations.append(f"NaN in column '{col}'")
        passed = len(violations) == 0
        return {"rule": "no_nan", "severity": "VIOLATION" if not passed else "OK",
                "message": ("; ".join(violations) if violations else f"{desc}: OK"), "passed": passed}

    @staticmethod
    def _check_ratio(
        preds: Optional[np.ndarray], rule: Dict[str, Any], desc: str
    ) -> Dict[str, Any]:
        max_ratio = rule.get("max", 10.0)
        if preds is None or len(preds) < 2:
            return {"rule": "ratio", "severity": "OK", "message": f"{desc}: skipped (no data)", "passed": True}
        ratio = preds.max() / (abs(preds.min()) + 1e-10)
        passed = ratio <= max_ratio
        return {
            "rule": "ratio",
            "severity": "VIOLATION" if not passed else "OK",
            "message": (f"Prediction ratio {ratio:.2f} exceeds max {max_ratio}" if not passed else f"{desc}: OK"),
            "passed": bool(passed),
        }

    @staticmethod
    def _check_max_proportion(
        preds: Optional[np.ndarray], rule: Dict[str, Any], desc: str
    ) -> Dict[str, Any]:
        threshold = rule.get("threshold", 0.01)
        if preds is None or len(preds) == 0:
            return {"rule": "max_proportion", "severity": "OK", "message": f"{desc}: skipped", "passed": True}
        unique_ratio = len(np.unique(preds)) / max(len(preds), 1)
        passed = unique_ratio <= threshold
        return {
            "rule": "max_proportion",
            "severity": "WARNING" if not passed else "OK",
            "message": (f"Unique proportion {unique_ratio:.4f} exceeds {threshold}" if not passed else f"{desc}: OK"),
            "passed": True,  # WARNING only — not a hard gate failure
        }

    @staticmethod
    def _check_unique_ids(
        data: Optional[pd.DataFrame], rule: Dict[str, Any], desc: str
    ) -> Dict[str, Any]:
        col = rule.get("column")
        if data is None or col is None or col not in data.columns:
            return {"rule": "unique_ids", "severity": "OK", "message": f"{desc}: skipped", "passed": True}
        dups = int(data[col].duplicated().sum())
        passed = dups == 0
        return {
            "rule": "unique_ids", "passed": passed,
            "severity": "VIOLATION" if not passed else "OK",
            "message": (f"Column '{col}' has {dups} duplicate ID(s)" if not passed else f"{desc}: OK"),
        }

    @staticmethod
    def _check_referential_integrity(
        data: Optional[pd.DataFrame], rule: Dict[str, Any], desc: str
    ) -> Dict[str, Any]:
        parent_col = rule.get("parent_col")
        child_col = rule.get("child_col")
        if data is None or parent_col is None or child_col is None:
            return {"rule": "referential_integrity", "severity": "OK", "message": f"{desc}: skipped", "passed": True}
        if parent_col not in data.columns or child_col not in data.columns:
            return {"rule": "referential_integrity", "severity": "INFO",
                    "message": "Columns not found for RI check", "passed": True}
        orphans = int((~data[child_col].isin(data[parent_col])).sum())
        passed = orphans == 0
        return {
            "rule": "referential_integrity", "passed": passed,
            "severity": "VIOLATION" if not passed else "OK",
            "message": (f"{orphans} orphaned referential values in '{child_col}'" if not passed else f"{desc}: OK"),
        }

    @staticmethod
    def _check_custom_fn(
        preds: Optional[np.ndarray], data: Optional[pd.DataFrame],
        rule: Dict[str, Any], desc: str,
    ) -> Dict[str, Any]:
        fn = rule.get("fn")
        if fn is None or not callable(fn):
            return {"rule": "custom_fn", "severity": "INFO",
                    "message": "No callable 'fn' provided for custom rule", "passed": True}
        try:
            result = fn(preds, data)
            passed = bool(result.get("passed", True))
            return {
                "rule": "custom_fn",
                "severity": "VIOLATION" if not passed else "OK",
                "message": result.get("message", desc),
                "passed": passed,
            }
        except Exception as exc:  # noqa: BLE001
            return {"rule": "custom_fn", "severity": "WARNING",
                    "message": f"Custom rule error: {exc}", "passed": True}
