"""
validation/range_validator.py
------------------------------
Domain-aware range validation for Hard Gate 1.

Covers three classes of rule:
  1. Bound rules      — standard min/max per column
  2. Positivity rules — columns that must always be ≥ 0 (financial/physical)
  3. Logical rules    — cross-column inequalities (col_a ≤ col_b, etc.)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

class RuleSeverity(str, Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class BoundRule:
    """Min/max constraint on a single column."""
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    severity: RuleSeverity = RuleSeverity.ERROR
    description: str = ""


@dataclass
class PositivityRule:
    """Enforces column >= 0 (or > 0 if strict=True)."""
    column: str
    strict: bool = False          # True → must be > 0; False → must be >= 0
    severity: RuleSeverity = RuleSeverity.ERROR
    description: str = ""


@dataclass
class LogicalRule:
    """Cross-column inequality: left_col <op> right_col."""
    left_col: str
    operator: str                 # "<=", "<", ">=", ">", "=="
    right_col: str
    severity: RuleSeverity = RuleSeverity.ERROR
    description: str = ""


# ---------------------------------------------------------------------------
# Violation result
# ---------------------------------------------------------------------------

@dataclass
class RangeViolation:
    rule_type: str
    severity: str
    column: str
    message: str
    offending_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "type": f"RANGE_{self.rule_type}",
            "column": self.column,
            "offending_count": self.offending_count,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_OP_MAP = {
    "<=": lambda a, b: a <= b,
    "<":  lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">":  lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


class RangeValidator:
    """
    Validates numeric columns against bound, positivity, and logical rules.

    Configuration example (from config.yaml):
      validation:
        range:
          bounds:
            - column: age
              min_value: 0
              max_value: 130
              severity: ERROR
            - column: interest_rate
              min_value: 0.0
              max_value: 1.0
              severity: WARNING
          positivity:
            - column: loan_amount
              strict: true
            - column: account_balance
              strict: false
          logical:
            - left_col: loan_amount
              operator: "<="
              right_col: credit_limit
              severity: ERROR
    """

    def __init__(
        self,
        bound_rules: Optional[List[BoundRule]] = None,
        positivity_rules: Optional[List[PositivityRule]] = None,
        logical_rules: Optional[List[LogicalRule]] = None,
    ) -> None:
        self.bound_rules: List[BoundRule] = bound_rules or []
        self.positivity_rules: List[PositivityRule] = positivity_rules or []
        self.logical_rules: List[LogicalRule] = logical_rules or []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RangeValidator":
        """Factory: build from project config dict."""
        range_cfg = config.get("validation", {}).get("range", {})

        bound_rules = [
            BoundRule(
                column=r["column"],
                min_value=r.get("min_value"),
                max_value=r.get("max_value"),
                severity=RuleSeverity(r.get("severity", "ERROR")),
                description=r.get("description", ""),
            )
            for r in range_cfg.get("bounds", [])
        ]

        positivity_rules = [
            PositivityRule(
                column=r["column"],
                strict=r.get("strict", False),
                severity=RuleSeverity(r.get("severity", "ERROR")),
                description=r.get("description", ""),
            )
            for r in range_cfg.get("positivity", [])
        ]

        logical_rules = [
            LogicalRule(
                left_col=r["left_col"],
                operator=r["operator"],
                right_col=r["right_col"],
                severity=RuleSeverity(r.get("severity", "ERROR")),
                description=r.get("description", ""),
            )
            for r in range_cfg.get("logical", [])
        ]

        return cls(
            bound_rules=bound_rules,
            positivity_rules=positivity_rules,
            logical_rules=logical_rules,
        )

    def validate(self, df: pd.DataFrame) -> List[RangeViolation]:
        """Runs all registered range rules against the DataFrame."""
        violations: List[RangeViolation] = []
        self._check_bounds(df, violations)
        self._check_positivity(df, violations)
        self._check_logical(df, violations)
        return violations

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_bounds(
        self, df: pd.DataFrame, violations: List[RangeViolation]
    ) -> None:
        # [ML] Soft validator — separates hard anomalies from valid novelties
        try:
            from validation.soft_validator import SoftValidator
            _sv: SoftValidator = SoftValidator()
        except Exception:  # noqa: BLE001
            _sv = None  # type: ignore[assignment]

        for rule in self.bound_rules:
            col = rule.column
            if col not in df.columns:
                logger.warning("BoundRule: column '%s' not in DataFrame — skipped.", col)
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            if rule.min_value is not None:
                breach_mask = (series < rule.min_value).reindex(df.index, fill_value=False)
                breach = int(breach_mask.sum())
                if breach > 0:
                    if _sv is not None:
                        cls = _sv.classify_violations(df, col, breach_mask)
                        hard, soft = cls["hard_count"], cls["soft_count"]
                    else:
                        hard, soft = breach, 0
                    if hard > 0:
                        violations.append(RangeViolation(
                            rule_type="BOUND_VIOLATION",
                            severity=rule.severity.value, column=col,
                            offending_count=hard,
                            message=(
                                f"[ML:HARD] Column '{col}': {hard} value(s) below minimum "
                                f"{rule.min_value} (anomaly-confirmed errors). "
                                f"Min observed: {series.min():.4g}. {rule.description}"
                            ),
                        ))
                    if soft > 0:
                        violations.append(RangeViolation(
                            rule_type="BOUND_VIOLATION",
                            severity="WARNING", column=col,
                            offending_count=soft,
                            message=(
                                f"[ML:SOFT] Column '{col}': {soft} value(s) below minimum "
                                f"{rule.min_value} (soft anomaly — possible valid novelty). "
                                f"Min observed: {series.min():.4g}. {rule.description}"
                            ),
                        ))

            if rule.max_value is not None:
                breach_mask = (series > rule.max_value).reindex(df.index, fill_value=False)
                breach = int(breach_mask.sum())
                if breach > 0:
                    if _sv is not None:
                        cls = _sv.classify_violations(df, col, breach_mask)
                        hard, soft = cls["hard_count"], cls["soft_count"]
                    else:
                        hard, soft = breach, 0
                    if hard > 0:
                        violations.append(RangeViolation(
                            rule_type="BOUND_VIOLATION",
                            severity=rule.severity.value, column=col,
                            offending_count=hard,
                            message=(
                                f"[ML:HARD] Column '{col}': {hard} value(s) above maximum "
                                f"{rule.max_value} (anomaly-confirmed errors). "
                                f"Max observed: {series.max():.4g}. {rule.description}"
                            ),
                        ))
                    if soft > 0:
                        violations.append(RangeViolation(
                            rule_type="BOUND_VIOLATION",
                            severity="WARNING", column=col,
                            offending_count=soft,
                            message=(
                                f"[ML:SOFT] Column '{col}': {soft} value(s) above maximum "
                                f"{rule.max_value} (soft anomaly — possible valid novelty). "
                                f"Max observed: {series.max():.4g}. {rule.description}"
                            ),
                        ))

    def _check_positivity(
        self, df: pd.DataFrame, violations: List[RangeViolation]
    ) -> None:
        for rule in self.positivity_rules:
            col = rule.column
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            series = df[col].dropna()
            if rule.strict:
                bad = (series <= 0).sum()
                op_desc = "must be strictly positive (> 0)"
            else:
                bad = (series < 0).sum()
                op_desc = "must be non-negative (≥ 0)"

            if bad > 0:
                violations.append(RangeViolation(
                    rule_type="POSITIVITY_VIOLATION",
                    severity=rule.severity.value,
                    column=col,
                    offending_count=int(bad),
                    message=(
                        f"Column '{col}' {op_desc}: {bad} invalid value(s) found. "
                        f"Min observed: {series.min():.4g}. {rule.description}"
                    ),
                ))

    def _check_logical(
        self, df: pd.DataFrame, violations: List[RangeViolation]
    ) -> None:
        for rule in self.logical_rules:
            left, right, op = rule.left_col, rule.right_col, rule.operator
            if left not in df.columns or right not in df.columns:
                logger.warning(
                    "LogicalRule '%s %s %s': one or both columns missing — skipped.",
                    left, op, right,
                )
                continue

            op_fn = _OP_MAP.get(op)
            if op_fn is None:
                logger.error("LogicalRule: unsupported operator '%s'.", op)
                continue

            pair = df[[left, right]].dropna()
            if pair.empty:
                continue

            bad = (~op_fn(pair[left], pair[right])).sum()
            if bad > 0:
                violations.append(RangeViolation(
                    rule_type="LOGICAL_INEQUALITY_VIOLATION",
                    severity=rule.severity.value,
                    column=f"{left}:{right}",
                    offending_count=int(bad),
                    message=(
                        f"Logical rule violated: '{left}' {op} '{right}' — "
                        f"{bad} row(s) break this constraint. {rule.description}"
                    ),
                ))
