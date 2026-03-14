"""
validation/range_validator.py
------------------------------
Domain-aware range validation for Hard Gate 1.

Covers three classes of rule:
  1. Bound rules      — standard min/max per column (from config.yaml)
  2. Positivity rules — columns that must always be ≥ 0 (financial/physical)
  3. Logical rules    — cross-column inequalities (col_a ≤ col_b, etc.)
  4. Auto-inferred IQR bounds — kicks in for ANY numeric column with no
     explicit config rule, using (Q1 - 3×IQR, Q3 + 3×IQR) soft bounds.
     This ensures generic/unknown datasets still get range coverage.

Fix 2: Auto-infer IQR-based soft bounds for numeric columns with no config rule.
Fix 4: SoftValidator is fitted once per validate() call (not once per rule),
       avoiding N identical IsolationForest fits on the same DataFrame.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np
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
          auto_infer_bounds: true   # default true — IQR soft-bounds for uncovered cols
          bounds:
            - column: age
              min_value: 0
              max_value: 130
              severity: ERROR
          positivity:
            - column: loan_amount
              strict: true
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
        auto_infer_bounds: bool = True,
    ) -> None:
        self.bound_rules: List[BoundRule] = bound_rules or []
        self.positivity_rules: List[PositivityRule] = positivity_rules or []
        self.logical_rules: List[LogicalRule] = logical_rules or []
        self.auto_infer_bounds: bool = auto_infer_bounds

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

        auto_infer = bool(range_cfg.get("auto_infer_bounds", True))

        return cls(
            bound_rules=bound_rules,
            positivity_rules=positivity_rules,
            logical_rules=logical_rules,
            auto_infer_bounds=auto_infer,
        )

    def validate(self, df: pd.DataFrame) -> List[RangeViolation]:
        """Runs all registered range rules against the DataFrame.

        Fix 4: SoftValidator is fitted ONCE here, then reused for all column
        checks — avoids N identical IsolationForest fits on the same DataFrame.
        """
        violations: List[RangeViolation] = []

        # Fix 4: Fit SoftValidator ONCE per validate() call
        _sv = None
        try:
            from validation.soft_validator import SoftValidator
            _sv = SoftValidator()
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if len(num_cols) >= 2 and _sv._available:
                X_all = df[num_cols].fillna(df[num_cols].median())
                _sv._fit(X_all.values)
                logger.debug("SoftValidator: fitted on %d rows x %d cols (once)", len(df), len(num_cols))
        except Exception as exc:  # noqa: BLE001
            logger.debug("SoftValidator unavailable: %s", exc)
            _sv = None

        self._check_bounds(df, violations, _sv)
        self._check_positivity(df, violations, _sv)
        self._check_logical(df, violations)

        # Fix 2: Auto-infer IQR bounds for numeric columns not covered by any rule
        if self.auto_infer_bounds:
            self._check_auto_iqr(df, violations, _sv)

        return violations

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_bounds(
        self, df: pd.DataFrame, violations: List[RangeViolation],
        _sv: Any,
    ) -> None:
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

            for bound_val, is_min in ((rule.min_value, True), (rule.max_value, False)):
                if bound_val is None:
                    continue
                if is_min:
                    breach_mask = (series < bound_val).reindex(df.index, fill_value=False)
                    direction = "below minimum"
                    obs_str = f"Min observed: {series.min():.4g}"
                else:
                    breach_mask = (series > bound_val).reindex(df.index, fill_value=False)
                    direction = "above maximum"
                    obs_str = f"Max observed: {series.max():.4g}"

                breach = int(breach_mask.sum())
                if breach == 0:
                    continue

                if _sv is not None and _sv._fitted:
                    cls_result = _sv.classify_violations(df, col, breach_mask)
                    hard, soft = cls_result["hard_count"], cls_result["soft_count"]
                else:
                    hard, soft = breach, 0

                if hard > 0:
                    violations.append(RangeViolation(
                        rule_type="BOUND_VIOLATION",
                        severity=rule.severity.value, column=col,
                        offending_count=hard,
                        message=(
                            f"[ML:HARD] Column '{col}': {hard} value(s) {direction} "
                            f"{bound_val} (anomaly-confirmed errors). "
                            f"{obs_str}. {rule.description}"
                        ),
                    ))
                if soft > 0:
                    violations.append(RangeViolation(
                        rule_type="BOUND_VIOLATION",
                        severity="WARNING", column=col,
                        offending_count=soft,
                        message=(
                            f"[ML:SOFT] Column '{col}': {soft} value(s) {direction} "
                            f"{bound_val} (soft anomaly — possible valid novelty). "
                            f"{obs_str}. {rule.description}"
                        ),
                    ))

    def _check_positivity(
        self, df: pd.DataFrame, violations: List[RangeViolation],
        _sv: Any,
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

    def _check_auto_iqr(
        self, df: pd.DataFrame, violations: List[RangeViolation],
        _sv: Any,
    ) -> None:
        """Fix 2: For numeric columns NOT covered by any config BoundRule,
        compute IQR-based soft bounds (Q1 - 3×IQR, Q3 + 3×IQR) and flag
        extreme outliers as WARNING. Uses SoftValidator to downgrade
        confirmed-valid novelties further to explicit soft notes.
        """
        covered: Set[str] = {r.column for r in self.bound_rules}
        positivity_covered: Set[str] = {r.column for r in self.positivity_rules}

        for col in df.select_dtypes(include="number").columns:
            if col in covered:
                continue  # already has explicit config rule

            series = df[col].dropna()
            if len(series) < 20:
                continue  # too few rows for IQR to be meaningful

            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue  # constant or near-constant column

            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr

            breach_mask = ((series < lower) | (series > upper)).reindex(df.index, fill_value=False)
            breach = int(breach_mask.sum())
            if breach == 0:
                continue

            if _sv is not None and _sv._fitted:
                cls_result = _sv.classify_violations(df, col, breach_mask)
                hard, soft = cls_result["hard_count"], cls_result["soft_count"]
            else:
                hard, soft = breach, 0

            if hard > 0:
                violations.append(RangeViolation(
                    rule_type="AUTO_IQR_OUTLIER",
                    severity="WARNING",
                    column=col,
                    offending_count=hard,
                    message=(
                        f"[AUTO-IQR] Column '{col}': {hard} extreme outlier(s) "
                        f"outside ±3×IQR bounds [{lower:.4g}, {upper:.4g}]. "
                        f"Range: {series.min():.4g}–{series.max():.4g}. "
                        "Add a config BoundRule to promote this to ERROR."
                    ),
                ))
            if soft > 0:
                logger.debug(
                    "[AUTO-IQR] Column '%s': %d soft novelties beyond IQR bounds (valid per IsolationForest).",
                    col, soft,
                )
