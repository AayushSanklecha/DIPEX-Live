"""
validation/null_validator.py
-----------------------------
Dedicated null-threshold validation engine for Hard Gate 1.

Architecture note:
  - ``critical_fields``: any single null → CRITICAL severity (instant hard-fail)
  - ``column_thresholds``: per-column override of the global missing-value budget
  - ``global_threshold``: default budget applied to all other columns

Severity levels (in ascending order of severity):
  WARNING   < threshold exceeded on a non-critical column, configurable
  ERROR     > global threshold exceeded
  CRITICAL  > any null in a declared critical field
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class NullViolation:
    column: str
    severity: str           # "WARNING" | "ERROR" | "CRITICAL"
    null_count: int
    null_pct: float
    threshold: float
    is_critical_field: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "severity": self.severity,
            "type": "NULL_VIOLATION",
            "null_count": self.null_count,
            "null_pct": round(self.null_pct, 6),
            "threshold": self.threshold,
            "is_critical_field": self.is_critical_field,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class NullValidator:
    """
    Multi-tier null-threshold validator.

    Configuration example (from config.yaml):
      validation:
        null:
          global_threshold: 0.10
          critical_fields: [transaction_id, patient_id, account_number]
          column_thresholds:
            loan_amount: 0.0       # zero tolerance
            description: 0.30      # looser budget for free-text fields
          warn_threshold: 0.05     # anything above this but below error gets WARNING
    """

    def __init__(
        self,
        global_threshold: float = 0.10,
        critical_fields: Optional[List[str]] = None,
        column_thresholds: Optional[Dict[str, float]] = None,
        warn_threshold: float = 0.05,
    ) -> None:
        self.global_threshold = global_threshold
        self.critical_fields: List[str] = critical_fields or []
        self.column_thresholds: Dict[str, float] = column_thresholds or {}
        self.warn_threshold = warn_threshold

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NullValidator":
        """Factory: constructs NullValidator from the project config dict."""
        null_cfg = config.get("validation", {}).get("null", {})
        return cls(
            global_threshold=null_cfg.get("global_threshold", 0.10),
            critical_fields=null_cfg.get("critical_fields", []),
            column_thresholds=null_cfg.get("column_thresholds", {}),
            warn_threshold=null_cfg.get("warn_threshold", 0.05),
        )

    def validate(self, df: pd.DataFrame, dataset_id: str = "default") -> List[NullViolation]:
        """
        Runs null validation on every column in the DataFrame.

        Returns:
            Ordered list of NullViolation objects (CRITICAL first).
        """
        violations: List[NullViolation] = []
        n_rows = len(df)

        if n_rows == 0:
            logger.warning("NullValidator received an empty DataFrame.")
            return violations

        null_counts = df.isnull().sum()
        null_pcts = null_counts / n_rows

        for col in df.columns:
            null_count = int(null_counts[col])
            null_pct = float(null_pcts[col])
            is_critical = col in self.critical_fields
            threshold = self.column_thresholds.get(col, self.global_threshold)

            # [RL] Dynamic threshold override for non-critical, non-hardcoded columns
            if not is_critical and col not in self.column_thresholds:
                try:
                    from validation.rl_threshold_tuner import get_rl_tuner
                    threshold = get_rl_tuner().get_threshold(dataset_id, col, default=threshold)
                except Exception:  # noqa: BLE001
                    pass  # fallback: keep config threshold

            if is_critical and null_count > 0:
                violation = NullViolation(
                    column=col,
                    severity="CRITICAL",
                    null_count=null_count,
                    null_pct=null_pct,
                    threshold=0.0,
                    is_critical_field=True,
                    message=(
                        f"[CRITICAL] Column '{col}' is a critical field — "
                        f"{null_count} null value(s) found ({null_pct:.2%}). "
                        "Zero nulls are permitted in critical fields."
                    ),
                )
                violations.append(violation)
                logger.error(violation.message)

            elif null_pct > threshold:
                severity = "ERROR"
                violation = NullViolation(
                    column=col,
                    severity=severity,
                    null_count=null_count,
                    null_pct=null_pct,
                    threshold=threshold,
                    is_critical_field=False,
                    message=(
                        f"Column '{col}' has {null_pct:.2%} missing values, "
                        f"exceeding the allowed threshold of {threshold:.2%}."
                    ),
                )
                violations.append(violation)
                logger.error(violation.message)

            elif null_pct > self.warn_threshold:
                violation = NullViolation(
                    column=col,
                    severity="WARNING",
                    null_count=null_count,
                    null_pct=null_pct,
                    threshold=threshold,
                    is_critical_field=False,
                    message=(
                        f"Column '{col}' has {null_pct:.2%} missing values "
                        f"(above warning level {self.warn_threshold:.2%})."
                    ),
                )
                violations.append(violation)
                logger.warning(violation.message)

        # Return CRITICAL first for readability in audit logs
        violations.sort(key=lambda v: {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}[v.severity])
        return violations
