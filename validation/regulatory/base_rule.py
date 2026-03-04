"""
validation/regulatory/base_rule.py
------------------------------------
Abstract base class for all regulatory domain rules.

Every domain rule must implement `evaluate(df) -> List[RegulatoryViolation]`.
This enforces a uniform contract across banking, healthcare, and future domains.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegulatoryViolation:
    """Structured result from a single regulatory rule evaluation."""
    rule_name: str
    domain: str                    # "banking" | "healthcare" | "generic"
    severity: str                  # "WARNING" | "ERROR" | "CRITICAL"
    column: str                    # Primary column involved (or "N/A")
    offending_count: int           # Number of offending rows (0 = schema-level)
    message: str
    remediation: str = ""          # Suggested fix for the data engineer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "domain": self.domain,
            "severity": self.severity,
            "type": "REGULATORY_VIOLATION",
            "column": self.column,
            "offending_count": self.offending_count,
            "message": self.message,
            "remediation": self.remediation,
        }


class BaseRegulatoryRule(abc.ABC):
    """
    Abstract base for all domain-specific regulatory rules.

    Subclasses declare:
      - ``name``   — unique rule identifier
      - ``domain`` — domain this rule belongs to
      - ``evaluate(df)`` — returns list of RegulatoryViolation objects
    """

    name: str = "base_rule"
    domain: str = "generic"

    @abc.abstractmethod
    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        """
        Evaluate the rule against the given DataFrame.

        Returns:
            Empty list if the rule passes; one or more RegulatoryViolation
            objects if violations are detected.
        """
        ...

    def _col_missing(self, col: str, df: pd.DataFrame) -> bool:
        """Helper: log and return True if the column doesn't exist."""
        if col not in df.columns:
            logger.debug(
                "[%s] Column '%s' not found — rule skipped.", self.name, col
            )
            return True
        return False
