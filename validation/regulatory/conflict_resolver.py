"""
validation/regulatory/conflict_resolver.py
-------------------------------------------
Rule Conflict Resolution Engine for DIPEX.

Detects when two or more rules produce contradicting verdicts on the same
column (e.g., RangeValidator passes a column but PositiveAmountRule fails it)
and resolves the conflict according to a configured strategy.

Resolution strategies
---------------------
strictest_wins   — CRITICAL > ERROR > WARNING > PASS. The most severe verdict
                   for each column wins. This is the safe, conservative default.
domain_priority  — The rule matching the primary configured domain takes
                   precedence. If the primary domain rule passes, conflicting
                   rules from other domains are downgraded to WARNING.
advisory_only    — Conflicts are detected and logged but neither rule's verdict
                   is overridden. A RULE_CONFLICT advisory violation is appended.

Usage
-----
    from validation.regulatory.conflict_resolver import RuleConflictResolver

    resolver = RuleConflictResolver(strategy="strictest_wins", primary_domain="banking")
    resolved, conflict_report = resolver.resolve(violations)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_rule import RegulatoryViolation

logger = logging.getLogger(__name__)

# Severity ordering (lower index = higher severity)
_SEVERITY_RANK: Dict[str, int] = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}


class RuleConflictResolver:
    """
    Detects and resolves conflicts between regulatory rules that target
    the same column with contradicting severity verdicts.

    Parameters
    ----------
    strategy       : Resolution strategy: 'strictest_wins' | 'domain_priority' | 'advisory_only'
    primary_domain : Active pipeline domain (used by 'domain_priority' strategy)
    """

    def __init__(
        self,
        strategy: str = "strictest_wins",
        primary_domain: str = "generic",
    ) -> None:
        if strategy not in ("strictest_wins", "domain_priority", "advisory_only"):
            raise ValueError(
                f"Unknown conflict strategy '{strategy}'. "
                "Use 'strictest_wins', 'domain_priority', or 'advisory_only'."
            )
        self.strategy = strategy
        self.primary_domain = primary_domain.lower()

    def resolve(
        self,
        violations: List[RegulatoryViolation],
    ) -> Tuple[List[RegulatoryViolation], List[Dict[str, Any]]]:
        """
        Detect and resolve conflicts in a list of violations.

        A conflict exists when two violations target the same column but come
        from different domains (or different rules) — signalling that the rules
        disagree on the status of that column.

        Returns
        -------
        resolved_violations : List[RegulatoryViolation]
            Violations after conflict resolution applied.
        conflict_report : List[Dict]
            Human-readable record of each conflict and its resolution.
        """
        if not violations:
            return [], []

        # Group violations by column
        by_column: Dict[str, List[RegulatoryViolation]] = {}
        for v in violations:
            by_column.setdefault(v.column, []).append(v)

        resolved: List[RegulatoryViolation] = []
        conflict_report: List[Dict[str, Any]] = []

        for col, col_violations in by_column.items():
            if len(col_violations) <= 1:
                # No conflict possible with a single rule
                resolved.extend(col_violations)
                continue

            # Check if there is a genuine conflict: multiple rules, multiple domains
            domains = {v.domain for v in col_violations}
            rules = {v.rule_name for v in col_violations}
            if len(rules) <= 1:
                resolved.extend(col_violations)
                continue

            # Conflict detected
            conflict_entry = {
                "column": col,
                "conflicting_rules": [v.rule_name for v in col_violations],
                "conflicting_domains": sorted(domains),
                "severities": {v.rule_name: v.severity for v in col_violations},
                "strategy_applied": self.strategy,
            }

            if self.strategy == "strictest_wins":
                # Keep only the highest-severity violation(s) for this column
                min_rank = min(_SEVERITY_RANK.get(v.severity, 99) for v in col_violations)
                winners = [v for v in col_violations
                           if _SEVERITY_RANK.get(v.severity, 99) == min_rank]
                resolved.extend(winners)
                conflict_entry["resolution"] = (
                    f"Strictest severity '{winners[0].severity}' wins from rule "
                    f"'{winners[0].rule_name}'. "
                    f"{len(col_violations) - len(winners)} lower-severity verdict(s) suppressed."
                )
                logger.info(
                    "[ConflictResolver] Column '%s': %d conflicting rules resolved via "
                    "strictest_wins → '%s' from '%s'.",
                    col, len(col_violations), winners[0].severity, winners[0].rule_name,
                )

            elif self.strategy == "domain_priority":
                # Primary domain rule wins; others downgraded to WARNING advisory
                primary_violations = [v for v in col_violations if v.domain == self.primary_domain]
                other_violations = [v for v in col_violations if v.domain != self.primary_domain]

                if primary_violations:
                    resolved.extend(primary_violations)
                    # Downgrade others to WARNING advisories
                    for v in other_violations:
                        advisory = RegulatoryViolation(
                            rule_name=f"{v.rule_name}_advisory",
                            domain=v.domain,
                            severity="WARNING",
                            column=v.column,
                            offending_count=v.offending_count,
                            message=f"[ADVISORY — domain_priority] {v.message}",
                            remediation=v.remediation,
                        )
                        resolved.append(advisory)
                    conflict_entry["resolution"] = (
                        f"Primary domain '{self.primary_domain}' rule took precedence. "
                        f"{len(other_violations)} non-primary rule(s) downgraded to WARNING."
                    )
                else:
                    # No primary domain rule for this column — fall back to strictest_wins
                    min_rank = min(_SEVERITY_RANK.get(v.severity, 99) for v in col_violations)
                    winners = [v for v in col_violations
                               if _SEVERITY_RANK.get(v.severity, 99) == min_rank]
                    resolved.extend(winners)
                    conflict_entry["resolution"] = (
                        "No primary domain rule found — fallback to strictest_wins."
                    )
                logger.info(
                    "[ConflictResolver] Column '%s': domain_priority applied "
                    "(primary=%s).", col, self.primary_domain,
                )

            else:  # advisory_only
                # Keep ALL violations but append a RULE_CONFLICT advisory
                resolved.extend(col_violations)
                advisory = RegulatoryViolation(
                    rule_name="rule_conflict",
                    domain="meta",
                    severity="WARNING",
                    column=col,
                    offending_count=len(col_violations),
                    message=(
                        f"[RULE_CONFLICT] Column '{col}' is targeted by {len(col_violations)} "
                        f"rules from domains {sorted(domains)} with conflicting outcomes. "
                        f"Rules: {[v.rule_name for v in col_violations]}. "
                        f"No automatic resolution applied (advisory_only mode)."
                    ),
                    remediation=(
                        "Review conflicting rule configurations. Consider switching to "
                        "'strictest_wins' or 'domain_priority' strategy to auto-resolve."
                    ),
                )
                resolved.append(advisory)
                conflict_entry["resolution"] = (
                    "advisory_only: all violations kept, RULE_CONFLICT advisory added."
                )
                logger.info(
                    "[ConflictResolver] Column '%s': advisory added for %d conflicting rules.",
                    col, len(col_violations),
                )

            conflict_report.append(conflict_entry)

        return resolved, conflict_report

    def summarize(self, conflict_report: List[Dict[str, Any]]) -> str:
        """Returns a one-line summary of the conflict report for logging."""
        if not conflict_report:
            return "No rule conflicts detected."
        cols = [r["column"] for r in conflict_report]
        return (
            f"{len(conflict_report)} rule conflict(s) resolved via '{self.strategy}' "
            f"on column(s): {cols}."
        )
