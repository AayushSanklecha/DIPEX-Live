"""
analyst/mentorship_engine.py
------------------------------
Senior analyst — Mentorship and Quality Review module.

Acts as a senior analyst reviewing junior/mid-level work, enforcing:
  - SQL logic and performance antipatterns
  - Dashboard visualization standards
  - Overconfidence warnings in analysis
  - KPI naming and formula completeness standards
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("dipex.mentorship")


# SQL antipatterns to flag
_SQL_ANTIPATTERNS: List[Dict[str, str]] = [
    {
        "pattern": r"\bSELECT\s+\*\b",
        "severity": "WARNING",
        "flag": "SELECT_STAR",
        "message": "Avoid SELECT * in production. Explicitly name required columns.",
    },
    {
        "pattern": r"\bNOT IN\s*\(",
        "severity": "WARNING",
        "flag": "NOT_IN_SUBQUERY",
        "message": "NOT IN with subquery fails on NULLs. Prefer LEFT JOIN / NOT EXISTS.",
    },
    {
        "pattern": r"\bFUNCTION\s+\w+\([^)]*\)\s+ON\s+\w+\.",
        "severity": "INFO",
        "flag": "FUNCTION_ON_INDEX_COL",
        "message": "Applying function to indexed column prevents index use. Refactor if possible.",
    },
    {
        "pattern": r"\bORDER BY\s+\d+\b",
        "severity": "INFO",
        "flag": "ORDER_BY_POSITION",
        "message": "ORDER BY column position is brittle. Use explicit column names.",
    },
    {
        "pattern": r"\bSELECT\b.+\bGROUP BY\b.+\bHAVING\b.+\bORDER BY\b",
        "severity": "INFO",
        "flag": "COMPLEX_QUERY_NO_COMMENT",
        "message": "Complex query detected. Add inline comments for maintainability.",
    },
    {
        "pattern": r"\b(CURSOR|FOREACH|WHILE)\b",
        "severity": "WARNING",
        "flag": "LOOP_IN_SQL",
        "message": "Row-by-row processing in SQL is slow. Prefer set-based operations.",
    },
]

# Dashboard quality checklist
_DASHBOARD_CHECKS: List[Dict[str, Any]] = [
    {"key": "title", "label": "Chart title present", "required": True},
    {"key": "x_axis.label", "label": "X-axis label present", "required": True},
    {"key": "y_axis.label", "label": "Y-axis label present", "required": True},
    {"key": "confidence_badge", "label": "Confidence badge present", "required": True},
    {"key": "gold_artefact_id", "label": "References Gold artefact", "required": True},
    {"key": "palette", "label": "Color palette defined (color-blind safe)", "required": False},
    {"key": "misleading_flags", "label": "No misleading visualizations", "required": True},
]


class MentorshipEngine:
    """
    Senior analyst review layer. Reviews work produced by junior/mid analysts
    and flags quality issues, logical errors, and standard violations.
    """

    def review_sql(self, query: str) -> Dict[str, Any]:
        """
        Review SQL query for logic issues, performance antipatterns, and correctness.

        Args:
            query: SQL query string to review

        Returns:
            {issues: [{severity, flag, message}], overall_quality: str, approved: bool}
        """
        issues: List[Dict[str, str]] = []

        for antipattern in _SQL_ANTIPATTERNS:
            if re.search(antipattern["pattern"], query, re.IGNORECASE | re.DOTALL):
                issues.append({
                    "severity": antipattern["severity"],
                    "flag": antipattern["flag"],
                    "message": antipattern["message"],
                })

        # Security: SQL injection risk
        if re.search(r"(--|\bEXEC\b|\bEXECUTE\b|;\s*DROP|;\s*DELETE|;\s*TRUNCATE)", query, re.IGNORECASE):
            issues.append({
                "severity": "CRITICAL",
                "flag": "SQL_INJECTION_RISK",
                "message": "Potential SQL injection pattern detected. Review immediately.",
            })

        # Check for missing WHERE on UPDATE/DELETE
        if re.search(r"\b(UPDATE|DELETE)\b", query, re.IGNORECASE) and \
                not re.search(r"\bWHERE\b", query, re.IGNORECASE):
            issues.append({
                "severity": "CRITICAL",
                "flag": "MISSING_WHERE_CLAUSE",
                "message": "UPDATE or DELETE without WHERE clause — will affect all rows!",
            })

        critical = [i for i in issues if i["severity"] == "CRITICAL"]
        warnings = [i for i in issues if i["severity"] == "WARNING"]

        quality = "excellent" if not issues else "good" if not critical and not warnings else "needs_review" if not critical else "rejected"

        return {
            "issues": issues,
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "overall_quality": quality,
            "approved": quality in ("excellent", "good"),
            "recommendation": (
                "SQL approved ✓" if quality in ("excellent", "good")
                else f"SQL rejected — {len(critical)} critical issue(s) found. Fix before use."
            ),
        }

    def approve_dashboard(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review a chart specification for quality and completeness.

        Args:
            spec: Chart spec dict (as generated by visualization_engine)

        Returns:
            {approved: bool, checks: [...], overall_quality: str}
        """
        checks = []
        failed_required = []

        def get_nested(d: Dict, key: str) -> Any:
            """Get value from dotted path key."""
            parts = key.split(".")
            cur = d
            for p in parts:
                if isinstance(cur, dict):
                    cur = cur.get(p)
                else:
                    return None
            return cur

        for chk in _DASHBOARD_CHECKS:
            key = chk["key"]
            label = chk["label"]
            required = chk["required"]

            val = get_nested(spec, key)

            # Special handling for misleading_flags: present means FAIL
            if key == "misleading_flags":
                flags = val or []
                high_severity = [f for f in flags if f.get("severity") == "WARNING"]
                passed = len(high_severity) == 0
                detail = f"{len(high_severity)} WARNING-level misleading flags" if not passed else "Clean"
            else:
                passed = val is not None and val != "" and val != []
                detail = f"Value: {str(val)[:60]}" if passed else "Not set"

            checks.append({
                "check": label,
                "passed": passed,
                "required": required,
                "detail": detail,
            })

            if required and not passed:
                failed_required.append(label)

        approved = len(failed_required) == 0
        quality = "approved" if approved else "rejected"

        return {
            "approved": approved,
            "overall_quality": quality,
            "checks": checks,
            "failed_required": failed_required,
            "recommendation": (
                "Dashboard spec approved ✓" if approved
                else f"Dashboard rejected — {len(failed_required)} required check(s) failed: "
                     + ", ".join(failed_required)
            ),
        }

    def review_junior_analysis(
        self,
        gold_artefact: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Review a Junior analyst's Gold artefact for quality and overconfidence.

        Checks:
        - Confidence badge present (QA-gated)
        - No overconfident language in summary
        - Assumptions documented
        - Sample size adequate
        - Limitations noted

        Returns:
            {approved: bool, flags: [...], recommendation: str}
        """
        flags: List[Dict[str, str]] = []

        # Check confidence badge
        if not gold_artefact.get("confidence_score"):
            flags.append({
                "severity": "CRITICAL",
                "flag": "MISSING_CONFIDENCE_SCORE",
                "message": "No confidence score — analysis has not been QA-gated.",
            })

        # Check for overconfident language
        summary = str(gold_artefact.get("summary", ""))
        overconfident_words = ["definitely", "certainly", "always", "never", "perfect", "guaranteed", "proven"]
        found = [w for w in overconfident_words if w in summary.lower()]
        if found:
            flags.append({
                "severity": "WARNING",
                "flag": "OVERCONFIDENT_LANGUAGE",
                "message": f"Overconfident language detected: {', '.join(found)}. Use probabilistic language.",
            })

        # Check for assumptions documentation
        if not gold_artefact.get("assumptions") and not gold_artefact.get("notes"):
            flags.append({
                "severity": "INFO",
                "flag": "MISSING_ASSUMPTIONS",
                "message": "No assumptions or notes documented. Senior analysts should always state assumptions.",
            })

        # Check sample size
        n = gold_artefact.get("n_rows") or gold_artefact.get("record_count") or 0
        if n < 30:
            flags.append({
                "severity": "WARNING",
                "flag": "SMALL_SAMPLE",
                "message": f"Sample size n={n} is very small. Results may not generalize. State this limitation.",
            })

        critical = [f for f in flags if f["severity"] == "CRITICAL"]
        warnings = [f for f in flags if f["severity"] == "WARNING"]
        approved = len(critical) == 0

        return {
            "approved": approved,
            "flags": flags,
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "recommendation": (
                "Analysis approved ✓ — proceed with reporting." if approved
                else f"Analysis rejected — {len(critical)} critical issue(s) must be resolved."
            ),
        }

    def enforce_kpi_standards(
        self,
        kpi_dict: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Enforce KPI naming and definition standards:
        - Name: Title Case, no underscores, max 40 chars
        - Formula: non-empty, uses words not column names
        - Owner: must be set
        - Threshold: should be numeric
        - Unit: should be set

        Returns:
            {compliant: bool, violations: [...], compliant_kpis: [...]}
        """
        violations = []
        compliant_kpis = []

        for name, kpi in kpi_dict.items():
            kpi_violations = []

            # Name standards
            if any(c in name for c in "_-"):
                kpi_violations.append(f"Name '{name}': use spaces not underscores/dashes")
            if len(name) > 40:
                kpi_violations.append(f"Name '{name}': exceeds 40 characters")
            if name != name.title():
                kpi_violations.append(f"Name '{name}': should be Title Case")

            # Formula
            if not kpi.get("formula") or kpi["formula"] == "DEFINE_FORMULA":
                kpi_violations.append(f"KPI '{name}': formula is not defined")

            # Owner
            if not kpi.get("owner"):
                kpi_violations.append(f"KPI '{name}': owner is not set")

            # Threshold
            threshold = kpi.get("threshold")
            if threshold is not None:
                try:
                    float(str(threshold).replace(">", "").replace("<", "").replace("=", "").strip())
                except ValueError:
                    kpi_violations.append(f"KPI '{name}': threshold '{threshold}' is not numeric-parseable")

            if kpi_violations:
                violations.extend(kpi_violations)
            else:
                compliant_kpis.append(name)

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "compliant_kpis": compliant_kpis,
            "non_compliant_kpis": [n for n in kpi_dict if n not in compliant_kpis],
            "recommendation": (
                f"All {len(kpi_dict)} KPIs meet standards ✓" if not violations
                else f"{len(violations)} standard violation(s) found — review before publishing."
            ),
        }

    def review_logic(
        self,
        operation: str,
        df: "pd.DataFrame",
        target_col: Optional[str] = None,
    ) -> "MentorshipReview":
        """
        Backward-compat API for AnalystOrchestrator.
        Performs a lightweight logic and sanity review on a Gold artefact DataFrame.

        Returns a MentorshipReview with a score and approved flag.
        """
        comments: List["ReviewComment"] = []
        score = 100.0

        # Basic sanity checks
        if df is None or len(df) == 0:
            comments.append(ReviewComment(
                severity="CRITICAL",
                flag="EMPTY_DATAFRAME",
                message="DataFrame is empty — no results to review."
            ))
            score -= 50.0

        # Column count
        if df is not None and len(df.columns) == 0:
            comments.append(ReviewComment(
                severity="CRITICAL",
                flag="NO_COLUMNS",
                message="DataFrame has no columns."
            ))
            score -= 40.0

        # Target column existence
        if target_col and df is not None and target_col not in df.columns:
            comments.append(ReviewComment(
                severity="WARNING",
                flag="TARGET_COL_MISSING",
                message=f"Target column '{target_col}' not found in DataFrame."
            ))
            score -= 15.0

        # Large null rate
        if df is not None and len(df) > 0:
            null_rate = float(df.isnull().mean().mean())
            if null_rate > 0.30:
                comments.append(ReviewComment(
                    severity="WARNING",
                    flag="HIGH_NULL_RATE",
                    message=f"Overall null rate {null_rate:.1%} exceeds 30%. Data quality is low."
                ))
                score -= 10.0

        critical_flags = [c for c in comments if c.severity == "CRITICAL"]
        approved = len(critical_flags) == 0 and score >= 60.0

        return MentorshipReview(
            operation=operation,
            approved=approved,
            score=max(0.0, score),
            comments=comments,
        )


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dc_m


@_dc_m
class ReviewComment:
    """Single review comment from MentorshipEngine."""
    severity: str     # CRITICAL | WARNING | INFO
    flag: str
    message: str


@_dc_m
class MentorshipReview:
    """Review result from MentorshipEngine.review_logic()."""
    operation: str
    approved: bool
    score: float
    comments: List[ReviewComment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "approved": self.approved,
            "score": self.score,
            "critical_count": sum(1 for c in self.comments if c.severity == "CRITICAL"),
            "warning_count": sum(1 for c in self.comments if c.severity == "WARNING"),
        }


# ══════════════════════════════════════════════════════════════════════════════
# TEST-CONTRACT DATACLASS + OVERRIDES
# ══════════════════════════════════════════════════════════════════════════════

@_dc_m
class SQLReviewResult:
    """
    Return type for MentorshipEngine.review_sql() — test-contract compatible.
    Attributes align with test assertions: r.warnings, r.blockers, r.score, r.approved.
    """
    warnings: int
    blockers: int
    score: float
    approved: bool
    issues: list


def _review_sql_result(query: str) -> "SQLReviewResult":
    """
    Stateless SQL review function returning SQLReviewResult dataclass.
    Consumed by the patched MentorshipEngine.review_sql().
    """
    import re as _re
    issues_list = []

    # Check antipatterns from _SQL_ANTIPATTERNS list
    for ap in _SQL_ANTIPATTERNS:
        pattern = ap["pattern"]
        # Fix: SELECT_STAR pattern has a broken \b after * (non-word char)
        # Use a more permissive pattern that avoids the word boundary issue
        if "STAR" in ap.get("flag", "") or r"\*\b" in pattern:
            # Replace the trailing \b with (?=\s|$|,) to avoid the word-boundary issue
            fixed_pattern = pattern.replace(r"\*\b", r"\*(?=\s|$|,|;)")
            matched = _re.search(fixed_pattern, query, _re.IGNORECASE | _re.DOTALL)
        else:
            matched = _re.search(pattern, query, _re.IGNORECASE | _re.DOTALL)
        if matched:
            issues_list.append({"severity": ap["severity"], "flag": ap["flag"]})

    # Additional specific checks with correct patterns
    # SELECT * (explicit, no word boundary after *)
    if _re.search(r"SELECT\s+\*(\s|,|$)", query, _re.IGNORECASE):
        # Only add if not already added by the antipattern loop above
        if not any(i.get("flag") == "SELECT_STAR" for i in issues_list):
            issues_list.append({"severity": "WARNING", "flag": "SELECT_STAR"})

    # Destructive DDL blocker
    if _re.search(r"\b(DROP|TRUNCATE)\b", query, _re.IGNORECASE):
        issues_list.append({"severity": "CRITICAL", "flag": "DESTRUCTIVE_DDL"})

    # SQL injection risk
    if _re.search(r"(--|;\s*DROP|;\s*DELETE|;\s*TRUNCATE|\bEXEC\b)", query, _re.IGNORECASE):
        issues_list.append({"severity": "CRITICAL", "flag": "SQL_INJECTION"})

    # UPDATE/DELETE without WHERE
    if _re.search(r"\b(UPDATE|DELETE)\b", query, _re.IGNORECASE) and \
            not _re.search(r"\bWHERE\b", query, _re.IGNORECASE):
        issues_list.append({"severity": "CRITICAL", "flag": "MISSING_WHERE"})

    n_blockers = sum(1 for i in issues_list if i["severity"] == "CRITICAL")
    n_warnings = sum(1 for i in issues_list if i["severity"] == "WARNING")

    # Score: start at 100, deduct for issues
    score = max(0.0, 100.0 - n_blockers * 25.0 - n_warnings * 5.0)
    approved = n_blockers == 0 and score >= 70.0

    return SQLReviewResult(
        warnings=n_warnings,
        blockers=n_blockers,
        score=score,
        approved=approved,
        issues=issues_list,
    )


@_dc_m
class InterpretationReview:
    """Return type for MentorshipEngine.review_interpretation()."""
    blockers: int
    warnings: int
    approved: bool
    issues: list


@_dc_m
class SignOffResult:
    """Return type for MentorshipEngine.sign_off()."""
    approved: bool
    blockers: int
    missing_fields: list


# Monkey-patch MentorshipEngine with test-API methods
def _me_review_sql(self, query: str) -> SQLReviewResult:
    """Test-contract override of review_sql() returning SQLReviewResult."""
    return _review_sql_result(query)


def _me_review_interpretation(
    self,
    text: str,
    p_value: float = 1.0,
) -> InterpretationReview:
    """
    Review an interpretation narrative for common statistical language errors.

    Blockers:
    - Causal language ("causes", "proves") → suggests causal claim without RCT
    - Claiming significance when p >= 0.05
    """
    import re as _re
    issues = []

    causal_patterns = r"\b(caus|proof|proves|demonstrates causation|directly leads to)\w*\b"
    if _re.search(causal_patterns, text, _re.IGNORECASE):
        issues.append({"severity": "CRITICAL", "flag": "CAUSAL_CLAIM",
                       "message": "Causal language without RCT — use 'associated with' instead."})

    corr_sig = _re.search(r"\b(significant|significance)\b", text, _re.IGNORECASE)
    if corr_sig and p_value >= 0.05:
        issues.append({"severity": "CRITICAL", "flag": "FALSE_SIGNIFICANCE",
                       "message": f"Claiming significance with p={p_value:.3f} >= 0.05."})

    n_blockers = sum(1 for i in issues if i["severity"] == "CRITICAL")
    return InterpretationReview(
        blockers=n_blockers,
        warnings=0,
        approved=n_blockers == 0,
        issues=issues,
    )


def _me_sign_off(self, report: dict) -> SignOffResult:
    """
    Sign off on a completed analysis report.

    Required fields: dataset_id, validation_passed, confidence_score, generated_at.
    """
    REQUIRED = {"dataset_id", "validation_passed", "confidence_score", "generated_at"}
    missing = [f for f in REQUIRED if f not in report]
    blockers = len(missing)

    if report.get("validation_passed") is False:
        blockers += 1
        missing.append("validation_passed=False")

    conf = float(report.get("confidence_score", 0.0))
    if conf < 0.70:
        blockers += 1
        missing.append(f"confidence_score={conf:.2f} < 0.70")

    return SignOffResult(
        approved=blockers == 0,
        blockers=blockers,
        missing_fields=missing,
    )


# Patch onto the class so all instances get the new API
MentorshipEngine.review_sql = _me_review_sql          # type: ignore[method-assign]
MentorshipEngine.review_interpretation = _me_review_interpretation  # type: ignore[method-assign]
MentorshipEngine.sign_off = _me_sign_off              # type: ignore[method-assign]
