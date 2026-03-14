"""
validation/compliance_decision.py
------------------------------------
Compliance-Aware Decision Layer for DIPEX.

This module translates raw ``RegulatoryViolation`` lists from the
``RegulatoryEngine`` into a structured ``ComplianceDecision`` object that:

  1. Calculates a ``compliance_penalty`` ∈ [-1.0, 0.0]
     — fed directly into ``ConfidenceVector.aggregate()``
  2. Produces a 3-tier decision: ``"allowed"`` | ``"conditional"`` | ``"blocked"``
  3. Identifies ``violating_columns`` for feature masking in ``PipelineBridge``
  4. Generates per-violation ``remediation_steps`` (LLM or static)
  5. Includes the conflict report from ``RuleConflictResolver``
  6. Feeds RL reward signal back into ``RLThresholdTuner`` (if enabled)
  7. Cross-references SHAP risk ranks with regulatory violations

Penalty formula
---------------
  penalty = -(n_critical × w_critical + n_error × w_error + n_warning × w_warning)
  clamped to [-1.0, 0.0]

  Default weights (from config):
      critical: 0.20, error: 0.10, warning: 0.02

Decision tiers
--------------
  blocked     — any CRITICAL violation (or critical_blocks_pipeline is True)
  conditional — only ERROR/WARNING violations present (below allowed_warning_count)
  allowed     — no violations
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from .regulatory.base_rule import RegulatoryViolation

logger = logging.getLogger("dipex.compliance.advisor")


# ─────────────────────────────────────────────────────────────────────────────
# ComplianceDecision dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplianceDecision:
    """
    The structured output of ComplianceAdvisor.evaluate().
    Consumed by PipelineBridge, ConfidenceVector, and ExecutiveReportGenerator.
    """
    decision: str                          # "allowed" | "conditional" | "blocked"
    compliance_penalty: float              # ∈ [-1.0, 0.0], subtracted from confidence
    violation_summary: List[Dict[str, Any]] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)   # LLM or static
    conflict_report: List[Dict[str, Any]] = field(default_factory=list)
    violating_columns: Set[str] = field(default_factory=set)     # for feature masking
    shap_ranked_violations: List[Dict[str, Any]] = field(default_factory=list)
    n_critical: int = 0
    n_error: int = 0
    n_warning: int = 0
    evaluated_at: str = ""
    domain: str = "generic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "compliance_penalty": self.compliance_penalty,
            "n_critical": self.n_critical,
            "n_error": self.n_error,
            "n_warning": self.n_warning,
            "violation_summary": self.violation_summary,
            "remediation_steps": self.remediation_steps,
            "conflict_report": self.conflict_report,
            "violating_columns": sorted(self.violating_columns),
            "shap_ranked_violations": self.shap_ranked_violations,
            "evaluated_at": self.evaluated_at,
            "domain": self.domain,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ComplianceAdvisor
# ─────────────────────────────────────────────────────────────────────────────

class ComplianceAdvisor:
    """
    Translates raw RegulatoryViolation lists into structured ComplianceDecisions.

    Usage::

        advisor = ComplianceAdvisor.from_config(config)
        decision = advisor.evaluate(violations, conflict_report, df, run_id="run-123")

        # Feed penalty into ConfidenceVector
        cv = ConfidenceVector.aggregate(..., compliance_penalty=decision.compliance_penalty)

        # Mask bad columns before AutoML
        compliant_features = [c for c in df.columns if c not in decision.violating_columns]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("compliance", {})
        weights = cfg.get("penalty_weights", {})
        self._w_critical = float(weights.get("critical", 0.20))
        self._w_error    = float(weights.get("error", 0.10))
        self._w_warning  = float(weights.get("warning", 0.02))
        self._critical_blocks = bool(cfg.get("critical_blocks_pipeline", True))
        self._allowed_warnings = int(cfg.get("allowed_warning_count", 10))
        self._audit_violations = bool(cfg.get("audit_violations", True))
        self._llm_remediation  = bool(cfg.get("llm_remediation", False))
        self._rl_feedback      = bool(cfg.get("rl_feedback", False))
        self._domain = (
            config.get("validation", {}).get("regulatory", {}).get("domain", "generic")
            if config else "generic"
        )
        self._config = config or {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ComplianceAdvisor":
        return cls(config)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        violations: List[RegulatoryViolation],
        conflict_report: Optional[List[Dict[str, Any]]] = None,
        df: Optional[pd.DataFrame] = None,
        run_id: str = "N/A",
    ) -> ComplianceDecision:
        """
        Evaluate a violation list and produce a ComplianceDecision.

        Parameters
        ----------
        violations     : Output of RegulatoryEngine.evaluate()
        conflict_report: Output of engine.get_last_conflict_report()
        df             : Original DataFrame (used for SHAP cross-reference if available)
        run_id         : Pipeline run identifier for audit logging

        Returns
        -------
        ComplianceDecision with penalty, tier decision, and all metadata.
        """
        conflict_report = conflict_report or []

        n_critical = sum(1 for v in violations if v.severity == "CRITICAL")
        n_error    = sum(1 for v in violations if v.severity == "ERROR")
        n_warning  = sum(1 for v in violations if v.severity == "WARNING")

        # ── 1. Penalty calculation ────────────────────────────────────────────
        raw_penalty = -(
            n_critical * self._w_critical
            + n_error  * self._w_error
            + n_warning * self._w_warning
        )
        compliance_penalty = max(-1.0, min(0.0, raw_penalty))

        # ── 2. Decision tier ─────────────────────────────────────────────────
        if n_critical > 0 and self._critical_blocks:
            decision = "blocked"
        elif n_error > 0 or n_warning > self._allowed_warnings:
            decision = "conditional"
        else:
            decision = "allowed"

        # ── 3. Violating columns (CRITICAL + ERROR only) ──────────────────────
        violating_columns: Set[str] = {
            v.column for v in violations
            if v.severity in ("CRITICAL", "ERROR") and v.column not in ("N/A", "")
        }

        # ── 4. Violation summary ─────────────────────────────────────────────
        violation_summary = [
            {
                "rule_name": v.rule_name,
                "domain": v.domain,
                "severity": v.severity,
                "column": v.column,
                "offending_count": v.offending_count,
                "message": v.message,
                "remediation": v.remediation,
                "shap_impact": None,   # filled in step 5
                "risk_rank": None,
            }
            for v in violations
        ]

        # ── 5. SHAP × Compliance cross-reference ─────────────────────────────
        shap_ranked = []
        if df is not None and violations:
            try:
                from validation.shap_explainer import explain_compliance_violations
                shap_ranked = explain_compliance_violations(df, violations, run_id=run_id)
                # Merge SHAP ranks back into violation_summary
                shap_by_col = {item["column"]: item for item in shap_ranked}
                for vs in violation_summary:
                    col_info = shap_by_col.get(vs["column"])
                    if col_info:
                        vs["shap_impact"] = col_info.get("shap_impact")
                        vs["risk_rank"]   = col_info.get("risk_rank")
            except Exception as exc:  # noqa: BLE001
                logger.debug("SHAP×Compliance cross-reference skipped: %s", exc)

        # ── 6. Remediation steps ─────────────────────────────────────────────
        remediation_steps = self._generate_remediation(violations, run_id)

        # ── 7. RL feedback ───────────────────────────────────────────────────
        if self._rl_feedback and violations:
            self._send_rl_feedback(violations)

        # ── 8. Audit log ─────────────────────────────────────────────────────
        evaluated_at = datetime.now(timezone.utc).isoformat()
        if self._audit_violations:
            self._audit(run_id, decision, compliance_penalty, violation_summary, evaluated_at)

        decision_obj = ComplianceDecision(
            decision=decision,
            compliance_penalty=compliance_penalty,
            violation_summary=violation_summary,
            remediation_steps=remediation_steps,
            conflict_report=conflict_report,
            violating_columns=violating_columns,
            shap_ranked_violations=shap_ranked,
            n_critical=n_critical,
            n_error=n_error,
            n_warning=n_warning,
            evaluated_at=evaluated_at,
            domain=self._domain,
        )

        logger.info(
            "[ComplianceAdvisor] run=%s domain=%s decision=%s penalty=%.3f "
            "violations(C=%d E=%d W=%d) masked_cols=%s",
            run_id, self._domain, decision, compliance_penalty,
            n_critical, n_error, n_warning, sorted(violating_columns),
        )
        return decision_obj

    # ------------------------------------------------------------------
    # Remediation
    # ------------------------------------------------------------------

    def _generate_remediation(
        self,
        violations: List[RegulatoryViolation],
        run_id: str,
    ) -> List[str]:
        """
        Generates remediation steps. Uses LLM if llm_remediation is True
        and LLM provider is available; otherwise uses static violation.remediation.
        """
        if not violations:
            return []

        if self._llm_remediation:
            try:
                from reporting_service.llm_provider import get_llm_provider
                provider = get_llm_provider(self._config)
                violation_dicts = [
                    {"rule": v.rule_name, "severity": v.severity, "message": v.message}
                    for v in violations
                ]
                narrative = provider.generate_compliance_remediation(
                    violations=violation_dicts,
                    domain=self._domain,
                    run_id=run_id,
                )
                if narrative and narrative.strip():
                    return [narrative]
            except Exception as exc:  # noqa: BLE001
                logger.debug("LLM remediation failed, using static: %s", exc)

        # Static fallback: deduplicate and prioritize CRITICAL
        seen: set = set()
        steps: List[str] = []
        sorted_v = sorted(violations, key=lambda v: {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}.get(v.severity, 3))
        for v in sorted_v:
            key = f"{v.rule_name}:{v.column}"
            if key not in seen:
                seen.add(key)
                steps.append(f"[{v.severity}] {v.rule_name}: {v.remediation}")
        return steps[:10]  # cap at 10 to prevent report overflow

    # ------------------------------------------------------------------
    # RL Feedback
    # ------------------------------------------------------------------

    def _send_rl_feedback(self, violations: List[RegulatoryViolation]) -> None:
        """
        Records compliance outcomes back into the RLThresholdTuner Q-table.
        One call per unique violating column (worst severity for that column).
        """
        try:
            from validation.rl_threshold_tuner import RLThresholdTuner
            tuner = RLThresholdTuner.get_instance()

            # Aggregate worst severity per column
            worst: Dict[str, str] = {}
            for v in violations:
                col = v.column
                if col in ("N/A", ""):
                    continue
                current_rank = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}.get(v.severity, 3)
                existing_rank = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}.get(
                    worst.get(col, "WARNING"), 3
                )
                if current_rank < existing_rank:
                    worst[col] = v.severity

            for col, severity in worst.items():
                tuner.record_compliance_outcome(
                    dataset_id="compliance_feedback",
                    column=col,
                    threshold=0.0,  # The tuner uses severity for reward
                    violation_severity=severity,
                )
                logger.debug("[RL] Compliance feedback recorded: col=%s severity=%s", col, severity)

        except Exception as exc:  # noqa: BLE001
            logger.debug("RL compliance feedback skipped: %s", exc)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _audit(
        self,
        run_id: str,
        decision: str,
        penalty: float,
        violation_summary: List[Dict[str, Any]],
        evaluated_at: str,
    ) -> None:
        """Writes violation audit record to audit/compliance.jsonl."""
        try:
            os.makedirs("audit", exist_ok=True)
            entry = {
                "event": "COMPLIANCE_EVALUATION",
                "run_id": run_id,
                "domain": self._domain,
                "decision": decision,
                "compliance_penalty": penalty,
                "violation_count": len(violation_summary),
                "violation_summary": violation_summary,
                "evaluated_at": evaluated_at,
            }
            with open("audit/compliance.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Compliance audit write failed: %s", exc)
