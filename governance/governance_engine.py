"""
governance/governance_engine.py
---------------------------------
Enterprise governance engine for DIPEX.

Enforces:
  - Confidence floor policies by domain
  - PII column enforcement (no PII in approved outputs)
  - Mandatory audit logging check
  - Data quality minimum thresholds
  - Regulatory domain-specific rules
  - Custom YAML-defined policies

Returns GovernanceDecision with pass/fail and violation list.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dipex.governance.governance_engine")

# [ML] Auto-PII detection
try:
    from governance.pii_detector import PiiDetector as _PiiDetector
    _pii_detector = _PiiDetector()
except Exception:  # noqa: BLE001
    _pii_detector = None


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyViolation:
    policy_id: str
    severity: str        # "CRITICAL" | "ERROR" | "WARNING"
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class GovernanceDecision:
    run_id: str
    decision: str                         # "PASS" | "WARN" | "BLOCK"
    violations: List[PolicyViolation] = field(default_factory=list)
    policies_checked: int = 0
    evaluated_at: str = ""

    def __post_init__(self) -> None:
        if not self.evaluated_at:
            self.evaluated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "decision": self.decision,
            "policies_checked": self.policies_checked,
            "violations": [v.to_dict() for v in self.violations],
            "critical_count": sum(1 for v in self.violations if v.severity == "CRITICAL"),
            "error_count": sum(1 for v in self.violations if v.severity == "ERROR"),
            "warning_count": sum(1 for v in self.violations if v.severity == "WARNING"),
            "evaluated_at": self.evaluated_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GovernanceEngine
# ─────────────────────────────────────────────────────────────────────────────

class GovernanceEngine:
    """
    Policy enforcement engine.

    Built-in policies:
      G001 — Confidence floor by domain
      G002 — No PII columns in approved outputs
      G003 — Audit log must exist and be non-empty
      G004 — Data quality gate must PASS
      G005 — Banking domain AML check
      G006 — Healthcare PHI non-disclosure
    """

    BUILTIN_POLICIES = [
        "G001_confidence_floor",
        "G002_pii_enforcement",
        "G003_audit_mandatory",
        "G004_data_quality_gate",
        "G005_banking_aml",
        "G006_healthcare_phi",
    ]

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("governance", {})
        self.domain: str = (config or {}).get("validation", {}).get("regulatory", {}).get("domain", "generic")
        self.pii_enforcement: bool = bool(cfg.get("pii_enforcement", True))
        self.min_confidence: float = float(cfg.get("min_confidence_banking", 0.75)) \
            if self.domain == "banking" else float(cfg.get("min_confidence", 0.60))
        self.audit_log_path: str = (config or {}).get("storage", {}).get("audit_log", "audit/audit.jsonl")
        self.policies_file: Optional[str] = cfg.get("policies_file")
        self._custom_policies: List[Dict] = self._load_custom_policies()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GovernanceEngine":
        return cls(config)

    def _load_custom_policies(self) -> List[Dict]:
        if not self.policies_file or not os.path.exists(self.policies_file):
            return []
        try:
            import yaml
            with open(self.policies_file, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            return raw.get("policies", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load custom policies: %s", exc)
            return []

    def evaluate(
        self,
        run_id: str,
        confidence_score: float,
        gate1_decision: str,
        gate2_decision: str,
        approved_output: Optional[Dict[str, Any]] = None,
        df_columns: Optional[List[str]] = None,
        pii_columns: Optional[List[str]] = None,
        # [ML] Optional: pass a sample DataFrame for auto PII detection
        df_sample: Optional[Any] = None,   # pd.DataFrame
    ) -> GovernanceDecision:
        """
        Run all governance policies and return a GovernanceDecision.

        If pii_columns is None and df_sample is provided, the ML PII detector
        will automatically scan for PII before applying G002.
        """
        violations: List[PolicyViolation] = []
        n_policies = 0

        # [ML] Auto-detect PII if not supplied and df_sample is available
        if pii_columns is None and df_sample is not None and _pii_detector is not None:
            try:
                import pandas as _pd
                if isinstance(df_sample, _pd.DataFrame):
                    pii_result   = _pii_detector.scan_dataframe(df_sample)
                    pii_columns  = pii_result.get("pii_columns", [])
                    df_columns   = df_columns or list(df_sample.columns)
                    if pii_columns:
                        logger.info(
                            "[ML] GovernanceEngine: auto-detected PII columns via NER+regex: %s",
                            pii_columns,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[ML] PII auto-detection failed: %s", exc)

        # G001 — Confidence floor
        n_policies += 1
        if confidence_score < self.min_confidence:
            violations.append(PolicyViolation(
                policy_id="G001", severity="ERROR",
                message=f"Confidence {confidence_score:.4f} below domain floor {self.min_confidence:.2f}",
                context={"confidence_score": confidence_score, "min_confidence": self.min_confidence},
            ))

        # G002 — PII enforcement
        n_policies += 1
        if self.pii_enforcement and pii_columns and df_columns:
            pii_in_output = [c for c in (pii_columns or []) if c in (df_columns or [])]
            if pii_in_output:
                violations.append(PolicyViolation(
                    policy_id="G002", severity="CRITICAL",
                    message=f"PII columns detected in approved output: {pii_in_output}",
                    context={"pii_columns": pii_in_output},
                ))

        # G003 — Audit log mandatory
        n_policies += 1
        if not os.path.exists(self.audit_log_path) or os.path.getsize(self.audit_log_path) == 0:
            violations.append(PolicyViolation(
                policy_id="G003", severity="ERROR",
                message="Audit log is missing or empty",
                context={"audit_log_path": self.audit_log_path},
            ))

        # G004 — Both gates must PASS
        n_policies += 1
        if gate1_decision != "PASS" or gate2_decision != "PASS":
            violations.append(PolicyViolation(
                policy_id="G004", severity="ERROR" if gate1_decision == "REJECT" else "WARNING",
                message=f"Data quality gates did not both PASS (gate1={gate1_decision}, gate2={gate2_decision})",
                context={"gate1": gate1_decision, "gate2": gate2_decision},
            ))

        # G005 — Banking AML (if domain == banking)
        if self.domain == "banking":
            n_policies += 1
            # Check for AML flags in approved_output
            if approved_output:
                narrative = approved_output.get("narrative", "")
                if "AML" in str(narrative).upper() or "SUSPICIOUS" in str(narrative).upper():
                    violations.append(PolicyViolation(
                        policy_id="G005", severity="CRITICAL",
                        message="AML/Suspicious activity flagged in approved output — regulatory disclosure required",
                        context={"domain": "banking"},
                    ))

        # G006 — Healthcare PHI non-disclosure
        if self.domain == "healthcare":
            n_policies += 1
            if pii_columns:
                violations.append(PolicyViolation(
                    policy_id="G006", severity="CRITICAL",
                    message=f"HIPAA: PHI columns must not appear in output: {pii_columns}",
                    context={"domain": "healthcare", "phi_columns": pii_columns},
                ))

        # Custom policies (from YAML)
        for policy in self._custom_policies:
            n_policies += 1
            self._evaluate_custom_policy(policy, confidence_score, gate1_decision, gate2_decision, violations)

        # Decision
        severities = {v.severity for v in violations}
        if "CRITICAL" in severities:
            decision = "BLOCK"
        elif "ERROR" in severities:
            decision = "BLOCK"
        elif "WARNING" in severities:
            decision = "WARN"
        else:
            decision = "PASS"

        gov_decision = GovernanceDecision(
            run_id=run_id,
            decision=decision,
            violations=violations,
            policies_checked=n_policies,
        )
        logger.info(
            "Governance: %s — %d policies checked, %d violations (CRITICAL=%d, ERROR=%d, WARN=%d)",
            decision, n_policies, len(violations),
            sum(1 for v in violations if v.severity == "CRITICAL"),
            sum(1 for v in violations if v.severity == "ERROR"),
            sum(1 for v in violations if v.severity == "WARNING"),
        )
        return gov_decision

    def _evaluate_custom_policy(
        self,
        policy: Dict,
        confidence_score: float,
        gate1: str,
        gate2: str,
        violations: List[PolicyViolation],
    ) -> None:
        """Evaluate a simple threshold-based custom policy from YAML."""
        try:
            policy_id = policy.get("id", "CUSTOM")
            check_type = policy.get("check", "min_confidence")
            severity = policy.get("severity", "WARNING")
            if check_type == "min_confidence":
                threshold = float(policy.get("value", 0.6))
                if confidence_score < threshold:
                    violations.append(PolicyViolation(
                        policy_id=policy_id, severity=severity,
                        message=policy.get("message", f"Confidence {confidence_score:.4f} below {threshold}"),
                        context={"confidence_score": confidence_score, "threshold": threshold},
                    ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Custom policy evaluation failed: %s", exc)
