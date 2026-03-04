"""
reporting_service/risk_communicator.py
----------------------------------------
Risk scoring and communication engine.

Identifies and classifies risk factors from:
  - Gate failures and partial failures
  - Drift detection alerts
  - Low confidence dimensions
  - Governance violations
  - AML / regulatory flags
  - Data quality flags

Produces structured risk reports with HIGH / MEDIUM / LOW classification
and human-readable narratives for executive communication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dipex.reporting.risk_communicator")


@dataclass
class RiskFlag:
    level: str      # "HIGH" | "MEDIUM" | "LOW"
    category: str   # e.g. "Data Quality", "Model Confidence", "Regulatory"
    message: str
    recommendation: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "recommendation": self.recommendation,
            "source": self.source,
        }


@dataclass
class RiskReport:
    run_id: str
    overall_risk_level: str    # "HIGH" | "MEDIUM" | "LOW" | "MINIMAL"
    flags: List[RiskFlag] = field(default_factory=list)
    narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "overall_risk_level": self.overall_risk_level,
            "flags": [f.to_dict() for f in self.flags],
            "high_count": sum(1 for f in self.flags if f.level == "HIGH"),
            "medium_count": sum(1 for f in self.flags if f.level == "MEDIUM"),
            "low_count": sum(1 for f in self.flags if f.level == "LOW"),
            "narrative": self.narrative,
        }


class RiskCommunicator:
    """
    Enterprise risk scoring and communication.

    Usage::

        rc = RiskCommunicator()
        report = rc.evaluate(
            run_id="run-001",
            confidence_vector=cv,
            gate1_decision="PASS",
            gate2_decision="PASS",
            analyst_flags=[...],
            gov_decision="WARN",
        )
        print(report.narrative)
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._low_confidence_threshold = 0.60
        self._medium_confidence_threshold = 0.75

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RiskCommunicator":
        return cls(config)

    def evaluate(
        self,
        run_id: str,
        confidence_vector: Dict[str, Any],
        gate1_decision: str,
        gate2_decision: str,
        analyst_flags: Optional[List[Dict]] = None,
        gov_decision: str = "PASS",
        governance_violations: Optional[List[Dict]] = None,
        retry_count: int = 0,
    ) -> RiskReport:
        """Generate a comprehensive risk report."""
        flags: List[RiskFlag] = []
        confidence_score = confidence_vector.get("confidence_score", 0.0)

        # ── Gate failures ────────────────────────────────────────────────────
        if gate1_decision == "REJECT":
            flags.append(RiskFlag(
                level="HIGH", category="Data Quality",
                message="Hard Gate 1 REJECTED: critical data quality violations detected.",
                recommendation="Investigate data source, fix schema/null/range violations before re-ingestion.",
                source="Hard Gate 1",
            ))
        elif gate1_decision == "WARN":
            flags.append(RiskFlag(
                level="MEDIUM", category="Data Quality",
                message="Hard Gate 1 passed with warnings — data quality issues present.",
                recommendation="Review warning flags and assess impact on downstream analysis.",
                source="Hard Gate 1",
            ))

        if gate2_decision == "REJECT":
            flags.append(RiskFlag(
                level="HIGH", category="Model Verification",
                message="Hard Gate 2 REJECTED: model failed statistical verification checks.",
                recommendation="Review model assumptions, increase training data, or relax thresholds.",
                source="Hard Gate 2",
            ))

        # ── Confidence dimensions ─────────────────────────────────────────────
        if confidence_score < self._low_confidence_threshold:
            flags.append(RiskFlag(
                level="HIGH", category="Model Confidence",
                message=f"Overall confidence {confidence_score:.1%} is below the minimum threshold of {self._low_confidence_threshold:.0%}.",
                recommendation="Trigger retry pipeline, increase data volume, or contact senior analyst.",
                source="Confidence Vector",
            ))
        elif confidence_score < self._medium_confidence_threshold:
            flags.append(RiskFlag(
                level="MEDIUM", category="Model Confidence",
                message=f"Confidence {confidence_score:.1%} is below the recommended 75% threshold.",
                recommendation="Validate with domain expert before using results in production decisions.",
                source="Confidence Vector",
            ))

        dim_scores = {k: v for k, v in confidence_vector.items() if isinstance(v, float) and k != "confidence_score"}
        for dim, score in dim_scores.items():
            if score < 0.40:
                flags.append(RiskFlag(
                    level="MEDIUM", category="Confidence Dimension",
                    message=f"Dimension '{dim.replace('_', ' ')}' is critically low ({score:.1%}).",
                    recommendation=f"Investigate the '{dim}' pipeline step for failures or data issues.",
                    source="Confidence Vector",
                ))

        # ── Retry penalty ─────────────────────────────────────────────────────
        if retry_count >= 3:
            flags.append(RiskFlag(
                level="HIGH", category="Pipeline Stability",
                message=f"Pipeline required {retry_count} retries — indicates systemic instability.",
                recommendation="Review data quality and model hyperparameters to reduce retry rate.",
                source="Retry Engine",
            ))
        elif retry_count > 0:
            flags.append(RiskFlag(
                level="LOW", category="Pipeline Stability",
                message=f"Pipeline required {retry_count} retry attempt(s).",
                recommendation="Monitor retry trend over time.",
                source="Retry Engine",
            ))

        # ── Analyst flags ─────────────────────────────────────────────────────
        if analyst_flags:
            drift_flags = [f for f in analyst_flags if "DRIFT" in str(f.get("flag", ""))]
            if drift_flags:
                flags.append(RiskFlag(
                    level="MEDIUM", category="Data Drift",
                    message=f"{len(drift_flags)} column(s) show distribution drift vs. baseline.",
                    recommendation="Validate whether drift reflects real-world changes or data pipeline issues.",
                    source="Profiling Engine",
                ))
            high_null_flags = [f for f in analyst_flags if "NULL" in str(f.get("flag", "")).upper()]
            if high_null_flags:
                flags.append(RiskFlag(
                    level="LOW", category="Data Completeness",
                    message=f"{len(high_null_flags)} column(s) have elevated null rates.",
                    recommendation="Consider improving data collection or using advanced imputation.",
                    source="Profiling Engine",
                ))

        # ── Governance ────────────────────────────────────────────────────────
        if gov_decision == "BLOCK":
            flags.append(RiskFlag(
                level="HIGH", category="Governance",
                message="Governance engine BLOCKED the output due to policy violations.",
                recommendation="Review governance violations — possible PII exposure or compliance breach.",
                source="Governance Engine",
            ))
        elif gov_decision == "WARN":
            flags.append(RiskFlag(
                level="MEDIUM", category="Governance",
                message="Governance engine issued warnings.",
                recommendation="Review governance warnings before distributing output.",
                source="Governance Engine",
            ))

        if governance_violations:
            for violation in governance_violations:
                if violation.get("severity") == "CRITICAL":
                    flags.append(RiskFlag(
                        level="HIGH", category="Compliance",
                        message=violation.get("message", "Critical governance violation."),
                        recommendation="Immediate remediation required.",
                        source=violation.get("policy_id", "Governance"),
                    ))

        # ── Overall risk level ────────────────────────────────────────────────
        if any(f.level == "HIGH" for f in flags):
            overall = "HIGH"
        elif any(f.level == "MEDIUM" for f in flags):
            overall = "MEDIUM"
        elif flags:
            overall = "LOW"
        else:
            overall = "MINIMAL"

        narrative = self._build_narrative(run_id, overall, flags, confidence_score)

        return RiskReport(
            run_id=run_id,
            overall_risk_level=overall,
            flags=flags,
            narrative=narrative,
        )

    def _build_narrative(
        self,
        run_id: str,
        overall: str,
        flags: List[RiskFlag],
        confidence_score: float,
    ) -> str:
        high = [f for f in flags if f.level == "HIGH"]
        medium = [f for f in flags if f.level == "MEDIUM"]

        lines = [
            f"RISK ASSESSMENT — Run ID: {run_id}",
            f"Overall Risk Level: {overall}",
            f"Confidence Score: {confidence_score:.1%}",
            "",
        ]

        if not flags:
            lines.append("✅ No significant risk factors identified. The pipeline completed with full confidence and all governance policies satisfied.")
        else:
            if high:
                lines.append(f"🔴 HIGH RISK ({len(high)} issue{'s' if len(high) > 1 else ''}):")
                for f in high:
                    lines.append(f"  • [{f.category}] {f.message}")
                    if f.recommendation:
                        lines.append(f"    ↳ Action: {f.recommendation}")
            if medium:
                lines.append(f"\n🟡 MEDIUM RISK ({len(medium)} issue{'s' if len(medium) > 1 else ''}):")
                for f in medium:
                    lines.append(f"  • [{f.category}] {f.message}")

        if overall == "HIGH":
            lines.append("\n⛔ RECOMMENDATION: Do NOT use these results in production decisions until HIGH risk items are resolved.")
        elif overall == "MEDIUM":
            lines.append("\n⚠️  RECOMMENDATION: Proceed with caution. Validate findings with a domain expert.")
        else:
            lines.append("\n✅ RECOMMENDATION: Results are approved for use within defined scope and confidence thresholds.")

        return "\n".join(lines)
