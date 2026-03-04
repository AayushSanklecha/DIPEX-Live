"""
output/approved_output.py
-------------------------
Step 8 — Approved Analytics Output.

Only results that have passed Hard Gate 1 and Hard Gate 2 are stored.
Every stored output carries a complete provenance envelope:

  - Confidence Score
  - QA status badge
  - Retry count
  - Validation checklist
  - Schema version
  - Drift report
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Envelope version for future schema evolution
ENVELOPE_VERSION = "1.0"

# QA status badges
QA_APPROVED = "APPROVED"
QA_REJECTED = "REJECTED"
QA_MARGINAL = "MARGINAL"


def _normalise_drift_report(drift: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures drift_report has a consistent structure for industry-grade storage."""
    if not drift:
        return {
            "baseline_available": False,
            "columns": {},
            "drifted_columns": [],
            "analyst_flags": [],
        }
    return {
        "baseline_available": bool(drift.get("columns")),
        "columns": drift.get("columns", {}),
        "drifted_columns": drift.get("drifted_columns", []),
        "analyst_flags": drift.get("analyst_flags", []),
    }


@dataclass
class ApprovedAnalyticsOutput:
    """
    Immutable envelope for approved analytics results.
    Only persisted when both Hard Gate 1 and Hard Gate 2 PASS.
    """

    run_id: str
    confidence_score: float
    qa_status_badge: str
    retry_count: int
    validation_checklist: Dict[str, Any]
    schema_version: str
    drift_report: Dict[str, Any]
    # Core results (sanitised — no in-memory estimators)
    narrative: str = ""
    proposal_summary: Dict[str, Any] = field(default_factory=dict)
    profile_summary: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    envelope_version: str = ENVELOPE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_version": self.envelope_version,
            "run_id": self.run_id,
            "confidence_score": round(self.confidence_score, 6),
            "qa_status_badge": self.qa_status_badge,
            "retry_count": self.retry_count,
            "validation_checklist": self.validation_checklist,
            "schema_version": self.schema_version,
            "drift_report": self.drift_report,
            "narrative": self.narrative,
            "proposal_summary": self.proposal_summary,
            "profile_summary": self.profile_summary,
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
        }


class ApprovedOutputStore:
    """
    Persists approved analytics outputs only when both gates pass.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        store_cfg = self.config.get("storage", {})
        out_dir = store_cfg.get("approved_output_dir", "data/approved_outputs")
        self._output_dir = Path(out_dir)
        # Optional: only store when confidence >= threshold (avoids storing retry-triggered runs)
        self._min_confidence = store_cfg.get("min_confidence_for_storage")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ApprovedOutputStore":
        return cls(config)

    def build_output(
        self,
        run_id: str,
        gate1_result: Dict[str, Any],
        gate2_result: Dict[str, Any],
        confidence_vector: Dict[str, Any],
        retry_count: int,
        schema_version: str,
        drift_report: Dict[str, Any],
        narrative: str = "",
        proposal_summary: Optional[Dict[str, Any]] = None,
        profile_summary: Optional[Dict[str, Any]] = None,
        fingerprint: Optional[str] = None,
    ) -> ApprovedAnalyticsOutput:
        """
        Builds the approved output envelope. Caller must ensure gates passed.
        """
        gate1_decision = gate1_result.get("decision", "REJECT")
        gate2_decision = gate2_result.get("decision", "REJECT")
        conf_score = float(confidence_vector.get("confidence_score", 0.0))
        all_passed = bool(confidence_vector.get("all_gates_passed", False))

        if gate1_decision == "REJECT" or gate2_decision == "REJECT":
            qa_badge = QA_REJECTED
        elif all_passed and conf_score >= 0.8:
            qa_badge = QA_APPROVED
        elif all_passed and conf_score >= 0.6:
            qa_badge = QA_MARGINAL
        else:
            qa_badge = QA_REJECTED

        validation_checklist = {
            "hard_gate_1": {
                "decision": gate1_decision,
                "violations": gate1_result.get("total_violations", 0),
                "warnings": gate1_result.get("total_warnings", 0),
                "passed": gate1_decision == "PASS",
            },
            "hard_gate_2": {
                "decision": gate2_result.get("decision", "REJECT"),
                "passed_checks": [
                    c.get("name", c.get("metric", ""))
                    for c in gate2_result.get("passed_checks", [])
                ],
                "failed_checks": [
                    c.get("name", c.get("metric", ""))
                    for c in gate2_result.get("failed_checks", [])
                ],
                "passed": gate2_decision == "PASS",
            },
            "confidence_vector": {
                "data_quality_score": confidence_vector.get("data_quality_score"),
                "statistical_score": confidence_vector.get("statistical_score"),
                "stability_score": confidence_vector.get("stability_score"),
                "drift_robustness_score": confidence_vector.get("drift_robustness_score"),
                "compliance_score": confidence_vector.get("compliance_score"),
                "retry_penalty_score": confidence_vector.get("retry_penalty_score"),
            },
        }

        return ApprovedAnalyticsOutput(
            run_id=run_id,
            confidence_score=conf_score,
            qa_status_badge=qa_badge,
            retry_count=retry_count,
            validation_checklist=validation_checklist,
            schema_version=schema_version,
            drift_report=_normalise_drift_report(drift_report or {}),
            narrative=narrative,
            proposal_summary=proposal_summary or {},
            profile_summary=profile_summary or {},
            fingerprint=fingerprint,
        )

    def store(
        self,
        output: ApprovedAnalyticsOutput,
        only_if_approved: bool = True,
        min_confidence: Optional[float] = None,
    ) -> bool:
        """
        Persists the output to disk. If only_if_approved=True (default),
        only stores when qa_status_badge is APPROVED or MARGINAL.
        If min_confidence is set, also requires confidence_score >= min_confidence.
        Returns True if stored, False otherwise.
        """
        if only_if_approved and output.qa_status_badge == QA_REJECTED:
            logger.info(
                "ApprovedOutputStore: skipping store for run_id=%s (qa_status=%s).",
                output.run_id,
                output.qa_status_badge,
            )
            return False

        if min_confidence is not None and output.confidence_score < min_confidence:
            logger.info(
                "ApprovedOutputStore: skipping store for run_id=%s "
                "(confidence %.4f < threshold %.4f).",
                output.run_id,
                output.confidence_score,
                min_confidence,
            )
            return False

        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"{output.run_id}_approved.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(output.to_dict(), f, indent=2, default=str, ensure_ascii=False)
            logger.info(
                "Approved analytics output stored: %s (qa=%s confidence=%.2f%%)",
                path,
                output.qa_status_badge,
                output.confidence_score * 100,
            )
            return True
        except OSError as exc:
            logger.error("Failed to store approved output to %s: %s", path, exc)
            return False
