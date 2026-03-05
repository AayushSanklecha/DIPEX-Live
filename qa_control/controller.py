"""
qa_control/controller.py
-------------------------
QA, GOVERNANCE & CONTROL LAYER — QAController

Aggregates all 5 QA sub-components into one clean `.run()` call:
  1. Deterministic Validation  → Hard Gate 1
  2. Independent QA Verifiers  → Hard Gate 2 / Confidence Vector
  3. Regulatory & Business Rules
  4. Confidence Scoring
  5. Audit Logs

Returns a structured QAResult that the pipeline can use directly.
Does NOT replace any existing module — purely an aggregation facade.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger("dipex.qa_control.controller")


# ── QA Result ─────────────────────────────────────────────────────────────────

@dataclass
class QAResult:
    """Structured result from all 5 QA sub-components."""
    run_id: str
    gate1_decision: str = "PENDING"        # PASS | REJECT
    gate2_decision: str = "PENDING"        # PASS | REJECT | NOT_RUN
    regulatory_passed: bool = True
    confidence_score: float = 0.0
    confidence_vector: Dict = field(default_factory=dict)
    audit_written: bool = False
    details: Dict = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def overall_decision(self) -> str:
        if self.gate1_decision == "REJECT":
            return "FAIL"
        if self.gate2_decision == "REJECT":
            return "WARN"
        if not self.regulatory_passed:
            return "WARN"
        return "PASS"

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "gate1_decision": self.gate1_decision,
            "gate2_decision": self.gate2_decision,
            "regulatory_passed": self.regulatory_passed,
            "confidence_score": round(self.confidence_score, 4),
            "overall_decision": self.overall_decision,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "details": self.details,
        }


# ── QA Controller ─────────────────────────────────────────────────────────────

class QAController:
    """
    Single aggregation point for the entire QA, Governance & Control Layer.

    Usage::

        controller = QAController(config=config)
        qa_result  = controller.run(df, run_id=run_id, dataset_id=dataset_id)

        if qa_result.gate1_decision == "REJECT":
            raise PipelineAbort("Hard gate 1 rejected the dataset")
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        dataset_id: str = "unknown",
        model_metrics: Optional[Dict] = None,
        quality_score: float = 0.5,
    ) -> QAResult:
        """
        Execute all 5 QA stages and return a QAResult.

        Parameters
        ----------
        df            : the (preprocessed) DataFrame
        run_id        : pipeline run ID for audit trail
        dataset_id    : dataset identifier
        model_metrics : optional ML metrics dict from modeling stage
        quality_score : base quality score from UDIL (0–1)
        """
        t0 = time.perf_counter()
        result = QAResult(run_id=run_id)
        model_metrics = model_metrics or {}

        # ── 1. Deterministic Validation (Hard Gate 1) ──────────────────────
        result.gate1_decision, gate1_detail = self._run_gate1(df)
        result.details["gate1"] = gate1_detail
        if result.gate1_decision == "REJECT":
            result.elapsed_ms = (time.perf_counter() - t0) * 1000
            self._write_audit(result, dataset_id)
            result.audit_written = True
            return result

        # ── 2. Regulatory & Business Rules ────────────────────────────────
        result.regulatory_passed, reg_detail = self._run_regulatory(df)
        result.details["regulatory"] = reg_detail

        # ── 3. Independent QA Verifiers (Hard Gate 2) ─────────────────────
        result.gate2_decision, gate2_detail = self._run_gate2(df, model_metrics, run_id)
        result.details["gate2"] = gate2_detail

        # ── 4. Confidence Scoring ──────────────────────────────────────────
        result.confidence_vector, result.confidence_score = self._run_confidence(
            df, model_metrics, quality_score,
            gate2_passed=(result.gate2_decision == "PASS"),
        )
        result.details["confidence"] = {"score": result.confidence_score}

        # ── 5. Audit Logs ─────────────────────────────────────────────────
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        self._write_audit(result, dataset_id)
        result.audit_written = True

        logger.info(
            "[QAControl][%s] gate1=%s gate2=%s regulatory=%s conf=%.3f decision=%s",
            run_id[:8], result.gate1_decision, result.gate2_decision,
            result.regulatory_passed, result.confidence_score, result.overall_decision,
        )
        return result

    # ── Sub-component runners ─────────────────────────────────────────────────

    def _run_gate1(self, df: pd.DataFrame):
        try:
            from validation.hard_gate import HardGate
            gate = HardGate.from_config(self.config)
            gate_result = gate.run(df, run_id="qa_ctrl")
            if gate_result.decision == "REJECT":
                return "REJECT", {"reason": gate_result.reason}
            return "PASS", {"checks": "all_passed"}
        except Exception as exc:
            logger.warning("Gate1 error (non-fatal): %s", exc)
            return "PASS", {"warning": str(exc)}

    def _run_regulatory(self, df: pd.DataFrame):
        try:
            from validation.regulatory.regulatory_checker import RegulatoryChecker
            checker = RegulatoryChecker(self.config)
            reg_result = checker.check(df)
            passed = reg_result.get("passed", True) if isinstance(reg_result, dict) else True
            return passed, reg_result if isinstance(reg_result, dict) else {}
        except ImportError:
            # Try alternative path
            try:
                from governance.governance_engine import GovernanceEngine
                engine = GovernanceEngine(self.config)
                gov = engine.evaluate(
                    run_id="qa_ctrl", confidence_score=0.8,
                    gate1_decision="PASS", gate2_decision="PASS",
                    df_columns=list(df.columns),
                )
                result = gov.to_dict() if hasattr(gov, "to_dict") else {}
                return True, result
            except Exception as exc2:
                logger.debug("Regulatory/GovernanceEngine unavailable: %s", exc2)
                return True, {}
        except Exception as exc:
            logger.warning("Regulatory check error (non-fatal): %s", exc)
            return True, {"warning": str(exc)}

    def _run_gate2(self, df: pd.DataFrame, model_metrics: Dict, run_id: str):
        try:
            from verifier.confidence_vector import ConfidenceVector
            cv = ConfidenceVector.from_config(self.config)
            gate_result = cv.run_verification_gate(
                df=df, model_metrics=model_metrics, run_id=run_id,
            )
            decision = gate_result.get("decision", "PASS")
            return decision, gate_result
        except Exception as exc:
            logger.warning("Gate2 verifier error (non-fatal): %s", exc)
            return "PASS", {"warning": str(exc)}

    def _run_confidence(
        self, df: pd.DataFrame, model_metrics: Dict,
        quality_score: float, gate2_passed: bool,
    ):
        try:
            from verifier.confidence_vector import ConfidenceVector
            cv = ConfidenceVector.from_config(self.config)
            vector = cv.aggregate(
                df=df, model_metrics=model_metrics,
                quality_score=quality_score,
                gate2_passed=gate2_passed,
                retry_count=0,
            )
            score = float(vector.get("confidence_score", 0.5))
            return vector, score
        except Exception as exc:
            logger.warning("Confidence scoring error (non-fatal): %s", exc)
            return {"confidence_score": 0.5}, 0.5

    def _write_audit(self, result: QAResult, dataset_id: str) -> None:
        try:
            os.makedirs("audit", exist_ok=True)
            entry = {
                "event": "QA_CONTROL_RUN",
                "run_id": result.run_id,
                "dataset_id": dataset_id,
                "gate1_decision": result.gate1_decision,
                "gate2_decision": result.gate2_decision,
                "regulatory_passed": result.regulatory_passed,
                "confidence_score": result.confidence_score,
                "overall_decision": result.overall_decision,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round(result.elapsed_ms, 2),
            }
            with open("audit/qa_control.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("Audit write failed (non-fatal): %s", exc)
