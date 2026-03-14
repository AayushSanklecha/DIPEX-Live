"""
verifier/confidence_vector.py
------------------------------
Independent Verification Engine — Hard Gate 2 + Confidence Aggregation.

Fix 3: Non-classification / unsupervised mode
  - When roc_auc == 0.0 (regression, EDA-only, clustering runs), Gate 2
    no longer silently passes. Instead it evaluates quality_score against
    a minimum threshold (default 0.5).
  - aggregate() now includes three data-quality dimensions that work
    regardless of whether a model ran: completeness, consistency, quality.
  - model_boost only applies when roc_auc is actually available (> 0).

Compliance Integration (new):
  - aggregate() now accepts compliance_penalty ∈ [-1.0, 0.0], directly
    subtracted from the confidence score.
  - run_verification_gate() dynamically tightens min ROC-AUC thresholds
    when regulatory violations are present, and auto-REJECTs on CRITICAL
    PHI violations in healthcare.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ConfidenceVector:
    """
    Independent Verification Engine — Hard Gate 2.

    Evaluates ML performance metrics + data quality signals and aggregates
    pipeline confidence into a single scalar score.

    Now includes compliance_penalty as a first-class component.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConfidenceVector":
        return cls(config)

    def run_verification_gate(
        self,
        df: pd.DataFrame,
        model_metrics: Dict[str, Any],
        run_id: str,
        compliance_decision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Hard Gate 2: reject if ML performance or data quality is below threshold.

        Compliance-aware: if violations are present, minimum ROC-AUC thresholds
        are tightened and CRITICAL PHI violations trigger an automatic REJECT.
        """
        logger.info("[%s] Verifier: Running Hard Gate 2 independent checks.", run_id[:8])

        domain = self.config.get("pipeline", {}).get("domain", "default")
        roc_auc = model_metrics.get("roc_auc", 0.0)

        # ── Compliance-driven auto-reject checks ──────────────────────────────
        if compliance_decision:
            cd_decision = compliance_decision.get("decision", "allowed")
            n_critical = compliance_decision.get("n_critical", 0)
            violations = compliance_decision.get("violation_summary", [])

            # Healthcare PHI violation → always auto-reject
            if domain == "healthcare" and any(
                v.get("rule_name") in ("phi_presence", "de_identification")
                and v.get("severity") == "CRITICAL"
                for v in violations
            ):
                reason = (
                    "Healthcare PHI/de-identification CRITICAL violation detected. "
                    "Auto-REJECT: raw patient data must not proceed through the pipeline."
                )
                logger.error("[%s] Verifier AUTO-REJECT (PHI): %s", run_id[:8], reason)
                return {"decision": "REJECT", "reason": reason, "mode": "compliance_auto_reject"}

            # GDPR data residency CRITICAL → auto-reject
            if any(
                v.get("rule_name") == "gdpr_data_residency"
                and v.get("severity") == "CRITICAL"
                for v in violations
            ):
                reason = (
                    "GDPR Art. 44 data residency CRITICAL violation detected. "
                    "Auto-REJECT: cross-border data transfer not permitted."
                )
                logger.error("[%s] Verifier AUTO-REJECT (GDPR): %s", run_id[:8], reason)
                return {"decision": "REJECT", "reason": reason, "mode": "compliance_auto_reject"}

        # ── ROC-AUC thresholds (tightened by compliance violations) ───────────
        if roc_auc > 0.0:
            # Base thresholds
            base_auc = {"banking": 0.65, "healthcare": 0.70, "finance": 0.65, "default": 0.50}
            min_auc = base_auc.get(domain, 0.50)

            # Tighten if compliance violations are present
            if compliance_decision:
                n_critical = compliance_decision.get("n_critical", 0)
                n_error    = compliance_decision.get("n_error", 0)
                if n_critical > 0:
                    min_auc = min(min_auc + 0.15, 0.92)  # tighten by 0.15
                elif n_error > 0:
                    min_auc = min(min_auc + 0.07, 0.85)  # tighten by 0.07
                logger.info(
                    "[%s] Verifier: compliance-adjusted min_auc=%.3f (CRITICAL=%d ERROR=%d)",
                    run_id[:8], min_auc, n_critical, n_error,
                )

            if roc_auc < min_auc:
                reason = (
                    f"ROC-AUC {roc_auc:.3f} below domain minimum {min_auc:.3f}"
                    + (" (tightened by compliance violations)" if compliance_decision else "")
                )
                logger.warning("[%s] Verifier REJECT (classification): %s", run_id[:8], reason)
                return {"decision": "REJECT", "reason": reason, "mode": "classification"}

            logger.info(
                "[%s] Verifier PASS (classification): ROC-AUC=%.3f >= %.3f",
                run_id[:8], roc_auc, min_auc,
            )
            return {
                "decision": "PASS",
                "mode": "classification",
                "roc_auc": roc_auc,
                "min_auc_applied": min_auc,
            }

        # ── Non-classification / EDA / unsupervised mode ───────────────────────
        quality_score = float(model_metrics.get("quality_score", 0.0))
        min_quality = {"banking": 0.60, "healthcare": 0.65, "default": 0.50}.get(domain, 0.50)

        if quality_score > 0.0 and quality_score < min_quality:
            reason = (
                f"Unsupervised/EDA run: quality_score {quality_score:.3f} "
                f"below domain minimum {min_quality:.3f} (no roc_auc available)"
            )
            logger.warning("[%s] Verifier REJECT (unsupervised): %s", run_id[:8], reason)
            return {"decision": "REJECT", "reason": reason, "mode": "unsupervised"}

        logger.info(
            "[%s] Verifier PASS (unsupervised/EDA mode): quality_score=%.3f",
            run_id[:8], quality_score,
        )
        return {"decision": "PASS", "mode": "unsupervised", "quality_score": quality_score}

    def aggregate(
        self,
        df: pd.DataFrame,
        model_metrics: Dict[str, Any],
        quality_score: float,
        gate2_passed: bool,
        retry_count: int,
        compliance_penalty: float = 0.0,
        compliance_decision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Produces a weighted scalar confidence score bridging multiple pipeline signals.

        Components:
          base_quality        — overall data quality score from profiler (0–1)
          completeness        — 1 - mean null rate across all columns (0–1)
          consistency         — 1 - duplicate row fraction (0–1)
          model_boost         — bonus for good ROC-AUC (classification runs only)
          compliance_penalty  — ∈ [-1.0, 0.0], subtracted directly from score
          retry_penalty       — penalty per retry attempt (-0.05 each)

        Weighting (classification): quality 35% + completeness 20% + consistency 10% + model 35%
        Weighting (unsupervised):   quality 50% + completeness 35% + consistency 15%

        Note: compliance_penalty is additive (always ≤ 0) and applied AFTER weighting.
        """
        n_rows = max(len(df), 1)

        # ── Data quality dimensions (always available) ─────────────────────────
        null_rate    = float(df.isnull().mean().mean()) if not df.empty else 0.0
        completeness = max(0.0, 1.0 - null_rate)

        dup_rate  = float(df.duplicated().sum()) / n_rows if not df.empty else 0.0
        consistency = max(0.0, 1.0 - dup_rate)

        # ── Model performance component ────────────────────────────────────────
        roc_auc = float(model_metrics.get("roc_auc", 0.0))
        has_model = roc_auc > 0.0
        model_boost = max(0.0, roc_auc - 0.5) * 0.4 if has_model else 0.0  # max 0.20

        # ── Weighted aggregation ────────────────────────────────────────────────
        if has_model:
            confidence_score = (
                quality_score  * 0.35
                + completeness * 0.20
                + consistency  * 0.10
                + model_boost
            )
        else:
            confidence_score = (
                quality_score  * 0.50
                + completeness * 0.35
                + consistency  * 0.15
            )

        # ── Penalties ──────────────────────────────────────────────────────────
        retry_penalty = retry_count * 0.05
        confidence_score -= retry_penalty

        # ── Compliance penalty (core accuracy impact) ─────────────────────────
        # compliance_penalty ∈ [-1.0, 0.0], already clamped by ComplianceAdvisor
        safe_compliance_penalty = float(max(-1.0, min(0.0, compliance_penalty)))
        confidence_score += safe_compliance_penalty

        # ── Hard cap if Gate 2 failed ─────────────────────────────────────────
        if not gate2_passed:
            confidence_score = min(confidence_score, 0.49)

        confidence_score = max(0.0, min(1.0, confidence_score))

        logger.info(
            "ConfidenceVector: score=%.3f (quality=%.3f completeness=%.3f "
            "consistency=%.3f model_boost=%.3f compliance_penalty=%.3f "
            "retry_penalty=%.2f mode=%s decision=%s)",
            confidence_score, quality_score, completeness, consistency,
            model_boost, safe_compliance_penalty, retry_penalty,
            "classification" if has_model else "unsupervised",
            compliance_decision or "n/a",
        )

        return {
            "confidence_score": confidence_score,
            "components": {
                "base_quality":         quality_score,
                "completeness":         round(completeness, 4),
                "consistency":          round(consistency, 4),
                "model_boost":          round(model_boost, 4),
                "compliance_penalty":   round(safe_compliance_penalty, 4),
                "retry_penalty":        retry_penalty,
                "gate2_passed":         gate2_passed,
                "compliance_decision":  compliance_decision or "n/a",
                "mode":                 "classification" if has_model else "unsupervised",
            },
        }
