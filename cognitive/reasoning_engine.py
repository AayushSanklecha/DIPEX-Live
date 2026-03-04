"""
cognitive/reasoning_engine.py
-------------------------------
Master orchestrator of analytical cognition — the "thinking" layer.

Acts as the senior cognitive layer that every analyst tier passes through:
  1. Sanity-checks outputs before surfacing
  2. Tracks all assumptions made during analysis
  3. Detects leakage patterns proactively
  4. Quantifies uncertainty on key metrics
  5. Calibrates expectations and filters publishable insights
  6. Flags silent inconsistencies and cross-metric contradictions
  7. Requests clarification when ambiguity exceeds threshold

This is NOT a rule engine — it mimics the hidden cognitive skills of
a senior analyst: "does this make sense?", "what am I assuming here?",
"have I checked for leakage?", "how confident am I really?"
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from cognitive.sanity_checker import SanityChecker, SanityViolation
from cognitive.assumption_tracker import AssumptionTracker, Assumption
from cognitive.leakage_sentinel import LeakageSentinel, LeakageWarning
from cognitive.uncertainty_quantifier import UncertaintyQuantifier, UncertaintyReport
from cognitive.expectation_calibrator import ExpectationCalibrator, InsightVerdict

logger = logging.getLogger("dipex.cognitive.reasoning_engine")


# ── Context & Finding Dataclasses ─────────────────────────────────────────────

@dataclass
class AnalysisContext:
    """Carries all metadata for a single analytical operation."""
    dataset_id:     str = ""
    operation:      str = ""
    analyst_tier:   str = "junior"         # junior | mid | senior
    target_col:     Optional[str] = None
    date_col:       Optional[str] = None
    group_col:      Optional[str] = None
    extra_rules:    List[Dict] = field(default_factory=list)
    stakeholder_expectation: Optional[str] = None
    session_id:     str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp:      str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class CognitiveFinding:
    """Structured output from one cognitive check cycle."""
    context:         AnalysisContext
    sanity_ok:       bool
    leakage_clean:   bool
    safe_to_publish: bool
    sanity_violations:   List[SanityViolation] = field(default_factory=list)
    leakage_warnings:    List[LeakageWarning]  = field(default_factory=list)
    assumptions:         List[Assumption]       = field(default_factory=list)
    uncertainty_reports: Dict[str, UncertaintyReport] = field(default_factory=dict)
    insight_verdicts:    List[InsightVerdict]   = field(default_factory=list)
    clarification_needed: bool = False
    clarification_reason: str = ""
    cognitive_score:     float = 1.0   # 0=do not use, 1=fully trusted

    def to_dict(self) -> Dict:
        return {
            "session_id": self.context.session_id,
            "dataset_id": self.context.dataset_id,
            "operation": self.context.operation,
            "sanity_ok": self.sanity_ok,
            "leakage_clean": self.leakage_clean,
            "safe_to_publish": self.safe_to_publish,
            "cognitive_score": round(self.cognitive_score, 4),
            "clarification_needed": self.clarification_needed,
            "clarification_reason": self.clarification_reason,
            "sanity_violations": [v.to_dict() for v in self.sanity_violations],
            "leakage_warnings": [w.to_dict() for w in self.leakage_warnings],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "uncertainty_reports": {
                k: v.to_dict() for k, v in self.uncertainty_reports.items()
            },
            "insight_verdicts": [iv.to_dict() for iv in self.insight_verdicts],
        }


# ── CognitiveReasoningEngine ──────────────────────────────────────────────────

class CognitiveReasoningEngine:
    """
    The analytical brain of the DIPEX system.

    Wraps every analytical operation with cognitive checks that replicate
    the hidden judgement skills of a senior analyst. Called automatically
    by the AnalystOrchestrator at each tier transition.

    Usage::

        engine  = CognitiveReasoningEngine(config)
        ctx     = AnalysisContext(dataset_id="sales", operation="risk_scoring")
        finding = engine.reason(gold_df, ctx, insights=["Revenue up 12%"])
        if not finding.safe_to_publish:
            handle_suppressed_output(finding)
    """

    # Ambiguity threshold: if cross-metric contradiction rate > this, request clarification
    AMBIGUITY_THRESHOLD = 0.15

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = config or {}
        self.sanity     = SanityChecker(config=cfg)
        self.sentinel   = LeakageSentinel(config=cfg)
        self.quantifier = UncertaintyQuantifier()
        self.calibrator = ExpectationCalibrator()
        self.assumptions = AssumptionTracker()
        self._config    = cfg

    # ── Public API ────────────────────────────────────────────────────────────

    def reason(
        self,
        df: pd.DataFrame,
        ctx: AnalysisContext,
        insights: Optional[List[str]] = None,
        target_df: Optional[pd.DataFrame] = None,   # for leakage: test set
        key_metrics: Optional[List[str]] = None,    # columns to quantify uncertainty
    ) -> CognitiveFinding:
        """
        Run full cognitive scan on output DataFrame and return a CognitiveFinding.

        Parameters
        ----------
        df           : The Gold artefact DataFrame to check
        ctx          : AnalysisContext describing the operation
        insights     : List of insight strings to evaluate and label
        key_metrics  : Columns to compute uncertainty intervals for
        """
        finding = CognitiveFinding(context=ctx, sanity_ok=True,
                                   leakage_clean=True, safe_to_publish=True)

        # ── 1. Sanity Gate ──────────────────────────────────────────────────
        violations = self.sanity.check(df, ctx.dataset_id, ctx.extra_rules)
        finding.sanity_violations = violations
        finding.sanity_ok = not any(v.severity == "CRITICAL" for v in violations)

        # ── 2. Leakage Sentinel ─────────────────────────────────────────────
        lk_warnings = self.sentinel.check(
            df,
            target_col=ctx.target_col,
            date_col=ctx.date_col,
            group_col=ctx.group_col,
            test_df=target_df,
        )
        finding.leakage_warnings = lk_warnings
        finding.leakage_clean = not any(w.severity == "CRITICAL" for w in lk_warnings)

        # ── 3. Silent inconsistency (cross-metric contradiction) ─────────────
        contradiction_rate, contradiction_detail = self._detect_contradictions(df)
        if contradiction_rate > self.AMBIGUITY_THRESHOLD:
            finding.clarification_needed = True
            finding.clarification_reason = (
                f"Cross-column contradiction rate {contradiction_rate:.1%} exceeds "
                f"threshold {self.AMBIGUITY_THRESHOLD:.0%}. Detail: {contradiction_detail}"
            )

        # ── 4. Uncertainty Quantification ────────────────────────────────────
        cols_to_quantify = key_metrics or df.select_dtypes("number").columns[:5].tolist()
        for col in cols_to_quantify:
            if col in df.columns:
                finding.uncertainty_reports[col] = \
                    self.quantifier.quantify_dataframe_column(df, col)

        # ── 5. Insight Calibration ───────────────────────────────────────────
        if insights:
            for stmt in insights:
                conf = self._estimate_confidence(stmt, finding)
                verdict = self.calibrator.evaluate(
                    statement=stmt, confidence=conf,
                    stakeholder_expected=ctx.stakeholder_expectation,
                )
                finding.insight_verdicts.append(verdict)

        # ── 6. Auto-record assumptions ───────────────────────────────────────
        finding.assumptions = self._auto_record_assumptions(df, ctx)

        # ── 7. Cognitive Score ───────────────────────────────────────────────
        finding.cognitive_score = self._compute_cognitive_score(finding)
        finding.safe_to_publish = (
            finding.sanity_ok
            and finding.leakage_clean
            and not finding.clarification_needed
            and finding.cognitive_score >= 0.5
            and self.assumptions.safe_to_publish()
        )

        # Log summary
        logger.info(
            "[CognitiveReasoningEngine] %s/%s — sanity=%s leakage=%s "
            "score=%.2f safe=%s",
            ctx.dataset_id, ctx.operation,
            "✓" if finding.sanity_ok else "✗",
            "✓" if finding.leakage_clean else "✗",
            finding.cognitive_score, finding.safe_to_publish,
        )
        return finding

    def annotate_result(
        self, result: Dict, finding: CognitiveFinding
    ) -> Dict:
        """
        Inject cognitive metadata into any result dict before it is returned
        to the caller. Every output should carry this data.
        """
        result["_cognitive"] = {
            "sanity_ok": finding.sanity_ok,
            "leakage_clean": finding.leakage_clean,
            "safe_to_publish": finding.safe_to_publish,
            "cognitive_score": finding.cognitive_score,
            "clarification_needed": finding.clarification_needed,
            "n_assumptions": len(finding.assumptions),
            "n_flagged_assumptions": sum(1 for a in finding.assumptions if a.flagged()),
            "n_violations": len(finding.sanity_violations),
            "n_leakage_warnings": len(finding.leakage_warnings),
            "uncertainty_q_hats": {
                k: getattr(v, 'q_hat', 0.0) for k, v in finding.uncertainty_reports.items()
            },
        }
        return result

    def request_clarification(self, finding: CognitiveFinding) -> Optional[str]:
        """Generate a clarification request string when ambiguity is detected."""
        if not finding.clarification_needed:
            return None
        return (
            f"⚠ CLARIFICATION NEEDED before publishing results for "
            f"dataset '{finding.context.dataset_id}':\n"
            f"{finding.clarification_reason}\n\n"
            f"Please verify the following assumptions:\n"
            + "\n".join(f"  • {a.statement}" for a in finding.assumptions if a.flagged())
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_contradictions(
        self, df: pd.DataFrame
    ) -> tuple[float, str]:
        """
        Look for cross-column logical contradictions:
        e.g. refunds > revenue, active_users > total_users.
        Returns (contradiction_rate, detail_string).
        """
        num_df = df.select_dtypes("number")
        if num_df.shape[1] < 2:
            return 0.0, ""
        # Heuristic: columns with "total" should be >= columns with "active"/"new"
        total_cols  = [c for c in num_df.columns if "total" in c.lower()]
        partial_cols = [c for c in num_df.columns
                        if any(k in c.lower() for k in ["active", "new", "sub",
                                                          "refund", "return"])]
        contradiction_rows = 0
        detail_parts = []
        for tc in total_cols:
            for pc in partial_cols:
                if tc == pc:
                    continue
                try:
                    bad = (df[pc] > df[tc]).sum()
                    if bad > 0:
                        contradiction_rows += bad
                        detail_parts.append(f"{pc} > {tc} ({bad} rows)")
                except Exception:  # noqa: BLE001
                    pass
        rate = contradiction_rows / max(len(df), 1)
        return rate, "; ".join(detail_parts[:3])

    def _estimate_confidence(
        self, statement: str, finding: CognitiveFinding
    ) -> float:
        """Heuristically estimate confidence for a text insight statement."""
        base = 0.85
        # Penalties
        if not finding.sanity_ok:
            base -= 0.25
        if not finding.leakage_clean:
            base -= 0.30
        if finding.clarification_needed:
            base -= 0.15
        crit_uncertainty = sum(
            1 for u in finding.uncertainty_reports.values()
            if getattr(u, "q_hat", 0.0) > 0.5
        )
        base -= crit_uncertainty * 0.05
        # Boosts
        if "significant" in statement.lower() or "p-value" in statement.lower():
            base += 0.05
        if "95% ci" in statement.lower() or "confidence interval" in statement.lower():
            base += 0.03
        return max(0.05, min(base, 1.0))

    def _auto_record_assumptions(
        self, df: pd.DataFrame, ctx: AnalysisContext
    ) -> List[Assumption]:
        assumptions = []
        null_rate = df.isnull().mean().mean()
        if null_rate > 0.01:
            assumptions.append(self.assumptions.record(
                f"Missing values ({null_rate:.1%} overall) are handled by imputation "
                f"or exclusion — may bias results",
                category="data", confidence=0.7, risk_if_wrong="MEDIUM",
                dataset_id=ctx.dataset_id, analysis_step=ctx.operation,
            ))
        if not ctx.target_col:
            assumptions.append(self.assumptions.record(
                "No target column specified — analysis is unsupervised/descriptive",
                category="scope", confidence=0.9, risk_if_wrong="LOW",
                dataset_id=ctx.dataset_id,
            ))
        if len(df) < 30:
            assumptions.append(self.assumptions.record(
                f"Very small sample (n={len(df)}) — statistical results may not be reliable",
                category="data", confidence=0.5, risk_if_wrong="HIGH",
                dataset_id=ctx.dataset_id,
            ))
        return assumptions

    def _compute_cognitive_score(self, finding: CognitiveFinding) -> float:
        """Weighted score: 1.0 = fully trusted, 0.0 = suppress."""
        score = 1.0
        crit_sanity   = sum(1 for v in finding.sanity_violations if v.severity == "CRITICAL")
        crit_leakage  = sum(1 for w in finding.leakage_warnings  if w.severity == "CRITICAL")
        high_unc      = sum(1 for u in finding.uncertainty_reports.values()
                            if getattr(u, "q_hat", 0.0) > 0.5)
        score -= crit_sanity   * 0.25
        score -= crit_leakage  * 0.35
        score -= high_unc      * 0.05
        if finding.clarification_needed:
            score -= 0.10
        flagged_assumptions = sum(1 for a in finding.assumptions if a.flagged())
        score -= flagged_assumptions * 0.03
        return round(max(0.0, min(score, 1.0)), 4)
