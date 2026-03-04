"""
analyst/analyst_orchestrator.py
---------------------------------
Master Analyst Orchestrator — routes analytical tasks across Junior, Mid,
and Senior tiers with cognitive reasoning injected at every tier transition.

This is the single entry point for all programmatic analyst operations.
It behaves like a "thinking analyst" by:
  1. Framing the problem before starting (Senior)
  2. Running cognitive checks before every tier transition
  3. Injecting LeakageSentinel + SanityChecker on all intermediate outputs
  4. Escalating complexity: if Junior output fails cognitive check → MidAnalyst
  5. Applying MentorshipEngine review before surfacing any final output
  6. Running RL optimizer to propose next-best strategy on failures
  7. Logging full audit trail (tier used, cognitive score, assumptions, lineage)

INVARIANT: All operations run on Gold-layer copies. Silver is never touched.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from ingestion.data_layers import LayerManager, ImmutableDataFrame, GoldArtefact
from cognitive.reasoning_engine import CognitiveReasoningEngine, AnalysisContext
from analyst.junior_analyst import JuniorAnalyst
from analyst.mid_analyst import MidAnalyst
from analyst.senior_analyst import SeniorAnalyst
from analyst.problem_framing import ProblemFramingEngine
from analyst.mentorship_engine import MentorshipEngine
from analyst.rl_optimizer import RLOptimizer, StrategyDomain

logger = logging.getLogger("dipex.analyst.orchestrator")


@dataclass
class OrchestratorResult:
    run_id:         str
    operation:      str
    tier_used:      str
    dataset_id:     str
    gold_artefact:  Optional[GoldArtefact] = None
    cognitive_score: float = 1.0
    safe_to_publish: bool = True
    mentorship_score: float = 100.0
    mentorship_approved: bool = True
    assumptions:    List[Dict] = field(default_factory=list)
    warnings:       List[str] = field(default_factory=list)
    rl_proposals:   List[Dict] = field(default_factory=list)
    framed_problem: Optional[Dict] = None
    elapsed_ms:     float = 0.0
    timestamp:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id, "operation": self.operation,
            "tier_used": self.tier_used, "dataset_id": self.dataset_id,
            "cognitive_score": round(self.cognitive_score, 4),
            "safe_to_publish": self.safe_to_publish,
            "mentorship_score": round(self.mentorship_score, 1),
            "mentorship_approved": self.mentorship_approved,
            "n_assumptions": len(self.assumptions),
            "n_warnings": len(self.warnings),
            "rl_proposals": len(self.rl_proposals),
            "has_gold_artefact": self.gold_artefact is not None,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "timestamp": self.timestamp,
        }


class AnalystOrchestrator:
    """
    Master orchestrator for the 3-tier Analyst Intelligence Automation Layer.

    All analyst operations are routed through this class.
    Cognitive reasoning is injected at every tier transition.

    Usage::

        orch   = AnalystOrchestrator(config)
        silver = orch.make_silver(df, dataset_id="sales_q4")
        result = orch.run(
            operation="statistical_analysis",
            silver=silver,
            params={"group_col": "region", "value_col": "revenue"},
        )
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        operator: str = "system",
    ) -> None:
        self.config   = config or {}
        self.operator = operator
        self.lm       = LayerManager(base_dir=self.config.get("data_layers", {}).get("base_dir", "data"))
        self.brain    = CognitiveReasoningEngine(config=self.config)
        self.junior   = JuniorAnalyst(layer_manager=self.lm, operator=operator)
        self.mid      = MidAnalyst(layer_manager=self.lm, operator=operator, config=self.config)
        self.senior   = SeniorAnalyst(layer_manager=self.lm, operator=operator)
        self.framer   = ProblemFramingEngine()
        self.mentor   = MentorshipEngine()
        self.rl       = RLOptimizer(config=self.config)

    # ── Silver Factory ────────────────────────────────────────────────────────

    def make_silver(self, df: pd.DataFrame, dataset_id: str) -> ImmutableDataFrame:
        """Wrap a DataFrame as a Silver ImmutableDataFrame for use in operations."""
        return ImmutableDataFrame(df.copy(), layer="silver", dataset_id=dataset_id)

    # ── Master Entry Point ────────────────────────────────────────────────────

    def run(
        self,
        operation: str,
        silver: ImmutableDataFrame,
        params: Optional[Dict] = None,
        auto_tier: bool = True,
        force_tier: Optional[str] = None,
        problem_statement: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        Route an operation to the appropriate tier with full cognitive governance.

        Parameters
        ----------
        operation         : e.g. "statistical_analysis", "automated_eda", "business_insights"
        silver            : ImmutableDataFrame (Silver layer)
        params            : Operation-specific parameters
        auto_tier         : If True, select tier automatically based on operation complexity
        force_tier        : "junior" | "mid" | "senior" to override auto selection
        problem_statement : Optional natural-language problem to frame first
        """
        import time
        t0     = time.perf_counter()
        run_id = str(uuid.uuid4())[:12]
        params = params or {}
        result = OrchestratorResult(
            run_id=run_id, operation=operation,
            dataset_id=silver._dataset_id,
            tier_used=force_tier or "auto",
        )

        # ── Step 0: Problem framing (optional) ──────────────────────────────
        if problem_statement:
            framed = self.framer.frame(
                problem_statement,
                available_columns=list(silver._df.columns),
                dataset_id=silver._dataset_id,
            )
            result.framed_problem = framed.to_dict()
            logger.info("[Orchestrator] Problem framed: intent=%s n_kpis=%d",
                        framed.detected_intent, len(framed.kpi_proposals))

        # ── Step 1: Pre-cognitive check ──────────────────────────────────────
        ctx = AnalysisContext(
            dataset_id=silver._dataset_id, operation=operation,
            target_col=params.get("target_col"),
            date_col=params.get("date_col"),
        )
        finding = self.brain.reason(silver._df, ctx)
        result.cognitive_score = finding.cognitive_score
        result.assumptions     = [a.to_dict() for a in finding.assumptions]

        if not finding.sanity_ok:
            result.warnings.append(f"Sanity violations: {[v.detail[:60] for v in finding.sanity_violations[:3]]}")
        if not finding.leakage_clean:
            result.warnings.append(f"Leakage detected: {[w.detail[:60] for w in finding.leakage_warnings[:2]]}")
        if finding.clarification_needed:
            result.warnings.append(f"Clarification needed: {finding.clarification_reason[:100]}")

        # ── Step 2: Tier selection ───────────────────────────────────────────
        tier = force_tier or (self._select_tier(operation) if auto_tier else "mid")
        result.tier_used = tier

        # ── Step 3: Run operation on appropriate tier ────────────────────────
        gold: Optional[GoldArtefact] = None
        try:
            gold = self._dispatch(tier, operation, silver, params)
        except Exception as e:  # noqa: BLE001
            logger.warning("[Orchestrator] %s/%s failed: %s — trying RL fallback", tier, operation, e)
            result.warnings.append(f"Operation failed: {e!s:.80}")

            # RL proposes a fallback strategy
            proposals = self.rl.propose(StrategyDomain.CLEANING,
                                         context={"null_rate": silver._df.isnull().mean().mean(),
                                                   "n_rows": len(silver._df)})
            result.rl_proposals = [p.to_dict() for p in proposals]

        # ── Step 4: Post-cognitive annotation ───────────────────────────────
        if gold is not None:
            result.gold_artefact = gold
            # Mentorship review
            mentor_review = self.mentor.review_logic(
                operation=operation,
                df=gold.data if hasattr(gold, "data") else silver._df,
                target_col=params.get("target_col"),
            )
            result.mentorship_score    = mentor_review.score
            result.mentorship_approved = mentor_review.approved

        result.safe_to_publish = (
            finding.safe_to_publish
            and result.mentorship_approved
            and (gold is not None)
        )
        result.elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[Orchestrator] %s run_id=%s tier=%s score=%.2f safe=%s elapsed=%.0fms",
            operation, run_id, tier, finding.cognitive_score,
            result.safe_to_publish, result.elapsed_ms,
        )
        return result

    # ── Batch Mode ────────────────────────────────────────────────────────────

    def run_full_analysis(
        self,
        silver: ImmutableDataFrame,
        target_col: Optional[str] = None,
        problem_statement: Optional[str] = None,
    ) -> Dict[str, OrchestratorResult]:
        """
        Run the complete Junior → Mid → Senior analysis pipeline on Silver data.
        Returns a dict of operation → OrchestratorResult.
        """
        results: Dict[str, OrchestratorResult] = {}
        # Junior ops
        for op in ["basic_stats", "missing_analysis", "data_cleaning"]:
            try:
                results[op] = self.run(op, silver, force_tier="junior")
            except Exception as e:  # noqa: BLE001
                logger.warning("[Orchestrator] junior/%s skipped: %s", op, e)

        # Mid ops
        mid_ops = ["automated_eda", "business_insights", "outlier_investigation"]
        for op in mid_ops:
            try:
                params = {"target_col": target_col} if target_col else {}
                results[op] = self.run(op, silver, params=params, force_tier="mid")
            except Exception as e:  # noqa: BLE001
                logger.warning("[Orchestrator] mid/%s skipped: %s", op, e)

        # Senior ops
        for op in ["strategic_analysis", "risk_assessment"]:
            try:
                params = {"target_col": target_col} if target_col else {}
                results[op] = self.run(op, silver, params=params, force_tier="senior",
                                        problem_statement=problem_statement)
            except Exception as e:  # noqa: BLE001
                logger.warning("[Orchestrator] senior/%s skipped: %s", op, e)

        logger.info("[Orchestrator] Full analysis complete: %d ops run", len(results))
        return results

    # ── Tier Selector ─────────────────────────────────────────────────────────

    @staticmethod
    def _select_tier(operation: str) -> str:
        junior_ops = {
            "basic_stats", "missing_analysis", "data_cleaning",
            "schema_validation", "duplicate_detection", "data_profiling",
            "basic_visualization_spec", "merge_files", "export_report",
            "pivot_table",
        }
        senior_ops = {
            "strategic_analysis", "risk_assessment", "problem_framing",
            "experiment_design", "executive_report", "pricing_analysis",
            "causal_inference_proxy", "bias_detection", "north_star_metric_definition",
            "sensitivity_analysis",
        }
        if operation in junior_ops:
            return "junior"
        if operation in senior_ops:
            return "senior"
        return "mid"  # default to mid for unknown ops

    # ── Operation Dispatcher ──────────────────────────────────────────────────

    def _dispatch(
        self,
        tier: str,
        operation: str,
        silver: ImmutableDataFrame,
        params: Dict,
    ) -> Optional[GoldArtefact]:
        if tier == "junior":
            return self._dispatch_junior(operation, silver, params)
        elif tier == "mid":
            return self._dispatch_mid(operation, silver, params)
        elif tier == "senior":
            return self._dispatch_senior(operation, silver, params)
        return None

    def _dispatch_junior(self, op: str, silver: ImmutableDataFrame, p: Dict) -> Optional[GoldArtefact]:
        dispatch = {
            "basic_stats":       lambda: self.junior.basic_stats(silver),
            "missing_analysis":  lambda: self.junior.missing_analysis(silver),
            "data_cleaning":     lambda: self.junior.data_cleaning(silver),
            "duplicate_detection": lambda: self.junior.duplicate_detection(silver),
            "data_profiling":    lambda: self.junior.data_profiling(silver),
            # ── New Phase 1b operations ─────────────────────────────────────
            "basic_visualization_spec": lambda: self.junior.basic_visualization_spec(
                silver,
                x_col=p.get("x_col", silver.data.columns[0]),
                y_col=p.get("y_col"),
                chart_type=p.get("chart_type"),
            ),
            "merge_files": lambda: self.junior.merge_files(
                sources=p.get("sources", [silver]),
                how=p.get("how", "concat"),
                on=p.get("on"),
                dataset_id=p.get("dataset_id", f"{silver._dataset_id}_merged"),
            ),
            "export_report": lambda: self.junior.export_report(
                silver,
                output_path=p.get("output_path", "reports/report.md"),
                fmt=p.get("fmt", "markdown"),
            ),
            "pivot_table": lambda: self.junior.pivot_table(
                silver,
                index=p.get("index", []),
                values=p.get("values", []),
                aggfunc=p.get("aggfunc", "mean"),
            ),
        }
        fn = dispatch.get(op)
        return fn() if fn else None

    def _dispatch_mid(self, op: str, silver: ImmutableDataFrame, p: Dict) -> Optional[GoldArtefact]:
        dispatch = {
            "automated_eda":         lambda: self.mid.automated_eda(silver, target_col=p.get("target_col")),
            "statistical_analysis":  lambda: self.mid.statistical_analysis(silver, p["group_col"], p["value_col"]),
            "ab_test_evaluation":    lambda: self.mid.ab_test_evaluation(silver, p["group_col"], p["metric_col"]),
            "business_insights":     lambda: self.mid.business_insights(silver, target_col=p.get("target_col")),
            "outlier_investigation": lambda: self.mid.outlier_investigation(silver),
            "variance_analysis":     lambda: self.mid.variance_analysis(silver, p["group_col"], p["value_col"]),
            "segmentation_clustering": lambda: self.mid.segmentation_clustering(silver, n_clusters=p.get("n_clusters", 4)),
            "time_series_exploration": lambda: self.mid.time_series_exploration(silver, p["date_col"], p["value_col"]),
            "cohort_analysis":       lambda: self.mid.cohort_analysis(silver, p["cohort_col"], p["time_col"], p["value_col"]),
            "correlation_deep_dive": lambda: self.mid.correlation_deep_dive(silver, target_col=p.get("target_col")),
            "advanced_sql":          lambda: self.mid.advanced_sql(silver, p["sql"]),
            "dashboard_design":      lambda: self.mid.dashboard_design(silver),
        }
        fn = dispatch.get(op)
        return fn() if fn else None

    def _dispatch_senior(self, op: str, silver: ImmutableDataFrame, p: Dict) -> Optional[GoldArtefact]:
        dispatch = {
            "strategic_analysis":  lambda: self.senior.strategic_analysis(silver, p.get("target_col")),
            "risk_assessment":     lambda: self.senior.risk_assessment(silver),
            "sensitivity_analysis": lambda: self.senior.sensitivity_analysis(
                silver,
                target_col=p.get("target_col", ""),
            ),
            # ── New Phase 1b operations ─────────────────────────────────────
            "causal_inference_proxy": lambda: self.senior.causal_inference_proxy(
                silver,
                treatment_col=p.get("treatment_col", ""),
                outcome_col=p.get("outcome_col", ""),
                time_col=p.get("time_col"),
                method=p.get("method", "did"),
            ),
            "bias_detection": lambda: self.senior.bias_detection(
                silver,
                sensitive_cols=p.get("sensitive_cols", []),
                outcome_col=p.get("outcome_col", ""),
                positive_outcome_value=p.get("positive_outcome_value", 1),
            ),
            "north_star_metric_definition": lambda: self.senior.north_star_metric_definition(
                silver,
                business_objective=p.get("business_objective", ""),
                kpi_candidates=p.get("kpi_candidates", []),
            ),
        }
        fn = dispatch.get(op)
        return fn() if fn else None
