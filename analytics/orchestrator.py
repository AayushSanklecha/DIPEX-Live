"""
analytics/orchestrator.py
---------------------------
AI & ANALYTICS SERVICE LAYER — Analytics Orchestrator

AnalyticsOrchestrator.run() sequences all 4 AI/Analytics sub-components:
  1. Automated EDA         (eda/auto_eda.py)
  2. Feature Engineering   (feature_engineering/engineer.py)
  3. Insight Ranking       (proposal/insight_ranker.py)
  4. LLM Summarization     (reporting_service/llm_provider.py)

Returns an AnalyticsResult that the pipeline can store and forward to
the Presentation Layer (reports, dashboard, exports).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("dipex.analytics.orchestrator")


# ── Analytics Result ──────────────────────────────────────────────────────────

@dataclass
class AnalyticsResult:
    """Combined output from all 4 AI & Analytics sub-components."""
    run_id: str = ""
    eda_report: Dict = field(default_factory=dict)
    feature_manifest: Dict = field(default_factory=dict)
    enriched_df: Optional[pd.DataFrame] = None
    insights: List[str] = field(default_factory=list)
    insight_ranking: Dict = field(default_factory=dict)
    llm_summary: str = ""
    elapsed_ms: float = 0.0
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self, include_df: bool = False) -> Dict:
        out = {
            "run_id": self.run_id,
            "eda_report": self.eda_report,
            "feature_manifest": self.feature_manifest,
            "insights": self.insights,
            "insight_ranking": self.insight_ranking,
            "llm_summary": self.llm_summary,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "errors": self.errors,
        }
        if include_df and self.enriched_df is not None:
            out["enriched_shape"] = list(self.enriched_df.shape)
        return out


# ── Analytics Orchestrator ────────────────────────────────────────────────────

class AnalyticsOrchestrator:
    """
    Sequences the full AI & Analytics Service Layer.

    Usage::

        orchestrator = AnalyticsOrchestrator(config=config)
        result = orchestrator.run(df, target_col="churn", run_id=run_id)

        print(result.insights)        # list of EDA-derived insights
        print(result.llm_summary)     # LLM narrative
        enriched = result.enriched_df  # feature-engineered DataFrame
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def run(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        run_id: Optional[str] = None,
        qa_result: Optional[Dict] = None,
    ) -> AnalyticsResult:
        """
        Execute EDA → Feature Engineering → Insight Ranking → LLM Summary.

        Parameters
        ----------
        df         : preprocessed DataFrame (after QA layer)
        target_col : supervised target column (optional)
        run_id     : pipeline run identifier
        qa_result  : optional QAResult dict for LLM context

        Returns
        -------
        AnalyticsResult
        """
        t0 = time.perf_counter()
        result = AnalyticsResult(run_id=run_id or "")

        if df is None or df.empty:
            logger.warning("[AnalyticsOrchestrator] Empty DataFrame — skipping analytics.")
            return result

        # ── Stage A: Automated EDA ─────────────────────────────────────────
        result.eda_report, result.insights = self._run_eda(df, run_id)

        # ── Stage B: Feature Engineering ──────────────────────────────────
        result.enriched_df, result.feature_manifest = self._run_feature_engineering(
            df, target_col
        )
        working_df = result.enriched_df if result.enriched_df is not None else df

        # ── Stage C: Insight Ranking ───────────────────────────────────────
        result.insight_ranking = self._run_insight_ranking(working_df, target_col, run_id)

        # ── Stage D: LLM Summarization ─────────────────────────────────────
        result.llm_summary = self._run_llm_summary(result, qa_result, run_id)

        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[AnalyticsOrchestrator][%s] eda=%d insights, fe=%s, llm=%d chars, elapsed=%.0fms",
            (run_id or "")[:8], len(result.insights),
            result.feature_manifest.get("final_shape", "n/a"),
            len(result.llm_summary), result.elapsed_ms,
        )
        return result

    # ── Sub-stage runners ─────────────────────────────────────────────────────

    def _run_eda(
        self, df: pd.DataFrame, run_id: Optional[str]
    ) -> tuple[Dict, List[str]]:
        try:
            from eda.auto_eda import AutoEDA
            eda = AutoEDA(config=self.config)
            report = eda.run(df, run_id=run_id)
            return report.to_dict(), report.insights
        except Exception as exc:
            logger.warning("[AnalyticsOrchestrator] EDA failed (non-fatal): %s", exc)
            return {}, []

    def _run_feature_engineering(
        self, df: pd.DataFrame, target_col: Optional[str]
    ) -> tuple[Optional[pd.DataFrame], Dict]:
        try:
            from feature_engineering.engineer import FeatureEngineer
            fe = FeatureEngineer(config=self.config)
            fe_result = fe.transform(df, target_col=target_col)
            return fe_result.df, fe_result.to_dict()
        except Exception as exc:
            logger.warning("[AnalyticsOrchestrator] Feature engineering failed (non-fatal): %s", exc)
            return df, {}

    def _run_insight_ranking(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
        run_id: Optional[str],
    ) -> Dict:
        try:
            from proposal.insight_ranker import InsightRanker
            ranker = InsightRanker(config=self.config)
            # InsightRanker expects a DataFrame; target_col is optional context
            ranking = ranker.rank(df, target_col=target_col)
            return ranking if isinstance(ranking, dict) else {"raw": str(ranking)}
        except Exception as exc:
            logger.debug("[AnalyticsOrchestrator] InsightRanker unavailable: %s", exc)
            return {}

    def _run_llm_summary(
        self,
        result: AnalyticsResult,
        qa_result: Optional[Dict],
        run_id: Optional[str],
    ) -> str:
        try:
            from reporting_service.llm_provider import get_llm_provider
            llm = get_llm_provider(self.config)

            context = {
                "run_id": run_id or "",
                "dataset_shape": result.eda_report.get("dataset_shape"),
                "insights": result.insights[:5],  # top-5 to keep prompt lean
                "top_insight": result.insight_ranking.get("top_insight_candidate"),
                "feature_net_added": result.feature_manifest.get("net_features_added", 0),
                "qa_decision": (qa_result or {}).get("overall_decision", "UNKNOWN"),
                "confidence_score": (qa_result or {}).get("confidence_score", 0.0),
            }
            summary = llm.generate_summary(context, run_id=run_id or "")
            return summary if isinstance(summary, str) else ""
        except Exception as exc:
            logger.debug("[AnalyticsOrchestrator] LLM summary unavailable: %s", exc)
            # Build a rule-based fallback summary
            parts = []
            if result.insights:
                parts.append("Key findings: " + "; ".join(result.insights[:3]))
            if result.feature_manifest.get("net_features_added"):
                parts.append(
                    f"{result.feature_manifest['net_features_added']} engineered features added."
                )
            return " ".join(parts) if parts else "Analytics summary unavailable."
