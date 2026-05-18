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
    """
    Combined output from all AI & Analytics sub-components.
    Extended with comprehensive analytics fields for professional reporting.
    """
    run_id: str = ""
    # ── Core (existing) ────────────────────────────────────────────────────────
    eda_report: Dict = field(default_factory=dict)
    feature_manifest: Dict = field(default_factory=dict)
    enriched_df: Optional[pd.DataFrame] = None
    insights: List[str] = field(default_factory=list)
    insight_ranking: Dict = field(default_factory=dict)
    llm_summary: str = ""
    eda_html_report_path: Optional[str] = None
    actions_log: Dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    errors: Dict[str, str] = field(default_factory=dict)
    retrain_required: bool = False

    # ── NEW: Pipeline Stage Timeline ───────────────────────────────────────────
    pipeline_stage_log: List[Dict] = field(default_factory=list)
    # Each entry: {stage, status, duration_ms, rows_in, rows_out, action_summary, timestamp}

    # ── NEW: Ingestion Metrics ─────────────────────────────────────────────────
    ingestion_metrics: Dict = field(default_factory=dict)
    # {source_type, bytes_ingested, total_gb, chunks_processed, schema_drift,
    #  null_rate, duplicate_rate, processing_speed_mbps, is_partial, file_size_mb}

    # ── NEW: Feature Importance ────────────────────────────────────────────────
    feature_importance: Dict = field(default_factory=dict)
    # {feature_name: importance_score, ...} — top-15 sorted by importance

    # ── NEW: Data Governance Summary ───────────────────────────────────────────
    governance_summary: Dict = field(default_factory=dict)
    # {pii_detected: int, redactions: int, pii_columns: [], governance_decision,
    #  bronze_checksum, silver_checksum, gold_checksum, compliance_status}

    # ── NEW: Data Lineage ──────────────────────────────────────────────────────
    data_lineage: Dict = field(default_factory=dict)
    # {raw: {rows, cols, checksum},
    #  bronze: {rows, cols, checksum, transforms: []},
    #  silver: {rows, cols, checksum, transforms: []},
    #  gold:   {rows, cols, checksum, transforms: []}}

    # ── NEW: Cross-Domain Rule Violations ─────────────────────────────────────
    cross_domain_flags: List[Dict] = field(default_factory=list)
    # [{rule_name, severity, domain, description, affected_columns, recommended_action}]

    # ── NEW: Statistical Significance Tests ────────────────────────────────────
    statistical_tests: Dict = field(default_factory=dict)
    # {normality: [{col, statistic, p_value, is_normal, interpretation}],
    #  stationarity: [{col, adf_stat, p_value, is_stationary}],
    #  homogeneity: [{group_col, levene_stat, p_value, equal_variance}]}

    # ── NEW: Bias & Fairness Report ────────────────────────────────────────────
    bias_fairness_report: Dict = field(default_factory=dict)
    # {checked: bool, groups_analyzed: [], results: [
    #   {group_col, group_value, sample_size, positive_rate, parity_ratio, disparate_impact,
    #    status: PASS|WARN|FAIL, interpretation}]}

    # ── NEW: Anomaly Deep Dive ─────────────────────────────────────────────────
    anomaly_deep_dive: Dict = field(default_factory=dict)
    # {if_contamination, per_column: [{col, anomaly_count, z_score_max, if_score_mean}]}

    # ── NEW: Regulatory Summary ────────────────────────────────────────────────
    regulatory_summary: Dict = field(default_factory=dict)
    # {domains_checked: [], rules_total, rules_passed, rules_warned, rules_failed,
    #  domain_results: {domain: {status, rules_checked, violations: []}}}

    def to_dict(self, include_df: bool = False) -> Dict:
        out = {
            "run_id": self.run_id,
            "eda_report": self.eda_report,
            "feature_manifest": self.feature_manifest,
            "insights": self.insights,
            "insight_ranking": self.insight_ranking,
            "llm_summary": self.llm_summary,
            "eda_html_report_path": self.eda_html_report_path,
            "actions_log": self.actions_log,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "errors": self.errors,
            "retrain_required": self.retrain_required,
            # New fields:
            "pipeline_stage_log": self.pipeline_stage_log,
            "ingestion_metrics": self.ingestion_metrics,
            "feature_importance": self.feature_importance,
            "governance_summary": self.governance_summary,
            "data_lineage": self.data_lineage,
            "cross_domain_flags": self.cross_domain_flags,
            "statistical_tests": self.statistical_tests,
            "bias_fairness_report": self.bias_fairness_report,
            "anomaly_deep_dive": self.anomaly_deep_dive,
            "regulatory_summary": self.regulatory_summary,
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

        # ── Stage D.5: AutoCorrector (Apply Data Prep) ─────────────────────
        try:
            from preprocessing.auto_corrector import AutoCorrector
            corrector = AutoCorrector(target_col=target_col)
            result.enriched_df, result.actions_log = corrector.apply(working_df, result.eda_report)
        except Exception as exc:
            logger.warning("[AnalyticsOrchestrator] AutoCorrector failed: %s", exc)
            result.actions_log = {}

        # ── Stage D.6: Anomaly Scoring ─────────────────────────────────────
        try:
            from preprocessing.anomaly_scorer import AnomalyScorer
            scorer = AnomalyScorer(config=self.config)
            result.enriched_df, anomaly_report = scorer.score(result.enriched_df, run_id=run_id, target_col=target_col)
            result.eda_report["anomaly_scoring"] = anomaly_report.to_dict()
        except Exception as exc:
            logger.warning("[AnalyticsOrchestrator] AnomalyScorer failed: %s", exc)

        # ── Stage E: Enrich eda_report with histogram bins (for exec report charts) ──
        result.eda_report = self._enrich_eda_with_histograms(result.eda_report, df)

        # ── Stage F: Statistical Significance Tests ────────────────────────────
        try:
            from analytics.advanced_analytics import run_statistical_tests
            result.statistical_tests = run_statistical_tests(working_df, target_col=target_col)
            logger.debug("[AnalyticsOrchestrator] Stage F: stat tests done — %d normality, %d stationarity",
                         len(result.statistical_tests.get("normality", [])),
                         len(result.statistical_tests.get("stationarity", [])))
        except Exception as exc:
            logger.warning("[AnalyticsOrchestrator] Stage F (stat tests) failed (non-fatal): %s", exc)
            result.statistical_tests = {}

        # ── Stage G: Feature Importance ────────────────────────────────────────
        try:
            from analytics.advanced_analytics import compute_feature_importance
            result.feature_importance = compute_feature_importance(working_df, target_col=target_col)
            logger.debug("[AnalyticsOrchestrator] Stage G: feature importance — %d features ranked",
                         len(result.feature_importance))
        except Exception as exc:
            logger.warning("[AnalyticsOrchestrator] Stage G (feature importance) failed (non-fatal): %s", exc)
            result.feature_importance = {}

        # ── Stage H: Bias & Fairness Analysis ──────────────────────────────────
        try:
            from analytics.advanced_analytics import run_bias_fairness_analysis
            result.bias_fairness_report = run_bias_fairness_analysis(working_df, target_col=target_col)
            logger.debug("[AnalyticsOrchestrator] Stage H: bias/fairness done — %d group results",
                         len(result.bias_fairness_report.get("results", [])))
        except Exception as exc:
            logger.warning("[AnalyticsOrchestrator] Stage H (bias/fairness) failed (non-fatal): %s", exc)
            result.bias_fairness_report = {}

        # ── Stage I: Anomaly Deep Dive ─────────────────────────────────────────
        try:
            from analytics.advanced_analytics import run_anomaly_deep_dive
            result.anomaly_deep_dive = run_anomaly_deep_dive(working_df)
            logger.debug("[AnalyticsOrchestrator] Stage I: anomaly deep dive — %d total anomalies, %d columns",
                         result.anomaly_deep_dive.get("total_anomalies", 0),
                         len(result.anomaly_deep_dive.get("per_column", [])))
        except Exception as exc:
            logger.warning("[AnalyticsOrchestrator] Stage I (anomaly deep dive) failed (non-fatal): %s", exc)
            result.anomaly_deep_dive = {}

        # ── Stage J: Auto-Domain Regulatory Summary ────────────────────────────
        try:
            from validation.regulatory.auto_domain_detector import detect_domains
            detected_domains = detect_domains(working_df)
            # Aggregate violations from cross_domain_flags if populated upstream
            rules_total = len(result.cross_domain_flags)
            rules_failed = sum(1 for f in result.cross_domain_flags if f.get("severity") in ("CRITICAL", "ERROR"))
            rules_warned = sum(1 for f in result.cross_domain_flags if f.get("severity") == "WARNING")
            result.regulatory_summary = {
                "domains_checked": detected_domains,
                "rules_total": rules_total,
                "rules_passed": max(0, rules_total - rules_failed - rules_warned),
                "rules_warned": rules_warned,
                "rules_failed": rules_failed,
                "auto_detected": True,
            }
            logger.debug("[AnalyticsOrchestrator] Stage J: regulatory summary — domains=%s", detected_domains)
        except Exception as exc:
            logger.warning("[AnalyticsOrchestrator] Stage J (regulatory summary) failed (non-fatal): %s", exc)
            result.regulatory_summary = {}

        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[AnalyticsOrchestrator][%s] eda=%d insights, fe=%s, llm=%d chars, "
            "stat_tests=%d, fi=%d, anomalies=%d, elapsed=%.0fms",
            (run_id or "")[:8], len(result.insights),
            result.feature_manifest.get("final_shape", "n/a"),
            len(result.llm_summary),
            len(result.statistical_tests.get("normality", [])),
            len(result.feature_importance),
            result.anomaly_deep_dive.get("total_anomalies", 0),
            result.elapsed_ms,
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
            from preprocessing.feature_engineer import FeatureEngineer
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

    def _enrich_eda_with_histograms(
        self,
        eda_report: Dict,
        df: Optional[pd.DataFrame],
    ) -> Dict:
        """
        Stage E — Attach histogram bins to eda_report.numeric_stats so the
        executive report can render Chart.js distribution charts inline.

        No separate file is created — everything goes into the single report.
        """
        if df is None or df.empty or not eda_report:
            return eda_report

        try:
            import numpy as np

            numeric_stats = eda_report.get("numeric_stats", {})
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            for col in num_cols[:16]:  # cap at 16 charts
                series = df[col].dropna()
                if len(series) < 5:
                    continue
                counts, bin_edges = np.histogram(series, bins=18)
                col_stats = numeric_stats.get(col, {})
                if not isinstance(col_stats, dict):
                    col_stats = {}
                col_stats["histogram_bins"]   = [round(float(b), 4) for b in bin_edges[:-1]]
                col_stats["histogram_counts"] = [int(c) for c in counts]
                if "mean" not in col_stats:
                    col_stats["mean"] = round(float(series.mean()), 4)
                if "skewness" not in col_stats and "skew" not in col_stats:
                    col_stats["skew"] = round(float(series.skew()), 3)
                numeric_stats[col] = col_stats

            eda_report["numeric_stats"] = numeric_stats

        except Exception as exc:
            logger.debug("[AnalyticsOrchestrator] Histogram enrichment skipped: %s", exc)

        return eda_report

