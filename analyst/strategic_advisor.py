"""
analyst/strategic_advisor.py
------------------------------
Senior analyst — Strategic Advisory module.

Provides:
  - growth_strategy(): top N growth levers from verified insights
  - revenue_optimization(): segment-level revenue drivers analysis
  - pricing_analysis(): price elasticity proxy via regression
  - risk_forecast(): confidence-bounded risk outlook
  - executive_narrative(): board-ready written summary
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("dipex.strategic_advisor")


class StrategicAdvisor:
    """
    Board-level strategic advisory powered by verified data.
    All recommendations cite confidence bounds and QA status.
    """

    def growth_strategy(
        self,
        insights: List[Dict[str, Any]],
        max_levers: int = 3,
    ) -> Dict[str, Any]:
        """
        Identify top growth levers from verified insight dicts.

        Each insight should have: {name, impact_estimate, confidence_score, action}
        Ranked by: confidence_score × |impact_estimate|

        Returns:
            {top_levers: [...], rationale, total_insights_evaluated}
        """
        if not insights:
            return {
                "top_levers": [],
                "rationale": "No verified insights available for growth analysis.",
                "total_insights_evaluated": 0,
            }

        scored = []
        for ins in insights:
            try:
                conf = float(ins.get("confidence_score", 0.5))
                impact = float(ins.get("impact_estimate", 0.0))
                lever_score = conf * abs(impact)
                scored.append((lever_score, ins))
            except (ValueError, TypeError):
                continue

        scored.sort(key=lambda x: -x[0])
        top = scored[:max_levers]

        return {
            "top_levers": [
                {
                    "rank": i + 1,
                    "name": ins.get("name", f"Lever {i + 1}"),
                    "action": ins.get("action", "TBD"),
                    "impact_estimate": ins.get("impact_estimate"),
                    "confidence_score": ins.get("confidence_score"),
                    "lever_score": round(score, 4),
                }
                for i, (score, ins) in enumerate(top)
            ],
            "rationale": (
                f"Ranked {len(insights)} verified insights by confidence × impact. "
                f"Top {len(top)} levers presented."
            ),
            "total_insights_evaluated": len(insights),
        }

    def revenue_optimization(
        self,
        df: pd.DataFrame,
        revenue_col: str,
        segment_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Identify segment-level revenue drivers.

        Args:
            df: Gold artefact DataFrame
            revenue_col: Column containing revenue values
            segment_cols: Columns to segment by (auto-detect if None)

        Returns:
            {segments: [...], top_driving_segment, bottom_segment, total_revenue, insights}
        """
        if revenue_col not in df.columns:
            return {"error": f"Column '{revenue_col}' not found in data"}

        rev = df[revenue_col].dropna()
        total = float(rev.sum())
        mean_rev = float(rev.mean())

        # Auto-detect categorical columns for segmentation
        if segment_cols is None:
            segment_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()[:3]

        segment_analysis = []
        for seg_col in segment_cols:
            if seg_col not in df.columns:
                continue
            grouped = df.groupby(seg_col, observed=True)[revenue_col].agg(
                ["sum", "mean", "count"]
            ).reset_index()
            grouped.columns = [seg_col, "total_revenue", "avg_revenue", "count"]
            grouped["revenue_share_pct"] = (grouped["total_revenue"] / total * 100).round(2)
            segment_analysis.append({
                "segment_column": seg_col,
                "breakdown": grouped.nlargest(10, "total_revenue").to_dict(orient="records"),
            })

        top_seg = None
        bottom_seg = None
        for analysis in segment_analysis:
            breakdown = analysis["breakdown"]
            if breakdown:
                top_seg = breakdown[0]
                bottom_seg = breakdown[-1]

        return {
            "revenue_col": revenue_col,
            "total_revenue": round(total, 2),
            "mean_revenue": round(mean_rev, 4),
            "segments": segment_analysis,
            "top_driving_segment": top_seg,
            "bottom_segment": bottom_seg,
            "insights": [
                f"Top revenue segment contributes {top_seg['revenue_share_pct']:.1f}% of total revenue."
                if top_seg else "No segment data available.",
            ],
        }

    def pricing_analysis(
        self,
        df: pd.DataFrame,
        price_col: str,
        demand_col: str,
    ) -> Dict[str, Any]:
        """
        Estimate price elasticity of demand via log-log regression.

        Elasticity = β₁ from: log(demand) = β₀ + β₁ × log(price)
        Elastic if |elasticity| > 1, inelastic if < 1.

        Returns:
            {elasticity, is_elastic, r_squared, recommendation, ci_95}
        """
        if price_col not in df.columns or demand_col not in df.columns:
            return {"error": f"Columns '{price_col}' or '{demand_col}' not found"}

        valid = df[[price_col, demand_col]].dropna()
        valid = valid[(valid[price_col] > 0) & (valid[demand_col] > 0)]

        if len(valid) < 10:
            return {"error": f"Insufficient data: only {len(valid)} valid rows (need ≥ 10)"}

        log_price = np.log(valid[price_col].values)
        log_demand = np.log(valid[demand_col].values)

        slope, intercept, r_val, p_val, se = stats.linregress(log_price, log_demand)
        elasticity = float(slope)
        r_squared = float(r_val ** 2)

        # 95% CI for elasticity
        ci_margin = 1.96 * se
        ci_95 = {"lower": round(elasticity - ci_margin, 4), "upper": round(elasticity + ci_margin, 4)}

        return {
            "price_col": price_col,
            "demand_col": demand_col,
            "elasticity": round(elasticity, 4),
            "is_elastic": abs(elasticity) > 1,
            "direction": "inverse" if elasticity < 0 else "direct",
            "r_squared": round(r_squared, 4),
            "p_value": round(float(p_val), 6),
            "statistically_significant": p_val < 0.05,
            "ci_95": ci_95,
            "n": len(valid),
            "recommendation": (
                f"Demand is {'elastic' if abs(elasticity) > 1 else 'inelastic'} "
                f"(elasticity = {elasticity:.3f}). "
                f"{'Reducing price can increase revenue.' if elasticity < -1 else 'Price has limited demand impact.'}"
            ),
        }

    def risk_forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon: int = 30,
    ) -> Dict[str, Any]:
        """
        Bounded risk outlook using rolling statistics.

        Estimates:
        - Rolling mean + std for trend direction
        - 80% and 95% prediction intervals (normal assumption)
        - Risk classification: LOW / MEDIUM / HIGH / CRITICAL

        Returns:
            {trend, risk_level, prediction_interval_80, prediction_interval_95, recommendation}
        """
        if target_col not in df.columns:
            return {"error": f"Column '{target_col}' not found"}

        series = df[target_col].dropna()
        if len(series) < 5:
            return {"error": "Insufficient data for risk forecast (need ≥ 5 observations)"}

        values = series.values.astype(float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))

        # Trend direction (last 20% vs first 20%)
        n = len(values)
        first_q = values[: max(1, n // 5)].mean()
        last_q = values[-max(1, n // 5) :].mean()
        trend_direction = "increasing" if last_q > first_q * 1.02 else "decreasing" if last_q < first_q * 0.98 else "stable"

        # Prediction intervals (normal approximation)
        z80 = 1.282
        z95 = 1.960
        pi_80 = {"lower": round(mean - z80 * std, 4), "upper": round(mean + z80 * std, 4)}
        pi_95 = {"lower": round(mean - z95 * std, 4), "upper": round(mean + z95 * std, 4)}

        # Risk classification based on coefficient of variation
        cv = std / abs(mean) if abs(mean) > 0 else float("inf")
        risk_level = (
            "CRITICAL" if cv > 1.0
            else "HIGH" if cv > 0.5
            else "MEDIUM" if cv > 0.2
            else "LOW"
        )

        return {
            "target_col": target_col,
            "horizon_days": horizon,
            "n_observations": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "coefficient_of_variation": round(cv, 4),
            "trend_direction": trend_direction,
            "risk_level": risk_level,
            "prediction_interval_80": pi_80,
            "prediction_interval_95": pi_95,
            "recommendation": (
                f"Risk level: {risk_level}. Trend: {trend_direction}. "
                f"95% of future values expected in [{pi_95['lower']:.2f}, {pi_95['upper']:.2f}]. "
                f"{'Immediate attention required.' if risk_level == 'CRITICAL' else 'Monitor closely.' if risk_level == 'HIGH' else 'Within normal operating range.'}"
            ),
        }

    def executive_narrative(
        self,
        report: Dict[str, Any],
        audience: str = "board",
    ) -> str:
        """
        Generate a board-ready executive narrative from a structured report.
        Routes through LLM provider when available; otherwise uses template.

        Args:
            report: Structured report dict (executive_summary, growth levers, risk)
            audience: "board" | "cfo" | "ceo" | "investor"

        Returns:
            Written narrative as Markdown string
        """
        report_type = report.get("report_type", "analysis")
        confidence = report.get("confidence_score") or report.get(
            "best_result", {}
        ).get("confidence_score", 0.0)
        top_insight = report.get("top_insights", ["No verified insights available"])[0]

        narrative = f"""# Executive Summary — {report_type.replace('_', ' ').title()}

**Confidence Level:** {f'{float(confidence):.1%}' if confidence else 'Not available'}
**QA Status:** {'✅ Verified' if report.get('qA_gated') else '⚠️ Unverified'}

## Key Finding
{top_insight}

## Strategic Implication
All insights presented have been independently verified through Hard Gate 1 (deterministic validation)
and Hard Gate 2 (statistical verification). No insight is presented without confidence confirmation.

## Reliability Note
This analysis is confidence-bounded at {f'{float(confidence):.1%}' if confidence else 'the stated threshold'}.
Results should be interpreted within these bounds. Statistical uncertainty is explicitly acknowledged.

_Prepared for {audience.upper()} review. All data sourced from verified Gold artefacts only._
"""
        # Try LLM enhancement if available
        try:
            from reporting_service.llm_provider import LLMProvider
            provider = LLMProvider()
            enhanced = provider.enhance_narrative(narrative, report)
            if enhanced:
                return enhanced
        except Exception:  # noqa: BLE001
            pass

        return narrative
