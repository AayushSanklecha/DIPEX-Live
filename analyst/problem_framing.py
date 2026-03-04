"""
analyst/problem_framing.py
----------------------------
Senior analyst — Problem Framing module.

Translates vague business questions into structured, measurable KPI definitions.
Provides:
  - translate_to_kpi(): vague question → structured KPI definition
  - identify_north_star(): list of KPIs → primary metric + supporting metrics
  - design_measurement_framework(): objective → leading + lagging indicators
  - disambiguate_metric(): metric name + df → clarified definition + uniqueness check
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("dipex.problem_framing")


@dataclass
class KPICandidate:
    """Structured KPI definition produced by problem framing."""
    name: str
    formula: str
    measurement_type: str          # "rate", "count", "ratio", "average", "sum"
    direction: str                 # "higher_is_better" | "lower_is_better" | "target_is_better"
    target_value: Optional[float]
    leading_or_lagging: str       # "leading" | "lagging"
    confidence_level: str          # "high" | "medium" | "low"
    notes: str = ""


# Keyword → KPI mapping patterns
_QUESTION_PATTERNS: List[Tuple[str, str, str, str]] = [
    (r"churn|attrition|leaving", "Churn Rate", "churn_events / total_customers", "lower_is_better"),
    (r"revenue|sales|income", "Total Revenue", "SUM(revenue)", "higher_is_better"),
    (r"conversion|convert|signup", "Conversion Rate", "conversions / total_visitors", "higher_is_better"),
    (r"retention|loyal|keep", "Retention Rate", "retained_customers / total_customers", "higher_is_better"),
    (r"cost|spend|expense", "Cost Efficiency", "revenue / total_cost", "higher_is_better"),
    (r"engagement|active|usage", "Engagement Rate", "active_users / total_users", "higher_is_better"),
    (r"quality|error|defect|complaint", "Quality Score", "1 - error_rate", "higher_is_better"),
    (r"satisfaction|nps|rating|happy", "Customer Satisfaction Score", "AVG(satisfaction_rating)", "higher_is_better"),
    (r"growth|grow|expand", "Growth Rate", "(current - previous) / previous", "higher_is_better"),
    (r"risk|fraud|anomal", "Risk Score", "anomaly_count / total_transactions", "lower_is_better"),
    (r"latency|speed|time|fast", "Response Time P95", "PERCENTILE(latency, 0.95)", "lower_is_better"),
    (r"profit|margin", "Profit Margin", "profit / revenue", "higher_is_better"),
]


class ProblemFraming:
    """
    Senior analyst cognitive module for translating business questions
    into measurable, actionable KPI frameworks.
    """

    def translate_to_kpi(
        self,
        vague_question: str,
        domain: str = "general",
    ) -> List[KPICandidate]:
        """
        Translate a vague business question into structured KPI candidate(s).

        Args:
            vague_question: e.g. "Why are customers leaving?"
            domain: Domain context ("banking", "healthcare", "finance", "general")

        Returns:
            List of KPI candidates (best match first)
        """
        question_lower = vague_question.lower()
        matched: List[KPICandidate] = []

        for pattern, name, formula, direction in _QUESTION_PATTERNS:
            if re.search(pattern, question_lower):
                matched.append(KPICandidate(
                    name=name,
                    formula=formula,
                    measurement_type=self._infer_measurement_type(formula),
                    direction=direction,
                    target_value=None,
                    leading_or_lagging=self._infer_leading_lagging(name),
                    confidence_level="medium",
                    notes=f"Auto-identified from question: '{vague_question[:80]}...' (domain={domain})",
                ))

        if not matched:
            # Fallback: generic KPI
            matched.append(KPICandidate(
                name="Custom Business Metric",
                formula="DEFINE_FORMULA",
                measurement_type="count",
                direction="higher_is_better",
                target_value=None,
                leading_or_lagging="lagging",
                confidence_level="low",
                notes=f"No pattern matched for: '{vague_question[:80]}'. Manual definition required.",
            ))

        logger.info("ProblemFraming: translated question to %d KPI candidates", len(matched))
        return matched

    def identify_north_star(
        self,
        kpi_list: List[KPICandidate],
    ) -> Dict[str, Any]:
        """
        Identify the primary (North Star) metric from a list of KPI candidates.

        North Star selection heuristics:
        1. Prefer "lagging" indicators (direct business outcome)
        2. Prefer "higher_is_better" direction
        3. Prefer "high" confidence_level
        4. Business outcome KPIs > operational KPIs

        Returns:
            {north_star: KPI, supporting_metrics: [KPIs], rationale: str}
        """
        if not kpi_list:
            return {"north_star": None, "supporting_metrics": [], "rationale": "No KPIs provided."}

        # Score each KPI
        scored = []
        for kpi in kpi_list:
            score = 0
            if kpi.leading_or_lagging == "lagging":
                score += 3
            if kpi.confidence_level == "high":
                score += 2
            elif kpi.confidence_level == "medium":
                score += 1
            if kpi.direction == "higher_is_better":
                score += 1
            if any(kw in kpi.name.lower() for kw in ["revenue", "profit", "retention", "satisfaction"]):
                score += 2
            scored.append((score, kpi))

        scored.sort(key=lambda x: -x[0])
        north_star = scored[0][1]
        supporting = [kpi for _, kpi in scored[1:]]

        return {
            "north_star": {
                "name": north_star.name,
                "formula": north_star.formula,
                "direction": north_star.direction,
            },
            "supporting_metrics": [
                {"name": k.name, "formula": k.formula} for k in supporting[:4]
            ],
            "rationale": (
                f"Selected '{north_star.name}' as North Star metric: "
                f"it is a {north_star.leading_or_lagging} indicator with "
                f"{north_star.confidence_level} confidence. "
                f"Formula: {north_star.formula}"
            ),
        }

    def design_measurement_framework(
        self,
        objective: str,
        domain: str = "general",
    ) -> Dict[str, Any]:
        """
        Design a structured measurement framework for a business objective.

        Returns:
            {objective, leading_indicators: [...], lagging_indicators: [...],
             guardrail_metrics: [...], measurement_cadence, review_trigger}
        """
        kpis = self.translate_to_kpi(objective, domain=domain)
        leading = [k for k in kpis if k.leading_or_lagging == "leading"]
        lagging = [k for k in kpis if k.leading_or_lagging == "lagging"]

        return {
            "objective": objective,
            "domain": domain,
            "leading_indicators": [
                {"name": k.name, "formula": k.formula, "direction": k.direction}
                for k in leading
            ],
            "lagging_indicators": [
                {"name": k.name, "formula": k.formula, "direction": k.direction}
                for k in lagging
            ],
            "guardrail_metrics": [
                {"name": "Data Quality Score", "threshold": ">= 0.90"},
                {"name": "Confidence Score", "threshold": ">= 0.70"},
            ],
            "measurement_cadence": "weekly",
            "review_trigger": f"If {kpis[0].name if kpis else 'primary KPI'} drops > 10% week-over-week",
        }

    def disambiguate_metric(
        self,
        metric_name: str,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Clarify metric definition using the actual dataset.
        Checks: uniqueness, column existence, naming conflicts, type validation.

        Args:
            metric_name: Column name or derived metric name
            df: Gold artefact DataFrame

        Returns:
            Disambiguation report dict
        """
        if metric_name not in df.columns:
            return {
                "metric_name": metric_name,
                "found_in_data": False,
                "similar_columns": [
                    c for c in df.columns if metric_name.lower() in c.lower()
                ][:5],
                "recommendation": f"Column '{metric_name}' not found. Check similar columns above.",
            }

        series = df[metric_name]
        return {
            "metric_name": metric_name,
            "found_in_data": True,
            "dtype": str(series.dtype),
            "is_unique": bool(series.nunique() == len(series)),
            "null_pct": round(float(series.isna().mean()) * 100, 2),
            "sample_values": series.dropna().head(5).tolist(),
            "cardinality": int(series.nunique()),
            "recommendation": (
                "Metric is well-defined and unique." if series.nunique() == len(series)
                else "Metric is NOT unique — cannot be used as a primary key or ID."
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_measurement_type(formula: str) -> str:
        formula_lower = formula.lower()
        if "sum(" in formula_lower:
            return "sum"
        if "avg(" in formula_lower or "mean" in formula_lower:
            return "average"
        if "/" in formula:
            return "ratio"
        if "count" in formula_lower:
            return "count"
        return "rate"

    @staticmethod
    def _infer_leading_lagging(name: str) -> str:
        leading_keywords = ["engagement", "latency", "quality", "risk", "conversion"]
        lagging_keywords = ["revenue", "profit", "churn", "retention", "satisfaction", "growth"]
        name_lower = name.lower()
        for kw in leading_keywords:
            if kw in name_lower:
                return "leading"
        for kw in lagging_keywords:
            if kw in name_lower:
                return "lagging"
        return "lagging"


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY SHIMS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProposedKPI:
    name: str
    formula: str
    direction: str
    leading_or_lagging: str = "lagging"


@dataclass
class FramedProblem:
    """Legacy API: returned by ProblemFramingEngine.frame()."""
    detected_intent: str
    vague_question: str
    kpi_proposals: List[KPICandidate]
    measurement_framework: Dict[str, Any]
    available_columns: List[str]
    dataset_id: str
    north_star: Optional[str] = None
    ambiguity_score: float = 0.0
    clarification_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_intent": self.detected_intent,
            "vague_question": self.vague_question,
            "kpi_count": len(self.kpi_proposals),
            "measurement_framework": self.measurement_framework,
            "dataset_id": self.dataset_id,
            "north_star": self.north_star,
            "ambiguity_score": self.ambiguity_score,
            "clarification_questions": self.clarification_questions,
        }


class ProblemFramingEngine(ProblemFraming):
    """
    Backward-compat alias for ProblemFraming with legacy frame() API
    expected by AnalystOrchestrator and tests.
    """

    # Ambiguity signals: phrases that indicate vagueness
    _VAGUE_PHRASES = [
        "things", "better", "improve", "metrics", "stuff", "help me",
        "understand", "look at", "check", "see if", "figure out",
    ]
    _CLARIFICATION_TEMPLATES = [
        "Which specific metric should be tracked?",
        "What time period should the analysis cover?",
        "Which customer segment is in scope?",
        "What would a successful outcome look like?",
        "Are there known confounding factors to control for?",
    ]

    def frame(
        self,
        question: str,
        available_columns: Optional[List[str]] = None,
        dataset_id: str = "",
        domain: str = "general",
    ) -> FramedProblem:
        """Wraps translate_to_kpi + design_measurement_framework."""
        kpis = self.translate_to_kpi(question, domain=domain)
        framework = self.design_measurement_framework(question, domain=domain)

        # Intent detection — map to a keyword that reflects the intent clearly
        intent_map = [
            (r"churn|attrition|leaving|churning", "churn"),
            (r"revenue|sales|income|revenue", "revenue"),
            (r"risk|fraud|anomal", "risk"),
            (r"growth|grow|expand|user base", "growth"),
            (r"satisfaction|nps|rating|happy", "satisfaction"),
            (r"conversion|convert|signup", "conversion"),
            (r"retention|loyal|keep", "retention"),
            (r"cost|spend|expense", "cost"),
            (r"engagement|active|usage", "engagement"),
            (r"quality|error|defect", "quality"),
        ]
        import re as _re
        detected_intent = "measurement"
        for pattern, intent_keyword in intent_map:
            if _re.search(pattern, question, _re.IGNORECASE):
                detected_intent = intent_keyword
                break

        # North star: the primary KPI from the proposals, or first framework KPI
        north_star: Optional[str] = None
        if kpis:
            north_star = kpis[0].name
        elif framework.get("north_star_metric"):
            north_star = framework["north_star_metric"]

        # Ambiguity score: [0,1] — higher = more vague
        q_lower = question.lower()
        q_words = set(q_lower.split())
        vague_hits = sum(1 for p in self._VAGUE_PHRASES if p in q_lower)
        # Word count penalty (short questions are more ambiguous)
        word_count = len(question.split())
        brevity_penalty = max(0.0, 1.0 - word_count / 12.0)
        ambiguity_score = round(
            min(1.0, (vague_hits * 0.25) + brevity_penalty * 0.5), 3
        )

        # Clarification questions: generated when question is vague
        clarification_questions: List[str] = []
        if ambiguity_score > 0.2 or not kpis:
            # Always emit at least N=2 questions for vague queries
            clarification_questions = self._CLARIFICATION_TEMPLATES[:max(2, vague_hits + 1)]

        return FramedProblem(
            detected_intent=detected_intent,
            vague_question=question,
            kpi_proposals=kpis,
            measurement_framework=framework,
            available_columns=available_columns or [],
            dataset_id=dataset_id,
            north_star=north_star,
            ambiguity_score=ambiguity_score,
            clarification_questions=clarification_questions,
        )
