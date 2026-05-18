"""
reporting_service/insight_narrator.py
---------------------------------------
Deterministic InsightNarrator — transforms raw pipeline statistics into
plain-language reasoning, interpretations, and actionable guidance.

No LLM required. Every interpretation is computed from the actual statistics
using domain-knowledge rules. Works even when HF_API_KEY is absent.

Sub-narrators:
  ColumnInterpreter      — shape, skew, zero-values, nulls, preprocessing advice
  CorrelationNarrator    — business implication + multicollinearity warnings
  MissingValueAdvisor    — MCAR/MAR/MNAR pattern hints + imputation strategy
  OutlierExplainer       — what IQR/Z-score outliers likely represent
  AnomalyExplainer       — isolation-forest anomaly rate interpretation
  DataQualityReasoner    — plain-language per-flag explanations
  ModelMetricInterpreter — ROC-AUC / F1 / accuracy in practical terms
  NarrativeBuilder       — full 300-500 word structured summary narrative
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dipex.reporting.insight_narrator")


# ── Distribution shape detection ──────────────────────────────────────────────

def _detect_shape(skew: float, kurtosis: float = 0.0) -> Tuple[str, str]:
    """
    Returns (shape_name, explanation) based on skewness and kurtosis values.
    """
    abs_skew = abs(skew)
    if abs_skew < 0.3:
        if abs(kurtosis) < 1.5:
            return "Normal", "approximately normally distributed — symmetric bell curve"
        elif kurtosis < -1.5:
            return "Uniform-like", "flatter than normal — values spread evenly with few extremes"
        else:
            return "Leptokurtic", "heavier tails than normal — extreme values are more common"
    elif abs_skew < 1.0:
        if skew > 0:
            return "Mildly right-skewed", "slightly more values below the mean, with a moderate upper tail"
        else:
            return "Mildly left-skewed", "slightly more values above the mean, with a moderate lower tail"
    elif abs_skew < 2.0:
        if skew > 0:
            return "Right-skewed (positive)", "most values cluster at the low end with a long tail of high values"
        else:
            return "Left-skewed (negative)", "most values cluster at the high end with a long tail of low values"
    else:
        if skew > 0:
            return "Strongly right-skewed", "extreme positive tail — a few very high values dominate"
        else:
            return "Strongly left-skewed", "extreme negative tail — a few very low values drive the distribution"


def _preprocessing_advice(shape: str, null_pct: float, zero_pct: float) -> str:
    """Return preprocessing recommendations based on detected shape."""
    advice = []
    if "right-skew" in shape.lower():
        advice.append("consider log-transform or square-root transform before linear modelling")
    if "left-skew" in shape.lower():
        advice.append("consider square or exponential transform if this is a target variable")
    if null_pct > 0.1:
        advice.append(f"investigate {null_pct:.0%} missing rate — prefer multiple imputation over mean-fill")
    elif null_pct > 0.01:
        advice.append("low null rate — median imputation is acceptable")
    if zero_pct > 0.05:
        advice.append(f"{zero_pct:.0%} zeros — distinguish true zeros from missing-coded-as-zero before modelling")
    if not advice:
        return "No preprocessing concerns identified for this column."
    return "Recommendation: " + "; ".join(advice) + "."


# ── Column Interpreter ─────────────────────────────────────────────────────────

class ColumnInterpreter:
    """
    Generates per-column plain-language interpretation from numeric stats.
    """

    def interpret(
        self,
        col: str,
        stats: Dict[str, Any],
        null_pct: float = 0.0,
        zero_pct: float = 0.0,
        actions_log: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, str]:
        """
        Returns dict with keys: shape, meaning, recommendation, flag (optional).
        """
        skew = float(stats.get("skewness", stats.get("skew", 0.0)) or 0.0)
        kurt = float(stats.get("kurtosis", 0.0) or 0.0)
        mean = float(stats.get("mean", 0.0) or 0.0)
        std  = float(stats.get("std",  0.0) or 0.0)
        cv   = (std / mean) if mean != 0 else 0.0

        shape_name, shape_desc = _detect_shape(skew, kurt)

        # Build meaning sentence
        if abs(skew) < 0.3:
            meaning = (
                f"'{col}' is {shape_desc}. "
                f"With mean={mean:.2f} and std={std:.2f}, values are consistent and predictable. "
                "Linear models will handle this column well without transformation."
            )
        elif skew > 1.0:
            meaning = (
                f"'{col}' is {shape_desc} (skew={skew:.2f}). "
                f"Most records have values near the lower range, but a subset of high values pulls the mean up. "
                "This is common in revenue, spend, and count data — it means the average overstates 'typical'."
            )
        elif skew < -1.0:
            meaning = (
                f"'{col}' is {shape_desc} (skew={skew:.2f}). "
                f"Most records are concentrated near the upper end. "
                "This often appears in percentage or rate columns approaching a ceiling (e.g. satisfaction scores)."
            )
        else:
            meaning = (
                f"'{col}' is {shape_desc} (skew={skew:.2f}, mean={mean:.2f}). "
                "The distribution has a moderate directional lean — mild transformation may improve model fit."
            )

        # High coefficient of variation warning
        if cv > 1.0:
            meaning += (
                f" High variability (CV={cv:.2f}) suggests this column spans a wide range — "
                "consider normalisation or scaling."
            )

        if actions_log and col in actions_log:
            action_desc = actions_log[col].get("action", "")
            reason_desc = actions_log[col].get("reason", "")
            recommendation = f"✅ Applied: {action_desc} ({reason_desc})"
        else:
            recommendation = _preprocessing_advice(shape_name, null_pct, zero_pct)

        flag = None
        if null_pct > 0.20 and (not actions_log or col not in actions_log):
            flag = f"⚠️ {null_pct:.0%} missing — needs manual review"
        elif zero_pct > 0.30 and (not actions_log or col not in actions_log):
            flag = f"⚠️ {zero_pct:.0%} zeros — verify whether these are true zeros"

        result = {"shape": shape_name, "meaning": meaning, "recommendation": recommendation}
        if flag:
            result["flag"] = flag
        return result


# ── Correlation Narrator ───────────────────────────────────────────────────────

class CorrelationNarrator:
    """Generates business-level explanations for feature correlation pairs."""

    def narrate(self, col_a: str, col_b: str, r: float) -> str:
        abs_r = abs(r)
        direction = "positive" if r > 0 else "negative"
        if abs_r > 0.7:
            strength = "strongly"
            implication = (
                f"'{col_a}' and '{col_b}' move {direction}ly together (r={r:.2f}). "
                f"This is a strong linear relationship. "
            )
            if r > 0:
                implication += (
                    f"When {col_a} increases, {col_b} tends to increase proportionally. "
                )
            else:
                implication += (
                    f"When {col_a} increases, {col_b} tends to decrease. "
                )
            implication += (
                "⚠️ Multicollinearity risk: including both in a linear model will inflate "
                "coefficient variance and may produce unstable estimates. "
                "Consider removing one, combining them (ratio/difference), or using regularisation (Ridge/Lasso)."
            )
        elif abs_r > 0.4:
            strength = "moderately"
            implication = (
                f"'{col_a}' and '{col_b}' are moderately {direction}ly correlated (r={r:.2f}). "
                "There is a meaningful but imperfect linear relationship — "
                "both features likely provide some overlapping information. "
                "Keep both unless feature count is a concern; a tree-based model will handle the overlap naturally."
            )
        else:
            strength = "weakly"
            implication = (
                f"'{col_a}' and '{col_b}' are weakly correlated (r={r:.2f}). "
                "No meaningful linear relationship detected. "
                "These features are independent sources of information — safe to include both in any model."
            )
        return implication

    def multicollinearity_warning(self, pairs: List[Dict[str, Any]]) -> Optional[str]:
        """Returns a summary multicollinearity warning if any strong correlations exist."""
        strong = [p for p in pairs if abs(float(p.get("r", p.get("correlation", 0)))) > 0.7]
        if not strong:
            return None
        cols = set()
        for p in strong:
            cols.add(p.get("a", p.get("col_a", "")))
            cols.add(p.get("b", p.get("col_b", "")))
        return (
            f"⚠️ {len(strong)} strongly correlated pair(s) detected involving: {', '.join(cols)}. "
            "This may cause unstable coefficients in linear/logistic regression. "
            "XGBoost/RandomForest are naturally robust to multicollinearity."
        )


# ── Missing Value Advisor ──────────────────────────────────────────────────────

class MissingValueAdvisor:
    """
    Advises on missingness type (MCAR/MAR/MNAR) and imputation strategy
    based on null percentage alone — without access to raw data.
    """

    def advise(self, col: str, null_pct: float) -> str:
        if null_pct < 0.01:
            return (
                f"'{col}' has <1% missing values. "
                "Likely random missingness (MCAR) — safe to drop or impute with median/mode."
            )
        elif null_pct < 0.05:
            return (
                f"'{col}' has {null_pct:.1%} missing values. "
                "Low missingness — likely MCAR. Median imputation is acceptable. "
                "No significant impact to model performance expected."
            )
        elif null_pct < 0.15:
            return (
                f"'{col}' has {null_pct:.1%} missing values. "
                "Moderate missingness — investigate whether it is MAR (Missing At Random): "
                "does the column tend to be null for certain customer segments or time periods? "
                "If so, use multiple imputation (MICE) or a missingness indicator feature."
            )
        elif null_pct < 0.40:
            return (
                f"'{col}' has {null_pct:.1%} missing values. "
                "High missingness — this column may be MNAR (Missing Not At Random), "
                "meaning the fact of it being missing carries information itself. "
                "Consider: (1) adding a binary 'was_{col}_missing' flag feature, "
                "(2) multiple imputation with auxiliary variables, "
                "(3) excluding from modelling and treating as a separate signal."
            )
        else:
            return (
                f"'{col}' has {null_pct:.1%} missing values — very high. "
                "This column has limited analytical value in its current form. "
                "Do not impute mechanically. Investigate the data collection process. "
                "Consider dropping from the feature set unless domain knowledge suggests the absence is itself meaningful."
            )


# ── Outlier Explainer ──────────────────────────────────────────────────────────

class OutlierExplainer:
    """Interprets what detected outliers are likely to represent."""

    def explain(self, col: str, count: int, pct: float, method: str) -> str:
        method_desc = {
            "IQR":    "values outside 1.5× the interquartile range",
            "Z-Score": "values more than 3 standard deviations from the mean",
            "Isolation Forest": "rows anomalous across multiple features simultaneously",
        }.get(method, f"values flagged by {method}")

        if pct < 0.02:
            risk_level = "low"
            interpretation = (
                f"Only {count} rows ({pct:.1%}) flagged — very few outliers. "
                "These are likely genuine exceptional cases or minor data entry errors. "
                "Safe to winsorise at the 1st/99th percentile if desired."
            )
        elif pct < 0.08:
            risk_level = "moderate"
            interpretation = (
                f"{count} rows ({pct:.1%}) are outliers in '{col}' ({method_desc}). "
                "This is a notable proportion. Possible causes: "
                "(1) batch errors in data ingestion, "
                "(2) a genuine bimodal population (e.g. different customer segments), "
                "(3) seasonal spikes or one-off events. "
                "Investigate before removing — do these rows belong to a systematic group?"
            )
        else:
            risk_level = "high"
            interpretation = (
                f"High outlier rate in '{col}': {count} rows ({pct:.1%}) flagged ({method_desc}). "
                "This suggests a structural issue — possibly: "
                "(1) two distinct subpopulations mixed in one column, "
                "(2) systematic measurement errors, or "
                "(3) a non-IID data issue. "
                "⚠️ Do NOT simply drop these rows — investigate root cause first."
            )
        return interpretation


# ── Anomaly Explainer ──────────────────────────────────────────────────────────

class AnomalyExplainer:
    """Interprets the dataset-level Isolation Forest anomaly rate."""

    def explain(self, anomaly_pct: float, n_rows: int) -> str:
        n_anomalies = int(anomaly_pct * n_rows)
        if anomaly_pct < 0.02:
            return (
                f"Only {n_anomalies} rows ({anomaly_pct:.1%}) are flagged as multi-variate anomalies "
                "by the Isolation Forest detector. This is a very low rate — consistent with genuine "
                "exceptional cases (high-value customers, edge transactions). "
                "These rows are unlikely to harm model training at this volume."
            )
        elif anomaly_pct < 0.06:
            return (
                f"{n_anomalies} rows ({anomaly_pct:.1%}) show anomalous patterns across multiple features simultaneously "
                "(Isolation Forest). This moderate rate may indicate: "
                "(1) a small cluster of genuinely unusual records (e.g. VIP accounts, test data), or "
                "(2) early signs of data collection drift. "
                "Recommended: review a sample of flagged rows before model training. "
                "Rows with anomaly_flag=−1 in the cleaned dataset are the ones to inspect."
            )
        else:
            return (
                f"⚠️ High anomaly rate: {n_anomalies} rows ({anomaly_pct:.1%}) are flagged as multi-variate outliers. "
                "At this rate, there are likely: "
                "(1) systematic data quality issues (schema changes, ETL bugs), "
                "(2) multiple distinct sub-populations mixed in the dataset, or "
                "(3) significant data drift from a previous collection period. "
                "Training a model on data with this level of anomalies risks learning spurious patterns. "
                "Prioritise root-cause investigation before proceeding to modelling."
            )


# ── Data Quality Reasoner ──────────────────────────────────────────────────────

_FLAG_EXPLANATIONS: Dict[str, str] = {
    "HIGH_NULL":
        "A column has an unacceptably high proportion of missing values. "
        "This reduces statistical power and can bias models if missingness is not random.",
    "DRIFT":
        "The feature distribution has shifted compared to a reference dataset. "
        "This is a leading indicator that a deployed model will degrade in performance — "
        "also known as covariate shift.",
    "ZERO_VALUE":
        "An unusual proportion of values are exactly zero. "
        "This may represent missing data coded as zero, or a genuine structural zero (e.g. no purchases in a period). "
        "The distinction critically affects which models and transformations are valid.",
    "SCHEMA_MISMATCH":
        "The incoming dataset's schema does not match the expected schema. "
        "Columns may be renamed, retyped, or removed — downstream pipelines may break silently.",
    "DUPLICATE_ROWS":
        "Duplicate records inflate sample sizes and bias training towards over-represented cases. "
        "This can cause models to memorise duplicated patterns rather than generalise.",
    "DATE_FUTURE":
        "Records contain dates in the future, which likely represent data entry errors "
        "or misformatted timestamps. Future dates in event-time data cause target leakage in time-series models.",
    "CARDINALITY":
        "A categorical column has extremely high cardinality relative to sample size. "
        "This causes sparse one-hot encodings and risks overfitting to rare categories.",
    "CONSTANT":
        "A column has zero variance — all values are identical. "
        "Constant columns add no information to any model and should be dropped.",
    "ANOMALY_RATE":
        "More than 5% of rows are flagged as multi-variate anomalies by the Isolation Forest detector. "
        "This rate is high enough to potentially compromise model training quality.",
}

class DataQualityReasoner:
    """Translates validation flag names into plain-language explanations."""

    def explain_flag(self, flag_category: str, message: str = "") -> str:
        key = flag_category.upper().replace(" ", "_")
        # Try exact match, then prefix match
        explanation = _FLAG_EXPLANATIONS.get(key)
        if not explanation:
            for k, v in _FLAG_EXPLANATIONS.items():
                if key.startswith(k) or k.startswith(key):
                    explanation = v
                    break
        if not explanation:
            explanation = "A data quality issue was detected that may affect model accuracy or pipeline reliability."
        return explanation

    def remediation(self, flag_category: str, severity: str) -> str:
        key = flag_category.upper().replace(" ", "_")
        remediations = {
            "HIGH_NULL":       "Investigate source system. Apply multiple imputation or add a 'was_missing' indicator feature.",
            "DRIFT":           "Retrain or recalibrate any deployed models. Alert data engineering to pipeline changes.",
            "ZERO_VALUE":      "Audit data collection logic. Treat suspicious zeros as missing before imputation.",
            "SCHEMA_MISMATCH": "Halt pipeline. Fix schema alignment between source and expected contract.",
            "DUPLICATE_ROWS":  "Deduplicate on a natural key (customer_id + date). Investigate upstream deduplication logic.",
            "DATE_FUTURE":     "Filter or cap dates at today's date. Investigate ETL timezone configuration.",
            "CARDINALITY":     "Use target encoding or embedding for this feature instead of one-hot encoding.",
            "CONSTANT":        "Drop this column from the feature set.",
            "ANOMALY_RATE":    "Review anomalous rows manually. Retrain the anomaly detector if this rate is expected.",
        }
        action = remediations.get(key, "Review this flag with the data engineering team.")
        prefix = "🔴 Immediate action required: " if severity == "CRITICAL" else "🟡 Recommended: "
        return prefix + action


# ── Model Metric Interpreter ───────────────────────────────────────────────────

class ModelMetricInterpreter:
    """Translates model evaluation metrics into practical business language."""

    def interpret_roc_auc(self, auc: float) -> str:
        if auc >= 0.95:
            tier = "exceptional"
            detail = (
                "The model separates positives from negatives with near-perfect discrimination. "
                "Validate that this is not due to target leakage (a feature derived from or correlated "
                "with the label in a non-causal way)."
            )
        elif auc >= 0.90:
            tier = "excellent"
            detail = (
                "The model correctly ranks a randomly chosen positive above a negative 90%+ of the time. "
                "Ready for deployment in most business applications."
            )
        elif auc >= 0.80:
            tier = "good"
            detail = (
                "The model has strong discriminative power. "
                "Performance gap from a naive baseline is meaningful. "
                "Further feature engineering or hyperparameter tuning may push this higher."
            )
        elif auc >= 0.70:
            tier = "fair"
            detail = (
                "The model performs above chance but leaves significant room for improvement. "
                "Consider: richer features, class balance correction, or a more expressive model family."
            )
        else:
            tier = "poor"
            detail = (
                "The model has limited discriminative ability. "
                "Investigate whether the target variable is correctly defined, "
                "the feature set is informative, and the training data is not corrupted."
            )
        return f"ROC-AUC = {auc:.4f} ({tier}). {detail}"

    def interpret_f1(self, f1: float) -> str:
        if f1 >= 0.90:
            return f"F1-Score = {f1:.4f} — excellent precision-recall balance."
        elif f1 >= 0.75:
            return f"F1-Score = {f1:.4f} — good balance between precision and recall. Suitable for most use cases."
        elif f1 >= 0.60:
            return (
                f"F1-Score = {f1:.4f} — moderate. Consider whether false positives or false negatives "
                "are more costly in this domain, and tune the classification threshold accordingly."
            )
        else:
            return (
                f"F1-Score = {f1:.4f} — low. The model is struggling with either precision or recall (or both). "
                "Check for class imbalance; consider SMOTE, class-weight adjustment, or a different model."
            )

    def interpret_accuracy(self, acc: float, is_imbalanced: bool = False) -> str:
        suffix = ""
        if is_imbalanced:
            suffix = (
                " Note: accuracy can be misleading on imbalanced datasets — "
                "a model predicting the majority class always achieves high accuracy. "
                "Rely on ROC-AUC and F1 for imbalanced problems."
            )
        if acc >= 0.90:
            return f"Accuracy = {acc:.1%} — high.{suffix}"
        elif acc >= 0.75:
            return f"Accuracy = {acc:.1%} — acceptable.{suffix}"
        else:
            return f"Accuracy = {acc:.1%} — below acceptable threshold for most production systems.{suffix}"

    def interpret_all(self, metrics: Dict[str, Any]) -> List[str]:
        """Interpret any model metrics dict, returning a list of explanation strings."""
        explanations = []
        for key, val in metrics.items():
            k = key.lower().replace("-", "_").replace(" ", "_")
            try:
                v = float(val)
            except (TypeError, ValueError):
                explanations.append(f"**{key}**: {val}")
                continue
            if "roc" in k or "auc" in k:
                explanations.append(self.interpret_roc_auc(v))
            elif "f1" in k:
                explanations.append(self.interpret_f1(v))
            elif "accuracy" in k or "acc" == k:
                explanations.append(self.interpret_accuracy(v))
            elif "precision" in k:
                explanations.append(
                    f"Precision = {v:.4f} — of all rows predicted positive, {v:.0%} were actually positive."
                )
            elif "recall" in k or "sensitivity" in k:
                explanations.append(
                    f"Recall = {v:.4f} — the model caught {v:.0%} of all actual positives."
                )
            else:
                explanations.append(f"**{key}**: {val}")
        return explanations


# ── NarrativeBuilder ──────────────────────────────────────────────────────────

class NarrativeBuilder:
    """
    Assembles a full structured 300-500 word narrative from all pipeline outputs.
    Provides a rich, human-readable report section regardless of whether an LLM is available.
    """

    def __init__(self) -> None:
        self._col_interp   = ColumnInterpreter()
        self._corr_narr    = CorrelationNarrator()
        self._miss_adv     = MissingValueAdvisor()
        self._outlier_exp  = OutlierExplainer()
        self._anomaly_exp  = AnomalyExplainer()
        self._dq_reasoner  = DataQualityReasoner()
        self._model_interp = ModelMetricInterpreter()

    def build(
        self,
        eda_report: Optional[Dict[str, Any]] = None,
        model_metrics: Optional[Dict[str, Any]] = None,
        analyst_flags: Optional[List[Dict]] = None,
        risk_flags: Optional[List[Dict]] = None,
        gate1_decision: str = "PASS",
        gate2_decision: str = "PASS",
        confidence_score: float = 0.0,
        col_count: int = 0,
        row_count: int = 0,
        run_id: str = "",
        actions_log: Optional[Dict[str, Dict[str, str]]] = None,
        user_context: Optional[str] = None,
    ) -> str:
        eda = eda_report or {}
        summary = eda.get("summary", {})
        n_rows = int(summary.get("n_rows", row_count))
        n_cols = int(summary.get("n_cols", col_count))
        null_pct = float(summary.get("overall_null_pct", 0.0))
        anomaly_pct = float(summary.get("anomaly_pct", 0.0))
        numeric_stats = eda.get("numeric_stats", {})
        correlations = eda.get("correlations", [])
        outliers = eda.get("outliers", {})
        insights = eda.get("insights", [])

        gate_ok = gate1_decision == "PASS" and gate2_decision == "PASS"
        sections: List[str] = []

        # ── Section 1: Dataset Overview ──────────────────────────────────────
        gate_str = "✅ All validation gates PASSED" if gate_ok else "❌ One or more validation gates FAILED"
        dq_status = (
            "The dataset is clean and suitable for downstream analysis."
            if null_pct < 0.05 and (analyst_flags or []) == []
            else "Several data quality concerns were identified — see details below."
        )
        sections.append(
            f"## Dataset Overview\n\n"
            f"This pipeline processed **{n_rows:,} rows × {n_cols} columns**. "
            f"{gate_str}. Confidence score: **{confidence_score:.1%}**. "
            f"Overall null rate: **{null_pct:.1%}**. "
            f"Isolation Forest anomaly rate: **{anomaly_pct:.1%}** ({int(anomaly_pct * n_rows)} rows). "
            f"{dq_status}"
        )

        # ── Section 2: Key EDA Findings ──────────────────────────────────────
        if insights:
            findings_text = "\n".join(f"- {ins}" for ins in insights[:5])
            sections.append(f"## Key EDA Findings\n\n{findings_text}")

        # Interpret top 3 numeric columns
        col_interpretations = []
        for col, stats in list(numeric_stats.items())[:3]:
            if not isinstance(stats, dict):
                continue
            null_c = float(eda.get("missing_values", {}).get(col, {}).get("null_pct", 0.0))
            interp = self._col_interp.interpret(col, stats, null_pct=null_c, actions_log=actions_log)
            col_interpretations.append(f"- **{col}**: {interp['meaning']}")
        if col_interpretations:
            sections.append(
                "## Column Interpretations\n\n" + "\n".join(col_interpretations)
            )

        # ── Section 3: Correlation Analysis ──────────────────────────────────
        if correlations:
            top_corr = sorted(
                correlations,
                key=lambda x: abs(float(x.get("correlation", x.get("r", 0)))),
                reverse=True,
            )[:3]
            corr_texts = []
            for c in top_corr:
                a = c.get("col_a", c.get("a", ""))
                b = c.get("col_b", c.get("b", ""))
                r = float(c.get("correlation", c.get("r", 0)))
                corr_texts.append(f"- {self._corr_narr.narrate(a, b, r)}")
            mc_warn = self._corr_narr.multicollinearity_warning(
                [{"a": c.get("col_a", ""), "b": c.get("col_b", ""), "r": float(c.get("correlation", 0))}
                 for c in correlations]
            )
            corr_section = "## Correlation Analysis\n\n" + "\n".join(corr_texts)
            if mc_warn:
                corr_section += f"\n\n{mc_warn}"
            sections.append(corr_section)

        # ── Section 4: Anomaly Analysis ───────────────────────────────────────
        if anomaly_pct > 0:
            sections.append(
                "## Anomaly Analysis\n\n"
                + self._anomaly_exp.explain(anomaly_pct, n_rows)
            )

        # ── Section 5: Model Performance ──────────────────────────────────────
        if model_metrics:
            metric_interps = self._model_interp.interpret_all(model_metrics)
            if metric_interps:
                sections.append(
                    "## Model Performance Interpretation\n\n"
                    + "\n\n".join(metric_interps)
                )

        # ── Section 6: Data Quality Actions ──────────────────────────────────
        dq_flags = (analyst_flags or []) + (risk_flags or [])
        if dq_flags:
            action_lines = []
            for flag in dq_flags[:5]:
                cat = flag.get("category", flag.get("type", "Unknown"))
                sev = flag.get("level", flag.get("severity", "WARNING"))
                explanation = self._dq_reasoner.explain_flag(cat)
                remediation = self._dq_reasoner.remediation(cat, sev)
                action_lines.append(
                    f"- **[{sev}] {cat}**: {explanation} {remediation}"
                )
            sections.append(
                "## Data Quality Actions Required\n\n" + "\n".join(action_lines)
            )

        # ── Section 7: Recommendation ─────────────────────────────────────────
        if gate_ok and confidence_score >= 0.80 and anomaly_pct < 0.06:
            rec = (
                "✅ **This dataset is approved for downstream modelling and reporting.** "
                "All validation gates passed. Data quality is within acceptable thresholds. "
                "Proceed with feature selection and model training using the enriched dataset."
            )
        elif gate_ok and confidence_score >= 0.60:
            rec = (
                "🟡 **Conditional approval.** Gates passed but one or more quality concerns exist. "
                "Address the data quality actions above before model training. "
                "Re-run the pipeline after fixes to achieve full confidence."
            )
        else:
            rec = (
                "🔴 **Not approved for downstream use.** "
                "Validation gates failed or confidence is too low. "
                "Investigate all flagged issues and re-run the pipeline after remediation."
            )
        sections.append(f"## Recommendation\n\n{rec}")

        # ── Section 8: Analyst Context (if provided) ───────────────────────────
        if user_context and user_context.strip():
            # Keyword-boost hint for downstream callers
            sections.append(
                f"## Analyst Context\n\n"
                f"*The following context was provided by the analyst and guided this report:*\n\n"
                f"> {user_context.strip()}"
            )

        return "\n\n---\n\n".join(sections)

    def build_hf_prompt(
        self,
        verified_result: Dict[str, Any],
        eda_report: Optional[Dict[str, Any]] = None,
        model_metrics: Optional[Dict[str, Any]] = None,
        analyst_flags: Optional[List[Dict]] = None,
        actions_log: Optional[Dict[str, Dict[str, str]]] = None,
        user_context: Optional[str] = None,
        user_context_image_description: Optional[str] = None,
        max_words: int = 400,
    ) -> str:
        """
        Build a rich, structured prompt for the HuggingFace Inference API.
        Includes pre-computed column interpretations, correlation narratives,
        and anomaly context so the LLM can produce a more insightful response.
        """
        eda = eda_report or {}
        summary = eda.get("summary", {})
        numeric_stats = eda.get("numeric_stats", {})
        correlations  = eda.get("correlations", [])
        insights      = eda.get("insights", [])
        gate      = str(verified_result.get("gate_decision", "PASS"))
        confidence = float(verified_result.get("confidence_score", 0.0))
        n_rows    = int(summary.get("n_rows", verified_result.get("row_count", 0)))
        n_cols    = int(summary.get("n_cols", verified_result.get("col_count", 0)))
        null_pct  = float(summary.get("overall_null_pct", 0.0))
        anom_pct  = float(summary.get("anomaly_pct", 0.0))

        # Pre-compute column interpretations (top 4)
        col_summaries: List[str] = []
        for col, stats in list(numeric_stats.items())[:4]:
            if not isinstance(stats, dict):
                continue
            null_c = float(eda.get("missing_values", {}).get(col, {}).get("null_pct", 0.0))
            interp = ColumnInterpreter().interpret(col, stats, null_pct=null_c, actions_log=actions_log)
            col_summaries.append(f"  - {col}: {interp['shape']} — {interp['meaning'][:120]}...")

        # Pre-compute AutoCorrector actions log summaries
        action_summaries: List[str] = []
        if actions_log:
            for col, action in list(actions_log.items())[:5]:
                act = action.get("action", "")
                rsn = action.get("reason", "")
                action_summaries.append(f"  - {col}: Applied {act} ({rsn})")

        # Pre-compute top correlations
        corr_summaries: List[str] = []
        for c in sorted(correlations, key=lambda x: abs(float(x.get("correlation", 0))), reverse=True)[:3]:
            a = c.get("col_a", "")
            b = c.get("col_b", "")
            r = float(c.get("correlation", 0))
            narr = CorrelationNarrator().narrate(a, b, r)
            corr_summaries.append(f"  - {narr[:140]}...")

        # Pre-compute model metric interpretations
        metric_summaries: List[str] = []
        if model_metrics:
            for interp in ModelMetricInterpreter().interpret_all(model_metrics)[:3]:
                metric_summaries.append(f"  - {interp[:140]}")

        prompt = (
            f"You are a Senior Data Scientist writing an executive analytics report for business stakeholders.\n\n"
            f"VERIFIED PIPELINE RESULTS (do not invent any additional facts):\n"
            f"- Dataset: {n_rows:,} rows × {n_cols} columns\n"
            f"- Validation gates: {gate}\n"
            f"- Confidence score: {confidence:.1%}\n"
            f"- Overall null rate: {null_pct:.1%}\n"
            f"- Anomaly rate (Isolation Forest): {anom_pct:.1%}\n\n"
        )
        if col_summaries:
            prompt += f"PRE-COMPUTED COLUMN INSIGHTS:\n" + "\n".join(col_summaries) + "\n\n"
        if corr_summaries:
            prompt += f"CORRELATION ANALYSIS:\n" + "\n".join(corr_summaries) + "\n\n"
        if metric_summaries:
            prompt += f"MODEL PERFORMANCE CONTEXT:\n" + "\n".join(metric_summaries) + "\n\n"
        if action_summaries:
            prompt += f"DATA PREPARATION APPLIED (DO NOT RECOMMEND THESE, THEY ARE ALREADY DONE):\n" + "\n".join(action_summaries) + "\n\n"
        if insights:
            prompt += "AUTO-EDA INSIGHTS:\n" + "\n".join(f"  - {i}" for i in insights[:4]) + "\n\n"

        # Inject user context and image descriptions if available
        if user_context and user_context.strip():
            prompt += f"ANALYST CONTEXT (provided by the user — incorporate into your analysis):\n"
            prompt += f"  {user_context.strip()}\n\n"
        if user_context_image_description and user_context_image_description.strip():
            prompt += f"REFERENCE IMAGE INSIGHT (extracted from uploaded image):\n"
            prompt += f"  {user_context_image_description.strip()}\n\n"

        prompt += (
            f"Write a structured executive report with these sections:\n"
            f"1. **Executive Summary** — what was analysed and the overall verdict (2-3 sentences)\n"
            f"2. **Analytical Insights** — what the data patterns mean in business/domain terms\n"
            f"3. **Data Quality Assessment** — key quality issues and their business impact\n"
            f"4. **Model Readiness** — is this data ready for ML? what steps are needed?\n"
            f"5. **Recommended Actions** — 3-5 concrete, prioritised next steps\n\n"
            f"RULES: Use ONLY the facts provided above. No speculation. No hallucination. "
            f"Write in clear, professional business language. Max {max_words} words total.\n\n"
            f"Executive Report:"
        )
        return prompt


# ── New Sub-Narrators (ip Part B) ─────────────────────────────────────────────


class StatisticalSignificanceNarrator:
    """
    Narrates the results of statistical hypothesis tests.

    Interprets chi-square, t-tests, Mann-Whitney U, ANOVA, and KS tests
    in plain language with effect-size context.
    """

    def narrate(self, stat_tests: Optional[Dict]) -> str:
        """
        Produce a plain-language narrative for statistical test results.

        Parameters
        ----------
        stat_tests : dict like {
            "chi_square": [{"column": "...", "p_value": 0.02, "statistic": 12.4}],
            "t_tests":    [{"column": "...", "p_value": 0.21, "statistic": 1.3}],
        }
        """
        if not stat_tests:
            return "*No statistical tests were performed for this dataset.*"

        lines = ["### Statistical Significance Analysis\n"]

        # ── Chi-Square (independence tests) ─────────────────────────────────
        chi = stat_tests.get("chi_square", []) or []
        if chi:
            lines.append("**Chi-Square Independence Tests:**")
            for t in chi[:5]:
                col = t.get("column", "?")
                p   = float(t.get("p_value", 1.0))
                chi_stat = float(t.get("statistic", 0.0))
                if p < 0.001:
                    sig = "**highly significant** (p < 0.001) — very strong evidence of association"
                elif p < 0.01:
                    sig = "**highly significant** (p < 0.01)"
                elif p < 0.05:
                    sig = "**significant** (p < 0.05)"
                elif p < 0.10:
                    sig = "borderline (p < 0.10) — weak evidence, treat with caution"
                else:
                    sig = "not significant (p = {:.3f}) — no evidence of association".format(p)
                lines.append(f"  - `{col}`: χ² = {chi_stat:.2f}, p = {p:.4f} → {sig}")

        # ── T-tests ──────────────────────────────────────────────────────────
        t_tests = stat_tests.get("t_tests", []) or []
        if t_tests:
            lines.append("\n**Independent Samples T-Tests:**")
            for t in t_tests[:5]:
                col = t.get("column", "?")
                p   = float(t.get("p_value", 1.0))
                st  = float(t.get("statistic", 0.0))
                dir_hint = "higher" if st > 0 else "lower"
                if p < 0.05:
                    interp = f"Group A shows {dir_hint} mean values — **statistically significant** at α=0.05"
                else:
                    interp = "No significant difference between groups at α=0.05"
                lines.append(f"  - `{col}`: t = {st:.2f}, p = {p:.4f} → {interp}")

        # ── ANOVA ─────────────────────────────────────────────────────────────
        anova = stat_tests.get("anova", []) or []
        if anova:
            lines.append("\n**ANOVA Tests (multi-group mean comparison):**")
            for t in anova[:3]:
                col = t.get("column", "?")
                p   = float(t.get("p_value", 1.0))
                f_s = float(t.get("statistic", 0.0))
                sig = "significant group differences exist" if p < 0.05 else "no significant group differences"
                lines.append(f"  - `{col}`: F = {f_s:.2f}, p = {p:.4f} → {sig}")

        # ── KS tests ──────────────────────────────────────────────────────────
        ks = stat_tests.get("ks_tests", []) or []
        if ks:
            lines.append("\n**Kolmogorov-Smirnov Distribution Tests:**")
            for t in ks[:3]:
                col = t.get("column", "?")
                p   = float(t.get("p_value", 1.0))
                d   = float(t.get("statistic", 0.0))
                sig = "distributions are **significantly different**" if p < 0.05 else "distributions appear similar"
                lines.append(f"  - `{col}`: D = {d:.4f}, p = {p:.4f} → {sig}")

        if len(lines) == 1:
            return "*No significant tests found in results.*"

        lines.append(
            "\n> **Interpretation note**: Statistical significance (p < 0.05) indicates "
            "a pattern is unlikely due to chance, but does not imply practical significance. "
            "Always consider effect size and domain context."
        )
        return "\n".join(lines)


class FeatureImportanceNarrator:
    """
    Narrates feature importance scores from ML models.
    Ranks features by importance, identifies dominant features,
    and warns about suspected target leakage.
    """

    LEAKAGE_THRESHOLD = 0.30  # Single feature with >30% importance → suspect leakage

    def narrate(
        self,
        importances: Optional[Dict[str, float]],
        model_name: str = "",
        task_type: str = "classification",
    ) -> str:
        """
        Parameters
        ----------
        importances : {"col_name": importance_score, ...} (already normalised 0-1)
        model_name  : Name of the model that produced these importances
        task_type   : 'classification' or 'regression'
        """
        if not importances:
            return "*Feature importance data not available.*"

        total = sum(importances.values()) or 1.0
        ranked = sorted(importances.items(), key=lambda x: -x[1])
        top = ranked[:10]

        lines = [f"### Feature Importance Analysis ({model_name or 'Model'})\n"]

        # Top feature summary
        if top:
            top_col, top_score = top[0]
            pct = top_score / total
            if pct > self.LEAKAGE_THRESHOLD:
                lines.append(
                    f"⚠️  **Leakage Warning**: `{top_col}` accounts for {pct:.1%} of total feature "
                    f"importance. A single feature dominating >{self.LEAKAGE_THRESHOLD:.0%} of "
                    "importance is a common target leakage signal. Verify this feature would be "
                    "available at inference time."
                )
            else:
                lines.append(
                    f"The most predictive feature is **`{top_col}`** ({pct:.1%} of total importance). "
                    f"{'No single feature dominant — importance is well-distributed.' if pct < 0.15 else ''}"
                )

        # Cumulative importance
        cumulative = 0.0
        n_for_80 = 0
        for _, score in ranked:
            cumulative += score / total
            n_for_80 += 1
            if cumulative >= 0.80:
                break

        lines.append(
            f"\n**Concentration**: {n_for_80} feature(s) explain 80% of model signal "
            f"(out of {len(ranked)} total features)."
        )
        if n_for_80 == 1:
            lines.append("⚠️  Very high importance concentration — verify for data leakage.")
        elif n_for_80 >= len(ranked) * 0.7:
            lines.append("✅ Well-distributed importance — model relies on many features.")

        # Top 10 ranked list
        lines.append("\n**Top Features (ranked by importance):**")
        for i, (col, score) in enumerate(top, 1):
            bar = "█" * int((score / total) * 30)
            lines.append(f"  {i:2d}. `{col}` {bar} {score / total:.2%}")

        return "\n".join(lines)


class BiasAndFairnessNarrator:
    """
    Narrates bias and fairness metrics in plain language.

    Interprets demographic parity, equal opportunity, and disparate
    impact ratios for protected groups.
    """

    DI_THRESHOLD      = 0.80  # Disparate Impact < 80% → adverse impact (4/5ths rule)
    DP_DIFF_THRESHOLD = 0.10  # Demographic Parity difference > 10% → concern

    def narrate(self, bias_report: Optional[Dict]) -> str:
        """
        Parameters
        ----------
        bias_report : {
            "protected_attribute": str,
            "metrics": {
                "demographic_parity_diff": float,
                "disparate_impact":        float,
                "equal_opportunity_diff":  float,
                "predictive_parity_diff":  float,
            },
            "group_positive_rates": {"Group A": 0.45, "Group B": 0.28, ...},
        }
        """
        if not bias_report:
            return "*Bias/fairness analysis was not performed for this dataset.*"

        attr    = bias_report.get("protected_attribute", "protected_attribute")
        metrics = bias_report.get("metrics", {}) or {}
        group_rates = bias_report.get("group_positive_rates", {}) or {}

        lines = [f"### Bias & Fairness Analysis — Protected Attribute: `{attr}`\n"]

        di  = metrics.get("disparate_impact")
        dpd = metrics.get("demographic_parity_diff")
        eod = metrics.get("equal_opportunity_diff")
        ppd = metrics.get("predictive_parity_diff")

        # Disparate Impact (4/5ths rule)
        if di is not None:
            di = float(di)
            if di < self.DI_THRESHOLD:
                lines.append(
                    f"⚠️  **Adverse Impact Detected** (Disparate Impact = {di:.3f} < {self.DI_THRESHOLD}). "
                    "Under the 4/5ths rule used by EEOC and GDPR, a DI ratio below 0.80 indicates "
                    "potential discrimination. Immediate fairness review required."
                )
            elif di < 0.90:
                lines.append(
                    f"⚡ **Moderate disparity** (DI = {di:.3f}). Below 0.90 — monitor closely. "
                    "May be acceptable with documented business justification."
                )
            else:
                lines.append(f"✅ Disparate Impact = {di:.3f} — above the 0.80 threshold (compliant).")

        # Demographic Parity Difference
        if dpd is not None:
            dpd = float(dpd)
            flag = "⚠️ " if abs(dpd) > self.DP_DIFF_THRESHOLD else "✅"
            lines.append(
                f"{flag} Demographic Parity Difference = {dpd:+.3f} "
                f"({'concerning' if abs(dpd) > self.DP_DIFF_THRESHOLD else 'acceptable'}). "
                "Positive values mean the majority group receives more favourable outcomes."
            )

        # Equal Opportunity Difference
        if eod is not None:
            eod = float(eod)
            flag = "⚠️ " if abs(eod) > 0.10 else "✅"
            lines.append(
                f"{flag} Equal Opportunity Difference = {eod:+.3f} "
                "(True Positive Rate gap between protected groups)."
            )

        # Group rates breakdown
        if group_rates:
            lines.append("\n**Positive Outcome Rates by Group:**")
            for group, rate in sorted(group_rates.items(), key=lambda x: -x[1]):
                bar = "█" * int(float(rate) * 20)
                lines.append(f"  - {group}: {bar} {float(rate):.1%}")

        lines.append(
            "\n> **Note**: Fairness metrics depend on the definition of 'fairness' (demographic parity, "
            "equal opportunity, or individual fairness). Consult domain experts and legal counsel before "
            "making decisions based on these metrics."
        )
        return "\n".join(lines)


class AnomalyDeepDiveNarrator:
    """
    Narrates anomaly detection results with business-context interpretation.

    Interprets isolation forest and z-score anomaly rates, identifies
    potential causes, and provides remediation guidance.
    """

    def narrate(
        self,
        anomaly_report: Optional[Dict],
        total_rows: int = 0,
    ) -> str:
        """
        Parameters
        ----------
        anomaly_report : {
            "method":        str,  IsolationForest | ZScore | IQR
            "anomaly_count": int,
            "anomaly_rate":  float,
            "top_anomalous_columns": [{"column": str, "anomaly_contribution": float}],
            "sample_anomalies": [...],
        }
        """
        if not anomaly_report:
            return "*Anomaly deep-dive analysis not available.*"

        method = anomaly_report.get("method", "Isolation Forest")
        count  = int(anomaly_report.get("anomaly_count", 0))
        rate   = float(anomaly_report.get("anomaly_rate", 0.0))
        top_cols = anomaly_report.get("top_anomalous_columns", []) or []

        lines = [f"### Anomaly Deep-Dive Analysis ({method})\n"]

        # Rate interpretation
        if rate < 0.005:
            severity = "✅ Very low anomaly rate"
            advice   = "Dataset appears clean — anomalies may be rare but legitimate edge cases."
        elif rate < 0.03:
            severity = "⚡ Moderate anomaly rate"
            advice   = "Investigate flagged records — may reflect data entry errors or genuine outliers."
        elif rate < 0.10:
            severity = "⚠️  Elevated anomaly rate"
            advice   = "High anomaly volume suggests systematic data quality issues or concept drift."
        else:
            severity = "🔴 Critical anomaly rate"
            advice   = "Dataset heavily contaminated — validate source data integrity before use."

        n_str = f"{count:,}" if total_rows == 0 else f"{count:,} / {total_rows:,}"
        lines.append(
            f"{severity}: {n_str} anomalies detected ({rate:.2%} of records). {advice}"
        )

        # Top contributing columns
        if top_cols:
            lines.append("\n**Top Columns Contributing to Anomalies:**")
            for item in top_cols[:6]:
                col  = item.get("column", "?")
                cont = float(item.get("anomaly_contribution", 0.0))
                lines.append(f"  - `{col}`: {cont:.1%} contribution to anomaly score")

        # Likely causes
        lines.append("\n**Likely Anomaly Causes:**")
        if rate > 0.05:
            lines.append("  - Systematic data quality issues (data pipeline errors, schema drift)")
        if top_cols:
            col0 = top_cols[0].get("column", "?")
            lines.append(f"  - Extreme values in `{col0}` — verify data type and range constraints")
        lines.append("  - Genuine rare events (fraud, equipment failure, market shocks)")
        lines.append("  - Label/target drift since last model training")

        # Remediation
        lines.append(
            "\n**Recommended Actions:**\n"
            "  1. Export anomalous records for manual review by domain expert\n"
            "  2. Check ETL pipeline for data corruption between {source} and {sink}\n"
            "  3. If genuine, consider separate model for anomalous sub-population\n"
            "  4. Update anomaly detection threshold if false-positive rate is high"
        )
        return "\n".join(lines)


class RegulatoryComplianceNarrator:
    """
    Narrates regulatory compliance results across multiple domains in plain language.

    Prioritises CRITICAL > ERROR > WARNING violations and provides
    a board-level compliance summary.
    """

    def narrate(
        self,
        regulatory_reports: Optional[Dict],
        domains: Optional[List[str]] = None,
    ) -> str:
        """
        Parameters
        ----------
        regulatory_reports : {
            "banking": {"violations": [RegulatoryViolation.to_dict()], "score": 0.85},
            "gdpr":    {"violations": [...], "score": 0.93},
            ...
        }
        """
        if not regulatory_reports:
            return "*No regulatory compliance data available.*"

        lines = ["### Multi-Domain Regulatory Compliance Summary\n"]

        total_critical = 0
        total_error    = 0
        total_warning  = 0
        domain_summaries = []

        for domain, report in regulatory_reports.items():
            if not isinstance(report, dict):
                continue
            violations = report.get("violations", []) or []
            score      = float(report.get("score", 1.0))

            n_crit = sum(1 for v in violations if v.get("severity") == "CRITICAL")
            n_err  = sum(1 for v in violations if v.get("severity") == "ERROR")
            n_warn = sum(1 for v in violations if v.get("severity") == "WARNING")
            total_critical += n_crit
            total_error    += n_err
            total_warning  += n_warn

            if n_crit > 0:
                status_icon = "🔴"
            elif n_err > 0:
                status_icon = "🟠"
            elif n_warn > 0:
                status_icon = "🟡"
            else:
                status_icon = "✅"

            domain_summaries.append(
                f"  {status_icon} **{domain.upper()}**: score={score:.1%} | "
                f"{n_crit} CRITICAL, {n_err} ERROR, {n_warn} WARNING"
            )

        # Executive verdict
        if total_critical > 0:
            verdict = (
                f"🔴 **NON-COMPLIANT**: {total_critical} critical violation(s) require immediate remediation "
                "before this dataset can be used in regulated processes."
            )
        elif total_error > 0:
            verdict = (
                f"🟠 **CONDITIONAL COMPLIANCE**: {total_error} error-level violations must be addressed. "
                "Escalate to compliance officer within 5 business days."
            )
        elif total_warning > 0:
            verdict = (
                f"🟡 **CAUTION**: {total_warning} warning(s) identified. Compliant with conditions. "
                "Review warnings with compliance team."
            )
        else:
            verdict = "✅ **FULLY COMPLIANT**: All regulatory checks passed across all domains."

        lines.append(verdict)
        lines.append("\n**Domain Breakdown:**")
        lines.extend(domain_summaries)

        if total_critical + total_error > 0:
            lines.append(
                "\n> **Board Action Required**: Non-compliance findings must be documented in the "
                "Risk Register and reported to Compliance Committee. Data processing may need to "
                "be suspended until critical violations are resolved."
            )

        return "\n".join(lines)


class DataLineageNarrator:
    """
    Narrates data lineage and provenance information in plain language.

    Interprets transformation steps, data sources, schema versions,
    and quality scores at each pipeline stage.
    """

    def narrate(self, lineage: Optional[Dict]) -> str:
        """
        Parameters
        ----------
        lineage : {
            "dataset_id": str,
            "source_type": str,
            "ingestion_timestamp": str,
            "transformations": [{"stage": str, "operation": str, "rows_in": int, "rows_out": int}],
            "quality_gates": [{"gate": str, "passed": bool, "score": float}],
            "schema_version": str,
            "fingerprint": str,
        }
        """
        if not lineage:
            return "*Data lineage information not available.*"

        ds_id      = lineage.get("dataset_id", "Unknown")
        source     = lineage.get("source_type", "Unknown")
        ts         = lineage.get("ingestion_timestamp", "Unknown")
        schema_ver = lineage.get("schema_version", "N/A")
        fp         = lineage.get("fingerprint", "")
        transforms = lineage.get("transformations", []) or []
        gates      = lineage.get("quality_gates", []) or []

        lines = [f"### Data Lineage & Provenance — `{ds_id}`\n"]

        lines.append(f"**Source**: {source} | **Ingested**: {ts} | **Schema**: v{schema_ver}")
        if fp:
            lines.append(f"**Data Fingerprint**: `{fp[:24]}…`")

        # Transformation pipeline
        if transforms:
            lines.append("\n**Transformation Pipeline:**")
            for i, step in enumerate(transforms, 1):
                stage   = step.get("stage", f"Stage {i}")
                op      = step.get("operation", "transform")
                rows_in = int(step.get("rows_in", 0))
                rows_out= int(step.get("rows_out", 0))
                drop_pct = 0.0 if rows_in == 0 else (rows_in - rows_out) / rows_in
                arrow = "→"
                lines.append(
                    f"  {i}. **{stage}** ({op}): {rows_in:,} rows {arrow} {rows_out:,} rows "
                    f"({drop_pct:.1%} removed)"
                )
                if drop_pct > 0.10:
                    lines.append(f"     ⚠️  High row drop rate ({drop_pct:.1%}) — investigate upstream quality.")

        # Quality gate summary
        if gates:
            passed = sum(1 for g in gates if g.get("passed", False))
            lines.append(f"\n**Quality Gates**: {passed}/{len(gates)} passed")
            for g in gates:
                icon = "✅" if g.get("passed", False) else "❌"
                gate_name = g.get("gate", "Gate")
                score = g.get("score", None)
                score_str = f" (score: {float(score):.2f})" if score is not None else ""
                lines.append(f"  {icon} {gate_name}{score_str}")

        lines.append(
            "\n> **Audit Note**: This lineage record is cryptographically linked to the "
            "source data fingerprint. Any downstream tampering will cause a checksum mismatch."
        )
        return "\n".join(lines)


# ── Module-level helper ────────────────────────────────────────────────────────

def get_narrator() -> NarrativeBuilder:
    """Returns a shared NarrativeBuilder instance."""
    return NarrativeBuilder()
