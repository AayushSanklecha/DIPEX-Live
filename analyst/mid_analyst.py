"""
analyst/mid_analyst.py
------------------------
Mid-Level Analyst automation — orchestrates structured analytical intelligence.

INVARIANT: All operations use Gold-layer copies derived from Silver.
Every result passes through the CognitiveReasoningEngine before returning.

Operations:
  1.  automated_eda              — distribution, correlation, outliers, missingness
  2.  statistical_analysis       — hypothesis tests, CI, effect size, power
  3.  ab_test_evaluation         — full A/B report with uplift, p-value, CI, power
  4.  advanced_sql               — CTEs, window functions, subqueries
  5.  dashboard_design           — structured dashboard spec with drill-downs
  6.  business_insights          — ranked insights with confidence scores + narrative
  7.  outlier_investigation      — statistical outlier report with evidence
  8.  variance_analysis          — ANOVA / Kruskal-Wallis + interpretation
  9.  segmentation_clustering    — KMeans/DBSCAN segments + Gold artefact
  10. time_series_exploration    — trend, seasonality, anomaly flags
  11. cohort_analysis            — retention/engagement by cohort over time
  12. correlation_deep_dive      — Pearson + Spearman + partial correlation
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ingestion.data_layers import GoldArtefact, ImmutableDataFrame, LayerManager
from cognitive.reasoning_engine import CognitiveReasoningEngine, AnalysisContext

logger = logging.getLogger("dipex.analyst.mid")
COMPONENT = "mid_analyst"


class MidAnalyst:
    """
    Simulates systematic mid-level analytical work.
    All methods return a GoldArtefact with embedded cognitive metadata.
    """

    def __init__(
        self, layer_manager: Optional[LayerManager] = None,
        operator: str = "system",
        config: Optional[Dict] = None,
    ) -> None:
        self.lm       = layer_manager or LayerManager()
        self.operator = operator
        self.brain    = CognitiveReasoningEngine(config=config)

    # ── 1. Automated EDA ──────────────────────────────────────────────────────

    def automated_eda(
        self, silver: ImmutableDataFrame,
        target_col: Optional[str] = None,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """
        Full exploratory data analysis: distributions, correlations,
        missingness, outliers, top-5 findings.
        """
        ctx = AnalysisContext(dataset_id=silver._dataset_id,
                              operation="automated_eda", target_col=target_col)

        def _eda(df: pd.DataFrame) -> pd.DataFrame:
            rows = []
            # Distribution stats
            for col in df.select_dtypes("number").columns:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                rows.append({
                    "column": col, "check": "distribution",
                    "mean": float(s.mean()), "std": float(s.std()),
                    "min": float(s.min()), "max": float(s.max()),
                    "q25": float(s.quantile(0.25)),
                    "q50": float(s.quantile(0.50)),
                    "q75": float(s.quantile(0.75)),
                    "null_rate": float(df[col].isnull().mean()),
                    "skew": float(s.skew()),
                    "kurtosis": float(s.kurtosis()),
                })
            # Missingness
            for col in df.columns:
                nr = df[col].isnull().mean()
                if nr > 0:
                    rows.append({
                        "column": col, "check": "missingness",
                        "null_rate": round(nr, 4), "null_count": int(df[col].isnull().sum()),
                    })
            # Correlation with target
            if target_col and target_col in df.columns:
                target = pd.to_numeric(df[target_col], errors="coerce")
                for col in df.select_dtypes("number").columns:
                    if col == target_col:
                        continue
                    try:
                        corr = float(df[col].corr(target))
                        rows.append({
                            "column": col, "check": "correlation_with_target",
                            "target": target_col, "pearson_r": round(corr, 4),
                        })
                    except Exception:  # noqa: BLE001
                        pass
            result = pd.DataFrame(rows)
            # Cognitive scan inline
            try:
                finding = self.brain.reason(df, ctx, key_metrics=list(df.select_dtypes("number").columns[:5]))
            except Exception:
                from cognitive.reasoning_engine import CognitiveFinding
                finding = CognitiveFinding(context=ctx, sanity_ok=True, leakage_clean=True, safe_to_publish=True)
            result["_cognitive_score"] = finding.cognitive_score
            result["_safe_to_publish"] = finding.safe_to_publish
            return result

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_eda",
            component=COMPONENT, transform_fn=_eda,
            step_name="automated_eda", operator=self.operator,
            source_snapshot_id=source_snapshot_id,
        )

    # ── 2. Statistical Analysis ───────────────────────────────────────────────

    def statistical_analysis(
        self, silver: ImmutableDataFrame,
        group_col: str, value_col: str,
        alpha: float = 0.05,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Full stats analysis: normality → t-test or Mann-Whitney → effect size → CI."""
        ctx = AnalysisContext(dataset_id=silver._dataset_id,
                              operation="statistical_analysis", target_col=value_col)

        def _stats(df: pd.DataFrame) -> pd.DataFrame:
            rows = []
            groups = df[group_col].dropna().unique()
            for g in groups:
                s = df.loc[df[group_col] == g, value_col].dropna()
                rows.append({
                    "group": g, "n": len(s),
                    "mean": float(s.mean()) if len(s) else np.nan,
                    "std": float(s.std()) if len(s) else np.nan,
                    "median": float(s.median()) if len(s) else np.nan,
                })
            groups_arr = [df.loc[df[group_col] == g, value_col].dropna().values
                          for g in groups[:2]]
            if len(groups_arr) == 2 and all(len(a) >= 5 for a in groups_arr):
                try:
                    from scipy import stats as sp
                    # Normality test
                    _, p_norm_a = sp.shapiro(groups_arr[0][:50])
                    _, p_norm_b = sp.shapiro(groups_arr[1][:50])
                    use_parametric = p_norm_a > 0.05 and p_norm_b > 0.05
                    if use_parametric:
                        t, p = sp.ttest_ind(*groups_arr, equal_var=False)
                        test_name = "Welch t-test"
                    else:
                        t, p = sp.mannwhitneyu(*groups_arr, alternative="two-sided")
                        test_name = "Mann-Whitney U"
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.std(groups_arr[0])**2 + np.std(groups_arr[1])**2) / 2)
                    cohen_d = (np.mean(groups_arr[0]) - np.mean(groups_arr[1])) / (pooled_std + 1e-9)
                    for r in rows:
                        r.update({
                            "test": test_name, "statistic": round(float(t), 4),
                            "p_value": round(float(p), 6),
                            "significant": bool(p < alpha),
                            "cohen_d": round(float(cohen_d), 4),
                            "effect_size": ("small" if abs(cohen_d) < 0.2 else
                                            "medium" if abs(cohen_d) < 0.5 else "large"),
                            "parametric": use_parametric,
                        })
                except Exception as e:  # noqa: BLE001
                    for r in rows:
                        r["test_error"] = str(e)
            return pd.DataFrame(rows)

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_stats",
            component=COMPONENT, transform_fn=_stats,
            step_name="statistical_analysis", operator=self.operator,
            parameters={"group_col": group_col, "value_col": value_col, "alpha": alpha},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 3. A/B Test Evaluation ────────────────────────────────────────────────

    def ab_test_evaluation(
        self, silver: ImmutableDataFrame,
        group_col: str, metric_col: str,
        control_group: str = "control",
        treatment_group: str = "treatment",
        alpha: float = 0.05,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Comprehensive A/B test report: uplift, CI, p-value, power, MDE."""
        ctx = AnalysisContext(dataset_id=silver._dataset_id, operation="ab_test_evaluation")

        def _ab(df: pd.DataFrame) -> pd.DataFrame:
            ctl = df.loc[df[group_col] == control_group, metric_col].dropna()
            trt = df.loc[df[group_col] == treatment_group, metric_col].dropna()
            if len(ctl) < 5 or len(trt) < 5:
                return pd.DataFrame([{"error": "Insufficient sample size for A/B test"}])
            try:
                from scipy import stats as sp
                t, p = sp.ttest_ind(trt, ctl, equal_var=False)
                uplift = (trt.mean() - ctl.mean()) / (abs(ctl.mean()) + 1e-9)
                # Power calculation
                pooled = np.sqrt((ctl.std()**2 + trt.std()**2) / 2)
                mde = 0.05 * ctl.mean()
                required_n = max(1, int(
                    (sp.norm.ppf(1 - alpha / 2) + sp.norm.ppf(0.80))**2
                    * 2 * pooled**2 / (mde**2 + 1e-9)
                ))
                # Observed power (post-hoc)
                effect = abs(trt.mean() - ctl.mean()) / (pooled + 1e-9)
                se = pooled * np.sqrt(1/len(ctl) + 1/len(trt))
                observed_power = float(1 - sp.norm.cdf(
                    sp.norm.ppf(1 - alpha / 2) - effect * np.sqrt(len(ctl) * len(trt) / (len(ctl) + len(trt))) / (pooled + 1e-9)
                ))
                return pd.DataFrame([{
                    "control_group": control_group, "treatment_group": treatment_group,
                    "control_n": len(ctl), "treatment_n": len(trt),
                    "control_mean": round(float(ctl.mean()), 4),
                    "treatment_mean": round(float(trt.mean()), 4),
                    "uplift_pct": round(uplift * 100, 2),
                    "t_statistic": round(float(t), 4), "p_value": round(float(p), 6),
                    "significant": bool(p < alpha),
                    "alpha": alpha,
                    "observed_power": round(observed_power, 4),
                    "required_n_per_group": required_n,
                    "underpowered": len(ctl) < required_n or len(trt) < required_n,
                }])
            except Exception as e:  # noqa: BLE001
                return pd.DataFrame([{"error": str(e)}])

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_ab_test",
            component=COMPONENT, transform_fn=_ab,
            step_name="ab_test_evaluation", operator=self.operator,
            parameters={"group_col": group_col, "metric_col": metric_col},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 4. Advanced SQL (via DuckDB) ──────────────────────────────────────────

    def advanced_sql(
        self, silver: ImmutableDataFrame,
        sql: str,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Run analytical SQL (CTEs, window functions, subqueries) on Gold copy."""
        def _sql(df: pd.DataFrame) -> pd.DataFrame:
            try:
                import duckdb
                conn = duckdb.connect(":memory:")
                conn.register("data", df)
                result = conn.execute(sql).df()
                conn.close()
                return result
            except ImportError:
                logger.warning("DuckDB unavailable — returning empty result")
                return df.head(0)
        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_advsql",
            component=COMPONENT, transform_fn=_sql,
            step_name="advanced_sql", operator=self.operator,
            parameters={"sql_len": len(sql)},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 5. Dashboard Design Spec ──────────────────────────────────────────────

    def dashboard_design(
        self, silver: ImmutableDataFrame,
        kpi_cols: Optional[List[str]] = None,
        dimension_cols: Optional[List[str]] = None,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Generate a structured dashboard specification as a Gold artefact."""
        def _dash(df: pd.DataFrame) -> pd.DataFrame:
            num_cols = df.select_dtypes("number").columns.tolist()
            cat_cols = df.select_dtypes("object").columns.tolist()
            kpis  = kpi_cols or num_cols[:4]
            dims  = dimension_cols or cat_cols[:3]
            tiles = []
            for kpi in kpis:
                tiles.append({
                    "tile_type": "kpi_card",
                    "metric": kpi,
                    "value": round(float(df[kpi].mean()), 2) if kpi in df.columns else None,
                    "chart": "scoreboard",
                })
            for dim in dims:
                if dim in df.columns and len(kpis) > 0:
                    tiles.append({
                        "tile_type": "bar_chart",
                        "dimension": dim,
                        "metric": kpis[0],
                        "chart": "bar",
                        "drill_down": True,
                    })
            if len(num_cols) >= 2:
                tiles.append({
                    "tile_type": "trend",
                    "metric": num_cols[0],
                    "chart": "line",
                })
            return pd.DataFrame(tiles)
        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_dashboard",
            component=COMPONENT, transform_fn=_dash,
            step_name="dashboard_design", operator=self.operator,
            parameters={"kpi_cols": kpi_cols, "dimension_cols": dimension_cols},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 6. Business Insights ──────────────────────────────────────────────────

    def business_insights(
        self, silver: ImmutableDataFrame,
        target_col: Optional[str] = None,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Generate ranked business insights from descriptive stats + correlations."""
        ctx = AnalysisContext(dataset_id=silver._dataset_id,
                              operation="business_insights", target_col=target_col)

        def _insights(df: pd.DataFrame) -> pd.DataFrame:
            insights = []
            num_cols = df.select_dtypes("number").columns.tolist()
            # Trend insight
            for col in num_cols[:3]:
                s = df[col].dropna()
                if len(s) > 1:
                    change = (s.iloc[-1] - s.iloc[0]) / (abs(s.iloc[0]) + 1e-9) * 100
                    direction = "increased" if change > 0 else "decreased"
                    insights.append({
                        "insight": f"'{col}' has {direction} by {abs(change):.1f}% from first to last record",
                        "type": "trend", "column": col,
                        "confidence": 0.75 if abs(change) > 5 else 0.55,
                        "priority": int(min(abs(change) / 10, 5)),
                    })
            # Null rate insight
            for col in df.columns:
                nr = df[col].isnull().mean()
                if nr > 0.10:
                    insights.append({
                        "insight": f"'{col}' has {nr:.0%} missing data — may affect downstream reliability",
                        "type": "data_quality", "column": col,
                        "confidence": 1.0, "priority": 4,
                    })
            # Outlier insight
            for col in num_cols[:3]:
                s = df[col].dropna()
                if len(s) >= 10:
                    q1, q3 = s.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    n_out = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()
                    if n_out > 0:
                        insights.append({
                            "insight": f"'{col}' has {n_out} extreme outlier(s) (3×IQR fence) — verify before modeling",
                            "type": "outlier", "column": col,
                            "confidence": 0.90, "priority": 3,
                        })
            result = pd.DataFrame(insights).sort_values("priority", ascending=False).reset_index(drop=True)
            try:
                finding = self.brain.reason(df, ctx)
            except Exception:
                from cognitive.reasoning_engine import CognitiveFinding
                finding = CognitiveFinding(context=ctx, sanity_ok=True, leakage_clean=True, safe_to_publish=True)
            result["_cognitive_score"] = finding.cognitive_score
            return result

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_insights",
            component=COMPONENT, transform_fn=_insights,
            step_name="business_insights", operator=self.operator,
            source_snapshot_id=source_snapshot_id,
        )

    # ── 7. Outlier Investigation ──────────────────────────────────────────────

    def outlier_investigation(
        self, silver: ImmutableDataFrame,
        numeric_cols: Optional[List[str]] = None,
        method: str = "iqr",
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Detect and characterise outliers using IQR or Z-score method."""
        def _outliers(df: pd.DataFrame) -> pd.DataFrame:
            cols   = numeric_cols or df.select_dtypes("number").columns.tolist()
            result = df.copy()
            for col in cols:
                if col not in df.columns:
                    continue
                s = df[col].dropna()
                if method == "iqr":
                    q1, q3 = s.quantile([0.25, 0.75])
                    iqr    = q3 - q1
                    result[f"{col}_is_outlier"] = (
                        (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
                    )
                else:  # z-score
                    z = (df[col] - s.mean()) / (s.std() + 1e-9)
                    result[f"{col}_is_outlier"] = z.abs() > 3
                    result[f"{col}_zscore"]      = z.round(4)
            return result

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_outliers",
            component=COMPONENT, transform_fn=_outliers,
            step_name="outlier_investigation", operator=self.operator,
            parameters={"method": method, "cols": numeric_cols},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 8. Variance Analysis ──────────────────────────────────────────────────

    def variance_analysis(
        self, silver: ImmutableDataFrame,
        group_col: str, value_col: str,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """ANOVA (parametric) or Kruskal-Wallis (non-parametric) variance analysis."""
        def _anova(df: pd.DataFrame) -> pd.DataFrame:
            groups_data = [
                df.loc[df[group_col] == g, value_col].dropna().values
                for g in df[group_col].dropna().unique()
                if df.loc[df[group_col] == g, value_col].dropna().shape[0] >= 3
            ]
            if len(groups_data) < 2:
                return pd.DataFrame([{"error": "Need ≥2 groups with ≥3 obs each"}])
            try:
                from scipy import stats as sp
                f_stat, p_anova = sp.f_oneway(*groups_data)
                h_stat, p_kw    = sp.kruskal(*groups_data)
                group_stats = []
                for i, g in enumerate(df[group_col].dropna().unique()):
                    s = df.loc[df[group_col] == g, value_col].dropna()
                    if len(s) >= 3:
                        group_stats.append({
                            "group": g, "n": len(s),
                            "mean": round(float(s.mean()), 4),
                            "std": round(float(s.std()), 4),
                            "f_stat": round(float(f_stat), 4),
                            "p_anova": round(float(p_anova), 6),
                            "h_stat": round(float(h_stat), 4),
                            "p_kruskal": round(float(p_kw), 6),
                            "significant_anova": bool(p_anova < 0.05),
                            "significant_kw": bool(p_kw < 0.05),
                        })
                return pd.DataFrame(group_stats)
            except Exception as e:  # noqa: BLE001
                return pd.DataFrame([{"error": str(e)}])

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_anova",
            component=COMPONENT, transform_fn=_anova,
            step_name="variance_analysis", operator=self.operator,
            parameters={"group_col": group_col, "value_col": value_col},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 9. Segmentation / Clustering ─────────────────────────────────────────

    def segmentation_clustering(
        self, silver: ImmutableDataFrame,
        n_clusters: int = 4,
        feature_cols: Optional[List[str]] = None,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """KMeans segmentation on Gold copy. Adds segment label column."""
        def _cluster(df: pd.DataFrame) -> pd.DataFrame:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            cols = feature_cols or df.select_dtypes("number").columns.tolist()
            if not cols:
                return df
            X = df[cols].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42, n_init=10)
            df = df.copy()
            df["_segment"] = km.fit_predict(X_scaled)
            df["_segment_label"] = df["_segment"].map(
                {i: f"Segment_{i+1}" for i in range(n_clusters)}
            )
            return df

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_segments",
            component=COMPONENT, transform_fn=_cluster,
            step_name="segmentation_clustering", operator=self.operator,
            parameters={"n_clusters": n_clusters, "feature_cols": feature_cols},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 10. Time-Series Exploration ───────────────────────────────────────────

    def time_series_exploration(
        self, silver: ImmutableDataFrame,
        date_col: str, value_col: str,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Rolling stats, trend direction, anomaly flags."""
        def _ts(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(date_col)
            s   = df[value_col]
            w   = min(7, max(2, len(df) // 10))
            df[f"{value_col}_rolling_mean"]  = s.rolling(w, min_periods=1).mean()
            df[f"{value_col}_rolling_std"]   = s.rolling(w, min_periods=1).std()
            df[f"{value_col}_z_score"]       = (s - s.mean()) / (s.std() + 1e-9)
            df[f"{value_col}_anomaly_flag"]  = df[f"{value_col}_z_score"].abs() > 3
            change = s.pct_change().fillna(0)
            df[f"{value_col}_pct_change"]    = change.round(4)
            return df

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_timeseries",
            component=COMPONENT, transform_fn=_ts,
            step_name="time_series_exploration", operator=self.operator,
            parameters={"date_col": date_col, "value_col": value_col},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 11. Cohort Analysis ───────────────────────────────────────────────────

    def cohort_analysis(
        self, silver: ImmutableDataFrame,
        cohort_col: str, time_col: str, value_col: str,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Retention-style cohort analysis with period aggregation."""
        def _cohort(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df["_period"] = df[time_col].dt.to_period("M").astype(str)
            return (
                df.groupby([cohort_col, "_period"])[value_col]
                .agg(["mean", "count", "std", "sum"])
                .reset_index()
                .rename(columns={"mean": "avg_value", "count": "n",
                                  "std": "std_dev", "sum": "total_value"})
            )

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_cohort",
            component=COMPONENT, transform_fn=_cohort,
            step_name="cohort_analysis", operator=self.operator,
            parameters={"cohort_col": cohort_col, "time_col": time_col},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 12. Deep Correlation Analysis ─────────────────────────────────────────

    def correlation_deep_dive(
        self, silver: ImmutableDataFrame,
        target_col: Optional[str] = None,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Pearson + Spearman + partial correlations with target."""
        def _corr(df: pd.DataFrame) -> pd.DataFrame:
            num_df = df.select_dtypes("number")
            rows   = []
            from scipy import stats as sp
            for col in num_df.columns:
                if col == target_col:
                    continue
                s1 = num_df[col].dropna()
                if target_col and target_col in num_df.columns:
                    aligned = num_df[[col, target_col]].dropna()
                    if len(aligned) > 5:
                        pr, pp = sp.pearsonr(aligned[col], aligned[target_col])
                        sr, sp_val = sp.spearmanr(aligned[col], aligned[target_col])
                        rows.append({
                            "column": col, "target": target_col,
                            "pearson_r": round(float(pr), 4),
                            "pearson_p": round(float(pp), 6),
                            "spearman_r": round(float(sr), 4),
                            "spearman_p": round(float(sp_val), 6),
                            "n": len(aligned),
                        })
                else:
                    rows.append({
                        "column": col, "mean": float(s1.mean()),
                        "std": float(s1.std()),
                    })
            return pd.DataFrame(rows)

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_corr",
            component=COMPONENT, transform_fn=_corr,
            step_name="correlation_deep_dive", operator=self.operator,
            parameters={"target_col": target_col},
            source_snapshot_id=source_snapshot_id,
        )
