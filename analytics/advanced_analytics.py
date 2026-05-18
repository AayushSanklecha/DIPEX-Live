"""
analytics/advanced_analytics.py
----------------------------------
Advanced analytics stages for professional reporting.

Provides computation of:
  - Statistical significance tests (normality, stationarity, homogeneity)
  - Feature importance via RandomForest proxy (no labels required)
  - Bias & Fairness analysis (disparate impact, statistical parity)
  - Anomaly deep dive (Isolation Forest + Z-score analysis)

All functions are non-fatal: exceptions are caught per-column and
logged as DEBUG, never crashing the pipeline.

Used by: AnalyticsOrchestrator (analytics/orchestrator.py)
Output:  Stored in AnalyticsResult fields (statistical_tests, feature_importance,
         bias_fairness_report, anomaly_deep_dive)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.analytics.advanced")

_MAX_COLS_FOR_STAT_TESTS = 30
_MAX_ROWS_FOR_STAT_TESTS = 100_000
_MIN_ROWS_FOR_STAT_TESTS = 8
_NORMALITY_ALPHA = 0.05
_STATIONARITY_ALPHA = 0.05
_DISPARATE_IMPACT_THRESHOLD = 0.8   # 80% rule (4/5 rule)
_ANOMALY_CONTAMINATION = 0.05       # 5% expected anomaly rate


# ── Statistical Significance Tests ────────────────────────────────────────────

def run_statistical_tests(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Run normality (Shapiro-Wilk), stationarity (ADF), and homogeneity (Levene) tests.
    
    Returns dict with keys: normality, stationarity, homogeneity.
    """
    result: Dict[str, Any] = {
        "normality": [],
        "stationarity": [],
        "homogeneity": [],
        "meta": {},
    }
    # Limit columns and rows to prevent excessive compute
    num_cols = df.select_dtypes(include="number").columns.tolist()[:_MAX_COLS_FOR_STAT_TESTS]
    sample = df[num_cols].dropna()
    if len(sample) > _MAX_ROWS_FOR_STAT_TESTS:
        sample = sample.sample(_MAX_ROWS_FOR_STAT_TESTS, random_state=42)
    if len(sample) < _MIN_ROWS_FOR_STAT_TESTS:
        result["meta"]["skipped"] = "Too few rows for statistical tests"
        return result

    # ── Normality: Shapiro-Wilk (n<5000) or D'Agostino (n>=5000) ─────────────
    try:
        from scipy import stats as sp_stats
        for col in num_cols:
            try:
                vals = sample[col].dropna()
                if len(vals) < _MIN_ROWS_FOR_STAT_TESTS:
                    continue
                if len(vals) <= 5000:
                    stat, p = sp_stats.shapiro(vals[:5000])
                    test_name = "Shapiro-Wilk"
                else:
                    stat, p = sp_stats.normaltest(vals)
                    test_name = "D'Agostino-Pearson"
                is_normal = bool(p > _NORMALITY_ALPHA)
                result["normality"].append({
                    "column": col,
                    "test": test_name,
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p), 6),
                    "is_normal": is_normal,
                    "interpretation": (
                        f"Distribution appears normal (p={p:.4f} > {_NORMALITY_ALPHA}). "
                        "Parametric tests can be applied."
                        if is_normal else
                        f"Non-normal distribution (p={p:.4f} ≤ {_NORMALITY_ALPHA}). "
                        "Use non-parametric tests (Mann-Whitney, Kruskal-Wallis)."
                    ),
                })
            except Exception as exc:  # noqa: BLE001
                logger.debug("Normality test failed for '%s': %s", col, exc)
    except ImportError:
        logger.debug("scipy not installed — skipping normality tests")

    # ── Stationarity: ADF Test ─────────────────────────────────────────────────
    try:
        from statsmodels.tsa.stattools import adfuller
        for col in num_cols[:10]:  # limit to 10 cols for time series
            try:
                vals = sample[col].dropna()
                if len(vals) < 20:
                    continue
                adf_stat, p, _, _, _, _ = adfuller(vals, autolag="AIC")
                is_stationary = bool(p <= _STATIONARITY_ALPHA)
                result["stationarity"].append({
                    "column": col,
                    "adf_statistic": round(float(adf_stat), 4),
                    "p_value": round(float(p), 6),
                    "is_stationary": is_stationary,
                    "interpretation": (
                        f"Series is stationary (p={p:.4f}). "
                        "Suitable for time series models without differencing."
                        if is_stationary else
                        f"Series is non-stationary (p={p:.4f}). "
                        "Apply differencing or ARIMA models."
                    ),
                })
            except Exception as exc:  # noqa: BLE001
                logger.debug("ADF test failed for '%s': %s", col, exc)
    except ImportError:
        logger.debug("statsmodels not installed — skipping stationarity tests")

    # ── Homogeneity: Levene test (if target_col available) ─────────────────────
    if target_col and target_col in df.columns:
        try:
            from scipy import stats as sp_stats
            groups = df[target_col].dropna().unique()[:5]  # limit to 5 groups
            for col in num_cols[:10]:
                try:
                    group_vals = [df[df[target_col] == g][col].dropna() for g in groups]
                    group_vals = [g for g in group_vals if len(g) > 1]
                    if len(group_vals) < 2:
                        continue
                    lev_stat, p = sp_stats.levene(*group_vals)
                    result["homogeneity"].append({
                        "column": col,
                        "group_column": target_col,
                        "levene_statistic": round(float(lev_stat), 4),
                        "p_value": round(float(p), 6),
                        "equal_variance": bool(p > _NORMALITY_ALPHA),
                        "interpretation": (
                            f"Equal variances across groups (p={p:.4f}). ANOVA assumptions satisfied."
                            if p > _NORMALITY_ALPHA else
                            f"Unequal variances across groups (p={p:.4f}). Use Welch ANOVA or non-parametric tests."
                        ),
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Levene test failed for '%s': %s", col, exc)
        except ImportError:
            pass

    result["meta"]["columns_tested"] = len(num_cols)
    result["meta"]["rows_tested"] = len(sample)
    return result


# ── Feature Importance ────────────────────────────────────────────────────────

def compute_feature_importance(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    n_estimators: int = 50,
    max_features: int = 15,
) -> Dict[str, float]:
    """
    Compute feature importance using:
    - RandomForest (supervised) if target_col is provided and usable
    - MutualInformation (unsupervised) as fallback
    - Correlation-to-first-numeric-col as final fallback

    Returns {feature_name: importance_score} sorted descending, top max_features.
    """
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return {}

    # Supervised path
    if target_col and target_col in df.columns:
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder
            X = num_df.drop(columns=[target_col], errors="ignore").fillna(0)
            y = df[target_col].dropna()
            X = X.loc[y.index]
            if len(X) < 30 or X.shape[1] == 0:
                raise ValueError("Not enough data for RF")
            is_clf = y.nunique() <= 20
            model = (RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                     if is_clf else
                     RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1))
            model.fit(X, y)
            imp = dict(zip(X.columns, model.feature_importances_))
            sorted_imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:max_features])
            logger.info("[FI] Random Forest importance: %d features", len(sorted_imp))
            return {k: round(float(v), 4) for k, v in sorted_imp.items()}
        except Exception as exc:  # noqa: BLE001
            logger.debug("RF importance failed: %s — trying MI fallback", exc)

    # Unsupervised: Mutual Information with a proxy target (first column)
    try:
        from sklearn.feature_selection import mutual_info_regression
        target_proxy = num_df.iloc[:, 0]
        features = num_df.iloc[:, 1:].fillna(0)
        if len(features) < 30 or features.shape[1] == 0:
            raise ValueError("Too few rows or features")
        mi = mutual_info_regression(features, target_proxy, random_state=42)
        imp = dict(zip(features.columns, mi))
        sorted_imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:max_features])
        logger.info("[FI] Mutual Information (proxy): %d features", len(sorted_imp))
        return {k: round(float(v), 4) for k, v in sorted_imp.items()}
    except Exception as exc:  # noqa: BLE001
        logger.debug("MI importance failed: %s — using correlation fallback", exc)

    # Final fallback: absolute correlation with first column
    try:
        corr = num_df.corr().iloc[0].abs().drop(num_df.columns[0]).sort_values(ascending=False)
        result = dict(corr.head(max_features))
        return {k: round(float(v), 4) for k, v in result.items()}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Correlation fallback failed: %s", exc)
        return {}


# ── Bias & Fairness Analysis ──────────────────────────────────────────────────

def run_bias_fairness_analysis(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    sensitive_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run disparate impact (80% rule) and statistical parity analysis.

    Parameters
    ----------
    df             : Input DataFrame
    target_col     : Binary outcome column (e.g., "approved", "loan_granted")
    sensitive_cols : Protected attribute columns (autodetected if None)

    Returns dict with: checked, groups_analyzed, results.
    """
    result: Dict[str, Any] = {
        "checked": False,
        "groups_analyzed": [],
        "results": [],
        "meta": {},
    }

    # Autodetect sensitive columns
    if not sensitive_cols:
        sensitive_hints = {"gender", "race", "ethnicity", "age_group", "nationality",
                           "religion", "disability", "marital_status"}
        sensitive_cols = [
            c for c in df.columns
            if any(h in c.lower() for h in sensitive_hints)
        ]

    if not sensitive_cols:
        result["meta"]["skipped"] = "No protected attribute columns detected"
        return result

    # Autodetect target
    if not target_col:
        binary_hints = {"approved", "accepted", "granted", "hired", "promoted",
                        "defaulted", "rejected", "passed"}
        for col in df.columns:
            if any(h in col.lower() for h in binary_hints) and df[col].nunique() <= 5:
                target_col = col
                break

    if not target_col or target_col not in df.columns:
        result["meta"]["skipped"] = "No binary outcome column found for fairness analysis"
        return result

    result["checked"] = True
    result["groups_analyzed"] = sensitive_cols

    try:
        outcome = pd.to_numeric(df[target_col], errors="coerce")
        # If non-numeric (e.g., "yes/no", "approved/denied"), encode
        if outcome.isna().all():
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            outcome = pd.Series(le.fit_transform(df[target_col].fillna("unknown")), index=df.index)
        overall_rate = outcome.mean() if len(outcome) > 0 else 0.5
    except Exception as exc:  # noqa: BLE001
        result["meta"]["error"] = str(exc)
        return result

    for sens_col in sensitive_cols:
        try:
            groups = df[sens_col].dropna().unique()
            for grp in groups[:10]:  # cap at 10 groups
                mask = df[sens_col] == grp
                grp_outcome = outcome[mask]
                n = len(grp_outcome)
                if n < 10:
                    continue
                grp_rate = grp_outcome.mean()
                parity_ratio = (grp_rate / overall_rate) if overall_rate > 0 else float("nan")
                disparate_impact = (grp_rate / overall_rate) if overall_rate > 0 else float("nan")
                status = "PASS"
                if not math.isnan(disparate_impact):
                    if disparate_impact < _DISPARATE_IMPACT_THRESHOLD:
                        status = "FAIL"
                    elif disparate_impact < 0.9:
                        status = "WARN"

                result["results"].append({
                    "group_column": sens_col,
                    "group_value": str(grp),
                    "sample_size": n,
                    "positive_rate": round(float(grp_rate), 4),
                    "overall_positive_rate": round(float(overall_rate), 4),
                    "parity_ratio": round(float(parity_ratio), 4) if not math.isnan(parity_ratio) else None,
                    "disparate_impact": round(float(disparate_impact), 4) if not math.isnan(disparate_impact) else None,
                    "status": status,
                    "interpretation": (
                        f"Group '{grp}' positive rate {grp_rate:.1%} vs. overall {overall_rate:.1%}. "
                        f"Disparate impact ratio: {parity_ratio:.2f} "
                        f"({'⚠️ Below 80% threshold' if status == 'FAIL' else '✓ Within acceptable range'})."
                    ),
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("Bias calc error for col '%s', group '%s': %s", sens_col, grp, exc)

    return result


# ── Anomaly Deep Dive ─────────────────────────────────────────────────────────

def run_anomaly_deep_dive(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run Isolation Forest + per-column Z-score analysis to identify anomalies.

    Returns dict with: if_contamination, per_column, total_anomalies, anomaly_indices.
    """
    result: Dict[str, Any] = {
        "if_contamination": _ANOMALY_CONTAMINATION,
        "per_column": [],
        "total_anomalies": 0,
        "anomaly_indices": [],
        "method": "IsolationForest + Z-score",
    }

    num_df = df.select_dtypes(include="number").fillna(0)
    if num_df.shape[0] < 20 or num_df.shape[1] == 0:
        result["meta"] = "Too few rows/columns for anomaly analysis"
        return result

    # Per-column Z-score analysis
    for col in num_df.columns:
        try:
            vals = num_df[col].dropna()
            mean, std = vals.mean(), vals.std()
            if std == 0:
                continue
            z_scores = (vals - mean) / std
            anomaly_mask = z_scores.abs() > 3
            anom_count = int(anomaly_mask.sum())
            max_z = float(z_scores.abs().max()) if len(z_scores) > 0 else 0.0
            if anom_count > 0:
                result["per_column"].append({
                    "column": col,
                    "anomaly_count": anom_count,
                    "anomaly_pct": round(anom_count / len(vals) * 100, 2),
                    "z_score_max": round(max_z, 2),
                    "mean": round(float(mean), 4),
                    "std": round(float(std), 4),
                    "interpretation": (
                        f"Column '{col}': {anom_count} value(s) ({anom_count/len(vals):.1%}) are >3σ from mean. "
                        f"Max Z-score: {max_z:.2f}."
                    ),
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("Z-score anomaly failed for col '%s': %s", col, exc)

    # Isolation Forest (global anomaly detection)
    try:
        from sklearn.ensemble import IsolationForest
        sample_df = num_df.sample(min(50_000, len(num_df)), random_state=42)
        isoforest = IsolationForest(contamination=_ANOMALY_CONTAMINATION, random_state=42, n_jobs=-1)
        preds = isoforest.fit_predict(sample_df)
        anomaly_idx = sample_df.index[preds == -1].tolist()
        result["total_anomalies"] = len(anomaly_idx)
        result["anomaly_indices"] = anomaly_idx[:100]  # cap returned indices
        logger.info("[AnomalyDeepDive] IsolationForest: %d anomalies detected", len(anomaly_idx))
    except ImportError:
        logger.debug("sklearn not installed — IsolationForest skipped")
    except Exception as exc:  # noqa: BLE001
        logger.debug("IsolationForest failed: %s", exc)

    # Enrich per_column with IF context
    result["per_column"].sort(key=lambda x: x["z_score_max"], reverse=True)
    return result
