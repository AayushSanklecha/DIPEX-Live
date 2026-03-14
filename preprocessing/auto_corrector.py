"""
preprocessing/auto_corrector.py
-------------------------------
Automatically applies deterministic data corrections based on EDA findings.
Generates an `actions_log` documenting every change made to the DataFrame.

Transformations:
- Skew > 2.0 (all positive): log1p
- 1.0 < Skew <= 2.0 (all positive): sqrt
- null_pct > 20%: ML Imputation (IterativeImputer) + indicator column
- 1% < null_pct <= 20%: median imputation
- zero_pct > 30%: replace 0 with NaN -> median impute
- Constant / All-null: drop column
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class AutoCorrector:
    """
    Applies data preparation fixes automatically, returning the modified
    DataFrame and a log of actions taken.
    """
    
    def __init__(self, target_col: str):
        self.target_col = target_col

    def apply(self, df: pd.DataFrame, eda_report: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
        """
        Apply automatic corrections to df based on eda_report insights.
        Returns:
            (corrected_df, actions_log)
        """
        # Work on a copy to avoid SettingWithCopyWarnings
        df_out = df.copy()
        actions_log: Dict[str, Dict[str, str]] = {}
        
        col_stats = eda_report.get("column_stats", {})
        
        # 1. Process columns for missingness and constants
        cols_to_drop = []
        cols_to_ml_impute = []
        
        for col in df_out.columns:
            if col == self.target_col:
                continue
                
            # Get missing stats
            stats = col_stats.get(col, {})
            null_pct = float(stats.get("null_pct", 0.0))
            
            # Check for drop conditions
            if null_pct == 1.0:
                cols_to_drop.append(col)
                actions_log[col] = {"action": "dropped_all_null", "reason": "100% missing values"}
                continue
                
            # If numeric, check for constant
            if isinstance(stats, dict):
                std = float(stats.get("std", 1.0))
                if std == 0.0 and null_pct < 1.0:
                    cols_to_drop.append(col)
                    actions_log[col] = {"action": "dropped_constant", "reason": "Zero variance (constant value)"}
                    continue
                    
            # 2. Missing value handling
            if null_pct > 0.20:
                # High missingness -> create indicator and flag for ML imputation
                indicator_col = f"{col}_is_missing"
                df_out[indicator_col] = df_out[col].isna().astype(int)
                actions_log[col] = {"action": "ml_imputed_high_missing", "reason": f"{null_pct*100:.1f}% missing, indicator added"}
                if pd.api.types.is_numeric_dtype(df_out[col]):
                    cols_to_ml_impute.append(col)
                else:
                    # For categoricals, just fill with 'MISSING' string if high null
                    df_out[col] = df_out[col].fillna("MISSING")
                    
            elif null_pct > 0.0:
                # Low missingness -> median (numeric) or mode (categorical)
                if pd.api.types.is_numeric_dtype(df_out[col]):
                    median_val = df_out[col].median()
                    df_out[col] = df_out[col].fillna(median_val)
                    actions_log[col] = {"action": "median_imputation", "reason": f"{null_pct*100:.1f}% missing"}
                else:
                    mode_val = df_out[col].mode(dropna=True)
                    if not mode_val.empty:
                        df_out[col] = df_out[col].fillna(mode_val[0])
                        actions_log[col] = {"action": "mode_imputation", "reason": f"{null_pct*100:.1f}% missing"}
        
        # Drop columns
        if cols_to_drop:
            df_out = df_out.drop(columns=cols_to_drop)
            
        # 3. Perform ML Imputation for numeric high-null columns
        if cols_to_ml_impute:
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa
                from sklearn.impute import IterativeImputer
                from sklearn.ensemble import ExtraTreesRegressor
                
                # ML Imputation logic
                # Only use numeric columns for the imputer's training data
                numeric_cols = df_out.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    logger.info("Running IterativeImputer (ML Imputation) on %d columns", len(cols_to_ml_impute))
                    
                    # We use a fast tree-based imputer
                    imputer = IterativeImputer(
                        estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
                        max_iter=5,
                        random_state=42,
                        add_indicator=False # We already added custom indicators
                    )
                    
                    # Fit transform only on numeric part
                    imputed_array = imputer.fit_transform(df_out[numeric_cols])
                    df_out[numeric_cols] = imputed_array
                    
                else:
                    # Fallback if not enough numeric columns
                    for c in cols_to_ml_impute:
                        df_out[c] = df_out[c].fillna(df_out[c].median())
                        actions_log[c] = {"action": "median_imputation", "reason": f"Fallback from ML imputer ({null_pct*100:.1f}% missing)"}
                        
            except ImportError as ie:
                logger.warning("Could not run IterativeImputer due to missing dependency: %s", ie)
                for c in cols_to_ml_impute:
                    df_out[c] = df_out[c].fillna(df_out[c].median())
                    actions_log[c] = {"action": "median_imputation", "reason": "Fallback: ML tools missing"}
            except Exception as e:
                logger.warning("IterativeImputer failed: %s", e)
                for c in cols_to_ml_impute:
                    df_out[c] = df_out[c].fillna(df_out[c].median())
                    actions_log[c] = {"action": "median_imputation", "reason": "Fallback: ML imputation failed"}

        # 4. Shape Transformations and Zero Handling (Numeric only)
        for col, stats in col_stats.items():
            if col == self.target_col or col not in df_out.columns:
                continue
            if not isinstance(stats, dict):
                continue
                
            col_series = df_out[col]
            
            # Suspicious Zero Handling
            zero_pct = float(stats.get("zero_pct", 0.0) or 0.0)
            if zero_pct > 0.30:
                # Check if it's already an indicator-like column (e.g. 0 and 1 only)
                unique_vals = col_series.dropna().unique()
                if len(unique_vals) > 2:
                    # Replace 0 with NaN, then impute
                    df_out[col] = df_out[col].replace(0, np.nan)
                    median_val = df_out[col].median()
                    df_out[col] = df_out[col].fillna(median_val)
                    actions_log[col] = {"action": "zero_to_nan_imputed", "reason": f"Suspicious zeros ({zero_pct*100:.1f}%), replaced with median"}
                    # Refresh series context
                    col_series = df_out[col]
            
            # Skew Transformations
            skew = float(stats.get("skewness", 0.0))
            min_val = float(stats.get("min", col_series.min()))
            
            if min_val >= 0:
                if skew > 2.0:
                    df_out[col] = np.log1p(col_series)
                    # Don't overwrite higher-priority logs (like ml_imputed) unless there is none
                    if col not in actions_log or "imput" not in actions_log[col]["action"]:
                        actions_log[col] = {"action": "log1p_transform", "reason": f"Highly right-skewed (skew={skew:.2f})"}
                    else:
                        actions_log[col]["reason"] += f" + log1p(skew={skew:.2f})"
                        
                elif 1.0 < skew <= 2.0:
                    df_out[col] = np.sqrt(col_series)
                    if col not in actions_log or "imput" not in actions_log[col]["action"]:
                        actions_log[col] = {"action": "sqrt_transform", "reason": f"Right-skewed (skew={skew:.2f})"}
                    else:
                        actions_log[col]["reason"] += f" + sqrt(skew={skew:.2f})"

        return df_out, actions_log
