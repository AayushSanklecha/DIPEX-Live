"""
preprocessing/feature_engineer.py
----------------------------------
Enterprise-grade feature engineering engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.preprocessing.feature_engineer")

@dataclass
class FeatureEngineeringReport:
    run_id: str
    features_added: List[str] = field(default_factory=list)
    transformations_applied: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "features_added": self.features_added,
            "transformations_applied": self.transformations_applied,
            "warnings": self.warnings,
        }

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("preprocessing", {})
        self.enabled: bool = bool(cfg.get("feature_engineering", True))
        self.lag_specs: List[Dict] = cfg.get("lag_features", [])
        self.rolling_specs: List[Dict] = cfg.get("rolling_features", [])
        self.calendar_cols: List[str] = cfg.get("calendar_columns", [])
        self.freq_encode_cols: List[str] = cfg.get("frequency_encode", [])
        self.target_encode_specs: List[Dict] = cfg.get("target_encode", [])
        self.log_transform_cols: List[str] = cfg.get("log_transform", [])
        self.poly_degree: int = int(cfg.get("polynomial_degree", 2))
        self.poly_cols: List[str] = cfg.get("polynomial_columns", [])
        self.binning_specs: List[Dict] = cfg.get("binning", [])
        self.interaction_specs: List[List[str]] = cfg.get("interactions", [])
        self.zscore_cols: List[str] = cfg.get("zscore_scale", [])
        self.minmax_cols: List[str] = cfg.get("minmax_scale", [])
        
        # [ML] Deep Feature Synthesis
        dfs_val = cfg.get("dfs_enabled", cfg.get("deep_feature_synthesis", True))
        self.dfs_enabled:    bool  = bool(dfs_val)
        self.dfs_max_feats:  int   = int(cfg.get("dfs_max_features", 50))
        self.dfs_corr_thresh: float = float(cfg.get("dfs_corr_threshold", 0.05))
        
        # Real-World Data Robustness Toggles
        self.high_cardinality_limit: int = int(cfg.get("high_cardinality_limit", 200))
        self.auto_log_skew_threshold: float = float(cfg.get("auto_log_skew_threshold", 2.0))
        self.handle_class_imbalance: bool = bool(cfg.get("handle_class_imbalance", True))
        self.imbalance_ratio_threshold: float = float(cfg.get("imbalance_ratio_threshold", 5.0))
        self.config = config or {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureEngineer":
        return cls(config)

    def engineer(self, df: pd.DataFrame, run_id: str = "", target_col: Optional[str] = None) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
        if not self.enabled:
            return df, FeatureEngineeringReport(run_id=run_id)

        report = FeatureEngineeringReport(run_id=run_id)
        df = df.copy()

        # [Robustness] Recover numeric types that might have been coerced to object during cleaning/replace
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
                    
        df = self._lag_features(df, report)
        df = self._rolling_features(df, report)
        df = self._calendar_features(df, report)
        df = self._frequency_encode(df, report)
        df = self._target_encode(df, report, target_col)
        df = self._log_transform(df, report)
        df = self._auto_log_correction(df, report)
        df = self._polynomial_features(df, report)
        df = self._binning(df, report)
        df = self._interactions(df, report)
        df = self._zscore_scale(df, report)
        df = self._minmax_scale(df, report)

        if self.dfs_enabled:
            df = self._synthesize_features(df, report, target_col)

        df = self._encode_remaining_objects(df, report)
        
        if self.handle_class_imbalance and target_col and target_col in df.columns:
            df = self._handle_class_imbalance(df, report, target_col)

        logger.info("Feature engineering complete: %d new features added.", len(report.features_added))
        return df, report

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Any:
        """Compatibility alias for AnalyticsOrchestrator."""
        # Wrap the result in a dummy object that matches EngineeredFeatures if needed,
        # or just return the df if the orchestrator is updated.
        # To be safe, let's make it return an object with .df attribute.
        df_out, report = self.engineer(df, target_col=target_col)
        
        class CompatResult:
            def __init__(self, df, report):
                self.df = df
                self.report = report
            def to_dict(self):
                return {
                    "features_added": self.report.features_added,
                    "transformations": self.report.transformations_applied,
                    "net_features_added": len(self.report.features_added)
                }
        return CompatResult(df_out, report)

    def _lag_features(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for spec in self.lag_specs:
            col = spec.get("column")
            lags = spec.get("lags", [1])
            if col not in df.columns: continue
            for lag in lags:
                new_col = f"{col}_lag_{lag}"
                df[new_col] = df[col].shift(lag)
                report.features_added.append(new_col)
        return df

    def _rolling_features(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for spec in self.rolling_specs:
            col = spec.get("column")
            windows = spec.get("windows", [7])
            stats_list = spec.get("stats", ["mean"])
            if col not in df.columns: continue
            for w in windows:
                roller = df[col].rolling(window=w, min_periods=1)
                for stat in stats_list:
                    fn = getattr(roller, stat, None)
                    if fn:
                        new_col = f"{col}_roll{w}_{stat}"
                        df[new_col] = fn()
                        report.features_added.append(new_col)
        return df

    def _calendar_features(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for col in self.calendar_cols:
            if col not in df.columns: continue
            dt = pd.to_datetime(df[col], errors="coerce")
            for attr, new_col in [("year", f"{col}_year"), ("month", f"{col}_month"), ("day", f"{col}_day"), ("dayofweek", f"{col}_dayofweek"), ("quarter", f"{col}_quarter"), ("hour", f"{col}_hour")]:
                df[new_col] = getattr(dt.dt, attr)
                report.features_added.append(new_col)
            df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            report.features_added.append(f"{col}_is_weekend")
        return df

    def _frequency_encode(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for col in self.freq_encode_cols:
            if col not in df.columns: continue
            if df[col].nunique() > self.high_cardinality_limit:
                new_col = f"{col}_hash_enc"
                df[new_col] = df[col].astype(str).apply(lambda x: hash(x) % 1000).astype(int)
                report.features_added.append(new_col)
                df = df.drop(columns=[col])
                continue
            freq_map = df[col].value_counts(normalize=True)
            new_col = f"{col}_freq_enc"
            df[new_col] = df[col].map(freq_map)
            report.features_added.append(new_col)
        return df

    def _target_encode(self, df: pd.DataFrame, report: FeatureEngineeringReport, target_col: Optional[str]) -> pd.DataFrame:
        for spec in self.target_encode_specs:
            col = spec.get("column")
            tgt = spec.get("target") or target_col
            if col not in df.columns or not tgt or tgt not in df.columns: continue
            if df[col].nunique() > self.high_cardinality_limit:
                new_col = f"{col}_hash_enc"
                df[new_col] = df[col].astype(str).apply(lambda x: hash(x) % 1000).astype(int)
                report.features_added.append(new_col)
                df = df.drop(columns=[col])
                continue
            enc_map = df.groupby(col)[tgt].mean()
            new_col = f"{col}_target_enc"
            df[new_col] = df[col].map(enc_map)
            report.features_added.append(new_col)
        return df

    def _log_transform(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for col in self.log_transform_cols:
            if col not in df.columns: continue
            new_col = f"{col}_log1p"
            df[new_col] = np.log1p(df[col].clip(lower=0))
            report.features_added.append(new_col)
        return df

    def _auto_log_correction(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        if self.auto_log_skew_threshold <= 0: return df
        num_cols = df.select_dtypes(include="number").columns
        for col in num_cols:
            if col.endswith("_log1p") or f"{col}_log1p" in df.columns or f"{col}_auto_log1p" in df.columns:
                continue
            try:
                skew = df[col].skew()
                if not np.isnan(skew) and abs(skew) > self.auto_log_skew_threshold:
                    # Only apply if all non-null values are non-negative (log1p safety)
                    if (df[col].dropna() >= 0).all():
                        new_col = f"{col}_auto_log1p"
                        df[new_col] = np.log1p(df[col])
                        # Force numeric type to avoid object coercion with NaNs
                        df[new_col] = pd.to_numeric(df[new_col], errors="coerce")
                        logger.debug("Applied auto-log to %s, skew=%.2f", col, skew)
                        report.features_added.append(new_col)
                        report.transformations_applied.append({"type": "auto_log1p", "column": col, "skew": float(skew)})
            except Exception: continue
        return df

    def _polynomial_features(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        if not self.poly_cols: return df
        try:
            from sklearn.preprocessing import PolynomialFeatures
            cols = [c for c in self.poly_cols if c in df.columns]
            if not cols: return df
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            poly_arr = poly.fit_transform(df[cols].fillna(0))
            poly_names = poly.get_feature_names_out(cols)
            for i, name in enumerate(poly_names):
                if name not in cols:
                    df[name] = poly_arr[:, i]
                    report.features_added.append(name)
        except Exception as exc:
            logger.warning("[FeatureEngineer] Polynomial features failed (non-fatal): %s", exc)
            report.warnings.append(f"polynomial_features error: {exc}")
        return df

    def _binning(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for spec in self.binning_specs:
            col, bins, strategy = spec.get("column"), int(spec.get("bins", 5)), spec.get("strategy", "quantile")
            if col not in df.columns: continue
            new_col = f"{col}_bin{bins}"
            try:
                if strategy == "quantile": df[new_col] = pd.qcut(df[col], q=bins, labels=False, duplicates="drop")
                else: df[new_col] = pd.cut(df[col], bins=bins, labels=False)
                report.features_added.append(new_col)
            except Exception as exc:
                logger.warning("[FeatureEngineer] Binning failed for '%s' (non-fatal): %s", col, exc)
                report.warnings.append(f"binning error on '{col}': {exc}")
        return df

    def _interactions(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for pair in self.interaction_specs:
            if len(pair) < 2: continue
            c1, c2 = pair[0], pair[1]
            if c1 not in df.columns or c2 not in df.columns: continue
            new_col = f"{c1}_x_{c2}"
            df[new_col] = df[c1] * df[c2]
            report.features_added.append(new_col)
        return df

    def _zscore_scale(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for col in self.zscore_cols:
            if col not in df.columns: continue
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                new_col = f"{col}_zscore"
                df[new_col] = (df[col] - mean) / std
                report.features_added.append(new_col)
        return df

    def _minmax_scale(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for col in self.minmax_cols:
            if col not in df.columns: continue
            mn, mx = df[col].min(), df[col].max()
            if mx > mn:
                new_col = f"{col}_minmax"
                df[new_col] = (df[col] - mn) / (mx - mn)
                report.features_added.append(new_col)
        return df

    def _synthesize_features(self, df: pd.DataFrame, report: FeatureEngineeringReport, target_col: Optional[str] = None) -> pd.DataFrame:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if target_col and target_col in num_cols: num_cols.remove(target_col)
        if len(num_cols) < 2: return df
        try:
            corr = df[num_cols].corr().abs()
            added = 0
            for i, c1 in enumerate(num_cols):
                for c2 in num_cols[i + 1:]:
                    if added >= self.dfs_max_feats: break
                    corr_val = corr.loc[c1, c2] if c1 in corr.index and c2 in corr.columns else 0.0
                    if not np.isnan(corr_val) and corr_val >= self.dfs_corr_thresh:
                        # Use per-column median instead of 0 — avoids distributional distortion
                        fill_c1 = df[c1].median() if df[c1].notna().any() else 0.0
                        fill_c2 = df[c2].median() if df[c2].notna().any() else 0.0
                        df[f"dfs_{c1}_div_{c2}"] = df[c1].fillna(fill_c1) / (df[c2].fillna(fill_c2).abs() + 1e-8)
                        df[f"dfs_{c1}_mul_{c2}"] = df[c1].fillna(fill_c1) * df[c2].fillna(fill_c2)
                        added += 2
                if added >= self.dfs_max_feats: break
        except Exception as exc:
            logger.warning("[FeatureEngineer] Deep Feature Synthesis failed (non-fatal): %s", exc)
            report.warnings.append(f"dfs_synthesis error: {exc}")
        return df

    def _encode_remaining_objects(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        object_cols = df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            n_unique = df[col].nunique()
            if n_unique > self.high_cardinality_limit:
                new_col = f"{col}_hash_enc"
                df[new_col] = df[col].astype(str).apply(lambda x: hash(x) % 1000).astype(int)
                df = df.drop(columns=[col])
                report.features_added.append(new_col)
            else:
                try:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str).fillna("UNKNOWN"))
                except Exception as exc:
                    logger.warning("[FeatureEngineer] LabelEncoder failed for '%s' (non-fatal): %s", col, exc)
                    report.warnings.append(f"label_encode error on '{col}': {exc}")
        return df

    def _handle_class_imbalance(self, df: pd.DataFrame, report: FeatureEngineeringReport, target_col: str) -> pd.DataFrame:
        if df[target_col].nunique() > 10: return df
        v_counts = df[target_col].value_counts()
        if len(v_counts) < 2: return df
        ratio = v_counts.iloc[0] / (v_counts.iloc[-1] or 1)
        if ratio > self.imbalance_ratio_threshold:
            try:
                from imblearn.over_sampling import SMOTE
                X = df.drop(columns=[target_col])
                y = df[target_col]
                num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
                logger.debug("SMOTE: Processing %d numeric columns: %s", len(num_cols), num_cols)
                X_num = X[num_cols].fillna(0)
                # Clamp k_neighbors: must be >= 1 and < minority class sample count
                minority_count = int(v_counts.iloc[-1])
                k = max(1, min(5, minority_count - 1))
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_res, y_res = smote.fit_resample(X_num, y)
                logger.debug("SMOTE: Resampled shape: %s", X_res.shape)
                df_res = pd.DataFrame(X_res, columns=num_cols)
                df_res[target_col] = y_res
                other_cols = [c for c in X.columns if c not in num_cols]
                for c in other_cols: df_res[c] = df[c].mode()[0] if not df[c].empty else 0
                return df_res
            except Exception as exc:
                logger.warning("[FeatureEngineer] SMOTE/resample failed (non-fatal): %s", exc)
                report.warnings.append(f"class_imbalance error: {exc}")
        return df
