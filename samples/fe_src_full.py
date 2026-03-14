class FeatureEngineer:
    """
    Declarative feature engineering engine driven by config YAML.

    Config stanza example::

        preprocessing:
          feature_engineering: true
          lag_features:
            - column: sales
              lags: [1, 7, 14]
          rolling_features:
            - column: sales
              windows: [7, 30]
              stats: [mean, std]
          calendar_columns: [order_date]
          frequency_encode: [category, region]
          target_encode:
            - column: segment
              target: revenue
          log_transform: [amount, revenue]
          polynomial_degree: 2
          polynomial_columns: [age, income]
          binning:
            - column: age
              bins: 5
              strategy: quantile   # 'quantile' or 'uniform'
          interactions:
            - [amount, frequency]
          zscore_scale: [amount]
          minmax_scale: [age]
    """

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
        self.dfs_enabled:    bool  = bool(cfg.get("deep_feature_synthesis", True))
        self.dfs_max_feats:  int   = int(cfg.get("dfs_max_features", 50))
        self.dfs_corr_thresh: float = float(cfg.get("dfs_corr_threshold", 0.05))
        
        # Real-World Data Robustness Toggles
        self.high_cardinality_limit: int = int(cfg.get("high_cardinality_limit", 200))
        self.auto_log_skew_threshold: float = float(cfg.get("auto_log_skew_threshold", 2.0))
        self.handle_class_imbalance: bool = bool(cfg.get("handle_class_imbalance", True))
        self.imbalance_ratio_threshold: float = float(cfg.get("imbalance_ratio_threshold", 5.0))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureEngineer":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def engineer(
        self,
        df: pd.DataFrame,
        run_id: str = "",
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
        """Apply all configured feature engineering transforms."""
        if not self.enabled:
            return df, FeatureEngineeringReport(run_id=run_id)

        report = FeatureEngineeringReport(run_id=run_id)
        df = df.copy()

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

        # [ML] Deep Feature Synthesis — automated interaction feature generation
        if self.dfs_enabled:
            df = self._synthesize_features(df, report, target_col)

        # Prevent string category crashes in downstream tree models 
        df = self._encode_remaining_objects(df, report)
        
        # Handle severe class skews
        if self.handle_class_imbalance and target_col and target_col in df.columns:
            df = self._handle_class_imbalance(df, report, target_col)

        logger.info(
            "Feature engineering complete: %d new features added.",
            len(report.features_added),
        )
        return df, report

    # ── Private transforms ───────────────────────────────────────────────────

    def _lag_features(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for spec in self.lag_specs:
            col = spec.get("column")
            lags = spec.get("lags", [1])
            if col not in df.columns:
                report.warnings.append(f"Lag: column '{col}' not found.")
                continue
            for lag in lags:
                new_col = f"{col}_lag_{lag}"
                df[new_col] = df[col].shift(lag)
                report.features_added.append(new_col)
                report.transformations_applied.append({"type": "lag", "column": col, "lag": lag, "new_col": new_col})
        return df

    def _rolling_features(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for spec in self.rolling_specs:
            col = spec.get("column")
            windows = spec.get("windows", [7])
            stats_list = spec.get("stats", ["mean"])
            if col not in df.columns:
                report.warnings.append(f"Rolling: column '{col}' not found.")
                continue
            for w in windows:
                roller = df[col].rolling(window=w, min_periods=1)
                for stat in stats_list:
                    fn = getattr(roller, stat, None)
                    if fn is None:
                        report.warnings.append(f"Rolling stat '{stat}' not supported.")
                        continue
                    new_col = f"{col}_roll{w}_{stat}"
                    df[new_col] = fn()
                    report.features_added.append(new_col)
                    report.transformations_applied.append({"type": "rolling", "column": col, "window": w, "stat": stat, "new_col": new_col})
        return df

    def _calendar_features(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for col in self.calendar_cols:
            if col not in df.columns:
                report.warnings.append(f"Calendar: column '{col}' not found.")
                continue
            dt = pd.to_datetime(df[col], errors="coerce")
            for attr, new_col in [
                ("year", f"{col}_year"),
                ("month", f"{col}_month"),
                ("day", f"{col}_day"),
                ("dayofweek", f"{col}_dayofweek"),
                ("quarter", f"{col}_quarter"),
                ("hour", f"{col}_hour"),
            ]:
                df[new_col] = getattr(dt.dt, attr)
                report.features_added.append(new_col)
            df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            report.features_added.append(f"{col}_is_weekend")
            report.transformations_applied.append({"type": "calendar", "column": col})
        return df

    def _frequency_encode(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for col in self.freq_encode_cols:
            if col not in df.columns:
                report.warnings.append(f"Freq encode: column '{col}' not found.")
                continue
            
            # High-Cardinality Guard
            if df[col].nunique() > self.high_cardinality_limit:
                new_col = f"{col}_hash_enc"
                df[new_col] = df[col].astype(str).apply(lambda x: hash(x) % 64).astype(int)
                report.features_added.append(new_col)
                report.transformations_applied.append({
                    "type": "hash_encoding", "column": col, "new_col": new_col, 
                    "reason": f"cardinality ({df[col].nunique()}) > limit ({self.high_cardinality_limit})"
                })
                # We drop the original early to save memory here
                df = df.drop(columns=[col])
                continue
                
            freq_map = df[col].value_counts(normalize=True)
            new_col = f"{col}_freq_enc"
            df[new_col] = df[col].map(freq_map)
            report.features_added.append(new_col)
            report.transformations_applied.append({"type": "frequency_encoding", "column": col, "new_col": new_col})
        return df

    def _target_encode(
        self, df: pd.DataFrame, report: FeatureEngineeringReport, target_col: Optional[str]
    ) -> pd.DataFrame:
        for spec in self.target_encode_specs:
            col = spec.get("column")
            tgt = spec.get("target") or target_col
            if col not in df.columns:
                report.warnings.append(f"Target encode: column '{col}' not found.")
                continue
            if not tgt or tgt not in df.columns:
                report.warnings.append(f"Target encode: target '{tgt}' not found.")
                continue
                
            # High-Cardinality Guard
            if df[col].nunique() > self.high_cardinality_limit:
                new_col = f"{col}_hash_enc"
                df[new_col] = df[col].astype(str).apply(lambda x: hash(x) % 64).astype(int)
                report.features_added.append(new_col)
                report.transformations_applied.append({
                    "type": "hash_encoding", "column": col, "new_col": new_col, 
                    "reason": f"cardinality ({df[col].nunique()}) > limit ({self.high_cardinality_limit})"
                })
                df = df.drop(columns=[col])
                continue
                
            enc_map = df.groupby(col)[tgt].mean()
            new_col = f"{col}_target_enc"
            df[new_col] = df[col].map(enc_map)
            report.features_added.append(new_col)
            report.transformations_applied.append({"type": "target_encoding", "column": col, "target": tgt, "new_col": new_col})
        return df

    def _log_transform(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for col in self.log_transform_cols:
            if col not in df.columns:
                report.warnings.append(f"Log transform: column '{col}' not found.")
                continue
            new_col = f"{col}_log1p"
            df[new_col] = np.log1p(df[col].clip(lower=0))
            report.features_added.append(new_col)
            report.transformations_applied.append({"type": "log1p", "column": col, "new_col": new_col})
        return df

    def _auto_log_correction(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        """[ML] Auto-detect heavily skewed numeric columns and apply log1p to normalize."""
        if self.auto_log_skew_threshold <= 0:
            return df
            
        num_cols = df.select_dtypes(include="number").columns
        for col in num_cols:
            # Skip if already log-transformed manually or if too many zeros/negatives
            if col.endswith("_log1p") or f"{col}_log1p" in df.columns:
                continue
            
            try:
                skew = df[col].skew()
                if not np.isnan(skew) and abs(skew) > self.auto_log_skew_threshold:
                    # Only apply if all non-null values are non-negative (log1p safety)
                    if (df[col].dropna() >= 0).all():
                        new_col = f"{col}_auto_log1p"
                        df[new_col] = np.log1p(df[col])
                        report.features_added.append(new_col)
                        report.transformations_applied.append({
                            "type": "auto_log1p", 
                            "column": col, 
                            "skew": float(skew), 
                            "new_col": new_col
                        })
                        logger.info(f"[ML] Auto-skew corrected '{col}' (skew={skew:.2f}) -> {new_col}")
            except Exception: # noqa: BLE001
                continue
        return df

    def _polynomial_features(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        if not self.poly_cols:
            return df
        try:
            from sklearn.preprocessing import PolynomialFeatures
        except ImportError:
            report.warnings.append("Polynomial features skipped — scikit-learn not installed.")
            return df
        cols = [c for c in self.poly_cols if c in df.columns]
        if not cols:
            return df
        poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False, interaction_only=False)
        poly_arr = poly.fit_transform(df[cols].fillna(0))
        poly_names = poly.get_feature_names_out(cols)
        for i, name in enumerate(poly_names):
            if name in cols:
                continue  # skip originals
            df[name] = poly_arr[:, i]
            report.features_added.append(name)
        report.transformations_applied.append({"type": "polynomial", "degree": self.poly_degree, "columns": cols})
        return df

    def _binning(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for spec in self.binning_specs:
            col = spec.get("column")
            bins = int(spec.get("bins", 5))
            strategy = spec.get("strategy", "quantile")
            if col not in df.columns:
                report.warnings.append(f"Binning: column '{col}' not found.")
                continue
            new_col = f"{col}_bin{bins}"
            try:
                if strategy == "quantile":
                    df[new_col] = pd.qcut(df[col], q=bins, labels=False, duplicates="drop")
                else:
                    df[new_col] = pd.cut(df[col], bins=bins, labels=False)
                report.features_added.append(new_col)
                report.transformations_applied.append({"type": "binning", "column": col, "bins": bins, "strategy": strategy, "new_col": new_col})
            except Exception as exc:  # noqa: BLE001
                report.warnings.append(f"Binning failed for '{col}': {exc}")
        return df

    def _interactions(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for pair in self.interaction_specs:
            if len(pair) < 2:
                continue
            c1, c2 = pair[0], pair[1]
            if c1 not in df.columns or c2 not in df.columns:
                report.warnings.append(f"Interaction: '{c1}' or '{c2}' not found.")
                continue
            new_col = f"{c1}_x_{c2}"
            df[new_col] = df[c1] * df[c2]
            report.features_added.append(new_col)
            report.transformations_applied.append({"type": "interaction", "col1": c1, "col2": c2, "new_col": new_col})
        return df

    def _zscore_scale(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for col in self.zscore_cols:
            if col not in df.columns:
                report.warnings.append(f"Z-score scale: column '{col}' not found.")
                continue
            mean, std = df[col].mean(), df[col].std()
            if std == 0:
                continue
            new_col = f"{col}_zscore"
            df[new_col] = (df[col] - mean) / std
            report.features_added.append(new_col)
            report.transformations_applied.append({"type": "zscore", "column": col, "mean": float(mean), "std": float(std), "new_col": new_col})
        return df

    def _minmax_scale(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        for col in self.minmax_cols:
            if col not in df.columns:
                report.warnings.append(f"MinMax scale: column '{col}' not found.")
                continue
            mn, mx = df[col].min(), df[col].max()
            r = mx - mn
            if r == 0:
                continue
            new_col = f"{col}_minmax"
            df[new_col] = (df[col] - mn) / r
            report.features_added.append(new_col)
            report.transformations_applied.append({"type": "minmax", "column": col, "min": float(mn), "max": float(mx), "new_col": new_col})
        return df

    def _synthesize_features(
        self,
        df:         pd.DataFrame,
        report:     FeatureEngineeringReport,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        [ML] Deep Feature Synthesis — automatically generate interaction features
        between numeric column pairs that show meaningful correlation.

        Generated transforms (per correlated pair):
          - ratio:   col_a / (col_b + eps)
          - product: col_a * col_b
          - log of each (if all-positive)
        """
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if target_col and target_col in num_cols:
            num_cols = [c for c in num_cols if c != target_col]

        if len(num_cols) < 2:
            return df

        added = 0
        cap = self.dfs_max_feats

        # Correlation-guided pair selection
        try:
            corr = df[num_cols].corr().abs()
        except Exception:  # noqa: BLE001
            return df

        seen: set = set()
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i + 1:]:
                if added >= cap:
                    break
                pair_key = tuple(sorted((c1, c2)))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                corr_val = corr.loc[c1, c2] if c1 in corr.index and c2 in corr.columns else 0.0
                if np.isnan(corr_val) or corr_val < self.dfs_corr_thresh:
                    continue

                s1, s2 = df[c1].fillna(0), df[c2].fillna(0)

                # Ratio feature
                if added < cap:
                    col_name = f"dfs_{c1}_div_{c2}"
                    df[col_name] = s1 / (s2.abs() + 1e-8)
                    report.features_added.append(col_name)
                    report.transformations_applied.append({"type": "dfs_ratio", "c1": c1, "c2": c2})
                    added += 1

                # Product feature
                if added < cap:
                    col_name = f"dfs_{c1}_mul_{c2}"
                    df[col_name] = s1 * s2
                    report.features_added.append(col_name)
                    report.transformations_applied.append({"type": "dfs_product", "c1": c1, "c2": c2})
                    added += 1

            if added >= cap:
                break

        logger.info("[ML] DFS: generated %d interaction features from %d numeric columns.",
                    added, len(num_cols))
        return df

    def _encode_remaining_objects(self, df: pd.DataFrame, report: FeatureEngineeringReport) -> pd.DataFrame:
        """Fallback: label encodes any remaining raw string columns so models don't crash."""
        object_cols = df.select_dtypes(include=["object"]).columns
        if not len(object_cols):
            return df
            
        try:
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            return df
            
        for col in object_cols:
            n_unique = df[col].nunique()
            if n_unique > self.high_cardinality_limit:
                new_col = f"{col}_hash_enc"
                df[new_col] = df[col].apply(lambda x: hash(str(x)) % 1000)
                df.drop(columns=[col], inplace=True)
                report.features_added.append(new_col)
                report.transformations_applied.append({
                    "type": "auto_hash_fallback", 
                    "column": col, 
                    "unique_count": n_unique
                })
                logger.info(f"[ML] High-cardinality guard: converted '{col}' ({n_unique} unique) to hash-encoding.")
            else:
                le = LabelEncoder()
                # Convert anything left to string avoiding NaNs breaking the encoder
                df[col] = le.fit_transform(df[col].astype(str).fillna("UNKNOWN"))
                report.transformations_applied.append({
                    "type": "label_encoding_fallback", 
                    "column": col, 
                    "classes_count": len(le.classes_)
                })
            
        return df

    def _handle_class_imbalance(
        self, df: pd.DataFrame, report: FeatureEngineeringReport, target_col: str
    ) -> pd.DataFrame:
        """Detect severe classification target imbalance and apply SMOTE to rebalance."""
        if not pd.api.types.is_numeric_dtype(df[target_col]) and not pd.api.types.is_object_dtype(df[target_col]):
            return df
            
        # Only classification tasks (continuous targets have high unqiue counts)
        if df[target_col].nunique() > 10:
            return df
            
        value_counts = df[target_col].value_counts()
        if len(value_counts) < 2:
            return df
            
        majority_count = value_counts.iloc[0]
        minority_count = value_counts.iloc[-1]
        
        if minority_count == 0:
            return df
            
        imbalance_ratio = majority_count / minority_count
        
        if imbalance_ratio > self.imbalance_ratio_threshold:
            try:
                from imblearn.over_sampling import SMOTE
                logger.info(f"Target '{target_col}' severely imbalanced (ratio {imbalance_ratio:.1f}:1). Applying SMOTE.")
                
                # Separate features to ensure we only apply SMOTE on numbers
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # Drop non-numerics for SMOTE if any are left
                num_cols = X.select_dtypes(include="number").columns
                X_num = X[num_cols]
                
                # We need at least 6 samples in minority class to run default SMOTE with k_neighbors=5
                k_neighbors = min(5, minority_count - 1)
                if k_neighbors < 1:
                     report.warnings.append(f"Imbalance ratio {imbalance_ratio:.1f}:1 bypasses SMOTE (minority count {minority_count} too small).")
                     return df
                     
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X_num.fillna(0), y)
                
                # Merge back
                df_resampled = pd.DataFrame(X_resampled, columns=num_cols.tolist())
                df_resampled[target_col] = y_resampled
                
                # Carry over non-numerics by repeating/padding (a bit naive but safe) - we already label encoded so should be empty
                other_cols = [c for c in X.columns if c not in num_cols]
                for c in other_cols:
                    # Not ideal for SMOTE but works as a fallback
                    df_resampled[c] = df[c].mode()[0] if not df[c].empty else 0
                    
                report.transformations_applied.append({
                    "type": "smote_rebalancing",
                    "target_column": target_col,
                    "imbalance_ratio_before": float(imbalance_ratio),
                    "rows_added": len(df_resampled) - len(df)
                })
                
                return df_resampled
            except ImportError:
                report.warnings.append(f"Severe imbalance ({imbalance_ratio:.1f}:1) detected but imbalanced-learn is not installed.")
                
        return df
