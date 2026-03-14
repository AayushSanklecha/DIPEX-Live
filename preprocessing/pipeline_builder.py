"""
preprocessing/pipeline_builder.py
-----------------------------------
YAML-driven sklearn ColumnTransformer + Pipeline builder.

Reads the `preprocessing` section of config.yaml and produces a
fitted sklearn Pipeline that can be applied to any DataFrame.

Supported transformations:
  numeric  : StandardScaler | MinMaxScaler | RobustScaler | None
  impute   : SimpleImputer (mean/median/most_frequent/constant)
  categorical : OneHotEncoder | OrdinalEncoder | TargetEncoder (if sklearn ≥1.3)
  text     : TfidfVectorizer on a single column
  passthrough: keep raw

Usage::

    from preprocessing.pipeline_builder import PipelineBuilder
    builder = PipelineBuilder(config)
    sklearn_pipeline = builder.build(df, target_col="churn")
    X_transformed = sklearn_pipeline.fit_transform(X_train)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder,
    RobustScaler, StandardScaler,
)

logger = logging.getLogger("dipex.preprocessing.pipeline_builder")


class PipelineBuilder:
    """
    YAML-config-driven sklearn Pipeline builder.

    Parameters
    ----------
    config : dict
        Full DIPEX config (expects config['preprocessing'] section).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config.get("preprocessing", {})

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PipelineBuilder":
        return cls(config)

    def build(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        text_cols: Optional[List[str]] = None,
    ) -> Pipeline:
        """
        Build and return a sklearn Pipeline based on config + DataFrame schema.

        If column lists are not provided, they are auto-detected from dtypes.
        """
        # Auto-detect columns
        exclude = {target_col} if target_col else set()
        if numeric_cols is None:
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
        if categorical_cols is None:
            categorical_cols = [
                c for c in df.select_dtypes(include=["object", "category"]).columns
                if c not in exclude and c not in (text_cols or [])
            ]
        if text_cols is None:
            text_cols = []

        logger.info(
            "Building pipeline: %d numeric, %d categorical, %d text columns.",
            len(numeric_cols), len(categorical_cols), len(text_cols),
        )

        transformers: List[Tuple] = []

        # ── Numeric transformer ───────────────────────────────────────────────
        if numeric_cols:
            impute_strategy = self.cfg.get("cleaning", {})
            if isinstance(impute_strategy, bool): impute_strategy = {}
            impute_strategy = impute_strategy.get("imputation_strategy", "mean") if isinstance(impute_strategy, dict) else "mean"
            if impute_strategy not in ("mean", "median", "most_frequent", "constant"):
                impute_strategy = "mean"

            fe_cfg = self.cfg.get("feature_engineering", {})
            if isinstance(fe_cfg, bool): fe_cfg = {}
            scaler_name = fe_cfg.get("scaler", "standard") if isinstance(fe_cfg, dict) else "standard"
            scaler = {
                "standard": StandardScaler(),
                "minmax":   MinMaxScaler(),
                "robust":   RobustScaler(),
                "none":     "passthrough",
            }.get(scaler_name, StandardScaler())

            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy=impute_strategy)),
                ("scaler",  scaler),
            ])
            transformers.append(("numeric", num_pipe, numeric_cols))

        # ── Categorical transformer ───────────────────────────────────────────
        if categorical_cols:
            fe_cfg = self.cfg.get("feature_engineering", {})
            if isinstance(fe_cfg, bool): fe_cfg = {}
            encoder_name = fe_cfg.get("encoder", "onehot") if isinstance(fe_cfg, dict) else "onehot"

            # Filter out extremely high-cardinality categoricals which would
            # explode OHE output size and slow everything down significantly
            _MAX_OHE_CARDINALITY = 50
            safe_cat_cols = []
            for c in categorical_cols:
                n_uniq = df[c].nunique(dropna=True)
                if n_uniq <= _MAX_OHE_CARDINALITY:
                    safe_cat_cols.append(c)
                else:
                    logger.debug(
                        "[Builder] Dropping cat col '%s' (%d unique > %d OHE limit)",
                        c, n_uniq, _MAX_OHE_CARDINALITY,
                    )
            categorical_cols = safe_cat_cols

            if categorical_cols:
                if encoder_name == "ordinal":
                    encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                else:  # default onehot
                    encoder = OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        max_categories=_MAX_OHE_CARDINALITY,
                    )

                cat_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", encoder),
                ])
                transformers.append(("categorical", cat_pipe, categorical_cols))

        # ── Text transformer ──────────────────────────────────────────────────
        for tcol in text_cols:
            from sklearn.feature_extraction.text import TfidfVectorizer
            text_pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=200, sublinear_tf=True)),
            ])
            transformers.append((f"text_{tcol}", text_pipe, tcol))

        # ── Assemble ColumnTransformer ────────────────────────────────────────
        if not transformers:
            logger.warning("No columns found for pipeline — returning identity passthrough.")
            col_transformer = ColumnTransformer(
                transformers=[("passthrough", "passthrough", [])],
                remainder="passthrough",
            )
        else:
            col_transformer = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                verbose_feature_names_out=False,
            )

        full_pipeline = Pipeline([
            ("preprocessor", col_transformer),
        ])

        logger.info("sklearn Pipeline built successfully.")
        return full_pipeline

    def get_feature_names(
        self,
        fitted_pipeline: Pipeline,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> List[str]:
        """Extract output feature names from a fitted pipeline."""
        try:
            preprocessor = fitted_pipeline.named_steps["preprocessor"]
            return list(preprocessor.get_feature_names_out())
        except Exception:  # noqa: BLE001
            return [f"feature_{i}" for i in range(
                fitted_pipeline.transform(df.head(1)).shape[1]
            )]
