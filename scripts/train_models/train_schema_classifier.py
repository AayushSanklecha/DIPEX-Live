"""
scripts/train_models/train_schema_classifier.py
------------------------------------------------
Colab-ready training script for the Schema Classifier.

Goal: given a DataFrame's column metadata (dtypes, cardinality, null rate,
sample values), predict the domain schema type:
  - tabular_finance | tabular_banking | tabular_healthcare |
    time_series | geospatial | survey | ecommerce | generic

Anti-Overfitting: 70/15/15 stratified split, class-weighted loss,
L2 regularization, early stopping.

Usage (Google Colab):
    !pip install lightgbm scikit-learn pandas numpy joblib
    !python train_schema_classifier.py --data schema_samples.csv --out models/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_schema_classifier")

# ── Schema labels ─────────────────────────────────────────────────────────────

SCHEMA_LABELS = [
    "tabular_finance", "tabular_banking", "tabular_healthcare",
    "time_series", "geospatial", "survey", "ecommerce", "generic",
]

# ── Feature extraction from column metadata ───────────────────────────────────

def extract_column_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-column features that describe the schema.

    Returns a 1-row DataFrame representing this dataset's schema profile.
    """
    import re
    n_rows, n_cols = df.shape
    num_cols   = df.select_dtypes(include="number").columns.tolist()
    cat_cols   = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols  = df.select_dtypes(include=["datetime"]).columns.tolist()

    # Column name pattern counts
    def _count_pattern(cols, pat):
        r = re.compile(pat, re.I)
        return sum(1 for c in cols if r.search(c))

    features = {
        "n_cols":           n_cols,
        "n_rows":           min(n_rows, 1_000_000),
        "numeric_ratio":    len(num_cols) / max(n_cols, 1),
        "categorical_ratio": len(cat_cols) / max(n_cols, 1),
        "datetime_ratio":   len(date_cols) / max(n_cols, 1),
        "overall_null_rate": float(df.isnull().mean().mean()),
        "mean_cardinality": float(
            df[cat_cols].apply(lambda s: s.nunique()).mean()
            if cat_cols else 0.0
        ),
        # Domain keyword counts in column names
        "has_finance_cols": _count_pattern(df.columns, r"revenue|profit|ebitda|eps|market.cap|p.e.ratio"),
        "has_banking_cols": _count_pattern(df.columns, r"loan|transaction|balance|aml|kyc|tier|collateral|repayment"),
        "has_health_cols":  _count_pattern(df.columns, r"patient|diagnosis|icd|drug|dosage|vital|bmi|blood"),
        "has_geo_cols":     _count_pattern(df.columns, r"lat|lon|latitude|longitude|geom|postcode|zipcode|city|country"),
        "has_ts_cols":      _count_pattern(df.columns, r"date|time|timestamp|dt|period|year|month|quarter"),
        "has_survey_cols":  _count_pattern(df.columns, r"q\d+|question|response|likert|rating|scale|agree"),
        "has_ecom_cols":    _count_pattern(df.columns, r"sku|product|cart|order|basket|checkout|customer|refund|chargeback"),
        "has_id_cols":      _count_pattern(df.columns, r"\bid\b|uuid|key|pk|surrogate"),
        "has_flag_cols":    _count_pattern(df.columns, r"flag|is_|has_|bool"),
        "has_amount_cols":  _count_pattern(df.columns, r"amount|price|cost|fee|total|value|sum"),
    }
    return pd.DataFrame([features])


def build_synthetic_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Build a synthetic labelled dataset for schema classification training.
    Each sample is a metadata feature vector describing one dataset's schema.
    """
    rng = np.random.RandomState(42)
    records = []

    label_generators = {
        "tabular_finance": lambda: {
            "n_cols": rng.randint(6, 25), "n_rows": rng.randint(500, 100_000),
            "numeric_ratio": rng.uniform(0.6, 0.95), "categorical_ratio": rng.uniform(0.02, 0.15),
            "datetime_ratio": rng.uniform(0.0, 0.1), "overall_null_rate": rng.uniform(0.0, 0.15),
            "mean_cardinality": rng.uniform(2, 10), "has_finance_cols": rng.randint(2, 7),
            "has_banking_cols": 0, "has_health_cols": 0, "has_geo_cols": 0,
            "has_ts_cols": rng.randint(0, 2), "has_survey_cols": 0, "has_ecom_cols": 0,
            "has_id_cols": rng.randint(0, 2), "has_flag_cols": rng.randint(0, 2),
            "has_amount_cols": rng.randint(1, 4),
        },
        "tabular_banking": lambda: {
            "n_cols": rng.randint(8, 30), "n_rows": rng.randint(1000, 500_000),
            "numeric_ratio": rng.uniform(0.5, 0.85), "categorical_ratio": rng.uniform(0.1, 0.25),
            "datetime_ratio": rng.uniform(0.05, 0.2), "overall_null_rate": rng.uniform(0.0, 0.20),
            "mean_cardinality": rng.uniform(2, 8), "has_finance_cols": rng.randint(0, 3),
            "has_banking_cols": rng.randint(3, 9), "has_health_cols": 0, "has_geo_cols": 0,
            "has_ts_cols": rng.randint(1, 4), "has_survey_cols": 0, "has_ecom_cols": 0,
            "has_id_cols": rng.randint(1, 3), "has_flag_cols": rng.randint(0, 2),
            "has_amount_cols": rng.randint(2, 5),
        },
        "tabular_healthcare": lambda: {
            "n_cols": rng.randint(8, 40), "n_rows": rng.randint(100, 50_000),
            "numeric_ratio": rng.uniform(0.3, 0.6), "categorical_ratio": rng.uniform(0.2, 0.45),
            "datetime_ratio": rng.uniform(0.05, 0.15), "overall_null_rate": rng.uniform(0.05, 0.35),
            "mean_cardinality": rng.uniform(3, 25), "has_finance_cols": 0,
            "has_banking_cols": 0, "has_health_cols": rng.randint(3, 9), "has_geo_cols": 0,
            "has_ts_cols": rng.randint(0, 3), "has_survey_cols": 0, "has_ecom_cols": 0,
            "has_id_cols": rng.randint(1, 3), "has_flag_cols": rng.randint(0, 3),
            "has_amount_cols": rng.randint(0, 2),
        },
        "time_series": lambda: {
            "n_cols": rng.randint(2, 12), "n_rows": rng.randint(100, 1_000_000),
            "numeric_ratio": rng.uniform(0.6, 1.0), "categorical_ratio": rng.uniform(0.0, 0.1),
            "datetime_ratio": rng.uniform(0.1, 0.5), "overall_null_rate": rng.uniform(0.0, 0.1),
            "mean_cardinality": rng.uniform(2, 5), "has_finance_cols": rng.randint(0, 2),
            "has_banking_cols": 0, "has_health_cols": 0, "has_geo_cols": 0,
            "has_ts_cols": rng.randint(2, 6), "has_survey_cols": 0, "has_ecom_cols": 0,
            "has_id_cols": rng.randint(0, 2), "has_flag_cols": 0,
            "has_amount_cols": rng.randint(0, 2),
        },
        "geospatial": lambda: {
            "n_cols": rng.randint(3, 15), "n_rows": rng.randint(100, 500_000),
            "numeric_ratio": rng.uniform(0.4, 0.8), "categorical_ratio": rng.uniform(0.1, 0.3),
            "datetime_ratio": rng.uniform(0.0, 0.1), "overall_null_rate": rng.uniform(0.0, 0.2),
            "mean_cardinality": rng.uniform(2, 50), "has_finance_cols": 0,
            "has_banking_cols": 0, "has_health_cols": 0, "has_geo_cols": rng.randint(2, 5),
            "has_ts_cols": rng.randint(0, 2), "has_survey_cols": 0, "has_ecom_cols": 0,
            "has_id_cols": rng.randint(0, 2), "has_flag_cols": 0,
            "has_amount_cols": 0,
        },
        "survey": lambda: {
            "n_cols": rng.randint(10, 60), "n_rows": rng.randint(50, 10_000),
            "numeric_ratio": rng.uniform(0.2, 0.6), "categorical_ratio": rng.uniform(0.3, 0.7),
            "datetime_ratio": rng.uniform(0.0, 0.05), "overall_null_rate": rng.uniform(0.05, 0.3),
            "mean_cardinality": rng.uniform(2, 7), "has_finance_cols": 0,
            "has_banking_cols": 0, "has_health_cols": 0, "has_geo_cols": 0,
            "has_ts_cols": 0, "has_survey_cols": rng.randint(3, 12), "has_ecom_cols": 0,
            "has_id_cols": rng.randint(0, 1), "has_flag_cols": rng.randint(0, 2),
            "has_amount_cols": 0,
        },
        "ecommerce": lambda: {
            "n_cols": rng.randint(8, 30), "n_rows": rng.randint(500, 1_000_000),
            "numeric_ratio": rng.uniform(0.4, 0.75), "categorical_ratio": rng.uniform(0.15, 0.4),
            "datetime_ratio": rng.uniform(0.05, 0.15), "overall_null_rate": rng.uniform(0.0, 0.2),
            "mean_cardinality": rng.uniform(3, 30), "has_finance_cols": 0,
            "has_banking_cols": 0, "has_health_cols": 0, "has_geo_cols": rng.randint(0, 2),
            "has_ts_cols": rng.randint(1, 3), "has_survey_cols": 0, "has_ecom_cols": rng.randint(3, 8),
            "has_id_cols": rng.randint(1, 3), "has_flag_cols": rng.randint(0, 2),
            "has_amount_cols": rng.randint(2, 5),
        },
        "generic": lambda: {
            "n_cols": rng.randint(2, 20), "n_rows": rng.randint(10, 100_000),
            "numeric_ratio": rng.uniform(0.1, 0.9), "categorical_ratio": rng.uniform(0.05, 0.8),
            "datetime_ratio": rng.uniform(0.0, 0.3), "overall_null_rate": rng.uniform(0.0, 0.4),
            "mean_cardinality": rng.uniform(2, 100), "has_finance_cols": 0,
            "has_banking_cols": 0, "has_health_cols": 0, "has_geo_cols": 0,
            "has_ts_cols": 0, "has_survey_cols": 0, "has_ecom_cols": 0,
            "has_id_cols": rng.randint(0, 2), "has_flag_cols": rng.randint(0, 2),
            "has_amount_cols": 0,
        },
    }

    per_label = n_samples // len(SCHEMA_LABELS)
    for label in SCHEMA_LABELS:
        gen = label_generators[label]
        for _ in range(per_label):
            rec = gen()
            rec["schema_label"] = label
            records.append(rec)

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def train(args) -> None:
    """Full training pipeline for schema classifier."""
    logger.info("=== Schema Classifier Training ===")
    t0 = time.perf_counter()

    # ── Load or generate data ─────────────────────────────────────────────────
    if args.data and os.path.exists(args.data):
        logger.info("Loading real dataset: %s", args.data)
        df = pd.read_csv(args.data)
    else:
        logger.info("Generating synthetic training data (%d samples)...", args.n_samples)
        df = build_synthetic_dataset(args.n_samples)
        if args.save_synthetic:
            df.to_csv(args.save_synthetic, index=False)
            logger.info("Saved synthetic data to %s", args.save_synthetic)

    # ── Prepare X, y ─────────────────────────────────────────────────────────
    label_col = "schema_label"
    if label_col not in df.columns:
        logger.error("Missing '%s' column in dataset. Aborting.", label_col)
        sys.exit(1)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    X = df.drop(columns=[label_col])

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    logger.info("Dataset: %d rows × %d features, %d classes", len(X), len(X.columns), len(le.classes_))

    # ── 70/15/15 stratified split ─────────────────────────────────────────────
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15 / 0.85, stratify=y_trainval, random_state=42,
    )
    logger.info("Split: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    # ── Train LightGBM with class weights ─────────────────────────────────────
    try:
        import lightgbm as lgb
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = dict(zip(classes.tolist(), weights.tolist()))

        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            learning_rate=0.05,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(25, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
    except ImportError:
        logger.warning("LightGBM not available, falling back to RandomForest")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    from sklearn.metrics import classification_report, accuracy_score

    y_pred_val  = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    val_acc  = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)
    gap      = abs(val_acc - test_acc)

    logger.info("Val  accuracy: %.3f", val_acc)
    logger.info("Test accuracy: %.3f", test_acc)
    logger.info("Overfitting gap: %.3f (threshold: 0.03)", gap)

    if gap > 0.03:
        logger.warning("⚠️  Overfitting detected! Val-Test gap %.3f > 0.03", gap)
    else:
        logger.info("✅ Quality gate passed — no overfitting detected")

    # ── 5-Fold Cross-Validation ───────────────────────────────────────────────
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    try:
        cv_scores = cross_val_score(
            model.__class__(**model.get_params()) if hasattr(model, "get_params") else model,
            X_trainval, y_trainval,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy", n_jobs=-1,
        )
        logger.info("5-Fold CV: mean=%.3f ± %.3f", cv_scores.mean(), cv_scores.std())
        if cv_scores.std() > 0.05:
            logger.warning("⚠️  High CV variance: std=%.3f > 0.05", cv_scores.std())
    except Exception as exc:
        logger.warning("CV scoring failed: %s", exc)

    print("\n=== Classification Report (Holdout) ===")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    # ── Save model ───────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    import joblib
    model_path = os.path.join(args.out, "schema_classifier.joblib")
    le_path    = os.path.join(args.out, "schema_label_encoder.joblib")
    joblib.dump(model, model_path)
    joblib.dump(le,    le_path)

    metadata = {
        "model_type": type(model).__name__,
        "val_accuracy": round(float(val_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "overfitting_gap": round(float(gap), 4),
        "quality_gate_passed": gap <= 0.03,
        "feature_names": list(X.columns),
        "schema_labels": list(le.classes_),
        "training_time_s": round(time.perf_counter() - t0, 2),
    }
    meta_path = os.path.join(args.out, "schema_classifier_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Model saved to %s", model_path)
    logger.info("Metadata saved to %s", meta_path)
    logger.info("Training completed in %.1fs", time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser(description="Train Schema Classifier for ADAP platform")
    parser.add_argument("--data", default=None, help="Path to real labelled dataset CSV")
    parser.add_argument("--out", default="models/schema_classifier", help="Output directory for saved model")
    parser.add_argument("--n-samples", type=int, default=4000, dest="n_samples",
                        help="Number of synthetic samples to generate")
    parser.add_argument("--save-synthetic", default=None, dest="save_synthetic",
                        help="If set, save synthetic training data to this path")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
