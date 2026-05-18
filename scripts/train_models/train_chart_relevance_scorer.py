"""
scripts/train_models/train_chart_relevance_scorer.py
------------------------------------------------------
Colab-ready training script for the Chart Relevance Scorer.

Goal: given a column's metadata and the current analysis context,
predict whether a chart type is relevant:
  0=irrelevant | 1=marginally useful | 2=highly recommended

Anti-Overfitting: 70/15/15 split, cross-entropy loss,
L2 regularisation, class-weighted training.

Chart types scored:
  histogram | boxplot | scatter | line | heatmap | bar | pie | violin

Usage (Google Colab):
    !pip install lightgbm scikit-learn pandas numpy joblib
    !python train_chart_relevance_scorer.py --out models/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_chart_relevance_scorer")

CHART_TYPES = ["histogram", "boxplot", "scatter", "line", "heatmap", "bar", "pie", "violin"]
RELEVANCE_LABELS = [0, 1, 2]  # 0=irrelevant, 1=useful, 2=highly recommended

# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_relevance_features(
    dtype: str,
    n_unique: int,
    n_rows: int,
    null_rate: float,
    skewness: float,
    is_target: bool,
    chart_type: str,
) -> Dict:
    """Feature vector describing column + chart type combination."""
    return {
        "is_numeric":     int(dtype in ("int64", "float64", "int32", "float32")),
        "is_categorical": int(dtype in ("object", "category", "bool")),
        "is_datetime":    int("datetime" in str(dtype)),
        "n_unique":       min(n_unique, 10000),
        "n_unique_log":   float(np.log1p(n_unique)),
        "n_rows_log":     float(np.log1p(n_rows)),
        "null_rate":      float(null_rate),
        "abs_skewness":   float(abs(skewness)),
        "is_target":      int(is_target),
        "is_binary":      int(n_unique <= 2),
        "is_high_card":   int(n_unique > 50),
        "is_low_card":    int(n_unique <= 10),
        **{f"chart_{c}": int(c == chart_type) for c in CHART_TYPES},
    }


def build_synthetic_relevance_data(n_per_combo: int = 30) -> pd.DataFrame:
    """
    Generate synthetic training data for chart relevance scoring.
    Encodes domain knowledge: e.g. histograms are highly relevant for
    numeric columns; pie charts are only recommended for low-cardinality.
    """
    rng = np.random.RandomState(42)
    records = []

    for chart in CHART_TYPES:
        for _ in range(n_per_combo):
            dtype = rng.choice(["float64", "int64", "object", "category", "datetime64"])
            n_unique = int(rng.choice([2, 5, 10, 25, 50, 100, 500, 1000]))
            n_rows   = int(rng.choice([100, 1000, 10000, 100000]))
            null_rate = float(rng.uniform(0, 0.4))
            skewness  = float(rng.normal(0, 2))
            is_target = bool(rng.choice([True, False]))

            is_numeric = dtype in ("float64", "int64")
            is_cat     = dtype in ("object", "category")
            is_dt      = "datetime" in str(dtype)
            is_binary  = n_unique <= 2
            is_low_c   = n_unique <= 10
            is_high_c  = n_unique > 50

            # Domain rules for relevance labels
            if chart == "histogram":
                label = 2 if is_numeric and not is_binary else (1 if is_numeric else 0)
            elif chart == "boxplot":
                label = 2 if is_numeric and n_rows >= 30 else (1 if is_numeric else 0)
            elif chart == "scatter":
                label = 2 if is_numeric and not is_target else (1 if is_numeric else 0)
            elif chart == "line":
                label = 2 if is_dt or (is_numeric and n_rows > 50) else (1 if is_numeric else 0)
            elif chart == "heatmap":
                label = 2 if is_numeric and not is_high_c else (1 if is_numeric else 0)
            elif chart == "bar":
                label = 2 if is_cat and is_low_c else (1 if is_cat else 0)
            elif chart == "pie":
                label = 2 if is_binary else (1 if is_low_c and is_cat else 0)
            elif chart == "violin":
                label = 2 if is_numeric and n_rows >= 100 else (1 if is_numeric and n_rows >= 30 else 0)
            else:
                label = 0

            # Add noise
            if rng.random() < 0.07:  # 7% label noise for robustness
                label = int(rng.choice(RELEVANCE_LABELS))

            feat = extract_relevance_features(
                dtype, n_unique, n_rows, null_rate, skewness, is_target, chart
            )
            feat["label"] = label
            records.append(feat)

    return pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)


def train(args) -> None:
    logger.info("=== Chart Relevance Scorer Training ===")
    t0 = time.perf_counter()

    logger.info("Generating synthetic data (%d samples per chart-type)...", args.n_per_combo)
    df = build_synthetic_relevance_data(args.n_per_combo)
    logger.info("Dataset: %d rows × %d features", len(df), len(df.columns) - 1)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    y = df["label"].values
    X = df.drop(columns=["label"])

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=42)

    logger.info("Split: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, min_child_samples=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            learning_rate=0.05, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=-1)],
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        model.fit(X_train, y_train)

    y_val_pred  = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    val_acc  = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    gap      = abs(val_acc - test_acc)

    logger.info("Val accuracy:  %.3f", val_acc)
    logger.info("Test accuracy: %.3f", test_acc)
    logger.info("Overfitting gap: %.3f", gap)
    if gap > 0.05:
        logger.warning("⚠️  Overfitting gap %.3f > 0.05", gap)
    else:
        logger.info("✅ Quality gate passed")

    print("\n=== Chart Relevance Report (Holdout) ===")
    print(classification_report(y_test, y_test_pred, target_names=["irrelevant", "useful", "recommended"]))

    os.makedirs(args.out, exist_ok=True)
    import joblib
    model_path = os.path.join(args.out, "chart_relevance_scorer.joblib")
    joblib.dump(model, model_path)

    meta = {
        "model_type": type(model).__name__,
        "val_accuracy": round(float(val_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "overfitting_gap": round(float(gap), 4),
        "quality_gate_passed": gap <= 0.05,
        "feature_names": list(X.columns),
        "training_time_s": round(time.perf_counter() - t0, 2),
    }
    with open(os.path.join(args.out, "chart_relevance_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Saved model to %s (%.1fs)", model_path, time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser(description="Train Chart Relevance Scorer")
    parser.add_argument("--out", default="models/chart_relevance", help="Output directory")
    parser.add_argument("--n-per-combo", type=int, default=50, dest="n_per_combo",
                        help="Synthetic samples per chart-type combination")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
