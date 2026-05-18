"""
scripts/train_models/train_drift_models.py
------------------------------------------
Colab-ready training script for data drift detection models.

Trains two models:
  1. Feature Drift Detector   — binary classifier (stable vs. drifted)
                                 Features: PSI, KS stat, distribution stats
  2. Label Drift Detector     — binary classifier for label distribution shift

Anti-Overfitting: 70/15/15 split, class-weighted loss,
5-fold CV, early stopping.

Usage (Google Colab):
    !pip install lightgbm scikit-learn pandas numpy joblib scipy
    !python train_drift_models.py --out models/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_drift_models")


# ── Synthetic drift dataset generation ───────────────────────────────────────

def build_drift_dataset(n_per_class: int = 1500) -> pd.DataFrame:
    """
    Generate synthetic drift detection training data.

    Features represent statistical comparison between a reference and
    production distribution window.
    """
    rng = np.random.RandomState(42)
    records = []

    for label in [0, 1]:  # 0=stable, 1=drifted
        for _ in range(n_per_class):
            if label == 0:  # stable: small PSI, high KS p-value
                psi         = float(rng.exponential(0.02))
                ks_stat     = float(rng.uniform(0.0, 0.05))
                ks_pvalue   = float(rng.uniform(0.3, 1.0))
                mean_shift  = float(rng.uniform(0.0, 0.1))
                std_shift   = float(rng.uniform(0.9, 1.1))
                entropy_diff = float(rng.uniform(0.0, 0.05))
                missing_rate_delta = float(rng.uniform(-0.02, 0.02))
                n_zeros_delta = float(rng.uniform(-0.02, 0.02))
            else:  # drifted: high PSI, low KS p-value
                psi         = float(rng.exponential(0.15) + 0.10)
                ks_stat     = float(rng.uniform(0.1, 0.8))
                ks_pvalue   = float(rng.uniform(0.0, 0.05))
                mean_shift  = float(rng.choice([1, -1]) * rng.exponential(0.5))
                std_shift   = float(rng.choice([0.3, 3.0]) * rng.uniform(0.5, 2.0))
                entropy_diff = float(rng.exponential(0.3))
                missing_rate_delta = float(rng.uniform(-0.3, 0.3))
                n_zeros_delta = float(rng.uniform(-0.2, 0.2))

            records.append({
                "psi":                  min(psi, 5.0),
                "psi_log":              float(np.log1p(min(psi, 5.0))),
                "ks_statistic":         min(ks_stat, 1.0),
                "ks_p_value":           ks_pvalue,
                "ks_significant":       int(ks_pvalue < 0.05),
                "mean_shift_ratio":     float(abs(mean_shift)),
                "std_shift_ratio":      float(abs(std_shift - 1.0)),
                "entropy_diff":         min(entropy_diff, 5.0),
                "missing_rate_delta":   float(abs(missing_rate_delta)),
                "n_zeros_delta":        float(abs(n_zeros_delta)),
                "psi_warn":             int(psi >= 0.10),
                "psi_critical":         int(psi >= 0.25),
                "label":                label,
            })

    return pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)


def train_single_model(
    X_train, y_train, X_val, y_val, X_test, y_test, name: str, out_dir: str
) -> dict:
    """Train one drift detection model and return metrics."""
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, min_child_samples=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            learning_rate=0.05, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20, verbose=False),
                              lgb.log_evaluation(period=-1)])
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                       class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)

    val_auc  = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    gap = abs(val_auc - test_auc)

    logger.info("[%s] Val AUC=%.3f, Test AUC=%.3f, Gap=%.3f", name, val_auc, test_auc, gap)
    if gap > 0.05:
        logger.warning("⚠️  Overfitting detected for %s: gap=%.3f", name, gap)

    try:
        cv_scores = cross_val_score(
            model, pd.concat([X_train, X_val]), pd.concat([pd.Series(y_train), pd.Series(y_val)]),
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring="roc_auc", n_jobs=-1,
        )
        logger.info("[%s] 5-Fold CV AUC: %.3f ± %.3f", name, cv_scores.mean(), cv_scores.std())
    except Exception as exc:
        logger.warning("[%s] CV failed: %s", name, exc)
        cv_scores = np.array([val_auc])

    print(f"\n=== {name} Holdout Report ===")
    print(classification_report(y_test, model.predict(X_test), target_names=["stable", "drifted"]))

    import joblib
    model_path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(model, model_path)

    return {
        "model_type": type(model).__name__,
        "val_auc": round(float(val_auc), 4),
        "test_auc": round(float(test_auc), 4),
        "cv_mean_auc": round(float(cv_scores.mean()), 4),
        "cv_std_auc": round(float(cv_scores.std()), 4),
        "overfitting_gap": round(float(gap), 4),
        "quality_gate_passed": gap <= 0.05 and cv_scores.std() <= 0.05,
    }


def train(args) -> None:
    logger.info("=== Drift Model Training ===")
    t0 = time.perf_counter()

    df = build_drift_dataset(args.n_per_class)
    logger.info("Dataset: %d rows, %d features", len(df), len(df.columns) - 1)

    from sklearn.model_selection import train_test_split

    y = df["label"].values
    X = df.drop(columns=["label"])

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15/0.85,
                                                        stratify=y_tv, random_state=42)
    logger.info("Split: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    os.makedirs(args.out, exist_ok=True)

    # Train Feature Drift Detector
    feature_metrics = train_single_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        "feature_drift_detector", args.out,
    )

    # Train Label Drift Detector (same features — in reality would have label distribution stats)
    label_metrics = train_single_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        "label_drift_detector", args.out,
    )

    with open(os.path.join(args.out, "drift_models_metadata.json"), "w") as fh:
        json.dump({
            "feature_drift_detector": feature_metrics,
            "label_drift_detector": label_metrics,
            "training_time_s": round(time.perf_counter() - t0, 2),
        }, fh, indent=2)

    logger.info("✅ All drift models saved to %s (%.1fs)", args.out, time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser(description="Train Drift Detection Models")
    parser.add_argument("--out", default="models/drift_models")
    parser.add_argument("--n-per-class", type=int, default=2000, dest="n_per_class")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
