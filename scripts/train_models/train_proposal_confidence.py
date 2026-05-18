"""
scripts/train_models/train_proposal_confidence.py
---------------------------------------------------
Colab-ready training script for the Proposal Confidence Scorer.

Goal: given an analytical insight proposal's features (source, support metrics,
domain, p-value, effect size), predict confidence in the proposal:
  0=low | 1=medium | 2=high

Anti-Overfitting: 70/15/15 split, class-weighted training,
Optuna hyperparameter tuning (20 trials), 5-fold CV.

Usage (Google Colab):
    !pip install lightgbm scikit-learn pandas numpy joblib optuna
    !python train_proposal_confidence.py --out models/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_proposal_confidence")

CONFIDENCE_LABELS = [0, 1, 2]  # 0=low, 1=medium, 2=high

# ── Synthetic data generation ────────────────────────────────────────────────

def build_proposal_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """
    Synthetic labelled data for proposal confidence scoring.

    Features derived from insight properties:
      - source_type: EDA | ML | Statistical | LLM | RL
      - n_supporting_rows: how many rows support the finding
      - sample_pct: % of dataset supporting the finding
      - p_value: statistical significance
      - effect_size: Cohen's d or Cramér's V
      - insight_rank: position in ranked insight list (1-20)
      - domain_confidence: domain-level prior confidence (0-1)
      - is_anomaly_based: based on anomaly detection
      - model_cv_score: CV score of the underlying model (if applicable)
    """
    rng = np.random.RandomState(42)
    records = []

    for _ in range(n_samples):
        p_value     = float(rng.exponential(0.15))
        effect_size = float(rng.exponential(0.3))
        n_rows      = int(rng.choice([50, 200, 500, 2000, 10000, 100000]))
        sample_pct  = float(rng.uniform(0.01, 1.0))
        rank        = int(rng.randint(1, 21))
        domain_conf = float(rng.uniform(0.4, 1.0))
        model_cv    = float(rng.uniform(0.5, 0.99))
        is_anomaly  = int(rng.choice([0, 1]))
        source_enc  = int(rng.choice([0, 1, 2, 3, 4]))  # EDA/ML/Stat/LLM/RL

        # Confidence label rules (domain knowledge)
        score = 0.0
        if p_value < 0.01: score += 2.0
        elif p_value < 0.05: score += 1.0
        if effect_size >= 0.5: score += 2.0
        elif effect_size >= 0.2: score += 1.0
        if sample_pct >= 0.5: score += 1.5
        elif sample_pct >= 0.1: score += 0.5
        if rank <= 3: score += 1.5
        elif rank <= 8: score += 0.5
        if model_cv >= 0.9: score += 1.5
        elif model_cv >= 0.75: score += 0.5
        if domain_conf >= 0.85: score += 1.0
        if is_anomaly: score += 0.5
        if source_enc == 1: score += 0.5  # ML > LLM

        # Add noise
        score += rng.normal(0, 0.5)

        if score >= 5.0:
            label = 2
        elif score >= 2.5:
            label = 1
        else:
            label = 0

        records.append({
            "p_value":        min(p_value, 1.0),
            "effect_size":    min(effect_size, 3.0),
            "n_rows_log":     float(np.log1p(n_rows)),
            "sample_pct":     sample_pct,
            "insight_rank":   rank,
            "domain_conf":    domain_conf,
            "model_cv_score": model_cv,
            "is_anomaly_based": is_anomaly,
            "source_type_enc": source_enc,
            "p_value_log":    float(np.log1p(p_value)),
            "label":          label,
        })

    return pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)


def train(args) -> None:
    logger.info("=== Proposal Confidence Scorer Training ===")
    t0 = time.perf_counter()

    df = build_proposal_dataset(args.n_samples)
    y = df["label"].values
    X = df.drop(columns=["label"])
    logger.info("Dataset: %d rows × %d features, 3 classes", len(X), len(X.columns))

    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=42
    )
    logger.info("Split: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    # ── Optional Optuna tuning ────────────────────────────────────────────────
    best_params = {
        "n_estimators": 300, "max_depth": 5, "min_child_samples": 5,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_lambda": 1.0, "learning_rate": 0.05,
    }

    if args.tune:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                import lightgbm as lgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth":    trial.suggest_int("max_depth", 3, 8),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                    "subsample":    trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_lambda":   trial.suggest_float("reg_lambda", 0.1, 5.0),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "random_state": 42, "n_jobs": -1, "verbose": -1,
                    "class_weight": "balanced",
                }
                m = lgb.LGBMClassifier(**params)
                m.fit(X_train, y_train)
                return accuracy_score(y_val, m.predict(X_val))

            study = optuna.create_study(direction="maximize",
                                         pruner=optuna.pruners.MedianPruner())
            study.optimize(objective, n_trials=args.tune_trials, show_progress_bar=False)
            best_params.update(study.best_params)
            logger.info("Best Optuna params: %s", study.best_params)
        except ImportError:
            logger.warning("Optuna not available — using default params")

    # ── Train final model ─────────────────────────────────────────────────────
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**best_params, class_weight="balanced",
                                    random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(25, verbose=False),
                              lgb.log_evaluation(period=-1)])
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_acc  = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    gap = abs(val_acc - test_acc)

    logger.info("Val accuracy:  %.3f", val_acc)
    logger.info("Test accuracy: %.3f", test_acc)
    logger.info("Overfitting gap: %.3f", gap)

    # 5-fold CV
    try:
        cv_scores = cross_val_score(model, X_tv, y_tv,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                     scoring="accuracy", n_jobs=-1)
        logger.info("5-Fold CV: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())
    except Exception as exc:
        logger.warning("CV failed: %s", exc)
        cv_scores = np.array([val_acc])

    print("\n=== Proposal Confidence Report (Holdout) ===")
    print(classification_report(y_test, model.predict(X_test),
                                 target_names=["low", "medium", "high"]))

    os.makedirs(args.out, exist_ok=True)
    import joblib
    model_path = os.path.join(args.out, "proposal_confidence_scorer.joblib")
    joblib.dump(model, model_path)
    with open(os.path.join(args.out, "proposal_confidence_metadata.json"), "w") as fh:
        json.dump({
            "model_type": type(model).__name__,
            "val_accuracy": round(float(val_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "overfitting_gap": round(float(gap), 4),
            "quality_gate_passed": gap <= 0.05 and cv_scores.std() <= 0.05,
            "best_params": best_params,
            "training_time_s": round(time.perf_counter() - t0, 2),
        }, fh, indent=2)
    logger.info("Saved to %s", model_path)


def main():
    parser = argparse.ArgumentParser(description="Train Proposal Confidence Scorer")
    parser.add_argument("--out",   default="models/proposal_confidence")
    parser.add_argument("--n-samples", type=int, default=3000, dest="n_samples")
    parser.add_argument("--tune",  action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--tune-trials", type=int, default=20, dest="tune_trials")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
