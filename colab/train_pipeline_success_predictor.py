"""
colab/train_pipeline_success_predictor.py
-------------------------------------------
DIPEX — Google Colab Training Script: ML Pipeline Success Predictor
Run this on Google Colab (CPU sufficient, ~1 min).

Outputs
-------
  models/pipeline_success_predictor.pkl

Instructions
------------
1. (Optional) Set RUN_LOG_CSV to a CSV of past pipeline run outcomes.
   Expected columns: null_rate, drift_detected, quality_score, row_count,
                     n_columns, anomaly_count, schema_match, known_dataset,
                     cv_score, columns_drifted, success  (0/1)
2. Without a CSV, a synthetic dataset is generated from realistic distributions.
3. Run all cells. Download the .pkl → place in models/.

The verifier/pipeline_success_predictor.py will auto-detect and use it.
"""

# !pip install scikit-learn pandas numpy joblib -q

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

print("=== DIPEX Pipeline Success Predictor Training ===")

# ─── Config ───────────────────────────────────────────────────────────────────

RUN_LOG_CSV     = None     # Path to real run log CSV or None for synthetic
N_SYNTHETIC     = 4_000    # Samples if no real data

FEATURE_NAMES   = [
    "null_rate", "drift_detected", "quality_score", "row_count_k",
    "n_columns", "anomaly_count", "schema_match", "known_dataset",
    "cv_score", "columns_drifted",
]

# ─── Step 1: Generate synthetic training data ─────────────────────────────────

def _generate_synthetic(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "null_rate":      rng.beta(1, 10, n),                    # mostly low
        "drift_detected": rng.choice([0, 1], p=[0.7, 0.3], size=n).astype(float),
        "quality_score":  rng.beta(8, 2, n),                     # mostly high
        "row_count_k":    rng.exponential(10, n),                 # in thousands
        "n_columns":      rng.integers(3, 60, n).astype(float),
        "anomaly_count":  rng.poisson(2, n).astype(float),
        "schema_match":   rng.choice([0, 1], p=[0.1, 0.9], size=n).astype(float),
        "known_dataset":  rng.choice([0, 1], p=[0.15, 0.85], size=n).astype(float),
        "cv_score":       rng.beta(5, 2, n),                      # model quality
        "columns_drifted": rng.poisson(1, n).astype(float),
    })

    # Realistic success rule (with noise)
    logit_success = (
        - df["null_rate"] * 4.0
        - df["drift_detected"] * 2.5
        + df["quality_score"] * 3.0
        + np.log1p(df["row_count_k"]) * 0.3
        + df["schema_match"] * 2.0
        + df["cv_score"] * 2.5
        - df["anomaly_count"] * 0.3
        - df["columns_drifted"] * 0.4
        + rng.normal(0, 0.5, n)    # noise
    )
    prob_success = 1 / (1 + np.exp(-logit_success))
    df["success"] = (rng.uniform(size=n) < prob_success).astype(int)

    print(f"Synthetic data: {n} rows, {df['success'].mean():.1%} success rate")
    return df

if RUN_LOG_CSV and os.path.exists(RUN_LOG_CSV):
    df = pd.read_csv(RUN_LOG_CSV)
    print(f"Loaded run log: {df.shape}")
else:
    df = _generate_synthetic(N_SYNTHETIC)

# ─── Step 2: Prepare features ─────────────────────────────────────────────────

X = df[FEATURE_NAMES].fillna(df[FEATURE_NAMES].median()).values
y = df["success"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")
print(f"Class balance — Train: {y_train.mean():.1%} success, Val: {y_val.mean():.1%}")

# ─── Step 3: Train ───────────────────────────────────────────────────────────

print("\nTraining RandomForestClassifier...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train)

y_pred  = clf.predict(X_val)
y_proba = clf.predict_proba(X_val)[:, 1]
val_acc = (y_pred == y_val).mean()
val_auc = roc_auc_score(y_val, y_proba)
print(f"Val accuracy: {val_acc:.4f}  |  ROC-AUC: {val_auc:.4f}")
print(classification_report(y_val, y_pred, target_names=["failure", "success"]))

# Feature importance
feat_imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                  key=lambda x: -x[1])
print("\nTop feature importances:")
for fname, imp in feat_imp[:7]:
    print(f"  {fname:25s}: {imp:.4f}")

# ─── Step 4: Cross-validation ─────────────────────────────────────────────────

cv_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── Step 5: Save artifact ───────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/pipeline_success_predictor.pkl")
print("\nArtifact saved: models/pipeline_success_predictor.pkl")
print(">> Download → place in your project's models/ directory.")
print(">> PipelineSuccessPredictor will auto-load it on the next pipeline run.")
