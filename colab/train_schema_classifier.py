"""
colab/train_schema_classifier.py
----------------------------------
DIPEX — Google Colab Training Script: ML Schema Semantic Type Classifier
Run this on Google Colab (free tier is sufficient).

Outputs
-------
  models/schema_classifier.pkl
  models/schema_label_encoder.pkl

Instructions
------------
1. Open this file in Google Colab.
2. Adjust TRAINING_CSV if you have domain-specific labelled data.
3. Run all cells. Training takes < 2 minutes on Colab CPU.
4. Download the two .pkl files and place them in your project's models/ folder.
   The runtime will auto-detect and use them.

If you don't have labelled data, a synthetic dataset is generated
automatically from the heuristic labels in Step 1.
"""

# ─── Step 0: Install dependencies ──────────────────────────────────────────────
# !pip install scikit-learn pandas numpy joblib -q

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

print("All imports OK.")

# ─── Step 1: Generate or load labelled training data ──────────────────────────

# If you have a real labelled CSV with columns [col_name, dtype, semantic_type, ...features],
# set this path. Otherwise, leave as None to use synthetic generation.
TRAINING_CSV = None  # e.g., "/content/drive/MyDrive/dipex_schema_labels.csv"

SEMANTIC_LABELS = [
    "id", "age", "amount", "date", "category", "text",
    "phone", "email", "boolean", "zipcode", "percentage",
    "score", "count", "name", "unknown",
]

def _synthetic_feature_row(sem_type: str) -> dict:
    """Generate a plausible feature vector for a given semantic type."""
    import random
    r = random.random

    base = {
        "null_rate":      round(r() * 0.15, 4),
        "unique_rate":    round(r(), 4),
        "is_numeric":     0.0,
        "is_string":      0.0,
        "is_datetime":    0.0,
        "mean_val":       0.0,
        "std_val":        0.0,
        "min_val":        0.0,
        "max_val":        0.0,
        "skew_val":       0.0,
        "all_integer":    0.0,
        "max_lt_200":     0.0,
        "max_lt_1":       0.0,
        "all_positive":   0.0,
        "n_distinct":     round(r() * 1000),
        "email_pattern":  0.0,
        "phone_pattern":  0.0,
        "mean_str_len":   0.0,
        "high_cardinality": 0.0,
        "low_cardinality":  0.0,
        "label":          sem_type,
    }

    if sem_type == "id":
        base.update({"is_numeric": r() > 0.3, "unique_rate": 0.95 + r() * 0.05,
                     "all_integer": 1.0, "high_cardinality": 1.0})
    elif sem_type == "age":
        base.update({"is_numeric": 1.0, "mean_val": 30 + r()*20, "max_lt_200": 1.0,
                     "all_integer": 1.0, "all_positive": 1.0, "max_val": 80 + r()*20})
    elif sem_type == "amount":
        base.update({"is_numeric": 1.0, "mean_val": 500 + r()*5000,
                     "std_val": 200 + r()*2000, "all_positive": r() > 0.2})
    elif sem_type == "date":
        base.update({"is_datetime": 1.0, "is_numeric": 0.0})
    elif sem_type == "email":
        base.update({"is_string": 1.0, "email_pattern": 0.8 + r()*0.2,
                     "mean_str_len": 20 + r()*15})
    elif sem_type == "phone":
        base.update({"is_string": 1.0, "phone_pattern": 0.7 + r()*0.3,
                     "mean_str_len": 10 + r()*5})
    elif sem_type == "category":
        base.update({"is_string": 1.0, "unique_rate": 0.001 + r()*0.05,
                     "low_cardinality": 1.0})
    elif sem_type == "text":
        base.update({"is_string": 1.0, "mean_str_len": 50 + r()*200, "unique_rate": 0.8+r()*0.2})
    elif sem_type == "boolean":
        base.update({"is_numeric": r() > 0.5, "unique_rate": 0.001,
                     "low_cardinality": 1.0, "max_lt_1": 1.0})
    elif sem_type == "percentage":
        base.update({"is_numeric": 1.0, "max_lt_1": 1.0, "all_positive": 1.0, "max_val": r()})
    elif sem_type == "score":
        base.update({"is_numeric": 1.0, "max_lt_200": 1.0, "all_positive": 1.0})
    elif sem_type == "count":
        base.update({"is_numeric": 1.0, "all_integer": 1.0, "all_positive": 1.0, "low_cardinality": 0.3})
    elif sem_type == "name":
        base.update({"is_string": 1.0, "mean_str_len": 10 + r()*15, "unique_rate": 0.7+r()*0.3})
    elif sem_type == "zipcode":
        base.update({"is_string": r() > 0.5, "mean_str_len": 5 + r()*2})

    return base

print("Generating synthetic training data (500 samples per class × 15 classes)...")
N_PER_CLASS = 500
rows = []
for label in SEMANTIC_LABELS:
    for _ in range(N_PER_CLASS):
        rows.append(_synthetic_feature_row(label))

df_train = pd.DataFrame(rows)
print(f"Training set: {len(df_train)} rows × {df_train.shape[1]} columns.")
print(df_train["label"].value_counts())

# ─── Step 2: Prepare X, y ─────────────────────────────────────────────────────

feature_cols = [c for c in df_train.columns if c != "label"]
X = df_train[feature_cols].values.astype(np.float32)
le = LabelEncoder()
y = le.fit_transform(df_train["label"].values)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# ─── Step 3: Train RandomForestClassifier ─────────────────────────────────────

print("Training RandomForestClassifier...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train)
val_acc = (clf.predict(X_val) == y_val).mean()
print(f"Validation accuracy: {val_acc:.4f}")
print(classification_report(y_val, clf.predict(X_val), target_names=le.classes_))

# ─── Step 4: Save artifacts ───────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/schema_classifier.pkl")
joblib.dump(le,  "models/schema_label_encoder.pkl")
print("\nArtifacts saved:")
print("  models/schema_classifier.pkl")
print("  models/schema_label_encoder.pkl")
print("\n>> Download these two files and place them in your project's models/ directory.")
print(">> The DIPEX runtime will auto-detect and use them at ingestion time.")
