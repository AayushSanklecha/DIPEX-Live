"""
scripts/train_confidence_scorer.py
------------------------------------
Trains the ProposalConfidenceScorer with real-world datasets.

Strategy:
  1. Download 20+ diverse real-world datasets from sklearn bundled + UCI repository
  2. For each dataset, run it through DIPEX's own quality pipeline to compute
     the exact 9-feature vector the scorer uses
  3. Label each dataset as high_confidence (1) or low_confidence (0)
     based on actual measured quality metrics
  4. Augment with synthetic variation to reach 3000+ training samples
  5. Train RandomForestClassifier + calibrate with cross-validation
  6. Save to models/proposal_confidence.pkl

Run with:
  python scripts/train_confidence_scorer.py
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("confidence_trainer")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
OUTPUT_PKL = os.path.join(MODELS_DIR, "proposal_confidence.pkl")

# ── Feature names (must match ml_confidence_scorer.py exactly) ───────────────
FEATURE_NAMES = [
    "drift_flag", "quality_score", "null_rate",
    "sample_size_k", "n_columns", "cv_score",
    "flag_severity_max", "columns_drifted", "proposer_type_enc",
]

# ── Real-world dataset loaders ────────────────────────────────────────────────

def _load_sklearn_datasets() -> list[tuple[str, pd.DataFrame, str | None]]:
    """Load diverse real datasets bundled with sklearn + seaborn."""
    datasets = []

    # sklearn bundled
    from sklearn import datasets as skds

    # Iris — clean, small, multi-class
    d = skds.load_iris(as_frame=True)
    df = d.frame; datasets.append(("iris", df, "target"))

    # Diabetes — regression, no nulls, biomedical
    d = skds.load_diabetes(as_frame=True)
    df = d.frame; datasets.append(("diabetes", df, "target"))

    # Breast cancer — high-dimensional, binary
    d = skds.load_breast_cancer(as_frame=True)
    df = d.frame; datasets.append(("breast_cancer", df, "target"))

    # Wine — clean, multi-class, chemical
    d = skds.load_wine(as_frame=True)
    df = d.frame; datasets.append(("wine", df, "target"))

    # Digits — image features, multi-class
    d = skds.load_digits(as_frame=True)
    df = d.frame; datasets.append(("digits", df, "target"))

    # California Housing — regression, geographic
    try:
        d = skds.fetch_california_housing(as_frame=True)
        df = d.frame; datasets.append(("california_housing", df, "MedHouseVal"))
    except Exception: pass

    # Olivetti faces — high-dim
    try:
        d = skds.fetch_olivetti_faces()
        df = pd.DataFrame(d.data)
        df["target"] = d.target
        datasets.append(("olivetti_faces", df, "target"))
    except Exception: pass

    # seaborn bundled datasets
    try:
        import seaborn as sns
        tips = sns.load_dataset("tips")
        datasets.append(("tips", tips, "tip"))

        titanic = sns.load_dataset("titanic").dropna(subset=["survived"])
        datasets.append(("titanic", titanic, "survived"))

        planets = sns.load_dataset("planets").dropna()
        datasets.append(("planets", planets, None))

        mpg = sns.load_dataset("mpg").dropna()
        datasets.append(("mpg", mpg, "mpg"))

        diamonds = sns.load_dataset("diamonds").sample(2000, random_state=42)
        datasets.append(("diamonds", diamonds, "price"))

        flights = sns.load_dataset("flights")
        datasets.append(("flights", flights, "passengers"))

        penguins = sns.load_dataset("penguins").dropna()
        datasets.append(("penguins", penguins, "species"))

        fmri = sns.load_dataset("fmri")
        datasets.append(("fmri", fmri, None))

    except Exception as e:
        log.warning("Seaborn datasets partial: %s", e)

    return datasets


def _load_uci_datasets() -> list[tuple[str, pd.DataFrame, str | None]]:
    """Download small UCI datasets directly (CSV accessible via URL)."""
    import requests as req

    uci_sources = [
        # Adult income (census) — socioeconomic, binary classification
        ("adult_income",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
         None,
         ["age","workclass","fnlwgt","education","education_num","marital",
          "occupation","relationship","race","sex","capital_gain","capital_loss",
          "hours_week","country","income"]),

        # Heart disease — medical, binary
        ("heart_disease",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
         None,
         ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
          "exang","oldpeak","slope","ca","thal","target"]),

        # Car evaluation — categorical, multi-class
        ("car_eval",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
         None,
         ["buying","maint","doors","persons","lug_boot","safety","class"]),

        # Abalone — biology, regression
        ("abalone",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
         None,
         ["sex","length","diameter","height","whole_weight","shucked_weight",
          "viscera_weight","shell_weight","rings"]),

        # Wine quality — continuous, regression
        ("wine_quality_red",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
         ";",
         None),

        # Wine quality white
        ("wine_quality_white",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
         ";",
         None),
    ]

    datasets = []
    for name, url, sep, cols in uci_sources:
        try:
            resp = req.get(url, timeout=15)
            if resp.status_code != 200:
                log.warning("UCI [%s]: HTTP %s", name, resp.status_code)
                continue
            from io import StringIO
            delimiter = sep or ","
            df = pd.read_csv(StringIO(resp.text), header=None if cols else "infer",
                             sep=delimiter, names=cols if cols else None,
                             na_values=["?", " ?", "? "])
            datasets.append((name, df, None))
            log.info("UCI [%s]: %d rows x %d cols", name, len(df), len(df.columns))
        except Exception as e:
            log.warning("UCI [%s] failed: %s", name, e)

    return datasets


# ── Pipeline quality feature extractor ───────────────────────────────────────

def _extract_pipeline_features(
    name: str,
    df: pd.DataFrame,
    target_col: str | None,
    proposer_type: int = 0,
) -> dict | None:
    """
    Run the dataset through DIPEX's actual quality pipeline and extract the
    9-feature vector used by ProposalConfidenceScorer.
    """
    try:
        if df.empty or len(df) < 10:
            return None

        # 1. Null rate
        null_rate = float(df.isnull().mean().mean())

        # 2. Quality score (inverse of null + type issues)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            # Encode categoricals
            df_enc = df.copy()
            for col in df_enc.select_dtypes(include="object"):
                df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
            numeric_df = df_enc.select_dtypes(include=[np.number])

        quality_score = float(np.clip(1.0 - null_rate - (0.05 * int(numeric_df.empty)), 0, 1))

        # 3. Anomaly count via IsolationForest
        anomaly_count = 0
        drift_flag = 0.0
        columns_drifted = 0
        num_clean = numeric_df.fillna(numeric_df.median())

        if len(num_clean) >= 20 and not num_clean.empty:
            try:
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(n_estimators=50, contamination="auto",
                                      random_state=42)
                preds = iso.fit_predict(num_clean)
                anomaly_count = int((preds == -1).sum())
            except Exception:
                pass

            # 4. Drift simulation: check if std of each column is anomalously high
            stds = num_clean.std()
            means = num_clean.mean()
            # Flag columns where CV (coeff of variation) > 2 as "drifted"
            cv_ratios = (stds / (means.abs() + 1e-8)).abs()
            columns_drifted = int((cv_ratios > 2.0).sum())
            drift_flag = float(columns_drifted > 0)

        # 5. Flag severity (nulls → severity 1, >20% → severity 2, >50% → severity 3)
        col_null_rates = df.isnull().mean()
        flag_severity_max = 0
        if (col_null_rates > 0.50).any():
            flag_severity_max = 3
        elif (col_null_rates > 0.20).any():
            flag_severity_max = 2
        elif (col_null_rates > 0.05).any():
            flag_severity_max = 1

        # 6. CV score — quick cross-validated model score on this dataset
        cv_score = 0.5
        if target_col and target_col in df.columns and len(num_clean) >= 20:
            try:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
                from sklearn.preprocessing import LabelEncoder

                y = df[target_col].dropna()
                X_ = num_clean.loc[y.index].fillna(0)
                if target_col in X_.columns:
                    X_ = X_.drop(columns=[target_col])

                if len(X_.columns) > 0 and len(y) >= 10:
                    is_cls = y.nunique() < 20
                    if is_cls:
                        y_enc = LabelEncoder().fit_transform(y.astype(str))
                        model = RandomForestClassifier(n_estimators=30, random_state=42)
                        scoring = "roc_auc" if y.nunique() == 2 else "accuracy"
                        skf = StratifiedKFold(n_splits=min(3, y.nunique()), shuffle=True, random_state=42)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            scores = cross_val_score(model, X_, y_enc, cv=skf, scoring=scoring)
                        cv_score = float(np.clip(scores.mean(), 0, 1))
                    else:
                        model = RandomForestRegressor(n_estimators=30, random_state=42)
                        kf = KFold(n_splits=3, shuffle=True, random_state=42)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            scores = cross_val_score(model, X_, y, cv=kf, scoring="r2")
                        cv_score = float(np.clip(scores.mean(), 0, 1))
            except Exception:
                pass

        # 7. Assemble feature vector
        feat = {
            "drift_flag":        drift_flag,
            "quality_score":     round(quality_score, 4),
            "null_rate":         round(null_rate, 4),
            "sample_size_k":     round(len(df) / 1000.0, 4),
            "n_columns":         float(len(df.columns)),
            "cv_score":          round(cv_score, 4),
            "flag_severity_max": float(flag_severity_max),
            "columns_drifted":   float(columns_drifted),
            "proposer_type_enc": float(proposer_type),
            # metadata for labeling
            "_name":             name,
            "_null_rate":        null_rate,
            "_cv_score":         cv_score,
            "_quality_score":    quality_score,
        }
        return feat

    except Exception as e:
        log.warning("[%s] feature extraction failed: %s", name, e)
        return None


# ── Label: high_confidence (1) or low_confidence (0) ─────────────────────────

def _label(feat: dict) -> int:
    """
    High confidence = pipeline produces a quality, actionable result.
    Uses a continuous scoring approach to get balanced classes from
    real-world data where quality_score is often uniformly high.
    """
    cv  = feat["_cv_score"]
    nr  = feat["_null_rate"]
    sev = feat["flag_severity_max"]

    # High confidence: strong CV score, low nulls, no severe flags
    if cv >= 0.70 and nr <= 0.08 and sev <= 1:
        return 1
    # Medium-good: decent CV, manageable nulls
    if cv >= 0.60 and nr <= 0.15 and sev <= 2:
        return 1
    # Everything else = low confidence
    return 0


# ── Low-quality synthetic samples: simulate messy real-world datasets ─────────

def _make_low_quality_samples(n: int = 500) -> list[dict]:
    """
    Simulate features of messy, real-world datasets that would produce
    low_confidence proposals: high nulls, weak models, severe drift.
    Grounded in realistic failure modes (missing medical data, sparse surveys,
    noisy IoT streams, incomplete financial records).
    """
    rng = np.random.default_rng(7)
    samples = []

    scenarios = [
        # (drift_flag, quality_score, null_rate, sample_k, n_cols, cv_score, severity, drifted, ptype)
        # Sparse survey data — lots of missing answers
        dict(drift_flag=0.0, quality_score=0.35, null_rate=0.55, sample_size_k=0.5,
             n_columns=20.0, cv_score=0.48, flag_severity_max=3, columns_drifted=0.0, proposer_type_enc=5),
        # IoT sensor drift — high temporal drift, weak model
        dict(drift_flag=1.0, quality_score=0.60, null_rate=0.12, sample_size_k=10.0,
             n_columns=8.0, cv_score=0.51, flag_severity_max=2, columns_drifted=4.0, proposer_type_enc=0),
        # Financial data with many missing cells
        dict(drift_flag=0.0, quality_score=0.42, null_rate=0.38, sample_size_k=2.0,
             n_columns=30.0, cv_score=0.52, flag_severity_max=3, columns_drifted=0.0, proposer_type_enc=3),
        # Small dataset, underfit model
        dict(drift_flag=0.0, quality_score=0.70, null_rate=0.04, sample_size_k=0.02,
             n_columns=5.0, cv_score=0.55, flag_severity_max=1, columns_drifted=1.0, proposer_type_enc=1),
        # Healthcare records — high null rate for optional fields
        dict(drift_flag=1.0, quality_score=0.50, null_rate=0.30, sample_size_k=1.2,
             n_columns=45.0, cv_score=0.50, flag_severity_max=2, columns_drifted=3.0, proposer_type_enc=4),
        # Heavily drifted time series
        dict(drift_flag=1.0, quality_score=0.55, null_rate=0.08, sample_size_k=5.0,
             n_columns=12.0, cv_score=0.44, flag_severity_max=2, columns_drifted=8.0, proposer_type_enc=2),
    ]

    per_scenario = n // len(scenarios)
    for base in scenarios:
        for _ in range(per_scenario):
            row = base.copy()
            for k in ["quality_score", "null_rate", "cv_score"]:
                row[k] = float(np.clip(row[k] + rng.normal(0, 0.05), 0.0, 1.0))
            row["_quality_score"] = row["quality_score"]
            row["_null_rate"]     = row["null_rate"]
            row["_cv_score"]      = row["cv_score"]
            row["label"]          = 0   # explicitly low_confidence
            samples.append(row)
    return samples


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment(rows: list[dict], target_n: int = 3000) -> list[dict]:
    """
    Augment the real-data rows with synthetic variations to reach target_n.
    Adds gaussian noise to numeric features keeping labels consistent.
    """
    rng = np.random.default_rng(42)
    augmented = list(rows)
    while len(augmented) < target_n:
        row = rows[rng.integers(0, len(rows))].copy()
        for k in FEATURE_NAMES:
            val = row[k]
            noise = rng.normal(0, 0.04)
            row[k] = float(np.clip(val + noise, 0.0, 1.0 if k != "n_columns" else 200.0))
        augmented.append(row)
    return augmented


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    log.info("=" * 60)
    log.info("DIPEX Confidence Scorer Training")
    log.info("=" * 60)

    # 1. Collect datasets
    log.info("\n[1/4] Loading real-world datasets...")
    all_datasets = []
    all_datasets += _load_sklearn_datasets()
    log.info("Sklearn/seaborn: %d datasets", len(all_datasets))

    uci = _load_uci_datasets()
    all_datasets += uci
    log.info("UCI: %d datasets", len(uci))

    log.info("Total datasets: %d", len(all_datasets))

    # 2. Extract features for each dataset + each proposer type
    log.info("\n[2/4] Extracting pipeline quality features...")
    rows = []
    proposer_types = list(range(8))  # 0-7 match _PROPOSER_TYPE_MAP

    for name, df, target_col in all_datasets:
        # Run with multiple proposer types for diversity
        for ptype in proposer_types:
            feat = _extract_pipeline_features(name, df, target_col, ptype)
            if feat:
                feat["label"] = _label(feat)
                rows.append(feat)
                if ptype == 0:
                    log.info("  [%s] rows=%d null=%.2f qual=%.2f cv=%.2f → label=%d",
                             name, len(df), feat["null_rate"],
                             feat["quality_score"], feat["cv_score"], feat["label"])

    log.info("Real feature rows: %d", len(rows))
    if len(rows) < 5:
        log.error("Too few training rows — check network access for UCI downloads")
        sys.exit(1)

    # 3. Add realistic low-quality samples + augment
    log.info("\n[3/4] Adding low-quality samples + augmenting to 3000+ training samples...")
    low_quality_rows = _make_low_quality_samples(500)
    rows += low_quality_rows
    log.info("After adding low-quality samples: %d rows", len(rows))

    rows = _augment(rows, target_n=3000)
    log.info("After augmentation: %d rows", len(rows))

    # Build X, y
    X = np.array([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows], dtype=np.int32)

    pos_rate = y.mean()
    log.info("Class balance: high_conf=%.1f%% low_conf=%.1f%%",
             pos_rate * 100, (1 - pos_rate) * 100)

    # 4. Train
    log.info("\n[4/4] Training RandomForest + probability calibration...")

    # Cross-validate first to report honest metrics
    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc_scores = cross_val_score(base_rf, X, y, cv=cv, scoring="roc_auc")
        f1_scores  = cross_val_score(base_rf, X, y, cv=cv, scoring="f1")

    log.info("5-fold CV  AUC: %.4f ± %.4f", auc_scores.mean(), auc_scores.std())
    log.info("5-fold CV  F1 : %.4f ± %.4f", f1_scores.mean(), f1_scores.std())

    # Train final model on all data with probability calibration
    final_model = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        method="isotonic",
        cv=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X, y)

    # Save
    joblib.dump(final_model, OUTPUT_PKL)
    log.info("\n✅ Model saved → %s", OUTPUT_PKL)

    # Sanity check
    proba = final_model.predict_proba(X[:5])
    log.info("Sample probas: %s", [round(p[1], 3) for p in proba])

    log.info("\n" + "=" * 60)
    log.info("Training complete.")
    log.info("  AUC  : %.4f", auc_scores.mean())
    log.info("  F1   : %.4f", f1_scores.mean())
    log.info("  Rows : %d", len(rows))
    log.info("  Saved: %s", OUTPUT_PKL)
    log.info("=" * 60)


if __name__ == "__main__":
    train()
