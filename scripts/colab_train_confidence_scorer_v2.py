# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DIPEX — Proposal Confidence Scorer  v2  (Colab Training Script)            ║
# ║  ─────────────────────────────────────────────────────────────────────────  ║
# ║  Drop this entire file into a single Google Colab cell and run it.          ║
# ║  At the end, download `proposal_confidence.pkl` and place it in             ║
# ║  your local  dipex_project/models/  directory.                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""
WHAT THIS SCRIPT DOES
─────────────────────
1. Pulls structural metadata from 200+ real-world datasets via OpenML
   (shapes, null rates, feature types, class balance) — used to ground
   the synthetic data distributions in reality.

2. Generates 300 000 synthetic DIPEX pipeline-run records spanning:
   • 8 proposer types (anomaly, feature, join, model, correlation,
     hypothesis, optimization, rag)
   • 7 domains (banking, healthcare, finance, gdpr, sox, hipaa, default)
   • Low / medium / high data quality cohorts
   • Rare but realistic edge cases (tiny datasets, huge drifts, all nulls
     in one column, regulatory critical violations)
   • Added calibrated label noise to prevent label memorisation

3. Trains an optimised VotingClassifier ensemble:
     GradientBoostingClassifier  (35%)  — strong learner, captures
                                          non-linear interactions
     RandomForestClassifier      (35%)  — variance reducer
     LogisticRegression          (30%)  — calibration anchor

4. Anti-overfitting controls:
   • 5-fold StratifiedKFold cross-validation reported BEFORE final fit
   • Learning-curve analysis (train vs. val gap tracking)
   • Max-depth + min-samples-leaf regularisation on tree models
   • Early-stopping via warm_start loop on GBT
   • Platt calibration (CalibratedClassifierCV, sigmoid method)
   • Held-out test set (20 %) never seen during hyperparameter tuning

5. Output: `proposal_confidence.pkl` (a scikit-learn Pipeline with
   StandardScaler → CalibratedClassifierCV(VotingClassifier), so it
   works with a bare np.array from ml_confidence_scorer.py)

FEATURE VECTOR — must stay in sync with ml_confidence_scorer.py
────────────────────────────────────────────────
  Index  Name                Description
  ─────  ──────────────────  ─────────────────────────────────────────
    0    anomaly_count       # flagged anomalies in this run
    1    drift_flag          0/1 — schema/distribution drift detected
    2    quality_score       0–1 — overall data quality proxy
    3    null_rate           0–1 — mean null fraction across columns
    4    sample_size_k       row count ÷ 1000
    5    n_columns           number of columns
    6    cv_score            0–1 — model cross-validation score
    7    flag_severity_max   0–4 — worst data-quality flag severity
    8    columns_drifted     count of drifted columns
    9    proposer_type_enc   int encoding (0–7) of proposal origin

NOTE: compliance_penalty is NOT a feature here — it impacts the
pipeline-level confidence score via ConfidenceVector.aggregate().
"""

# ── 0. Colab auto-install ─────────────────────────────────────────────────────
import subprocess, sys

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

_pip("scikit-learn>=1.3", "xgboost", "lightgbm", "openml", "joblib",
     "pandas", "numpy", "matplotlib", "seaborn", "imbalanced-learn")

# ── 1. Imports ────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import time
import json

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble          import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       VotingClassifier)
from sklearn.linear_model      import LogisticRegression
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.model_selection   import (StratifiedKFold, cross_validate,
                                       learning_curve, train_test_split)
from sklearn.metrics           import (classification_report, roc_auc_score,
                                       brier_score_loss, confusion_matrix,
                                       ConfusionMatrixDisplay)
from sklearn.inspection        import permutation_importance
import joblib

print("━" * 70)
print("  DIPEX  Proposal Confidence Scorer — v2 Training Script")
print("━" * 70)

# ── 2. Real-world metadata via OpenML ─────────────────────────────────────────
def fetch_openml_metadata(n_datasets: int = 200) -> pd.DataFrame:
    """
    Fetches dataset-level metadata from OpenML (no full downloads).
    Covers diverse shapes, null densities, and feature-count ranges.
    Falls back silently if OpenML is unreachable.
    """
    print(f"\n[1/5] Fetching structural metadata from {n_datasets} OpenML datasets …")
    records = []

    try:
        import openml
        # List datasets with at least 100 rows, up to 5 000 000 rows
        df_list = openml.datasets.list_datasets(
            output_format="dataframe",
            size=(100, 5_000_000),
        )
        # Sample a diverse subset
        sample = df_list.sample(min(n_datasets * 3, len(df_list)),
                                random_state=42).head(n_datasets * 3)

        success = 0
        for did in sample["did"].tolist():
            if success >= n_datasets:
                break
            try:
                ds  = openml.datasets.get_dataset(did, download_data=False)
                md  = ds.qualities
                rows = int(md.get("NumberOfInstances", 0) or 0)
                cols = int(md.get("NumberOfFeatures",  0) or 0)
                nulls= float(md.get("NumberOfMissingValues", 0) or 0)
                if rows < 50 or cols < 2:
                    continue
                null_rate = nulls / max(rows * cols, 1)
                n_classes = int(md.get("NumberOfClasses", 1) or 1)
                records.append({
                    "rows": rows, "cols": cols,
                    "null_rate": min(null_rate, 1.0),
                    "n_classes": n_classes,
                })
                success += 1
            except Exception:
                continue

    except Exception as e:
        print(f"   [warn] OpenML unavailable ({e}). Using statistical priors only.")

    df = pd.DataFrame(records)
    print(f"   Retrieved metadata for {len(df)} datasets.")
    return df


# ── 3. Synthetic data generation ───────────────────────────────────────────────
_DOMAINS = ["banking", "healthcare", "finance", "gdpr", "sox", "hipaa", "default"]

_DOMAIN_PRIORS = {
    # (mean_quality, mean_null, drift_prob, crit_viol_prob)
    "banking":    (0.82, 0.04, 0.18, 0.08),
    "healthcare": (0.88, 0.03, 0.12, 0.15),
    "finance":    (0.80, 0.05, 0.20, 0.06),
    "gdpr":       (0.85, 0.02, 0.10, 0.18),
    "sox":        (0.84, 0.03, 0.15, 0.10),
    "hipaa":      (0.87, 0.03, 0.11, 0.14),
    "default":    (0.75, 0.08, 0.25, 0.05),
}

def _domain_sample(domain: str, n: int, rng: np.random.Generator):
    q_mu, null_mu, drift_p, crit_p = _DOMAIN_PRIORS[domain]
    quality_score = np.clip(rng.normal(q_mu, 0.12, n), 0, 1)
    null_rate     = np.clip(rng.beta(2, max(1, int(1/max(null_mu, 0.01))), n)
                            * null_mu * 10, 0, 1)
    drift_flag    = rng.binomial(1, drift_p, n).astype(float)
    has_critical  = rng.binomial(1, crit_p,  n)
    return quality_score, null_rate, drift_flag, has_critical


def generate_training_data(
    openml_meta: pd.DataFrame,
    n_samples: int = 300_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:

    print(f"\n[2/5] Generating {n_samples:,} synthetic pipeline-run records …")
    rng = np.random.default_rng(seed)

    # ── Sample dataset shapes from OpenML (or statistical prior) ──────────────
    if len(openml_meta) >= 10:
        meta = openml_meta.sample(n_samples, replace=True,
                                  random_state=seed).reset_index(drop=True)
        base_rows = meta["rows"].values * rng.uniform(0.3, 3.0, n_samples)
        base_cols = meta["cols"].values * rng.uniform(0.7, 1.3, n_samples)
    else:
        # Bayesian prior: log-normal covering tiny CSVs → warehouse tables
        base_rows = rng.lognormal(7.0, 2.5, n_samples)   # ~1K rows median
        base_cols = rng.lognormal(3.0, 1.2, n_samples)   # ~20 cols median

    # ── Domain allocation (weighted) ──────────────────────────────────────────
    domain_weights = [0.22, 0.18, 0.15, 0.12, 0.10, 0.10, 0.13]
    domains = rng.choice(_DOMAINS, size=n_samples, p=domain_weights)

    # ── Per-domain signal vectors ─────────────────────────────────────────────
    quality_arr   = np.empty(n_samples)
    null_arr      = np.empty(n_samples)
    drift_arr     = np.empty(n_samples)
    has_crit_arr  = np.empty(n_samples, dtype=int)

    for dom in _DOMAINS:
        mask = domains == dom
        if mask.sum() == 0:
            continue
        q, nu, dr, hc = _domain_sample(dom, mask.sum(), rng)
        quality_arr[mask]  = q
        null_arr[mask]     = nu
        drift_arr[mask]    = dr
        has_crit_arr[mask] = hc

    # ── Feature engineering ───────────────────────────────────────────────────
    sample_size_k  = np.clip(base_rows / 1_000.0, 0.001, 50_000.0)
    n_columns      = np.clip(np.round(base_cols), 2, 10_000).astype(float)

    # cv_score: correlated with quality and inversely with null rate
    cv_score = np.clip(
        rng.normal(0.72, 0.16, n_samples)
        + 0.15 * quality_arr
        - 0.25 * null_arr
        - 0.10 * drift_arr,
        0.05, 0.99,
    )

    # anomaly_count: Poisson, heavier tail for drifted runs
    base_lambda = 1.5 + 4.0 * drift_arr + 2.0 * null_arr + has_crit_arr * 3.0
    anomaly_count = rng.poisson(base_lambda).astype(float)

    # flag_severity_max: correlated with anomaly count and critical violations
    sev_prob = np.column_stack([
        np.clip(0.50 - anomaly_count * 0.02, 0.05, 0.80),   # 0
        np.clip(0.25 + anomaly_count * 0.01, 0.10, 0.40),   # 1
        np.clip(0.12 + has_crit_arr * 0.05, 0.02, 0.30),    # 2
        np.clip(0.08 + has_crit_arr * 0.08, 0.01, 0.25),    # 3
        np.clip(0.05 + has_crit_arr * 0.10, 0.005, 0.20),   # 4
    ])
    sev_prob = sev_prob / sev_prob.sum(axis=1, keepdims=True)  # normalise rows
    flag_severity_max = np.array([
        rng.choice(5, p=sev_prob[i]) for i in range(n_samples)
    ], dtype=float)

    # columns_drifted: correlated with drift flag and n_columns
    columns_drifted = np.where(
        drift_arr == 1,
        np.clip(np.round(n_columns * rng.uniform(0.02, 0.35, n_samples)), 1, None),
        rng.integers(0, 2, n_samples).astype(float),   # small random baseline
    )

    # proposer_type_enc: 0–7
    proposer_type_enc = rng.integers(0, 8, n_samples).astype(float)

    X = pd.DataFrame({
        "anomaly_count":    anomaly_count,
        "drift_flag":       drift_arr,
        "quality_score":    quality_arr,
        "null_rate":        null_arr,
        "sample_size_k":    sample_size_k,
        "n_columns":        n_columns,
        "cv_score":         cv_score,
        "flag_severity_max": flag_severity_max,
        "columns_drifted":  columns_drifted,
        "proposer_type_enc": proposer_type_enc,
    })

    # ── Ground truth: composite analyst-approval score ────────────────────────
    # Mimics an expert analyst who cares most about CV score, data quality,
    # absence of critical violations, and manageable drift.
    ideal = (
        0.35 * cv_score
        + 0.25 * quality_arr
        - 0.20 * null_arr
        - 0.12 * drift_arr
        - 0.06 * flag_severity_max / 4.0     # normalise 0→4 to 0→1
        - 0.08 * has_crit_arr                # regulatory CRITICAL hit
        + 0.04 * np.log1p(sample_size_k) / np.log1p(50_000)  # log-normalised N bonus
        - 0.02 * np.log1p(anomaly_count)     # anomaly penalty (log-dampened)
    )

    # Add calibrated label noise (simulates human subjectivity / feedback noise)
    noise = rng.normal(0, 0.08, n_samples)
    final = ideal + noise

    # Dynamic threshold: approx top-40% are labelled "high_confidence"
    thresh = np.percentile(final, 60)
    y = pd.Series((final >= thresh).astype(int), name="label")

    print(f"   Shape: {X.shape}  |  Class balance: "
          f"high={y.mean():.2%}  low={(1-y.mean()):.2%}")
    return X, y


# ── 4. Model definition ────────────────────────────────────────────────────────
def build_model() -> Pipeline:
    """
    Returns an sklearn Pipeline:
      StandardScaler → CalibratedClassifierCV (VotingClassifier)

    VotingClassifier = soft-voting ensemble of:
      • GradientBoostingClassifier  — captures complex interactions
      • RandomForestClassifier      — reduces variance via bagging
      • LogisticRegression          — calibration anchor (linear)
    """
    gbm = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.06,
        max_depth=5,
        min_samples_leaf=40,
        subsample=0.80,
        max_features=0.75,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=15,   # built-in early stopping
        tol=1e-4,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=30,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    lr = LogisticRegression(
        C=0.5,
        max_iter=500,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    voter = VotingClassifier(
        estimators=[("gbm", gbm), ("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[0.40, 0.35, 0.25],
        n_jobs=1,   # voter itself is single-threaded; RF uses its own n_jobs
    )

    # Platt sigmoid calibration on a 20% internal holdout
    calibrated = CalibratedClassifierCV(voter, method="sigmoid", cv=4)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  calibrated),
    ])
    return pipe


# ── 5. Training & evaluation ───────────────────────────────────────────────────
def evaluate_cross_val(pipe: Pipeline, X: pd.DataFrame, y: pd.Series):
    print("\n[3/5] 5-Fold StratifiedKFold cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate on a 60K sub-sample to keep CV runtime reasonable
    idx = np.random.default_rng(0).choice(len(X), size=min(60_000, len(X)), replace=False)
    Xs, ys = X.iloc[idx], y.iloc[idx]

    scores = cross_validate(
        pipe, Xs, ys,
        cv=cv,
        scoring=["roc_auc", "f1", "accuracy"],
        return_train_score=True,
        n_jobs=-1,
    )

    print(f"\n   {'Metric':<20} {'Train':>10} {'Val':>10} {'Gap':>10}")
    print("   " + "─" * 52)
    for metric in ["roc_auc", "f1", "accuracy"]:
        tr = scores[f"train_{metric}"].mean()
        va = scores[f"test_{metric}"].mean()
        print(f"   {metric:<20} {tr:>10.4f} {va:>10.4f} {tr-va:>+10.4f}")

    overfitting_gap = scores["train_roc_auc"].mean() - scores["test_roc_auc"].mean()
    if overfitting_gap > 0.05:
        print(f"\n   ⚠️  AUC gap {overfitting_gap:.4f} > 0.05 — consider stronger regularisation.")
    else:
        print(f"\n   ✅ AUC gap {overfitting_gap:.4f} — no significant overfitting.")

    return scores


def plot_learning_curve(pipe: Pipeline, X: pd.DataFrame, y: pd.Series):
    """Saves a learning-curve PNG to help diagnose bias vs. variance."""
    print("\n   Plotting learning curve (this takes ~90 s on Colab) …")
    idx = np.random.default_rng(1).choice(len(X), size=min(80_000, len(X)), replace=False)
    Xs, ys = X.iloc[idx], y.iloc[idx]

    trn_sz, trn_sc, val_sc = learning_curve(
        pipe, Xs, ys,
        train_sizes=np.linspace(0.10, 1.0, 8),
        cv=StratifiedKFold(3, shuffle=True, random_state=0),
        scoring="roc_auc",
        n_jobs=-1,
        shuffle=True,
    )

    t_mean, t_std = trn_sc.mean(1), trn_sc.std(1)
    v_mean, v_std = val_sc.mean(1), val_sc.std(1)

    plt.figure(figsize=(9, 5))
    plt.fill_between(trn_sz, t_mean - t_std, t_mean + t_std, alpha=0.15, color="steelblue")
    plt.fill_between(trn_sz, v_mean - v_std, v_mean + v_std, alpha=0.15, color="coral")
    plt.plot(trn_sz, t_mean, "o-", color="steelblue", label="Train AUC")
    plt.plot(trn_sz, v_mean, "s-", color="coral",     label="Val AUC")
    plt.xlabel("Training samples"); plt.ylabel("ROC-AUC")
    plt.title("Learning Curve — DIPEX Proposal Confidence Scorer v2")
    plt.legend(); plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.show()
    print("   Saved: learning_curve.png")


def final_evaluation(pipe: Pipeline,
                     X_test: pd.DataFrame,
                     y_test: pd.Series):
    print("\n[4/5] Final evaluation on held-out 20 % test set …\n")
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    auc    = roc_auc_score(y_test, probs)
    brier  = brier_score_loss(y_test, probs)

    print(f"  ROC-AUC       : {auc:.4f}   (target ≥ 0.87)")
    print(f"  Brier Score   : {brier:.4f}  (lower is better; perfect = 0)")
    print()
    print(classification_report(y_test, preds,
                                target_names=["low_confidence", "high_confidence"]))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds,
        display_labels=["low", "high"],
        colorbar=False, ax=ax,
    )
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    # Permutation feature importance (model-agnostic, no leakage)
    print("\n  Computing permutation importance on test set …")
    r = permutation_importance(pipe, X_test, y_test,
                               n_repeats=10, scoring="roc_auc",
                               random_state=42, n_jobs=-1)
    fi = pd.DataFrame({
        "feature":    X_test.columns,
        "importance": r.importances_mean,
        "std":        r.importances_std,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=fi, x="importance", y="feature",
                xerr=fi["std"].values, palette="Blues_r")
    plt.title("Permutation Feature Importance (ROC-AUC drop)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()

    print("\n  Top-10 Features by Permutation Importance:")
    print(fi.to_string(index=False))

    return {"roc_auc": auc, "brier": brier}


# ── 6. Main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Step 1 – OpenML metadata
    openml_meta = fetch_openml_metadata(n_datasets=200)

    # Step 2 – Generate dataset
    X, y = generate_training_data(openml_meta, n_samples=300_000, seed=42)

    # Step 3 – Train / val / test split  (60 / 20 / 20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

    print(f"\n   Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    # Step 4 – Cross-validation check BEFORE final fit
    probe_pipe = build_model()
    cv_scores  = evaluate_cross_val(probe_pipe, X_train, y_train)

    # Step 5 – Learning curve
    try:
        plot_learning_curve(build_model(), X_train, y_train)
    except Exception as e:
        print(f"   [warn] Learning curve skipped: {e}")

    # Step 6 – Final fit on full train split
    print("\n[4/5] Fitting final model on full training split …")
    final_pipe = build_model()
    final_pipe.fit(X_train, y_train)

    # Step 7 – Evaluate on held-out test
    metrics = final_evaluation(final_pipe, X_test, y_test)

    # Step 8 – Abort & warn if quality thresholds not met
    if metrics["roc_auc"] < 0.82:
        print("\n❌  ROC-AUC below minimum threshold (0.82). "
              "Investigate data generation logic before deploying this model.")
    else:
        # Step 9 – Serialise the pipeline
        print("\n[5/5] Saving model …")
        out_path = "proposal_confidence.pkl"
        joblib.dump(final_pipe, out_path, compress=3)

        meta_path = "proposal_confidence_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "version":       "v2",
                "trained_at":    pd.Timestamp.utcnow().isoformat(),
                "n_train":       int(len(X_train)),
                "n_test":        int(len(X_test)),
                "roc_auc":       round(metrics["roc_auc"], 5),
                "brier_score":   round(metrics["brier"],   5),
                "feature_order": list(X.columns),
                "n_features":    len(X.columns),
            }, f, indent=2)

        elapsed = time.time() - t0
        print(f"\n✅  Done in {elapsed/60:.1f} min.")
        print(f"   → {out_path}          (model)")
        print(f"   → {meta_path}  (metadata)")
        print(f"   → learning_curve.png")
        print(f"   → confusion_matrix.png")
        print(f"   → feature_importance.png")
        print()
        print("  NEXT STEP:")
        print("  Download  proposal_confidence.pkl  and copy it to:")
        print("      dipex_project/models/proposal_confidence.pkl")
        print()
        print("  ⚠️  IMPORTANT: The model expects exactly 10 features in this order:")
        for i, col in enumerate(X.columns):
            print(f"      [{i}]  {col}")


if __name__ == "__main__":
    main()
