"""
DIPEX â€” Industry-Grade Colab Training Script PART 1 (Models 1-3)
=================================================================
Run on Google Colab (T4/A100). After this, run train_colab_part2.py.

INSTRUCTIONS:
1. Upload both part files to Colab
2. Run: !pip install -q openml xgboost lightgbm mapie shap
3. Mount Drive when prompted
4. Run this file, then part2
"""
import os, bz2, json, time, warnings, urllib.request, hashlib, datetime
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, mean_squared_error)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.datasets import fetch_openml, fetch_kddcup99
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# â”€â”€ Google Drive Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    from google.colab import drive
    drive.mount("/content/drive")
    SAVE_DIR = "/content/drive/MyDrive/dipex_models"
except Exception:
    SAVE_DIR = "models"   # local fallback

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
np.random.seed(42)

# â”€â”€ Shared utilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def header(title):
    print(f"\n{'â•'*65}\n  {title}\n{'â•'*65}")

def split3(X, y, stratify=True):
    """Strict 3-way split. No data leakage."""
    s = y if stratify else None
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=s)
    s2 = y_tmp if stratify else None
    X_v, X_te, y_v, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=s2)
    return X_tr, X_v, X_te, y_tr, y_v, y_te

def safe_df(df, target_col):
    """Convert Categorical dtypes, get_dummies, return float32 X and encoded y."""
    feat = df.drop(columns=[target_col]).copy()
    for c in feat.columns:
        if hasattr(feat[c], "cat"):
            feat[c] = feat[c].astype(str)
    feat = feat.fillna("__NA__")
    X = pd.get_dummies(feat, drop_first=True).values.astype(np.float32)
    tgt = df[target_col].copy()
    if hasattr(tgt, "cat"):
        tgt = tgt.astype(str)
    y = LabelEncoder().fit_transform(tgt.fillna("missing"))
    return X, y

import copy
def _lc_clone(model):
    """Strip early_stopping_rounds/early_stopping before passing to learning_curve.
    XGBoost/LightGBM need eval_set when early_stopping is set,
    but sklearn's learning_curve never passes eval_set internally.
    """
    m = copy.deepcopy(model)
    try:
        m.set_params(early_stopping_rounds=None)
    except Exception:
        pass
    try:
        m.set_params(early_stopping=False)
    except Exception:
        pass
    try:
        m.set_params(n_estimators=300)
    except Exception:
        pass
    return m

def overfit_report(name, model, X_tr, y_tr, X_v, y_v, X_te, y_te,
                   dummy_score, scoring="accuracy"):
    """
    Full overfitting / underfitting diagnosis.
    Prints learning curve + explicit verdict.
    """
    header(f"ðŸ”¬ DIAGNOSIS â€” {name}")
    pred_fn = model.predict

    tr_s  = accuracy_score(y_tr, pred_fn(X_tr))
    val_s = accuracy_score(y_v,  pred_fn(X_v))
    te_s  = accuracy_score(y_te, pred_fn(X_te))
    gap   = tr_s - val_s

    print(f"  Dummy baseline  : {dummy_score:.4f}")
    print(f"  Train accuracy  : {tr_s:.4f}")
    print(f"  Val   accuracy  : {val_s:.4f}")
    print(f"  Test  accuracy  : {te_s:.4f}")
    print(f"  Train-Val gap   : {gap:.4f}")
    improvement = val_s - dummy_score

    # Learning curve â€” _lc_clone strips early_stopping_rounds so XGB/LGB won't crash
    X_all = np.vstack([X_tr, X_v])
    y_all = np.concatenate([y_tr, y_v])
    sizes = np.linspace(0.10, 1.0, 6)
    print(f"\n  \U0001f4c8 Learning Curve")
    print(f"  {'N_train':>8}  {'Train':>7}  {'Val':>7}  {'Gap':>7}  Status")
    try:
        lc_model = _lc_clone(model)
        lc_sizes, tr_lc, val_lc = learning_curve(
            lc_model, X_all, y_all, train_sizes=sizes, cv=3,
            scoring=scoring, n_jobs=-1, error_score=0.0)
        for n, tr, vl in zip(lc_sizes, tr_lc.mean(1), val_lc.mean(1)):
            g = tr - vl
            flag = "âš  OVERFIT" if g > 0.10 else "âœ…"
            print(f"  {n:>8d}  {tr:>7.4f}  {vl:>7.4f}  {g:>7.4f}  {flag}")
    except Exception as lc_err:
        # Custom models (e.g. StackedPSP) may not implement sklearn fit() interface
        print(f"  âš  Learning curve skipped ({type(lc_err).__name__})")
        print(f"    Manual: Train={tr_s:.4f}  Val={val_s:.4f}  Test={te_s:.4f}  Gap={gap:.4f}")

    # Verdict
    print(f"\n  VERDICT:")
    if gap > 0.10:
        print(f"  âŒ OVERFIT â€” Train-Val gap={gap:.3f} > 0.10")
        print(f"     Fix: increase regularization, more data, reduce depth")
    elif improvement < 0.10:
        print(f"  âŒ UNDERFIT â€” only {improvement:.3f} above dummy")
        print(f"     Fix: more features, larger model, more training data")
    else:
        print(f"  âœ… HEALTHY â€” gap={gap:.3f}, beats dummy by +{improvement:.3f}")

    return {"train": tr_s, "val": val_s, "test": te_s,
            "gap": gap, "dummy": dummy_score, "improvement": improvement}

def save(model, name, metrics: dict, extra: dict = None):
    path = f"{SAVE_DIR}/{name}.pkl"
    joblib.dump(model, path)
    joblib.dump(model, f"models/{name}.pkl")
    entry = {
        "version": "2.0",
        "trained_at": datetime.datetime.utcnow().isoformat(),
        "metrics": metrics,
        **(extra or {}),
    }
    print(f"\n  âœ…  Saved â†’ {path}")
    return entry

def load_openml_safe(name):
    try:
        ds = fetch_openml(name=name, version="active", as_frame=True, parser="auto")
        df = ds.frame if hasattr(ds, "frame") else pd.concat(
            [ds.data, ds.target.rename("__target__")], axis=1)
        return df
    except Exception as e:
        print(f"    âš  {name}: {e}")
        return None

import copy
def _lc_clone(model):
    """Clone a model without early_stopping_rounds for use inside learning_curve.
    XGBoost/LightGBM require eval_set when early_stopping_rounds is set,
    but learning_curve never passes eval_set internally.
    """
    m = copy.deepcopy(model)
    try:
        # XGBoost
        m.set_params(early_stopping_rounds=None, n_estimators=300)
    except Exception:
        pass
    try:
        # LightGBM
        m.set_params(early_stopping_rounds=None, n_estimators=300)
    except Exception:
        pass
    # Remove early_stopping attr for HistGBT / sklearn estimators
    if hasattr(m, 'early_stopping'):
        try: m.set_params(early_stopping=False)
        except Exception: pass
    return m

registry = {}

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODEL 1 â€” SCHEMA SEMANTIC TYPE CLASSIFIER
# Architecture: XGBoost on 25 real column-level statistical features
# Data: 100+ OpenML datasets â†’ ~25,000 labeled column instances
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
header("MODEL 1 â€” Schema Semantic Type Classifier")
t0 = time.time()

SCHEMA_DATASETS = [
    "adult","titanic","iris","wine","diabetes","breast-cancer","heart-c",
    "glass","vehicle","segment","letter","abalone","bank-marketing","eeg-eye-state",
    "credit-g","yeast","hypothyroid","anneal","mushroom","car","waveform-5000",
    "blood-transfusion-service-center","monks-problems-1","splice","optdigits",
    "kr-vs-kp","tic-tac-toe","sonar","ionosphere","mfeat-factors","kc1",
    "mfeat-karhunen","mfeat-zernike","australian","hepatitis","dermatology",
    "balance-scale","ecoli","haberman","heart-statlog","colic",
    "page-blocks","pendigits","primary-tumor","soybean","zoo","lymph",
    "vote","flags","spambase",
    "dna","JapaneseVowels","autos","har",
]

SEMANTIC_RULES = {
    "email":      lambda s: bool(s.str.contains(r"@.*\.", regex=True, na=False).mean() > 0.3),
    "phone":      lambda s: bool(s.str.replace(r"\D","",regex=True).str.len().between(7,15).mean() > 0.5) if s.dtype == object else False,
    "url":        lambda s: bool(s.str.contains(r"https?://|www\.", regex=True, na=False).mean() > 0.2) if s.dtype == object else False,
    "boolean":    lambda s: s.nunique() <= 2,
    "id":         lambda s: s.is_unique and len(s) > 10,
    "date":       lambda s: bool(s.astype(str).str.match(r"\d{4}[-/]\d{2}").any()) if s.dtype == object else False,
    "percentage": lambda s: (s.between(0,100).mean() > 0.9) if pd.api.types.is_numeric_dtype(s) else False,
    "score":      lambda s: (s.between(0,10).mean() > 0.9) if pd.api.types.is_numeric_dtype(s) else False,
    "currency":   lambda s: bool(s.astype(str).str.contains(r"[$â‚¬Â£Â¥]|\d+\.\d{2}$", regex=True).mean() > 0.2) if s.dtype == object else False,
    "category":   lambda s: s.dtype == object and 2 < s.nunique() <= 30,
    "numeric":    lambda s: pd.api.types.is_numeric_dtype(s) and s.nunique() > 20,
    "text":       lambda s: s.dtype == object and s.str.split().str.len().mean() > 3 if s.dtype == object else False,
}

def col_features(series: pd.Series) -> dict | None:
    try:
        s = series.dropna()
        if len(s) < 10:
            return None
        is_num = pd.api.types.is_numeric_dtype(s)
        ss = s.astype(str)
        f = {
            "null_rate":        float(series.isnull().mean()),
            "unique_ratio":     float(s.nunique() / max(len(s), 1)),
            "digit_ratio":      float(ss.str.replace(r"\D","",regex=True).str.len().mean() / max(ss.str.len().mean(), 1)),
            "alpha_ratio":      float(ss.str.replace(r"[^a-zA-Z]","",regex=True).str.len().mean() / max(ss.str.len().mean(), 1)),
            "mean_len":         float(ss.str.len().mean()),
            "std_len":          float(ss.str.len().std() or 0),
            "has_at":           float(ss.str.contains("@").mean()),
            "has_dot":          float(ss.str.contains(r"\.", regex=True).mean()),
            "has_slash":        float(ss.str.contains("/").mean()),
            "has_dollar":       float(ss.str.contains(r"[$â‚¬Â£]", regex=True).mean()),
            "is_numeric_dtype": float(is_num),
            "num_unique":       float(s.nunique()),
            "entropy":          float(-(s.value_counts(normalize=True).apply(np.log2) * s.value_counts(normalize=True)).sum()) if s.nunique() > 1 else 0.0,
            "zero_ratio":       float((s == 0).mean()) if is_num else 0.0,
            "negative_ratio":   float((s < 0).mean()) if is_num else 0.0,
            "min_val":          float(s.min()) if is_num else 0.0,
            "max_val":          float(s.max()) if is_num else 0.0,
            "mean_val":         float(s.mean()) if is_num else 0.0,
            "skewness":         float(s.skew()) if is_num else 0.0,
            "between_0_1":      float(s.between(0,1).mean()) if is_num else 0.0,
            "between_0_100":    float(s.between(0,100).mean()) if is_num else 0.0,
            "between_0_10":     float(s.between(0,10).mean()) if is_num else 0.0,
            "is_unique":        float(len(s) == s.nunique()),
            "top1_freq":        float(s.value_counts(normalize=True).iloc[0]) if s.nunique() > 0 else 1.0,
        }
        return f
    except Exception:
        return None

def infer_label(series: pd.Series) -> str | None:
    for label, rule in SEMANTIC_RULES.items():
        try:
            if rule(series.dropna()):
                return label
        except Exception:
            pass
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "category"

print("  Extracting column instances from OpenML datasets...")
schema_rows = []
for ds_name in SCHEMA_DATASETS:
    df = load_openml_safe(ds_name)
    if df is None:
        continue
    for col in df.columns:
        feat = col_features(df[col])
        if feat is None:
            continue
        label = infer_label(df[col])
        if label is None:
            continue
        feat["label"] = label
        schema_rows.append(feat)

df_sc = pd.DataFrame(schema_rows)
ct = df_sc["label"].value_counts()
df_sc = df_sc[df_sc["label"].isin(ct[ct >= 10].index)].copy()
print(f"  Column instances: {len(df_sc):,}")
print(df_sc["label"].value_counts().to_string())

SFEAT = [c for c in df_sc.columns if c != "label"]
X_sc = df_sc[SFEAT].values.astype(np.float32)
y_sc_raw = df_sc["label"].values
le_sc = LabelEncoder()
y_sc = le_sc.fit_transform(y_sc_raw)
joblib.dump(le_sc, f"{SAVE_DIR}/schema_label_encoder.pkl")
joblib.dump(le_sc, "models/schema_label_encoder.pkl")

X_sctr, X_scv, X_scte, y_sctr, y_scv, y_scte = split3(X_sc, y_sc, stratify=True)
print(f"  Split: train={len(X_sctr):,}  val={len(X_scv):,}  test={len(X_scte):,}")

dummy_sc = DummyClassifier(strategy="most_frequent").fit(X_sctr, y_sctr)
dummy_score_sc = accuracy_score(y_scv, dummy_sc.predict(X_scv))

clf_sc = XGBClassifier(
    n_estimators=1000, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
    use_label_encoder=False, eval_metric="mlogloss",
    early_stopping_rounds=40, n_jobs=-1, random_state=42,
    verbosity=0,
)
clf_sc.fit(X_sctr, y_sctr,
           eval_set=[(X_scv, y_scv)], verbose=False)

metrics_sc = overfit_report(
    "Schema Classifier", clf_sc,
    X_sctr, y_sctr, X_scv, y_scv, X_scte, y_scte, dummy_score_sc)

print(f"\n  Classification Report (test):")
print(classification_report(y_scte, clf_sc.predict(X_scte),
      target_names=le_sc.classes_, zero_division=0))

# SHAP
print("  Computing SHAP feature importance...")
explainer_sc = shap.TreeExplainer(clf_sc)
shap_vals = explainer_sc.shap_values(X_scv[:200])
shap.summary_plot(shap_vals, X_scv[:200], feature_names=SFEAT,
                  plot_type="bar", show=False)
plt.savefig(f"{SAVE_DIR}/schema_shap.png", bbox_inches="tight")
plt.close()

registry["schema_classifier"] = save(clf_sc, "schema_classifier", metrics_sc,
    {"n_samples": len(df_sc), "n_classes": len(le_sc.classes_),
     "best_iteration": clf_sc.best_iteration})
print(f"  â± Model 1 done in {time.time()-t0:.0f}s")

# Save Model 1 registry entry
joblib.dump(registry, f"{SAVE_DIR}/_registry_part1.pkl")
print(f"\n  âœ…  Model 1 registry saved â†’ {SAVE_DIR}/_registry_part1.pkl")
print(f"  Registry so far: {list(registry.keys())}")

print("""
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  âœ…  Part 1 complete â€” Model 1 (Schema Classifier) trained.

  Next steps (run each in a NEW cell):
    %run train_model2_drift.py    â† Model 2: Drift Detector
    %run train_model3_psp.py      â† Model 3: Pipeline Success Predictor
    %run train_colab_part2.py     â† Models 4, 5, 6
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
""")

