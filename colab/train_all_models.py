"""
colab/train_all_models.py
--------------------------
Trains ALL DIPEX ML models locally and saves .pkl artifacts to models/.

Run: python colab/train_all_models.py
     (from project root: c:/Users/sankl/Desktop/dipex)

Models produced
---------------
  models/schema_classifier.pkl          + schema_label_encoder.pkl
  models/drift_autoencoder.pkl          + drift_scaler.pkl
  models/pipeline_success_predictor.pkl
  models/nlp_query_classifier.pkl       + nlp_query_vectorizer.pkl
  models/proposal_confidence.pkl
  models/chart_relevance_scorer.pkl

Estimated time: ~3 minutes on i5-1235U + 16 GB RAM.
"""

from __future__ import annotations
import os, sys, time, random
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score
)

os.makedirs("models", exist_ok=True)

def header(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def tick(msg: str, t0: float) -> None:
    print(f"  ✅  {msg}  ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema Classifier
# ─────────────────────────────────────────────────────────────────────────────
header("1 / 6  Schema Semantic-Type Classifier")
t0 = time.time()

SEMANTIC_LABELS = [
    "id","age","amount","date","category","text","phone","email",
    "boolean","zipcode","percentage","score","count","name","unknown"
]

def _synthetic_row(sem_type: str) -> dict:
    r = random.random
    base = {
        "null_rate": round(r()*0.15, 4),
        "unique_rate": round(r(), 4),
        "is_numeric": 0.0, "is_string": 0.0, "is_datetime": 0.0,
        "mean_val": 0.0, "std_val": 0.0, "min_val": 0.0, "max_val": 0.0,
        "skew_val": 0.0, "all_integer": 0.0, "max_lt_200": 0.0,
        "max_lt_1": 0.0, "all_positive": 0.0,
        "n_distinct": round(r()*1000),
        "email_pattern": 0.0, "phone_pattern": 0.0, "mean_str_len": 0.0,
        "high_cardinality": 0.0, "low_cardinality": 0.0, "label": sem_type,
    }
    if sem_type == "id":
        base.update({"is_numeric": r()>0.3, "unique_rate": 0.95+r()*0.05,
                     "all_integer": 1.0, "high_cardinality": 1.0})
    elif sem_type == "age":
        base.update({"is_numeric": 1.0, "mean_val": 30+r()*20,
                     "max_lt_200": 1.0, "all_integer": 1.0,
                     "all_positive": 1.0, "max_val": 80+r()*20})
    elif sem_type == "amount":
        base.update({"is_numeric": 1.0, "mean_val": 500+r()*5000,
                     "std_val": 200+r()*2000, "all_positive": r()>0.2})
    elif sem_type == "date":
        base.update({"is_datetime": 1.0})
    elif sem_type == "email":
        base.update({"is_string": 1.0, "email_pattern": 0.8+r()*0.2,
                     "mean_str_len": 20+r()*15})
    elif sem_type == "phone":
        base.update({"is_string": 1.0, "phone_pattern": 0.7+r()*0.3,
                     "mean_str_len": 10+r()*5})
    elif sem_type == "category":
        base.update({"is_string": 1.0, "unique_rate": 0.001+r()*0.05,
                     "low_cardinality": 1.0})
    elif sem_type == "text":
        base.update({"is_string": 1.0, "mean_str_len": 50+r()*200,
                     "unique_rate": 0.8+r()*0.2})
    elif sem_type == "boolean":
        base.update({"unique_rate": 0.001, "low_cardinality": 1.0,
                     "max_lt_1": 1.0})
    elif sem_type == "percentage":
        base.update({"is_numeric": 1.0, "max_lt_1": 1.0,
                     "all_positive": 1.0, "max_val": r()})
    elif sem_type == "score":
        base.update({"is_numeric": 1.0, "max_lt_200": 1.0, "all_positive": 1.0})
    elif sem_type == "count":
        base.update({"is_numeric": 1.0, "all_integer": 1.0, "all_positive": 1.0})
    elif sem_type == "name":
        base.update({"is_string": 1.0, "mean_str_len": 10+r()*15,
                     "unique_rate": 0.7+r()*0.3})
    elif sem_type == "zipcode":
        base.update({"is_string": r()>0.5, "mean_str_len": 5+r()*2})
    return base

print("  Generating 7,500 synthetic rows ...")
rows = [_synthetic_row(lbl) for lbl in SEMANTIC_LABELS for _ in range(500)]
df_sc = pd.DataFrame(rows)
feat_cols = [c for c in df_sc.columns if c != "label"]
X_sc = df_sc[feat_cols].values.astype(np.float32)
le = LabelEncoder()
y_sc = le.fit_transform(df_sc["label"].values)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_sc, y_sc, test_size=0.15, random_state=42, stratify=y_sc
)
clf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=2,
    class_weight="balanced", n_jobs=-1, random_state=42
)
clf.fit(X_tr, y_tr)
val_acc = accuracy_score(y_val, clf.predict(X_val))
print(f"  Val Accuracy: {val_acc:.4f}")
print(classification_report(y_val, clf.predict(X_val),
                             target_names=le.classes_, zero_division=0))
joblib.dump(clf, "models/schema_classifier.pkl")
joblib.dump(le,  "models/schema_label_encoder.pkl")
tick("schema_classifier.pkl + schema_label_encoder.pkl saved", t0)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Drift Autoencoder
# ─────────────────────────────────────────────────────────────────────────────
header("2 / 6  Drift Autoencoder")
t0 = time.time()

rng = np.random.default_rng(42)
N = 5000
df_drift = pd.DataFrame(rng.standard_normal((N, 10)),
                         columns=[f"feat_{i}" for i in range(10)])
df_drift["amount"] = rng.exponential(scale=1000, size=N)
df_drift["age"]    = rng.integers(18, 90, size=N).astype(float)
df_drift["score"]  = rng.uniform(0, 1, size=N)

num_cols = df_drift.select_dtypes(include="number").columns.tolist()
X_ae = df_drift[num_cols].fillna(0).values
scaler_ae = StandardScaler()
X_sc_ae = scaler_ae.fit_transform(X_ae)

X_tr_ae, X_val_ae = train_test_split(X_sc_ae, test_size=0.1, random_state=42)
dim = X_ae.shape[1]
h = (max(dim*2, 8), max(dim, 4), max(dim*2, 8))
ae = MLPRegressor(
    hidden_layer_sizes=h, activation="relu", solver="adam",
    max_iter=500, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=20, random_state=42, verbose=False
)
ae.fit(X_tr_ae, X_tr_ae)

val_pred = ae.predict(X_val_ae)
val_err  = np.mean(np.square(X_val_ae - val_pred), axis=1)
p95      = np.percentile(val_err, 95)
print(f"  Val MSE mean: {val_err.mean():.6f}  |  P95 threshold: {p95:.6f}")

joblib.dump(ae,       "models/drift_autoencoder.pkl")
joblib.dump(scaler_ae,"models/drift_scaler.pkl")
tick("drift_autoencoder.pkl + drift_scaler.pkl saved", t0)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Pipeline Success Predictor
# ─────────────────────────────────────────────────────────────────────────────
header("3 / 6  Pipeline Success Predictor")
t0 = time.time()

PSP_FEATURES = [
    "null_rate","drift_detected","quality_score","row_count_k",
    "n_columns","anomaly_count","schema_match","known_dataset",
    "cv_score","columns_drifted"
]

def _gen_psp(n=4000):
    rng2 = np.random.default_rng(42)
    df = pd.DataFrame({
        "null_rate":       rng2.beta(1, 10, n),
        "drift_detected":  rng2.choice([0,1], p=[0.7,0.3], size=n).astype(float),
        "quality_score":   rng2.beta(8, 2, n),
        "row_count_k":     rng2.exponential(10, n),
        "n_columns":       rng2.integers(3, 60, n).astype(float),
        "anomaly_count":   rng2.poisson(2, n).astype(float),
        "schema_match":    rng2.choice([0,1], p=[0.1,0.9], size=n).astype(float),
        "known_dataset":   rng2.choice([0,1], p=[0.15,0.85], size=n).astype(float),
        "cv_score":        rng2.beta(5, 2, n),
        "columns_drifted": rng2.poisson(1, n).astype(float),
    })
    logit = (
        -df["null_rate"]*4 - df["drift_detected"]*2.5
        + df["quality_score"]*3 + np.log1p(df["row_count_k"])*0.3
        + df["schema_match"]*2 + df["cv_score"]*2.5
        - df["anomaly_count"]*0.3 - df["columns_drifted"]*0.4
        + rng2.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["success"] = (rng2.uniform(size=n) < prob).astype(int)
    return df

df_psp = _gen_psp()
X_psp = df_psp[PSP_FEATURES].values
y_psp = df_psp["success"].values
X_ptr, X_pval, y_ptr, y_pval = train_test_split(
    X_psp, y_psp, test_size=0.15, random_state=42, stratify=y_psp
)
clf_psp = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", n_jobs=-1, random_state=42
)
clf_psp.fit(X_ptr, y_ptr)
psp_auc = roc_auc_score(y_pval, clf_psp.predict_proba(X_pval)[:,1])
cv_psp  = cross_val_score(clf_psp, X_psp, y_psp, cv=5,
                           scoring="roc_auc", n_jobs=-1)
print(f"  Val ROC-AUC: {psp_auc:.4f}  |  5-Fold CV: {cv_psp.mean():.4f} ± {cv_psp.std():.4f}")
print(classification_report(y_pval, clf_psp.predict(X_pval),
                             target_names=["failure","success"], zero_division=0))
joblib.dump(clf_psp, "models/pipeline_success_predictor.pkl")
tick("pipeline_success_predictor.pkl saved", t0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. NLP Query Classifier
# ─────────────────────────────────────────────────────────────────────────────
header("4 / 6  NLP Query Intent Classifier")
t0 = time.time()

SEED_CORPUS = [
    ("show me top 10 customers by revenue","top_n"),
    ("top 5 products by sales","top_n"),
    ("best 20 regions by profit","top_n"),
    ("highest revenue accounts","top_n"),
    ("which customers have the most orders","top_n"),
    ("show top 50 rows by score","top_n"),
    ("worst 10 products by margin","bottom_n"),
    ("bottom 5 stores by sales","bottom_n"),
    ("lowest revenue segments","bottom_n"),
    ("least active users","bottom_n"),
    ("what is the total revenue","aggregate"),
    ("sum of sales by region","aggregate"),
    ("average order value","aggregate"),
    ("mean income per segment","aggregate"),
    ("total profit last year","aggregate"),
    ("what is the max price","aggregate"),
    ("minimum cost per product","aggregate"),
    ("calculate total earnings","aggregate"),
    ("show customers where age > 30","filter"),
    ("filter revenue greater than 1000","filter"),
    ("find rows where status is active","filter"),
    ("customers from New York","filter"),
    ("orders where quantity > 5","filter"),
    ("show records with nulls in revenue","filter"),
    ("sales trend over time","trend"),
    ("show monthly revenue trend","trend"),
    ("how has profit changed over the year","trend"),
    ("plot revenue by month","trend"),
    ("weekly trend for orders","trend"),
    ("compare revenue across regions","compare"),
    ("difference between segment A and B","compare"),
    ("year over year comparison","compare"),
    ("correlation between age and income","correlation"),
    ("is revenue correlated with spend","correlation"),
    ("relationship between price and demand","correlation"),
    ("distribution of revenue","distribution"),
    ("histogram of age","distribution"),
    ("spread of order values","distribution"),
    ("how many unique customers","count_distinct"),
    ("distinct products sold","count_distinct"),
    ("number of unique orders","count_distinct"),
    ("revenue by category","group_by"),
    ("group sales by region and product","group_by"),
    ("average price per brand","group_by"),
    ("breakdown by segment","group_by"),
    ("daily sales for last 30 days","time_series"),
    ("monthly revenue last 12 months","time_series"),
    ("show order count by week","time_series"),
    ("last 7 days performance","time_series"),
    ("show me the data","general"),
    ("describe the table","general"),
    ("what are the columns","general"),
    ("give me a summary","general"),
    ("overview of the dataset","general"),
]

texts  = [t for t,_ in SEED_CORPUS]
labels = [l for _,l in SEED_CORPUS]
vec  = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
svc  = LinearSVC(max_iter=3000, random_state=42)
pipe = SKPipeline([("vec", vec), ("svc", svc)])
cv_nlp = cross_val_score(pipe, texts, labels, cv=5, scoring="accuracy")
print(f"  5-Fold CV Accuracy: {cv_nlp.mean():.4f} ± {cv_nlp.std():.4f}")
pipe.fit(texts, labels)
joblib.dump(pipe.named_steps["svc"], "models/nlp_query_classifier.pkl")
joblib.dump(pipe.named_steps["vec"], "models/nlp_query_vectorizer.pkl")
tick("nlp_query_classifier.pkl + nlp_query_vectorizer.pkl saved", t0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Proposal Confidence Scorer
# ─────────────────────────────────────────────────────────────────────────────
header("5 / 6  Proposal Confidence Scorer")
t0 = time.time()

PROP_FEATURES = [
    "anomaly_count","drift_flag","quality_score","null_rate",
    "sample_size_k","n_columns","cv_score","flag_severity_max",
    "columns_drifted","proposer_type_enc"
]

def _gen_proposal(n=3000):
    rng3 = np.random.default_rng(99)
    df = pd.DataFrame({
        "anomaly_count":    rng3.poisson(3, n).astype(float),
        "drift_flag":       rng3.choice([0,1], p=[0.6,0.4], size=n).astype(float),
        "quality_score":    rng3.beta(6, 2, n),
        "null_rate":        rng3.beta(1, 8, n),
        "sample_size_k":    rng3.exponential(20, n),
        "n_columns":        rng3.integers(3, 60, n).astype(float),
        "cv_score":         rng3.beta(5, 2, n),
        "flag_severity_max":rng3.integers(1, 5, n).astype(float),
        "columns_drifted":  rng3.poisson(1, n).astype(float),
        "proposer_type_enc":rng3.integers(-1, 8, n).astype(float),
    })
    logit = (df["cv_score"]*3 + df["quality_score"]*2
             - df["null_rate"]*3 - df["drift_flag"]*1.5
             + rng3.normal(0, 0.5, n))
    prob = 1 / (1 + np.exp(-logit))
    df["high_conf"] = (rng3.uniform(size=n) < prob).astype(int)
    return df

df_prop = _gen_proposal()
X_prop = df_prop[PROP_FEATURES].values
y_prop = df_prop["high_conf"].values
X_ptr2, X_pval2, y_ptr2, y_pval2 = train_test_split(
    X_prop, y_prop, test_size=0.2, random_state=42, stratify=y_prop
)
clf_prop = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight="balanced",
    n_jobs=-1, random_state=42
)
clf_prop.fit(X_ptr2, y_ptr2)
pauc = roc_auc_score(y_pval2, clf_prop.predict_proba(X_pval2)[:,1])
print(f"  Val ROC-AUC: {pauc:.4f}")
print(classification_report(y_pval2, clf_prop.predict(X_pval2),
                             target_names=["low_conf","high_conf"], zero_division=0))
joblib.dump(clf_prop, "models/proposal_confidence.pkl")
tick("proposal_confidence.pkl saved", t0)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Chart Relevance Scorer
# ─────────────────────────────────────────────────────────────────────────────
header("6 / 6  Chart Relevance Scorer")
t0 = time.time()

CHART_FEATURES = [
    "row_density","col_density","num_ratio","cat_ratio","first_cat_card",
    "skew_val","mean_corr","null_rate","has_dt","intent_enc"
]
CHART_TYPES = ["bar","line","scatter","heatmap","histogram","box","pie"]

def _gen_chart(n=3000):
    rng4 = np.random.default_rng(77)
    df = pd.DataFrame({
        "row_density":    rng4.uniform(0, 1, n),
        "col_density":    rng4.uniform(0, 1, n),
        "num_ratio":      rng4.uniform(0, 1, n),
        "cat_ratio":      rng4.uniform(0, 1, n),
        "first_cat_card": rng4.uniform(0, 0.3, n),
        "skew_val":       rng4.normal(0, 3, n),
        "mean_corr":      rng4.uniform(0, 1, n),
        "null_rate":      rng4.beta(1, 10, n),
        "has_dt":         rng4.choice([0,1], p=[0.5,0.5], size=n).astype(float),
        "intent_enc":     rng4.uniform(0, 1, n),
    })
    def _best(row):
        if row["has_dt"] > 0.5 or row["intent_enc"] < 0.2: return "line"
        if row["cat_ratio"] > 0.6 and row["first_cat_card"] < 0.15: return "pie"
        if row["cat_ratio"] > 0.3: return "bar"
        if row["mean_corr"] > 0.7: return "scatter"
        if abs(row["skew_val"]) > 2: return "histogram"
        if row["num_ratio"] > 0.8 and row["col_density"] > 0.5: return "heatmap"
        return "box"
    df["best_chart"] = df.apply(_best, axis=1)
    return df

df_chart = _gen_chart()
X_chart = df_chart[CHART_FEATURES].values
y_chart = df_chart["best_chart"].values
X_ctr, X_cval, y_ctr, y_cval = train_test_split(
    X_chart, y_chart, test_size=0.2, random_state=42
)
clf_chart = RandomForestClassifier(
    n_estimators=200, max_depth=8, n_jobs=-1, random_state=42
)
clf_chart.fit(X_ctr, y_ctr)
cacc = accuracy_score(y_cval, clf_chart.predict(X_cval))
print(f"  Val Accuracy: {cacc:.4f}")
print(classification_report(y_cval, clf_chart.predict(X_cval), zero_division=0))
joblib.dump(clf_chart, "models/chart_relevance_scorer.pkl")
tick("chart_relevance_scorer.pkl saved", t0)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ALL MODELS TRAINED SUCCESSFULLY")
print("="*60)
saved = [f for f in os.listdir("models") if f.endswith(".pkl")]
for f in sorted(saved):
    size_kb = os.path.getsize(f"models/{f}") // 1024
    print(f"  ✅  models/{f}  ({size_kb} KB)")
print(f"\n  Total: {len(saved)} .pkl files in models/")
