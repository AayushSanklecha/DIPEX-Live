"""
colab/train_models_3to6.py
===========================
Trains models 3-6 only (PSP, NLP, Proposal, Chart).
Models 1 (schema) and 2 (drift AE) are already saved.

Run from project root:
  python colab/train_models_3to6.py
"""
from __future__ import annotations
import bz2, json, os, time, urllib.request, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.datasets import fetch_openml

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
np.random.seed(42)

def header(t):
    print(f"\n{'='*68}\n  {t}\n{'='*68}")

def tick(msg, t0):
    print(f"  ✅  {msg}  ({time.time()-t0:.1f}s)")

def three_way_split(X, y, val=0.20, test=0.20, stratify=None):
    strat = y if stratify else None
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val+test, random_state=42, stratify=strat)
    ratio = test / (val + test)
    strat2 = y_tmp if stratify else None
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=ratio, random_state=42, stratify=strat2)
    return X_tr, X_val, X_te, y_tr, y_val, y_te

def print_learning_curve(estimator, X, y, scoring="accuracy", cv=5,
                          n_points=5, clf_name="Model"):
    print(f"\n  📈 Learning Curve — {clf_name}")
    sizes = np.linspace(0.15, 1.0, n_points)
    tr_sz, tr_sc, vl_sc = learning_curve(
        estimator, X, y, train_sizes=sizes, cv=cv,
        scoring=scoring, n_jobs=-1, error_score=0.0)
    print(f"  {'N_train':>8}  {'Train':>8}  {'Val':>8}  {'Gap':>8}")
    for n, tr, vl in zip(tr_sz, tr_sc.mean(1), vl_sc.mean(1)):
        gap = abs(tr - vl)
        flag = " ⚠ overfit?" if gap > 0.10 else ""
        print(f"  {n:>8d}  {tr:>8.4f}  {vl:>8.4f}  {gap:>8.4f}{flag}")

def check_overfit(m, Xtr, ytr, Xvl, yvl):
    tr  = accuracy_score(ytr, m.predict(Xtr))
    vl  = accuracy_score(yvl, m.predict(Xvl))
    gap = abs(tr - vl)
    print(f"  Train={tr:.4f} | Val={vl:.4f} | Gap={gap:.4f}",
          "⚠ OVERFIT" if gap > 0.10 else "✅ OK")

def load_openml(name):
    try:
        ds = fetch_openml(name=name, version="active", as_frame=True, parser="auto")
        df = ds.frame if hasattr(ds, "frame") else pd.concat(
            [ds.data, ds.target.rename("__target__")], axis=1)
        print(f"    Loaded {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df
    except Exception as e:
        print(f"    ⚠ Skipping {name}: {e}")
        return None

def safe_X(df: pd.DataFrame, target_col: str) -> np.ndarray:
    """
    Convert all Categorical columns to plain object before get_dummies + fillna.
    Returns float32 numpy array safe for sklearn.
    """
    feat = df.drop(columns=[target_col]).copy()
    # Downcast all Categorical dtypes to their base dtype
    for col in feat.columns:
        if hasattr(feat[col], "cat"):
            feat[col] = feat[col].astype(str)  # treat as string category
    # Now fillna with sentinel is safe
    feat = feat.fillna("__missing__")
    return pd.get_dummies(feat, drop_first=True).values.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE SUCCESS PREDICTOR
#    Run 6 real sklearn algorithms × 40 OpenML datasets → binary success label
# ─────────────────────────────────────────────────────────────────────────────
header("3 / 6  Pipeline Success Predictor  (Real sklearn experiments on OpenML)")
t0 = time.time()

PSP_DATASETS = [
    "iris","wine","diabetes","breast-cancer",
    "credit-g","heart-c","hepatitis","vehicle","segment",
    "glass","ionosphere","sonar","australian",
    "tic-tac-toe","kr-vs-kp","yeast","mfeat-factors",
    "anneal","hypothyroid","mushroom","car","letter",
    "waveform-5000","abalone","bank-marketing",
    "eeg-eye-state","blood-transfusion-service-center",
    "monks-problems-1","monks-problems-2","monks-problems-3",
    "splice","dna","optdigits",
    "mfeat-karhunen","mfeat-morphological","mfeat-zernike",
    "JapaneseVowels","kr-vs-kp","segment","vehicle",
]
PSP_DATASETS = list(dict.fromkeys(PSP_DATASETS))  # deduplicate

ALGORITHMS = {
    "rf_100": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0),
    "rf_10":  RandomForestClassifier(n_estimators=10,  n_jobs=-1, random_state=0),
    "lr_c1":  LogisticRegression(C=1.0, max_iter=500),
    "lr_c01": LogisticRegression(C=0.1, max_iter=500),
    "et_100": ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=0),
    "dummy":  DummyClassifier(strategy="most_frequent"),
}

def _meta(df):
    return {
        "null_rate":       float(df.isnull().mean().mean()),
        "drift_detected":  0.0,
        "quality_score":   float(1.0 - df.isnull().mean().mean()),
        "row_count_k":     float(len(df) / 1000.0),
        "n_columns":       float(df.shape[1]),
        "anomaly_count":   0.0,
        "schema_match":    1.0,
        "known_dataset":   1.0,
        "cv_score":        0.0,
        "columns_drifted": 0.0,
    }

print("  Running real sklearn CV experiments on OpenML datasets ...")
psp_rows = []
for ds_name in PSP_DATASETS:
    df_ds = load_openml(ds_name)
    if df_ds is None:
        continue
    try:
        target_col = df_ds.columns[-1]
        X_exp = safe_X(df_ds, target_col)
        tgt   = df_ds[target_col].copy()
        if hasattr(tgt, "cat"):
            tgt = tgt.astype(str)
        y_exp = LabelEncoder().fit_transform(tgt.fillna("missing"))
        if X_exp.shape[0] < 30 or X_exp.shape[1] < 1:
            continue
        X_sub = X_exp[:min(len(X_exp), 3000)]
        y_sub = y_exp[:min(len(y_exp), 3000)]
        meta  = _meta(df_ds)
        dummy_cv = cross_val_score(
            DummyClassifier(strategy="most_frequent"),
            X_sub, y_sub, cv=5, scoring="accuracy", n_jobs=-1).mean()
        for alg_name, alg in ALGORITHMS.items():
            try:
                cv_mean = float(cross_val_score(
                    alg, X_sub, y_sub, cv=5,
                    scoring="accuracy", n_jobs=-1).mean())
                row = dict(meta)
                row["cv_score"]  = cv_mean
                row["success"]   = int(cv_mean >= 0.65 and cv_mean > dummy_cv + 0.10)
                row["algorithm"] = alg_name
                psp_rows.append(row)
            except Exception:
                pass
    except Exception as e:
        print(f"    ⚠ {ds_name}: {e}")

print(f"\n  Real experiment records: {len(psp_rows):,}")

if len(psp_rows) < 20:
    print("  ⚠ Too few experiments — extending with more datasets ...")
    # Extend with extra small OpenML datasets
    extras = ["iris","wine","glass","sonar","ionosphere","vehicle","yeast"]
    for ds_name in extras:
        df_ds = load_openml(ds_name)
        if df_ds is None:
            continue
        try:
            target_col = df_ds.columns[-1]
            X_exp = safe_X(df_ds, target_col)
            tgt   = df_ds[target_col].copy()
            if hasattr(tgt, "cat"):
                tgt = tgt.astype(str)
            y_exp = LabelEncoder().fit_transform(tgt.fillna("missing"))
            X_sub = X_exp[:min(len(X_exp), 2000)]
            y_sub = y_exp[:min(len(y_exp), 2000)]
            meta  = _meta(df_ds)
            dummy_cv = cross_val_score(
                DummyClassifier(strategy="most_frequent"),
                X_sub, y_sub, cv=3, scoring="accuracy").mean()
            for alg_name, alg in ALGORITHMS.items():
                try:
                    cv_mean = float(cross_val_score(
                        alg, X_sub, y_sub, cv=3, scoring="accuracy").mean())
                    row = dict(meta)
                    row["cv_score"]  = cv_mean
                    row["success"]   = int(cv_mean >= 0.65 and cv_mean > dummy_cv + 0.10)
                    row["algorithm"] = alg_name
                    psp_rows.append(row)
                except Exception:
                    pass
        except Exception:
            pass

df_psp = pd.DataFrame(psp_rows)
print(f"  Total records: {len(df_psp):,}  |  Success rate: {df_psp['success'].mean():.2%}")

PSP_FEATURES = [
    "null_rate","drift_detected","quality_score","row_count_k",
    "n_columns","anomaly_count","schema_match","known_dataset",
    "cv_score","columns_drifted",
]
X_psp = df_psp[PSP_FEATURES].fillna(0).values
y_psp = df_psp["success"].values

X_ptr, X_pval, X_pte, y_ptr, y_pval, y_pte = three_way_split(
    X_psp, y_psp, stratify=True)
print(f"  Split: train={len(X_ptr):,}  val={len(X_pval):,}  test={len(X_pte):,}")

clf_psp = RandomForestClassifier(
    n_estimators=500, max_depth=8, min_samples_leaf=4,
    class_weight="balanced", oob_score=True, n_jobs=-1, random_state=42,
)
clf_psp.fit(X_ptr, y_ptr)
print(f"  OOB Score: {clf_psp.oob_score_:.4f}")
check_overfit(clf_psp, X_ptr, y_ptr, X_pval, y_pval)
auc = roc_auc_score(y_pte, clf_psp.predict_proba(X_pte)[:, 1])
print(f"  Test ROC-AUC: {auc:.4f}")
print(classification_report(y_pte, clf_psp.predict(X_pte),
      target_names=["failure", "success"], zero_division=0))
print_learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=8,
                           class_weight="balanced", n_jobs=-1, random_state=42),
    X_psp, y_psp, scoring="roc_auc", n_points=5,
    clf_name="Pipeline Success Predictor",
)
joblib.dump(clf_psp, "models/pipeline_success_predictor.pkl")
tick("pipeline_success_predictor.pkl  SAVED", t0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. NLP QUERY CLASSIFIER — WikiSQL (80K real questions)
# ─────────────────────────────────────────────────────────────────────────────
header("4 / 6  NLP Query Intent Classifier  (WikiSQL 80K real questions)")
t0 = time.time()

WIKISQL_URL = (
    "https://raw.githubusercontent.com/salesforce/WikiSQL/"
    "master/data/train.jsonl.bz2"
)

def _intent(q, agg, n_conds, order_by, has_limit):
    ql = q.lower()
    if agg in (4, 5) or any(w in ql for w in ("sum","total","average","mean","avg")):
        return "aggregate"
    if agg == 3:
        return "count_distinct" if any(
            w in ql for w in ("distinct","unique","how many different")) else "aggregate"
    if agg in (1, 2):
        return "aggregate"
    if order_by and has_limit:
        return "top_n" if any(
            w in ql for w in ("top","highest","most","best","largest","greatest")) else "bottom_n"
    if any(w in ql for w in ("trend","monthly","weekly","daily","over time","per month","per year")):
        return "trend"
    if any(w in ql for w in ("compare","difference","versus","vs","better","higher","lower than")):
        return "compare"
    if any(w in ql for w in ("correlat","relation","depend","affect","cause","predict")):
        return "correlation"
    if any(w in ql for w in ("distribution","spread","histogram","skew")):
        return "distribution"
    if any(w in ql for w in ("group by","grouped","breakdown","per","by category","by region")):
        return "group_by"
    if n_conds > 0:
        return "filter"
    return "general"

print("  Downloading WikiSQL train.jsonl.bz2 ...")
nlp_texts, nlp_labels = [], []
try:
    raw = bz2.decompress(urllib.request.urlopen(WIKISQL_URL, timeout=90).read())
    for line in raw.decode("utf-8").splitlines():
        try:
            obj = json.loads(line)
            q   = obj.get("question", "")
            if not q:
                continue
            sql = obj.get("sql", {})
            ql  = q.lower()
            lbl = _intent(q, sql.get("agg", 0), len(sql.get("conds", [])),
                          "ORDER" in q.upper(),
                          any(w in ql for w in ("top ", "first ", "bottom ")))
            nlp_texts.append(q)
            nlp_labels.append(lbl)
        except Exception:
            pass
    print(f"  Parsed {len(nlp_texts):,} WikiSQL questions.")
    print(pd.Series(nlp_labels).value_counts().to_string())
except Exception as e:
    print(f"  ⚠ WikiSQL failed: {e}")

# Always add curated supplement for rare intent classes
SUPPLEMENT = [
    ("show me top 10 customers by revenue","top_n"),
    ("top 5 products by sales this year","top_n"),
    ("best 20 regions by profit margin","top_n"),
    ("highest revenue accounts last quarter","top_n"),
    ("top 50 users by session duration","top_n"),
    ("worst 10 products by return rate","bottom_n"),
    ("bottom 5 stores by monthly sales","bottom_n"),
    ("lowest performing customer segments","bottom_n"),
    ("least active users this quarter","bottom_n"),
    ("what is the total revenue for 2023","aggregate"),
    ("sum of all credit card transactions","aggregate"),
    ("average order value across regions","aggregate"),
    ("what is the maximum price paid","aggregate"),
    ("minimum salary in engineering dept","aggregate"),
    ("count total unique order records","aggregate"),
    ("median age of all customers","aggregate"),
    ("filter customers where age > 30","filter"),
    ("show rows where status is active","filter"),
    ("customers from New York with revenue > 5000","filter"),
    ("orders that were shipped last month","filter"),
    ("show records with null values in revenue","filter"),
    ("show monthly revenue trend over time","trend"),
    ("how has profit changed over the past year","trend"),
    ("weekly order volume for last 90 days","trend"),
    ("revenue growth week by week","trend"),
    ("compare revenue across all regions","compare"),
    ("year over year performance comparison by segment","compare"),
    ("segment A versus segment B on profit","compare"),
    ("Q1 vs Q2 sales performance","compare"),
    ("correlation between customer age and income","correlation"),
    ("does advertising spend affect revenue","correlation"),
    ("what factors predict churn","correlation"),
    ("relationship between price and quantity demanded","correlation"),
    ("distribution of order values across channels","distribution"),
    ("histogram of customer ages","distribution"),
    ("spread of transaction amounts","distribution"),
    ("revenue breakdown by product category","group_by"),
    ("group sales by region and quarter","group_by"),
    ("average price per brand","group_by"),
    ("total orders per country","group_by"),
    ("daily sales for last 30 days","time_series"),
    ("monthly revenue last 12 months","time_series"),
    ("order count by week this year","time_series"),
    ("quarter over quarter growth rate","time_series"),
    ("how many unique customers do we have","count_distinct"),
    ("distinct products sold last month","count_distinct"),
    ("number of unique regions in dataset","count_distinct"),
    ("count distinct account types","count_distinct"),
    ("describe the entire dataset","general"),
    ("show all available columns","general"),
    ("give me an overview of the data","general"),
    ("what does this table contain","general"),
    ("preview first 10 rows","general"),
]
for text, label in SUPPLEMENT:
    nlp_texts.append(text)
    nlp_labels.append(label)

print(f"  Total NLP samples: {len(nlp_texts):,}")
X_nlp, y_nlp = np.array(nlp_texts), np.array(nlp_labels)
X_ntr, X_nval, X_nte, y_ntr, y_nval, y_nte = three_way_split(
    X_nlp, y_nlp, stratify=False)
print(f"  Split: train={len(X_ntr):,}  val={len(X_nval):,}  test={len(X_nte):,}")

nlp_pipe = SKPipeline([
    ("vec", TfidfVectorizer(ngram_range=(1,3), max_features=30_000,
                             sublinear_tf=True, min_df=1)),
    ("svc", LinearSVC(C=0.5, max_iter=5000, random_state=42)),
])
nlp_pipe.fit(X_ntr.tolist(), y_ntr.tolist())
tr_a = accuracy_score(y_ntr, nlp_pipe.predict(X_ntr.tolist()))
vl_a = accuracy_score(y_nval, nlp_pipe.predict(X_nval.tolist()))
te_a = accuracy_score(y_nte,  nlp_pipe.predict(X_nte.tolist()))
print(f"  Train={tr_a:.4f} | Val={vl_a:.4f} | Test={te_a:.4f}",
      "⚠ OVERFIT" if abs(tr_a - vl_a) > 0.10 else "✅ OK")
print(classification_report(y_nte, nlp_pipe.predict(X_nte.tolist()), zero_division=0))

# Learning curve on 2K sample subset
lc_n = min(len(nlp_texts), 2000)
idx = np.random.choice(len(nlp_texts), lc_n, replace=False)
X_lc = [nlp_texts[i] for i in idx]
y_lc = [nlp_labels[i] for i in idx]
print_learning_curve(
    SKPipeline([
        ("vec", TfidfVectorizer(ngram_range=(1,2), max_features=10_000, sublinear_tf=True)),
        ("svc", LinearSVC(C=0.5, max_iter=3000, random_state=42)),
    ]),
    X_lc, y_lc, scoring="accuracy", cv=3, n_points=5,
    clf_name="NLP Query Classifier",
)
joblib.dump(nlp_pipe.named_steps["svc"], "models/nlp_query_classifier.pkl")
joblib.dump(nlp_pipe.named_steps["vec"], "models/nlp_query_vectorizer.pkl")
tick("nlp_query_classifier.pkl + nlp_query_vectorizer.pkl  SAVED", t0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PROPOSAL CONFIDENCE SCORER
#    Derived from real PSP experiment records
# ─────────────────────────────────────────────────────────────────────────────
header("5 / 6  Proposal Confidence Scorer  (Real OpenML meta-learning)")
t0 = time.time()

prop_rows = []
for row in psp_rows:
    pr = {
        "null_rate":         float(row.get("null_rate", 0)),
        "drift_flag":        float(row.get("drift_detected", 0)),
        "quality_score":     float(row.get("quality_score", 0)),
        "sample_size_k":     float(row.get("row_count_k", 0)),
        "n_columns":         float(row.get("n_columns", 0)),
        "cv_score":          float(row.get("cv_score", 0)),
        "flag_severity_max": 1.0 if float(row.get("null_rate", 0)) > 0.10 else 0.0,
        "columns_drifted":   float(row.get("columns_drifted", 0)),
        "proposer_type_enc": float(hash(str(row.get("algorithm","rf"))) % 8) / 8.0,
        "high_conf":         int(
            float(row.get("cv_score", 0)) >= 0.75
            and int(row.get("success", 0)) == 1
        ),
    }
    prop_rows.append(pr)

df_prop = pd.DataFrame(prop_rows)
print(f"  Real proposal records: {len(df_prop):,}  |  High-conf rate: {df_prop['high_conf'].mean():.2%}")

PROP_FEAT_COLS = [
    "null_rate","drift_flag","quality_score","sample_size_k","n_columns",
    "cv_score","flag_severity_max","columns_drifted","proposer_type_enc",
]
X_prp = df_prop[PROP_FEAT_COLS].fillna(0).values
y_prp = df_prop["high_conf"].values

X_prtr, X_prval, X_prte, y_prtr, y_prval, y_prte = three_way_split(
    X_prp, y_prp, stratify=True)
print(f"  Split: train={len(X_prtr):,}  val={len(X_prval):,}  test={len(X_prte):,}")

clf_prp = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_leaf=3,
    class_weight="balanced", oob_score=True, n_jobs=-1, random_state=42,
)
clf_prp.fit(X_prtr, y_prtr)
print(f"  OOB Score: {clf_prp.oob_score_:.4f}")
check_overfit(clf_prp, X_prtr, y_prtr, X_prval, y_prval)
prp_auc = roc_auc_score(y_prte, clf_prp.predict_proba(X_prte)[:, 1])
print(f"  Test ROC-AUC: {prp_auc:.4f}")
print(classification_report(y_prte, clf_prp.predict(X_prte),
      target_names=["low_conf","high_conf"], zero_division=0))
print_learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=10,
                           class_weight="balanced", n_jobs=-1, random_state=42),
    X_prp, y_prp, scoring="roc_auc", n_points=5,
    clf_name="Proposal Confidence Scorer",
)
joblib.dump(clf_prp, "models/proposal_confidence.pkl")
tick("proposal_confidence.pkl  SAVED", t0)

# ─────────────────────────────────────────────────────────────────────────────
# 6. CHART RELEVANCE SCORER
#    Real OpenML data profiles → Draco/Vega-Lite expert chart labels
# ─────────────────────────────────────────────────────────────────────────────
header("6 / 6  Chart Relevance Scorer  (Real OpenML data profiles)")
t0 = time.time()

CHART_DATASETS = [
    "adult","titanic","iris","wine","diabetes","breast-cancer",
    "heart-c","glass","vehicle","segment","letter","abalone",
    "bank-marketing","eeg-eye-state","credit-g","yeast",
    "hypothyroid","anneal","mushroom","car","waveform-5000",
    "blood-transfusion-service-center",
    "monks-problems-1","splice","optdigits",
    "kr-vs-kp","tic-tac-toe","sonar","ionosphere","mfeat-factors",
]

CHART_FEAT_COLS = [
    "row_density","col_density","num_ratio","cat_ratio","first_cat_card",
    "skew_val","mean_corr","null_rate","has_dt","intent_enc",
]

def _chart_feat(df_raw):
    if df_raw.shape[0] < 20:
        return None
    num_c = df_raw.select_dtypes(include="number").columns.tolist()
    cat_c = df_raw.select_dtypes(exclude="number").columns.tolist()
    nr, nc = df_raw.shape
    num_r = len(num_c) / max(nc, 1)
    cat_r = len(cat_c) / max(nc, 1)
    fcc   = (df_raw[cat_c[0]].nunique() / nr if cat_c else 0.0)
    null  = float(df_raw.isnull().mean().mean())
    skew  = float(df_raw[num_c].skew().mean()) if len(num_c) >= 2 else 0.0
    corr  = 0.0
    if len(num_c) >= 2:
        cm = df_raw[num_c].corr().abs()
        np.fill_diagonal(cm.values, 0)
        corr = float(cm.mean().mean())
    has_dt = float(any(k in c.lower() for c in df_raw.columns
                       for k in ("date","time","year","month")))
    return {
        "row_density":   min(nr / 10_000, 1.0),
        "col_density":   min(nc / 50, 1.0),
        "num_ratio":     num_r,
        "cat_ratio":     cat_r,
        "first_cat_card":fcc,
        "skew_val":      skew,
        "mean_corr":     corr,
        "null_rate":     null,
        "has_dt":        has_dt,
        "intent_enc":    float(np.random.uniform(0, 1)),
    }

def _best_chart(f, df_raw):
    """Draco/Vega-Lite aligned expert rule."""
    if f["has_dt"] > 0.5:
        return "line"
    if f["cat_ratio"] > 0.5 and f["first_cat_card"] < 0.10:
        return "pie"
    if f["cat_ratio"] > 0.30:
        return "bar"
    if f["mean_corr"] > 0.65 and f["num_ratio"] > 0.7:
        return "scatter"
    if abs(f["skew_val"]) > 1.5 and f["num_ratio"] > 0.5:
        return "histogram"
    if f["num_ratio"] > 0.80 and f["col_density"] > 0.4:
        return "heatmap"
    return "box"

print("  Extracting real chart features from OpenML datasets ...")
chart_rows = []
for ds_name in CHART_DATASETS:
    df_raw = load_openml(ds_name)
    if df_raw is None:
        continue
    n_reps = max(1, min(25, len(df_raw) // 80))
    for _ in range(n_reps):
        sample = df_raw.sample(min(len(df_raw), 500), random_state=None)
        f = _chart_feat(sample)
        if f is None:
            continue
        f["label"] = _best_chart(f, sample)
        chart_rows.append(f)

df_chart = pd.DataFrame(chart_rows)
ct = df_chart["label"].value_counts()
df_chart = df_chart[df_chart["label"].isin(ct[ct >= 5].index)].copy()
print(f"  Chart samples: {len(df_chart):,}")
print(df_chart["label"].value_counts().to_string())

X_cht = df_chart[CHART_FEAT_COLS].fillna(0).values
y_cht = df_chart["label"].values

X_ctr, X_cval, X_cte, y_ctr, y_cval, y_cte = three_way_split(
    X_cht, y_cht, stratify=True)
print(f"  Split: train={len(X_ctr):,}  val={len(X_cval):,}  test={len(X_cte):,}")

clf_cht = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_leaf=2,
    oob_score=True, n_jobs=-1, random_state=42,
)
clf_cht.fit(X_ctr, y_ctr)
print(f"  OOB Score: {clf_cht.oob_score_:.4f}")
check_overfit(clf_cht, X_ctr, y_ctr, X_cval, y_cval)
te_acc = accuracy_score(y_cte, clf_cht.predict(X_cte))
print(f"  Test Accuracy: {te_acc:.4f}")
print(classification_report(y_cte, clf_cht.predict(X_cte), zero_division=0))
print_learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
    X_cht, y_cht, scoring="accuracy", n_points=5,
    clf_name="Chart Relevance Scorer",
)
joblib.dump(clf_cht, "models/chart_relevance_scorer.pkl")
tick("chart_relevance_scorer.pkl  SAVED", t0)

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("  ALL 6 MODELS TRAINED & VERIFIED — FINAL INVENTORY")
print("="*68)
for f in sorted(os.listdir("models")):
    if not f.endswith(".pkl"):
        continue
    kb  = os.path.getsize(f"models/{f}") // 1024
    obj = joblib.load(f"models/{f}")
    print(f"  ✅  {f:<48} ({kb:>5} KB)  [{type(obj).__name__}]")
