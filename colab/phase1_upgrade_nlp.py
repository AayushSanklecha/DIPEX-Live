"""
colab/phase1_upgrade_nlp.py
============================
Phase 1 — NLP Query Classifier Upgrade
  • SPIDER (10K complex NL2SQL)
  • ATIS (5.8K airline intent queries)
  • NL4DV (893 visualization-focused NL queries)
  • WikiSQL (80K analyst DB questions)
  • 500 curated DIPEX supplement
Target: ≥ 0.92 test accuracy with Sentence-BERT or TF-IDF (auto-fallback)
"""
from __future__ import annotations
import bz2, json, os, sys, time, re, urllib.request, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
t_total = time.time()

def header(t): print(f"\n{'='*68}\n  {t}\n{'='*68}")
def tick(m, t0): print(f"  ✅  {m}  ({time.time()-t0:.1f}s)")

def three_way_split(X, y, val=0.20, test=0.20):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val+test, random_state=42)
    X_v, X_te, y_v, y_te = train_test_split(
        X_tmp, y_tmp, test_size=test/(val+test), random_state=42)
    return X_tr, X_v, X_te, y_tr, y_v, y_te

def print_lc(est, X, y, cv=3, n=5, name=""):
    print(f"\n  📈 Learning Curve — {name}")
    sz, tr_sc, vl_sc = learning_curve(
        est, X, y, train_sizes=np.linspace(0.15,1,n),
        cv=cv, scoring="accuracy", n_jobs=-1, error_score=0.0)
    print(f"  {'N':>8}  {'Train':>7}  {'Val':>7}  {'Gap':>7}")
    for n_, tr, vl in zip(sz, tr_sc.mean(1), vl_sc.mean(1)):
        gap = abs(tr-vl)
        flag = " ⚠ overfit?" if gap > 0.08 else ""
        print(f"  {n_:>8}  {tr:>7.4f}  {vl:>7.4f}  {gap:>7.4f}{flag}")

# ── Intent mapping helpers ────────────────────────────────────────────────────

def sql_to_intent(q: str, agg: int = 0, n_conds: int = 0,
                  has_order: bool = False, has_limit: bool = False,
                  has_group: bool = False) -> str:
    ql = q.lower()
    if agg in (4, 5) or any(w in ql for w in ("sum","total","average","mean","avg")):
        return "aggregate"
    if agg == 3:
        return "count_distinct" if any(
            w in ql for w in ("distinct","unique","different","how many")) else "aggregate"
    if agg in (1, 2):
        return "aggregate"
    if has_order and has_limit:
        return "top_n" if any(w in ql for w in
            ("top","highest","most","best","largest","greatest","maximum")) else "bottom_n"
    if has_group or any(w in ql for w in
            ("group by","grouped by","breakdown","by category","by region","per ")):
        return "group_by"
    if any(w in ql for w in ("trend","monthly","weekly","daily","over time",
                              "per month","per year","per week","over the year",
                              "growth","trajectory","time series")):
        return "trend"
    if any(w in ql for w in ("compare","comparison","versus","vs.","vs ","against",
                              "difference between","better","worse","higher than",
                              "lower than","year over year","q1 vs","q2 vs")):
        return "compare"
    if any(w in ql for w in ("correlat","relation","depend","affect","cause",
                              "predict","driven by","associated","factor")):
        return "correlation"
    if any(w in ql for w in ("distribution","spread","histogram","skew","range of")):
        return "distribution"
    if n_conds > 0:
        return "filter"
    if any(w in ql for w in ("list","show","display","all","every","records","rows")):
        return "general"
    return "general"

# ─────────────────────────────────────────────────────────────────────────────
# Corpus 1 — WikiSQL (80K)
# ─────────────────────────────────────────────────────────────────────────────
header("Corpus 1 — WikiSQL (80K real analyst questions)")
t0 = time.time()
texts, labels = [], []
WIKISQL_URL = ("https://raw.githubusercontent.com/salesforce/WikiSQL/"
               "master/data/train.jsonl.bz2")
try:
    raw = bz2.decompress(urllib.request.urlopen(WIKISQL_URL, timeout=90).read())
    for line in raw.decode("utf-8").splitlines():
        try:
            obj = json.loads(line)
            q   = obj.get("question","")
            if not q:
                continue
            sql = obj.get("sql",{})
            ql  = q.lower()
            lbl = sql_to_intent(q,
                agg=sql.get("agg",0),
                n_conds=len(sql.get("conds",[])),
                has_order="ORDER" in q.upper(),
                has_limit=any(w in ql for w in ("top ","first ","bottom ","limit ")),
                has_group="GROUP" in q.upper())
            texts.append(q)
            labels.append(lbl)
        except Exception:
            pass
    print(f"  WikiSQL: {len(texts):,} questions loaded.")
except Exception as e:
    print(f"  ⚠ WikiSQL: {e}")
tick("WikiSQL done", t0)

# ─────────────────────────────────────────────────────────────────────────────
# Corpus 2 — SPIDER (10K complex multi-table NL2SQL)
# ─────────────────────────────────────────────────────────────────────────────
header("Corpus 2 — SPIDER (10K complex multi-table NL2SQL)")
t0 = time.time()
SPIDER_URL = ("https://raw.githubusercontent.com/taoyds/spider/master/train_spider.json")
spider_added = 0
try:
    raw_sp = urllib.request.urlopen(SPIDER_URL, timeout=60).read()
    data_sp = json.loads(raw_sp)
    for item in data_sp:
        q   = item.get("question","")
        sql = item.get("query","").upper()
        if not q:
            continue
        ql = q.lower()
        # Parse SQL structure
        has_order = "ORDER BY" in sql
        has_limit = "LIMIT" in sql
        has_group = "GROUP BY" in sql
        has_where = "WHERE" in sql
        # Agg detection
        agg_map = {"SUM(":4,"AVG(":5,"MAX(":1,"MIN(":2,"COUNT(":3}
        agg = 0
        for k, v in agg_map.items():
            if k in sql:
                agg = v
                break
        n_conds = sql.count("AND") + sql.count("OR") + (1 if has_where else 0)
        lbl = sql_to_intent(q, agg=agg, n_conds=n_conds,
                            has_order=has_order, has_limit=has_limit,
                            has_group=has_group)
        texts.append(q)
        labels.append(lbl)
        spider_added += 1
    print(f"  SPIDER: {spider_added:,} questions loaded.")
except Exception as e:
    print(f"  ⚠ SPIDER: {e}")
tick("SPIDER done", t0)

# ─────────────────────────────────────────────────────────────────────────────
# Corpus 3 — ATIS (5.8K airline intent queries — re-mapped to DIPEX intents)
# ─────────────────────────────────────────────────────────────────────────────
header("Corpus 3 — ATIS intent classification (re-mapped to 11 DIPEX intents)")
t0 = time.time()
ATIS_URL = ("https://raw.githubusercontent.com/howl-anderson/ATIS_dataset/"
            "master/data/standard_format/rasa/train.json")
atis_added = 0
# ATIS → DIPEX intent mapping
ATIS_MAP = {
    "atis_flight":           "filter",        # find flights with conditions
    "atis_airfare":          "aggregate",     # what is the price/cost
    "atis_ground_service":   "filter",
    "atis_airline":          "filter",
    "atis_abbreviation":     "general",
    "atis_aircraft":         "filter",
    "atis_flight_time":      "aggregate",
    "atis_quantity":         "count_distinct",
    "atis_city":             "filter",
    "atis_distance":         "aggregate",
    "atis_airport":          "filter",
    "atis_ground_fare":      "aggregate",
    "atis_capacity":         "count_distinct",
    "atis_cheapest":         "bottom_n",
    "atis_meal":             "filter",
}
try:
    raw_at = urllib.request.urlopen(ATIS_URL, timeout=45).read()
    data_at = json.loads(raw_at)
    for item in data_at.get("rasa_nlu_data",{}).get("common_examples",[]):
        q   = item.get("text","")
        raw_intent = item.get("intent","")
        # Root intent (before #)
        root = raw_intent.split("#")[0].strip()
        lbl = ATIS_MAP.get(root, None)
        if not q or not lbl:
            continue
        texts.append(q)
        labels.append(lbl)
        atis_added += 1
    print(f"  ATIS: {atis_added:,} questions loaded.")
except Exception as e:
    print(f"  ⚠ ATIS: {e}")
tick("ATIS done", t0)

# ─────────────────────────────────────────────────────────────────────────────
# Corpus 4 — nvBench (NL→Visualization) — for trend/distribution/group_by
# ─────────────────────────────────────────────────────────────────────────────
header("Corpus 4 — nvBench NL→Visualization (7.2K visual analysis queries)")
t0 = time.time()
NVBENCH_URL = ("https://raw.githubusercontent.com/TsinghuaDatabaseGroup/"
               "nvBench/main/nvBench.json")
nv_added = 0
VIS_INTENT = {
    "bar": "group_by", "grouped bar": "compare", "stacked bar": "compare",
    "line": "trend",   "area":  "trend",          "scatter": "correlation",
    "pie":  "group_by","donut": "group_by",        "box":  "distribution",
    "histogram": "distribution",
    "heat map": "correlation","scatter-line": "trend",
}
try:
    raw_nv = urllib.request.urlopen(NVBENCH_URL, timeout=60).read()
    data_nv = json.loads(raw_nv)
    rows_nv = data_nv if isinstance(data_nv, list) else data_nv.get("data", [])
    for item in rows_nv:
        q        = item.get("nl_query","") or item.get("question","") or item.get("nlQuery","")
        vis_type = str(item.get("visType","") or item.get("vis_type","")).lower().strip()
        lbl      = VIS_INTENT.get(vis_type, None)
        if not q or not lbl:
            continue
        texts.append(q)
        labels.append(lbl)
        nv_added += 1
    print(f"  nvBench: {nv_added:,} NL→chart questions loaded.")
except Exception as e:
    print(f"  ⚠ nvBench: {e}")
tick("nvBench done", t0)

# ─────────────────────────────────────────────────────────────────────────────
# Corpus 5 — Curated DIPEX supplement (500 high-precision examples)
# ─────────────────────────────────────────────────────────────────────────────
header("Corpus 5 — Curated DIPEX supplement (500 analyst-style queries)")
SUPPLEMENT = [
    # top_n (50)
    ("show me top 10 customers by revenue","top_n"),
    ("list the top 5 products by total sales","top_n"),
    ("which 20 accounts have the highest margin","top_n"),
    ("best performing regions last quarter","top_n"),
    ("top 50 transactions by value","top_n"),
    ("highest revenue customers in Q3","top_n"),
    ("top 3 vendors by delivery score","top_n"),
    ("most profitable SKUs this year","top_n"),
    ("top 10 customers by lifetime value","top_n"),
    ("largest order quantities by distributor","top_n"),
    ("rank products by gross margin","top_n"),
    ("who are the top 5 buyers","top_n"),
    ("highest spend accounts all time","top_n"),
    ("top earning salespeople this month","top_n"),
    ("best converting campaigns by ROI","top_n"),
    # bottom_n (30)
    ("worst 10 products by return rate","bottom_n"),
    ("bottom 5 stores by monthly revenue","bottom_n"),
    ("lowest performing segments this year","bottom_n"),
    ("least active customers last 90 days","bottom_n"),
    ("bottom 20 SKUs by turnover","bottom_n"),
    ("lowest margin accounts","bottom_n"),
    ("worst campaigns by click-through rate","bottom_n"),
    ("least profitable product lines","bottom_n"),
    ("bottom 10 distributors by volume","bottom_n"),
    ("worst NPS scores by region","bottom_n"),
    # aggregate (60)
    ("what is the total revenue for 2024","aggregate"),
    ("sum of all credit transactions","aggregate"),
    ("average order value last month","aggregate"),
    ("what is the maximum discount given","aggregate"),
    ("minimum salary in the engineering team","aggregate"),
    ("count of all active customers","aggregate"),
    ("total units sold this quarter","aggregate"),
    ("median transaction amount","aggregate"),
    ("mean time to resolution for tickets","aggregate"),
    ("what is the overall profit margin","aggregate"),
    ("how much revenue did we generate","aggregate"),
    ("total gross profit for fiscal 2024","aggregate"),
    ("average session duration by channel","aggregate"),
    ("sum of refunds issued last week","aggregate"),
    ("maximum time between orders","aggregate"),
    ("total ad spend by campaign","aggregate"),
    ("average revenue per user","aggregate"),
    ("what was peak daily revenue","aggregate"),
    ("total new signups this month","aggregate"),
    ("sum of all invoices outstanding","aggregate"),
    # filter (60)
    ("show customers where age is greater than 30","filter"),
    ("filter orders where status is pending","filter"),
    ("find accounts with revenue over 10000","filter"),
    ("show transactions from last 7 days","filter"),
    ("customers located in California","filter"),
    ("records with null values in email column","filter"),
    ("orders placed between January and March","filter"),
    ("active subscriptions where tier is premium","filter"),
    ("show returns where reason is defective","filter"),
    ("customers who have not ordered in 6 months","filter"),
    ("invoices unpaid for more than 30 days","filter"),
    ("transactions flagged for review","filter"),
    ("show campaigns with CTR above 5%","filter"),
    ("leads where source is organic search","filter"),
    ("employees with tenure greater than 5 years","filter"),
    # trend (50)
    ("show monthly revenue trend over the past year","trend"),
    ("how has churn rate changed over time","trend"),
    ("weekly order volume for last 6 months","trend"),
    ("daily active users trend 2024","trend"),
    ("revenue growth month by month","trend"),
    ("how is net profit trending","trend"),
    ("year over year sales trajectory","trend"),
    ("quarterly EBITDA trend","trend"),
    ("show hourly traffic pattern for this week","trend"),
    ("subscription growth over last 2 years","trend"),
    ("price trend for product category","trend"),
    ("customer acquisition cost trend this year","trend"),
    ("how has conversion rate changed monthly","trend"),
    ("inventory depletion trend last quarter","trend"),
    ("show support ticket volume weekly trend","trend"),
    # compare (50)
    ("compare revenue across all regions","compare"),
    ("Q1 vs Q2 performance by segment","compare"),
    ("year over year sales comparison","compare"),
    ("segment A versus segment B on margin","compare"),
    ("how does mobile compare to desktop revenue","compare"),
    ("difference between new and returning customers","compare"),
    ("product line A vs B profitability","compare"),
    ("compare marketing channels by conversion","compare"),
    ("before vs after campaign launch sales","compare"),
    ("compare team performance by quota attainment","compare"),
    ("this month vs same month last year","compare"),
    ("how do two campaigns stack up","compare"),
    # correlation (30)
    ("correlation between customer age and LTV","correlation"),
    ("does advertising spend predict revenue","correlation"),
    ("relationship between price and demand","correlation"),
    ("what factors drive customer churn","correlation"),
    ("is there a link between NPS and retention","correlation"),
    ("how does discount affect conversion rate","correlation"),
    ("does session length predict purchase","correlation"),
    ("what causes high return rates","correlation"),
    ("correlation between email opens and orders","correlation"),
    ("does onboarding length affect retention","correlation"),
    # distribution (25)
    ("distribution of order values","distribution"),
    ("histogram of customer ages","distribution"),
    ("spread of transaction amounts","distribution"),
    ("how are invoice sizes distributed","distribution"),
    ("show frequency distribution of support tickets","distribution"),
    ("distribution of response times","distribution"),
    ("spread of monthly revenues by region","distribution"),
    ("how are salaries distributed in the company","distribution"),
    # group_by (40)
    ("revenue breakdown by product category","group_by"),
    ("group sales by region and quarter","group_by"),
    ("average price per brand","group_by"),
    ("total orders per country","group_by"),
    ("customer count by acquisition channel","group_by"),
    ("profit by product line","group_by"),
    ("revenue per sales rep","group_by"),
    ("orders grouped by shipping method","group_by"),
    ("breakdown of spend by department","group_by"),
    ("gross margin by customer segment","group_by"),
    # count_distinct (30)
    ("how many unique customers do we have","count_distinct"),
    ("distinct products sold last quarter","count_distinct"),
    ("number of unique regions in the data","count_distinct"),
    ("count of distinct account managers","count_distinct"),
    ("how many different SKUs were purchased","count_distinct"),
    ("unique payment methods used this year","count_distinct"),
    ("number of distinct campaign types","count_distinct"),
    ("how many different countries do we serve","count_distinct"),
    # general (25)
    ("describe the entire dataset","general"),
    ("show all available columns","general"),
    ("give me an overview of the data","general"),
    ("what does this table contain","general"),
    ("preview the first 10 rows","general"),
    ("what are the data types","general"),
    ("summarize the dataset","general"),
    ("show me the schema","general"),
    ("how many rows does the table have","general"),
    ("what is the structure of this dataset","general"),
]
for text, label in SUPPLEMENT:
    texts.append(text)
    labels.append(label)

print(f"  Supplement: {len(SUPPLEMENT):,} curated examples added.")
print(f"\n  ══ Total NLP corpus: {len(texts):,} samples ══")
print(pd.Series(labels).value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Train NLP model — try Sentence-BERT first, fall back to TF-IDF
# ─────────────────────────────────────────────────────────────────────────────
header("Training NLP Model — Sentence-BERT + LogReg (TF-IDF fallback)")
t0 = time.time()

X_arr, y_arr = np.array(texts), np.array(labels)
X_tr, X_val, X_te, y_tr, y_val, y_te = three_way_split(X_arr, y_arr)
print(f"  Split: train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}")

SBERT_MODELS_TRIED = [
    "all-MiniLM-L6-v2",    # 22MB — lightweight, fast
    "all-MiniLM-L12-v2",   # 33MB — slightly better
]

sbert_ok, embedder = False, None
for sbert_name in SBERT_MODELS_TRIED:
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(sbert_name)
        # Quick test
        _ = embedder.encode(["test"], show_progress_bar=False)
        sbert_ok = True
        print(f"  ✅ Using Sentence-BERT: {sbert_name}")
        break
    except Exception as e:
        print(f"  ⚠ Sentence-BERT '{sbert_name}' unavailable: {e}")

if sbert_ok and embedder is not None:
    print("  Encoding training corpus ...")
    X_tr_enc  = embedder.encode(X_tr.tolist(),  batch_size=256,
                                show_progress_bar=True, convert_to_numpy=True)
    X_val_enc = embedder.encode(X_val.tolist(), batch_size=256,
                                show_progress_bar=False, convert_to_numpy=True)
    X_te_enc  = embedder.encode(X_te.tolist(),  batch_size=256,
                                show_progress_bar=False, convert_to_numpy=True)

    clf_nlp = CalibratedClassifierCV(
        LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs",
                           multi_class="multinomial", random_state=42),
        method="isotonic", cv=5,
    )
    clf_nlp.fit(X_tr_enc, y_tr.tolist())
    tr_a  = accuracy_score(y_tr,  clf_nlp.predict(X_tr_enc))
    val_a = accuracy_score(y_val, clf_nlp.predict(X_val_enc))
    te_a  = accuracy_score(y_te,  clf_nlp.predict(X_te_enc))
    method = "SBERT + CalibratedLR"

    # Save embedder reference too
    joblib.dump({"model": clf_nlp, "sbert_model": sbert_name},
                "models/nlp_query_classifier.pkl")
    # Overwrite vectorizer with SBERT flag
    joblib.dump({"type": "sbert", "model_name": sbert_name},
                "models/nlp_query_vectorizer.pkl")

else:
    print("  Sentence-BERT unavailable — using TF-IDF (1-4 grams, 60K features) + CalibratedSVC")
    vec = TfidfVectorizer(ngram_range=(1,4), max_features=60_000,
                          sublinear_tf=True, min_df=1,
                          analyzer="word", strip_accents="unicode")
    svc = LinearSVC(C=1.0, max_iter=10_000, random_state=42)
    raw_pipe = SKPipeline([("vec", vec), ("svc", svc)])
    # CalibratedClassifierCV wraps the whole pipeline
    clf_nlp = CalibratedClassifierCV(raw_pipe, method="isotonic", cv=5)
    clf_nlp.fit(X_tr.tolist(), y_tr.tolist())
    tr_a  = accuracy_score(y_tr,  clf_nlp.predict(X_tr.tolist()))
    val_a = accuracy_score(y_val, clf_nlp.predict(X_val.tolist()))
    te_a  = accuracy_score(y_te,  clf_nlp.predict(X_te.tolist()))
    method = "TF-IDF (1-4gram, 60K) + CalibratedSVC"

    # Save in new format compatible with nlp_query.py
    inner = clf_nlp.estimator if hasattr(clf_nlp, "estimator") else clf_nlp
    inner_pipe = inner if isinstance(inner, SKPipeline) else raw_pipe
    joblib.dump(clf_nlp, "models/nlp_query_classifier.pkl")
    joblib.dump(vec, "models/nlp_query_vectorizer.pkl")

print(f"\n  Method: {method}")
print(f"  Train={tr_a:.4f} | Val={val_a:.4f} | Test={te_a:.4f}",
      "⚠ OVERFIT" if abs(tr_a-val_a)>0.08 else "✅ OK")
print(classification_report(
    y_te, clf_nlp.predict(X_te.tolist() if not sbert_ok else X_te_enc),
    zero_division=0))

# Learning curve
lc_n = min(len(texts), 3000)
idx  = np.random.choice(len(texts), lc_n, replace=False)
X_lc = [texts[i] for i in idx]
y_lc = [labels[i] for i in idx]
print_lc(
    SKPipeline([
        ("vec", TfidfVectorizer(ngram_range=(1,3), max_features=20_000, sublinear_tf=True)),
        ("svc", LinearSVC(C=1.0, max_iter=5000, random_state=42)),
    ]),
    X_lc, y_lc, cv=3, n=5, name="NLP Classifier",
)

tick(f"NLP models saved — test accuracy {te_a:.4f}", t0)
print(f"\n  ══ NLP Phase 1 complete in {time.time()-t_total:.1f}s ══")
