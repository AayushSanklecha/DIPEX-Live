# @title DIPEX — Complete Production ML Trainer (v3 Final + Exotic + NLP)
# =========================================================================
# ONE FILE — PASTE ENTIRELY INTO A SINGLE COLAB CELL (or run as script).
#
# Trains ALL 6 production artifacts in one shot:
#   1. drift_autoencoder.pkl
#   2. drift_scaler.pkl
#   3. drift_pca.pkl
#   4. schema_classifier.pkl  ← 21 base + 10 exotic types + NLP column-name intelligence
#   5. schema_label_encoder.pkl
#   6. chart_relevance_scorer.pkl
#
# SCHEMA CLASSIFIER is the most advanced:
#   Stage 1 — Keyword override on column NAME  (deterministic, 0ms)
#   Stage 2 — TF-IDF char n-gram + LogReg       (NLP on column name)
#   Stage 3 — RandomForest on 25 data features  (statistical fallback)
#
# Expected runtime on Colab CPU: ~4–6 minutes
# =========================================================================

# ── Step 0: Auto-install ──────────────────────────────────────────────────
import subprocess, sys
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
_pip("openml", "scikit-learn", "pandas", "numpy", "joblib")

# ── Step 1: Imports ───────────────────────────────────────────────────────
import os, re, time, logging, warnings, hashlib
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_breast_cancer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dipex_v3_final")

MODELS_DIR = "/content/dipex_models"
os.makedirs(MODELS_DIR, exist_ok=True)
RNG = np.random.default_rng(2025)

def _hdr(t):
    log.info("\n" + "="*70)
    log.info(f"  {t}")
    log.info("="*70)

log.info(f"Setup complete. Output: {MODELS_DIR}")


# =========================================================================
# TOP-LEVEL CLASSES (must be at module level so joblib can pickle them)
# =========================================================================

class CombinedSchemaClassifier:
    """
    Two-stage schema classifier:
      Stage A: Original base RF (21 types) — used when confidence >= threshold
      Stage B: Exotic patch RF (10 exotic types) — used when base is uncertain
    Picklable because it is defined at module top level.
    """
    def __init__(self, clf_base, clf_patch, le_base, le_extended, threshold=0.55):
        self.clf_base    = clf_base
        self.clf_patch   = clf_patch
        self.le_base     = le_base
        self.le_extended = le_extended
        self.threshold   = threshold

    def predict(self, X):
        X = np.nan_to_num(np.array(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            proba   = self.clf_base.predict_proba(X)
            conf    = proba.max(axis=1)
            base_p  = self.le_base.inverse_transform(self.clf_base.predict(X))
        except Exception:
            base_p = np.array(["unknown"] * len(X))
            conf   = np.zeros(len(X))

        results = base_p.astype(object).copy()
        mask    = conf < self.threshold
        if mask.any():
            try:
                enc   = self.clf_patch.predict(X[mask])
                preds = self.le_extended.inverse_transform(enc)
                results[mask] = preds
            except Exception:
                pass
        return results

    def predict_proba_max(self, X):
        X = np.nan_to_num(np.array(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            return self.clf_base.predict_proba(X).max(axis=1)
        except Exception:
            return np.zeros(len(X))


class NLPAugmentedSchemaClassifier:
    """
    Three-stage schema classifier:
      Stage 1 — Keyword override on column name (deterministic)
      Stage 2 — TF-IDF char n-gram + Logistic Regression on column name
      Stage 3 — Statistical model (CombinedSchemaClassifier or base RF)
    Picklable because it is defined at module top level.
    """
    def __init__(self, clf_stat, clf_nlp, le_nlp, lexicon, threshold=0.65):
        self.clf_stat  = clf_stat   # CombinedSchemaClassifier or base RF + LE
        self.clf_nlp   = clf_nlp    # sklearn Pipeline: TF-IDF + LR
        self.le_nlp    = le_nlp     # LabelEncoder for NLP clf
        self.lexicon   = lexicon    # dict: label -> [keywords]
        self.threshold = threshold

    def _keyword_score(self, col_name: str) -> str | None:
        name   = col_name.lower()
        name   = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        name   = re.sub(r'[_\-\.\s]+', ' ', name).strip()
        tokens = set(name.split())
        scores = {}
        for lbl, kws in self.lexicon.items():
            kw_tokens = set(' '.join(kws).split())
            direct    = len(tokens & kw_tokens)
            substr    = sum(1 for kw in kws if kw in name or name in kw)
            scores[lbl] = direct + substr
        best = max(scores, key=scores.get)
        return best if scores[best] >= 2 else None

    def predict_single(self, stat_features, col_name: str = "") -> str:
        sf = np.nan_to_num(np.array(stat_features, dtype=np.float32).reshape(1, -1),
                           nan=0.0, posinf=0.0, neginf=0.0)

        # Stage 1 — keyword
        if col_name:
            kw = self._keyword_score(col_name)
            if kw:
                return kw

        # Stage 2 — NLP
        if col_name:
            try:
                proba = self.clf_nlp.predict_proba([col_name])[0]
                if proba.max() >= self.threshold:
                    return self.le_nlp.inverse_transform([proba.argmax()])[0]
            except Exception:
                pass

        # Stage 3 — statistical
        try:
            preds = self.clf_stat.predict(sf)
            return str(preds[0])
        except Exception:
            return "unknown"

    def predict_batch(self, rows) -> list:
        return [self.predict_single(f, n) for f, n in rows]

    def explain(self, stat_features, col_name: str = "") -> dict:
        sf = np.nan_to_num(np.array(stat_features, dtype=np.float32).reshape(1, -1),
                           nan=0.0, posinf=0.0, neginf=0.0)
        if col_name:
            kw = self._keyword_score(col_name)
            if kw:
                return {"stage": 1, "label": kw, "reason": "keyword_override"}
            try:
                proba = self.clf_nlp.predict_proba([col_name])[0]
                conf  = float(proba.max())
                pred  = self.le_nlp.inverse_transform([proba.argmax()])[0]
                if conf >= self.threshold:
                    return {"stage": 2, "label": pred, "reason": "nlp_name", "conf": round(conf, 4)}
                return {"stage": 3, "label": self.predict_single(sf, ""),
                        "reason": f"nlp_low_conf({conf:.3f})"}
            except Exception:
                pass
        return {"stage": 3, "label": self.predict_single(sf, ""),
                "reason": "stats_only"}


# =========================================================================
# SEMANTIC LEXICON (for NLP stage)
# =========================================================================
SEMANTIC_LEXICON = {
    "id":            ["id","identifier","uuid","guid","record_id","row_id","uid","customer_id","order_id","txn_id","ref_no","serial_no"],
    "age":           ["age","years","yrs","age_years","patient_age","year_of_birth","yob","tenure_years"],
    "amount":        ["amount","price","cost","revenue","income","salary","fee","charge","payment","balance","total","tax","profit","loss","budget","spend","fare","rent","invoice_value"],
    "date":          ["date","datetime","timestamp","created_at","updated_at","dob","joining_date","expiry_date","due_date","start_date","end_date","transaction_date","posting_date"],
    "category":      ["category","type","class","group","status","state","segment","tier","level","tag","kind","mode","channel","bucket","cohort","cluster","department","region","zone"],
    "text":          ["text","description","comment","note","remarks","narrative","summary","detail","message","feedback","review","reason","explanation","title","subject"],
    "phone":         ["phone","mobile","cell","contact","telephone","tel","phone_number","mobile_no","contact_no","phn","mob"],
    "email":         ["email","mail","e_mail","email_address","email_id","emailid","contact_email","user_email"],
    "boolean":       ["is_","has_","flag","active","enabled","disabled","deleted","verified","confirmed","approved","boolean","bool","indicator","eligible"],
    "zipcode":       ["zip","zipcode","postal","postal_code","postcode","pin","pin_code","pincode","area_code","pcode"],
    "percentage":    ["rate","ratio","percent","pct","percentage","proportion","share","margin","yield","roi","cagr","coverage","growth_rate","churn_rate","interest_rate"],
    "score":         ["score","rating","rank","grade","gpa","marks","points","fico","credit_score","risk_score","nps","satisfaction","priority","severity","confidence"],
    "count":         ["count","num","number","qty","quantity","volume","frequency","views","clicks","impressions","sessions","visits","events","cnt","n_","num_of_"],
    "name":          ["name","full_name","first_name","last_name","surname","customer_name","user_name","username","display_name","employee_name","vendor_name","company_name"],
    "url":           ["url","link","uri","href","endpoint","website","domain","web_address","page_url","profile_url","image_url","redirect_url","callback_url"],
    "ip_address":    ["ip","ip_address","ipv4","ipv6","client_ip","server_ip","source_ip","remote_ip","inet","mac_id","network_addr"],
    "coordinates":   ["lat","latitude","lon","longitude","lng","coord","coordinates","geo_lat","geo_lon","gps_lat","gps_long","location_lat","location_lon","x_coord","y_coord"],
    "duration":      ["duration","elapsed","time_taken","response_time","latency","tenure_days","session_duration","hold_time","wait_time","seconds","minutes","hours","interval","ttl"],
    "address":       ["address","addr","street","lane","road","avenue","location","residence","billing_address","shipping_address","house_no","building","locality","address_line"],
    "currency_code": ["currency","currency_code","ccy","ccy_code","iso_currency","transaction_currency","base_currency","invoice_currency"],
    "swift_code":    ["swift","swift_code","bic","bic_code","swift_bic","bank_code","routing_bic"],
    "iban":          ["iban","international_bank_account","bank_account_iban","iban_number"],
    "ssn":           ["ssn","social_security","social_security_number","sin","national_insurance","nin","tax_id_us"],
    "pan_number":    ["pan","pan_no","pan_number","pan_card","panno","cust_pan","income_tax_pan","it_pan"],
    "passport":      ["passport","passport_no","passport_number","travel_doc","passport_id"],
    "vin":           ["vin","vehicle_identification","chassis_no","chassis_number","vin_number"],
    "mac_address":   ["mac","mac_address","hardware_address","physical_address","device_mac","mac_id","nic_address"],
    "credit_card":   ["credit_card","card_number","card_no","cc_number","ccnum","debit_card","masked_card","card_last4"],
    "ticker_symbol": ["ticker","symbol","stock_symbol","stock_ticker","scrip","equity_symbol","ticker_symbol","bse_code","nse_code","isin"],
    "hash_value":    ["hash","checksum","md5","sha256","sha1","digest","fingerprint","token_hash","hex_digest"],
    "unknown":       ["misc","other","extra","temp","col","field","raw","custom","var","unknown"],
}

ALL_LABELS = list(SEMANTIC_LEXICON.keys())  # 31 total labels


# =========================================================================
# FEATURE EXTRACTION (shared by schema + exotic patch)
# =========================================================================
def extract_stat_features(s: pd.Series) -> list:
    """25-feature statistical vector from a pandas Series."""
    n_total = max(len(s), 1)
    s_clean = s.dropna()
    is_num  = pd.api.types.is_numeric_dtype(s)
    is_str  = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

    num_vals = pd.to_numeric(s_clean, errors='coerce').dropna() if not is_num else s_clean.copy()
    str_vals = s_clean.astype(str) if is_str else pd.Series([], dtype=str)
    num_vals = num_vals[~np.isinf(num_vals)]

    try:    all_int = float((num_vals == num_vals.apply(int)).all()) if len(num_vals) > 0 else 0.0
    except: all_int = 0.0

    def sv(v): return float(v) if not (np.isnan(float(v)) or np.isinf(float(v))) else 0.0

    return [
        sv(s.isnull().mean()),                                           # 0  null_rate
        sv(s_clean.nunique() / n_total),                                 # 1  unique_rate
        float(is_num), float(is_str),                                    # 2,3
        float(pd.api.types.is_datetime64_any_dtype(s)),                  # 4  is_datetime
        sv(num_vals.mean()  if len(num_vals) else 0.0),                  # 5  mean_val
        sv(num_vals.std()   if len(num_vals) else 0.0),                  # 6  std_val
        sv(num_vals.min()   if len(num_vals) else 0.0),                  # 7  min_val
        sv(num_vals.max()   if len(num_vals) else 0.0),                  # 8  max_val
        sv(num_vals.skew()  if len(num_vals) > 3 else 0.0),              # 9  skew_val
        all_int,                                                         # 10 all_integer
        float(bool(len(num_vals) > 0 and float(num_vals.max()) < 200)),  # 11 max_lt_200
        float(bool(len(num_vals) > 0 and float(num_vals.max()) <= 1.0)), # 12 max_lt_1
        float(bool(len(num_vals) > 0 and (num_vals >= 0).all())),        # 13 all_positive
        float(s_clean.nunique()),                                        # 14 n_distinct
        sv(str_vals.str.contains(r"@.*\.", na=False).mean()   if len(str_vals) else 0), # 15 email
        sv(str_vals.str.contains(r"^\+?\d[\d\s\-()]{7,}$", na=False).mean() if len(str_vals) else 0), # 16 phone
        sv(str_vals.str.len().mean() if len(str_vals) else 0),           # 17 str_len
        float(s_clean.nunique() / n_total > 0.9),                       # 18 high_card
        float(s_clean.nunique() / n_total < 0.05),                      # 19 low_card
        sv(str_vals.str.contains(r"https?://|www\.", na=False).mean()   if len(str_vals) else 0), # 20 url
        sv(str_vals.str.match(r"^(\d{1,3}\.){3}\d{1,3}$", na=False).mean() if len(str_vals) else 0), # 21 ip
        float(bool(len(num_vals) > 0 and ((num_vals >= -180) & (num_vals <= 180)).all())), # 22 coord_range
        float(bool(len(num_vals) > 0 and (num_vals % 1 != 0).mean() > 0.8)), # 23 coord_prec
        sv(str_vals.str.match(r"^[A-Z]{3}$", na=False).mean() if len(str_vals) else 0),   # 24 curr
    ]


# =========================================================================
# SYNTHETIC COLUMN GENERATORS (all 31 types)
# =========================================================================
def _make_series(label: str, n: int) -> pd.Series:
    null_p = RNG.uniform(0.0, 0.35)
    def _null(s):
        s = s.copy()
        if null_p > 0:
            idx = RNG.choice(len(s), max(1, int(len(s)*null_p)), replace=False)
            s.iloc[idx] = np.nan
        return s
    alpha = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    if label == "id":           return _null(pd.Series(np.arange(10000, 10000+n)))
    if label == "age":          return _null(pd.Series(RNG.normal(35, 12, n).clip(0, 110)))
    if label == "amount":       return _null(pd.Series(RNG.normal(RNG.uniform(-1e5,1e5), RNG.uniform(100,1e4), n)))
    if label == "date":         return _null(pd.Series(pd.date_range("2010-01-01", periods=n, freq="D").strftime("%Y-%m-%d")))
    if label == "category":     return _null(pd.Series(RNG.choice(["Cat_A","Cat_B","Cat_C","Cat_D","Cat_E"], n)))
    if label == "text":         return _null(pd.Series([" ".join(RNG.choice(["lorem","ipsum","dolor","sit","amet"], 8).tolist()) for _ in range(n)]))
    if label == "phone":        return _null(pd.Series([f"+1-{RNG.integers(200,999)}-{RNG.integers(100,999)}-{RNG.integers(1000,9999)}" for _ in range(n)]))
    if label == "email":        return _null(pd.Series([f"user{RNG.integers(0,9999)}@gmail.com" for _ in range(n)]))
    if label == "boolean":      return _null(pd.Series(RNG.choice([True, False], n)))
    if label == "zipcode":      return _null(pd.Series(RNG.integers(10000, 99999, n)))
    if label == "percentage":   return _null(pd.Series(RNG.uniform(0, 1, n).astype(float)))
    if label == "score":        return _null(pd.Series(RNG.normal(50, 15, n).clip(0, 100)))
    if label == "count":        return _null(pd.Series(RNG.poisson(RNG.uniform(1, 100), n)))
    if label == "name":         return _null(pd.Series([f"{RNG.choice(['Alice','Bob','Carlos'])} Smith" for _ in range(n)]))
    if label == "url":          return _null(pd.Series([f"https://example.com/page{RNG.integers(1,100)}" for _ in range(n)]))
    if label == "ip_address":   return _null(pd.Series([f"192.168.{RNG.integers(0,255)}.{RNG.integers(1,254)}" for _ in range(n)]))
    if label == "coordinates":  return _null(pd.Series(RNG.uniform(-90, 90, n).round(6)))
    if label == "duration":     return _null(pd.Series(RNG.integers(0, 7200, n).astype(float)))
    if label == "address":      return _null(pd.Series([f"{RNG.integers(1,9999)} Main St, London" for _ in range(n)]))
    if label == "currency_code":return _null(pd.Series(RNG.choice(["USD","EUR","GBP","JPY","INR"], n)))
    if label == "unknown":      return _null(pd.Series(RNG.normal(0, 1e8, n)))
    # ── Exotic types ──────────────────────────────────────────────
    if label == "swift_code":
        banks = ["HDFC","BOFA","CITI","HSBC","DEUT","BNPA","ICIC","AXIS"]
        cnts  = ["IN","US","GB","DE","FR","SG","AE","AU"]
        locs  = ["BB","3N","LN","FF","PP","HH"]
        return _null(pd.Series([f"{RNG.choice(banks)}{RNG.choice(cnts)}{RNG.choice(locs)}" for _ in range(n)]))
    if label == "iban":
        cnts = ["GB","DE","FR","IN","AE","NL","IT"]
        return _null(pd.Series([f"{RNG.choice(cnts)}{RNG.integers(10,99)}" + "".join([str(RNG.integers(0,10)) for _ in range(RNG.integers(10,20))]) for _ in range(n)]))
    if label == "ssn":
        return _null(pd.Series([f"{RNG.integers(100,999)}-{RNG.integers(10,99)}-{RNG.integers(1000,9999)}" for _ in range(n)]))
    if label == "pan_number":
        return _null(pd.Series(["".join(RNG.choice(alpha, 5).tolist()) + "".join([str(RNG.integers(0,10)) for _ in range(4)]) + RNG.choice(alpha) for _ in range(n)]))
    if label == "passport":
        return _null(pd.Series(["".join(RNG.choice(alpha, RNG.integers(1,3)).tolist()) + "".join([str(RNG.integers(0,10)) for _ in range(7)]) for _ in range(n)]))
    if label == "vin":
        vc = list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
        return _null(pd.Series(["".join(RNG.choice(vc, 17).tolist()) for _ in range(n)]))
    if label == "mac_address":
        sep = RNG.choice([":", "-"])
        return _null(pd.Series([sep.join([f"{RNG.integers(0,256):02X}" for _ in range(6)]) for _ in range(n)]))
    if label == "credit_card":
        return _null(pd.Series([f"****-****-****-{RNG.integers(1000,9999)}" for _ in range(n)]))
    if label == "ticker_symbol":
        pool = ["AAPL","GOOG","MSFT","TSLA","NVDA","RELIANCE.NS","INFY.NS","TCS.NS","HDFCBANK.NS"]
        return _null(pd.Series(RNG.choice(pool, n).tolist()))
    if label == "hash_value":
        return _null(pd.Series([hashlib.md5(str(RNG.integers(0,999999)).encode()).hexdigest() for _ in range(n)]))
    # fallback
    return _null(pd.Series(RNG.normal(0, 1, n)))


# =========================================================================
# REAL-WORLD DATA LOADER
# =========================================================================
def load_datasets(max_openml=30):
    _hdr("Step 1/4 — Loading Real-World Datasets")
    dfs = []
    for fn in [fetch_california_housing, load_diabetes, load_wine, load_breast_cancer]:
        try:
            b = fn(); dfs.append(pd.DataFrame(b.data))
            log.info(f"  [+] sklearn: {fn.__name__}")
        except Exception: pass

    try:
        import openml
        ids = [31, 29, 1590, 1461, 37, 40691, 1510, 4534, 180, 40685, 43, 847, 554, 531, 40981, 40984]
        ok = 0
        for did in ids[:max_openml]:
            try:
                ds = openml.datasets.get_dataset(did, download_data=True,
                     download_qualities=False, download_features_meta_data=False)
                X, _, _, _ = ds.get_data(dataset_format="dataframe")
                num = X.select_dtypes(include="number").dropna(axis=1, how='all')
                if num.shape[1] >= 2 and len(num) >= 50:
                    dfs.append(num); ok += 1
                    log.info(f"  [+] OpenML {did}: {num.shape}")
            except Exception: pass
        log.info(f"  ✓ {ok} OpenML datasets loaded.")
    except ImportError:
        log.warning("  [!] openml not installed — using sklearn datasets only.")
    return dfs


# =========================================================================
# DRIFT AUTOENCODER — trains drift_autoencoder.pkl, drift_scaler.pkl, drift_pca.pkl
# =========================================================================
def train_drift(dfs):
    _hdr("Step 2/4 — Drift Autoencoder")
    N = 15; blocks = []

    def _inject(X, null_frac=0.15, out_frac=0.08):
        X = X.astype(float).copy()
        n, m = X.shape
        X[RNG.random((n,m)) < null_frac] = np.nan
        for r in RNG.choice(n, max(1, int(n*out_frac)), replace=False):
            c = int(RNG.integers(0, m))
            if not np.isnan(X[r,c]):
                std = np.nanstd(X[:,c]) + 1e-4
                ch  = RNG.integers(0, 4)
                if   ch == 0: X[r,c] *= 100
                elif ch == 1: X[r,c] *= -1
                elif ch == 2: X[r,c]  = 0.0
                else:         X[r,c] += RNG.choice([-1,1]) * std * 15
        return X

    for df in dfs:
        arr = df.values.astype(float)
        arr = np.pad(arr, ((0,0),(0,max(0,N-arr.shape[1])))) if arr.shape[1] < N else arr[:,:N]
        c = np.nan_to_num(StandardScaler().fit_transform(np.nan_to_num(arr)), 0)
        d = np.nan_to_num(_inject(c, 0.20, 0.15), 0)
        s = c * RNG.uniform(1.2, 1.8) + RNG.normal(0, 1)
        blocks.extend([c, d, s])

    corpus = np.clip(np.vstack(blocks), -10, 10)
    RNG.shuffle(corpus)
    log.info(f"  Corpus: {corpus.shape[0]:,} rows × {corpus.shape[1]} features")

    sc  = StandardScaler()
    csc = sc.fit_transform(corpus)
    pca = PCA(n_components=12, random_state=42)
    cp  = pca.fit_transform(csc)
    log.info(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    ae = MLPRegressor(hidden_layer_sizes=(12,6,12), activation="relu",
                      solver="adam", max_iter=800, learning_rate_init=0.002,
                      early_stopping=True, random_state=42)
    ae.fit(cp, cp)
    mse = float(np.mean(np.square(cp - ae.predict(cp))))
    log.info(f"  Train MSE: {mse:.6f}  iterations: {ae.n_iter_}")

    joblib.dump(ae,  os.path.join(MODELS_DIR, "drift_autoencoder.pkl"))
    joblib.dump(sc,  os.path.join(MODELS_DIR, "drift_scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "drift_pca.pkl"))
    log.info("  ✓ drift_autoencoder.pkl + drift_scaler.pkl + drift_pca.pkl")


# =========================================================================
# SCHEMA CLASSIFIER — trains schema_classifier.pkl, schema_label_encoder.pkl
# Includes: 21 base types + 10 exotic types + NLP column-name intelligence
# =========================================================================
def _nlp_name_samples(n_per=250):
    """Generate (col_name_str, label) pairs for TF-IDF training."""
    X, y = [], []
    for lbl, kws in SEMANTIC_LEXICON.items():
        for _ in range(n_per):
            kw = str(RNG.choice(kws))
            ch = RNG.integers(0, 5)
            if   ch == 0: name = kw
            elif ch == 1: name = f"{RNG.choice(['cust','user','txn','inv','raw'])}_{kw}"
            elif ch == 2: name = f"{kw}_{RNG.choice(['id','no','val','dt','cd','num'])}"
            elif ch == 3: name = kw.replace("_"," ").title().replace(" ","")
            else:
                parts = kw.split("_")
                name  = "".join(p[0] for p in parts if p) + "_" + (parts[-1] if parts else kw)
            X.append(name); y.append(lbl)
    return X, y


def train_schema():
    _hdr("Step 3/4 — Schema Classifier (31 types + NLP)")

    # ── A. Generate synthetic training data ───────────────────────────────
    log.info("  Generating synthetic data for 31 labels (500 samples each)...")
    X_list, y_list = [], []
    for lbl in ALL_LABELS:
        ok = 0
        for _ in range(700):        # 700 attempts → target 500 successes
            try:
                s   = _make_series(lbl, int(RNG.integers(50, 400)))
                fv  = extract_stat_features(s)
                X_list.append(fv); y_list.append(lbl); ok += 1
                if ok >= 500: break
            except Exception: pass
        log.info(f"    {lbl:<20}: {ok} samples")

    X = np.nan_to_num(np.array(X_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_list)

    # ── B. Train base + exotic RF ─────────────────────────────────────────
    le_base = LabelEncoder()
    y_enc   = le_base.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.20, stratify=y_enc, random_state=42)

    clf_base = RandomForestClassifier(n_estimators=400, max_depth=16, min_samples_leaf=4,
                                      n_jobs=-1, random_state=42, class_weight="balanced",
                                      oob_score=True)
    cv = cross_val_score(clf_base, X_tr, y_tr,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="accuracy")
    log.info(f"  5-Fold CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

    clf_base.fit(X_tr, y_tr)
    log.info(f"  OOB accuracy:       {clf_base.oob_score_:.3f}")
    log.info(f"  Test accuracy:      {clf_base.score(X_te, y_te):.3f}  ← real number")

    # Retrain on ALL data for production
    clf_base.fit(X, y_enc)

    # Patch RF for exotic types (will cover the low-confidence cases)
    exotic_labels = [l for l in ALL_LABELS if l not in [
        "id","age","amount","date","category","text","phone","email",
        "boolean","zipcode","percentage","score","count","name","unknown",
        "url","ip_address","coordinates","duration","address","currency_code"
    ]]
    le_ext = LabelEncoder()
    le_ext.fit(ALL_LABELS)

    X_ex, y_ex_str = [], []
    for lbl in exotic_labels:
        for _ in range(400):
            try:
                s  = _make_series(lbl, int(RNG.integers(50, 300)))
                fv = extract_stat_features(s)
                X_ex.append(fv); y_ex_str.append(lbl)
            except Exception: pass

    X_ex   = np.nan_to_num(np.array(X_ex, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_ex   = le_ext.transform(np.array(y_ex_str))
    clf_ex = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=4,
                                    n_jobs=-1, random_state=42, class_weight="balanced")
    clf_ex.fit(X_ex, y_ex)
    log.info(f"  Exotic RF accuracy: {clf_ex.score(X_ex, y_ex):.3f}")

    combined = CombinedSchemaClassifier(clf_base, clf_ex, le_base, le_ext, threshold=0.55)

    # ── C. NLP classifier on column names ─────────────────────────────────
    log.info("\n  Training NLP column-name classifier (TF-IDF char-ngram + LogReg)...")
    X_names, y_nlp_str = _nlp_name_samples(n_per=250)
    le_nlp   = LabelEncoder()
    y_nlp    = le_nlp.fit_transform(y_nlp_str)
    nlp_clf  = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5),
                                  min_df=1, max_features=5000)),
        ("lr",    LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced",
                                     multi_class="multinomial", solver="lbfgs")),
    ])
    nlp_clf.fit(X_names, y_nlp)
    log.info(f"  NLP accuracy: {nlp_clf.score(X_names, y_nlp):.3f}")

    # ── D. Wrap into NLPAugmentedSchemaClassifier ─────────────────────────
    final = NLPAugmentedSchemaClassifier(combined, nlp_clf, le_nlp, SEMANTIC_LEXICON, threshold=0.65)

    # Self-test
    log.info("\n  Self-test (Stage-1 keyword override):")
    tests = [
        ("gps_latitude",    "coordinates"), ("customer_pan_no",   "pan_number"),
        ("swift_bic_code",  "swift_code"),  ("transaction_amount","amount"),
        ("email_address",   "email"),       ("is_active_flag",    "boolean"),
        ("hash_checksum",   "hash_value"),  ("stock_ticker",      "ticker_symbol"),
    ]
    dummy = [0.0] * 25
    passes = 0
    for col, expected in tests:
        got = final.predict_single(dummy, col)
        ok  = "✓" if got == expected else "✗"
        if got == expected: passes += 1
        log.info(f"    {ok}  {col:<30} → {got}")
    log.info(f"  Self-test: {passes}/{len(tests)} passed")

    # ── E. Save ───────────────────────────────────────────────────────────
    joblib.dump(final,  os.path.join(MODELS_DIR, "schema_classifier.pkl"))
    joblib.dump(le_ext, os.path.join(MODELS_DIR, "schema_label_encoder.pkl"))
    log.info("  ✓ schema_classifier.pkl + schema_label_encoder.pkl")


# =========================================================================
# CHART RELEVANCE SCORER — trains chart_relevance_scorer.pkl
# =========================================================================
def train_chart():
    _hdr("Step 4/4 — Chart Relevance Scorer")
    TYPES = ["bar","line","scatter","heatmap","histogram","box","pie"]
    X_list, y_list = [], []

    for t in TYPES:
        for _ in range(600):
            nr  = RNG.integers(30, 1000)
            nc  = RNG.integers(2, 20)
            num = RNG.uniform(0.1, 1.0)
            fv  = [
                min(nr/10000, 1.0), min(nc/50, 1.0), num, 1-num,
                RNG.uniform(0, 0.5), RNG.uniform(-2, 2),
                RNG.uniform(0, 1),   RNG.uniform(0, 0.5),
                float(RNG.random() > 0.5), RNG.uniform(0, 1),
            ]
            X_list.append(fv); y_list.append(t)

    X = np.array(X_list, dtype=np.float32)
    X += RNG.normal(0, 0.01, X.shape).astype(np.float32)
    y = np.array(y_list)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=6,
                                 n_jobs=-1, random_state=42, class_weight="balanced",
                                 oob_score=True)
    cv = cross_val_score(clf, X_tr, y_tr,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="accuracy")
    log.info(f"  5-Fold CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

    clf.fit(X_tr, y_tr)
    log.info(f"  OOB accuracy:       {clf.oob_score_:.3f}")
    log.info(f"  Test accuracy:      {clf.score(X_te, y_te):.3f}  ← real number")

    clf.fit(X, y)   # retrain on all data for production
    joblib.dump(clf, os.path.join(MODELS_DIR, "chart_relevance_scorer.pkl"))
    log.info("  ✓ chart_relevance_scorer.pkl")


# =========================================================================
# MAIN — Run everything
# =========================================================================
if __name__ == "__main__":
    t0   = time.time()
    dfs  = load_datasets(max_openml=30)

    train_drift(dfs)
    train_schema()
    train_chart()

    _hdr("ALL 6 ARTIFACTS SAVED")
    log.info(f"  Total time: {time.time()-t0:.0f} seconds")
    log.info(f"  Output dir: {MODELS_DIR}")
    for f in sorted(os.listdir(MODELS_DIR)):
        size = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
        log.info(f"    {f:<40} {size:>8.1f} KB")

    log.info("""
  NEXT STEP — Download all .pkl files:

    from google.colab import files
    import glob
    for f in glob.glob("/content/dipex_models/*.pkl"):
        files.download(f)
        print("Downloaded:", f)

  Then copy them to:  dipex_project/models/
""")
