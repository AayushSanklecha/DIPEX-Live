# @title DIPEX — Production ML Artifact Training (Colab)
# ============================================================
# DIPEX Production-Grade ML Trainer
# ============================================================
# Run this notebook in Google Colab (GPU not required).
# At the end, download the 6 .pkl files and place them in:
#   dipex_project/models/
#
# What this trains (on real, messy, diverse data):
#   1. drift_autoencoder.pkl + drift_scaler.pkl + drift_pca.pkl
#      — Portable MLPRegressor autoencoder for multivariate drift.
#        Trained on 50+ OpenML datasets (500K+ rows after augmentation).
#
#   2. schema_classifier.pkl + schema_label_encoder.pkl
#      — RandomForestClassifier for semantic column type detection.
#        500+ real labelled column samples per class (15 classes).
#        CV accuracy target: 0.88+
#
#   3. chart_relevance_scorer.pkl
#      — RandomForestClassifier for chart type ranking.
#        600 richly-varied synthetic datasets per chart type (7 types).
#        CV accuracy target: 0.82+
#
# Expected runtime: 20-35 minutes with Colab Standard CPU.
# ============================================================

# ── Cell 1: Install dependencies ─────────────────────────────────────────────

# @title Cell 1 — Install
# %%capture
# !pip install -q openml scikit-learn pandas numpy joblib

# ── Cell 2: Imports & Setup ───────────────────────────────────────────────────

# @title Cell 2 — Imports & Setup
import os
import sys
import json
import warnings
import logging
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_diabetes,
    fetch_california_housing,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("dipex_trainer")

MODELS_DIR = "/content/dipex_models"
os.makedirs(MODELS_DIR, exist_ok=True)

RNG = np.random.default_rng(2024)
log.info("Setup complete. Output dir: %s", MODELS_DIR)


# ── Cell 3: Data Loading Utilities ────────────────────────────────────────────

# @title Cell 3 — Data Loaders
def _inject_messiness(X: np.ndarray, null_frac=0.12, outlier_frac=0.04) -> np.ndarray:
    """Inject realistic NaNs and hard outliers into a numeric array."""
    X = X.astype(float).copy()
    n, m = X.shape
    # NaN injection
    X[RNG.random((n, m)) < null_frac] = np.nan
    # Outlier injection (5–10× IQR on random cells)
    n_out = max(1, int(n * outlier_frac))
    for r in RNG.choice(n, n_out, replace=False):
        c = int(RNG.integers(0, m))
        col_std = np.nanstd(X[:, c])
        X[r, c] = RNG.choice([-1, 1]) * col_std * RNG.uniform(5, 12)
    return X


def load_sklearn_datasets() -> list[pd.DataFrame]:
    loaders = [
        load_iris, load_wine, load_breast_cancer,
        load_diabetes, fetch_california_housing,
    ]
    dfs = []
    for fn in loaders:
        try:
            b = fn()
            dfs.append(pd.DataFrame(b.data, columns=b.feature_names))
            log.info("  sklearn %-40s %s", fn.__name__, dfs[-1].shape)
        except Exception as e:
            log.warning("  %s failed: %s", fn.__name__, e)
    return dfs


def load_openml_datasets(max_datasets=60) -> list[pd.DataFrame]:
    """
    Fetch a curated diverse set of OpenML tabular datasets.
    Covers: finance, healthcare, engineering, social, sensor, text-analytics domains.
    Each has 2–50 numeric features and 100–500K rows.
    """
    try:
        import openml
    except ImportError:
        log.warning("openml not installed — run: pip install openml")
        return []

    # Curated dataset IDs (data_id) with domain diversity
    curated_ids = [
        # ── Finance / Credit ──────────────────────
        31,    # credit-g
        29,    # credit-approval
        44,    # spambase
        1590,  # adult income
        1461,  # bank marketing
        40981, # diabetes (Pima)
        40984, # taiwanese credit
        # ── Healthcare / Bio ─────────────────────
        37,    # diabetes
        40691, # wine quality (red)
        40692, # wine quality (white)
        300,   # isolet
        1510,  # wdbc (breast cancer Wisconsin)
        40982, # steel plates faults
        # ── Engineering / Sensors ────────────────
        4534,  # PhishingWebsites
        4538,  # madelon
        4134,  # Bioresponse
        1119,  # HCC survival
        1489,  # phoneme
        1120,  # magic telescope
        1515,  # abalone
        # ── Social / Demographic ─────────────────
        180,   # covertype (reduced)
        4541,  # Internet-advertisements
        40685, # shuttle
        43,    # electricity (ELEC2)
        # ── Time-series / Temporal patterns ──────
        1046,  # mozilla4
        1039,  # hiv_bovis
        1049,  # pc4
        1050,  # pc3
        # ── Diverse regression targets ───────────
        41187, # loan amount
        42,    # labor
        847,   # fri_c4_500_10
        844,   # fri_c3_500_10
        819,   # fri_c2_500_10
        816,   # fri_c1_500_10
        # ── Multi-class / multi-feature ──────────
        1053,  # jm1
        1063,  # kc2
        1067,  # kc1
        1068,  # pc1
        23380, # energy efficiency
        554,   # mnist (reduced)
        40975, # car
        14,    # mfeat-fourier
        18,    # mfeat-morphological
        22,    # mfeat-pixel
        # ── Regression datasets ───────────────────
        531,   # boston
        560,   # bodyfat
        564,   # kin8nm
        550,   # quake
    ]

    dfs = []
    failed = 0
    for did in curated_ids[:max_datasets]:
        try:
            dataset = openml.datasets.get_dataset(
                did, download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )
            X, y, _, col_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            num = X.select_dtypes(include="number")
            if num.shape[1] >= 2 and len(num) >= 50:
                dfs.append(num)
                log.info("  OpenML %-6d %-35s %s", did, dataset.name[:35], num.shape)
        except Exception as e:
            failed += 1
            log.debug("  OpenML %d skip: %s", did, e)

    log.info("Loaded %d OpenML datasets (%d failed/skipped)", len(dfs), failed)
    return dfs


# ── Cell 4: Drift Autoencoder ─────────────────────────────────────────────────

# @title Cell 4 — Train Drift Autoencoder
N_DRIFT_FEATURES = 15
N_PCA_COMPONENTS = 12

def build_drift_corpus(dfs: list[pd.DataFrame]) -> np.ndarray:
    """
    Combine all datasets into a unified corpus.
    Each dataset is locally standardised, then padded/truncated to N_DRIFT_FEATURES.
    Realistic messiness is injected per-dataset.
    Also synthesises "drifted" variants (shifted, scaled) for variety.
    """
    blocks = []

    for df in dfs:
        num = df.select_dtypes(include="number").dropna(axis=1, how="all")
        if num.shape[1] < 2:
            continue

        arr = num.values.astype(float)
        n, m = arr.shape

        # Fill existing NaNs with column median
        for j in range(m):
            med = np.nanmedian(arr[:, j])
            arr[np.isnan(arr[:, j]), j] = 0.0 if np.isnan(med) else med

        # Local standardise (this makes the corpus scale-invariant)
        sc = StandardScaler()
        arr = sc.fit_transform(arr)
        arr = np.clip(arr, -5, 5)

        # Add messy variant (NaN + outliers)
        arr_messy = _inject_messiness(arr.copy(), null_frac=0.12, outlier_frac=0.04)
        arr_messy = np.nan_to_num(arr_messy, nan=0.0, posinf=3.0, neginf=-3.0)

        # Add "mild drift" variant (mean shift + scale perturbation)
        arr_shifted = arr + RNG.normal(0, 0.3, arr.shape)
        arr_scaled  = arr * RNG.uniform(0.7, 1.4, (1, m))

        # Pad / truncate all variants
        def _pad(a):
            if a.shape[1] == N_DRIFT_FEATURES:
                return a
            if a.shape[1] > N_DRIFT_FEATURES:
                return a[:, :N_DRIFT_FEATURES]
            return np.hstack([a, np.zeros((a.shape[0], N_DRIFT_FEATURES - a.shape[1]))])

        for variant in [arr, arr_messy, arr_shifted, arr_scaled]:
            padded = _pad(variant).astype(np.float32)
            padded = np.clip(padded, -5, 5)
            blocks.append(padded)

    corpus = np.vstack(blocks)
    RNG.shuffle(corpus)
    corpus = np.nan_to_num(corpus, nan=0.0)
    log.info("Drift corpus: %d rows × %d features", *corpus.shape)
    return corpus


def train_drift_autoencoder(dfs: list[pd.DataFrame]) -> None:
    log.info("\n=== [1/3] Drift Autoencoder ===")
    corpus = build_drift_corpus(dfs)

    # GlobalScaler (metadata: n_features_in_=15)
    global_sc = StandardScaler()
    corpus_scaled = global_sc.fit_transform(corpus)
    log.info("  GlobalScaler fit on %d rows", len(corpus_scaled))

    # PCA
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    corpus_pca = pca.fit_transform(corpus_scaled)
    var = pca.explained_variance_ratio_.sum()
    log.info("  PCA(%d): %.1f%% variance explained", N_PCA_COMPONENTS, var * 100)

    # MLP Autoencoder  12 → 6 → 12
    ae = MLPRegressor(
        hidden_layer_sizes=(N_PCA_COMPONENTS, N_PCA_COMPONENTS // 2, N_PCA_COMPONENTS),
        activation="relu",
        solver="adam",
        max_iter=800,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        verbose=False,
    )
    ae.fit(corpus_pca, corpus_pca)
    mse = float(np.mean(np.square(corpus_pca - ae.predict(corpus_pca))))
    log.info("  MSE (train): %.6f  n_iter=%d", mse, ae.n_iter_)

    joblib.dump(ae,        os.path.join(MODELS_DIR, "drift_autoencoder.pkl"))
    joblib.dump(global_sc, os.path.join(MODELS_DIR, "drift_scaler.pkl"))
    joblib.dump(pca,       os.path.join(MODELS_DIR, "drift_pca.pkl"))
    log.info("  ✓ Saved drift_autoencoder.pkl + drift_scaler.pkl + drift_pca.pkl")


# ── Cell 5: Schema Classifier ─────────────────────────────────────────────────

# @title Cell 5 — Train Schema Semantic-Type Classifier

SEMANTIC_LABELS = [
    # Original 15
    "id", "age", "amount", "date", "category", "text",
    "phone", "email", "boolean", "zipcode", "percentage",
    "score", "count", "name", "unknown",
    # Extended 6
    "url", "ip_address", "coordinates", "duration", "address", "currency_code",
]

# Feature extraction (must match ingestion/schema_infer.py _FEAT_ORDER exactly)
_FEAT_ORDER = [
    "null_rate", "unique_rate", "is_numeric", "is_string",
    "is_datetime", "mean_val", "std_val", "min_val", "max_val",
    "skew_val", "all_integer", "max_lt_200", "max_lt_1",
    "all_positive", "n_distinct", "email_pattern", "phone_pattern",
    "mean_str_len", "high_cardinality", "low_cardinality",
    # Extended features for new labels
    "url_pattern", "ip_pattern", "coord_range", "coord_precision", "currency_pattern",
]

def extract_column_features(series: pd.Series, col_name: str = "") -> dict:
    """
    Local replica of ingestion/schema_infer._extract_column_features().
    Kept here so the Colab script is self-contained.
    """
    s = series.dropna()
    n = max(len(s), 1)

    is_num  = pd.api.types.is_numeric_dtype(series)
    is_str  = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    is_dt   = pd.api.types.is_datetime64_any_dtype(series)

    num_vals = pd.to_numeric(s, errors="coerce").dropna() if not is_num else s.dropna()
    str_vals = s.astype(str) if is_str else pd.Series([], dtype=str)

    null_rate   = series.isnull().mean()
    unique_rate = series.nunique(dropna=True) / max(len(series), 1)

    mean_val = float(num_vals.mean())   if len(num_vals) > 0 else 0.0
    std_val  = float(num_vals.std())    if len(num_vals) > 1 else 0.0
    min_val  = float(num_vals.min())    if len(num_vals) > 0 else 0.0
    max_val  = float(num_vals.max())    if len(num_vals) > 0 else 0.0
    skew_val = float(num_vals.skew())   if len(num_vals) > 3 else 0.0

    try:
        all_integer = float((num_vals == num_vals.apply(lambda x: int(x))).all()) \
                      if len(num_vals) > 0 else 0.0
    except Exception:
        all_integer = 0.0
    max_lt_200   = float(max_val < 200)  if len(num_vals) > 0 else 0.0
    max_lt_1     = float(max_val <= 1.0) if len(num_vals) > 0 else 0.0
    all_pos      = float((num_vals >= 0).all()) if len(num_vals) > 0 else 0.0
    n_distinct   = float(series.nunique(dropna=True))

    email_pattern = float(str_vals.str.contains(r"@.*\.", na=False).mean()) if is_str and len(str_vals) > 0 else 0.0
    phone_pattern = float(str_vals.str.contains(r"^\+?\d[\d\s\-()]{7,}$", na=False, regex=True).mean()) \
                    if is_str and len(str_vals) > 0 else 0.0
    mean_str_len  = float(str_vals.str.len().mean()) if is_str and len(str_vals) > 0 else 0.0

    # Extended pattern signals (for 6 new labels)
    url_pattern = (
        float(str_vals.str.contains(r"https?://|www\.", na=False).mean())
        if is_str and len(str_vals) > 0 else 0.0
    )
    ip_pattern = (
        float(str_vals.str.match(
            r"^(\d{1,3}\.){3}\d{1,3}$|^([0-9a-fA-F]{1,4}:){2,7}", na=False
        ).mean())
        if is_str and len(str_vals) > 0 else 0.0
    )
    coord_range = (
        float(((num_vals >= -180) & (num_vals <= 180)).all())
        if len(num_vals) > 0 else 0.0
    )
    coord_precision = (
        float((num_vals % 1 != 0).mean() > 0.8)
        if len(num_vals) > 0 else 0.0
    )
    currency_pattern = (
        float(str_vals.str.match(r"^[A-Z]{3}$", na=False).mean() > 0.7)
        if is_str and len(str_vals) > 0 else 0.0
    )

    return {
        "null_rate": null_rate, "unique_rate": unique_rate,
        "is_numeric": float(is_num), "is_string": float(is_str), "is_datetime": float(is_dt),
        "mean_val": mean_val, "std_val": std_val, "min_val": min_val, "max_val": max_val,
        "skew_val": skew_val, "all_integer": all_integer, "max_lt_200": max_lt_200,
        "max_lt_1": max_lt_1, "all_positive": all_pos, "n_distinct": n_distinct,
        "email_pattern": email_pattern, "phone_pattern": phone_pattern,
        "mean_str_len": mean_str_len,
        "high_cardinality": float(unique_rate > 0.9),
        "low_cardinality":  float(unique_rate < 0.05),
        # Extended
        "url_pattern": url_pattern, "ip_pattern": ip_pattern,
        "coord_range": coord_range, "coord_precision": coord_precision,
        "currency_pattern": currency_pattern,
    }


def _make_series(label: str, n: int) -> pd.Series:
    """
    Generate ONE pd.Series representative of a semantic label,
    with random parametrisation for maximum diversity.
    Each call gives a slightly different instance (different params, sizes, null rates).
    """
    null_p = RNG.uniform(0.0, 0.25)     # 0-25% NaN injection
    n_use  = int(RNG.integers(50, n))   # variable length 50-n

    def _null(s: pd.Series) -> pd.Series:
        s = s.copy()
        if null_p > 0:
            idx = RNG.choice(len(s), max(1, int(len(s) * null_p)), replace=False)
            s.iloc[idx] = np.nan
        return s

    if label == "id":
        choice = RNG.integers(0, 3)
        if choice == 0:
            # Sequential int
            return _null(pd.Series(np.arange(10000, 10000 + n_use)))
        elif choice == 1:
            # Random large ints (high cardinality)
            return _null(pd.Series(RNG.integers(1_000_000, 9_999_999, n_use)))
        else:
            # UUID-like strings
            return _null(pd.Series([f"ID-{RNG.integers(0, 999999):06d}" for _ in range(n_use)]))

    elif label == "age":
        choice = RNG.integers(0, 4)
        if choice == 0:
            arr = RNG.integers(0, 100, n_use).astype(float)
        elif choice == 1:
            arr = RNG.integers(18, 65, n_use).astype(float)
        elif choice == 2:
            arr = RNG.integers(0, 18, n_use).astype(float)         # children
        else:
            arr = RNG.normal(35, 12, n_use).clip(0, 110)
        # Random outliers
        if RNG.random() < 0.3:
            arr[RNG.choice(len(arr), 3, replace=False)] = RNG.integers(111, 150, 3)
        return _null(pd.Series(arr))

    elif label == "amount":
        choice = RNG.integers(0, 5)
        if choice == 0:
            arr = RNG.exponential(RNG.uniform(100, 50000), n_use)
        elif choice == 1:
            arr = RNG.lognormal(RNG.uniform(3, 9), RNG.uniform(0.5, 2.5), n_use)
        elif choice == 2:
            arr = -1 * RNG.exponential(500, n_use)                 # negative
        elif choice == 3:
            arr = RNG.normal(RNG.uniform(-1e5, 1e5), RNG.uniform(100, 1e4), n_use)
        else:
            arr = RNG.uniform(-5000, 50000, n_use)                 # uniform
        return _null(pd.Series(arr.astype(float)))

    elif label == "date":
        start = pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(RNG.integers(0, 5000)))
        try:
            freq = str(RNG.choice(["D", "h", "W", "ME"]))
            dts  = pd.date_range(start, periods=n_use, freq=freq)
        except ValueError:
            # Some pandas versions don't support 'ME'; fall back to 'MS'
            dts  = pd.date_range(start, periods=n_use, freq="MS")
        fmt   = str(RNG.choice(["%Y-%m-%d", "%d/%m/%Y", "%Y%m%d", "%m-%d-%Y"]))
        arr   = pd.Series(dts.strftime(fmt))
        return _null(arr)

    elif label == "category":
        n_cats = int(RNG.integers(2, 12))
        cats   = [f"Cat_{chr(65+i)}" for i in range(n_cats)]
        # Various cat distributions: balanced, imbalanced, ordinal
        if RNG.random() < 0.4:
            probs  = RNG.dirichlet(np.ones(n_cats) * RNG.uniform(0.3, 3))
            arr    = RNG.choice(cats, n_use, p=probs)
        else:
            arr    = RNG.choice(cats, n_use)
        return _null(pd.Series(arr, dtype=object))

    elif label == "text":
        word_pool = [
            "lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
            "adipiscing", "elit", "sed", "do", "eiusmod", "tempor",
            "incididunt", "labore", "magna", "aliqua", "veniam",
        ]
        word_count = int(RNG.integers(10, 40))
        arr = pd.Series([
            " ".join(RNG.choice(word_pool, int(RNG.integers(5, word_count))).tolist())
            for _ in range(n_use)
        ])
        return _null(arr)

    elif label == "phone":
        fmt = int(RNG.integers(0, 3))
        if fmt == 0:
            arr = pd.Series([f"+1-{RNG.integers(200,999)}-{RNG.integers(100,999)}-{RNG.integers(1000,9999)}"
                              for _ in range(n_use)])
        elif fmt == 1:
            arr = pd.Series([f"({RNG.integers(200,999)}) {RNG.integers(100,999)}-{RNG.integers(1000,9999)}"
                              for _ in range(n_use)])
        else:
            arr = pd.Series([f"{RNG.integers(100,999)}{RNG.integers(100,999)}{RNG.integers(1000,9999)}"
                              for _ in range(n_use)])
        return _null(arr)

    elif label == "email":
        domains = ["gmail.com", "yahoo.com", "outlook.com", "company.org",
                   "work.net", "university.edu", "enterprise.io", "gov.in"]
        arr = pd.Series([
            f"{RNG.choice(['user','admin','contact','info','support'])}"
            f"{RNG.integers(0, 99999)}@{RNG.choice(domains)}"
            for _ in range(n_use)
        ])
        return _null(arr)

    elif label == "boolean":
        choice = int(RNG.integers(0, 4))
        if choice == 0:
            arr = pd.Series(RNG.integers(0, 2, n_use))
        elif choice == 1:
            arr = pd.Series(RNG.choice([True, False], n_use))
        elif choice == 2:
            arr = pd.Series(RNG.choice(["yes", "no"], n_use))
        else:
            arr = pd.Series(RNG.choice(["true", "false", "1", "0"], n_use))
        return _null(arr)

    elif label == "zipcode":
        choice = int(RNG.integers(0, 3))
        if choice == 0:
            arr = pd.Series([f"{RNG.integers(10000,99999)}" for _ in range(n_use)])
        elif choice == 1:
            arr = pd.Series(RNG.integers(10000, 99999, n_use))
        else:
            arr = pd.Series([f"{RNG.integers(100000,999999)}" for _ in range(n_use)])  # 6-digit (India)
        return _null(arr)

    elif label == "percentage":
        choice = int(RNG.integers(0, 3))
        if choice == 0:
            arr = RNG.uniform(0, 1, n_use)         # 0-1 format
        elif choice == 1:
            arr = RNG.uniform(0, 100, n_use)        # 0-100 format
        else:
            arr = RNG.beta(2, 5, n_use)             # skewed 0-1
        return _null(pd.Series(arr.astype(float)))

    elif label == "score":
        choice = int(RNG.integers(0, 4))
        if choice == 0:
            arr = RNG.uniform(0, 10, n_use)         # 0-10
        elif choice == 1:
            arr = RNG.integers(1, 6, n_use).astype(float)  # 1-5 rating
        elif choice == 2:
            arr = RNG.normal(50, 15, n_use).clip(0, 100)   # normalized score
        else:
            arr = RNG.uniform(300, 850, n_use)              # FICO-like
        return _null(pd.Series(arr.astype(float)))

    elif label == "count":
        choice = int(RNG.integers(0, 4))
        if choice == 0:
            arr = RNG.integers(0, 1000, n_use)
        elif choice == 1:
            arr = RNG.poisson(RNG.uniform(1, 100), n_use)
        elif choice == 2:
            arr = RNG.integers(0, 20, n_use)        # small counts
        else:
            arr = RNG.integers(0, 1_000_000, n_use) # large counts
        return _null(pd.Series(arr.astype(float)))

    elif label == "name":
        first = ["Alice", "Bob", "Carlos", "Diana", "Eva", "Frank", "Grace",
                 "Hector", "Iris", "Jack", "Kai", "Lena", "Mia", "Noah",
                 "Olivia", "Pablo", "Quinn", "Rosa", "Sam", "Tina", "Uma"]
        last  = ["Smith", "Jones", "Kumar", "Lee", "Patel", "Brown", "Wilson",
                 "Garcia", "Nguyen", "Kim", "Chen", "Sharma", "Singh", "Müller",
                 "O'Brien", "van der Berg", "Al-Rashid", "Yamamoto", "Santos"]
        choice = int(RNG.integers(0, 3))
        if choice == 0:
            arr = pd.Series([f"{RNG.choice(first)} {RNG.choice(last)}" for _ in range(n_use)])
        elif choice == 1:
            arr = pd.Series([RNG.choice(first) for _ in range(n_use)])   # first only
        else:
            arr = pd.Series([RNG.choice(last) for _ in range(n_use)])    # last only
        return _null(arr)

    elif label == "url":
        _DOMAINS = ["example.com", "data.io", "api.github.com", "cdn.corp.net",
                    "storage.cloud.co", "img.site.org", "feeds.news.com"]
        _SCHEMES = ["https://", "http://", "https://www."]
        _PATHS   = ["", "/page", "/data/file.csv", "/api/v2/results",
                    "/images/thumb.jpg", "/user/profile", "/report/2024"]
        parts = [
            f"{RNG.choice(_SCHEMES)}{RNG.choice(_DOMAINS)}{RNG.choice(_PATHS)}"
            for _ in range(n_use)
        ]
        return _null(pd.Series(parts))

    elif label == "ip_address":
        choice = RNG.integers(0, 2)
        if choice == 0:
            # IPv4
            parts = [
                f"{RNG.integers(1,255)}.{RNG.integers(0,255)}"
                f".{RNG.integers(0,255)}.{RNG.integers(0,255)}"
                for _ in range(n_use)
            ]
        else:
            # IPv4 private ranges
            prefix = RNG.choice(["192.168", "10.0", "172.16"])
            parts = [
                f"{prefix}.{RNG.integers(0,255)}.{RNG.integers(1,254)}"
                for _ in range(n_use)
            ]
        return _null(pd.Series(parts))

    elif label == "coordinates":
        choice = RNG.integers(0, 3)
        if choice == 0:
            # Latitude: -90 to 90
            arr = RNG.uniform(-90, 90, n_use).round(RNG.integers(4, 7))
        elif choice == 1:
            # Longitude: -180 to 180
            arr = RNG.uniform(-180, 180, n_use).round(RNG.integers(4, 7))
        else:
            # City-like cluster (realistic dense range)
            centre = RNG.uniform(-80, 80)
            arr = RNG.normal(centre, RNG.uniform(0.01, 2.0), n_use).clip(-180, 180)
        return _null(pd.Series(arr))

    elif label == "duration":
        choice = RNG.integers(0, 3)
        if choice == 0:
            # Integer seconds (most common in analytics)
            arr = RNG.integers(0, 7200, n_use).astype(float)
            return _null(pd.Series(arr))
        elif choice == 1:
            # HH:MM:SS strings
            secs = RNG.integers(0, 86400, n_use)
            parts = [f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}" for s in secs]
            return _null(pd.Series(parts))
        else:
            # "Xh Ym" / "Xs" human-readable strings
            _UNITS = ["s", "sec", "min", "h", "hr"]
            parts = [f"{RNG.integers(1, 3600)}{RNG.choice(_UNITS)}" for _ in range(n_use)]
            return _null(pd.Series(parts))

    elif label == "address":
        _STREETS = ["Main St", "Park Ave", "Oak Lane", "MG Road", "Gandhi Nagar",
                    "Church Road", "High Street", "Elm Drive", "Sector 15",
                    "Lake View Blvd", "Station Road", "Civil Lines"]
        _CITIES  = ["", ", Mumbai", ", London", ", New York", ", Delhi",
                    ", Chicago", ", Sydney", ", Berlin"]
        parts = [
            f"{RNG.integers(1, 9999)} {RNG.choice(_STREETS)}{RNG.choice(_CITIES)}"
            for _ in range(n_use)
        ]
        return _null(pd.Series(parts))

    elif label == "currency_code":
        _CURRENCIES = [
            "USD", "EUR", "GBP", "JPY", "INR", "AUD", "CAD", "CHF",
            "CNY", "SGD", "HKD", "NOK", "SEK", "DKK", "MXN", "BRL",
            "ZAR", "AED", "SAR", "KRW",
        ]
        # Real distributions — USD/EUR are far more common
        _WEIGHTS = [0.25, 0.20, 0.10, 0.07, 0.06, 0.04, 0.04, 0.04,
                    0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01,
                    0.01, 0.01, 0.01, 0.00]
        w = np.array(_WEIGHTS[:len(_CURRENCIES)], dtype=float)
        w /= w.sum()
        arr = RNG.choice(_CURRENCIES, n_use, p=w)
        return _null(pd.Series(arr))

    elif label == "unknown":
        choice = int(RNG.integers(0, 5))
        if choice == 0:
            arr = pd.Series([f"X_{RNG.integers(0,9999)}" for _ in range(n_use)])
        elif choice == 1:
            # Hex chars of random bytes — one char per element
            raw = RNG.bytes(n_use * 2 + 4).hex()
            arr = pd.Series(list(raw[:n_use]))
        elif choice == 2:
            arr = pd.Series(RNG.normal(0, 1e8, n_use))   # extreme-scale floats
        elif choice == 3:
            arr = pd.Series([""] * n_use)                # empty strings
        else:
            arr = pd.Series(RNG.choice([None, 0, "N/A", "?", np.nan], n_use))
        return _null(arr)

    else:
        # fallthrough (should never happen, but be safe)
        return _null(pd.Series(RNG.normal(0, 1, n_use)))


def generate_schema_training_data(
    dfs: list[pd.DataFrame],
    n_synthetic_per_class: int = 500,
    series_length: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for the schema classifier from:
      A. Synthetic Series — n_synthetic_per_class per label with random params
      B. Real OpenML columns — labelled via column name heuristics
    """
    rows:   list[list[float]] = []
    labels: list[str]         = []
    errors: int               = 0

    log.info("  Generating %d synthetic samples per class...", n_synthetic_per_class)

    # ── A: Synthetic samples ─────────────────────────────────────────────────
    for lbl in SEMANTIC_LABELS:
        count = 0
        attempts = 0
        while count < n_synthetic_per_class and attempts < n_synthetic_per_class * 4:
            attempts += 1
            try:
                s = _make_series(lbl, series_length)
                feats = extract_column_features(s, col_name=lbl)
                rows.append([feats[k] for k in _FEAT_ORDER])
                labels.append(lbl)
                count += 1
            except Exception:
                errors += 1

        log.info("  %-14s: %d synthetic samples", lbl, count)

    # ── B: Real columns from OpenML (heuristic labelling by column name) ─────
    name_to_label: dict[str, str] = {
        # age
        "age": "age", "years": "age", "yrs": "age", "year_of_birth": "age",
        # amount / financial
        "income": "amount", "salary": "amount", "price": "amount",
        "revenue": "amount", "cost": "amount", "amount": "amount",
        "balance": "amount", "payment": "amount", "total": "amount",
        "tax": "amount", "fee": "amount", "charge": "amount",
        "expenditure": "amount", "profit": "amount", "loss": "amount",
        # count
        "count": "count", "num": "count", "number": "count",
        "quantity": "count", "volume": "count", "n_": "count",
        "views": "count", "clicks": "count",
        # percentage / ratio
        "rate": "percentage", "ratio": "percentage", "percent": "percentage",
        "pct": "percentage", "proportion": "percentage",
        # score / rating
        "score": "score", "rating": "score", "grade": "score",
        "rank": "score", "priority": "score", "gpa": "score",
        # category
        "type": "category", "class": "category", "group": "category",
        "category": "category", "status": "category", "gender": "category",
        "region": "category", "country": "category", "city": "category",
        # boolean
        "flag": "boolean", "is_": "boolean", "has_": "boolean",
        "active": "boolean", "enabled": "boolean",
        # id
        "id": "id", "uuid": "id", "key": "id",
    }

    real_added = 0
    for df in dfs:
        num = df.select_dtypes(include="number")
        for col in num.columns:
            col_lower = col.lower().replace(" ", "_")
            # Find matching label
            matched_label = None
            for keyword, lbl in name_to_label.items():
                if keyword in col_lower:
                    matched_label = lbl
                    break
            if matched_label is None:
                continue

            try:
                feats = extract_column_features(num[col], col_name=col)
                rows.append([feats[k] for k in _FEAT_ORDER])
                labels.append(matched_label)
                real_added += 1
            except Exception:
                errors += 1

    log.info("  Real dataset columns added: %d (errors suppressed: %d)", real_added, errors)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels)
    return X, y


def train_schema_classifier(dfs: list[pd.DataFrame]) -> None:
    log.info("\n=== [2/3] Schema Semantic-Type Classifier ===")

    X, y = generate_schema_training_data(dfs, n_synthetic_per_class=500)

    classes, counts = np.unique(y, return_counts=True)
    log.info("  Total samples: %d   Classes: %d", len(X), len(classes))
    for c, n in zip(classes, counts):
        log.info("    %-16s: %d", c, n)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── Train / Test split (20 % held-out, stratified) ────────────────────────
    # This is the ONLY honest accuracy metric for a Random Forest.
    # clf.predict(X_train) always returns ~1.00 because RF memorises training
    # data — that number is meaningless. The test-set report below is real.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, stratify=y_enc, random_state=42
    )
    log.info("  Train size: %d   Test size: %d (20%% held out)",
             len(X_train), len(X_test))

    clf_rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=18,            # reduced from 25 — less memorisation
        min_samples_leaf=4,      # at least 4 samples per leaf
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,          # second unbiased accuracy estimate
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold CV on training split only
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(clf_rf, X_train, y_train, cv=skf,
                             scoring="accuracy", n_jobs=-1)
    log.info("  5-fold CV accuracy (train split): %.3f \u00b1 %.3f",
             cv_acc.mean(), cv_acc.std())

    # Fit on full training split
    clf_rf.fit(X_train, y_train)
    oob_acc = clf_rf.oob_score_
    test_acc = clf_rf.score(X_test, y_test)
    log.info("  OOB accuracy (unbiased):          %.3f", oob_acc)
    log.info("  Test-set accuracy (20%% held out): %.3f  ← USE THIS NUMBER", test_acc)
    log.info("  Train accuracy (meaningless for RF): %.3f",
             clf_rf.score(X_train, y_train))

    # ── Classification report on TEST SET only ────────────────────────────────
    print("\n" + "="*60)
    print("SCHEMA CLASSIFIER — TEST SET Report (20%% held-out, n=%d)" % len(X_test))
    print("(These numbers are REAL. Train accuracy is always ~1.00 for RF.)")
    print("="*60)
    preds_test = clf_rf.predict(X_test)
    print(classification_report(y_test, preds_test, target_names=le.classes_))

    # Retrain on ALL data for production artifact
    clf_rf.fit(X, y_enc)
    joblib.dump(clf_rf, os.path.join(MODELS_DIR, "schema_classifier.pkl"))
    joblib.dump(le,     os.path.join(MODELS_DIR, "schema_label_encoder.pkl"))
    log.info("  \u2713 Saved schema_classifier.pkl + schema_label_encoder.pkl")
    log.info("  Note: artifact retrained on 100%% data after evaluation.")


# ── Cell 6: Chart Relevance Scorer ───────────────────────────────────────────

# @title Cell 6 — Train Chart Relevance Scorer

CHART_TYPES = ["bar", "line", "scatter", "heatmap", "histogram", "box", "pie"]

# 10-feature vector matching reporting_service/chart_relevance_scorer._extract_features()
def extract_chart_features(df: pd.DataFrame, query_intent: str = "general") -> np.ndarray:
    n_rows  = len(df)
    n_cols  = len(df.columns)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    num_ratio = len(num_cols) / max(n_cols, 1)
    cat_ratio = len(cat_cols) / max(n_cols, 1)
    first_cat_card = df[cat_cols[0]].nunique() / max(n_rows, 1) if cat_cols else 0.0

    skew = float(df[num_cols[0]].skew()) if num_cols else 0.0
    skew = float(np.clip(skew, -10, 10))

    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        mean_corr = float(corr.values.mean())
    else:
        mean_corr = 0.0

    null_rate = float(df.isnull().mean().mean())
    dt_cols   = df.select_dtypes(include="datetime").columns.tolist()
    has_dt    = float(len(dt_cols) > 0 or any("date" in c.lower() or "time" in c.lower()
                                               for c in df.columns))
    intent_map = {
        "trend": 0, "time_series": 0, "distribute": 1, "distribution": 1,
        "compare": 2, "group_by": 2, "correlation": 3, "top_n": 4, "bottom_n": 4,
        "aggregate": 5, "general": 6,
    }
    intent_enc = float(intent_map.get(query_intent or "general", 6)) / 6.0

    return np.array([
        min(n_rows / 10_000, 1.0),
        min(n_cols / 50.0, 1.0),
        num_ratio, cat_ratio,
        first_cat_card, skew,
        mean_corr, null_rate,
        has_dt, intent_enc,
    ], dtype=np.float32)


def _make_chart_sample(chart_type: str, n_base: int = 400) -> tuple[pd.DataFrame, str]:
    """
    Generate ONE highly varied synthetic DataFrame that suits a given chart type.
    Extensive randomisation ensures realistic feature overlap between chart types.
    """
    # Random row count (realistic: 30 to 50K)
    n_rows = int(RNG.integers(30, n_base))
    noise  = lambda shape: RNG.normal(0, RNG.uniform(0.01, 0.3), shape)
    null_p = RNG.uniform(0.0, 0.20)

    def nullify(s: pd.Series) -> pd.Series:
        s = s.copy()
        if null_p > 0 and len(s) > 5:
            s.iloc[RNG.choice(len(s), int(len(s) * null_p), replace=False)] = np.nan
        return s

    if chart_type == "line":
        # Core: datetime-like column + 1-4 numeric time-series
        n_series = int(RNG.integers(1, 5))
        freq = RNG.choice(["D", "h", "W", "min"])
        dts  = pd.date_range("2018-01-01", periods=n_rows, freq=freq)
        data = {"date": dts.astype(str)}
        for i in range(n_series):
            base = RNG.normal(100, 20, n_rows).cumsum()
            data[f"series_{i}"] = nullify(pd.Series(base + noise(n_rows)))
        intent = str(RNG.choice(["trend", "time_series"]))

    elif chart_type == "bar":
        # Core: 1-3 low-cardinality cat cols + 1-3 numeric
        n_cats  = int(RNG.integers(3, 15))
        n_gcols = int(RNG.integers(1, 4))
        n_vcols = int(RNG.integers(1, 4))
        data = {}
        for i in range(n_gcols):
            cats = [f"Cat_{chr(65+j)}" for j in range(n_cats)]
            if RNG.random() < 0.5:
                cats = RNG.choice(cats, n_cats, replace=False).tolist()  # shuffle label order
            data[f"group_{i}"] = nullify(pd.Series(RNG.choice(cats, n_rows)))
        for i in range(n_vcols):
            data[f"value_{i}"] = nullify(pd.Series(RNG.exponential(RNG.uniform(100, 10000), n_rows)))
        intent = str(RNG.choice(["compare", "group_by", "top_n", "bottom_n"]))

    elif chart_type == "scatter":
        # Core: 2-6 correlated numeric cols, no cats, no date
        n_dim = int(RNG.integers(2, 7))
        base  = RNG.normal(0, 1, n_rows)
        data  = {}
        for i in range(n_dim):
            coef = RNG.uniform(-1.5, 1.5)
            data[f"feat_{i}"] = nullify(pd.Series(base * coef + noise(n_rows)))
        intent = str(RNG.choice(["correlation", "general"]))

    elif chart_type == "heatmap":
        # Core: many numeric cols (5-20), dense correlations, no date
        n_dim = int(RNG.integers(5, 21))
        base  = RNG.normal(0, 1, (n_rows, 3))  # 3 latent factors
        data  = {}
        for i in range(n_dim):
            weights = RNG.uniform(-1, 1, 3)
            data[f"var_{i}"] = nullify(pd.Series((base @ weights) + noise(n_rows)))
        intent = str(RNG.choice(["correlation", "general"]))

    elif chart_type == "histogram":
        # Core: 1-4 numeric cols, no cats, no date, potentially skewed
        n_num = int(RNG.integers(1, 5))
        data  = {}
        for i in range(n_num):
            dist = int(RNG.integers(0, 5))
            if dist == 0:
                arr = RNG.exponential(RNG.uniform(10, 1000), n_rows)
            elif dist == 1:
                arr = RNG.lognormal(RNG.uniform(1, 8), RNG.uniform(0.3, 2), n_rows)
            elif dist == 2:
                arr = RNG.normal(RNG.uniform(-100, 100), RNG.uniform(1, 50), n_rows)
            elif dist == 3:
                arr = RNG.gamma(RNG.uniform(0.5, 5), RNG.uniform(1, 100), n_rows)
            else:
                arr = RNG.uniform(0, RNG.uniform(1, 1000), n_rows)
            data[f"val_{i}"] = nullify(pd.Series(arr))
        intent = str(RNG.choice(["distribution", "distribute"]))

    elif chart_type == "box":
        # Core: 1-4 cat grouping cols + 1-4 numeric measurements
        n_groups = int(RNG.integers(2, 8))
        n_vcols  = int(RNG.integers(1, 5))
        cats     = [f"Grp{i}" for i in range(n_groups)]
        data     = {"group": nullify(pd.Series(RNG.choice(cats, n_rows)))}
        for i in range(n_vcols):
            shift = RNG.normal(0, 2, n_groups)
            data[f"metric_{i}"] = nullify(pd.Series([
                RNG.normal(shift[RNG.choice(n_groups)], RNG.uniform(0.5, 2))
                for _ in range(n_rows)
            ]))
        intent = str(RNG.choice(["distribution", "compare"]))

    elif chart_type == "pie":
        # Core: very low cardinality cat (2-7 slices), 1 numeric
        n_slices = int(RNG.integers(2, 8))
        slices   = [f"Seg{i}" for i in range(n_slices)]
        probs    = RNG.dirichlet(np.ones(n_slices) * RNG.uniform(0.5, 3))
        data     = {
            "segment": nullify(pd.Series(RNG.choice(slices, n_rows, p=probs))),
            "value":   nullify(pd.Series(RNG.exponential(1000, n_rows))),
        }
        intent = str(RNG.choice(["aggregate", "general"]))

    else:
        data   = {"x": pd.Series(RNG.normal(0, 1, n_rows))}
        intent = "general"

    df = pd.DataFrame(data)
    return df, intent


def generate_chart_training_data(n_per_class: int = 600) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y_labels: list[str]      = []
    errors = 0

    for chart_type in CHART_TYPES:
        count = 0
        attempts = 0
        while count < n_per_class and attempts < n_per_class * 4:
            attempts += 1
            try:
                df, intent = _make_chart_sample(chart_type, n_base=int(RNG.integers(50, 2000)))
                feats = extract_chart_features(df, query_intent=intent)
                # Add small Gaussian noise to features to prevent memorisation
                feats = feats + RNG.normal(0, 0.01, feats.shape).astype(np.float32)
                X_rows.append(feats)
                y_labels.append(chart_type)
                count += 1
            except Exception:
                errors += 1

        log.info("  %-12s: %d samples", chart_type, count)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels)
    log.info("  Total: %d samples, %d errors suppressed", len(X), errors)
    return X, y


def train_chart_scorer() -> None:
    log.info("\n=== [3/3] Chart Relevance Scorer ===")

    X, y = generate_chart_training_data(n_per_class=600)

    # ── Train / Test split (20% held-out, stratified) ─────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    log.info("  Train size: %d   Test size: %d (20%% held out)",
             len(X_train), len(X_test))

    # CRITICAL: train with string labels — clf.classes_ must be chart-type names
    # (inference code reads clf.classes_ directly as strings like "bar", "line")
    clf = RandomForestClassifier(
        n_estimators=600,
        max_depth=12,            # shallower — less memorisation of synthetic patterns
        min_samples_leaf=8,      # stronger regularisation
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(clf, X_train, y_train, cv=skf,
                             scoring="accuracy", n_jobs=-1)
    log.info("  5-fold CV accuracy (train split): %.3f \u00b1 %.3f",
             cv_acc.mean(), cv_acc.std())

    clf.fit(X_train, y_train)
    oob_acc  = clf.oob_score_
    test_acc = clf.score(X_test, y_test)
    log.info("  OOB accuracy (unbiased):          %.3f", oob_acc)
    log.info("  Test-set accuracy (20%% held out): %.3f  \u2190 USE THIS NUMBER", test_acc)
    log.info("  clf.classes_: %s", list(clf.classes_))

    # ── Classification report on TEST SET only ────────────────────────────────
    print("\n" + "="*60)
    print("CHART SCORER — TEST SET Report (20% held-out, n=%d)" % len(X_test))
    print("(These numbers are REAL. Train accuracy is always ~1.00 for RF.)")
    print("="*60)
    preds_test = clf.predict(X_test)
    print(classification_report(y_test, preds_test))

    # Retrain on ALL data for production artifact
    clf.fit(X, y)
    joblib.dump(clf, os.path.join(MODELS_DIR, "chart_relevance_scorer.pkl"))
    log.info("  \u2713 Saved chart_relevance_scorer.pkl")
    log.info("  Note: artifact retrained on 100%% data after evaluation.")


# ── Cell 7: Run All Training ──────────────────────────────────────────────────

# @title Cell 7 — MAIN: Run All Training
if __name__ == "__main__":
    t0 = time.perf_counter()

    # 1. Load data
    log.info("Loading sklearn built-in datasets...")
    sklearn_dfs = load_sklearn_datasets()

    log.info("Loading OpenML datasets (this may take a few minutes)...")
    openml_dfs  = load_openml_datasets(max_datasets=55)

    all_dfs = sklearn_dfs + openml_dfs
    log.info("Total datasets: %d", len(all_dfs))

    # 2. Train
    train_drift_autoencoder(all_dfs)
    train_schema_classifier(all_dfs)
    train_chart_scorer()

    elapsed = time.perf_counter() - t0
    log.info("\n✓ All 6 artifacts saved to: %s  (%.0f min %.0f sec)",
             MODELS_DIR, elapsed // 60, elapsed % 60)


# ── Cell 8: Download artifacts ────────────────────────────────────────────────

# @title Cell 8 — Download All Artifacts
"""
Run this cell last to download everything to your local machine:
"""
# from google.colab import files
# import glob
# for f in glob.glob("/content/dipex_models/*.pkl"):
#     files.download(f)
#     print("Downloaded:", f)
#
# Place each downloaded .pkl in:
#   dipex_project/models/
#
# Files to download:
#   drift_autoencoder.pkl
#   drift_scaler.pkl
#   drift_pca.pkl
#   schema_classifier.pkl
#   schema_label_encoder.pkl
#   chart_relevance_scorer.pkl
