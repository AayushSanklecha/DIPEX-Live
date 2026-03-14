"""
scripts/train_all_ml_artifacts.py
===================================
Master training script — trains all 3 missing ML model artifacts on
real-world messy tabular data and saves them to models/.

Artifacts produced
------------------
  models/drift_autoencoder.pkl    — MLPRegressor autoencoder (PCA-space)
  models/drift_scaler.pkl         — StandardScaler metadata (n_features_in_=15)
  models/drift_pca.pkl            — PCA(n_components=12) for portable embedding
  models/schema_classifier.pkl    — RandomForestClassifier for semantic types
  models/schema_label_encoder.pkl — LabelEncoder for semantic type labels
  models/chart_relevance_scorer.pkl — RandomForestClassifier for chart ranking

Run from project root:
    python scripts/train_all_ml_artifacts.py

All three trainers use deliberately messy data:
  - random NaN injection (5-20 % per column)
  - random outlier injection (1-5 % of rows, 5-10× IQR)
  - mixed scales, skew, correlated noise
  - imbalanced classes (semantic types are naturally imbalanced)
"""

from __future__ import annotations

import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("train_artifacts")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.join(os.path.dirname(__file__), "..")
MODELS    = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)
sys.path.insert(0, ROOT)

RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inject_messiness(
    X: np.ndarray,
    null_frac: float = 0.10,
    outlier_frac: float = 0.03,
) -> np.ndarray:
    """
    Add realistic messiness to a numeric array in-place:
      - Random NaN injection (null_frac of all cells)
      - Outlier injection (outlier_frac of rows get one column ×8)
    """
    X = X.astype(float).copy()
    n, m = X.shape

    # NaN injection
    nan_mask = RNG.random((n, m)) < null_frac
    X[nan_mask] = np.nan

    # Outlier injection (hard outliers — 5–10× IQR)
    n_out = max(1, int(n * outlier_frac))
    out_rows = RNG.choice(n, n_out, replace=False)
    out_cols = RNG.integers(0, m, n_out)
    for r, c in zip(out_rows, out_cols):
        X[r, c] = (RNG.random() * 10 + 5) * np.nanstd(X[:, c])

    return X


def _load_sklearn_datasets() -> list[pd.DataFrame]:
    """Return a list of DataFrames from sklearn bundled datasets."""
    from sklearn import datasets as skds

    loaders = [
        skds.load_iris,
        skds.load_wine,
        skds.load_breast_cancer,
        skds.load_diabetes,
        skds.fetch_california_housing,
    ]
    dfs: list[pd.DataFrame] = []
    for loader in loaders:
        try:
            bunch = loader()
            df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
            dfs.append(df)
            log.info("  Loaded sklearn %-30s  shape=%s", loader.__name__, df.shape)
        except Exception as exc:
            log.warning("  %s failed: %s", loader.__name__, exc)
    return dfs


def _load_openml_datasets() -> list[pd.DataFrame]:
    """Return DataFrames from a curated list of OpenML datasets (numeric cols only)."""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        return []

    targets = [
        ("titanic",       "1"),
        ("credit-g",      "31"),
        ("adult",         "1590"),
        ("bank-marketing","1461"),
        ("diabetes",      "37"),
        ("steel-plates-fault", "40982"),
        ("electricity",   "44156"),
        ("wine-quality-red","40691"),
    ]
    dfs: list[pd.DataFrame] = []
    for name, did in targets:
        try:
            bunch = fetch_openml(data_id=int(did), as_frame=True, parser="auto")
            num = bunch.data.select_dtypes(include="number")
            if num.shape[1] >= 2:
                dfs.append(num)
                log.info("  OpenML %-30s  shape=%s", name, num.shape)
        except Exception as exc:
            log.warning("  OpenML %s skipped: %s", name, exc)
    return dfs


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DRIFT AUTOENCODER
# ═════════════════════════════════════════════════════════════════════════════

N_DRIFT_FEATURES = 15  # fixed-width for the PCA projection


def _build_drift_corpus(dfs: list[pd.DataFrame]) -> np.ndarray:
    """
    Combine all datasets into a single messy corpus for autoencoder training.
    Each dataset is:
      1. Standardised column-wise (mean=0, std=1) — makes training scale-invariant
      2. Zero-padded / truncated to N_DRIFT_FEATURES columns
      3. Messiness injected
    """
    from sklearn.preprocessing import StandardScaler

    blocks: list[np.ndarray] = []
    for df in dfs:
        num = df.select_dtypes(include="number").dropna(axis=1, how="all")
        if num.shape[1] < 2:
            continue
        arr = num.values.astype(float)
        n, m = arr.shape

        # Fill existing NaNs with column median before standardising
        for j in range(m):
            col_med = np.nanmedian(arr[:, j])
            arr[np.isnan(arr[:, j]), j] = col_med if not np.isnan(col_med) else 0.0

        # Column-wise standardise
        sc = StandardScaler()
        arr = sc.fit_transform(arr)

        # Inject realistic messiness (NaN → fill 0 for network)
        arr = _inject_messiness(arr, null_frac=0.12, outlier_frac=0.04)
        arr = np.nan_to_num(arr, nan=0.0, posinf=3.0, neginf=-3.0)
        arr = np.clip(arr, -5, 5)

        # Pad / truncate to fixed width
        if m < N_DRIFT_FEATURES:
            pad = np.zeros((n, N_DRIFT_FEATURES - m))
            arr = np.hstack([arr, pad])
        else:
            arr = arr[:, :N_DRIFT_FEATURES]

        blocks.append(arr)

    corpus = np.vstack(blocks)
    RNG.shuffle(corpus)
    log.info("Drift corpus: %d rows × %d features", *corpus.shape)
    return corpus.astype(np.float32)


def train_drift_autoencoder(dfs: list[pd.DataFrame]) -> None:
    """
    Train: StandardScaler → PCA(n_components=12) → MLPRegressor autoencoder.
    Separate local StandardScaler is fit at inference time (on baseline);
    the PCA and autoencoder are pre-trained for distribution-agnostic embedding.
    """
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor

    log.info("\n=== [1/3] Drift Autoencoder ===")
    corpus = _build_drift_corpus(dfs)

    # Step 1: Scaler over the flattened corpus (carries n_features_in_=15)
    scaler = StandardScaler()
    corpus_scaled = scaler.fit_transform(corpus)
    log.info("  StandardScaler fit: %d cols", scaler.n_features_in_)

    # Step 2: PCA — reduce to 12 principal components
    n_pca = min(12, N_DRIFT_FEATURES, corpus_scaled.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    corpus_pca = pca.fit_transform(corpus_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    log.info("  PCA(%d) explains %.1f%% variance", n_pca, var_explained * 100)

    # Step 3: MLP Autoencoder in PCA space (n_pca → bottleneck → n_pca)
    bottleneck = max(n_pca // 2, 4)
    ae = MLPRegressor(
        hidden_layer_sizes=(n_pca, bottleneck, n_pca),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )
    ae.fit(corpus_pca, corpus_pca)
    recon_err = np.mean(np.square(corpus_pca - ae.predict(corpus_pca)))
    log.info("  Autoencoder MSE (train): %.6f", recon_err)

    # Save
    joblib.dump(ae,     os.path.join(MODELS, "drift_autoencoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS, "drift_scaler.pkl"))
    joblib.dump(pca,    os.path.join(MODELS, "drift_pca.pkl"))
    log.info("  Saved drift_autoencoder.pkl, drift_scaler.pkl, drift_pca.pkl")


# ═════════════════════════════════════════════════════════════════════════════
# 2.  SCHEMA (SEMANTIC TYPE) CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════

SEMANTIC_LABELS = [
    "id", "age", "amount", "date", "category", "text",
    "phone", "email", "boolean", "zipcode", "percentage",
    "score", "count", "name", "unknown",
]

N_PER_CLASS = 350  # synthetic samples per label


def _gen_series(label: str) -> list[pd.Series]:
    """Generate a batch of synthetic pd.Series that clearly represent each label."""
    series_list: list[pd.Series] = []

    def make(arr, null_p=0.05):
        s = pd.Series(arr, dtype=object if isinstance(arr[0], str) else None)
        if null_p > 0:
            s[RNG.random(len(s)) < null_p] = np.nan
        return s

    n = N_PER_CLASS
    if label == "id":
        # high cardinality integers or UUIDs
        series_list.append(make(RNG.integers(100000, 999999, n)))
        series_list.append(make([f"ID-{i:06d}" for i in RNG.integers(0, 999999, n)]))

    elif label == "age":
        # integers 0-110, sometimes with outliers
        series_list.append(make(RNG.integers(0, 100, n)))
        arr = RNG.integers(18, 65, n).astype(float)
        arr[RNG.choice(n, 5)] = RNG.integers(110, 150, 5)  # outliers
        series_list.append(make(arr, null_p=0.08))

    elif label == "amount":
        # monetary: right-skewed floats, large range
        series_list.append(make(RNG.exponential(1000, n)))
        series_list.append(make(RNG.lognormal(6, 2, n)))
        series_list.append(make(-1 * RNG.exponential(500, n)))  # negative amounts

    elif label == "date":
        # date strings
        base = pd.date_range("2015-01-01", periods=n, freq="6h")
        series_list.append(pd.Series(base.astype(str)))
        series_list.append(pd.Series(base.strftime("%Y/%m/%d")))

    elif label == "category":
        # low cardinality strings (3-10 distinct values)
        cats = ["A", "B", "C", "D", "E"]
        series_list.append(make(RNG.choice(cats, n)))
        cats2 = ["male", "female", "other"]
        series_list.append(make(RNG.choice(cats2, n)))
        cats3 = ["active", "inactive", "pending", "churned"]
        series_list.append(make(RNG.choice(cats3, n)))

    elif label == "text":
        # high-cardinality long strings
        words = ["lorem", "ipsum", "dolor", "amet", "consectetur", "adipiscing"]
        series_list.append(make(
            [" ".join(RNG.choice(words, RNG.integers(5, 20))) for _ in range(n)]
        ))

    elif label == "phone":
        series_list.append(make(
            [f"+1-{RNG.integers(200,999)}-{RNG.integers(100,999)}-{RNG.integers(1000,9999)}"
             for _ in range(n)]
        ))
        series_list.append(make(
            [f"({RNG.integers(200,999)}) {RNG.integers(100,999)}-{RNG.integers(1000,9999)}"
             for _ in range(n)]
        ))

    elif label == "email":
        domains = ["gmail.com", "yahoo.com", "work.org", "company.net", "uni.edu"]
        series_list.append(make(
            [f"user{RNG.integers(0,99999)}@{RNG.choice(domains)}" for _ in range(n)]
        ))

    elif label == "boolean":
        series_list.append(make(RNG.integers(0, 2, n)))
        series_list.append(make(RNG.choice([True, False], n)))
        series_list.append(make(RNG.choice(["yes", "no", "true", "false"], n)))

    elif label == "zipcode":
        series_list.append(make([f"{RNG.integers(10000,99999)}" for _ in range(n)]))
        series_list.append(make(RNG.integers(10000, 99999, n)))

    elif label == "percentage":
        series_list.append(make(RNG.uniform(0, 1, n)))     # 0-1 scale
        series_list.append(make(RNG.uniform(0, 100, n)))   # 0-100 scale

    elif label == "score":
        series_list.append(make(RNG.uniform(0, 10, n)))
        series_list.append(make(RNG.integers(1, 6, n)))    # 1-5 rating

    elif label == "count":
        series_list.append(make(RNG.integers(0, 1000, n)))
        series_list.append(make(RNG.poisson(50, n)))

    elif label == "name":
        first_names = ["Alice", "Bob", "Carlos", "Diana", "Eva", "Frank",
                       "Grace", "Hector", "Iris", "Jack", "Kai", "Lena"]
        last_names  = ["Smith", "Jones", "Kumar", "Lee", "Patel", "Brown",
                       "Wilson", "Garcia", "Nguyen", "Kim", "Chen", "Sharma"]
        series_list.append(make(
            [f"{RNG.choice(first_names)} {RNG.choice(last_names)}" for _ in range(n)]
        ))
        series_list.append(make([RNG.choice(first_names) for _ in range(n)]))

    elif label == "unknown":
        # Mixed garbage columns
        series_list.append(make(RNG.bytes(n).hex()[:n]))
        series_list.append(make([f"X{RNG.integers(0,9999)}" for _ in range(n)]))

    return series_list


def train_schema_classifier(dfs: list[pd.DataFrame]) -> None:
    """
    Build a LabelEncoder + RandomForestClassifier that maps column-level
    feature vectors (20 features) to semantic type labels.
    Training data = synthetic columns per label + real sklearn dataset columns.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    # Import project's own feature extractor
    from ingestion.schema_infer import _extract_column_features

    log.info("\n=== [2/3] Schema Semantic-Type Classifier ===")

    _FEAT_ORDER = [
        "null_rate", "unique_rate", "is_numeric", "is_string",
        "is_datetime", "mean_val", "std_val", "min_val", "max_val",
        "skew_val", "all_integer", "max_lt_200", "max_lt_1",
        "all_positive", "n_distinct", "email_pattern", "phone_pattern",
        "mean_str_len", "high_cardinality", "low_cardinality",
    ]

    rows: list[list[float]] = []
    labels: list[str] = []

    # ── Synthetic samples ─────────────────────────────────────────────────────
    for lbl in SEMANTIC_LABELS:
        series_batch = _gen_series(lbl)
        for s in series_batch:
            try:
                feats = _extract_column_features(s, col_name=lbl)
                rows.append([feats[k] for k in _FEAT_ORDER])
                labels.append(lbl)
            except Exception:
                pass

    # ── Columns from real sklearn datasets (hand-labeled) ────────────────────
    real_label_hints: dict[str, str] = {
        # iris / wine / breast_cancer
        "sepal length (cm)": "amount", "sepal width (cm)":  "amount",
        "petal length (cm)": "amount", "petal width (cm)":  "amount",
        "alcohol":           "percentage", "malic_acid":     "amount",
        "mean radius":       "amount",    "mean texture":    "amount",
        "mean smoothness":   "percentage","mean symmetry":   "percentage",
        "mean fractal dimension": "percentage",
        # california housing
        "MedInc":   "amount",  "HouseAge": "age",  "AveRooms": "count",
        "AveBedrms":"count",   "Population":"count","AveOccup": "count",
        "Latitude":  "score",  "Longitude": "score",
        # diabetes
        "age": "age", "bmi": "score", "bp": "score",
    }
    for df in dfs:
        num = df.select_dtypes(include="number")
        for col in num.columns:
            lbl = real_label_hints.get(col)
            if lbl is None:
                continue
            try:
                feats = _extract_column_features(num[col], col_name=col)
                rows.append([feats[k] for k in _FEAT_ORDER])
                labels.append(lbl)
            except Exception:
                pass

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels)
    log.info("  Total training samples: %d across %d classes", len(X), len(np.unique(y)))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    # Quick cross-val check
    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
    log.info("  5-fold CV accuracy: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    clf.fit(X, y_enc)
    log.info("  Train accuracy: %.3f", clf.score(X, y_enc))

    joblib.dump(clf, os.path.join(MODELS, "schema_classifier.pkl"))
    joblib.dump(le,  os.path.join(MODELS, "schema_label_encoder.pkl"))
    log.info("  Saved schema_classifier.pkl, schema_label_encoder.pkl")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  CHART RELEVANCE SCORER
# ═════════════════════════════════════════════════════════════════════════════

CHART_TYPES = ["bar", "line", "scatter", "heatmap", "histogram", "box", "pie"]


def _make_chart_dataset(chart_type: str, n: int = 180) -> list[tuple[pd.DataFrame, str]]:
    """
    Generate synthetic DataFrames that ideally suit a given chart type.
    Returns [(df, query_intent), ...].
    """
    samples: list[tuple[pd.DataFrame, str]] = []

    for _ in range(n):
        if chart_type == "line":
            # Time-series-like: datetime index, a few numeric series
            n_rows = RNG.integers(50, 500)
            df = pd.DataFrame({
                "date":  pd.date_range("2020-01-01", periods=n_rows, freq="D"),
                "value": RNG.normal(100, 15, n_rows).cumsum(),
                "trend": np.linspace(0, 50, n_rows) + RNG.normal(0, 5, n_rows),
            })
            intent = RNG.choice(["trend", "time_series"])

        elif chart_type == "bar":
            # Low-cardinality categorical + numeric aggregates
            cats = [f"Cat{i}" for i in range(RNG.integers(3, 10))]
            n_rows = RNG.integers(50, 300)
            df = pd.DataFrame({
                "category": RNG.choice(cats, n_rows),
                "value":    RNG.exponential(1000, n_rows),
                "count":    RNG.integers(1, 50, n_rows),
            })
            intent = RNG.choice(["compare", "group_by", "top_n"])

        elif chart_type == "scatter":
            # Two or more correlated numeric columns, no date
            n_rows = RNG.integers(50, 400)
            x = RNG.normal(0, 1, n_rows)
            df = pd.DataFrame({
                "feature_a": x,
                "feature_b": x * RNG.uniform(0.5, 1.5) + RNG.normal(0, 0.3, n_rows),
                "feature_c": RNG.normal(0, 1, n_rows),
            })
            intent = "correlation"

        elif chart_type == "heatmap":
            # Wide table, many numeric columns, high correlation density
            n_rows = RNG.integers(50, 200)
            n_cols = RNG.integers(5, 15)
            base = RNG.normal(0, 1, (n_rows, n_cols))
            # Introduce correlated structure
            base += RNG.normal(0, 0.2, (n_rows, 1))
            df = pd.DataFrame(base, columns=[f"var_{i}" for i in range(n_cols)])
            intent = "correlation"

        elif chart_type == "histogram":
            # Mostly numeric, skewed distributions — no date, no cat
            n_rows = RNG.integers(100, 500)
            df = pd.DataFrame({
                "value":   RNG.exponential(500, n_rows),
                "score":   RNG.normal(50, 15, n_rows),
            })
            intent = RNG.choice(["distribution", "distribute"])

        elif chart_type == "box":
            # Mix of categorical grouping + numeric values
            cats = [f"Group{i}" for i in range(RNG.integers(2, 6))]
            n_rows = RNG.integers(50, 400)
            df = pd.DataFrame({
                "group":  RNG.choice(cats, n_rows),
                "metric": RNG.normal(0, 1, n_rows) + RNG.choice([0, 2, 4], n_rows),
                "extra":  RNG.uniform(0, 1, n_rows),
            })
            intent = RNG.choice(["distribution", "compare"])

        elif chart_type == "pie":
            # Very low-cardinality categorical — 2-6 exclusive groups
            cats = [f"Slice{i}" for i in range(RNG.integers(2, 7))]
            n_rows = RNG.integers(30, 200)
            df = pd.DataFrame({
                "segment": RNG.choice(cats, n_rows),
                "revenue": RNG.exponential(1000, n_rows),
            })
            intent = RNG.choice(["aggregate", "general"])

        else:
            continue

        # Inject messiness
        for col in df.select_dtypes(include="number").columns:
            mask = RNG.random(len(df)) < 0.07
            df.loc[mask, col] = np.nan

        samples.append((df, str(intent)))

    return samples


def train_chart_scorer(dfs: list[pd.DataFrame]) -> None:
    """
    Train a RandomForestClassifier to map (DataFrame features, intent) → best chart type.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    from reporting_service.chart_relevance_scorer import _extract_features as _chart_feats

    log.info("\n=== [3/3] Chart Relevance Scorer ===")

    X_rows: list[np.ndarray] = []
    y_labels: list[str] = []

    for chart_type in CHART_TYPES:
        samples = _make_chart_dataset(chart_type, n=180)
        for df, intent in samples:
            try:
                feat = _chart_feats(df, query_intent=intent)
                X_rows.append(feat)
                y_labels.append(chart_type)
            except Exception:
                pass

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels)
    log.info("  Total training samples: %d across %d chart types", len(X), len(CHART_TYPES))

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    log.info("  5-fold CV accuracy: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    clf.fit(X, y)
    log.info("  Train accuracy: %.3f", clf.score(X, y))

    joblib.dump(clf, os.path.join(MODELS, "chart_relevance_scorer.pkl"))
    log.info("  Saved chart_relevance_scorer.pkl")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    t0 = time.perf_counter()

    log.info("Loading real-world datasets…")
    dfs = _load_sklearn_datasets()
    dfs += _load_openml_datasets()
    log.info("Total datasets loaded: %d", len(dfs))

    train_drift_autoencoder(dfs)
    train_schema_classifier(dfs)
    train_chart_scorer(dfs)

    elapsed = time.perf_counter() - t0
    log.info("\n✓ All artifacts saved to: %s  (%.1fs)", MODELS, elapsed)
    log.info("  drift_autoencoder.pkl   drift_scaler.pkl   drift_pca.pkl")
    log.info("  schema_classifier.pkl   schema_label_encoder.pkl")
    log.info("  chart_relevance_scorer.pkl")
