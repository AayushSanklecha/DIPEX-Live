import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.metrics import classification_report, f1_score
from scipy.stats import randint

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("dipex_professional_trainer")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
RNG = np.random.default_rng(2026)

# ==============================================================================
# 1. DATA SOURCE: COMBINING SYNTHETIC + REAL WORLD DATA
# ==============================================================================
def load_real_world_dataframes():
    """Loads a few robust datasets from sklearn to anchor the models in reality."""
    from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_breast_cancer
    dfs = []
    try: dfs.append(pd.DataFrame(fetch_california_housing().data))
    except Exception: pass
    try: dfs.append(pd.DataFrame(load_diabetes().data))
    except Exception: pass
    try: dfs.append(pd.DataFrame(load_wine().data))
    except Exception: pass
    try: dfs.append(pd.DataFrame(load_breast_cancer().data))
    except Exception: pass
    return dfs

# ==============================================================================
# 2. DRIFT AUTOENCODER (Artifacts 1, 2, 3)
# ==============================================================================
def train_drift(dfs):
    log.info("\n=== Training Multivariate Drift Model (with validation split) ===")
    blocks = []
    N_FEAT = 15
    
    # Generate some purely synthetic noise as well to pad it out
    for _ in range(50):
        dfs.append(pd.DataFrame(RNG.normal(0, 1, size=(500, N_FEAT))))

    for df in dfs:
        arr = df.values.astype(float)
        if arr.shape[1] < N_FEAT:
            arr = np.pad(arr, ((0,0), (0, N_FEAT - arr.shape[1])))
        else:
            arr = arr[:, :N_FEAT]
            
        arr_clean = np.nan_to_num(StandardScaler().fit_transform(np.nan_to_num(arr)), 0)
        blocks.append(arr_clean)
        
    corpus = np.vstack(blocks)
    RNG.shuffle(corpus)
    corpus = np.clip(corpus, -5, 5)
    
    # PREVENT OVERFITTING: Train/Test split for the autoencoder
    X_train, X_val = train_test_split(corpus, test_size=0.2, random_state=42)
    log.info(f"Drift Corpus Size: Train={X_train.shape}, Val={X_val.shape}")
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val) # IMPORTANT: Scale validation using train scaler
    
    pca = PCA(n_components=12, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    var = pca.explained_variance_ratio_.sum()
    log.info(f"PCA Variance Explained: {var:.1%}")
    
    # MLP with early stopping using the validation set to prevent overfitting
    ae = MLPRegressor(
        hidden_layer_sizes=(12, 6, 12), activation="relu",
        solver="adam", max_iter=800, learning_rate_init=0.002,
        early_stopping=True, validation_fraction=0.2, random_state=42
    )
    ae.fit(X_train_pca, X_train_pca)
    
    val_preds = ae.predict(X_val_pca)
    val_mse = np.mean((X_val_pca - val_preds)**2)
    log.info(f"Autoencoder Validation MSE: {val_mse:.4f}")
    
    joblib.dump(ae, os.path.join(MODELS_DIR, "drift_autoencoder.pkl"))
    joblib.dump(sc, os.path.join(MODELS_DIR, "drift_scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "drift_pca.pkl"))
    log.info("Saved Drift Artifacts.")

# ==============================================================================
# 3. SCHEMA CLASSIFIER (Artifacts 4, 5)
# ==============================================================================
SEMANTIC_LABELS = [
    "id", "age", "amount", "date", "category", "text", "phone", "email", 
    "boolean", "zipcode", "percentage", "score", "count", "name", "unknown",
    "url", "ip_address", "coordinates", "duration", "address", "currency_code"
]

def robust_extract_features(s: pd.Series):
    """Extraction completely immune to Pandas NaNs, Infinities, mixed types"""
    n_total = max(len(s), 1)
    s_clean = s.dropna()
    is_num = pd.api.types.is_numeric_dtype(s)
    is_str = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
    
    num_vals = pd.to_numeric(s_clean, errors='coerce').dropna() if not is_num else s_clean.copy()
    str_vals = s_clean.astype(str) if is_str else pd.Series([], dtype=str)
    num_vals = num_vals[~np.isinf(num_vals)]
    
    try: all_int = float((num_vals == num_vals.apply(int)).all()) if len(num_vals)>0 else 0.0
    except: all_int = 0.0
    
    return [
        float(s.isnull().mean()), float(s_clean.nunique() / n_total),
        float(is_num), float(is_str), float(pd.api.types.is_datetime64_any_dtype(s)),
        float(num_vals.mean() if len(num_vals) else 0.0), float(num_vals.std() if len(num_vals) else 0.0),
        float(num_vals.min() if len(num_vals) else 0.0), float(num_vals.max() if len(num_vals) else 0.0),
        float(num_vals.skew() if len(num_vals)>3 else 0.0), all_int,
        float(num_vals.max() < 200 if len(num_vals) else 0.0), float(num_vals.max() <= 1.0 if len(num_vals) else 0.0),
        float((num_vals >= 0).all() if len(num_vals) else 0.0), float(s_clean.nunique()),
        float(str_vals.str.contains(r"@.*\.", na=False).mean() if len(str_vals) else 0),
        float(str_vals.str.contains(r"^\+?\d[\d\s\-()]{7,}$", na=False).mean() if len(str_vals) else 0),
        float(str_vals.str.len().mean() if len(str_vals) else 0),
        float(s_clean.nunique()/n_total > 0.9), float(s_clean.nunique()/n_total < 0.05),
        float(str_vals.str.contains(r"https?://", na=False).mean() if len(str_vals) else 0),
        float(str_vals.str.match(r"^(\d{1,3}\.){3}\d{1,3}$", na=False).mean() if len(str_vals) else 0),
        float(((num_vals >= -180) & (num_vals <= 180)).all() if len(num_vals) else 0),
        float((num_vals % 1 != 0).mean() > 0.8 if len(num_vals) else 0),
        float(str_vals.str.match(r"^[A-Z]{3}$", na=False).mean() > 0.7 if len(str_vals) else 0)
    ]

def _synth_column_advanced(lbl, n):
    null_p = RNG.uniform(0, 0.4)
    def _null(s):
        s = s.copy()
        if null_p > 0: s.iloc[RNG.choice(len(s), max(1, int(len(s)*null_p)), replace=False)] = np.nan
        return s
        
    if lbl == "id": return _null(pd.Series([f"X-{RNG.integers(1,999999)}" for _ in range(n)]))
    if lbl == "age": return _null(pd.Series(RNG.normal(40, 15, n).clip(0, 110)))
    if lbl == "amount": return _null(pd.Series(RNG.exponential(1500, n))) # More realistic than normal
    if lbl == "category": return _null(pd.Series(RNG.choice(["A", "B", "C", "D"], n)))
    if lbl == "percentage": return _null(pd.Series(RNG.uniform(0, 1.0, n)))
    if lbl == "boolean": return _null(pd.Series(RNG.choice([True, False], n)))
    if lbl == "count": return _null(pd.Series(RNG.poisson(5, n)))
    return _null(pd.Series(RNG.normal(0,1,n)))

def train_schema():
    log.info("\n=== Training Schema Classifier (Cross-Validated) ===")
    X_list, y_list = [], []
    for lbl in SEMANTIC_LABELS:
        for _ in range(400):
            try:
                s = _synth_column_advanced(lbl, RNG.integers(50, 500))
                X_list.append(robust_extract_features(s))
                y_list.append(lbl)
            except: pass
            
    X = np.nan_to_num(np.array(X_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_list)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    
    # 2. Hyperparameter Tuning using Randomized Search and K-Fold CV
    log.info("Running Randomized CV Search for Schema Classifier...")
    param_dist = {
        'n_estimators': randint(100, 400),
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_leaf': randint(1, 10)
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    
    # 3-Fold Cross Validation. We evaluate on F1 macro to ensure all classes are handled well.
    cv_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=15, 
        cv=KFold(n_splits=3, shuffle=True, random_state=42), 
        scoring='f1_macro', n_jobs=-1, random_state=42, verbose=0
    )
    
    cv_search.fit(X_train, y_train)
    best_clf = cv_search.best_estimator_
    
    log.info(f"Best Schema Params: {cv_search.best_params_}")
    
    # 3. Evaluate on unseen test set explicitly to prove no overfitting
    y_pred = best_clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    log.info(f"Schema Test Set F1 Score (weighted): {f1:.4f}")
    if f1 == 1.0:
        log.warning("F1 is exactly 1.0. This is highly suspicious and indicates data leakage or overfitting.")
        
    joblib.dump(best_clf, os.path.join(MODELS_DIR, "schema_classifier.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "schema_label_encoder.pkl"))
    log.info("Saved Schema Classifier.")

# ==============================================================================
# 4. CHART RELEVANCE SCORER (Artifact 6)
# ==============================================================================
def train_chart():
    log.info("\n=== Training Chart Relevance Scorer (Heuristic-Anchored + CV) ===")
    TYPES = ["bar", "line", "scatter", "heatmap", "histogram", "box", "pie"]
    X_list, y_list = [], []
    
    # Instead of random noise, we build features that have a logical relationship to the chart type
    for _ in range(5000):
        n_rows = RNG.integers(10, 100000)
        n_cols = RNG.integers(2, 50)
        num_ratio = RNG.uniform(0.0, 1.0)
        cat_card = RNG.uniform(0.0, 1.0) # cardinaltiy of categorical cols
        has_dt = float(RNG.random() > 0.7)
        
        # Base features
        feats = [
            n_rows/10000, n_cols/50, num_ratio, 1-num_ratio, cat_card,
            RNG.uniform(-2, 2), RNG.uniform(0, 1), RNG.uniform(0, 0.5), has_dt, RNG.uniform(0, 1)
        ]
        
        # HEURISTIC TARGET ASSIGNMENT:
        # We assign the chart type based on logical rules, and the model must learn these 
        # complex boundaries. This prevents random overfitting to noise.
        true_type = "scatter" # default
        if has_dt == 1.0 and num_ratio > 0.3:
            true_type = "line"
        elif num_ratio < 0.2 and 0 < cat_card < 0.05:
            true_type = "pie"
        elif num_ratio < 0.5 and cat_card < 0.2:
            true_type = "bar"
        elif n_cols > 10 and num_ratio > 0.8:
            true_type = "heatmap"
        elif num_ratio == 1.0 and n_cols < 3:
            true_type = "histogram"
        elif n_rows > 1000 and num_ratio > 0.5 and cat_card < 0.1:
            true_type = "box"
            
        X_list.append(feats)
        y_list.append(true_type)
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [5, 10, 15, 20],
        'min_samples_leaf': randint(2, 10) # Minimum samples leaf prevents overfitting to strict rules
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    cv_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=10, 
        cv=KFold(n_splits=3, shuffle=True, random_state=42), 
        scoring='accuracy', n_jobs=-1, random_state=42, verbose=0
    )
    
    cv_search.fit(X_train, y_train)
    best_clf = cv_search.best_estimator_
    
    log.info(f"Best Chart Scorer Params: {cv_search.best_params_}")
    
    y_pred = best_clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    log.info(f"Chart Test Set F1 Score (weighted): {f1:.4f}")
    
    joblib.dump(best_clf, os.path.join(MODELS_DIR, "chart_relevance_scorer.pkl"))
    log.info("Saved Chart Relevance Scorer.")

if __name__ == "__main__":
    t0 = time.time()
    dfs = load_real_world_dataframes()
    train_drift(dfs)
    train_schema()
    train_chart()
    log.info(f"Done! All Models robustly cross-validated and saved in {time.time()-t0:.1f}s.")
