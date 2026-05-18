# @title DIPEX — Train Schema Semantic-Type Classifier
# =========================================================================
# standalone script to train the Schema Semantic-Type Classifier.
# =========================================================================

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dipex_schema_trainer")

MODELS_DIR = "/content/dipex_models"
os.makedirs(MODELS_DIR, exist_ok=True)
RNG = np.random.default_rng(2025)

SEMANTIC_LABELS = [
    "id", "age", "amount", "date", "category", "text", "phone", "email", 
    "boolean", "zipcode", "percentage", "score", "count", "name", "unknown",
    "url", "ip_address", "coordinates", "duration", "address", "currency_code"
]

def robust_extract_features(s: pd.Series):
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
        float(s.isnull().mean()),                                      
        float(s_clean.nunique() / n_total),                            
        float(is_num), float(is_str), 
        float(pd.api.types.is_datetime64_any_dtype(s)),                
        float(num_vals.mean() if len(num_vals) else 0.0),              
        float(num_vals.std() if len(num_vals) else 0.0),               
        float(num_vals.min() if len(num_vals) else 0.0),               
        float(num_vals.max() if len(num_vals) else 0.0),               
        float(num_vals.skew() if len(num_vals)>3 else 0.0),            
        all_int,                                                       
        float(num_vals.max() < 200 if len(num_vals) else 0.0),         
        float(num_vals.max() <= 1.0 if len(num_vals) else 0.0),        
        float((num_vals >= 0).all() if len(num_vals) else 0.0),        
        float(s_clean.nunique()),                                      
        float(str_vals.str.contains(r"@.*\.", na=False).mean() if len(str_vals) else 0), 
        float(str_vals.str.contains(r"^\+?\d[\d\s\-()]{7,}$", na=False).mean() if len(str_vals) else 0), 
        float(str_vals.str.len().mean() if len(str_vals) else 0),      
        float(s_clean.nunique()/n_total > 0.9),                        
        float(s_clean.nunique()/n_total < 0.05),                       
        float(str_vals.str.contains(r"https?://|www\.", na=False).mean() if len(str_vals) else 0), 
        float(str_vals.str.match(r"^(\d{1,3}\.){3}\d{1,3}$", na=False).mean() if len(str_vals) else 0), 
        float(((num_vals >= -180) & (num_vals <= 180)).all() if len(num_vals) else 0), 
        float((num_vals % 1 != 0).mean() > 0.8 if len(num_vals) else 0), 
        float(str_vals.str.match(r"^[A-Z]{3}$", na=False).mean() > 0.7 if len(str_vals) else 0) 
    ]

def _make_diverse_series(label: str, n: int) -> pd.Series:
    null_p = RNG.uniform(0.0, 0.35)
    n_use  = n
    def _null(s: pd.Series) -> pd.Series:
        s = s.copy()
        if null_p > 0:
            idx = RNG.choice(len(s), max(1, int(len(s) * null_p)), replace=False)
            s.iloc[idx] = np.nan
        return s
    if label == "id": return _null(pd.Series(np.arange(10000, 10000 + n_use)))
    elif label == "age": return _null(pd.Series(RNG.normal(35, 12, n_use).clip(0, 110)))
    elif label == "amount": return _null(pd.Series(RNG.normal(RNG.uniform(-1e5, 1e5), RNG.uniform(100, 1e4), n_use)))
    elif label == "date": return _null(pd.Series(pd.date_range("2010-01-01", periods=n_use, freq="D").strftime("%Y-%m-%d")))
    elif label == "category": return _null(pd.Series(RNG.choice(["Cat_A", "Cat_B", "Cat_C", "Cat_D"], n_use)))
    elif label == "text": return _null(pd.Series([" ".join(RNG.choice(["lorem", "ipsum", "dolor", "sit"], 5).tolist()) for _ in range(n_use)]))
    elif label == "phone": return _null(pd.Series([f"+1-{RNG.integers(200,999)}-0000" for _ in range(n_use)]))
    elif label == "email": return _null(pd.Series([f"user{RNG.integers(0, 999)}@gmail.com" for _ in range(n_use)]))
    elif label == "boolean": return _null(pd.Series(RNG.choice([True, False], n_use)))
    elif label == "zipcode": return _null(pd.Series(RNG.integers(10000, 99999, n_use)))
    elif label == "percentage": return _null(pd.Series(RNG.uniform(0, 1, n_use).astype(float)))
    elif label == "score": return _null(pd.Series(RNG.normal(50, 15, n_use).clip(0, 100)))
    elif label == "count": return _null(pd.Series(RNG.poisson(RNG.uniform(1, 100), n_use)))
    elif label == "name": return _null(pd.Series([f"{RNG.choice(['Alice','Bob','Charlie'])} Smith" for _ in range(n_use)]))
    elif label == "url": return _null(pd.Series([f"https://example.com/page{RNG.integers(1,100)}" for _ in range(n_use)]))
    elif label == "ip_address": return _null(pd.Series([f"192.168.1.{RNG.integers(1,254)}" for _ in range(n_use)]))
    elif label == "coordinates": return _null(pd.Series(RNG.uniform(-90, 90, n_use).round(5)))
    elif label == "duration": return _null(pd.Series(RNG.integers(0, 7200, n_use).astype(float)))
    elif label == "address": return _null(pd.Series([f"{RNG.integers(1, 9999)} Main St, London" for _ in range(n_use)]))
    elif label == "currency_code": return _null(pd.Series(RNG.choice(["USD", "EUR", "GBP", "JPY", "INR"], n_use)))
    elif label == "unknown": return _null(pd.Series(RNG.normal(0, 1e8, n_use)))
    return _null(pd.Series(RNG.normal(0, 1, n_use)))

def train_schema():
    log.info("Training Schema Semantic Classifier...")
    X_list, y_list = [], []
    for lbl in SEMANTIC_LABELS:
        for _ in range(500):
            try:
                s = _make_diverse_series(lbl, RNG.integers(50, 400))
                X_list.append(robust_extract_features(s))
                y_list.append(lbl)
            except: pass
            
    X = np.nan_to_num(np.array(X_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_list)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.20, stratify=y_enc, random_state=42)
    clf = RandomForestClassifier(n_estimators=400, max_depth=16, min_samples_leaf=4, 
                                 n_jobs=-1, random_state=42, class_weight="balanced", oob_score=True)
    
    cv_acc = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")
    log.info(f" CV Accuracy:       {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    
    clf.fit(X_train, y_train)
    log.info(f" OOB Accuracy:      {clf.oob_score_:.3f}")
    log.info(f" Test Accuracy:     {clf.score(X_test, y_test):.3f}")
    
    clf.fit(X, y_enc)
    joblib.dump(clf, os.path.join(MODELS_DIR, "schema_classifier.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "schema_label_encoder.pkl"))
    log.info("Saved models to schema_classifier.pkl, schema_label_encoder.pkl")

if __name__ == "__main__":
    train_schema()
