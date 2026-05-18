# @title DIPEX — Train Multivariate Anomaly Detector
# =========================================================================
# Standalone script to train the Isolation Forest Anomaly Detector.
# Used in Robust Data Triage or Preprocessing to flag corrupted rows.
# =========================================================================

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing, load_diabetes

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dipex_anomaly_trainer")

MODELS_DIR = "/content/dipex_models"
os.makedirs(MODELS_DIR, exist_ok=True)
RNG = np.random.default_rng(2025)

def inject_real_world_messiness(X: np.ndarray, null_frac=0.10, outlier_frac=0.15) -> np.ndarray:
    X = X.astype(float).copy()
    n, m = X.shape
    
    mask_null = RNG.random((n, m)) < null_frac
    X[mask_null] = np.nan
    
    n_outliers = max(1, int(n * outlier_frac))
    for r in RNG.choice(n, n_outliers, replace=False):
        c = int(RNG.integers(0, m))
        if not np.isnan(X[r, c]):
            col_std = np.nanstd(X[:, c]) + 1e-4
            choice = RNG.integers(0, 4)
            if choice == 0: X[r, c] *= 100       
            elif choice == 1: X[r, c] *= -1      
            elif choice == 2: X[r, c] = 0.0      
            else: X[r, c] += RNG.choice([-1, 1]) * col_std * 15 
            
    return X


def train_anomaly_detector():
    log.info("Training Multivariate Anomaly Detector (Isolation Forest)...")
    blocks = []
    N_FEAT = 15
    
    # Load some baseline data
    dfs = []
    try: dfs.append(pd.DataFrame(fetch_california_housing().data))
    except: pass
    try: dfs.append(pd.DataFrame(load_diabetes().data))
    except: pass
    
    # Generate some synthetic data blocks
    for _ in range(50):
        arr = RNG.normal(0, RNG.uniform(1, 10), (RNG.integers(100, 1000), RNG.integers(5, 15)))
        dfs.append(pd.DataFrame(arr))
        
    for df in dfs:
        arr = df.values.astype(float)
        if arr.shape[1] < N_FEAT:
            arr = np.pad(arr, ((0,0), (0, N_FEAT - arr.shape[1])))
        else:
            arr = arr[:, :N_FEAT]
            
        # Standardize for the pipeline
        arr_clean = np.nan_to_num(StandardScaler().fit_transform(np.nan_to_num(arr)), 0)
        arr_dirty = np.nan_to_num(inject_real_world_messiness(arr_clean, 0.10, 0.20), 0)
        
        blocks.extend([arr_clean, arr_dirty])
        
    corpus = np.vstack(blocks)
    RNG.shuffle(corpus)
    corpus = np.clip(corpus, -20, 20)
    
    log.info(f"Constructed Anomaly Corpus: {corpus.shape[0]:,} rows × {corpus.shape[1]} features")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('iforest', IsolationForest(n_estimators=200, contamination=0.10, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(corpus)
    
    # Test on some new dirty data
    test_clean = RNG.normal(0, 1, (1000, N_FEAT))
    test_dirty = inject_real_world_messiness(test_clean, 0.0, 0.50)
    test_dirty = np.nan_to_num(test_dirty, 0)
    
    preds_clean = pipeline.predict(test_clean)
    preds_dirty = pipeline.predict(test_dirty)
    
    clean_anomaly_rate = (preds_clean == -1).mean()
    dirty_anomaly_rate = (preds_dirty == -1).mean()
    
    log.info(f" Baseline Anomaly Rate (Testing Setup): {clean_anomaly_rate:.2%}")
    log.info(f" Corrupted Anomaly Rate (Testing Setup): {dirty_anomaly_rate:.2%}")
    
    joblib.dump(pipeline, os.path.join(MODELS_DIR, "anomaly_detector.pkl"))
    log.info("Saved models to anomaly_detector.pkl")

if __name__ == "__main__":
    train_anomaly_detector()
