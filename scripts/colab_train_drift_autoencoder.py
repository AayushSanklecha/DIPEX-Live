# @title DIPEX — Train Drift Autoencoder
# =========================================================================
# Standalone script to train the Multivariate Drift Autoencoder.
# =========================================================================

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dipex_drift_trainer")

MODELS_DIR = "/content/dipex_models"
os.makedirs(MODELS_DIR, exist_ok=True)
RNG = np.random.default_rng(2025)

def inject_real_world_messiness(X: np.ndarray, null_frac=0.15, outlier_frac=0.08) -> np.ndarray:
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

def load_robust_datasets(max_openml=40):
    dfs = []
    try: dfs.append(pd.DataFrame(fetch_california_housing().data))
    except: pass
    try: dfs.append(pd.DataFrame(load_diabetes().data))
    except: pass
    try: dfs.append(pd.DataFrame(load_breast_cancer().data))
    except: pass

    try:
        import openml
        curated_ids = [31, 29, 1590, 1461, 37, 40691, 1510, 4534, 180, 40685, 43, 847, 554, 531, 40981, 40984, 1119, 1489, 41187, 4541]
        for did in curated_ids[:max_openml]:
            try:
                ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False, download_features_meta_data=False)
                X, _, _, _ = ds.get_data(dataset_format="dataframe")
                num = X.select_dtypes(include="number").dropna(axis=1, how='all')
                if num.shape[1] >= 2 and len(num) >= 50:
                    dfs.append(num)
            except: pass
    except ImportError:
        log.warning("openml package not installed. Proceeding with basic datasets.")
    return dfs

def train_drift():
    log.info("Fetching Real-World Data (Fault-Tolerant)")
    dfs = load_robust_datasets()
    log.info(f"Loaded {len(dfs)} datasets for Drift training.")
    
    blocks = []
    N_FEAT = 15
    
    for df in dfs:
        arr = df.values.astype(float)
        if arr.shape[1] < N_FEAT:
            arr = np.pad(arr, ((0,0), (0, N_FEAT - arr.shape[1])))
        else:
            arr = arr[:, :N_FEAT]
            
        arr_clean = np.nan_to_num(StandardScaler().fit_transform(np.nan_to_num(arr)), 0)
        arr_dirty = np.nan_to_num(inject_real_world_messiness(arr_clean, 0.20, 0.15), 0)
        arr_shift = arr_clean * RNG.uniform(1.2, 1.8) + RNG.normal(0, 1)
        
        blocks.extend([arr_clean, arr_dirty, arr_shift])
        
    corpus = np.vstack(blocks)
    RNG.shuffle(corpus)
    corpus = np.clip(corpus, -10, 10)
    
    log.info(f"Constructed Drift Corpus: {corpus.shape[0]:,} rows × {corpus.shape[1]} features")
    
    sc = StandardScaler()
    corpus_scaled = sc.fit_transform(corpus)
    
    pca = PCA(n_components=12, random_state=42)
    corpus_pca = pca.fit_transform(corpus_scaled)
    var = pca.explained_variance_ratio_.sum()
    log.info(f"PCA Variance Explained: {var:.1%}")
    
    ae = MLPRegressor(
        hidden_layer_sizes=(12, 6, 12), activation="relu",
        solver="adam", max_iter=800, learning_rate_init=0.002,
        early_stopping=True, verbose=False, random_state=42
    )
    ae.fit(corpus_pca, corpus_pca)
    mse = float(np.mean(np.square(corpus_pca - ae.predict(corpus_pca))))
    
    log.info(f"Final Autoencoder MSE: {mse:.6f}")
    
    joblib.dump(ae, os.path.join(MODELS_DIR, "drift_autoencoder.pkl"))
    joblib.dump(sc, os.path.join(MODELS_DIR, "drift_scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "drift_pca.pkl"))
    log.info("Saved: drift_autoencoder.pkl, drift_scaler.pkl, drift_pca.pkl")

if __name__ == "__main__":
    train_drift()
