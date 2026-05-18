# @title DIPEX — Train Chart Relevance Scorer
# =========================================================================
# Standalone script to train the Chart Relevance Scorer.
# =========================================================================

import os
import time
import logging
import warnings
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dipex_chart_trainer")

MODELS_DIR = "/content/dipex_models"
os.makedirs(MODELS_DIR, exist_ok=True)
RNG = np.random.default_rng(2025)

def train_chart():
    log.info("Training Chart Relevance Scorer...")
    TYPES = ["bar", "line", "scatter", "heatmap", "histogram", "box", "pie"]
    X_list, y_list = [], []
    
    for t in TYPES:
        for _ in range(600): 
            n_rows, n_cols = RNG.integers(30, 1000), RNG.integers(2, 20)
            num_ratio = RNG.uniform(0.1, 1.0)
            feats = [
                min(n_rows/10000, 1.0), min(n_cols/50, 1.0), 
                num_ratio, 1-num_ratio,
                RNG.uniform(0, 0.5), # cat card
                RNG.uniform(-2, 2),  # skew
                RNG.uniform(0, 1),   # mean_corr
                RNG.uniform(0, 0.5), # null_rate
                float(RNG.random() > 0.5), # has dt
                RNG.uniform(0, 1)    # intent_enc
            ]
            X_list.append(feats)
            y_list.append(t)
            
    X = np.array(X_list, dtype=np.float32)
    X += RNG.normal(0, 0.01, X.shape) 
    y = np.array(y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=6,
                                 n_jobs=-1, random_state=42, class_weight="balanced", oob_score=True)
    
    cv_acc = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")
    log.info(f" CV Accuracy:       {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    
    clf.fit(X_train, y_train)
    log.info(f" OOB Accuracy:      {clf.oob_score_:.3f}")
    log.info(f" Test Accuracy:     {clf.score(X_test, y_test):.3f}")
    
    clf.fit(X, y)
    joblib.dump(clf, os.path.join(MODELS_DIR, "chart_relevance_scorer.pkl"))
    log.info("Saved models to chart_relevance_scorer.pkl")

if __name__ == "__main__":
    train_chart()
