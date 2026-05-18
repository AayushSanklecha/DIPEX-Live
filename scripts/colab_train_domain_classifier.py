# @title DIPEX — Train Data Domain Classifier
# =========================================================================
# Standalone script to train the Data Domain Classifier using NLP.
# Replaces heuristic domain detection in AnalystBrain.
# =========================================================================

import os
import time
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dipex_domain_trainer")

MODELS_DIR = "/content/dipex_models"
os.makedirs(MODELS_DIR, exist_ok=True)
RNG = np.random.default_rng(2025)

DOMAINS = {
    "finance": ["revenue", "price", "stock", "interest", "dividend", "yield", "currency", "credit", "debit", "balance", "trade", "investment"],
    "medical": ["patient", "diagnosis", "blood", "pressure", "heart_rate", "dose", "symptom", "treatment", "doctor", "hospital", "mortality"],
    "hr": ["employee", "salary", "wage", "department", "hire_date", "attrition", "performance", "bonus", "role", "manager", "absence"],
    "ecommerce": ["cart", "product", "sku", "checkout", "conversion", "session", "click", "order", "return_rate", "shipping", "review"],
    "general": ["id", "name", "date", "status", "count", "value", "type", "description", "location", "category", "index", "code"]
}

def generate_domain_data(n_samples=2000):
    texts, labels = [], []
    for _ in range(n_samples):
        domain = RNG.choice(list(DOMAINS.keys()))
        num_cols = RNG.integers(5, 15)
        # Sample words from the chosen domain, plus some general words/noise
        words = RNG.choice(DOMAINS[domain], min(num_cols, len(DOMAINS[domain])), replace=False).tolist()
        words += RNG.choice(DOMAINS["general"], RNG.integers(1, 4), replace=False).tolist()
        # Create a string representation of the schema
        schema_text = " ".join(words)
        texts.append(schema_text)
        labels.append(domain)
    return texts, labels

def train_domain():
    log.info("Training Data Domain Classifier (NLP)...")
    X, y = generate_domain_data(n_samples=3000)
    
    # NLP Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=500)),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")
    log.info(f" CV Accuracy:       {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    
    pipeline.fit(X_train, y_train)
    log.info(f" Test Accuracy:     {pipeline.score(X_test, y_test):.3f}")
    
    pipeline.fit(X, y)
    joblib.dump(pipeline, os.path.join(MODELS_DIR, "domain_classifier.pkl"))
    log.info("Saved models to domain_classifier.pkl")

if __name__ == "__main__":
    train_domain()
