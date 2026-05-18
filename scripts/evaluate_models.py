import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Let's import the specific training scripts we have so we can test them exactly as they are built.
try:
    from scripts.colab_train_schema_classifier import robust_extract_features, _make_diverse_series, SEMANTIC_LABELS, RNG
except ImportError:
    print("Could not load schema classifier")
try:
    from scripts.colab_train_domain_classifier import generate_domain_data
except ImportError:
    print("Could not load domain classifier")

def evaluate_schema_classifier():
    print("\n====================")
    print("Evaluating Schema Classifier (Random Forest)")
    
    # 1. Generate messy testing data strictly separated from training
    X_list, y_list = [], []
    for lbl in SEMANTIC_LABELS:
        for _ in range(100): # 100 test samples per class
            try:
                # _make_diverse_series automatically injects heavy NaNs and messy data at random intervals
                s = _make_diverse_series(lbl, RNG.integers(50, 400))
                X_list.append(robust_extract_features(s))
                y_list.append(lbl)
            except: pass
            
    X_test = np.nan_to_num(np.array(X_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_test = np.array(y_list)
    
    clf = joblib.load('models/schema_classifier.pkl')
    le = joblib.load('models/schema_label_encoder.pkl')
    
    y_test_enc = le.transform(y_test)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    
    acc = accuracy_score(y_test_enc, preds)
    f1 = f1_score(y_test_enc, preds, average='weighted')
    
    # AUROC for multiclass
    try:
        auc = roc_auc_score(y_test_enc, probs, multi_class='ovr', average='weighted')
    except:
        auc = 0.0 # fallback if classes are missing in test split
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Weighted F1:   {f1:.4f}")
    print(f"Weighted AUROC:  {auc:.4f}")

    print("\nParameters:")
    print(f"- Trees: {clf.n_estimators}")
    print(f"- Max Depth: {clf.max_depth}")
    try:
        print(f"- Out-of-Bag (OOB) Training Score: {clf.oob_score_:.4f}")
    except AttributeError:
        pass

def evaluate_domain_classifier():
    print("\n====================")
    print("Evaluating Domain Classifier (TF-IDF + Random Forest)")
    
    # 1. Generate strictly messy test data
    X_test, y_test = generate_domain_data(n_samples=500)
    
    pipeline = joblib.load('models/domain_classifier.pkl')
    
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    
    try:
        auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
        
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Weighted F1:   {f1:.4f}")
    print(f"Weighted AUROC:  {auc:.4f}")
    
    rf = pipeline.named_steps['clf']
    print("\nParameters:")
    print(f"- TF-IDF Features: {len(pipeline.named_steps['tfidf'].vocabulary_)}")
    print(f"- Trees: {rf.n_estimators}")
    print(f"- Max Depth: {rf.max_depth}")

def get_anomaly_parameters():
    print("\n====================")
    print("Anomaly Detector (Isolation Forest)")
    
    pipeline = joblib.load('models/anomaly_detector.pkl')
    iforest = pipeline.named_steps['iforest']
    
    print("\nParameters:")
    print(f"- Trees (n_estimators): {iforest.n_estimators}")
    print(f"- Target Contamination (Prior): {iforest.contamination}")
    print(f"- Max Samples per Tree: {iforest.max_samples}")
    print("- Note: AUROC/Accuracy is not applicable here as Isolation Forest is an unsupervised algorithm.")

if __name__ == '__main__':
    evaluate_schema_classifier()
    evaluate_domain_classifier()
    get_anomaly_parameters()
    print("\n====================")
