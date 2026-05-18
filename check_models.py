"""
check_models.py
Functional smoke-test for all 6 Apr-13 models.
Uses the correct internal structure of each artifact.
Run: python check_models.py
"""
import sys, warnings, json
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import joblib
from pathlib import Path

M = Path('models')
results = {}

# ── 1. Schema Classifier ─────────────────────────────────────────────────────
# Structure: Pipeline(steps=['model'])  +  schema_feature_registry dict
try:
    clf  = joblib.load(M / 'schema_classifier.pkl')    # Pipeline, step='model'
    le   = joblib.load(M / 'schema_label_encoder.pkl') # LabelEncoder
    reg  = joblib.load(M / 'schema_feature_registry.pkl')
    n_f  = reg['n_stat'] + reg['n_nlp']               # total feature count
    X    = np.zeros((1, n_f))
    pred = clf.predict(X)
    label = le.inverse_transform(pred)[0]
    labels = reg.get('schema_labels', [])
    results['schema_classifier'] = f'PASS  ->  n_features={n_f}  predicted="{label}"  n_classes={len(labels)}'
except Exception as e:
    results['schema_classifier'] = f'FAIL  ->  {e}'

# ── 2. Domain Classifier ─────────────────────────────────────────────────────
# Structure: Pipeline(steps=['model'])  +  domain_registry dict
try:
    dom_clf = joblib.load(M / 'domain_classifier.pkl')
    dom_le  = joblib.load(M / 'domain_label_encoder.pkl')
    dom_reg = joblib.load(M / 'domain_registry.pkl')
    n_f2    = dom_reg['n_features']
    X2      = np.zeros((1, n_f2))
    pred2   = dom_clf.predict(X2)
    label2  = dom_le.inverse_transform(pred2)[0]
    domains = dom_reg.get('domain_labels', [])
    results['domain_classifier'] = f'PASS  ->  n_features={n_f2}  predicted="{label2}"  n_domains={len(domains)}'
except Exception as e:
    results['domain_classifier'] = f'FAIL  ->  {e}'

# ── 3. Drift Autoencoder ─────────────────────────────────────────────────────
# Structure: dict with PyTorch-style state_dict (enc.0.weight, dec.0.weight, ...)
# Keys: type, input_dim, h1, h2, dropout, state_dict, scaler, threshold, feat_names
try:
    drift_art  = joblib.load(M / 'drift_pipeline.pkl')
    feat_names = drift_art.get('feat_names', joblib.load(M / 'drift_feature_names.pkl'))
    scaler     = drift_art['scaler']          # StandardScaler
    sd         = drift_art['state_dict']      # PyTorch state_dict
    input_dim  = drift_art['input_dim']
    threshold  = drift_art.get('threshold', None)
    n_in       = scaler.n_features_in_
    X3         = np.zeros((1, n_in))
    Xs         = scaler.transform(X3)         # (1, n_in)

    # PyTorch-style manual forward: encoder path enc.0=Linear, enc.1=BN, enc.4=Linear
    def relu(x): return np.maximum(0, x)
    def bn_forward(x, w, b, rm, rv, eps=1e-5):
        # Inference-mode batch norm: normalize then affine
        x_norm = (x - rm) / np.sqrt(rv + eps)
        return x_norm * w + b

    # enc.0.weight: (h1=85, input=20) → Xs(1,20) @ W0.T(20,85) = (1,85)
    W0, b0 = sd['enc.0.weight'].numpy(), sd['enc.0.bias'].numpy()
    W1, b1 = sd['enc.1.weight'].numpy(), sd['enc.1.bias'].numpy()
    rm1 = sd['enc.1.running_mean'].numpy(); rv1 = sd['enc.1.running_var'].numpy()
    # enc.4.weight: (h2=30, h1=85) → h(1,85) @ W4.T(85,30) = (1,30)
    W4, b4 = sd['enc.4.weight'].numpy(), sd['enc.4.bias'].numpy()

    h  = relu(bn_forward(Xs @ W0.T + b0, W1, b1, rm1, rv1))  # (1,85)
    z  = h @ W4.T + b4                        # (1,30) latent

    # decoder path: dec.0=Linear, dec.1=BN, dec.4=Linear
    Wd0, bd0 = sd['dec.0.weight'].numpy(), sd['dec.0.bias'].numpy()
    Wd1, bd1 = sd['dec.1.weight'].numpy(), sd['dec.1.bias'].numpy()
    rmd = sd['dec.1.running_mean'].numpy();  rvd = sd['dec.1.running_var'].numpy()
    Wd4, bd4 = sd['dec.4.weight'].numpy(), sd['dec.4.bias'].numpy()

    hd  = relu(bn_forward(z @ Wd0.T + bd0, Wd1, bd1, rmd, rvd))
    rec = hd @ Wd4.T + bd4                    # reconstruction

    mse = float(np.mean((Xs - rec) ** 2))
    h1_dim = drift_art.get('h1', W0.shape[0])
    h2_dim = drift_art.get('h2', W4.shape[0])
    results['drift_autoencoder'] = (
        f'PASS  ->  input_dim={input_dim}  h1={h1_dim}  h2={h2_dim}  '
        f'recon_mse={mse:.6f}  threshold={threshold}  n_feat_names={len(feat_names)}'
    )
except Exception as e:
    results['drift_autoencoder'] = f'FAIL  ->  {e}'

# ── 4. Anomaly Detector ──────────────────────────────────────────────────────
# Structure: sklearn Pipeline(steps=['scaler','model'])  +  threshold dict
try:
    anom_pipe   = joblib.load(M / 'anomaly_detector.pkl')   # Pipeline
    anom_thresh = joblib.load(M / 'anomaly_threshold.pkl')  # dict
    n_feats     = anom_thresh['n_features']
    threshold   = anom_thresh['threshold']
    feat_names  = anom_thresh.get('feat_names', [])
    X4          = np.zeros((5, n_feats))
    scores      = anom_pipe.decision_function(X4)
    flagged     = int(np.sum(scores < threshold))
    results['anomaly_detector'] = (
        f'PASS  ->  n_features={n_feats}  threshold={threshold:.4f}  '
        f'flagged={flagged}/5  sample_score={scores[0]:.4f}'
    )
except Exception as e:
    results['anomaly_detector'] = f'FAIL  ->  {e}'

# ── 5. Chart Relevance Scorer ────────────────────────────────────────────────
# Structure: Pipeline(steps=['model'])  +  chart_registry dict
# Note: LGBMClassifier expects 30 features (23 stat + 7 NLP in registry)
#       Use n_features_in_ from the model itself as the authoritative count.
try:
    chart_pipe  = joblib.load(M / 'chart_relevance_scorer.pkl')
    chart_reg   = joblib.load(M / 'chart_registry.pkl')
    chart_model = chart_pipe.named_steps['model']          # LGBMClassifier
    n_f5        = chart_model.n_features_in_               # 30 — authoritative
    chart_types = chart_reg['chart_types']
    n_chart     = chart_reg['n_chart_types']
    X5          = np.zeros((1, n_f5))
    pred5       = chart_pipe.predict(X5)
    label5      = chart_types[int(pred5[0])] if int(pred5[0]) < len(chart_types) else str(pred5[0])
    results['chart_relevance_scorer'] = (
        f'PASS  ->  n_features={n_f5}  n_chart_types={n_chart}  predicted="{label5}"'
    )
except Exception as e:
    results['chart_relevance_scorer'] = f'FAIL  ->  {e}'

# ── 6. Proposal Confidence Scorer ────────────────────────────────────────────
# Structure: sklearn Pipeline  +  confidence_metadata.json
try:
    conf_pipe = joblib.load(M / 'proposal_confidence.pkl')
    with open(M / 'confidence_metadata.json') as f:
        meta = json.load(f)
    n_f6  = conf_pipe[0].n_features_in_
    X6    = np.zeros((1, n_f6))
    prob  = float(conf_pipe.predict_proba(X6)[0][1])
    ece   = meta.get('ece_after', 'N/A')
    auc   = meta.get('val_auc_cal', 'N/A')
    results['proposal_confidence'] = (
        f'PASS  ->  n_features={n_f6}  confidence={prob:.4f}  ECE={ece}  AUC={auc}'
    )
except Exception as e:
    results['proposal_confidence'] = f'FAIL  ->  {e}'

# ── Print results ─────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('  APR 13 @ 20:57 -- MODEL FUNCTIONAL SMOKE TEST')
print('=' * 70)
passes = sum(1 for v in results.values() if v.startswith('PASS'))
total  = len(results)
print(f'  {passes}/{total} models operational\n')
for name, status in results.items():
    tag = status[:4]
    print(f'  [{tag}]  {name}')
    print(f'         {status[8:]}')
    print()
print('=' * 70)
if passes == total:
    print('  ALL 6 APR-13 MODELS ARE WORKING AND INTEGRATED.')
else:
    print(f'  WARNING: {total - passes} model(s) failed.')
print('=' * 70)
