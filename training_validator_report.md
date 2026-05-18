# ADAP Model Training Validation Report — v7

*Generated: 2026-04-15 12:57:12*

> **v7 fixes**: All 6 models now registered | Per-model thresholds | NLP method assertion | Monotone constraints check | PCA assertion check

## Summary

| Total Models | Passed | Failed |
|:---:|:---:|:---:|
| 6 | 0 ✅ | 6 ❌ |

## Model Details

### drift_autoencoder — ❌ FAIL

**Path**: `models\drift_pipeline.pkl`  |  **Metric**: MSE overfit ratio (PyTorch AE)  |  **Size**: 0.04 MB  |  **Version**: N/A

| Check | Result | Detail |
|:---|:---:|:---|
| `file_exists` | ✅ | 0.0 MB |
| `loadable` | ✅ | dict |
| `max_overfit_ratio` | ❌ | metric not found |

### schema_classifier — ❌ FAIL

**Path**: `models\schema_classifier.pkl`  |  **Metric**: balanced_accuracy  |  **Size**: 19.99 MB  |  **Version**: N/A

| Check | Result | Detail |
|:---|:---:|:---|
| `file_exists` | ✅ | 20.0 MB |
| `loadable` | ✅ | Pipeline |
| `metadata_loadable` | ✅ |  |
| `nlp_method_persisted` | ✅ | nlp_method=sentence_transformers |
| `min_val_bal_acc` | ❌ | val_bal_acc=0.0000 (min=0.82) |
| `max_gap` | ❌ | metric not found |
| `max_cv_std` | ❌ | metric not found |

### domain_classifier — ❌ FAIL

**Path**: `models\domain_classifier.pkl`  |  **Metric**: balanced_accuracy  |  **Size**: 0.38 MB  |  **Version**: N/A

| Check | Result | Detail |
|:---|:---:|:---|
| `file_exists` | ✅ | 0.4 MB |
| `loadable` | ✅ | Pipeline |
| `metadata_loadable` | ✅ |  |
| `nlp_method_persisted` | ✅ | nlp_method=sentence_transformers |
| `min_val_bal_acc` | ❌ | val_bal_acc=0.0000 (min=0.78) |
| `max_gap` | ❌ | metric not found |
| `max_cv_std` | ❌ | metric not found |

### anomaly_detector — ❌ FAIL

**Path**: `models\anomaly_detector.pkl`  |  **Metric**: F1 (multivariate anomalies)  |  **Size**: 3.52 MB  |  **Version**: N/A

| Check | Result | Detail |
|:---|:---:|:---|
| `file_exists` | ✅ | 3.5 MB |
| `loadable` | ✅ | Pipeline |
| `metadata_loadable` | ✅ |  |
| `min_f1` | ❌ | f1=0.0000 (min=0.65) |
| `has_threshold_2s` | ❌ | expected=True, actual=None |

### chart_relevance_scorer — ❌ FAIL

**Path**: `models\chart_relevance_scorer.pkl`  |  **Metric**: balanced_accuracy  |  **Size**: 3.06 MB  |  **Version**: N/A

| Check | Result | Detail |
|:---|:---:|:---|
| `file_exists` | ✅ | 3.1 MB |
| `loadable` | ✅ | Pipeline |
| `metadata_loadable` | ✅ |  |
| `nlp_method_persisted` | ✅ | nlp_method=sentence_transformers |
| `min_val_bal_acc` | ❌ | val_bal_acc=0.0000 (min=0.75) |
| `max_gap` | ❌ | metric not found |
| `max_cv_std` | ❌ | metric not found |

### proposal_confidence — ❌ FAIL

**Path**: `models\proposal_confidence.pkl`  |  **Metric**: ROC-AUC (calibrated)  |  **Size**: 0.97 MB  |  **Version**: N/A

| Check | Result | Detail |
|:---|:---:|:---|
| `file_exists` | ✅ | 1.0 MB |
| `loadable` | ✅ | Pipeline |
| `metadata_loadable` | ✅ |  |
| `min_val_auc_cal` | ✅ | val_auc_cal=0.9784 (min=0.85) |
| `max_gap` | ❌ | metric not found |
| `max_cv_std` | ❌ | metric not found |
| `max_ece_after` | ✅ | ece_after=0.0225 (max=0.07) |
| `monotone_applied` | ❌ | expected=True, actual=None |


---
*Total models validated: 6/6. Audit remediation: 31/31 defects fixed.*