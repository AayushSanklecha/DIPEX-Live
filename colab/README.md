# DIPEX Model Training — Colab & Local Guide

## Overview

DIPEX uses **6 pre-trained ML model artifacts** stored in `models/`.  
All models are trained on real-world datasets in the Colab notebooks listed below.  
The artifacts are already committed — you only need to retrain if you want to improve or customize a model.

---

## Model Inventory

| Artifact | Purpose | Notebook |
|---|---|---|
| `schema_classifier.pkl` + `schema_label_encoder.pkl` | Semantic column type inference (15 labels: id, age, amount, date…) | `train_schema_classifier.ipynb` |
| `pipeline_success_predictor.pkl` | Pre-run success prediction (10 features) | `train_pipeline_predictor.ipynb` |
| `proposal_confidence.pkl` + `proposal_quantile_model.pkl` | XGBoost confidence scorer + quantile regressor | `train_proposal_models.ipynb` |
| `chart_relevance_scorer.pkl` | Ranks chart types for a given DataFrame + intent | `train_chart_ranker.ipynb` |
| `nlp_query_classifier.pkl` + `nlp_query_vectorizer.pkl` | TF-IDF + LinearSVC for natural-language query routing | `train_nlp_query.ipynb` |
| `drift_autoencoder.pkl` + `drift_scaler.pkl` + `drift_vae.pt` | Multivariate drift detection (VAE) | `train_drift_models.ipynb` |

---

## How to Retrain (Colab)

1. Open the relevant notebook in [Google Colab](https://colab.research.google.com).
2. Mount your Google Drive or clone this repo.
3. Run all cells. The notebook saves the PKL/PT files to `models/`.
4. Download and replace the files in this repo's `models/` directory.
5. Commit and push.

---

## Feature Compatibility

> [!IMPORTANT]
> The 20 feature names extracted in `ingestion/schema_infer.py::_extract_column_features()` 
> **must exactly match** those used during Colab training.  
> If you add/remove features during retraining, update both the notebook AND `_FEAT_ORDER` 
> in `schema_infer.py` at the same time.

---

## Runtime Behavior

- All models are loaded lazily at first use (no startup cost if unused).
- If a PKL is missing or corrupt, the module **falls back to heuristic rules** and logs a warning.
- The pipeline never breaks due to a missing model — it degrades to rule-based inference.

---

## Integration Points

| Model | Loaded by | Used in stage |
|---|---|---|
| `schema_classifier` | `ingestion/schema_infer.py` | Ingestion → UDIL enrichment |
| `pipeline_success_predictor` | `modeling/` | Pre-run risk assessment |
| `proposal_confidence` | `proposal/ml_confidence_scorer.py` | Proposal Layer scoring |
| `chart_relevance_scorer` | `reporting_service/` | Executive Report chart selection |
| `nlp_query_classifier` | `query_engine/` | Natural-language query routing |
| `drift_autoencoder` | `profiling/drift_detector.py` | Stage 3 PSI / VAE drift detection |
