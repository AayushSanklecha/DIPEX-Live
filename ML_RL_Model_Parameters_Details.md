# ADAP System: Machine Learning and Reinforcement Learning Models & Parameters Details

This document consolidates the complete architectural configurations, hyperparameters, and implementation details for all ML and RL models utilized in the ADAP platform. It supplements the main report by fully resolving missing or non-explicit parameters.

## 1. NLP-Augmented Schema Classifier
* **Framework:** 3-Stage Cascade (Regex → TF-IDF + Logistic Regression → LightGBM)
* **Stage 1 (Regex):** 19 compiled patterns, matching threshold >= 90% across 200 random samples.
* **Stage 2 (TF-IDF + LR):** 
  * Character n-grams TF-IDF (n ∈ {2,...,5}), 10,000 features. 
  * Logistic Regression: `C=5.0`, `solver='lbfgs'`, `multi_class='multinomial'`.
* **Stage 3 (LightGBM):** Evaluates 58 features (30 statistical, 28 NLP embedding similarities).
  * **Hyperparameters:** `n_estimators=400`, `max_depth=8`, `learning_rate=0.05`, `num_leaves=127`.
  * **Ensemble Weights:** Stage 3 prediction weight = 0.70; Stage 2 weight = 0.30.

## 2. Drift Autoencoder
* **Framework:** PyTorch Multi-Layer Perceptron (MLP)
* **Architecture:** Autoencoder bottleneck structure.
  * Explicit Dimensions: `20 -> 85 -> 30 -> 85 -> 20`
* **Regularization:** 1D Batch Normalization applied at hidden layers.
* **Inference Details:** No reference window required. Decision threshold `τ = 0.785` (calibrated to 95th percentile of clean training reconstruction MSE).

## 3. Anomaly Detector
* **Framework:** IsolationForest (scikit-learn Pipeline)
* **Pipeline:** `StandardScaler` → `IsolationForest`
* **Hyperparameters:** `n_estimators=200`
* **Scoring Mechanism:** Anomaly score computed via expected path length bounds; flagged if score deviates toward 1.0.

## 4. Proposal Confidence Scorer
* **Framework:** Platt-Calibrated VotingClassifier Ensemble
* **Base Architectures & Weights:**
  * LightGBM (Weight: `0.40`)
  * RandomForest (Weight: `0.35`)
  * Logistic Regression (Weight: `0.25`)
* **Input Feature Vector:** 24 dimensions.
* **Calibration:** Platt Scaling applied to map outputs to rigorous confidence thresholds (ECE ≤ 0.07).

## 5. Chart Relevance Scorer
* **Framework:** LightGBM Classifier Pipeline
* **Pipeline:** `StandardScaler` → `LGBMClassifier`
* **Input Features:** 30 statistical data dimension signals.
* **Output:** Softmax probabilities over 7 predefined target chart types (histogram, bar, scatter, line, box, heatmap, pie).

## 6. Domain Classifier
* **Framework:** RandomForestClassifier Pipeline
* **Pipeline:** `StandardScaler` → `RandomForestClassifier`
* **Hyperparameters:** `n_estimators=300`
* **Input Features:** 53 dataset-level features.
* **Output Scale:** Maps dynamically to 7 regulatory domains (banking, healthcare, finance, ecommerce, government, insurance, generic).

## 7. Reinforcement Learning (RL) Engine
### 7.1 Thompson Sampling Bandit
* **Arms Layout:** 9 posteriors (evaluated across 3 decision axes).
* **Prior Configuration:** Weakly informative `Beta(2, 2)` constraints.
* **Update Scheme:** Closed-form Beta posterior updates after every run: `α_new = α + r`, `β_new = β + (1 - r)`.

### 7.2 Proximal Policy Optimization (PPO) Actor-Critic
* **Action Space:** 8 discrete decision axes (yielding 11,664 possible configuration outcomes).
* **Hyperparameters:** Clipping Parameter `ε = 0.20`.
* **Actor (Policy) Network:** Shared MLP backbone.
  * Dimensions: `12 (Input) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> 8 independent SoftMax heads`. (Approx. 9,000 parameters)
* **Critic (Value) Network:** 
  * Dimensions: `12 (Input) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(1)` (Scalar state-value estimator `V(s)`).

## 8. AutoML Downstream Target Models
When stage 7 identifies model proposals contextually, Optuna fine-tunes models within standard bounds, regulated under the strict anti-overfitting constraints (early stopping set to `25` epochs):
* **LightGBM Candidates:** `max_depth=8`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `learning_rate=0.05`, `n_estimators=500`.
* **XGBoost Candidates:** `max_depth=6`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `reg_alpha=0.1`, `learning_rate=0.05`, `n_estimators=500`.
* **Random Forest Candidates:** `n_estimators=200`, `max_depth=8`, `min_samples_leaf=5`, `max_features="sqrt"`.
