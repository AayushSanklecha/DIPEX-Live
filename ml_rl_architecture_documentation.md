# Machine Learning & Reinforcement Learning in the Project

The following documentation strictly details the exact Machine Learning (ML) and Reinforcement Learning (RL) architectures natively implemented in the repository, referencing exact source files (e.g., `scripts/train_individual/`, `modeling/`, `learning/`). 

-----------------------------------
DRIFT AUTOENCODER
-----------------------------------

### 1. Purpose
- **Usage:** Identifies distributional data drift over time by detecting when new data no longer fits the expected schema geometries.
- **Stage:** Pre-Analysis / Data Intake Pipeline.

---

### 2. Problem Type
- **Type:** Anomaly Detection / Drift Estimation.

---

### 3. Input Features
- **Features:** Numeric "drift fingerprints" extracted from numeric schema columns.
- **Types:** Numeric arrays (scaled metrics representing distributional properties like variance, quantiles, and extreme bounds). 

---

### 4. Training Data
- **Type:** Tabular numeric features extracted from datasets.
- **Source:** Historical pipeline data and augmented real-world open-data datasets (e.g., from OpenML). Includes noise injection simulating high-missingness drift and scale shifts.

---

### 5. Training Method
- **Method:** Batch training using PyTorch neural networks.
- **Frequency:** Periodic offline training via the script `01_drift_autoencoder.py`.

---

### 6. Model Details
- **Model Name:** PyTorch Custom Symmetric Bottleneck Autoencoder (`_DriftAE`).
- **Key Characteristics:** 
  - Uses `BatchNorm1d` + `GELU` activation + `Dropout`.
  - Linear bottleneck (no activation) for stable reconstruction.
- **Key Hyperparameters:**
  - dynamically optimized via `Optuna` (e.g., hidden layers `h1` [24-96], `h2` [4-48], learning rate, and dropout).
  - Uses `AdamW` optimizer and `ReduceLROnPlateau` scheduler.

---

### 7. Evaluation Metrics
- **Metrics used:** 
  - Mean Squared Error (MSE) for reconstruction loss.
  - MAD-based anomaly scores (Median Absolute Deviation) combining reconstruction error and latent space distance.
- **Why:** MSE ensures the autoencoder perfectly learns normal data shapes. The MAD threshold provides a statistical bound that is highly robust to heavily skewed outlier distributions compared to standard mean+3σ.

---

### 8. Realistic Performance Expectations
- **Performance:** Excellent at catching massive structural metric shifts (e.g., a column changing from inches to centimeters). Moderate sensitivity to highly subtle, slowly developing covariance drift.

---

### 9. Why This Model Was Chosen
- **Why:** Autoencoders naturally map the boundaries of "normal" data without explicitly needing thousands of labeled "drift" examples. 
- **Alternatives:** 
  - *Isolation Forest*: Less capable of capturing deep, nonlinear correlations between massive amounts of numeric schema metrics.

---

### 10. Libraries Used
- **Libraries:** `torch` (PyTorch) for the neural architecture and GPU acceleration, `optuna` for hyperparameter optimization, `scikit-learn` for `RobustScaler`.

---

### 11. Inference / Usage in Pipeline
- **Flow:** Pipeline extracts standard geometry fingerprints from new data → passes to Autoencoder → Autoencoder reconstructs it. If the combined reconstruction + latent distance score exceeds the MAD threshold, a structural drift alert is triggered.

---

### 12. Limitations
- **Limitations:** Fails on entirely categorical columns (only processes numeric fingerprints). Highly dependent on robust scaling.

-----------------------------------
SCHEMA SEMANTIC-TYPE CLASSIFIER
-----------------------------------

### 1. Purpose
- **Usage:** Infers the structural dictionary type of a column (e.g., identifying a column as 'boolean', 'id', 'age', 'percentage', or 'duration').
- **Stage:** Pre-Analysis Profiling.

---

### 2. Problem Type
- **Type:** Multi-class Classification.

---

### 3. Input Features
- **Features:** A combination of statistical features (min, max, skew, null-rates) and NLP word embeddings of the column name.
- **Types:** Mixed (Numeric statistics and Dense Vector NLP embeddings).

---

### 4. Training Data
- **Type:** Tabular rows representing individual columns. 
- **Source:** Hardcoded template structures and massive pools of real OpenML datasets. Augmented via Gaussian noise interpolation (SMOTE-like logic) to handle imbalanced classes.

---

### 5. Training Method
- **Method:** Batch training using an Early-Stopping loop over 5-fold Stratified Cross-Validation arrays.

---

### 6. Model Details
- **Model Name:** `LightGBM` (LGBMClassifier).
- **Key Hyperparameters:**
  - Evaluated via Optuna: `n_estimators` (500–3000), `max_depth` (4–10), `num_leaves`, `learning_rate` (log uniform), and L1/L2 regularization (`reg_alpha`/`reg_lambda`).

---

### 7. Evaluation Metrics
- **Metrics used:** 
  - Balanced Accuracy.
- **Why:** Balanced accuracy is explicitly used to heavily penalize the model if it ignores minority schema structures (like recognizing rare 'duration' columns versus ubiquitous 'id' columns).

---

### 8. Realistic Performance Expectations
- **Performance:** Highly accurate (expected 90%+ Balanced Accuracy). NLP string comparisons coupled with statistical boundaries make typical business columns trivial to map.

---

### 9. Why This Model Was Chosen
- **Why:** LightGBM handles the massive sparsity of hybrid NLP/statistical features beautifully and is incredibly fast to train. 
- **Alternatives:** 
  - *RandomForest*: Too slow to train over nested cross-validation bounds.
  - *Deep Networks*: Overkill for a tabular inference logic with fewer than 200 dimensions.

---

### 10. Libraries Used
- **Libraries:** `lightgbm`, `scikit-learn` (LabelEncoder, StratifiedKFold), `optuna`.

---

### 11. Inference / Usage in Pipeline
- **Flow:** New user dataset is uploaded → DataFrame columns are split → NLP vector + stats calculated for each column → LightGBM outputs the most probable string label.

---

### 12. Limitations
- **Limitations:** Assumes column headers map somewhat closely to English structural nouns. Obfuscated or completely blank column names rely solely on the numerical statistics which drops accuracy.

-----------------------------------
DOMAIN CLASSIFIER
-----------------------------------

### 1. Purpose
- **Usage:** Categorizes the encompassing business domain of the entire dataset (e.g., Banking, Healthcare, Government, E-commerce).
- **Stage:** Pre-Analysis / Context tagging.

---

### 2. Problem Type
- **Type:** Multi-class Classification.

---

### 3. Input Features
- **Features:** Dataset-wide aggregations (log_n_rows, n_cols, null_rates) combined with keyword match flags (`kw_banking`, `kw_ecommerce`) and a mean-pooled NLP vector over all column names.
- **Types:** Numeric structural metrics and Dense Vector metrics.

---

### 4. Training Data
- **Type:** Tabular representations of entire datasets. 
- **Source:** Human-curated "gold labels" pulled from explicit mappings, backed by a 3-layer NLP consensus engine for unlabeled data limits.

---

### 5. Training Method
- **Method:** Batch supervised training with tight Optuna-based early stopping to prevent saturation/overfitting.

---

### 6. Model Details
- **Model Name:** `LightGBM` (LGBMClassifier).
- **Key Hyperparameters:**
  - Heavy regularization limits intentionally passed to Optuna: constrained `num_leaves` (<200) and higher `min_child_samples` to prevent the model from memorizing highly sparse single-domain data.

---

### 7. Evaluation Metrics
- **Metrics used:** 
  - Balanced Accuracy (scored on split Holdout datasets).
- **Why:** Evaluates the capacity of the model to identify rare business matrices (like 'Government') without biasing purely toward common examples.

---

### 8. Realistic Performance Expectations
- **Performance:** Expect moderate to high bound accuracy (70-85%). Ambiguous datasets (e.g. generalized "staff logs") might straddle multiple domains.

---

### 9. Why This Model Was Chosen
- **Why:** Ensembling gradient boosted trees works perfectly against categorical NLP-mean features combined with discrete boolean keyword flags.

---

### 10. Libraries Used
- **Libraries:** `lightgbm`, `scikit-learn`, `numpy`.

---

### 11. Inference / Usage in Pipeline
- **Flow:** The dataset structure is reduced into a single matrix row → LightGBM outputs the domain. The pipeline then conditionally enforces regulatory checks based on this (e.g., applying CCPA rules if "Healthcare" is detected).

---

### 12. Limitationsq
- **Limitations:** Subject to severe data sparsity if trained on generic academic sets. Struggles with highly generalized datasets that contain no distinctive business terminology.

-----------------------------------
AUTOMATED MACHINE LEARNING (AutoML) DOWNSTREAM
-----------------------------------

### 1. Purpose
- **Usage:** Dynamically models the end-user's provided target column (Y) against the rest of their dataset (X).
- **Stage:** Final Model Training / Hypothesis testing.

---

### 2. Problem Type
- **Type:** Binary Classification, Multiclass Classification, or Target Regression (Dynamic based on user requests).

---

### 3. Input Features
- **Features:** Whatever features exist in the user's uploaded dataset after imputation and validation loops. 

---

### 4. Training Data
- **Type:** The user's specific live dataset. 
- **Size:** Handled in core pipeline memory (Pandas). 

---

### 5. Training Method
- **Method:** Just-In-Time (JIT) batch training generated while the pipeline is running.

---

### 6. Model Details
- **Model Name:** Competition between `XGBoost`, `LightGBM`, and `RandomForest`.
- **Key Hyperparameters:**
  - These are evaluated conditionally based on the task size. XGBoost and LightGBM are allowed to optimize tree depths and learning rates natively. Models are then calibrated using `CalibratedClassifierCV` (Platt Scaling) to ensure confidence probabilities are mathematically truthful.

---

### 7. Evaluation Metrics
- **Metrics used:** 
  - Target dependent: F1-Score, ROC-AUC, RMSE, MAE.
- **Why:** Typical supervised evaluation metrics natively verify modeling bounds.

---

### 8. Realistic Performance Expectations
- **Performance:** Wholly dependent on the quality of the user's data. 

---

### 9. Why This Model Was Chosen
- **Why:** Tree-based ensemble estimators (XGB/LGBM) require minimal scaling, ignore outliers naturally, process missing values natively, and train blazingly fast on tabular enterprise data.

---

### 10. Libraries Used
- **Libraries:** `xgboost`, `lightgbm`, `sklearn.ensemble`.

---

### 11. Inference / Usage in Pipeline
- **Flow:** The best algorithm is serialized into a final `.pkl` file via `joblib`, attached to a SHAP explainer matrix, and returned as the core payload in the Executive Report.

---

### 12. Limitations
- **Limitations:** These architectures cannot process raw images or sound embeddings; they are strictly designed for structured tabular arrays. 

-----------------------------------
PPO REINFORCEMENT LEARNING AGENT
-----------------------------------

### 1. Purpose
- **Usage:** Orchestrates and optimizes the pipeline run configurations for *future* pipeline executions based on past runtime successes and failures.
- **Stage:** Stage 7 (Confidence Aggregation & RL Feedback).

---

### 2. Problem Type
- **Type:** Reinforcement Learning.

---

### 3. Input Features
- **Features:** Pipeline environment states: memory usage bounds, execution time, validation drop-rates, model confidence scores, and categorical logic-gate passes.

---

### 4. Training Data
- **Type:** Episodic logs generated recursively by the `PipelineBridge`. 
- **Source:** Generated strictly by preceding pipeline runs locally.

---

### 5. Training Method
- **Method:** Incremental online training (PPO Agent updates weights after pipeline completion).

---

### 6. Model Details
- **Model Name:** Proximal Policy Optimization (PPO) using a PyTorch Multi-Layer Perceptron (Actor-Critic framework).
- **Key Hyperparameters:**
  - Standard RL hyperparameters exist: Discount factors (Gamma), Actor-Critic learning rates via `Adam`, and exploration bounds (Epsilon). 

---

### 7. Evaluation Metrics
- **Metrics used:** 
  - Reward Value Calculation.
- **Why:** Agent success stringently relates back to a calculated reward function balancing highest accuracy against lowest execution latency and minimal data quarantines.

---

### 8. Realistic Performance Expectations
- **Performance:** Requires a "cold start" period of roughly 15-30 runs before the agent learns how to navigate the dataset distributions intelligently without forcing the pipeline into slow fallback states.

---

### 9. Why This Model Was Chosen
- **Why:** PPO is the current enterprise state-of-the-art for stable, continuous-action-space environments.
- **Alternatives:** 
  - *Deep Q-Networks (DQN)*: Too unstable dealing with continuous outputs (like dynamically shifting imputation threshold floats).

---

### 10. Libraries Used
- **Libraries:** `torch` (PyTorch).

---

### 11. Inference / Usage in Pipeline
- **Flow:** Model observes the start-state of the data, proposes operational configuration limits (Action Space), Pipeline runs with those limits, Engine calculates a score (Reward), PPO agent updates network logic.

---

### 12. Limitations
- **Limitations:** Incredibly difficult to debug heuristically if the agent converges on an edge-case shortcut (e.g. learning to just skip complex validations entirely to maximize speed rewards).

-----------------------------------
ANOMALY ISOLATION FOREST 
-----------------------------------

### 1. Purpose
- **Usage:** Systematically quarantines extreme data outliers right before model training.
- **Stage:** Dynamic Fallback / Pre-Processing.

---

### 2. Problem Type
- **Type:** Unsupervised Anomaly Detection.

---

### 3. Input Features
- **Features:** Tabular numeric columns.

---

### 4. Training Data
- **Type:** Live active DataFrame rows passing through the pipeline.

---

### 5. Training Method
- **Method:** JIT batch inference directly on the data.

---

### 6. Model Details
- **Model Name:** `IsolationForest` (Scikit-Learn). 
- **Key Hyperparameters:** Default bounding parameters with dynamically calculated `contamination` limits depending on the data size thresholds.

---

### 7. Evaluation Metrics
- **Metrics used:** 
  - No explicit evaluation metrics implemented. (It behaves strictly as an unsupervised data truncator based on topological separation distances).

---

### 8. Realistic Performance Expectations
- **Performance:** Exceptionally fast at finding obvious corrupted numerical data spikes. 

---

### 9. Why This Model Was Chosen
- **Why:** Very low computational overhead compared to training Neural Autoencoders for one-off user data pruning.

---

### 10. Libraries Used
- **Libraries:** `sklearn.ensemble.IsolationForest`.

---

### 11. Inference / Usage in Pipeline
- **Flow:** Data passes through → IF calculates path lengths to isolate points → Returns binary mapping array → Pipeline drops extreme outliers.

---

### 12. Limitations
- **Limitations:** Cannot compute semantic outliers (e.g. an anomaly in textual fields). Requires completely non-null numeric geometries to function.

-----------------------------------

## Models Not Used (But Relevant)
- **Deep Neural Networks (Dense Tabular Transformers, TabNet)**: Not implemented. The core modeling engine purposefully restricts downstream creation to XGB/LGBM because Transformers for tabular datasets are dramatically slower, exponentially more compute-heavy, and offer effectively zero performance gain over Gradient Boosted Trees for matrices with under 1 million rows.
- **Recurrent Neural Networks (LSTMs)**: Not implemented. There is no heavy longitudinal time-series forecasting capability inherently coded into the current automated ML gates.
- **Large Language Models (LLMs) for ML Evaluation**: Not implemented. The `ReportingService` aggregates metric floats mathematically and rigidly formats HTML; no generative AI is used to "hallucinate" interpretation strings, preventing non-deterministic logic from corrupting executive audit logs.
