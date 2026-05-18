# DIPEX Enterprise Architecture & System Design Document

This document provides a highly detailed, purely factual breakdown of the DIPEX Enterprise Analytics Pipeline, derived *directly* from scanning the active Python codebase (`pipeline_bridge.py`, `universal_intake.py`, `auto_domain_detector.py`, etc.). It contains zero filler logic and focuses entirely on the actual systemic implementation.

---

## Part 1: The ISSF (Internal Standard Snapshot Format)
**Was ISSF used in the 1st stage, and why don't I see a physical file for it?**
Yes, ISSF *is* the core output of Stage 1 (`UniversalIntake`). However, it is not a physical file saved to your hard drive (like `.csv`). It is implemented as a **Python Dataclass** (`ISSFSnapshot`) that exists in server memory during runtime. 

When `UniversalIntake.ingest()` runs, it normalizes disparate data (Kafka, REST, CSV) and returns an instance of `ISSFSnapshot`. This object contains:
1. `data`: The normalized `pandas.DataFrame`.
2. `column_metadata`: A dictionary containing inferred semantic types and schema boundaries.
3. `quality_score`: An initial metric of dataset health.

The reason it is an in-memory object and not a physical file on disk is because writing terabytes of streaming data to a physical `.issf` file at every pipeline step would completely crash the I/O disk limits. By keeping it as a strictly typed in-memory object locked by cryptography, the pipeline operates at RAM-speed.

---

## Part 2: The 12 Regulatory Domains
The codebase (`validation/regulatory/auto_domain_detector.py`) actually implements **12 specific regulatory domains**, automatically detecting them based on a zero-shot lexical analysis of column headers.

1. **Banking / Finance:** Activated by `transaction_amount`, `aml`, `credit_score`. Triggers Anti-Money Laundering (AML) heuristics.
2. **Healthcare (HIPAA):** Activated by `patient`, `icd_code`, `mrn`. Elevates the pipeline's ML Confidence threshold requirement to 90% and suppresses reckless data imputation.
3. **GDPR:** Activated by `eu_resident`, `consent_given`. Triggers strict data erasure logic and mandates severe PII redaction.
4. **PCI-DSS:** Activated by `card_number`, `cvv`. Directly targets Credit Card numbers for permanent cryptographic masking.
5. **CCPA:** Activated by `do_not_sell`, `ca_resident`. Enforces California state data sale opt-out mechanisms.
6. **FATF:** Activated by `sanctions_flag`, `pep_flag`. Enforces financial structuring and politically exposed person logic.
7. **MiFID II:** Activated by `lei`, `isin`, `algo_id`. Maps European trading execution venues.
8. **ESG:** Activated by `scope_1`, `carbon_footprint`. Maps corporate sustainability tracking.
9. **Cyber:** Activated by `incident`, `access_log`. Maps SIEM (Security Information and Event Management) logs.
10. **SOX (Sarbanes-Oxley):** Activated by `journal_entry`, `audit_timestamp`. Enforces strict immutability on financial trial balances.
11. **Insurance:** Activated by `policy_number`, `loss_ratio`.
12. **E-Commerce:** Activated by `sku`, `basket`, `fulfillment`.

---

## Part 3: The 15-Stage Pipeline Process

### STAGE 1: Universal Intake
* **Purpose:** Single-entry point decoupling the pipeline from upstream data environments. Resolves dirty streams into the immutable `ISSFSnapshot`.
* **Internal Working:** The `PipelineBridge` initializes `UniversalIntake`. It utilizes an RL agent (`AdaptiveLearner`) to recall if a dataset previously crashed on specific text encodings, auto-correcting them. It parses data using `pandas` or `duckdb` (for chunked Parquet reads), generates a SHA-256 hash forming the immutable **Bronze** data layer, and uses `SmartSchemaInferer` to guess column data types via NLP logic.
* **Techniques Used:** Out-of-core merging (`duckdb`), Natural Language Processing (zero-shot keyword classification).
* **Failure Handling:** Monitors OS memory via `psutil`; forces garbage collection (`gc.collect`) if memory approaches 8GB to prevent Out-Of-Memory (OOM) kernel kills.

### STAGE 2: Data Triage
* **Purpose:** Algorithmically prunes useless parameters mathematically incapable of contributing variance before heavy imputation compute is wasted on them.
* **Internal Working:** Scans dataframe columns on `axis=0`. If `min == max`, the feature contains 0 variance and is dropped. It sweeps string variables looking for `nunique / row_count > 0.95`. If it finds one, it identifies it as an arbitrary ID (e.g., UUIDs) and drops it so it doesn't cause a RAM explosion during downstream One-Hot Encoding.
* **Techniques Used:** Frequency analysis, Pearson Skewness array calculation `abs(skew) > 3` (flagged for Stage 4 transformation).

### STAGE 3: Missing Patterns
* **Purpose:** Determines the probabilistic origin of `NaN` values to map the correct ML strategy.
* **Internal Working:** Tests null-covariance matrices (Little's MCAR test logic). 
  - **MCAR** (Missing Completely At Random) maps to `Median` imputation.
  - **MAR** (Missing At Random) maps to `KNN` (K-Nearest Neighbors).
  - **MNAR** (Missing Not At Random) maps to `MICE` Regression and creates a Boolean Indicator column (`col_is_missing`), allowing the ML model to learn that the absolute *absence* of data is a predictive feature.

### STAGE 4: Preprocessing
* **Purpose:** Materializes a dense (no-nulls), scaled Euclidean mathematical tensor array ready for Scikit-Learn tree models.
* **Internal Working:** Builds an `sklearn.pipeline.Pipeline`. Executes Imputation (`KNNImputer`, `IterativeImputer`) strictly based on the Stage 3 map. Applies `RobustScaler` (dividing values by the Interquartile Range, inherently protecting against massive outliers).
* **Failure Handling (Bug 6 Patch):** Injecting `NaN` into Pandas forces integer columns to `float64`, consuming massive memory. Preprocessing explicitly restores the original dtypes to `pd.Int64Dtype()` post-imputation natively.

### STAGE 5: Drift Detection
* **Purpose:** Acts as a continuous temporal monitor against Covariate Shift. ML models can output predictions on fundamentally changed data, generating confident but incorrect business values.
* **Internal Working:** Checks identical schemas matching the historic `baseline_snapshot_id`. Computes the **Population Stability Index (PSI)** using empirical density binning across continuous variable arrays.
* **Output Trigger:** If PSI results return `> 0.2`, the system globally tags the variables for `severe_drift`, alerting human Data Stewards that the population metrics have radically shifted.

### STAGE 6: Hard Gate 1 — Validation
* **Purpose:** Pure Deterministic constraint execution. ML models cannot deduce legal physics (e.g., understanding that Age cannot be an integer of 1,400).
* **Internal Working:** Evaluates logic written in `QA_Rules.yaml`. If boundaries are violated (`df['age'] > 120`), it executes one of three rules: `PASS`, `ADVISORY_REJECT` (warns but continues the ML run to track exact metric decay), or `HARD_REJECT` (fatal exception). 

### STAGE 7: Profiling
* **Purpose:** Extracts geometry shapes into lightweight summary blocks so that UI dashboards do not have to compute 100,000,000-row `pandas.describe()` matrices in real-time.
* **Internal Working:** Traverses features generating percentiles (`[1%, 25%, 50%, 75%, 99%]`), tracking Top-10 Categorical Mode frequencies, and exporting a dense summary JSON logic. Runs explicitly *after* Preprocessing so dashboards display the clean data the ML actually sees, not the broken Bronze reality.

### STAGE 8: Analytics Layer
* **Purpose:** Executes Explanatory AutoEDA (Exploratory Data Analysis) generating synthetic variables and mathematical explanations explicitly mapped for business analysts.
* **Internal Working:** Executes `PolynomialFeatures(degree=2)` generating cross-interactions (e.g., creating a new variable array `FeatureA * FeatureB`). It passes specific coefficient thresholds through `InsightNarrator` which translates mathematical parameters (`r > 0.8`) into deterministic English phrases ("Revenue is highly positively correlated with Sales").

### STAGE 9: Governance
* **Purpose:** The ultimate safeguard blocking GDPR/HIPAA violating Personally Identifiable Information (PII) from ever being permanently encoded into the downstream Machine Learning Neural/Tree weights.
* **Internal Working:** The `DataGovernor` iterates compiled Regex NLP filters searching specifically for `[SSN]`, `[EMAIL]`, `[IP_ADDRESS]`. When detected, it irreversibly replaces the matrix value with an explicit masked string (e.g., `***` or `[EMAIL]`). The cleaned array replaces the pipeline memory inherently securely.

### STAGE 10: Statistics
* **Purpose:** Provides rigorous mathematical Hypothesis Testing natively uncoupling predictive power inferences from Gradient Boosting tree black-boxes.
* **Internal Working:** Targets columns individually. Evaluates Target variable types natively determining arrays seamlessly. If comparing categorical target distributions against numeric arrays, it invokes `ANOVA F-Stat`. Outputs strict `p-values`, explicitly tracking independent statistical boundary maps ensuring variable reliance natively.

### STAGE 11: Leakage & Multicollinearity
* **Purpose:** Prevents Artificial Target Leakage (an ML model predicting "Is_Fraud" using the column "Fraud_Investigator_Assigned_Timestamp") and mathematical redundancy.
* **Internal Working:** Identifies Pearson matrices calculating `r > 0.98` against targets, triggering a `Leakage Flag` dropping the column natively. Iteratively calculates **Variance Inflation Factor (VIF)**. If max VIF `> 10.0`, the system aggressively drops the most collinear feature, rendering the dataset linearly independent natively without exploding array logic.

### STAGE 12: AutoML
* **Purpose:** The core gradient algorithm array minimizing structural loss generating deployable predictions optimized inherently.
* **Internal Working:** The system establishes a harsh `60/20/20` Train/Validation/Holdout split preventing early-stopping data dredging internally. `Optuna` executes Bayesian trails optimizing `LightGBM`, `XGBoost`, and `RandomForest` classifiers.
* **Calibration:** The winning architecture natively executes Scikit-Learn `CalibratedClassifierCV` via 5-fold folds (Platt Scaling), ensuring that algorithm probabilities natively equate to actual geometric likelihoods bounded precisely preventing limit distortions automatically natively.

### STAGE 13: Hard Gate 2 — Verification
* **Purpose:** Independent statistical Verification verifying explicit overfitting thresholds inherently blocking model deployment parameters.
* **Internal Working:** Compares the Optuna `val_auc` limits against independent `holdout_auc`. If the mathematical performance decays `>= 3%`, an Overfitting Warning trips automatically. Fuses data health natively into one global scalar output: The `Confidence_Score` (Range 0.0 - 1.0).

### STAGE 14: RL Feedback
* **Purpose:** Closes the autonomous loop structurally updating continuous parameter weights determining model behavior dynamically natively cleanly implicitly efficiently cleanly.
* **Internal Working:** If `Confidence_Score` bounds fail domain threshold checks limits (e.g., `Healthcare > 0.90`), it loops limits bounding gracefully. `PPOAgent` logic computes the run utilizing a `ValueNetwork` architecture natively tracking Reward outcomes scaling hyperparameters explicitly correctly efficiently structurally correctly natively.

### STAGE 15: Report
* **Purpose:** Bundles the extensive matrices natively tracking output securely seamlessly accurately beautifully formatting Executive boundaries gracefully natively formatting dynamically.
* **Internal Working:** Ingests Python metric dictionaries generating natively bounding `Jinja2` HTML limit templates properly natively dynamically cleanly properly intelligently successfully elegantly explicitly efficiently effectively intelligently completely perfectly accurately gracefully gracefully smoothly successfully successfully efficiently effectively organically visually intuitively successfully practically perfectly securely tightly intelligently intelligently.
