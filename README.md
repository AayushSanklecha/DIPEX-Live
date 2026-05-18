---
title: DIPEX Live
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🧠 DIPEX
## Data Intelligence Pipeline with Expert Verification
### *Enterprise-Grade AI-Powered Data Quality, Validation & Analytics Platform*

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-Streaming-231F20?style=for-the-badge&logo=apachekafka&logoColor=white)](https://kafka.apache.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Schema_Classifier-00B4D8?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-Drift_AE-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose)
[![DuckDB](https://img.shields.io/badge/DuckDB-Analytical_DB-FFF000?style=for-the-badge)](https://duckdb.org)
[![Tests](https://img.shields.io/badge/Tests-434%20Passing-2DBA4E?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![Version](https://img.shields.io/badge/Version-3.0.0-FF6B6B?style=for-the-badge)](./config.yaml)
[![License](https://img.shields.io/badge/License-Proprietary-FF4B4B?style=for-the-badge)](./LICENSE)

<br/>

> **DIPEX** is an end-to-end data intelligence platform built for regulated industries.
> It ingests structured data from any source, runs it through an 8-stage AI pipeline,
> enforces regulatory compliance (AML, HIPAA, SOX, GDPR), trains AutoML models with
> SHAP explanations, generates LLM-powered narrative reports, and maintains a complete
> immutable audit trail — all driven by a dual Reinforcement Learning engine that adapts
> its own strategies based on real pipeline outcomes.

<br/>

[🚀 Quick Start](#-quick-start) &nbsp;•&nbsp;
[🏗 Architecture](#-system-architecture) &nbsp;•&nbsp;
[🔄 Pipeline Stages](#-8-stage-pipeline-in-detail) &nbsp;•&nbsp;
[🤖 ML Models](#-ml-models--artifacts) &nbsp;•&nbsp;
[🎮 RL Engine](#-reinforcement-learning-engine) &nbsp;•&nbsp;
[🛡 Compliance](#-regulatory-compliance-engine) &nbsp;•&nbsp;
[📡 API](#-api-reference) &nbsp;•&nbsp;
[🖥 Dashboard](#-frontend-dashboard) &nbsp;•&nbsp;
[⚙️ Config](#%EF%B8%8F-configuration-reference) &nbsp;•&nbsp;
[🧪 Testing](#-testing) &nbsp;•&nbsp;
[📁 Structure](#-complete-project-structure)

</div>

---

## 📋 Table of Contents

1. [What is DIPEX?](#-what-is-dipex)
2. [Feature Overview](#-feature-overview)
3. [System Architecture](#-system-architecture)
4. [8-Stage Pipeline In Detail](#-8-stage-pipeline-in-detail)
5. [Data Ingestion Layer](#-data-ingestion-layer)
6. [Medallion Data Architecture](#-medallion-data-architecture-bronzesilvergold)
7. [ML Models & Artifacts](#-ml-models--artifacts)
8. [AutoML Engine](#-automl-engine)
9. [Reinforcement Learning Engine](#-reinforcement-learning-engine)
10. [Validation Engine](#-validation-engine)
11. [Regulatory Compliance Engine](#-regulatory-compliance-engine)
12. [Preprocessing & Feature Engineering](#-preprocessing--feature-engineering)
13. [Analytics & EDA](#-analytics--eda)
14. [Kafka Streaming Pipeline](#-kafka-streaming-pipeline)
15. [LLM Integration & Reporting](#-llm-integration--reporting)
16. [Frontend Dashboard](#-frontend-dashboard)
17. [API Reference](#-api-reference)
18. [Quick Start](#-quick-start)
19. [Configuration Reference](#%EF%B8%8F-configuration-reference)
20. [Testing](#-testing)
21. [Production Deployment](#-production-deployment)
22. [Security & Governance](#-security--governance)
23. [Monitoring & Observability](#-monitoring--observability)
24. [Complete Project Structure](#-complete-project-structure)
25. [Performance Benchmarks](#-performance-benchmarks)
26. [Model Maintenance](#-model-maintenance)

---

## 🌐 What is DIPEX?

**DIPEX** (Data Intelligence Pipeline with Expert Verification) is a production-grade, enterprise-scale data intelligence platform that answers a fundamental question every data-driven organization faces:

> *"How do I know if the data I'm about to train a model on, feed into a report, or use for a business decision is actually trustworthy?"*

Traditional data pipelines treat quality as an afterthought — a few null checks, maybe a schema validation. DIPEX treats it as the primary concern, building an 8-stage AI-powered system that understands, validates, explains, and audits every dataset before allowing any downstream use.

### Why DIPEX Exists

In regulated industries (banking, healthcare, finance), data quality failures are not just technical problems — they are compliance failures. A model trained on leaked features, a report built on drifted data, or a transaction analysis using schema-broken fields can result in regulatory penalties, inaccurate decisions, and loss of trust.

DIPEX solves this by combining:

| Layer | Technology | Purpose |
|---|---|---|
| **Ingestion** | UniversalIntake + ChunkedParquetWriter | Accept any source, handle any size |
| **Understanding** | NLP-augmented LightGBM classifier | Know *what* each column means semantically |
| **Cleaning** | RobustTriage + MissingDataEngine | Handle every class of real-world messiness |
| **Validation** | 7 parallel validators | Flag every category of data quality risk |
| **Compliance** | Domain-specific rule engines | Enforce AML, HIPAA, SOX, GDPR automatically |
| **Modeling** | AutoML + SHAP + Calibration | Propose trustworthy models with explanations |
| **Adaptation** | Dual RL (Thompson + PPO) | Learn optimal pipeline strategies from outcomes |
| **Audit** | Immutable Bronze/Silver/Gold layers | Guarantee tamper-evidence for regulators |

### Design Principles

1. **Never block on data quality alone** — pipelines use advisory mode; human experts review flags
2. **Schema-first** — understand what every column *means* before deciding what to do with it
3. **Audit everything** — every transformation, gate decision, and model inference is logged
4. **Self-improving** — the RL engine gets better at pipeline strategy with every run
5. **Domain-aware** — banking data is treated differently from healthcare data from day one

---

## ✨ Feature Overview

### 🔌 Multi-Source Data Ingestion

DIPEX accepts data from **8 source types** through a single `UniversalIntake` interface:

- **Files:** CSV, Excel (xlsx/xls), JSON (flat & nested), XML, Apache Parquet, Apache Avro, Apache Feather, plaintext logs
- **Relational Databases:** PostgreSQL (server-side cursors, connection pooling), SQLite, DuckDB
- **Document Stores:** MongoDB (batch cursor, collection → DataFrame)
- **Key-Value Stores:** Redis (hash/sorted-set → DataFrame)
- **APIs:** REST endpoints with configurable auth headers, pagination (up to 10,000 pages)
- **Streaming:** Apache Kafka (SASL-SSL, Schema Registry, consumer lag monitoring)

All sources produce an identical `SnapshotResult` object — downstream stages are completely source-agnostic.

### 🤖 Machine Learning Intelligence

Nine pre-trained production ML artifacts trained on 100+ real-world OpenML, PMLB, and UCI datasets:

- **Schema Classifier** (LightGBM, 19.5 MB): Classifies any column into 1 of 31 semantic types using 58 features (30 statistical + 28 NLP embeddings)
- **Drift Autoencoder** (PyTorch MLP, 43.6 KB): Detects multivariate distribution shift using reconstruction error with a learned threshold
- **Anomaly Detector** (IsolationForest, 3.4 MB): Row-level anomaly scoring with AUROC 0.961
- **Confidence Scorer** (Calibrated Ensemble, 946 KB): Produces calibrated confidence probabilities with ECE 0.0225
- **Chart Relevance Scorer** (LightGBM, 2.99 MB): Ranks 7 chart types by dataset suitability
- **Domain Classifier** (RandomForest, 372 KB): Assigns datasets to 1 of 7 regulatory domains
- **PPO Policy Network** (NumPy MLP, 311 KB): Deep RL actor for 8-axis pipeline strategy decisions
- **PPO Value Network** (NumPy MLP, 275 KB): Deep RL critic for advantage estimation

### 🛡 Quality & Compliance

- Dual quality gate architecture (soft QA gate + hard statistical gate)
- 7 parallel validation sub-modules running concurrently per pipeline run
- Banking AML/SAR: BSA $10K transaction threshold, structuring detection, KYC field validation
- Healthcare HIPAA: PHI detection in free text, SSN pattern matching, de-identification flags
- Finance SOX: Basel III Capital Adequacy (8% minimum), net position limits, revenue recognition
- GDPR: PII detection, consent metadata validation, data residency checks
- All violations produce LLM-generated remediation narratives

### 📊 Analytics & Reporting

- Automated EDA with self-contained HTML reports (histograms, correlation heatmaps, box plots)
- Statistical engine: descriptive stats, Pearson/Spearman correlation, simple and multivariate regression
- Power BI-style 10-section accordion dashboard in the React frontend
- LLM-powered narrative reports via Ollama (llama3) or HuggingFace (Mistral-7B)
- SHAP waterfall charts, feature importance bars, calibration reliability diagrams
- KPI sparklines and anomaly density heatmaps

### 🎮 Dual Reinforcement Learning

Two complementary RL systems that together learn the optimal pipeline strategy:

- **Thompson Sampling Bandit** (always-on, no GPU): Beta-Bernoulli posteriors over 3 decision axes, converges in ~80 runs
- **PPO Actor-Critic** (deep RL, post shadow-mode): 8-axis action space, GAE advantage estimation, rollback protection, trained on 1,000 synthetic episodes

### 🏛 Governance & Auditability

- Immutable Bronze/Silver/Gold medallion data layers with SHA-256 checksums
- Append-only JSONL audit log per pipeline run
- Full data lineage tracking (`lineage_id` traceable from Gold back to Bronze)
- JWT-authenticated API with RBAC enforcement
- Prometheus metrics exporter + Grafana dashboard support

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DIPEX v3 SYSTEM OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   INGESTION  │  │  PROCESSING  │  │  VALIDATION  │  │  ANALYTICS   │  │
│  │              │  │              │  │              │  │              │  │
│  │ File Upload  │  │ DataCleaner  │  │ Range / Null │  │ Auto EDA     │  │
│  │ PostgreSQL   │  │ FeatureEng   │  │ Schema /     │  │ Descriptive  │  │
│  │ MongoDB      │  │ NLP Analyzer │  │ Leakage /    │  │ Correlation  │  │
│  │ Kafka        │  │ RobustTriage │  │ Drift /      │  │ Regression   │  │
│  │ REST API     │  │ MissingData  │  │ Anomaly /    │  │ AutoML       │  │
│  │ DuckDB       │  │ TempSplitter │  │ Multicollin  │  │ SHAP         │  │
│  │ Redis        │  │ RLSelector   │  │ ZeroValue    │  │ Narrative    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │                 │           │
│         ▼                 ▼                 ▼                 ▼           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    MEDALLION DATA ARCHITECTURE                      │  │
│  │  [Bronze: Raw Snapshot]──►[Silver: Validated]──►[Gold: Analysis]   │  │
│  │   SHA-256 fingerprinted    Enriched schema       Full lineage_id   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                   DUAL QUALITY GATE SYSTEM                          │  │
│  │  Gate 1 (QA): Q = w_null*(1-null_rate) + w_schema*conformance + …  │  │
│  │  Gate 2 (Hard): p = confidence_scorer(drift+anomaly+gate outcomes)  │  │
│  │  Decision: p≥0.70→PASS | p≥0.55→WARN | p<0.55→FAIL                │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     RL ADAPTATION LAYER                             │  │
│  │  Thompson Sampling Bandit (always-on) + PPO Actor-Critic (deep RL)  │  │
│  │  Updates pipeline strategy based on gate outcomes and model metrics  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Backend (:8000) │ React SPA (:3000) │ Kafka │ Prometheus  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User/System
    │ POST /api/pipeline/run (file + config)
    ▼
UniversalIntake          → validate, format-detect, create Bronze snapshot (SHA-256)
    │
    ▼
SchemaInferencer         → classify each column (31 types, NLP-augmented cascade)
    │
    ▼
RobustTriage             → tiered null handling, zero-inflation, outlier repair
DataCleaner              → coercion, deduplication, boolean normalization
FeatureEngineer          → derived features, log transforms, temporal decomposition
    │
    ▼
7 Parallel Validators    → range, null, schema, leakage, drift, VIF, zero-value
ComplianceEngine         → domain-specific rules (AML/HIPAA/SOX/GDPR)
    │
    ▼
AutoEDA                  → HTML report: distributions, correlations, outlier plots
StatisticsEngine         → descriptive, Pearson/Spearman, regression
    │
    ▼
AutoML Racer             → 4 candidate models, Optuna tuning, SHAP explanations
LeakageDetector          → pre-fit guard: drops |r|≥0.98 features, ID-like columns
Calibrator               → Platt sigmoid scaling if ECE > 0.05
    │
    ▼
Gate 1 (QA)              → quality score Q ∈ [0,1]
Gate 2 (Hard)            → confidence score p ∈ [0,1] → PASS/WARN/FAIL
    │
    ▼
PPOAgent.record_outcome  → update Thompson + PPO posteriors, save checkpoint
AuditWriter              → append immutable record to audit/audit.jsonl
LLMReporter (async)      → generate narrative HTML report via Ollama/HuggingFace
    │
    ▼
PipelineResult           → returned to caller / frontend over WebSocket
```

---

## 🔄 8-Stage Pipeline In Detail

Every pipeline run executes exactly 8 stages in sequence. Each stage exposes a standard `StageResult` interface enabling granular testing and composition.

---

### Stage 1 — Universal Ingestion

**Module:** `ingestion/universal_intake.py` (36 KB)

The `UniversalIntake` class is the single entry point for all data formats and sources. Its responsibilities:

#### Format Auto-Detection

Rather than trusting file extensions (which are frequently wrong), DIPEX inspects file content using a multi-pass heuristic:
1. Magic bytes (Parquet magic, Avro header, ZIP signature for Excel)
2. First 512 bytes parsed as JSON to detect JSON arrays/objects
3. CSV dialect detection (delimiter, quoting, encoding)
4. XML root tag detection

```python
from ingestion.universal_intake import UniversalIntake, SourceConfig

cfg = SourceConfig(
    source_type="file",
    dataset_id="q1_transactions",
    data_mode="batch",
    path="data/transactions.csv",
    file_format="auto",          # auto-detect from content
)
intake = UniversalIntake.from_yaml("config.yaml")
snapshot = intake.ingest(cfg)
# snapshot.df         → cleaned DataFrame
# snapshot.bronze_path → immutable Parquet location
# snapshot.checksum   → SHA-256 hex digest
# snapshot.lineage_id → globally unique run identifier
```

#### Bronze Layer Creation

Before any transformation, `UniversalIntake` writes an immutable Bronze snapshot:

```
data/bronze/<dataset_id>/<timestamp>_bronze.parquet    ← raw data, no changes
data/bronze/<dataset_id>/<timestamp>_bronze.json       ← sidecar: shape, checksum, source metadata
```

The SHA-256 checksum is computed over the raw Parquet bytes and stored in the sidecar. Any downstream system that reads Bronze data first verifies this checksum via `ImmutabilityGuard`.

#### Large File Handling

For files exceeding `large_data.chunk_threshold_mb` (default 128 MB):

```
File → ChunkedParquetReader (100K rows per chunk)
                    │
                    ▼ (for each chunk)
              ChunkedParquetWriter → data/tmp/<chunk_000N>.parquet
                    │
                    ▼ (after all chunks)
            DuckDB UNION ALL → single merged Bronze Parquet
```

Supports up to **50 GB per ingestion job**. If process RSS exceeds `large_data.max_mem_rss_gb` (default 8 GB), the pipeline pauses, flushes current chunks, and continues.

---

### Stage 2 — Schema Detection

**Module:** `ingestion/schema_infer.py` (49 KB — largest single module)

The `NLPAugmentedSchemaClassifier` determines the **semantic meaning** of every column, not just its data type.

#### 3-Stage Cascade

```
Column Name + Column Values
        │
        ├─── Stage 1: Keyword Lexicon (O(1) per column)
        │             Compiled regex patterns for 31 type signatures
        │             email: r'[@].*\.[a-z]{2,}', IBAN: r'[A-Z]{2}\d{2}[A-Z0-9]+'
        │             If high-confidence match (>0.90) → use directly
        │
        ├─── Stage 2: TF-IDF + LogisticRegression on column NAME
        │             Character n-gram (min=2, max=5) vectorizer
        │             Trained on 50,000 real-world column name examples
        │             Provides a distribution over 31 types
        │
        └─── Stage 3: LightGBM on 58 FEATURES from column VALUES
                      30 statistical features + 28 NLP similarity scores
                      Final prediction: argmax of weighted ensemble
```

#### 31 Semantic Types Detected

| Category | Types |
|---|---|
| **Identity** | `id`, `name`, `email`, `phone`, `ssn`, `iban`, `pan_number`, `passport`, `vin`, `mac_address`, `credit_card`, `hash_value` |
| **Numeric** | `age`, `amount`, `percentage`, `score`, `count`, `duration`, `coordinates` |
| **Temporal** | `date` |
| **Categorical** | `category`, `boolean`, `currency_code`, `swift_code`, `zipcode`, `ticker_symbol`, `ip_address`, `url` |
| **Text** | `text`, `address` |
| **Unknown** | `unknown` |

#### 30 Statistical Features

`null_rate` · `unique_rate` · `is_numeric` · `is_string` · `is_datetime` · `mean_val` · `std_val` · `min_val` · `max_val` · `skew_val` · `all_integer` · `max_lt_200` · `max_lt_1` · `all_positive` · `n_distinct` · `email_pattern` · `phone_pattern` · `mean_str_len` · `high_cardinality` · `low_cardinality` · `url_pattern` · `ip_pattern` · `coord_range` · `coord_precision` · `currency_pattern` · `has_negatives` · `zero_fraction` · `mixed_types` · `all_uppercase` · `numeric_string_fraction`

#### 28 NLP Similarity Scores

Sentence-Transformer embeddings (`all-MiniLM-L6-v2`) of the column name are compared against 21 semantic type anchor phrases and 7 domain anchor phrase sets, producing cosine similarity scores used as features by Stage 3.

---

### Stage 3 — Preprocessing

**Module:** `preprocessing/` (13 files, ~200 KB total)

The preprocessing layer is implemented as a series of composable transformers, each operating on the schema-annotated DataFrame:

#### RobustTriage (`robust_triage.py`, 44 KB)

The most complex preprocessing component. Handles three tiers of null severity:

```
Per-column null rate:
  > 90%        → DROP COLUMN ENTIRELY (too sparse to be useful)
  25% – 90%    → Apply medium_null_strategy (ffill | bfill | median | mean)
  < 25%        → Standard imputation (median for numeric, mode for categorical)
```

Additional triage operations:
- **Zero-inflation repair:** Columns with >50% zeros have zeros converted to NaN, then re-imputed using `zero_impute_strategy` (median/mean/mode/KNN)
- **Mixed-type coercion:** Pass 1 — `pd.to_numeric(errors='coerce')`. Pass 2 (regex fallback) — extract numeric prefix from strings like `"42 kg"` or `"$1,200.00"`. If coercion introduces >15% new NaN (configurable via `mixed_type_loss_threshold`), the original string column is preserved
- **Near-zero variance:** Columns with unique_rate < 1/high_cardinality_limit are dropped
- **High cardinality:** String columns with >200 unique values beyond a threshold are hash-bucketed (64 buckets) rather than one-hot encoded
- **Class imbalance:** Detected when majority:minority class ratio ≥ 5.0. Action: SMOTE (oversample synthetic minority samples) or class_weight="balanced" for models

#### DataCleaner (`cleaner.py`, 20 KB)

- **Null imputation** with method selection: `median` (numeric, default), `mode` (categorical, default), `KNN` (columns with high mutual information with target)
- **Outlier handling:** IQR clipping at configurable factor (default 1.5). Values beyond Q1 – 1.5×IQR or Q3 + 1.5×IQR are clipped, not removed
- **Duplicate removal:** Exact-match deduplication → near-duplicate detection via min-hash approximation
- **Boolean normalization:** `yes/no`, `true/false`, `Y/N`, `1/0`, `True/False` → standard Python bool
- **Date coercion:** Attempts `pd.to_datetime` with multiple format strings for columns identified as `date` by schema stage

#### FeatureEngineer (`feature_engineer.py`, 22 KB)

- **Temporal decomposition:** `created_at` → `created_at_year`, `created_at_month`, `created_at_day`, `created_at_dayofweek`, `created_at_is_weekend`, `created_at_quarter`, `created_at_hour`
- **Log transforms:** Auto-applied to numeric columns where `abs(skewness) > auto_log_skew_threshold` (default 1.0)
- **Interaction features:** High-MI numeric column pairs (Pearson |r| < 0.90 to avoid collinearity) generate ratio and product features
- **Polynomial features:** Optional, degree-2, for small feature sets (<20 columns) when enabled

#### MissingDataEngine (`missing_data_engine.py`, 31 KB)

Diagnoses the *type* of missingness before choosing an imputation strategy:
- **MCAR** (Missing Completely At Random): No correlation between missingness and any other variable → median/mean imputation appropriate
- **MAR** (Missing At Random): Missingness correlates with observed variables → model-based imputation (KNN, regression imputation)
- **MNAR** (Missing Not At Random): Missingness depends on the missing value itself → flag and add binary `is_missing_{col}` indicator column

#### NLPColumnAnalyzer (`nlp_column_analyzer.py`, 22 KB)

Uses spaCy (`en_core_web_sm`) to extract semantic context from column names:
- Camel-case splitting: `customerFirstName` → `customer first name`
- Acronym expansion: `amt` → `amount`, `ccy` → `currency`
- Lexical normalization for schema feature extraction

#### TemporalSplitter (`temporal_splitter.py`, 12 KB)

For time-ordered datasets, regular random cross-validation would constitute data leakage (future data used to train, past data used to evaluate). TemporalSplitter implements:
- **Sliding window CV:** Train on window [t, t+k], validate on [t+k, t+k+m]. Window slides forward by m each split
- **Temporal holdout:** Most recent n% of data reserved for final evaluation only

#### RLFeatureSelector (`rl_feature_selector.py`, 10 KB)

When datasets have >100 columns, the RL agent recommends a feature selection strategy:
- **Drift-heavy datasets:** Prefer features with low PSI (stable across time)
- **Quality-heavy datasets:** Prefer features with low null rate and high variance
- **Balanced:** Standard SHAP-based recursive feature elimination

---

### Stage 4 — Validation

**Module:** `validation/` (15 files)

Seven validators run **concurrently** using Python threading. Each validates a specific risk category and returns a list of `ValidationFinding` objects with: `column`, `check_type`, `severity` (WARNING/ERROR/CRITICAL), `value`, `threshold`, and `message`.

See [Validation Engine](#-validation-engine) for full detail on each validator.

---

### Stage 5 — EDA (Exploratory Data Analysis)

**Module:** `eda/auto_eda.py`

Produces a self-contained HTML report (no external dependencies at render time) covering:

- **Distribution plots** for all numeric columns: histogram + KDE overlay + summary statistics
- **Categorical frequency bars** for all string/category columns, sorted by frequency
- **Correlation heatmap** (Pearson for numeric, Cramér's V for categorical)
- **Missing value matrix** (white = present, black = missing, sorted by missingness rate)
- **Outlier box plots** with IQR annotations and flagged outlier counts
- **Class distribution** (for supervised tasks): bar chart of target variable distribution

Output: `reports/eda_<run_id>.html`

---

### Stage 6 — Statistical Analytics

**Module:** `analytics/`

Computes and returns a structured `AnalyticsResult` object:

- **Descriptive statistics:** mean, median, mode, std, variance, min, max, 5th/25th/75th/95th percentiles, kurtosis, skewness, for all numeric columns
- **Correlation analysis:** Pearson matrix + Spearman matrix for numeric columns; Cramér's V matrix for categorical pairs
- **Regression analysis:** Simple OLS for each numeric column vs the target; multivariate OLS for top-10 SHAP features vs target; returns R², RMSE, coefficients, p-values
- **Anomaly density:** Per-column fraction of rows with anomaly scores above threshold, per-region anomaly concentration maps

---

### Stage 7 — AutoML Proposal

**Module:** `proposal/automl.py`

See [AutoML Engine](#-automl-engine) for full detail.

---

### Stage 8 — Verification & Audit

**Module:** `verifier/`

The final stage aggregates all previous results:

1. **Gate 1 evaluation:** Computes QA quality score Q from null rates, schema conformance, anomaly density, and duplicate rate
2. **Gate 2 evaluation:** Runs the `ProposalConfidenceScorer` on 24 pipeline-run features → `p ∈ [0,1]`
3. **Decision:** `PASS` (p ≥ domain_threshold) | `WARN` (p ≥ 0.55) | `FAIL` (p < 0.55)
4. **RL update:** Calls `PPOAgent.record_outcome()` and updates Thompson Sampling posteriors
5. **Audit write:** Appends a complete run record to `audit/audit.jsonl` (append-only, never overwritten)
6. **LLM report** (async, non-blocking): Triggers narrative HTML report generation
7. **Returns:** `PipelineResult` with: gate_decision, confidence_score, anomaly_count, drift_flag, schema_summary, shap_importances, compliance_findings, lineage_id

---

## 🔌 Data Ingestion Layer

### Source Type Reference

```python
from ingestion.universal_intake import SourceConfig, UniversalIntake

# ── File ─────────────────────────────────────────────────────────────────────
cfg = SourceConfig(source_type="file", path="data/sales.csv", file_format="auto")

# ── PostgreSQL ───────────────────────────────────────────────────────────────
cfg = SourceConfig(
    source_type="database", db_type="postgresql",
    connection_string="postgresql://user:pass@host:5432/db",
    table_or_query="SELECT * FROM transactions WHERE date >= '2024-01-01'",
)

# ── MongoDB ──────────────────────────────────────────────────────────────────
cfg = SourceConfig(
    source_type="database", db_type="mongodb",
    connection_string="mongodb://host:27017",
    table_or_query="payments",   # collection name
    mongo_db="dipex_prod",
)

# ── REST API ─────────────────────────────────────────────────────────────────
cfg = SourceConfig(
    source_type="api",
    api_url="https://api.example.com/v1/records",
    api_headers={"Authorization": "Bearer <token>"},
    api_pagination_key="next_cursor",
)

# ── Apache Kafka ──────────────────────────────────────────────────────────────
cfg = SourceConfig(
    source_type="stream",
    kafka_bootstrap="localhost:9092",
    kafka_topic="dipex.raw_events",
    kafka_group_id="dipex-pipeline",
)
```

### Connector Architecture

```
ingestion/connectors/
├── postgres_connector.py    # SQLAlchemy + psycopg2, server-side cursors
├── mongodb_connector.py     # pymongo, batch cursor, schema inference from BSON
├── duckdb_connector.py      # native duckdb, Parquet merge engine
├── redis_connector.py       # redis-py, hash/sorted-set to DataFrame
└── sqlite_connector.py      # sqlite3, lightweight embedded

ingestion/readers/
├── parquet_reader.py        # pyarrow, snappy/gzip/zstd decompression
├── avro_reader.py           # fastavro, schema evolution support
├── feather_reader.py        # pyarrow IPC format
├── xml_reader.py            # lxml, auto-flatten nested elements
└── api_reader.py            # httpx, adaptive retry, pagination loop
```

### Large Data Pipeline

```yaml
large_data:
  max_total_gb: 50              # hard cap per job (OverflowError if exceeded)
  chunk_size_rows: 100_000      # rows per Pandas/DuckDB chunk
  chunk_threshold_mb: 128       # switch to ChunkedParquetWriter above this
  max_mem_rss_gb: 8             # pause & flush if process RSS exceeds this
  tmp_dir: data/tmp             # temp Parquet chunks (cleaned up on success)
  use_duckdb_merge: true        # DuckDB UNION ALL is fastest merge strategy
  parquet_compression: snappy   # chunk compression: snappy | gzip | none
  cleanup_tmp_on_success: true  # auto-remove data/tmp after merge
  kafka_max_messages: 10_000_000
  api_max_pages: 10_000
  db_fetch_chunk_size: 50_000   # server-side cursor batch size
  sample_rows_analytics: 500_000  # max rows for analytics/reporting layer
```

---

## 🏛 Medallion Data Architecture (Bronze/Silver/Gold)

DIPEX implements a strict three-tier immutable data lake architecture:

### Bronze Layer — Raw Immutable Snapshots

```
data/bronze/
└── <dataset_id>/
    ├── <timestamp>_bronze.parquet      ← exact copy of raw data, no transformations
    └── <timestamp>_bronze.json         ← sidecar metadata
        {
          "dataset_id": "q1_transactions",
          "snapshot_id": "snap_20260415_143022",
          "source_type": "file",
          "original_path": "uploads/q1_transactions.csv",
          "row_count": 150000,
          "column_count": 24,
          "sha256": "a3f8b2c1d4e5f6...",    ← computed over Parquet bytes
          "created_at": "2026-04-15T14:30:22Z",
          "ingest_duration_s": 1.34
        }
```

**Immutability guarantee:** `ImmutabilityGuard.verify(path)` recomputes SHA-256 and raises `ChecksumMismatchError` if the file was modified after creation. Called automatically before any Stage 2+ access.

### Silver Layer — Validated & Enriched

```
data/silver/
└── <dataset_id>/
    └── <snapshot_id>_issf.parquet      ← cleaned, type-annotated, schema-enriched
```

Silver artifacts contain all Bronze rows (minus true duplicates) with:
- Correct dtypes per schema classification
- Null values imputed or flagged
- Column-level quality scores as metadata attributes
- SHA-256 checksum (same immutability guarantee)

### Gold Layer — Analysis-Ready

```
data/gold/
└── <dataset_id>/
    └── <operation>_gold.parquet        ← analyst-derived subset/aggregation
```

Gold artifacts are produced by authorized analyst operations (filters, aggregations, joins) on Silver data. Every Gold file includes a `lineage_id` that traces back through Silver to the original Bronze snapshot, providing complete data provenance.

### Verification CLI

```bash
# Verify a specific layer
python main.py layer-verify \
  --dataset-id q1_transactions \
  --snapshot-id snap_20260415_143022 \
  --layer silver
# ✅ Layer VERIFIED: silver/q1_transactions/snap_20260415_143022 
#    checksum=a3f8b2... | shape=(150000, 24) | created=2026-04-15T14:30:22Z

# Verify all layers for a dataset
python main.py layer-verify --dataset-id q1_transactions --all-layers
```

---

## 🤖 ML Models & Artifacts

> **Active Model Version:** Apr 15 2026 @ 22:09 (sourced from `latest_newest_models/`)
> **Runtime Location:** `models/` directory
> **Source Folder:** `latest_newest_models/` (do not delete — source of truth)
> **Smoke Test:** `python check_models.py` (runs real inference on all 6 core models)

### Model Registry Overview

| # | Model File | Size | Date | Architecture | Primary Metric |
|---|---|---|---|---|---|
| 1 | `schema_classifier.pkl` | 19.5 MB | Apr 15 | LightGBM Pipeline | Acc 94.7% |
| 2 | `schema_label_encoder.pkl` | 1.3 KB | Apr 15 | LabelEncoder | — |
| 3 | `schema_feature_registry.pkl` | 2.5 KB | Apr 15 | dict (metadata) | — |
| 4 | `drift_pipeline.pkl` | 43.6 KB | Apr 15 | PyTorch MLP AE | Detection@σ=0.3: 89.4% |
| 5 | `drift_feature_names.pkl` | 0.2 KB | Apr 15 | list | — |
| 6 | `anomaly_detector.pkl` | 3.4 MB | Apr 15 | IsolationForest Pipeline | AUROC 0.961 |
| 7 | `anomaly_threshold.pkl` | 0.4 KB | Apr 15 | dict (threshold=0.0089) | — |
| 8 | `domain_classifier.pkl` | 372 KB | Apr 15 | RandomForest Pipeline | Acc 96.1% |
| 9 | `domain_label_encoder.pkl` | 0.6 KB | Apr 15 | LabelEncoder | — |
| 10 | `domain_registry.pkl` | 1.6 KB | Apr 15 | dict (53 features, 6 domains) | — |
| 11 | `chart_relevance_scorer.pkl` | 2.99 MB | Apr 15 | LightGBM Pipeline (30 feat) | Acc 90.9% |
| 12 | `chart_registry.pkl` | 0.6 KB | Apr 15 | dict (7 chart types) | — |
| 13 | `proposal_confidence.pkl` | 946 KB | Apr 15 | Calibrated VotingClassifier | AUC 0.9784, ECE 0.0225 |
| 14 | `confidence_metadata.json` | 1.6 KB | Apr 15 | JSON (metrics) | — |
| 15 | `rl_ppo_policy.pkl` | 311 KB | Apr 10 | NumPy MLP Actor | Mean Reward ≥ 0.65 |
| 16 | `rl_ppo_value.pkl` | 275 KB | Apr 10 | NumPy MLP Critic | — |

### Training Quality Gates (Applied to All Models)

Before any model artifact is saved, it must pass three quality checks:

| Gate | Threshold | Logic |
|---|---|---|
| Minimum validation score | Model-specific | e.g., schema: balanced_acc ≥ 0.82, confidence: AUC ≥ 0.85 |
| Val–Holdout gap (overfitting) | ≤ 3–5% | gap = max(val_score − hold_score, 0) — penalizes overfitting only |
| CV Standard Deviation | ≤ 3.5–5% | Measures stability across folds |
| Suspiciously perfect | hold < ceiling (0.985–1.01) | Flags possible data leakage |

These gates are enforced in `scripts/train_individual/00_shared_utils.py` (`quality_gate()` function) and post-verified by `utils/training_validator.py`.

---

### Model 1 — Schema Classifier

**File:** `schema_classifier.pkl` + `schema_label_encoder.pkl` + `schema_feature_registry.pkl`

**Architecture: 3-Stage NLP-Augmented Cascade**

```
Column Input
    │
    ├── Stage 1: Regex Lexicon (compiled patterns)
    │   email    → r'[@].*\.[a-z]{2,}'
    │   IBAN     → r'[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,}'
    │   phone    → r'(\+?\d[\d\s\-()]{7,}\d)'
    │   IP addr  → r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    │   URL      → r'https?://[^\s]+'
    │   ...19 more patterns
    │   If match_confidence > 0.90 → return immediately
    │
    ├── Stage 2: TF-IDF + LogisticRegression on column NAME
    │   Vectorizer: CharNGram(min_n=2, max_n=5), 10,000 features
    │   Provides probability distribution across 31 types
    │
    └── Stage 3: LightGBM on 58 FEATURES from column VALUES
        30 statistical + 28 NLP embedding similarity scores
        Final prediction: weighted blend of Stage 2 + Stage 3

sklearn Pipeline:
  steps = [('model', LGBMClassifier(n_estimators=400, max_depth=8))]
```

**Feature Set (58 total):**
- 30 statistical: null_rate, unique_rate, is_numeric, is_string, is_datetime, mean, std, min, max, skew, all_integer, max_lt_200, max_lt_1, all_positive, n_distinct, email_match%, phone_match%, mean_str_len, high_cardinality, low_cardinality, url_match%, ip_match%, coord_range, coord_precision, currency_match%, has_negatives, zero_fraction, mixed_types, all_uppercase, numeric_string_fraction
- 28 NLP: cosine similarity of column name embedding vs 21 semantic type anchors + 7 domain anchors

**Training:** 60+ OpenML + PMLB + UCI datasets × 4 messiness augmentation variants = 500,000+ training samples

**Metrics:**
```
Holdout Accuracy:     94.7%
5-Fold CV Mean:       93.9%
5-Fold CV Std:        1.2%
Val-Holdout Gap:      0.8%  (PASS — no overfitting)
```

---

### Model 2 — Drift Autoencoder

**File:** `drift_pipeline.pkl` + `drift_feature_names.pkl`

**Architecture: PyTorch MLP Autoencoder with BatchNorm**

```python
# Encoder: 20 → 85 → 30
enc.0: Linear(20, 85) + enc.1: BatchNorm1d(85) + ReLU → enc.4: Linear(85, 30)

# Decoder: 30 → 85 → 20
dec.0: Linear(30, 85) + dec.1: BatchNorm1d(85) + ReLU → dec.4: Linear(85, 20)

# Compression ratio: 20 → 30 (latent) — note: h1=85 is hidden width, not latent dim
```

**Drift Pipeline Dict Structure:**
```python
{
  'type':          'autoencoder',
  'input_dim':     20,          # number of statistical features per dataset
  'h1':            85,          # hidden layer width
  'h2':            30,          # latent dimension
  'dropout':       0.1,
  'state_dict':    {...},       # PyTorch weight tensors (enc.0.weight, etc.)
  'scaler':        StandardScaler(),
  'valid_mask':    np.array([...]),  # (20,) bool mask for valid feature columns
  'latent_centroid': np.array([...]),
  'recon_scale':   float,
  'lat_scale':     float,
  'threshold':     0.785,       # MSE threshold for DRIFT / NO-DRIFT decision
  'architecture':  '20-85-30-85-20-bn',
  'feat_names':    [20-element list of feature names],
  'version':       '20260415_2209',
}
```

**Inference at Runtime:**
```python
X_raw = extract_20_features(dataframe)          # 20 numerical statistics
X_scaled = scaler.transform(X_raw)              # StandardScaler
# Forward pass through encoder
h = relu(bn(X_scaled @ W_enc0.T + b_enc0))      # (1, 85)
z = h @ W_enc4.T + b_enc4                       # (1, 30) latent
# Forward pass through decoder
hd = relu(bn(z @ W_dec0.T + b_dec0))            # (1, 85)
rec = hd @ W_dec4.T + b_dec4                    # (1, 20) reconstruction
mse = mean((X_scaled - rec) ** 2)
drift_detected = mse > threshold                 # 0.785
```

**Detection Performance by Shift Magnitude:**

| Σ Shift | Detection Rate | False Positive Rate |
|---|---|---|
| 0.1 (subtle) | 61.3% | 5.0% |
| 0.3 (moderate) | 89.4% | 4.2% |
| 0.5 (clear) | 97.1% | 3.8% |
| 1.0 (severe) | 99.8% | 3.1% |

---

### Model 3 — Anomaly Detector

**File:** `anomaly_detector.pkl` + `anomaly_threshold.pkl`

**Architecture:**
```python
sklearn Pipeline:
  StandardScaler()
  → IsolationForest(n_estimators=200, contamination=0.10, random_state=42)

Threshold dict:
  {'threshold': 0.0089, 'n_features': 20, 'feat_names': [...]}
```

**How It Works:**
IsolationForest builds 200 random isolation trees. Anomaly score = average path length to isolate a sample (shorter path = more anomalous). The learned threshold (0.0089 on the `decision_function` scale) converts continuous scores to binary anomaly/clean labels.

**Training Data:** 60+ real datasets with synthetic row-level corruption:
- Null injection (5–15% of rows per column)
- Outlier injection (2% of rows at 3–10× IQR)
- Sign flips on numeric columns
- Zero runflation (replacing values with 0)

**Metrics:**
```
AUROC:        0.961
Precision@5%FPR: 0.887
F1 ≥ 0.65    (quality gate threshold)
Latency:      1.2 ms per 1,000 rows
```

---

### Model 4 — Proposal Confidence Scorer

**File:** `proposal_confidence.pkl` + `confidence_metadata.json`

**Architecture:**
```python
sklearn Pipeline:
  StandardScaler(n_features_in_=24)
  → CalibratedClassifierCV(
      estimator=VotingClassifier([
          ('lgb', LGBMClassifier(n_estimators=300, ...)),   # weight 40%
          ('rf',  RandomForestClassifier(n_estimators=200)), # weight 35%
          ('lr',  LogisticRegression(C=1.0)),                # weight 25%
      ]),
      cv=4, method='sigmoid'   # Platt scaling
    )
```

**24 Input Features:**

| Feature | Description |
|---|---|
| `anomaly_count` | Total anomalous rows detected |
| `drift_flag` | Binary: drift detected? |
| `quality_score` | Gate 1 composite score |
| `null_rate` | Overall dataset null rate |
| `sample_size_k` | Dataset size in thousands |
| `n_columns` | Number of feature columns |
| `cv_score` | AutoML best CV score |
| `flag_severity_max` | Max validator severity (0–3) |
| `columns_drifted` | Count of drifted columns |
| `proposer_type_enc` | Encoded model family |
| `compliance_penalty` | Sum of compliance penalty weights |
| `n_compliance_violations` | Number of compliance violations |
| `leakage_severity` | Max leakage finding severity |
| `vif_max` | Maximum VIF score detected |
| `zero_inflation_cols` | Columns with >50% zeros |
| `missing_pattern_mnar` | Columns with MNAR missingness |
| `target_is_binary` | Binary vs multiclass task |
| `n_numeric_cols` | Feature composition: numeric |
| `n_categorical_cols` | Feature composition: categorical |
| `n_datetime_cols` | Feature composition: datetime |
| `domain_enc` | Encoded regulatory domain |
| `is_high_stakes` | Banking or healthcare domain? |
| `data_age_days` | Days since dataset creation |
| `retry_count` | Number of pipeline retries for this run |

**Calibration Impact:**

| Stage | ECE | Reliability |
|---|---|---|
| Before calibration | 0.091 | Raw VotingClassifier |
| After Platt scaling | **0.0225** | Calibrated output |
| Improvement | **75.3% reduction** | — |

---

### Model 5 — Chart Relevance Scorer

**File:** `chart_relevance_scorer.pkl` + `chart_registry.pkl`

**Architecture:**
```python
sklearn Pipeline:
  StandardScaler(n_features_in_=30)   # 30 features (23 stat + 7 NLP — use model's n_features_in_)
  → LGBMClassifier(n_estimators=400, max_depth=12, class_weight='balanced')
```

**7 Output Classes (Chart Types):**
`histogram` · `bar` · `scatter` · `line` · `box` · `heatmap` · `pie`

**Key Input Features (30 total):**
- Is numeric? Is categorical? Is datetime? Is high-cardinality?
- Unique rate, null rate, n_columns, n_rows
- Autocorrelation (Ljung-Box p-value) — low p-value → line chart more relevant
- Bimodality coefficient (Sarle's b) — b > 0.555 → histogram/box plot relevant
- Skewness, kurtosis
- NLP domain embeddings: 7 domain similarity scores

**Registry Structure:**
```python
{
  'chart_types': ['histogram', 'bar', 'scatter', 'line', 'box', 'heatmap', 'pie'],
  'n_chart_types': 7,
  'features': [...23 stat feature names...],  # registry has 23, model needs 30
  'label_method': 'statistical',
  'nlp_method': 'sentence_transformers',
  'version': '20260415_2209',
}
```

> **Important:** The LightGBM model expects **30 features** (model.n_features_in_ = 30), while the registry lists 23. Always use `chart_pipe.named_steps['model'].n_features_in_` as the authoritative feature count when constructing inference inputs.

**Metrics:** Holdout Accuracy **90.9%** | CV 91.3% ± 1.8%

---

### Model 6 — Domain Classifier

**File:** `domain_classifier.pkl` + `domain_label_encoder.pkl` + `domain_registry.pkl`

**Architecture:**
```python
sklearn Pipeline:
  StandardScaler(n_features_in_=53)
  → RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced')
```

**7 Output Domains:** `banking` · `healthcare` · `finance` · `ecommerce` · `government` · `insurance` · `generic`

**53 Input Features:** Dataset-level statistical aggregates (column count, row count, numeric/categorical/datetime ratios, mean null rate, mean unique rate, etc.) + 28 NLP domain similarity scores from dataset name and column name ensemble.

**Domain Assignment Impact:** The domain drives which regulatory rule engine is activated and sets higher confidence thresholds (banking: 0.85, healthcare: 0.90 vs default: 0.70).

---

### Models 7 & 8 — PPO RL Agent

**Files:** `rl_ppo_policy.pkl` (311 KB) + `rl_ppo_value.pkl` (275 KB)

See [Reinforcement Learning Engine](#-reinforcement-learning-engine) for full detail.

---

### Training Reports — `models/reports/`

All 7 JSON training reports from the v7 Colab training run are stored in `models/reports/` and read at validation time by `utils/training_validator.py`:

| Report File | Key Fields |
|---|---|
| `schema_classifier_v7_report.json` | `val_bal_acc`, `hold_bal_acc`, `gap`, `cv_std`, `nlp_method`, `_version` |
| `domain_classifier_v7_report.json` | `val_bal_acc`, `hold_bal_acc`, `gap`, `cv_std`, `nlp_method` |
| `drift_autoencoder_v7_report.json` | `train_mse`, `val_mse`, `overfit_ratio`, `threshold` |
| `anomaly_detector_v7_report.json` | `f1`, `auroc`, `threshold_2s`, `contamination` |
| `chart_relevance_scorer_v7_report.json` | `val_bal_acc`, `hold_bal_acc`, `gap`, `cv_std` |
| `confidence_scorer_v7_report.json` | `val_auc_cal`, `gap`, `cv_std`, `ece_after`, `monotone_applied` |
| `post_training_validation.json` | End-to-end v7 summary: all 6 model gate results |

```bash
# Run the full training validator (reads models/reports/*.json)
python -m utils.training_validator .
```

---

## 🏎 AutoML Engine

**Module:** `proposal/automl.py`

### Task Detection & Model Selection

The AutoML engine first infers the task type from the target column:
- Binary if 2 unique values
- Multi-class if 3–20 unique values
- Regression if numeric with >20 unique values

Then races up to 4 model candidates:

| Task | Candidates | Primary Metric |
|---|---|---|
| Binary Classification | LogisticRegression, RandomForest, XGBoost, LightGBM | ROC-AUC |
| Multi-class | LogisticRegression, RandomForest, XGBoost, LightGBM | Weighted Accuracy |
| Regression | Ridge, RandomForest, XGBoost, LightGBM | R² |

### Hyperparameter Tuning

**Primary — Optuna TPE (Tree of Parzen Estimators, 50 trials):**

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth':    trial.suggest_int('max_depth', 3, 12),
        'learning_rate':trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':    trial.suggest_float('subsample', 0.6, 1.0),
        'reg_alpha':    trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':   trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    model = LGBMClassifier(**params, random_state=42)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50, timeout=120)
```

**Fallback — RandomizedSearchCV (20 draws, 3-fold CV):** Used automatically when `optuna` is unavailable.

### Cross-Validation Strategy (RL-Selected)

The `PPOAgent` recommends one of three CV strategies per run:

| Strategy | When Used | Implementation |
|---|---|---|
| `temporal_cv` | Time-ordered data, banking/drift-heavy | Sliding window with gap |
| `stratified_kfold` | Classification with class imbalance | StratifiedKFold(n_splits=5) |
| `kfold` | Balanced regression tasks | KFold(n_splits=5, shuffle=True) |

### SHAP Explanations

After model selection, SHAP values explain every prediction:

```python
import shap

if model_type in ('LGBMClassifier', 'RandomForestClassifier', 'XGBClassifier'):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # exact, fast
elif model_type == 'LogisticRegression':
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)  # fast approximation
```

SHAP summaries stored in `PipelineResult.shap_importances` and rendered as waterfall charts in the dashboard.

### Pre-Fit Leakage Detection

`ModelingLeakageDetector` (`modeling/leakage_detector.py`) runs before every `fit()`:

| Check | Trigger | Severity | Action |
|---|---|---|---|
| ID-like uniqueness | unique_rate ≥ 99% | CRITICAL | Auto-drop from X |
| Target-proximate name | col name in (`outcome`, `result`, `churn`, `default`, `flag`, `label`, `target`) | WARNING | Flag for review |
| High Pearson with target | \|r\| ≥ 0.98 | CRITICAL | Auto-drop from X |
| Elevated Pearson with target | \|r\| ≥ 0.90 | WARNING | Flag for review |
| High Cramér's V (categorical) | V ≥ 0.95 | CRITICAL | Auto-drop from X |
| Elevated Cramér's V | V ≥ 0.85 | WARNING | Flag for review |

All findings recorded in `LeakageReport` appended to the run's audit record.

### Post-Fit Calibration

`Calibrator` (`modeling/calibrator.py`) applies Platt sigmoid or isotonic regression scaling if the raw model's ECE > 0.05:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(base_model, cv=4, method='sigmoid')
calibrated.fit(X_train, y_train)
# ECE improvement: 0.091 → 0.027 typical range
```

---

## 🎮 Reinforcement Learning Engine

DIPEX implements **two complementary RL systems** that together learn optimal pipeline execution strategies:

### System 1 — Thompson Sampling Bandit (Always-On)

**Module:** `learning/rl_agent/agent.py` (shadow mode decision maker)

**Problem Formulation:** At each pipeline run, the system must choose:
- *Which CV strategy to use?* (temporal_cv, stratified_kfold, kfold)
- *How strict should the confidence gate be?* (tight ≥0.70, balanced ≥0.55, loose ≥0.40)
- *What prior to use for the ranker?* (drift_heavy, quality_heavy, balanced)

**Algorithm: Beta-Bernoulli Thompson Sampling**

```
For each arm a in each decision axis:
  Maintain Beta(α_a, β_a) posterior distribution

On each pipeline run:
  1. For each axis: sample θ_a ~ Beta(α_a, β_a) for each arm
  2. Select arm with highest sample: a* = argmax(θ_a)
  3. Execute pipeline with chosen strategy
  4. Observe reward r ∈ {0, 1}
  5. Update: α_{a*} += r, β_{a*} += (1-r)
```

**Why Thompson Sampling?**

- **No hyperparameters:** The Beta-Bernoulli conjugate update is exact — no learning rate to tune
- **Automatic exploration-exploitation:** Early (α=β≈1): Beta(1,1)=Uniform → high variance → explores freely. Late (α>>1, β≈1): Beta(100,2) ≈ spike at 0.98 → barely explores
- **Converges in ~80 runs:** Simulation shows cumulative regret flattens below 2% by run 150

**State Persistence:**
```json
{
  "cv_strategy": {
    "temporal_cv":      {"alpha": 23, "beta": 4},
    "stratified_kfold": {"alpha": 61, "beta": 9},
    "kfold":            {"alpha": 12, "beta": 18}
  },
  "confidence_gate": {
    "tight":   {"alpha": 45, "beta": 7},
    "balanced":{"alpha": 31, "beta": 12},
    "loose":   {"alpha": 8,  "beta": 24}
  },
  "ranker_prior": {
    "drift_heavy":    {"alpha": 19, "beta": 5},
    "quality_heavy":  {"alpha": 28, "beta": 8},
    "balanced":       {"alpha": 35, "beta": 11}
  },
  "total_pulls": 142
}
```

### System 2 — PPO Actor-Critic Agent (Deep RL)

**Module:** `learning/rl_agent/` (9 files)

**Shadow Mode:** For the first 20 real pipeline episodes, the PPO agent operates in shadow mode — it uses Thompson Sampling decisions but observes all state-action-reward transitions. This bootstraps the replay buffer with real data before PPO training begins.

**State Space (12-dimensional):**

```python
state = [
    n_rows / 1_000_000,          # normalized row count
    n_cols / 100,                 # normalized column count
    null_rate,                    # overall null fraction
    anomaly_rate,                 # fraction of anomalous rows
    drift_psi,                   # PSI drift metric
    data_health / 100,           # composite health score
    domain_is_banking,            # binary
    domain_is_healthcare,         # binary
    domain_is_finance,            # binary
    prior_confidence_score,       # last run's confidence
    quarantine_frac,             # fraction of rows quarantined
    retry_count / 5,             # normalized retry count
]
```

**Action Space (8-axis discrete):**

```python
AXES = {
    'cv_strategy':           ['temporal', 'stratified', 'kfold'],
    'cv_folds':              [3, 5, 10],
    'imputation':            ['median', 'knn', 'mice'],
    'outlier_policy':        ['clip', 'quarantine', 'winsorize'],
    'model_complexity':      ['low', 'medium', 'high'],
    'confidence_threshold':  [0.40, 0.55, 0.70, 0.85],
    'retry_budget':          [0, 1, 2, 3],
    'feature_selection':     ['none', 'shap_top20', 'rl_selected'],
}
# Total action combinations: 3×3×3×3×3×4×4×3 = 11,664
```

**Policy Network Architecture:**
```
Input(12) → Linear(12, 64) → ReLU → Linear(64, 32) → ReLU
         → 8 separate head Linear(32, n_arms_per_axis)
         → Softmax per head → sample one arm per axis
```

**PPO Update (every 32 transitions):**
```python
# GAE Advantage Estimation
advantages = gae(rewards, values, gamma=0.99, lam=0.95)

# Clipped Surrogate Objective
ratio = exp(log_prob_new - log_prob_old)
L_clip = mean(min(ratio * A, clip(ratio, 1-ε, 1+ε) * A))  # ε=0.2

# Value Loss
L_value = mean((V(s) - returns)**2)

# Total Loss (maximized via gradient ascent)
Loss = L_clip - 0.5 * L_value + 0.01 * entropy
```

**Rollback Protection:**
```python
ROLLBACK_WINDOW = 5
ROLLBACK_DROP_THRESHOLD = 0.20

# After each update:
recent_mean = mean(last_5_rewards)
drop = (best_reward - recent_mean) / best_reward
if drop > 0.20:
    policy.weights = best_checkpoint_weights  # revert
    logger.warning("Rollback triggered")
```

**Training Statistics (Synthetic Pre-Training):**
- Episodes: 1,000
- 8 scenario types: clean_small, dirty_large, banking_aml, healthcare_phi, high_null, high_drift, ecommerce_fraud, time_series
- Quality gate: eval_mean_reward ≥ 0.65, eval_std ≤ 0.09
- Training curves saved: `models/rl_training_curves.png`

### Reward Signal

```python
reward_components = {
    'pipeline_success': 0.33 if gate_decision in ("PASS", "WARN") else 0.0,
    'model_quality':    0.33 if model_auc >= confidence_threshold else 0.0,
    'data_health':      0.34 * (data_health_score / 100.0),
}
# Adjustment bonuses:
if user_approved_plan:     reward += 0.05   # user confirms pre-analysis plan
if quarantine_frac < 0.02: reward += 0.03   # very few rows quarantined
if retry_count == 0:       reward += 0.05   # no retries needed

reward = clip(sum(components) + noise(0, 0.05), 0, 1)
```

---

## 🔍 Validation Engine

**Module:** `validation/` (15 files + 2 sub-packages)

All seven validators operate in **advisory mode by default** (`advisory_mode: true` in `config.yaml`). They produce structured `ValidationFinding` objects that are aggregated by the quality gates — they never unilaterally halt the pipeline.

### Validator 1 — Range Validator

**File:** `validation/range_validator.py`

Checks that column values fall within expected business ranges:

- **IQR outlier detection:** Values beyond `Q1 – factor×IQR` or `Q3 + factor×IQR` (default factor=1.5) are flagged as outliers. Count and fraction reported.
- **Domain-specific range checks:** For columns identified as `age`: 0–125. For `percentage`: 0–100. For `probability`/`score`: 0–1.
- **Zero-inflation detection:** Columns where zero fraction > `high_zero_threshold` (50%) are flagged with a WARNING — may indicate invalid data encoding
- **Business rule ranges:** Configured per column in `config.yaml` under `validation.range_rules`

### Validator 2 — Null Validator

**File:** `validation/null_validator.py`

- **Per-column null rate** vs. configurable `null_threshold` (default 0.99 — advisory)
- **Required field enforcement:** Columns in `config.hard_gate_1.critical_columns` raise CRITICAL if any null found
- **Null cascade detection:** If nulls in column A are strongly correlated with nulls in column B, flags a potential data join failure

### Validator 3 — Schema Validator

**File:** `validation/schema_validator.py`

- **Type conformance:** Compares actual column dtype vs. ML-inferred semantic type. E.g., a column inferred as `date` containing only integers raises a WARNING
- **Cardinality consistency:** A column inferred as `category` with >1,000 unique values raises a WARNING (possible free-text contamination)
- **Schema drift vs registry:** If a schema registry exists from a previous run, checks for added/removed columns and type changes

### Validator 4 — Leakage Detector

**File:** `validation/leakage_detector.py`

Separate from the pre-fit `ModelingLeakageDetector`. This one runs during validation (Stage 4), before modeling, to give early warning:

- **Pearson |r| ≥ 0.98 with target:** CRITICAL — column likely encodes the target
- **Pearson |r| ≥ 0.90 with target:** WARNING — high correlation, review required
- **Cramér's V ≥ 0.95 (categorical):** CRITICAL
- **ID-like uniqueness:** unique_rate ≥ `id_uniqueness_threshold` (0.99) → CRITICAL
- **Target-proximate name patterns:** Column names matching `_id$|^outcome|^result|^is_churn` → WARNING

### Validator 5 — Drift Detector

**File:** `validation/drift_detector.py`

Uses the `drift_pipeline.pkl` autoencoder to detect distribution shift:

```python
drift_art = joblib.load('models/drift_pipeline.pkl')
X_features = extract_20_statistical_features(df)
X_scaled = drift_art['scaler'].transform(X_features)
# Forward pass through PyTorch AE weights
reconstruction_mse = autoencoder_forward(X_scaled, drift_art['state_dict'])
drift_detected = reconstruction_mse > drift_art['threshold']  # 0.785
severity = 'HIGH' if mse > 2*threshold else 'MODERATE' if mse > threshold else 'LOW'
```

Also computes Population Stability Index (PSI) per column vs. a reference baseline:
- PSI < 0.10 → NO_DRIFT (green)
- PSI 0.10–0.25 → MODERATE_DRIFT (yellow) — warning
- PSI > 0.25 → HIGH_DRIFT (red) — error

### Validator 6 — Multicollinearity Detector

**File:** `validation/multicollinearity_detector.py`

Computes Variance Inflation Factor (VIF) for all numeric features:

```
VIF_j = 1 / (1 - R²_j)   where R²_j = R² of regressing column j on all others
```

| VIF | Severity | Action |
|---|---|---|
| VIF > 10 | ERROR | Flag pair, recommend drop |
| VIF 5–10 | WARNING | Flag pair, suggest review |
| VIF < 5 | OK | No action |

Limited to `validation.multicollinearity.max_features_for_vif` (default 100) to keep computation tractable.

### Validator 7 — Zero Value Detector

**File:** `validation/zero_value_detector.py`

Domain-aware zero analysis specifically designed for financial and healthcare data where zeros are often invalid:
- `amount`/`revenue` columns with >50% zeros → ERROR (likely data encoding issue)
- `age` column with any zeros → WARNING (age=0 is invalid in most business contexts)
- `quantity` columns with >80% zeros → WARNING

---

## 🏛 Regulatory Compliance Engine

**Module:** `validation/regulatory/`

The compliance engine activates after domain classification assigns a regulatory framework. It runs 4 domain-specific rule sets in parallel, each producing `ComplianceViolation` objects with: `domain`, `rule_name`, `severity`, `affected_columns`, `record_count`, and `remediation_hint`.

### Banking — AML/SAR Rules

```yaml
banking:
  aml_amount_column: transaction_amount
  aml_threshold: 10000.0           # US Bank Secrecy Act §5313
  allow_zero_amounts: false
  currency_column: currency
  loan_ratio:
    loan_col: loan_amount
    value_col: collateral_value
    max_ltv: 0.90                  # 90% Loan-to-Value cap
  repayment:
    repayment_col: repayment_amount
    balance_col: outstanding_balance
```

**Rules Enforced:**
1. Transactions ≥ $10,000 without a corresponding SAR record → CRITICAL
2. Structuring detection: clusters of transactions 10–20% below $10,000 → ERROR
3. Round-number clustering: >15% of transactions are exact round numbers → WARNING
4. Missing mandatory AML fields (`transaction_id`, `timestamp`, `counterparty_id`) → WARNING
5. Zero transaction amounts when `allow_zero_amounts=false` → ERROR
6. Loan-to-Value ratio exceeding `max_ltv` → WARNING
7. Repayment amount exceeding outstanding balance → ERROR
8. Missing currency codes for multi-currency datasets → WARNING

### Healthcare — HIPAA Rules

```yaml
healthcare:
  age_column: patient_age
  min_age: 0
  max_age: 125
  diagnosis_columns: [diagnosis_code]
  text_columns_for_phi_scan: [notes, comments, description]
  allowed_phi_columns: []
```

**Rules Enforced:**
1. SSN pattern (`\d{3}-\d{2}-\d{4}`) found in non-SSN columns → CRITICAL
2. Phone numbers detected in free-text columns → ERROR
3. Date-of-birth columns without de-identification annotation → WARNING
4. Patient names detectable via spaCy NER in text fields → WARNING
5. Age values outside [min_age, max_age] range → ERROR
6. Missing ICD-10 diagnosis codes where required → WARNING
7. PHI columns without encryption metadata → WARNING

### Finance — SOX Rules

```yaml
finance:
  revenue_columns: [net_revenue, gross_revenue, segment_revenue]
  capital_adequacy:
    tier1_col: tier1_capital
    rwa_col: risk_weighted_assets
    min_car: 0.08                  # Basel III minimum 8%
  net_position:
    position_column: net_position
    max_long: 1000000
    max_short: 500000
```

**Rules Enforced:**
1. Capital Adequacy Ratio (Tier 1 Capital / RWA) < 8% → CRITICAL
2. Net position exceeding `max_long` or `max_short` → ERROR
3. Revenue recognition: credit memo `is_credit_memo=True` with positive revenue → ERROR
4. Missing audit trail columns (`modified_at`, `modified_by`) → WARNING
5. Negative revenue in non-credit-memo rows → WARNING

### GDPR Rules

```yaml
gdpr:
  residency_column: data_region
  allowed_regions: [EU, EEA, DE, FR, UK]
  consent_column: consent_given
  phi_columns: [patient_id, ssn, date_of_birth, full_name, address, phone, email]
```

**Rules Enforced:**
1. PII columns (`phi_columns`) without `consent_given=True` → CRITICAL
2. Data from non-allowed residency regions → ERROR
3. Missing `data_subject_consent`, `processing_basis`, or `retention_date` metadata → WARNING
4. PII columns without `is_anonymized` flag → WARNING
5. Data subjects with `right_to_erasure=True` still present in dataset → CRITICAL

### Compliance Penalty System

Violations subtract from the pipeline confidence score:

```yaml
compliance:
  penalty_weights:
    critical: 0.20    # per CRITICAL violation
    error:    0.10    # per ERROR violation
    warning:  0.02    # per WARNING violation
  critical_blocks_pipeline: false   # CRITICAL can optionally halt pipeline
  audit_violations: true            # all violations written to audit/compliance.jsonl
  llm_remediation: true             # LLM generates per-violation remediation guidance
  rl_feedback: true                 # violation count feeds back to RL threshold tuner
```

**Example:** A dataset with 2 CRITICAL violations and 3 WARNINGs loses:
`2 × 0.20 + 3 × 0.02 = 0.46` from its confidence score — highly likely to produce a WARN or FAIL gate decision.

---

## ⚙️ Preprocessing & Feature Engineering

### Tiered Null Handling Strategy

```
Column null rate analysis:
┌──────────────────────────────────────────────────────────┐
│ > 90% null   → DROP COLUMN                              │
│               (data/bronze still preserves original)    │
├──────────────────────────────────────────────────────────┤
│ 25%–90% null → Apply medium_null_strategy:              │
│               ffill  : forward-fill (time-series safe)   │
│               bfill  : backward-fill                     │
│               median : median imputation                 │
│               mean   : mean imputation                   │
├──────────────────────────────────────────────────────────┤
│ < 25% null   → Standard imputation:                     │
│               numeric   → median                        │
│               categorical → mode                        │
│               high-MI cols → KNN(n_neighbors=5)         │
└──────────────────────────────────────────────────────────┘
```

### Feature Generation Examples

```python
# Temporal decomposition (auto-applied to 'date' type columns)
df['order_date_year']      = df['order_date'].dt.year
df['order_date_month']     = df['order_date'].dt.month
df['order_date_dayofweek'] = df['order_date'].dt.dayofweek
df['order_date_is_weekend']= df['order_date'].dt.dayofweek.isin([5,6]).astype(int)
df['order_date_quarter']   = df['order_date'].dt.quarter
df['order_date_hour']      = df['order_date'].dt.hour  # if datetime has time component

# Log transforms (auto-applied when |skew| > 1.0)
df['revenue_log'] = np.log1p(df['revenue'])

# Ratio features (for high-MI numeric pairs, |r| < 0.90)
df['revenue_per_unit'] = df['revenue'] / (df['quantity'] + 1e-8)

# SMOTE class balancing (when majority:minority ≥ 5.0)
from imblearn.over_sampling import SMOTE
X_balanced, y_balanced = SMOTE(k_neighbors=5).fit_resample(X_train, y_train)
```

### Configuration

```yaml
preprocessing:
  impute_strategy: median           # median | mean | mode | knn
  scale_strategy: standard          # standard | minmax | robust
  encode_strategy: onehot           # onehot | label | target
  outlier_method: iqr
  outlier_factor: 1.5
  drop_col_null_threshold: 0.90
  handle_class_imbalance: true
  imbalance_ratio_threshold: 5.0
  auto_log_skew_threshold: 1.0
  feature_engineering: true
  high_cardinality_limit: 200
  triage:
    medium_null_lower: 0.25
    medium_null_upper: 0.90
    medium_null_strategy: ffill
    high_zero_threshold: 0.50
    zero_to_nan: true
    zero_impute_strategy: median
    mixed_type_coerce: true
    mixed_type_loss_threshold: 0.15
    coerce_regex_fallback: true
    drop_near_zero_variance: true
    auto_log_transform: true
    skew_threshold: 2.0
    auto_resample: true
    resample_strategy: smote       # smote | oversample | undersample
```

---

## 📊 Analytics & EDA

### Automated EDA

`eda/auto_eda.py` generates a self-contained HTML report (no CDN dependencies):

- **Table of contents** with links to each section
- **Dataset overview:** shape, dtypes, memory usage, ingest timestamp
- **Per-column cards:** For each column — distribution plot, summary stats box, top-5 values, null fraction bar
- **Correlation matrix:** Interactive heatmap (Plotly-embedded), sortable by correlation strength
- **Missing data matrix:** Visual pattern of missingness (missingness correlations revealed)
- **Outlier summary table:** Count + fraction + IQR bounds per flagged column
- **Schema annotation table:** ML-inferred type, confidence, and recommended validation rules

### Statistical Engine

`analytics/` computes:

```python
result = {
    'descriptive': {
        'col_name': {
            'mean': float, 'median': float, 'std': float,
            'min': float, 'max': float, 'skew': float, 'kurt': float,
            'q05': float, 'q25': float, 'q75': float, 'q95': float,
            'null_rate': float, 'unique_rate': float,
        }
    },
    'correlation': {
        'pearson':  pd.DataFrame,   # numeric × numeric
        'spearman': pd.DataFrame,   # numeric × numeric (rank-based)
        'cramers_v':pd.DataFrame,   # categorical × categorical
    },
    'regression': {
        'simple_ols': [{
            'feature': str, 'coef': float, 'r2': float,
            'rmse': float, 'p_value': float,
        }],
        'multivariate_ols': {
            'r2': float, 'adj_r2': float, 'rmse': float,
            'coefficients': {feature: float},
        },
    },
    'anomaly_density': {
        'overall_rate': float,
        'per_column':   {col: float},
        'hot_regions':  [(row_start, row_end, density)],
    },
}
```

---

## 📡 Kafka Streaming Pipeline

### Topic Architecture

```
External Producer
    │ produce JSON events
    ▼
┌─────────────────────┐
│ dipex.raw_events    │  ← inbound: raw data events (JSON per record)
└──────────┬──────────┘
           │ DIPEX Consumer Group
           ▼
┌─────────────────────┐
│ dipex.cleaned       │  ← internal: preprocessed, validated batches
└──────────┬──────────┘
           │
           ▼ (after validation + gate decisions)
┌─────────────────────┐   ┌────────────────────────┐   ┌───────────────────┐
│ dipex.gold_outputs  │   │ dipex.drift_alerts     │   │ dipex.rl_signals  │
│ gate + confidence   │   │ PSI per column         │   │ reward signals    │
│ SHAP importances    │   │ autoencoder MSE        │   │ for bandit update │
└─────────────────────┘   └────────────────────────┘   └───────────────────┘
```

### Window Configuration

```yaml
streaming:
  kafka_bootstrap: kafka:29092
  consumer_group: dipex-pipeline
  topics:
    raw_events:  dipex.raw_events
    cleaned:     dipex.cleaned
    gold_outputs:dipex.gold_outputs
    drift_alerts:dipex.drift_alerts
    rl_signals:  dipex.rl_signals
  window_config:
    tumbling_5m:               # 5-minute non-overlapping windows
      type: tumbling
      size_s: 300
    sliding_1m:                # 30-second advance, 60-second windows
      type: sliding
      size_s: 60
      advance_s: 30
  max_queue_depth: 1000
  late_data_tolerance_s: 30
  consumer_lag_warn_threshold: 1000
  consumer_lag_crit_threshold: 10000
```

### Starting Kafka Mode

```bash
# 1. Start Kafka infrastructure
docker-compose up -d kafka zookeeper schema-registry

# 2. Verify topics are created
docker-compose exec kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 --list

# 3. Start the DIPEX pipeline consumer (processes incoming events)
$env:PYTHONPATH="."
python scripts/start_kafka_pipeline.py

# 4. Produce test events (separate terminal)
python scripts/produce_kafka_test_data.py --continuous --delay 1.0

# 5. Monitor gold outputs (separate terminal)
python scripts/watch_kafka_results.py

# 6. Debug consumer lag
python scripts/debug_kafka_health.py
```

### Security Configuration (`.env`)

```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_GROUP_ID=dipex-pipeline
KAFKA_SECURITY_PROTOCOL=SASL_SSL        # or PLAINTEXT for dev
KAFKA_SASL_MECHANISM=SCRAM-SHA-256
KAFKA_SASL_USERNAME=dipex_user
KAFKA_SASL_PASSWORD=your_strong_password
SCHEMA_REGISTRY_URL=http://localhost:8081
```

---

## 🤖 LLM Integration & Reporting

**Module:** `reporting_service/llm_provider.py`

### Provider Configuration

```yaml
llm:
  provider: ollama                 # ollama | huggingface
  model: llama3                   # or: mistralai/Mistral-7B-Instruct-v0.2
  max_prompt_tokens: 3000
  fallback_on_error: true          # never block pipeline on LLM failure
  pii_redaction: true              # strip PII from data before sending
  audit_prompts: true              # log all prompts/completions
  cost_tracking: true
```

### HuggingFace Configuration

```bash
# .env
LLM_PROVIDER=huggingface
HF_API_KEY=hf_...
HF_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
HF_FALLBACK_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### What the LLM Produces

For each pipeline run, the LLM generates a multi-section HTML narrative report:

1. **Executive Summary** — 3-paragraph plain-English overview of the dataset and key findings
2. **Data Quality Assessment** — narrative explanation of quality issues, their root causes, and business impact
3. **Schema Insights** — per-column type surprises, recommended validation rules
4. **Anomaly Analysis** — what the anomalies look like, potential causes, recommended investigation
5. **Drift Analysis** — how the dataset has shifted, which features are most unstable
6. **Compliance Summary** — plain-English explanation of each violation and its regulatory significance
7. **Remediation Roadmap** — prioritized list of actions to take before using this data
8. **Model Recommendation** — which AutoML model was selected and why, confidence in the recommendation

**PII Redaction:** Before any data is sent to the LLM, all columns identified as PHI/PII (by domain classifier + regex patterns) are replaced with `[REDACTED]`. Only aggregate statistics (means, std, null rates) are included in the prompt.

---

## 🖥 Frontend Dashboard

**Tech Stack:** React 18, Vite (build tool), JSX, Vanilla CSS (component-scoped)
**Port:** `http://localhost:3000`
**State Management:** Custom `PipelineSessionContext` (React Context API)

### Page 1 — Run Pipeline (`RunPipeline.jsx`, 177 KB)

The primary interface for data upload and analysis. Organized into 4 phases:

**Phase 1: Data Source Selection**

Four ingestion mode tabs, each with mode-specific configuration:

- **📁 File Upload:** Drag-and-drop zone supporting CSV, Excel, JSON, Parquet, Avro, XML. Row range filter (`rowRange`) and column range filter (`colRange`) allow uploading only a slice of a large file. Live preview shows first 10 rows before submission.
- **🗄 Database:** Connection string builder for PostgreSQL (host/port/db/user/pass), MongoDB (URI), DuckDB (file path). Table name or custom SQL query input. Test connection button.
- **🌊 Kafka:** Topic selector dropdown, consumer group ID input, offset reset control (latest/earliest), lag meter showing current consumer lag in real time via WebSocket.
- **🌐 REST API:** Endpoint URL, HTTP method, auth header builder (Bearer/Basic/API Key), pagination config (cursor field, max pages).

**Phase 2: Intelligence Hub**

Triggered automatically after data is ingested. Shows:

- Pre-analysis planning panel: estimated processing time, schema preview (column count, type distribution pie chart)
- Active regulatory domain badges (Banking, Healthcare, Finance, GDPR — shown always when active, even with no violations)
- Analyst instruction hints: mode-specific guidance (e.g., "For Kafka: ensure consumer group ID matches your environment", "For API: verify auth token expiry")
- Column type distribution heatmap

**Phase 3: Pipeline Execution Progress**

Real-time stage-by-stage progress bar updating via WebSocket (`ws://localhost:8000/ws/{run_id}`):

```
[■■■■■■■■□□] Stage 4/8: Validation (Range Validator) — 2.1s elapsed
  → 3 findings: 1 WARNING (null_rate), 2 INFO (outliers)
```

**Phase 4: Results Panel (10-Section Accordion)**

| Section | Content |
|---|---|
| 1. Executive Summary | Gate decision badge, confidence gauge, quality score, anomaly count, key metrics |
| 2. Schema Analysis | 31-type column type breakdown table, confidence per column, type distribution chart |
| 3. Data Quality | Null rate per column bar chart, outlier density, zero-inflation flags, duplicate count |
| 4. Anomaly Detection | Row-level anomaly heatmap, score distribution, top-10 most anomalous rows |
| 5. Data Drift | Feature-wise PSI bars (green/yellow/red), autoencoder MSE gauge, drift timeline |
| 6. Compliance Report | Violation table with domain badge, severity, affected columns, remediation text |
| 7. AutoML Results | Model comparison table (4 candidates), winner highlight, CV score chart |
| 8. SHAP Explanations | Waterfall chart (top-15 features), importance bar chart, feature-value scatter |
| 9. Statistical Analysis | Correlation heatmap, distribution histograms, regression summary |
| 10. Audit Trail | Full run record JSON viewer, lineage_id, Bronze/Silver/Gold checksums |

### Page 2 — Analytics (`Analytics.jsx`, 93 KB)

- Historical pipeline run list with multi-filter (date range, gate decision, domain, dataset_id)
- KPI sparklines for 4 metrics over time: average confidence score, anomaly rate, drift detections, gate pass rate
- Model performance comparison table across multiple runs on the same dataset
- Export panel: download full run data as CSV, JSON, or Parquet

### Page 3 — API Docs (`ApiDocs.jsx`, 9 KB)

- Auto-generated from OpenAPI schema
- Live request builder with response preview
- Authentication test panel

### WebSocket Real-Time Integration

```javascript
// Established automatically when pipeline run starts
const ws = new WebSocket(`ws://localhost:8000/ws/${run_id}`);

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  // update.stage_num: 1-8
  // update.stage_name: "Validation"
  // update.status: "running" | "complete" | "error"
  // update.duration_s: float
  // update.findings_count: int
  updateProgressBar(update);
};
```

---

## 📚 API Reference

**Base URL:** `http://localhost:8000`
**Auth:** Bearer JWT (optional in dev; required when `DIPEX_AUTH_STRICT=true`)
**Rate Limit:** 120 requests/minute, burst: 20

### Pipeline Endpoints

```http
POST /api/pipeline/run
Content-Type: multipart/form-data

file=@data/sales.csv
target_col=churn
domain=banking           # optional; auto-detected if omitted
analyst_instructions=... # optional natural language guidance
```

```json
{
  "run_id": "a3f8b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
  "status": "PASS",
  "confidence_score": 0.847,
  "quality_score": 0.912,
  "anomaly_count": 23,
  "anomaly_rate": 0.0015,
  "drift_flag": false,
  "drift_severity": "LOW",
  "schema_summary": {"id": 2, "amount": 4, "date": 1, "category": 3, "boolean": 2},
  "proposed_model": "LightGBM",
  "cv_auc": 0.921,
  "shap_top5": [["balance", 0.34], ["age", 0.21], ...],
  "compliance_violations": [],
  "domain": "banking",
  "lineage_id": "lin_20260415_143022_a3f8",
  "run_duration_s": 7.4
}
```

```http
POST /api/pipeline/simple-run
Content-Type: multipart/form-data
# Simplified endpoint: source_kind=live (Kafka) or file
```

```http
GET /api/results/
?page=1&page_size=20
&status=PASS,WARN
&domain=banking
&date_from=2026-04-01
&date_to=2026-04-15
```

```http
GET /api/results/{run_id}
# Full PipelineResult including SHAP values, compliance detail, audit record
```

```http
GET /api/stats/
# Aggregate statistics across all runs:
# { total_runs, pass_rate, avg_confidence, avg_anomaly_rate, domain_breakdown, ... }
```

### Data Endpoints

```http
POST /api/ingest/
POST /api/ingest/v2/          # enhanced: includes pre-analysis planning response
POST /api/preprocess/         # preprocessing only, returns cleaned DataFrame
GET  /api/explorer/           # data profiling and exploration tools
POST /api/exports/            # export result as CSV/JSON/Parquet
```

### Report & Audit

```http
POST /api/report/generate
{ "run_id": "a3f8b2c1-..." }
# Triggers async LLM narrative report generation
# Returns: { "report_path": "reports/report_a3f8.html" }

GET /api/audit/
?run_id=a3f8b2c1-...
&date_from=2026-04-01
# Browse append-only audit trail

GET /api/analytics/{run_id}
# Retrieve full analytics snapshot (descriptive stats, correlation, regression)
```

### WebSocket

```
WS /ws/{run_id}
# Real-time stage progress during pipeline execution
# Messages: { stage_num, stage_name, status, duration_s, findings_count }
```

### Feedback & Cohort

```http
POST /api/feedback/          # Submit analyst feedback on a run
GET  /api/cohort/            # Cohort analysis across run groups
GET  /api/run/               # Alias for /api/results/
```

---

## 🚀 Quick Start

> **Step 0 is mandatory** on first install — it sets up Python dependencies, downloads the spaCy language model, and creates all required runtime directories.

### Step 0 — First-Time Setup

```bat
:: Windows
setup.bat
```

```bash
# Linux / macOS
chmod +x setup.sh && ./setup.sh
```

The setup script runs:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm    # required for NLP column analysis
mkdir -p data/bronze data/silver data/gold \
         data/uploads data/snapshots data/tmp \
         audit reports models output
```

---

### Option A — Docker Compose (Recommended)

Starts the complete stack: API + Frontend + Kafka + Schema Registry + demo databases + Prometheus + Grafana.

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View API logs
docker-compose logs -f dipex-api

# Access points:
# API:            http://localhost:8000
# Frontend:       http://localhost:3000
# API Docs:       http://localhost:8000/docs
# Kafka UI:       http://localhost:8080
# Grafana:        http://localhost:3001  (admin/admin)
# Prometheus:     http://localhost:9090
```

---

### Option B — Local Development

```bash
# Terminal 1 — Backend (API server with hot-reload)
$env:PYTHONPATH="."           # Windows PowerShell
# export PYTHONPATH="."       # Linux/macOS
uvicorn api.app:app --reload --port 8000

# Terminal 2 — Frontend (Vite dev server with HMR)
cd frontend
npm install
npm run dev
# Opens: http://localhost:3000
```

---

### Option C — CLI (10 Sub-Commands)

```bash
$env:PYTHONPATH="."

# ── Full Pipeline ─────────────────────────────────────────────────────────────
python main.py run \
  --source data/sales.csv \
  --target churn \
  --domain banking

# ── Preprocess Only ───────────────────────────────────────────────────────────
python main.py preprocess \
  --source data/raw.csv \
  --target revenue \
  --output data/clean.csv

# ── Statistical Analysis Only ─────────────────────────────────────────────────
python main.py stats \
  --source data/sales.csv \
  --target churn \
  --output reports/stats.json

# ── SQL Query via DuckDB ──────────────────────────────────────────────────────
python main.py query \
  --source data/sales.csv \
  --sql "SELECT region, COUNT(*), SUM(amount) FROM df GROUP BY region ORDER BY 2 DESC"

# ── Generate Report for Past Run ──────────────────────────────────────────────
python main.py report --run-id a3f8b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c

# ── Universal Intake (Any Source → Full Pipeline) ──────────────────────────────
python main.py intake \
  --source-type file \
  --path data/transactions.parquet \
  --format parquet \
  --target fraud_flag \
  --dataset-id transactions_q1

# ── Batch Ingest Multiple Files ───────────────────────────────────────────────
python main.py batch-ingest \
  --sources data/jan.csv,data/feb.csv,data/mar.csv \
  --target revenue \
  --batch-mode incremental

# ── Verify Data Layer Integrity ───────────────────────────────────────────────
python main.py layer-verify \
  --dataset-id transactions_q1 \
  --snapshot-id snap_20260415_143022 \
  --layer silver

# ── Validate All Layers for a Dataset ────────────────────────────────────────
python main.py layer-verify \
  --dataset-id transactions_q1 \
  --all-layers

# ── Health Check ─────────────────────────────────────────────────────────────
python main.py health
```

---

### Option D — Model Verification

```bash
# Functional smoke test (runs real inference on all 6 core models)
$env:PYTHONIOENCODING="utf-8"
python check_models.py

# Expected output:
# [PASS]  schema_classifier         n_features=58  predicted="boolean"  n_classes=20
# [PASS]  domain_classifier         n_features=53  predicted="healthcare"  n_domains=6
# [PASS]  drift_autoencoder         input_dim=20  h1=85  h2=30  threshold=0.785
# [PASS]  anomaly_detector          n_features=20  threshold=0.0089  sample_score=0.0105
# [PASS]  chart_relevance_scorer    n_features=30  n_chart_types=7  predicted="histogram"
# [PASS]  proposal_confidence       n_features=24  ECE=0.0225  AUC=0.9784
# ALL 6 MODELS ARE WORKING AND INTEGRATED.

# Full training validator (reads models/reports/*.json quality gate reports)
python -m utils.training_validator .
```

---

### Option E — Demo Scripts

```bash
$env:PYTHONPATH="."

# All 4 source types, all at once
python scripts/demo_all_paths.py

# Individual source demos
python scripts/demo_01_postgres.py    # PostgreSQL ingestion
python scripts/demo_02_mongodb.py     # MongoDB ingestion
python scripts/demo_03_kafka.py       # Kafka streaming
python scripts/demo_04_api.py         # REST API ingestion

# Run all sources in batch
python scripts/run_all_sources.py
```

---

## ⚙️ Configuration Reference

All configuration lives in `config.yaml`. Environment-specific values and secrets live in `.env` and override `config.yaml` at runtime.

### `config.yaml` — Full Annotated Reference

```yaml
environment: development          # development | staging | production

# ── Medallion Data Layer Paths ────────────────────────────────────────────────
data_layers:
  bronze_dir: data/bronze         # raw immutable snapshots
  silver_dir: data/silver         # validated, enriched
  gold_dir: data/gold             # analyst-ready exports
  model_registry: data/model_registry
  audit_dir: audit                # append-only audit JSONL

# ── Pipeline Execution ────────────────────────────────────────────────────────
pipeline:
  domain: default                 # default | banking | healthcare | finance | gdpr
  auto_stage_timeout_s: 300       # max wall-clock seconds per stage
  confidence:
    threshold: 0.7                # Gate 2 minimum confidence → PASS
    domain_thresholds:
      default: 0.7
      banking: 0.85               # higher stakes → tighter gate
      healthcare: 0.9
      finance: 0.8
  retry:
    max_retries: 3
    backoff_base_s: 2             # exponential backoff base
    escalate_after: 3             # raise exception after N retries

# ── Gate 1 (QA Gate) ──────────────────────────────────────────────────────────
hard_gate_1:
  max_null_rate: 0.99             # advisory: only warns, never halts
  critical_columns: []            # columns that MUST NOT be null
  allow_schema_drift: true        # first-time datasets allowed
  regulatory_domain: default
  advisory_mode: true             # CRITICAL: gate flags but NEVER halts

# ── Gate 2 (Hard Statistical Gate) ───────────────────────────────────────────
hard_gate_2:
  min_sample_size: 30             # minimum rows for statistical validity
  drift_psi_threshold: 0.2        # PSI above this → HIGH_DRIFT
  stability_cv_threshold: 0.3     # CV of column stability scores

# ── Validation Engine ─────────────────────────────────────────────────────────
validation:
  strict_mode: false
  advisory_mode: true
  multicollinearity_threshold: 10.0
  null_threshold: 0.99
  leakage:
    correlation_hard_threshold: 0.98   # auto-drop
    correlation_warn_threshold: 0.90
    cramers_v_hard_threshold: 0.95
    cramers_v_warn_threshold: 0.85
    id_uniqueness_threshold: 0.99
    drop_critical: true
  multicollinearity:
    vif_error_threshold: 10.0
    vif_warn_threshold: 5.0
    corr_hard_threshold: 0.95
    drop_on_error: true
    max_features_for_vif: 100

# ── Regulatory Rule Engine ────────────────────────────────────────────────────
  regulatory:
    domains: [banking]
    halt_on_critical: false
    conflict_resolution: strictest_wins
    banking:
      amount_columns: [transaction_amount, loan_amount, fee_amount]
      allow_zero_amounts: false
      aml_amount_column: transaction_amount
      aml_threshold: 10000.0
    healthcare:
      age_column: patient_age
      min_age: 0
      max_age: 125
      text_columns_for_phi_scan: [notes, comments, description]
    finance:
      capital_adequacy:
        tier1_col: tier1_capital
        rwa_col: risk_weighted_assets
        min_car: 0.08
    gdpr:
      residency_column: data_region
      allowed_regions: [EU, EEA, DE, FR, UK]
      consent_column: consent_given

# ── Compliance Penalties ──────────────────────────────────────────────────────
compliance:
  penalty_weights:
    critical: 0.20
    error: 0.10
    warning: 0.02
  critical_blocks_pipeline: false
  audit_violations: true
  allowed_warning_count: 10
  llm_remediation: true
  rl_feedback: true

# ── Reinforcement Learning ────────────────────────────────────────────────────
rl:
  epsilon_min: 0.05
  epsilon_max: 0.3
  drift_epsilon_boost: 0.3
  ewc_lambda: 0.9
  instability_threshold: -0.1
  instability_window: 3
  sandbox_mode: false             # true → RL decisions not persisted
  max_episodes: 500

# ── LLM Integration ───────────────────────────────────────────────────────────
llm:
  provider: ollama                # ollama | huggingface
  model: llama3
  max_prompt_tokens: 3000
  fallback_on_error: true
  pii_redaction: true
  audit_prompts: true
  cost_tracking: true

# ── Kafka Streaming ───────────────────────────────────────────────────────────
streaming:
  kafka_bootstrap: kafka:29092
  consumer_group: dipex-pipeline
  max_queue_depth: 1000
  late_data_tolerance_s: 30
  consumer_lag_warn_threshold: 1000
  consumer_lag_crit_threshold: 10000

# ── Large Data Handling ───────────────────────────────────────────────────────
large_data:
  max_total_gb: 50
  chunk_size_rows: 100_000
  chunk_threshold_mb: 128
  max_mem_rss_gb: 8
  tmp_dir: data/tmp
  use_duckdb_merge: true
  parquet_compression: snappy
  cleanup_tmp_on_success: true
  sample_rows_analytics: 500_000

# ── API Server ────────────────────────────────────────────────────────────────
api:
  host: 0.0.0.0
  port: 8000
  reload: false
  cors_origins: [http://localhost:3000, http://localhost:8080]
  rate_limit_rpm: 120
  rate_limit_burst: 20
  jwt_expire_mins: 60
  jwt_refresh_hours: 24

# ── Security ──────────────────────────────────────────────────────────────────
security:
  encryption_at_rest: false
  audit_access_log: true
  pii_detection: true
  pii_mask_char: '***'
  rbac_enforce: true

# ── Monitoring ────────────────────────────────────────────────────────────────
monitoring:
  prometheus_scrape_interval: 15s
  grafana_port: 3001
  thresholds:
    pipeline_failure_rate: 0.2
    confidence_median_min: 0.7
    kafka_lag_warn: 1000
    kafka_lag_crit: 10000
    llm_tokens_per_hr: 500000
```

### Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `huggingface` |
| `HF_API_KEY` | — | HuggingFace API key |
| `HF_MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.2` | Primary HF model |
| `HF_FALLBACK_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Fallback HF model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker |
| `KAFKA_GROUP_ID` | `dipex-pipeline` | Consumer group ID |
| `KAFKA_SECURITY_PROTOCOL` | `PLAINTEXT` | `PLAINTEXT` or `SASL_SSL` |
| `KAFKA_SASL_MECHANISM` | — | `SCRAM-SHA-256` etc. |
| `KAFKA_SASL_USERNAME` | — | Kafka SASL username |
| `KAFKA_SASL_PASSWORD` | — | Kafka SASL password |
| `DIPEX_AUTH_STRICT` | `false` | Enforce JWT on all API endpoints |
| `JWT_SECRET_KEY` | auto-generated | Must set explicitly in production |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `MONGO_URI` | — | MongoDB connection URI |
| `REDIS_URL` | — | Redis connection URL |
| `DUCKDB_PATH` | `data/dipex.duckdb` | DuckDB file path |

---

## 🧪 Testing

DIPEX has a comprehensive `pytest` test suite covering unit tests, integration tests, and model smoke tests.

### Running Tests

```bash
$env:PYTHONPATH="."

# ── Recommended: Core tests (no external infrastructure needed) ───────────────
python -m pytest tests/ --ignore=tests/legacy/ -m "not integration" -v

# ── Kafka health tests (mocked — no Kafka broker needed) ──────────────────────
python -m pytest tests/test_kafka_health.py -v

# ── Full suite (requires: docker-compose up -d kafka) ─────────────────────────
python -m pytest tests/ --ignore=tests/legacy/ -v

# ── Legacy security and RBAC tests ────────────────────────────────────────────
python -m pytest tests/legacy/test_security.py -v

# ── Complete test suite ───────────────────────────────────────────────────────
python -m pytest tests/ -v

# ── With HTML coverage report ─────────────────────────────────────────────────
python -m pytest tests/ --ignore=tests/legacy/ -m "not integration" \
  --cov=. \
  --cov-report=html:reports/coverage \
  --cov-report=term-missing

# ── Model functional smoke test ───────────────────────────────────────────────
python check_models.py

# ── Training validator (reads models/reports/*.json) ──────────────────────────
python -m utils.training_validator .

# ── Pipeline integration test (requires running API) ──────────────────────────
python scripts/test_pipeline_integration.py

# ── Legacy triage tests ───────────────────────────────────────────────────────
python scripts/triage_legacy_tests.py
```

### Test Suite Coverage

| Test Scope | Files | Tests | Status |
|---|---|---|---|
| Core unit tests | `tests/test_*.py` | 320 | All passing |
| Integration tests | `tests/test_*_integration.py` | 114 | All passing |
| Kafka (mocked) | `tests/test_kafka_health.py` | 23 | All passing |
| Legacy security | `tests/legacy/test_security.py` | 40 | All passing |
| **Total** | — | **497** | **All passing** |

### Model Smoke Test Output

```
======================================================================
  APR 15 @ 22:09 -- MODEL FUNCTIONAL SMOKE TEST
======================================================================
  6/6 models operational

  [PASS]  schema_classifier
           n_features=58  predicted="boolean"  n_classes=20

  [PASS]  domain_classifier
           n_features=53  predicted="healthcare"  n_domains=6

  [PASS]  drift_autoencoder
           input_dim=20  h1=85  h2=30  threshold=0.785  n_feat_names=20

  [PASS]  anomaly_detector
           n_features=20  threshold=0.0089  flagged=0/5  sample_score=0.0105

  [PASS]  chart_relevance_scorer
           n_features=30  n_chart_types=7  predicted="histogram"

  [PASS]  proposal_confidence
           n_features=24  confidence=0.0121  ECE=0.0225  AUC=0.9784

======================================================================
  ALL 6 APR-15 MODELS ARE WORKING AND INTEGRATED.
======================================================================
```

---

## 🐳 Production Deployment

### Full Stack Deployment

```bash
# Build all Docker images and start services
docker-compose up -d --build

# Verify all services are healthy
docker-compose ps
# Should show: dipex-api, dipex-frontend, kafka, zookeeper,
#              schema-registry, kafka-ui, prometheus, grafana
#              all with status "healthy" or "running"

# Confirm API is responding
curl http://localhost:8000/health
# {"status": "healthy", "version": "3.0.0", "models_loaded": 6}

# Confirm models are loaded
python check_models.py
```

### Pre-Production Checklist

| Step | Action | Command |
|---|---|---|
| 1 | Set JWT secret | `JWT_SECRET_KEY=<32+-char random>` in `.env` |
| 2 | Enable auth enforcement | `DIPEX_AUTH_STRICT=true` in `.env` |
| 3 | Configure LLM backend | Set `LLM_PROVIDER` + credentials in `.env` |
| 4 | Configure Kafka TLS | `KAFKA_SECURITY_PROTOCOL=SASL_SSL` + credentials |
| 5 | Set database URLs | `DATABASE_URL`, `MONGO_URI`, `REDIS_URL` in `.env` |
| 6 | Run model smoke test | `python check_models.py` — all 6 must PASS |
| 7 | Run training validator | `python -m utils.training_validator .` |
| 8 | Verify layer integrity | `python main.py layer-verify --all-layers` |
| 9 | Run API health check | `curl http://localhost:8000/health` |
| 10 | Run integration tests | `python scripts/test_pipeline_integration.py` |

### Scaling Considerations

- **API:** Stateless FastAPI workers behind nginx reverse proxy (see `scripts/nginx.conf`)
- **Kafka:** Increase `streaming.max_queue_depth` and consumer group replicas for high-throughput
- **Large Data:** Increase `large_data.max_mem_rss_gb` and `large_data.chunk_size_rows` based on available RAM
- **LLM:** For high-volume report generation, use HuggingFace Inference Endpoints vs. local Ollama

---

## 🔒 Security & Governance

### Authentication & Authorization

```yaml
api:
  jwt_expire_mins: 60
  jwt_refresh_hours: 24

security:
  rbac_enforce: true
  audit_access_log: true    # every API call logged to audit/access.log
  pii_detection: true       # auto-detect and mask PII in API responses
  pii_mask_char: '***'
  encryption_at_rest: false # enable for regulated deployments
```

### Data Immutability

Bronze and Silver layer artifacts are SHA-256 fingerprinted at creation. Any access that fails the integrity check raises `ChecksumMismatchError`:

```python
from ingestion.immutability_guard import ImmutabilityGuard, ChecksumMismatchError

guard = ImmutabilityGuard()
try:
    guard.verify("data/bronze/sales/snap001_bronze.parquet")
    # Recomputes SHA-256, compares to sidecar JSON
except ChecksumMismatchError as e:
    # File was tampered with — block all downstream processing
    audit_writer.log_tamper_event(str(e))
    raise
```

### Audit Trail

Every pipeline run produces an append-only JSONL record:

```json
{
  "run_id": "a3f8b2c1-...",
  "timestamp": "2026-04-15T14:30:22Z",
  "dataset_id": "q1_transactions",
  "lineage_id": "lin_20260415_143022_a3f8",
  "gate_decision": "PASS",
  "confidence_score": 0.847,
  "schema_summary": {"id": 2, "amount": 4},
  "anomaly_count": 23,
  "drift_flag": false,
  "compliance_violations": [],
  "proposed_model": "LightGBM",
  "cv_auc": 0.921,
  "bronze_checksum": "a3f8b2c1...",
  "silver_checksum": "b4f9c3d2...",
  "user_id": "analyst@company.com",
  "pipeline_version": "3.0.0"
}
```

---

## 📈 Monitoring & Observability

### Prometheus Metrics

DIPEX exposes metrics at `/metrics` (Prometheus scrape format):

```
dipex_pipeline_runs_total{status="PASS|WARN|FAIL"} 142
dipex_pipeline_duration_seconds{quantile="0.5"} 7.4
dipex_anomaly_rate{} 0.0015
dipex_confidence_score{quantile="0.5"} 0.847
dipex_drift_detections_total{severity="LOW|MODERATE|HIGH"} 5
dipex_compliance_violations_total{domain="banking",severity="WARNING"} 12
dipex_kafka_consumer_lag{topic="dipex.raw_events"} 0
dipex_llm_tokens_used_total{provider="ollama"} 45000
```

### Alert Rules (`monitoring/alert_rules.yml`)

| Alert | Condition | Severity |
|---|---|---|
| `HighPipelineFailureRate` | failure_rate > 20% | CRITICAL |
| `LowConfidenceMedian` | median_confidence < 0.70 | WARNING |
| `HighKafkaLag` | consumer_lag > 1,000 | WARNING |
| `CriticalKafkaLag` | consumer_lag > 10,000 | CRITICAL |
| `HighLLMTokenUsage` | tokens/hr > 500,000 | WARNING |
| `RetryEscalations` | retry_escalations/hr > 5 | WARNING |

### Grafana Dashboard

Access at `http://localhost:3001` (admin / admin). Pre-configured panels:
- Pipeline run rate and success rate over time
- Confidence score distribution histogram
- Anomaly rate trend
- Kafka consumer lag per topic
- LLM token usage per provider
- Gate decision breakdown (PASS/WARN/FAIL pie chart)

---

## 📁 Complete Project Structure

```
dipex_project/
│
├── api/                              # FastAPI REST backend
│   ├── app.py                        # App factory, middleware, CORS, startup
│   ├── metrics.py                    # Prometheus /metrics endpoint
│   ├── preview_plan.py               # Pre-analysis planning API
│   ├── middleware/                   # Rate limiter, JWT auth, request logger
│   └── routes/                       # 17 route modules
│       ├── pipeline_run.py           # POST /api/pipeline/run (main endpoint)
│       ├── results.py                # GET /api/results/
│       ├── analytics.py              # GET /api/analytics/{run_id}
│       ├── ingest.py                 # POST /api/ingest/
│       ├── ingest_v2.py              # POST /api/ingest/v2/ (enhanced)
│       ├── preprocess.py             # POST /api/preprocess/
│       ├── report.py                 # POST /api/report/generate
│       ├── audit.py                  # GET /api/audit/
│       ├── feedback.py               # POST /api/feedback/
│       ├── stats.py                  # GET /api/stats/
│       ├── exports.py                # POST /api/exports/
│       ├── explorer.py               # GET /api/explorer/
│       ├── cohort.py                 # GET /api/cohort/
│       ├── run.py                    # GET /api/run/
│       ├── analyst.py                # POST /api/analyst/
│       ├── instruction_parser.py     # POST /api/instructions/parse
│       └── __init__.py
│
├── analytics/                        # Statistical analysis engine
├── audit/                            # Runtime: append-only audit JSONL files
│
├── data/                             # Runtime data directories
│   ├── bronze/                       # Immutable raw snapshots (SHA-256)
│   ├── silver/                       # Validated, schema-enriched snapshots
│   ├── gold/                         # Analyst-derived exports
│   ├── snapshots/                    # Legacy snapshot storage
│   ├── uploads/                      # Temporary API upload staging
│   ├── tmp/                          # Chunked Parquet writer temp dir
│   └── dipex.duckdb                  # Embedded analytical database
│
├── docs/                             # Documentation
│   ├── DIPEX_IEEE_Research_Paper.md  # Full IEEE-style research paper
│   └── regulatory_matrices.md        # Compliance rule documentation
│
├── eda/                              # Automated EDA → self-contained HTML
├── feature_engineering/              # Legacy proxy → preprocessing/
│
├── frontend/                         # React 18 SPA (Vite)
│   └── src/
│       ├── pages/                    # RunPipeline, Analytics, ApiDocs
│       ├── components/               # Shell, charts, AnalysisPlanModal
│       ├── api/                      # API client (fetch wrappers)
│       ├── context/                  # PipelineSessionContext
│       └── utils/                    # Frontend utilities
│
├── ingestion/                        # Multi-source data ingestion (28 files)
│   ├── universal_intake.py           # Single intake interface (36 KB)
│   ├── schema_infer.py               # NLP-augmented classifier (49 KB)
│   ├── pipeline_bridge.py            # Full pipeline orchestrator (98 KB)
│   ├── pipeline_bridge_helpers.py    # Helpers for pipeline_bridge
│   ├── data_layers.py               # Bronze/Silver/Gold management (19 KB)
│   ├── immutability_guard.py         # SHA-256 enforcement (12 KB)
│   ├── lineage.py                    # Data lineage tracking (9 KB)
│   ├── kafka_pipeline.py             # Kafka consumer/producer (16 KB)
│   ├── batch_processor.py            # Multi-source batch ingestion (16 KB)
│   ├── stream_processor.py           # Streaming window processor (18 KB)
│   ├── adaptive_learner.py           # Adaptive learning from ingestion (19 KB)
│   ├── quality_gate.py               # Gate 1 + Gate 2 implementations (16 KB)
│   ├── normaliser.py                 # Data normalisation utilities (12 KB)
│   ├── feedback_controller.py        # RL feedback from ingestion (20 KB)
│   ├── data_rescue.py                # Recovery from corrupt/malformed data (25 KB)
│   ├── websocket_handler.py          # WebSocket stage-progress pusher (26 KB)
│   ├── connectors/                   # postgres, mongodb, redis, duckdb, sqlite
│   └── readers/                      # parquet, avro, feather, xml, api
│
├── latest_newest_models/             # SOURCE OF TRUTH — Do NOT delete
│   ├── *.pkl                         # 14 model artifacts (Apr 15 2026 @ 22:09)
│   └── reports/                      # 7 v7 training validation reports
│
├── learning/                         # Reinforcement learning engine
│   └── rl_agent/                     # PPO Actor-Critic system
│       ├── agent.py                  # PPOAgent main class (12 KB)
│       ├── policy_network.py         # Actor: 8-axis action head (9 KB)
│       ├── value_network.py          # Critic: V(s) estimator (5 KB)
│       ├── ppo_trainer.py            # GAE + clipped PPO update (20 KB)
│       ├── replay_buffer.py          # Transition storage (5 KB)
│       ├── reward_shaper.py          # Multi-component reward (9 KB)
│       ├── state_encoder.py          # 12-dim context encoder (6 KB)
│       └── action_space.py           # 8-axis discrete actions (5 KB)
│
├── modeling/                         # Model-layer ML utilities
│   ├── leakage_detector.py           # Pre-fit leakage guard (heavy use)
│   ├── calibrator.py                 # Platt/isotonic calibration
│   └── automl_trainer.py             # AutoML race orchestrator
│
├── models/                           # Active runtime model store (loaded on startup)
│   ├── schema_classifier.pkl         # 19.5 MB — LightGBM, Apr 15 2026
│   ├── schema_label_encoder.pkl      # 1.3 KB
│   ├── schema_feature_registry.pkl   # 2.5 KB — 58 features, 20 types
│   ├── drift_pipeline.pkl            # 43.6 KB — PyTorch MLP AE
│   ├── drift_feature_names.pkl       # 0.2 KB — 20 feature names
│   ├── anomaly_detector.pkl          # 3.4 MB — IsolationForest
│   ├── anomaly_threshold.pkl         # 0.4 KB — threshold=0.0089
│   ├── domain_classifier.pkl         # 372 KB — RandomForest, 6 domains
│   ├── domain_label_encoder.pkl      # 0.6 KB
│   ├── domain_registry.pkl           # 1.6 KB — 53 features
│   ├── chart_relevance_scorer.pkl    # 2.99 MB — LightGBM, 30 features
│   ├── chart_registry.pkl            # 0.6 KB — 7 chart types
│   ├── proposal_confidence.pkl       # 946 KB — Calibrated VotingClassifier
│   ├── confidence_metadata.json      # 1.6 KB — ECE=0.0225, AUC=0.9784
│   ├── rl_ppo_policy.pkl             # 311 KB — PPO actor, Apr 10 2026
│   ├── rl_ppo_value.pkl              # 275 KB — PPO critic, Apr 10 2026
│   ├── rl_training_curves.png        # PPO training diagnostic
│   └── reports/                      # v7 training validation reports
│       ├── schema_classifier_v7_report.json
│       ├── domain_classifier_v7_report.json
│       ├── drift_autoencoder_v7_report.json
│       ├── anomaly_detector_v7_report.json
│       ├── chart_relevance_scorer_v7_report.json
│       ├── confidence_scorer_v7_report.json
│       └── post_training_validation.json
│
├── monitoring/                       # Prometheus alert rules, Grafana dashboards
├── preprocessing/                    # Data cleaning + feature engineering (13 files)
│   ├── cleaner.py                    # Core data cleaner (20 KB)
│   ├── feature_engineer.py           # Feature generation (22 KB)
│   ├── robust_triage.py              # Tiered null/outlier/imbalance (44 KB)
│   ├── nlp_column_analyzer.py        # spaCy column name analysis (22 KB)
│   ├── missing_data_engine.py        # MCAR/MAR/MNAR diagnosis (31 KB)
│   ├── analyst_brain.py              # Analyst decision layer (42 KB)
│   ├── anomaly_scorer.py             # Per-stage anomaly tagging (11 KB)
│   ├── auto_corrector.py             # Auto-fix common data errors (8 KB)
│   ├── pipeline_builder.py           # Sklearn pipeline builder (8 KB)
│   ├── rl_feature_selector.py        # RL-guided feature selection (10 KB)
│   ├── temporal_splitter.py          # Temporal CV splits (12 KB)
│   ├── missing_pattern_analyzer.py   # MCAR/MAR/MNAR patterns (12 KB)
│   └── __init__.py
│
├── profiling/                        # Data profiling utilities
├── proposal/                         # AutoML model racing + SHAP
├── qa_control/                       # Quality assurance + calibration
├── reporting_service/                # LLM-powered narrative reports
│   └── llm_provider.py              # Ollama + HuggingFace provider
│
├── scripts/                          # 50+ utility, training, and demo scripts
│   ├── colab_train_production_v7.py  # Latest Colab training (133 KB)
│   ├── colab_train_production_v6.py  # Previous version (70 KB)
│   ├── train_individual/             # Per-model v7 training scripts
│   │   ├── 00_shared_utils.py        # Shared utilities + quality gates (65 KB)
│   │   ├── 01_drift_autoencoder.py   # Drift AE training
│   │   ├── 02_schema_classifier.py   # Schema classifier training
│   │   ├── 03_domain_classifier.py   # Domain classifier training
│   │   ├── 04_anomaly_detector.py    # Anomaly detector training
│   │   ├── 05_chart_relevance_scorer.py
│   │   ├── 06_confidence_scorer.py
│   │   └── 07_post_validation.py     # Post-training validation script
│   ├── train_models/                 # Production training modules
│   │   ├── train_schema_classifier.py
│   │   ├── train_drift_models.py
│   │   ├── train_anomaly_detector.py
│   │   ├── train_domain_classifier.py
│   │   ├── train_chart_relevance_scorer.py
│   │   ├── train_proposal_confidence.py
│   │   └── train_rl_ppo_agent.py
│   ├── demo_01_postgres.py           # PostgreSQL source demo
│   ├── demo_02_mongodb.py            # MongoDB source demo
│   ├── demo_03_kafka.py              # Kafka streaming demo
│   ├── demo_04_api.py                # REST API source demo
│   ├── demo_all_paths.py             # All 4 source demos
│   ├── start_kafka_pipeline.py       # Kafka consumer entrypoint
│   ├── produce_kafka_test_data.py    # Kafka test data producer
│   ├── watch_kafka_results.py        # Monitor Kafka gold outputs
│   ├── test_pipeline_integration.py  # Full integration test (35 KB)
│   ├── generate_sample_data.to       # Sample dataset generator
│   └── evaluate_models.py            # Model evaluation scripts
│
├── stats/                            # Statistical computation modules
├── tests/                            # pytest test suite (497 tests)
│   └── legacy/                       # Legacy security + integration tests
│
├── training/                         # Large-scale training utilities
│   └── large_scale/                  # E-commerce, banking data generators
│
├── utils/                            # Shared utilities
│   ├── model_loader.py               # Lazy model artifact loader
│   ├── training_validator.py         # 6-model quality gate validator
│   └── numpy_compat.py               # NumPy version compatibility
│
├── validation/                       # 7 validation sub-modules + compliance
│   ├── range_validator.py
│   ├── null_validator.py
│   ├── schema_validator.py
│   ├── leakage_detector.py
│   ├── drift_detector.py
│   ├── multicollinearity_detector.py
│   ├── zero_value_detector.py
│   ├── compliance_decision.py
│   ├── regulatory/                   # Banking, HIPAA, SOX, GDPR rule engines
│   └── governance/                   # RBAC policy enforcement
│
├── verifier/                         # Final stage: gate + RL + audit
│
├── check_models.py                   # Model functional smoke test (run inference)
├── main.py                           # CLI entry point (10 sub-commands)
├── simple_pipeline.py                # Simplified pipeline for testing
├── config.yaml                       # Central configuration (281 lines)
├── docker-compose.yml                # Full stack deployment
├── Dockerfile                        # API container definition
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Package metadata + build config
├── pytest.ini                        # Test discovery configuration
├── setup.bat                         # Windows one-command setup
└── setup.sh                          # Linux/macOS one-command setup
```

---

## 📊 Performance Benchmarks

### End-to-End Pipeline Latency

| Dataset | Rows | Columns | Stage 1–4 | Stage 5–8 | Total |
|---|---|---|---|---|---|
| Small | 1,000 | 10 | 0.3 s | 1.1 s | **1.4 s** |
| Medium | 10,000 | 25 | 0.7 s | 2.8 s | **3.5 s** |
| Large | 100,000 | 40 | 2.1 s | 5.3 s | **7.4 s** |
| Very Large | 500,000 | 50 | 7.9 s | 18.2 s | **26.1 s** |
| Chunked | 10,000,000 | 20 | ~90 s | N/A | **~90 s** (ingestion only) |

### Model Inference Latency

| Model | Input | Latency |
|---|---|---|
| Schema Classifier | 100 columns | < 5 ms |
| Domain Classifier | 1 dataset | < 2 ms |
| Drift Autoencoder | 1 dataset | 3 ms |
| Anomaly Detector | 1,000 rows | 1.2 ms |
| Anomaly Detector | 100,000 rows | 98 ms |
| Chart Relevance Scorer | 1 dataset | < 1 ms |
| Confidence Scorer | 1 run record | < 1 ms |

### Active Model Quality Metrics (Apr 15 2026)

| Model | Metric | Value | Gate Threshold |
|---|---|---|---|
| Schema Classifier | Holdout Balanced Accuracy | **94.7%** | ≥ 82% |
| Domain Classifier | Holdout Balanced Accuracy | **96.1%** | ≥ 78% |
| Drift Autoencoder | Detection Rate @ σ=0.3 | **89.4%** | Overfit ratio ≤ 2.5× |
| Anomaly Detector | AUROC | **0.961** | F1 ≥ 0.65 |
| Chart Relevance Scorer | Holdout Balanced Accuracy | **90.9%** | ≥ 75% |
| Confidence Scorer | AUC (calibrated) | **0.9784** | ≥ 0.85 |
| Confidence Scorer | ECE (after Platt) | **0.0225** | ≤ 0.07 |
| PPO Agent | Eval Mean Reward | **≥ 0.65** | ≥ 0.65, std ≤ 0.09 |

---

## 🔧 Model Maintenance

### Retraining Models

All 6 core models were trained using `scripts/train_individual/` scripts on Google Colab (GPU recommended for faster training):

```python
# Paste scripts in order into Colab cells:
# Cell 1: 00_shared_utils.py    (REQUIRED — shared utilities, quality gates)
# Cell 2: 01_drift_autoencoder.py
# Cell 3: 02_schema_classifier.py
# Cell 4: 03_domain_classifier.py
# Cell 5: 04_anomaly_detector.py
# Cell 6: 05_chart_relevance_scorer.py
# Cell 7: 06_confidence_scorer.py
# Cell 8: 07_post_validation.py   (validates all 6, prints final report)

# Colab setup cell (run first):
# !pip install -q openml lightgbm scikit-learn imbalanced-learn \
#     optuna shap joblib sentence-transformers requests scipy \
#     pmlb ucimlrepo pyarrow fastparquet torch

# After training: download from /content/adap_models/
# Copy to: latest_newest_models/  AND  models/
```

### Deploying New Models

```bash
# 1. Copy new models to latest_newest_models/ (preserve original)
cp /path/to/new/*.pkl latest_newest_models/
cp /path/to/new/reports/*.json latest_newest_models/reports/

# 2. Sync to active models/ directory
$src = "latest_newest_models"
$dst = "models"
Copy-Item "$src\*.pkl" $dst -Force
Copy-Item "$src\*.json" $dst -Force
Copy-Item "$src\reports\*" "models\reports\" -Force

# 3. Verify all models load and produce valid output
python check_models.py

# 4. Run training validator
python -m utils.training_validator .

# 5. Run full test suite
python -m pytest tests/ --ignore=tests/legacy/ -m "not integration" -v
```

### Model Versioning

Model versions are tracked via the `_version` field in training reports:
```json
{"_version": "20260415_2209", ...}
```

The version format is `YYYYMMDD_HHMM` from the Colab training run timestamp. Compare versions using the `post_training_validation.json` report.

---

## 👤 Author & License

**DIPEX v3.0.0** is proprietary software developed by **Aayush Sanklecha**.

Department of Computer Science and Engineering  
All rights reserved. Unauthorized reproduction, distribution, or modification is prohibited.

---

<div align="center">

*Built for data quality, model trustworthiness, and regulatory peace of mind.*

**DIPEX v3.0.0** | Python 3.12+ | FastAPI | React 18 | LightGBM | PyTorch | Apache Kafka | DuckDB

</div>
