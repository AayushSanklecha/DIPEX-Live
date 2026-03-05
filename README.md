 Data Intelligence Platform for Expert Analysis

>  · Production-Grade AI-Augmented Data Analytics Platform

this  is an end-to-end data intelligence platform that ingests, validates, analyses, and reports on structured data using a multi-tier analyst intelligence layer, reinforcement learning, and governed LLM summarization — all with enterprise-grade security, observability, and CI/CD.

---

## Architecture Overview — 5 Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCE LAYER                           │
│           CSV │ Excel │ Database │ API │ Kafka Streams          │
│                   datasource/router.py                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DATA PROCESSING LAYER                          │
│  ─ Data Ingestion         ingestion/universal_intake.py         │
│  ─ Normalization          ingestion/normaliser.py               │
│  ─ Data Profiling         profiling/profiler.py                 │
│  ─ Streaming Window       ingestion/streaming_window.py  (NEW)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│             QA, GOVERNANCE & CONTROL LAYER                      │
│  ─ Deterministic Validation   validation/hard_gate.py           │
│  ─ Independent QA Verifiers   verifier/confidence_vector.py     │
│  ─ Regulatory & Business Rules validation/regulatory/           │
│  ─ Confidence Scoring         verifier/confidence_vector.py     │
│  ─ Audit Logs                 audit/audit.jsonl                 │
│                   qa_control/controller.py  (NEW)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              AI & ANALYTICS SERVICE LAYER                       │
│  ─ Automated EDA          eda/auto_eda.py              (NEW)    │
│  ─ Feature Engineering    feature_engineering/engineer.py (NEW) │
│  ─ Insight Ranking        proposal/insight_ranker.py            │
│  ─ Retry & Strategy       pipeline_bridge._retry_engine_loop()  │
│  ─ LLM Summarization      reporting_service/llm_provider.py     │
│                   analytics/orchestrator.py  (NEW)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                            │
│  ─ Dashboards    dashboard/index.html                           │
│  ─ Reports       reporting_service/executive_report.py          │
│  ─ APIs          api/routes/ (12 route modules)                 │
│  ─ Exports       api/routes/exports.py  (NEW)                   │
│                  /api/export/{csv|json|parquet|report}          │
└─────────────────────────────────────────────────────────────────┘
```

**Pipeline Stages** (`ingestion/pipeline_bridge.py`):

| Stage | Name | Layer | Outcome if Fail |
|---|---|---|---|
| 0 | **Streaming Window** | Data Processing | Skip (non-streaming), warn |
| 1 | Preprocessing | Data Processing | Stop, log |
| 2 | **Hard Gate 1** | QA/Control | Abort pipeline |
| 3 | Profiling | Data Processing | Warn, continue |
| 4 | **AI & Analytics** | AI & Analytics Svc | Warn, continue |
| 5 | Governance | QA/Control | Warn, continue |
| 5 | Statistical Analysis | QA/Control | Warn, continue |
| 6 | ML Modeling | AI & Analytics Svc | Stop, log |
| 7 | **Hard Gate 2** | QA/Control | Trigger Retry Engine |
| 8 | Confidence Vector | QA/Control | Trigger Retry Engine |
| 9 | **Retry Engine** | AI & Analytics Svc | Escalate to audit |
| 10 | Experience Memory | AI & Analytics Svc | Warn |
| 11 | RL Update | AI & Analytics Svc | Sandbox safe, warn |
| 12 | Executive Report | Presentation | Fallback to rule-based |
| 13 | Audit Trail | QA/Control | Warn |

---


## Quick Start

### Prerequisites

- Python 3.9+
- Docker + Docker Compose
- (Optional) Kafka — bundled in `docker-compose.yml`

### Install

```bash
git clone https://github.com/YOUR_ORG/dipproj.git
cd dipproj
pip install -r requirements.txt
```

### Run Locally (File-Based, No Docker)

```bash
python -m uvicorn api.app:app --reload --port 8000
```

### Run Full Stack (Docker Compose)

```bash
# Start everything: API + Dashboard + Kafka + Prometheus + Grafana
docker-compose up -d

# View API docs
open http://localhost:8000/docs

# View Dashboard
open http://localhost:3000

# View Grafana
open http://localhost:3001   # admin / use GRAFANA_PASSWORD env var
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET_KEY` | `"dipex-dev-secret-..."` | **Must override in production** |
| `JWT_EXPIRE_MINS` | `60` | Access token lifetime |
| `JWT_REFRESH_EXPIRE_HOURS` | `24` | Refresh token lifetime |
| `DIPEX_ENV` | `development` | `development` / `staging` / `production` |
| `DIPEX_ENCRYPTION_KEY` | `""` | Base64 Fernet key for at-rest encryption |
| `DIPEX_KEY_SALT` | `"dipex-default-salt-v1"` | PBKDF2 salt |
| `LLM_PROVIDER` | `ollama` | `ollama` / `openai` / `gemini` / `anthropic` |
| `OPENAI_API_KEY` | `""` | Required if `LLM_PROVIDER=openai` |
| `KAFKA_BOOTSTRAP` | `kafka:9092` | Kafka broker address |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Comma-separated CORS origins |
| `RATE_LIMIT_RPM` | `120` | API rate limit (requests/minute) |
| `RATE_LIMIT_BURST` | `20` | Rate limit burst size |
| `AUDIT_DIR` | `audit` | Directory for audit log files |
| `MODEL_REGISTRY_PATH` | `data/model_registry` | Model registry directory |
| `GRAFANA_PASSWORD` | `dipex_grafana_2024` | Grafana admin password |

> **Security**: Never commit secrets. Use environment variables or a secrets manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault).

---

## Key Features

### Data Layer (Bronze → Silver → Gold)
- **Immutable Bronze snapshots**: SHA-256 checksummed, write-once
- **Silver cleaning**: imputation, dedup, dtype coercion, GDPR governance
- **Gold outputs**: only QA-approved (Gate 1 + Gate 2 + Confidence ≥ threshold)

### Hard Gates
- **Gate 1**: schema, null, range, type, regulatory rules — any fail = pipeline abort
- **Gate 2**: statistical validity, drift, stability, domain rules — fail = retry engine

### Retry Engine (Bandit-Driven)
- Max 3 retries per run (configurable)
- UCB1 strategy selection — never same path twice
- Escalation to `audit/retry_escalations.jsonl` on budget exhaustion

### Security
- **RBAC**: `require_role()` dependency enforced on all mutation endpoints
- **Encryption at-rest**: Fernet symmetric (DIPEX_ENCRYPTION_KEY) with PBKDF2 key derivation
- **Governance**: PII detection with ML NER + regex scanners

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Path | Role Required | Description |
|---|---|---|---|
| GET | `/` | Public | Service info |
| GET | `/health` | Public | Health check (DB, registry, uptime) |
| GET | `/metrics` | Public | Operational metrics JSON |
| GET | `/prom-metrics` | Public | Prometheus scrape endpoint |
| POST | `/ingest/file` | ANALYST | Upload file (UDIL, returns snapshot) |
| POST | `/api/pipeline/run` | ANALYST | **Unified**: upload + full 13-stage pipeline in one call |
| POST | `/api/run/` | ANALYST | Run pipeline on previously uploaded file |
| GET | `/api/results` | VIEWER | List approved results |
| GET | `/api/audit/` | VIEWER | Audit trail JSONL |

Full interactive docs at `/docs` (Swagger UI) or `/redoc`.

---

## Dashboard Pages

| Page | URL | Description |
|---|---|---|
| Overview | `/dashboard/index.html` | KPI cards, pipeline status, recent runs |

---

## Monitoring

### Prometheus
- Scrape endpoint: `GET /prom-metrics`
- Config: `monitoring/prometheus.yml`
- 15+ metrics: pipeline runs, gate decisions, confidence distribution, retries, LLM tokens, Kafka lag, RL violations, drift events

### Grafana
- Port: `3001` (Docker Compose)
- Datasource auto-provisioned from `monitoring/grafana/datasources/prometheus.yml`

### Alert Rules
- 15 Prometheus alert rules in `monitoring/alert_rules.yml`
- Covers: pipeline failure rate, confidence median, retry escalations, RL safety violations, severe drift, LLM cost spikes, Kafka consumer lag, API down

---

## CI/CD Pipeline

GitHub Actions (`.github/workflows/ci.yml`):

```
Lint (ruff) → Test (pytest + 60% coverage gate) → Security Scan (bandit) → Build (Docker) → Deploy (main only)
```

---

## Testing

```bash
# Full suite
python -m pytest tests/ -q

# With coverage
python -m pytest tests/ --cov=. --cov-report=term-missing

# Individual suites
python -m pytest tests/test_security.py -v          # Security layer
python -m pytest tests/test_streaming.py -v          # Streaming
python -m pytest tests/test_rl_safety.py -v          # RL safety rails
python -m pytest tests/test_llm_governance.py -v     # LLM governance
python -m pytest tests/test_isolation_guarantee.py -v # Data immutability
python -m pytest tests/test_analyst_intelligence.py -v # Analyst layer
```

---

## Project Structure

```
dipex/
├── api/                    # FastAPI application
│   ├── app.py              # Main app: middleware, routes, /health, /metrics
│   ├── metrics.py          # Prometheus registry (15+ metrics)
│   └── routes/             # 15 route modules
├── analyst/                # 3-tier analyst intelligence (16 modules)
├── auth/                   # JWT + RBAC + RL auth tuner
├── governance/             # PII detector, governance engine, policy registry
├── ingestion/              # 13-stage pipeline, connectors, stream processor
├── learning/               # RL update engine, experience memory, safety rails
├── middleware/             # Rate limiter, audit access log
├── modeling/               # Model trainer, registry, calibration
├── monitoring/             # prometheus.yml, grafana/, alert_rules.yml
├── reporting_service/      # LLM provider (governed), executive report
├── security/               # Encryption at-rest (Fernet + PBKDF2)
├── stats/                  # Hypothesis tests, descriptive stats
├── tests/                  # 19+ test modules, 500+ test cases
├── validation/             # Hard Gate 1 + confidence vector
├── verifier/               # Hard Gate 2 + 5 verifiers (drift/stability/domain/...)
├── dashboard/              # HTML dashboard pages (8 pages)
├── .github/workflows/      # CI/CD (ci.yml)
├── docker-compose.yml      # Full stack: API + Kafka + Prometheus + Grafana
├── Dockerfile              # Multi-stage build, non-root user, HEALTHCHECK
└── config.yaml             # All subsystem configuration
```

---

## License

This is a personal project and an original idea of Aayush Sanklecha and Pranav gund . 
