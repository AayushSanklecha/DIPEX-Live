# DIPEX — Data Intelligence Platform for Expert Analysis

> **DIP2026-23191** · Production-Grade AI-Augmented Data Analytics Platform

DIPEX is an end-to-end data intelligence platform that ingests, validates, analyses, and reports on structured data using a multi-tier analyst intelligence layer, reinforcement learning, and governed LLM summarization — all with enterprise-grade security, observability, and CI/CD.

---

## Architecture Overview

```
Raw Data ──► Bronze (Immutable) ──► Hard Gate 1 ──► Silver (Frozen)
                                                          │
                Retry Engine ◄── Hard Gate 2 ◄── Profiling + Governance + Stats + ML
                      │                                   │
              Experience Memory                    Confidence Vector
                      │                                   │
               RL Policy Update                    Gold Output (if ≥ threshold)
                      │                                   │
                LLM Report ◄────────────────────── Approved Storage
```

**13-Stage Pipeline** (in `ingestion/pipeline_bridge.py`):

| Stage | Name | Outcome if Fail |
|---|---|---|
| 1 | Preprocessing | Stop, log |
| 2 | **Hard Gate 1** | Abort pipeline, no RL update |
| 3 | Profiling | Warn, continue |
| 4 | Governance | Warn, continue |
| 5 | Statistics | Warn, continue |
| 6 | ML Modeling | Stop, log |
| 7 | **Hard Gate 2** | Trigger Retry Engine |
| 8 | Confidence Vector | Trigger Retry Engine |
| 9 | **Retry Engine** | Escalate to audit log |
| 10 | Experience Memory | Warn |
| 11 | RL Update | Sandbox safe, warn |
| 12 | Executive Report | Fallback to rule-based |
| 13 | Audit Trail | Warn |

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

### Analyst Intelligence (3 Tiers)
- **Junior**: basic stats, data cleaning, SQL queries, pivot tables, visualizations, reports
- **Mid**: EDA, cohort analysis, statistical tests, dashboard design, insight generation
- **Senior**: problem framing, experiment design, causal inference, strategic advisory, mentorship

### Hard Gates
- **Gate 1**: schema, null, range, type, regulatory rules — any fail = pipeline abort
- **Gate 2**: statistical validity, drift, stability, domain rules — fail = retry engine

### Retry Engine (Bandit-Driven)
- Max 3 retries per run (configurable)
- UCB1 strategy selection — never same path twice
- Escalation to `audit/retry_escalations.jsonl` on budget exhaustion

### Reinforcement Learning
- Meta-RL UCB1 (3 strategy families) + regret minimization
- EWC smoothing (λ=0.90) — prevents catastrophic forgetting
- Drift-conditioned ε boost (PSI > 0.2 → ε = 0.30)
- Forbidden target protection (`assert_target_allowed`)
- Sandbox mode: all writes are dry-run only

### LLM Governance
- Provider abstraction: only `reporting_service/llm_provider.py` may call LLMs
- PII redaction before prompt and after response
- Governance gate: refuses summarization of non-approved results
- Prompt audit trail: SHA-256 hashes to `audit/llm_prompts.jsonl`
- Cost tracker: cumulative token usage to `audit/llm_cost_log.jsonl`

### Security
- **JWT** (HS256): access + refresh tokens, RBAC roles (VIEWER/ANALYST/ADMIN/API_SERVICE)
- **RBAC**: `require_role()` dependency enforced on all mutation endpoints
- **Encryption at-rest**: Fernet symmetric (DIPEX_ENCRYPTION_KEY) with PBKDF2 key derivation
- **Audit access log**: daily-rotating `audit/access_log_YYYY-MM-DD.jsonl` per request
- **PII detection**: ML NER + 8-pattern regex scanner across all DataFrame columns

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Path | Role Required | Description |
|---|---|---|---|
| GET | `/` | Public | Service info |
| GET | `/health` | Public | Health check (DB, registry, uptime) |
| GET | `/metrics` | Public | Operational metrics JSON |
| GET | `/prom-metrics` | Public | Prometheus scrape endpoint |
| POST | `/auth/token` | Public | Login → JWT tokens |
| GET | `/auth/me` | VIEWER | Current user info |
| POST | `/ingest/file` | ANALYST | Upload file (UDIL, returns snapshot) |
| POST | `/api/pipeline/run` | ANALYST | **Unified**: upload + full 13-stage pipeline in one call |
| POST | `/api/run/` | ANALYST | Run pipeline on previously uploaded file |
| GET | `/api/results` | VIEWER | List approved results |
| POST | `/analyst/run` | ANALYST | Run single analyst operation |
| GET | `/analyst/operations` | VIEWER | List all analyst operations |
| POST | `/analyst/frame-problem` | ANALYST | Business question → KPI framework |
| POST | `/analyst/design-experiment` | ANALYST | A/B test design + power calc |
| POST | `/governance/evaluate` | ANALYST | Run governance checks |
| GET | `/drift/detect` | ANALYST | PSI + KL drift detection |
| GET | `/api/audit/` | VIEWER | Audit trail JSONL |

Full interactive docs at `/docs` (Swagger UI) or `/redoc`.

---

## Dashboard Pages

| Page | URL | Description |
|---|---|---|
| Overview | `/dashboard/index.html` | KPI cards, pipeline status, recent runs |
| Analyst Ops | `/dashboard/analyst_ops.html` | Browse + run Junior/Mid/Senior operations |
| RL Status | `/dashboard/rl_status.html` | Policy weights, reward history, safety log |
| Streaming | `/dashboard/streaming.html` | Kafka lag, window buffers, live event stream |
| Lineage | `/dashboard/lineage.html` | Bronze→Silver→Gold trace with HMAC hashes |
| Drift Monitor | `/dashboard/index.html#drift` | PSI, KL, JS, Wasserstein drift history (section in main dashboard) |

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

This project is part of the DIP2026-23191 research programme.
