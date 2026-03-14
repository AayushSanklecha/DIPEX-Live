# 🎯 DIPEX DEMO GUIDE

## ⚡ 30-Second Quick Start

**Docker (full stack):**
```bash
docker-compose up -d
```

**Local dev:**
```bash
# Terminal 1 — backend
uvicorn api.app:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
```

Then open: **http://localhost:3000**

---

## 📊 Live Demo Steps (~2 minutes)

### 1. Run a Pipeline (30 seconds)
- Go to **http://localhost:3000/run** (Run Pipeline page)
- Choose a source type:
  - **File Upload** — drag & drop a CSV (or use `samples/titanic.csv`)
  - **Database** — paste a DB URI, click **List Tables**, pick a table
  - **REST API** — enter an endpoint URL
  - **Kafka Stream** — enter a topic name
- (Optional) Enter a **Dataset ID** label — auto-generated from filename if blank
- (Optional) Enter a **Target Column** — auto-detected if blank
- Click **▶ Execute Pipeline**
- Watch the result panel: Gate decisions, confidence score, stage log

### 2. Explore the Dashboard (30 seconds)
- Go to **http://localhost:3000/** (System Dashboard)
- You'll see:
  - **Gate 1 / Gate 2 status** pills + **Confidence** + **Quality Score**
  - **KPI Cards** — avg, min, max, std for all numeric columns
  - **Quick Views** — auto-generated area/line charts
- Click **Schema** tab → toggle columns on/off → KPI cards update live
- Use the **Columns** dropdown (top left) → uncheck columns → charts narrow
- Use the **Rows** dropdown to filter by column value

### 3. View Reports (15 seconds)
- Go to **http://localhost:3000/reports**
- See the history of all pipeline runs with status badges
- Click any run to see full details (or navigate to `/` for the latest)

### 4. Check API Docs (15 seconds)
- **Built-in**: http://localhost:3000/api-docs
- **Swagger**: http://localhost:8000/docs

---

## 🌐 API Demo (For Technical Judges)

### Check Health
```bash
curl http://localhost:8000/health
```

### Upload & Run Pipeline (one command)
```bash
curl -X POST http://localhost:8000/api/pipeline/simple-run \
  -F "source_kind=file" \
  -F "file=@samples/titanic.csv" \
  -F "dataset_id=titanic_demo"
```

### Database Click-to-Ingest
```bash
# 1. List tables
curl "http://localhost:8000/api/db/tables?uri=postgresql://user:pass@host/db"

# 2. Run pipeline on a specific table
curl -X POST http://localhost:8000/api/pipeline/simple-run \
  -F "source_kind=database" \
  -F "db_uri=postgresql://user:pass@host/db" \
  -F "db_table=customers"
```

### Get Latest Result (with sample rows)
```bash
curl http://localhost:8000/api/results/latest
```

---

## 📁 Sample Data

- `samples/titanic.csv` — Titanic dataset
- Any CSV/Excel/JSON/Parquet file works

---

## ✨ Key Features to Highlight

| Feature | Detail |
|---|---|
| **4 source types** | File, Database (click-to-ingest), REST API, Kafka stream |
| **Auto target detection** | Finds target column automatically when left blank |
| **Zero-Config Robustness** | Auto-infers statistical rules (IQR bounds, empirical PSI) for noisy/messy data |
| **Dual-gate QA** | Gate 1 (quality) + Gate 2 (statistical) with retry engine |
| **Interactive Dashboard** | Column selector, row filters, live chart updates |
| **Immutable snapshots** | SHA-256 checksummed Parquet snapshots (Bronze layer) |
| **LLM narrative** | Governed summarization via Ollama/OpenAI/Gemini |
| **Full audit trail** | Every run logged to `audit/audit.jsonl` |
| **Prometheus/Grafana** | 15+ metrics, alert rules, Grafana dashboards |

---

## 🎯 What Makes DIPEX Stand Out

1. **Click-to-Ingest Databases** — list tables via URI, ingest with one click
2. **Dual Hard Gates** — enforce quality before any output is produced
3. **Bandit-Driven Retry Engine** — UCB1 strategy, never repeats same path
4. **Live Column/Row Filtering** — slice data in the dashboard without re-running
5. **Complete Observability** — Prometheus metrics, Grafana dashboards, JSONL audit

---

## 🚨 Troubleshooting

**Port already in use?**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

**Dashboard not loading?**
1. Check backend: `curl http://localhost:8000/health`
2. Check frontend: `curl http://localhost:3000`
3. Make sure `VITE_API_URL` in `frontend/.env` points to the backend

**No sample rows in dashboard after run?**  
The snapshot file (`data/snapshots/<run_id>_issf.parquet`) must exist. Re-run the pipeline via `/run` page — this is now fixed in v1.1.

---

## ⏱️ Timing Guide

| Action | Time |
|---|---|
| Start servers | ~10s |
| Upload & run pipeline | ~20–40s |
| View dashboard | instant |
| DB table listing | ~2s |
| Full demo | ~2 minutes |

---

## 🎓 Architecture at a Glance

| Layer | Tech |
|---|---|
| **Backend** | FastAPI (async, OpenAPI docs) |
| **Frontend** | React + Vite + Recharts |
| **Data** | Pandas, Parquet snapshots, DuckDB |
| **ML** | Scikit-learn, custom confidence vector |
| **LLM** | Ollama / OpenAI / Gemini / Anthropic |
| **Streaming** | Kafka (confluent-kafka), sliding window |
| **Monitoring** | Prometheus + Grafana |
| **CI/CD** | GitHub Actions (lint → test → build → deploy) |

---

**Ready to impress! 🚀**
