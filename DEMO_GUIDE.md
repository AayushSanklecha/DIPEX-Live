# 🎯 QUICK DEMO GUIDE FOR JUDGES

## ⚡ 30-Second Quick Start

```bash
./start.sh
```

Then open: **http://localhost:8000/dashboard**

---

## 📊 Live Demo Steps (2 minutes)

### 1. **Upload Dataset** (15 seconds)
- Click **"📤 Upload & Run Pipeline"** button
- Select a CSV/Excel file (or use sample: `samples/titanic.csv`)
- Click **"Run Pipeline"**

### 2. **Watch Automated Processing** (30 seconds)
You'll see real-time execution of:
1. ✅ Data upload
2. ✅ Data cleaning (null handling, type detection)
3. ✅ EDA (statistics, distributions)
4. ✅ Anomaly detection (outliers, quality checks)
5. ✅ Visualization generation
6. ✅ Report creation

### 3. **View Results** (15 seconds)
Results appear immediately with:
- Run ID and dataset info
- Quality score
- Gate decision (PASS/WARN/FAIL)
- Stage-by-stage execution log

### 4. **Check Reports** (30 seconds)
- Click **"📊 View Reports"** button
- See list of all pipeline runs
- Click **"View"** on any run for detailed statistics
- Charts, anomalies, and insights displayed

### 5. **System Health** (15 seconds)
- Click **"🏥 System Health"** button
- See uptime, database status, system state

---

## 🎬 Alternative: Automated Demo

```bash
python demo.py
```

This runs an automated demonstration using sample data and shows:
- Data loading
- Cleaning process
- EDA execution
- Anomaly detection
- Report generation

---

## 🌐 API Demo (For Technical Judges)

### 1. Check API Health
```bash
curl http://localhost:8000/health
```

### 2. View API Documentation
Open: **http://localhost:8000/docs**

### 3. Upload & Process (One Command)
```bash
curl -X POST http://localhost:8000/api/pipeline/simple-run \
  -F "source_kind=file" \
  -F "file=@samples/titanic.csv"
```

### 4. View Results
```bash
curl http://localhost:8000/api/results
```

---

## 📁 Sample Data Locations

- `samples/titanic.csv` - Titanic dataset
- `samples/*.csv` - Any other CSV files you add
- Supports: CSV, Excel, JSON, Parquet

---

## ✨ Key Features to Highlight

### 1. **Upload Dataset**
- ✅ Drag & drop or click to upload
- ✅ Multiple format support (CSV, Excel, JSON, Parquet)
- ✅ Database connections (PostgreSQL, MongoDB, Neo4j)
- ✅ Live API ingestion

### 2. **Clean Data**
- ✅ Automatic null value handling
- ✅ Type detection and conversion
- ✅ Outlier capping
- ✅ Data validation

### 3. **Run EDA Automatically**
- ✅ Descriptive statistics (mean, std, quartiles)
- ✅ Distribution analysis
- ✅ Correlation detection
- ✅ Missing value reports

### 4. **Detect Anomalies**
- ✅ Statistical outlier detection
- ✅ IQR-based anomaly flagging
- ✅ Quality scoring
- ✅ Data quality gates

### 5. **Generate Charts/Dashboards**
- ✅ Live metrics dashboard
- ✅ KPI cards (runs, pass rate, quality)
- ✅ Interactive tables
- ✅ Real-time updates

### 6. **Generate Simple Report**
- ✅ HTML reports with insights
- ✅ Summary statistics table
- ✅ Anomaly highlights
- ✅ Downloadable format

---

## 🎯 What Makes This Stand Out

1. **Real-time Processing**: Watch data flow through pipeline stages
2. **Zero Configuration**: Works immediately, no setup required
3. **Auto-Detection**: Automatically finds target column, handles data types
4. **Quality Gates**: PASS/WARN/FAIL decisions based on data quality
5. **Audit Trail**: Complete log of all operations
6. **Professional UI**: Clean, modern dashboard
7. **RESTful API**: Full programmatic access
8. **Production-Ready**: Error handling, logging, health checks

---

## 🚨 Troubleshooting

**Port already in use?**
```bash
lsof -ti:8000 | xargs kill -9
./start.sh
```

**Dashboard not loading?**
- Wait 5 seconds after server starts
- Check: http://localhost:8000/health
- Refresh browser

**No sample data?**
- Place any CSV file in `samples/` directory
- Or upload via dashboard

---

## 📸 Demo Screenshots Flow

1. **Home Screen**: Clean dashboard with quick action buttons
2. **Upload**: Drag & drop interface
3. **Processing**: Real-time stage execution log
4. **Results**: Quality score, gate decision, full details
5. **Reports**: List of all runs with statistics
6. **Details**: Individual run breakdown with insights

---

## ⏱️ Timing Guide

| Action | Time | Total |
|--------|------|-------|
| Start server | 5s | 5s |
| Upload file | 5s | 10s |
| Process pipeline | 10-30s | 40s |
| View results | 5s | 45s |
| Show reports | 10s | 55s |
| API demo | 20s | 75s |

**Total demo time**: ~1-2 minutes

---

## 💡 Key Talking Points

1. **"This automates the entire analyst workflow"**
   - No manual cleaning, no manual analysis
   - Upload → Results in 30 seconds

2. **"Built for production"**
   - Error handling, logging, audit trail
   - RESTful API, health monitoring

3. **"Flexible data intake"**
   - Files, databases, APIs, streams
   - Auto-detects formats and types

4. **"Quality gates ensure reliability"**
   - Data quality scoring
   - Anomaly detection
   - PASS/WARN/FAIL decisions

5. **"Complete observability"**
   - Stage-by-stage execution log
   - Audit trail for compliance
   - Metrics and health monitoring

---

## 🎓 For Judges: Architecture Highlights

- **Backend**: FastAPI (modern, async, fast)
- **Data Processing**: Pandas, NumPy (industry standard)
- **Analytics**: DuckDB (in-process OLAP)
- **Frontend**: Vanilla JS (no framework bloat)
- **Storage**: Immutable data layers (Bronze/Silver/Gold)
- **API**: RESTful, OpenAPI/Swagger docs

---

**Ready to impress! 🚀**
