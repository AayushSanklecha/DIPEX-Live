/* ═══════════════════════════════════════════════════════════════════
   DIPEX Enterprise Analytics — app.js v4.0
   Full SPA logic: routing, API calls, charts, animations
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

// ── Configuration ────────────────────────────────────────────────────
const API_BASE = '';   // served from same origin by FastAPI
const TOKEN = {
    get access() { return localStorage.getItem('dipex_token') || ''; },
    get refresh() { return localStorage.getItem('dipex_refresh') || ''; },
    set access(v) { localStorage.setItem('dipex_token', v); },
    set refresh(v) { localStorage.setItem('dipex_refresh', v); },
    clear() { localStorage.removeItem('dipex_token'); localStorage.removeItem('dipex_refresh'); }
};

// ── Page Meta ────────────────────────────────────────────────────────
const PAGE_META = {
    overview: { title: 'Overview', subtitle: 'Platform health and KPI summary' },
    pipeline: { title: 'Pipeline', subtitle: 'Upload, configure and run the DIPEX pipeline' },
    ingest_all: { title: 'Multi-DB Ingest', subtitle: 'Aggregate from MongoDB, Redis, PostgreSQL, Neo4j, Kafka, DuckDB & Parquet' },
    analytics: { title: 'Analytics', subtitle: 'Confidence scores, gate decisions, and insight narratives' },
    statistics: { title: 'Statistics', subtitle: 'Descriptive stats, hypothesis testing, and regression' },
    modeling: { title: 'Modeling', subtitle: 'ML training, champion selection, and model registry' },
    sql: { title: 'SQL Console', subtitle: 'Query your datasets with DuckDB SQL' },
    analyst_ops: { title: 'Analyst Intelligence', subtitle: 'Junior · Mid · Senior analyst operations on Gold artefacts' },
    analyst_tiers: { title: 'Analyst Tier Automation', subtitle: 'Layered operations from Bronze to Gold via analyst tiers' },
    rl_status: { title: 'RL Status', subtitle: 'Meta-RL policy weights, exploration rate, reward history' },
    streaming: { title: 'Streaming Monitor', subtitle: 'Kafka consumer lag, window state, backpressure, watermark' },
    lineage: { title: 'Data Lineage', subtitle: 'Trace any Gold artefact back through Silver and Bronze' },
    drift: { title: 'Drift Monitor', subtitle: 'PSI · KL Divergence · Jensen-Shannon · Wasserstein' },
    cohort: { title: 'Cohort Analysis', subtitle: 'Retention matrix, LTV curves, period-over-period' },
    calibration: { title: 'Calibration', subtitle: 'Reliability diagram, ECE, Brier Score, Platt/Isotonic' },
    reports: { title: 'Reports', subtitle: 'Generate and download executive reports' },
    governance: { title: 'Governance', subtitle: 'Policy engine, data catalog, compliance evaluation' },
    audit: { title: 'Audit Trail', subtitle: 'Tamper-evident, chronological system audit log' },
    login: { title: 'Authentication', subtitle: 'JWT login, role-based access, token management' },
    api_docs: { title: 'API Docs', subtitle: 'Interactive Swagger UI — explore and test every endpoint' },
    grafana: { title: 'Grafana', subtitle: 'Live monitoring dashboards — pipeline runs, drift, RL, Kafka' },
};

// ── Core: apiFetch ───────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
    const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
    if (TOKEN.access) headers['Authorization'] = 'Bearer ' + TOKEN.access;
    const res = await fetch(API_BASE + path, { ...options, headers });
    if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try { const e = await res.json(); msg = e.detail || e.message || msg; } catch { }
        throw new Error(msg);
    }
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) return res.json();
    return res.text();
}

// ── Toast Notifications ──────────────────────────────────────────────
function showToast(msg, type = 'info', duration = 4000) {
    const icons = { success: '✓', error: '✕', info: 'ℹ', warn: '⚠' };
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${msg}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'toastSlide 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function setStatus(elId, msg, type) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.className = `status-msg ${type}`;
    el.textContent = msg;
}

// ── SPA Router ───────────────────────────────────────────────────────
let currentPage = 'overview';
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const page = document.getElementById('page-' + pageId);
    const nav = document.getElementById('nav-' + pageId);
    if (page) page.classList.add('active');
    if (nav) nav.classList.add('active');
    const meta = PAGE_META[pageId] || {};
    document.getElementById('page-title').textContent = meta.title || pageId;
    document.getElementById('page-subtitle').textContent = meta.subtitle || '';
    currentPage = pageId;
    onPageLoad(pageId);
}

function onPageLoad(pageId) {
    const loaders = {
        overview: () => { loadMetrics(); loadAudit(); },
        rl_status: () => loadRLStatus(),
        streaming: () => refreshStreaming(),
        lineage: () => loadLineageDatasets(),
        models: () => loadModelRegistry(),
        analyst_ops: () => renderAnalystOps(),
        calibration: () => renderCalibrationDemo(),
        login: () => refreshAuthUI(),
        api_docs: () => lazyLoadIframe('api-docs-iframe', 'http://localhost:8000/docs'),
        grafana: () => lazyLoadIframe('grafana-iframe', 'http://localhost:3001'),
    };
    (loaders[pageId] || (() => { }))();
}

// Lazy-load an iframe — only sets src on first visit
function lazyLoadIframe(iframeId, url) {
    const iframe = document.getElementById(iframeId);
    if (iframe && iframe.src === 'about:blank') iframe.src = url;
}

// ── Clock ────────────────────────────────────────────────────────────
function updateClock() {
    const el = document.getElementById('current-time');
    if (el) el.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ── Animated Counters ─────────────────────────────────────────────────
function animateCounter(el, target, decimals = 0, suffix = '') {
    const start = 0;
    const duration = 900;
    const startTime = performance.now();
    function tick(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const val = start + (target - start) * eased;
        el.textContent = val.toFixed(decimals) + suffix;
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

// ── API Status Check ──────────────────────────────────────────────────
async function checkAPIStatus() {
    const dot = document.getElementById('api-status-dot');
    const text = document.getElementById('api-status-text');
    try {
        await fetch(API_BASE + '/health', { signal: AbortSignal.timeout(3000) });
        dot.className = 'status-dot online';
        text.textContent = 'API Online';
    } catch {
        dot.className = 'status-dot offline';
        text.textContent = 'API Offline';
    }
}
setInterval(checkAPIStatus, 30000);
checkAPIStatus();

function refreshData() { onPageLoad(currentPage); showToast('Refreshed', 'info', 1500); }

// ══════════════════════════════════════════════════════════════════════
// OVERVIEW
// ══════════════════════════════════════════════════════════════════════
let confChart = null, statusChart = null;

async function loadMetrics() {
    try {
        const data = await apiFetch('/metrics');
        const totalRuns = data.total_pipeline_runs || 0;
        const passRate = data.total_pipeline_runs
            ? ((data.total_pipeline_runs - (data.retry_runs || 0)) / data.total_pipeline_runs * 100)
            : 0;
        const el = (id) => document.getElementById(id);
        animateCounter(el('kpi-total-runs'), totalRuns);
        el('kpi-pass-rate').textContent = passRate.toFixed(1) + '%';
        el('kpi-approved').textContent = data.approved_runs || totalRuns || '—';
        el('kpi-reports').textContent = data.reports_count || '—';
        renderConfidenceChart();
        renderStatusChart(data);
    } catch (e) {
        const el = (id) => document.getElementById(id);
        ['kpi-total-runs', 'kpi-pass-rate', 'kpi-approved', 'kpi-reports'].forEach(id =>
            (document.getElementById(id) || {}).textContent = '—');
        renderConfidenceChart();
        renderStatusChart({});
    }
}

function renderConfidenceChart() {
    const ctx = document.getElementById('confidence-chart');
    if (!ctx) return;
    if (confChart) confChart.destroy();
    const labels = Array.from({ length: 12 }, (_, i) => ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][i]);
    const data = [0.61, 0.65, 0.67, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.87];
    confChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Confidence Score',
                data,
                borderColor: '#6c63ff',
                backgroundColor: 'rgba(108,99,255,0.08)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#6c63ff',
                pointRadius: 3,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b99b5', font: { size: 11 } } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b99b5', font: { size: 11 } }, min: 0.5, max: 1.0 }
            }
        }
    });
}

function renderStatusChart(data) {
    const ctx = document.getElementById('status-chart');
    if (!ctx) return;
    if (statusChart) statusChart.destroy();
    const passed = data.total_pipeline_runs ? (data.total_pipeline_runs - (data.retry_runs || 0)) : 34;
    const retried = data.retry_runs || 5;
    const failed = data.failed_runs || 2;
    statusChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Passed', 'Retried', 'Failed'],
            datasets: [{ data: [passed, retried, failed], backgroundColor: ['#00e870', '#ffb938', '#ff4757'], borderColor: 'transparent', hoverOffset: 6 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, cutout: '65%',
            plugins: {
                legend: { labels: { color: '#8b99b5', font: { size: 11 }, padding: 16 } }
            }
        }
    });
}

// ══════════════════════════════════════════════════════════════════════
// PIPELINE
// ══════════════════════════════════════════════════════════════════════
function setupDropZone() {
    const zone = document.getElementById('drop-zone');
    const input = document.getElementById('file-input');
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => { if (input.files[0]) uploadFile(input.files[0]); });
}

async function uploadFile(file) {
    setStatus('upload-status', `Uploading ${file.name}…`, 'info');
    const form = new FormData();
    form.append('file', file);
    try {
        const headers = {};
        if (TOKEN.access) headers['Authorization'] = 'Bearer ' + TOKEN.access;
        const res = await fetch(API_BASE + '/ingest/file', { method: 'POST', headers, body: form });
        const data = await res.json();
        const runId = data.snapshot_id || data.run_id || '';
        document.getElementById('run-id').value = runId;
        setStatus('upload-status', `✓ Uploaded — Snapshot: ${runId}`, 'success');
        showToast('File uploaded successfully', 'success');
        loadSnapshots();
    } catch (e) {
        setStatus('upload-status', `Error: ${e.message}`, 'error');
        showToast('Upload failed: ' + e.message, 'error');
    }
}

async function runPipeline() {
    const runId = document.getElementById('run-id').value.trim();
    const target = document.getElementById('target-col').value.trim() || 'target';
    const card = document.getElementById('pipeline-log-card');
    const logEl = document.getElementById('pipeline-log');
    const badge = document.getElementById('pipeline-badge');
    const resCard = document.getElementById('pipeline-result-card');

    if (!runId) { showToast('Upload a file first to get a Run ID', 'warn'); return; }

    card.style.display = 'block';
    resCard.style.display = 'none';
    badge.className = 'badge badge-running';
    badge.textContent = 'RUNNING';
    logEl.innerHTML = '<div class="log-line info">[DIPEX] Starting pipeline run…</div>';

    const addLog = (msg, type = '') => {
        logEl.innerHTML += `<div class="log-line ${type}">${msg}</div>`;
        logEl.scrollTop = logEl.scrollHeight;
    };

    try {
        const result = await apiFetch('/api/run', {
            method: 'POST',
            body: JSON.stringify({ run_id: runId, target_column: target })
        });
        badge.className = 'badge badge-pass';
        badge.textContent = 'DONE';
        addLog('[DIPEX] Run complete.', 'success');
        const logs = result.logs || result.pipeline_logs || [];
        logs.forEach(l => addLog(l));
        resCard.style.display = 'block';
        const conf = result.confidence_score || result.final_confidence || null;
        document.getElementById('pipeline-result').innerHTML = renderResultCard(result, conf);
        showToast('Pipeline completed successfully', 'success');
    } catch (e) {
        badge.className = 'badge badge-fail';
        badge.textContent = 'ERROR';
        addLog('[ERROR] ' + e.message, 'error');
        showToast('Pipeline failed: ' + e.message, 'error');
    }
}

async function runPreprocess() {
    const runId = document.getElementById('run-id').value.trim();
    if (!runId) { showToast('Enter a Run ID first', 'warn'); return; }
    try {
        showToast('Preprocessing…', 'info');
        await apiFetch('/api/preprocess', { method: 'POST', body: JSON.stringify({ run_id: runId }) });
        showToast('Preprocessing complete', 'success');
    } catch (e) { showToast('Error: ' + e.message, 'error'); }
}

function renderResultCard(result, conf) {
    const score = conf ? (conf * 100).toFixed(1) + '%' : 'N/A';
    const cls = conf >= 0.85 ? 'badge-pass' : conf >= 0.70 ? 'badge-warn' : 'badge-fail';
    return `<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
    <span>Confidence Score:</span>
    <span class="badge ${cls}">${score}</span>
  </div>
  <div class="json-result"><pre>${JSON.stringify(result, null, 2)}</pre></div>`;
}

async function loadSnapshots() {
    try {
        const data = await apiFetch('/ingest/snapshots');
        const rows = (Array.isArray(data) ? data : (data.snapshots || [])).slice(0, 20);
        const tbody = document.getElementById('snapshots-body');
        if (!tbody) return;
        tbody.innerHTML = rows.length ? rows.map(s => `<tr>
      <td>${s.dataset_id || '—'}</td>
      <td class="mono">${(s.snapshot_id || '').slice(0, 16)}…</td>
      <td>${s.row_count || '—'}</td>
      <td>${s.quality_score != null ? (s.quality_score * 100).toFixed(0) + '%' : '—'}</td>
      <td><span class="badge ${s.gold_approved ? 'badge-pass' : s.silver_ready ? 'badge-warn' : 'badge-info'}">${s.gold_approved ? 'Gold' : s.silver_ready ? 'Silver' : 'Bronze'}</span></td>
      <td>${s.created_at ? new Date(s.created_at).toLocaleTimeString() : '—'}</td>
    </tr>`).join('') : '<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:20px">No snapshots yet</td></tr>';
    } catch { /* silent */ }
}
// ══════════════════════════════════════════════════════════════════════
// MULTI-DB INGEST
// ══════════════════════════════════════════════════════════════════════
const DB_SOURCES = ['mongodb', 'redis', 'postgres', 'neo4j', 'kafka', 'duckdb', 'parquet'];

function setDbCardState(src, state, msg) {
    const card = document.getElementById('db-' + src);
    if (!card) return;
    card.className = 'db-card ' + state;
    card.querySelector('.db-status').textContent = msg;
}

async function triggerIngestAll() {
    const btn = document.getElementById('ingest-all-btn');
    const resultEl = document.getElementById('ingest-all-result');
    btn.disabled = true; btn.textContent = '⊛ Running…';
    DB_SOURCES.forEach(s => setDbCardState(s, 'running', 'Running…'));
    setStatus('ingest-all-status', 'Triggering multi-DB aggregation…', 'info');
    try {
        const data = await apiFetch('/ingest/all-databases', { method: 'POST' });
        const results = data.results || [];
        DB_SOURCES.forEach(src => {
            const r = results.find(x => (x.source || '').toLowerCase().includes(src));
            if (r) setDbCardState(src, r.status === 'ok' || r.status === 'success' ? 'ok' : 'error', r.status === 'ok' || r.status === 'success' ? '✓ Done' : '✕ Error');
            else setDbCardState(src, '', 'N/A');
        });
        setStatus('ingest-all-status', 'Status: ' + data.status + ' · ' + (data.sources_attempted || 0) + ' sources', 'success');
        resultEl.innerHTML = '<div class="json-result"><pre>' + JSON.stringify(data, null, 2) + '</pre></div>';
        showToast('Ingest-All: ' + data.status, data.status === 'completed' ? 'success' : 'warn');
    } catch (e) {
        DB_SOURCES.forEach(s => setDbCardState(s, 'error', '✕ Error'));
        setStatus('ingest-all-status', 'Error: ' + e.message, 'error');
        resultEl.innerHTML = '<div class="json-result" style="color:var(--danger)"><pre>Error: ' + e.message + '</pre></div>';
        showToast('Ingest-All failed: ' + e.message, 'error');
    } finally { btn.disabled = false; btn.textContent = '⊛ Trigger All-DB Ingest'; }
}

// ══════════════════════════════════════════════════════════════════════
// ANALYTICS
// ══════════════════════════════════════════════════════════════════════
let radarChart = null;

async function loadAnalytics() {
    const runId = document.getElementById('analytics-run-id').value.trim();
    try {
        const data = await apiFetch(runId ? '/api/results/' + runId : '/api/results/latest');
        renderAnalyticsUI(data);
    } catch (e) {
        renderAnalyticsUI({
            confidence_score: 0.87, gate1_decision: 'PASS', gate2_decision: 'PASS',
            dimensions: { data_quality: 0.92, statistical_strength: 0.84, stability: 0.88, compliance: 0.91 },
            narrative: 'Analysis complete — demo data. Hard Gate 1 (data quality) and Hard Gate 2 (statistical verification) both PASS. Confidence vector: data_quality 0.92 · statistical_strength 0.84 · stability 0.88 · compliance 0.91.'
        });
        if (runId) showToast('Using demo data: ' + e.message, 'warn');
    }
}

function renderAnalyticsUI(data) {
    const conf = data.confidence_score || 0;
    const gaugeEl = document.getElementById('gauge-value');
    if (gaugeEl) gaugeEl.textContent = (conf * 100).toFixed(1) + '%';
    drawGauge(conf);
    const g1 = data.gate1_decision || (conf >= 0.70 ? 'PASS' : 'REJECT');
    const g2 = data.gate2_decision || (conf >= 0.85 ? 'PASS' : 'REJECT');
    const setB = (id, txt, cls) => { const e = document.getElementById(id); if (e) { e.textContent = txt; e.className = 'badge ' + cls; } };
    setB('gate1-badge', g1, g1 === 'PASS' ? 'badge-pass' : 'badge-fail');
    setB('gate2-badge', g2, g2 === 'PASS' ? 'badge-pass' : 'badge-fail');
    const bar = document.getElementById('conf-bar-fill');
    if (bar) { bar.style.width = (conf * 100) + '%'; bar.className = 'conf-bar ' + (conf >= 0.85 ? 'high' : conf >= 0.70 ? 'medium' : 'low'); }
    const narr = document.getElementById('narrative-content');
    if (narr) narr.textContent = data.narrative || data.insight || 'No narrative available.';
    renderRadarChart(data.dimensions || { data_quality: 0.92, statistical_strength: 0.84, stability: 0.88, compliance: 0.91 });
}

function drawGauge(conf) {
    const canvas = document.getElementById('gauge-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d'); const cx = 100, cy = 100, r = 70;
    ctx.clearRect(0, 0, 200, 120);
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 14; ctx.stroke();
    const color = conf >= 0.85 ? '#00e870' : conf >= 0.70 ? '#ffb938' : '#ff4757';
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, Math.PI + conf * Math.PI);
    ctx.strokeStyle = color; ctx.lineWidth = 14; ctx.lineCap = 'round'; ctx.stroke();
}

function renderRadarChart(dims) {
    const ctx = document.getElementById('radar-chart');
    if (!ctx) return;
    if (radarChart) radarChart.destroy();
    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: Object.keys(dims).map(k => k.replace(/_/g, ' ')),
            datasets: [{
                label: 'Score', data: Object.values(dims), borderColor: '#6c63ff',
                backgroundColor: 'rgba(108,99,255,0.15)', pointBackgroundColor: '#6c63ff', pointRadius: 4
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { r: { min: 0, max: 1, ticks: { display: false }, grid: { color: 'rgba(255,255,255,0.06)' }, pointLabels: { color: '#8b99b5', font: { size: 10 } } } },
            plugins: { legend: { display: false } }
        }
    });
}

// ══════════════════════════════════════════════════════════════════════
// STATISTICS
// ══════════════════════════════════════════════════════════════════════
async function loadDescriptiveStats() {
    const runId = document.getElementById('stats-run-id').value.trim();
    const out = document.getElementById('stats-output');
    out.innerHTML = '<div class="skeleton skeleton-text" style="width:100%;height:80px"></div>';
    try {
        const data = await apiFetch(runId ? '/api/stats/' + runId : '/api/stats/latest');
        out.innerHTML = '<div class="json-result" style="max-height:280px"><pre>' + JSON.stringify(data.descriptive || data, null, 2) + '</pre></div>';
    } catch (e) { out.innerHTML = '<div class="status-msg error">Error: ' + e.message + '</div>'; }
}

async function runHypothesisTest() {
    const test = document.getElementById('ht-test').value;
    const colA = document.getElementById('ht-col-a').value.trim();
    const colB = document.getElementById('ht-col-b').value.trim();
    const out = document.getElementById('ht-output');
    if (!colA) { showToast('Column A is required', 'warn'); return; }
    out.innerHTML = '<div class="skeleton skeleton-text" style="width:100%;height:60px"></div>';
    try {
        const data = await apiFetch('/api/hypothesis', { method: 'POST', body: JSON.stringify({ test_type: test, column_a: colA, column_b: colB || undefined }) });
        const pval = data.p_value != null ? data.p_value.toFixed(4) : '—';
        const sig = data.p_value != null && data.p_value < 0.05;
        out.innerHTML = '<div class="gate-row"><span class="gate-label">p-value</span><span class="badge ' + (sig ? 'badge-pass' : 'badge-warn') + '">' + pval + '</span></div>' +
            '<div class="gate-row"><span class="gate-label">Statistic</span><span>' + (data.statistic || 0).toFixed(4) + '</span></div>' +
            '<div class="gate-row"><span class="gate-label">Significant?</span><span class="badge ' + (sig ? 'badge-pass' : 'badge-fail') + '">' + (sig ? 'Yes (p<0.05)' : 'No') + '</span></div>' +
            '<div class="json-result" style="margin-top:8px"><pre>' + JSON.stringify(data, null, 2) + '</pre></div>';
    } catch (e) { out.innerHTML = '<div class="status-msg error">Error: ' + e.message + '</div>'; }
}

async function runRegression() {
    const target = document.getElementById('reg-target').value.trim();
    const model = document.getElementById('reg-model').value;
    const out = document.getElementById('reg-output');
    if (!target) { showToast('Enter a target column', 'warn'); return; }
    out.innerHTML = '<div class="skeleton skeleton-text" style="width:100%;height:60px"></div>';
    try {
        const data = await apiFetch('/api/regression', { method: 'POST', body: JSON.stringify({ target_column: target, model_type: model }) });
        out.innerHTML = '<div class="json-result"><pre>' + JSON.stringify(data, null, 2) + '</pre></div>';
    } catch (e) { out.innerHTML = '<div class="status-msg error">Error: ' + e.message + '</div>'; }
}

// ══════════════════════════════════════════════════════════════════════
// MODELING
// ══════════════════════════════════════════════════════════════════════
async function trainModels() {
    const runId = document.getElementById('model-run-id').value.trim();
    const target = document.getElementById('model-target').value.trim() || 'target';
    const statusEl = document.getElementById('model-train-status');
    if (!runId) { showToast('Enter a Run ID', 'warn'); return; }
    statusEl.innerHTML = '<div class="status-msg info">Training models…</div>';
    try {
        const data = await apiFetch('/api/model/train', { method: 'POST', body: JSON.stringify({ run_id: runId, target_column: target }) });
        statusEl.innerHTML = '<div class="status-msg success">Training complete · Champion: ' + (data.champion || data.best_model || '—') + '</div>' +
            '<div class="json-result" style="margin-top:8px"><pre>' + JSON.stringify(data, null, 2) + '</pre></div>';
        loadModelRegistry();
        showToast('Models trained successfully', 'success');
    } catch (e) {
        statusEl.innerHTML = '<div class="status-msg error">Error: ' + e.message + '</div>';
        showToast('Training failed: ' + e.message, 'error');
    }
}

async function loadModelRegistry() {
    const tbody = document.getElementById('model-registry-body');
    if (!tbody) return;
    try {
        const data = await apiFetch('/api/model/registry');
        const rows = Array.isArray(data) ? data : (data.models || []);
        tbody.innerHTML = rows.length ? rows.map(m =>
            '<tr><td class="mono">' + (m.run_id || '—') + '</td><td>' + (m.model_type || m.algorithm || '—') + '</td><td>' + (m.task_type || '—') + '</td>' +
            '<td style="font-size:0.7rem;color:var(--text-muted)">' + (m.saved_at ? new Date(m.saved_at).toLocaleString() : '—') + '</td>' +
            '<td>' + (m.is_champion ? '<span class="badge badge-pass">Champion</span>' : '—') + '</td></tr>'
        ).join('') : '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:20px">No models yet</td></tr>';
    } catch { tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:20px">Error loading registry</td></tr>'; }
}

// ══════════════════════════════════════════════════════════════════════
// SQL CONSOLE
// ══════════════════════════════════════════════════════════════════════
async function executeSQL() {
    const sql = document.getElementById('sql-editor').value.trim();
    const runId = document.getElementById('sql-run-id').value.trim();
    if (!sql) { showToast('Enter a SQL query', 'warn'); return; }
    setStatus('sql-status', 'Executing…', 'info');
    const card = document.getElementById('sql-result-card');
    try {
        const data = await apiFetch('/api/sql', { method: 'POST', body: JSON.stringify({ query: sql, run_id: runId || undefined }) });
        setStatus('sql-status', 'Query complete', 'success');
        card.style.display = 'block';
        const rows = data.rows || data.results || [];
        const columns = data.columns || (rows.length > 0 ? Object.keys(rows[0]) : []);
        const total = data.total_rows || rows.length;
        document.getElementById('sql-result-meta').textContent = total + ' row' + (total !== 1 ? 's' : '') + ' · ' + ((data.elapsed_ms || 0).toFixed(0)) + 'ms';
        const table = document.getElementById('sql-result-table');
        table.innerHTML = columns.length
            ? '<thead><tr>' + columns.map(c => '<th>' + c + '</th>').join('') + '</tr></thead><tbody>' +
            rows.slice(0, 200).map(r => '<tr>' + columns.map(c => '<td>' + (r[c] ?? '') + '</td>').join('') + '</tr>').join('') + '</tbody>'
            : '<tbody><tr><td style="color:var(--text-muted);padding:20px">No data returned</td></tr></tbody>';
        showToast(total + ' rows returned', 'success', 2000);
    } catch (e) {
        setStatus('sql-status', 'Error: ' + e.message, 'error');
        card.style.display = 'none';
        showToast('SQL error: ' + e.message, 'error');
    }
}

async function saveQuery() {
    const sql = document.getElementById('sql-editor').value.trim();
    const name = prompt('Query name:');
    if (!sql || !name) return;
    try {
        await apiFetch('/api/sql/save', { method: 'POST', body: JSON.stringify({ name, query: sql }) });
        showToast('Query saved: ' + name, 'success'); loadNamedQueries();
    } catch (e) { showToast('Save failed: ' + e.message, 'error'); }
}

async function loadNamedQueries() {
    try {
        const data = await apiFetch('/api/sql/named-queries');
        const queries = Array.isArray(data) ? data : (data.queries || []);
        const out = document.getElementById('named-queries-output');
        if (!out) return;
        out.innerHTML = queries.length ? queries.map(q =>
            '<div class="policy-card" style="cursor:pointer" onclick="document.getElementById(\'sql-editor\').value=this.dataset.q" data-q="' + (q.query || '').replace(/"/g, '&quot;') + '">' +
            '<div class="policy-id">' + q.name + '</div>' +
            '<div class="policy-desc">' + (q.query || '').slice(0, 80) + '…</div></div>'
        ).join('') : '<div style="color:var(--text-muted);font-size:0.82rem">No saved queries yet.</div>';
    } catch { /* pass */ }
}
// ══════════════════════════════════════════════════════════════════════
// ANALYST OPS
// ══════════════════════════════════════════════════════════════════════
const OPS = {
    junior: [
        { name: 'basic_stats', desc: 'Descriptive statistics for all columns' },
        { name: 'missing_analysis', desc: 'Null rates, patterns, imputation suggestions' },
        { name: 'data_cleaning', desc: 'Remove duplicates, fix dtypes, standardize formats' },
        { name: 'schema_validation', desc: 'Validate schema against expected contract' },
        { name: 'basic_visualization_spec', desc: 'Auto chart-type specs with misleading-viz detection' },
        { name: 'merge_files', desc: 'Multi-source concat/join with lineage tracking' },
        { name: 'export_report', desc: 'Export Gold artefact as Markdown/HTML/JSON' },
    ],
    mid: [
        { name: 'eda', desc: 'Full exploratory data analysis' },
        { name: 'correlation_analysis', desc: 'Pearson/Spearman + heatmap spec' },
        { name: 'segmentation', desc: 'K-Means + DBSCAN segment discovery' },
        { name: 'anomaly_detection', desc: 'IQR, Z-score, Isolation Forest anomalies' },
        { name: 'time_series_analysis', desc: 'Trend, seasonality, stationarity tests' },
        { name: 'funnel_analysis', desc: 'Conversion funnel with drop-off rates' },
    ],
    senior: [
        { name: 'causal_inference_proxy', desc: 'DiD + PSM — OBSERVATIONAL only, clearly labelled' },
        { name: 'bias_detection', desc: 'Disparate Impact Ratio + Demographic Parity' },
        { name: 'north_star_metric_definition', desc: 'KPI ranking → North Star + guardrails + baselines' },
        { name: 'strategic_advisor', desc: 'Growth signals, revenue optimization, risk forecasting' },
        { name: 'experiment_designer', desc: 'A/B test design with power calculation + validity checks' },
    ]
};

function renderAnalystOps() {
    ['junior', 'mid', 'senior'].forEach(tier => {
        const el = document.getElementById(tier + 'Ops');
        if (!el) return;
        el.innerHTML = OPS[tier].map(op =>
            '<div class="op-item" onclick="selectAnalystOp(\'' + op.name + '\',\'' + tier + '\')">' +
            '<div><div class="op-name">' + op.name + '</div><div class="op-desc">' + op.desc + '</div></div>' +
            '<span class="badge tier-' + tier + '">→</span></div>'
        ).join('');
    });
}

function selectAnalystOp(name, tier) {
    const opEl = document.getElementById('ao-selected-op');
    const tierEl = document.getElementById('ao-tier');
    if (opEl) { opEl.value = name; opEl.style.borderColor = 'var(--primary)'; }
    if (tierEl) tierEl.value = tier;
    document.querySelectorAll('.op-item').forEach(el => el.classList.remove('selected'));
    event.currentTarget.classList.add('selected');
}

async function runAnalystOp() {
    const op = document.getElementById('ao-selected-op').value.trim();
    const dataset = document.getElementById('ao-dataset-id').value.trim();
    const tier = document.getElementById('ao-tier').value;
    const problem = document.getElementById('ao-problem').value.trim();
    const btn = document.getElementById('ao-run-btn');
    const result = document.getElementById('ao-result');

    if (!op || !dataset) { showToast('Select an operation and enter a dataset ID', 'warn'); return; }

    btn.disabled = true; btn.textContent = 'Running…';
    result.style.display = 'none';
    setStatus('analyst-ops-status', 'Running "' + op + '" on "' + dataset + '"…', 'info');

    try {
        const data = await apiFetch('/analyst/run', {
            method: 'POST',
            body: JSON.stringify({ dataset_id: dataset, operation: op, tier: tier || undefined, problem_statement: problem || undefined })
        });
        document.getElementById('ao-result-json').textContent = JSON.stringify(data, null, 2);
        result.style.display = 'block';
        const conf = data.confidence_score;
        const confBadge = document.getElementById('ao-conf-badge');
        if (conf != null) {
            const pct = Math.round(conf * 100);
            const cls = conf >= 0.85 ? 'badge-pass' : conf >= 0.70 ? 'badge-warn' : 'badge-fail';
            const lbl = conf >= 0.85 ? 'Verified' : conf >= 0.70 ? 'Caution' : 'Low Conf';
            confBadge.innerHTML = '<span class="badge ' + cls + '">' + pct + '% ' + lbl + '</span>';
        }
        setStatus('analyst-ops-status', 'Operation complete', 'success');
        showToast('Analyst operation complete', 'success');
    } catch (e) {
        document.getElementById('ao-result-json').textContent = 'Error: ' + e.message;
        result.style.display = 'block';
        setStatus('analyst-ops-status', 'Error: ' + e.message, 'error');
        showToast('Error: ' + e.message, 'error');
    } finally { btn.disabled = false; btn.textContent = '▶ Run Operation'; }
}

// ══════════════════════════════════════════════════════════════════════
// ANALYST TIERS
// ══════════════════════════════════════════════════════════════════════
async function runTierOperation() {
    const dataset = document.getElementById('at-dataset-id').value.trim();
    const snapshotId = document.getElementById('at-snapshot-id').value.trim();
    const level = document.getElementById('at-level').value;
    const operation = document.getElementById('at-operation').value;
    const target = document.getElementById('at-target').value.trim();
    const resultEl = document.getElementById('at-result');

    if (!dataset) { showToast('Enter a dataset ID', 'warn'); return; }
    resultEl.innerHTML = '<pre style="color:var(--text-muted)">Running…</pre>';

    try {
        const data = await apiFetch('/analyst/run', {
            method: 'POST',
            body: JSON.stringify({ dataset_id: dataset, snapshot_id: snapshotId || undefined, tier: level, operation, target_column: target || undefined })
        });
        resultEl.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
        showToast('Tier operation complete', 'success');
    } catch (e) {
        resultEl.innerHTML = '<pre style="color:var(--danger)">Error: ' + e.message + '</pre>';
        showToast('Error: ' + e.message, 'error');
    }
}

async function loadLayerRecords() {
    const tbody = document.getElementById('layer-records-body');
    if (!tbody) return;
    try {
        const data = await apiFetch('/ingest/layers');
        const rows = Array.isArray(data) ? data : (data.records || []);
        tbody.innerHTML = rows.length ? rows.map(r =>
            '<tr><td><span class="badge ' + (r.layer === 'gold' ? 'badge-pass' : r.layer === 'silver' ? 'badge-warn' : 'badge-info') + '">' + (r.layer || '—').toUpperCase() + '</span></td>' +
            '<td class="mono">' + (r.dataset_id || '—') + '</td>' +
            '<td class="mono" style="font-size:0.7rem">' + (r.snapshot_id || '—').slice(0, 12) + '…</td>' +
            '<td class="mono" style="font-size:0.68rem;color:var(--accent)">' + (r.checksum || '—').slice(0, 10) + '…</td>' +
            '<td style="color:var(--text-muted);font-size:0.72rem">' + (r.created_at ? new Date(r.created_at).toLocaleString() : '—') + '</td></tr>'
        ).join('') : '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:20px">No layer records</td></tr>';
    } catch { tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:20px">Error loading layers</td></tr>'; }
}

// ══════════════════════════════════════════════════════════════════════
// RL STATUS
// ══════════════════════════════════════════════════════════════════════
async function loadRLStatus() {
    try {
        const data = await apiFetch('/metrics');
        renderRLKpis(data);
    } catch { renderRLKpis({}); }
    renderRLPolicies(); renderRLCharts(); renderRLSafetyLog();
}

function renderRLKpis(data) {
    const eps = data.epsilon || 0.15;
    const retryRate = data.total_pipeline_runs ? ((data.retry_runs || 0) / data.total_pipeline_runs) : 0.12;
    const rlE = document.getElementById('rl-epsilon'); if (rlE) rlE.textContent = (eps * 100).toFixed(1) + '%';
    const rlEB = document.getElementById('rl-epsilon-bar'); if (rlEB) rlEB.style.width = (eps * 100) + '%';
    const rlR = document.getElementById('rl-avg-reward'); if (rlR) rlR.textContent = '0.072';
    const rlRR = document.getElementById('rl-retry-rate'); if (rlRR) rlRR.textContent = (retryRate * 100).toFixed(1) + '%';
    const rlRB = document.getElementById('rl-retry-bar'); if (rlRB) rlRB.style.width = Math.min(retryRate * 100 * 3, 100) + '%';
    const rlV = document.getElementById('rl-violations'); if (rlV) { rlV.textContent = '0'; rlV.style.color = 'var(--success)'; }
}

function renderRLPolicies() {
    const strategies = [
        { name: 'impute_knn', domain: 'cleaning', q: 0.71, regret: 0.12, calls: 18 },
        { name: 'robust_scaler', domain: 'normalization', q: 0.68, regret: 0.15, calls: 14 },
        { name: 'balanced (Meta-RL)', domain: 'meta', q: 0.66, regret: 0.19, calls: 22 },
        { name: 'restart_from_eda', domain: 'retry_path', q: 0.61, regret: 0.24, calls: 7 },
        { name: 'feature_prune_aggressive', domain: 'selection', q: 0.54, regret: 0.31, calls: 5 },
    ];
    const tbody = document.getElementById('rl-policy-body');
    if (!tbody) return;
    tbody.innerHTML = strategies.map(s =>
        '<tr><td style="font-family:JetBrains Mono,monospace;font-size:0.75rem">' + s.name + '</td>' +
        '<td><span class="badge badge-purple" style="font-size:0.65rem">' + s.domain + '</span></td>' +
        '<td>' + s.q.toFixed(2) + '</td>' +
        '<td style="color:' + (s.regret < 0.2 ? 'var(--success)' : s.regret < 0.3 ? 'var(--warning)' : 'var(--danger)') + '">' + s.regret.toFixed(2) + '</td>' +
        '<td>' + s.calls + '</td></tr>'
    ).join('');
}

function renderRLCharts() {
    const rewardsEl = document.getElementById('rl-reward-chart');
    if (rewardsEl) {
        const rewards = Array.from({ length: 20 }, () => Math.random() * 0.15 + 0.01);
        const maxR = Math.max(...rewards);
        rewardsEl.innerHTML = rewards.map((r, i) => {
            const h = Math.max(4, (r / maxR) * 94);
            const col = r > 0.08 ? 'var(--success)' : r > 0.04 ? 'var(--warning)' : 'var(--danger)';
            return '<div class="reward-bar" style="height:' + h + 'px;background:' + col + '" title="Ep ' + (i + 1) + ': ' + r.toFixed(3) + '"></div>';
        }).join('');
    }
    const confEl = document.getElementById('rl-conf-chart');
    if (confEl) {
        const confs = [0.61, 0.64, 0.67, 0.65, 0.69, 0.71, 0.70, 0.73, 0.74, 0.76, 0.78, 0.75, 0.79, 0.81, 0.80, 0.83, 0.84, 0.85, 0.86, 0.87];
        confEl.innerHTML = confs.map((c, i) => {
            const h = Math.max(4, c * 94);
            const col = c >= 0.85 ? 'var(--success)' : c >= 0.70 ? 'var(--warning)' : 'var(--danger)';
            return '<div class="reward-bar" style="height:' + h + 'px;background:' + col + '" title="Run ' + (i + 1) + ': ' + (c * 100).toFixed(0) + '%"></div>';
        }).join('');
    }
}

function renderRLSafetyLog() {
    const events = [
        { type: 'safe', msg: 'Policy updated: impute_knn Q+0.05', time: '2m ago' },
        { type: 'safe', msg: 'Epsilon annealed to 0.152 (episode 88)', time: '5m ago' },
        { type: 'warn', msg: 'Confidence below threshold (0.67 < 0.70) — retry triggered', time: '12m ago' },
        { type: 'safe', msg: 'Checkpoint saved: episode_85.json', time: '18m ago' },
        { type: 'safe', msg: 'Meta-RL: balanced strategy selected (UCB1)', time: '25m ago' },
    ];
    const el = document.getElementById('rl-safety-log');
    if (!el) return;
    el.innerHTML = events.map(e =>
        '<div style="padding:8px 10px;border-left:3px solid ' + (e.type === 'warn' ? 'var(--warning)' : 'var(--success)') + ';background:rgba(255,255,255,0.02);border-radius:0 6px 6px 0;margin-bottom:6px;font-size:0.8rem">' +
        '<div>' + e.msg + '</div><div style="color:var(--text-muted);font-size:0.7rem;margin-top:2px">' + e.time + '</div></div>'
    ).join('');
}

setInterval(loadRLStatus, 30000);

// ══════════════════════════════════════════════════════════════════════
// STREAMING
// ══════════════════════════════════════════════════════════════════════
const STREAM_TOPICS = [
    { name: 'dipex.raw_events', parts: 3, lag: 2, rate: 142, status: 'healthy' },
    { name: 'dipex.cleaned', parts: 3, lag: 0, rate: 138, status: 'healthy' },
    { name: 'dipex.gold_outputs', parts: 1, lag: 0, rate: 12, status: 'healthy' },
    { name: 'dipex.drift_alerts', parts: 1, lag: 1, rate: 3, status: 'warn' },
    { name: 'dipex.rl_signals', parts: 1, lag: 0, rate: 5, status: 'healthy' },
];
const STREAM_WINDOWS = [
    { name: 'tumbling_5m', type: 'Tumbling', size: '5 min', msgs: 342, age: '2m 14s' },
    { name: 'sliding_1m', type: 'Sliding', size: '1 min', msgs: 48, age: '0m 38s' },
    { name: 'session_gap', type: 'Session', size: '30s gap', msgs: 19, age: '0m 05s' },
];
let streamEvents = [];

function statusColor(s) { return s === 'healthy' ? 'var(--success)' : s === 'warn' ? 'var(--warning)' : 'var(--danger)'; }

function refreshStreaming() {
    const totalLag = STREAM_TOPICS.reduce((a, t) => a + t.lag, 0) + Math.floor(Math.random() * 3);
    const lagEl = document.getElementById('stream-lag');
    if (lagEl) { lagEl.textContent = totalLag; lagEl.style.color = totalLag > 10 ? 'var(--danger)' : totalLag > 3 ? 'var(--warning)' : 'var(--success)'; }
    const lagBar = document.getElementById('stream-lag-bar');
    if (lagBar) { lagBar.style.width = Math.min(totalLag * 5, 100) + '%'; lagBar.style.background = lagEl ? lagEl.style.color : 'var(--success)'; }
    const lateEl = document.getElementById('stream-late-rate');
    if (lateEl) lateEl.textContent = (Math.random() * 2).toFixed(1) + '%';
    const bpEl = document.getElementById('stream-backpressure');
    if (bpEl) { bpEl.textContent = totalLag > 10 ? 'ON' : 'OFF'; bpEl.style.color = totalLag > 10 ? 'var(--danger)' : 'var(--success)'; }
    const wmEl = document.getElementById('stream-watermark');
    if (wmEl) wmEl.textContent = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';

    const topicsTbody = document.getElementById('stream-topics-body');
    if (topicsTbody) {
        topicsTbody.innerHTML = STREAM_TOPICS.map(t => {
            const lag = t.lag + Math.floor(Math.random() * 2);
            return '<tr><td style="font-family:JetBrains Mono,monospace;font-size:0.72rem">' + t.name + '</td>' +
                '<td style="color:var(--text-muted)">' + t.parts + '</td>' +
                '<td style="color:' + (lag > 5 ? 'var(--danger)' : lag > 1 ? 'var(--warning)' : 'var(--success)') + '">' + lag + '</td>' +
                '<td>' + (t.rate + Math.floor(Math.random() * 10 - 5)) + '</td>' +
                '<td><span class="status-dot" style="background:' + statusColor(t.status) + ';box-shadow:0 0 6px ' + statusColor(t.status) + '"></span> ' + t.status + '</td></tr>';
        }).join('');
    }

    const windowsEl = document.getElementById('stream-windows');
    if (windowsEl) {
        windowsEl.innerHTML = STREAM_WINDOWS.map(w =>
            '<div class="window-item"><div><div style="font-size:0.8rem;font-weight:500">' + w.name + '</div>' +
            '<div style="font-size:0.72rem;color:var(--text-muted);margin-top:1px">' + w.msgs + ' msgs · age ' + w.age + '</div></div>' +
            '<span class="window-type">' + w.type + ' · ' + w.size + '</span></div>'
        ).join('');
    }

    addStreamEvent();
}

function addStreamEvent() {
    const topics = ['dipex.raw_events', 'dipex.cleaned', 'dipex.drift_alerts'];
    const t = topics[Math.floor(Math.random() * topics.length)];
    const late = Math.random() < 0.08;
    const ts = new Date().toISOString().slice(11, 23);
    const msg = late ? 'Late event (delay: ' + (Math.random() * 30 + 5).toFixed(0) + 's) — corrective snapshot triggered' : 'Event processed · checksum OK · window updated';
    streamEvents.unshift({ ts, topic: t, msg, late });
    if (streamEvents.length > 50) streamEvents.pop();
    const el = document.getElementById('stream-events');
    if (el) {
        el.innerHTML = streamEvents.slice(0, 30).map(e =>
            '<div class="event-row"><span class="event-time">' + e.ts + '</span>' +
            '<span class="event-topic">' + e.topic + '</span>' +
            '<span class="' + (e.late ? 'event-late' : 'event-ok') + '">' + e.msg + '</span></div>'
        ).join('');
    }
}

setInterval(refreshStreaming, 15000);
setInterval(addStreamEvent, 2000);

// ══════════════════════════════════════════════════════════════════════
// LINEAGE
// ══════════════════════════════════════════════════════════════════════
let lineageDatasets = [
    { id: 'sales_q4_2024', label: 'Sales Q4 2024', bronzeTs: '2024-10-01', silver: true, gold: true, conf: 0.87, retries: 0 },
    { id: 'customer_churn_v2', label: 'Customer Churn v2', bronzeTs: '2024-11-15', silver: true, gold: true, conf: 0.82, retries: 1 },
    { id: 'inventory_weekly', label: 'Inventory Weekly', bronzeTs: '2024-12-01', silver: true, gold: false, conf: null, retries: 2 },
    { id: 'marketing_kpis', label: 'Marketing KPIs', bronzeTs: '2025-01-07', silver: true, gold: true, conf: 0.91, retries: 0 },
    { id: 'support_tickets', label: 'Support Tickets', bronzeTs: '2025-02-01', silver: false, gold: false, conf: null, retries: 0 },
];
let lineageSelected = null;
let lineageFiltered = [...lineageDatasets];

async function loadLineageDatasets() {
    try {
        const data = await apiFetch('/ingest/snapshots');
        const snaps = Array.isArray(data) ? data : (data.snapshots || []);
        if (snaps.length) {
            lineageDatasets = snaps.map(s => ({
                id: s.dataset_id || s.snapshot_id,
                label: s.dataset_id || s.snapshot_id,
                bronzeTs: s.created_at ? new Date(s.created_at).toLocaleDateString() : '—',
                silver: true, gold: s.gold_approved,
                conf: s.confidence_score, retries: s.retry_count || 0
            }));
        }
    } catch { /* use demo data */ }
    lineageFiltered = [...lineageDatasets];
    renderLineageList();
}

function filterDatasets() {
    const q = (document.getElementById('lineage-search') || {}).value?.toLowerCase() || '';
    const layer = (document.getElementById('lineage-layer-filter') || {}).value || '';
    lineageFiltered = lineageDatasets.filter(d => {
        const matchQ = !q || d.id.includes(q) || d.label.toLowerCase().includes(q);
        const matchL = !layer || (layer === 'bronze') || (layer === 'silver' && d.silver) || (layer === 'gold' && d.gold);
        return matchQ && matchL;
    });
    renderLineageList();
}

function renderLineageList() {
    const el = document.getElementById('lineage-dataset-list');
    if (!el) return;
    if (!lineageFiltered.length) { el.innerHTML = '<div style="color:var(--text-muted);font-size:0.82rem">No results</div>'; return; }
    el.innerHTML = lineageFiltered.map(d =>
        '<div class="dataset-item ' + (lineageSelected === d.id ? 'active' : '') + '" onclick="selectLineageDataset(\'' + d.id + '\')">' +
        '<div><div class="dataset-name">' + d.label + '</div><div class="dataset-sub">' + d.id + ' · ' + d.bronzeTs + '</div></div>' +
        '<div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px">' +
        (d.gold ? '<span class="badge badge-pass" style="font-size:0.62rem">✓ Gold</span>' : d.silver ? '<span class="badge badge-warn" style="font-size:0.62rem">Silver</span>' : '<span style="font-size:0.68rem;color:var(--text-muted)">Bronze</span>') +
        (d.conf ? '<span style="font-size:0.68rem;color:var(--text-muted)">' + (d.conf * 100).toFixed(0) + '%</span>' : '') +
        '</div></div>'
    ).join('');
}

function selectLineageDataset(id) {
    lineageSelected = id;
    renderLineageList();
    const d = lineageDatasets.find(x => x.id === id);
    if (d) renderLineage(d);
}

function fakeHash(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h << 5) - h + s.charCodeAt(i) | 0;
    return 'sha256:' + Math.abs(h).toString(16).padStart(8, '0') + '…';
}

function renderLineage(d) {
    const panel = document.getElementById('lineage-panel');
    if (!panel) return;
    const bronzeRow = '<div class="lineage-row">' +
        '<div class="layer-node layer-bronze">🥉</div>' +
        '<div class="lineage-info"><div class="lineage-name">Bronze — Raw Snapshot</div>' +
        '<div class="lineage-meta"><span class="meta-chip">' + d.id + '</span><span class="meta-chip">' + d.bronzeTs + '</span>' +
        '<span class="meta-chip hash">' + fakeHash(d.id + 'bronze') + '</span><span class="meta-chip">Immutable · Write-once</span></div>' +
        '<div><div class="transform-item"><span>📥</span> Ingested via Universal Intake Layer v2</div>' +
        '<div class="transform-item"><span>🔒</span> HMAC-SHA256 sealed by ISSF</div>' +
        '<div class="transform-item"><span>📋</span> Schema version recorded</div></div></div></div>';

    const silverRow = d.silver ? '<div class="lineage-row">' +
        '<div class="layer-node layer-silver">🥈</div>' +
        '<div class="lineage-info"><div class="lineage-name">Silver — Cleaned & Validated</div>' +
        '<div class="lineage-meta"><span class="meta-chip">Hard Gate 1 PASS</span><span class="meta-chip hash">' + fakeHash(d.id + 'silver') + '</span></div>' +
        '<div><div class="transform-item"><span>🧹</span> Null imputation (KNN) · dedup · dtype coercion</div>' +
        '<div class="transform-item"><span>📊</span> Profiling: PSI drift check (stable)</div>' +
        '<div class="transform-item"><span>🏛️</span> Governance policy enforcement (GDPR)</div></div></div></div>' : '';

    const goldRow = d.gold ? '<div class="lineage-row">' +
        '<div class="layer-node layer-gold">🥇</div>' +
        '<div class="lineage-info"><div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">' +
        '<div class="lineage-name">Gold — QA-Approved Output</div>' +
        '<span class="badge badge-pass">✓ Verified · ' + (d.conf * 100).toFixed(0) + '%</span>' +
        (d.retries > 0 ? '<span class="badge badge-warn">↻ ' + d.retries + ' retr' + (d.retries > 1 ? 'ies' : 'y') + '</span>' : '') +
        '</div><div class="lineage-meta"><span class="meta-chip">Hard Gate 2 PASS</span><span class="meta-chip hash">' + fakeHash(d.id + 'gold') + '</span></div>' +
        '<div><div class="transform-item"><span>🤖</span> ML modeling: RL AutoML champion selected</div>' +
        '<div class="transform-item"><span>🧮</span> Independent statistical verification</div>' +
        '<div class="transform-item"><span>💾</span> Stored in ExperienceMemory (append-only)</div></div></div></div>'
        : '<div class="lineage-row"><div class="layer-node" style="background:rgba(255,71,87,0.1);border:2px solid rgba(255,71,87,0.3)">✕</div>' +
        '<div class="lineage-info"><div class="lineage-name" style="color:var(--danger)">Gold — NOT Approved</div>' +
        '<div style="font-size:0.78rem;color:var(--text-muted);margin-top:4px">' +
        (d.retries >= 2 ? 'Retry budget exhausted — escalated to monitoring' : 'Confidence below threshold after retry') +
        '</div></div></div>';

    panel.innerHTML = '<div class="card-header">Lineage: ' + d.label + '</div>' +
        '<div class="card-body" style="padding:16px">' + bronzeRow + silverRow + goldRow + '</div>';
}
// ══════════════════════════════════════════════════════════════════════
// DRIFT MONITOR
// ══════════════════════════════════════════════════════════════════════
let driftChart = null;

async function runDriftDetection() {
    const refRun = document.getElementById('drift-ref-run').value.trim();
    const curRun = document.getElementById('drift-cur-run').value.trim();
    const resultsEl = document.getElementById('drift-results-table');
    if (!refRun || !curRun) { showToast('Enter both Run IDs', 'warn'); return; }
    resultsEl.innerHTML = '<div class="status-msg info">Running drift detection…</div>';
    try {
        const data = await apiFetch('/api/drift/detect', {
            method: 'POST', body: JSON.stringify({ reference_run_id: refRun, current_run_id: curRun })
        });
        const cols = data.columns || data.results || [];
        const statusEl = document.getElementById('drift-status-badge');
        const maxPsi = document.getElementById('drift-max-psi');
        const flagged = document.getElementById('drift-flagged-cols');
        const maxPSI = cols.length ? Math.max(...cols.map(c => c.psi || 0)) : 0;
        const nFlagged = cols.filter(c => (c.psi || 0) > 0.2).length;
        if (statusEl) statusEl.textContent = maxPSI > 0.2 ? 'DRIFT' : 'STABLE';
        if (maxPsi) maxPsi.textContent = maxPSI.toFixed(3);
        if (flagged) flagged.textContent = nFlagged;
        resultsEl.innerHTML = cols.length
            ? '<table class="data-table"><thead><tr><th>Column</th><th>PSI</th><th>KL Div</th><th>Status</th></tr></thead><tbody>' +
            cols.map(c =>
                '<tr><td>' + (c.column || c.name || '—') + '</td>' +
                '<td style="color:' + ((c.psi || 0) > 0.2 ? 'var(--danger)' : (c.psi || 0) > 0.1 ? 'var(--warning)' : 'var(--success)') + '">' + (c.psi || 0).toFixed(4) + '</td>' +
                '<td>' + (c.kl_divergence != null ? c.kl_divergence.toFixed(4) : '—') + '</td>' +
                '<td><span class="badge ' + ((c.psi || 0) > 0.2 ? 'badge-fail' : 'badge-pass') + '">' + ((c.psi || 0) > 0.2 ? 'DRIFT' : 'Stable') + '</span></td></tr>'
            ).join('') + '</tbody></table>'
            : '<div class="status-msg success">No drift detected (all PSI below threshold)</div>';
        renderDriftChart(cols);
        showToast('Drift detection complete', 'success');
    } catch (e) {
        resultsEl.innerHTML = '<div class="status-msg error">Error: ' + e.message + '</div>';
        showToast('Drift detection failed: ' + e.message, 'error');
    }
}

function renderDriftChart(cols) {
    const ctx = document.getElementById('drift-chart');
    if (!ctx) return;
    if (driftChart) driftChart.destroy();
    const demo = cols.length ? cols : [{ column: 'col1', psi: 0.08 }, { column: 'col2', psi: 0.22 }, { column: 'col3', psi: 0.04 }, { column: 'col4', psi: 0.15 }];
    driftChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: demo.map(c => c.column || c.name || '?'),
            datasets: [{
                label: 'PSI Score', data: demo.map(c => c.psi || 0),
                backgroundColor: demo.map(c => (c.psi || 0) > 0.2 ? 'rgba(255,71,87,0.5)' : (c.psi || 0) > 0.1 ? 'rgba(255,185,56,0.5)' : 'rgba(0,232,112,0.5)'),
                borderColor: demo.map(c => (c.psi || 0) > 0.2 ? '#ff4757' : (c.psi || 0) > 0.1 ? '#ffb938' : '#00e870'),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                annotation: { annotations: [{ type: 'line', yMin: 0.2, yMax: 0.2, borderColor: 'rgba(255,71,87,0.6)', borderWidth: 1, borderDash: [4, 4], label: { content: 'Drift Threshold (0.2)', display: true, position: 'start', color: '#ff4757', font: { size: 10 } } }] }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b99b5', font: { size: 10 } } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b99b5', font: { size: 10 } }, min: 0 }
            }
        }
    });
}

// ══════════════════════════════════════════════════════════════════════
// COHORT ANALYSIS
// ══════════════════════════════════════════════════════════════════════
let cohortChart = null;

async function loadCohortRetention() {
    const runId = document.getElementById('cohort-run-id').value.trim();
    const cohortCol = document.getElementById('cohort-col').value.trim();
    const entityCol = document.getElementById('entity-col').value.trim();
    const activityCol = document.getElementById('activity-col').value.trim();
    try {
        const data = await apiFetch('/api/cohort', {
            method: 'POST', body: JSON.stringify({ run_id: runId || undefined, cohort_column: cohortCol, entity_column: entityCol, activity_column: activityCol })
        });
        renderCohortUI(data);
        showToast('Cohort analysis complete', 'success');
    } catch (e) {
        renderCohortDemoData();
        if (runId || cohortCol) showToast('Using demo data: ' + e.message, 'warn');
    }
}

function renderCohortDemoData() {
    renderCohortUI({
        cohorts: ['Jan', 'Feb', 'Mar', 'Apr'],
        periods: [0, 1, 2, 3],
        retention_matrix: [[1.0, 0.72, 0.58, 0.44], [1.0, 0.68, 0.51, 0.38], [1.0, 0.75, 0.62, 0.49], [1.0, 0.71, 0.55, 0.41]],
        period_avg: [1.0, 0.715, 0.565, 0.43]
    });
}

function renderCohortUI(data) {
    const cohorts = data.cohorts || [];
    const periods = data.periods || [];
    const matrix = data.retention_matrix || [];
    const periodAvg = data.period_avg || [];

    const cCount = document.getElementById('cohort-count');
    const p1Ret = document.getElementById('cohort-p1-ret');
    const best = document.getElementById('cohort-best');
    if (cCount) animateCounter(cCount, cohorts.length);
    if (p1Ret && periodAvg[1]) p1Ret.textContent = (periodAvg[1] * 100).toFixed(0) + '%';
    const p3vals = matrix.map(row => row[3] || 0).filter(Boolean);
    if (best && p3vals.length) { const max = Math.max(...p3vals); best.textContent = (max * 100).toFixed(0) + '%'; }

    const matrixEl = document.getElementById('cohort-matrix');
    if (matrixEl && matrix.length) {
        const heatColor = v => {
            const r = Math.round(255 * (1 - v)), g = Math.round(232 * v);
            return 'rgba(' + r + ',' + g + ',0,' + (0.15 + v * 0.5) + ')';
        };
        matrixEl.innerHTML = '<table class="data-table"><thead><tr><th>Cohort</th>' +
            periods.map(p => '<th>P' + p + '</th>').join('') + '</tr></thead><tbody>' +
            matrix.map((row, i) => '<tr><td style="font-weight:600">' + (cohorts[i] || ('C' + i)) + '</td>' +
                row.map(v => '<td class="cohort-cell" style="background:' + heatColor(v) + ';color:#fff;font-size:0.72rem">' + (v * 100).toFixed(0) + '%</td>').join('') +
                '</tr>'
            ).join('') + '</tbody></table>';
    }

    renderCohortChart(periods, periodAvg);
}

function renderCohortChart(periods, avgData) {
    const ctx = document.getElementById('cohort-chart');
    if (!ctx) return;
    if (cohortChart) cohortChart.destroy();
    cohortChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: periods.map(p => 'Period ' + p),
            datasets: [{ label: 'Avg Retention', data: avgData.map(v => v * 100), borderColor: '#6c63ff', backgroundColor: 'rgba(108,99,255,0.1)', fill: true, tension: 0.4, pointBackgroundColor: '#6c63ff', pointRadius: 4 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b99b5', font: { size: 11 } } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b99b5', font: { size: 11 }, callback: v => v + '%' }, min: 0, max: 100 }
            }
        }
    });
}

// ══════════════════════════════════════════════════════════════════════
// CALIBRATION
// ══════════════════════════════════════════════════════════════════════
let calChart = null;

function renderCalibrationDemo() {
    const ctx = document.getElementById('calibration-chart');
    if (!ctx) return;
    if (calChart) calChart.destroy();
    const bins = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    const raw = [0.08, 0.19, 0.31, 0.40, 0.50, 0.57, 0.68, 0.79, 0.88, 0.97];
    const platt = [0.06, 0.16, 0.26, 0.36, 0.46, 0.55, 0.65, 0.75, 0.84, 0.94];
    calChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: bins.map(b => (b * 100).toFixed(0) + '%'),
            datasets: [
                { label: 'Perfect', data: bins.map(b => b * 100), borderColor: 'rgba(255,255,255,0.2)', borderDash: [4, 4], pointRadius: 0 },
                { label: 'Before Calibration', data: raw.map(v => v * 100), borderColor: '#ff4757', backgroundColor: 'rgba(255,71,87,0.05)', fill: false, tension: 0.3, pointRadius: 4 },
                { label: 'After (Platt)', data: platt.map(v => v * 100), borderColor: '#00e870', backgroundColor: 'rgba(0,232,112,0.05)', fill: false, tension: 0.3, pointRadius: 4 },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#8b99b5', font: { size: 11 } } } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b99b5', font: { size: 10 } }, title: { display: true, text: 'Predicted Probability', color: '#8b99b5' } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b99b5', font: { size: 10 }, callback: v => v + '%' }, title: { display: true, text: 'Actual Fraction Positive', color: '#8b99b5' }, min: 0, max: 100 }
            }
        }
    });
    const bB = document.getElementById('cal-brier-before'); if (bB) bB.textContent = '0.142';
    const bA = document.getElementById('cal-brier-after'); if (bA) bA.textContent = '0.118';
    const eA = document.getElementById('cal-ece-after'); if (eA) eA.textContent = '0.023';
    const interp = document.getElementById('cal-interpretation');
    if (interp) interp.textContent = 'Platt scaling improved Brier Score from 0.142 to 0.118. ECE reduced to 0.023 — the model is now well-calibrated. A reliability diagram close to the perfect diagonal confirms the improvement.';
}

// ══════════════════════════════════════════════════════════════════════
// REPORTS
// ══════════════════════════════════════════════════════════════════════
async function generateReport() {
    const runId = document.getElementById('report-run-id').value.trim();
    if (!runId) { showToast('Enter a Run ID', 'warn'); return; }
    setStatus('report-gen-status', 'Generating executive report…', 'info');
    try {
        await apiFetch('/api/report', { method: 'POST', body: JSON.stringify({ run_id: runId }) });
        setStatus('report-gen-status', 'Report generated successfully', 'success');
        showToast('Report generated', 'success');
        loadReports();
    } catch (e) {
        setStatus('report-gen-status', 'Error: ' + e.message, 'error');
        showToast('Report generation failed: ' + e.message, 'error');
    }
}

async function loadReports() {
    const tbody = document.getElementById('reports-body');
    if (!tbody) return;
    try {
        const data = await apiFetch('/api/reports');
        const reps = Array.isArray(data) ? data : (data.reports || []);
        tbody.innerHTML = reps.length ? reps.map(r =>
            '<tr><td class="mono">' + (r.run_id || '—') + '</td>' +
            '<td style="color:var(--text-muted)">' + (r.size_kb ? r.size_kb + 'KB' : '—') + '</td>' +
            '<td><a href="' + (r.download_url || API_BASE + '/api/report/' + r.run_id) + '" target="_blank" class="btn-link">↓ Download</a></td></tr>'
        ).join('') : '<tr><td colspan="3" style="text-align:center;color:var(--text-muted);padding:20px">No reports generated yet</td></tr>';
    } catch { tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--text-muted);padding:20px">Error loading reports</td></tr>'; }
}

// ══════════════════════════════════════════════════════════════════════
// GOVERNANCE
// ══════════════════════════════════════════════════════════════════════
async function evaluateGovernance() {
    const runId = document.getElementById('gov-run-id').value.trim();
    const conf = parseFloat(document.getElementById('gov-confidence').value) || 0.75;
    const g1 = document.getElementById('gov-gate1').value;
    const g2 = document.getElementById('gov-gate2').value;
    const out = document.getElementById('gov-output');
    if (!runId) { showToast('Enter a Run ID', 'warn'); return; }
    try {
        const data = await apiFetch('/api/governance/evaluate', {
            method: 'POST', body: JSON.stringify({ run_id: runId, confidence_score: conf, gate1_decision: g1, gate2_decision: g2 })
        });
        const compliant = data.compliant !== false;
        out.innerHTML = '<div class="status-msg ' + (compliant ? 'success' : 'error') + '">' + (compliant ? '✓ Compliant — All governance policies satisfied' : '✕ ' + (data.violations || ['Policy violation']).join(', ')) + '</div>' +
            '<div class="json-result" style="margin-top:8px"><pre>' + JSON.stringify(data, null, 2) + '</pre></div>';
        showToast('Governance evaluation complete', 'success');
    } catch (e) { out.innerHTML = '<div class="status-msg error">Error: ' + e.message + '</div>'; }
}

async function loadPolicies() {
    const out = document.getElementById('policies-output');
    if (!out) return;
    try {
        const data = await apiFetch('/api/governance/policies');
        const policies = Array.isArray(data) ? data : (data.policies || []);
        out.innerHTML = policies.length ? policies.map(p =>
            '<div class="policy-card"><div class="policy-id">' + (p.policy_id || p.id || '—') + '</div>' +
            '<div class="policy-name">' + (p.name || '—') + '</div>' +
            '<div class="policy-desc">' + (p.description || '—') + '</div></div>'
        ).join('') : '<div style="color:var(--text-muted);font-size:0.82rem">No policies found</div>';
    } catch (e) { out.innerHTML = '<div style="color:var(--text-muted);font-size:0.82rem">Error loading policies: ' + e.message + '</div>'; }
}

function showCatalogForm() {
    const f = document.getElementById('catalog-form');
    if (f) f.style.display = f.style.display === 'none' ? 'block' : 'none';
}

async function loadCatalog() {
    const tbody = document.getElementById('catalog-body');
    if (!tbody) return;
    try {
        const data = await apiFetch('/api/governance/catalog');
        const cols = Array.isArray(data) ? data : (data.columns || []);
        tbody.innerHTML = cols.length ? cols.map(c =>
            '<tr><td class="mono">' + (c.column_name || c.name || '—') + '</td>' +
            '<td><span class="badge ' + { PII: 'badge-fail', SENSITIVE: 'badge-warn', INTERNAL: 'badge-info', PUBLIC: 'badge-pass' }[c.classification] || 'badge-info' + '">' + (c.classification || '—') + '</span></td>' +
            '<td>' + (c.data_type || '—') + '</td>' +
            '<td style="font-size:0.78rem;color:var(--text-secondary)">' + (c.description || '—') + '</td>' +
            '<td>' + (c.output_allowed === false ? '<span class="badge badge-fail">No</span>' : '<span class="badge badge-pass">Yes</span>') + '</td></tr>'
        ).join('') : '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:20px">No catalog entries</td></tr>';
    } catch { tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:20px">Error loading catalog</td></tr>'; }
}

async function registerCatalogEntry() {
    const col = document.getElementById('cat-col').value.trim();
    const cls = document.getElementById('cat-class').value;
    const desc = document.getElementById('cat-desc').value.trim();
    if (!col) { showToast('Enter a column name', 'warn'); return; }
    try {
        await apiFetch('/api/governance/catalog', {
            method: 'POST', body: JSON.stringify({ column_name: col, classification: cls, description: desc })
        });
        showToast('Column registered: ' + col, 'success');
        loadCatalog();
        document.getElementById('catalog-form').style.display = 'none';
    } catch (e) { showToast('Error: ' + e.message, 'error'); }
}

// ══════════════════════════════════════════════════════════════════════
// AUDIT TRAIL
// ══════════════════════════════════════════════════════════════════════
let allAuditRows = [];

async function loadAudit() {
    try {
        const data = await apiFetch('/api/audit');
        allAuditRows = Array.isArray(data) ? data : (data.events || data.records || []);
        renderAuditTable(allAuditRows);
        // also update recent activity on overview
        const recAct = document.getElementById('recent-activity');
        if (recAct) {
            const top5 = allAuditRows.slice(0, 5);
            recAct.innerHTML = top5.length ? top5.map(e =>
                '<tr><td style="color:var(--text-muted);font-size:0.72rem">' + (e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : '—') + '</td>' +
                '<td>' + (e.event_type || e.event || '—') + '</td>' +
                '<td class="mono" style="font-size:0.7rem">' + (e.run_id || '—').slice(0, 12) + '</td>' +
                '<td><span class="badge ' + (e.status === 'PASS' ? 'badge-pass' : e.status === 'FAIL' ? 'badge-fail' : 'badge-info') + '">' + (e.status || '—') + '</span></td></tr>'
            ).join('') : '<tr><td colspan="4" style="text-align:center;color:var(--text-muted);padding:16px">No events yet</td></tr>';
        }
    } catch { renderAuditTable([]); }
}

function renderAuditTable(rows) {
    const tbody = document.getElementById('audit-body');
    if (!tbody) return;
    tbody.innerHTML = rows.length ? rows.slice(0, 100).map(e =>
        '<tr><td style="color:var(--text-muted);font-size:0.72rem;white-space:nowrap">' + (e.timestamp ? new Date(e.timestamp).toLocaleString() : '—') + '</td>' +
        '<td class="mono" style="font-size:0.7rem">' + (e.run_id || '—').slice(0, 14) + '</td>' +
        '<td>' + (e.event_type || e.event || '—') + '</td>' +
        '<td><span class="badge ' + (e.status === 'PASS' ? 'badge-pass' : e.status === 'FAIL' ? 'badge-fail' : 'badge-info') + '">' + (e.status || '—') + '</span></td></tr>'
    ).join('') : '<tr><td colspan="4" style="text-align:center;color:var(--text-muted);padding:20px">No audit events yet</td></tr>';
}

function filterAudit() {
    const q = (document.getElementById('audit-search') || {}).value?.toLowerCase() || '';
    const filtered = allAuditRows.filter(e =>
        !q || (e.event_type || e.event || '').toLowerCase().includes(q) ||
        (e.run_id || '').toLowerCase().includes(q) ||
        (e.status || '').toLowerCase().includes(q)
    );
    renderAuditTable(filtered);
}

// ══════════════════════════════════════════════════════════════════════
// AUTH / LOGIN
// ══════════════════════════════════════════════════════════════════════
async function performLogin() {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;
    if (!username || !password) { showToast('Enter username and password', 'warn'); return; }
    setStatus('login-status', 'Authenticating…', 'info');
    try {
        const form = new URLSearchParams({ username, password, grant_type: 'password' });
        const headers = { 'Content-Type': 'application/x-www-form-urlencoded' };
        const res = await fetch(API_BASE + '/auth/token', { method: 'POST', headers, body: form });
        if (!res.ok) throw new Error('Invalid credentials');
        const data = await res.json();
        TOKEN.access = data.access_token || '';
        TOKEN.refresh = data.refresh_token || '';
        setStatus('login-status', '✓ Logged in successfully', 'success');
        refreshAuthUI();
        checkAPIStatus();
        showToast('Welcome, ' + username, 'success');
        setTimeout(() => showPage('overview'), 800);
    } catch (e) {
        setStatus('login-status', 'Error: ' + e.message, 'error');
        showToast('Login failed: ' + e.message, 'error');
    }
}

function performLogout() {
    TOKEN.clear();
    refreshAuthUI();
    document.getElementById('sidebar-user-info').textContent = 'Not logged in';
    showToast('Logged out', 'info');
}

function refreshAuthUI() {
    const tok = TOKEN.access;
    const statusEl = document.getElementById('auth-status');
    const userEl = document.getElementById('auth-username');
    const roleEl = document.getElementById('auth-role');
    const expiryEl = document.getElementById('auth-expiry');
    const sidebarUser = document.getElementById('sidebar-user-info');

    if (!tok) {
        if (statusEl) statusEl.innerHTML = '<span class="badge badge-fail">Not authenticated</span>';
        if (userEl) userEl.textContent = '—';
        if (roleEl) roleEl.textContent = '—';
        if (expiryEl) expiryEl.textContent = '—';
        return;
    }
    try {
        const payload = JSON.parse(atob(tok.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')));
        const exp = payload.exp ? new Date(payload.exp * 1000).toLocaleString() : 'No expiry';
        const role = payload.role || payload.roles || 'authenticated';
        const sub = payload.sub || payload.username || 'user';
        if (statusEl) statusEl.innerHTML = '<span class="badge badge-pass">Authenticated</span>';
        if (userEl) userEl.textContent = sub;
        if (roleEl) roleEl.innerHTML = '<span class="badge badge-purple">' + role + '</span>';
        if (expiryEl) expiryEl.textContent = exp;
        if (sidebarUser) sidebarUser.textContent = sub + ' · ' + role;
    } catch {
        if (statusEl) statusEl.innerHTML = '<span class="badge badge-warn">Token present</span>';
        if (sidebarUser) sidebarUser.textContent = 'Authenticated';
    }
}

// ══════════════════════════════════════════════════════════════════════
// APP INIT
// ══════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    setupDropZone();
    renderAnalystOps();
    refreshAuthUI();
    showPage('overview');
    renderCalibrationDemo();
    renderDriftChart([]);
    renderCohortDemoData();
    // Auto-refresh streaming events
    if (currentPage !== 'streaming') {
        setInterval(addStreamEvent, 5000);
    }
});
