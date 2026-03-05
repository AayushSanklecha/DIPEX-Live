/**
 * DIPEX — Power BI Styled Dashboard · app.js
 * Presentation Layer · v2.0
 */

'use strict';

/* ═══ Config ═══════════════════════════════════════════════════ */
const API = 'http://localhost:8000';
const POLL_MS = 30_000;  // auto-refresh interval

/* ═══ State ════════════════════════════════════════════════════ */
const state = {
  runs: [],
  filteredRuns: [],
  reports: [],
  activeFilter: 'ALL',
  activeSource: 'ALL',
  tableSearch: '',
  charts: {},
  uploadedFile: null,
  uploading: false,
};

/* ═══ Init ═════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  startClock();
  checkApiHealth();
  refreshAll();
  setInterval(refreshAll, POLL_MS);
  setInterval(checkApiHealth, 10_000);
  setupDropZone();
});

/* ═══ Clock ════════════════════════════════════════════════════ */
function startClock() {
  const el = document.getElementById('ribbon-time');
  const tick = () => {
    const now = new Date();
    el.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };
  tick(); setInterval(tick, 1000);
}

/* ═══ Tab switching ════════════════════════════════════════════ */
function switchTab(tab) {
  document.querySelectorAll('.pbi-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.rtab').forEach(b => b.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
  document.getElementById(`rtab-${tab}`).classList.add('active');
  if (tab === 'report') loadReports();
}

/* ═══ API health ══════════════════════════════════════════════ */
async function checkApiHealth() {
  const pip = document.getElementById('api-pip');
  const lbl = document.getElementById('api-status-label');
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    if (r.ok) {
      pip.classList.add('online');
      lbl.textContent = 'API Online';
    } else throw new Error('bad status');
  } catch {
    pip.classList.remove('online');
    lbl.textContent = 'API Offline';
  }
}

/* ═══ Refresh all ═════════════════════════════════════════════ */
async function refreshAll() {
  await Promise.all([loadRuns(), loadMetrics()]);
}

/* ═══ Load run data ═══════════════════════════════════════════ */
async function loadRuns() {
  try {
    const r = await fetch(`${API}/api/export/results/json?limit=500`);
    if (!r.ok) throw new Error('no data');
    const data = await r.json();
    state.runs = (data.results || []).reverse();
  } catch {
    state.runs = [];
  }
  applyFilters();
  updateKPIs();
  renderCharts();
}

/* ═══ Load metrics ════════════════════════════════════════════ */
async function loadMetrics() {
  try {
    const r = await fetch(`${API}/metrics`);
    if (!r.ok) return;
    const d = await r.json();
    document.getElementById('kpi-rpts').textContent = d.reports_generated ?? '—';
  } catch { /* silent */ }
}

/* ═══ Filters ═════════════════════════════════════════════════ */
function applyFilter(btn, filter) {
  document.querySelectorAll('.slicer-pill').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  state.activeFilter = filter;
  applyFilters();
  renderCharts();
}

function applySourceFilter(val) {
  state.activeSource = val;
  applyFilters();
  renderCharts();
}

function applyFilters() {
  let runs = state.runs;
  if (state.activeFilter !== 'ALL')
    runs = runs.filter(r => (r.gate_decision || '').toUpperCase() === state.activeFilter);
  if (state.activeSource !== 'ALL')
    runs = runs.filter(r => (r.source_type || '').toLowerCase() === state.activeSource.toLowerCase());
  if (state.tableSearch)
    runs = runs.filter(r =>
      (r.run_id || '').toLowerCase().includes(state.tableSearch) ||
      (r.dataset_id || '').toLowerCase().includes(state.tableSearch)
    );
  state.filteredRuns = runs;
  renderTable(runs);
}

function filterTable(q) {
  state.tableSearch = q.toLowerCase();
  applyFilters();
}

/* ═══ KPI Tiles ═══════════════════════════════════════════════ */
function updateKPIs() {
  const runs = state.filteredRuns;
  const total = runs.length;
  const passed = runs.filter(r => (r.gate_decision || '').toUpperCase() === 'PASS').length;
  const passRate = total ? ((passed / total) * 100).toFixed(1) : '—';
  const avgConf = total
    ? (runs.reduce((s, r) => s + (Number(r.confidence_score) || 0), 0) / total).toFixed(2)
    : '—';

  document.getElementById('kpi-runs').textContent = total || '—';
  document.getElementById('kpi-pass').textContent = passRate !== '—' ? passRate + '%' : '—';
  document.getElementById('kpi-conf').textContent = avgConf !== '—' ? avgConf : '—';

  // deltas from full run set
  document.getElementById('kpi-pass-delta').textContent = passed ? `${passed} PASS / ${total - passed} not passed` : '';
  document.getElementById('kpi-runs-delta').textContent = `${total} total (filtered)`;

  // Sparklines
  renderSparkline('spark-runs', runs.map((_, i) => i + 1), '#6366f1');
  renderSparkline('spark-pass', runs.map(r => r.gate_decision === 'PASS' ? 1 : 0), '#12b76a');
  renderSparkline('spark-conf', runs.map(r => Number(r.confidence_score) || 0), '#F2C811');
  renderSparkline('spark-rpts', runs.map((_, i) => Math.round(i * 0.7)), '#118dff');
}

/* ═══ Sparklines ══════════════════════════════════════════════ */
function renderSparkline(id, data, color) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  if (state.charts[id]) { state.charts[id].destroy(); }
  state.charts[id] = new Chart(canvas, {
    type: 'line',
    data: {
      labels: data.map(() => ''),
      datasets: [{ data, borderColor: color, borderWidth: 1.5, fill: true,
        backgroundColor: color + '22', pointRadius: 0, tension: 0.4 }],
    },
    options: {
      responsive: false, plugins: { legend: { display: false } },
      scales: { x: { display: false }, y: { display: false } },
      animation: false,
    },
  });
}

/* ═══ Charts ══════════════════════════════════════════════════ */
function renderCharts() {
  renderDonut();
  renderTrend();
  renderSourceBar();
  renderStageBar();
}

const CHART_DEFAULTS = {
  color: '#94a3b8',
  grid: { color: 'rgba(255,255,255,.06)', drawBorder: false },
  tooltip: {
    backgroundColor: '#0f3460', borderColor: 'rgba(255,255,255,.1)', borderWidth: 1,
    titleColor: '#e8eaf6', bodyColor: '#94a3b8', cornerRadius: 6, padding: 10,
  },
};

function destroyChart(id) {
  if (state.charts[id]) { state.charts[id].destroy(); delete state.charts[id]; }
}

/* Gate Decision Donut */
function renderDonut() {
  destroyChart('donut');
  const runs = state.filteredRuns;
  const pass = runs.filter(r => r.gate_decision === 'PASS').length;
  const warn = runs.filter(r => r.gate_decision === 'WARN').length;
  const fail = runs.filter(r => r.gate_decision === 'FAIL').length;
  const total = runs.length;

  const pct = total ? ((pass / total) * 100).toFixed(0) + '%' : '—';
  document.getElementById('donut-pass-pct').textContent = pct;

  const ctx = document.getElementById('chart-donut').getContext('2d');
  state.charts['donut'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Pass', 'Warn', 'Fail'],
      datasets: [{
        data: [pass || 0, warn || 0, fail || 0],
        backgroundColor: ['#12b76a', '#f79009', '#f04438'],
        borderColor: '#16213e', borderWidth: 3, hoverBorderWidth: 3,
      }],
    },
    options: {
      responsive: true, cutout: '68%',
      plugins: {
        legend: { position: 'bottom', labels: { color: '#94a3b8', font: { size: 11 }, padding: 14, boxWidth: 12 } },
        tooltip: { ...CHART_DEFAULTS.tooltip, callbacks: {
          label: ctx => ` ${ctx.label}: ${ctx.parsed} (${total ? ((ctx.parsed / total)*100).toFixed(1) : 0}%)`,
        }},
      },
    },
  });
}

/* Confidence Trend Line */
function renderTrend() {
  destroyChart('trend');
  const runs = state.filteredRuns.slice(-30);
  const labels = runs.map((r, i) => r.timestamp ? r.timestamp.substring(5, 16) : `Run ${i + 1}`);
  const confData = runs.map(r => +(Number(r.confidence_score) || 0).toFixed(3));

  const ctx = document.getElementById('chart-trend').getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 200);
  grad.addColorStop(0, 'rgba(242,200,17,.35)');
  grad.addColorStop(1, 'rgba(242,200,17,.0)');

  state.charts['trend'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Confidence Score', data: confData, borderColor: '#F2C811',
        backgroundColor: grad, borderWidth: 2, pointRadius: confData.length < 20 ? 4 : 0,
        pointBackgroundColor: '#F2C811', tension: 0.4, fill: true,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: CHART_DEFAULTS.tooltip },
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 8, font: { size: 10 } }, grid: CHART_DEFAULTS.grid },
        y: { min: 0, max: 1, ticks: { color: '#64748b', font: { size: 10 } }, grid: CHART_DEFAULTS.grid },
      },
    },
  });
}

/* Source Type Bar */
function renderSourceBar() {
  destroyChart('source');
  const runs = state.filteredRuns;
  const freq = {};
  runs.forEach(r => { const s = r.source_type || 'unknown'; freq[s] = (freq[s] || 0) + 1; });
  const labels = Object.keys(freq);
  const data = Object.values(freq);
  const colors = ['#6366f1', '#12b76a', '#F2C811', '#f04438', '#06b6d4', '#a855f7'];

  const ctx = document.getElementById('chart-source').getContext('2d');
  state.charts['source'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels.length ? labels : ['No data'],
      datasets: [{
        label: 'Runs', data: data.length ? data : [0],
        backgroundColor: labels.map((_, i) => colors[i % colors.length] + 'cc'),
        borderColor: labels.map((_, i) => colors[i % colors.length]),
        borderWidth: 1, borderRadius: 4,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      plugins: { legend: { display: false }, tooltip: CHART_DEFAULTS.tooltip },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: CHART_DEFAULTS.grid },
        y: { ticks: { color: '#94a3b8', font: { size: 12 } }, grid: { display: false } },
      },
    },
  });
}

/* Stage Performance Horizontal Bar */
function renderStageBar() {
  destroyChart('stages');
  const labels = ['Source Router','Streaming Window','Preprocessing','Hard Gate 1','Profiling','AI Analytics','Governance','ML Modeling','Hard Gate 2','Retry Engine','Report'];
  const passes = state.filteredRuns.length;
  // Simulate stage pass counts from run data
  const data = [passes, passes, passes,
    state.filteredRuns.filter(r => r.gate1_decision === 'PASS').length,
    passes,
    passes,
    passes,
    passes,
    state.filteredRuns.filter(r => r.gate2_decision === 'PASS').length,
    state.filteredRuns.filter(r => r.gate_decision !== 'FAIL').length,
    passes,
  ];

  const ctx = document.getElementById('chart-stages').getContext('2d');
  state.charts['stages'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Passed', data,
        backgroundColor: 'rgba(18,183,106,.7)', borderColor: '#12b76a',
        borderWidth: 1, borderRadius: 4,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: CHART_DEFAULTS.tooltip },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 10 }, maxRotation: 35 }, grid: CHART_DEFAULTS.grid },
        y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: CHART_DEFAULTS.grid },
      },
    },
  });
}

/* ═══ Runs Table ══════════════════════════════════════════════ */
function renderTable(runs) {
  const tbody = document.getElementById('runs-tbody');
  if (!runs.length) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty-row">No runs match the current filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = runs.slice(0, 100).map(r => {
    const conf = Number(r.confidence_score) || 0;
    const confColor = conf >= 0.8 ? '#12b76a' : conf >= 0.5 ? '#f79009' : '#f04438';
    const gate1 = r.gate1_decision || 'PENDING';
    const gate2 = r.gate2_decision || 'PENDING';
    const dec   = r.gate_decision || 'PENDING';
    const ts    = r.timestamp ? r.timestamp.substring(0, 16).replace('T', ' ') : '—';

    return `<tr>
      <td class="run-id-cell" title="${r.run_id || ''}">${(r.run_id || '—').substring(0, 8)}…</td>
      <td>${escHtml(r.dataset_id || '—')}</td>
      <td>${escHtml(r.source_type || '—')}</td>
      <td>${r.row_count ? Number(r.row_count).toLocaleString() : '—'}</td>
      <td><span class="badge-pill ${gate1}">${gate1}</span></td>
      <td><span class="badge-pill ${gate2}">${gate2}</span></td>
      <td>
        <div class="conf-bar">
          <div class="conf-track"><div class="conf-fill" style="width:${Math.min(conf*100,100).toFixed(0)}%;background:${confColor}"></div></div>
          <span class="conf-val">${conf.toFixed(2)}</span>
        </div>
      </td>
      <td><span class="badge-pill ${dec}">${dec}</span></td>
      <td style="font-size:11px;color:#64748b">${ts}</td>
      <td><a class="btn-dl-sm" href="${API}/api/export/report/${r.run_id}" download title="Download report">⤓</a></td>
    </tr>`;
  }).join('');
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* ═══ Run Pipeline ════════════════════════════════════════════ */
function showPage(p) {
  if (p === 'run_pipeline') toggleRunPanel();
}

function toggleRunPanel() {
  document.getElementById('run-panel').classList.toggle('hidden');
}

function hideRunPanel() {
  document.getElementById('run-panel').classList.add('hidden');
}

function setupDropZone() {
  const dz  = document.getElementById('run-drop-zone');
  const inp = document.getElementById('run-file-input');
  if (!dz || !inp) return;

  dz.addEventListener('click', () => inp.click());
  inp.addEventListener('change', () => handleFile(inp.files[0]));

  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragging'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragging'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('dragging');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

function handleFile(file) {
  if (!file) return;
  state.uploadedFile = file;
  const nameEl = document.getElementById('run-file-name');
  nameEl.textContent = `📎 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  nameEl.classList.remove('hidden');
}

async function executePipeline() {
  if (state.uploading) return;
  if (!state.uploadedFile) { toast('Please select a file first.', 'error'); return; }

  state.uploading = true;
  const btn = document.getElementById('run-btn');
  const logEl = document.getElementById('run-log');
  const resEl = document.getElementById('run-result');
  btn.disabled = true; btn.textContent = '⏳ Running…';
  logEl.classList.remove('hidden'); resEl.classList.add('hidden');
  logEl.innerHTML = '';

  function log(msg) { logEl.innerHTML += `<div>[${new Date().toLocaleTimeString()}] ${escHtml(msg)}</div>`; logEl.scrollTop = 9999; }

  try {
    log(`Uploading ${state.uploadedFile.name}…`);
    const fd = new FormData();
    fd.append('file', state.uploadedFile);
    const upRes = await fetch(`${API}/ingest/upload`, { method: 'POST', body: fd });
    if (!upRes.ok) throw new Error(`Upload failed: ${upRes.statusText}`);
    const upData = await upRes.json();
    const snapshotId = upData.snapshot_id || upData.id;
    log(`Uploaded ✓  Snapshot: ${snapshotId}`);

    log('Running full pipeline…');
    const target = document.getElementById('run-target').value.trim();
    const runBody = { snapshot_id: snapshotId };
    if (target) runBody.target_col = target;

    const runRes = await fetch(`${API}/pipeline/run`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(runBody),
    });
    const runData = await runRes.json();

    if (!runRes.ok) {
      log(`Pipeline error: ${runData.detail || runRes.statusText}`);
      showResult(resEl, false, `Pipeline failed: ${runData.detail || runRes.statusText}`);
    } else {
      const dec = runData.gate_decision || 'UNKNOWN';
      log(`Gate 1: ${runData.gate1_decision || '?'}`);
      log(`Gate 2: ${runData.gate2_decision || '?'}`);
      log(`Confidence: ${(runData.confidence_score || 0).toFixed(3)}`);
      log(`Decision: ${dec}`);
      log('Pipeline complete.');
      showResult(resEl, dec === 'PASS', `Decision: ${dec} · Run ID: ${(runData.run_id || '').substring(0, 8)}…`);
      setTimeout(refreshAll, 1500);
    }
  } catch (e) {
    log(`Error: ${e.message}`);
    showResult(resEl, false, e.message);
  } finally {
    state.uploading = false;
    btn.disabled = false; btn.textContent = '▶ Run Full Pipeline';
  }
}

function showResult(el, ok, msg) {
  el.className = `run-result ${ok ? 'pass' : 'fail'}`;
  el.textContent = (ok ? '✓ ' : '✗ ') + msg;
  el.classList.remove('hidden');
}

/* ═══ Reports Tab ═════════════════════════════════════════════ */
async function loadReports() {
  try {
    const r = await fetch(`${API}/api/export/list`);
    if (!r.ok) throw new Error();
    const d = await r.json();
    state.reports = d.available_reports || [];
  } catch {
    state.reports = [];
  }
  renderReports();
}

function renderReports() {
  const list = document.getElementById('report-list');
  const empty = document.getElementById('report-empty');

  // Update stats
  document.getElementById('rstat-total').textContent = state.reports.length || '0';

  if (!state.reports.length) {
    empty.style.display = 'flex';
    return;
  }

  empty.style.display = 'none';
  const sizeTotal = state.reports.reduce((s, r) => s + (r.size_bytes || 0), 0);
  document.getElementById('rstat-size').textContent = formatBytes(sizeTotal);
  document.getElementById('rstat-latest').textContent = state.reports[0]?.filename?.substring(0, 12) + '…' || '—';

  list.innerHTML = state.reports.map((rep, idx) => `
    <div class="report-item">
      <div class="report-file-icon">📄</div>
      <div class="report-meta">
        <div class="report-name">${escHtml(rep.filename)}</div>
        <div class="report-info">${formatBytes(rep.size_bytes)} · Executive HTML Report</div>
      </div>
      <div class="report-actions">
        <button class="btn-view" onclick="viewReport(${idx})">👁 View</button>
        <a class="btn-dl" href="${API}${rep.download_url}" download="${escHtml(rep.filename)}">⤓ Download</a>
      </div>
    </div>
  `).join('');
}

function viewReport(idx) {
  const rep = state.reports[idx];
  if (!rep) return;
  const viewer = document.getElementById('report-viewer');
  const iframe  = document.getElementById('report-iframe');
  const titleEl = document.getElementById('viewer-title');
  const dlBtn   = document.getElementById('viewer-dl-btn');

  iframe.src = `${API}${rep.download_url}`;
  titleEl.textContent = rep.filename;
  dlBtn.href = `${API}${rep.download_url}`;
  dlBtn.download = rep.filename;
  viewer.classList.remove('hidden');
  viewer.scrollIntoView({ behavior: 'smooth' });
}

function closeViewer() {
  const viewer = document.getElementById('report-viewer');
  viewer.classList.add('hidden');
  document.getElementById('report-iframe').src = 'about:blank';
}

/* ═══ Utilities ═══════════════════════════════════════════════ */
function formatBytes(b) {
  if (!b) return '0 B';
  if (b < 1024) return b + ' B';
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
  return (b / (1024 * 1024)).toFixed(2) + ' MB';
}

/* ═══ Toast ═══════════════════════════════════════════════════ */
function toast(msg, type = 'info') {
  const ctr = document.getElementById('toast-ctr');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  t.onclick = () => t.remove();
  ctr.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}
