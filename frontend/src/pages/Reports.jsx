import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area,
    PieChart as RechartsPieChart, Pie, Cell, ScatterChart, Scatter, ZAxis
} from 'recharts';
import {
    Search, Loader2, AlertTriangle, FileCheck, Info,
    CheckCircle2, XCircle, AlertCircle, Clock, Database,
    Upload, Radio, Globe, FileText, RefreshCw, Download,
    ChevronRight, Shield, Zap, BarChart2, TrendingUp,
    Copy, Check, ChevronDown, ChevronUp, PieChart
} from 'lucide-react';
import { ResultsService, getCachedData } from '../api/client';
import './Reports.css';

const API_BASE = import.meta.env.VITE_API_URL || '';

// ── Helpers ───────────────────────────────────────────────────────────────────

const fmtNum = (n, dec = 2) => {
    if (n == null || isNaN(n)) return '—';
    return Number(n).toFixed(dec);
};

const fmtPct = (n) => n != null ? `${(n * 100).toFixed(1)}%` : '—';

const relTime = (ts) => {
    if (!ts) return '—';
    try {
        const diff = Date.now() - new Date(ts).getTime();
        const m = Math.floor(diff / 60000);
        if (m < 1) return 'just now';
        if (m < 60) return `${m}m ago`;
        const h = Math.floor(m / 60);
        if (h < 24) return `${h}h ago`;
        return `${Math.floor(h / 24)}d ago`;
    } catch { return ts; }
};

const SOURCE_MAP = {
    file: { Icon: Upload, label: 'File Upload', cls: 'src-file' },
    database: { Icon: Database, label: 'Database', cls: 'src-db' },
    live: { Icon: Radio, label: 'Kafka Stream', cls: 'src-kafka' },
    api: { Icon: Globe, label: 'REST API', cls: 'src-api' },
};

const SourceBadge = ({ kind }) => {
    const s = SOURCE_MAP[kind] || SOURCE_MAP.file;
    const Icon = s.Icon;
    return <span className={`rpt-src-badge ${s.cls}`}><Icon size={12} /> {s.label}</span>;
};

const GateBadge = ({ decision }) => {
    const d = (decision || 'UNKNOWN').toUpperCase();
    const cls = d === 'PASS' ? 'pass' : d === 'FAIL' ? 'fail' : 'unknown';
    return <span className={`gate-badge ${cls}`}>{d}</span>;
};

// ── 1. Run History Sidebar ────────────────────────────────────────────────────

const RunSidebar = ({ runs, activeId, onSelect }) => (
    <aside className="rpt-sidebar">
        <div className="sidebar-header">
            <Clock size={14} />
            <span>Run History</span>
            <span className="sidebar-count">{runs.length}</span>
        </div>
        <div className="sidebar-list">
            {runs.length === 0 && (
                <div className="sidebar-empty">No runs yet.</div>
            )}
            {[...runs].reverse().map(r => (
                <button
                    key={r.run_id}
                    className={`sidebar-item ${r.run_id === activeId ? 'active' : ''}`}
                    onClick={() => onSelect(r.run_id)}
                >
                    <div className="si-top">
                        <code className="si-runid">{r.run_id?.slice(0, 10)}…</code>
                        <GateBadge decision={r.gate_decision} />
                    </div>
                    <div className="si-meta">
                        <span>{r.dataset_id || 'Unknown'}</span>
                        <span>{relTime(r.timestamp)}</span>
                    </div>
                    {r.confidence_score != null && (
                        <div className="si-conf-bar">
                            <div className="si-conf-fill" style={{ width: `${r.confidence_score * 100}%` }} />
                        </div>
                    )}
                </button>
            ))}
        </div>
    </aside>
);

// ── 2. Report Header ──────────────────────────────────────────────────────────

const ReportHeader = ({ d }) => {
    const [copied, setCopied] = useState(false);
    const copy = () => {
        navigator.clipboard.writeText(d.run_id || '');
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
    };
    return (
        <div className="rpt-header">
            <div className="rpt-header-top">
                <div>
                    <h1 className="rpt-dataset-title">{d.dataset_id || 'Pipeline Run'}</h1>
                    <div className="rpt-header-meta">
                        <button className="runid-copy" onClick={copy} title="Copy Run ID">
                            <code>{d.run_id}</code>
                            {copied ? <Check size={11} /> : <Copy size={11} />}
                        </button>
                        &nbsp;·&nbsp;
                        {d.timestamp && <span>{new Date(d.timestamp).toLocaleString()}</span>}
                        {d.source_kind && <><span className="sep">·</span><SourceBadge kind={d.source_kind} /></>}
                        {d.target_col && <><span className="sep">·</span><span className="target-col-badge">🎯 {d.target_col}</span></>}
                    </div>
                </div>
            </div>
            <div className="rpt-kpi-chips">
                <div className={`kpi-chip ${(d.gate_decision || '').toLowerCase()}`}>
                    <span className="chip-label">Gate Decision</span>
                    <span className="chip-val">{d.gate_decision || '—'}</span>
                </div>
                <div className="kpi-chip neutral">
                    <span className="chip-label">Confidence</span>
                    <span className="chip-val">{fmtPct(d.confidence_score)}</span>
                </div>
                <div className="kpi-chip neutral">
                    <span className="chip-label">Quality Score</span>
                    <span className="chip-val">{fmtPct(d.quality_score)}</span>
                </div>
                <div className="kpi-chip neutral">
                    <span className="chip-label">Rows</span>
                    <span className="chip-val">{(d.row_count || 0).toLocaleString()}</span>
                </div>
                <div className="kpi-chip neutral">
                    <span className="chip-label">Columns</span>
                    <span className="chip-val">{d.col_count || '—'}</span>
                </div>
                {d.retry_count > 0 && (
                    <div className="kpi-chip warn">
                        <span className="chip-label">Retries</span>
                        <span className="chip-val">{d.retry_count}</span>
                    </div>
                )}
            </div>
        </div>
    );
};

// ── 3. Pipeline Stage Timeline ────────────────────────────────────────────────

const StageTimeline = ({ stages }) => {
    if (!stages?.length) return null;
    const icon = (s) => {
        const st = (s.status || '').toUpperCase();
        if (st === 'PASS') return <CheckCircle2 size={18} className="stage-icon pass" />;
        if (st === 'FAIL') return <XCircle size={18} className="stage-icon fail" />;
        if (st === 'WARN') return <AlertCircle size={18} className="stage-icon warn" />;
        return <div className="stage-dot unknown" />;
    };
    return (
        <div className="rpt-section">
            <div className="rpt-section-label"><Zap size={14} /> Pipeline Stage Timeline</div>
            <div className="timeline-wrap">
                {stages.map((s, i) => (
                    <React.Fragment key={i}>
                        <div className={`timeline-step ${(s.status || 'unknown').toLowerCase()}`}>
                            {icon(s)}
                            <div className="timeline-name">{s.name}</div>
                            {s.duration_ms != null && (
                                <div className="timeline-dur">{s.duration_ms}ms</div>
                            )}
                            {s.reason && (
                                <div className="timeline-reason">{s.reason}</div>
                            )}
                        </div>
                        {i < stages.length - 1 && <div className="timeline-connector" />}
                    </React.Fragment>
                ))}
            </div>
        </div>
    );
};

// ── 4. Confidence Gauge (SVG arc) ──────────────────────────────────────────────

const ConfidenceGauge = ({ value }) => {
    const pct = Math.min(1, Math.max(0, value || 0));
    const r = 70, cx = 90, cy = 90;
    const start = Math.PI, end = 2 * Math.PI;
    const totalArc = end - start;
    const arcEnd = start + totalArc * pct;

    const polar = (ang) => ({
        x: cx + r * Math.cos(ang),
        y: cy + r * Math.sin(ang),
    });

    const p1 = polar(start), p2 = polar(arcEnd);
    const largeArc = pct > 0.5 ? 1 : 0;
    const trackPath = `M ${polar(start).x} ${polar(start).y} A ${r} ${r} 0 1 1 ${polar(end - 0.001).x} ${polar(end - 0.001).y}`;
    const fillPath = pct > 0.001 ? `M ${p1.x} ${p1.y} A ${r} ${r} 0 ${largeArc} 1 ${p2.x} ${p2.y}` : '';

    const colour = pct >= 0.75 ? '#34d399' : pct >= 0.5 ? '#fbbf24' : '#f87171';

    return (
        <div className="gauge-wrap">
            <svg width="180" height="100" viewBox="0 0 180 100">
                <path d={trackPath} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="12" strokeLinecap="round" />
                {fillPath && (
                    <path d={fillPath} fill="none" stroke={colour} strokeWidth="12" strokeLinecap="round"
                        style={{ filter: `drop-shadow(0 0 6px ${colour}88)` }} />
                )}
                <text x={cx} y={cy - 8} textAnchor="middle" fill={colour} fontSize="22" fontWeight="800">
                    {Math.round(pct * 100)}%
                </text>
                <text x={cx} y={cy + 10} textAnchor="middle" fill="#8b949e" fontSize="10">
                    confidence
                </text>
            </svg>
        </div>
    );
};

// ── 5. Gate Decision Cards ────────────────────────────────────────────────────

const GateCards = ({ gate1, gate2, dims }) => {
    const card = (label, decision, description) => {
        const d = (decision || 'UNKNOWN').toUpperCase();
        return (
            <div className={`gate-card ${d.toLowerCase()}`}>
                <div className="gate-card-top">
                    {d === 'PASS' ? <CheckCircle2 size={20} className="gc-icon" /> : <XCircle size={20} className="gc-icon" />}
                    <h3>{label}</h3>
                    <GateBadge decision={d} />
                </div>
                <p className="gate-card-desc">{description}</p>
            </div>
        );
    };
    return (
        <div className="rpt-section">
            <div className="rpt-section-label"><Shield size={14} /> Gate Decisions</div>
            <div className="gate-cards-row">
                {card(
                    'Gate 1 — Data Quality',
                    gate1,
                    gate1 === 'PASS'
                        ? 'Raw data passed quality thresholds — null rates, type consistency, and cardinality checks all within acceptable bounds.'
                        : 'Raw data failed quality checks. Possible causes: high null rates, type errors, or low-cardinality columns. Review the heatmap below.',
                )}
                {card(
                    'Gate 2 — Statistical QA',
                    gate2,
                    gate2 === 'PASS'
                        ? 'Processed data passed statistical validation — feature distributions stable and correlation structure is sound for modelling.'
                        : 'Statistical QA failed. Feature distributions may be unstable or degenerate. The model metrics below reflect this structural weakness.',
                )}
            </div>
        </div>
    );
};

// ── 6. Model Metrics Table ────────────────────────────────────────────────────

const MODEL_THRESHOLDS = {
    accuracy: 0.7, f1: 0.6, auc: 0.65, auc_roc: 0.65,
    precision: 0.6, recall: 0.6, r2: 0.5,
};

const MetricsTable = ({ metrics }) => {
    const [sortCol, setSortCol] = useState(null);
    const [sortAsc, setSortAsc] = useState(false);
    if (!metrics || !Object.keys(metrics).length) return null;

    const rows = Object.entries(metrics)
        .filter(([, v]) => v != null && typeof v !== 'object')
        .map(([k, v]) => ({ key: k, val: typeof v === 'number' ? v : parseFloat(v) }));

    if (!rows.length) return null;

    const sorted = sortCol
        ? [...rows].sort((a, b) => (sortAsc ? 1 : -1) * ((a.val ?? 0) - (b.val ?? 0)))
        : rows;

    const colour = (k, v) => {
        const threshold = MODEL_THRESHOLDS[k.toLowerCase()];
        if (threshold == null || isNaN(v)) return '';
        return v >= threshold ? 'metric-good' : 'metric-bad';
    };

    return (
        <div className="rpt-section">
            <div className="rpt-section-label"><BarChart2 size={14} /> Model Metrics</div>
            <p className="rpt-section-hint">
                These metrics reflect how well the AutoML model fit the processed data. Green cells meet the quality threshold; red cells fall below it.
                A low metric alongside a passing Gate 2 suggests the model's confidence is justified by data structure, not just fit.
            </p>
            <div className="metrics-table-wrap">
                <table className="metrics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th onClick={() => { setSortCol('val'); setSortAsc(s => !s); }} className="sortable">
                                Value {sortAsc ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                            </th>
                            <th>Threshold</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sorted.map(({ key, val }) => {
                            const thr = MODEL_THRESHOLDS[key.toLowerCase()];
                            const cls = colour(key, val);
                            const pct = !isNaN(val) && val <= 1;
                            return (
                                <tr key={key} className={cls}>
                                    <td className="metric-key">{key.replace(/_/g, ' ')}</td>
                                    <td className="metric-val">{pct ? fmtPct(val) : fmtNum(val, 4)}</td>
                                    <td className="metric-thr">{thr ? (pct ? fmtPct(thr) : fmtNum(thr, 2)) : '—'}</td>
                                    <td>{thr != null ? (val >= thr ? <span className="ms-pass">✓ Pass</span> : <span className="ms-fail">✗ Below</span>) : <span className="ms-na">—</span>}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

// ── 7. Risk Flags ─────────────────────────────────────────────────────────────

const SEV_ORDER = { HIGH: 0, MEDIUM: 1, LOW: 2 };
const SEV_ICON = { HIGH: '🔴', MEDIUM: '🟡', LOW: '🟢' };

const RiskFlags = ({ flags }) => {
    if (!flags?.length) return null;
    const sorted = [...flags].sort((a, b) =>
        (SEV_ORDER[a.severity] ?? 1) - (SEV_ORDER[b.severity] ?? 1)
    );
    return (
        <div className="rpt-section">
            <div className="rpt-section-label"><AlertTriangle size={14} /> Risk Flags</div>
            <p className="rpt-section-hint">
                Risk flags are graduated warnings that don't necessarily cause a gate failure but highlight areas the data team should investigate.
                Unlike gate decisions (binary), flags give a prioritised action list.
            </p>
            <div className="risk-flags-list">
                {sorted.map((f, i) => (
                    <div key={i} className={`risk-flag sev-${(f.severity || 'medium').toLowerCase()}`}>
                        <span className="rf-icon">{SEV_ICON[f.severity] || '🟡'}</span>
                        <div className="rf-body">
                            {f.column && <span className="rf-col">{f.column}</span>}
                            <span className="rf-desc">{f.description}</span>
                        </div>
                        <span className={`rf-sev sev-${(f.severity || '').toLowerCase()}`}>{f.severity}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

// ── 8. Data Quality Heatmap ───────────────────────────────────────────────────

const QualityHeatmap = ({ rows, colMeta }) => {
    const MAX_ROWS = 30, MAX_COLS = 8;
    if (!rows?.length) return null;

    // Pick top columns (prefer those with metadata, then first N)
    const allCols = Object.keys(rows[0] || {});
    const cols = allCols.slice(0, MAX_COLS);
    const displayRows = rows.slice(0, MAX_ROWS);

    // Compute per-column mean/std for outlier detection
    const colStats = {};
    cols.forEach(c => {
        const vals = rows.map(r => parseFloat(r[c])).filter(v => !isNaN(v));
        if (!vals.length) return;
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
        colStats[c] = { mean, std };
    });

    // Null %  and outlier % per col
    const colNullPct = c => rows.filter(r => r[c] == null || r[c] === '').length / rows.length;
    const colOutPct = c => {
        if (!colStats[c]) return 0;
        const { mean, std } = colStats[c];
        return std === 0 ? 0 : rows.filter(r => Math.abs(parseFloat(r[c]) - mean) > 2 * std).length / rows.length;
    };

    const cellClass = (col, val) => {
        if (val == null || val === '') return 'hm-null';
        const s = colStats[col];
        if (s && s.std > 0 && Math.abs(parseFloat(val) - s.mean) > 2 * s.std) return 'hm-outlier';
        return 'hm-clean';
    };

    return (
        <div className="rpt-section">
            <div className="rpt-section-label"><TrendingUp size={14} /> Data Quality Heatmap</div>
            <p className="rpt-section-hint">
                Each cell is colour-coded by its quality status: <span className="legend-clean">■ clean</span>{' '}
                <span className="legend-outlier">■ outlier (&gt;2σ)</span>{' '}
                <span className="legend-null">■ null/missing</span>.
                Column headers show null% and outlier% to quickly spot problematic columns.
            </p>
            <div className="heatmap-wrap">
                <table className="heatmap-table">
                    <thead>
                        <tr>
                            <th className="hm-row-num">#</th>
                            {cols.map(c => (
                                <th key={c}>
                                    <div className="hm-col-label">{c}</div>
                                    <div className="hm-col-badges">
                                        <span className="hm-null-pct">{(colNullPct(c) * 100).toFixed(0)}% null</span>
                                        {colOutPct(c) > 0 && <span className="hm-out-pct">{(colOutPct(c) * 100).toFixed(0)}% out</span>}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {displayRows.map((row, ri) => (
                            <tr key={ri}>
                                <td className="hm-row-num">{ri + 1}</td>
                                {cols.map(c => {
                                    const v = row[c];
                                    const cls = cellClass(c, v);
                                    return (
                                        <td key={c} className={`hm-cell ${cls}`} title={String(v ?? 'null')}>
                                            {v != null && v !== '' ? String(v).substring(0, 12) : '∅'}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="heatmap-note">
                Showing {displayRows.length} of {rows.length} rows · {cols.length} of {allCols.length} columns
            </div>
        </div>
    );
};

// ── 8.5 Smart Visual Analysis ───────────────────────────────────────────────────────

const SmartVisuals = ({ rows, colMeta, targetCol }) => {
    if (!rows || rows.length === 0) return null;

    // Smart logic to identify column types and generate chart data
    const visuals = useMemo(() => {
        const allCols = Object.keys(rows[0] || {});
        const numericCols = [];
        const catCols = [];

        allCols.forEach(col => {
            // ignore target column for general grouping
            if (col === targetCol) return;

            let isNumeric = true;
            let uniqueVals = new Set();
            for (let i = 0; i < Math.min(rows.length, 100); i++) {
                const val = rows[i][col];
                if (val != null && val !== '') {
                    uniqueVals.add(val);
                    if (isNaN(Number(val))) {
                        isNumeric = false;
                    }
                }
            }

            if (isNumeric && uniqueVals.size > 5) {
                numericCols.push(col);
            } else {
                // If it's a string, or low cardinality numeric
                if (uniqueVals.size > 0 && uniqueVals.size <= 20) {
                    catCols.push(col);
                }
            }
        });

        const charts = [];

        // 1. Generate Class Distribution for Target Column if it exists and is categorical
        if (targetCol) {
            const counts = {};
            rows.forEach(r => {
                const val = String(r[targetCol] || 'Missing');
                counts[val] = (counts[val] || 0) + 1;
            });
            if (Object.keys(counts).length > 0 && Object.keys(counts).length <= 15) {
                const isVeryLowCard = Object.keys(counts).length <= 4;
                charts.push({
                    id: 'target-dist',
                    title: `Target Distribution: ${targetCol}`,
                    type: isVeryLowCard ? 'pie' : 'bar',
                    data: Object.entries(counts).map(([k, v]) => ({ name: k, count: v })).sort((a, b) => b.count - a.count),
                    dataKey: 'count',
                    color: '#8b5cf6'
                });
            }
        }

        // 2. Generate Bar/Pie charts for top categorical columns
        catCols.slice(0, 3).forEach(col => {
            const counts = {};
            rows.forEach(r => {
                const val = (r[col] == null || r[col] === '') ? 'Missing' : String(r[col]).substring(0, 20);
                counts[val] = (counts[val] || 0) + 1;
            });
            const numKeys = Object.keys(counts).length;
            if (numKeys > 0) {
                charts.push({
                    id: `cat-${col}`,
                    title: `Category Breakdown: ${col}`,
                    type: numKeys <= 5 ? 'pie' : 'bar',
                    data: Object.entries(counts).map(([k, v]) => ({ name: k, count: v })).sort((a, b) => b.count - a.count).slice(0, 10),
                    dataKey: 'count',
                    color: '#3b82f6'
                });
            }
        });

        // 3. Generate Area/Line charts for numeric distribution approximation
        numericCols.slice(0, 3).forEach(col => {
            // create 10 bins
            let min = Infinity, max = -Infinity;
            const vals = [];
            rows.forEach(r => {
                const v = Number(r[col]);
                if (!isNaN(v) && r[col] != null && r[col] !== '') {
                    if (v < min) min = v;
                    if (v > max) max = v;
                    vals.push(v);
                }
            });

            if (min < Infinity && max > -Infinity && min !== max) {
                const binCount = 10;
                const range = max - min;
                const binSize = range / binCount;
                const bins = Array(binCount).fill(0);

                vals.forEach(v => {
                    let binIdx = Math.floor((v - min) / binSize);
                    if (binIdx >= binCount) binIdx = binCount - 1;
                    bins[binIdx]++;
                });

                const data = bins.map((count, i) => {
                    const binStart = min + i * binSize;
                    const binEnd = min + (i + 1) * binSize;
                    return {
                        name: `${binStart.toFixed(1)}-${binEnd.toFixed(1)}`,
                        frequency: count
                    };
                });

                charts.push({
                    id: `num-${col}`,
                    title: `Distribution: ${col}`,
                    type: 'area',
                    data: data,
                    dataKey: 'frequency',
                    color: '#10b981'
                });
            }
        });

        // 4. Generate a Scatter plot to show correlation if we have at least 2 numeric features
        if (numericCols.length >= 2) {
            const colX = numericCols[0];
            const colY = numericCols[1];

            const scatterData = rows.map(r => ({
                x: Number(r[colX]),
                y: Number(r[colY])
            })).filter(r => !isNaN(r.x) && !isNaN(r.y));

            if (scatterData.length > 0) {
                charts.push({
                    id: `scatter-${colX}-${colY}`,
                    title: `Correlation: ${colX} vs ${colY}`,
                    type: 'scatter',
                    data: scatterData,
                    dataKey: 'y',
                    colX: colX,
                    colY: colY,
                    color: '#f43f5e'
                });
            }
        }

        // 5. Data Completeness (Missing Values)
        const missingCounts = {};
        allCols.forEach(col => {
            let missing = 0;
            rows.forEach(r => {
                if (r[col] == null || r[col] === '') missing++;
            });
            if (missing > 0) {
                missingCounts[col] = missing;
            }
        });

        if (Object.keys(missingCounts).length > 0) {
            charts.push({
                id: 'missing-values',
                title: 'Missing Values count (top 10)',
                type: 'bar',
                data: Object.entries(missingCounts)
                    .map(([k, v]) => ({ name: k, count: v }))
                    .sort((a, b) => b.count - a.count).slice(0, 10),
                dataKey: 'count',
                color: '#ef4444' // red for missing data
            });
        }

        // 6. Outlier / Anomaly Detection (Z-Score > 3)
        const outlierCounts = {};
        numericCols.forEach(col => {
            const vals = [];
            let sum = 0;
            rows.forEach(r => {
                const v = Number(r[col]);
                if (!isNaN(v) && r[col] != null && r[col] !== '') {
                    vals.push(v);
                    sum += v;
                }
            });

            if (vals.length > 2) {
                const mean = sum / vals.length;
                let varianceSum = 0;
                vals.forEach(v => {
                    varianceSum += Math.pow(v - mean, 2);
                });
                const stdDev = Math.sqrt(varianceSum / vals.length);

                if (stdDev > 0) {
                    let outliers = 0;
                    vals.forEach(v => {
                        const zScore = Math.abs((v - mean) / stdDev);
                        // Using a Z-Score of 3 as standard anomaly threshold
                        if (zScore > 3) outliers++;
                    });

                    if (outliers > 0) {
                        outlierCounts[col] = outliers;
                    }
                }
            }
        });

        if (Object.keys(outlierCounts).length > 0) {
            charts.push({
                id: 'statistical-anomalies',
                title: 'Data Anomalies (>3 Std Dev)',
                type: 'bar',
                data: Object.entries(outlierCounts)
                    .map(([k, v]) => ({ name: k, count: v }))
                    .sort((a, b) => b.count - a.count).slice(0, 10),
                dataKey: 'count',
                color: '#f59e0b' // Warning Orange
            });
        }

        return charts;
    }, [rows, targetCol]);

    if (!visuals || visuals.length === 0) return null;

    const PIE_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

    return (
        <div className="rpt-section">
            <div className="rpt-section-label"><PieChart size={14} /> Smart Visual Analysis</div>
            <p className="rpt-section-hint">
                These charts are automatically generated by analyzing the dataset's cardinality and variance on the fly. Categorical distributions and continuous feature spread are crucial for identifying data skew before modeling.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginTop: '16px' }}>
                {visuals.map(chart => (
                    <div key={chart.id} className="chart-card" style={{ background: '#161b22', border: '1px solid #21262d', borderRadius: '10px', padding: '16px' }}>
                        <h4 style={{ color: '#e6edf3', margin: '0 0 16px 0', fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{chart.title}</h4>
                        <div style={{ width: '100%', height: '220px' }}>
                            <ResponsiveContainer>
                                {chart.type === 'bar' ? (
                                    <BarChart data={chart.data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
                                        <XAxis dataKey="name" tick={{ fill: '#8b949e', fontSize: 10 }} axisLine={false} tickLine={false} />
                                        <YAxis tick={{ fill: '#8b949e', fontSize: 10 }} axisLine={false} tickLine={false} />
                                        <Tooltip
                                            contentStyle={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: '8px', color: '#e6edf3', fontSize: '12px' }}
                                            itemStyle={{ color: chart.color }}
                                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                        />
                                        <Bar dataKey={chart.dataKey} fill={chart.color} radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                ) : chart.type === 'pie' ? (
                                    <RechartsPieChart margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                                        <Pie
                                            data={chart.data}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={60}
                                            outerRadius={80}
                                            paddingAngle={5}
                                            dataKey={chart.dataKey}
                                            stroke="none"
                                        >
                                            {chart.data.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip
                                            contentStyle={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: '8px', color: '#e6edf3', fontSize: '12px' }}
                                            itemStyle={{ color: '#fff' }}
                                        />
                                    </RechartsPieChart>
                                ) : chart.type === 'scatter' ? (
                                    <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                                        <XAxis type="number" dataKey="x" name={chart.colX} tick={{ fill: '#8b949e', fontSize: 10 }} axisLine={false} tickLine={false} />
                                        <YAxis type="number" dataKey="y" name={chart.colY} tick={{ fill: '#8b949e', fontSize: 10 }} axisLine={false} tickLine={false} />
                                        <ZAxis type="number" range={[20, 20]} />
                                        <Tooltip
                                            cursor={{ strokeDasharray: '3 3' }}
                                            contentStyle={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: '8px', color: '#e6edf3', fontSize: '12px' }}
                                        />
                                        <Scatter data={chart.data} fill={chart.color} opacity={0.6} />
                                    </ScatterChart>
                                ) : (
                                    <AreaChart data={chart.data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                        <defs>
                                            <linearGradient id={`grad-${chart.id}`} x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={chart.color} stopOpacity={0.8} />
                                                <stop offset="95%" stopColor={chart.color} stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
                                        <XAxis dataKey="name" tick={{ fill: '#8b949e', fontSize: 10 }} axisLine={false} tickLine={false} />
                                        <YAxis tick={{ fill: '#8b949e', fontSize: 10 }} axisLine={false} tickLine={false} />
                                        <Tooltip
                                            contentStyle={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: '8px', color: '#e6edf3', fontSize: '12px' }}
                                            cursor={{ stroke: 'rgba(255,255,255,0.1)' }}
                                        />
                                        <Area type="monotone" dataKey={chart.dataKey} stroke={chart.color} strokeWidth={2} fillOpacity={1} fill={`url(#grad-${chart.id})`} />
                                    </AreaChart>
                                )}
                            </ResponsiveContainer>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// ── 9. Narrative ──────────────────────────────────────────────────────────────

const NarrativeSection = ({ narrative, runId, onRegenerate, generating }) => (
    <div className="rpt-section">
        <div className="rpt-section-label"><FileCheck size={14} /> Executive Narrative</div>
        <p className="rpt-section-hint">
            The narrative is the interpretation layer — it translates all the charts and numbers above into plain English
            that a non-technical stakeholder can read, act on, and share. Every visual above is data; this is the story.
        </p>
        <div className="narrative-card">
            {narrative ? (
                <>
                    <h3 className="narrative-title">{narrative.title}</h3>
                    <p className="narrative-body">{narrative.body}</p>
                </>
            ) : (
                <p className="no-narrative">No narrative generated for this run.</p>
            )}
        </div>
        <button className="regenerate-btn" onClick={onRegenerate} disabled={generating}>
            {generating ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />}
            {generating ? 'Generating…' : 'Regenerate Narrative'}
        </button>
    </div>
);

// ── Export Bar ────────────────────────────────────────────────────────────────

const ExportBar = ({ hasReport, reportUrl, runId, onGenerate, generating }) => (
    <div className="export-bar" style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
            <div className="export-bar-left">
                <span className="export-label">📄 Report</span>
                {hasReport
                    ? <span className="export-ready">HTML report ready</span>
                    : <span className="export-none">No HTML report generated yet</span>}
            </div>
            <div className="export-bar-right">
                {!hasReport && (
                    <button className="export-btn generate" onClick={onGenerate} disabled={generating}>
                        {generating ? <Loader2 size={14} className="spin" /> : <Zap size={14} />}
                        {generating ? 'Generating…' : 'Generate Full Report'}
                    </button>
                )}
                {hasReport && reportUrl && (
                    <a href={`${API_BASE}${reportUrl}`} target="_blank" rel="noreferrer" className="export-btn download">
                        <Download size={14} /> Download HTML
                    </a>
                )}
            </div>
        </div>
    </div>
);

// ── Main Reports Page ─────────────────────────────────────────────────────────

const Reports = () => {
    const cachedAll = getCachedData('/api/results');
    const cachedLatest = getCachedData('/api/results/latest');

    const [runList, setRunList] = useState(cachedAll?.runs || []);
    const [activeId, setActiveId] = useState(cachedLatest?.run_id || null);
    const [reportData, setReportData] = useState(cachedLatest || null);
    const [loading, setLoading] = useState(!cachedLatest);
    const [error, setError] = useState(null);
    const [generating, setGenerating] = useState(false);
    const [searchId, setSearchId] = useState('');

    // ── Load run list ────────────────────────────────────────────────────────
    useEffect(() => {
        ResultsService.getAllResults()
            .then(d => setRunList(d.runs || []))
            .catch(() => { });
    }, []);

    // ── Load a specific run ──────────────────────────────────────────────────
    const loadRun = useCallback(async (id) => {
        setLoading(true); setError(null);
        try {
            const data = await ResultsService.getResult(id);
            setReportData(data);
            setActiveId(id);
            setError(null);
        } catch {
            setError('Could not load run. Check the run ID.');
        } finally {
            setLoading(false);
        }
    }, []);

    // ── Load latest on mount ─────────────────────────────────────────────────
    useEffect(() => {
        (async () => {
            if (!loading && !reportData) setLoading(true);
            setError(null);
            try {
                const data = await ResultsService.getLatestResult();
                if (!data || !data.run_id) {
                    setError('No pipeline runs found. Run a pipeline to generate a report.');
                } else {
                    setReportData(data);
                    setActiveId(data.run_id);
                }
            } catch {
                setError('No pipeline runs found. Run a pipeline to generate a report.');
            } finally {
                setLoading(false);
            }
        })();
    }, []);

    const handleSearch = (e) => {
        e.preventDefault();
        if (searchId.trim()) loadRun(searchId.trim());
    };

    const handleRegenerate = async () => {
        if (!activeId) return;
        setGenerating(true);
        try {
            await fetch(`${API_BASE}/report/executive`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ run_id: activeId }),
            });
            await loadRun(activeId);
        } catch { /* silent */ }
        finally { setGenerating(false); }
    };

    // ── Radar data ───────────────────────────────────────────────────────────
    const radarData = useMemo(() => {
        if (!reportData?.dimensions) return [];
        const d = reportData.dimensions;
        return [
            { subject: 'Data Quality', A: Math.round((d.data_quality || 0) * 100) },
            { subject: 'Statistical Strength', A: Math.round((d.statistical_strength || 0) * 100) },
            { subject: 'Stability', A: Math.round((d.stability || 0) * 100) },
            { subject: 'Compliance', A: Math.round((d.compliance || 0) * 100) },
        ];
    }, [reportData?.dimensions]);

    return (
        <div className="rpt-layout">
            {/* Sidebar */}
            <RunSidebar runs={runList} activeId={activeId} onSelect={loadRun} />

            {/* Main */}
            <div className="rpt-main">
                {/* Search bar */}
                <form onSubmit={handleSearch} className="rpt-search-bar">
                    <Search size={16} className="rpt-search-icon" />
                    <input
                        value={searchId}
                        onChange={e => setSearchId(e.target.value)}
                        placeholder="Jump to a Run ID…"
                        className="rpt-search-input"
                    />
                    <button type="submit" className="rpt-search-btn" disabled={loading}>
                        {loading ? <Loader2 size={14} className="spin" /> : 'Lookup'}
                    </button>
                </form>

                {error && (
                    <div className="rpt-error">
                        <AlertTriangle size={16} /> {error}
                    </div>
                )}

                {loading && !error && (
                    <div className="rpt-loading">
                        <Loader2 size={32} className="spin" />
                        <p>Loading pipeline telemetry…</p>
                    </div>
                )}

                {!loading && reportData && (
                    <>
                        {/* ── 2. Header ── */}
                        <ReportHeader d={reportData} />

                        {/* ── 3. Stage Timeline ── */}
                        <StageTimeline stages={reportData.stages} />

                        {/* ── 4. Gauge + Radar ── */}
                        <div className="rpt-section">
                            <div className="rpt-section-label"><Info size={14} /> Confidence &amp; Dimension Analysis</div>
                            <p className="rpt-section-hint">
                                The gauge shows the <strong>single confidence score</strong> — the model's overall certainty in the pipeline's output.
                                The radar breaks it down into four dimensions so you can see <em>why</em> the score is what it is:
                                are the inputs clean? are the statistical patterns strong? was the run stable? did compliance gate pass?
                            </p>
                            <div className="gauge-radar-row">
                                <div className="gauge-panel">
                                    <ConfidenceGauge value={reportData.confidence_score} />
                                    <div className="gauge-sub">
                                        {reportData.confidence_score >= 0.75
                                            ? '✅ High confidence — results are reliable.'
                                            : reportData.confidence_score >= 0.5
                                                ? '⚠️ Moderate confidence — review risk flags.'
                                                : '❌ Low confidence — results need validation.'}
                                    </div>
                                </div>
                                <div className="radar-panel">
                                    <ResponsiveContainer width="100%" height={260}>
                                        <RadarChart data={radarData}>
                                            <PolarGrid stroke="rgba(255,255,255,0.08)" />
                                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#8b949e', fontSize: 11 }} />
                                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#4a5568', fontSize: 9 }} tickCount={4} />
                                            <Radar name="Run" dataKey="A" stroke="#6366f1" fill="#6366f1" fillOpacity={0.25} strokeWidth={2} />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>

                        {/* ── 5. Gate Cards ── */}
                        <GateCards gate1={reportData.gate1_decision} gate2={reportData.gate2_decision} />

                        {/* ── 6. Model Metrics ── */}
                        <MetricsTable metrics={reportData.model_metrics} />

                        {/* ── 7. Risk Flags ── */}
                        <RiskFlags flags={reportData.risk_flags} />

                        {/* ── 8. Data Quality Heatmap ── */}
                        <QualityHeatmap rows={reportData.sample_rows} colMeta={reportData.column_metadata} />

                        {/* ── Smart Visuals ── */}
                        <SmartVisuals rows={reportData.sample_rows} colMeta={reportData.column_metadata} targetCol={reportData.target_col} />

                        {/* ── 9. Narrative ── */}
                        <NarrativeSection
                            narrative={reportData.narrative}
                            runId={activeId}
                            onRegenerate={handleRegenerate}
                            generating={generating}
                        />

                        {/* ── 10. Export Bar ── */}
                        <ExportBar
                            hasReport={reportData.has_report}
                            reportUrl={reportData.report_url}
                            runId={activeId}
                            onGenerate={handleRegenerate}
                            generating={generating}
                        />
                    </>
                )}
            </div>
        </div>
    );
};

export default Reports;
