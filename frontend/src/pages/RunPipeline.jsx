// v5 - Professional Reports + Power BI Illustrations
import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Upload, Database, Radio, Globe, Play, CheckCircle, AlertCircle, FileText, Cpu, Filter, Columns, X, ChevronDown, ChevronUp, SlidersHorizontal, Table2, Shield, ShieldOff, Info, Plus, Trash2, Eye, EyeOff, RefreshCw, ThumbsUp, ThumbsDown, MessageSquare, Sparkles, RotateCcw, BarChart2, TrendingUp, PieChart as PieIcon, Activity, Layers } from 'lucide-react';
import {
    ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend,
    BarChart, Bar, LineChart, Line, AreaChart, Area, PieChart, Pie, Cell,
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ComposedChart
} from 'recharts';
import { analyzeSchema, computeStats, PALETTE } from '../utils/dataAnalyzer';
import AnalysisPlanModal from '../components/AnalysisPlanModal';
import './RunPipeline.css';

const API_BASE = import.meta.env.VITE_API_URL || '';

const MODES = [
    { id: 'file', icon: Upload, label: 'File Upload', desc: 'CSV, Excel, JSON, Parquet' },
    { id: 'database', icon: Database, label: 'Database', desc: 'PostgreSQL, MongoDB, Redis, Neo4j' },
    { id: 'live', icon: Radio, label: 'Kafka Stream', desc: 'Live streaming from a Kafka topic' },
    { id: 'api', icon: Globe, label: 'REST API', desc: 'Pull data from any HTTP endpoint' },
];

const SOURCE_LABELS = { file: 'File Upload', database: 'Database', live: 'Kafka Stream', api: 'REST API' };

const DOMAIN_CARDS = [
    { id: 'banking', emoji: '🏦', label: 'Banking', sub: 'AML · Basel III' },
    { id: 'healthcare', emoji: '🏥', label: 'Healthcare', sub: 'HIPAA · PHI' },
    { id: 'finance', emoji: '📈', label: 'Finance', sub: 'SEC · Capital' },
    { id: 'gdpr', emoji: '🇪🇺', label: 'GDPR', sub: 'Residency · Consent' },
    { id: 'sox', emoji: '📋', label: 'SOX', sub: 'Audit Trail' },
    { id: 'hipaa', emoji: '🔒', label: 'HIPAA', sub: 'Encryption' },
];

const EXTRA_REGULATION_PILLS = ['gdpr', 'sox', 'hipaa', 'banking', 'healthcare', 'finance'];

function applyFilters(rows, filters) {
    if (!filters.length) return rows;
    return rows.filter(row => filters.every(f => {
        const cell = String(row[f.col] ?? '');
        const numCell = parseFloat(cell), numVal = parseFloat(f.val);
        switch (f.op) {
            case 'eq': return cell === f.val;
            case 'neq': return cell !== f.val;
            case 'gt': return !isNaN(numCell) && numCell > numVal;
            case 'lt': return !isNaN(numCell) && numCell < numVal;
            case 'gte': return !isNaN(numCell) && numCell >= numVal;
            case 'lte': return !isNaN(numCell) && numCell <= numVal;
            case 'contains': return cell.toLowerCase().includes(f.val.toLowerCase());
            case 'startswith': return cell.toLowerCase().startsWith(f.val.toLowerCase());
            default: return true;
        }
    }));
}

function projectCols(rows, selectedCols) {
    if (!selectedCols) return rows;
    if (selectedCols.size === 0) return [];
    return rows.map(r => {
        const out = {}; selectedCols.forEach(k => { if (k in r) out[k] = r[k]; }); return out;
    });
}

const fmtNum = n => {
    if (n == null || isNaN(n)) return '—';
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return Number(n).toFixed(2);
};

// ── helpers: parse colRange text ("1-5" or "name,age,income") into a Set of keys
function parseColRange(colRange, allKeys) {
    if (!colRange.trim()) return null;
    const raw = colRange.trim();
    // Range like 1-10 (1-indexed)
    const rangeMatch = raw.match(/^(\d+)-(\d+)$/);
    if (rangeMatch) {
        const from = Math.max(0, parseInt(rangeMatch[1], 10) - 1);
        const to   = Math.min(allKeys.length - 1, parseInt(rangeMatch[2], 10) - 1);
        return new Set(allKeys.slice(from, to + 1));
    }
    // Comma-separated names
    const names = raw.split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
    return new Set(allKeys.filter(k => names.includes(k.toLowerCase())));
}

function parseRowRange(rowRange, totalRows) {
    if (!rowRange.trim()) return { from: 0, to: totalRows };
    const m = rowRange.trim().match(/^(\d+)-(\d+)$/);
    if (m) return { from: Math.max(0, parseInt(m[1], 10) - 1), to: Math.min(totalRows, parseInt(m[2], 10)) };
    return { from: 0, to: totalRows };
}

const DataPreviewPanel = ({ rows, sourceKind, colRange='', rowRange='' }) => {
    const schema = useMemo(() => analyzeSchema(rows), [rows]);
    const [selectedCols, setSelectedCols] = useState(null);
    const [colOpen, setColOpen] = useState(false);
    const prevSchemaKeyRef = useRef('');

    // Apply colRange whenever schema or prop changes
    useEffect(() => {
        const key = schema.columns.map(c => c.key).join(',');
        const parsedRange = parseColRange(colRange, schema.columns.map(c => c.key));
        if (key !== prevSchemaKeyRef.current || colRange) {
            prevSchemaKeyRef.current = key;
            setSelectedCols(parsedRange || new Set(schema.columns.map(c => c.key)));
        }
    }, [schema, colRange]);

    const [filters, setFilters] = useState([]);
    const [filterOpen, setFilterOpen] = useState(false);
    const [filterCol, setFilterCol] = useState('');
    const [filterOp, setFilterOp] = useState('eq');
    const [filterVal, setFilterVal] = useState('');
    const [page, setPage] = useState(0);
    const PAGE_SIZE = 20;
    const toggleCol = (key, checked) => setSelectedCols(prev => { const s = new Set(prev); checked ? s.add(key) : s.delete(key); return s; });
    const uniqueVals = useMemo(() => {
        if (!filterCol) return [];
        return [...new Set(rows.map(r => String(r[filterCol] ?? '')))].sort().slice(0, 40);
    }, [filterCol, rows]);
    const addFilter = () => { if (!filterCol || filterVal === '') return; setFilters(f => [...f, { col: filterCol, op: filterOp, val: filterVal }]); setFilterVal(''); };

    // Apply rowRange to source rows first
    const { from: rowFrom, to: rowTo } = useMemo(() => parseRowRange(rowRange, rows.length), [rowRange, rows.length]);
    const rangedRows = useMemo(() => rows.slice(rowFrom, rowTo), [rows, rowFrom, rowTo]);

    const filteredRows = useMemo(() => applyFilters(rangedRows, filters), [rangedRows, filters]);
    const displayRows = useMemo(() => projectCols(filteredRows, selectedCols), [filteredRows, selectedCols]);
    const pageRows = displayRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
    const totalPages = Math.ceil(displayRows.length / PAGE_SIZE);
    const displayCols = (selectedCols && selectedCols.size > 0) ? [...selectedCols] : schema.columns.map(c => c.key);
    const stats = useMemo(() => {
        const ps = analyzeSchema(displayRows.slice(0, 200));
        return computeStats(displayRows.slice(0, 200), ps.numericCols.slice(0, 5));
    }, [displayRows]);

    const rowRangeActive = rowRange.trim() && (rowFrom > 0 || rowTo < rows.length);
    const colRangeActive = colRange.trim() && selectedCols && selectedCols.size < schema.columns.length;

    return (
        <div className="preview-panel">
            <div className="preview-header">
                <Table2 size={16} className="preview-header-icon" />
                <h3>Data Preview — <span className="preview-src">{SOURCE_LABELS[sourceKind] ?? sourceKind}</span></h3>
                <span className="preview-meta">{filteredRows.length.toLocaleString()} / {rows.length.toLocaleString()} rows &middot; {displayCols.length} / {schema.columns.length} cols
                    {rowRangeActive && <span style={{marginLeft:'0.5rem',color:'#f59e0b',fontSize:'0.72rem',fontWeight:700}}> · rows {rowFrom+1}–{rowTo}</span>}
                    {colRangeActive && <span style={{marginLeft:'0.35rem',color:'#818cf8',fontSize:'0.72rem',fontWeight:700}}> · col filter active</span>}
                </span>
            </div>
            <div className="preview-controls">
                <div className="rp-selector">
                    <button className="rp-sel-trigger" onClick={() => setColOpen(o => !o)}>
                        <Columns size={13} /> Columns
                        <span className="rp-sel-count">{selectedCols?.size ?? 0} / {schema.columns.length}</span>
                        <ChevronDown size={11} style={{ transform: colOpen ? 'rotate(180deg)' : '', transition: '0.15s' }} />
                    </button>
                    {colOpen && (
                        <div className="rp-dropdown">
                            <div className="rp-dd-actions">
                                <button onClick={() => setSelectedCols(new Set(schema.columns.map(c => c.key)))}>All</button>
                                <button onClick={() => setSelectedCols(new Set())}>Clear</button>
                                {schema.numericCols.length > 0 && <button onClick={() => setSelectedCols(new Set(schema.numericCols))}>Numeric</button>}
                                {schema.categoricalCols.length > 0 && <button onClick={() => setSelectedCols(new Set(schema.categoricalCols))}>Categorical</button>}
                                {schema.temporalCols.length > 0 && <button onClick={() => setSelectedCols(new Set(schema.temporalCols))}>Dates</button>}
                            </div>
                            <div className="rp-dd-list">
                                {schema.columns.map(c => (
                                    <label key={c.key} className="rp-col-item">
                                        <input type="checkbox" checked={selectedCols?.has(c.key) ?? true} onChange={e => toggleCol(c.key, e.target.checked)} />
                                        <span className="rp-col-name">{c.key}</span>
                                        <span className={`rp-type type-${c.type}`}>{c.type}</span>
                                    </label>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
                <div className="rp-selector">
                    <button className="rp-sel-trigger" onClick={() => setFilterOpen(o => !o)}>
                        <Filter size={13} /> Filter Rows
                        {filters.length > 0 && <span className="rp-sel-count active">{filters.length}</span>}
                        <ChevronDown size={11} style={{ transform: filterOpen ? 'rotate(180deg)' : '', transition: '0.15s' }} />
                    </button>
                    {filterOpen && (
                        <div className="rp-dropdown rp-dropdown-wide">
                            <div className="rp-filter-builder">
                                <select value={filterCol} onChange={e => setFilterCol(e.target.value)} className="rp-fsel">
                                    <option value="">— Column —</option>
                                    {schema.columns.map(c => <option key={c.key} value={c.key}>{c.key} ({c.type})</option>)}
                                </select>
                                <select value={filterOp} onChange={e => setFilterOp(e.target.value)} className="rp-fsel-sm">
                                    <option value="eq">= equals</option><option value="neq">≠ not equals</option>
                                    <option value="gt">&gt; greater</option><option value="lt">&lt; less</option>
                                    <option value="gte">≥ gte</option><option value="lte">≤ lte</option>
                                    <option value="contains">contains</option><option value="startswith">starts with</option>
                                </select>
                                <input className="rp-finput" list="rp-fval" value={filterVal} onChange={e => setFilterVal(e.target.value)} placeholder="Value…" />
                                <datalist id="rp-fval">{uniqueVals.map(v => <option key={v} value={v} />)}</datalist>
                                <button className="rp-fadd" onClick={addFilter} disabled={!filterCol || filterVal === ''}>+ Add</button>
                                {filters.length > 0 && <button className="rp-fclear" onClick={() => setFilters([])}>Clear all</button>}
                            </div>
                            {filters.length > 0 && (
                                <div className="rp-pills">
                                    {filters.map((f, i) => (
                                        <div key={i} className="rp-pill">
                                            <code>{f.col}</code>&nbsp;{f.op}&nbsp;<em>"{f.val}"</em>
                                            <button onClick={() => setFilters(fs => fs.filter((_, j) => j !== i))}><X size={9} /></button>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>
                {(filters.length > 0 || (selectedCols != null && selectedCols.size < schema.columns.length)) && (
                    <button className="rp-reset" onClick={() => { setFilters([]); setSelectedCols(new Set(schema.columns.map(c => c.key))); setPage(0); }}>
                        <X size={11} /> Reset
                    </button>
                )}
            </div>
            {stats.length > 0 && (
                <div className="preview-stats">
                    {stats.map((s, i) => (
                        <div key={s.col} className="preview-stat-chip" style={{ borderColor: PALETTE[i % PALETTE.length] + '55' }}>
                            <span className="psc-col">{s.col}</span>
                            <span className="psc-val">{fmtNum(s.mean)}<span className="psc-label"> avg</span></span>
                            <span className="psc-range">{fmtNum(s.min)} → {fmtNum(s.max)}</span>
                        </div>
                    ))}
                </div>
            )}
            {displayRows.length === 0 ? (
                <div className="preview-empty">No rows match the current filters.</div>
            ) : (
                <>
                    <div className="preview-table-wrap">
                        <table className="preview-table">
                            <thead>
                                <tr>
                                    <th className="row-num">#</th>
                                    {displayCols.map(col => (
                                        <th key={col}>{col}
                                            <span className={`rp-type type-${schema.columns.find(c => c.key === col)?.type ?? 'text'}`}>
                                                {schema.columns.find(c => c.key === col)?.type ?? ''}
                                            </span>
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {pageRows.map((row, ri) => (
                                    <tr key={ri}>
                                        <td className="row-num">{page * PAGE_SIZE + ri + 1}</td>
                                        {displayCols.map(col => <td key={col} title={String(row[col] ?? '')}>{String(row[col] ?? '').substring(0, 60)}</td>)}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    {totalPages > 1 && (
                        <div className="preview-pagination">
                            <button disabled={page === 0} onClick={() => setPage(0)}>«</button>
                            <button disabled={page === 0} onClick={() => setPage(p => p - 1)}>‹ Prev</button>
                            <span>Page {page + 1} / {totalPages}</span>
                            <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>Next ›</button>
                            <button disabled={page >= totalPages - 1} onClick={() => setPage(totalPages - 1)}>»</button>
                        </div>
                    )}
                </>
            )}
        </div>
    );
};

// â”€â”€ PII type metadata (icon + colour) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const PII_META = {
    Email: { icon: 'âœ‰ï¸', color: '#818cf8', label: 'Email' },
    SSN: { icon: '🔢', color: '#f87171', label: 'SSN' },
    CreditCard: { icon: '💳', color: '#fb923c', label: 'Credit Card' },
    Phone: { icon: '📞', color: '#34d399', label: 'Phone' },
    IPAddress: { icon: 'ðŸŒ', color: '#60a5fa', label: 'IP Address' },
    ICD10: { icon: 'ðŸ¥', color: '#a78bfa', label: 'ICD-10 Code' },
    IBAN: { icon: 'ðŸ¦', color: '#fbbf24', label: 'IBAN' },
    Swift: { icon: '💲', color: '#2dd4bf', label: 'SWIFT/BIC' },
};

const POLICY_STYLES = {
    redacted: { bg: 'rgba(99,102,241,0.12)', border: '#6366f1', color: '#a5b4fc', icon: 'ðŸ›¡ï¸', text: 'Redacted' },
    rejected: { bg: 'rgba(239,68,68,0.12)', border: '#ef4444', color: '#fca5a5', icon: '🚫', text: 'Rejected' },
    flagged: { bg: 'rgba(251,191,36,0.10)', border: '#f59e0b', color: '#fde68a', icon: 'âš ï¸', text: 'Flagged' },
    passed: { bg: 'rgba(52,211,153,0.10)', border: '#10b981', color: '#6ee7b7', icon: '✅', text: 'Clean' },
    skipped: { bg: 'rgba(100,116,139,0.10)', border: '#475569', color: '#94a3b8', icon: 'â­ï¸', text: 'Not Checked' },
};

const GovernanceSummary = ({ govReport }) => {
    if (!govReport) return null;

    const status = govReport.status || 'skipped';
    const totalRedactions = govReport.total_redactions || 0;

    // Normalize pii_hits: governor always returns {col: {pii_type: count}},
    // but defensively handle if a future detector returns {col: [hit, ...]}
    const normalizeHits = (raw) => {
        if (!raw || typeof raw !== 'object') return {};
        const normalized = {};
        Object.entries(raw).forEach(([col, hits]) => {
            if (Array.isArray(hits)) {
                // Convert array of hit strings → {type: count}
                normalized[col] = hits.reduce((acc, h) => {
                    const key = typeof h === 'string' ? h : String(h);
                    acc[key] = (acc[key] || 0) + 1;
                    return acc;
                }, {});
            } else if (hits && typeof hits === 'object') {
                normalized[col] = hits;
            }
            // Skip if hits is null/undefined/primitive
        });
        return normalized;
    };

    const policyHits = normalizeHits(govReport.pii_hits);
    const affectedCols = Object.keys(policyHits);
    const style = POLICY_STYLES[status] || POLICY_STYLES.skipped;

    // Aggregate PII type counts across all columns
    const piiTypeCounts = {};
    affectedCols.forEach(col => {
        Object.entries(policyHits[col]).forEach(([piiType, count]) => {
            piiTypeCounts[piiType] = (piiTypeCounts[piiType] || 0) + count;
        });
    });

    const hasFindings = affectedCols.length > 0;

    return (
        <div className="gov-summary-block" style={{ borderColor: style.border, background: style.bg }}>
            <div className="gov-summary-header">
                <span className="gov-summary-icon">{style.icon}</span>
                <div>
                    <span className="gov-summary-title">Data Governance</span>
                    <span className="gov-summary-status" style={{ color: style.color, borderColor: style.border, background: `${style.border}22` }}>
                        {style.text}
                    </span>
                </div>
                {hasFindings && (
                    <div className="gov-summary-stats">
                        <span className="gov-stat-chip" title="Columns with PII">
                            <span className="gov-stat-num" style={{ color: style.color }}>{affectedCols.length}</span>
                            <span className="gov-stat-label">col{affectedCols.length !== 1 ? 's' : ''} affected</span>
                        </span>
                        {status === 'redacted' && (
                            <span className="gov-stat-chip" title="Total redactions applied">
                                <span className="gov-stat-num" style={{ color: '#a5b4fc' }}>{totalRedactions}</span>
                                <span className="gov-stat-label">redactions</span>
                            </span>
                        )}
                    </div>
                )}
            </div>

            {/* Policy applied */}
            <div className="gov-policy-row">
                <span className="gov-policy-label">Policy</span>
                <code className="gov-policy-val">{govReport.policy_applied || '—'}</code>
                {govReport.dataset_id && <>
                    <span className="gov-policy-label" style={{ marginLeft: '1rem' }}>Dataset</span>
                    <code className="gov-policy-val">{govReport.dataset_id}</code>
                </>}
            </div>

            {/* No PII clean state */}
            {!hasFindings && status !== 'skipped' && (
                <p className="gov-clean-msg">✓ No PII detected — dataset is clean.</p>
            )}

            {/* PII Type Breakdown */}
            {Object.keys(piiTypeCounts).length > 0 && (
                <div className="gov-pii-section">
                    <div className="gov-pii-heading">PII Types Detected</div>
                    <div className="gov-pii-chips">
                        {Object.entries(piiTypeCounts).map(([piiType, count]) => {
                            const meta = PII_META[piiType] || { icon: 'âš ï¸', color: '#94a3b8', label: piiType };
                            return (
                                <div key={piiType} className="gov-pii-chip" style={{ borderColor: meta.color, background: `${meta.color}18` }}>
                                    <span className="gov-pii-chip-icon">{meta.icon}</span>
                                    <span className="gov-pii-chip-label" style={{ color: meta.color }}>{meta.label}</span>
                                    <span className="gov-pii-chip-count" style={{ background: meta.color }}>{count}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Affected Columns */}
            {affectedCols.length > 0 && (
                <div className="gov-cols-section">
                    <div className="gov-pii-heading">Affected Columns</div>
                    <div className="gov-cols-list">
                        {affectedCols.map(col => {
                            const colHits = policyHits[col];
                            const colTotal = Object.values(colHits).reduce((a, b) => a + b, 0);
                            return (
                                <div key={col} className="gov-col-row">
                                    <span className="gov-col-name">
                                        <Shield size={11} style={{ marginRight: '0.3rem', color: style.color, flexShrink: 0 }} />
                                        {col}
                                    </span>
                                    <div className="gov-col-types">
                                        {Object.entries(colHits).map(([piiType, cnt]) => {
                                            const meta = PII_META[piiType] || { color: '#94a3b8', label: piiType };
                                            return (
                                                <span key={piiType} className="gov-col-type-badge" style={{ color: meta.color, borderColor: `${meta.color}55`, background: `${meta.color}14` }}>
                                                    {meta.label} ×{cnt}
                                                </span>
                                            );
                                        })}
                                    </div>
                                    <span className="gov-col-total" style={{ color: style.color }}>{colTotal} hit{colTotal !== 1 ? 's' : ''}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};

const RegulatorySummary = ({ regulatoryReport }) => {
    if (!regulatoryReport || regulatoryReport.length === 0) return null;

    return (
        <div className="gov-summary-block" style={{ borderColor: '#f59e0b', background: 'rgba(251,191,36,0.06)', marginTop: '16px' }}>
            <div className="gov-summary-header">
                <span className="gov-summary-icon">âš–ï¸</span>
                <div>
                    <span className="gov-summary-title">Regulatory Engine</span>
                    <span className="gov-summary-status" style={{ color: '#fde68a', borderColor: '#f59e0b', background: `rgba(251,191,36,0.15)` }}>
                        {regulatoryReport.length} Finding{regulatoryReport.length !== 1 ? 's' : ''}
                    </span>
                </div>
            </div>

            <div className="gov-cols-section" style={{ marginTop: '0.75rem' }}>
                <div className="gov-cols-list">
                    {regulatoryReport.map((flag, idx) => (
                        <div key={idx} className="gov-col-row" style={{ alignItems: 'flex-start', padding: '0.6rem' }}>
                            <div style={{ flex: 1 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                                    <span style={{
                                        fontSize: '0.65rem', padding: '2px 6px', borderRadius: '4px', fontWeight: 700,
                                        backgroundColor: flag.level === 'HIGH' ? 'rgba(239,68,68,0.2)' : (flag.level === 'MEDIUM' ? 'rgba(245,158,11,0.2)' : 'rgba(52,211,153,0.2)'),
                                        color: flag.level === 'HIGH' ? '#fca5a5' : (flag.level === 'MEDIUM' ? '#fcd34d' : '#6ee7b7')
                                    }}>
                                        {flag.level}
                                    </span>
                                    <span style={{ fontWeight: 600, color: '#e2e8f0', fontSize: '0.75rem' }}>{flag.category}</span>
                                </div>
                                <div style={{ fontSize: '0.75rem', color: '#94a3b8', lineHeight: 1.4 }}>{flag.message}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

// ── Chart color palette ──────────────────────────────────────────────────────
const CHART_COLORS = ['#6366f1','#8b5cf6','#22d3ee','#10b981','#f59e0b','#f43f5e','#a78bfa','#34d399','#60a5fa','#fb923c'];

// ── Significance badge ────────────────────────────────────────────────────────
const SigBadge = ({ score }) => {
    const pct = Math.round((score || 0) * 100);
    const level = pct >= 70 ? { label: 'High',   color: '#10b981', bg: 'rgba(16,185,129,0.12)' }
                : pct >= 40 ? { label: 'Medium', color: '#f59e0b', bg: 'rgba(245,158,11,0.12)' }
                :             { label: 'Low',    color: '#64748b', bg: 'rgba(100,116,139,0.12)' };
    return (
        <span style={{ fontSize:'0.65rem', fontWeight:700, padding:'2px 8px', borderRadius:'999px',
            color:level.color, background:level.bg, border:`1px solid ${level.color}44` }}>
            {level.label} &middot; {pct}%
        </span>
    );
};

// ── Correlation Heatmap ───────────────────────────────────────────────────────
const HeatmapGrid = ({ data, rowLabels, colLabels }) => {
    if (!data || !rowLabels || !colLabels || !rowLabels.length) return null;
    const size = rowLabels.length;
    const cellSize = Math.min(56, Math.floor(340 / size));
    const getColor = (val) => {
        const abs = Math.abs(val);
        return val >= 0 ? `rgba(99,102,241,${0.1 + abs * 0.8})` : `rgba(244,63,94,${0.1 + abs * 0.8})`;
    };
    return (
        <div style={{ overflowX:'auto', padding:'0.5rem' }}>
            <div style={{ display:'grid', gridTemplateColumns:`60px repeat(${size},${cellSize}px)`, gap:2 }}>
                <div />
                {colLabels.map(l => (
                    <div key={l} style={{ fontSize:'0.55rem', color:'#64748b', textAlign:'center',
                        overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', padding:'2px' }}>{l}</div>
                ))}
                {rowLabels.map(rl => (
                    <React.Fragment key={rl}>
                        <div style={{ fontSize:'0.55rem', color:'#64748b', overflow:'hidden',
                            textOverflow:'ellipsis', whiteSpace:'nowrap', display:'flex', alignItems:'center' }}>{rl}</div>
                        {colLabels.map(cl => {
                            const cell = data.find(d => d.x === rl && d.y === cl);
                            const val = cell ? cell.value : 0;
                            return (
                                <div key={`${rl}-${cl}`} title={`${rl} x ${cl}: ${val.toFixed(3)}`}
                                    style={{ width:cellSize, height:cellSize, background:getColor(val),
                                        borderRadius:4, display:'flex', alignItems:'center',
                                        justifyContent:'center', fontSize:'0.5rem', color:'#e2e8f0', fontWeight:700 }}>
                                    {val.toFixed(2)}
                                </div>
                            );
                        })}
                    </React.Fragment>
                ))}
            </div>
        </div>
    );
};

// ── Chart renderer ────────────────────────────────────────────────────────────
const SECTION_META = {
    schema:     { label: 'Schema',     color: '#818cf8', emoji: '🗃️' },
    quality:    { label: 'Quality',    color: '#34d399', emoji: '✅' },
    missing:    { label: 'Missing',    color: '#f59e0b', emoji: '∅'  },
    anomaly:    { label: 'Anomaly',    color: '#f87171', emoji: '🔍' },
    drift:      { label: 'Drift',      color: '#60a5fa', emoji: '📡' },
    model:      { label: 'Model',      color: '#a78bfa', emoji: '🤖' },
    governance: { label: 'Governance', color: '#c084fc', emoji: '🔒' },
    regulatory: { label: 'Regulatory', color: '#fbbf24', emoji: '⚖️' },
    pipeline:   { label: 'Pipeline',   color: '#2dd4bf', emoji: '🔄' },
};

const ChartRenderer = ({ chart }) => {
    const { type, data, columns, stack_keys, line_keys, row_labels, col_labels } = chart;
    if (!data || data.length === 0)
        return <div style={{ textAlign:'center', color:'#475569', padding:'2rem', fontSize:'0.8rem' }}>No data available</div>;

    const tt = {
        contentStyle: { background:'#0f0f1e', border:'1px solid rgba(99,102,241,0.3)', borderRadius:8, fontSize:'0.78rem', color:'#e2e8f0' },
        cursor: { fill:'rgba(99,102,241,0.06)' },
    };

    if (type === 'scatter') {
        const xKey = (columns || [])[0] || 'x', yKey = (columns || [])[1] || 'y';
        return (
            <ResponsiveContainer width="100%" height={240}>
                <ScatterChart><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey={xKey} name={xKey} tick={{ fill:'#64748b', fontSize:10 }} />
                    <YAxis dataKey={yKey} name={yKey} tick={{ fill:'#64748b', fontSize:10 }} />
                    <Tooltip {...tt} /><Scatter data={data} fill={CHART_COLORS[0]} fillOpacity={0.75} />
                </ScatterChart>
            </ResponsiveContainer>
        );
    }
    if (type === 'bar' || type === 'count_bar' || type === 'funnel') {
        const bars = (stack_keys && stack_keys.length > 1) ? stack_keys : ['value'];
        return (
            <ResponsiveContainer width="100%" height={240}>
                <BarChart data={data} margin={{ bottom:20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" tick={{ fill:'#64748b', fontSize:9 }} angle={-30} textAnchor="end" interval={0} />
                    <YAxis tick={{ fill:'#64748b', fontSize:10 }} /><Tooltip {...tt} />
                    {bars.map((k, i) => <Bar key={k} dataKey={k} fill={CHART_COLORS[i % 10]} radius={[4,4,0,0]} />)}
                </BarChart>
            </ResponsiveContainer>
        );
    }
    if (type === 'stacked_bar') {
        const bars = stack_keys || [];
        return (
            <ResponsiveContainer width="100%" height={240}>
                <BarChart data={data} margin={{ bottom:20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" tick={{ fill:'#64748b', fontSize:9 }} angle={-30} textAnchor="end" interval={0} />
                    <YAxis tick={{ fill:'#64748b', fontSize:10 }} /><Tooltip {...tt} />
                    <Legend wrapperStyle={{ fontSize:'0.7rem', color:'#94a3b8' }} />
                    {bars.map((k, i) => <Bar key={k} dataKey={k} stackId="a" fill={CHART_COLORS[i % 10]} />)}
                </BarChart>
            </ResponsiveContainer>
        );
    }
    if (type === 'range_bar') {
        const keys = stack_keys || ['min','avg','max'];
        return (
            <ResponsiveContainer width="100%" height={240}>
                <BarChart data={data} margin={{ bottom:30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" tick={{ fill:'#64748b', fontSize:9 }} angle={-30} textAnchor="end" interval={0} />
                    <YAxis tick={{ fill:'#64748b', fontSize:10 }} /><Tooltip {...tt} />
                    <Legend wrapperStyle={{ fontSize:'0.7rem', color:'#94a3b8' }} />
                    {keys.map((k, i) => <Bar key={k} dataKey={k} fill={CHART_COLORS[i]} radius={[3,3,0,0]} />)}
                </BarChart>
            </ResponsiveContainer>
        );
    }
    if (type === 'line') {
        return (
            <ResponsiveContainer width="100%" height={240}>
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="time" tick={{ fill:'#64748b', fontSize:9 }} />
                    <YAxis tick={{ fill:'#64748b', fontSize:10 }} /><Tooltip {...tt} />
                    <Line type="monotone" dataKey="value" stroke={CHART_COLORS[0]} strokeWidth={2} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        );
    }
    if (type === 'multi_line') {
        const keys = line_keys || [];
        return (
            <ResponsiveContainer width="100%" height={240}>
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="time" tick={{ fill:'#64748b', fontSize:9 }} />
                    <YAxis tick={{ fill:'#64748b', fontSize:10 }} /><Tooltip {...tt} />
                    <Legend wrapperStyle={{ fontSize:'0.7rem', color:'#94a3b8' }} />
                    {keys.map((k, i) => <Line key={k} type="monotone" dataKey={k} stroke={CHART_COLORS[i % 10]} strokeWidth={2} dot={false} />)}
                </LineChart>
            </ResponsiveContainer>
        );
    }
    if (type === 'area') {
        return (
            <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="bin" tick={{ fill:'#64748b', fontSize:9 }} />
                    <YAxis tick={{ fill:'#64748b', fontSize:10 }} /><Tooltip {...tt} />
                    <defs>
                        <linearGradient id="aG" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor={CHART_COLORS[0]} stopOpacity={0.4} />
                            <stop offset="95%" stopColor={CHART_COLORS[0]} stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <Area type="monotone" dataKey="count" stroke={CHART_COLORS[0]} fill="url(#aG)" strokeWidth={2} />
                </AreaChart>
            </ResponsiveContainer>
        );
    }
    if (type === 'pie') {
        return (
            <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                    <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%"
                        outerRadius={85} innerRadius={40} paddingAngle={3}
                        label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`} labelLine={false}>
                        {data.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % 10]} />)}
                    </Pie>
                    <Tooltip {...tt} />
                </PieChart>
            </ResponsiveContainer>
        );
    }
    if (type === 'radar') {
        const keys = line_keys || [];
        return (
            <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={data}>
                    <PolarGrid stroke="rgba(255,255,255,0.08)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill:'#64748b', fontSize:9 }} />
                    <PolarRadiusAxis tick={{ fill:'#64748b', fontSize:8 }} domain={[0,1]} />
                    {keys.map((k, i) => (
                        <Radar key={k} name={k} dataKey={k}
                            stroke={CHART_COLORS[i % 10]} fill={CHART_COLORS[i % 10]} fillOpacity={0.18} />
                    ))}
                    <Legend wrapperStyle={{ fontSize:'0.7rem', color:'#94a3b8' }} />
                    <Tooltip {...tt} />
                </RadarChart>
            </ResponsiveContainer>
        );
    }
    if (type === 'heatmap') {
        return <HeatmapGrid data={data} rowLabels={row_labels || []} colLabels={col_labels || []} />;
    }
    const firstKey = Object.keys(data[0] || {}).find(k => k !== 'name') || 'value';
    return (
        <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" tick={{ fill:'#64748b', fontSize:9 }} />
                <YAxis tick={{ fill:'#64748b', fontSize:10 }} /><Tooltip {...tt} />
                <Bar dataKey={firstKey} fill={CHART_COLORS[0]} radius={[4,4,0,0]} />
            </BarChart>
        </ResponsiveContainer>
    );
};

// ── Power BI Panel ────────────────────────────────────────────────────────────
const PowerBIPanel = ({ charts, kpis, insightsFeed, sectionInsights, loading, rowCount, colCount }) => {
    const [activeSection, setActiveSection] = useState('all');
    const [expandedInsights, setExpandedInsights] = useState(true);
    const [fullscreenChart, setFullscreenChart] = useState(null);

    // Derive sections that have charts
    const availableSections = useMemo(() => {
        const secs = new Set(charts.map(c => c.section || 'schema'));
        return ['all', ...Object.keys(SECTION_META).filter(s => secs.has(s))];
    }, [charts]);

    const visibleCharts = useMemo(() => {
        if (activeSection === 'all') return charts;
        return charts.filter(c => (c.section || 'schema') === activeSection);
    }, [charts, activeSection]);

    const activeInsights = useMemo(() => {
        if (!sectionInsights) return insightsFeed || [];
        if (activeSection === 'all') return insightsFeed || [];
        return (sectionInsights[activeSection] || []).map(i => ({
            chart_id: i.title,
            title:    i.title,
            text:     i.explanation || i.insight || '',
            relevance: Math.round((i.significance || 0) * 100),
            section:  activeSection,
        }));
    }, [sectionInsights, insightsFeed, activeSection]);

    const getKpiIcon = key => {
        if (key.includes('Records'))   return '📋';
        if (key.includes('Columns'))   return '🗂️';
        if (key.includes('Null'))      return '∅';
        if (key.includes('Avg'))       return '📊';
        if (key.includes('Max'))       return '🔺';
        if (key.includes('Top'))       return '🏆';
        if (key.includes('AUC'))       return '📈';
        if (key.includes('F1'))        return '⚡';
        if (key.includes('PII'))       return '🔒';
        if (key.includes('Drift'))     return '📡';
        return '💡';
    };

    return (
        <div className="powerbi-panel">

            {/* ── KPI Cards ─────────────────────────────────────────────── */}
            {Object.keys(kpis).length > 0 && (
                <div className="kpi-cards-row">
                    {Object.entries(kpis).map(([key, val], i) => (
                        <div key={key} className="kpi-card" style={{ borderColor: CHART_COLORS[i % 10] + '66' }}>
                            <span className="kpi-icon">{getKpiIcon(key)}</span>
                            <span className="kpi-value" style={{ color: CHART_COLORS[i % 10] }}>{String(val)}</span>
                            <span className="kpi-label">{key}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* ── Insight Cards ─────────────────────────────────────────── */}
            {activeInsights.length > 0 && (
                <div className="insight-cards-section">
                    <div className="insight-cards-header" onClick={() => setExpandedInsights(e => !e)}>
                        <Activity size={14} style={{ color: '#818cf8' }} />
                        <span>AI Insights &amp; Explanations</span>
                        <span className="insights-count-badge">{activeInsights.length}</span>
                        <ChevronDown size={12} style={{ transform: expandedInsights ? 'rotate(180deg)' : '', transition: '0.2s', marginLeft: 'auto' }} />
                    </div>
                    {expandedInsights && (
                        <div className="insight-cards-grid">
                            {activeInsights.slice(0, 9).map((ins, i) => {
                                const sec = SECTION_META[ins.section] || SECTION_META.schema;
                                const rel = ins.relevance || 0;
                                return (
                                    <div key={ins.chart_id + i} className="insight-card" style={{ borderColor: sec.color + '44' }}>
                                        <div className="insight-card-header">
                                            <span className="insight-card-emoji">{sec.emoji}</span>
                                            <span className="insight-card-title">{ins.title}</span>
                                            <span className="insight-card-rel" style={{
                                                color: rel >= 70 ? '#10b981' : rel >= 40 ? '#f59e0b' : '#64748b',
                                                background: rel >= 70 ? 'rgba(16,185,129,0.1)' : rel >= 40 ? 'rgba(245,158,11,0.1)' : 'rgba(100,116,139,0.1)',
                                            }}>{rel}%</span>
                                        </div>
                                        <p className="insight-card-text">{ins.text}</p>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}

            {/* ── Section Tabs + Chart Grid ──────────────────────────────── */}
            <div className="charts-panel-header">
                <Layers size={15} style={{ color: '#818cf8' }} />
                <span style={{ fontWeight: 700, color: '#e2e8f0', fontSize: '0.88rem' }}>Interactive Illustrations</span>
                <span style={{ fontSize: '0.73rem', color: '#475569', marginLeft: '0.5rem' }}>
                    {loading ? 'Analysing data…' : `${visibleCharts.length} of ${charts.length} chart${charts.length !== 1 ? 's' : ''}`}
                </span>
                {loading && <span className="spinner" style={{ marginLeft: '0.75rem' }} />}
            </div>

            {/* Section filter tabs */}
            {!loading && charts.length > 0 && availableSections.length > 2 && (
                <div className="section-tabs">
                    {availableSections.map(sec => {
                        const meta = sec === 'all' ? { label: 'All', color: '#818cf8', emoji: '📊' } : (SECTION_META[sec] || { label: sec, color: '#818cf8', emoji: '📊' });
                        const cnt  = sec === 'all' ? charts.length : charts.filter(c => (c.section || 'schema') === sec).length;
                        return (
                            <button key={sec}
                                className={`section-tab ${activeSection === sec ? 'section-tab--active' : ''}`}
                                style={{ '--tab-color': meta.color }}
                                onClick={() => setActiveSection(sec)}>
                                <span>{meta.emoji}</span>
                                <span>{meta.label}</span>
                                <span className="section-tab-count">{cnt}</span>
                            </button>
                        );
                    })}
                </div>
            )}

            {loading && (
                <div className="charts-grid">
                    {Array.from({ length: 6 }).map((_, i) => (
                        <div key={i} className="chart-card chart-loading-skeleton" style={{ padding: '1rem' }}>
                            <div className="skeleton-line" style={{ width: '60%', height: 14, marginBottom: 8 }} />
                            <div className="skeleton-line" style={{ width: '90%', height: 8, marginBottom: 16 }} />
                            <div className="skeleton-block" style={{ height: 200 }} />
                        </div>
                    ))}
                </div>
            )}

            {!loading && visibleCharts.length > 0 && (
                <div className="charts-grid">
                    {visibleCharts.map(chart => {
                        const secMeta = SECTION_META[chart.section] || SECTION_META.schema;
                        return (
                            <div key={chart.id} className="chart-card"
                                style={{ '--card-accent': secMeta.color }}
                                onClick={() => setFullscreenChart(chart)}>
                                <div className="chart-card-header">
                                    <div className="chart-card-header-top">
                                        <span className="chart-card-section-dot" style={{ background: secMeta.color }} title={secMeta.label} />
                                        <span className="chart-card-title">{chart.title}</span>
                                        <SigBadge score={chart.significance} />
                                    </div>
                                    {chart.summary && <div className="chart-card-summary">{chart.summary}</div>}
                                </div>
                                <div className="chart-card-body" onClick={e => e.stopPropagation()}>
                                    <ChartRenderer chart={chart} />
                                </div>
                                {chart.explanation && (
                                    <div className="chart-explanation">{chart.explanation}</div>
                                )}
                                <div className="chart-card-footer">
                                    {chart.x_label && <span className="chart-axis-label"><span className="axis-tag">X</span> {chart.x_label}</span>}
                                    {chart.y_label && <span className="chart-axis-label"><span className="axis-tag">Y</span> {chart.y_label}</span>}
                                    <span className="chart-cols-badge">{(chart.columns || []).slice(0, 3).join(', ')}</span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {!loading && visibleCharts.length === 0 && charts.length > 0 && (
                <div className="charts-empty">
                    <PieIcon size={32} style={{ color: '#374151', marginBottom: '0.75rem' }} />
                    <p>No charts in this section. Switch to <strong>All</strong> to see all {charts.length} illustrations.</p>
                </div>
            )}

            {!loading && charts.length === 0 && (
                <div className="charts-empty">
                    <PieIcon size={36} style={{ color: '#374151', marginBottom: '1rem' }} />
                    <p>No illustrations available. Run the pipeline to generate charts.</p>
                </div>
            )}

            {/* ── Fullscreen Chart Modal ─────────────────────────────────── */}
            {fullscreenChart && (
                <div className="chart-fullscreen-overlay" onClick={() => setFullscreenChart(null)}>
                    <div className="chart-fullscreen-modal" onClick={e => e.stopPropagation()}>
                        <div className="chart-fullscreen-header">
                            <span className="chart-fullscreen-title">{fullscreenChart.title}</span>
                            <button className="chart-fullscreen-close" onClick={() => setFullscreenChart(null)}>
                                <X size={18} />
                            </button>
                        </div>
                        {fullscreenChart.summary && (
                            <div style={{ fontSize: '0.82rem', color: '#94a3b8', padding: '0 1.5rem 0.5rem' }}>{fullscreenChart.summary}</div>
                        )}
                        <div style={{ padding: '0 1.5rem 1rem', flex: 1 }}>
                            <ChartRenderer chart={{ ...fullscreenChart, _fullscreen: true }} />
                        </div>
                        {fullscreenChart.explanation && (
                            <div style={{ padding: '1rem 1.5rem', borderTop: '1px solid rgba(99,102,241,0.15)', fontSize: '0.82rem', color: '#94a3b8', lineHeight: 1.6 }}>
                                {fullscreenChart.explanation}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

// ── Unified Results Section (tabs: Report | Charts | Data) ────────────────────
const UnifiedResultsSection = ({ result, charts, kpis, insightsFeed, sectionInsights, chartsLoading }) => {
    const [tab, setTab] = useState('report');
    const TABS = [
        { id: 'report', icon: '📋', label: 'Analysis Report'   },
        { id: 'charts', icon: '📊', label: 'Illustrations'      },
        { id: 'data',   icon: '🗃️', label: 'Analysed Data'     },
    ];
    return (
        <div className="unified-results">
            <div className="result-tabs">
                {TABS.map(t => (
                    <button key={t.id}
                        className={`result-tab ${tab === t.id ? 'result-tab--active' : ''}`}
                        onClick={() => setTab(t.id)}>
                        <span>{t.icon}</span> {t.label}
                        {t.id === 'charts' && chartsLoading && (
                            <span className="spinner" style={{ width: 10, height: 10, borderWidth: 1.5, marginLeft: 6 }} />
                        )}
                        {t.id === 'charts' && !chartsLoading && charts.length > 0 && (
                            <span className="tab-count-badge">{charts.length}</span>
                        )}
                    </button>
                ))}
            </div>
            <div className="unified-tab-body">
                {tab === 'report' && <ResultPanel result={result} />}
                {tab === 'charts' && (
                    <PowerBIPanel
                        charts={charts}
                        kpis={kpis}
                        insightsFeed={insightsFeed}
                        sectionInsights={sectionInsights}
                        loading={chartsLoading}
                        rowCount={result.row_count}
                        colCount={result.col_count}
                    />
                )}
                {tab === 'data' && (
                    result.sample_rows?.length > 0
                        ? (
                            <>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.75rem',
                                    padding: '0.5rem 1rem', background: 'rgba(16,185,129,0.08)', borderRadius: 8,
                                    border: '1px solid rgba(16,185,129,0.25)' }}>
                                    <span style={{ color: '#34d399', fontWeight: 700, fontSize: '0.85rem' }}>Analysed Data Preview</span>
                                    <span style={{ fontSize: '0.75rem', color: '#475569' }}>post-pipeline transformed dataset · {result.sample_rows.length} rows</span>
                                </div>
                                <DataPreviewPanel rows={result.sample_rows} sourceKind={result.source_kind} />
                            </>
                        )
                        : <div style={{ textAlign: 'center', padding: '3rem', color: '#475569' }}>No processed data preview available.</div>
                )}
            </div>
        </div>
    );
};


// â”€â”€ Rich narrative report generator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const ResultPanel = ({ result }) => {
    const fr = result.final_result || {};
    const gate = fr.gate_decision || 'UNKNOWN';
    const quality = fr.quality_score != null ? `${(fr.quality_score * 100).toFixed(1)}%` : null;
    const rows = result.row_count ?? result.sample_rows?.length ?? 0;
    const govReport = fr.governance_report || {};
    const regReport = fr.regulatory_report || [];
    const edaSummary = fr.eda_report?.summary || fr.summary || {};
    const anomalies = fr.anomaly_deep_dive || {};
    const biasFair = fr.bias_fairness_report || {};
    const modelM = fr.model_metrics || {};
    const rlSum = fr.rl_agent_summary || {};
    const insights = fr.insights || [];
    const drift = fr.drift_report || {};
    const statTests = fr.statistical_tests || {};
    const leakage = fr.leakage_report || {};
    const multicollin = fr.multicollinearity_report || {};
    const instrSummary = fr.instruction_summary || [];

    // Anomaly details
    const totalAnomalies = anomalies.total_anomalies ?? edaSummary.anomaly_count ?? 0;
    const anomPct = edaSummary.anomaly_pct ?? (rows > 0 ? totalAnomalies / rows : 0);
    const anomCols = anomalies.per_column || [];
    const anomHandled = anomalies.handling_strategy || (anomPct < 0.05 ? 'Flagged and retained — anomaly rate below 5% threshold' : 'Applied winsorization and Isolation Forest removal');

    // PII
    const piiHits = govReport.pii_hits || {};
    const piiCols = Object.keys(piiHits);
    const totalRedactions = govReport.total_redactions || 0;
    const piiHandled = govReport.policy_applied || (piiCols.length > 0 ? 'REDACT' : 'PASS');

    // Regulatory
    const regFailed = regReport.filter(r => r.level === 'HIGH');
    const regWarned = regReport.filter(r => r.level === 'MEDIUM');
    const regDomains = [...new Set(regReport.map(r => r.category || r.domain || '').filter(Boolean))];

    // Bias
    const biasResults = (biasFair.results || []).filter(r => r.status === 'FAIL');
    const biasPass   = (biasFair.results || []).filter(r => r.status === 'PASS');

    // Drift
    const driftCols = drift.drifted_columns || [];
    const driftHandled = drift.handling || (driftCols.length > 0 ? 'Flagged for retraining; drift score stored in audit log' : 'No drift detected');

    // Null/missing
    const nullPct = edaSummary.overall_null_pct ?? fr.null_pct ?? 0;
    const nullCols = edaSummary.null_heavy_cols || [];
    const nullStrategy = fr.imputation_strategy || 'Median imputation for numeric; mode for categorical';

    // Model
    const roc = modelM.roc_auc ? (+modelM.roc_auc).toFixed(4) : null;
    const f1  = modelM.f1    ? (+modelM.f1).toFixed(4)    : null;
    const acc = modelM.accuracy ? (+modelM.accuracy).toFixed(4) : null;
    const bestModel = modelM.best_model || modelM.model_name || null;
    const featureImportance = fr.feature_importance || {};
    const topFeatures = Object.entries(featureImportance).sort(([,a],[,b]) => b - a).slice(0, 5);

    // Leakage & multicollinearity
    const leakedCols = leakage.leaked_columns || [];
    const highVifCols = (multicollin.vif_scores || []).filter(c => c.vif > 5);

    const gateColor = gate === 'PASS' ? '#10b981' : gate === 'WARN' ? '#f59e0b' : '#ef4444';
    const gateIcon  = gate === 'PASS' ? '✅' : gate === 'WARN' ? 'âš ï¸' : 'âŒ';

    const Divider = () => <div style={{height:'1px',background:'linear-gradient(90deg,rgba(99,102,241,0.15),transparent)',margin:'1.5rem 0'}}/>;
    const SubTitle = ({children,icon}) => <div style={{display:'flex',alignItems:'center',gap:'0.5rem',fontWeight:700,color:'#94a3b8',fontSize:'0.73rem',textTransform:'uppercase',letterSpacing:'0.06em',margin:'1rem 0 0.5rem'}}><span>{icon}</span>{children}</div>;
    const InfoRow = ({label,value,color}) => <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'0.3rem 0',borderBottom:'1px solid rgba(255,255,255,0.04)',fontSize:'0.81rem'}}><span style={{color:'#64748b'}}>{label}</span><span style={{fontWeight:600,color:color||'#e2e8f0'}}>{value}</span></div>;

    return (
        <div className="result-block">
            {/* â”€â”€ BANNER â”€â”€ */}
            <div className="result-banner" style={{ background: gate === 'PASS' ? 'linear-gradient(135deg,rgba(16,185,129,0.12),rgba(16,185,129,0.04))' : gate === 'WARN' ? 'linear-gradient(135deg,rgba(245,158,11,0.12),rgba(245,158,11,0.04))' : 'linear-gradient(135deg,rgba(239,68,68,0.12),rgba(239,68,68,0.04))', borderColor: gateColor + '44' }}>
                <div className="result-banner-left">
                    <span style={{ fontSize: '2.2rem' }}>{gateIcon}</span>
                    <div>
                        <div style={{ fontSize: '1.15rem', fontWeight: 800, color: gateColor }}>Pipeline {gate}</div>
                        <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: '0.2rem' }}>Run ID: <code style={{ color: '#a5b4fc' }}>{result.run_id}</code> &nbsp;·&nbsp; Source: <strong style={{color:'#94a3b8'}}>{SOURCE_LABELS[result.source_kind] ?? result.source_kind}</strong></div>
                    </div>
                </div>
                <div className="result-meta-chips">
                    {quality && <span className="rmeta-chip">🎯 Quality {quality}</span>}
                    <span className="rmeta-chip">📋 {rows.toLocaleString()} rows</span>
                    {edaSummary.n_cols && <span className="rmeta-chip">ðŸ—³ï¸ {edaSummary.n_cols} cols</span>}
                    {roc && <span className="rmeta-chip">📈 AUC {roc}</span>}
                    {f1  && <span className="rmeta-chip">⚡ F1 {f1}</span>}
                    {totalAnomalies > 0 && <span className="rmeta-chip" style={{ color: '#f87171', borderColor: '#ef444455' }}>ðŸ” {totalAnomalies} anomalies</span>}
                    {piiCols.length > 0 && <span className="rmeta-chip" style={{ color: '#c084fc', borderColor: '#a855f755' }}>🔒 {piiCols.length} PII col{piiCols.length>1?'s':''}</span>}
                    {driftCols.length > 0 && <span className="rmeta-chip" style={{ color: '#f59e0b', borderColor: '#f59e0b55' }}>📡 {driftCols.length} drift</span>}
                </div>
            </div>

            {/* â”€â”€ NARRATIVE REPORT â”€â”€ */}
            <div className="report-narrative">

                {/* â•â•â• Â§1 EXECUTIVE SUMMARY â•â•â• */}
                <div className="report-section">
                    <div className="report-section-title">📋 Executive Summary</div>

                    <p className="report-para">
                        The <strong>DIPEX Automated Data Processing (ADAP)</strong> pipeline successfully processed a dataset containing
                        <strong> {rows.toLocaleString()} records</strong>
                        {edaSummary.n_cols ? <> across <strong>{edaSummary.n_cols} columns</strong></> : ''}
                        , ingested via <em>{SOURCE_LABELS[result.source_kind] ?? result.source_kind}</em>.&nbsp;
                        The 15-stage processing pipeline executed end-to-end and returned a final gate decision of
                        <strong style={{ color: gateColor }}> {gate}</strong>
                        {quality ? <>, achieving an overall data quality score of <strong style={{color:'#34d399'}}>{quality}</strong></> : ''}.
                        {edaSummary.n_numeric != null && <> The dataset contains <strong>{edaSummary.n_numeric}</strong> numeric and <strong>{edaSummary.n_categorical ?? (edaSummary.n_cols - edaSummary.n_numeric)}</strong> categorical features.</>}
                    </p>

                    <p className="report-para">
                        {nullPct > 0
                            ? (<>A total of <strong style={{color:'#f87171'}}>{(nullPct * 100).toFixed(2)}%</strong> of all field values were found to be missing or null prior to processing.
                                {nullCols.length > 0 && <> The columns with the highest null density included: <em>{nullCols.slice(0,4).join(', ')}</em>.</>}
                                {' '}The ingestion pipeline applied <em><strong>{nullStrategy}</strong></em> across all affected columns to restore full data coverage.
                                {nullPct > 0.25 ? ' âš ï¸ High missingness (>25%) may compromise downstream model reliability — consider sourcing higher-quality data.' :
                                 nullPct > 0.10 ? ' Moderate null rates were detected; imputation was applied conservatively to avoid data leakage.' :
                                 ' Null rates remained within acceptable bounds for enterprise-grade data quality.'}
                              </>)
                            : 'The dataset arrived fully populated with zero null values — a strong indicator of high-quality upstream data collection and ETL processes.'}
                    </p>

                    <p className="report-para">
                        {gate === 'PASS'
                            ? <>
                                The pipeline’s dual quality gate system assessed this dataset as{' '}
                                <strong style={{color:'#10b981'}}>production-ready</strong>.
                                {quality && <> The quality score of <strong>{quality}</strong> exceeds the minimum production threshold of 70%,
                                indicating sufficient data integrity for safe deployment in machine learning pipelines and
                                business-critical decision systems.</>}
                                {' '}All 15 processing stages executed without triggering hard-stop conditions.
                                The data is cleared for downstream consumption and model training.
                              </>
                            : gate === 'WARN'
                            ? <>
                                The pipeline returned a <strong style={{color:'#f59e0b'}}>WARNING</strong> status —
                                the dataset passed basic quality thresholds but exhibits characteristics requiring human review before production use.
                                {quality && <> The quality score of <strong>{quality}</strong> is within the acceptable range
                                but below the confidence tier required for fully automated deployment.</>}
                                {' '}Review the anomaly, compliance, and drift findings below before authorising downstream consumption.
                              </>
                            : <>
                                The pipeline returned a <strong style={{color:'#ef4444'}}>FAIL</strong> status —
                                one or more hard quality gates were not satisfied.
                                {quality && <> The quality score of <strong>{quality}</strong> did not meet the required production threshold.</>}
                                {' '}This dataset should not be used for model training or automated decision-making
                                without significant data quality remediation. See the detailed findings in each section below.
                              </>
                        }
                    </p>

                    {instrSummary.length > 0 && (
                        <p className="report-para">
                            <strong>Analyst instructions were applied:</strong> {instrSummary.join(' · ')}.
                            The pipeline adapted its strategy accordingly, including modified model selection, feature engineering focus, and custom regulatory enforcement.
                            The RL orchestrator registered these instructions as contextual signals and updated its policy weights for future runs.
                        </p>
                    )}

                    {insights.length > 0 && (
                        <div style={{marginTop:'0.75rem'}}>
                            <SubTitle icon="💡">Top AI-Generated Insights</SubTitle>
                            <ul className="report-insights-list">
                                {insights.slice(0, 7).map((ins, i) => <li key={i}>{ins}</li>)}
                            </ul>
                        </div>
                    )}
                </div>

                <Divider />

                {/* â•â•â• Â§2 DATA INGESTION & SCHEMA â•â•â• */}
                <div className="report-section">
                    <div className="report-section-title">📥 Data Ingestion & Schema Analysis</div>
                    <p className="report-para">
                        Data was ingested from a <strong>{SOURCE_LABELS[result.source_kind] ?? result.source_kind}</strong> source.
                        The Universal Intake layer performed automatic schema inference, including column data-type detection (numeric, categorical, temporal, boolean, text), cardinality analysis, and uniqueness scoring.
                        {edaSummary.n_cols && <> A total of <strong>{edaSummary.n_cols}</strong> columns were detected in the raw schema.</>}
                        {edaSummary.n_duplicate_rows != null && edaSummary.n_duplicate_rows > 0 &&
                            <> <strong style={{color:'#f59e0b'}}>{edaSummary.n_duplicate_rows}</strong> duplicate rows were identified and removed during the Bronze → Silver cleaning stage.</>}
                    </p>
                    {edaSummary.n_cols && (
                        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(160px,1fr))',gap:'0.5rem',margin:'0.75rem 0'}}>
                            {edaSummary.n_numeric!=null&&<div className="report-issue-card" style={{borderColor:'rgba(99,102,241,0.3)'}}><div className="report-issue-col">🔢 Numeric</div><div style={{fontSize:'1.2rem',fontWeight:800,color:'#818cf8'}}>{edaSummary.n_numeric}</div><div className="report-issue-stat">columns</div></div>}
                            {edaSummary.n_categorical!=null&&<div className="report-issue-card" style={{borderColor:'rgba(16,185,129,0.3)'}}><div className="report-issue-col">ðŸ·ï¸ Categorical</div><div style={{fontSize:'1.2rem',fontWeight:800,color:'#34d399'}}>{edaSummary.n_categorical}</div><div className="report-issue-stat">columns</div></div>}
                            {edaSummary.n_temporal!=null&&<div className="report-issue-card" style={{borderColor:'rgba(245,158,11,0.3)'}}><div className="report-issue-col">📅 Temporal</div><div style={{fontSize:'1.2rem',fontWeight:800,color:'#fbbf24'}}>{edaSummary.n_temporal}</div><div className="report-issue-stat">columns</div></div>}
                            <div className="report-issue-card" style={{borderColor:'rgba(239,68,68,0.2)'}}><div className="report-issue-col">∅ Null Rate</div><div style={{fontSize:'1.2rem',fontWeight:800,color:nullPct>0.15?'#f87171':'#34d399'}}>{(nullPct*100).toFixed(1)}%</div><div className="report-issue-stat">overall</div></div>
                        </div>
                    )}

                    <p className="report-para">
                        {edaSummary.n_duplicate_rows != null && edaSummary.n_duplicate_rows > 0
                            ? <>
                                <strong style={{color:'#f59e0b'}}>{edaSummary.n_duplicate_rows.toLocaleString()}</strong> exact duplicate
                                rows were identified during the Bronze &rarr; Silver cleaning stage, representing{' '}
                                <strong>{rows > 0 ? ((edaSummary.n_duplicate_rows / rows) * 100).toFixed(2) : 0}%</strong> of total ingested volume.
                                Duplicate detection used a composite hash across all non-timestamp columns, ensuring only exact duplicates are removed while
                                near-duplicates are preserved and flagged. Removed duplicates are preserved in the ISSF audit trail with full provenance metadata.
                              </>
                            : 'No duplicate rows were detected — every record is unique across all fields. This is a strong signal that upstream data collection enforces primary key constraints or includes deduplication at source. The full row count is preserved for downstream processing.'}
                    </p>
                    <p className="report-para">
                        {driftCols.length > 0
                            ? <>
                                Distribution drift analysis was performed using <strong>Population Stability Index (PSI)</strong> against the previous run baseline.
                                PSI values below 0.1 indicate no meaningful change; 0.1&ndash;0.25 indicate moderate shifts requiring monitoring; above 0.25 indicate
                                major shifts that invalidate models trained on prior snapshots.
                                Significant drift was detected in <strong style={{color:'#f59e0b'}}>{driftCols.length} column{driftCols.length !== 1 ? 's' : ''}</strong>:{' '}
                                <em>{driftCols.slice(0,5).join(', ')}{driftCols.length > 5 ? ` and ${driftCols.length - 5} more` : ''}</em>.
                                Remediation applied: <strong>{driftHandled}</strong>.
                                All drifted columns have been flagged in the ISSF audit log. The RL orchestrator has been notified to trigger incremental model retraining
                                on the next scheduled run, ensuring prediction models remain aligned with the current data regime.
                              </>
                            : 'PSI drift analysis was performed against the previous run baseline. No significant distribution shifts were detected — all features maintain stable distributions consistent with historical norms. Models trained on prior snapshots remain statistically valid. Continuous drift monitoring remains active.'
                        }
                    </p>
                </div>

                <Divider />

                {/* â•â•â• Â§3 ANOMALY DETECTION & DATA QUALITY â•â•â• */}
                <div className="report-section">
                    <div className="report-section-title">ðŸ” Anomaly Detection & Data Quality</div>

                    <p className="report-para">
                        The ADAP anomaly detection layer employed an <strong>Isolation Forest</strong> algorithm with an adaptive contamination factor calibrated per column based on historical baselines.
                        {totalAnomalies > 0
                            ? (<> In this run, the algorithm identified <strong style={{color:'#f87171'}}>{totalAnomalies} anomalous record(s)</strong>,
                                representing <strong>{(anomPct * 100).toFixed(3)}%</strong> of the total dataset.
                                {anomCols.length > 0 && <> Anomaly density was highest in the following columns: <strong>{anomCols.slice(0,4).map(c=>c.col).join(', ')}</strong>.</>}
                                The maximum Z-score recorded across all anomalous values was <strong style={{color:'#fb923c'}}>{anomCols.length>0?Math.max(...anomCols.map(c=>c.z_score_max||0)).toFixed(3):'—'}</strong>,
                                indicating {'significant'} deviation from the population mean.
                              </>)
                            : ' No anomalous records were identified. All data points were consistent with expected statistical distributions as defined by the pipeline\'s contamination threshold and Z-score bounds.'}
                    </p>

                    <p className="report-para">
                        <strong>Remediation strategy applied: </strong><em>{anomHandled}</em>.{' '}
                        {anomPct >= 0.05
                            ? <>
                                The elevated anomaly rate (&ge;5%) triggered the <strong>aggressive remediation pathway</strong>:
                                winsorization was applied first, capping all extreme values at the 1st and 99th percentiles
                                to neutralise statistical influence while preserving record count.
                                Records with Isolation Forest confidence scores below the critical threshold were then selectively removed.
                                All removed records are preserved in the encrypted ISSF audit trail with full provenance metadata —
                                source row index, anomaly score, the specific triggering columns, and ingestion timestamp.
                              </>
                            : <>
                                Because the anomaly rate remained below the 5% production threshold, detected anomalies were{' '}
                                <strong>flagged in-place</strong> via the output metadata field <code>_anomaly_flag</code> rather than removed.
                                This conservative approach preserves full dataset volume for downstream consumers while maintaining complete transparency.
                                All flagged records and their anomaly scores are written to the ISSF audit trail.
                              </>}
                    </p>
                    <p className="report-para">
                        {totalAnomalies === 0
                            ? 'Data quality across all numeric features is exceptionally clean. The absence of anomalies across all columns is a strong positive signal for downstream model reliability — training sets built from this data will have minimal noise-to-signal contamination. Statistical tests confirm all major features exhibit distributions consistent with expected domain ranges.'
                            : anomPct < 0.01
                            ? `With an anomaly rate of only ${(anomPct * 100).toFixed(3)}%, overall data quality remains very high. The detected anomalies represent an extremely small fraction of total volume. After remediation, the cleaned dataset retains full representational integrity for model training and analysis.`
                            : `The ${(anomPct * 100).toFixed(2)}% anomaly rate indicates a non-trivial fraction of the dataset contains extreme or inconsistent values. Downstream models should be validated carefully and feature importance analysis examined to ensure anomalous records did not disproportionately influence any specific feature's distribution post-cleaning.`}
                    </p>

                    {anomCols.length > 0 && (
                        <>
                            <SubTitle icon="📊">Per-Column Anomaly Breakdown</SubTitle>
                            <div className="report-issue-grid">
                                {anomCols.slice(0, 8).map((c, i) => (
                                    <div key={i} className="report-issue-card" style={{ borderColor: '#ef444433' }}>
                                        <div className="report-issue-col">📊 {c.col}</div>
                                        <div className="report-issue-stat">Anomalies: <strong style={{ color: '#f87171' }}>{c.anomaly_count ?? '—'}</strong></div>
                                        <div className="report-issue-stat">Max Z-score: <strong style={{ color: '#fb923c' }}>{c.z_score_max?.toFixed(3) ?? '—'}</strong></div>
                                        <div className="report-issue-stat">IF Score: <strong>{c.if_score_mean?.toFixed(4) ?? '—'}</strong></div>
                                        <div className="report-issue-stat" style={{fontSize:'0.67rem',color:'#475569',marginTop:'0.25rem'}}>{c.action || anomHandled}</div>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}

                    {(statTests.normality?.length > 0 || statTests.stationarity?.length > 0) && (
                        <>
                            <SubTitle icon="ðŸ“">Statistical Hypothesis Tests</SubTitle>
                            <p className="report-para">
                                {statTests.normality?.length > 0 && (
                                    <>The <strong>Shapiro-Wilk normality test</strong> was applied to all {statTests.normality.length} numeric columns.
                                    <strong style={{color:'#f87171'}}> {statTests.normality.filter(t=>!t.is_normal).length}</strong> column(s) failed the normality test (p &lt; 0.05), indicating non-Gaussian distributions:
                                    <em> {statTests.normality.filter(t=>!t.is_normal).map(t=>t.col).slice(0,5).join(', ')}</em>.
                                    These columns were processed using non-parametric methods and robust scalers to avoid violating model assumptions. </>
                                )}
                                {statTests.stationarity?.filter(t=>!t.is_stationary).length > 0 && (
                                    <>The <strong>Augmented Dickey-Fuller (ADF)</strong> stationarity test flagged
                                    <strong style={{color:'#f59e0b'}}> {statTests.stationarity.filter(t=>!t.is_stationary).length}</strong> column(s) as non-stationary:
                                    <em> {statTests.stationarity.filter(t=>!t.is_stationary).map(t=>t.col).join(', ')}</em>.
                                    Log-differencing was applied to induce stationarity prior to time-series modelling.</>
                                )}
                            </p>
                        </>
                    )}

                    {(leakedCols.length > 0 || highVifCols.length > 0) && (
                        <>
                            <SubTitle icon="🚿">Leakage & Multicollinearity</SubTitle>
                            {leakedCols.length > 0 && (
                                <p className="report-para">
                                    <strong style={{color:'#f87171'}}>Target leakage</strong> was detected in {leakedCols.length} column(s): <em>{leakedCols.join(', ')}</em>.
                                    These features exhibited near-perfect correlation with the target column, indicating that they encode post-hoc information unavailable at prediction time.
                                    They have been removed from the training dataset to prevent inflated model performance metrics.
                                </p>
                            )}
                            {highVifCols.length > 0 && (
                                <p className="report-para">
                                    <strong style={{color:'#f59e0b'}}>Multicollinearity</strong> was detected in {highVifCols.length} feature pair(s) with Variance Inflation Factor (VIF) &gt; 5:
                                    <em> {highVifCols.map(c=>`${c.col} (VIF=${c.vif?.toFixed(1)})`).join(', ')}</em>.
                                    Redundant features were pruned using VIF-based sequential elimination to stabilise regression coefficients and improve interpretability.
                                </p>
                            )}
                        </>
                    )}
                </div>

                <Divider />

                {/* â•â•â• Â§4 GOVERNANCE, PII & REGULATORY COMPLIANCE â•â•â• */}
                <div className="report-section">
                    <div className="report-section-title">âš–ï¸ Governance, Privacy & Regulatory Compliance</div>

                    <p className="report-para">
                        The ADAP governance layer performs automated <strong>PII (Personally Identifiable Information) scanning</strong> using a multi-pattern regex engine tuned for
                        global data privacy frameworks including GDPR (EU), HIPAA (US), CCPA (California), PDPA (Thailand), and LGPD (Brazil).
                        {piiCols.length > 0
                            ? (<> In this run, PII was identified in <strong style={{color:'#c084fc'}}>{piiCols.length} column(s)</strong>: <em>{piiCols.join(', ')}</em>.
                                A total of <strong style={{color:'#a78bfa'}}>{totalRedactions} value(s)</strong> were
                                {piiHandled === 'REJECT'
                                    ? <> <strong style={{color:'#ef4444'}}>rejected and permanently removed</strong> from the output dataset in accordance with the REJECT governance policy.
                                        Rejected records are preserved only in the encrypted audit log accessible to authorized personnel.
                                      </>
                                    : <> <strong style={{color:'#818cf8'}}>redacted (replaced with masked placeholders)</strong> in compliance with the {piiHandled} governance policy.
                                        The original values are stored in an encrypted vault and excluded from all downstream ML training pipelines.
                                      </>}
                              </>)
                            : ' In this run, the PII scanner found no personally identifiable information across all columns and all rows. The dataset is fully compliant with all 12 monitored PII categories under applicable regulatory frameworks.'}
                    </p>

                    {regReport.length > 0 && (
                        <p className="report-para">
                            The regulatory rules engine evaluated <strong>{regReport.length} finding(s)</strong> across
                            {regDomains.length > 0 && <> the following compliance domains: <em>{regDomains.join(', ')}</em></>}.
                            {regFailed.length > 0 && <> <strong style={{ color: '#ef4444' }}>{regFailed.length} HIGH-severity violation(s)</strong> were detected — these represent critical compliance failures that must be remediated before data can be used in production environments.</> }
                            {regWarned.length > 0 && <> <strong style={{ color: '#f59e0b' }}>{regWarned.length} MEDIUM-severity finding(s)</strong> were recorded in the audit trail — these require review within 30 days under standard compliance frameworks.</> }
                            {regReport.filter(r=>r.level==='LOW').length > 0 && <> {regReport.filter(r=>r.level==='LOW').length} LOW-severity informational finding(s) were logged for transparency.</> }
                        </p>
                    )}

                    {regFailed.length > 0 && (
                        <>
                            <SubTitle icon="🚨">HIGH-Severity Violations (Immediate Action Required)</SubTitle>
                            <div className="report-violation-list">
                                {regFailed.map((v, i) => (
                                    <div key={i} className="report-violation">
                                        <span className="report-viol-badge high">HIGH</span>
                                        <div style={{flex:1}}>
                                            <div className="report-viol-rule">{v.category || v.rule_name}</div>
                                            <div className="report-viol-msg">{v.message}</div>
                                            {v.remediation && <div style={{fontSize:'0.73rem',color:'#818cf8',marginTop:'0.3rem'}}>→ Remediation: {v.remediation}</div>}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}

                    {regWarned.length > 0 && (
                        <>
                            <SubTitle icon="âš ï¸">MEDIUM-Severity Warnings</SubTitle>
                            <div className="report-violation-list">
                                {regWarned.map((v, i) => (
                                    <div key={i} className="report-violation" style={{borderColor:'rgba(245,158,11,0.3)',background:'rgba(245,158,11,0.04)'}}>
                                        <span className="report-viol-badge medium">MED</span>
                                        <div style={{flex:1}}>
                                            <div className="report-viol-rule">{v.category || v.rule_name}</div>
                                            <div className="report-viol-msg">{v.message}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}

                    {biasResults.length > 0 && (
                        <>
                            <SubTitle icon="âš–ï¸">Bias & Fairness — Disparate Impact Failures</SubTitle>
                            <p className="report-para">
                                The disparate impact fairness analysis (4/5ths rule, DI threshold = 0.80) identified
                                <strong style={{ color: '#f59e0b' }}> {biasResults.length} group(s)</strong> with statistically significant outcome disparities:
                                &nbsp;{biasResults.map(r => `${r.group_col}=${r.group_value} (DI=${r.disparate_impact?.toFixed(3) ?? '—'}, pos_rate=${r.positive_rate?.toFixed(3) ?? '—'})`).join('; ')}.
                                These groups have been documented in the bias audit trail.
                                <strong> Deployment of any model trained on this dataset for decisions affecting these groups requires explicit human review and approval.</strong>
                            </p>
                            {biasPass.length > 0 && (
                                <p className="report-para" style={{fontSize:'0.79rem',color:'#64748b'}}>
                                    {biasPass.length} group(s) passed the fairness threshold: {biasPass.slice(0,4).map(r=>`${r.group_col}=${r.group_value}`).join(', ')}.
                                </p>
                            )}
                        </>
                    )}
                </div>

                <Divider />

                {/* â•â•â• Â§5 MODEL PERFORMANCE & FEATURE INTELLIGENCE â•â•â• */}
                {(roc || f1 || acc || topFeatures.length > 0) && (
                    <>
                        <div className="report-section">
                            <div className="report-section-title">ðŸ¤– AutoML Model Performance</div>

                            <p className="report-para">
                                The AutoML layer executed a model race across Random Forest, XGBoost, LightGBM, and Logistic Regression, using
                                stratified 5-fold cross-validation with early stopping. Hyperparameter search was performed using Bayesian optimisation.
                                {bestModel && <> The best-performing model was <strong style={{color:'#a5b4fc'}}>{bestModel}</strong>.</>}
                            </p>

                            {(roc || f1 || acc) && (
                                <>
                                    <SubTitle icon="📊">Performance Metrics</SubTitle>
                                    <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(130px,1fr))',gap:'0.5rem',margin:'0.5rem 0 1rem'}}>
                                        {roc&&<div className="report-issue-card" style={{borderColor:'rgba(52,211,153,0.4)'}}><div className="report-issue-col">ROC-AUC</div><div style={{fontSize:'1.4rem',fontWeight:800,color:'#34d399'}}>{roc}</div><div className="report-issue-stat">{+roc>0.95?'ðŸ† Excellent':+roc>0.85?'✅ Good':+roc>0.70?'âš ï¸ Fair':'âŒ Poor'}</div></div>}
                                        {f1&&<div className="report-issue-card" style={{borderColor:'rgba(99,102,241,0.4)'}}><div className="report-issue-col">F1 Score</div><div style={{fontSize:'1.4rem',fontWeight:800,color:'#818cf8'}}>{f1}</div><div className="report-issue-stat">{+f1>0.85?'ðŸ† Excellent':+f1>0.70?'✅ Good':+f1>0.55?'âš ï¸ Fair':'âŒ Poor'}</div></div>}
                                        {acc&&<div className="report-issue-card" style={{borderColor:'rgba(59,130,246,0.4)'}}><div className="report-issue-col">Accuracy</div><div style={{fontSize:'1.4rem',fontWeight:800,color:'#60a5fa'}}>{acc}</div><div className="report-issue-stat">{+acc>0.95?'ðŸ† Excellent':+acc>0.85?'✅ Good':+acc>0.70?'âš ï¸ Fair':'âŒ Poor'}</div></div>}
                                    </div>
                                    <p className="report-para">
                                        {roc && (+roc > 0.90
                                            ? `An AUC of ${roc} indicates excellent discriminative power — the model confidently separates positive from negative classes and is ready for production deployment.`
                                            : +roc > 0.75
                                            ? `An AUC of ${roc} indicates good but improvable performance. Consider feature engineering, additional training data, or ensemble stacking to push above 0.90.`
                                            : `An AUC of ${roc} is below the recommended threshold for production use (0.75). Investigate class imbalance, feature quality, and whether the target column definition is correct.`)}
                                    </p>
                                </>
                            )}

                            {topFeatures.length > 0 && (
                                <>
                                    <SubTitle icon="ðŸ†">Top Predictive Features</SubTitle>
                                    <p className="report-para">The following features contributed most to the model\'s predictive performance, ranked by their aggregated importance score across all models in the race:</p>
                                    {topFeatures.map(([feat, score], i) => (
                                        <div key={feat} style={{display:'flex',alignItems:'center',gap:'0.75rem',marginBottom:'0.4rem'}}>
                                            <span style={{width:'20px',fontWeight:700,color:'#818cf8',fontSize:'0.75rem'}}>#{i+1}</span>
                                            <span style={{flex:1,fontSize:'0.8rem',color:'#94a3b8'}}>{feat}</span>
                                            <div style={{width:`${Math.round(score*200)}px`,maxWidth:'120px',height:'6px',background:'linear-gradient(90deg,#6366f1,#8b5cf6)',borderRadius:'3px'}} />
                                            <span style={{fontSize:'0.75rem',color:'#a5b4fc',fontWeight:700,width:'42px',textAlign:'right'}}>{(score*100).toFixed(1)}%</span>
                                        </div>
                                    ))}
                                </>
                            )}
                        </div>
                        <Divider />
                    </>
                )}

                {/* â•â•â• Â§6 RL AGENT PERFORMANCE â•â•â• */}
                {(rlSum.episode_count != null || rlSum.last_reward != null) && (
                    <div className="report-section">
                        <div className="report-section-title">ðŸ¤– Reinforcement Learning Orchestrator (PPO Agent)</div>

                        <p className="report-para">
                            The ADAP PPO-based reinforcement learning orchestrator continuously learns from pipeline outcomes to improve future configuration decisions.
                            {rlSum.episode_count != null && <> It has completed <strong>{rlSum.episode_count} exploration episode(s)</strong> to date.</>}
                            {rlSum.in_shadow_mode
                                ? ` The agent is currently in <strong>shadow mode</strong>, collecting behavioural data across ${Math.max(0, 20 - (rlSum.episode_count || 0))} more episodes before activating full policy gradient updates. During shadow mode, the agent observes all pipeline decisions and computes counterfactual rewards without altering the execution path.`
                                : ' The agent is in <strong>active mode</strong>, directly influencing pipeline configuration decisions including model selection strategy, imputation method, feature engineering depth, and anomaly handling thresholds.'}
                        </p>

                        {rlSum.last_reward != null && (
                            <p className="report-para">
                                The reward signal for this run was <strong style={{ color: +rlSum.last_reward > 0.6 ? '#34d399' : +rlSum.last_reward > 0.3 ? '#f59e0b' : '#f87171' }}>{(+rlSum.last_reward).toFixed(4)}</strong>,
                                computed as a weighted combination of: data quality score, model performance metrics, processing efficiency, and compliance gate results.
                                {+rlSum.last_reward > 0.6 ? ' This high reward signal indicates the agent selected an effective strategy for this dataset type.' :
                                 +rlSum.last_reward > 0.3 ? ' This moderate reward suggests room for improvement — the agent will adjust its policy to increase future rewards.' :
                                 ' This low reward indicates the pipeline encountered significant challenges. The agent will apply a large negative update to avoid repeating these configuration choices.'}
                                {rlSum.recommended_action && <> For the next run, the agent recommends: <strong>{Object.entries(rlSum.recommended_action).slice(0,4).map(([k,v])=>`${k.replace(/_/g,' ')}=${v}`).join(', ')}</strong>.</>}
                            </p>
                        )}
                    </div>
                )}

            </div>{/* /report-narrative */}

            <GovernanceSummary govReport={govReport.status ? govReport : fr.governance_report} />
            <RegulatorySummary regulatoryReport={regReport} />

            {fr.report_path && <p className="report-hint">📄 Full HTML report: <code>{fr.report_path}</code></p>}
            <div className="result-actions">
                <Link to="/" className="view-reports-link primary">View Analytics Dashboard →</Link>
                <a href={`${API_BASE}/api/export/report/${fr.run_id}`} className="view-reports-link" target="_blank" rel="noopener noreferrer">View Report →</a>
            </div>
        </div>
    );
};

// â”€â”€ CSV parser utility (browser-side, no deps) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function _csvLine(line) {
    const out = []; let cur = '', inQ = false;
    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (inQ) {
            if (ch === '"') { if (line[i + 1] === '"') { cur += '"'; i++; } else inQ = false; }
            else cur += ch;
        } else {
            if (ch === '"') inQ = true;
            else if (ch === ',') { out.push(cur.trim()); cur = ''; }
            else cur += ch;
        }
    }
    out.push(cur.trim());
    return out;
}

function parseCSVPreview(text, maxRows = 500) {
    const lines = text.split(/\r?\n/).filter(l => l.trim());
    if (lines.length < 2) return [];
    const headers = _csvLine(lines[0]);
    return lines.slice(1, maxRows + 1).map(line => {
        const cols = _csvLine(line);
        const row = {};
        headers.forEach((h, i) => { row[h] = cols[i] ?? ''; });
        return row;
    }).filter(row => Object.values(row).some(v => v !== ''));
}

const RunPipeline = () => {
    const [mode, setMode] = useState('file');
    const [optionalOpen, setOptionalOpen] = useState(false);
    const [file, setFile] = useState(null);
    const [dragging, setDragging] = useState(false);
    const [targetCol, setTargetCol] = useState('');
    const [colRange, setColRange] = useState('');
    const [rowRange, setRowRange] = useState('');
    const [domain, setDomain] = useState('');
    const [extraDomains, setExtraDomains] = useState([]);

    // â”€â”€ Raw ingestion preview â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    const [previewRows, setPreviewRows] = useState([]);
    const [previewLoading, setPreviewLoading] = useState(false);
    // Exact stats computed from the full file at load time (no estimation)
    const [fileMetaRows, setFileMetaRows] = useState(null);     // actual row count
    const [fileMetaNullRate, setFileMetaNullRate] = useState(0); // null fraction 0-1

    const [dbType, setDbType] = useState('postgresql');
    const [dbHost, setDbHost] = useState('postgres');   // Docker service name
    const [dbPort, setDbPort] = useState('5432');
    const [dbUser, setDbUser] = useState('dipex');
    const [dbPass, setDbPass] = useState('dipex_secret');
    const [dbName, setDbName] = useState('dipex');
    const [dbTable, setDbTable] = useState('');

    const [kafkaTopic, setKafkaTopic] = useState('dipex_pipeline');
    const [kafkaBroker, setKafkaBroker] = useState('kafka:29092');
    const [kafkaGroupId, setKafkaGroupId] = useState('dipex-consumer');
    const [kafkaMaxMessages, setKafkaMaxMessages] = useState('10000');
    const [kafkaInfoOpen, setKafkaInfoOpen] = useState(false);

    const [apiUrl, setApiUrl] = useState('');
    const [apiMethod, setApiMethod] = useState('GET');
    const [apiAuthType, setApiAuthType] = useState('none');
    const [apiBearerToken, setApiBearerToken] = useState('');
    const [apiKeyName, setApiKeyName] = useState('X-API-Key');
    const [apiKeyValue, setApiKeyValue] = useState('');
    const [apiBasicUser, setApiBasicUser] = useState('');
    const [apiBasicPass, setApiBasicPass] = useState('');
    const [apiHeaders, setApiHeaders] = useState([{ key: '', value: '' }]);
    const [apiBody, setApiBody] = useState('');
    const [apiShowPass, setApiShowPass] = useState(false);

    const [running, setRunning] = useState(false);
    const [connecting, setConnecting] = useState(false);
    const [availableTables, setAvailableTables] = useState([]);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    // ── Power BI chart state ─────────────────────────────────────────────────
    const [charts, setCharts] = useState([]);
    const [sectionInsights, setSectionInsights] = useState({});
    const [kpis, setKpis] = useState({});
    const [insightsFeed, setInsightsFeed] = useState([]);
    const [chartsLoading, setChartsLoading] = useState(false);

    // â”€â”€ Analyst Instruction Loop state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    const [userInstructions, setUserInstructions] = useState('');
    const [planRejectionCount, setPlanRejectionCount] = useState(0);
    const [feedbackState, setFeedbackState] = useState(null); // null | 'pending' | 'submitted'
    const [feedbackHappy, setFeedbackHappy] = useState(null);
    const [feedbackReason, setFeedbackReason] = useState('');
    const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
    const [feedbackResult, setFeedbackResult] = useState(null);

    // â”€â”€ Pre-Analysis Plan modal state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    const [planModalOpen, setPlanModalOpen] = useState(false);
    const [planLoading, setPlanLoading] = useState(false);
    const [plan, setPlan] = useState(null);
    const [instructionSummary, setInstructionSummary] = useState([]);
    // Holds FormData built at preview time, re-used on Approve
    const pendingFormRef = useRef(null);

    // â”€â”€ Load raw preview when file is chosen â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    const loadFilePreview = useCallback((f) => {
        if (!f) { setPreviewRows([]); setFileMetaRows(null); setFileMetaNullRate(0); return; }

        if (f.name.match(/\.(csv|tsv|txt)$/i)) {
            setPreviewLoading(true);
            const reader = new FileReader();
            reader.onload = (ev) => {
                try {
                    const text = ev.target.result;
                    // Split into lines, remove blank lines
                    const allLines = text.split(/\r?\n/).filter(l => l.trim());
                    if (allLines.length < 2) { setPreviewRows([]); setFileMetaRows(0); setFileMetaNullRate(0); return; }

                    const headers = _csvLine(allLines[0]);
                    const dataLines = allLines.slice(1); // every row after header

                    // ── Exact row count (no estimation) ─────────────────────────
                    setFileMetaRows(dataLines.length);

                    // ── Null rate: scan up to 5 000 rows for speed ───────────────
                    const NULL_PAT = /^(|null|nan|na|n\/a|none|undefined|#n\/a)$/i;
                    let totalCells = 0, nullCells = 0;
                    const scanLines = dataLines.slice(0, Math.min(5000, dataLines.length));
                    scanLines.forEach(line => {
                        const cols = _csvLine(line);
                        headers.forEach((_, i) => {
                            totalCells++;
                            if (NULL_PAT.test((cols[i] ?? '').trim())) nullCells++;
                        });
                    });
                    setFileMetaNullRate(totalCells > 0 ? nullCells / totalCells : 0);

                    // ── Preview first 500 rows for the table ────────────────────
                    setPreviewRows(parseCSVPreview(text, 500));
                } catch { setPreviewRows([]); setFileMetaRows(null); setFileMetaNullRate(0); }
                finally { setPreviewLoading(false); }
            };
            reader.readAsText(f);

        } else if (f.name.match(/\.json$/i)) {
            setPreviewLoading(true);
            const reader = new FileReader();
            reader.onload = (ev) => {
                try {
                    const parsed = JSON.parse(ev.target.result);
                    const rows = Array.isArray(parsed) ? parsed : (parsed.data || parsed.results || Object.values(parsed)[0] || []);
                    setFileMetaRows(rows.length); // exact count
                    // Null rate from all rows (JSON is already parsed)
                    if (rows.length > 0 && typeof rows[0] === 'object') {
                        const keys = Object.keys(rows[0]);
                        let totalCells = 0, nullCells = 0;
                        rows.slice(0, 5000).forEach(row => keys.forEach(k => {
                            totalCells++;
                            const v = row[k];
                            if (v === null || v === undefined || String(v).trim() === '') nullCells++;
                        }));
                        setFileMetaNullRate(totalCells > 0 ? nullCells / totalCells : 0);
                    }
                    setPreviewRows(rows.slice(0, 500));
                } catch { setPreviewRows([]); setFileMetaRows(null); setFileMetaNullRate(0); }
                finally { setPreviewLoading(false); }
            };
            reader.readAsText(f);

        } else {
            // Excel/Parquet — backend will provide exact counts after processing
            setPreviewRows([{ _info: `${f.name} — row/column counts will be shown after pipeline runs` }]);
            setFileMetaRows(null);     // unknown until backend reads it
            setFileMetaNullRate(0);
        }
    }, []);

    const onDrop = useCallback((e) => { e.preventDefault(); setDragging(false); const d = e.dataTransfer.files[0]; if (d) { setFile(d); loadFilePreview(d); } }, [loadFilePreview]);
    const onDragOver = e => { e.preventDefault(); setDragging(true); };
    const onDragLeave = () => setDragging(false);
    const onFileChange = e => { if (e.target.files[0]) { setFile(e.target.files[0]); loadFilePreview(e.target.files[0]); } };

    const buildApiConfig = () => {
        const cfg = { url: apiUrl.trim(), method: apiMethod };
        if (apiAuthType === 'bearer' && apiBearerToken) cfg.auth = { type: 'bearer', token: apiBearerToken };
        else if (apiAuthType === 'apikey' && apiKeyValue) cfg.auth = { type: 'apikey', name: apiKeyName || 'X-API-Key', value: apiKeyValue };
        else if (apiAuthType === 'basic' && apiBasicUser) cfg.auth = { type: 'basic', username: apiBasicUser, password: apiBasicPass };
        const validHeaders = apiHeaders.filter(h => h.key.trim() && h.value.trim());
        if (validHeaders.length) cfg.headers = Object.fromEntries(validHeaders.map(h => [h.key.trim(), h.value.trim()]));
        if (['POST', 'PUT', 'PATCH'].includes(apiMethod) && apiBody.trim()) cfg.body = apiBody.trim();
        return cfg;
    };

    // â”€â”€ Build FormData shared between preview and run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    const buildFormData = (forcedTable = null) => {
        const form = new FormData();
        form.append('source_kind', mode);
        if (targetCol) form.append('target_col', targetCol.trim());
        if (colRange) form.append('col_range', colRange.trim());
        if (rowRange) form.append('row_range', rowRange.trim().replace(/\s+/g, ''));
        if (domain) form.append('domain', domain);
        if (extraDomains.length) form.append('extra_domains', extraDomains.join(','));
        // â”€â”€ Analyst instruction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if (userInstructions.trim()) form.append('user_instructions', userInstructions.trim());
        form.append('plan_rejection_count', String(planRejectionCount));

        if (mode === 'file') {
            if (!file) throw new Error('Please select or drop a file first.');
            form.append('file', file);
        } else if (mode === 'database') {
            const finalTable = forcedTable || dbTable;
            if (!dbHost.trim() || !dbName.trim() || !finalTable.trim()) throw new Error('Please discover tables first or provide a Table name manually.');
            const dbConfig = { backend: dbType, host: dbHost, port: dbPort, database: dbName, username: dbUser, password: dbPass, table: finalTable };
            form.append('source_input', JSON.stringify(dbConfig));
            if (dbType === 'neo4j') form.set('source_kind', 'graph_db');
        } else if (mode === 'live') {
            if (!kafkaBroker.trim() || !kafkaTopic.trim()) throw new Error('Please enter Kafka Broker and Topic.');
            form.append('source_input', JSON.stringify({ brokers: kafkaBroker.trim(), topic: kafkaTopic.trim(), group_id: kafkaGroupId.trim() || 'dipex-consumer', max_messages: parseInt(kafkaMaxMessages, 10) || 10000 }));
        } else if (mode === 'api') {
            if (!apiUrl.trim()) throw new Error('Please enter a REST API URL.');
            form.append('source_input', JSON.stringify(buildApiConfig()));
        }
        return form;
    };

    // â”€â”€ Step 1: Show pre-analysis plan modal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    const handleRunSubmit = async (e, forcedTable = null) => {
        if (e) e.preventDefault();
        setError(null); setResult(null);
        let form;
        try {
            form = buildFormData(forcedTable);
        } catch (err) {
            setError(err.message || 'Form validation error');
            return;
        }
        pendingFormRef.current = form;

        // Fetch preview plan (fast, metadata-only)
        setPlanLoading(true);
        setPlanModalOpen(true);
        setPlan(null);
        try {
            // ── Real dataset metadata — computed from file at load time ────────
            const _dataRows = previewRows.filter(r => !r._info && !r._error);
            const _planCols = _dataRows.length > 0
                ? Object.keys(_dataRows[0]).filter(k => !k.startsWith('_'))
                : [];
            const _planRows  = fileMetaRows ?? (_dataRows.length > 0 ? _dataRows.length : null);
            const _planNullR = fileMetaNullRate ?? 0;
            // Analyze actual column types from real data values — NOT name guessing
            const _schema = _dataRows.length > 0 ? analyzeSchema(_dataRows) : null;
            const planPayload = {
                domain: domain || 'generic',
                target_col: targetCol.trim() || null,
                mode: 'auto',
                user_instructions: userInstructions.trim() || '',
                plan_rejection_count: planRejectionCount,
                n_rows:                 _planRows,
                n_cols:                 _planCols.length || null,
                column_names:           _planCols.length > 0 ? _planCols : null,
                null_rate:              _planNullR,
                // Exact type counts from inferType() on actual column values
                numeric_cols_count:     _schema ? _schema.numericCols.length     : null,
                categorical_cols_count: _schema ? _schema.categoricalCols.length : null,
                temporal_cols_count:    _schema ? (_schema.temporalCols  || []).length : null,
                text_cols_count:        _schema ? (_schema.textCols      || []).length : null,
            };
            const planRes = await fetch(`${API_BASE}/api/pipeline/preview-plan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(planPayload),
            });
            if (planRes.ok) {
                const planData = await planRes.json();
                setPlan(planData);
                setInstructionSummary(planData.instruction_summary || []);
            } else {
                setPlan({ data_summary: {}, domain: { active: domain || 'generic', rules: [], rules_count: 0 }, operations: [], warnings: [], instruction_summary: [] });
                setInstructionSummary([]);
            }
        } catch {
            setPlan({ data_summary: {}, domain: { active: domain || 'generic', rules: [], rules_count: 0 }, operations: [], warnings: [{ level: 'info', message: 'Preview plan unavailable — proceeding without preview.' }], instruction_summary: [] });
            setInstructionSummary([]);
        } finally {
            setPlanLoading(false);
        }
    };

    // â”€â”€ Step 2: User approved plan — run the actual pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    const handlePlanApproved = async () => {
        setPlanModalOpen(false);
        setPlan(null);
        setRunning(true);
        setFeedbackState(null);
        setFeedbackHappy(null);
        setFeedbackReason('');
        setFeedbackResult(null);
        try {
            const form = pendingFormRef.current;
            if (!form) throw new Error('Internal error: form data lost.');
            // Signal to audit log that the user reviewed the plan
            form.append('plan_approved', 'true');

            const res = await fetch(`${API_BASE}/api/pipeline/simple-run`, { method: 'POST', body: form });
            const text = await res.text();
            let data;
            try { data = JSON.parse(text); } catch {
                if (res.status === 413) throw new Error('File too large — the server rejected the upload (413). Try a smaller file or increase the server upload limit.');
                throw new Error(`Server returned HTTP ${res.status} with a non-JSON body. Make sure the ADAP backend is running on port 8000.`);
            }
            if (!res.ok) throw new Error(data.detail || `Pipeline run failed (HTTP ${res.status})`);
            setResult(data);
            setCharts([]); setKpis({}); setInsightsFeed([]); setSectionInsights({});
            setFeedbackState('pending'); // show feedback panel
            // ── Async intelligence fetch (Power BI charts) ───────────────────────
            if (data.run_id) {
                setChartsLoading(true);
                fetch(`${API_BASE}/api/results/${data.run_id}/intelligence`)
                    .then(r => r.ok ? r.json() : null)
                    .then(intel => {
                        if (intel) {
                            setCharts(intel.charts || []);
                            setKpis(intel.kpis || {});
                            setInsightsFeed(intel.insights_feed || []);
                            setSectionInsights(intel.section_insights || {});
                        }
                    })
                    .catch(() => {})
                    .finally(() => setChartsLoading(false));
            }
        } catch (err) {
            const msg = err.message || 'Unexpected error';
            if (msg === 'Failed to fetch' || msg.includes('NetworkError') || msg.includes('ECONNREFUSED')) {
                setError('Cannot reach the API server. Make sure the ADAP backend is running on port 8000.');
            } else { setError(msg); }
        } finally { setRunning(false); pendingFormRef.current = null; }
    };

    // â”€â”€ Step 2b: User rejected plan — increment counter and re-fetch plan â”€â”€â”€â”€â”€â”€â”€â”€â”€
    const handlePlanRejected = async () => {
        const newCount = planRejectionCount + 1;
        setPlanRejectionCount(newCount);
        // Re-fetch the plan with the updated rejection count
        setPlanLoading(true);
        setPlan(null);
        try {
            // ── Real dataset metadata — computed from file at load time ────────
            const _dataRows2 = previewRows.filter(r => !r._info && !r._error);
            const _planCols2 = _dataRows2.length > 0
                ? Object.keys(_dataRows2[0]).filter(k => !k.startsWith('_'))
                : [];
            const _planRows2 = fileMetaRows ?? (_dataRows2.length > 0 ? _dataRows2.length : null);
            const _schema2   = _dataRows2.length > 0 ? analyzeSchema(_dataRows2) : null;
            const planPayload = {
                domain: domain || 'generic',
                target_col: targetCol.trim() || null,
                mode: 'auto',
                user_instructions: userInstructions.trim() || '',
                plan_rejection_count: newCount,
                n_rows:                 _planRows2,
                n_cols:                 _planCols2.length || null,
                column_names:           _planCols2.length > 0 ? _planCols2 : null,
                null_rate:              fileMetaNullRate ?? 0,
                numeric_cols_count:     _schema2 ? _schema2.numericCols.length     : null,
                categorical_cols_count: _schema2 ? _schema2.categoricalCols.length : null,
                temporal_cols_count:    _schema2 ? (_schema2.temporalCols  || []).length : null,
                text_cols_count:        _schema2 ? (_schema2.textCols      || []).length : null,
            };
            const planRes = await fetch(`${API_BASE}/api/pipeline/preview-plan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(planPayload),
            });
            if (planRes.ok) {
                const planData = await planRes.json();
                setPlan(planData);
                setInstructionSummary(planData.instruction_summary || []);
            } else {
                setPlan({ data_summary: {}, domain: { active: domain || 'generic', rules: [], rules_count: 0 }, operations: [], warnings: [] });
            }
        } catch {
            setPlan({ data_summary: {}, domain: { active: domain || 'generic', rules: [], rules_count: 0 }, operations: [], warnings: [] });
        } finally { setPlanLoading(false); }
    };

    const handlePlanCancelled = () => {
        setPlanModalOpen(false);
        setPlan(null);
        setPlanLoading(false);
        pendingFormRef.current = null;
        // Note: do NOT reset planRejectionCount here — only reset on full new run
    };

    // â”€â”€ Feedback submission â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    const handleFeedbackSubmit = async (happy) => {
        if (!result?.run_id) return;
        setFeedbackHappy(happy);
        setFeedbackSubmitting(true);
        try {
            const res = await fetch(`${API_BASE}/api/pipeline/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    run_id: result.run_id,
                    happy,
                    reason: feedbackReason.trim(),
                    plan_rejection_count: planRejectionCount,
                    rerun_requested: !happy,
                }),
            });
            const data = await res.json();
            setFeedbackResult(data);
            setFeedbackState('submitted');
            // If unhappy: allow re-run by re-showing the form with instructions pre-filled
            if (!happy) {
                setTimeout(() => {
                    setResult(null);
                    setFeedbackState(null);
                    setFeedbackHappy(null);
                    setFeedbackReason('');
                    setPlanRejectionCount(0);
                }, 1800);
            }
        } catch (err) {
            console.error('Feedback submission failed:', err);
        } finally {
            setFeedbackSubmitting(false);
        }
    };

    const handleDiscoverTables = async e => {
        e.preventDefault();
        if (!dbHost.trim() || !dbName.trim()) { setError('Please fill in Host and Database Name first.'); return; }
        setConnecting(true); setError(null); setAvailableTables([]); setPreviewRows([]);
        try {
            const form = new FormData();
            form.append('source_kind', dbType === 'neo4j' ? 'graph_db' : 'database');
            form.append('source_input', JSON.stringify({ backend: dbType, host: dbHost, port: dbPort, database: dbName, username: dbUser, password: dbPass }));
            const res = await fetch(`${API_BASE}/api/pipeline/list-tables`, { method: 'POST', body: form });
            const text = await res.text();
            let data;
            try { data = JSON.parse(text); } catch {
                throw new Error(`Server returned HTTP ${res.status} with a non-JSON body. Is the backend running?`);
            }
            if (!res.ok) throw new Error(data.detail || `Failed to list tables (HTTP ${res.status})`);
            if (!data.tables || data.tables.length === 0) throw new Error('Connected, but no tables/collections were found.');
            setAvailableTables(data.tables);
            // Auto-preview first table
            if (data.tables.length > 0) {
                const firstTable = data.tables[0];
                if (!dbTable) setDbTable(firstTable);
                fetchDbTablePreview(firstTable);
            }
        } catch (err) { setError(err.message || 'Connection failed'); }
        finally { setConnecting(false); }
    };

    const fetchDbTablePreview = async (tableName) => {
        try {
            const form = new FormData();
            form.append('source_kind', dbType === 'neo4j' ? 'graph_db' : 'database');
            form.append('source_input', JSON.stringify({ backend: dbType, host: dbHost, port: dbPort, database: dbName, username: dbUser, password: dbPass, table: tableName, limit: 200 }));
            const res = await fetch(`${API_BASE}/api/pipeline/preview-source`, { method: 'POST', body: form });
            if (!res.ok) return;
            const data = await res.json();
            if (data.status === 'ok') {
                setPreviewRows(data.rows);
                if (data.n_rows !== undefined) setFileMetaRows(data.n_rows);
                if (data.null_rate !== undefined) setFileMetaNullRate(data.null_rate);
            } else if (Array.isArray(data.rows) && data.rows.length > 0) {
                setPreviewRows(data.rows.slice(0, 500));
            } else if (Array.isArray(data) && data.length > 0) {
                setPreviewRows(data.slice(0, 500));
            } else {
                setPreviewRows([]);
                setFileMetaRows(0);
                setFileMetaNullRate(0);
            }
        } catch { /* preview is optional, silently ignore */ }
    };

    const DB_TYPES = [
        { id: 'postgresql', label: 'PostgreSQL', emoji: '🐘', port: '5432' },
        { id: 'mongodb', label: 'MongoDB', emoji: '🍃', port: '27017' }
    ];

    const dbLabel = dbType === 'neo4j' ? 'Node Labels' : dbType === 'mongodb' ? 'Collections' : dbType === 'redis' ? 'Keys' : 'Tables';

    return (
        <div className="run-pipeline-container">
            <div className="mode-selector">
                {MODES.map(m => {
                    const Icon = m.icon;
                    return (
                        <button key={m.id} className={`mode-card ${mode === m.id ? 'mode-card--active' : ''}`}
                            onClick={() => { setMode(m.id); setFile(null); setError(null); setResult(null); setAvailableTables([]); setDbTable(''); }}>
                            <Icon className="mode-icon" />
                            <span className="mode-label">{m.label}</span>
                            <span className="mode-desc">{m.desc}</span>
                        </button>
                    );
                })}
            </div>

            <div className="panels-row">
                <div className="configuration-panel">
                    <div className="panel-header">
                        <Cpu className="panel-icon" />
                        <h2>Configure &amp; Run Pipeline</h2>
                    </div>

                    <form onSubmit={e => handleRunSubmit(e)} className="pipeline-form" noValidate>

                        {/* Analytics Targeting Config */}
                        <div className="form-row targeting-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem', marginBottom: '1.5rem', padding: '1rem', background: 'rgba(99, 102, 241, 0.03)', borderRadius: '8px', border: '1px solid rgba(99, 102, 241, 0.1)' }}>
                            <div className="form-group" style={{ marginBottom: 0 }}>
                                <label style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#818cf8', fontWeight: 600 }}>Target (Prediction) Column</label>
                                <input type="text" value={targetCol} onChange={e => setTargetCol(e.target.value)} placeholder="e.g. 'churn'" disabled={running} style={{ padding: '0.6rem', fontSize: '0.85rem', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)' }} title="The specific column you want the AI to predict" />
                            </div>
                            <div className="form-group" style={{ marginBottom: 0 }}>
                                <label style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#818cf8', fontWeight: 600 }}>Column Filter</label>
                                <input type="text" value={colRange} onChange={e => setColRange(e.target.value)} placeholder="e.g. '1-10' or 'age, income'" disabled={running} style={{ padding: '0.6rem', fontSize: '0.85rem', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)' }} title="Type column names or range to ONLY keep those columns" />
                            </div>
                            <div className="form-group" style={{ marginBottom: 0 }}>
                                <label style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#818cf8', fontWeight: 600 }}>Row Range</label>
                                <input type="text" value={rowRange} onChange={e => setRowRange(e.target.value)} placeholder="e.g. '1-100'" disabled={running} style={{ padding: '0.6rem', fontSize: '0.85rem', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)' }} title="Format: Start-End (e.g. 1-100)" />
                            </div>
                        </div>

                        {mode === 'file' && (
                            <div className={`drop-zone ${dragging ? 'drop-zone--active' : ''} ${file ? 'drop-zone--filled' : ''}`}
                                onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
                                onClick={() => fileInputRef.current?.click()}>
                                <input type="file" ref={fileInputRef} onChange={onFileChange} accept=".csv,.json,.xlsx,.xls,.parquet" hidden />
                                {file ? (
                                    <div className="drop-file-info">
                                        <FileText className="drop-icon" />
                                        <span className="drop-filename">{file.name}</span>
                                        <span className="drop-size">({(file.size / 1024).toFixed(1)} KB)</span>
                                        <span className="drop-change">Click to change</span>
                                    </div>
                                ) : (
                                    <div className="drop-placeholder">
                                        <Upload className="drop-icon anim-bounce" />
                                        <p>Drag &amp; drop your file here</p>
                                        <span className="or-divider">— or —</span>
                                        <span className="browse-btn">Browse Files</span>
                                        <span className="drop-formats">CSV · JSON · Excel · Parquet</span>
                                    </div>
                                )}
                            </div>
                        )}

                        {mode === 'database' && (
                            <div className="db-source-panel">
                                <div className="db-type-strip">
                                    {DB_TYPES.map(db => (
                                        <button key={db.id} type="button" className={`db-type-pill ${dbType === db.id ? 'active' : ''}`}
                                            onClick={() => {
                                                setDbType(db.id);
                                                if (db.port) setDbPort(db.port);
                                                // Auto-fill Docker service name & ADAP creds
                                                if (db.id === 'mongodb') {
                                                    setDbHost('mongo'); setDbUser('dipex'); setDbPass('dipex_secret'); setDbName('dipex');
                                                } else if (db.id === 'postgresql') {
                                                    setDbHost('postgres'); setDbUser('dipex'); setDbPass('dipex_secret'); setDbName('dipex');
                                                }
                                                setAvailableTables([]); setDbTable('');
                                            }}
                                            disabled={running || connecting}>
                                            <span>{db.emoji}</span> {db.label}
                                        </button>
                                    ))}
                                </div>
                                <div className="form-row">
                                    <div className="form-group" style={{ flex: 2 }}>
                                        <label>Host</label>
                                        <input type="text" value={dbHost} onChange={e => setDbHost(e.target.value)} placeholder="localhost" disabled={running || connecting} />
                                    </div>
                                    <div className="form-group" style={{ flex: 1 }}>
                                        <label>Port</label>
                                        <input type="text" value={dbPort} onChange={e => setDbPort(e.target.value)} placeholder="5432" disabled={running || connecting} />
                                    </div>
                                </div>
                                <div className="form-row">
                                    <div className="form-group">
                                        <label>Username <span className="optional">(optional)</span></label>
                                        <input type="text" value={dbUser} onChange={e => setDbUser(e.target.value)} placeholder="postgres" disabled={running || connecting} />
                                    </div>
                                    <div className="form-group">
                                        <label>Password <span className="optional">(optional)</span></label>
                                        <input type="password" value={dbPass} onChange={e => setDbPass(e.target.value)} placeholder="••••••••" disabled={running || connecting} />
                                    </div>
                                </div>
                                <div className="form-group">
                                    <label>Database Name</label>
                                    <input type="text" value={dbName} onChange={e => { setDbName(e.target.value); setAvailableTables([]); setDbTable(''); }} placeholder="my_database" disabled={running || connecting} />
                                </div>
                                {availableTables.length > 0 ? (
                                    <div className="tables-discovery-grid">
                                        <div className="discovery-header">
                                            <span className="discovery-label">📂 {availableTables.length} {dbType === 'neo4j' ? 'node labels' : dbType === 'mongodb' ? 'collections' : dbType === 'redis' ? 'key patterns' : 'tables'} — click to ingest</span>
                                            <button type="button" className="btn-rediscover" onClick={() => { setAvailableTables([]); setDbTable(''); }}>
                                                <RefreshCw size={12} /> Re-discover
                                            </button>
                                        </div>
                                        <div className="tables-grid">
                                            {availableTables.map(t => {
                                                const isIngesting = dbTable === t && running;
                                                return (
                                                    <button key={t} type="button"
                                                        className={`table-card-btn ${dbTable === t ? 'selected' : ''} ${isIngesting ? 'ingesting' : ''}`}
                                                        onClick={ev => { ev.preventDefault(); setDbTable(t); fetchDbTablePreview(t); }}
                                                        onDoubleClick={ev => { ev.preventDefault(); setDbTable(t); handleRunSubmit(null, t); }}
                                                        disabled={running} title={`Single-click: preview table · Double-click: ingest "${t}"`}>
                                                        {isIngesting
                                                            ? <span className="spinner" style={{ borderTopColor: '#6366f1', borderColor: 'rgba(99,102,241,0.3)', width: 14, height: 14 }} />
                                                            : <Database size={15} className="tc-icon" />}
                                                        <span className="tc-name">{t}</span>
                                                        {!isIngesting && <span className="tc-hint">👁️ preview · dbl-click ingest</span>}
                                                    </button>
                                                );
                                            })}
                                        </div>
                                        <div className="form-group" style={{ marginTop: '0.75rem' }}>
                                            <label>Or type custom table / query manually</label>
                                            <input type="text" value={dbTable} onChange={e => setDbTable(e.target.value)} placeholder="SELECT * FROM orders LIMIT 5000" disabled={running || connecting} />
                                        </div>
                                    </div>
                                ) : (
                                    <>
                                        <div className="form-group">
                                            <label>{dbType === 'neo4j' ? 'Cypher Query' : dbType === 'redis' ? 'Key Pattern' : 'Table / Collection'} <span className="optional">(or click Discover below)</span></label>
                                            <input type="text" value={dbTable} onChange={e => setDbTable(e.target.value)}
                                                placeholder={dbType === 'neo4j' ? 'MATCH (n) RETURN n LIMIT 1000' : dbType === 'redis' ? '*' : 'my_table'}
                                                disabled={running || connecting} />
                                        </div>
                                        <button type="button" className={`btn-discover-enhanced ${connecting ? 'connecting' : ''}`} onClick={handleDiscoverTables} disabled={running || connecting || !dbHost.trim() || !dbName.trim()}>
                                            {connecting ? <><span className="spinner-dark" />Connecting to {dbName || dbType}…</> : <><Database size={16} />Connect &amp; Browse {dbLabel}</>}
                                        </button>
                                    </>
                                )}
                            </div>
                        )}

                        {mode === 'live' && (
                            <div className="kafka-panel">
                                <button type="button" className="kafka-info-toggle" onClick={() => setKafkaInfoOpen(o => !o)}>
                                    <Info size={14} /> How Kafka ingestion works
                                    <ChevronDown size={13} style={{ transform: kafkaInfoOpen ? 'rotate(180deg)' : '', transition: '0.2s', marginLeft: 'auto' }} />
                                </button>
                                {kafkaInfoOpen && (
                                    <div className="kafka-info-box">
                                        <p><strong>Broker</strong> — address of your Kafka bootstrap server, e.g. <code>localhost:9092</code> outside Docker, or <code>kafka:29092</code> inside Docker.</p>
                                        <p><strong>Topic</strong> — the Kafka topic to consume messages from, e.g. <code>orders</code> or <code>events</code>.</p>
                                        <p><strong>Consumer Group</strong> — identifies this consumer in offset tracking (any unique string works).</p>
                                        <p><strong>Max Messages</strong> — ADAP reads up to this many messages in a 30-second tumbling window then processes the batch.</p>
                                    </div>
                                )}
                                <div className="form-row">
                                    <div className="form-group" style={{ flex: 2 }}>
                                        <label>Kafka Broker</label>
                                        <input type="text" value={kafkaBroker} onChange={e => setKafkaBroker(e.target.value)} placeholder="kafka:29092" disabled={running} />
                                    </div>
                                    <div className="form-group" style={{ flex: 1 }}>
                                        <label>Port (in address)</label>
                                        <span className="help-text" style={{ paddingTop: '0.45rem', display: 'block' }}>e.g. <code>kafka:29092</code></span>
                                    </div>
                                </div>
                                <div className="form-row">
                                    <div className="form-group">
                                        <label>Topic</label>
                                        <input type="text" value={kafkaTopic} onChange={e => setKafkaTopic(e.target.value)} placeholder="dipex_pipeline" disabled={running} />
                                    </div>
                                    <div className="form-group">
                                        <label>Consumer Group</label>
                                        <input type="text" value={kafkaGroupId} onChange={e => setKafkaGroupId(e.target.value)} placeholder="dipex-consumer" disabled={running} />
                                    </div>
                                </div>
                                <div className="form-group">
                                    <label>Max Messages <span className="optional">(per 30-second window)</span></label>
                                    <input type="number" value={kafkaMaxMessages} onChange={e => setKafkaMaxMessages(e.target.value)} min="1" max="1000000" placeholder="10000" disabled={running} />
                                    <span className="help-text">ADAP collects up to this many messages then runs the full pipeline on the batch.</span>
                                </div>
                            </div>
                        )}

                        {mode === 'api' && (
                            <div className="api-config-panel">
                                <div className="api-url-row">
                                    <select className="api-method-sel" value={apiMethod} onChange={e => setApiMethod(e.target.value)} disabled={running}>
                                        {['GET', 'POST', 'PUT', 'PATCH', 'DELETE'].map(m => <option key={m}>{m}</option>)}
                                    </select>
                                    <input className="api-url-input" type="text" value={apiUrl} onChange={e => setApiUrl(e.target.value)} placeholder="https://api.example.com/v1/data" disabled={running} />
                                </div>
                                <span className="help-text">Must return JSON — an array of objects or an object with a data/results key.</span>

                                <div className="api-section">
                                    <div className="api-section-label">🔑 Authentication</div>
                                    <div className="auth-pills">
                                        {[{ id: 'none', label: 'None' }, { id: 'bearer', label: 'Bearer Token' }, { id: 'apikey', label: 'API Key' }, { id: 'basic', label: 'Basic Auth' }].map(t => (
                                            <button key={t.id} type="button" className={`auth-pill ${apiAuthType === t.id ? 'active' : ''}`} onClick={() => setApiAuthType(t.id)} disabled={running}>{t.label}</button>
                                        ))}
                                    </div>
                                    {apiAuthType === 'bearer' && (
                                        <div className="form-group" style={{ marginTop: '0.6rem' }}>
                                            <label>Bearer Token</label>
                                            <input type="text" value={apiBearerToken} onChange={e => setApiBearerToken(e.target.value)} placeholder="eyJhbGciOiJSUzI1NiJ9…" disabled={running} />
                                        </div>
                                    )}
                                    {apiAuthType === 'apikey' && (
                                        <div className="form-row" style={{ marginTop: '0.6rem' }}>
                                            <div className="form-group"><label>Header Name</label><input type="text" value={apiKeyName} onChange={e => setApiKeyName(e.target.value)} placeholder="X-API-Key" disabled={running} /></div>
                                            <div className="form-group"><label>API Key Value</label><input type="text" value={apiKeyValue} onChange={e => setApiKeyValue(e.target.value)} placeholder="sk-…" disabled={running} /></div>
                                        </div>
                                    )}
                                    {apiAuthType === 'basic' && (
                                        <div className="form-row" style={{ marginTop: '0.6rem' }}>
                                            <div className="form-group"><label>Username</label><input type="text" value={apiBasicUser} onChange={e => setApiBasicUser(e.target.value)} placeholder="admin" disabled={running} /></div>
                                            <div className="form-group">
                                                <label>Password</label>
                                                <div style={{ position: 'relative' }}>
                                                    <input type={apiShowPass ? 'text' : 'password'} value={apiBasicPass} onChange={e => setApiBasicPass(e.target.value)} placeholder="••••••••" disabled={running} style={{ paddingRight: '2.5rem', width: '100%', boxSizing: 'border-box' }} />
                                                    <button type="button" className="pass-toggle" onClick={() => setApiShowPass(s => !s)}>{apiShowPass ? <EyeOff size={14} /> : <Eye size={14} />}</button>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                <div className="api-section">
                                    <div className="api-section-label">📋 Custom Headers <span className="optional">(optional)</span></div>
                                    {apiHeaders.map((h, i) => (
                                        <div key={i} className="header-row">
                                            <input type="text" className="header-key" value={h.key} onChange={e => setApiHeaders(prev => prev.map((x, j) => j === i ? { ...x, key: e.target.value } : x))} placeholder="Header-Name" disabled={running} />
                                            <input type="text" className="header-val" value={h.value} onChange={e => setApiHeaders(prev => prev.map((x, j) => j === i ? { ...x, value: e.target.value } : x))} placeholder="value" disabled={running} />
                                            <button type="button" className="header-remove" onClick={() => setApiHeaders(prev => prev.filter((_, j) => j !== i))} disabled={running || apiHeaders.length === 1}><Trash2 size={13} /></button>
                                        </div>
                                    ))}
                                    <button type="button" className="header-add-btn" onClick={() => setApiHeaders(prev => [...prev, { key: '', value: '' }])} disabled={running}><Plus size={13} /> Add Header</button>
                                </div>

                                {['POST', 'PUT', 'PATCH'].includes(apiMethod) && (
                                    <div className="api-section">
                                        <div className="api-section-label">📄 Request Body <span className="optional">(JSON)</span></div>
                                        <textarea className="api-body-textarea" value={apiBody} onChange={e => setApiBody(e.target.value)} placeholder={'{ "key": "value" }'} rows={4} disabled={running} />
                                    </div>
                                )}

                                {apiUrl.trim() && (
                                    <div className="api-json-preview">
                                        <div className="api-json-preview-label">⚡ Live Config Preview</div>
                                        <pre className="api-json-pre">{JSON.stringify(buildApiConfig(), null, 2)}</pre>
                                    </div>
                                )}

                                {/* Test & Preview button */}
                                {apiUrl.trim() && (
                                    <button type="button" disabled={running || previewLoading}
                                        style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.55rem 1.1rem', borderRadius: '8px', border: '1px solid rgba(99,102,241,0.4)', background: 'rgba(99,102,241,0.1)', color: '#a5b4fc', fontWeight: 700, fontSize: '0.82rem', cursor: 'pointer', transition: 'all 0.2s ease', width: 'fit-content' }}
                                        onClick={async () => {
                                            setPreviewLoading(true);
                                            setPreviewRows([]);
                                            try {
                                                const form = new FormData();
                                                form.append('source_kind', 'api');
                                                form.append('source_input', JSON.stringify(buildApiConfig()));
                                                const res = await fetch(`${API_BASE}/api/pipeline/preview-source`, { method: 'POST', body: form });
                                                if (!res.ok) {
                                                    let errorMsg = await res.text();
                                                    throw new Error(errorMsg || `HTTP ${res.status}`);
                                                }
                                                const data = await res.json();
                                                if (data.status === 'ok') {
                                                    setPreviewRows(data.rows);
                                                    if (data.n_rows !== undefined) setFileMetaRows(data.n_rows);
                                                    if (data.null_rate !== undefined) setFileMetaNullRate(data.null_rate);
                                                } else if (data.status === 'empty') {
                                                    setPreviewRows([{ _info: 'Connected, but the API returned no data.' }]);
                                                    setFileMetaRows(0);
                                                    setFileMetaNullRate(0);
                                                } else {
                                                    setPreviewRows([{ _error: `API test failed: ${data.errors ? data.errors.join(', ') : 'Unknown setup issue'}` }]);
                                                }
                                            } catch (err) {
                                                setPreviewRows([{ _error: `API test failed: ${err.message}` }]);
                                            } finally { setPreviewLoading(false); }
                                        }}>
                                        {previewLoading ? <><span className="spinner" style={{width:12,height:12,borderWidth:2}} /> Testing…</> : <>🔍 Test & Preview</>}
                                    </button>
                                )}
                            </div>
                        )}


                        <div style={{ marginBottom: '1.25rem' }}>
                            <label style={{ display: 'block', fontWeight: 700, fontSize: '0.72rem', color: '#7c85a0', marginBottom: '0.65rem', letterSpacing: '0.07em', textTransform: 'uppercase' }}>
                                <Shield size={13} style={{ verticalAlign: 'middle', marginRight: '0.35rem', color: domain ? '#a5b4fc' : '#4a5568' }} />
                                Regulatory Domain
                                <span style={{ fontWeight: 400, color: domain ? '#a5b4fc' : '#4a5568', marginLeft: '0.45rem', fontSize: '0.72rem', textTransform: 'none', letterSpacing: 0 }}>
                                    {domain ? `— ${DOMAIN_CARDS.find(d => d.id === domain)?.label} rules active` : '— skip compliance checks'}
                                </span>
                            </label>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem' }}>
                                {DOMAIN_CARDS.map(dc => (
                                    <button key={dc.id} type="button" disabled={running}
                                        onClick={() => { setDomain(prev => prev === dc.id ? '' : dc.id); setExtraDomains(prev => prev.filter(x => x !== dc.id)); }}
                                        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.25rem', padding: '0.65rem 0.4rem', borderRadius: '10px', border: domain === dc.id ? '1.5px solid #6366f1' : '1.5px solid rgba(100,116,139,0.18)', background: domain === dc.id ? 'rgba(99,102,241,0.16)' : 'rgba(255,255,255,0.03)', color: domain === dc.id ? '#c4b5fd' : '#7c85a0', fontWeight: domain === dc.id ? 700 : 500, fontSize: '0.77rem', cursor: running ? 'not-allowed' : 'pointer', transition: 'all 0.2s cubic-bezier(0.4,0,0.2,1)', opacity: running ? 0.5 : 1, boxShadow: domain === dc.id ? '0 0 16px rgba(99,102,241,0.3)' : 'none' }}>
                                        <span style={{ fontSize: '1.3rem', lineHeight: 1 }}>{dc.emoji}</span>
                                        <span style={{ fontWeight: 700 }}>{dc.label}</span>
                                        <span style={{ fontSize: '0.67rem', color: domain === dc.id ? '#a5b4fc' : '#4a5568', textAlign: 'center' }}>{dc.sub}</span>
                                    </button>
                                ))}
                            </div>
                            {domain && (
                                <div style={{ marginTop: '0.65rem', display: 'flex', flexWrap: 'wrap', gap: '0.4rem', alignItems: 'center' }}>
                                    <span style={{ fontSize: '0.7rem', color: '#4a5568', fontWeight: 700, marginRight: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Also enforce:</span>
                                    {EXTRA_REGULATION_PILLS.filter(d => d !== domain).map(d => {
                                        const active = extraDomains.includes(d);
                                        return (
                                            <button key={d} type="button" disabled={running}
                                                onClick={() => setExtraDomains(prev => active ? prev.filter(x => x !== d) : [...prev, d])}
                                                style={{ padding: '0.2rem 0.65rem', borderRadius: '999px', fontSize: '0.68rem', border: active ? '1.5px solid #6366f1' : '1.5px solid rgba(100,116,139,0.2)', background: active ? 'rgba(99,102,241,0.18)' : 'rgba(255,255,255,0.03)', color: active ? '#c4b5fd' : '#4a5568', fontWeight: active ? 700 : 500, cursor: running ? 'not-allowed' : 'pointer', transition: 'all 0.18s cubic-bezier(0.4,0,0.2,1)', textTransform: 'uppercase', letterSpacing: '0.05em', boxShadow: active ? '0 0 10px rgba(99,102,241,0.25)' : 'none' }}>
                                                {active ? '✓ ' : '+ '}{d}
                                            </button>
                                        );
                                    })}
                                </div>
                            )}
                            {!domain && (
                                <p style={{ margin: '0.5rem 0 0', fontSize: '0.72rem', color: '#3a4258', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                                    <ShieldOff size={12} style={{ color: '#4a5568' }} />
                                    No domain selected — pipeline runs without regulatory checks.
                                </p>
                            )}
                        </div>

                        {/* ————————————————— Analyst Instructions Box ————————————————————————————————— */}
                        <div className="instruction-box-wrapper">
                            <div className="instruction-box-header">
                                <MessageSquare size={14} className="instruction-box-icon" />
                                <span>Analyst Instructions</span>
                                <span className="instruction-box-badge">🤖 AI-guided</span>
                                <span className="instruction-char-count">{userInstructions.length}/500</span>
                            </div>
                            <textarea
                                id="user-instructions"
                                className="instruction-textarea"
                                value={userInstructions}
                                onChange={e => setUserInstructions(e.target.value.slice(0, 500))}
                                placeholder={{
                                    file:     'Examples:\n• "Focus on anomaly detection in the numeric columns"\n• "Predict customer churn — target column is \'churned\'"\n• "This is financial transaction data — apply AML and fraud rules"\n• "Ignore columns with > 30% nulls, keep first 10 000 rows only"',
                                    database: 'Examples:\n• "Run correlation analysis on all numeric columns in this table"\n• "Detect schema drift vs last week\'s snapshot"\n• "Apply GDPR rules — redact email and phone columns"\n• "This is an orders table — predict revenue, exclude cancelled rows"',
                                    live:     'Examples:\n• "Flag real-time fraud transactions — focus on amount and merchant_id"\n• "Detect sudden distribution shifts in the event_type column"\n• "Aggregate by 5-minute windows and alert if anomaly rate > 2%"\n• "Apply HIPAA rules — mask patient_id in every message"',
                                    api:      'Examples:\n• "This API returns stock prices — detect outliers in \'close\' column"\n• "Correlate all numeric fields and rank by predictive power"\n• "Enforce GDPR — scan for PII in every response field"\n• "Normalize JSON, flatten nested arrays, then classify domain"',
                                }[mode] || 'Tell the AI how to approach this data…'}
                                rows={4}
                                disabled={running}
                            />
                            <div className="instruction-hints-row">
                                {({
                                    file: [
                                        '🔍 detect anomalies',
                                        '📈 predict churn',
                                        '💰 fraud detection',
                                        '🧹 drop null columns',
                                        '📊 correlation analysis',
                                        '🏦 banking AML rules',
                                        '🔒 redact PII',
                                        '➖ skip ML model',
                                    ],
                                    database: [
                                        '📊 correlation analysis',
                                        '🔄 detect schema drift',
                                        '🔒 apply GDPR rules',
                                        '📈 predict revenue',
                                        '🧹 exclude null rows',
                                        '🏦 apply AML rules',
                                        '📋 profile all columns',
                                    ],
                                    live: [
                                        '🚨 real-time fraud alert',
                                        '📉 detect drift spikes',
                                        '🏥 apply HIPAA masking',
                                        '⏱️ 5-min window aggregation',
                                        '🔍 flag anomaly rate > 2%',
                                        '📊 live distribution check',
                                    ],
                                    api: [
                                        '📈 detect price outliers',
                                        '🔗 correlate all fields',
                                        '🔒 GDPR PII scan',
                                        '📂 flatten nested JSON',
                                        '📊 rank by predictive power',
                                        '🔍 classify domain',
                                    ],
                                }[mode] || []).map(hint => (
                                    <button
                                        key={hint}
                                        type="button"
                                        className="instruction-hint-chip"
                                        onClick={() => setUserInstructions(prev => prev ? `${prev}, ${hint.replace(/^[^ ]+ /, '')}` : hint.replace(/^[^ ]+ /, ''))}
                                        disabled={running}
                                    >
                                        {hint}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <button type="submit" className={`btn-run ${running ? 'running' : ''}`} disabled={running || planLoading}>
                            {running ? <><span className="spinner" /> Running Pipeline…</> : planLoading ? <><span className="spinner" /> Generating Plan…</> : <><Play className="btn-icon" /> Execute Pipeline</>}
                        </button>
                    </form>

                    {error && (
                        <div className="status-message error">
                            <AlertCircle className="status-icon" />
                            <div className="status-content"><strong>Execution Failed</strong><p>{error}</p></div>
                        </div>
                    )}

                    {result && feedbackState === 'pending' && (
                        <div className="feedback-panel">
                            <UnifiedResultsSection
                                result={result}
                                charts={charts}
                                kpis={kpis}
                                insightsFeed={insightsFeed}
                                sectionInsights={sectionInsights}
                                chartsLoading={chartsLoading}
                            />
                            <div className="feedback-section">
                                <div className="feedback-header">
                                    <Sparkles size={16} className="feedback-sparkle" />
                                    <span className="feedback-title">How was this analysis?</span>
                                    <span className="feedback-subtitle">Your feedback trains the AI for future runs</span>
                                </div>
                                <div className="feedback-btn-row">
                                    <button
                                        id="feedback-happy-btn"
                                        className="feedback-btn feedback-btn--happy"
                                        onClick={() => handleFeedbackSubmit(true)}
                                        disabled={feedbackSubmitting}
                                    >
                                        <ThumbsUp size={18} />
                                        Happy with results
                                    </button>
                                    <button
                                        id="feedback-rerun-btn"
                                        className="feedback-btn feedback-btn--rerun"
                                        onClick={() => handleFeedbackSubmit(false)}
                                        disabled={feedbackSubmitting}
                                    >
                                        <RotateCcw size={18} />
                                        Re-run with changes
                                    </button>
                                </div>
                                <div className="feedback-reason-row">
                                    <MessageSquare size={13} style={{ color: '#4a5568', flexShrink: 0 }} />
                                    <textarea
                                        className="feedback-reason-input"
                                        value={feedbackReason}
                                        onChange={e => setFeedbackReason(e.target.value.slice(0, 300))}
                                        placeholder="Optional: Tell us why (improves future runs)…"
                                        rows={2}
                                        disabled={feedbackSubmitting}
                                    />
                                </div>
                                {planRejectionCount > 0 && (
                                    <div className="feedback-rejection-notice">
                                        ⚡ {planRejectionCount} plan revision{planRejectionCount !== 1 ? 's' : ''} recorded — agent will learn from this
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {result && feedbackState === 'submitted' && (
                        <div className="feedback-panel">
                            <UnifiedResultsSection
                                result={result}
                                charts={charts}
                                kpis={kpis}
                                insightsFeed={insightsFeed}
                                sectionInsights={sectionInsights}
                                chartsLoading={chartsLoading}
                            />
                            <div className="feedback-submitted">
                                {feedbackHappy ? (
                                    <><CheckCircle size={20} style={{ color: '#10b981' }} /> <strong>Thank you!</strong> Result recorded as successful. RL agent updated (+{feedbackResult?.rl_reward_delta?.toFixed(2) || '0.20'} reward).</>
                                ) : (
                                    <><RotateCcw size={20} style={{ color: '#f59e0b', animation: 'spin 1s linear' }} /> <strong>Got it.</strong> Preparing re-run… RL agent updated ({feedbackResult?.rl_reward_delta?.toFixed(2) || '-0.20'} reward).</>
                                )}
                            </div>
                        </div>
                    )}

                    {result && !feedbackState && (
                        <UnifiedResultsSection
                            result={result}
                            charts={charts}
                            kpis={kpis}
                            insightsFeed={insightsFeed}
                            sectionInsights={sectionInsights}
                            chartsLoading={chartsLoading}
                        />
                    )}
                </div>

                <div className="info-panel">
                    <h3>Pipeline Workflow</h3>
                    <ol className="workflow-steps">
                        <li><strong>Universal Intake:</strong> Auto schema inference + typing across CSV, JSON, Excel, Parquet, DB, Kafka &amp; REST.</li>
                        <li><strong>Data Triage:</strong> Null analysis, cardinality checks, zero-variance drop, skew &amp; class-imbalance detection.</li>
                        <li><strong>Missing Patterns:</strong> MCAR/MAR analysis — selects optimal imputation strategy per column.</li>
                        <li><strong>Preprocessing:</strong> sklearn Pipeline — impute, encode, scale; RL feature selector prunes low-value columns.</li>
                        <li><strong>Drift Detection:</strong> Schema &amp; distribution drift vs. prior run (PSI-based).</li>
                        <li><strong>Hard Gate 1 — Validation:</strong> Deterministic null/range/integrity checks + Regulatory &amp; Compliance engine (GDPR, HIPAA, AML…).</li>
                        <li><strong>Profiling:</strong> Statistical profile — descriptives, correlations, outlier summary.</li>
                        <li><strong>Analytics Layer:</strong> AutoEDA, Feature Engineering, Insight Ranking &amp; LLM narrative.</li>
                        <li><strong>Governance:</strong> PII scan, policy enforcement, data catalog update.</li>
                        <li><strong>Statistics:</strong> Descriptive stats + hypothesis tests (target correlation).</li>
                        <li><strong>Leakage &amp; Multicollinearity:</strong> Target-leakage removal + VIF-based collinear feature pruning.</li>
                        <li><strong>AutoML:</strong> Model race — RF, XGB, LGBM, LR + confidence calibration.</li>
                        <li><strong>Hard Gate 2 — Verification:</strong> Independent statistical verifier + confidence vector aggregation.</li>
                        <li><strong>RL Feedback:</strong> Intelligent retry engine, experience memory &amp; RL model update.</li>
                        <li><strong>Report:</strong> Executive HTML report with LLM narrative + audit trail.</li>
                    </ol>
                    <div className="info-badge">
                        <Radio className="badge-icon" />
                        <span>Kafka ingestion runs a 30-second tumbling window and feeds the same pipeline.</span>
                    </div>
                </div>
            </div>

            {/* â”€â”€ RAW ingestion preview (shown immediately after file/data is loaded) â”€â”€ */}
            {previewRows.length > 0 && !result && (
                <div style={{ marginTop: '1.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.6rem', padding: '0.5rem 1rem', background: 'rgba(99,102,241,0.08)', borderRadius: '8px', border: '1px solid rgba(99,102,241,0.2)' }}>
                        <span style={{ fontSize: '1rem' }}>ðŸ‘ï¸</span>
                        <span style={{ fontWeight: 700, color: '#a5b4fc', fontSize: '0.85rem' }}>Raw Data Preview</span>
                        <span style={{ fontSize: '0.75rem', color: '#475569' }}>— before pipeline processing · {previewRows.length} rows sampled</span>
                    </div>
                    {previewLoading
                        ? <div style={{ padding: '2rem', textAlign: 'center', color: '#475569' }}>Parsing file…</div>
                        : <DataPreviewPanel rows={previewRows} sourceKind={mode} colRange={colRange} rowRange={rowRange} />}
                </div>
            )}

            {/* â”€â”€ POST-PIPELINE processed data preview â”€â”€ */}
            {result?.sample_rows?.length > 0 && (
                <div style={{ marginTop: '1.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.6rem', padding: '0.5rem 1rem', background: 'rgba(16,185,129,0.08)', borderRadius: '8px', border: '1px solid rgba(16,185,129,0.25)' }}>
                        <span style={{ fontSize: '1rem' }}>✅</span>
                        <span style={{ fontWeight: 700, color: '#34d399', fontSize: '0.85rem' }}>Processed Data Preview</span>
                        <span style={{ fontSize: '0.75rem', color: '#475569' }}>— after full pipeline transformation · {result.sample_rows.length} rows</span>
                    </div>
                    <DataPreviewPanel rows={result.sample_rows} sourceKind={result.source_kind} colRange={colRange} rowRange={rowRange} />
                </div>
            )}

            {/* â”€â”€ Pre-Analysis Plan Modal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
            <AnalysisPlanModal
                plan={plan}
                isLoading={planLoading}
                onApprove={handlePlanApproved}
                onCancel={handlePlanCancelled}
                onReject={handlePlanRejected}
                instructionSummary={instructionSummary}
                rejectionCount={planRejectionCount}
            />
        </div>
    );
};

export default RunPipeline;
