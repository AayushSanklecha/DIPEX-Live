// v2 - cache bust
import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import { Upload, Database, Radio, Globe, Play, CheckCircle, AlertCircle, FileText, Cpu, Filter, Columns, X, ChevronDown, ChevronUp, SlidersHorizontal, Table2, Shield, ShieldOff, Info, Plus, Trash2, Eye, EyeOff, RefreshCw } from 'lucide-react';
import { analyzeSchema, computeStats, PALETTE } from '../utils/dataAnalyzer';
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

const DataPreviewPanel = ({ rows, sourceKind }) => {
    const schema = useMemo(() => analyzeSchema(rows), [rows]);
    const [selectedCols, setSelectedCols] = useState(null);
    const [colOpen, setColOpen] = useState(false);
    const prevSchemaKeyRef = useRef('');
    useEffect(() => {
        const key = schema.columns.map(c => c.key).join(',');
        if (key !== prevSchemaKeyRef.current) { prevSchemaKeyRef.current = key; setSelectedCols(new Set(schema.columns.map(c => c.key))); }
    }, [schema]);
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
    const filteredRows = useMemo(() => applyFilters(rows, filters), [rows, filters]);
    const displayRows = useMemo(() => projectCols(filteredRows, selectedCols), [filteredRows, selectedCols]);
    const pageRows = displayRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
    const totalPages = Math.ceil(displayRows.length / PAGE_SIZE);
    const displayCols = (selectedCols && selectedCols.size > 0) ? [...selectedCols] : schema.columns.map(c => c.key);
    const stats = useMemo(() => {
        const ps = analyzeSchema(displayRows.slice(0, 200));
        return computeStats(displayRows.slice(0, 200), ps.numericCols.slice(0, 5));
    }, [displayRows]);

    return (
        <div className="preview-panel">
            <div className="preview-header">
                <Table2 size={16} className="preview-header-icon" />
                <h3>Data Preview — <span className="preview-src">{SOURCE_LABELS[sourceKind] ?? sourceKind}</span></h3>
                <span className="preview-meta">{filteredRows.length.toLocaleString()} / {rows.length.toLocaleString()} rows &middot; {displayCols.length} / {schema.columns.length} cols</span>
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

// ── PII type metadata (icon + colour) ────────────────────────────────────────
const PII_META = {
    Email: { icon: '✉️', color: '#818cf8', label: 'Email' },
    SSN: { icon: '🔢', color: '#f87171', label: 'SSN' },
    CreditCard: { icon: '💳', color: '#fb923c', label: 'Credit Card' },
    Phone: { icon: '📞', color: '#34d399', label: 'Phone' },
    IPAddress: { icon: '🌐', color: '#60a5fa', label: 'IP Address' },
    ICD10: { icon: '🏥', color: '#a78bfa', label: 'ICD-10 Code' },
    IBAN: { icon: '🏦', color: '#fbbf24', label: 'IBAN' },
    Swift: { icon: '💱', color: '#2dd4bf', label: 'SWIFT/BIC' },
};

const POLICY_STYLES = {
    redacted: { bg: 'rgba(99,102,241,0.12)', border: '#6366f1', color: '#a5b4fc', icon: '🛡️', text: 'Redacted' },
    rejected: { bg: 'rgba(239,68,68,0.12)', border: '#ef4444', color: '#fca5a5', icon: '🚫', text: 'Rejected' },
    flagged: { bg: 'rgba(251,191,36,0.10)', border: '#f59e0b', color: '#fde68a', icon: '⚠️', text: 'Flagged' },
    passed: { bg: 'rgba(52,211,153,0.10)', border: '#10b981', color: '#6ee7b7', icon: '✅', text: 'Clean' },
    skipped: { bg: 'rgba(100,116,139,0.10)', border: '#475569', color: '#94a3b8', icon: '⏭️', text: 'Not Checked' },
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
                            const meta = PII_META[piiType] || { icon: '⚠️', color: '#94a3b8', label: piiType };
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
                <span className="gov-summary-icon">⚖️</span>
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

const ResultPanel = ({ result }) => (
    <div className="result-block">
        <div className="result-header">
            <CheckCircle className="result-icon success-icon" />
            <h3>Pipeline Complete</h3>
        </div>
        <div className="result-grid">
            <div className="result-item"><span className="result-label">Run ID</span><span className="result-value mono">{result.run_id}</span></div>
            <div className="result-item">
                <span className="result-label">Gate Decision</span>
                <span className={`badge badge-${result.final_result?.gate_decision || 'UNKNOWN'}`}>{result.final_result?.gate_decision || '—'}</span>
            </div>
            <div className="result-item">
                <span className="result-label">Quality Score</span>
                <span className="result-value">{result.final_result?.quality_score != null ? `${(result.final_result.quality_score * 100).toFixed(1)}%` : '—'}</span>
            </div>
            <div className="result-item"><span className="result-label">Source</span><span className="result-value">{SOURCE_LABELS[result.source_kind] ?? result.source_kind}</span></div>
            <div className="result-item"><span className="result-label">Dataset</span><span className="result-value">{result.dataset_id}</span></div>
            <div className="result-item"><span className="result-label">Rows ingested</span><span className="result-value">{result.row_count?.toLocaleString() ?? result.sample_rows?.length ?? '—'}</span></div>
        </div>

        <GovernanceSummary govReport={result.final_result?.governance_report} />
        <RegulatorySummary regulatoryReport={result.final_result?.regulatory_report} />

        {result.final_result?.report_path && <p className="report-hint">📄 Report saved to: <code>{result.final_result.report_path}</code></p>}
        <div className="result-actions">
            <a href="/" className="view-reports-link primary">View Analytics Dashboard →</a>
            <a href="/reports" className="view-reports-link">View Report →</a>
        </div>
    </div>
);

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

    const onDrop = useCallback((e) => { e.preventDefault(); setDragging(false); const d = e.dataTransfer.files[0]; if (d) setFile(d); }, []);
    const onDragOver = e => { e.preventDefault(); setDragging(true); };
    const onDragLeave = () => setDragging(false);
    const onFileChange = e => { if (e.target.files[0]) setFile(e.target.files[0]); };

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

    const handleRunSubmit = async (e, forcedTable = null) => {
        if (e) e.preventDefault();
        setRunning(true); setError(null); setResult(null);
        try {
            const form = new FormData();
            form.append('source_kind', mode);
            if (targetCol) form.append('target_col', targetCol.trim());
            if (colRange) form.append('col_range', colRange.trim());
            if (rowRange) form.append('row_range', rowRange.trim().replace(/\s+/g, ''));
            if (domain) form.append('domain', domain);
            if (extraDomains.length) form.append('extra_domains', extraDomains.join(','));

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

            const res = await fetch(`${API_BASE}/api/pipeline/simple-run`, { method: 'POST', body: form });
            const text = await res.text();
            let data;
            try { data = JSON.parse(text); } catch {
                // Server returned non-JSON (HTML error page, 413, Nginx 502, etc.)
                if (res.status === 413) throw new Error(`File too large — the server rejected the upload (413). Try a smaller file or increase the server upload limit.`);
                throw new Error(`Server returned HTTP ${res.status} with a non-JSON body. Make sure the DIPEX backend is running on port 8000.`);
            }
            if (!res.ok) throw new Error(data.detail || `Pipeline run failed (HTTP ${res.status})`);
            setResult(data);
        } catch (err) {
            const msg = err.message || 'Unexpected error';
            if (msg === 'Failed to fetch' || msg.includes('NetworkError') || msg.includes('ECONNREFUSED')) {
                setError('Cannot reach the API server. Make sure the DIPEX backend is running on port 8000.');
            } else { setError(msg); }
        } finally { setRunning(false); }
    };

    const handleDiscoverTables = async e => {
        e.preventDefault();
        if (!dbHost.trim() || !dbName.trim()) { setError('Please fill in Host and Database Name first.'); return; }
        setConnecting(true); setError(null); setAvailableTables([]);
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
        } catch (err) { setError(err.message || 'Connection failed'); }
        finally { setConnecting(false); }
    };

    const DB_TYPES = [
        { id: 'postgresql', label: 'PostgreSQL', emoji: '🐘', port: '5432' },
        { id: 'mysql', label: 'MySQL', emoji: '🐬', port: '3306' },
        { id: 'mongodb', label: 'MongoDB', emoji: '🍃', port: '27017' },
        { id: 'redis', label: 'Redis', emoji: '🔴', port: '6379' },
        { id: 'neo4j', label: 'Neo4j', emoji: '🔷', port: '7687' },
        { id: 'duckdb', label: 'DuckDB', emoji: '🦆', port: '' },
        { id: 'sqlite', label: 'SQLite', emoji: '📁', port: '' },
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

                    <form onSubmit={e => handleRunSubmit(e)} className="pipeline-form">

                        {/* Analytics Targeting Config */}
                        {/* Analytics Targeting Config */}
                        <div className="form-row targeting-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '1.5rem', padding: '1rem', background: 'rgba(99, 102, 241, 0.03)', borderRadius: '8px', border: '1px solid rgba(99, 102, 241, 0.1)' }}>
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
                                                // Auto-fill Docker service name & DIPEX creds
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
                                                        onClick={ev => { ev.preventDefault(); setDbTable(t); handleRunSubmit(null, t); }}
                                                        disabled={running} title={`Click to ingest "${t}"`}>
                                                        {isIngesting
                                                            ? <span className="spinner" style={{ borderTopColor: '#6366f1', borderColor: 'rgba(99,102,241,0.3)', width: 14, height: 14 }} />
                                                            : <Database size={15} className="tc-icon" />}
                                                        <span className="tc-name">{t}</span>
                                                        {!isIngesting && <span className="tc-hint">click to ingest</span>}
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
                                        <p><strong>Max Messages</strong> — DIPEX reads up to this many messages in a 30-second tumbling window then processes the batch.</p>
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
                                    <span className="help-text">DIPEX collects up to this many messages then runs the full pipeline on the batch.</span>
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
                                    <div className="api-section-label">🔐 Authentication</div>
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

                        <button type="submit" className={`btn-run ${running ? 'running' : ''}`} disabled={running}>
                            {running ? <><span className="spinner" /> Running Pipeline…</> : <><Play className="btn-icon" /> Execute Pipeline</>}
                        </button>
                    </form>

                    {error && (
                        <div className="status-message error">
                            <AlertCircle className="status-icon" />
                            <div className="status-content"><strong>Execution Failed</strong><p>{error}</p></div>
                        </div>
                    )}

                    {result && <ResultPanel result={result} />}
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

            {result?.sample_rows?.length > 0 && (
                <DataPreviewPanel rows={result.sample_rows} sourceKind={result.source_kind} />
            )}
        </div>
    );
};

export default RunPipeline;
