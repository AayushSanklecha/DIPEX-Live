import React, { useState, useCallback } from 'react';
import { Database, Search, RefreshCw, Table2, ChevronLeft, ChevronRight, Eye, Loader2, AlertTriangle, CheckCircle2 } from 'lucide-react';
import './DataExplorer.css';

const API_BASE = import.meta.env.VITE_API_URL || '';
const PAGE_SIZE = 50;

const DB_TYPES = [
    { id: 'postgresql', label: 'PostgreSQL', emoji: '🐘', defaultPort: 5432 },
    { id: 'mongodb', label: 'MongoDB', emoji: '🍃', defaultPort: 27017 },
];

export default function DataExplorer() {
    const [backend, setBackend] = useState('postgresql');
    const [host, setHost] = useState('postgres');   // Docker service name, reachable by the API container
    const [port, setPort] = useState('5432');
    const [dbName, setDbName] = useState('dipex');
    const [user, setUser] = useState('dipex');
    const [pass, setPass] = useState('dipex_secret');

    const [tables, setTables] = useState([]);
    const [table, setTable] = useState('');
    const [loading, setLoading] = useState(false);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [error, setError] = useState(null);
    const [connected, setConnected] = useState(false);

    const [rows, setRows] = useState([]);
    const [columns, setCols] = useState([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(0);
    const [search, setSearch] = useState('');
    const [searchInput, setSearchInput] = useState('');

    const baseBody = () => ({
        backend,
        host,
        port: parseInt(port, 10),
        database: dbName,
        username: user || null,
        password: pass || null,
    });

    // ── Connect: list tables ──────────────────────────────────────────────────
    const handleConnect = useCallback(async () => {
        setLoading(true); setError(null); setConnected(false); setTables([]); setRows([]); setCols([]); setTable('');
        try {
            const res = await fetch(`${API_BASE}/api/explorer/connect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(baseBody()),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
            setTables(data.tables || []);
            setConnected(true);
        } catch (e) { setError(e.message); }
        finally { setLoading(false); }
    }, [backend, host, port, dbName, user, pass]);

    // ── Preview table ─────────────────────────────────────────────────────────
    const fetchPreview = useCallback(async (tbl, pg, srch) => {
        if (!tbl) return;
        setPreviewLoading(true); setError(null);
        try {
            const res = await fetch(`${API_BASE}/api/explorer/preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...baseBody(), table: tbl, limit: PAGE_SIZE, offset: pg * PAGE_SIZE, search: srch || null }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
            setRows(data.rows || []);
            setCols(data.columns || []);
            setTotal(data.total || 0);
        } catch (e) { setError(e.message); }
        finally { setPreviewLoading(false); }
    }, [backend, host, port, dbName, user, pass]);

    const handleSelectTable = (tbl) => {
        setTable(tbl); setPage(0); setSearch(''); setSearchInput('');
        fetchPreview(tbl, 0, '');
    };

    const handleSearch = () => {
        const s = searchInput.trim();
        setSearch(s); setPage(0);
        fetchPreview(table, 0, s);
    };

    const handlePage = (delta) => {
        const np = page + delta;
        setPage(np);
        fetchPreview(table, np, search);
    };

    const totalPages = Math.ceil(total / PAGE_SIZE);
    const dbConf = DB_TYPES.find(d => d.id === backend);

    return (
        <div className="explorer-page">
            {/* ── Sidebar: Connection Panel ────────────────────────────── */}
            <aside className="explorer-sidebar">
                <div className="exp-sidebar-header">
                    <Database size={18} className="exp-brand-icon" />
                    <span>Data Explorer</span>
                </div>

                {/* DB Type */}
                <div className="exp-field-group">
                    <label>Database Engine</label>
                    <div className="exp-engine-btns">
                        {DB_TYPES.map(db => (
                            <button
                                key={db.id}
                                className={`exp-engine-btn ${backend === db.id ? 'active' : ''}`}
                                onClick={() => {
                                    setBackend(db.id);
                                    setPort(String(db.defaultPort));
                                    // use correct Docker service names and creds
                                    if (db.id === 'postgresql') {
                                        setHost('postgres'); setDbName('dipex'); setUser('dipex'); setPass('dipex_secret');
                                    } else {
                                        setHost('mongo'); setDbName('dipex'); setUser('dipex'); setPass('dipex_secret');
                                    }
                                    setConnected(false); setTables([]); setRows([]); setTable('');
                                }}
                            >
                                <span className="exp-emoji">{db.emoji}</span> {db.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Connection fields */}
                <div className="exp-field-group">
                    <label>Host</label>
                    <input className="exp-input" value={host} onChange={e => setHost(e.target.value)} placeholder="localhost" />
                </div>
                <div className="exp-field-row">
                    <div className="exp-field-group">
                        <label>Port</label>
                        <input className="exp-input" value={port} onChange={e => setPort(e.target.value)} placeholder="5432" />
                    </div>
                    <div className="exp-field-group" style={{ flex: 2 }}>
                        <label>Database Name</label>
                        <input className="exp-input" value={dbName} onChange={e => setDbName(e.target.value)} placeholder="dipex_demo" />
                    </div>
                </div>
                <div className="exp-field-group">
                    <label>Username</label>
                    <input className="exp-input" value={user} onChange={e => setUser(e.target.value)} placeholder="hackathon" />
                </div>
                <div className="exp-field-group">
                    <label>Password</label>
                    <input className="exp-input" type="password" value={pass} onChange={e => setPass(e.target.value)} placeholder="password" />
                </div>

                <button className="exp-connect-btn" onClick={handleConnect} disabled={loading}>
                    {loading ? <Loader2 size={15} className="spin" /> : <Database size={15} />}
                    {loading ? 'Connecting…' : 'Connect & List Tables'}
                </button>

                {connected && (
                    <div className="exp-connected-badge">
                        <CheckCircle2 size={13} /> Connected to {dbConf?.label}
                    </div>
                )}

                {/* Table List */}
                {tables.length > 0 && (
                    <div className="exp-tables-list">
                        <div className="exp-tables-header">
                            <Table2 size={13} /> {tables.length} Table{tables.length !== 1 ? 's' : ''}
                        </div>
                        {tables.map(t => (
                            <button
                                key={t}
                                className={`exp-table-item ${table === t ? 'active' : ''}`}
                                onClick={() => handleSelectTable(t)}
                            >
                                {t}
                            </button>
                        ))}
                    </div>
                )}
            </aside>

            {/* ── Main : Data Table ────────────────────────────────────── */}
            <main className="explorer-main">
                {!table && !error && (
                    <div className="exp-empty">
                        <Database size={60} className="exp-empty-icon" />
                        <h2>Connect to a Database</h2>
                        <p>Fill in your connection details on the left, click <strong>Connect &amp; List Tables</strong>, then select a table to preview its raw data.</p>
                        <div className="exp-quick-tips">
                            <div className="exp-tip"><span>🐘</span><div><strong>PostgreSQL (DIPEX container)</strong><br />Host: <code>postgres</code> &nbsp;Port: <code>5432</code><br />DB: <code>dipex</code> &nbsp;User: <code>dipex</code> &nbsp;Pass: <code>dipex_secret</code><br />Tables: <code>hackathon_users</code>, <code>hackathon_transactions</code></div></div>
                            <div className="exp-tip"><span>🍃</span><div><strong>MongoDB (DIPEX container)</strong><br />Host: <code>mongo</code> &nbsp;Port: <code>27017</code><br />DB: <code>dipex</code> &nbsp;User: <code>dipex</code> &nbsp;Pass: <code>dipex_secret</code><br />Collection: <code>hackathon_device_logs</code></div></div>
                        </div>
                    </div>
                )}

                {error && (
                    <div className="exp-error">
                        <AlertTriangle size={18} />
                        <span>{error}</span>
                    </div>
                )}

                {table && (
                    <>
                        {/* Toolbar */}
                        <div className="exp-toolbar">
                            <div className="exp-table-title">
                                <Table2 size={16} />
                                <span className="exp-table-name">{table}</span>
                                <span className="exp-total-badge">{total.toLocaleString()} rows</span>
                            </div>
                            <div className="exp-search-bar">
                                <input
                                    className="exp-search-input"
                                    value={searchInput}
                                    onChange={e => setSearchInput(e.target.value)}
                                    onKeyDown={e => e.key === 'Enter' && handleSearch()}
                                    placeholder="Search…"
                                />
                                <button className="exp-search-btn" onClick={handleSearch}><Search size={14} /></button>
                                <button className="exp-refresh-btn" onClick={() => fetchPreview(table, page, search)} disabled={previewLoading}>
                                    <RefreshCw size={14} className={previewLoading ? 'spin' : ''} />
                                </button>
                            </div>
                        </div>

                        {/* Data Grid */}
                        <div className="exp-grid-wrapper">
                            {previewLoading ? (
                                <div className="exp-loading"><Loader2 size={28} className="spin" /> Loading data…</div>
                            ) : rows.length === 0 ? (
                                <div className="exp-loading">No rows found.</div>
                            ) : (
                                <table className="exp-table">
                                    <thead>
                                        <tr>
                                            {columns.map(col => (
                                                <th key={col} className="exp-th">{col}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {rows.map((row, ri) => (
                                            <tr key={ri} className={`exp-tr ${ri % 2 === 1 ? 'odd' : ''}`}>
                                                {columns.map(col => (
                                                    <td key={col} className={`exp-td ${row[col] == null ? 'null-cell' : ''}`}>
                                                        {row[col] == null ? <span className="exp-null">NULL</span> : String(row[col])}
                                                    </td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            )}
                        </div>

                        {/* Pagination */}
                        {totalPages > 1 && (
                            <div className="exp-pagination">
                                <button className="exp-page-btn" disabled={page === 0} onClick={() => handlePage(-1)}>
                                    <ChevronLeft size={14} /> Prev
                                </button>
                                <span className="exp-page-info">Page {page + 1} / {totalPages} &nbsp;·&nbsp; Rows {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} of {total.toLocaleString()}</span>
                                <button className="exp-page-btn" disabled={page >= totalPages - 1} onClick={() => handlePage(1)}>
                                    Next <ChevronRight size={14} />
                                </button>
                            </div>
                        )}
                    </>
                )}
            </main>
        </div>
    );
}
