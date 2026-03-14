import React, { useState, useEffect } from 'react';
import { BookOpen, ExternalLink, CheckCircle, ChevronDown, ChevronRight } from 'lucide-react';
import './ApiDocs.css';

const API_BASE = import.meta.env.VITE_API_URL || '';
// For links that open in a new tab (Swagger, Redoc) we need the real API host,
// since nginx doesn't proxy /docs or /redoc. Falls back to localhost:8000 in dev.
const DOCS_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ── Endpoints catalog ────────────────────────────────────────────────────────
const ENDPOINTS = [
    {
        category: '🚀 Pipeline',
        routes: [
            {
                method: 'POST', path: '/api/pipeline/run',
                summary: 'Ingest a file and run the full DIPEX pipeline in one step',
                body: [
                    { name: 'file', type: 'File (multipart)', required: true, desc: 'The file to run through the pipeline (CSV, Excel, JSON, Parquet)' },
                    { name: 'target_col', type: 'string', required: false, desc: 'Target column for supervised ML — auto-detected if blank' },
                    { name: 'dataset_id', type: 'string', required: false, desc: 'Stable dataset label; defaults to filename stem' },
                    { name: 'file_format', type: 'string', required: false, desc: 'Force format: csv | excel | json | xml | parquet' },
                    { name: 'skip_stages', type: 'string', required: false, desc: 'Comma-separated stages to skip, e.g. modeling,rl_update' },
                ],
            },
            {
                method: 'POST', path: '/api/pipeline/simple-run',
                summary: 'One-click pipeline — accepts file, database, Kafka, or REST API source',
                body: [
                    { name: 'source_kind', type: 'string', required: true, desc: 'file | database | graph_db | api | live (Kafka)' },
                    { name: 'source_input', type: 'string', required: false, desc: 'DB URI, Kafka topic/broker JSON, or API URL. Not needed for file uploads.' },
                    { name: 'file', type: 'File (multipart)', required: false, desc: 'Required only when source_kind=file' },
                    { name: 'target_col', type: 'string', required: false, desc: 'Optional target column name' },
                    { name: 'dataset_id', type: 'string', required: false, desc: 'Optional dataset label' },
                ],
            },
        ],
    },
    {
        category: '📥 Ingestion',
        routes: [
            { method: 'POST', path: '/api/ingest/', summary: 'Upload a file and assign a run_id for deferred pipeline execution' },
            { method: 'POST', path: '/api/ingest/direct', summary: 'Upload and immediately push data through UniversalIntake — returns schema snapshot' },
            { method: 'POST', path: '/api/ingest/fetch', summary: 'Pull data from an already-connected database into DIPEX and run intake' },
            { method: 'POST', path: '/api/ingest/v2/source', summary: 'V2 multi-source ingest — file, database, stream, API — returns full ISSF snapshot' },
        ],
    },
    {
        category: '📊 Data & Stats',
        routes: [
            { method: 'POST', path: '/api/run/', summary: 'Execute pipeline for a previously uploaded run_id (needs target_column)' },
            { method: 'GET', path: '/api/stats/', summary: 'EDA statistics for a dataset — distributions, outliers, null rates' },
            { method: 'GET', path: '/api/results/latest', summary: 'Retrieve the latest pipeline result' },
            { method: 'GET', path: '/api/results/{run_id}', summary: 'Retrieve the pipeline result for a specific run' },
        ],
    },
    {
        category: '📋 Reports & Audit',
        routes: [
            { method: 'GET', path: '/report/', summary: 'View the HTML executive report for the most recent run' },
            { method: 'GET', path: '/api/reports/', summary: 'List all available reports' },
            { method: 'GET', path: '/api/audit/logs', summary: 'Get paginated pipeline audit logs in JSONL format' },
            { method: 'POST', path: '/api/preprocess/', summary: 'Run preprocessing only — imputation, encoding, normalization' },
        ],
    },
    {
        category: '📤 Exports',
        routes: [
            { method: 'GET', path: '/api/export/csv', summary: 'Download processed dataset as CSV' },
            { method: 'GET', path: '/api/export/json', summary: 'Download processed dataset as JSON' },
            { method: 'GET', path: '/api/export/parquet', summary: 'Download processed dataset as Parquet' },
            { method: 'GET', path: '/api/export/report', summary: 'Download the executive report as PDF/HTML' },
            { method: 'GET', path: '/api/export/list', summary: 'List all available export files' },
        ],
    },
    {
        category: '🤖 AI & Analytics',
        routes: [
            { method: 'POST', path: '/analyst/run', summary: 'Trigger the full analyst pipeline for a given dataset_id' },
            { method: 'POST', path: '/api/cohort', summary: 'Run cohort retention analysis on the current dataset' },
        ],
    },
];

const METHOD_COLORS = { GET: '#10b981', POST: '#6366f1', PUT: '#f59e0b', DELETE: '#ef4444' };

// ── Route Card ───────────────────────────────────────────────────────────────
const RouteCard = ({ route }) => {
    const [open, setOpen] = useState(false);
    return (
        <div className="route-card">
            <div className="route-header" onClick={() => setOpen(!open)}>
                <span className="method-badge" style={{ background: METHOD_COLORS[route.method] }}>
                    {route.method}
                </span>
                <code className="route-path">{route.path}</code>
                <span className="route-summary">{route.summary}</span>
                <span className="route-chevron">
                    {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                </span>
            </div>
            {open && route.body && (
                <div className="route-body">
                    <table className="params-table">
                        <thead>
                            <tr>
                                <th>Parameter</th>
                                <th>Type</th>
                                <th>Required</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            {route.body.map((p) => (
                                <tr key={p.name}>
                                    <td><code>{p.name}</code></td>
                                    <td><span className="type-badge">{p.type}</span></td>
                                    <td>{p.required ? <span className="req-yes">Yes</span> : <span className="req-no">No</span>}</td>
                                    <td>{p.desc}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

// ── Main Page ────────────────────────────────────────────────────────────────
const ApiDocs = () => {
    const [apiAlive, setApiAlive] = useState(null);

    useEffect(() => {
        fetch(`${API_BASE}/health`)
            .then((r) => setApiAlive(r.ok))
            .catch(() => setApiAlive(false));
    }, []);

    return (
        <div className="api-docs-container">
            <div className="api-docs-header">
                <div className="api-docs-title">
                    <BookOpen className="api-docs-icon" />
                    <div>
                        <h2>API Reference</h2>
                        <p>Complete endpoint documentation for the DIPEX backend</p>
                    </div>
                </div>
                <div className="api-docs-actions">
                    {apiAlive != null && (
                        <span className={`api-status-badge ${apiAlive ? 'alive' : 'dead'}`}>
                            <CheckCircle size={14} />
                            {apiAlive ? 'API Online' : 'API Offline'}
                        </span>
                    )}
                    <a href={`${DOCS_BASE}/docs`} target="_blank" rel="noopener noreferrer" className="swagger-link">
                        <ExternalLink size={14} />
                        Swagger UI
                    </a>
                    <a href={`${DOCS_BASE}/redoc`} target="_blank" rel="noopener noreferrer" className="swagger-link">
                        <ExternalLink size={14} />
                        ReDoc
                    </a>
                </div>
            </div>

            {ENDPOINTS.map((section) => (
                <div key={section.category} className="endpoint-section">
                    <h3 className="section-title">{section.category}</h3>
                    {section.routes.map((r) => (
                        <RouteCard key={r.path + r.method} route={r} />
                    ))}
                </div>
            ))}
        </div>
    );
};

export default ApiDocs;
