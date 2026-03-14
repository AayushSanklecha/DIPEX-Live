import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
    BarChart, Bar, PieChart, Pie, Cell,
    ScatterChart, Scatter, XAxis, YAxis, ZAxis,
    CartesianGrid, Tooltip as ChartTooltip, Legend,
    ResponsiveContainer, AreaChart, Area, Treemap,
} from 'recharts';
import {
    Upload, Database, Radio, Globe,
    TrendingUp, BarChart2, PieChart as PieIcon,
    RefreshCw, Layers, Zap, Clock,
    ChevronUp, ChevronDown, Minus, FileText, CheckCircle2,
    Filter, Columns, X, SlidersHorizontal, ArrowRight, Lightbulb,
    Briefcase, Activity, Code, User,
} from 'lucide-react';
import {
    analyzeSchema, computeStats, recommendCharts,
    PALETTE,
} from '../utils/dataAnalyzer';
import { ResultsService, getCachedData } from '../api/client';
import './Dashboard.css';

// ── Helpers ──────────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_URL || '';


const fmtNum = (n, decimals = 2) => {
    if (n === undefined || n === null || isNaN(n)) return '—';
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return Number(n).toFixed(decimals);
};

const trendIcon = (val) => {
    if (val > 0) return <ChevronUp size={14} className="trend-up" />;
    if (val < 0) return <ChevronDown size={14} className="trend-down" />;
    return <Minus size={14} className="trend-flat" />;
};

// ── Custom tooltip ────────────────────────────────────────────────────────────

const PowerTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="power-tooltip">
            {label && <p className="pt-label">{label}</p>}
            {payload.map((p, i) => (
                <p key={i} style={{ color: p.color || PALETTE[i] }} className="pt-row">
                    <span className="pt-name">{p.name ?? p.dataKey}:</span>
                    <span className="pt-val">{fmtNum(p.value)}</span>
                </p>
            ))}
        </div>
    );
};

// ── KPI Card ──────────────────────────────────────────────────────────────────

const KpiCard = ({ stat, color }) => {
    const trend = stat.mean - stat.median;
    return (
        <div className="kpi-card" style={{ '--accent': color }}>
            <div className="kpi-top">
                <span className="kpi-col">{stat.col}</span>
                <span className="kpi-count">{stat.count.toLocaleString()} rows</span>
            </div>
            <div className="kpi-value">{fmtNum(stat.mean)}</div>
            <div className="kpi-label">avg</div>
            <div className="kpi-meta">
                <span><span className="meta-label">Min</span>{fmtNum(stat.min)}</span>
                <span><span className="meta-label">Max</span>{fmtNum(stat.max)}</span>
                <span><span className="meta-label">σ</span>{fmtNum(stat.stddev)}</span>
            </div>
            <div className="kpi-trend">
                {trendIcon(trend)}
                <span>Mean vs Median Δ: {fmtNum(Math.abs(trend))}</span>
            </div>
            <div className="kpi-bar" style={{ width: `${Math.min(100, (stat.mean / stat.max) * 100)}%` }} />
        </div>
    );
};

// ── Chart Deck ────────────────────────────────────────────────────────────────

const ChartDeck = ({ charts }) => {
    if (!charts?.length) return null;

    const renderChart = (ch, idx) => {
        const color = PALETTE[idx % PALETTE.length];

        switch (ch.type) {
            case 'line':
            case 'multiline':
                return (
                    <div className="deck-card span-2" key={idx}>
                        <div className="deck-card-header">
                            <TrendingUp size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <ResponsiveContainer width="100%" height={220}>
                            <AreaChart data={ch.data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                <defs>
                                    <linearGradient id={`grad${idx}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                                        <stop offset="95%" stopColor={color} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                                <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} tickFormatter={v => fmtNum(v, 0)} />
                                <ChartTooltip content={<PowerTooltip />} />
                                <Area type="monotone" dataKey={ch.valueCol} name={ch.valueCol}
                                    stroke={color} strokeWidth={2.5}
                                    fill={`url(#grad${idx})`} dot={false} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'bar':
                return (
                    <div className="deck-card" key={idx}>
                        <div className="deck-card-header">
                            <BarChart2 size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <ResponsiveContainer width="100%" height={220}>
                            <BarChart data={ch.data} layout="vertical" margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                                <XAxis type="number" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} tickFormatter={v => fmtNum(v, 0)} />
                                <YAxis dataKey="name" type="category" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={90} />
                                <ChartTooltip content={<PowerTooltip />} />
                                <Bar dataKey="value" fill={color} radius={[0, 6, 6, 0]}>
                                    {ch.data.map((_, i) => (
                                        <Cell key={i} fill={PALETTE[i % PALETTE.length]} fillOpacity={0.85} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'pie':
                return (
                    <div className="deck-card" key={idx}>
                        <div className="deck-card-header">
                            <PieIcon size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <ResponsiveContainer width="100%" height={220}>
                            <PieChart>
                                <Pie data={ch.data} cx="50%" cy="50%"
                                    innerRadius={55} outerRadius={85}
                                    paddingAngle={3} dataKey="value">
                                    {ch.data.map((_, i) => (
                                        <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                                    ))}
                                </Pie>
                                <ChartTooltip content={<PowerTooltip />} />
                                <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'scatter':
                return (
                    <div className="deck-card" key={idx}>
                        <div className="deck-card-header">
                            <Layers size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <ResponsiveContainer width="100%" height={220}>
                            <ScatterChart margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="x" name={ch.xCol} tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} tickFormatter={v => fmtNum(v, 0)} />
                                <YAxis dataKey="y" name={ch.yCol} tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} tickFormatter={v => fmtNum(v, 0)} />
                                <ZAxis range={[30, 30]} />
                                <ChartTooltip cursor={{ strokeDasharray: '3 3' }} content={<PowerTooltip />} />
                                <Scatter data={ch.data} fill={color} fillOpacity={0.7} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'histogram':
                return (
                    <div className="deck-card" key={idx}>
                        <div className="deck-card-header">
                            <BarChart2 size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <ResponsiveContainer width="100%" height={220}>
                            <BarChart data={ch.data} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="bin" tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                                <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                                <ChartTooltip content={<PowerTooltip />} />
                                <Bar dataKey="count" fill={color} radius={[4, 4, 0, 0]} barSize={40} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'boxplot':
                if (!ch.stats) return null;
                return (
                    <div className="deck-card" key={idx}>
                        <div className="deck-card-header">
                            <Layers size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <div className="boxplot-viz">
                            <div className="boxplot-stats-grid">
                                <div><label>Min</label><span>{fmtNum(ch.stats.min)}</span></div>
                                <div><label>Q1</label><span>{fmtNum(ch.stats.q1)}</span></div>
                                <div><label>Med</label><span className="med">{fmtNum(ch.stats.median)}</span></div>
                                <div><label>Q3</label><span>{fmtNum(ch.stats.q3)}</span></div>
                                <div><label>Max</label><span>{fmtNum(ch.stats.max)}</span></div>
                            </div>
                            <div className="boxplot-track">
                                <div className="boxplot-whisker" style={{
                                    left: `${((ch.stats.lowerFence - ch.stats.min) / (ch.stats.max - ch.stats.min)) * 100}%`,
                                    width: `${((ch.stats.upperFence - ch.stats.lowerFence) / (ch.stats.max - ch.stats.min)) * 100}%`
                                }} />
                                <div className="boxplot-box" style={{
                                    left: `${((ch.stats.q1 - ch.stats.min) / (ch.stats.max - ch.stats.min)) * 100}%`,
                                    width: `${((ch.stats.q3 - ch.stats.q1) / (ch.stats.max - ch.stats.min)) * 100}%`
                                }} />
                                <div className="boxplot-median" style={{
                                    left: `${((ch.stats.median - ch.stats.min) / (ch.stats.max - ch.stats.min)) * 100}%`
                                }} />
                            </div>
                        </div>
                    </div>
                );

            case 'treemap':
                return (
                    <div className="deck-card span-2" key={idx}>
                        <div className="deck-card-header">
                            <PieIcon size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <ResponsiveContainer width="100%" height={300}>
                            <Treemap
                                data={ch.data}
                                dataKey="value"
                                ratio={4 / 3}
                                stroke="#fff"
                                fill={color}
                            >
                                <ChartTooltip content={<PowerTooltip />} />
                            </Treemap>
                        </ResponsiveContainer>
                    </div>
                );

            case 'hbar':
                return (
                    <div className="deck-card" key={idx}>
                        <div className="deck-card-header">
                            <BarChart2 size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <ResponsiveContainer width="100%" height={260}>
                            <BarChart data={ch.data} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                                <XAxis type="number" hide />
                                <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: '#94a3b8' }} width={80} axisLine={false} tickLine={false} />
                                <ChartTooltip content={<PowerTooltip />} />
                                <Bar dataKey="value" fill={color} radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'table':
                return (
                    <div className="deck-card span-all" key={idx}>
                        <div className="deck-card-header">
                            <FileText size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <div className="dashboard-table-preview">
                            <table className="mini-table">
                                <thead>
                                    <tr>
                                        {ch.columns.map(c => <th key={c}>{c}</th>)}
                                    </tr>
                                </thead>
                                <tbody>
                                    {ch.data.map((r, i) => (
                                        <tr key={i}>
                                            {ch.columns.map(c => <td key={c}>{String(r[c] ?? '')}</td>)}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                );

            default:
                return null;
        }
    };

    return (
        <div className="deck-grid">
            {charts.map((ch, i) => renderChart(ch, i))}
        </div>
    );
};

// ── ML Intelligence Chart Deck ────────────────────────────────────────────────
const IntelChartDeck = ({ charts }) => {
    if (!charts?.length) return null;

    const renderIntelChart = (ch, idx) => {
        const color = PALETTE[idx % PALETTE.length];

        switch (ch.type) {
            case 'scatter':
                const [xCol, yCol] = ch.columns;
                return (
                    <div className="deck-card" key={ch.id}>
                        <div className="deck-card-header">
                            <Layers size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <p className="intel-insight-small">{ch.insight}</p>
                        <ResponsiveContainer width="100%" height={200}>
                            <ScatterChart margin={{ top: 15, right: 20, left: 10, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                                <XAxis type="number" dataKey={xCol} tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => fmtNum(v)} />
                                <YAxis type="number" dataKey={yCol} tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => fmtNum(v)} width={50} />
                                <ChartTooltip content={<PowerTooltip />} cursor={{ strokeDasharray: '3 3' }} />
                                <ZAxis range={[30, 30]} />
                                <Scatter data={ch.data} fill={color} opacity={0.7} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'bar':
                return (
                    <div className="deck-card" key={ch.id}>
                        <div className="deck-card-header">
                            <BarChart2 size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <p className="intel-insight-small">{ch.insight}</p>
                        <ResponsiveContainer width="100%" height={200}>
                            <BarChart data={ch.data} layout="vertical" margin={{ top: 15, right: 20, left: 20, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#21262d" />
                                <XAxis type="number" tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => fmtNum(v)} />
                                <YAxis dataKey="name" type="category" tick={{ fill: '#94a3b8', fontSize: 11 }} tickLine={false} axisLine={false} width={80} />
                                <ChartTooltip content={<PowerTooltip />} />
                                <Bar dataKey="value" fill={color} radius={[0, 4, 4, 0]}>
                                    {ch.data.map((_, i) => (
                                        <Cell key={i} fill={PALETTE[i % PALETTE.length]} fillOpacity={0.8} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'stacked_bar':
                return (
                    <div className="deck-card span-2" key={ch.id}>
                        <div className="deck-card-header">
                            <BarChart2 size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <p className="intel-insight-small">{ch.insight}</p>
                        <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={ch.data} margin={{ top: 15, right: 20, left: 10, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#21262d" />
                                <XAxis dataKey="name" tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} />
                                <YAxis tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => fmtNum(v)} width={50} />
                                <ChartTooltip content={<PowerTooltip />} />
                                <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', color: '#94a3b8', paddingTop: '10px' }} />
                                {ch.stack_keys?.map((k, i) => (
                                    <Bar key={k} dataKey={k} stackId="a" fill={PALETTE[i % PALETTE.length]} radius={i === ch.stack_keys.length - 1 ? [4, 4, 0, 0] : [0, 0, 0, 0]} />
                                ))}
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'line':
            case 'area':
                return (
                    <div className="deck-card" key={ch.id}>
                        <div className="deck-card-header">
                            <TrendingUp size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <p className="intel-insight-small">{ch.insight}</p>
                        <ResponsiveContainer width="100%" height={200}>
                            <AreaChart data={ch.data} margin={{ top: 15, right: 20, left: 10, bottom: 5 }}>
                                <defs>
                                    <linearGradient id={`igrad${idx}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                                        <stop offset="95%" stopColor={color} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#21262d" />
                                <XAxis dataKey={ch.type === 'area' ? 'bin' : 'time'} tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} />
                                <YAxis tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => fmtNum(v)} width={50} />
                                <ChartTooltip content={<PowerTooltip />} />
                                <Area type="monotone" dataKey={ch.type === 'area' ? 'count' : 'value'} stroke={color} strokeWidth={2} fill={`url(#igrad${idx})`} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'multi_line':
                return (
                    <div className="deck-card span-2" key={ch.id}>
                        <div className="deck-card-header">
                            <TrendingUp size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <p className="intel-insight-small">{ch.insight}</p>
                        <ResponsiveContainer width="100%" height={250}>
                            <AreaChart data={ch.data} margin={{ top: 15, right: 20, left: 10, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#21262d" />
                                <XAxis dataKey="time" tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} />
                                <YAxis tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => fmtNum(v)} width={50} />
                                <ChartTooltip content={<PowerTooltip />} />
                                <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', color: '#94a3b8', paddingTop: '10px' }} />
                                {ch.line_keys?.map((k, i) => (
                                    <Area key={k} type="monotone" dataKey={k} stroke={PALETTE[i % PALETTE.length]} strokeWidth={2} fillOpacity={0} />
                                ))}
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                );

            case 'pie':
                return (
                    <div className="deck-card" key={ch.id}>
                        <div className="deck-card-header">
                            <PieIcon size={16} className="deck-icon" style={{ color }} />
                            <h3>{ch.title}</h3>
                        </div>
                        <p className="intel-insight-small">{ch.insight}</p>
                        <ResponsiveContainer width="100%" height={200}>
                            <PieChart>
                                <Pie data={ch.data} cx="50%" cy="50%" innerRadius={50} outerRadius={75} paddingAngle={2} dataKey="value">
                                    {ch.data.map((_, i) => (
                                        <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                                    ))}
                                </Pie>
                                <ChartTooltip content={<PowerTooltip />} />
                                <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                );

            default:
                return null;
        }
    };

    return (
        <div className="deck-grid">
            {charts.map((ch, i) => renderIntelChart(ch, i))}
        </div>
    );
};

// ── Source Badge ──────────────────────────────────────────────────────────────

const SourceBadge = ({ kind }) => {
    const map = {
        file: { icon: FileText, label: 'File Upload', cls: 'src-file' },
        database: { icon: Database, label: 'Database', cls: 'src-db' },
        live: { icon: Radio, label: 'Kafka Stream', cls: 'src-kafka' },
        api: { icon: Globe, label: 'REST API', cls: 'src-api' },
    };
    const src = map[kind] || { icon: Zap, label: kind, cls: 'src-file' };
    const Icon = src.icon;
    return (
        <span className={`source-badge ${src.cls}`}>
            <Icon size={12} /> {src.label}
        </span>
    );
};

// ── Column Selector Panel ─────────────────────────────────────────────────────

const ColumnSelectorPanel = ({ schema, selectedCols, onToggleCol, onSelectAll, onClearAll }) => {
    const [open, setOpen] = useState(false);
    if (!schema?.columns.length) return null;

    return (
        <div className="selector-panel">
            <button className="selector-trigger" onClick={() => setOpen(o => !o)}>
                <Columns size={14} />
                Columns
                <span className="selector-count">{selectedCols.size} / {schema.columns.length}</span>
                <ChevronDown size={12} style={{ transform: open ? 'rotate(180deg)' : '', transition: '0.2s' }} />
            </button>

            {open && (
                <div className="selector-dropdown">
                    <div className="selector-actions">
                        <button onClick={onSelectAll}>All</button>
                        <button onClick={onClearAll}>Clear</button>
                        {schema.numericCols.length > 0 && (
                            <button onClick={() => { onClearAll(); schema.numericCols.forEach(c => onToggleCol(c, true)); }}>
                                Numeric only
                            </button>
                        )}
                        {schema.categoricalCols.length > 0 && (
                            <button onClick={() => { onClearAll(); schema.categoricalCols.forEach(c => onToggleCol(c, true)); }}>
                                Categorical only
                            </button>
                        )}
                    </div>
                    <div className="selector-list">
                        {schema.columns.map(c => (
                            <label key={c.key} className={`selector-item ${selectedCols.has(c.key) ? 'selected' : ''}`}>
                                <input
                                    type="checkbox"
                                    checked={selectedCols.has(c.key)}
                                    onChange={e => onToggleCol(c.key, e.target.checked)}
                                />
                                <span className="sel-col-name">{c.key}</span>
                                <span className={`type-badge type-${c.type}`}>{c.type}</span>
                            </label>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

// ── Row Filter Panel ──────────────────────────────────────────────────────────

const RowFilterPanel = ({ schema, rows, activeFilters, onAddFilter, onRemoveFilter, onClearFilters }) => {
    const [open, setOpen] = useState(false);
    const [filterCol, setFilterCol] = useState('');
    const [filterOp, setFilterOp] = useState('eq');
    const [filterVal, setFilterVal] = useState('');

    const colOptions = schema?.columns ?? [];
    const activeCount = activeFilters.length;

    const handleAdd = () => {
        if (!filterCol || filterVal === '') return;
        onAddFilter({ col: filterCol, op: filterOp, val: filterVal });
        setFilterVal('');
    };

    // Unique values for the selected column (for autocomplete)
    const uniqueVals = useMemo(() => {
        if (!filterCol || !rows.length) return [];
        const vals = [...new Set(rows.map(r => String(r[filterCol] ?? '')))].sort();
        return vals.slice(0, 50);
    }, [filterCol, rows]);

    return (
        <div className="selector-panel">
            <button className="selector-trigger" onClick={() => setOpen(o => !o)}>
                <Filter size={14} />
                Rows
                {activeCount > 0 && <span className="selector-count active">{activeCount} active</span>}
                <ChevronDown size={12} style={{ transform: open ? 'rotate(180deg)' : '', transition: '0.2s' }} />
            </button>

            {open && (
                <div className="selector-dropdown wide">
                    {/* Filter builder */}
                    <div className="filter-builder">
                        <select value={filterCol} onChange={e => setFilterCol(e.target.value)} className="filter-select">
                            <option value="">— Pick column —</option>
                            {colOptions.map(c => (
                                <option key={c.key} value={c.key}>{c.key} ({c.type})</option>
                            ))}
                        </select>

                        <select value={filterOp} onChange={e => setFilterOp(e.target.value)} className="filter-select-sm">
                            <option value="eq">= equals</option>
                            <option value="neq">≠ not equals</option>
                            <option value="gt">&gt; greater</option>
                            <option value="lt">&lt; less</option>
                            <option value="gte">≥ gte</option>
                            <option value="lte">≤ lte</option>
                            <option value="contains">contains</option>
                            <option value="startswith">starts with</option>
                        </select>

                        <input
                            className="filter-input"
                            list="filter-val-list"
                            value={filterVal}
                            onChange={e => setFilterVal(e.target.value)}
                            placeholder="Value…"
                        />
                        <datalist id="filter-val-list">
                            {uniqueVals.map(v => <option key={v} value={v} />)}
                        </datalist>

                        <button className="filter-add-btn" onClick={handleAdd} disabled={!filterCol || filterVal === ''}>
                            + Add
                        </button>
                    </div>

                    {/* Active filters */}
                    {activeCount > 0 && (
                        <div className="filter-pills">
                            {activeFilters.map((f, i) => (
                                <div key={i} className="filter-pill">
                                    <code>{f.col}</code>&nbsp;{f.op}&nbsp;<em>"{f.val}"</em>
                                    <button onClick={() => onRemoveFilter(i)} className="pill-remove"><X size={10} /></button>
                                </div>
                            ))}
                            <button className="filter-clear" onClick={onClearFilters}>Clear all</button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

// ── Schema Table ──────────────────────────────────────────────────────────────

const SchemaTable = ({ schema, selectedCols, onToggleCol }) => (
    <div className="schema-table-wrapper">
        <table className="schema-table">
            <thead>
                <tr>
                    <th>Include</th>
                    <th>Column</th>
                    <th>Detected Type</th>
                </tr>
            </thead>
            <tbody>
                {schema.columns.map(c => (
                    <tr key={c.key} className={selectedCols?.has(c.key) ? 'row-selected' : 'row-dim'}>
                        <td>
                            <input type="checkbox"
                                checked={selectedCols?.has(c.key) ?? true}
                                onChange={e => onToggleCol?.(c.key, e.target.checked)}
                                className="schema-check"
                            />
                        </td>
                        <td className="col-name">{c.key}</td>
                        <td><span className={`type-badge type-${c.type}`}>{c.type}</span></td>
                    </tr>
                ))}
            </tbody>
        </table>
    </div>
);

// ── Row Filter logic ──────────────────────────────────────────────────────────

function applyFilters(rows, filters) {
    if (!filters.length) return rows;
    return rows.filter(row => filters.every(f => {
        const cellRaw = row[f.col];
        const cell = String(cellRaw ?? '');
        const val = f.val;
        const numCell = parseFloat(cell);
        const numVal = parseFloat(val);

        switch (f.op) {
            case 'eq': return cell === val;
            case 'neq': return cell !== val;
            case 'gt': return !isNaN(numCell) && numCell > numVal;
            case 'lt': return !isNaN(numCell) && numCell < numVal;
            case 'gte': return !isNaN(numCell) && numCell >= numVal;
            case 'lte': return !isNaN(numCell) && numCell <= numVal;
            case 'contains': return cell.toLowerCase().includes(val.toLowerCase());
            case 'startswith': return cell.toLowerCase().startsWith(val.toLowerCase());
            default: return true;
        }
    }));
}

// Narrow rows to only include selected columns.
// allSelected=true  → show all columns (user hasn't touched the selector)
// selectedCols.size > 0 + allSelected=false  → project to chosen columns
// selectedCols.size === 0 + allSelected=false → user explicitly cleared all; return [] so the UI shows "no columns" message
function projectCols(rows, selectedCols, allSelected) {
    if (allSelected) return rows;
    if (!selectedCols || selectedCols.size === 0) return [];
    return rows.map(r => {
        const out = {};
        selectedCols.forEach(k => { if (k in r) out[k] = r[k]; });
        return out;
    });
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

const Dashboard = () => {
    // ── Synchronous Init from Cache ──
    const cachedData = getCachedData('/api/results/latest');

    const buildInitRows = (data) => {
        if (!data) return [];
        if (data.sample_rows?.length) return data.sample_rows;
        if (data.preview_data?.length) return data.preview_data;
        const metrics = data.final_result ?? data;
        const entries = Object.entries(metrics).filter(([, v]) => typeof v === 'number');
        return entries.length ? [Object.fromEntries(entries)] : [];
    };

    const initRows = buildInitRows(cachedData);
    const initSchema = initRows.length ? analyzeSchema(initRows) : null;
    const initIntel = cachedData?.run_id ? getCachedData(`/api/results/${cachedData.run_id}/intelligence`) : null;

    const [reportData, setReportData] = useState(cachedData || null);
    const [rows, setRows] = useState(initRows);
    const [schema, setSchema] = useState(initSchema);
    const [phase, setPhase] = useState(cachedData ? 'ready' : 'idle');
    const [sourceKind, setSourceKind] = useState(cachedData?.source_kind || '');
    const [activeTab, setActiveTab] = useState('overview');
    const [lastSync, setLastSync] = useState(cachedData ? new Date() : null);
    const [intelligence, setIntelligence] = useState(initIntel);
    const [persona, setPersona] = useState('executive'); // 'executive', 'analyst', 'data_scientist'

    // ── Selection state ──────────────────────────────────────────────────────
    // allSelected=true means "show all" — separate from selectedCols.size===0
    const [selectedCols, setSelectedCols] = useState(new Set(initSchema?.columns?.map(c => c.key) || []));
    const [allSelected, setAllSelected] = useState(true);
    const [rowFilters, setRowFilters] = useState([]);

    const syncRef = useRef(null);

    // ── Helpers for column selector ──────────────────────────────────────────
    const toggleCol = useCallback((key, checked) => {
        setAllSelected(false);
        setSelectedCols(prev => {
            const next = new Set(prev);
            checked ? next.add(key) : next.delete(key);
            return next;
        });
    }, []);

    const selectAll = useCallback(() => {
        if (!schema) return;
        setAllSelected(true);
        setSelectedCols(new Set(schema.columns.map(c => c.key)));
    }, [schema]);

    const clearAll = useCallback(() => {
        setAllSelected(false);
        setSelectedCols(new Set());
    }, []);

    // ── Derived: filtered + projected rows ───────────────────────────────────
    const filteredRows = useMemo(() => {
        let r = applyFilters(rows, rowFilters);
        r = projectCols(r, selectedCols, allSelected);
        return r;
    }, [rows, rowFilters, selectedCols, allSelected]);

    // ── Derived: schema + charts + stats from filteredRows ───────────────────
    const { activeSchema, stats, charts } = useMemo(() => {
        if (!filteredRows.length) return { activeSchema: schema, stats: [], charts: [] };
        const sc = analyzeSchema(filteredRows);
        const st = computeStats(filteredRows, sc.numericCols);
        const ch = recommendCharts(sc, filteredRows);
        return { activeSchema: sc, stats: st, charts: ch };
    }, [filteredRows, schema]);

    // ── Auto-pull latest pipeline result ─────────────────────────────────────
    const fetchLatestResult = useCallback(async () => {
        try {
            const data = await ResultsService.getLatestResult();
            if (!data) { setPhase(p => (p === 'idle' || p === 'loading') ? 'empty' : p); return; }

            setReportData(data);
            setSourceKind(data.source_kind || '');
            setLastSync(new Date());

            const intelData = await ResultsService.getIntelligence(data.run_id);
            if (intelData) setIntelligence(intelData);

            let incoming = [];
            if (data.sample_rows?.length) incoming = data.sample_rows;
            else if (data.preview_data?.length) incoming = data.preview_data;

            if (incoming.length) {
                const sc = analyzeSchema(incoming);
                setSchema(sc);
                setRows(incoming);
                // Auto-select all on first load or when run changes
                setAllSelected(true);
                setSelectedCols(new Set(sc.columns.map(c => c.key)));
            } else {
                const synthetic = buildSyntheticRows(data);
                if (synthetic.length) {
                    const sc = analyzeSchema(synthetic);
                    setSchema(sc);
                    setRows(synthetic);
                    setAllSelected(true);
                    setSelectedCols(new Set(sc.columns.map(c => c.key)));
                } else {
                    setPhase(p => p === 'idle' || p === 'loading' ? 'empty' : p);
                }
            }
            setPhase('ready');
        } catch {
            if (phase === 'idle' || phase === 'loading') setPhase('empty');
        }
    }, [phase]);

    useEffect(() => {
        if (phase === 'idle') setPhase('loading');
        fetchLatestResult();
        syncRef.current = setInterval(fetchLatestResult, 10_000);
        return () => clearInterval(syncRef.current);
    }, []);

    const buildSyntheticRows = (data) => {
        if (!data) return [];
        const metrics = data.final_result ?? data;
        const entries = Object.entries(metrics).filter(([, v]) => typeof v === 'number');
        return entries.length ? [Object.fromEntries(entries)] : [];
    };

    // ── Filter helpers ────────────────────────────────────────────────────────
    const addFilter = useCallback(f => setRowFilters(prev => [...prev, f]), []);
    const removeFilter = useCallback(i => setRowFilters(prev => prev.filter((_, idx) => idx !== i)), []);
    const clearFilters = useCallback(() => setRowFilters([]), []);

    // ── Render states ─────────────────────────────────────────────────────────

    if (phase === 'loading') return (
        <div className="bi-dashboard-v2">
            <div className="dash-loading">
                <div className="loader-ring" />
                <p>Loading pipeline telemetry…</p>
            </div>
        </div>
    );

    if (phase === 'empty') return (
        <div className="bi-dashboard-v2">
            <div className="dash-empty">
                <div className="empty-icon-wrap"><Zap size={48} className="empty-icon" /></div>
                <h2>No Pipeline Data Yet</h2>
                <p>Run a pipeline first — the dashboard will automatically generate visualisations after each run.</p>
                <a href="/run" className="btn-run-pipeline">
                    <CheckCircle2 size={16} /> Run Your First Pipeline
                </a>
            </div>
        </div>
    );

    // ── Ready ─────────────────────────────────────────────────────────────────

    const gate1 = reportData?.gate1_decision;
    const gate2 = reportData?.gate2_decision;
    const totalRows = reportData?.row_count ?? rows.length;
    const totalCols = reportData?.col_count ?? schema?.columns.length ?? 0;
    const selSize = allSelected ? totalCols : selectedCols.size;
    // Confidence / quality may live at top-level (results endpoint) or nested (run endpoint)
    const confScore = reportData?.confidence_score
        ?? reportData?.final_result?.confidence_score
        ?? (reportData?.final_result?.confidence_vector?.confidence_score);
    const qualScore = reportData?.quality_score
        ?? reportData?.final_result?.quality_score;
    const targetCol = reportData?.target_col
        ?? reportData?.final_result?.target_column_used
        ?? reportData?.target_column_used;

    // Filter charts based on persona and selectedCols
    const getFilteredIntelCharts = (chartsList) => {
        if (!chartsList) return [];

        // 1. Filter out charts that use columns we have unchecked
        let validCharts = chartsList.filter(c => {
            // Collect all columns this chart depends on
            const deps = [];
            if (c.xCol) deps.push(c.xCol);
            if (c.yCol) deps.push(c.yCol);
            if (c.valueCol) deps.push(c.valueCol);
            if (c.bin) deps.push(c.bin); // If named
            if (c.stack_keys) deps.push(...c.stack_keys);
            if (c.line_keys) deps.push(...c.line_keys);

            // Allow if all dependencies are in selectedCols (or if allSelected is true)
            if (allSelected) return true;
            return deps.every(d => selectedCols.has(d));
        });

        // 2. Filter by persona
        if (persona === 'executive') {
            return validCharts.filter(c => ['bar', 'pie', 'line', 'area', 'hbar', 'stacked_bar', 'treemap'].includes(c.type));
        }
        if (persona === 'data_scientist') {
            return validCharts.filter(c => ['scatter', 'histogram', 'boxplot', 'multi_line', 'line'].includes(c.type));
        }
        return validCharts;
    };

    const displayCharts = getFilteredIntelCharts(intelligence?.charts) || [];

    return (
        <div className="bi-dashboard-v2">

            {/* ── Master Header ──────────────────────────────────────────── */}
            <div className="bi-master-header">
                <div className="bi-title-area">
                    <div className="bi-logo-dot" />
                    <div>
                        <h1 className="bi-title">
                            {reportData?.dataset_id ?? 'Pipeline Analytics'}
                        </h1>
                        <div className="bi-subtitle">
                            Run&nbsp;<code>{reportData?.run_id?.slice(0, 8) ?? '—'}</code>
                            &nbsp;·&nbsp;
                            {filteredRows.length.toLocaleString()} / {totalRows.toLocaleString()} rows
                            &nbsp;·&nbsp;
                            {selSize} / {totalCols} cols
                            {targetCol && <>&nbsp;·&nbsp;<span className="target-col-hint">🎯 {targetCol}</span></>}
                            {sourceKind && <>&nbsp;·&nbsp;<SourceBadge kind={sourceKind} /></>}
                        </div>
                    </div>
                </div>
                <div className="bi-toolbar">
                    <div className="persona-switcher">
                        <button className={`persona-btn ${persona === 'executive' ? 'active' : ''}`} onClick={() => setPersona('executive')}>
                            <Briefcase size={14} /> Executive
                        </button>
                        <button className={`persona-btn ${persona === 'analyst' ? 'active' : ''}`} onClick={() => setPersona('analyst')}>
                            <Activity size={14} /> Analyst
                        </button>
                        <button className={`persona-btn ${persona === 'data_scientist' ? 'active' : ''}`} onClick={() => setPersona('data_scientist')}>
                            <Code size={14} /> Data Scientist
                        </button>
                    </div>
                    {lastSync && (
                        <span className="bi-sync">
                            <Clock size={13} /> {lastSync.toLocaleTimeString()}
                        </span>
                    )}
                    <button className="bi-refresh" onClick={() => { setPhase('loading'); fetchLatestResult(); }}>
                        <RefreshCw size={14} /> Refresh
                    </button>
                </div>
            </div>

            {/* ── Gate Status Bar ────────────────────────────────────────── */}
            {(gate1 || gate2) && (
                <div className="gate-status-bar">
                    <div className={`gate-chip ${(gate1 || '').toLowerCase()}`}>
                        Gate 1 — Quality: <strong>{gate1 ?? '—'}</strong>
                    </div>
                    <div className={`gate-chip ${(gate2 || '').toLowerCase()}`}>
                        Gate 2 — Stats: <strong>{gate2 ?? '—'}</strong>
                    </div>
                    {confScore != null && (
                        <div className="gate-chip confidence">
                            Confidence: <strong>{(confScore * 100).toFixed(1)}%</strong>
                        </div>
                    )}
                    {qualScore != null && (
                        <div className="gate-chip quality">
                            Quality Score: <strong>{(qualScore * 100).toFixed(1)}%</strong>
                        </div>
                    )}
                </div>
            )}

            {/* ── Filter Bar (Hidden for Executive) ──────────────────────── */}
            {persona !== 'executive' && (
                <div className="filter-bar">
                    <div className="filter-bar-left">
                        <SlidersHorizontal size={14} className="filter-bar-icon" />
                        <span className="filter-bar-label">Focus view:</span>

                        {/* Column Selector */}
                        {schema && (
                            <ColumnSelectorPanel
                                schema={schema}
                                selectedCols={selectedCols}
                                onToggleCol={toggleCol}
                                onSelectAll={selectAll}
                                onClearAll={clearAll}
                            />
                        )}

                        {/* Row Filter */}
                        {schema && rows.length > 0 && (
                            <RowFilterPanel
                                schema={schema}
                                rows={rows}
                                activeFilters={rowFilters}
                                onAddFilter={addFilter}
                                onRemoveFilter={removeFilter}
                                onClearFilters={clearFilters}
                            />
                        )}
                    </div>

                    {/* Active filter pills summary in bar */}
                    {rowFilters.length > 0 && (
                        <div className="filter-bar-pills">
                            {rowFilters.map((f, i) => (
                                <div key={i} className="filter-pill-sm">
                                    {f.col}&nbsp;<em>{f.op}</em>&nbsp;"{f.val}"
                                    <button onClick={() => removeFilter(i)}><X size={9} /></button>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="filter-bar-right">
                        {(rowFilters.length > 0 || !allSelected) && (
                            <button className="filter-reset" onClick={() => { clearFilters(); selectAll(); }}>
                                <X size={12} /> Reset
                            </button>
                        )}
                    </div>
                </div>
            )}

            {/* ── Tabs ──────────────────────────────────────────────────── */}
            <div className="bi-tabs">
                {['overview', 'charts', 'schema', 'narrative'].map(t => {
                    if (persona === 'executive' && t === 'schema') return null;
                    return (
                        <button
                            key={t}
                            className={`bi-tab ${activeTab === t ? 'bi-tab--active' : ''}`}
                            onClick={() => setActiveTab(t)}
                        >
                            {t.charAt(0).toUpperCase() + t.slice(1)}
                        </button>
                    )
                })}
            </div>

            {/* ── Overview ──────────────────────────────────────────────── */}
            {activeTab === 'overview' && (
                <div className="tab-pane">
                    {intelligence ? (
                        <div className="dashboard-intel-view">
                            {/* KPI Row - Executives see Narrative summary here too */}
                            {persona === 'executive' && reportData?.narrative && (
                                <div className="executive-summary-banner">
                                    <p>{reportData.narrative.body.substring(0, 350)}...</p>
                                </div>
                            )}

                            <div className="intel-kpi-row">
                                {Object.entries(intelligence.kpis || {}).map(([k, v], i) => (
                                    <div key={k} className="intel-kpi-card" style={{ '--accent': PALETTE[i % PALETTE.length] }}>
                                        <div className="intel-kpi-val">{typeof v === 'number' && k.includes('Total') ? v.toLocaleString() : fmtNum(v)}</div>
                                        <div className="intel-kpi-lbl">{k}</div>
                                    </div>
                                ))}
                            </div>

                            <div className="intel-main-split">
                                {/* Top 3 Multi-variate Charts */}
                                <div className="intel-top-charts" style={{ flex: persona === 'executive' ? '1.5' : '1' }}>
                                    <div className="section-label">
                                        {persona === 'data_scientist' ? '🔬 Statistical Distributions' : '📈 Executive Visualizations'}
                                        <span className="intel-badge">AI Selected</span>
                                    </div>
                                    <div className="intel-chart-deck">
                                        <IntelChartDeck charts={displayCharts.slice(0, persona === 'executive' ? 4 : 3)} />
                                    </div>
                                </div>

                                {/* Insights Feed (Hidden for Data Scientists usually) */}
                                {persona !== 'data_scientist' && (
                                    <div className="intel-insights-feed">
                                        <div className="section-label">
                                            <Lightbulb size={14} style={{ color: '#f59e0b', marginRight: '6px' }} />
                                            {persona === 'executive' ? 'Key Takeaways' : 'Automated Insights Feed'}
                                        </div>
                                        <div className="intel-feed-list">
                                            {(intelligence.insights_feed || []).slice(0, persona === 'executive' ? 5 : 10).map((insight, i) => (
                                                <div key={i} className="intel-feed-card">
                                                    <div className="feed-card-header">
                                                        {persona !== 'executive' && <span className="feed-relevance">{insight.relevance}% score</span>}
                                                        <h4 className="feed-title">{insight.title}</h4>
                                                    </div>
                                                    <p className="feed-text">{insight.text}</p>
                                                    <button className="feed-jump-btn" onClick={() => setActiveTab('charts')}>
                                                        View Chart <ArrowRight size={12} style={{ marginLeft: '4px' }} />
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    ) : filteredRows.length === 0 ? (
                        <div className="no-stats">No rows match the current filters.</div>
                    ) : (
                        <>
                            {stats.length ? (
                                <>
                                    <div className="section-label">📊 Key Metrics ({filteredRows.length.toLocaleString()} rows · {selSize} cols)</div>
                                    <div className="kpi-ribbon">
                                        {stats.map((s, i) => (
                                            <KpiCard key={s.col} stat={s} color={PALETTE[i % PALETTE.length]} />
                                        ))}
                                    </div>
                                </>
                            ) : (
                                <div className="no-stats">No numeric columns selected — choose numeric columns to see KPIs.</div>
                            )}

                            {charts.length > 0 && (
                                <>
                                    <div className="section-label">📈 Quick Views</div>
                                    <div className="deck-grid">
                                        {charts.slice(0, 2).map((ch, i) => {
                                            const color = PALETTE[i % PALETTE.length];
                                            if (ch.type !== 'line' && ch.type !== 'multiline') return null;
                                            return (
                                                <div className="deck-card span-2" key={i}>
                                                    <div className="deck-card-header">
                                                        <TrendingUp size={16} className="deck-icon" style={{ color }} />
                                                        <h3>{ch.title}</h3>
                                                    </div>
                                                    <ResponsiveContainer width="100%" height={200}>
                                                        <AreaChart data={ch.data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                                            <defs>
                                                                <linearGradient id={`ovg${i}`} x1="0" y1="0" x2="0" y2="1">
                                                                    <stop offset="5%" stopColor={color} stopOpacity={0.25} />
                                                                    <stop offset="95%" stopColor={color} stopOpacity={0} />
                                                                </linearGradient>
                                                            </defs>
                                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                                                            <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                                                            <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} tickFormatter={v => fmtNum(v, 0)} />
                                                            <ChartTooltip content={<PowerTooltip />} />
                                                            <Area type="monotone" dataKey={ch.valueCol} stroke={color} strokeWidth={2} fill={`url(#ovg${i})`} dot={false} />
                                                        </AreaChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </>
                            )}
                        </>
                    )}
                </div>
            )}

            {/* ── Charts ────────────────────────────────────────────────── */}
            {activeTab === 'charts' && (
                <div className="tab-pane">
                    {intelligence ? (
                        <>
                            <div className="section-label">
                                🎯 Statistically Scored Analysis ({displayCharts.length} total visualisations)
                            </div>
                            <p className="rpt-section-hint">
                                The combinatorial ML engine scanned combinations across data types
                                and ranked these charts by mathematical significance (variance, correlation).
                            </p>
                            <IntelChartDeck charts={displayCharts} />
                        </>
                    ) : filteredRows.length === 0 ? (
                        <div className="no-stats">No rows match the current filters.</div>
                    ) : charts.length ? (
                        <>
                            <div className="section-label">
                                🎯 Generic Auto-Generated Charts ({charts.length}) — {filteredRows.length.toLocaleString()} rows · {selSize} cols
                            </div>
                            <ChartDeck charts={charts} />
                        </>
                    ) : (
                        <div className="no-stats">Try selecting more columns (numeric + categorical or date) to generate charts.</div>
                    )}
                </div>
            )}

            {/* ── Schema ────────────────────────────────────────────────── */}
            {activeTab === 'schema' && schema && (
                <div className="tab-pane">
                    <div className="section-label">🗄️ Detected Schema — click checkboxes to include/exclude columns</div>
                    <div className="schema-summary">
                        <span className="scm-pill numeric">{activeSchema?.numericCols.length ?? 0} Numeric</span>
                        <span className="scm-pill categorical">{activeSchema?.categoricalCols.length ?? 0} Categorical</span>
                        <span className="scm-pill temporal">{activeSchema?.temporalCols.length ?? 0} Temporal</span>
                        <span className="scm-pill text">{activeSchema?.textCols.length ?? 0} Text</span>
                    </div>
                    <SchemaTable schema={schema} selectedCols={selectedCols} onToggleCol={toggleCol} />
                </div>
            )}

            {/* ── Narrative ─────────────────────────────────────────────── */}
            {activeTab === 'narrative' && (
                <div className="tab-pane">
                    {reportData?.narrative ? (
                        <div className="narrative-block">
                            <h2>{reportData.narrative.title}</h2>
                            <p className="narrative-body">{reportData.narrative.body}</p>
                        </div>
                    ) : (
                        <div className="no-stats">No LLM narrative available for this run.</div>
                    )}
                    {reportData?.final_result?.report_path && (
                        <div className="report-path-hint">
                            📄 Report saved to: <code>{reportData.final_result.report_path}</code>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default Dashboard;
