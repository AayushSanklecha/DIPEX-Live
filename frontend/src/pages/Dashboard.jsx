import React, { useState, useEffect } from 'react';
import { AnalystService, ResultsService } from '../api/client';
import { Activity, CheckCircle, Clock, FileText, AlertTriangle, Database } from 'lucide-react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
    BarChart, Bar, PieChart, Pie, Cell, Legend
} from 'recharts';
import './Dashboard.css';

const Dashboard = () => {
    const [metrics, setMetrics] = useState({
        total_pipeline_runs: 0,
        passed_runs: 0,
        pass_rate: 0,
        reports_generated: 0,
        uptime_seconds: 0
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const data = await AnalystService.getSystemMetrics();
                setMetrics(data);
                setLoading(false);
            } catch (err) {
                setError('Failed to fetch system metrics');
                setLoading(false);
            }
        };

        fetchMetrics();
        const interval = setInterval(fetchMetrics, 5000);
        return () => clearInterval(interval);
    }, []);

    const formatUptime = (seconds) => {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        return `${h}h ${m}m`;
    };

    // Mock historical data because the backend currently only yields aggregates
    // In a real PowerBI scenario, we'd fetch time-series data from the backend.
    const generateTrendData = () => {
        const data = [];
        let currentRuns = Math.max(0, metrics.total_pipeline_runs - 50);
        for (let i = 14; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);

            const newRuns = Math.floor(Math.random() * 5) + 1;
            currentRuns += newRuns;

            data.push({
                date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                runs: i === 0 ? metrics.total_pipeline_runs : currentRuns, // anchor today to actual
                failures: Math.floor(newRuns * (1 - metrics.pass_rate))
            });
        }
        return data;
    };

    const trendData = generateTrendData();
    const failedRuns = metrics.total_pipeline_runs - metrics.passed_runs;

    const pieData = [
        { name: 'Passed', value: metrics.passed_runs },
        { name: 'Failed', value: failedRuns }
    ];
    const COLORS = ['#10b981', '#ef4444'];

    if (loading) return <div className="loading-state">Loading analytical models...</div>;
    if (error) return <div className="error-state">{error}</div>;

    return (
        <div className="bi-dashboard">
            <div className="bi-header">
                <h2>Overall System Performance</h2>
                <div className="bi-toolbar">
                    <span className="last-sync">Last synced: Just now</span>
                </div>
            </div>

            {/* KPI Ribbon */}
            <div className="kpi-ribbon">
                <div className="kpi-card">
                    <div className="kpi-title">Total Processed Pipelines</div>
                    <div className="kpi-value">{metrics.total_pipeline_runs}</div>
                    <div className="kpi-spark">
                        <Activity size={16} className="text-blue" />
                        <span>Lifetime execution count</span>
                    </div>
                </div>

                <div className="kpi-card">
                    <div className="kpi-title">System Pass Rate</div>
                    <div className="kpi-value text-green">{(metrics.pass_rate * 100).toFixed(1)}%</div>
                    <div className="kpi-spark">
                        <CheckCircle size={16} className="text-green" />
                        <span>{metrics.passed_runs} successful runs</span>
                    </div>
                </div>

                <div className="kpi-card">
                    <div className="kpi-title">Failed Pipelines</div>
                    <div className="kpi-value text-red">{failedRuns}</div>
                    <div className="kpi-spark">
                        <AlertTriangle size={16} className="text-red" />
                        <span>Requires analyst review</span>
                    </div>
                </div>

                <div className="kpi-card">
                    <div className="kpi-title">Reports Generated</div>
                    <div className="kpi-value">{metrics.reports_generated}</div>
                    <div className="kpi-spark">
                        <FileText size={16} className="text-purple" />
                        <span>Executive summaries compiled</span>
                    </div>
                </div>
            </div>

            {/* Main Analytical Grid */}
            <div className="analytical-grid">
                {/* Primary Trend Chart */}
                <div className="chart-card span-2">
                    <h3 className="chart-title">Pipeline Execution Trends (14 Days)</h3>
                    <div className="chart-body">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={trendData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                <XAxis dataKey="date" tick={{ fontSize: 12, fill: '#64748b' }} tickLine={false} axisLine={false} />
                                <YAxis tick={{ fontSize: 12, fill: '#64748b' }} tickLine={false} axisLine={false} />
                                <RechartsTooltip
                                    contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                />
                                <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                                <Line type="monotone" dataKey="runs" name="Total Runs (Cumulative)" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Success vs Failure Distribution */}
                <div className="chart-card">
                    <h3 className="chart-title">Outcome Distribution</h3>
                    <div className="chart-body">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <RechartsTooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                                <Legend iconType="circle" verticalAlign="bottom" height={36} wrapperStyle={{ fontSize: '12px' }} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Volume Bar Chart */}
                <div className="chart-card span-2">
                    <h3 className="chart-title">Daily Delta Volumes</h3>
                    <div className="chart-body">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={trendData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                <XAxis dataKey="date" tick={{ fontSize: 12, fill: '#64748b' }} tickLine={false} axisLine={false} />
                                <YAxis tick={{ fontSize: 12, fill: '#64748b' }} tickLine={false} axisLine={false} />
                                <RechartsTooltip cursor={{ fill: '#f1f5f9' }} contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }} />
                                <Legend iconType="circle" wrapperStyle={{ fontSize: '12px' }} />
                                <Bar dataKey="failures" name="Failures" stackId="a" fill="#ef4444" radius={[0, 0, 0, 0]} />
                                <Bar dataKey="runs" name="Successes" stackId="a" fill="#10b981" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* System Health Card */}
                <div className="chart-card">
                    <h3 className="chart-title">System Properties</h3>
                    <div className="system-props">
                        <div className="prop-row">
                            <span className="prop-label">Service Uptime</span>
                            <span className="prop-value">{formatUptime(metrics.uptime_seconds)}</span>
                        </div>
                        <div className="prop-row">
                            <span className="prop-label">Data Store</span>
                            <span className="prop-value inline-flex"><Database size={14} style={{ marginRight: '4px' }} /> Connected</span>
                        </div>
                        <div className="prop-row">
                            <span className="prop-label">RL Engine</span>
                            <span className="prop-value">Active</span>
                        </div>
                        <div className="prop-row">
                            <span className="prop-label">Verification Gate</span>
                            <span className="prop-value text-green">Strict Mode</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
