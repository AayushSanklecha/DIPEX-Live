import React, { useState, useEffect } from 'react';
import { ResultsService } from '../api/client';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import { Search, Loader2, AlertTriangle, FileCheck, Info } from 'lucide-react';
import './Reports.css';

const Reports = () => {
    const [runId, setRunId] = useState('');
    const [reportData, setReportData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Load latest on mount
    useEffect(() => {
        fetchLatest();
    }, []);

    const fetchLatest = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await ResultsService.getLatestResult();
            if (!data) {
                setError('No previous pipeline runs found.');
            } else {
                setReportData(data);
                setRunId(data.run_id);
            }
        } catch (err) {
            setError('Failed to fetch the latest report.');
        } finally {
            setLoading(false);
        }
    };

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!runId.trim()) return;

        setLoading(true);
        setError(null);
        try {
            const data = await ResultsService.getResult(runId);
            setReportData(data);
        } catch (err) {
            setError('Run ID not found or could not be retrieved.');
            setReportData(null);
        } finally {
            setLoading(false);
        }
    };

    const formatRadarData = (dimensions) => {
        if (!dimensions) return [];
        return [
            { subject: 'Data Quality', A: dimensions.data_quality * 100, fullMark: 100 },
            { subject: 'Statistical Strength', A: dimensions.statistical_strength * 100, fullMark: 100 },
            { subject: 'Stability', A: dimensions.stability * 100, fullMark: 100 },
            { subject: 'Compliance', A: dimensions.compliance * 100, fullMark: 100 },
        ];
    };

    return (
        <div className="reports-container">
            <div className="search-bar">
                <form onSubmit={handleSearch}>
                    <div className="search-input-wrapper">
                        <Search className="search-icon" />
                        <input
                            type="text"
                            placeholder="Search by Run ID (e.g. 8a7b6c5d)"
                            value={runId}
                            onChange={(e) => setRunId(e.target.value)}
                        />
                        <button type="submit" className="btn-search" disabled={loading}>
                            {loading ? <Loader2 className="spin" /> : 'Lookup'}
                        </button>
                    </div>
                </form>
            </div>

            {error && (
                <div className="report-error">
                    <AlertTriangle />
                    <span>{error}</span>
                </div>
            )}

            {loading && !error && (
                <div className="report-loading">
                    <Loader2 className="spin large" />
                    <p>Analyzing pipeline telemetry...</p>
                </div>
            )}

            {!loading && reportData && (
                <div className="report-content">
                    <div className="report-header">
                        <div className="header-title">
                            <h2>Run: {reportData.run_id}</h2>
                            <span className={`status-badge ${reportData.status.toLowerCase()}`}>
                                {reportData.status}
                            </span>
                        </div>
                        <div className="header-meta">
                            <span>Dataset: <strong>{reportData.dataset_id}</strong></span>
                            <span>Rows: <strong>{reportData.row_count?.toLocaleString() || 0}</strong></span>
                            <span>Columns: <strong>{reportData.col_count?.toLocaleString() || 0}</strong></span>
                        </div>
                    </div>

                    <div className="report-grid">
                        <div className="narrative-panel">
                            <div className="panel-heading">
                                <FileCheck className="heading-icon" />
                                <h3>Executive Summary</h3>
                            </div>
                            <div className="narrative-body">
                                {reportData.narrative ? (
                                    <>
                                        <h4>{reportData.narrative.title}</h4>
                                        <p>{reportData.narrative.body}</p>
                                        <div className="gate-metrics">
                                            <div className={`gate-pill ${reportData.gate1_decision?.toLowerCase()}`}>
                                                Gate 1 (Quality): {reportData.gate1_decision}
                                            </div>
                                            <div className={`gate-pill ${reportData.gate2_decision?.toLowerCase()}`}>
                                                Gate 2 (Stats): {reportData.gate2_decision}
                                            </div>
                                        </div>
                                    </>
                                ) : (
                                    <p className="no-narrative">No narrative generated for this run.</p>
                                )}
                            </div>

                            <div className="confidence-score-box">
                                <span className="score-label">Aggregated Confidence</span>
                                <span className="score-value">
                                    {((reportData.confidence_score || 0) * 100).toFixed(1)}%
                                </span>
                            </div>
                        </div>

                        <div className="radar-panel">
                            <div className="panel-heading">
                                <Info className="heading-icon" />
                                <h3>Dimension Analysis</h3>
                            </div>
                            <div className="radar-wrapper">
                                {reportData.dimensions ? (
                                    <ResponsiveContainer width="100%" height={300}>
                                        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={formatRadarData(reportData.dimensions)}>
                                            <PolarGrid stroke="#e2e8f0" />
                                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 12 }} />
                                            <PolarRadiusAxis
                                                angle={30}
                                                domain={[0, 100]}
                                                tick={{ fill: '#94a3b8', fontSize: 10 }}
                                                tickCount={5}
                                            />
                                            <Radar
                                                name="Run metrics"
                                                dataKey="A"
                                                stroke="#3b82f6"
                                                fill="#3b82f6"
                                                fillOpacity={0.5}
                                            />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="no-dimensions">
                                        <p>Dimension data not available for this run.</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Reports;
