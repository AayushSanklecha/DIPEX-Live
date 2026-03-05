import React, { useState } from 'react';
import { AnalystService } from '../api/client';
import { Upload, Play, CheckCircle, AlertCircle } from 'lucide-react';
import './RunPipeline.css';

const RunPipeline = () => {
    const [datasetId, setDatasetId] = useState('Invistico_Airline');
    const [running, setRunning] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleRun = async (e) => {
        e.preventDefault();
        if (!datasetId.trim()) return;

        setRunning(true);
        setError(null);
        setResult(null);

        try {
            const response = await AnalystService.runPipeline(datasetId);
            setResult(response.status || 'Success');
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to start pipeline');
        } finally {
            setRunning(false);
        }
    };

    return (
        <div className="run-pipeline-container">
            <div className="configuration-panel">
                <div className="panel-header">
                    <Upload className="panel-icon" />
                    <h2>Configure Pipeline Run</h2>
                </div>

                <form onSubmit={handleRun} className="pipeline-form">
                    <div className="form-group">
                        <label htmlFor="datasetId">Dataset Identifier</label>
                        <input
                            type="text"
                            id="datasetId"
                            value={datasetId}
                            onChange={(e) => setDatasetId(e.target.value)}
                            placeholder="e.g. Invistico_Airline"
                            disabled={running}
                        />
                        <span className="help-text">Enter the dataset name to process.</span>
                    </div>

                    <button
                        type="submit"
                        className={`btn-run ${running ? 'running' : ''}`}
                        disabled={running || !datasetId.trim()}
                    >
                        {running ? (
                            <>Running Pipeline...</>
                        ) : (
                            <><Play className="btn-icon" /> Execute Pipeline</>
                        )}
                    </button>
                </form>

                {error && (
                    <div className="status-message error">
                        <AlertCircle className="status-icon" />
                        <div className="status-content">
                            <strong>Execution Failed</strong>
                            <p>{error}</p>
                        </div>
                    </div>
                )}

                {result && (
                    <div className="status-message success">
                        <CheckCircle className="status-icon" />
                        <div className="status-content">
                            <strong>Execution Completed</strong>
                            <p>Pipeline finished with status: {result}</p>
                            <a href="/reports" className="view-reports-link">View Detailed Report →</a>
                        </div>
                    </div>
                )}
            </div>

            <div className="info-panel">
                <h3>Pipeline Workflow</h3>
                <ol className="workflow-steps">
                    <li><strong>Universal Intake:</strong> Automatic schema inference and typing.</li>
                    <li><strong>Data Cleansing:</strong> Imputation, encoding, and standardization.</li>
                    <li><strong>Quality Gates:</strong> Strict validation of statistical thresholds.</li>
                    <li><strong>Model Execution:</strong> Training and evaluation.</li>
                    <li><strong>Report Generation:</strong> Automated insights and executive summaries.</li>
                </ol>
            </div>
        </div>
    );
};

export default RunPipeline;
