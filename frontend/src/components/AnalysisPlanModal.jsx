/**
 * AnalysisPlanModal.jsx — v2 (Analyst Instruction Loop)
 * --------------------------------------------------------
 * Pre-Analysis Approval modal that shows what ADAP will do BEFORE running.
 * Displays: instruction summary, data summary, operations, regulatory rules, warnings.
 * User can Approve (starts full pipeline), Reject (try a different approach), or Cancel.
 *
 * Props:
 *   plan               : { data_summary, domain, operations, warnings, instruction_summary } — from /api/pipeline/preview-plan
 *   onApprove          : () => void — start the pipeline
 *   onCancel           : () => void — dismiss the modal
 *   onReject           : () => void — user wants a different plan (increments rejection count)
 *   isLoading          : boolean    — while plan is being generated
 *   instructionSummary : string[]   — what the AI understood from analyst instructions
 *   rejectionCount     : number     — how many times user has said "try again"
 */

import React from 'react';
import './AnalysisPlanModal.css';

// ── Operation status icons ────────────────────────────────────────────────────
const OP_ICONS = {
  null_imputation:      '🧹',
  outlier_detection:    '🎯',
  pii_scan:             '🔒',
  regulatory_compliance:'⚖️',
  automl:               '🤖',
  unsupervised:         '🔬',
  feature_engineering:  '⚙️',
  schema_validation:    '📋',
};

const OP_COLORS = {
  planned:                { bg: 'rgba(99,102,241,0.1)',  border: 'rgba(99,102,241,0.25)', text: '#818cf8' },
  skipped:                { bg: 'rgba(100,116,139,0.1)', border: 'rgba(100,116,139,0.2)', text: '#64748b' },
  warning:                { bg: 'rgba(245,158,11,0.1)',  border: 'rgba(245,158,11,0.25)', text: '#f59e0b' },
  'skipped (per instruction)': { bg: 'rgba(56,189,248,0.06)', border: 'rgba(56,189,248,0.2)', text: '#38bdf8' },
};

const WARN_STYLES = {
  warning: { bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.3)', icon: '⚠️', text: '#f59e0b' },
  info:    { bg: 'rgba(99,102,241,0.08)', border: 'rgba(99,102,241,0.25)', icon: 'ℹ️', text: '#818cf8' },
  error:   { bg: 'rgba(239,68,68,0.08)',  border: 'rgba(239,68,68,0.3)',   icon: '🚨', text: '#ef4444' },
};

const AnalysisPlanModal = ({
  plan,
  onApprove,
  onCancel,
  onReject,
  isLoading = false,
  instructionSummary = [],
  rejectionCount = 0,
}) => {
  if (!isLoading && !plan) return null;

  const summary     = plan?.data_summary || {};
  const domain      = plan?.domain || {};
  const ops         = plan?.operations || [];
  const warns       = plan?.warnings || [];
  const hasCritical = warns.some(w => w.level === 'error');

  // Merge instruction_summary from plan (server) with the prop (parent state)
  const allInstructionSummary = [
    ...(instructionSummary || []),
    ...(plan?.instruction_summary || []),
  ].filter((v, i, a) => a.indexOf(v) === i); // deduplicate

  return (
    <div className="apm-overlay" role="dialog" aria-modal="true" aria-label="Pre-Analysis Plan">
      <div className="apm-modal">
        {/* Header */}
        <div className="apm-header">
          <div className="apm-header-icon">📋</div>
          <div>
            <h2 className="apm-title">Pre-Analysis Plan</h2>
            <p className="apm-subtitle">
              Review what ADAP will do before committing to a run
              {rejectionCount > 0 && (
                <span className="apm-rejection-badge"> · {rejectionCount} revision{rejectionCount !== 1 ? 's' : ''}</span>
              )}
            </p>
          </div>
          <button className="apm-close" onClick={onCancel} aria-label="Close">✕</button>
        </div>

        {isLoading ? (
          <div className="apm-loading">
            <div className="apm-spinner" />
            <p>
              {rejectionCount > 0
                ? `Generating revised plan (attempt ${rejectionCount + 1})…`
                : 'Analysing dataset schema — this takes 2-5 seconds…'}
            </p>
          </div>
        ) : (
          <>
            {/* ── Instruction Intelligence Summary ──────────────────────────── */}
            {allInstructionSummary.length > 0 && (
              <div className="apm-instruction-summary">
                <div className="apm-instruction-summary-header">
                  <span className="apm-instruction-brain">🧠</span>
                  <span>AI understood your instructions</span>
                </div>
                <ul className="apm-instruction-list">
                  {allInstructionSummary.map((item, i) => (
                    <li key={i} className="apm-instruction-item">
                      <span className="apm-instruction-check">✓</span>
                      <span dangerouslySetInnerHTML={{ __html: item.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Data Summary */}
            <div className="apm-section">
              <div className="apm-section-title">📊 Dataset Summary</div>
              <div className="apm-summary-grid">
                <div className="apm-stat">
                  <span className="apm-stat-val">{(summary.n_rows || 0).toLocaleString()}</span>
                  <span className="apm-stat-label">Rows</span>
                </div>
                <div className="apm-stat">
                  <span className="apm-stat-val">{summary.n_cols || '—'}</span>
                  <span className="apm-stat-label">Columns</span>
                </div>
                <div className="apm-stat">
                  <span className="apm-stat-val" style={{ color: (summary.overall_null_pct || 0) > 20 ? '#f59e0b' : '#e2e8f0' }}>
                    {(summary.overall_null_pct || 0).toFixed(1)}%
                  </span>
                  <span className="apm-stat-label">Null Rate</span>
                </div>
                <div className="apm-stat">
                  <span className="apm-stat-val">{summary.numeric_cols || 0}</span>
                  <span className="apm-stat-label">Numeric</span>
                </div>
                <div className="apm-stat">
                  <span className="apm-stat-val">{summary.categorical_cols || 0}</span>
                  <span className="apm-stat-label">Categorical</span>
                </div>
                <div className="apm-stat">
                  <span className="apm-stat-val" style={{ color: summary.duplicate_rows > 0 ? '#f59e0b' : '#e2e8f0' }}>
                    {summary.duplicate_rows || 0}
                  </span>
                  <span className="apm-stat-label">Duplicates</span>
                </div>
              </div>

              {summary.columns_to_drop?.length > 0 && (
                <div className="apm-drop-warn">
                  <span style={{ color: '#f59e0b', fontWeight: 700 }}>⚠️ Auto-drop:</span>{' '}
                  {summary.columns_to_drop.join(', ')} will be removed ({'>'} 90% null)
                </div>
              )}

              {summary.rows_to_quarantine_est > 0 && (
                <div className="apm-drop-warn" style={{ borderColor: 'rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.05)' }}>
                  <span style={{ color: '#ef4444', fontWeight: 700 }}>🔴 ~{summary.rows_to_quarantine_est} rows</span>{' '}
                  may be quarantined ({'>'} 80% null per row)
                </div>
              )}
            </div>

            {/* Domain / Regulatory */}
            <div className="apm-section">
              <div className="apm-section-title">⚖️ Regulatory Scope</div>
              <div className="apm-domain-row">
                <div className="apm-domain-badge">
                  {domain.active || 'generic'}
                </div>
                {domain.detected?.length > 0 && domain.detected[0] !== domain.selected && (
                  <span className="apm-auto-detect">🤖 auto-detected from column names</span>
                )}
                {domain.rules_count > 0 && (
                  <span className="apm-rule-count">{domain.rules_count} rules</span>
                )}
              </div>
              {domain.rules?.length > 0 && (
                <div className="apm-rule-list">
                  {domain.rules.slice(0, 6).map((r, i) => (
                    <span key={i} className="apm-rule-chip">⚖️ {r}</span>
                  ))}
                  {domain.rules.length > 6 && (
                    <span className="apm-rule-chip apm-rule-chip-more">+{domain.rules.length - 6} more</span>
                  )}
                </div>
              )}
            </div>

            {/* Operations */}
            <div className="apm-section">
              <div className="apm-section-title">⚙️ Planned Operations ({ops.length})</div>
              <div className="apm-ops-list">
                {ops.map((op, i) => {
                  const statusKey = op.status?.toLowerCase() || 'planned';
                  const st = OP_COLORS[statusKey] || OP_COLORS.planned;
                  return (
                    <div key={i} className="apm-op-item" style={{ background: st.bg, border: `1px solid ${st.border}` }}>
                      <span className="apm-op-icon">{OP_ICONS[op.op] || '⚙️'}</span>
                      <div className="apm-op-text">
                        <div className="apm-op-label">{op.label}</div>
                        {op.detail && <div className="apm-op-detail">{op.detail}</div>}
                      </div>
                      <span className="apm-op-status" style={{ color: st.text }}>{op.status}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Warnings */}
            {warns.length > 0 && (
              <div className="apm-section">
                <div className="apm-section-title">⚠️ Notices ({warns.length})</div>
                <div className="apm-warns-list">
                  {warns.map((w, i) => {
                    const ws = WARN_STYLES[w.level] || WARN_STYLES.info;
                    return (
                      <div key={i} className="apm-warn-item" style={{ background: ws.bg, border: `1px solid ${ws.border}` }}>
                        <span>{ws.icon}</span>
                        <span style={{ color: ws.text, fontSize: '0.82rem' }}>{w.message}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}

        {/* Footer */}
        <div className="apm-footer">
          <button className="apm-btn cancel" onClick={onCancel}>
            Cancel
          </button>
          {!isLoading && onReject && (
            <button
              className="apm-btn reject"
              onClick={onReject}
              id="apm-reject-btn"
              title="Generate a revised plan and record this rejection in the RL system"
            >
              🔄 Try different approach
            </button>
          )}
          {!isLoading && (
            <button
              className={`apm-btn approve ${hasCritical ? 'warn' : ''}`}
              onClick={onApprove}
              id="apm-approve-btn"
            >
              {hasCritical ? '⚠️ Approve Anyway' : '✅ Approve & Run Pipeline'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisPlanModal;
