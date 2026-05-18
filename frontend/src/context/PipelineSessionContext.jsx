/**
 * PipelineSessionContext.jsx
 * ---------------------------
 * Data State Awareness (DSA) Context — tracks pipeline session state
 * across all frontend components.
 *
 * Provides:
 *   - Bronze/Silver/Gold tier tracking
 *   - Pre-analysis plan approval workflow
 *   - Data freshness indicators
 *   - Stage completion tracking
 *
 * Usage:
 *   // Wrap app in provider:
 *   <PipelineSessionProvider>...</PipelineSessionProvider>
 *
 *   // In any component:
 *   const { session, setTier, approvePlan } = usePipelineSession();
 */

import React, { createContext, useContext, useReducer, useCallback } from 'react';

// Use relative URL so calls work on HF Spaces, localhost, and any host.
// The built frontend is served by FastAPI on the same origin, so /pipeline/...
// resolves correctly without needing a hardcoded host.
// window.ADAP_API_BASE can still override this for split-host setups.
const API = window.ADAP_API_BASE || '';

// ── State shape ───────────────────────────────────────────────────────────────

const INITIAL_STATE = {
  // Current pipeline session
  runId: null,
  status: 'idle',          // idle | plan_pending | running | completed | error
  progress: 0,             // 0-100

  // Data State Awareness tiers
  tier: null,              // null | 'bronze' | 'silver' | 'gold'
  tierTimestamps: {
    bronze: null,
    silver: null,
    gold:   null,
  },

  // Pre-analysis plan
  plan: null,
  planApproved: false,
  planLoading: false,

  // Stage tracking
  completedStages: [],
  currentStage: null,
  stageLog: [],

  // Latest result snippet
  lastResult: null,
  confidence: null,
  gate: null,

  // Global config for this session
  domain: 'generic',
  targetCol: null,
  mode: 'auto',
};

// ── Actions ───────────────────────────────────────────────────────────────────

const ACTIONS = {
  SET_PLAN_LOADING:  'SET_PLAN_LOADING',
  SET_PLAN:          'SET_PLAN',
  APPROVE_PLAN:      'APPROVE_PLAN',
  CANCEL_PLAN:       'CANCEL_PLAN',
  START_RUN:         'START_RUN',
  UPDATE_PROGRESS:   'UPDATE_PROGRESS',
  ADVANCE_STAGE:     'ADVANCE_STAGE',
  SET_TIER:          'SET_TIER',
  COMPLETE_RUN:      'COMPLETE_RUN',
  ERROR_RUN:         'ERROR_RUN',
  RESET_SESSION:     'RESET_SESSION',
  SET_CONFIG:        'SET_CONFIG',
};

// ── Reducer ───────────────────────────────────────────────────────────────────

const sessionReducer = (state, action) => {
  switch (action.type) {
    case ACTIONS.SET_PLAN_LOADING:
      return { ...state, planLoading: action.payload };

    case ACTIONS.SET_PLAN:
      return {
        ...state,
        plan: action.payload,
        planLoading: false,
        status: 'plan_pending',
      };

    case ACTIONS.APPROVE_PLAN:
      return {
        ...state,
        planApproved: true,
        status: 'running',
        progress: 0,
        completedStages: [],
        currentStage: 'ingestion',
      };

    case ACTIONS.CANCEL_PLAN:
      return {
        ...state,
        plan: null,
        planApproved: false,
        planLoading: false,
        status: 'idle',
      };

    case ACTIONS.START_RUN:
      return {
        ...state,
        runId: action.payload.runId,
        status: 'running',
        progress: 5,
        tier: null,
        completedStages: [],
        stageLog: [],
        lastResult: null,
        confidence: null,
        gate: null,
      };

    case ACTIONS.UPDATE_PROGRESS:
      return {
        ...state,
        progress: Math.min(action.payload, 100),
      };

    case ACTIONS.ADVANCE_STAGE: {
      const { stage, result } = action.payload;
      const stageEntry = { stage, timestamp: new Date().toISOString(), result };
      return {
        ...state,
        currentStage: stage,
        completedStages: [...state.completedStages, stage],
        stageLog: [...state.stageLog, stageEntry],
      };
    }

    case ACTIONS.SET_TIER: {
      const tier = action.payload;
      const now = new Date().toISOString();
      const newTimestamps = { ...state.tierTimestamps, [tier]: now };
      // Ensure monotonic tier progression (bronze → silver → gold)
      const tierOrder = ['bronze', 'silver', 'gold'];
      const currentIdx = tierOrder.indexOf(tier);
      // Only advance, never regress
      const effectiveTier = currentIdx >= tierOrder.indexOf(state.tier || 'bronze') ? tier : state.tier;
      return {
        ...state,
        tier: effectiveTier,
        tierTimestamps: newTimestamps,
      };
    }

    case ACTIONS.COMPLETE_RUN:
      return {
        ...state,
        status: 'completed',
        progress: 100,
        tier: 'gold',
        currentStage: null,
        lastResult: action.payload.result,
        confidence: action.payload.confidence,
        gate: action.payload.gate,
        tierTimestamps: {
          ...state.tierTimestamps,
          gold: new Date().toISOString(),
        },
      };

    case ACTIONS.ERROR_RUN:
      return {
        ...state,
        status: 'error',
        currentStage: null,
        lastResult: action.payload,
      };

    case ACTIONS.RESET_SESSION:
      return { ...INITIAL_STATE };

    case ACTIONS.SET_CONFIG:
      return {
        ...state,
        domain: action.payload.domain || state.domain,
        targetCol: action.payload.targetCol !== undefined ? action.payload.targetCol : state.targetCol,
        mode: action.payload.mode || state.mode,
      };

    default:
      return state;
  }
};

// ── Context ───────────────────────────────────────────────────────────────────

const PipelineSessionContext = createContext(null);

export const PipelineSessionProvider = ({ children }) => {
  const [session, dispatch] = useReducer(sessionReducer, INITIAL_STATE);

  // ── Config ─────────────────────────────────────────────────────────────────
  const setConfig = useCallback((config) => {
    dispatch({ type: ACTIONS.SET_CONFIG, payload: config });
  }, []);

  // ── Plan workflow ───────────────────────────────────────────────────────────
  const fetchPreviewPlan = useCallback(async (formData) => {
    dispatch({ type: ACTIONS.SET_PLAN_LOADING, payload: true });
    try {
      const res = await fetch(`${API}/pipeline/preview-plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          domain: formData.domain || 'generic',
          target_col: formData.target_col || null,
          mode: formData.mode || 'auto',
          user_context: formData.user_context || '',
          // Pass column names for analysis if we have them
          column_names: formData.column_names || [],
          n_rows: formData.n_rows || null,
          n_cols: formData.n_cols || null,
          null_rate: formData.null_rate || null,
        }),
      });
      const plan = await res.json();
      dispatch({ type: ACTIONS.SET_PLAN, payload: plan });
      return plan;
    } catch (err) {
      console.warn('[PipelineSession] Preview plan failed, using fallback:', err.message);
      // Generate a basic plan without the API
      const fallbackPlan = {
        data_summary: {
          n_rows: formData.n_rows || 0,
          n_cols: formData.n_cols || 0,
          overall_null_pct: (formData.null_rate || 0) * 100,
          numeric_cols: 0,
          categorical_cols: 0,
          duplicate_rows: 0,
          columns_to_drop: [],
          rows_to_quarantine_est: 0,
          target_col: formData.target_col || null,
        },
        domain: {
          selected: formData.domain || 'generic',
          detected: [],
          active: formData.domain || 'generic',
          rules_count: 0,
          rules: [],
        },
        operations: [
          { op: 'pii_scan', label: 'PII Scan', detail: 'Email, SSN, credit card detection', status: 'planned' },
          { op: 'null_imputation', label: 'Null Imputation', detail: 'Median/KNN imputation', status: 'planned' },
          { op: 'outlier_detection', label: 'Outlier Detection', detail: 'IQR winsorize', status: 'planned' },
          formData.target_col
            ? { op: 'automl', label: 'AutoML', detail: 'XGBoost + LightGBM 5-fold CV', status: 'planned' }
            : { op: 'unsupervised', label: 'Unsupervised Analysis', detail: 'Isolation Forest + K-Means', status: 'planned' },
        ],
        warnings: [],
        plan_elapsed_ms: 0,
      };
      dispatch({ type: ACTIONS.SET_PLAN, payload: fallbackPlan });
      return fallbackPlan;
    }
  }, []);

  const approvePlan = useCallback(() => {
    dispatch({ type: ACTIONS.APPROVE_PLAN });
  }, []);

  const cancelPlan = useCallback(() => {
    dispatch({ type: ACTIONS.CANCEL_PLAN });
  }, []);

  // ── Run lifecycle ───────────────────────────────────────────────────────────
  const startRun = useCallback((runId) => {
    dispatch({ type: ACTIONS.START_RUN, payload: { runId } });
  }, []);

  const updateProgress = useCallback((pct) => {
    dispatch({ type: ACTIONS.UPDATE_PROGRESS, payload: pct });
  }, []);

  const advanceStage = useCallback((stage, result = null) => {
    dispatch({ type: ACTIONS.ADVANCE_STAGE, payload: { stage, result } });
    // Auto-advance tier based on stage
    if (['ingestion', 'schema_validation'].includes(stage)) {
      dispatch({ type: ACTIONS.SET_TIER, payload: 'bronze' });
    } else if (['preprocessing', 'quality_gate1', 'analyst_brain'].includes(stage)) {
      dispatch({ type: ACTIONS.SET_TIER, payload: 'silver' });
    } else if (['model_training', 'confidence_vector'].includes(stage)) {
      dispatch({ type: ACTIONS.SET_TIER, payload: 'gold' });
    }
  }, []);

  const completeRun = useCallback((result, confidence, gate) => {
    dispatch({ type: ACTIONS.COMPLETE_RUN, payload: { result, confidence, gate } });
  }, []);

  const errorRun = useCallback((error) => {
    dispatch({ type: ACTIONS.ERROR_RUN, payload: error });
  }, []);

  const resetSession = useCallback(() => {
    dispatch({ type: ACTIONS.RESET_SESSION });
  }, []);

  // ── Derived getters ─────────────────────────────────────────────────────────
  const tierState = {
    label: session.tier
      ? session.tier.charAt(0).toUpperCase() + session.tier.slice(1)
      : 'Unprocessed',
    icon: { bronze: '🥉', silver: '🥈', gold: '🥇', null: '📦' }[session.tier] || '📦',
    color: { bronze: '#cd7c4a', silver: '#94a3b8', gold: '#eab308', null: '#475569' }[session.tier] || '#475569',
  };

  const isRunning = session.status === 'running';
  const canRun = session.status === 'idle' || session.status === 'completed' || session.status === 'error';

  return (
    <PipelineSessionContext.Provider value={{
      session,
      tier: session.tier,
      tierState,
      isRunning,
      canRun,
      // Plan
      fetchPreviewPlan,
      approvePlan,
      cancelPlan,
      // Run lifecycle
      startRun,
      updateProgress,
      advanceStage,
      completeRun,
      errorRun,
      resetSession,
      // Config
      setConfig,
    }}>
      {children}
    </PipelineSessionContext.Provider>
  );
};

// ── Hook ──────────────────────────────────────────────────────────────────────

export const usePipelineSession = () => {
  const ctx = useContext(PipelineSessionContext);
  if (!ctx) {
    throw new Error('usePipelineSession must be used inside PipelineSessionProvider');
  }
  return ctx;
};

// ── Tier Badge Component ──────────────────────────────────────────────────────

export const TierBadge = ({ size = 'sm' }) => {
  const { tier, tierState } = usePipelineSession();
  if (!tier) return null;

  const styles = {
    sm: { fontSize: '0.72rem', padding: '0.15rem 0.5rem' },
    md: { fontSize: '0.82rem', padding: '0.25rem 0.7rem' },
  };

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '0.25rem',
      borderRadius: '20px',
      background: `${tierState.color}22`,
      border: `1px solid ${tierState.color}44`,
      color: tierState.color,
      fontWeight: 700,
      ...styles[size],
    }}>
      {tierState.icon} {tierState.label}
    </span>
  );
};

export default PipelineSessionContext;
