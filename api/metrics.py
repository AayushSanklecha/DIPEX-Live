"""
api/metrics.py
---------------
Production-grade Prometheus metrics registry for DIPEX.

Exposes:
  - Pipeline run counters (by gate decision, dataset)
  - Stage failure counters (by stage name)
  - Pipeline duration histogram (p50, p95, p99 SLA tracking)
  - Hard Gate decision counters (Gate 1 / Gate 2)
  - Retry engine counters (total retries, strategy used)
  - Confidence score histogram (distribution tracking)
  - LLM token usage counters (by provider, model)
  - Drift detection counters (by dataset, severity level)
  - QA failure rate per gate
  - Anomaly counters
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
from typing import Tuple

# ── Pipeline Execution Metrics ────────────────────────────────────────────────

dipex_pipeline_runs_total = Counter(
    "dipex_pipeline_runs_total",
    "Total number of pipeline runs by final gate decision.",
    ["gate_decision"],          # PASS | FAIL
)

dipex_pipeline_duration_seconds = Histogram(
    "dipex_pipeline_duration_seconds",
    "End-to-end pipeline duration in seconds.",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

dipex_stage_failures_total = Counter(
    "dipex_stage_failures_total",
    "Number of times a specific pipeline stage failed.",
    ["stage"],
)

# ── Gate Decision Metrics ─────────────────────────────────────────────────────

dipex_gate_decisions_total = Counter(
    "dipex_gate_decisions_total",
    "Hard Gate decisions by gate number and outcome.",
    ["gate", "decision"],       # gate: 1|2, decision: PASS|REJECT
)

# ── Confidence Score Distribution ─────────────────────────────────────────────

dipex_confidence_score = Histogram(
    "dipex_confidence_score",
    "Distribution of confidence scores produced by the Confidence Vector stage.",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
)

# ── Retry Engine Metrics ──────────────────────────────────────────────────────

dipex_retry_total = Counter(
    "dipex_retry_total",
    "Total retry attempts triggered by the Retry Engine.",
    ["strategy"],               # bandit strategy name
)

dipex_retry_escalations_total = Counter(
    "dipex_retry_escalations_total",
    "Number of runs where retry budget exhausted (escalated to monitoring).",
)

# ── LLM Token Usage ───────────────────────────────────────────────────────────

dipex_llm_tokens_total = Counter(
    "dipex_llm_tokens_total",
    "Total LLM tokens consumed by provider and model.",
    ["provider", "model", "direction"],  # direction: prompt|response
)

dipex_llm_calls_total = Counter(
    "dipex_llm_calls_total",
    "Total LLM API calls by provider.",
    ["provider"],
)

dipex_llm_governance_blocks_total = Counter(
    "dipex_llm_governance_blocks_total",
    "Number of LLM summary requests blocked by governance gate (unapproved result).",
)

# ── Drift Detection Metrics ───────────────────────────────────────────────────

dipex_drift_detected_total = Counter(
    "dipex_drift_detected_total",
    "Drift events detected by PSI threshold.",
    ["dataset_id", "severity"],     # severity: mild|moderate|severe
)

# ── ML / Model Metrics ────────────────────────────────────────────────────────

dipex_model_predictions_total = Counter(
    "dipex_model_predictions_total",
    "Total predictions made by model type.",
    ["model"],
)

dipex_model_roc_auc = Histogram(
    "dipex_model_roc_auc",
    "Distribution of model ROC-AUC scores recorded.",
    buckets=[0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
)

# ── RL Metrics ────────────────────────────────────────────────────────────────

dipex_rl_policy_updates_total = Counter(
    "dipex_rl_policy_updates_total",
    "Number of RL policy updates (Meta-RL, regret, EWC).",
    ["policy"],
)

dipex_rl_safety_violations_total = Counter(
    "dipex_rl_safety_violations_total",
    "Number of RL safety violations (forbidden target attempts).",
)

dipex_rl_rollbacks_total = Counter(
    "dipex_rl_rollbacks_total",
    "Number of RL instability-triggered rollbacks.",
)

# ── Streaming / Consumer Lag ──────────────────────────────────────────────────

dipex_streaming_consumer_lag = Gauge(
    "dipex_streaming_consumer_lag_messages",
    "Current Kafka consumer lag in messages.",
    ["topic"],
)

dipex_streaming_late_events_total = Counter(
    "dipex_streaming_late_events_total",
    "Number of late-arriving stream events processed.",
    ["source"],
)

# ── QA Failure Rates ──────────────────────────────────────────────────────────

dipex_qa_failures_total = Counter(
    "dipex_qa_failures_total",
    "QA failures by gate and check type.",
    ["gate", "check"],
)

# ── Anomaly Detection ─────────────────────────────────────────────────────────

dipex_anomalies_detected_total = Counter(
    "dipex_anomalies_detected_total",
    "Number of anomalies detected by the profiling or statistical stage.",
    ["column", "type"],
)


# ── Helper: Prometheus scrape response ───────────────────────────────────────

def get_metrics_response() -> Tuple[bytes, str]:
    """
    Returns (body_bytes, content_type) for the Prometheus scrape endpoint.
    Compatible with FastAPI Response: Response(content=body, media_type=ct).
    """
    body = generate_latest()
    return body, CONTENT_TYPE_LATEST
