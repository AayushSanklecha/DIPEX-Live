"""
proposal/proposal_router.py
-----------------------------
Proposal Router — Step 4 of the DIPEX 13-Stage Pipeline.

The Proposal Router examines the pipeline context (profiling signals,
gate results, drift metrics, domain) and autonomously routes the dataset
to the most appropriate analyst operation combination.

Architecture
------------
This module is PURELY ADVISORY / AI-ASSISTIVE. It:
  - Generates candidate hypotheses and technical suggestions
  - Proposes which analyst operations should run next
  - Assigns a rationale + confidence score to each proposal
  - Does NOT execute anything or modify any data

Design Pattern: Rule-Based Dispatch + UCB1 Bandit Scoring
  - Deterministic rules pattern-match against pipeline signals
  - Confidence-weighted ranking surfaces top proposals
  - Proposals are consumed by the pipeline bridge to schedule analyst ops

Proposal Structure
------------------
Each proposal dict has:
  {
    "operation":    str,       # e.g. "eda_summary", "frame_problem"
    "tier":         str,       # "junior" | "mid" | "senior"
    "rationale":    str,       # Why this operation is recommended
    "confidence":   float,     # 0.0 – 1.0
    "priority":     int,       # 1 = highest
    "estimated_ms": int,       # rough-order-of-magnitude estimate
  }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dipex.proposal.router")


# ── Signal thresholds ─────────────────────────────────────────────────────────

_HIGH_NULL_RATE      = 0.15   # >15% nulls in any column → data quality alert
_HIGH_SKEWNESS       = 2.0    # |skewness| >2 → transformation proposal
_HIGH_DRIFT_PSI      = 0.20   # PSI >0.20 → drift-triggered RL boost signal
_LOW_CONFIDENCE      = 0.75   # confidence <0.75 → causal / experiment proposal
_HIGH_CARDINALITY    = 0.80   # >80% unique → PK / high-cardinality alert
_HIGH_CORRELATION    = 0.90   # inter-column correlation >0.90 → collinearity flag
_MIN_ROWS_STATS      = 30     # minimum rows for statistical tests
_MIN_ROWS_ML         = 100    # minimum rows for ML modeling recommendations


class ProposalRouter:
    """
    Autonomously proposes analyst operations based on pipeline signals.

    Usage::

        router = ProposalRouter(domain="banking", config=config)
        proposals = router.route(
            profile=profiling_report,
            gate_result=hard_gate1_result,
            drift_report=drift_report,
            confidence_score=0.72,
        )
        # returns list of proposal dicts, sorted by priority
    """

    def __init__(
        self,
        domain: str = "default",
        config: Optional[Dict[str, Any]] = None,
        max_proposals: int = 10,
    ) -> None:
        self.domain        = domain.lower()
        self.config        = config or {}
        self.max_proposals = max_proposals

        # UCB1-style selection counter (op → call count)
        self._call_counts: Dict[str, int] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def route(
        self,
        profile: Optional[Dict[str, Any]] = None,
        gate_result: Optional[Any] = None,
        drift_report: Optional[Dict[str, Any]] = None,
        confidence_score: float = 1.0,
        row_count: int = 0,
        extra_signals: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Examine pipeline signals and return a prioritised list of proposals.

        Parameters
        ----------
        profile          : Output of Profiler.profile() — column/dataset statistics
        gate_result      : GateResult from HardGate.run()
        drift_report     : Drift detector report with PSI/KL scores
        confidence_score : Confidence vector scalar from Stage 8
        row_count        : Total rows in dataset
        extra_signals    : Any extra k/v signals to factor in

        Returns
        -------
        List of proposal dicts, sorted by priority (1 = highest), capped at max_proposals.
        """
        proposals: List[Dict[str, Any]] = []
        signals = extra_signals or {}

        # ── Layer 1: Data Quality Proposals ──────────────────────────────────
        if profile:
            proposals.extend(self._profile_proposals(profile, row_count))

        # ── Layer 2: Gate Failure Proposals ──────────────────────────────────
        if gate_result and hasattr(gate_result, "decision"):
            proposals.extend(self._gate_proposals(gate_result))

        # ── Layer 3: Drift Proposals ──────────────────────────────────────────
        if drift_report:
            proposals.extend(self._drift_proposals(drift_report))

        # ── Layer 4: Confidence-Driven Proposals ──────────────────────────────
        proposals.extend(self._confidence_proposals(confidence_score, row_count))

        # ── Layer 5: Domain-Specific Proposals ───────────────────────────────
        proposals.extend(self._domain_proposals())

        # ── Layer 6: Default Baseline (always run) ────────────────────────────
        proposals.extend(self._baseline_proposals(row_count))

        # ── Layer 7: Streaming Window-Size Proposals ────────────────────────
        if signals.get("is_streaming", False):
            proposals.extend(self._streaming_proposals(signals))

        # Deduplicate by operation name (keep highest confidence)
        seen: Dict[str, Dict] = {}
        for p in proposals:
            op = p["operation"]
            if op not in seen or p["confidence"] > seen[op]["confidence"]:
                seen[op] = p

        # Sort: primary by priority (asc), secondary by confidence (desc)
        ranked = sorted(seen.values(), key=lambda x: (x["priority"], -x["confidence"]))
        result = ranked[:self.max_proposals]

        # Update call counts for UCB1 tracking
        for p in result:
            op = p["operation"]
            self._call_counts[op] = self._call_counts.get(op, 0) + 1

        logger.info(
            "ProposalRouter [domain=%s]: %d proposals generated, top=%s",
            self.domain, len(result),
            result[0]["operation"] if result else "none",
        )
        return result

    # ── Internal layers ───────────────────────────────────────────────────────

    def _profile_proposals(
        self, profile: Dict[str, Any], row_count: int
    ) -> List[Dict[str, Any]]:
        """Generate proposals from data profiling signals."""
        proposals: List[Dict[str, Any]] = []
        columns = profile.get("columns", {})
        flags   = profile.get("analyst_flags", [])

        # High null rate → cleaning proposal
        high_null_cols = [
            col for col, meta in columns.items()
            if isinstance(meta, dict) and meta.get("null_rate", 0) > _HIGH_NULL_RATE
        ]
        if high_null_cols:
            proposals.append(self._proposal(
                operation="data_cleaning",
                tier="junior",
                rationale=(
                    f"{len(high_null_cols)} column(s) have >15% null rate "
                    f"({', '.join(high_null_cols[:3])}{'...' if len(high_null_cols) > 3 else ''}). "
                    "Imputation or exclusion strategy required."
                ),
                confidence=0.93,
                priority=1,
                estimated_ms=500,
            ))

        # High skewness → transformation proposal
        skewed_cols = [
            col for col, meta in columns.items()
            if isinstance(meta, dict) and abs(meta.get("skewness", 0)) > _HIGH_SKEWNESS
        ]
        if skewed_cols:
            proposals.append(self._proposal(
                operation="basic_stats",
                tier="junior",
                rationale=(
                    f"Skewed distributions detected in {len(skewed_cols)} column(s). "
                    "Log or Box-Cox transformation may improve normality for downstream models."
                ),
                confidence=0.82,
                priority=2,
                estimated_ms=300,
            ))

        # High-cardinality columns → PK detection
        pk_cols = [
            col for col, meta in columns.items()
            if isinstance(meta, dict) and meta.get("cardinality_tier") == "unique"
        ]
        if pk_cols:
            proposals.append(self._proposal(
                operation="generate_insights",
                tier="mid",
                rationale=(
                    f"{len(pk_cols)} column(s) appear to be primary key candidates "
                    f"({', '.join(pk_cols[:3])}). Recommend unique constraint validation."
                ),
                confidence=0.78,
                priority=3,
                estimated_ms=400,
            ))

        # High correlation → collinearity warning
        corr_flags = [f for f in flags if f.get("flag_type") == "high_correlation"]
        if corr_flags:
            proposals.append(self._proposal(
                operation="correlation_analysis",
                tier="mid",
                rationale=(
                    f"{len(corr_flags)} high-correlation pair(s) detected. "
                    "Consider PCA, VIF analysis, or feature exclusion to reduce multicollinearity."
                ),
                confidence=0.85,
                priority=2,
                estimated_ms=600,
            ))

        # Large dataset → EDA
        if row_count >= _MIN_ROWS_STATS:
            proposals.append(self._proposal(
                operation="eda_summary",
                tier="mid",
                rationale=(
                    f"Dataset has {row_count:,} rows. "
                    "Full EDA recommended before pipeline proceeds."
                ),
                confidence=0.90,
                priority=1,
                estimated_ms=1200,
            ))

        return proposals

    def _gate_proposals(self, gate_result: Any) -> List[Dict[str, Any]]:
        """Generate proposals based on Hard Gate 1 result."""
        proposals: List[Dict[str, Any]] = []
        if gate_result.decision == "REJECT":
            for failure in gate_result.failures[:3]:
                col    = failure.get("column", "unknown")
                reason = failure.get("message", "validation failure")
                proposals.append(self._proposal(
                    operation="data_cleaning",
                    tier="junior",
                    rationale=(
                        f"Gate 1 REJECT — column '{col}': {reason[:100]}. "
                        "Data cleaning required before reprocessing."
                    ),
                    confidence=0.97,
                    priority=1,
                    estimated_ms=800,
                ))

        if gate_result.warnings:
            proposals.append(self._proposal(
                operation="regulatory_compliance_check",
                tier="mid",
                rationale=(
                    f"{len(gate_result.warnings)} Gate 1 warning(s). "
                    "Review regulatory alignment and data contract gaps."
                ),
                confidence=0.80,
                priority=2,
                estimated_ms=400,
            ))

        return proposals

    def _drift_proposals(self, drift_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate proposals when data drift is detected."""
        proposals: List[Dict[str, Any]] = []
        overall_psi = drift_report.get("overall_psi", 0.0)
        severity    = drift_report.get("severity", "none")

        if severity in ("severe", "moderate") or overall_psi > _HIGH_DRIFT_PSI:
            proposals.append(self._proposal(
                operation="design_experiment",
                tier="senior",
                rationale=(
                    f"Data drift detected (PSI={overall_psi:.3f}, severity={severity}). "
                    "A/B or time-series experiment recommended to isolate drift root cause."
                ),
                confidence=0.88,
                priority=1,
                estimated_ms=2000,
            ))
            proposals.append(self._proposal(
                operation="cohort_analysis",
                tier="mid",
                rationale=(
                    "Drift detected — cohort segmentation recommended to identify "
                    "which population subgroups are most affected."
                ),
                confidence=0.82,
                priority=2,
                estimated_ms=900,
            ))

        return proposals

    def _confidence_proposals(
        self, confidence: float, row_count: int
    ) -> List[Dict[str, Any]]:
        """Generate proposals when pipeline confidence is below threshold."""
        proposals: List[Dict[str, Any]] = []

        if confidence < _LOW_CONFIDENCE:
            proposals.append(self._proposal(
                operation="causal_inference",
                tier="senior",
                rationale=(
                    f"Confidence score {confidence:.2%} is below threshold {_LOW_CONFIDENCE:.0%}. "
                    "Causal inference may identify confounders suppressing confidence."
                ),
                confidence=0.86,
                priority=2,
                estimated_ms=3000,
            ))

        if row_count >= _MIN_ROWS_ML and confidence < _LOW_CONFIDENCE:
            proposals.append(self._proposal(
                operation="automl_model_selection",
                tier="senior",
                rationale=(
                    "Low confidence with sufficient row count → "
                    "AutoML model class selection proposal warranted."
                ),
                confidence=0.80,
                priority=3,
                estimated_ms=5000,
            ))

        return proposals

    def _domain_proposals(self) -> List[Dict[str, Any]]:
        """Domain-specific baseline proposals for known domains."""
        domain_ops: Dict[str, List[Dict[str, Any]]] = {
            "banking": [
                self._proposal(
                    "aml_pattern_detection", "senior",
                    "Banking domain: AML pattern detection recommended for all transaction datasets.",
                    0.88, 2, 2500,
                ),
            ],
            "healthcare": [
                self._proposal(
                    "outlier_detection", "mid",
                    "Healthcare domain: Clinical outlier detection is critical before reporting.",
                    0.90, 1, 1500,
                ),
            ],
            "finance": [
                self._proposal(
                    "time_series_analysis", "senior",
                    "Finance domain: Time-series decomposition recommended for trading/reporting data.",
                    0.87, 2, 3000,
                ),
            ],
        }
        return domain_ops.get(self.domain, [])

    def _baseline_proposals(self, row_count: int) -> List[Dict[str, Any]]:
        """Baseline proposals always generated regardless of signals."""
        props = [
            self._proposal(
                "generate_report", "junior",
                "Generate summary report for stakeholder review.",
                0.70, 5, 800,
            ),
        ]
        if row_count >= _MIN_ROWS_STATS:
            props.append(self._proposal(
                "hypothesis_testing", "mid",
                "Statistical hypothesis tests recommended to validate key assumptions.",
                0.75, 4, 1000,
            ))
        return props

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def _proposal(
        operation: str,
        tier: str,
        rationale: str,
        confidence: float,
        priority: int,
        estimated_ms: int,
    ) -> Dict[str, Any]:
        return {
            "operation":    operation,
            "tier":         tier,
            "rationale":    rationale[:400],
            "confidence":   round(min(max(confidence, 0.0), 1.0), 4),
            "priority":     priority,
            "estimated_ms": estimated_ms,
        }

    def top_operation(self, proposals: List[Dict[str, Any]]) -> Optional[str]:
        """Return the name of the highest-priority proposal, or None."""
        return proposals[0]["operation"] if proposals else None

    def filter_by_tier(
        self, proposals: List[Dict[str, Any]], tier: str
    ) -> List[Dict[str, Any]]:
        """Filter proposals to a specific analyst tier."""
        return [p for p in proposals if p["tier"] == tier]

    def _streaming_proposals(
        self, signals: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Layer 7 — Streaming Window-Size Proposals.

        Triggered when extra_signals contains 'is_streaming': True.

        Bases window recommendations on:
          - event_rate_per_sec : events per second (default 100)
          - target_latency_ms  : latency budget in ms (default 5000)
          - has_late_data      : bool — whether late events are common
          - session_gap_ms     : optional inactivity gap for session windows

        Proposes:
          - Tumbling window  : non-overlapping fixed windows for aggregation
          - Sliding window   : overlapping windows for rolling analytics
          - Session window   : gap-based (if session_gap_ms provided)
          - Watermark depth  : late-data tolerance configuration
        """
        proposals: List[Dict[str, Any]] = []

        event_rate     = float(signals.get("event_rate_per_sec", 100))
        target_latency = float(signals.get("target_latency_ms", 5000))
        has_late_data  = bool(signals.get("has_late_data", False))
        session_gap_ms = signals.get("session_gap_ms")

        # Tumbling window — non-overlapping, bounded by latency budget
        tumble_ms  = int(min(max(target_latency, 1000), 300_000))
        tumble_sec = tumble_ms // 1000
        proposals.append(self._proposal(
            operation="streaming_window_config",
            tier="senior",
            rationale=(
                f"Streaming data detected (event_rate={event_rate:.0f}/s, "
                f"latency_budget={target_latency:.0f}ms). "
                f"RECOMMENDED: Tumbling window of {tumble_sec}s "
                f"({int(event_rate * tumble_sec):,} events/window expected). "
                "Use for non-overlapping aggregation (sum, count, avg)."
            ),
            confidence=0.85,
            priority=1,
            estimated_ms=200,
        ))

        # Sliding window — 25% slide for 4x temporal resolution
        slide_ms  = max(tumble_ms // 4, 500)
        slide_sec = slide_ms // 1000
        proposals.append(self._proposal(
            operation="streaming_sliding_window",
            tier="senior",
            rationale=(
                f"ALTERNATIVE: Sliding window (size={tumble_sec}s, "
                f"slide={slide_sec}s) for 4x temporal resolution "
                "in rolling analytics and continuous anomaly detection."
            ),
            confidence=0.78,
            priority=2,
            estimated_ms=200,
        ))

        # Session window — gap-based (only if session_gap_ms provided)
        if session_gap_ms:
            proposals.append(self._proposal(
                operation="streaming_session_window",
                tier="senior",
                rationale=(
                    f"Session window (inactivity_gap={session_gap_ms}ms) recommended "
                    "for user-session analytics where natural activity boundaries "
                    "define the aggregation unit."
                ),
                confidence=0.82,
                priority=2,
                estimated_ms=150,
            ))

        # Watermark depth — late-data tolerance (only if late data present)
        if has_late_data:
            watermark_ms = max(tumble_ms // 2, 1000)
            proposals.append(self._proposal(
                operation="streaming_watermark_config",
                tier="senior",
                rationale=(
                    f"Late events detected. Set watermark depth to "
                    f"{watermark_ms}ms ({watermark_ms // 1000}s). "
                    "Events arriving later create a corrective Gold snapshot "
                    "rather than silently overwriting in place."
                ),
                confidence=0.91,
                priority=1,
                estimated_ms=100,
            ))

        logger.info(
            "[ProposalRouter] Streaming proposals: %d (rate=%.0f/s, latency=%.0fms)",
            len(proposals), event_rate, target_latency,
        )
        return proposals

    def bandit_summary(self) -> Dict[str, int]:
        """Return the UCB1 call-count tracker for all proposed operations."""
        return dict(self._call_counts)
