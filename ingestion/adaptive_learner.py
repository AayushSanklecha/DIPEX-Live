"""
ingestion/adaptive_learner.py
------------------------------
Persistent, self-improving intelligence layer for the DIPEX ingestion pipeline.

After EVERY operation — success OR failure — the learner:
  1. Records what happened (strategy, outcome, error type, quality score, elapsed ms)
  2. Analyses root cause of failure or reason for success
  3. Builds a knowledge base of "what works for what data"
  4. Recommends improved strategy for the NEXT ingestion of the same or similar data
  5. Persists all learning to a JSON knowledge base (data/adaptive_kb.json)

Learning dimensions
-------------------
Format learning     : Which file format / encoding / delimiter worked for a source
Encoding learning   : Which encoding succeeded after failed attempts
Quality learning    : Common quality issues per dataset → proactive fix suggestions
Schema learning     : Drift history → predict future drift
Error learning      : Which errors occur at which data volume / source type
Pipeline learning   : Which downstream stages fail and why

Strategy improvement
--------------------
The learner uses a simple Bayesian-style success counter:
  success_rate = successes / total_attempts  (per strategy key)
Future strategy selection prioritises strategies with highest historical success rate.

Storage
-------
data/adaptive_kb.json  — persistent knowledge base
  {
    "source_strategies": {...},   # best strategy per source pattern
    "error_patterns": {...},       # error type → root cause + fix
    "quality_patterns": {...},     # dataset_id → common quality issues
    "schema_patterns": {...},      # dataset_id → drift history
    "global_stats": {...}          # total runs, pass rate, avg quality
  }
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dipex.ingestion.adaptive_learner")


# ── Outcome Record ────────────────────────────────────────────────────────────

@dataclass
class IngestionOutcome:
    """Complete record of a single ingestion attempt."""
    dataset_id: str
    source_type: str          # file | api | database | stream
    timestamp: str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    success: bool             = True
    quality_score: float      = 0.0
    validation_status: str    = "PENDING"
    row_count: int            = 0
    schema_version: Optional[str] = None
    elapsed_ms: float         = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    strategy_used: Dict       = field(default_factory=dict)   # encoding, delimiter, format, etc.
    stage_failures: List[str] = field(default_factory=list)   # which pipeline stages failed
    quality_issues: List[str] = field(default_factory=list)   # quality gate violations
    schema_drift: bool        = False
    schema_drift_type: Optional[str] = None  # ADDITIVE | MISSING | TYPE_CHANGE
    source_uri: str           = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# ── Strategy Recommendation ───────────────────────────────────────────────────

@dataclass
class StrategyRecommendation:
    """What the learner recommends for the next attempt."""
    dataset_id: str
    recommended_format: Optional[str] = None
    recommended_encoding: Optional[str] = None
    recommended_delimiter: Optional[str] = None
    skip_stages: List[str] = field(default_factory=list)
    quality_pre_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.5   # 0.0–1.0
    reason: str = ""


# ── Adaptive Learner ──────────────────────────────────────────────────────────

class AdaptiveLearner:
    """
    Self-improving intelligence layer for the ingestion pipeline.

    Usage::

        learner = AdaptiveLearner()

        # Record outcome after ingestion
        outcome = IngestionOutcome(dataset_id="sales", source_type="file",
                                   success=True, quality_score=0.92, ...)
        learner.record(outcome)

        # Get strategy recommendation BEFORE next ingestion
        rec = learner.recommend("sales", source_type="file", hints={"path": "sales.csv"})
        print(rec.recommended_encoding, rec.reason)
    """

    KB_PATH = "data/adaptive_kb.json"

    def __init__(self, kb_path: str = "data/adaptive_kb.json") -> None:
        self.kb_path = kb_path
        os.makedirs(os.path.dirname(kb_path) if os.path.dirname(kb_path) else ".", exist_ok=True)
        self._kb: Dict[str, Any] = self._load()

    # ── Record outcome ────────────────────────────────────────────────────────

    def record(self, outcome: IngestionOutcome) -> None:
        """Record outcome and update knowledge base."""
        self._update_source_strategies(outcome)
        self._update_error_patterns(outcome)
        self._update_quality_patterns(outcome)
        self._update_schema_patterns(outcome)
        self._update_global_stats(outcome)
        self._analyse_root_cause(outcome)
        self._save()

        status = "SUCCESS" if outcome.success else "FAILURE"
        logger.info(
            "[AdaptiveLearner] Recorded %s — dataset=%s quality=%.2f stage_fails=%s",
            status, outcome.dataset_id, outcome.quality_score,
            outcome.stage_failures or "none",
        )

    # ── Recommend strategy ────────────────────────────────────────────────────

    def recommend(
        self,
        dataset_id: str,
        source_type: str = "file",
        hints: Optional[Dict] = None,
    ) -> StrategyRecommendation:
        """
        Recommend the best ingestion strategy based on historical learning.
        Called BEFORE an ingestion attempt.
        """
        hints = hints or {}
        rec   = StrategyRecommendation(dataset_id=dataset_id)
        reasons: List[str] = []

        # ── 1. Best format/encoding/delimiter from past successes ─────────────
        strat_key = self._source_key(dataset_id, source_type)
        strats    = self._kb.get("source_strategies", {}).get(strat_key, {})
        if strats:
            best = self._best_strategy(strats)
            rec.recommended_format    = best.get("format")
            rec.recommended_encoding  = best.get("encoding")
            rec.recommended_delimiter = best.get("delimiter")
            rate = best.get("success_rate", 0.0)
            rec.confidence = min(0.95, 0.5 + rate * 0.45)
            reasons.append(
                f"Format/encoding learnt from {best.get('attempts', 1)} past attempt(s) "
                f"(success_rate={rate:.0%})"
            )

        # ── 2. Quality pre-checks from historical issues ──────────────────────
        qp = self._kb.get("quality_patterns", {}).get(dataset_id, {})
        common_issues = qp.get("common_issues", [])
        if common_issues:
            rec.quality_pre_checks = common_issues[:5]
            rec.warnings.extend([f"Expected quality issue: {i}" for i in common_issues[:3]])
            reasons.append(f"Quality pre-checks from {len(common_issues)} historical issues")

        # ── 3. Schema drift warnings ──────────────────────────────────────────
        sp = self._kb.get("schema_patterns", {}).get(dataset_id, {})
        if sp.get("drift_count", 0) > 0:
            last_drift = sp.get("last_drift_type")
            rec.warnings.append(
                f"Schema drift detected {sp['drift_count']} time(s) in history — last: {last_drift}"
            )
            reasons.append("Schema drift history noted")

        # ── 4. Stage failure recommendations ─────────────────────────────────
        ep = self._kb.get("error_patterns", {})
        for etype, edata in ep.items():
            if edata.get("source_type") == source_type and edata.get("count", 0) >= 2:
                rec.warnings.append(
                    f"Recurring error '{etype}' ({edata['count']}x): {edata.get('suggested_fix', '')}"
                )

        # ── 5. Skip stages that always fail for this dataset ─────────────────
        pipeline_fails = self._kb.get("pipeline_stage_fails", {}).get(dataset_id, {})
        always_fail = [
            stage for stage, info in pipeline_fails.items()
            if info.get("attempts", 0) >= 3 and info.get("successes", 0) == 0
        ]
        if always_fail:
            rec.skip_stages = always_fail
            rec.warnings.append(f"Skipping consistently-failing stages: {always_fail}")

        rec.reason = "; ".join(reasons) if reasons else "No prior learning for this dataset."
        return rec

    # ── Root cause analysis ───────────────────────────────────────────────────

    def _analyse_root_cause(self, outcome: IngestionOutcome) -> str:
        """Derive root cause and suggested fix for a failure."""
        if outcome.success:
            return ""

        error_type = outcome.error_type or "UNKNOWN"
        root_cause_map = {
            "DATA_FORMAT_ERROR":   ("Unrecognised or malformed file format",
                                    "Try specifying fmt= explicitly or check file integrity"),
            "ENCODING_ERROR":      ("Character encoding mismatch",
                                    "Try encoding='latin-1' or 'utf-8-sig'; check BOM"),
            "SCHEMA_ERROR":        ("Schema drift — column removed or type changed",
                                    "Review schema_registry history and align source schema"),
            "QUALITY_GATE_ERROR":  ("Data quality below threshold",
                                    "Increase null/duplicate tolerance or clean source before ingest"),
            "API_TIMEOUT_ERROR":   ("API endpoint timed out",
                                    "Increase timeout_s; check network / API status"),
            "API_RESPONSE_ERROR":  ("API returned non-2xx status",
                                    "Check authentication credentials and endpoint URL"),
            "DB_CONNECTION_ERROR": ("Database connection failed",
                                    "Verify credentials, host, port; check firewall rules"),
            "STREAM_LAG_ERROR":    ("Consumer lag exceeded threshold",
                                    "Scale consumer count; reduce window size; check backpressure"),
        }
        reason, fix = root_cause_map.get(
            error_type,
            ("Unknown error type", "Review error logs for clues"),
        )
        # Store in KB
        ep = self._kb.setdefault("error_patterns", {})
        ep.setdefault(error_type, {"count": 0, "source_type": outcome.source_type,
                                    "root_cause": reason, "suggested_fix": fix, "examples": []})
        ep[error_type]["count"] += 1
        if outcome.error_message and len(ep[error_type]["examples"]) < 5:
            ep[error_type]["examples"].append(outcome.error_message[:200])
        logger.info(
            "[AdaptiveLearner] Root cause: %s → %s | Fix: %s", error_type, reason, fix
        )
        return reason

    # ── Knowledge base updates ────────────────────────────────────────────────

    def _update_source_strategies(self, outcome: IngestionOutcome) -> None:
        strat_key = self._source_key(outcome.dataset_id, outcome.source_type)
        strategies = self._kb.setdefault("source_strategies", {})
        entry = strategies.setdefault(strat_key, {})

        # Build strategy fingerprint from what was used
        strategy = outcome.strategy_used or {}
        fmt  = strategy.get("format", "unknown")
        enc  = strategy.get("encoding", "utf-8")
        delim = strategy.get("delimiter", "auto")
        strategy_fp = f"{fmt}|{enc}|{delim}"

        attempts = entry.setdefault(strategy_fp, {
            "format": fmt, "encoding": enc, "delimiter": delim,
            "attempts": 0, "successes": 0, "success_rate": 0.0,
            "avg_quality": 0.0, "avg_ms": 0.0,
        })
        attempts["attempts"] += 1
        if outcome.success:
            attempts["successes"] += 1
        attempts["success_rate"] = attempts["successes"] / attempts["attempts"]
        # Running averages
        n = attempts["attempts"]
        attempts["avg_quality"] = ((attempts["avg_quality"] * (n - 1)) + outcome.quality_score) / n
        attempts["avg_ms"]      = ((attempts["avg_ms"]      * (n - 1)) + outcome.elapsed_ms)     / n

    def _update_error_patterns(self, outcome: IngestionOutcome) -> None:
        if outcome.success or not outcome.error_type:
            return
        self._analyse_root_cause(outcome)  # already updates error_patterns

    def _update_quality_patterns(self, outcome: IngestionOutcome) -> None:
        qp = self._kb.setdefault("quality_patterns", {})
        entry = qp.setdefault(outcome.dataset_id, {
            "runs": 0, "fail_count": 0, "warn_count": 0,
            "avg_quality": 0.0, "common_issues": [],
        })
        entry["runs"] += 1
        n = entry["runs"]
        entry["avg_quality"] = ((entry["avg_quality"] * (n - 1)) + outcome.quality_score) / n
        if outcome.validation_status == "FAILED":
            entry["fail_count"] += 1
        elif outcome.validation_status == "WARN":
            entry["warn_count"] += 1
        # Accumulate recurring quality issues
        for issue in outcome.quality_issues:
            if issue not in entry["common_issues"]:
                entry["common_issues"].append(issue)
            # Keep top 20
            entry["common_issues"] = entry["common_issues"][:20]

    def _update_schema_patterns(self, outcome: IngestionOutcome) -> None:
        sp = self._kb.setdefault("schema_patterns", {})
        entry = sp.setdefault(outcome.dataset_id, {
            "versions_seen": [], "drift_count": 0, "last_drift_type": None,
        })
        if outcome.schema_version and outcome.schema_version not in entry["versions_seen"]:
            entry["versions_seen"].append(outcome.schema_version)
        if outcome.schema_drift:
            entry["drift_count"] += 1
            entry["last_drift_type"] = outcome.schema_drift_type

    def _update_global_stats(self, outcome: IngestionOutcome) -> None:
        gs = self._kb.setdefault("global_stats", {
            "total_runs": 0, "successes": 0, "failures": 0,
            "total_rows": 0, "avg_quality": 0.0,
        })
        gs["total_runs"] += 1
        if outcome.success:
            gs["successes"] += 1
        else:
            gs["failures"] += 1
        gs["total_rows"] += outcome.row_count
        n = gs["total_runs"]
        gs["avg_quality"] = ((gs["avg_quality"] * (n - 1)) + outcome.quality_score) / n

    # ── Pipeline stage tracking ───────────────────────────────────────────────

    def record_stage_outcome(self, dataset_id: str, stage: str, success: bool) -> None:
        """Track success/failure per pipeline stage per dataset."""
        psf = self._kb.setdefault("pipeline_stage_fails", {})
        entry = psf.setdefault(dataset_id, {})
        s     = entry.setdefault(stage, {"attempts": 0, "successes": 0})
        s["attempts"] += 1
        if success:
            s["successes"] += 1
        self._save()

    # ── Public reporting ──────────────────────────────────────────────────────

    def get_insights(self, dataset_id: Optional[str] = None) -> Dict:
        """Return human-readable insights from the knowledge base."""
        gs = self._kb.get("global_stats", {})
        insights: Dict = {
            "global_stats": gs,
            "total_datasets": len(self._kb.get("quality_patterns", {})),
        }
        if dataset_id:
            insights["quality_history"] = self._kb.get("quality_patterns", {}).get(dataset_id, {})
            insights["schema_history"]  = self._kb.get("schema_patterns", {}).get(dataset_id, {})
        insights["top_errors"] = sorted(
            self._kb.get("error_patterns", {}).items(),
            key=lambda x: x[1].get("count", 0), reverse=True,
        )[:5]
        return insights

    def get_strategy_report(self) -> List[Dict]:
        """Return all learned strategies with success rates."""
        out = []
        for key, strategies in self._kb.get("source_strategies", {}).items():
            best = self._best_strategy(strategies)
            if best:
                out.append({"source": key, "best_strategy": best})
        return sorted(out, key=lambda x: x["best_strategy"].get("success_rate", 0), reverse=True)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _source_key(dataset_id: str, source_type: str) -> str:
        return f"{source_type}::{dataset_id}"

    @staticmethod
    def _best_strategy(strategies: Dict) -> Dict:
        """Return strategy with highest success_rate (tie-break: lowest avg_ms)."""
        if not strategies:
            return {}
        return max(
            strategies.values(),
            key=lambda s: (s.get("success_rate", 0), -s.get("avg_ms", 999999)),
        )

    def _load(self) -> Dict:
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                pass
        return {}

    def _save(self) -> None:
        with open(self.kb_path, "w", encoding="utf-8") as f:
            json.dump(self._kb, f, indent=2, default=str)
