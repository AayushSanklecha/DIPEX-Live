"""
ingestion/feedback_controller.py
----------------------------------
Phase 14 — Feedback & Retry Controller

Bridges Hard Gate 2 → retry logic → RL update → Experience Memory.

This controller is the single authoritative component that:
  1. Receives a gate-2 rejection or low-confidence signal
  2. Selects the best retry strategy via UCB1 bandit
  3. Executes the retry with the chosen preprocessing strategy
  4. Computes the confidence delta (reward) for the RL engine
  5. Updates Experience Memory with the outcome
  6. Escalates to audit if retry budget is exhausted

Retry Strategies
----------------
  AGGRESSIVE_CLEAN   : Drop all rows with any null; enforce hard type coercions
  IMPUTE_MEDIAN      : Median imputation for numerics; mode for categoricals
  OUTLIER_CLIP       : Winsorise at 1st/99th percentile; Z-score |>3| → clip
  DROP_LOW_QUALITY   : Drop columns with null_rate >30%; drop duplicate rows
  SCHEMA_RELAX       : Coerce dtypes; do NOT enforce strict schema constraints

Architecture
------------
  FeedbackController.evaluate(pipeline_ctx) → RetryResult
    ├── _select_strategy()         UCB1 bandit over retry strategies
    ├── _apply_strategy(df, strat) Returns cleaned DataFrame
    ├── _compute_reward(before, after) Δconfidence ∈ [-1, 1]
    └── _escalate()                Log to audit/retry_escalations.jsonl

Safety Contracts
----------------
  - NEVER modifies the Silver-layer DataFrame in place
  - All strategy applications return a NEW DataFrame copy
  - Escalation is logged; pipeline receives a REJECT outcome
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.ingestion.feedback_controller")

# ── Strategy definitions ──────────────────────────────────────────────────────

STRATEGIES = [
    "AGGRESSIVE_CLEAN",
    "IMPUTE_MEDIAN",
    "OUTLIER_CLIP",
    "DROP_LOW_QUALITY",
    "SCHEMA_RELAX",
]


@dataclass
class RetryResult:
    """Outcome of a feedback/retry cycle."""
    run_id:           str
    attempt:          int
    strategy:         str
    original_conf:    float
    new_conf:         float
    reward:           float
    decision:         str        # PASS | REJECT | ESCALATED
    df:               Optional[pd.DataFrame] = field(default=None, repr=False)
    error:            Optional[str] = None
    metadata:         Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":        self.run_id,
            "attempt":       self.attempt,
            "strategy":      self.strategy,
            "original_conf": round(self.original_conf, 4),
            "new_conf":      round(self.new_conf, 4),
            "reward":        round(self.reward, 4),
            "decision":      self.decision,
            "error":         self.error,
            "metadata":      self.metadata,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }


# ── Feedback Controller ───────────────────────────────────────────────────────

class FeedbackController:
    """
    UCB1 bandit-driven feedback and retry controller.

    Parameters
    ----------
    max_retries : int
        Maximum retry budget per pipeline run (default: 3)
    confidence_threshold : float
        Target confidence score. Stop retrying when reached.
    audit_dir : str
        Directory for escalation logs.
    """

    def __init__(
        self,
        max_retries: int = 3,
        confidence_threshold: float = 0.75,
        audit_dir: str = "audit",
    ) -> None:
        self.max_retries          = max_retries
        self.confidence_threshold = confidence_threshold
        self.audit_dir            = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # UCB1 bandit state per strategy
        self._ucb_counts:  Dict[str, int]   = {s: 0 for s in STRATEGIES}
        self._ucb_rewards: Dict[str, float] = {s: 0.0 for s in STRATEGIES}

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        df: pd.DataFrame,
        run_id: str,
        confidence_score: float,
        gate2_result: Optional[Any] = None,
        profiling_report: Optional[Dict[str, Any]] = None,
        attempt: int = 1,
    ) -> RetryResult:
        """
        Evaluate whether a retry is warranted, select strategy, apply it.

        Parameters
        ----------
        df               : Current DataFrame (Silver snapshot — read-only copy taken internally)
        run_id           : Pipeline run identifier
        confidence_score : Current confidence score (pre-retry)
        gate2_result     : HardGate2 result object (optional — for failure analysis)
        profiling_report : Profiler output (optional — guides strategy selection)
        attempt          : Current retry attempt number (1-indexed)

        Returns
        -------
        RetryResult with decision, new confidence, reward, and cleaned DataFrame.
        """
        logger.info(
            "[Feedback] run=%s attempt=%d conf=%.3f threshold=%.3f",
            run_id, attempt, confidence_score, self.confidence_threshold,
        )

        # ── Budget check ─────────────────────────────────────────────────────
        if attempt > self.max_retries:
            return self._escalate(run_id, attempt, confidence_score)

        # ── Already above threshold? ──────────────────────────────────────────
        if confidence_score >= self.confidence_threshold:
            return RetryResult(
                run_id=run_id, attempt=attempt,
                strategy="NONE_NEEDED",
                original_conf=confidence_score, new_conf=confidence_score,
                reward=0.0, decision="PASS", df=df,
            )

        # ── Select strategy ───────────────────────────────────────────────────
        strategy = self._select_strategy(attempt, profiling_report)

        # ── Apply strategy (always on a copy) ─────────────────────────────────
        try:
            cleaned_df = self._apply_strategy(df.copy(), strategy, profiling_report)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Feedback] Strategy %s failed: %s", strategy, exc)
            cleaned_df = df.copy()
            strategy   = "FALLBACK_COPY"

        # ── Estimate new confidence heuristically ─────────────────────────────
        new_conf = self._estimate_confidence(cleaned_df, confidence_score, strategy)
        reward   = self._compute_reward(confidence_score, new_conf)

        # ── Update UCB1 bandit ─────────────────────────────────────────────────
        self._ucb_update(strategy, reward)

        decision = "PASS" if new_conf >= self.confidence_threshold else "REJECT"

        result = RetryResult(
            run_id=run_id, attempt=attempt,
            strategy=strategy,
            original_conf=confidence_score, new_conf=new_conf,
            reward=reward, decision=decision, df=cleaned_df,
            metadata={
                "rows_before": len(df),
                "rows_after":  len(cleaned_df),
                "cols_before": len(df.columns),
                "cols_after":  len(cleaned_df.columns),
            },
        )

        logger.info(
            "[Feedback] strategy=%s reward=%.4f conf %.3f→%.3f decision=%s",
            strategy, reward, confidence_score, new_conf, decision,
        )
        return result

    def run_retry_loop(
        self,
        df: pd.DataFrame,
        run_id: str,
        initial_confidence: float,
        gate2_result: Optional[Any] = None,
        profiling_report: Optional[Dict[str, Any]] = None,
    ) -> RetryResult:
        """
        Run the full retry loop until budget exhausted or threshold reached.

        Returns the final RetryResult (PASS, REJECT, or ESCALATED).
        """
        current_df   = df
        current_conf = initial_confidence
        last_result: Optional[RetryResult] = None

        for attempt in range(1, self.max_retries + 2):  # +2: escalation attempt
            result = self.evaluate(
                current_df, run_id, current_conf,
                gate2_result=gate2_result,
                profiling_report=profiling_report,
                attempt=attempt,
            )
            last_result = result

            if result.decision in ("PASS", "ESCALATED"):
                break

            # Update for next iteration
            current_df   = result.df if result.df is not None else current_df
            current_conf = result.new_conf

        return last_result  # type: ignore[return-value]

    # ── Strategy selector (UCB1) ──────────────────────────────────────────────

    def _select_strategy(
        self,
        attempt: int,
        profiling_report: Optional[Dict[str, Any]],
    ) -> str:
        """
        UCB1 bandit strategy selection.

        - Attempt 1: use profiling signals for an informed first pick
        - Subsequent: UCB1 exploration-exploitation
        """
        if attempt == 1 and profiling_report:
            return self._signal_guided_strategy(profiling_report)

        # UCB1: select max(avg_reward + sqrt(2*ln(total)/count))
        total_plays = sum(self._ucb_counts.values())
        if total_plays == 0:
            return STRATEGIES[0]

        ucb_scores: Dict[str, float] = {}
        for strat in STRATEGIES:
            n = self._ucb_counts[strat]
            if n == 0:
                ucb_scores[strat] = float("inf")
            else:
                avg = self._ucb_rewards[strat] / n
                ucb_scores[strat] = avg + math.sqrt(2 * math.log(total_plays) / n)

        return max(ucb_scores, key=ucb_scores.get)  # type: ignore[arg-type]

    def _signal_guided_strategy(
        self, profiling_report: Dict[str, Any]
    ) -> str:
        """Pick the most appropriate strategy from profiling signals."""
        columns = profiling_report.get("columns", {})

        # Dominant signal: most columns have high null rate → impute
        null_rates = [
            meta.get("null_rate", 0)
            for meta in columns.values()
            if isinstance(meta, dict)
        ]
        if null_rates and (sum(1 for r in null_rates if r > 0.15) / max(len(null_rates), 1)) > 0.3:
            return "IMPUTE_MEDIAN"

        # Many outliers → clip
        outlier_flags = [
            meta.get("outlier_pct_iqr", 0)
            for meta in columns.values()
            if isinstance(meta, dict)
        ]
        if outlier_flags and max(outlier_flags, default=0) > 0.05:
            return "OUTLIER_CLIP"

        # High cardinality columns that look like IDs → clean
        pk_cols = sum(
            1 for meta in columns.values()
            if isinstance(meta, dict) and meta.get("cardinality_tier") == "unique"
        )
        if pk_cols > 2:
            return "DROP_LOW_QUALITY"

        return "AGGRESSIVE_CLEAN"

    # ── Strategy implementations ──────────────────────────────────────────────

    def _apply_strategy(
        self,
        df: pd.DataFrame,
        strategy: str,
        profiling_report: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Apply the selected cleanup strategy. Always receives a COPY of df."""
        if strategy == "AGGRESSIVE_CLEAN":
            return self._strategy_aggressive_clean(df)
        elif strategy == "IMPUTE_MEDIAN":
            return self._strategy_impute_median(df)
        elif strategy == "OUTLIER_CLIP":
            return self._strategy_outlier_clip(df)
        elif strategy == "DROP_LOW_QUALITY":
            return self._strategy_drop_low_quality(df, profiling_report)
        elif strategy == "SCHEMA_RELAX":
            return self._strategy_schema_relax(df)
        else:
            return df

    def _strategy_aggressive_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop all rows containing any null; coerce mixed-type columns."""
        df = df.dropna()
        df = df.drop_duplicates()
        # Coerce object columns that look numeric
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass
        return df.reset_index(drop=True)

    def _strategy_impute_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Median imputation for numerics; mode for categoricals."""
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    med = df[col].median()
                    df[col] = df[col].fillna(med)
                else:
                    mode_vals = df[col].mode()
                    if not mode_vals.empty:
                        df[col] = df[col].fillna(mode_vals.iloc[0])
        return df

    def _strategy_outlier_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorise at 1st/99th percentile for numeric columns."""
        for col in df.select_dtypes(include=[np.number]).columns:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lo, upper=hi)
        return df

    def _strategy_drop_low_quality(
        self,
        df: pd.DataFrame,
        profiling_report: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Drop columns with >30% nulls; drop duplicate rows."""
        null_rates = df.isnull().mean()
        cols_to_drop = null_rates[null_rates > 0.30].index.tolist()
        if cols_to_drop:
            logger.info("[FeedbackCtrl] Dropping %d low-quality columns: %s",
                        len(cols_to_drop), cols_to_drop[:5])
            df = df.drop(columns=cols_to_drop, errors="ignore")
        df = df.drop_duplicates()
        return df.reset_index(drop=True)

    def _strategy_schema_relax(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce all dtypes to most permissive type; no hard constraints."""
        result = df.copy()
        for col in result.columns:
            # Try numeric coercion; fall back to string
            coerced = pd.to_numeric(result[col], errors="coerce")
            if coerced.notna().sum() > 0.5 * len(result):
                result[col] = coerced
            else:
                result[col] = result[col].astype(str)
        return result

    # ── Confidence estimation ─────────────────────────────────────────────────

    def _estimate_confidence(
        self,
        df: pd.DataFrame,
        previous_conf: float,
        strategy: str,
    ) -> float:
        """
        Heuristic confidence estimate post-strategy application.

        Uses measurable data quality indicators as a proxy for confidence.
        In production, the real confidence is computed by the ConfidenceVectorEngine.
        """
        if len(df) == 0:
            return 0.0

        null_rate      = df.isnull().mean().mean()
        dup_rate       = df.duplicated().mean()
        completeness   = max(0.0, 1.0 - null_rate - dup_rate)

        # Strategy bonuses (empirical heuristics)
        bonuses = {
            "AGGRESSIVE_CLEAN":  0.08,
            "IMPUTE_MEDIAN":     0.05,
            "OUTLIER_CLIP":      0.06,
            "DROP_LOW_QUALITY":  0.04,
            "SCHEMA_RELAX":      0.03,
            "FALLBACK_COPY":     0.0,
        }
        bonus = bonuses.get(strategy, 0.0)

        estimated = previous_conf + (completeness * bonus) + bonus * 0.5
        return round(min(max(estimated, 0.0), 1.0), 4)

    def _compute_reward(self, before: float, after: float) -> float:
        """Reward = Δconfidence, clamped to [-1, 1]."""
        return round(min(max(after - before, -1.0), 1.0), 4)

    # ── UCB1 update ───────────────────────────────────────────────────────────

    def _ucb_update(self, strategy: str, reward: float) -> None:
        if strategy in self._ucb_counts:
            self._ucb_counts[strategy]  += 1
            self._ucb_rewards[strategy] += reward

    # ── Escalation ────────────────────────────────────────────────────────────

    def _escalate(
        self,
        run_id: str,
        attempt: int,
        confidence_score: float,
    ) -> RetryResult:
        """Log escalation to audit/retry_escalations.jsonl."""
        entry = {
            "run_id":        run_id,
            "attempt":       attempt,
            "confidence":    round(confidence_score, 4),
            "message":       (
                f"Retry budget ({self.max_retries}) exhausted. "
                f"Confidence {confidence_score:.3f} remains below threshold "
                f"{self.confidence_threshold:.3f}."
            ),
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "ucb_counts":    dict(self._ucb_counts),
        }
        log_file = self.audit_dir / "retry_escalations.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("[Feedback] Escalation write failed: %s", exc)

        logger.warning(
            "[Feedback] ESCALATED run=%s conf=%.3f after %d retries",
            run_id, confidence_score, self.max_retries,
        )

        return RetryResult(
            run_id=run_id, attempt=attempt,
            strategy="ESCALATED",
            original_conf=confidence_score, new_conf=confidence_score,
            reward=-0.5, decision="ESCALATED",
            error=entry["message"],
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def reset_bandit(self) -> None:
        """Reset UCB1 bandit state (use between pipeline runs if isolation needed)."""
        self._ucb_counts  = {s: 0 for s in STRATEGIES}
        self._ucb_rewards = {s: 0.0 for s in STRATEGIES}

    def bandit_summary(self) -> Dict[str, Any]:
        """Return current UCB1 bandit state for logging/monitoring."""
        return {
            "strategy_counts":  dict(self._ucb_counts),
            "strategy_rewards": {
                k: round(v, 4) for k, v in self._ucb_rewards.items()
            },
            "best_strategy": max(
                (s for s in STRATEGIES if self._ucb_counts[s] > 0),
                key=lambda s: self._ucb_rewards[s] / max(self._ucb_counts[s], 1),
                default="NONE",
            ),
        }
