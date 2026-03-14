"""
preprocessing/anomaly_scorer.py
---------------------------------
Row-Level Anomaly Scoring using Isolation Forest.

Answers the question: "Is this row as a whole unusual, given
the rest of the dataset?" — without requiring a known target.

Design:
  - Fits an Isolation Forest on the cleaned numeric feature set
  - Outputs TWO new columns on the DataFrame:
      `anomaly_score`  : continuous score ∈ (-inf, 0.5] — more negative = more anomalous
      `anomaly_flag`   : int8 — 1 = normal, -1 = anomalous (sklearn convention)
  - Emits a WARNING if anomaly_pct > warning_threshold
  - Emits an ERROR if anomaly_pct > error_threshold
  - Falls back gracefully if sklearn is unavailable (skips silently)

All decisions are fully logged and included in AnomalyReport.

Config stanza (all optional)::

    preprocessing:
      anomaly_scoring:
        enabled: true
        contamination: 0.05         # expected fraction of anomalies (0.0–0.5)
        warning_threshold: 0.05     # >5% anomalies → WARNING
        error_threshold: 0.20       # >20% anomalies → ERROR (pipeline concern)
        n_estimators: 100
        max_samples: auto
        random_state: 42
        output_score_col: anomaly_score
        output_flag_col: anomaly_flag
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.preprocessing.anomaly_scorer")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalyReport:
    """Audit record of anomaly scoring results."""
    run_id: str = ""
    enabled: bool = True
    n_rows: int = 0
    n_anomalous: int = 0
    anomaly_pct: float = 0.0
    contamination: float = 0.05
    severity: str = "OK"        # "OK" | "WARNING" | "ERROR" | "SKIPPED"
    method: str = "isolation_forest"
    features_used: List[str] = field(default_factory=list)
    top_anomalous_indices: List[int] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "enabled": self.enabled,
            "n_rows": self.n_rows,
            "n_anomalous": self.n_anomalous,
            "anomaly_pct": round(self.anomaly_pct, 6),
            "contamination": self.contamination,
            "severity": self.severity,
            "method": self.method,
            "features_used": self.features_used,
            "top_anomalous_indices": self.top_anomalous_indices[:20],
            "message": self.message,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyScorer:
    """
    Unsupervised row-level anomaly scorer using Isolation Forest.

    Usage::

        scorer = AnomalyScorer(config=config)
        df_scored, report = scorer.score(df, run_id="abc123")

        # df_scored now contains two new columns:
        #   anomaly_score  — continuous (more negative = stranger)
        #   anomaly_flag   — -1 anomaly, 1 normal

        if report.severity == "ERROR":
            logger.error("High anomaly rate: %s", report.message)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("preprocessing", {}).get("anomaly_scoring", {})
        self.enabled: bool             = bool(cfg.get("enabled", True))
        self.contamination: float      = float(cfg.get("contamination", 0.05))
        self.warn_threshold: float     = float(cfg.get("warning_threshold", 0.05))
        self.error_threshold: float    = float(cfg.get("error_threshold", 0.20))
        self.n_estimators: int         = int(cfg.get("n_estimators", 50))    # 50 is fast-enough for anomaly detection
        self.max_samples: Any          = cfg.get("max_samples", "auto")
        self.random_state: int         = int(cfg.get("random_state", 42))
        self.score_col: str            = str(cfg.get("output_score_col", "anomaly_score"))
        self.flag_col: str             = str(cfg.get("output_flag_col", "anomaly_flag"))
        # Performance caps — override via config if needed
        self.max_rows_for_fit: int     = int(cfg.get("max_rows_for_fit", 5000))
        self.max_features_for_fit: int = int(cfg.get("max_features_for_fit", 50))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AnomalyScorer":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        df: pd.DataFrame,
        run_id: str = "",
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, AnomalyReport]:
        """
        Fit Isolation Forest on numeric features of df and append anomaly
        score / flag columns.

        Parameters
        ----------
        df         : Input DataFrame (should already be cleaned)
        run_id     : Pipeline run ID for logging
        target_col : Target column to EXCLUDE from anomaly features

        Returns
        -------
        (df_scored, AnomalyReport)
          df_scored includes two new columns: anomaly_score, anomaly_flag
        """
        report = AnomalyReport(run_id=run_id, enabled=self.enabled)
        df = df.copy()

        if not self.enabled:
            report.severity = "SKIPPED"
            report.message = "Anomaly scoring disabled by config."
            return df, report

        if df is None or df.empty:
            report.severity = "SKIPPED"
            report.message = "Empty DataFrame — skipping anomaly scoring."
            return df, report

        # ── Feature selection: numeric, exclude target ────────────────────────
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col
            and c != self.score_col
            and c != self.flag_col
        ]

        if len(num_cols) == 0:
            report.severity = "SKIPPED"
            report.message = "No numeric feature columns available for anomaly scoring."
            return df, report

        if len(df) < 20:
            report.severity = "SKIPPED"
            report.message = f"Too few rows ({len(df)}) for reliable Isolation Forest."
            return df, report

        report.features_used = num_cols
        report.n_rows = len(df)
        report.contamination = self.contamination

        # ── Prepare feature matrix (capped for performance) ──────────────────
        # Cap to top N features to avoid slow high-dimensional fitting
        feat_cols = num_cols[:self.max_features_for_fit]
        X_score = df[feat_cols].fillna(0.0)
        # Sample rows for fitting (IF is a ratio metric, sampling is valid)
        if len(X_score) > self.max_rows_for_fit:
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(len(X_score), size=self.max_rows_for_fit, replace=False)
            X_fit = X_score.iloc[sample_idx]
        else:
            X_fit = X_score

        # ── Fit Isolation Forest ──────────────────────────────────────────────
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            report.severity = "SKIPPED"
            report.method = "none"
            report.message = "scikit-learn not installed — anomaly scoring skipped."
            logger.warning(report.message)
            return df, report

        try:
            clf = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                max_samples=self.max_samples,
                random_state=self.random_state,
                n_jobs=-1,
            )

            # Fit on sample, score ALL rows
            clf.fit(X_fit)
            df[self.score_col] = clf.score_samples(X_score)
            df[self.flag_col]  = clf.predict(X_score).astype(np.int8)

            n_anomalous  = int((df[self.flag_col] == -1).sum())
            anomaly_pct  = n_anomalous / len(df)

            report.n_anomalous = n_anomalous
            report.anomaly_pct = anomaly_pct

            # Top anomalous row indices (lowest score = most anomalous)
            top_idx = (
                df[self.score_col]
                .nsmallest(min(20, n_anomalous))
                .index
                .tolist()
            )
            report.top_anomalous_indices = [int(i) for i in top_idx]

            # ── Severity classification ───────────────────────────────────────
            if anomaly_pct > self.error_threshold:
                report.severity = "ERROR"
                report.message = (
                    f"{anomaly_pct:.2%} of rows are anomalous "
                    f"(error threshold: {self.error_threshold:.2%}). "
                    "Possible data quality or pipeline failure. Review source data."
                )
                logger.error("[AnomalyScorer] %s", report.message)

            elif anomaly_pct > self.warn_threshold:
                report.severity = "WARNING"
                report.message = (
                    f"{anomaly_pct:.2%} of rows flagged as anomalous "
                    f"(warn threshold: {self.warn_threshold:.2%}). "
                    "Investigate anomalous_flag == -1 rows before modelling."
                )
                logger.warning("[AnomalyScorer] %s", report.message)

            else:
                report.severity = "OK"
                report.message = (
                    f"Anomaly rate {anomaly_pct:.2%} is within acceptable range "
                    f"(threshold: {self.warn_threshold:.2%})."
                )
                logger.info("[AnomalyScorer][%s] %s", run_id[:8], report.message)

        except Exception as exc:
            report.severity = "SKIPPED"
            report.method = "exception"
            report.message = f"Anomaly scoring failed: {exc}"
            logger.warning("[AnomalyScorer] %s", report.message)
            # Remove partial columns if they exist
            df = df.drop(columns=[self.score_col, self.flag_col], errors="ignore")

        return df, report
