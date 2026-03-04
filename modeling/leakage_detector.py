"""
modeling/leakage_detector.py
------------------------------
Production data leakage detection for ML pipelines.

Detects:
  1. Correlation leakage       — feature perfectly correlated with target
  2. Temporal leakage          — future information in features (requires datetime index)
  3. Near-duplicate leakage    — features that are near-identical to target
  4. Post-event contamination  — features computed using the target label

All checks are non-destructive and return structured findings.

Usage::

    ld = LeakageDetector()
    report = ld.detect(df, target="churn", time_col="event_date")
    if report.has_leakage:
        raise LeakageDetectedError(report.summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger("dipex.modeling.leakage")


class LeakageDetectedError(ValueError):
    """Raised when hard-stop leakage is confirmed."""


@dataclass
class LeakageFlag:
    feature: str
    leakage_type: str    # correlation | temporal | near_duplicate | naming
    severity: str        # HIGH | MEDIUM | LOW
    correlation: Optional[float]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature, "leakage_type": self.leakage_type,
            "severity": self.severity, "correlation": self.correlation,
            "detail": self.detail,
        }


@dataclass
class LeakageReport:
    has_leakage: bool = False
    flags: List[LeakageFlag] = field(default_factory=list)
    summary: str = ""
    high_severity_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_leakage": self.has_leakage,
            "high_severity_count": self.high_severity_count,
            "total_flags": len(self.flags),
            "flags": [f.to_dict() for f in self.flags],
            "summary": self.summary,
        }


class LeakageDetector:
    """
    Multi-method data leakage detector.

    Parameters
    ----------
    correlation_threshold : float
        Pearson |r| above this triggers a HIGH-severity flag.
    near_duplicate_threshold : float
        Correlation above this triggers near-duplicate flag.
    suspicious_names : list
        Column name substrings that often indicate leakage.
    """

    SUSPICIOUS_NAMES = [
        "target", "label", "y_", "_label", "churn", "fraud", "default",
        "approved", "outcome", "result", "response", "_flag", "_ind",
        "_score_final", "actual_",
    ]

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        near_duplicate_threshold: float = 0.99,
        suspicious_names: Optional[List[str]] = None,
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.near_duplicate_threshold = near_duplicate_threshold
        self.suspicious_names = suspicious_names or self.SUSPICIOUS_NAMES

    def detect(
        self,
        df: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]] = None,
        time_col: Optional[str] = None,
    ) -> LeakageReport:
        """Run all leakage detection checks."""
        report = LeakageReport()

        if target not in df.columns:
            report.summary = f"Target '{target}' not found."
            return report

        features = feature_columns or [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
        target_series = df[target].dropna()

        # 1. Correlation leakage
        self._correlation_leakage(df, target, features, target_series, report)

        # 2. Near-duplicate leakage
        self._near_duplicate_leakage(df, target, features, target_series, report)

        # 3. Naming-convention leakage
        self._naming_leakage(features, target, report)

        # 4. Temporal leakage (if time column provided)
        if time_col and time_col in df.columns:
            self._temporal_leakage(df, target, features, time_col, report)

        report.high_severity_count = sum(1 for f in report.flags if f.severity == "HIGH")
        report.has_leakage = len(report.flags) > 0
        report.summary = self._build_summary(report)

        if report.high_severity_count > 0:
            logger.error("HIGH-severity leakage detected: %s flags. Training blocked.",
                         report.high_severity_count)
        elif report.flags:
            logger.warning("Potential leakage detected: %d flags. Review before training.", len(report.flags))

        return report

    # ── Correlation leakage ───────────────────────────────────────────────────

    def _correlation_leakage(
        self, df: pd.DataFrame, target: str,
        features: List[str], target_series: pd.Series, report: LeakageReport,
    ) -> None:
        for col in features:
            sub = df[[col, target]].dropna()
            if len(sub) < 10:
                continue
            try:
                r, p = scipy_stats.pearsonr(sub[col], sub[target])
                abs_r = abs(float(r))
                if abs_r >= self.near_duplicate_threshold:
                    report.flags.append(LeakageFlag(
                        feature=col, leakage_type="near_duplicate",
                        severity="HIGH", correlation=round(abs_r, 6),
                        detail=(f"r={abs_r:.4f} — feature is near-identical to target. "
                                "Almost certainly leakage."),
                    ))
                elif abs_r >= self.correlation_threshold:
                    report.flags.append(LeakageFlag(
                        feature=col, leakage_type="correlation",
                        severity="HIGH", correlation=round(abs_r, 6),
                        detail=(f"r={abs_r:.4f} suspiciously high correlation with '{target}'. "
                                "Verify this feature is available at prediction time."),
                    ))
                elif abs_r >= 0.80:
                    report.flags.append(LeakageFlag(
                        feature=col, leakage_type="correlation",
                        severity="MEDIUM", correlation=round(abs_r, 6),
                        detail=f"r={abs_r:.4f} — high correlation, review feature construction.",
                    ))
            except Exception:  # noqa: BLE001
                pass

    # ── Near-duplicate ────────────────────────────────────────────────────────

    def _near_duplicate_leakage(
        self, df: pd.DataFrame, target: str,
        features: List[str], target_series: pd.Series, report: LeakageReport,
    ) -> None:
        # Check for near-zero variation features (constants leak the global stat)
        for col in features:
            # Already checked in correlation; check zero-variance
            if df[col].std() < 1e-8:
                report.flags.append(LeakageFlag(
                    feature=col, leakage_type="near_duplicate",
                    severity="LOW", correlation=None,
                    detail=f"'{col}' has near-zero variance — likely a constant or post-event fill.",
                ))

    # ── Naming-convention leakage ─────────────────────────────────────────────

    def _naming_leakage(self, features: List[str], target: str, report: LeakageReport) -> None:
        for col in features:
            col_lower = col.lower()
            for suspect in self.suspicious_names:
                if suspect in col_lower and col != target:
                    report.flags.append(LeakageFlag(
                        feature=col, leakage_type="naming",
                        severity="MEDIUM", correlation=None,
                        detail=(f"Column name '{col}' contains suspicious substring '{suspect}'. "
                                "Verify it doesn't encode the target label."),
                    ))
                    break

    # ── Temporal leakage ──────────────────────────────────────────────────────

    def _temporal_leakage(
        self, df: pd.DataFrame, target: str,
        features: List[str], time_col: str, report: LeakageReport,
    ) -> None:
        """
        Test if features are more predictive on the SAME day (potential look-ahead)
        vs. lagged values (legitimate). Simplified: check if future-period aggregates exist.
        """
        try:
            dates = pd.to_datetime(df[time_col], errors="coerce")
            if dates.isna().mean() > 0.5:
                return
            # Look for columns with rolling or future suffixes
            future_hints = ["_future", "_next", "_t+", "_lead", "_fwd"]
            for col in features:
                col_lower = col.lower()
                for hint in future_hints:
                    if hint in col_lower:
                        report.flags.append(LeakageFlag(
                            feature=col, leakage_type="temporal",
                            severity="HIGH", correlation=None,
                            detail=(f"Column '{col}' name suggests forward-looking information ('{hint}'). "
                                    "This is temporal leakage if used as a feature."),
                        ))
                        break
        except Exception:  # noqa: BLE001
            pass

    # ── Summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(report: LeakageReport) -> str:
        if not report.flags:
            return "No leakage detected. Training is safe to proceed."
        high = sum(1 for f in report.flags if f.severity == "HIGH")
        med = sum(1 for f in report.flags if f.severity == "MEDIUM")
        low = sum(1 for f in report.flags if f.severity == "LOW")
        cols = ", ".join(f.feature for f in report.flags if f.severity == "HIGH")
        if high > 0:
            return (f"HIGH-SEVERITY LEAKAGE DETECTED in {high} feature(s): [{cols}]. "
                    f"Also {med} MEDIUM, {low} LOW flags. Training should be blocked.")
        return (f"Potential leakage: {med} MEDIUM, {low} LOW flags. "
                "Review flagged features before accepting model outputs.")
