"""
validation/drift_detector.py
-----------------------------
Schema & statistical drift detector.

Compares the incoming DataFrame's schema against a persisted
schema fingerprint for the same dataset_id, detecting:

  • Column additions   → WARNING  (may be enrichment or accident)
  • Column deletions   → ERROR    (data contract violation)
  • dtype changes      → WARNING  (silent precision or type shift)
  • Distribution drift → WARNING  (PSI-based, numeric columns only)

Schema fingerprints are stored as JSON files in data/schema_registry/.
On first run for a dataset, the fingerprint is written (no violations raised).

All thresholds are config-driven.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.validation.drift_detector")

_REGISTRY_DIR = os.path.join("data", "schema_registry")


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriftViolation:
    column: Optional[str]
    severity: str        # WARNING | ERROR
    drift_type: str      # column_added | column_removed | dtype_changed | distribution_drift
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "severity": self.severity,
            "drift_type": self.drift_type,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class DriftReport:
    dataset_id: str
    run_id: str
    is_first_run: bool
    violations: List[DriftViolation] = field(default_factory=list)
    schema_written: bool = False
    retrain_required: bool = False
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def has_errors(self) -> bool:
        return any(v.severity == "ERROR" for v in self.violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "is_first_run": self.is_first_run,
            "schema_written": self.schema_written,
            "retrain_required": self.retrain_required,
            "violations": [v.to_dict() for v in self.violations],
            "has_errors": self.has_errors,
            "evaluated_at": self.evaluated_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class SchemaDriftDetector:
    """
    Detect schema and distribution drift versus the last stored fingerprint.

    Config stanza (all optional)::

        validation:
          drift:
            registry_dir: data/schema_registry
            psi_threshold: 0.2        # PSI above this → distribution_drift WARNING
            psi_bins: 10              # number of bins for PSI computation
            track_distribution: true  # set false to skip PSI (faster)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("validation", {}).get("drift", {})
        self.registry_dir: str    = cfg.get("registry_dir", _REGISTRY_DIR)
        self.psi_threshold: float = float(cfg.get("psi_threshold", 0.2))
        self.psi_bins: int        = int(cfg.get("psi_bins", 10))
        self.track_dist: bool     = bool(cfg.get("track_distribution", True))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SchemaDriftDetector":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        run_id: str = "",
    ) -> DriftReport:
        """
        Compare df against the stored schema fingerprint for dataset_id.

        If no fingerprint exists (first run), writes one and returns
        a clean DriftReport with is_first_run=True.
        """
        os.makedirs(self.registry_dir, exist_ok=True)
        fingerprint_path = os.path.join(
            self.registry_dir, f"{dataset_id}_schema.json"
        )

        report = DriftReport(dataset_id=dataset_id, run_id=run_id, is_first_run=False)
        current_schema = self._extract_schema(df)

        if not os.path.exists(fingerprint_path):
            # First run — write fingerprint and return clean report
            self._write_fingerprint(fingerprint_path, current_schema, df)
            report.is_first_run = True
            report.schema_written = True
            logger.info(
                "[DriftDetector] First run for dataset_id='%s' — schema fingerprint written.",
                dataset_id,
            )
            return report

        # Load stored fingerprint
        with open(fingerprint_path, encoding="utf-8") as f:
            stored = json.load(f)

        stored_schema: Dict[str, str] = stored.get("schema", {})
        stored_stats: Dict[str, Any]  = stored.get("statistics", {})

        # ── 1. Column-level schema checks ────────────────────────────────────
        current_cols = set(current_schema.keys())
        stored_cols  = set(stored_schema.keys())

        for col in current_cols - stored_cols:
            report.violations.append(DriftViolation(
                column=col, severity="WARNING", drift_type="column_added",
                message=f"Column '{col}' is new — not seen in prior runs.",
                details={"dtype": current_schema[col]},
            ))
            logger.warning("[DriftDetector] Column ADDED: '%s'", col)

        for col in stored_cols - current_cols:
            report.violations.append(DriftViolation(
                column=col, severity="ERROR", drift_type="column_removed",
                message=f"Column '{col}' has disappeared (data contract violation).",
                details={"expected_dtype": stored_schema[col]},
            ))
            logger.error("[DriftDetector] Column REMOVED: '%s'", col)

        for col in current_cols & stored_cols:
            if current_schema[col] != stored_schema[col]:
                report.violations.append(DriftViolation(
                    column=col, severity="WARNING", drift_type="dtype_changed",
                    message=(
                        f"dtype of '{col}' changed: "
                        f"{stored_schema[col]} → {current_schema[col]}"
                    ),
                    details={
                        "previous_dtype": stored_schema[col],
                        "current_dtype": current_schema[col],
                    },
                ))
                logger.warning(
                    "[DriftDetector] dtype drift '%s': %s → %s",
                    col, stored_schema[col], current_schema[col],
                )

        # ── 2. Distribution drift (PSI) ───────────────────────────────────────
        if self.track_dist and stored_stats:
            for col in current_cols & stored_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                psi = self._compute_psi(
                    df[col].dropna().values,
                    stored_stats.get(col, {})
                )
                if psi is not None and psi > self.psi_threshold:
                    report.violations.append(DriftViolation(
                        column=col, severity="WARNING",
                        drift_type="distribution_drift",
                        message=(
                            f"Distribution of '{col}' has shifted significantly "
                            f"(PSI={psi:.3f}, threshold={self.psi_threshold})."
                        ),
                        details={"psi": round(psi, 4), "threshold": self.psi_threshold},
                    ))
                    logger.warning(
                        "[DriftDetector] Distribution drift '%s' PSI=%.3f", col, psi
                    )

        # Update fingerprint with current data (rolling update)
        self._write_fingerprint(fingerprint_path, current_schema, df)
        report.schema_written = True

        # ── 3. Evaluate Retraining Trigger (Priority 5 Gap Fix) ───────────────
        dist_drift_count = sum(1 for v in report.violations if v.drift_type == "distribution_drift")
        schema_errors = sum(1 for v in report.violations if v.drift_type == "column_removed")
        
        if dist_drift_count > 0 or schema_errors > 0:
            report.retrain_required = True
            logger.warning(
                "🚨 DRIFT ALERT: %d feature(s) drifted, %d schema error(s). RETRAINING REQUIRED.",
                dist_drift_count, schema_errors
            )

        logger.info(
            "[DriftDetector] dataset_id='%s' — %d violations (%d errors).",
            dataset_id, len(report.violations), schema_errors
        )
        return report

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_schema(df: pd.DataFrame) -> Dict[str, str]:
        return {col: str(df[col].dtype) for col in df.columns}

    def _write_fingerprint(
        self,
        path: str,
        schema: Dict[str, str],
        df: pd.DataFrame,
    ) -> None:
        statistics: Dict[str, Any] = {}
        for col in df.select_dtypes(include=np.number).columns:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            # Fix 5: Store empirical histogram buckets (20 bins) alongside summary stats.
            # PSI prior reconstruction uses these empirical buckets — no Gaussian assumption.
            counts, bin_edges = np.histogram(s.values, bins=20)
            bucket_pcts = (counts / max(counts.sum(), 1)).tolist()
            statistics[col] = {
                "mean": float(s.mean()),
                "std":  float(s.std()),
                "min":  float(s.min()),
                "max":  float(s.max()),
                "p25":  float(s.quantile(0.25)),
                "p50":  float(s.quantile(0.50)),
                "p75":  float(s.quantile(0.75)),
                "n":    int(len(s)),
                # Empirical histogram for PSI (Fix 5)
                "hist_bins":   bin_edges.tolist(),   # len = 21
                "hist_pcts":   bucket_pcts,          # len = 20, sums to ~1.0
            }
        fingerprint = {
            "schema":     schema,
            "statistics": statistics,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2)

    def _compute_psi(
        self,
        current_vals: np.ndarray,
        stored_stats: Dict[str, Any],
    ) -> Optional[float]:
        """
        Population Stability Index (PSI) — Fix 5: Empirical Prior.

        Uses the stored histogram buckets (hist_bins / hist_pcts) instead of
        fitting a Normal distribution to (mean, std). This is accurate for
        skewed, bimodal, or heavy-tailed real-world distributions.

        Falls back to the Normal approximation for legacy fingerprints that
        pre-date this fix (those without 'hist_bins' key).
        """
        if not stored_stats or len(current_vals) < self.psi_bins:
            return None
        try:
            # ── Prefer empirical prior (Fix 5) ────────────────────────────────
            if "hist_bins" in stored_stats and "hist_pcts" in stored_stats:
                bins = np.array(stored_stats["hist_bins"])     # shape: (n_bins+1,)
                prior_pct = np.array(stored_stats["hist_pcts"])  # shape: (n_bins,)

                s_min, s_max = float(bins[0]), float(bins[-1])
                if s_max <= s_min:
                    return None

                clipped = np.clip(current_vals, s_min, s_max)
                cur_hist, _ = np.histogram(clipped, bins=bins)
                cur_pct = cur_hist / max(cur_hist.sum(), 1)

                # Smooth to avoid log(0)
                prior_pct = np.clip(prior_pct, 1e-6, None)
                prior_pct = prior_pct / prior_pct.sum()
                cur_pct   = np.clip(cur_pct, 1e-6, None)

                psi = float(np.sum((cur_pct - prior_pct) * np.log(cur_pct / prior_pct)))
                return psi

            # ── Legacy fallback: Normal approximation ─────────────────────────
            s_min = stored_stats.get("min", np.nanmin(current_vals))
            s_max = stored_stats.get("max", np.nanmax(current_vals))
            if s_max <= s_min:
                return None
            bins = np.linspace(s_min, s_max, self.psi_bins + 1)
            clipped = np.clip(current_vals, s_min, s_max)
            cur_hist, _ = np.histogram(clipped, bins=bins)
            cur_pct = cur_hist / max(cur_hist.sum(), 1)
            prior_mean = stored_stats.get("mean", (s_min + s_max) / 2)
            prior_std  = stored_stats.get("std",  (s_max - s_min) / 4)
            if prior_std <= 0:
                return None
            from scipy.stats import norm  # type: ignore
            prior_pct = np.diff(norm.cdf(bins, loc=prior_mean, scale=prior_std))
            prior_pct = np.clip(prior_pct, 1e-6, None)
            prior_pct /= prior_pct.sum()
            cur_pct = np.clip(cur_pct, 1e-6, None)
            psi = float(np.sum((cur_pct - prior_pct) * np.log(cur_pct / prior_pct)))
            return psi

        except Exception as exc:
            logger.debug("PSI computation failed: %s", exc)
            return None
