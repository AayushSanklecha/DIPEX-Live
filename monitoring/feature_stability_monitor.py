"""
monitoring/feature_stability_monitor.py
-----------------------------------------
Cross-run feature importance stability monitor.

Problem: If SHAP/feature importance rankings change drastically between
pipeline runs (e.g., top-3 features last run: [age, income, balance],
this run: [transaction_id, random_col, zip_code]), it almost certainly
means a data pipeline issue upstream — not a real signal. Without this
check, the model silently produces wrong results with no warning.

What this module does:
  1. Persists the feature importance ranking from the winning model each run
  2. On every subsequent run, computes rank correlation (Kendall's τ) between
     the current and previous importance rankings
  3. Emits:
     - PASS if τ > stability_threshold (importances are consistent)
     - WARNING if τ ∈ [degraded_threshold, stability_threshold)
     - ERROR if τ < degraded_threshold (dramatic re-ranking — likely data issue)
  4. Logs top features added/removed between runs for human review
  5. Stores the history file under data/feature_stability/

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

logger = logging.getLogger("dipex.monitoring.feature_stability_monitor")

_STABILITY_DIR = os.path.join("data", "feature_stability")


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StabilityReport:
    dataset_id: str
    run_id: str
    is_first_run: bool
    status: str                        # PASS | WARNING | ERROR | SKIPPED
    tau: Optional[float]               # Kendall's τ (-1..1); None if first run
    stability_threshold: float
    degraded_threshold: float
    current_top_features: List[str]
    previous_top_features: List[str]
    features_gained: List[str]         # new top features vs last run
    features_lost: List[str]           # dropped from top features vs last run
    message: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "is_first_run": self.is_first_run,
            "status": self.status,
            "kendall_tau": round(self.tau, 4) if self.tau is not None else None,
            "stability_threshold": self.stability_threshold,
            "degraded_threshold": self.degraded_threshold,
            "current_top_features": self.current_top_features,
            "previous_top_features": self.previous_top_features,
            "features_gained": self.features_gained,
            "features_lost": self.features_lost,
            "message": self.message,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Monitor
# ─────────────────────────────────────────────────────────────────────────────

class FeatureStabilityMonitor:
    """
    Track and validate feature importance stability across pipeline runs.

    Config stanza (all optional)::

        monitoring:
          feature_stability:
            stability_dir: data/feature_stability
            stability_threshold: 0.70    # Kendall τ >= this → PASS
            degraded_threshold: 0.40     # Kendall τ >= this → WARNING; below → ERROR
            top_n: 15                    # number of top features to compare
            min_features: 3              # min features in importance dict to run check
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("monitoring", {}).get("feature_stability", {})
        self.stability_dir: str     = cfg.get("stability_dir", _STABILITY_DIR)
        self.stable_thresh: float   = float(cfg.get("stability_threshold", 0.70))
        self.degraded_thresh: float = float(cfg.get("degraded_threshold", 0.40))
        self.top_n: int             = int(cfg.get("top_n", 15))
        self.min_features: int      = int(cfg.get("min_features", 3))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureStabilityMonitor":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        feature_importances: Dict[str, float],
        dataset_id: str,
        run_id: str = "",
    ) -> StabilityReport:
        """
        Compare current feature importances with the stored history.

        Parameters
        ----------
        feature_importances : dict mapping feature_name → importance_score
        dataset_id          : used to namespace the history file
        run_id              : pipeline run ID for logging

        Returns
        -------
        StabilityReport
        """
        os.makedirs(self.stability_dir, exist_ok=True)
        history_path = os.path.join(self.stability_dir, f"{dataset_id}_importance.json")

        current_top = self._top_n_sorted(feature_importances)
        current_top_names = [f for f, _ in current_top]

        if len(current_top_names) < self.min_features:
            report = StabilityReport(
                dataset_id=dataset_id, run_id=run_id, is_first_run=True,
                status="SKIPPED", tau=None,
                stability_threshold=self.stable_thresh,
                degraded_threshold=self.degraded_thresh,
                current_top_features=current_top_names,
                previous_top_features=[],
                features_gained=[], features_lost=[],
                message=f"Too few features ({len(current_top_names)}) for stability check.",
            )
            self._write_history(history_path, feature_importances, current_top_names, run_id)
            return report

        if not os.path.exists(history_path):
            # First run — write and return clean report
            self._write_history(history_path, feature_importances, current_top_names, run_id)
            report = StabilityReport(
                dataset_id=dataset_id, run_id=run_id, is_first_run=True,
                status="PASS", tau=None,
                stability_threshold=self.stable_thresh,
                degraded_threshold=self.degraded_thresh,
                current_top_features=current_top_names,
                previous_top_features=[],
                features_gained=[], features_lost=[],
                message="First run — feature importance baseline written.",
            )
            logger.info(
                "[FeatureStability] First run for dataset_id='%s' — baseline written.",
                dataset_id,
            )
            return report

        # Load prior history
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        prev_top_names: List[str] = history.get("top_features", [])
        prev_importances: Dict[str, float] = history.get("importances", {})

        # Compute Kendall τ on shared feature set
        tau = self._kendall_tau(
            feature_importances, prev_importances, current_top_names, prev_top_names
        )

        # Features gained / lost vs last run (set comparison on top-N)
        current_set = set(current_top_names)
        prev_set    = set(prev_top_names)
        gained = sorted(current_set - prev_set)
        lost   = sorted(prev_set - current_set)

        # Assign status
        if tau is None:
            status = "SKIPPED"
            message = "Kendall τ could not be computed (no shared features)."
        elif tau >= self.stable_thresh:
            status = "PASS"
            message = (
                f"Feature importance is stable (τ={tau:.3f} ≥ {self.stable_thresh})."
            )
        elif tau >= self.degraded_thresh:
            status = "WARNING"
            message = (
                f"Feature importance degraded (τ={tau:.3f}). "
                f"Lost: {lost}. Gained: {gained}. Review upstream data."
            )
        else:
            status = "ERROR"
            message = (
                f"Feature importance severely unstable (τ={tau:.3f} < {self.degraded_thresh}). "
                f"Top features completely changed — likely upstream data issue. "
                f"Lost: {lost}. Gained: {gained}."
            )

        if status == "ERROR":
            logger.error("[FeatureStability] %s", message)
        elif status == "WARNING":
            logger.warning("[FeatureStability] %s", message)
        else:
            logger.info("[FeatureStability] %s", message)

        # Update history with current run
        self._write_history(history_path, feature_importances, current_top_names, run_id)

        return StabilityReport(
            dataset_id=dataset_id, run_id=run_id, is_first_run=False,
            status=status, tau=tau,
            stability_threshold=self.stable_thresh,
            degraded_threshold=self.degraded_thresh,
            current_top_features=current_top_names,
            previous_top_features=prev_top_names,
            features_gained=gained, features_lost=lost,
            message=message,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _top_n_sorted(
        self, importances: Dict[str, float]
    ) -> List[tuple]:
        """Return top-N (feature, score) sorted by score descending."""
        return sorted(importances.items(), key=lambda x: x[1], reverse=True)[: self.top_n]

    def _kendall_tau(
        self,
        current: Dict[str, float],
        previous: Dict[str, float],
        current_top: List[str],
        prev_top: List[str],
    ) -> Optional[float]:
        """
        Compute Kendall's τ on the union of top features from both runs.
        Assigns 0 importance to features missing from either run.
        """
        shared = list(set(current_top) | set(prev_top))
        if len(shared) < 2:
            return None
        try:
            curr_scores = np.array([current.get(f, 0.0) for f in shared])
            prev_scores = np.array([previous.get(f, 0.0) for f in shared])
            # Kendall τ via scipy
            from scipy.stats import kendalltau  # type: ignore
            tau, _ = kendalltau(curr_scores, prev_scores)
            return float(tau) if not np.isnan(tau) else None
        except ImportError:
            # Manual concordant/discordant count
            n = len(shared)
            concordant = discordant = 0
            for i in range(n):
                for j in range(i + 1, n):
                    sign_c = np.sign(curr_scores[i] - curr_scores[j])
                    sign_p = np.sign(prev_scores[i] - prev_scores[j])
                    if sign_c == sign_p:
                        concordant += 1
                    elif sign_c != 0 and sign_p != 0:
                        discordant += 1
            denom = n * (n - 1) / 2
            return float((concordant - discordant) / denom) if denom > 0 else None
        except Exception:
            return None

    def _write_history(
        self,
        path: str,
        importances: Dict[str, float],
        top_features: List[str],
        run_id: str,
    ) -> None:
        record = {
            "importances": {k: float(v) for k, v in importances.items()},
            "top_features": top_features,
            "run_id": run_id,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
