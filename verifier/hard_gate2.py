"""
verifier/hard_gate2.py
-----------------------
Step 5 — Independent Verifier Stack (Hard Gate 2).

This module evaluates a single model candidate using an ensemble of
independent verifiers. Any failing check results in a REJECT decision:

  - Statistical validity (hypothesis test, p-value, confidence interval)
  - Baseline comparison (must beat naive model)
  - Stability (cross-split consistency)
  - Permutation validation (performance vs label shuffling)
  - Leakage detection (time-aware split enforcement)
  - Drift robustness (stable across temporal windows)
  - Compliance verification (domain/regulatory rules on outputs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging
import numpy as np
import pandas as pd

from .statistical_verifier import StatisticalVerifier
from .baseline_verifier import BaselineVerifier
from .stability_verifier import StabilityVerifier
from .drift_verifier import DriftVerifier
from .domain_verifier import DomainVerifier
from .confidence_aggregator import ConfidenceAggregator
from profiling.drift_detector import DriftDetector

# [ML] Pipeline success predictor — warns before expensive verification
try:
    from verifier.pipeline_success_predictor import PipelineSuccessPredictor as _PSP
    _psp = _PSP()
except Exception:  # noqa: BLE001
    _psp = None

logger = logging.getLogger(__name__)


_BLOCKING_BOOL = True  # any failed check blocks the candidate


@dataclass
class HardGate2Result:
    """Structured output from Hard Gate 2."""

    run_id: str
    decision: str  # "PASS" | "REJECT"
    reason: str
    suppress_learning: bool
    failed_checks: List[Dict[str, Any]] = field(default_factory=list)
    passed_checks: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "decision": self.decision,
            "reason": self.reason,
            "suppress_learning": self.suppress_learning,
            "failed_checks": self.failed_checks,
            "passed_checks": self.passed_checks,
            "confidence": self.confidence or {},
        }


class HardGate2:
    """
    Orchestrates the independent verifier stack for a single candidate.

    Expected inputs (artifacts dict mirrors AutoMLProposer output):
      - estimator: fitted sklearn estimator
      - X_train, y_train: training data
      - y_true_val, y_pred_val: validation targets / predictions
      - timestamp_col: optional name of timestamp column in original df
      - df: original DataFrame (for temporal checks & drift)
      - baseline_df: optional baseline DataFrame for drift robustness
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config or {}

        v_cfg = self.config.get("pipeline", {}).get("verifier", {})
        qa_cfg = self.config.get("pipeline", {}).get("qa_gate", {})

        self._p_value_threshold: float = float(
            v_cfg.get("statistical_p_value", 0.05)
        )
        self._min_baseline_improvement: float = float(
            v_cfg.get("min_baseline_improvement", 0.05)
        )
        self._cv_stability_threshold: float = float(
            v_cfg.get("cv_stability_threshold", 0.1)
        )
        self._max_drift_psi: float = float(qa_cfg.get("max_drift_psi", 0.25))

        self._stat = StatisticalVerifier(p_value_threshold=self._p_value_threshold)
        self._baseline = BaselineVerifier(
            min_improvement=self._min_baseline_improvement
        )
        self._stability = StabilityVerifier(
            max_std_threshold=self._cv_stability_threshold
        )
        self._drift_verifier = DriftVerifier(psi_threshold=self._max_drift_psi)
        self._aggregator = ConfidenceAggregator()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HardGate2":
        return cls(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_candidate(
        self,
        run_id: str,
        candidate: Dict[str, Any],
        artifacts: Dict[str, Any],
        domain_rules: Optional[List[Dict[str, Any]]] = None,
        baseline_df: Optional[pd.DataFrame] = None,
    ) -> HardGate2Result:
        """
        Evaluates a single model candidate and returns a HardGate2Result.
        """
        model_type = candidate.get("model_type", "unknown")
        task_type = candidate.get("task", "classification")
        metric_name = candidate.get("metric_name", "metric")
        metric_value = float(candidate.get("metric_value", 0.0))

        y_true_val = np.asarray(artifacts.get("y_true_val"))
        y_pred_val = np.asarray(artifacts.get("y_pred_val"))
        estimator = artifacts.get("estimator")
        X_train = artifacts.get("X_train")
        y_train = artifacts.get("y_train")
        df: Optional[pd.DataFrame] = artifacts.get("df")
        timestamp_col: Optional[str] = artifacts.get("timestamp_col")

        failed: List[Dict[str, Any]] = []
        passed: List[Dict[str, Any]] = []

        # [ML] Pre-flight: predict run success before running expensive verifier stack
        if _psp is not None:
            ctx = {
                "null_rate":       artifacts.get("null_rate", 0.0),
                "drift_detected":  artifacts.get("drift_detected", False),
                "quality_score":   artifacts.get("quality_score", 1.0),
                "row_count":       len(X_train) if X_train is not None else 0,
                "n_columns":       X_train.shape[1] if hasattr(X_train, "shape") else 0,
                "anomaly_count":   artifacts.get("anomaly_count", 0),
                "schema_match":    artifacts.get("schema_match", True),
                "cv_score":        metric_value,
                "columns_drifted": artifacts.get("columns_drifted", 0),
            }
            psp_result = _psp.predict(ctx)
            logger.info(
                "[ML] HardGate2 pre-flight prediction: %s (prob=%.3f) — %s",
                psp_result["prediction"], psp_result["success_prob"],
                "; ".join(psp_result.get("warnings", [])) or "no warnings",
            )
            if psp_result["prediction"] == "LIKELY_FAILURE":
                logger.warning(
                    "[ML] PipelineSuccessPredictor flagged LIKELY_FAILURE — "
                    "proceeding with verifier stack but expect failures."
                )

        # 1. Statistical validity (hypothesis test + simple CI)
        stat_res = self._stat.verify(
            y_true=y_true_val, y_pred=y_pred_val, task_type=task_type
        )
        stat_res = self._attach_confidence_interval(
            stat_res, task_type=task_type, y_true=y_true_val, y_pred=y_pred_val
        )
        self._bucket("statistical", stat_res, failed, passed)

        # 2. Baseline comparison (must beat naive model)
        base_res = self._baseline.verify(
            model_score=metric_value, task_type=task_type, y_train=y_train
        )
        self._bucket("baseline", base_res, failed, passed)

        # 3. Stability (cross-split consistency)
        if estimator is not None and X_train is not None and y_train is not None:
            stab_res = self._stability.verify(
                model=estimator, X=X_train, y=np.asarray(y_train)
            )
        else:
            stab_res = {
                "metric": "cv_stability_std",
                "value": None,
                "mean_score": metric_value,
                "passed": True,
                "detail": "Stability not evaluated (missing estimator or training data).",
            }
        self._bucket("stability", stab_res, failed, passed)

        # 4. Permutation validation
        perm_res = self._permutation_validation(
            y_true=y_true_val, y_pred=y_pred_val, task_type=task_type, metric_name=metric_name
        )
        self._bucket("permutation", perm_res, failed, passed)

        # 5. Leakage detection (time separation enforced)
        leak_res = self._leakage_check(
            df=df,
            timestamp_col=timestamp_col,
            y_true_val=y_true_val,
            artifacts=artifacts,
        )
        if leak_res is not None:
            self._bucket("leakage", leak_res, failed, passed)

        # 6. Drift robustness (stable across temporal windows)
        drift_robust_res = self._temporal_robustness_check(
            df=df,
            baseline_df=baseline_df,
            timestamp_col=timestamp_col,
            y_true_val=y_true_val,
            y_pred_val=y_pred_val,
            task_type=task_type,
        )
        if drift_robust_res is not None:
            self._bucket("drift_robustness", drift_robust_res, failed, passed)

        # 7. Compliance verification (domain / regulatory rules)
        domain_verifier = DomainVerifier(domain_rules or [])
        domain_res = domain_verifier.verify(y_pred_val)
        self._bucket("domain", domain_res, failed, passed)

        # Aggregate confidence vector
        verifier_vector: Dict[str, Dict[str, Any]] = {
            "statistical": stat_res,
            "baseline": base_res,
            "stability": stab_res,
            "permutation": perm_res,
            "domain": domain_res,
        }
        if drift_robust_res is not None:
            verifier_vector["drift_robustness"] = drift_robust_res
        if leak_res is not None:
            verifier_vector["leakage"] = leak_res

        confidence = self._aggregator.aggregate(verifier_vector)

        has_failures = any(not chk.get("passed", False) for chk in failed)

        if has_failures or not confidence.get("all_gates_passed", True):
            decision = "REJECT"
            reason = (
                "Hard Gate 2 REJECTED: one or more verification checks failed. "
                f"First failure: {failed[0]['detail'] if failed else 'confidence penalty'}"
            )
            suppress_learning = True
        else:
            decision = "PASS"
            reason = "Hard Gate 2 PASSED — all verifier checks satisfied."
            suppress_learning = False

        logger.info(
            "Hard Gate 2 decision for run_id=%s, model=%s, task=%s → %s",
            run_id,
            model_type,
            task_type,
            decision,
        )

        return HardGate2Result(
            run_id=run_id,
            decision=decision,
            reason=reason,
            suppress_learning=suppress_learning,
            failed_checks=failed,
            passed_checks=passed,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bucket(
        name: str,
        res: Dict[str, Any],
        failed: List[Dict[str, Any]],
        passed: List[Dict[str, Any]],
    ) -> None:
        record = {"name": name, **res}
        if res.get("passed", False):
            passed.append(record)
        else:
            failed.append(record)

    def _permutation_validation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str,
        metric_name: str,
        n_permutations: int = 100,
    ) -> Dict[str, Any]:
        """
        Simple permutation test: compares observed performance to the
        distribution obtained by shuffling labels.
        """
        if y_true.size == 0:
            return {
                "metric": "permutation_p_value",
                "value": 1.0,
                "passed": False,
                "detail": "No samples available for permutation validation.",
            }

        if task_type == "regression":
            # Lower MSE is better
            from sklearn.metrics import mean_squared_error

            obs = float(mean_squared_error(y_true, y_pred))

            perm_scores: List[float] = []
            for _ in range(n_permutations):
                perm = np.random.permutation(y_true)
                perm_scores.append(float(mean_squared_error(perm, y_pred)))

            perm_scores = np.asarray(perm_scores)
            # p-value: probability that permuted MSE is as low or lower than observed
            p_val = float(((perm_scores <= obs).sum() + 1) / (n_permutations + 1))
            passed = p_val < self._p_value_threshold
        else:
            # Classification accuracy
            from sklearn.metrics import accuracy_score

            obs = float(accuracy_score(y_true, y_pred))
            perm_scores = []
            for _ in range(n_permutations):
                perm = np.random.permutation(y_true)
                perm_scores.append(float(accuracy_score(perm, y_pred)))

            perm_scores = np.asarray(perm_scores)
            # p-value: probability that permuted accuracy is as high or higher than observed
            p_val = float(((perm_scores >= obs).sum() + 1) / (n_permutations + 1))
            passed = p_val < self._p_value_threshold

        return {
            "metric": "permutation_p_value",
            "value": float(p_val),
            "passed": bool(passed),
            "detail": (
                f"Permutation test p-value={p_val:.4f} "
                f"for metric '{metric_name}' (threshold={self._p_value_threshold})."
            ),
        }

    @staticmethod
    def _attach_confidence_interval(
        res: Dict[str, Any],
        task_type: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Attaches a simple normal-approximation confidence interval to
        the statistical verifier result when applicable.
        """
        if task_type != "classification":
            return res

        from sklearn.metrics import accuracy_score

        n = len(y_true)
        if n == 0:
            return res

        acc = float(accuracy_score(y_true, y_pred))
        z = 1.96  # ~95% CI
        se = np.sqrt(acc * (1.0 - acc) / n)
        lower = max(0.0, acc - z * se)
        upper = min(1.0, acc + z * se)

        res = dict(res)  # shallow copy
        res["confidence_interval"] = {
            "level": 1.0 - alpha,
            "lower": round(float(lower), 6),
            "upper": round(float(upper), 6),
            "center": round(acc, 6),
        }
        return res

    def _leakage_check(
        self,
        df: Optional[pd.DataFrame],
        timestamp_col: Optional[str],
        y_true_val: np.ndarray,
        artifacts: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Enforces basic time-based separation when a timestamp column and
        validation indices are available. If the earliest validation
        timestamp is earlier than the latest training timestamp, we flag
        potential leakage.
        """
        val_index = artifacts.get("val_index")
        train_index = artifacts.get("train_index")

        if (
            df is None
            or timestamp_col is None
            or timestamp_col not in df.columns
            or val_index is None
            or train_index is None
        ):
            return None

        ts_series = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
        train_ts = ts_series.loc[train_index].dropna()
        val_ts = ts_series.loc[val_index].dropna()

        if train_ts.empty or val_ts.empty:
            return None

        max_train = train_ts.max()
        min_val = val_ts.min()

        if min_val <= max_train:
            return {
                "metric": "time_leakage",
                "value": 1,
                "passed": False,
                "detail": (
                    "Potential time leakage detected: earliest validation timestamp "
                    f"{min_val} is not strictly after latest training timestamp {max_train}."
                ),
            }

        return {
            "metric": "time_leakage",
            "value": 0,
            "passed": True,
            "detail": (
                "Time-based split respected: validation window starts after training window."
            ),
        }

    def _temporal_robustness_check(
        self,
        df: Optional[pd.DataFrame],
        baseline_df: Optional[pd.DataFrame],
        timestamp_col: Optional[str],
        y_true_val: np.ndarray,
        y_pred_val: np.ndarray,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Checks that model performance is reasonably stable across temporal
        windows using the validation segment.
        """
        if df is None or timestamp_col is None or timestamp_col not in df.columns:
            return None

        # Use only rows that participated in validation (by index)
        # If validation indices not provided, fall back to full df.
        # For now, we approximate by binning validation rows by timestamp quantiles.
        ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
        mask = ts.notna()
        ts = ts[mask]
        if ts.empty:
            return None

        # Align y_true_val/y_pred_val length to ts length if necessary
        n = min(len(ts), len(y_true_val), len(y_pred_val))
        if n < 10:
            return None

        ts = ts.iloc[:n]
        yt = y_true_val[:n]
        yp = y_pred_val[:n]

        # Bin into temporal quartiles
        bins = pd.qcut(ts.rank(method="first"), q=4, labels=False)

        from sklearn.metrics import accuracy_score, mean_squared_error

        scores: List[float] = []
        for b in range(4):
            idx = bins[bins == b].index
            if len(idx) < 3:
                continue
            if task_type == "regression":
                scores.append(
                    float(
                        mean_squared_error(
                            yt[ts.index.get_indexer(idx)], yp[ts.index.get_indexer(idx)]
                        )
                    )
                )
            else:
                scores.append(
                    float(
                        accuracy_score(
                            yt[ts.index.get_indexer(idx)], yp[ts.index.get_indexer(idx)]
                        )
                    )
                )

        if not scores:
            return None

        scores_arr = np.asarray(scores)
        mean_score = float(scores_arr.mean())
        std_score = float(scores_arr.std(ddof=1)) if len(scores_arr) > 1 else 0.0

        # Threshold is aligned with cv_stability_threshold for consistency
        passed = std_score <= self._cv_stability_threshold

        return {
            "metric": "temporal_performance_std",
            "value": std_score,
            "mean_score": mean_score,
            "passed": bool(passed),
            "detail": (
                f"Temporal window performance std={std_score:.4f} "
                f"(threshold={self._cv_stability_threshold})."
            ),
        }

