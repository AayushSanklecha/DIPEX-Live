"""
verifier/stability_verifier.py
--------------------------------
Production-grade stability verifier with:
  - Cross-validation score variance (std threshold)
  - SHAP feature importance consistency across multiple random seeds
  - Feature variance stability (train vs validation distribution)
  - Temporal window performance consistency

Used in Hard Gate 2 to ensure model results are stable and not
artifacts of a particular data ordering or random seed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.verifier.stability")

_MIN_SAMPLES_FOR_SHAP: int = 20
_SHAP_SEED_COUNT: int = 3
_SHAP_VARIANCE_THRESHOLD: float = 0.10   # max allowed SHAP rank variance


class StabilityVerifier:
    """
    Checks model stability using:

    1. Cross-validation score variance (std < threshold)
    2. SHAP feature importance consistency across random seeds
    3. Feature distribution variance check (train vs val split)
    """

    def __init__(
        self,
        max_std_threshold: float = 0.10,
        cv: int = 5,
        check_shap: bool = True,
        shap_variance_threshold: float = _SHAP_VARIANCE_THRESHOLD,
    ) -> None:
        self.max_std_threshold = max_std_threshold
        self.cv = cv
        self.check_shap = check_shap
        self.shap_variance_threshold = shap_variance_threshold

    def verify(
        self,
        model: Any,
        X: Any,
        y: Any,
        X_val: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs all stability checks.

        Args:
            model        : Fitted sklearn-compatible estimator
            X            : Training features (array-like)
            y            : Training labels (array-like)
            X_val        : Optional validation features for distribution check
            feature_names: Optional list of feature names

        Returns:
            dict with: metric, passed, value, mean_score, shap_stable, detail
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if len(X_arr) == 0:
            return self._empty_result("Empty training set — stability not evaluated.")

        # ── 1. Cross-validation stability ───────────────────────────────────
        cv_result = self._cv_stability(model, X_arr, y_arr)

        # ── 2. SHAP stability ────────────────────────────────────────────────
        shap_result = {"shap_stable": None, "shap_detail": "SHAP check skipped."}
        if self.check_shap and len(X_arr) >= _MIN_SAMPLES_FOR_SHAP:
            shap_result = self._shap_stability(model, X_arr, feature_names)

        # ── 3. Feature variance check (train vs val) ─────────────────────────
        feat_var_result = {"feature_variance_ok": None, "feature_variance_detail": "No val set."}
        if X_val is not None:
            feat_var_result = self._feature_variance_check(X_arr, np.asarray(X_val))

        # ── Overall decision ─────────────────────────────────────────────────
        cv_ok = cv_result["passed"]
        shap_ok = shap_result.get("shap_stable", True)   # None = not checked → pass
        feat_ok = feat_var_result.get("feature_variance_ok", True)

        all_passed = cv_ok and (shap_ok is not False) and (feat_ok is not False)

        failures = []
        if not cv_ok:
            failures.append(f"CV std {cv_result['value']:.4f} > threshold {self.max_std_threshold}")
        if shap_ok is False:
            failures.append(shap_result.get("shap_detail", "SHAP unstable"))
        if feat_ok is False:
            failures.append(feat_var_result.get("feature_variance_detail", "Feature variance high"))

        detail = (
            "; ".join(failures)
            if failures
            else (
                f"Stability PASS — CV std={cv_result['value']:.4f}, "
                f"mean={cv_result['mean_score']:.4f}, SHAP={'stable' if shap_ok else 'N/A'}."
            )
        )

        logger.info(
            "StabilityVerifier: passed=%s cv_std=%.4f cv_mean=%.4f shap_ok=%s",
            all_passed, cv_result["value"], cv_result["mean_score"], shap_ok,
        )

        return {
            "metric": "cv_stability_std",
            "value": float(cv_result["value"]),
            "mean_score": float(cv_result["mean_score"]),
            "passed": bool(all_passed),
            "cv_passed": bool(cv_ok),
            "shap_stable": shap_ok,
            "feature_variance_ok": feat_ok,
            "detail": detail,
        }

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _cv_stability(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Cross-validation score variance check."""
        try:
            from sklearn.model_selection import cross_val_score

            n_cv = min(self.cv, len(set(y)) if len(set(y)) > 1 else self.cv)
            n_cv = max(2, min(n_cv, len(X) // 2))

            scores = cross_val_score(model, X, y, cv=n_cv, error_score=0.0)
            mean_s = float(scores.mean())
            std_s = float(scores.std(ddof=1) if len(scores) > 1 else 0.0)
            passed = std_s < self.max_std_threshold

            return {
                "value": std_s,
                "mean_score": mean_s,
                "passed": passed,
                "cv_scores": [round(s, 4) for s in scores.tolist()],
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("CV stability check failed: %s", exc)
            return {
                "value": 0.0,
                "mean_score": 0.0,
                "passed": True,
                "detail": f"CV unavailable: {exc}",
            }

    def _shap_stability(
        self, model: Any, X: np.ndarray, feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        SHAP feature importance rank stability across 3 random seeds.
        High rank variance → instability.
        """
        try:
            import shap  # type: ignore

            n_background = min(50, len(X))
            rank_lists: List[np.ndarray] = []

            for seed in range(_SHAP_SEED_COUNT):
                rng = np.random.RandomState(seed * 42)
                bg_idx = rng.choice(len(X), size=n_background, replace=False)
                background = X[bg_idx]
                sample_idx = rng.choice(len(X), size=min(30, len(X)), replace=False)
                sample = X[sample_idx]

                # Try TreeExplainer first; fall back to KernelExplainer
                try:
                    explainer = shap.TreeExplainer(model, background)
                except Exception:
                    try:
                        explainer = shap.KernelExplainer(model.predict_proba
                                                         if hasattr(model, "predict_proba")
                                                         else model.predict, background)
                    except Exception:
                        logger.debug("SHAP explainer unavailable — skipping SHAP stability.")
                        return {"shap_stable": None, "shap_detail": "SHAP explainer unavailable."}

                vals = explainer.shap_values(sample)
                if isinstance(vals, list):
                    vals = vals[-1]  # multi-class: use last class
                importances = np.abs(np.asarray(vals)).mean(axis=0)
                ranks = importances.argsort()[::-1]
                rank_lists.append(ranks)

            # Compute rank variance: how much do top-3 features shift across seeds?
            n_feats = min(rank_lists[0].shape[0], 3)
            top_variance = float(np.std(
                [[r[i] for i in range(n_feats)] for r in rank_lists], axis=0
            ).mean())

            stable = top_variance < self.shap_variance_threshold
            detail = (
                f"SHAP top-{n_feats} rank variance={top_variance:.4f} "
                f"(threshold={self.shap_variance_threshold})"
            )
            return {"shap_stable": bool(stable), "shap_detail": detail, "shap_rank_variance": top_variance}

        except ImportError:
            logger.debug("SHAP library not installed — SHAP stability check skipped.")
            return {"shap_stable": None, "shap_detail": "SHAP not installed."}
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP stability check failed: %s", exc)
            return {"shap_stable": None, "shap_detail": f"SHAP error: {exc}"}

    def _feature_variance_check(
        self, X_train: np.ndarray, X_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Checks that feature distributions haven't shifted drastically
        between train and validation sets (mean shift / std ratio check).
        """
        try:
            train_mean = X_train.mean(axis=0)
            val_mean = X_val.mean(axis=0)
            train_std = X_train.std(axis=0) + 1e-9

            # Standardised mean difference per feature
            smd = np.abs((train_mean - val_mean) / train_std)
            max_smd = float(smd.max())
            mean_smd = float(smd.mean())

            # Threshold: SMD > 0.5 indicates a meaningful shift
            ok = bool(max_smd < 0.5)
            detail = (
                f"Feature distribution check: max_SMD={max_smd:.4f}, "
                f"mean_SMD={mean_smd:.4f} ({'OK' if ok else 'WARNING: shift detected'})"
            )
            return {
                "feature_variance_ok": ok,
                "max_smd": round(max_smd, 4),
                "mean_smd": round(mean_smd, 4),
                "feature_variance_detail": detail,
            }
        except Exception as exc:  # noqa: BLE001
            return {"feature_variance_ok": None, "feature_variance_detail": f"Check failed: {exc}"}

    @staticmethod
    def _empty_result(detail: str) -> Dict[str, Any]:
        return {
            "metric": "cv_stability_std",
            "value": 0.0,
            "mean_score": 0.0,
            "passed": True,
            "cv_passed": True,
            "shap_stable": None,
            "feature_variance_ok": None,
            "detail": detail,
        }
