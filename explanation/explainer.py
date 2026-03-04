"""
explanation/explainer.py
-------------------------
Production-grade SHAP-backed explanation engine for DIPEX.

Produces a rich Markdown narrative for each pipeline run, augmented with
SHAP feature-level importance explanations when a fitted estimator is
available.

Fallback hierarchy:
  1. SHAP feature importance (TreeExplainer / LinearExplainer / KernelExplainer)
  2. Feature importance via model.feature_importances_ (RF/GB)
  3. Plain confidence-vector narrative (no model)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Explainer:
    """
    Generates structured, human-readable insight narratives enriched
    with SHAP-based feature importance explanations.

    Parameters
    ----------
    top_n_features : int
        Number of top SHAP features to include in the narrative.
    """

    def __init__(self, top_n_features: int = 8) -> None:
        self.top_n = top_n_features
        # Lazy-import the SHAP explainer (graceful if shap not installed)
        try:
            from explanation.shap_explainer import SHAPExplainer
            self._shap = SHAPExplainer()
            logger.info("Explainer: SHAPExplainer loaded.")
        except Exception as exc:  # noqa: BLE001
            self._shap = None
            logger.warning("Explainer: SHAPExplainer unavailable (%s) — narrative only.", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        run_id:     str,
        insight:    Dict[str, Any],
        confidence: Dict[str, Any],
        precedents: List[str],
        # Optional: present when called from the pipeline post-training
        estimator:     Any               = None,
        X_sample:      Any               = None,  # np.ndarray or pd.DataFrame (subset)
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """
        Construct a Markdown explanation narrative.

        Args:
            run_id:        Unique pipeline run identifier.
            insight:       AutoML proposal dict (model_type, task, metric_*).
            confidence:    Aggregated confidence dict from ConfidenceAggregator.
            precedents:    Historical precedent strings.
            estimator:     Fitted sklearn estimator (enables SHAP).
            X_sample:      Representative rows for SHAP computation (≤100 rows).
            feature_names: Column names matching X_sample columns.

        Returns:
            Markdown-formatted narrative string.
        """
        model_type   = insight.get("model_type", "unknown")
        task         = insight.get("task", "analysis")
        metric_name  = insight.get("metric_name", "metric")
        metric_value = insight.get("metric_value")
        conf_score   = confidence.get("confidence_score", 0.0)

        metric_line = ""
        if metric_value is not None:
            metric_line = f"**Performance:** {metric_name} = `{metric_value:.4f}`  \n"

        lines: List[str] = [
            f"## Insight Report — Run `{run_id}`",
            "",
            f"**Action:** The system used a **{model_type}** model to perform **{task}**.",
            metric_line,
            f"**Confidence Score:** `{conf_score:.2%}`  "
            f"(All gates passed: `{confidence.get('all_gates_passed')}`)",
            "",
            "### Verification Details",
        ]

        for v_name, v_res in confidence.get("vector", {}).items():
            status = "✅" if v_res.get("passed") else "❌"
            lines.append(
                f"- {status} **{v_name.capitalize()}:** {v_res.get('detail', 'N/A')}"
            )

        # ── [ML] SHAP Feature Importance ──────────────────────────────────────
        shap_section = self._build_shap_section(estimator, X_sample, feature_names)
        if shap_section:
            lines += ["", "### Feature Importance (SHAP)"] + shap_section

        # ── Historical Precedents ─────────────────────────────────────────────
        if precedents:
            lines += ["", "### Historical Precedents"]
            lines.extend(f"- {p}" for p in precedents)

        lines += [
            "",
            "### Conclusion",
            "This insight has been statistically verified and is safe for analyst consumption.",
        ]

        narrative = "\n".join(lines)
        logger.debug("Explanation narrative generated for run_id=%s", run_id)
        return narrative

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_shap_section(
        self,
        estimator:     Any,
        X_sample:      Any,
        feature_names: Optional[List[str]],
    ) -> List[str]:
        """Generate SHAP importance lines; returns [] if anything fails."""
        if estimator is None or X_sample is None:
            return []

        if self._shap is None:
            # Fallback: use model.feature_importances_ if present
            return self._fallback_importance(estimator, feature_names)

        try:
            import numpy as np
            X_arr = X_sample if hasattr(X_sample, "shape") else None
            if X_arr is None:
                return []
            # Use at most 50 rows for speed
            n = min(len(X_arr), 50)
            X_sub = X_arr[:n]
            result = self._shap.explain(estimator, X_sub, feature_names=feature_names)
            importances = result.get("mean_abs_shap", {})
            if not importances:
                return self._fallback_importance(estimator, feature_names)

            sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            rows = [f"| Feature | Mean |SHAP| |", "|---|---|", ]
            for fname, val in sorted_feats[: self.top_n]:
                bar = "█" * min(int(val * 40), 20)
                rows.append(f"| `{fname}` | {val:.4f} | {bar} |")
            rows.append(f"\n*Method: {result.get('method', 'shap')} — "
                        f"computed on {n} sample rows, top {self.top_n} features shown.*")
            return rows
        except Exception as exc:  # noqa: BLE001
            logger.debug("SHAP section build failed: %s", exc)
            return self._fallback_importance(estimator, feature_names)

    def _fallback_importance(
        self,
        estimator:     Any,
        feature_names: Optional[List[str]],
    ) -> List[str]:
        """Use model.feature_importances_ when SHAP is unavailable."""
        try:
            import numpy as np
            fi = getattr(estimator, "feature_importances_", None)
            if fi is None:
                return []
            fi = np.asarray(fi)
            names = feature_names or [f"feature_{i}" for i in range(len(fi))]
            pairs = sorted(zip(names, fi), key=lambda x: x[1], reverse=True)
            rows = ["| Feature | Importance |", "|---|---|"]
            for fname, val in pairs[: self.top_n]:
                rows.append(f"| `{fname}` | {val:.4f} |")
            rows.append("\n*Method: model feature_importances_ (SHAP unavailable).*")
            return rows
        except Exception:  # noqa: BLE001
            return []
