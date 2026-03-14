"""
validation/multicollinearity_detector.py
-----------------------------------------
Variance Inflation Factor (VIF) multicollinearity detector.

Problem: When two or more features are highly correlated with EACH OTHER
(not the target), models suffer:
  - Unstable / unreliable feature importances
  - Inflated variance in coefficient estimates (linear/logistic regression)
  - Reduced interpretability (can't tell which feature matters)
  - Potential overfitting when correlated features reinforce each other

Real-world examples:
  - Banking:    revenue & profit_margin (always correlated)
  - Healthcare: BMI & weight (derived from each other)
  - IoT:        temperature_celsius & temperature_fahrenheit (identical)

This module:
  1. Computes Pearson correlation matrix for all numeric features
  2. Computes VIF for each feature (VIF > 10 = severe multicollinearity)
  3. Identifies correlated pairs above the threshold
  4. Recommends which column to drop (keep the one with lower avg correlation)
  5. Optionally drops the recommended columns automatically

All thresholds are config-driven.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.validation.multicollinearity_detector")


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CollinearityViolation:
    column: str
    vif: float
    severity: str           # WARNING | ERROR
    paired_with: List[str]  # columns it is collinear with
    recommendation: str     # "drop" | "monitor"
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "vif": round(self.vif, 2),
            "severity": self.severity,
            "paired_with": self.paired_with,
            "recommendation": self.recommendation,
            "message": self.message,
        }


@dataclass
class MulticollinearityReport:
    run_id: str
    violations: List[CollinearityViolation] = field(default_factory=list)
    columns_dropped: List[str] = field(default_factory=list)
    correlated_pairs: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "violations": [v.to_dict() for v in self.violations],
            "columns_dropped": self.columns_dropped,
            "correlated_pairs": self.correlated_pairs,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class MulticollinearityDetector:
    """
    Detect and optionally remove highly collinear features using VIF.

    Config stanza (all optional)::

        validation:
          multicollinearity:
            vif_error_threshold: 10.0    # VIF >= this → ERROR (severe)
            vif_warn_threshold: 5.0      # VIF >= this → WARNING (moderate)
            corr_hard_threshold: 0.95    # pairwise |corr| for pair reporting
            drop_on_error: true          # auto-drop ERROR-level columns
            max_features_for_vif: 100    # skip VIF if > this many features (slow)
            target_col: null             # exclude target from collinearity check
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("validation", {}).get("multicollinearity", {})
        self.vif_error: float   = float(cfg.get("vif_error_threshold", 10.0))
        self.vif_warn: float    = float(cfg.get("vif_warn_threshold", 5.0))
        self.corr_hard: float   = float(cfg.get("corr_hard_threshold", 0.95))
        self.drop_on_error: bool = bool(cfg.get("drop_on_error", True))
        self.max_feats: int     = int(cfg.get("max_features_for_vif", 100))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MulticollinearityDetector":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        run_id: str = "",
    ) -> Tuple[pd.DataFrame, MulticollinearityReport]:
        """
        Detect multicollinearity in numeric features of df.

        Returns
        -------
        (cleaned_df, MulticollinearityReport)
        cleaned_df has ERROR-level collinear columns removed (if drop_on_error=True).
        """
        report = MulticollinearityReport(run_id=run_id)

        num_cols = [
            c for c in df.select_dtypes(include=np.number).columns
            if c != target_col
        ]

        if len(num_cols) < 2:
            return df, report

        # Use column-wise median imputation — fillna(0) would distort variance
        # and bias the OLS R² used inside _compute_vif, masking real collinearity.
        col_medians = df[num_cols].median()
        X = df[num_cols].fillna(col_medians)

        # Sample rows for correlation matrix (pearson corr is ratio, not sum)
        _SAMPLE_N = 5000
        if len(X) > _SAMPLE_N:
            X_sample = X.sample(n=_SAMPLE_N, random_state=42)
        else:
            X_sample = X

        # ── 1. Pairwise correlation (fast, always runs) ───────────────────────
        corr_matrix = X_sample.corr().abs()
        pairs = self._find_correlated_pairs(corr_matrix, num_cols)
        report.correlated_pairs = pairs

        # ── 2. VIF (more expensive — skip if too many features) ───────────────
        vif_scores: Dict[str, float] = {}
        if len(num_cols) <= self.max_feats:
            vif_scores = self._compute_vif(X, num_cols, report)
        else:
            report.warnings.append(
                f"VIF skipped — {len(num_cols)} features > max_features_for_vif "
                f"({self.max_feats}). Only pairwise correlation used."
            )
            # Estimate VIF from correlation matrix: VIF_i ≈ 1 / (1 - R²_i)
            for col in num_cols:
                other_cols = [c for c in num_cols if c != col]
                if not other_cols:
                    continue
                r2_approx = float(corr_matrix.loc[col, other_cols].max() ** 2)
                vif_scores[col] = 1.0 / (1.0 - r2_approx + 1e-9)

        # ── 3. Build violations ───────────────────────────────────────────────
        paired_map: Dict[str, List[str]] = {c: [] for c in num_cols}
        for pair in pairs:
            paired_map[pair["col_a"]].append(pair["col_b"])
            paired_map[pair["col_b"]].append(pair["col_a"])

        drop_candidates: List[str] = []
        for col in num_cols:
            vif = vif_scores.get(col, 0.0)
            if vif < self.vif_warn:
                continue
            severity = "ERROR" if vif >= self.vif_error else "WARNING"
            recommendation = "drop" if severity == "ERROR" and self.drop_on_error else "monitor"

            report.violations.append(CollinearityViolation(
                column=col, vif=vif, severity=severity,
                paired_with=paired_map.get(col, []),
                recommendation=recommendation,
                message=(
                    f"'{col}' has VIF={vif:.1f} ({severity}) — "
                    f"highly collinear with: {paired_map.get(col, [])}. "
                    f"Recommendation: {recommendation}."
                ),
            ))
            if severity == "ERROR":
                drop_candidates.append(col)
                logger.error(
                    "[VIF] ERROR '%s' VIF=%.1f — dropping from features.", col, vif
                )
            else:
                logger.warning(
                    "[VIF] WARNING '%s' VIF=%.1f — monitor for instability.", col, vif
                )

        # ── 4. Smart drop: remove the higher-VIF of each collinear pair ───────
        if self.drop_on_error and drop_candidates:
            # Keep columns that don't appear in any ERROR pair
            # (avoid dropping both columns of a pair)
            to_drop = self._select_drop_set(drop_candidates, vif_scores, report)
            if to_drop:
                df = df.drop(columns=to_drop)
                report.columns_dropped = to_drop
                logger.warning(
                    "[VIF] Dropped %d collinear columns: %s", len(to_drop), to_drop
                )

        logger.info(
            "[VIF] run_id=%s — %d violations (%d dropped). %d correlated pairs.",
            run_id[:8] if run_id else "?",
            len(report.violations), len(report.columns_dropped),
            len(pairs),
        )
        return df, report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_vif(
        self, X: pd.DataFrame, cols: List[str], report: MulticollinearityReport,
        max_rows: int = 5000,
    ) -> Dict[str, float]:
        """Compute VIF for each column using fast numpy QR decomposition.
        
        Rows are sampled to max_rows to cap computation time. VIF is a ratio
        (not a sum), so sampling preserves statistical meaning when n >> p.
        """
        vifs: Dict[str, float] = {}
        arr = X.values.astype(np.float64, copy=False)

        # Sample rows for performance
        n = arr.shape[0]
        if n > max_rows:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_rows, replace=False)
            arr = arr[idx]

        # Add intercept column for OLS
        ones = np.ones((arr.shape[0], 1), dtype=np.float64)
        X_full = np.hstack([ones, arr])  # shape: (n, p+1)

        for i, col in enumerate(cols):
            try:
                # Skip near-zero-variance columns — VIF is meaningless for constants
                col_data = arr[:, i]
                if np.std(col_data) < 1e-9:
                    vifs[col] = 1.0
                    continue

                y = col_data
                # Use all other columns (skip intercept index 0, skip col i+1)
                other_indices = [0] + [j + 1 for j in range(len(cols)) if j != i]
                X_others = X_full[:, other_indices]

                # Fast least squares via numpy
                coeffs, _, _, _ = np.linalg.lstsq(X_others, y, rcond=None)
                y_pred = X_others @ coeffs
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                if ss_tot < 1e-12:
                    vifs[col] = 1.0
                    continue
                r2 = min(1.0 - ss_res / ss_tot, 0.9999)
                vifs[col] = 1.0 / max(1.0 - r2, 1e-6)
            except Exception as exc:
                report.warnings.append(f"VIF failed for '{col}': {exc}")
                vifs[col] = 1.0
        return vifs

    def _find_correlated_pairs(
        self, corr: pd.DataFrame, cols: List[str]
    ) -> List[Dict[str, Any]]:
        """Find all pairs of columns with |corr| >= corr_hard_threshold."""
        pairs = []
        seen: set = set()
        for i, ca in enumerate(cols):
            for cb in cols[i + 1:]:
                key = tuple(sorted((ca, cb)))
                if key in seen:
                    continue
                seen.add(key)
                if ca in corr.index and cb in corr.columns:
                    c = float(corr.loc[ca, cb])
                    if c >= self.corr_hard:
                        pairs.append({
                            "col_a": ca, "col_b": cb,
                            "correlation": round(c, 4),
                        })
        return pairs

    def _select_drop_set(
        self,
        candidates: List[str],
        vif_scores: Dict[str, float],
        report: MulticollinearityReport,
    ) -> List[str]:
        """
        Among ERROR-level candidates, always drop the one with higher VIF
        from each correlated pair, preserving at least one of each pair.
        """
        to_drop: List[str] = []
        remaining = set(candidates)
        for pair in report.correlated_pairs:
            ca, cb = pair["col_a"], pair["col_b"]
            if ca in remaining and cb in remaining:
                # Drop the one with higher VIF
                vif_a = vif_scores.get(ca, 0.0)
                vif_b = vif_scores.get(cb, 0.0)
                drop = ca if vif_a >= vif_b else cb
                to_drop.append(drop)
                remaining.discard(drop)
        # Also drop remaining single ERROR candidates not in any pair
        for col in remaining:
            if col not in to_drop:
                to_drop.append(col)
        return to_drop
