"""
cognitive/leakage_sentinel.py
-------------------------------
Detects data leakage patterns before any model or analysis is surfaced.

Leakage types detected:
  1. Target leakage    — features that encode the target (post-hoc features)
  2. Temporal leakage  — future data used to predict the past
  3. ID/proxy leakage  — unique identifiers or proxies that trivially predict target
  4. Group leakage     — training/test contamination via group membership
  5. Duplicate leakage — near-identical rows in both train and test

All checks are heuristic; confidence scores reflect detection certainty.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.cognitive.leakage_sentinel")


# ── Warning Dataclass ─────────────────────────────────────────────────────────

@dataclass
class LeakageWarning:
    leakage_type:  str               # target | temporal | id | group | duplicate
    column:        Optional[str]
    severity:      str               # CRITICAL | WARNING
    detail:        str
    evidence:      str = ""
    confidence:    float = 0.9
    recommendation: str = ""

    def to_dict(self) -> Dict:
        return {
            "leakage_type": self.leakage_type, "column": self.column,
            "severity": self.severity, "detail": self.detail,
            "evidence": self.evidence[:200],
            "confidence": round(self.confidence, 4),
            "recommendation": self.recommendation,
        }


# ── LeakageSentinel ───────────────────────────────────────────────────────────

class LeakageSentinel:
    """
    Screens a DataFrame (or train/test pair) for leakage before analysis/modeling.
    Called automatically by the CognitiveReasoningEngine.
    """

    # Columns whose names suggest they are IDs or proxies
    ID_PATTERNS = [
        "id", "_id", "uuid", "key", "index",
        "user_id", "customer_id", "order_id", "session_id",
    ]
    # Temporal patterns (leakage when used as future-aware feature)
    TEMPORAL_OUTCOME_KEYWORDS = [
        "outcome", "result", "final", "label", "churned", "converted",
        "purchased", "clicked", "responded", "paid",
    ]
    # High-correlation threshold for target leakage detection
    TARGET_LEAKAGE_CORR_THRESHOLD = 0.95

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self, df: pd.DataFrame,
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        group_col: Optional[str] = None,
        train_df: Optional[pd.DataFrame] = None,
        test_df:  Optional[pd.DataFrame] = None,
    ) -> List[LeakageWarning]:
        warnings: List[LeakageWarning] = []
        warnings += self._check_id_leakage(df, target_col)
        if target_col and target_col in df.columns:
            warnings += self._check_target_leakage(df, target_col)
        if date_col:
            warnings += self._check_temporal_leakage(df, date_col, target_col)
        if train_df is not None and test_df is not None:
            warnings += self._check_group_leakage(train_df, test_df, group_col)
            warnings += self._check_duplicate_leakage(train_df, test_df)
        if warnings:
            logger.warning(
                "[LeakageSentinel] %d leakage warning(s) detected (%d CRITICAL)",
                len(warnings), sum(1 for w in warnings if w.severity == "CRITICAL"),
            )
        return warnings

    def is_clean(
        self, df: pd.DataFrame,
        target_col: Optional[str] = None, **kwargs
    ) -> bool:
        return not any(
            w.severity == "CRITICAL"
            for w in self.check(df, target_col, **kwargs)
        )

    # ── Detection Methods ─────────────────────────────────────────────────────

    def _check_id_leakage(
        self, df: pd.DataFrame, target_col: Optional[str]
    ) -> List[LeakageWarning]:
        warnings = []
        id_cols = [
            c for c in df.columns
            if any(pat in c.lower() for pat in self.ID_PATTERNS)
            and c != target_col
        ]
        for col in id_cols:
            # If an ID column has near-perfect correlation with target → leakage
            if target_col and target_col in df.columns:
                try:
                    encoded = pd.factorize(df[col])[0]
                    target  = pd.to_numeric(df[target_col], errors="coerce")
                    corr = abs(np.corrcoef(encoded, target.fillna(0))[0, 1])
                    if corr > 0.7:
                        warnings.append(LeakageWarning(
                            leakage_type="id", column=col, severity="CRITICAL",
                            detail=f"ID column '{col}' correlates {corr:.2f} with target — possible proxy leakage",
                            confidence=min(corr, 0.95),
                            recommendation=f"Remove '{col}' from feature set before modeling",
                        ))
                        continue
                except Exception:  # noqa: BLE001
                    pass
            # Just flag presence of ID columns as informational
            nuniq = df[col].nunique()
            if nuniq / max(len(df), 1) > 0.95:
                warnings.append(LeakageWarning(
                    leakage_type="id", column=col, severity="WARNING",
                    detail=f"'{col}' appears to be a unique identifier ({nuniq} unique/{len(df)} rows) — should not be used as feature",
                    confidence=0.9,
                    recommendation=f"Exclude '{col}' from feature engineering",
                ))
        return warnings

    def _check_target_leakage(
        self, df: pd.DataFrame, target_col: str
    ) -> List[LeakageWarning]:
        """High correlation with target may indicate post-hoc feature."""
        warnings = []
        target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        for col in df.select_dtypes(include="number").columns:
            if col == target_col:
                continue
            try:
                corr = abs(np.corrcoef(df[col].fillna(0), target)[0, 1])
                if corr >= self.TARGET_LEAKAGE_CORR_THRESHOLD:
                    warnings.append(LeakageWarning(
                        leakage_type="target", column=col, severity="CRITICAL",
                        detail=f"'{col}' has near-perfect correlation ({corr:.3f}) with target '{target_col}' — likely leakage",
                        evidence=f"Pearson r = {corr:.4f}",
                        confidence=corr,
                        recommendation=f"Investigate if '{col}' is derived from or encodes '{target_col}'",
                    ))
                elif corr >= 0.85:
                    warnings.append(LeakageWarning(
                        leakage_type="target", column=col, severity="WARNING",
                        detail=f"'{col}' has very high correlation ({corr:.3f}) with target — verify it is not a post-hoc feature",
                        confidence=0.75,
                    ))
            except Exception:  # noqa: BLE001
                pass
        return warnings

    def _check_temporal_leakage(
        self, df: pd.DataFrame, date_col: str, target_col: Optional[str]
    ) -> List[LeakageWarning]:
        """Flag features whose names suggest they are computed after the target."""
        warnings = []
        if date_col not in df.columns:
            return []
        outcome_cols = [
            c for c in df.columns if c != target_col
            and any(kw in c.lower() for kw in self.TEMPORAL_OUTCOME_KEYWORDS)
        ]
        for col in outcome_cols:
            warnings.append(LeakageWarning(
                leakage_type="temporal", column=col, severity="WARNING",
                detail=f"'{col}' name suggests it may encode a future outcome — verify feature is computed before the prediction point",
                confidence=0.7,
                recommendation="Ensure feature timestamp < target timestamp for all rows",
            ))
        return warnings

    def _check_group_leakage(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
        group_col: Optional[str],
    ) -> List[LeakageWarning]:
        if not group_col:
            return []
        if group_col not in train_df.columns or group_col not in test_df.columns:
            return []
        train_groups = set(train_df[group_col].dropna().unique())
        test_groups  = set(test_df[group_col].dropna().unique())
        overlap      = train_groups & test_groups
        if overlap:
            return [LeakageWarning(
                leakage_type="group", column=group_col, severity="CRITICAL",
                detail=f"{len(overlap)} groups appear in both train and test sets — SUTVA violation",
                evidence=f"Overlap groups (sample): {list(overlap)[:5]}",
                confidence=1.0,
                recommendation=f"Use group-aware splitting: split by '{group_col}' not by row index",
            )]
        return []

    def _check_duplicate_leakage(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> List[LeakageWarning]:
        try:
            combined = pd.merge(train_df, test_df, how="inner")
            n_dups   = len(combined)
            if n_dups > 0:
                return [LeakageWarning(
                    leakage_type="duplicate", column=None, severity="CRITICAL",
                    detail=f"{n_dups} exact-duplicate rows appear in both train and test",
                    confidence=1.0,
                    recommendation="Remove duplicates before splitting or use a dedup key",
                )]
        except Exception:  # noqa: BLE001
            pass
        return []
