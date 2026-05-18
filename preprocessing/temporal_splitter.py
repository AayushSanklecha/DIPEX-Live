"""
preprocessing/temporal_splitter.py
------------------------------------
Time-aware train/test splitting for temporal and panel data.

Problem: Standard k-fold cross-validation on time-series data leaks
future into the past — the model trains on data from 2024 to predict
2023 events. This makes CV scores wildly optimistic (e.g., 92% reported
but 74% actual on new data). This is one of the most common accuracy
killers in real-world industry datasets.

This module provides:
  1. Auto-detection of temporal columns in the DataFrame
  2. Walk-forward (expanding window) cross-validation splits
  3. Sliding window CV splits (fixed-size training window)
  4. A drop-in replacement for sklearn's StratifiedKFold/KFold that
     respects time ordering

All thresholds are config-driven. Falls back to standard k-fold
gracefully if no temporal column is detected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.preprocessing.temporal_splitter")


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalSplitInfo:
    """Metadata about the temporal split strategy chosen."""
    strategy: str               # "walk_forward" | "sliding_window" | "standard_kfold"
    temporal_col: Optional[str] # column used for ordering
    n_splits: int
    min_train_size: int
    max_test_size: int
    reason: str                 # human-readable explanation of choice
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "temporal_col": self.temporal_col,
            "n_splits": self.n_splits,
            "min_train_size": self.min_train_size,
            "max_test_size": self.max_test_size,
            "reason": self.reason,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Splitter
# ─────────────────────────────────────────────────────────────────────────────

class TemporalSplitter:
    """
    Time-aware cross-validation splitter.

    Config stanza (all optional)::

        preprocessing:
          temporal:
            strategy: "auto"            # "walk_forward" | "sliding_window" | "auto"
            n_splits: 5                 # number of CV folds
            temporal_col: null          # explicit column name; null = auto-detect
            min_train_fraction: 0.5     # min fraction of data in first training fold
            test_fraction: 0.20         # size of each test window
            sliding_window_size: null   # fixed window rows for sliding (null = expanding)
            auto_detect_patterns:       # column name patterns to auto-detect as datetime
              - date
              - time
              - timestamp
              - created
              - updated
              - dt
    """

    _DEFAULT_PATTERNS = ["date", "time", "timestamp", "created", "updated", "dt", "day", "month"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("preprocessing", {}).get("temporal", {})
        self.strategy: str          = cfg.get("strategy", "auto")
        self.n_splits: int          = int(cfg.get("n_splits", 5))
        self.explicit_col: Optional[str] = cfg.get("temporal_col")
        self.min_train_frac: float  = float(cfg.get("min_train_fraction", 0.5))
        self.test_frac: float       = float(cfg.get("test_fraction", 0.20))
        self.sliding_size: Optional[int] = cfg.get("sliding_window_size")
        self.detect_patterns: List[str]  = cfg.get(
            "auto_detect_patterns", self._DEFAULT_PATTERNS
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TemporalSplitter":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_temporal_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect the most likely temporal column in df.

        Priority order:
        1. Explicit config column (if valid)
        2. Datetime-dtype columns
        3. Object columns matching known name patterns and parseable as dates
        """
        if self.explicit_col:
            if self.explicit_col in df.columns:
                return self.explicit_col
            logger.warning(
                "[TemporalSplitter] Configured temporal_col '%s' not found in DataFrame.",
                self.explicit_col,
            )

        # Check for datetime dtype columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        # Check for pattern-matching object columns parseable as dates
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in self.detect_patterns):
                sample = df[col].dropna().head(20)
                try:
                    pd.to_datetime(sample, errors="raise")
                    return col
                except Exception:
                    continue

        return None

    def get_splits(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], TemporalSplitInfo]:
        """
        Compute cross-validation splits respecting temporal order.

        Returns
        -------
        splits : List of (train_indices, test_indices) tuples
        info   : TemporalSplitInfo with strategy chosen and metadata
        """
        n = len(df)
        temporal_col = self.detect_temporal_column(df)
        warnings: List[str] = []

        if temporal_col is None:
            # No temporal column found — fall back to standard sklearn splits
            info = TemporalSplitInfo(
                strategy="standard_kfold",
                temporal_col=None,
                n_splits=self.n_splits,
                min_train_size=1,
                max_test_size=n,
                reason="No temporal column detected — using standard k-fold.",
                warnings=["No temporal column found. Consider adding a timestamp column."],
            )
            splits = self._standard_splits(n, target_col, df)
            return splits, info

        # Sort by the temporal column
        try:
            sort_vals = pd.to_datetime(df[temporal_col], errors="coerce")
        except Exception:
            sort_vals = df[temporal_col]

        sort_order = sort_vals.argsort().values

        strategy = self.strategy
        if strategy == "auto":
            strategy = "sliding_window" if self.sliding_size else "walk_forward"

        if strategy == "walk_forward":
            splits = list(self._walk_forward_splits(sort_order, n))
        else:  # sliding_window
            splits = list(self._sliding_window_splits(sort_order, n))

        if not splits:
            warnings.append("Temporal splits produced 0 folds — falling back to standard k-fold.")
            splits = self._standard_splits(n, target_col, df)
            strategy = "standard_kfold"

        min_train = min(len(tr) for tr, _ in splits) if splits else 0
        max_test  = max(len(te) for _, te in splits) if splits else 0

        info = TemporalSplitInfo(
            strategy=strategy,
            temporal_col=temporal_col,
            n_splits=len(splits),
            min_train_size=min_train,
            max_test_size=max_test,
            reason=(
                f"Temporal column '{temporal_col}' detected. "
                f"Using {strategy} with {len(splits)} folds to prevent future leakage."
            ),
            warnings=warnings,
        )
        logger.info(
            "[TemporalSplitter] Strategy=%s col='%s' folds=%d "
            "min_train=%d max_test=%d",
            strategy, temporal_col, len(splits), min_train, max_test,
        )
        return splits, info

    def get_sklearn_cv(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """
        Return a sklearn-compatible CV object (or list of splits) for use
        in cross_val_score. If temporal, returns a list of (train, test) index
        tuples that sklearn's cross_val_score accepts.
        """
        splits, info = self.get_splits(df, target_col)
        if info.strategy == "standard_kfold":
            # Return proper sklearn object
            from sklearn.model_selection import StratifiedKFold, KFold
            if target_col and target_col in df.columns:
                try:
                    _target_nuniq = df[target_col].nunique()
                except Exception:  # noqa: BLE001 — unhashable
                    _target_nuniq = 20  # force KFold
                if _target_nuniq < 20:
                    return StratifiedKFold(n_splits=self.n_splits, shuffle=False)
            return KFold(n_splits=self.n_splits, shuffle=False)
        return splits  # sklearn accepts list of (train_idx, test_idx) directly

    # ── Private helpers ───────────────────────────────────────────────────────

    def _walk_forward_splits(
        self, order: np.ndarray, n: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Expanding window: train grows each fold, test is the next block."""
        min_train = max(int(n * self.min_train_frac), 2)
        test_size = max(int(n * self.test_frac), 1)
        start = min_train
        while start + test_size <= n:
            train_idx = order[:start]
            test_idx  = order[start:start + test_size]
            yield train_idx, test_idx
            start += test_size

    def _sliding_window_splits(
        self, order: np.ndarray, n: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Fixed-size training window slides forward each fold."""
        win = self.sliding_size or max(int(n * self.min_train_frac), 2)
        test_size = max(int(n * self.test_frac), 1)
        start = 0
        while start + win + test_size <= n:
            train_idx = order[start:start + win]
            test_idx  = order[start + win:start + win + test_size]
            yield train_idx, test_idx
            start += test_size

    def _standard_splits(
        self,
        n: int,
        target_col: Optional[str],
        df: pd.DataFrame,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Standard sklearn k-fold as fallback."""
        try:
            from sklearn.model_selection import StratifiedKFold, KFold
            if target_col and target_col in df.columns:
                try:
                    _target_nuniq = df[target_col].nunique()
                except Exception:  # noqa: BLE001 — unhashable
                    _target_nuniq = 20  # force KFold
                if _target_nuniq < 20:
                    cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                    y = df[target_col]
                    return list(cv.split(np.zeros(n), y))
            else:
                cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                return list(cv.split(np.zeros(n)))
        except Exception:
            # Basic fallback
            fold_size = n // self.n_splits
            idx = np.arange(n)
            splits = []
            for i in range(self.n_splits):
                test = idx[i * fold_size:(i + 1) * fold_size]
                train = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
                splits.append((train, test))
            return splits
