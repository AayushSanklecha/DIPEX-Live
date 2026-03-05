"""
ingestion/streaming_window.py
------------------------------
DATA PROCESSING LAYER — Streaming Window Engine

Provides explicit Tumbling, Sliding, and Session window implementations
that wrap the existing stream_processor.py logic and integrate with
PipelineBridge for Kafka / stream sources.

Window types
------------
- TumblingWindow  : Fixed-size, non-overlapping time buckets
- SlidingWindow   : Overlapping windows (size > step)
- SessionWindow   : Activity-based windows separated by gaps

All windows produce a list of pd.DataFrame micro-batches that can be
independently piped through the DIPEX 13-stage pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterator, List, Optional

import pandas as pd

logger = logging.getLogger("dipex.ingestion.streaming_window")


# ── Window Result ─────────────────────────────────────────────────────────────

@dataclass
class WindowBatch:
    """A single window batch produced by a streaming window engine."""
    window_id: str
    window_type: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    row_count: int
    data: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "window_type": self.window_type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "row_count": self.row_count,
        }


# ── Base Window ───────────────────────────────────────────────────────────────

class BaseWindow:
    """Abstract base for all window strategies."""

    window_type: str = "base"

    def __init__(self, timestamp_col: Optional[str] = None):
        self.timestamp_col = timestamp_col

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce timestamp column to datetime if present."""
        if self.timestamp_col and self.timestamp_col in df.columns:
            df = df.copy()
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors="coerce")
        return df

    def slice(self, df: pd.DataFrame) -> List[WindowBatch]:  # noqa: A003
        raise NotImplementedError

    def stream(self, df: pd.DataFrame) -> Iterator[WindowBatch]:
        """Yield batches one at a time (memory-efficient for large frames)."""
        yield from self.slice(df)


# ── Tumbling Window ───────────────────────────────────────────────────────────

class TumblingWindow(BaseWindow):
    """
    Fixed-size, non-overlapping time buckets.

    If no timestamp column is provided, falls back to row-count bucketing.
    """
    window_type = "tumbling"

    def __init__(
        self,
        window_seconds: float = 60.0,
        max_rows: int = 10_000,
        timestamp_col: Optional[str] = None,
    ):
        super().__init__(timestamp_col)
        self.window_seconds = window_seconds
        self.max_rows = max_rows

    def slice(self, df: pd.DataFrame) -> List[WindowBatch]:
        df = self._parse_timestamps(df)
        batches: List[WindowBatch] = []

        if self.timestamp_col and self.timestamp_col in df.columns:
            ts = df[self.timestamp_col]
            t_min = ts.min()
            t_max = ts.max()
            if pd.isna(t_min) or pd.isna(t_max):
                # No valid timestamps — fall back to row bucketing
                return self._row_bucketing(df)
            step = timedelta(seconds=self.window_seconds)
            current = t_min
            win_idx = 0
            while current <= t_max:
                end = current + step
                mask = (ts >= current) & (ts < end)
                chunk = df[mask].copy()
                if not chunk.empty:
                    batches.append(WindowBatch(
                        window_id=f"tumbling_{win_idx}",
                        window_type=self.window_type,
                        start_time=current.to_pydatetime() if hasattr(current, "to_pydatetime") else current,
                        end_time=end.to_pydatetime() if hasattr(end, "to_pydatetime") else end,
                        row_count=len(chunk),
                        data=chunk,
                    ))
                current = end
                win_idx += 1
        else:
            batches = self._row_bucketing(df)

        logger.info("TumblingWindow: produced %d batches from %d rows", len(batches), len(df))
        return batches

    def _row_bucketing(self, df: pd.DataFrame) -> List[WindowBatch]:
        batches = []
        for i, start in enumerate(range(0, len(df), self.max_rows)):
            chunk = df.iloc[start : start + self.max_rows].copy()
            batches.append(WindowBatch(
                window_id=f"tumbling_rows_{i}",
                window_type=self.window_type,
                start_time=None,
                end_time=None,
                row_count=len(chunk),
                data=chunk,
            ))
        return batches


# ── Sliding Window ────────────────────────────────────────────────────────────

class SlidingWindow(BaseWindow):
    """
    Overlapping windows: each window covers `window_seconds` of data,
    sliding forward by `step_seconds` at a time.
    """
    window_type = "sliding"

    def __init__(
        self,
        window_seconds: float = 120.0,
        step_seconds: float = 60.0,
        timestamp_col: Optional[str] = None,
    ):
        super().__init__(timestamp_col)
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds

    def slice(self, df: pd.DataFrame) -> List[WindowBatch]:
        df = self._parse_timestamps(df)
        if not self.timestamp_col or self.timestamp_col not in df.columns:
            logger.warning("SlidingWindow requires a timestamp_col — falling back to TumblingWindow")
            return TumblingWindow(window_seconds=self.window_seconds).slice(df)

        ts = df[self.timestamp_col]
        t_min = ts.min()
        t_max = ts.max()
        if pd.isna(t_min):
            return []

        window_td = timedelta(seconds=self.window_seconds)
        step_td = timedelta(seconds=self.step_seconds)

        batches = []
        current = t_min
        win_idx = 0
        while current <= t_max:
            end = current + window_td
            mask = (ts >= current) & (ts < end)
            chunk = df[mask].copy()
            if not chunk.empty:
                batches.append(WindowBatch(
                    window_id=f"sliding_{win_idx}",
                    window_type=self.window_type,
                    start_time=current.to_pydatetime() if hasattr(current, "to_pydatetime") else current,
                    end_time=end.to_pydatetime() if hasattr(end, "to_pydatetime") else end,
                    row_count=len(chunk),
                    data=chunk,
                ))
            current += step_td
            win_idx += 1

        logger.info("SlidingWindow: produced %d batches from %d rows", len(batches), len(df))
        return batches


# ── Session Window ────────────────────────────────────────────────────────────

class SessionWindow(BaseWindow):
    """
    Activity-based windows. A new session starts whenever the gap between
    consecutive events exceeds `gap_seconds`.
    """
    window_type = "session"

    def __init__(
        self,
        gap_seconds: float = 300.0,
        timestamp_col: Optional[str] = None,
        session_key_col: Optional[str] = None,
    ):
        super().__init__(timestamp_col)
        self.gap_seconds = gap_seconds
        self.session_key_col = session_key_col

    def slice(self, df: pd.DataFrame) -> List[WindowBatch]:
        df = self._parse_timestamps(df)
        if not self.timestamp_col or self.timestamp_col not in df.columns:
            logger.warning("SessionWindow requires a timestamp_col — no batches produced")
            return []

        df = df.sort_values(self.timestamp_col).copy()
        ts = df[self.timestamp_col]
        gap_td = pd.Timedelta(seconds=self.gap_seconds)

        # Compute session breaks
        time_diff = ts.diff()
        session_breaks = (time_diff > gap_td) | (time_diff.isna())
        df["_session_id"] = session_breaks.cumsum()

        batches = []
        for session_id, group in df.groupby("_session_id"):
            group = group.drop(columns=["_session_id"])
            batches.append(WindowBatch(
                window_id=f"session_{session_id}",
                window_type=self.window_type,
                start_time=group[self.timestamp_col].min().to_pydatetime(),
                end_time=group[self.timestamp_col].max().to_pydatetime(),
                row_count=len(group),
                data=group,
            ))

        logger.info("SessionWindow: produced %d sessions from %d rows", len(batches), len(df))
        return batches


# ── Streaming Window Engine ───────────────────────────────────────────────────

class StreamingWindowEngine:
    """
    Unified facade for the streaming window layer.

    Selects and runs the appropriate window strategy, then returns
    a list of WindowBatch objects ready for the DIPEX pipeline.

    Usage::

        engine = StreamingWindowEngine.from_config(config)
        batches = engine.process(df)
        for batch in batches:
            result = bridge.run(snapshot_from_batch(batch))
    """

    STRATEGIES = {
        "tumbling": TumblingWindow,
        "sliding": SlidingWindow,
        "session": SessionWindow,
    }

    def __init__(self, strategy: BaseWindow):
        self.strategy = strategy

    @classmethod
    def from_config(cls, config: dict) -> "StreamingWindowEngine":
        """Build from pipeline config dict (reads `streaming.window` section)."""
        cfg = config.get("streaming", {}).get("window", {})
        strategy_name = cfg.get("type", "tumbling").lower()
        ts_col = cfg.get("timestamp_col", None)

        if strategy_name == "tumbling":
            strategy = TumblingWindow(
                window_seconds=float(cfg.get("window_seconds", 60)),
                max_rows=int(cfg.get("max_rows", 10_000)),
                timestamp_col=ts_col,
            )
        elif strategy_name == "sliding":
            strategy = SlidingWindow(
                window_seconds=float(cfg.get("window_seconds", 120)),
                step_seconds=float(cfg.get("step_seconds", 60)),
                timestamp_col=ts_col,
            )
        elif strategy_name == "session":
            strategy = SessionWindow(
                gap_seconds=float(cfg.get("gap_seconds", 300)),
                timestamp_col=ts_col,
                session_key_col=cfg.get("session_key_col"),
            )
        else:
            logger.warning("Unknown window strategy '%s' — defaulting to tumbling", strategy_name)
            strategy = TumblingWindow(timestamp_col=ts_col)

        return cls(strategy)

    def process(self, df: pd.DataFrame) -> List[WindowBatch]:
        """Slice df into window batches using the configured strategy."""
        if df is None or df.empty:
            logger.warning("StreamingWindowEngine received empty DataFrame")
            return []
        return self.strategy.slice(df)

    def process_with_callback(
        self,
        df: pd.DataFrame,
        callback: Callable[[WindowBatch], None],
    ) -> int:
        """
        Process df and invoke callback for each batch.
        Returns the number of batches processed.
        """
        count = 0
        for batch in self.strategy.stream(df):
            callback(batch)
            count += 1
        return count
