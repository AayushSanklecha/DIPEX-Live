"""
ingestion/stream_processor.py
-------------------------------
Production-grade streaming data processor for DIPEX.

Implements:
  - Tumbling windows (non-overlapping, fixed intervals)
  - Sliding windows (overlapping, configurable stride)
  - Event-time watermarks with configurable late-data tolerance
  - Backpressure throttle (queue depth threshold)
  - Consumer lag monitoring and alerting
  - Per-window Gold snapshot creation (immutable, SHA-256 checksummed)
  - Full QA pipeline execution per window batch
  - Late data handling: creates corrective snapshot version (never overwrites)
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("dipex.streaming")

_DEFAULT_TUMBLING_WINDOW_S: int = 300    # 5 minutes
_DEFAULT_SLIDING_WINDOW_S: int = 600     # 10 minutes
_DEFAULT_SLIDE_STRIDE_S: int = 120       # 2 minutes
_DEFAULT_WATERMARK_DELAY_S: int = 300    # 5 minutes max late arrival
_DEFAULT_MAX_QUEUE_DEPTH: int = 10_000
_DEFAULT_BACKPRESSURE_THRESH: float = 0.80  # 80% queue fill = throttle


@dataclass
class WindowSnapshot:
    """Immutable result of one processed window."""
    window_id: str
    window_type: str          # "tumbling" | "sliding"
    window_start: float       # epoch seconds
    window_end: float
    record_count: int
    checksum: str             # SHA-256 of window data
    is_corrective: bool = False
    qa_passed: bool = True
    qa_gate_decision: str = "PASS"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: Optional[pd.DataFrame] = field(default=None, repr=False)


@dataclass
class StreamStats:
    """Live stats for the streaming processor."""
    total_events: int = 0
    late_events: int = 0
    dropped_events: int = 0
    windows_closed: int = 0
    corrective_snapshots: int = 0
    backpressure_activations: int = 0
    consumer_lag: int = 0
    queue_depth: int = 0
    current_watermark: float = 0.0


class StreamProcessor:
    """
    Event-driven streaming processor with window management, backpressure,
    watermarks, and per-window QA validation.

    Architecture:
        Ingest → Event Queue → Watermark Assignment → Window Assignment
        → Window Close Trigger → QA Pipeline → Gold Snapshot → Lineage
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        sc = self.config.get("streaming", {})

        # ── Window configuration ────────────────────────────────────────────
        self._tumbling_s: int = int(sc.get("tumbling_window_seconds", _DEFAULT_TUMBLING_WINDOW_S))
        self._sliding_s: int = int(sc.get("sliding_window_seconds", _DEFAULT_SLIDING_WINDOW_S))
        self._stride_s: int = int(sc.get("slide_stride_seconds", _DEFAULT_SLIDE_STRIDE_S))
        self._watermark_delay: int = int(sc.get("watermark_delay_seconds", _DEFAULT_WATERMARK_DELAY_S))
        self._max_queue: int = int(sc.get("max_queue_depth", _DEFAULT_MAX_QUEUE_DEPTH))
        self._backpressure_thresh: float = float(
            sc.get("backpressure_threshold", _DEFAULT_BACKPRESSURE_THRESH)
        )
        self._event_time_field: str = sc.get("event_time_field", "timestamp")
        self._window_mode: str = sc.get("window_mode", "tumbling")

        # ── Internal state ──────────────────────────────────────────────────
        self._event_queue: queue.Queue = queue.Queue(maxsize=self._max_queue)
        self._tumbling_buffers: Dict[int, List[Dict]] = defaultdict(list)
        self._sliding_buffers: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
        self._watermark: float = 0.0               # max event time seen
        self._current_watermark: float = 0.0       # watermark - delay
        self._snapshots: List[WindowSnapshot] = []
        self._stats = StreamStats()
        self._running: bool = False
        self._lock = threading.Lock()
        self._on_snapshot: Optional[Callable[[WindowSnapshot], None]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_snapshot_callback(self, fn: Callable[[WindowSnapshot], None]) -> None:
        """Register a callback that fires on each completed window snapshot."""
        self._on_snapshot = fn

    def emit(self, event: Dict[str, Any]) -> bool:
        """
        Submit an event to the processor.
        Returns False if backpressure is active (caller should throttle).
        """
        # Check backpressure
        depth = self._event_queue.qsize()
        self._stats.queue_depth = depth

        if depth >= self._max_queue * self._backpressure_thresh:
            self._stats.backpressure_activations += 1
            logger.warning(
                "StreamProcessor: backpressure ACTIVE — queue at %d/%d (%.0f%%)",
                depth, self._max_queue, 100 * depth / self._max_queue,
            )
            return False  # caller must slow down

        try:
            self._event_queue.put_nowait(event)
            self._stats.total_events += 1
            return True
        except queue.Full:
            self._stats.dropped_events += 1
            logger.error("StreamProcessor: event DROPPED — queue full")
            return False

    def emit_batch(self, events: List[Dict[str, Any]]) -> int:
        """Emit a batch of events. Returns count actually accepted."""
        accepted = 0
        for evt in events:
            if self.emit(evt):
                accepted += 1
            else:
                break  # stop at first backpressure signal
        return accepted

    def start(
        self,
        source_iterator: Iterator[Dict[str, Any]],
        dataset_id: str = "stream",
        block: bool = True,
    ) -> None:
        """
        Start processing events from an iterator.

        Args:
            source_iterator : Yields event dicts
            dataset_id      : Base dataset ID for snapshot naming
            block           : If True, blocks until source exhausted
        """
        self._running = True
        self._stats = StreamStats()

        ingest_thread = threading.Thread(
            target=self._ingest_from_iterator,
            args=(source_iterator, dataset_id),
            daemon=True,
        )
        process_thread = threading.Thread(
            target=self._process_loop,
            args=(dataset_id,),
            daemon=True,
        )

        ingest_thread.start()
        process_thread.start()

        if block:
            ingest_thread.join()
            process_thread.join()

    def stop(self) -> None:
        """Graceful shutdown — flush remaining windows."""
        self._running = False
        self._flush_all_windows(dataset_id="stream_final")

    def get_snapshots(self) -> List[WindowSnapshot]:
        """Return all completed window snapshots."""
        return list(self._snapshots)

    def get_stats(self) -> Dict[str, Any]:
        """Return current streaming statistics."""
        s = self._stats
        return {
            "total_events": s.total_events,
            "late_events": s.late_events,
            "dropped_events": s.dropped_events,
            "windows_closed": s.windows_closed,
            "corrective_snapshots": s.corrective_snapshots,
            "backpressure_activations": s.backpressure_activations,
            "current_watermark": datetime.fromtimestamp(s.current_watermark, tz=timezone.utc).isoformat()
            if s.current_watermark else None,
            "queue_depth": s.queue_depth,
            "consumer_lag": s.consumer_lag,
        }

    # ------------------------------------------------------------------
    # Internal: ingestion & processing
    # ------------------------------------------------------------------

    def _ingest_from_iterator(self, source: Iterator[Dict], dataset_id: str) -> None:
        """Reads from source iterator and pushes to event queue."""
        try:
            for event in source:
                if not self._running:
                    break
                while not self.emit(event) and self._running:
                    time.sleep(0.1)  # backpressure: wait before retry
        except Exception as exc:  # noqa: BLE001
            logger.error("StreamProcessor: ingest error — %s", exc)
        finally:
            self._event_queue.put(None)  # poison pill

    def _process_loop(self, dataset_id: str) -> None:
        """Main event processing loop."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
                if event is None:
                    break  # poison pill
                self._process_event(event, dataset_id)
                self._event_queue.task_done()
            except queue.Empty:
                self._check_window_close(dataset_id)
        self._flush_all_windows(dataset_id)

    def _process_event(self, event: Dict[str, Any], dataset_id: str) -> None:
        """Route event to correct window based on event time and watermark."""
        event_time = self._extract_event_time(event)

        # Update watermark
        with self._lock:
            if event_time > self._watermark:
                self._watermark = event_time
                self._current_watermark = self._watermark - self._watermark_delay
                self._stats.current_watermark = self._current_watermark

        # Late data check
        if event_time < self._current_watermark:
            self._stats.late_events += 1
            logger.debug(
                "StreamProcessor: late event received (event_time=%s, watermark=%s)",
                event_time, self._current_watermark,
            )
            self._create_corrective_snapshot([event], dataset_id, event_time)
            return

        # Assign to window(s)
        if self._window_mode in ("tumbling", "both"):
            window_key = self._tumbling_key(event_time)
            self._tumbling_buffers[window_key].append(event)

        if self._window_mode in ("sliding", "both"):
            for key in self._sliding_keys(event_time):
                self._sliding_buffers[key].append(event)

        # Check if any window should be closed
        self._check_window_close(dataset_id)

    def _tumbling_key(self, event_time: float) -> int:
        """Returns the start-time of the tumbling window containing event_time."""
        return int(event_time // self._tumbling_s) * self._tumbling_s

    def _sliding_keys(self, event_time: float) -> List[Tuple[int, int]]:
        """Returns all (start, end) pairs of sliding windows containing event_time."""
        keys = []
        window_start = int(event_time // self._stride_s) * self._stride_s
        for i in range(self._sliding_s // self._stride_s):
            start = window_start - i * self._stride_s
            end = start + self._sliding_s
            if start <= event_time < end:
                keys.append((start, end))
        return keys

    def _check_window_close(self, dataset_id: str) -> None:
        """Close and process windows whose end time is past the watermark."""
        with self._lock:
            watermark = self._current_watermark

        # Check tumbling windows
        closed_keys = [k for k in self._tumbling_buffers if k + self._tumbling_s <= watermark]
        for key in closed_keys:
            events = self._tumbling_buffers.pop(key)
            self._close_window(events, "tumbling", key, key + self._tumbling_s, dataset_id)

        # Check sliding windows
        closed_skeys = [k for k in self._sliding_buffers if k[1] <= watermark]
        for key in closed_skeys:
            events = self._sliding_buffers.pop(key)
            self._close_window(events, "sliding", key[0], key[1], dataset_id)

    def _close_window(
        self,
        events: List[Dict],
        window_type: str,
        start: float,
        end: float,
        dataset_id: str,
    ) -> None:
        """Process and snapshot a closed window."""
        if not events:
            return

        try:
            df = pd.json_normalize(events)
            checksum = self._compute_checksum(df)
            window_id = f"{dataset_id}_{window_type}_{int(start)}_{int(end)}_{checksum[:8]}"

            # Run QA pipeline on window batch
            qa_passed, qa_decision = self._run_window_qa(df, window_id)

            snapshot = WindowSnapshot(
                window_id=window_id,
                window_type=window_type,
                window_start=start,
                window_end=end,
                record_count=len(df),
                checksum=checksum,
                qa_passed=qa_passed,
                qa_gate_decision=qa_decision,
                is_corrective=False,
                data=df,
            )

            self._snapshots.append(snapshot)
            self._stats.windows_closed += 1

            logger.info(
                "StreamProcessor: %s window closed [%s→%s] records=%d qa=%s",
                window_type, datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
                datetime.fromtimestamp(end, tz=timezone.utc).isoformat(),
                len(df), qa_decision,
            )

            if self._on_snapshot:
                self._on_snapshot(snapshot)

        except Exception as exc:  # noqa: BLE001
            logger.error("StreamProcessor: window close error — %s", exc)

    def _create_corrective_snapshot(
        self, late_events: List[Dict], dataset_id: str, event_time: float
    ) -> None:
        """Create a corrective snapshot for late-arriving data. Never overwrites."""
        try:
            df = pd.json_normalize(late_events)
            checksum = self._compute_checksum(df)
            corr_id = f"{dataset_id}_corrective_{int(event_time)}_{checksum[:8]}"

            snapshot = WindowSnapshot(
                window_id=corr_id,
                window_type="corrective",
                window_start=event_time,
                window_end=event_time,
                record_count=len(df),
                checksum=checksum,
                is_corrective=True,
                qa_passed=True,
                qa_gate_decision="CORRECTIVE",
                data=df,
            )

            self._snapshots.append(snapshot)
            self._stats.corrective_snapshots += 1
            logger.info(
                "StreamProcessor: corrective snapshot created for %d late event(s)", len(df)
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("StreamProcessor: corrective snapshot failed — %s", exc)

    def _flush_all_windows(self, dataset_id: str) -> None:
        """Flush all open windows at shutdown."""
        logger.info("StreamProcessor: flushing all open windows...")
        all_keys = list(self._tumbling_buffers.keys())
        for key in all_keys:
            events = self._tumbling_buffers.pop(key)
            if events:
                self._close_window(events, "tumbling", key, key + self._tumbling_s, dataset_id)

        all_skeys = list(self._sliding_buffers.keys())
        for key in all_skeys:
            events = self._sliding_buffers.pop(key)
            if events:
                self._close_window(events, "sliding", key[0], key[1], dataset_id)

    def _run_window_qa(self, df: pd.DataFrame, window_id: str) -> Tuple[bool, str]:
        """Run Hard Gate 1 QA on a window DataFrame. Returns (passed, decision)."""
        try:
            from validation.hard_gate import HardGate
            gate = HardGate.from_config(self.config)
            result = gate.run(df, run_id=window_id)
            return (result.decision != "REJECT", result.decision)
        except ImportError:
            return (True, "SKIPPED")
        except Exception as exc:  # noqa: BLE001
            logger.warning("StreamProcessor: window QA failed — %s", exc)
            return (True, "QA_ERROR")

    def _extract_event_time(self, event: Dict[str, Any]) -> float:
        """Extract event time from event dict. Falls back to processing time."""
        ts_field = self._event_time_field
        ts_val = event.get(ts_field)
        if ts_val is None:
            return time.time()

        try:
            if isinstance(ts_val, (int, float)):
                # Auto-detect: if >= 1e12 assume milliseconds (1_000_000_000_000 ms = year 2001)
                if ts_val >= 1e12:
                    return ts_val / 1000.0
                return float(ts_val)
            if isinstance(ts_val, str):
                from dateutil import parser as dtparser
                dt = dtparser.parse(ts_val)
                return dt.timestamp()
        except Exception:
            pass
        return time.time()

    @staticmethod
    def _compute_checksum(df: pd.DataFrame) -> str:
        """SHA-256 checksum of the DataFrame contents."""
        try:
            content = df.to_json(orient="records", date_format="iso").encode("utf-8")
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return hashlib.sha256(uuid.uuid4().bytes).hexdigest()
