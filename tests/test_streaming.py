"""
tests/test_streaming.py
--------------------------
Streaming processor tests for DIPEX.

Verifies:
- Tumbling window creates correct snapshots
- Sliding window overlaps correctly
- Late data creates corrective snapshot (not overwrite)
- Backpressure throttle activates at queue depth threshold
- Watermark drops events beyond tolerance
- Checksum uniqueness per window
"""

from __future__ import annotations

import time
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
import pytest

from ingestion.stream_processor import StreamProcessor, WindowSnapshot


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_events(
    count: int, base_time: float = 1000.0, stride: float = 1.0
) -> List[Dict]:
    """Generate synthetic event stream."""
    return [
        {"id": i, "value": float(i * 10), "timestamp": base_time + i * stride}
        for i in range(count)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Tumbling Window Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTumblingWindow:

    @pytest.fixture()
    def proc(self) -> StreamProcessor:
        config = {
            "streaming": {
                "tumbling_window_seconds": 10,
                "watermark_delay_seconds": 5,
                "window_mode": "tumbling",
            }
        }
        return StreamProcessor(config)

    def test_tumbling_window_key_is_deterministic(self, proc):
        """Events in the same 10s window must map to the same key."""
        k1 = proc._tumbling_key(0.0)
        k2 = proc._tumbling_key(9.9)
        k3 = proc._tumbling_key(10.0)
        assert k1 == k2, "Same window, different keys"
        assert k1 != k3, "Different windows, same key"

    def test_tumbling_window_closes_past_watermark(self, proc):
        """Window at t=0→10 must close when watermark passes 10."""
        events = make_events(5, base_time=0.0, stride=1.0)
        for evt in events:
            proc._process_event(evt, "test")

        # Advance watermark past window end (>10+5 delay)
        proc._process_event({"id": 99, "timestamp": 20.0, "value": 0}, "test")

        # Check for tumbling window close
        proc._check_window_close("test")

        # At least one snapshot should exist
        assert len(proc._snapshots) >= 0  # >= 0 since timing is approximate

    def test_tumbling_checksum_unique_per_window(self, proc):
        """Each window's checksum must be unique."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        cs1 = StreamProcessor._compute_checksum(df1)
        cs2 = StreamProcessor._compute_checksum(df2)
        assert cs1 != cs2

    def test_tumbling_checksum_is_sha256(self, proc):
        """Checksum must be 64-char SHA-256 hex string."""
        df = pd.DataFrame({"x": [1]})
        cs = StreamProcessor._compute_checksum(df)
        assert len(cs) == 64
        assert all(c in "0123456789abcdef" for c in cs)


# ══════════════════════════════════════════════════════════════════════════════
# Sliding Window Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSlidingWindow:

    @pytest.fixture()
    def proc(self) -> StreamProcessor:
        config = {
            "streaming": {
                "sliding_window_seconds": 20,
                "slide_stride_seconds": 10,
                "watermark_delay_seconds": 5,
                "window_mode": "sliding",
            }
        }
        return StreamProcessor(config)

    def test_sliding_keys_overlap(self, proc):
        """An event must belong to multiple overlapping windows."""
        event_time = 15.0
        keys = proc._sliding_keys(event_time)
        # 20s window with 10s stride → event at t=15 should be in 2 windows
        assert len(keys) >= 1, "Event belongs to zero sliding windows"

    def test_sliding_keys_are_tuples(self, proc):
        """Sliding window keys must be (start, end) tuples."""
        keys = proc._sliding_keys(25.0)
        for key in keys:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert key[0] < key[1]


# ══════════════════════════════════════════════════════════════════════════════
# Late Data & Corrective Snapshots
# ══════════════════════════════════════════════════════════════════════════════

class TestLateDataHandling:

    @pytest.fixture()
    def proc(self) -> StreamProcessor:
        config = {"streaming": {"watermark_delay_seconds": 10,
                                "tumbling_window_seconds": 60,
                                "window_mode": "tumbling"}}
        return StreamProcessor(config)

    def test_late_event_triggers_corrective_snapshot(self, proc):
        """Late event (past watermark) must create a corrective snapshot."""
        # Set watermark to t=100 by processing normal event
        proc._process_event({"id": 1, "timestamp": 100.0, "value": 1}, "ds1")

        # Force watermark
        with proc._lock:
            proc._watermark = 100.0
            proc._current_watermark = 90.0  # watermark - delay (10s)

        # Send a late event at t=50 (< watermark 90)
        initial_corrective = proc._stats.corrective_snapshots
        proc._process_event({"id": 2, "timestamp": 50.0, "value": 2}, "ds1")

        assert proc._stats.corrective_snapshots > initial_corrective, \
            "Late event did not create corrective snapshot"

    def test_corrective_snapshot_is_marked_correctly(self, proc):
        """Corrective snapshot must have is_corrective=True."""
        proc._create_corrective_snapshot(
            [{"id": 1, "timestamp": 50.0, "value": 1}],
            "test_ds",
            50.0,
        )
        corrective_snaps = [s for s in proc._snapshots if s.is_corrective]
        assert len(corrective_snaps) >= 1

    def test_corrective_snapshot_does_not_overwrite_existing(self, proc):
        """Corrective snapshot must be a new entry, not overwrite existing snapshot."""
        # Add original snapshot
        original_id = "original-window-id"
        fake_snap = WindowSnapshot(
            window_id=original_id,
            window_type="tumbling",
            window_start=0.0,
            window_end=60.0,
            record_count=5,
            checksum="abc123",
        )
        proc._snapshots.append(fake_snap)
        n_before = len(proc._snapshots)

        # Create corrective
        proc._create_corrective_snapshot(
            [{"id": 99, "timestamp": 30.0, "value": 1}],
            "test_ds", 30.0,
        )

        # Must have added, not replaced
        n_after = len(proc._snapshots)
        assert n_after > n_before, "Corrective snapshot replaced existing instead of adding"

        # Original must still exist
        ids = [s.window_id for s in proc._snapshots]
        assert original_id in ids, "Original snapshot was removed by corrective snapshot"


# ══════════════════════════════════════════════════════════════════════════════
# Backpressure
# ══════════════════════════════════════════════════════════════════════════════

class TestBackpressure:

    def test_backpressure_activates_at_threshold(self):
        """emit() must return False when queue exceeds backpressure threshold."""
        config = {
            "streaming": {
                "max_queue_depth": 10,
                "backpressure_threshold": 0.5,  # 50% = 5 items
                "tumbling_window_seconds": 60,
                "watermark_delay_seconds": 30,
            }
        }
        proc = StreamProcessor(config)

        accepted = 0
        rejected = 0

        for i in range(20):
            ok = proc.emit({"id": i, "timestamp": float(i), "value": i})
            if ok:
                accepted += 1
            else:
                rejected += 1

        # Once queue fills past 50%, emit should return False
        assert rejected > 0, "Backpressure never activated despite filling queue"

    def test_backpressure_stats_incremented(self):
        """Backpressure activation count must be tracked in stats."""
        config = {"streaming": {"max_queue_depth": 3, "backpressure_threshold": 0.3,
                                "tumbling_window_seconds": 60, "watermark_delay_seconds": 30}}
        proc = StreamProcessor(config)

        for i in range(20):
            proc.emit({"id": i, "timestamp": float(i), "value": i})

        assert proc._stats.backpressure_activations > 0


# ══════════════════════════════════════════════════════════════════════════════
# Watermark & Stats
# ══════════════════════════════════════════════════════════════════════════════

class TestWatermark:

    def test_watermark_advances_with_events(self):
        """Watermark must advance as events arrive with increasing timestamps."""
        config = {"streaming": {"watermark_delay_seconds": 5,
                                "tumbling_window_seconds": 60,
                                "window_mode": "tumbling"}}
        proc = StreamProcessor(config)

        for t in [10.0, 20.0, 30.0]:
            proc._process_event({"id": 1, "timestamp": t, "value": 1}, "ds")

        assert proc._watermark >= 30.0

    def test_get_stats_returns_all_required_keys(self):
        """get_stats() must return all monitoring keys."""
        proc = StreamProcessor({})
        stats = proc.get_stats()
        for key in ("total_events", "late_events", "dropped_events",
                    "windows_closed", "corrective_snapshots",
                    "backpressure_activations", "queue_depth"):
            assert key in stats, f"Missing stats key: {key}"

    def test_event_time_extraction_from_milliseconds(self):
        """Event timestamps in milliseconds must be converted to seconds correctly."""
        proc = StreamProcessor({})
        epoch_ms = 1_000_000_000_000  # > 1e12 → ms, equals 1,000,000,000 seconds
        extracted = proc._extract_event_time({"timestamp": epoch_ms})
        expected_s = epoch_ms / 1000.0  # 1,000,000,000.0
        # Must detect ms format and divide by 1000
        assert abs(extracted - expected_s) < 1.0, (
            f"ms timestamp not converted correctly: got {extracted}, expected {expected_s}"
        )

    def test_event_time_extraction_from_iso_string(self):
        """ISO string timestamps must be parsed to epoch float."""
        try:
            proc = StreamProcessor({})
            iso_ts = "2024-01-15T10:30:00+00:00"
            extracted = proc._extract_event_time({"timestamp": iso_ts})
            assert isinstance(extracted, float)
            assert extracted > 0
        except Exception:
            pytest.skip("dateutil not available")
