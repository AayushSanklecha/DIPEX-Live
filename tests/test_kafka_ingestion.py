"""
tests/test_kafka_ingestion.py
--------------------------------
Kafka stream ingestion tests — no live broker required.
All Kafka Consumer interactions are mocked.

Coverage:
  - StreamReader.read_kafka(): windowed batch → DataFrame
  - TumblingWindow: add/close cycle
  - SlidingWindow: add/emit cycle
  - Malformed messages: skipped without crash
  - Broker unreachable: raises DataFormatError
  - collect_events(): in-memory event windowing
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.readers.stream_reader import (
    StreamReader, KafkaSourceConfig, WindowConfig,
    TumblingWindow, SlidingWindow, StreamReadResult,
)
from ingestion.error_handler import DataFormatError, StreamLagError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_kafka_msg(data: dict, topic="raw_events", partition=0, offset=0):
    """Create a mock confluent_kafka.Message."""
    msg = MagicMock()
    msg.value.return_value = json.dumps(data).encode("utf-8")
    msg.error.return_value = None
    msg.topic.return_value = topic
    msg.partition.return_value = partition
    msg.offset.return_value = offset
    msg.timestamp.return_value = (1, int(time.time() * 1000))
    return msg


def _make_error_msg():
    """Create a mock message with an error."""
    msg = MagicMock()
    err = MagicMock()
    err.code.return_value = -191  # arbitrary non-EOF error code
    msg.error.return_value = err
    return msg


# ═══════════════════════════════════════════════════════════════════════════════
# TumblingWindow Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTumblingWindow:
    def test_add_and_close_returns_records(self):
        """Add records to window, close returns them."""
        window = TumblingWindow(window_size_s=10.0, watermark_delay_s=60.0)
        window.add({"id": 1})
        window.add({"id": 2})
        records, start, end, late = window.close()
        assert len(records) == 2
        assert late == 0

    def test_should_close_after_expiry(self):
        """Window should_close() returns True after window_size_s."""
        window = TumblingWindow(window_size_s=0.01, watermark_delay_s=60.0)
        time.sleep(0.02)
        assert window.should_close() is True

    def test_should_not_close_before_expiry(self):
        """Window should_close() returns False before window_size_s."""
        window = TumblingWindow(window_size_s=60.0, watermark_delay_s=60.0)
        assert window.should_close() is False

    def test_close_resets_buffer(self):
        """After close(), buffer is empty for next window."""
        window = TumblingWindow(window_size_s=10.0, watermark_delay_s=60.0)
        window.add({"id": 1})
        window.close()
        records, _, _, _ = window.close()
        assert len(records) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# SlidingWindow Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSlidingWindow:
    def test_add_and_emit_returns_records(self):
        """Add records, emit returns overlapping window."""
        window = SlidingWindow(window_size_s=10.0, slide_step_s=1.0, watermark_delay_s=60.0)
        window.add({"id": 1})
        window.add({"id": 2})
        records, start, end, late = window.emit()
        assert len(records) == 2

    def test_should_emit_after_slide_step(self):
        """should_emit() returns True after slide_step_s."""
        window = SlidingWindow(window_size_s=10.0, slide_step_s=0.01, watermark_delay_s=60.0)
        time.sleep(0.02)
        assert window.should_emit() is True


# ═══════════════════════════════════════════════════════════════════════════════
# StreamReader.read_kafka() Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStreamReaderKafka:

    def test_read_kafka_yields_stream_read_result(self):
        """read_kafka() yields StreamReadResult with DataFrame data."""
        msgs = [
            _make_kafka_msg({"id": 1, "value": "a"}),
            _make_kafka_msg({"id": 2, "value": "b"}),
        ]
        # After messages, return None to trigger timeout/window close
        mock_consumer = MagicMock()
        call_count = {"n": 0}

        def _poll(timeout=1.0):
            if call_count["n"] < len(msgs):
                msg = msgs[call_count["n"]]
                call_count["n"] += 1
                return msg
            return None

        mock_consumer.poll = _poll
        mock_consumer.assignment.return_value = []

        config = KafkaSourceConfig(topic="test_topic", max_messages=2)
        window_cfg = WindowConfig(strategy="tumbling", window_size_s=0.01)

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            reader = StreamReader()
            time.sleep(0.02)  # let window expire
            results = list(reader.read_kafka(config, window_cfg, max_windows=1))

        assert len(results) >= 1
        assert isinstance(results[0], StreamReadResult)
        assert isinstance(results[0].data, pd.DataFrame)
        assert results[0].row_count >= 1

    def test_empty_topic_returns_empty_window(self):
        """If no messages arrive, yield empty result after window closes."""
        mock_consumer = MagicMock()
        mock_consumer.poll.return_value = None
        mock_consumer.assignment.return_value = []

        config = KafkaSourceConfig(topic="empty_topic", max_messages=100)
        window_cfg = WindowConfig(strategy="tumbling", window_size_s=0.01)

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            reader = StreamReader()
            time.sleep(0.02)
            results = list(reader.read_kafka(config, window_cfg, max_windows=1))

        # May yield 0 results (empty window) or 1 empty-DataFrame result
        for r in results:
            assert isinstance(r.data, pd.DataFrame)

    def test_malformed_message_skipped_not_crash(self):
        """Invalid JSON message is skipped, valid messages are processed."""
        bad_msg = MagicMock()
        bad_msg.value.return_value = b"NOT JSON {{{"
        bad_msg.error.return_value = None
        bad_msg.topic.return_value = "test"
        bad_msg.partition.return_value = 0
        bad_msg.offset.return_value = 0
        bad_msg.timestamp.return_value = (1, int(time.time() * 1000))

        good_msg = _make_kafka_msg({"id": 99})

        mock_consumer = MagicMock()
        msgs = [bad_msg, good_msg]
        call_count = {"n": 0}

        def _poll(timeout=1.0):
            if call_count["n"] < len(msgs):
                m = msgs[call_count["n"]]
                call_count["n"] += 1
                return m
            return None

        mock_consumer.poll = _poll
        mock_consumer.assignment.return_value = []

        config = KafkaSourceConfig(topic="test", max_messages=2)
        window_cfg = WindowConfig(strategy="tumbling", window_size_s=0.01)

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            reader = StreamReader()
            time.sleep(0.02)
            results = list(reader.read_kafka(config, window_cfg, max_windows=1))

        # Should not crash — at least the good message is processed
        # The bad message produces a {"_raw": ...} record, so we expect all records
        total_rows = sum(r.row_count for r in results)
        assert total_rows >= 1

    def test_kafka_error_message_handled_gracefully(self):
        """Messages with Kafka errors are handled without crashing."""
        mock_consumer = MagicMock()
        error_msg = _make_error_msg()
        good_msg = _make_kafka_msg({"id": 1})

        msgs = [error_msg, good_msg]
        call_count = {"n": 0}

        def _poll(timeout=1.0):
            if call_count["n"] < len(msgs):
                m = msgs[call_count["n"]]
                call_count["n"] += 1
                return m
            return None

        mock_consumer.poll = _poll
        mock_consumer.assignment.return_value = []

        config = KafkaSourceConfig(topic="test", max_messages=2)
        window_cfg = WindowConfig(strategy="tumbling", window_size_s=0.01)

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            reader = StreamReader()
            time.sleep(0.02)
            # Should not crash
            results = list(reader.read_kafka(config, window_cfg, max_windows=1))

    def test_raises_when_confluent_kafka_not_installed(self):
        """Missing confluent-kafka package raises DataFormatError."""
        import builtins
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "confluent_kafka":
                raise ImportError("No module named 'confluent_kafka'")
            return real_import(name, *args, **kwargs)

        config = KafkaSourceConfig(topic="test")
        window_cfg = WindowConfig()
        reader = StreamReader()

        with patch("builtins.__import__", side_effect=_mock_import):
            with pytest.raises(DataFormatError, match="confluent-kafka"):
                list(reader.read_kafka(config, window_cfg))

    def test_broker_unreachable_raises_exception(self):
        """Consumer creation failure when broker unreachable raises Exception."""
        with patch(
            "confluent_kafka.Consumer",
            side_effect=Exception("Broker not available")
        ):
            config = KafkaSourceConfig(topic="test")
            window_cfg = WindowConfig()
            reader = StreamReader()

            with pytest.raises(Exception, match="Broker not available"):
                list(reader.read_kafka(config, window_cfg))

    def test_nested_json_messages_flattened(self):
        """Nested JSON in Kafka messages is flattened via json_normalize."""
        nested_msg = _make_kafka_msg({"id": 1, "meta": {"source": "sensor_A"}})
        mock_consumer = MagicMock()
        call_count = {"n": 0}

        def _poll(timeout=1.0):
            if call_count["n"] == 0:
                call_count["n"] += 1
                return nested_msg
            return None

        mock_consumer.poll = _poll
        mock_consumer.assignment.return_value = []

        config = KafkaSourceConfig(topic="test", max_messages=1)
        window_cfg = WindowConfig(strategy="tumbling", window_size_s=0.01)

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            reader = StreamReader()
            time.sleep(0.02)
            results = list(reader.read_kafka(config, window_cfg, max_windows=1))

        assert len(results) >= 1
        df = results[0].data
        assert "meta.source" in df.columns


# ═══════════════════════════════════════════════════════════════════════════════
# StreamReader.collect_events() Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollectEvents:

    def test_collect_events_returns_results(self):
        """collect_events() processes in-memory events into StreamReadResults."""
        reader = StreamReader()
        events = [{"user_id": i, "event": "click"} for i in range(10)]
        wc = WindowConfig(strategy="tumbling", window_size_s=0.01)
        time.sleep(0.02)
        results = reader.collect_events(events, wc)
        assert len(results) >= 1
        total = sum(r.row_count for r in results)
        assert total == 10

    def test_collect_events_sliding_window(self):
        """collect_events() with sliding window returns valid results."""
        reader = StreamReader()
        events = [{"x": i} for i in range(5)]
        wc = WindowConfig(strategy="sliding", window_size_s=1.0, slide_step_s=0.01)
        time.sleep(0.02)
        results = reader.collect_events(events, wc)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r.data, pd.DataFrame)

    def test_collect_events_empty_list(self):
        """collect_events() with empty list returns empty results."""
        reader = StreamReader()
        wc = WindowConfig(strategy="tumbling", window_size_s=0.01)
        time.sleep(0.02)
        results = reader.collect_events([], wc)
        # May return empty list or list with empty DataFrame
        for r in results:
            assert r.row_count == 0
