"""
tests/test_kafka_pipeline.py
------------------------------
Unit tests for KafkaPipelineRunner — the Kafka-to-DIPEX pipeline bridge.

All tests use mocks — no live Kafka broker or DIPEX pipeline required.
Tests verify:
  - process_one_batch() routes DataFrame through pipeline
  - DLQ is written on pipeline failure
  - Stats are tracked correctly
  - Consecutive error limit halts the runner
  - Callback is invoked after each successful batch
  - Graceful shutdown on stop() signal
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from ingestion.kafka_pipeline import KafkaPipelineRunner


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id":        [1, 2, 3],
        "value":     [10.0, 20.0, 30.0],
        "category":  ["A", "B", "A"],
        "timestamp": [1700000000.0, 1700000001.0, 1700000002.0],
    })


@pytest.fixture()
def runner() -> KafkaPipelineRunner:
    return KafkaPipelineRunner({
        "kafka": {
            "bootstrap_servers": "localhost:9092",
            "topics": ["test-topic"],
            "group_id": "dipex-test",
            "batch_size": 100,
        },
        "pipeline": {
            "target_col": None,
            "skip_stages": [],
        },
    })


# ══════════════════════════════════════════════════════════════════════════════
# Importability
# ══════════════════════════════════════════════════════════════════════════════

class TestKafkaPipelineImport:

    def test_importable(self):
        from ingestion.kafka_pipeline import KafkaPipelineRunner
        assert KafkaPipelineRunner is not None

    def test_run_from_config_importable(self):
        from ingestion.kafka_pipeline import run_from_config
        assert run_from_config is not None


# ══════════════════════════════════════════════════════════════════════════════
# Stats API
# ══════════════════════════════════════════════════════════════════════════════

class TestKafkaPipelineStats:

    def test_initial_stats_all_zero(self, runner):
        stats = runner.get_stats()
        for key in ("batches_processed", "events_processed", "dlq_count",
                    "pipeline_errors", "consecutive_errors"):
            assert stats[key] == 0, f"Expected {key}=0, got {stats[key]}"

    def test_get_stats_returns_dict(self, runner):
        assert isinstance(runner.get_stats(), dict)


# ══════════════════════════════════════════════════════════════════════════════
# process_one_batch — mocked pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestProcessOneBatch:

    def _make_mock_pipeline(self):
        """Return a mock PipelineResult and mocked module paths."""
        mock_snapshot = MagicMock()
        mock_snapshot.row_count = 3

        mock_result = MagicMock()
        mock_result.summary.return_value = {
            "run_id": "test-run-123",
            "dataset_id": "kafka_batch",
            "gate_decision": "PASS",
            "stages": [],
            "model_metrics": {},
        }

        mock_intake = MagicMock()
        mock_intake.ingest_dataframe.return_value = mock_snapshot

        mock_bridge = MagicMock()
        mock_bridge.run.return_value = mock_result

        return mock_intake, mock_bridge

    def test_process_one_batch_returns_summary(self, runner, sample_df):
        mock_intake, mock_bridge = self._make_mock_pipeline()
        with patch("ingestion.kafka_pipeline.KafkaPipelineRunner._run_pipeline",
                   return_value={"gate_decision": "PASS", "dataset_id": "batch_1"}):
            result = runner.process_one_batch(sample_df, dataset_id="batch_1")
        assert isinstance(result, dict)
        assert result.get("gate_decision") == "PASS"

    def test_process_one_batch_uses_pipeline(self, runner, sample_df):
        run_called_with = []

        def fake_run_pipeline(df, dataset_id, **kwargs):
            run_called_with.append({"df_shape": df.shape, "dataset_id": dataset_id})
            return {"gate_decision": "PASS", "stages": []}

        with patch.object(runner, "_run_pipeline", side_effect=fake_run_pipeline):
            runner.process_one_batch(sample_df, dataset_id="test_batch")

        assert len(run_called_with) == 1
        assert run_called_with[0]["df_shape"] == sample_df.shape


# ══════════════════════════════════════════════════════════════════════════════
# DLQ — Dead Letter Queue
# ══════════════════════════════════════════════════════════════════════════════

class TestDeadLetterQueue:

    def test_dlq_written_on_failure(self, runner, sample_df, tmp_path, monkeypatch):
        """Pipeline failure must write batch to DLQ JSONL file."""
        monkeypatch.chdir(tmp_path)  # audit/ relative to tmp working dir

        runner._write_dlq(sample_df, "failed_batch_001", "MockError: something broke")

        dlq_path = tmp_path / "audit" / "kafka_dlq.jsonl"
        assert dlq_path.exists(), "DLQ file not created"

        with open(dlq_path) as f:
            entry = json.loads(f.readline())

        assert entry["dataset_id"] == "failed_batch_001"
        assert entry["row_count"] == len(sample_df)
        assert "MockError" in entry["error"]

    def test_dlq_stat_incremented_on_pipeline_error(self, runner, sample_df, tmp_path, monkeypatch):
        """DLQ count and pipeline_errors stats must increment on failure."""
        monkeypatch.chdir(tmp_path)

        def boom(df, dataset_id, **kwargs):
            raise RuntimeError("test failure")

        with patch.object(runner, "_run_pipeline", side_effect=boom):
            runner._process_batch(sample_df, "bad_batch", batch_idx=1)

        stats = runner.get_stats()
        assert stats["pipeline_errors"] >= 1
        assert stats["dlq_count"] >= 1

    def test_consecutive_error_limit_stops_runner(self, runner, sample_df, tmp_path, monkeypatch):
        """Exceeding max_consecutive_errors must set _running=False."""
        monkeypatch.chdir(tmp_path)
        runner._max_errors = 3
        runner._running    = True

        def boom(df, dataset_id, **kwargs):
            raise RuntimeError("force error")

        with patch.object(runner, "_run_pipeline", side_effect=boom):
            for i in range(4):
                runner._process_batch(sample_df, f"bad_{i}", batch_idx=i)

        assert runner._running is False, "Runner should have stopped after consecutive errors"


# ══════════════════════════════════════════════════════════════════════════════
# Callback
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchCallback:

    def test_callback_invoked_after_success(self, runner, sample_df, tmp_path, monkeypatch):
        """Batch callback must be called with DataFrame and result summary."""
        monkeypatch.chdir(tmp_path)

        callback_calls = []
        def my_callback(df, result):
            callback_calls.append({"df_len": len(df), "gate": result.get("gate_decision")})

        runner.set_batch_callback(my_callback)

        with patch.object(runner, "_run_pipeline",
                          return_value={"gate_decision": "PASS", "stages": []}):
            runner._process_batch(sample_df, "cb_batch", batch_idx=1)

        assert len(callback_calls) == 1
        assert callback_calls[0]["gate"] == "PASS"

    def test_callback_error_does_not_crash_runner(self, runner, sample_df, tmp_path, monkeypatch):
        """A crashing callback must not stop the pipeline."""
        monkeypatch.chdir(tmp_path)

        def bad_callback(df, result):
            raise ValueError("callback bug")

        runner.set_batch_callback(bad_callback)

        with patch.object(runner, "_run_pipeline",
                          return_value={"gate_decision": "PASS", "stages": []}):
            # Should not raise
            runner._process_batch(sample_df, "cb_batch", batch_idx=1)

        # Stats still updated
        assert runner.get_stats()["batches_processed"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# Shutdown
# ══════════════════════════════════════════════════════════════════════════════

class TestGracefulShutdown:

    def test_stop_sets_running_false(self, runner):
        runner._running = True
        runner.stop()
        assert runner._running is False

    def test_double_stop_is_safe(self, runner):
        runner._running = True
        runner.stop()
        runner.stop()  # Must not raise
        assert runner._running is False
