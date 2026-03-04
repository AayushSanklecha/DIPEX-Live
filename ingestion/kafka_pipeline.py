"""
ingestion/kafka_pipeline.py
-----------------------------
Full Kafka → DIPEX Pipeline integration.

Architecture:
    Kafka Topics
        │ (confluent-kafka or kafka-python)
        ▼
    KafkaConnector.stream()          ← batch DataFrames
        │
        ▼
    UniversalIntake.ingest_dataframe()  ← ISSF snapshot
        │
        ▼
    PipelineBridge.run()             ← 13-stage DIPEX pipeline
        │                               (preprocess, validate, profile,
        │                                governance, stats, model, report …)
        ▼
    Audit / DLQ                      ← on failure: dead-letter queue

Features:
  - Configurable topics, consumer group, batch size
  - Per-batch full pipeline execution (all 13 stages)
  - Dead-letter queue (DLQ) for failed batches → audit/kafka_dlq.jsonl
  - Consumer lag logging every N batches
  - SIGTERM + KeyboardInterrupt graceful shutdown with offset commit
  - Prometheus-compatible stats dict: events_processed, dlq_count, errors, lag
  - process_one_batch() API for unit-testing without a real Kafka broker
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd

logger = logging.getLogger("dipex.kafka_pipeline")


class KafkaPipelineRunner:
    """
    Subscribes to Kafka topics and runs every incoming batch through
    the full 13-stage DIPEX pipeline.

    Config structure::

        {
          "kafka": {
            "bootstrap_servers": "localhost:9092",
            "group_id":          "dipex-pipeline",
            "topics":            ["raw_events", "sensor_data"],
            "batch_size":        1000,
            "poll_timeout_s":    1.0,
            "auto_offset_reset": "latest",
            "value_deserializer": "json",
            "security_protocol": "PLAINTEXT",
            # SASL (optional)
            "sasl_mechanism":   "PLAIN",
            "sasl_username":    "",
            "sasl_password":    "",
          },
          "pipeline": {
            "target_col": null,       # supervised ML target (optional)
            "skip_stages": [],        # stages to skip
            "domain": "default",      # banking / healthcare / default
            "confidence": {"threshold": 0.70},
          },
          "lag_log_every_n_batches": 50,
          "max_consecutive_errors": 10,
        }

    Usage::

        runner = KafkaPipelineRunner(config)
        runner.start()        # blocks; Ctrl-C for graceful shutdown
        # or
        stats = runner.get_stats()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config   = config or {}
        self._running  = False
        self._lock     = threading.Lock()
        self._run_id   = str(uuid.uuid4())

        kafka_cfg = self.config.get("kafka", {})
        self._topics:     List[str] = kafka_cfg.get("topics", [])
        self._batch_size: int       = int(kafka_cfg.get("batch_size", 1_000))
        self._lag_every:  int       = int(self.config.get("lag_log_every_n_batches", 50))
        self._max_errors: int       = int(self.config.get("max_consecutive_errors", 10))

        self._stats: Dict[str, int] = {
            "batches_processed": 0,
            "events_processed":  0,
            "dlq_count":         0,
            "pipeline_errors":   0,
            "consecutive_errors": 0,
            "consumer_lag":      0,
        }
        self._on_batch_complete: Optional[Callable[[pd.DataFrame, Dict], None]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_batch_callback(self, fn: Callable[[pd.DataFrame, Dict], None]) -> None:
        """
        Register a callback invoked after each batch completes pipeline.
        fn(batch_df, pipeline_result_summary)
        """
        self._on_batch_complete = fn

    def start(self, block: bool = True) -> None:
        """
        Start consuming Kafka and routing batches through DIPEX pipeline.

        If block=True (default), this call blocks until shutdown.
        If block=False, starts background threads and returns.
        """
        self._running = True
        _setup_signal_handlers(self)

        logger.info(
            "[KafkaPipeline] Starting — run_id=%s topics=%s batch_size=%d",
            self._run_id[:8], self._topics, self._batch_size,
        )

        consumer_thread = threading.Thread(
            target=self._consume_loop,
            name="dipex-kafka-consumer",
            daemon=True,
        )
        consumer_thread.start()

        if block:
            try:
                consumer_thread.join()
            except KeyboardInterrupt:
                logger.info("[KafkaPipeline] KeyboardInterrupt — shutting down")
                self.stop()
                consumer_thread.join(timeout=10)

    def stop(self) -> None:
        """Signal graceful shutdown. Consumer will commit offsets and exit."""
        logger.info("[KafkaPipeline] Stop requested")
        with self._lock:
            self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """Return current pipeline statistics (Prometheus-friendly)."""
        with self._lock:
            return dict(self._stats)

    def process_one_batch(
        self,
        df: pd.DataFrame,
        dataset_id: str = "kafka_batch",
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single DataFrame through the full DIPEX pipeline.
        Designed for unit testing without a live Kafka broker.

        Returns the PipelineResult.summary() dict.
        """
        return self._run_pipeline(df, dataset_id=dataset_id, target_col=target_col)

    # ------------------------------------------------------------------
    # Internal: Consumer loop
    # ------------------------------------------------------------------

    def _consume_loop(self) -> None:
        """Main consumer loop: fetch batches and run pipeline."""
        try:
            from ingestion.connectors.kafka_connector import KafkaConnector
        except ImportError as exc:
            logger.error("[KafkaPipeline] Cannot import KafkaConnector: %s", exc)
            return

        kafka_cfg = dict(self.config.get("kafka", {}))
        kafka_cfg["batch_size"] = self._batch_size

        connector = KafkaConnector(kafka_cfg)
        batch_idx = 0

        try:
            for df_batch in connector.stream(chunk_size=self._batch_size):
                if not self._running:
                    logger.info("[KafkaPipeline] Shutdown signal received")
                    break

                if df_batch is None or df_batch.empty:
                    continue

                batch_idx += 1
                dataset_id = f"kafka_batch_{self._run_id[:8]}_{batch_idx}"

                logger.info(
                    "[KafkaPipeline] Batch %d: %d events from topics %s",
                    batch_idx, len(df_batch), self._topics,
                )

                self._process_batch(df_batch, dataset_id, batch_idx)

                # Consumer lag logging
                if batch_idx % self._lag_every == 0:
                    self._log_consumer_lag(connector)

        except Exception as exc:  # noqa: BLE001
            logger.error("[KafkaPipeline] Consumer loop crashed: %s", exc)
        finally:
            connector.close()
            logger.info(
                "[KafkaPipeline] Consumer stopped. Stats: %s", self.get_stats()
            )

    # ------------------------------------------------------------------
    # Internal: Batch processing
    # ------------------------------------------------------------------

    def _process_batch(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        batch_idx: int,
    ) -> None:
        """Route batch through pipeline, DLQ on failure."""
        try:
            pipeline_cfg = self.config.get("pipeline", {})
            target_col = pipeline_cfg.get("target_col")

            result_summary = self._run_pipeline(
                df,
                dataset_id=dataset_id,
                target_col=target_col,
                skip_stages=pipeline_cfg.get("skip_stages", []),
            )

            with self._lock:
                self._stats["batches_processed"] += 1
                self._stats["events_processed"]  += len(df)
                self._stats["consecutive_errors"]  = 0

            logger.info(
                "[KafkaPipeline] Batch %d complete — gate=%s conf=%.3f",
                batch_idx,
                result_summary.get("gate_decision", "?"),
                result_summary.get("confidence_vector", {}).get("confidence_score", 0.0)
                if isinstance(result_summary.get("confidence_vector"), dict)
                else 0.0,
            )

            if self._on_batch_complete:
                try:
                    self._on_batch_complete(df, result_summary)
                except Exception as cb_exc:  # noqa: BLE001
                    logger.warning("[KafkaPipeline] Callback error: %s", cb_exc)

        except Exception as exc:  # noqa: BLE001
            logger.error("[KafkaPipeline] Batch %d pipeline error: %s", batch_idx, exc)
            self._write_dlq(df, dataset_id, str(exc))
            with self._lock:
                self._stats["pipeline_errors"]    += 1
                self._stats["dlq_count"]          += 1
                self._stats["consecutive_errors"] += 1
                if self._stats["consecutive_errors"] >= self._max_errors:
                    logger.critical(
                        "[KafkaPipeline] %d consecutive errors — halting pipeline",
                        self._max_errors,
                    )
                    self._running = False

    # ------------------------------------------------------------------
    # Internal: DIPEX pipeline execution
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        target_col: Optional[str] = None,
        skip_stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Route a DataFrame through UniversalIntake → PipelineBridge.
        Returns PipelineResult.summary() dict.
        """
        try:
            from ingestion.universal_intake import UniversalIntake
            from ingestion.pipeline_bridge import PipelineBridge

            intake = UniversalIntake(config=self.config)
            snapshot = intake.ingest_dataframe(
                df,
                dataset_id=dataset_id,
                source_type="kafka",
            )

            bridge = PipelineBridge(config=self.config)
            result = bridge.run(
                snapshot=snapshot,
                target_col=target_col,
                skip_stages=skip_stages or [],
            )
            return result.summary()

        except ImportError as exc:
            logger.error("[KafkaPipeline] Import error in pipeline: %s", exc)
            # Fallback: minimal pipeline result
            return {
                "run_id": str(uuid.uuid4()),
                "dataset_id": dataset_id,
                "gate_decision": "SKIP",
                "error": f"Import error: {exc}",
                "stages": [],
            }
        except Exception as exc:
            raise

    # ------------------------------------------------------------------
    # Internal: Utilities
    # ------------------------------------------------------------------

    def _write_dlq(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        error: str,
    ) -> None:
        """Write a failed batch to the dead-letter queue JSONL file."""
        try:
            os.makedirs("audit", exist_ok=True)
            entry = {
                "event":        "KAFKA_DLQ",
                "dataset_id":   dataset_id,
                "run_id":       self._run_id,
                "row_count":    len(df),
                "columns":      list(df.columns),
                "error":        error,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                "sample_json":  df.head(5).to_json(orient="records"),
            }
            with open("audit/kafka_dlq.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.warning(
                "[KafkaPipeline] DLQ: %d rows written for dataset_id=%s",
                len(df), dataset_id,
            )
        except Exception as dlq_exc:  # noqa: BLE001
            logger.error("[KafkaPipeline] DLQ write failed: %s", dlq_exc)

    def _log_consumer_lag(self, connector: Any) -> None:
        """Log consumer lag from the Kafka connector."""
        try:
            lag = connector.get_consumer_lag()
            total_lag = sum(lag.values()) if lag else 0
            with self._lock:
                self._stats["consumer_lag"] = total_lag
            logger.info("[KafkaPipeline] Consumer lag: %d messages|%s", total_lag, lag)
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------
# Signal handler helper
# ------------------------------------------------------------------

def _setup_signal_handlers(runner: KafkaPipelineRunner) -> None:
    """Register SIGTERM handler for graceful shutdown in production containers."""
    def _on_sigterm(signum: int, frame: Any) -> None:
        logger.info("[KafkaPipeline] SIGTERM received — initiating graceful shutdown")
        runner.stop()

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (OSError, ValueError):
        # Not available on all platforms (e.g. non-main thread)
        pass


# ------------------------------------------------------------------
# Convenience: run from config file
# ------------------------------------------------------------------

def run_from_config(config: Dict[str, Any], block: bool = True) -> KafkaPipelineRunner:
    """
    Create and start a KafkaPipelineRunner from a config dict.

    Example::

        from ingestion.kafka_pipeline import run_from_config
        runner = run_from_config({
            "kafka": {
                "bootstrap_servers": "localhost:9092",
                "topics": ["events"],
                "group_id": "dipex",
            },
            "pipeline": {"target_col": "label"},
        })
    """
    runner = KafkaPipelineRunner(config)
    runner.start(block=block)
    return runner
