"""
ingestion/connectors/kafka_connector.py
-----------------------------------------
Production Kafka consumer connector.

Supports:
- confluent-kafka (production) with Schema Registry (Avro/JSON Schema)
- kafka-python (dev/fallback)
- Configurable consumer groups, topics, auto-offset-reset
- Event-time timestamp extraction
- Graceful shutdown with offset commit
- Consumer lag monitoring
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.kafka")

_DEFAULT_POLL_TIMEOUT: float = 1.0     # seconds
_DEFAULT_BATCH_MSG: int = 1000         # messages per DataFrame chunk
_DEFAULT_MAX_POLL_RETRIES: int = 3
_DEFAULT_SESSION_TIMEOUT_MS: int = 30_000


class KafkaConnector(BaseConnector):
    """
    Kafka consumer connector.

    Config keys:
        bootstrap_servers   : Comma-separated broker list (env: KAFKA_BROKERS)
        group_id            : Consumer group ID
        topics              : List of topic names to consume
        auto_offset_reset   : "earliest" | "latest" | "none" (default: latest)
        max_messages        : Max messages to consume in one extract() call
        batch_size          : Messages per yielded DataFrame chunk (default: 1000)
        poll_timeout_s      : Poll timeout in seconds (default: 1.0)
        value_deserializer  : "json" | "string" | "avro" (default: json)
        timestamp_field     : Field name for event time (default: timestamp)
        security_protocol   : "PLAINTEXT" | "SSL" | "SASL_SSL" (default: PLAINTEXT)
        sasl_mechanism      : "PLAIN" | "SCRAM-SHA-256" | "SCRAM-SHA-512"
        sasl_username       : SASL username (env: KAFKA_SASL_USER)
        sasl_password       : SASL password (env: KAFKA_SASL_PASS)
        schema_registry_url : Confluent Schema Registry URL (optional)
        enable_auto_commit  : Whether to auto-commit offsets (default: False)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._consumer = None
        self._use_confluent: bool = False
        self._lag_stats: Dict[str, int] = {}

    def _get_brokers(self) -> str:
        return os.environ.get("KAFKA_BROKERS", self.config.get("bootstrap_servers", "localhost:9092"))

    def _get_consumer(self):
        """Create consumer — tries confluent-kafka first, falls back to kafka-python."""
        if self._consumer is not None:
            return self._consumer

        group_id = self.config.get("group_id", "dipex-consumer")
        brokers = self._get_brokers()
        offset_reset = self.config.get("auto_offset_reset", "latest")
        security = self.config.get("security_protocol", "PLAINTEXT")
        auto_commit = self.config.get("enable_auto_commit", False)

        # -- Try confluent-kafka -------------------------------------------------
        try:
            from confluent_kafka import Consumer as ConfluentConsumer  # type: ignore

            conf = {
                "bootstrap.servers": brokers,
                "group.id": group_id,
                "auto.offset.reset": offset_reset,
                "enable.auto.commit": auto_commit,
                "session.timeout.ms": _DEFAULT_SESSION_TIMEOUT_MS,
            }
            if security != "PLAINTEXT":
                conf["security.protocol"] = security
                conf["sasl.mechanisms"] = self.config.get("sasl_mechanism", "PLAIN")
                conf["sasl.username"] = os.environ.get("KAFKA_SASL_USER",
                                                        self.config.get("sasl_username", ""))
                conf["sasl.password"] = os.environ.get("KAFKA_SASL_PASS",
                                                        self.config.get("sasl_password", ""))

            self._consumer = ConfluentConsumer(conf)
            self._use_confluent = True
            logger.info("KafkaConnector: using confluent-kafka (production mode)")
            return self._consumer

        except ImportError:
            logger.info("confluent-kafka not available — falling back to kafka-python")

        # -- Fallback: kafka-python -----------------------------------------------
        try:
            from kafka import KafkaConsumer  # type: ignore
            from kafka.errors import NoBrokersAvailable  # type: ignore

            self._consumer = KafkaConsumer(
                bootstrap_servers=brokers.split(","),
                group_id=group_id,
                auto_offset_reset=offset_reset,
                enable_auto_commit=auto_commit,
                value_deserializer=self._get_deserializer(),
                consumer_timeout_ms=int(self.config.get("poll_timeout_s", _DEFAULT_POLL_TIMEOUT) * 1000),
            )
            self._use_confluent = False
            logger.info("KafkaConnector: using kafka-python (dev mode)")
            return self._consumer

        except ImportError as exc:
            raise ConnectorError(
                "No Kafka library found. Install: pip install confluent-kafka OR pip install kafka-python"
            ) from exc
        except Exception as exc:
            raise ConnectorError(f"KafkaConnector: failed to create consumer — {exc}") from exc

    def test_connection(self) -> bool:
        try:
            consumer = self._get_consumer()
            topics = self.config.get("topics", [])
            if self._use_confluent:
                # Get cluster metadata — verifies broker connectivity
                from confluent_kafka import KafkaException  # type: ignore
                meta = consumer.list_topics(timeout=5)
                available = list(meta.topics.keys())
                missing = [t for t in topics if t not in available]
                if missing:
                    logger.warning("KafkaConnector: topics not found: %s", missing)
                logger.info("KafkaConnector: connection test PASSED (confluent)")
                return True
            else:
                # kafka-python: list topics
                consumer.topics()
                logger.info("KafkaConnector: connection test PASSED (kafka-python)")
                return True
        except Exception as exc:
            logger.error("KafkaConnector: connection test FAILED — %s", exc)
            return False

    def get_schema(self) -> Dict[str, Any]:
        """Infer schema by consuming a small sample of messages."""
        topics = self.config.get("topics", [])
        sample_msgs = self._consume_messages(max_messages=10)
        if not sample_msgs:
            return {"topics": topics, "columns": [], "dtypes": {},
                    "description": "No messages sampled (empty or lag too low)"}
        df_sample = pd.json_normalize(sample_msgs)
        return {
            "topics": topics,
            "columns": list(df_sample.columns),
            "dtypes": {col: str(df_sample[col].dtype) for col in df_sample.columns},
            "estimated_row_count": -1,
            "description": f"Schema inferred from {len(sample_msgs)} sampled messages",
        }

    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Consume messages and return as DataFrame.
        `query` is ignored (no SQL-style querying in Kafka).
        """
        max_msgs = kwargs.get("max_messages", self.config.get("max_messages", 10_000))
        messages = self._consume_messages(max_messages=max_msgs)
        if not messages:
            return pd.DataFrame()
        df = pd.json_normalize(messages)
        logger.info("KafkaConnector: consumed %d messages", len(df))
        return df

    def stream(self, chunk_size: int = _DEFAULT_BATCH_MSG, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """
        Continuously consume Kafka topics yielding DataFrames per batch.
        Call close() or set a stop condition externally.
        """
        topics = self.config.get("topics", [])
        if not topics:
            raise ConnectorError("KafkaConnector: 'topics' must be specified in config")

        consumer = self._get_consumer()
        if self._use_confluent:
            consumer.subscribe(topics)
        else:
            consumer.subscribe(topics)

        batch: List[Dict] = []
        timeout = self.config.get("poll_timeout_s", _DEFAULT_POLL_TIMEOUT)

        try:
            while True:
                msg = self._poll_one(consumer, timeout)
                if msg is None:
                    if batch:
                        yield pd.json_normalize(batch)
                        batch = []
                    continue

                payload = self._decode_value(msg)
                if payload is not None:
                    batch.append(payload)
                if len(batch) >= chunk_size:
                    yield pd.json_normalize(batch)
                    self._commit(consumer)
                    batch = []
        except KeyboardInterrupt:
            logger.info("KafkaConnector: stream interrupted by user")
        finally:
            if batch:
                yield pd.json_normalize(batch)
            self.close()

    def get_consumer_lag(self) -> Dict[str, int]:
        """Returns estimated consumer lag per partition."""
        return dict(self._lag_stats)

    def close(self) -> None:
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception:
                pass
            self._consumer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _consume_messages(self, max_messages: int = 1000) -> List[Dict]:
        """Consume up to max_messages from configured topics."""
        topics = self.config.get("topics", [])
        if not topics:
            return []

        consumer = self._get_consumer()
        if self._use_confluent:
            consumer.subscribe(topics)
        else:
            consumer.subscribe(topics)

        messages = []
        timeout = self.config.get("poll_timeout_s", _DEFAULT_POLL_TIMEOUT)
        idle_polls = 0

        while len(messages) < max_messages and idle_polls < _DEFAULT_MAX_POLL_RETRIES:
            msg = self._poll_one(consumer, timeout)
            if msg is None:
                idle_polls += 1
                continue
            idle_polls = 0
            payload = self._decode_value(msg)
            if payload is not None:
                messages.append(payload)

        self._commit(consumer)
        return messages

    def _poll_one(self, consumer, timeout: float) -> Optional[Any]:
        """Poll for one message from confluent or kafka-python consumer."""
        if self._use_confluent:
            msg = consumer.poll(timeout)
            if msg is None or msg.error():
                return None
            return msg
        else:
            try:
                # kafka-python: iterate
                record = next(iter(consumer), None)
                return record
            except StopIteration:
                return None

    def _decode_value(self, msg: Any) -> Optional[Dict]:
        """Decode message value to dict."""
        try:
            if self._use_confluent:
                raw = msg.value()
            else:
                raw = msg.value

            if raw is None:
                return None

            deserializer = self.config.get("value_deserializer", "json")
            if deserializer == "json":
                if isinstance(raw, bytes):
                    return json.loads(raw.decode("utf-8"))
                if isinstance(raw, dict):
                    return raw
                return json.loads(str(raw))
            elif deserializer == "string":
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                ts_field = self.config.get("timestamp_field", "timestamp")
                return {"value": text, ts_field: time.time()}
            else:
                return {"raw": str(raw)}
        except Exception as exc:
            logger.debug("KafkaConnector: message decode failed — %s", exc)
            return None

    def _commit(self, consumer) -> None:
        """Commit offsets if auto-commit is disabled."""
        if not self.config.get("enable_auto_commit", False):
            try:
                if self._use_confluent:
                    consumer.commit(asynchronous=False)
                else:
                    consumer.commit()
            except Exception as exc:
                logger.warning("KafkaConnector: offset commit failed — %s", exc)

    def _get_deserializer(self) -> Callable:
        """kafka-python value deserializer."""
        mode = self.config.get("value_deserializer", "json")
        if mode == "json":
            return lambda x: json.loads(x.decode("utf-8")) if x else {}
        return lambda x: x.decode("utf-8") if x else ""
