# utils/kafka_health.py
"""
Kafka broker health check with graceful degradation.

DIPEX works fully without Kafka. This utility checks broker
reachability before attempting stream operations.

Usage:
    from utils.kafka_health import kafka_is_available, require_kafka

    if kafka_is_available():
        start_stream_consumer()
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_kafka_available: Optional[bool] = None  # cached after first check


def kafka_is_available(timeout_ms: int = 3000) -> bool:
    """
    Returns True if the Kafka broker is reachable.
    Result is cached after the first call for the lifetime of the process.
    """
    global _kafka_available
    if _kafka_available is not None:
        return _kafka_available

    bootstrap = os.environ.get(
        "KAFKA_BOOTSTRAP_SERVERS",
        os.environ.get("KAFKA_BROKERS", "localhost:9092")
    )

    try:
        from confluent_kafka.admin import AdminClient
        admin = AdminClient({
            "bootstrap.servers": bootstrap,
            "socket.timeout.ms": timeout_ms,
            "api.version.request": True,
        })
        metadata = admin.list_topics(timeout=timeout_ms / 1000)
        _kafka_available = metadata is not None
        logger.info("Kafka broker reachable at %s", bootstrap)

    except ImportError:
        logger.warning(
            "confluent-kafka not installed — Kafka streaming unavailable. "
            "Install with: pip install confluent-kafka"
        )
        _kafka_available = False

    except Exception as exc:
        logger.warning(
            "Kafka broker NOT reachable at %s — "
            "stream ingestion will be skipped. "
            "Start Kafka with: docker-compose up -d kafka\n"
            "Error: %s",
            bootstrap,
            exc,
        )
        _kafka_available = False

    return _kafka_available


def require_kafka() -> None:
    """
    Raises RuntimeError with clear instructions if Kafka is not available.
    Use this only for operations where Kafka is truly mandatory.
    """
    if not kafka_is_available():
        bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        raise RuntimeError(
            f"Kafka broker is required but not reachable at {bootstrap}.\n"
            f"Start it with: docker-compose up -d kafka\n"
            f"Then verify: docker-compose exec kafka "
            f"kafka-topics.sh --bootstrap-server localhost:9092 --list"
        )


def reset_kafka_cache() -> None:
    """
    Resets the availability cache.
    Call this in tests that mock the Kafka connection.
    """
    global _kafka_available
    _kafka_available = None
