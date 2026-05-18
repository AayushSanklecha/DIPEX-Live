# tests/test_kafka_health.py
"""
Kafka health check utility tests.
These run without a live broker — we mock the connection.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset Kafka availability cache before each test."""
    from utils.kafka_health import reset_kafka_cache
    reset_kafka_cache()
    yield
    reset_kafka_cache()


def test_returns_false_when_broker_unreachable():
    """kafka_is_available() must return False (not raise) when broker is down."""
    with patch("confluent_kafka.admin.AdminClient") as mock_admin:
        mock_admin.return_value.list_topics.side_effect = Exception("Connection refused")
        from utils.kafka_health import kafka_is_available
        result = kafka_is_available(timeout_ms=100)
    assert result is False


def test_returns_true_when_broker_reachable():
    """kafka_is_available() must return True when AdminClient succeeds."""
    with patch("confluent_kafka.admin.AdminClient") as mock_admin:
        mock_meta = MagicMock()
        mock_admin.return_value.list_topics.return_value = mock_meta
        from utils.kafka_health import kafka_is_available
        result = kafka_is_available(timeout_ms=100)
    assert result is True


def test_result_is_cached():
    """Second call uses cache — AdminClient called only once."""
    with patch("confluent_kafka.admin.AdminClient") as mock_admin:
        mock_admin.return_value.list_topics.return_value = MagicMock()
        from utils.kafka_health import kafka_is_available
        kafka_is_available()
        kafka_is_available()
        kafka_is_available()
    assert mock_admin.call_count == 1


def test_require_kafka_raises_with_helpful_message():
    """require_kafka() must raise RuntimeError with docker-compose instructions."""
    from utils.kafka_health import require_kafka, reset_kafka_cache
    reset_kafka_cache()
    with patch("confluent_kafka.admin.AdminClient") as mock_admin:
        mock_admin.return_value.list_topics.side_effect = Exception("Connection refused")
        with pytest.raises(RuntimeError) as exc_info:
            require_kafka()
    assert "docker-compose" in str(exc_info.value).lower()


def test_graceful_degradation_no_crash_without_kafka():
    """
    The most important test: pipeline MUST NOT crash when Kafka is down.
    kafka_is_available() returns False — no exception propagates to caller.
    """
    with patch("confluent_kafka.admin.AdminClient") as mock_admin:
        mock_admin.return_value.list_topics.side_effect = ConnectionRefusedError
        from utils.kafka_health import kafka_is_available
        result = kafka_is_available(timeout_ms=100)
    assert result is False, "kafka_is_available() must never raise — must return False"
