# tests/test_kafka_connectivity.py
"""
Issue 11: Kafka broker connectivity smoke test.
Requires: docker-compose up -d (Kafka + Zookeeper must be running)
Run: pytest tests/test_kafka_connectivity.py -v -m integration
"""
import os
import uuid

import pytest

pytestmark = pytest.mark.integration  # skip in unit test runs


@pytest.fixture(scope="module")
def bootstrap_servers() -> str:
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def test_kafka_broker_is_reachable(bootstrap_servers: str) -> None:
    """Can we connect to the Kafka broker at all?"""
    from confluent_kafka.admin import AdminClient

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    metadata = admin.list_topics(timeout=5)

    assert metadata is not None, (
        f"Kafka broker at {bootstrap_servers} did not respond. "
        "Is docker-compose up? Is port 9092 mapped?"
    )


def test_required_topics_exist(bootstrap_servers: str) -> None:
    """All DIPEX topics must exist after docker-compose up."""
    from confluent_kafka.admin import AdminClient

    REQUIRED_TOPICS = [
        "raw_events",
        "cleaned",
        "gold_outputs",
        "drift_alerts",
        "rl_signals",
    ]

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    metadata = admin.list_topics(timeout=5)
    existing_topics = list(metadata.topics.keys())

    missing = [t for t in REQUIRED_TOPICS if t not in existing_topics]
    assert not missing, (
        f"Missing Kafka topics: {missing}. "
        "Run: docker-compose exec kafka kafka-topics.sh --create ... "
        "OR add topic auto-creation to docker-compose.yml "
        "KAFKA_AUTO_CREATE_TOPICS_ENABLE=true"
    )


def test_can_produce_and_consume_message(bootstrap_servers: str) -> None:
    """Full round-trip: produce a test message, consume it back."""
    from confluent_kafka import Consumer, Producer

    TEST_TOPIC = "dipex_connectivity_test"
    TEST_MESSAGE = f"dipex-smoke-{uuid.uuid4()}"

    # Produce
    p = Producer({"bootstrap.servers": bootstrap_servers})
    p.produce(TEST_TOPIC, TEST_MESSAGE.encode())
    p.flush(timeout=5)

    # Consume
    c = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": f"dipex-smoke-{uuid.uuid4()}",
        "auto.offset.reset": "earliest",
    })
    c.subscribe([TEST_TOPIC])
    msg = c.poll(timeout=5.0)
    c.close()

    assert msg is not None, (
        f"No message received from topic '{TEST_TOPIC}' within 5s"
    )
    assert msg.value().decode() == TEST_MESSAGE, (
        "Consumed message does not match produced message"
    )
