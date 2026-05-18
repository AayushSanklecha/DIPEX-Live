"""
scripts/demo_03_kafka.py
──────────────────────────────
DIPEX Demo — Kafka Stream Ingestion

Starts a background producer that sends IoT sensor readings,
then ingests them through the DIPEX pipeline using windowed streaming.

Prerequisites:
    docker-compose -f docker-compose.demo.yml up -d
    pip install confluent-kafka

Usage:
    python scripts/demo_03_kafka.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _demo_setup import configure_demo_environment
configure_demo_environment()

from ingestion.universal_intake import UniversalIntake, SourceConfig
from ingestion.readers.stream_reader import KafkaSourceConfig, WindowConfig

TOPIC = "iot_sensors"
BROKERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# ── IoT Sensor Simulator ─────────────────────────────────────────────

SENSOR_IDS = [f"sensor_{i:03d}" for i in range(1, 11)]
SENSOR_TYPES = ["temperature", "humidity", "pressure", "co2", "light"]
LOCATIONS = ["Building-A Floor-1", "Building-A Floor-2", "Building-B Floor-1",
             "Building-B Floor-3", "Warehouse-C"]


def generate_sensor_reading() -> dict:
    """Generate a realistic IoT sensor reading with occasional imperfections."""
    sensor_type = random.choice(SENSOR_TYPES)
    value_ranges = {
        "temperature": (18.0, 35.0),
        "humidity": (30.0, 90.0),
        "pressure": (980.0, 1040.0),
        "co2": (350.0, 1200.0),
        "light": (0.0, 1000.0),
    }
    lo, hi = value_ranges[sensor_type]
    value = round(random.uniform(lo, hi), 2)

    # 5% chance of anomalous reading (realistic sensor drift/failure)
    if random.random() < 0.05:
        value = round(value * random.choice([3.0, 0.1, -1.0]), 2)

    reading = {
        "sensor_id": random.choice(SENSOR_IDS),
        "sensor_type": sensor_type,
        "value": value,
        "unit": {"temperature": "C", "humidity": "%", "pressure": "hPa",
                 "co2": "ppm", "light": "lux"}[sensor_type],
        "location": random.choice(LOCATIONS),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "battery_pct": random.randint(10, 100),
    }

    # --- Real-world data quality issues ---
    # 8% chance sensor doesn't report its ID (connection glitch)
    if random.random() < 0.08:
        reading["sensor_id"] = None

    # 5% chance value is missing (sensor timeout)
    if random.random() < 0.05:
        reading["value"] = None

    # 3% chance location is missing (GPS module failure)
    if random.random() < 0.03:
        reading["location"] = None

    # 2% chance of corrupt timestamp
    if random.random() < 0.02:
        reading["timestamp"] = "INVALID_TS"

    return reading


def produce_messages(n_messages: int = 80, delay_s: float = 0.02) -> None:
    """Produce IoT sensor messages to Kafka topic."""
    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": BROKERS})
    print(f"  -> Producing {n_messages} IoT sensor readings to topic '{TOPIC}'...")

    for i in range(n_messages):
        reading = generate_sensor_reading()
        producer.produce(TOPIC, json.dumps(reading).encode("utf-8"))
        if (i + 1) % 20 == 0:
            producer.flush()
            print(f"     ... {i + 1}/{n_messages} sent")
        time.sleep(delay_s)

    producer.flush()
    print(f"  -> All {n_messages} messages produced.")


def consume_messages_directly(brokers: str, topic: str, n: int = 200, timeout: float = 10.0):
    """Consume messages directly from Kafka, returning parsed events."""
    from confluent_kafka import Consumer, KafkaError
    import json

    consumer = Consumer({
        "bootstrap.servers": brokers,
        "group.id": f"dipex-demo-direct-{int(time.time())}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([topic])

    events = []
    t0 = time.time()
    print(f"  -> Consuming from topic '{topic}'...")

    while len(events) < n and (time.time() - t0) < timeout:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                break
            continue
        try:
            record = json.loads(msg.value().decode("utf-8"))
            events.append(record)
        except Exception:
            pass

    consumer.close()
    print(f"  -> Consumed {len(events)} messages.")
    return events


def main():
    print("\n" + "=" * 70)
    print("  DIPEX DEMO - PATH 2: Kafka Stream Ingestion")
    print("=" * 70)

    # ── Step 1: Produce messages ──────────────────────────────────────
    print("\n-> Step 1: Producing IoT sensor data...")
    produce_messages(80, 0.02)
    time.sleep(1.0)

    # ── Step 2: Consume messages directly ─────────────────────────────
    print("\n-> Step 2: Consuming messages from Kafka...")
    events = consume_messages_directly(BROKERS, TOPIC, n=200, timeout=15.0)

    if not events:
        print("  [!!] No messages consumed. Is Kafka running?")
        return None

    # ── Step 3: Feed through pipeline using collect_events ────────────
    print(f"\n-> Step 3: Processing {len(events)} events through DIPEX pipeline...")
    print("  (using tumbling window strategy)")

    from ingestion.readers.stream_reader import StreamReader

    reader = StreamReader()
    window_cfg = WindowConfig(strategy="tumbling", window_size_s=60.0)
    stream_results = reader.collect_events(events, window_cfg)

    # Combine windowed results into one DataFrame
    import pandas as pd
    dfs = [r.data for r in stream_results if not r.data.empty]
    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Now run through full pipeline via file ingestion (Bronze -> pipeline)
    # Save temp CSV and ingest as file to get full pipeline treatment
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "iot_kafka_events.csv")
    combined_df.to_csv(tmp_path, index=False)

    source = SourceConfig(
        source_type="file",
        dataset_id="iot_sensor_stream",
        data_mode="stream",
        path=tmp_path,
        require_quality_pass=False,
        block_on_schema_break=False,
    )

    intake = UniversalIntake()
    snapshot = intake.ingest(source)

    # ── Show results ──────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  RESULTS")
    print("-" * 70)
    print(f"  Rows ingested      : {snapshot.row_count}")
    print(f"  Schema version     : {snapshot.schema_version}")
    print(f"  Quality score      : {snapshot.quality_score:.2%}")
    print(f"  Validation status  : {snapshot.validation_status}")
    print(f"  ISSF compliant     : {snapshot.is_compliant}")
    print(f"  Data mode          : {snapshot.data_mode}")
    print("-" * 70)

    if snapshot.data is not None and not snapshot.data.empty:
        if "sensor_type" in snapshot.data.columns:
            print("\n  Sensor types captured:")
            for stype, count in snapshot.data["sensor_type"].value_counts().items():
                print(f"    - {stype}: {count} readings")
        key_cols = [c for c in ["sensor_id", "sensor_type", "value", "unit", "location"]
                    if c in snapshot.data.columns]
        if key_cols:
            print(f"\n  Sample readings:")
            print(snapshot.data[key_cols].head(5).to_string(index=False))

    print("\n[OK] Kafka stream ingestion complete!\n")
    return snapshot


if __name__ == "__main__":
    main()
