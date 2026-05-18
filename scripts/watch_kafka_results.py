"""
scripts/watch_kafka_results.py
-------------------------------
Simple Kafka consumer to watch 'gold_outputs' topic and print processed results.
"""

import json
import argparse
from confluent_kafka import Consumer, KafkaError

def main():
    parser = argparse.ArgumentParser(description="DIPEX Kafka Result Monitor")
    parser.add_argument("--topic", type=str, default="gold_outputs", help="Kafka topic to watch")
    parser.add_argument("--brokers", type=str, default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--group", type=str, default="dipex-monitor", help="Consumer group ID")
    args = parser.parse_args()

    conf = {
        'bootstrap.servers': args.brokers,
        'group.id': args.group,
        'auto.offset.reset': 'latest'
    }

    consumer = Consumer(conf)
    consumer.subscribe([args.topic])

    print(f"Monitoring topic '{args.topic}' at {args.brokers}...")
    print("Waiting for results... (Press Ctrl+C to stop)")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Error: {msg.error()}")
                    break

            try:
                payload = json.loads(msg.value().decode('utf-8'))
                batch_id = payload.get('batch_id', 'unknown')
                summary = payload.get('summary', {})
                decision = summary.get('gate_decision', 'UNKNOWN')
                
                print(f"\n[RESULT] Batch: {batch_id}")
                print(f"         Decision: {decision}")
                print(f"         Confidence: {summary.get('confidence_vector', {}).get('confidence_score', 0.0):.4f}")
                print(f"         Stages: {len(summary.get('stages', []))} completed")
            except Exception as e:
                print(f"Error decoding message: {e}")
                print(f"Raw value: {msg.value()}")

    except KeyboardInterrupt:
        print("\nMonitor stopped.")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
