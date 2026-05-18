"""
scripts/produce_kafka_test_data.py
-----------------------------------
Produces sample JSON data to the Kafka 'raw_events' topic for DIPEX pipeline testing.
"""

import json
import time
import argparse
import random
import pandas as pd
from confluent_kafka import Producer

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def main():
    parser = argparse.ArgumentParser(description="DIPEX Kafka Test Data Producer")
    parser.add_argument("--continuous", action="store_true", help="Run in a continuous loop")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between messages in continuous mode (seconds)")
    parser.add_argument("--topic", type=str, default="raw_events", help="Kafka topic to produce to")
    parser.add_argument("--brokers", type=str, default="localhost:9092", help="Kafka bootstrap servers")
    args = parser.parse_args()

    # 1. Load sample data
    csv_path = "data/messy_test_data.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        df = pd.DataFrame([{
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "region": "North",
            "product": "Laptop",
            "sales": 1500.0,
            "units": 1,
            "customer_id": "C1234"
        }])

    # 2. Kafka Configuration
    conf = {
        'bootstrap.servers': args.brokers,
        'client.id': 'dipex-test-producer'
    }

    producer = Producer(conf)
    topic = args.topic

    print(f"Producing to topic '{topic}' at {conf['bootstrap.servers']}...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # 3. Produce records
            for _, row in df.iterrows():
                data = row.to_dict()
                
                # Update timestamp and jitter sales for "realism"
                data['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
                if 'sales' in data:
                    data['sales'] = round(data['sales'] * (0.8 + 0.4 * random.random()), 2)

                producer.produce(
                    topic, 
                    key=str(data.get('customer_id', '')),
                    value=json.dumps(data), 
                    callback=delivery_report
                )
                producer.poll(0)
                
                if args.continuous:
                    time.sleep(args.delay)
            
            if not args.continuous:
                break
    except KeyboardInterrupt:
        print("\nProducer stopped by user.")

    # 4. Wait for any outstanding messages to be delivered
    print("Flushing final messages...")
    producer.flush()
    print("Done!")

if __name__ == "__main__":
    main()
