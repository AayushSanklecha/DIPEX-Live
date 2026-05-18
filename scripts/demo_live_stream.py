import sys
import subprocess
import time
import json
import random
import uuid
from datetime import datetime

# Auto-install dependencies
try:
    from confluent_kafka import Producer
except ImportError:
    print("Installing confluent-kafka...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "confluent-kafka"])
    from confluent_kafka import Producer

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result """
    if err is not None:
        print(f"❌ Message delivery failed: {err}")
    else:
        # Print a cool minimal console UI for the stream
        print(f"🌊 [STREAM] → Topic: {msg.topic()} | Partition: {msg.partition()} | Offset: {msg.offset()}")

def generate_live_transaction():
    """Generates a realistic financial transaction/IoT packet"""
    cities = ["New York", "London", "Tokyo", "Berlin", "Dubai", "Mumbai", "Singapore", "Sydney"]
    merchants = ["Amazon", "Uber", "Starbucks", "Apple", "Netflix", "Walmart", "Local Cafe", "AirBnB"]
    
    # 2% chance of generating an anomaly (fraud)
    is_anomaly = random.random() < 0.02
    
    amount = round(random.uniform(5.0, 150.0), 2)
    if is_anomaly:
        amount = round(random.uniform(2000.0, 10000.0), 2)  # Huge fraud amount
        
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": f"USR_{random.randint(10000, 99999)}",
        "merchant": random.choice(merchants),
        "location": random.choice(cities),
        "amount": amount,
        "currency": "USD",
        "device_type": random.choice(["mobile", "desktop", "pos_terminal"]),
        "is_suspicious": is_anomaly
    }

def main():
    topic_name = "live_transactions"
    broker_url = "localhost:9092"
    
    # Conf matching the local docker-compose
    conf = {
        'bootstrap.servers': broker_url,
        'client.id': 'dipex-presentation-streamer'
    }

    print("==========================================================")
    print("🚀  DIPEX KAFKA STREAM SIMULATOR (PAYMENT GATEWAY)        ")
    print("==========================================================")
    print(f"Connecting to broker   : {broker_url}")
    print(f"Target Kafka Topic     : {topic_name}")
    print("==========================================================")
    print("Starting stream... Press Ctrl+C to stop.\n")
    
    try:
        producer = Producer(conf)
    except Exception as e:
        print(f"❌ Critical Error: Could not connect to Kafka. Is docker running? ({e})")
        sys.exit(1)

    try:
        while True:
            # Generate fake realtime event
            event_data = generate_live_transaction()
            payload = json.dumps(event_data)
            
            # Print physical packet for visual aesthetic in terminal
            print(f"📦 Payload: {payload[:80]}...")
            
            # Fire and forget into Kafka
            producer.produce(
                topic=topic_name, 
                key=event_data["user_id"],
                value=payload, 
                callback=delivery_report
            )
            
            # Trigger callbacks
            producer.poll(0)
            
            # Wait for next event (simulates live variable traffic)
            time.sleep(random.uniform(0.1, 0.8))
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping the live stream...")

    print("Flushing final messages to broker...")
    producer.flush()
    print("Done!")

if __name__ == "__main__":
    main()
