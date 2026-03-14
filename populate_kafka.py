import json
import time
import random
import uuid
from datetime import datetime
from kafka import KafkaProducer

def generate_transaction():
    tx_id = str(uuid.uuid4())
    user_id = f"U{random.randint(1000, 9999)}"
    amount = round(random.uniform(5.0, 5000.0), 2)
    currency = random.choice(["USD", "EUR", "GBP", "JPY"])
    # 5% chance of failing status
    status = random.choices(["COMPLETED", "FAILED", "PENDING"], weights=[0.85, 0.05, 0.10])[0]
    
    # Missing value injection for testing
    if random.random() < 0.02:
        amount = None
    if random.random() < 0.02:
        currency = None

    return {
        "tx_id": tx_id,
        "user_id": user_id,
        "amount": amount,
        "currency": currency,
        "status": status,
        "tx_date": datetime.utcnow().isoformat() + "Z",
        "device_ip": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
    }

def main():
    print("Connecting to Kafka at kafka:29092...")
    producer = KafkaProducer(
        bootstrap_servers=['kafka:29092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    topic = "dipex_pipeline"
    num_messages = 500
    
    print(f"Sending {num_messages} messages to topic '{topic}'...")
    for i in range(num_messages):
        msg = generate_transaction()
        producer.send(topic, msg)
        if (i + 1) % 100 == 0:
            print(f"Sent {i + 1}/{num_messages} messages...")
            time.sleep(0.1) # Small delay to ensure order and avoid overwhelming buffer
            
    producer.flush()
    print(f"Successfully populated {num_messages} messages into Kafka topic '{topic}'!")

if __name__ == "__main__":
    main()
