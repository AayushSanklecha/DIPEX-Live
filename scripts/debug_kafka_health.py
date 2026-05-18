
import os
from utils.kafka_health import kafka_is_available

def debug_health():
    print(f"KAFKA_BOOTSTRAP_SERVERS: {os.environ.get('KAFKA_BOOTSTRAP_SERVERS')}")
    print(f"KAFKA_BROKERS: {os.environ.get('KAFKA_BROKERS')}")
    
    print("Checking kafka_is_available(timeout_ms=5000)...")
    result = kafka_is_available(timeout_ms=5000)
    print(f"Result: {result}")

if __name__ == "__main__":
    # Manually load .env if needed
    from scripts.start_kafka_pipeline import _load_dotenv
    _load_dotenv()
    debug_health()
