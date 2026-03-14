import json
import time
import requests
from kafka import KafkaProducer

def main():
    print("Connecting to Kafka broker at kafka:29092...")
    producer = KafkaProducer(
        bootstrap_servers=['kafka:29092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    topic = "dipex_pipeline"
    url = 'https://stream.wikimedia.org/v2/stream/recentchange'
    
    print(f"Connecting to REAL live stream: {url}")
    print(f"Piping live Wikipedia edits to Kafka topic: '{topic}'...")
    print("Keep this running in the background while you ingest from the UI!")
    print("-" * 50)
    
    count = 0
    
    # We use stream=True to keep the connection open for the Server Sent Events
    try:
        headers = {
            'Accept': 'text/event-stream',
            'User-Agent': 'dipex-demo/1.0 (test@example.com)'
        }
        response = requests.get(url, stream=True, headers=headers)
        print(f"Server responded with status: {response.status_code}")
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                
                # SSE data lines start with "data: "
                if decoded_line.startswith('data: '):
                    try:
                        # Parse the real JSON payload from Wikipedia
                        data = json.loads(decoded_line[6:])
                        
                        # We extract a few interesting fields to make it look clean for the demo
                        clean_event = {
                            "event_id": data.get("meta", {}).get("id"),
                            "wiki": data.get("wiki"),
                            "user": data.get("user"),
                            "title": data.get("title"),
                            "event_type": data.get("type"),
                            "timestamp": data.get("meta", {}).get("dt"),
                            "bot": data.get("bot", False),
                            "length_old": data.get("length", {}).get("old"),
                            "length_new": data.get("length", {}).get("new")
                        }
                        
                        # Send to Kafka!
                        producer.send(topic, clean_event)
                        count += 1
                        
                        if count % 10 == 0:
                            print(f"[LIVE] Piped {count} real Wikipedia edits to Kafka...")
                            
                    except json.JSONDecodeError:
                        continue
                        
    except KeyboardInterrupt:
        print("\nStopping live stream...")
    finally:
        producer.flush()
        producer.close()

if __name__ == "__main__":
    main()
