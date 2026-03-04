"""
scripts/smart_router.py
-------------------------
Smart Database Router & Ingestion Pipeline

This script demonstrates the end-to-end flow the user requested:
1. You have raw data (CSV, JSON, key-value).
2. The router detects the "type" of data.
3. It uploads that data to the *correct* database (Postgres, Mongo, Redis).
4. DIPEX then reads from that database and processes it.

Usage:
  python scripts/smart_router.py
"""

import logging
import os
import sys
import pandas as pd
import json
import redis
from pymongo import MongoClient
from sqlalchemy import create_engine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.connectors.factory import ConnectorFactory
from ingestion.universal_intake import UniversalIntake

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dipex.smart_router")

# ── 1. Create Sample Data for Different Databases ─────────────────────────────

def create_sample_datasets():
    os.makedirs("data/samples", exist_ok=True)
    
    # 1. Relational / Tabular Data (Destined for PostgreSQL)
    df_sql = pd.DataFrame({
        "transaction_id": [101, 102, 103],
        "user_id": ["U1", "U2", "U1"],
        "amount": [250.0, 15.5, 900.0],
        "status": ["COMPLETED", "PENDING", "COMPLETED"]
    })
    df_sql.to_csv("data/samples/financial_tx.csv", index=False)
    
    # 2. Document / Nested Data (Destined for MongoDB)
    docs = [
        {"user_id": "U1", "profile": {"age": 30, "preferences": ["dark_mode", "email_alerts"]}, "history": [{"login": "2023-10-01"}]},
        {"user_id": "U2", "profile": {"age": 25, "preferences": ["sms_alerts"]}, "history": []}
    ]
    with open("data/samples/user_profiles.json", "w") as f:
        json.dump(docs, f)

    # 3. Key-Value / Session Data (Destined for Redis)
    sessions = {
        "session:U1:xyz": '{"cart_total": 45.0, "active": true}',
        "session:U2:abc": '{"cart_total": 0.0,  "active": false}'
    }
    with open("data/samples/live_sessions.json", "w") as f:
        json.dump(sessions, f)

    return "data/samples/financial_tx.csv", "data/samples/user_profiles.json", "data/samples/live_sessions.json"

# ── 2. The Smart Router ────────────────────────────────────────────────────────

class SmartDatabaseRouter:
    """Routes data to the correct DB, then pulls it into DIPEX."""
    
    def __init__(self):
        self.intake = UniversalIntake()
        
    def process_tabular_data(self, csv_path: str):
        """Flow: CSV -> PostgreSQL -> DIPEX"""
        logger.info("\n--- Processing Tabular Data ---")
        logger.info(f"1. Reading {csv_path}")
        df = pd.read_csv(csv_path)
        
        logger.info("2. Uploading to PostgreSQL (Database for structured/relational data)...")
        engine = create_engine("postgresql://admin:supersecret@localhost:5432/dipex")
        df.to_sql("financial_tx", engine, if_exists="replace", index=False)
        
        logger.info("3. DIPEX extracting from PostgreSQL and processing...")
        # Now use DIPEX to pull it back out and run ML profiling
        pg_config = {
            "source_type": "postgres",
            "uri": "postgresql://admin:supersecret@localhost:5432/dipex",
            "table": "financial_tx"
        }
        connector = ConnectorFactory.create("postgres", pg_config)
        df_extracted = connector.extract()
        
        snapshot = self.intake.ingest_dataframe(df_extracted, dataset_id="postgres_financial_tx")
        logger.info(f"4. DONE. ML Snapshot compliance: {snapshot.is_compliant}. Shape: {snapshot.data.shape}")

    def process_document_data(self, json_path: str):
        """Flow: JSON -> MongoDB -> DIPEX"""
        logger.info("\n--- Processing Document Data ---")
        logger.info(f"1. Reading {json_path}")
        with open(json_path, "r") as f:
            docs = json.load(f)
            
        logger.info("2. Uploading to MongoDB (Database for nested/unstructured documents)...")
        client = MongoClient("mongodb://admin:supersecret@localhost:27017/dipex?authSource=admin")
        db = client["dipex"]
        collection = db["user_profiles"]
        collection.drop() # clean for demo
        collection.insert_many(docs)
        
        logger.info("3. DIPEX extracting from MongoDB and processing...")
        mongo_config = {
            "source_type": "mongodb",
            "uri": "mongodb://admin:supersecret@localhost:27017/dipex?authSource=admin",
            "database": "dipex",
            "collection": "user_profiles"
        }
        connector = ConnectorFactory.create("mongodb", mongo_config)
        df_extracted = connector.extract()
        
        snapshot = self.intake.ingest_dataframe(df_extracted, dataset_id="mongo_user_profiles")
        logger.info(f"4. DONE. ML Snapshot compliance: {snapshot.is_compliant}. Shape: {snapshot.data.shape}")

    def process_key_value_data(self, kv_json_path: str):
        """Flow: Key-Value -> Redis -> DIPEX"""
        logger.info("\n--- Processing Key-Value Data ---")
        logger.info(f"1. Reading {kv_json_path}")
        with open(kv_json_path, "r") as f:
            kv_data = json.load(f)
            
        logger.info("2. Uploading to Redis (Database for fast session/cache data)...")
        r = redis.Redis.from_url("redis://:dipexredis@localhost:6379/0")
        for k, v in kv_data.items():
            r.set(k, v)
        
        logger.info("3. DIPEX extracting from Redis and processing...")
        redis_config = {
            "source_type": "redis",
            "host": "localhost",
            "port": 6379,
            "password": "dipexredis",
            "key_pattern": "session:*"
        }
        connector = ConnectorFactory.create("redis", redis_config)
        df_extracted = connector.extract()
        
        snapshot = self.intake.ingest_dataframe(df_extracted, dataset_id="redis_live_sessions")
        logger.info(f"4. DONE. ML Snapshot compliance: {snapshot.is_compliant}. Shape: {snapshot.data.shape}")


if __name__ == "__main__":
    logger.info("Generating sample files...")
    csv_file, json_file, kv_file = create_sample_datasets()
    
    router = SmartDatabaseRouter()
    
    # 1. Structured data goes to Postgres -> DIPEX
    router.process_tabular_data(csv_file)
    
    # 2. Nested JSON goes to Mongo -> DIPEX
    router.process_document_data(json_file)
    
    # 3. Fast keys go to Redis -> DIPEX
    router.process_key_value_data(kv_file)
    
    logger.info("\nAll data successfully routed to their respective databases and processed by DIPEX!")
