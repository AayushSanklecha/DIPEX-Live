"""
Seed MongoDB with real Tips dataset (seaborn restaurant data).
Run:  python samples/seed_mongodb.py
Needs: pip install pymongo  +  docker compose up -d mongodb
"""
import json, os
from pathlib import Path

try:
    from pymongo import MongoClient
    uri = os.getenv("MONGO_URI", "mongodb://admin:supersecret@localhost:27017/dipex?authSource=admin")
    docs = json.loads(Path("samples/tips.json").read_text())
    cli = MongoClient(uri, serverSelectionTimeoutMS=5000)
    col = cli["dipex"]["tips"]
    col.drop()
    col.insert_many(docs)
    print(f"✓ MongoDB  → {len(docs)} tip records inserted into dipex.tips")
    print(f"  Sample: {docs[0]}")
    cli.close()
except Exception as e:
    print(f"✕ MongoDB seed failed: {e}")
