"""
Seed Redis with Titanic passenger data (each row = one Redis hash).
Run:  python samples/seed_redis.py
Needs: pip install redis  +  docker compose up -d redis
"""
import json, os
from pathlib import Path
import pandas as pd

try:
    import redis
    r = redis.Redis(
        host=os.getenv("REDIS_HOST","localhost"),
        port=int(os.getenv("REDIS_PORT",6379)),
        password=os.getenv("REDIS_PASSWORD","dipexredis"),
        decode_responses=True,
    )
    r.ping()
    df = pd.read_csv("samples/titanic.csv").fillna("").head(200)
    pipe = r.pipeline()
    for i, row in df.iterrows():
        pipe.hset(f"dipex:passenger:{i}", mapping={k: str(v) for k, v in row.items()})
    pipe.set("dipex:passenger:count", len(df))
    pipe.execute()
    print(f"✓ Redis    → {len(df)} passengers stored as hashes (dipex:passenger:0 … {len(df)-1})")
    sample = r.hgetall("dipex:passenger:0")
    print(f"  Sample key dipex:passenger:0: Name={sample.get('Name','')} Survived={sample.get('Survived','')} Fare={sample.get('Fare','')}")
except Exception as e:
    print(f"✕ Redis seed failed: {e}")
