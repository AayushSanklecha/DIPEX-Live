"""
scripts/demo_02_mongodb.py
──────────────────────────────
DIPEX Demo — MongoDB Ingestion

Loads nested product catalog documents into MongoDB,
then ingests through the full DIPEX pipeline.

Prerequisites:
    docker-compose -f docker-compose.demo.yml up -d

Usage:
    python scripts/demo_02_mongodb.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _demo_setup import configure_demo_environment
configure_demo_environment()

from ingestion.universal_intake import UniversalIntake, SourceConfig
from ingestion.readers.db_reader import DBSourceConfig


SEED_FILE = Path(__file__).parent / "seed_data" / "seed_mongo.json"


def seed_mongodb() -> int:
    """Insert seed data into MongoDB. Returns document count."""
    from pymongo import MongoClient

    host = os.environ.get("MONGO_HOST", "localhost")
    port = int(os.environ.get("MONGO_PORT", 27017))
    client = MongoClient(host, port, serverSelectionTimeoutMS=5000)

    db = client["dipex_demo"]
    coll = db["product_catalog"]

    if coll.count_documents({}) == 0:
        with open(SEED_FILE, encoding="utf-8") as f:
            docs = json.load(f)
        coll.insert_many(docs)
        print(f"  -> Seeded {len(docs)} documents into MongoDB")
    else:
        print(f"  -> MongoDB already has {coll.count_documents({})} documents (skipping seed)")

    count = coll.count_documents({})
    client.close()
    return count


def main():
    print("\n" + "=" * 70)
    print("  DIPEX DEMO - PATH 1b: MongoDB Ingestion")
    print("=" * 70)

    print("\n-> Seeding MongoDB with product catalog...")
    doc_count = seed_mongodb()

    # Build a no-auth URI (Docker MongoDB has no auth by default)
    mongo_host = os.environ.get("MONGO_HOST", "localhost")
    mongo_port = os.environ.get("MONGO_PORT", "27017")
    mongo_uri = os.environ.get("MONGO_URI", f"mongodb://{mongo_host}:{mongo_port}/")
    os.environ["_DEMO_MONGO_URI"] = mongo_uri

    db_cfg = DBSourceConfig(
        backend="mongodb",
        host=mongo_host,
        port=int(mongo_port),
        database="dipex_demo",
        table_or_collection="product_catalog",
        dsn_env="_DEMO_MONGO_URI",  # use direct URI — no auth needed
    )

    source = SourceConfig(
        source_type="database",
        dataset_id="product_catalog",
        data_mode="batch",
        db_config=db_cfg,
        require_quality_pass=False,
        block_on_schema_break=False,
    )

    print(f"\n-> Reading {doc_count} documents from MongoDB `product_catalog`...")
    print("  (nested documents will be automatically flattened)")
    intake = UniversalIntake()
    snapshot = intake.ingest(source)

    print("\n" + "-" * 70)
    print("  RESULTS")
    print("-" * 70)
    print(f"  Rows ingested      : {snapshot.row_count}")
    print(f"  Schema version     : {snapshot.schema_version}")
    print(f"  Quality score      : {snapshot.quality_score:.2%}")
    print(f"  Validation status  : {snapshot.validation_status}")
    print(f"  ISSF compliant     : {snapshot.is_compliant}")
    print("-" * 70)

    if snapshot.data is not None and not snapshot.data.empty:
        print(f"\n  Columns detected ({len(snapshot.data.columns)}):")
        for col in sorted(snapshot.data.columns):
            if not col.startswith("_"):
                print(f"    - {col}")
        print(f"\n  Sample data (first 3 rows, key columns):")
        key_cols = [c for c in ["sku", "name", "brand", "price", "category"] if c in snapshot.data.columns]
        if key_cols:
            print(snapshot.data[key_cols].head(3).to_string(index=False))

    print("\n[OK] MongoDB ingestion complete! Nested docs flattened.\n")
    return snapshot


if __name__ == "__main__":
    main()
