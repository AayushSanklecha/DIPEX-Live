"""
scripts/demo_01_postgres.py
──────────────────────────────
DIPEX Demo — PostgreSQL Ingestion

Reads the `sales_orders` table from a local PostgreSQL instance
and runs it through the full DIPEX pipeline.

Prerequisites:
    docker-compose -f docker-compose.demo.yml up -d
    (wait ~10s for Postgres to seed)

Usage:
    python scripts/demo_01_postgres.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _demo_setup import configure_demo_environment
configure_demo_environment()

from ingestion.universal_intake import UniversalIntake, SourceConfig
from ingestion.readers.db_reader import DBSourceConfig


def main():
    print("\n" + "=" * 70)
    print("  DIPEX DEMO - PATH 1a: PostgreSQL Ingestion")
    print("=" * 70)

    db_cfg = DBSourceConfig(
        backend="postgres",
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        database=os.environ.get("POSTGRES_DB", "dipex_demo"),
        username_env="POSTGRES_USER",
        password_env="POSTGRES_PASSWORD",
        table_or_collection="sales_orders",
    )

    source = SourceConfig(
        source_type="database",
        dataset_id="ecommerce_sales",
        data_mode="batch",
        db_config=db_cfg,
        require_quality_pass=False,
        block_on_schema_break=False,
    )

    print("\n-> Connecting to PostgreSQL and reading `sales_orders` table...")
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
    print(f"  Snapshot ID        : {snapshot.snapshot_id[:16]}...")
    print("-" * 70)

    if snapshot.data is not None and not snapshot.data.empty:
        print("\n  Sample data (first 5 rows):")
        print(snapshot.data.head().to_string(index=False))

    print("\n[OK] PostgreSQL ingestion complete!\n")
    return snapshot


if __name__ == "__main__":
    main()
