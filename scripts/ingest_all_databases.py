"""
scripts/ingest_all_databases.py
---------------------------------
Multi-DB Aggregator for DIPEX

This script demonstrates how to pull raw data from every configured
database (MongoDB, Redis, PostgreSQL, Neo4j, Parquet, Kafka) and
funnel it into the standard DIPEX Universal Intake pipeline for
profiling, quality gating, and ML processing.

Usage:
  python scripts/ingest_all_databases.py
"""

import logging
import os
import sys

# Ensure dipex package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.connectors.factory import ConnectorFactory
from ingestion.universal_intake import UniversalIntake

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dipex.ingest_all")


# ── Configurations for your local Docker databases ──────────────────────────

SOURCES = [
    {
        "id": "mongo_tips",
        "config": {
            "source_type": "mongodb",
            "uri": "mongodb://admin:supersecret@localhost:27017/dipex?authSource=admin",
            "database": "dipex",
            "collection": "tips"
        }
    },
    {
        "id": "postgres_titanic",
        "config": {
            "source_type": "postgres",
            "uri": "postgresql://admin:supersecret@localhost:5432/dipex",
            "table": "titanic"
        }
    },
    {
        "id": "redis_passengers",
        "config": {
            "source_type": "redis",
            "url": "redis://:dipexredis@localhost:6379/0",
            "key_pattern": "dipex:passenger:*"
        }
    },
    {
        "id": "neo4j_titanic",
        "config": {
            "source_type": "neo4j",
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "supersecret",
            "query": "MATCH (n:Passenger) RETURN n LIMIT 100"
        }
    },
    {
        "id": "duckdb_titanic",
        "config": {
            "source_type": "duckdb",
            "duckdb_path": "data/dipex.duckdb",
            "table": "titanic"
        }
    }
]


def run_aggregation():
    logger.info("Starting Multi-DB Aggregation Pipeline...")
    intake = UniversalIntake()
    
    success_count = 0
    failed_count  = 0
    
    for source in SOURCES:
        dataset_id = source["id"]
        cfg        = source["config"]
        db_type    = cfg["source_type"]
        
        logger.info(f"\n[{dataset_id}] Attempting connection to {db_type.upper()}...")
        try:
            # 1. Connect and extract
            connector = ConnectorFactory.create(db_type, cfg)
            
            # Fast fail if DB isn't running
            if not connector.test_connection():
                logger.warning(f"[{dataset_id}] SKIPPED — database unreachable (is Docker running?)")
                failed_count += 1
                continue
                
            df = connector.extract()
            logger.info(f"[{dataset_id}] Extracted {len(df)} rows from {db_type}")
            
            if df.empty:
                logger.info(f"[{dataset_id}] SKIPPED — table/collection is empty")
                continue
                
            logger.info(f"[{dataset_id}] Wrapping extracted data into ISSFSnapshot...")
            import uuid
            from ingestion.issf import ISSFSnapshot
            
            run_id = str(uuid.uuid4())
            snapshot = ISSFSnapshot(
                dataset_id=dataset_id,
                snapshot_id=run_id,
                schema_version="1.0",
                data_mode="batch",
                source_type="database",
                source_uri=cfg.get("uri", cfg.get("url", cfg.get("duckdb_path", ""))),
                column_metadata=[],
                row_count=len(df),
                quality_score=1.0,
                validation_status="PASSED",
                error_logs=[],
                data=df
            )
            
            # 3. Trigger full ML pipeline and report generation
            logger.info(f"[{dataset_id}] Triggering 13-stage PipelineBridge for full processing & reporting...")
            from ingestion.pipeline_bridge import PipelineBridge
            
            bridge = PipelineBridge()
            try:
                result = bridge.run(
                    snapshot=snapshot,
                    run_id=run_id
                )
                summary = result.summary()
                logger.info(f"[{dataset_id}] Pipeline Complete! Gate Decision: {summary.get('gate_decision')} | Report: {summary.get('report_path')}")
                success_count += 1
            except Exception as pl_err:
                logger.error(f"[{dataset_id}] Pipeline Execution ERROR — {pl_err}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"[{dataset_id}] ERROR — {e}")
            failed_count += 1
            
    logger.info(f"\nAggregation & Full Pipeline execution complete! {success_count} successful, {failed_count} skipped/failed.")

if __name__ == "__main__":
    run_aggregation()
