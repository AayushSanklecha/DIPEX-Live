"""
simple_pipeline.py
------------------
Simplified pipeline orchestration for core workflow:
1. Load data
2. Clean data
3. Run EDA
4. Detect anomalies
5. Generate visualizations
6. Generate report
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def orchestrate_pipeline(run_id: str, target_col: str = None, source_path: str = None) -> bool:
    """
    Execute the simplified analytics pipeline by hooking into the actual Validation/Proposal/Verifier engines.
    """
    try:
        logger.info(f"Starting actual pipeline for run_id={run_id}, target={target_col}")
        from ingestion.pipeline_bridge import PipelineBridge
        from ingestion.universal_intake import UniversalIntake, SourceConfig
        from ingestion.issf import ISSFSnapshot
        import json

        snapshot = None

        if source_path and os.path.exists(source_path):
            logger.info(f"Triggering UniversalIntake for source: {source_path}")
            # Ensure config.yaml exists or create a default config
            try:
                intake = UniversalIntake.from_yaml("config.yaml")
            except Exception:
                intake = UniversalIntake()
                
            cfg = SourceConfig(
                source_type="file",
                dataset_id=run_id,
                data_mode="batch",
                path=source_path
            )
            snapshot = intake.ingest(cfg)
        else:
            # Fallback path if source is not provided or ingestion failed
            # Try mapping run_id to a known snapshot from the audit log
            snapshot_id = run_id
            audit_log = "audit/audit.jsonl"
            if os.path.exists(audit_log):
                with open(audit_log, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("run_id") == run_id and "snapshot_id" in entry:
                                snapshot_id = entry["snapshot_id"]
                                break
                        except Exception:
                            pass

            snapshot_path_json = Path(f"data/snapshots/{snapshot_id}_issf.json")
            snapshot_path_old = Path(f"data/snapshots/{snapshot_id}.json")
            
            target_path = snapshot_path_json if snapshot_path_json.exists() else snapshot_path_old

            if not target_path.exists():
                logger.error(f"No snapshot data found for run_id={run_id} / snapshot_id={snapshot_id}")
                return False
                
            with open(target_path, "r", encoding="utf-8") as f:
                snap_data = json.load(f)
                
            # Load actual data if available
            parquet_path = Path(f"data/snapshots/{snapshot_id}_issf.parquet")
            import pandas as pd
            if parquet_path.exists():
                df_snap = pd.read_parquet(parquet_path)
                logger.info(f"Loaded {len(df_snap)} rows from {parquet_path.name}")
            else:
                df_snap = None
                logger.warning(f"No parquet found at {parquet_path}")

            snapshot = ISSFSnapshot(
                dataset_id=snap_data.get("dataset_id", "unknown"),
                schema_version=snap_data.get("schema_version", "1.0"),
                data_mode=snap_data.get("data_mode", "batch"),
                source_type=snap_data.get("source_type", "file"),
                source_uri=snap_data.get("source_uri", "local"),
                column_metadata=[], # normally loaded
                row_count=snap_data.get("row_count", 0),
                quality_score=snap_data.get("quality_score", 1.0),
                validation_status=snap_data.get("validation_status", "PASS"),
                data=df_snap,
                error_logs=[],
                extra_meta={}
            )
        
        logger.info("Instantiating PipelineBridge...")
        bridge = PipelineBridge()
        
        # Run bridge
        if snapshot is None:
            logger.error("Failed to acquire snapshot for running pipeline.")
            return False
            
        result = bridge.run(snapshot, target_col=target_col, run_id=run_id)
        
        if result.gate_decision == "FAIL":
            logger.error(f"Pipeline failed at gate for run_id={run_id}")
            return False
            
        logger.info(f"Pipeline completed successfully for run_id={run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed for run_id={run_id}: {e}", exc_info=True)
        return False
