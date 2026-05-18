# main.py — MUST BE FIRST EXECUTABLE LINE (Issue 06)
import sys

_REQUIRED = (3, 12)   # Sprint 3: updated from (3, 11) — Python 3.12.7
if sys.version_info < _REQUIRED:
    raise RuntimeError(
        f"DIPEX v3 requires Python {_REQUIRED[0]}.{_REQUIRED[1]}+. "
        f"You are running {sys.version}. "
        f"Upgrade Python and rebuild your virtual environment."
    )

"""
main.py
-------
Entry point for the DIPEX Enterprise Analytics Platform v3.0.0

Sub-commands:
  serve        — Start the FastAPI REST API server
  run          — Run the full pipeline end-to-end
  preprocess   — Preprocess a CSV file (clean + feature engineer)
  stats        — Run statistical analysis on a CSV file
  report       — Generate an executive HTML report for a run_id
  query        — Execute a SQL query against a CSV file via DuckDB
  ingest-all   — Pull from all configured databases (Mongo, Redis, Postgres...)
"""

import argparse
import logging
import os
import yaml

# Issue 02: NumPy compatibility check — BEFORE any DIPEX module imports
from utils.numpy_compat import check_numpy_compatibility
check_numpy_compatibility()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("dipex.main")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DIPEX Enterprise Analytics Platform v3.0.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py serve --port 8000 --reload
  python main.py run --source data.csv --target churn
  python main.py preprocess --source data.csv --target churn --output cleaned.csv
  python main.py stats --source data.csv --target churn
  python main.py query --source data.csv --sql "SELECT COUNT(*) FROM df"
  python main.py report --run-id <run_id>
  python main.py ingest-all
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # ── serve ──────────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI REST API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    serve_parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")

    # ── run ────────────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run the full pipeline end-to-end")
    run_parser.add_argument("--source", type=str, required=True, help="Path to data file")
    run_parser.add_argument("--type", choices=["csv", "excel", "json"], default="csv")
    run_parser.add_argument("--target", type=str, default="target", help="Target column name")

    # ── preprocess ─────────────────────────────────────────────────────────────
    pp_parser = subparsers.add_parser("preprocess", help="Preprocess a CSV file")
    pp_parser.add_argument("--source", type=str, required=True, help="Input CSV path")
    pp_parser.add_argument("--target", type=str, default="", help="Target column (for FE)")
    pp_parser.add_argument("--output", type=str, default="", help="Output CSV path")

    # ── stats ──────────────────────────────────────────────────────────────────
    stats_parser = subparsers.add_parser("stats", help="Run statistical analysis on a CSV")
    stats_parser.add_argument("--source", type=str, required=True)
    stats_parser.add_argument("--target", type=str, default="", help="Target column for correlation")
    stats_parser.add_argument("--output", type=str, default="", help="Save JSON report to file")

    # ── query ──────────────────────────────────────────────────────────────────
    query_parser = subparsers.add_parser("query", help="Execute SQL on a CSV via DuckDB")
    query_parser.add_argument("--source", type=str, required=True)
    query_parser.add_argument("--sql", type=str, required=True, help="SQL query (use 'df' as table name)")
    query_parser.add_argument("--output", type=str, default="", help="Save result CSV to file")

    # ── report ─────────────────────────────────────────────────────────────────
    rpt_parser = subparsers.add_parser("report", help="Generate executive HTML report for a run")
    rpt_parser.add_argument("--run-id", type=str, required=True, help="Pipeline run ID")
    rpt_parser.add_argument("--output", type=str, default="", help="Custom output HTML path")

    # ── ingest-all ─────────────────────────────────────────────────────────────
    subparsers.add_parser("ingest-all", help="Pull data from all databases into DIPEX")


    # ── intake (UDIL v2) ───────────────────────────────────────────────────────
    intake_parser = subparsers.add_parser(
        "intake",
        help="Ingest any data source → ISSF snapshot → full downstream pipeline",
    )
    intake_parser.add_argument(
        "--source-type", choices=["file", "api", "database", "stream"], default="file",
        help="Data source type",
    )
    intake_parser.add_argument("--path", type=str, default="", help="File path (for file sources)")
    intake_parser.add_argument("--format", type=str, default=None,
                               help="File format hint (csv/excel/json/xml/parquet/avro/feather/log)")
    intake_parser.add_argument("--dataset-id", type=str, default="", help="Dataset identifier")
    intake_parser.add_argument("--target", type=str, default="", help="Target column for ML (optional)")
    intake_parser.add_argument("--mode", choices=["batch", "live", "stream"], default="batch",
                               help="Data mode")
    intake_parser.add_argument("--no-bridge", action="store_true",
                               help="Skip downstream pipeline bridge (ingest only)")
    intake_parser.add_argument("--allow-fail", action="store_true",
                               help="Don't exit 1 on quality/schema failure")

    # ── batch-ingest ───────────────────────────────────────────────────────────
    batch_parser = subparsers.add_parser(
        "batch-ingest",
        help="Run batch ingestion job across multiple sources from config",
    )
    batch_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    batch_parser.add_argument(
        "--batch-mode", choices=["full_refresh", "incremental", "partition"],
        default="incremental", help="Batch processing mode",
    )
    batch_parser.add_argument("--sources", type=str, default="",
                              help="Comma-separated file paths to ingest (for quick multi-file batch)")
    batch_parser.add_argument("--target", type=str, default="", help="Target column for ML")
    batch_parser.add_argument("--archive", action="store_true",
                              help="Archive snapshots older than configured days")

    # ── analyst ────────────────────────────────────────────────────────────────
    analyst_parser = subparsers.add_parser(
        "analyst",
        help="Run junior/senior analyst operations on Gold layer copies",
    )
    analyst_parser.add_argument("--dataset-id", type=str, required=True,
                                help="Source dataset identifier")
    analyst_parser.add_argument("--snapshot-id", type=str, required=True,
                                help="ISSF snapshot ID to derive Gold layer from")
    analyst_parser.add_argument(
        "--level", choices=["junior", "senior"], default="junior",
        help="Analyst level — junior (cleanup/agg/filter) | senior (stats/ML/risk)",
    )
    analyst_parser.add_argument(
        "--operation",
        choices=[
            "basic_cleaning", "remove_duplicates", "simple_aggregation",
            "filter_rows", "kpi_tracking", "percent_change", "sql_query",
            "manual_threshold_check",
            "feature_engineering", "risk_scoring", "windowed_aggregation",
            "executive_summary",
        ],
        default="basic_cleaning",
        help="Operation to run on Gold copy",
    )
    analyst_parser.add_argument("--target", type=str, default="",
                                help="Target column (for senior ops)")
    analyst_parser.add_argument("--group-by", type=str, default="",
                                help="Comma-separated column names for groupby ops")
    analyst_parser.add_argument("--query", type=str, default="",
                                help="Pandas query string for filter_rows")
    analyst_parser.add_argument("--output", type=str, default="output/analyst",
                                help="Output directory for Gold artefact export")

    # ── layer-verify ───────────────────────────────────────────────────────────
    lv_parser = subparsers.add_parser(
        "layer-verify",
        help="Verify checksum integrity of a Bronze or Silver layer artefact",
    )
    lv_parser.add_argument("--dataset-id", type=str, required=True)
    lv_parser.add_argument("--snapshot-id", type=str, required=True)
    lv_parser.add_argument("--layer", choices=["bronze", "silver", "gold"], default="silver")


    args = parser.parse_args()

    # ── Dispatch ───────────────────────────────────────────────────────────────
    if args.command == "serve":

        import uvicorn
        config = load_config()
        # Ensure runtime directories exist (local dev — Docker creates them at build)
        for _d in ("data/uploads", "data/snapshots", "data/model_registry",
                   "data/approved_outputs", "audit", "reports"):
            os.makedirs(_d, exist_ok=True)
        project_name = config.get("project", {}).get("name", "DIPEX")
        logger.info("Starting %s on %s:%d …", project_name, args.host, args.port)
        uvicorn.run(
            "api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level="info",
        )

    elif args.command == "run":
        import uuid
        from simple_pipeline import orchestrate_pipeline
        run_id = str(uuid.uuid4())
        # Copy source file into expected location
        if args.source and os.path.exists(args.source):
            os.makedirs("data/uploads", exist_ok=True)
            import shutil
            dest = f"data/uploads/{run_id}_sample.csv"
            shutil.copy(args.source, dest)
            logger.info("Source copied to %s", dest)
        logger.info("CLI pipeline run — source=%s  run_id=%s", args.source, run_id)
        success = orchestrate_pipeline(run_id, target_col=args.target, source_path=args.source)
        sys.exit(0 if success else 1)

    elif args.command == "preprocess":
        import pandas as pd
        from preprocessing.cleaner import DataCleaner
        from preprocessing.feature_engineer import FeatureEngineer
        config = load_config()
        df = pd.read_csv(args.source)
        logger.info("Preprocessing %s (%d rows × %d cols)", args.source, len(df), len(df.columns))
        cleaner = DataCleaner.from_config(config)
        df_clean, report = cleaner.clean(df)
        logger.info("Cleaning: %d→%d rows, %d dupes removed", report.rows_before, report.rows_after, report.duplicates_removed)
        fe = FeatureEngineer.from_config(config)
        df_final, fe_report = fe.engineer(df_clean, target_col=args.target or None)
        logger.info("Features added: %s", fe_report.features_added[:10])
        output = args.output or args.source.replace(".csv", "_preprocessed.csv")
        df_final.to_csv(output, index=False)
        logger.info("Preprocessed data saved to: %s", output)

    elif args.command == "stats":
        import json
        import pandas as pd
        from stats.descriptive import DescriptiveStats
        from stats.correlation import CorrelationAnalyzer
        config = load_config()
        df = pd.read_csv(args.source)
        logger.info("Running stats on %s", args.source)
        ds = DescriptiveStats()
        desc = ds.analyze(df)
        ca = CorrelationAnalyzer()
        corr = ca.analyze(df, target=args.target or None)
        full = {"descriptive": desc, "correlation": corr}
        if args.output:
            with open(args.output, "w") as f:
                json.dump(full, f, indent=2, default=str)
            logger.info("Stats report saved to %s", args.output)
        else:
            import pprint
            pprint.pprint(desc.get("summary", {}))

    elif args.command == "ingest-all":
        import scripts.ingest_all_databases as ia
        ia.run_aggregation()

    elif args.command == "query":
        import pandas as pd
        from query_engine.sql_engine import SQLEngine
        df = pd.read_csv(args.source)
        engine = SQLEngine()
        engine.register("df", df)
        result = engine.execute(args.sql)
        if result.success:
            result_df = pd.DataFrame(result.data, columns=result.columns)
            logger.info("Query returned %d rows in %.1fms", result.rows, result.elapsed_ms)
            if args.output:
                result_df.to_csv(args.output, index=False)
                logger.info("Results saved to %s", args.output)
            else:
                print(result_df.to_string(index=False))
        else:
            logger.error("Query failed: %s", result.error)
            sys.exit(1)

    elif args.command == "report":
        import json
        from reporting_service.executive_report import ExecutiveReportGenerator
        config = load_config()
        run_id = args.run_id
        reporter = ExecutiveReportGenerator.from_config(config)
        # Try to load run data from audit log
        audit_log = "audit/audit.jsonl"
        run_data = {}
        if os.path.exists(audit_log):
            with open(audit_log) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("data", {}).get("run_id") == run_id:
                            run_data.update(entry.get("data", {}))
                    except Exception:
                        pass
        report_path = reporter.generate(
            run_id=run_id,
            confidence_vector=run_data.get("confidence_vector", {"confidence_score": 0}),
            gate1_decision=run_data.get("gate_decision", "UNKNOWN"),
            gate2_decision=run_data.get("hard_gate_2_decision", "UNKNOWN"),
            narrative=run_data.get("narrative", "No narrative available."),
            analyst_flags=[],
            model_metrics=run_data.get("model_type", {}),
            risk_flags=[],
            fingerprint=run_data.get("fingerprint", ""),
            schema_version="1.0",
            row_count=0, col_count=0, flag_count=0, retry_count=0,
            gov_decision="N/A",
        )
        if report_path:
            logger.info("Executive report generated: %s", report_path)
            if args.output:
                import shutil
                shutil.copy(report_path, args.output)
        else:
            logger.error("Report generation failed.")
            sys.exit(1)

    elif args.command == "intake":
        import json as _json
        config = load_config()
        dataset_id = args.dataset_id or (
            os.path.splitext(os.path.basename(args.path))[0] if args.path else "intake"
        )
        from ingestion.universal_intake import UniversalIntake, SourceConfig
        intake = UniversalIntake.from_yaml("config.yaml")
        cfg = SourceConfig(
            source_type=args.source_type,
            dataset_id=dataset_id,
            data_mode=args.mode,
            path=args.path or None,
            file_format=getattr(args, "format", None),
        )
        try:
            snapshot = intake.ingest(cfg)
            logger.info(
                "Intake complete — dataset=%s rows=%d quality=%.2f status=%s schema=v%s",
                snapshot.dataset_id, snapshot.row_count, snapshot.quality_score,
                snapshot.validation_status, snapshot.schema_version,
            )
            if not args.no_bridge and snapshot.data is not None:
                from ingestion.pipeline_bridge import PipelineBridge
                bridge = PipelineBridge(config=config)
                bridge_result = bridge.run(
                    snapshot,
                    target_col=args.target or None,
                )
                logger.info(
                    "Pipeline bridge complete — gate=%s report=%s",
                    bridge_result.gate_decision, bridge_result.report_path or "N/A",
                )
                if bridge_result.gate_decision == "FAIL" and not args.allow_fail:
                    sys.exit(1)
        except Exception as exc:
            logger.exception("Intake failed: %s", exc)
            if not args.allow_fail:
                sys.exit(1)

    elif args.command == "batch-ingest":
        import json as _json
        config = load_config()
        from ingestion.universal_intake import SourceConfig
        from ingestion.batch_processor import BatchProcessor, BatchConfig
        src_paths = [p.strip() for p in args.sources.split(",") if p.strip()]
        if not src_paths:
            logger.error("--sources must be a non-empty comma-separated list of file paths")
            sys.exit(1)
        source_cfgs = [
            SourceConfig(
                source_type="file",
                dataset_id=os.path.splitext(os.path.basename(p))[0],
                data_mode="batch",
                path=p,
            )
            for p in src_paths
        ]
        batch_cfg = BatchConfig(
            mode=args.batch_mode,
            source_configs=source_cfgs,
            target_col=args.target or None,
            archive_after_days=30 if args.archive else 0,
            run_pipeline_bridge=True,
        )
        processor = BatchProcessor(config=config)
        result = processor.run(batch_cfg)
        logger.info(
            "Batch ingest complete — %d/%d sources succeeded, %d total rows",
            result.succeeded, result.total_sources, result.total_rows_ingested,
        )
        if result.failed > 0:
            logger.warning("%d sources failed: %s", result.failed, result.errors[:3])
            sys.exit(1)

    elif args.command == "analyst":
        config = load_config()
        import json as _json
        from ingestion.data_layers import ImmutableDataFrame, LayerManager
        from ingestion.universal_intake import UniversalIntake, SourceConfig
        from analyst.junior_analyst import JuniorAnalyst
        from analyst.senior_analyst import SeniorAnalyst

        # Load Silver snapshot data
        intake = UniversalIntake(config=config)
        lm = LayerManager()
        snap_path = os.path.join("data/snapshots", f"{args.snapshot_id}_issf.json")
        if not os.path.exists(snap_path):
            logger.error("Snapshot not found: %s", snap_path)
            sys.exit(1)
        with open(snap_path, encoding="utf-8") as f:
            snap_meta = _json.load(f)
        # Read the Silver parquet if available, else re-ingest
        parquet_path = snap_path.replace("_issf.json", "_issf.parquet")
        if os.path.exists(parquet_path):
            import pandas as pd
            df = pd.read_parquet(parquet_path)
        else:
            logger.warning("Silver parquet not found — re-reading from snapshot JSON only (summary mode)")
            sys.exit(1)

        silver = ImmutableDataFrame(df, layer="silver", dataset_id=args.dataset_id)
        jr = JuniorAnalyst(layer_manager=lm, operator="cli")
        sr = SeniorAnalyst(layer_manager=lm, operator="cli")
        op  = args.operation
        sid = args.snapshot_id

        try:
            if op == "basic_cleaning":
                gold = jr.basic_cleaning(silver, source_snapshot_id=sid)
            elif op == "remove_duplicates":
                gold = jr.remove_duplicates(silver, source_snapshot_id=sid)
            elif op == "filter_rows":
                gold = jr.filter_rows(silver, query=args.query or "index >= 0", source_snapshot_id=sid)
            elif op == "simple_aggregation":
                cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
                gold = jr.simple_aggregation(silver, group_by=cols or [silver.columns[0]],
                                              agg_map={}, source_snapshot_id=sid)
            elif op == "kpi_tracking":
                gold = jr.kpi_tracking(silver, kpi_definitions={}, source_snapshot_id=sid)
            elif op == "percent_change":
                gold = jr.percent_change(silver, col=args.target or silver.columns[0],
                                          source_snapshot_id=sid)
            elif op == "sql_query":
                gold = jr.sql_query(silver, sql="SELECT * FROM data", source_snapshot_id=sid)
            elif op == "manual_threshold_check":
                gold = jr.manual_threshold_check(silver, thresholds={}, source_snapshot_id=sid)
            elif op == "feature_engineering":
                num_cols = df.select_dtypes("number").columns.tolist()
                gold = sr.feature_engineering(silver, numeric_cols=num_cols, source_snapshot_id=sid)
            elif op == "risk_scoring":
                signals = {c: 1.0 / max(1, len(df.select_dtypes("number").columns))
                           for c in df.select_dtypes("number").columns}
                gold = sr.risk_scoring(silver, risk_signals=signals, source_snapshot_id=sid)
            elif op == "windowed_aggregation":
                col = args.target or df.select_dtypes("number").columns[0]
                gold = sr.windowed_aggregation(silver, value_col=col, source_snapshot_id=sid)
            elif op == "executive_summary":
                gold = sr.executive_summary(silver, target_col=args.target or None,
                                             source_snapshot_id=sid)
            else:
                logger.error("Unknown operation: %s", op)
                sys.exit(1)

            os.makedirs(args.output, exist_ok=True)
            out_path = os.path.join(args.output, f"{gold.dataset_id}.csv")
            jr.data_export(gold, output_path=out_path, fmt="csv")
            logger.info(
                "Analyst '%s' (%s) completed | Gold artefact: %s | lineage: %s",
                op, args.level, out_path, gold.lineage.lineage_id,
            )
        except Exception as exc:
            logger.error("Analyst operation failed: %s", exc)
            sys.exit(1)

    elif args.command == "layer-verify":
        from ingestion.data_layers import LayerManager
        from ingestion.immutability_guard import ChecksumMismatchError
        config = load_config()
        lm = LayerManager()
        try:
            report = lm.verify_layer(
                dataset_id=args.dataset_id,
                layer=args.layer,
                snapshot_id=args.snapshot_id,
            )
            if report.get("verified"):
                logger.info(
                    "✅ Layer VERIFIED: %s/%s/%s | checksum=%s… | shape=%s",
                    args.layer, args.dataset_id, args.snapshot_id,
                    report["checksum"][:24], report.get("shape"),
                )
            else:
                logger.error("❌ Layer UNVERIFIED: %s", report.get("error"))
                sys.exit(1)
        except ChecksumMismatchError as exc:
            logger.error("❌ CHECKSUM MISMATCH — possible tampering: %s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.error("Layer verify failed: %s", exc)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(0)



if __name__ == "__main__":
    main()
