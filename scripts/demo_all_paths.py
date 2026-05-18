"""
scripts/demo_all_paths.py
──────────────────────────────
DIPEX Demo — All Ingestion Paths

Master script that runs all 4 demo paths sequentially and prints
a unified comparison table. Perfect for judge presentations.

Prerequisites:
    docker-compose -f docker-compose.demo.yml up -d
    (wait ~15s for all services to be healthy)
    Internet connection (for API demo)

Usage:
    python scripts/demo_all_paths.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _demo_setup import configure_demo_environment
configure_demo_environment()


def run_safe(name: str, fn):
    """Run a demo function safely, catching errors."""
    try:
        t0 = time.perf_counter()
        snapshot = fn()
        elapsed = time.perf_counter() - t0
        return {
            "name": name,
            "status": "PASS",
            "rows": snapshot.row_count if snapshot else 0,
            "quality": f"{snapshot.quality_score:.0%}" if snapshot else "N/A",
            "schema": snapshot.schema_version if snapshot else "N/A",
            "time_s": f"{elapsed:.1f}s",
        }
    except Exception as exc:
        return {
            "name": name,
            "status": f"FAIL: {str(exc)[:40]}",
            "rows": 0,
            "quality": "N/A",
            "schema": "N/A",
            "time_s": "N/A",
        }


def main():
    print("\n")
    print("=" * 72)
    print("    DIPEX - Universal Data Intake Demo (All Paths)")
    print("")
    print("    Automating the workflow of a Data Analyst")
    print("    ANY source -> Normalised -> Quality-checked -> Pipeline-ready")
    print("=" * 72)
    print()

    results = []

    # ── PATH 1a: PostgreSQL ───────────────────────────────────────────
    print("[1/4] Running PostgreSQL ingestion...")
    from demo_01_postgres import main as pg_main
    results.append(run_safe("PostgreSQL (E-Commerce Sales)", pg_main))

    # ── PATH 1b: MongoDB ─────────────────────────────────────────────
    print("[2/4] Running MongoDB ingestion...")
    from demo_02_mongodb import main as mongo_main
    results.append(run_safe("MongoDB (Product Catalog)", mongo_main))

    # ── PATH 2: Kafka ────────────────────────────────────────────────
    print("[3/4] Running Kafka stream ingestion...")
    from demo_03_kafka import main as kafka_main
    results.append(run_safe("Kafka (IoT Sensor Stream)", kafka_main))

    # ── PATH 3: HTTP API ─────────────────────────────────────────────
    print("[4/4] Running HTTP API ingestion...")
    from demo_04_api import main as api_main
    results.append(run_safe("HTTP API (Crypto Market)", api_main))

    # ── Final Comparison Table ────────────────────────────────────────
    print("\n\n")
    print("=" * 80)
    print("  FINAL STATUS TABLE")
    print("=" * 80)
    print(f"  {'Source':<32} {'Status':<8} {'Rows':>5} {'Quality':>8} {'Schema':>7} {'Time':>7}")
    print("-" * 80)

    for r in results:
        name = r["name"][:31].ljust(31)
        st = r["status"][:7].ljust(7)
        rows = str(r["rows"]).rjust(5)
        qual = str(r["quality"]).rjust(8)
        sch = str(r["schema"]).rjust(7)
        tm = str(r["time_s"]).rjust(7)
        marker = "[OK]" if r["status"] == "PASS" else "[!!]"
        print(f"  {marker} {name} {st} {rows} {qual} {sch} {tm}")

    print("=" * 80)

    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    print(f"\n  Result: {passed}/{total} ingestion paths completed successfully.")
    print(f"  Pipeline: Read -> Bronze -> Normalise -> Schema -> Quality -> ISSF")
    print(f"  Zero manual intervention - fully automated data analyst workflow.\n")


if __name__ == "__main__":
    main()
