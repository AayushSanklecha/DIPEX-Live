"""
scripts/run_all_sources.py
---------------------------
Downloads real public datasets, then runs the DIPEX pipeline against
every supported data source and prints a pass/fail summary table.

Real datasets used:
  CSV     → Titanic (passenger survival — datasciencedojo GitHub)
  Excel   → Titanic (same data, 2-sheet Excel)
  JSON    → Tips (seaborn restaurant dataset)
  Parquet → NYC Yellow Taxi 2023 (NYC TLC open data)
  SQL     → Titanic → PostgreSQL table
  MongoDB → Tips documents
  Redis   → Titanic passenger hashes
  Neo4j   → Titanic passenger graph
  DuckDB  → Titanic table (in-process)

Usage:
  python scripts/run_all_sources.py                  # file sources only
  python scripts/run_all_sources.py --with-databases  # includes Docker DBs
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

# Force UTF-8 output on Windows so Unicode box chars don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
PY = sys.executable

WIDTH = 70

def banner(msg):
    print("\n" + "=" * WIDTH)
    print(f"  {msg}")
    print("=" * WIDTH)


def run(label, cmd, check_output=None, timeout=180):
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
        elapsed = time.time() - t0
        output  = (proc.stdout + proc.stderr).strip()
        passed  = proc.returncode == 0
        if passed and check_output and check_output not in output:
            passed = False
            detail = f"exit 0 but '{check_output}' not in output"
        else:
            detail = output[-300:] if not passed else (output[:200] if output else "ok")
        return passed, elapsed, detail
    except subprocess.TimeoutExpired:
        return False, timeout, "TIMEOUT"
    except Exception as e:
        return False, time.time() - t0, str(e)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-databases", action="store_true",
                        help="Also seed and run database-backed sources (requires Docker)")
    args = parser.parse_args()

    # ── Step 0: Download real datasets ───────────────────────────────────────
    banner("Step 0 — Downloading real public datasets")
    ok, t, detail = run("download", [PY, "scripts/download_real_datasets.py"],
                        check_output="[OK]")
    if not ok:
        print(f"  ✕ Download failed:\n{detail}")
        sys.exit(1)
    print(detail)

    results = []

    # ── Step 1: CSV (Titanic) → DIPEX intake ────────────────────────────────
    banner("Step 1 — CSV (Titanic) → DIPEX intake pipeline")
    ok, t, detail = run(
        "csv",
        [PY, "main.py", "intake",
         "--format", "csv",
         "--path",   "samples/titanic.csv",
         "--dataset-id", "titanic_csv",
         "--target", "Survived",
         "--allow-fail"],
        check_output="Intake complete"
    )
    results.append(("CSV  (Titanic)", ok, f"{t:.1f}s", detail[-120:]))
    print(f"  {'✓' if ok else '✕'} CSV ({t:.1f}s)")

    # ── Step 2: Excel (Titanic) → DIPEX intake ──────────────────────────────
    banner("Step 2 — Excel (Titanic 2-sheet) → DIPEX intake pipeline")
    ok, t, detail = run(
        "excel",
        [PY, "main.py", "intake",
         "--format", "excel",
         "--path",   "samples/titanic.xlsx",
         "--dataset-id", "titanic_excel",
         "--target", "Survived",
         "--allow-fail"],
        check_output="Intake complete"
    )
    results.append(("Excel (Titanic)", ok, f"{t:.1f}s", detail[-120:]))
    print(f"  {'✓' if ok else '✕'} Excel ({t:.1f}s)")

    # ── Step 3: JSON (Tips) → DIPEX intake ──────────────────────────────────
    banner("Step 3 — JSON (Tips/restaurant data) → DIPEX intake pipeline")
    ok, t, detail = run(
        "json",
        [PY, "main.py", "intake",
         "--format", "json",
         "--path",   "samples/tips.json",
         "--dataset-id", "tips_json",
         "--allow-fail"],
        check_output="Intake complete"
    )
    results.append(("JSON (Tips)", ok, f"{t:.1f}s", detail[-120:]))
    print(f"  {'✓' if ok else '✕'} JSON ({t:.1f}s)")

    # ── Step 4: Parquet (Diamonds) → DIPEX intake ───────────────────────────
    banner("Step 4 -- Parquet (Diamonds dataset) --> DIPEX intake pipeline")
    ok, t, detail = run(
        "parquet",
        [PY, "main.py", "intake",
         "--format", "parquet",
         "--path",   "samples/diamonds.parquet",
         "--dataset-id", "diamonds_parquet",
         "--allow-fail"],
        check_output="Intake complete"
    )
    results.append(("Parquet (Diamonds)", ok, f"{t:.1f}s", detail[-120:]))
    print(f"  {'OK' if ok else 'FAIL'} Parquet ({t:.1f}s)")


    # ── Step 5: Python stats analysis ───────────────────────────────────────
    banner("Step 5 — Python analysis (descriptive stats on Titanic)")
    ok, t, detail = run(
        "stats",
        [PY, "main.py", "stats",
         "--source", "samples/titanic.csv",
         "--target", "Survived",
         "--output", "output/titanic_stats.json"],
    )
    results.append(("Python stats", ok, f"{t:.1f}s", detail[-120:]))
    print(f"  {'✓' if ok else '✕'} Python stats ({t:.1f}s)")

    # ── Step 6: Python preprocess ────────────────────────────────────────────
    banner("Step 6 — Python preprocess (clean + feature engineering)")
    ok, t, detail = run(
        "preprocess",
        [PY, "main.py", "preprocess",
         "--source", "samples/titanic.csv",
         "--target", "Survived",
         "--output", "output/titanic_preprocessed.csv"],
    )
    results.append(("Python preprocess", ok, f"{t:.1f}s", detail[-120:]))
    print(f"  {'✓' if ok else '✕'} Preprocess ({t:.1f}s)")

    # ── Step 7: SQL via DuckDB ───────────────────────────────────────────────
    banner("Step 7 — SQL analysis via DuckDB (Titanic survival by class)")
    ok, t, detail = run(
        "sql",
        [PY, "main.py", "query",
         "--source", "samples/titanic.csv",
         "--sql",
         "SELECT Pclass, Sex, COUNT(*) as passengers, "
         "ROUND(AVG(CAST(Survived AS DOUBLE))*100,1) as survival_pct "
         "FROM df GROUP BY Pclass, Sex ORDER BY Pclass, Sex",
         "--output", "output/titanic_sql_result.csv"],
    )
    results.append(("SQL / DuckDB", ok, f"{t:.1f}s", detail[-120:]))
    print(f"  {'✓' if ok else '✕'} SQL/DuckDB ({t:.1f}s)")

    # ── Step 8: Full 13-stage pipeline run ───────────────────────────────────
    banner("Step 8 — Full 13-stage pipeline run (Titanic CSV)")
    ok, t, detail = run(
        "pipeline",
        [PY, "main.py", "run",
         "--source", "samples/titanic.csv",
         "--target", "Survived"],
        timeout=240
    )
    results.append(("Full Pipeline", ok, f"{t:.1f}s", detail[-120:]))
    print(f"  {'✓' if ok else '✕'} Full pipeline ({t:.1f}s)")

    # ── Step 9: Database sources (optional, needs Docker) ───────────────────
    if args.with_databases:
        for label, script, check in [
            ("DuckDB  (seed)", "samples/seed_duckdb.py",  "✓ DuckDB"),
            ("MongoDB (seed)", "samples/seed_mongodb.py", "✓ MongoDB"),
            ("Redis   (seed)", "samples/seed_redis.py",   "✓ Redis"),
            ("Neo4j   (seed)", "samples/seed_neo4j.py",   "✓ Neo4j"),
        ]:
            banner(f"Database seed — {label}")
            ok, t, detail = run(label, [PY, script], check_output=check, timeout=60)
            results.append((label, ok, f"{t:.1f}s", detail[-120:]))
            print(f"  {'✓' if ok else '✕'} {label} ({t:.1f}s)")

        banner("Database — DIPEX ingest-all (all configured databases)")
        ok, t, detail = run("ingest-all", [PY, "main.py", "ingest-all"], timeout=120)
        results.append(("All Databases", ok, f"{t:.1f}s", detail[-120:]))
        print(f"  {'✓' if ok else '✕'} ingest-all ({t:.1f}s)")

    # ── Final summary ────────────────────────────────────────────────────────
    banner("RESULTS SUMMARY")
    print(f"  {'Source':<26} {'Status':<10} {'Time':<8} Output")
    print("  " + "-" * (WIDTH - 2))
    passed_count = 0
    for label, ok, t, detail in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        note   = (detail or "").split("\n")[-1][:32]
        print(f"  {label:<26} {status:<10} {t:<8} {note}")
        if ok:
            passed_count += 1
    print()
    print(f"  {passed_count}/{len(results)} sources passed")
    print("=" * WIDTH)

    if not args.with_databases:
        print()
        print("  Outputs saved to output/ directory.")
        print()
        print("  To also test MongoDB / Redis / Neo4j / DuckDB:")
        print("    1. double-click  start.bat  (starts Docker stack)")
        print("    2. python scripts/run_all_sources.py --with-databases")

    sys.exit(0 if passed_count == len(results) else 1)


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    main()

