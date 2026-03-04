"""
scripts/verify_all.py
----------------------
One-shot verification script for all DIPEX extensions.

Run this after completing setup to confirm every integration
is correctly installed and activatable.

Usage:
    python scripts/verify_all.py
"""

from __future__ import annotations

import os
import sys

# ── Ensure project root is on Python path ────────────────────────────────────
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Windows ANSI color support ───────────────────────────────────────────────
# Enables PASS/FAIL colors in old cmd.exe and PowerShell on Windows.
# Silently skipped if colorama is not installed (colors still work in
# Windows Terminal, VS Code terminal, and any ANSI-capable shell).
try:
    import colorama
    colorama.just_fix_windows_console()
except ImportError:
    pass

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
SKIP = "\033[93m  SKIP\033[0m"

results = []


def check(name: str, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        results.append(True)
    except Exception as exc:
        print(f"{FAIL}  {name} — {exc}")
        results.append(False)


def check_import(module_path: str):
    parts = module_path.rsplit(".", 1)
    if len(parts) == 2:
        mod, attr = parts
        m = __import__(mod, fromlist=[attr])
        assert getattr(m, attr) is not None
    else:
        __import__(module_path)


print("\n" + "=" * 58)
print("  DIPEX Extension Verification Suite")
print("=" * 58 + "\n")

# ── 1. Library imports ──────────────────────────────────────────
print("[ Libraries ]")
check("duckdb",             lambda: check_import("duckdb"))
check("clickhouse_connect", lambda: check_import("clickhouse_connect"))
check("neo4j",              lambda: check_import("neo4j"))
check("elasticsearch",      lambda: check_import("elasticsearch"))

# ── 2. LLM Provider ────────────────────────────────────────────
print("\n[ LLM Provider ]")
check("HuggingFaceProvider importable",
      lambda: check_import("reporting_service.llm_provider.HuggingFaceProvider"))

check("get_llm_provider factory works",
      lambda: check_import("reporting_service.llm_provider.get_llm_provider"))

def _hf_fallback_produces_summary():
    os.environ.pop("HF_API_KEY", None)  # ensure no key → fallback mode
    from reporting_service.llm_provider import HuggingFaceProvider
    p = HuggingFaceProvider(config={})
    r = p.generate_summary({
        "confidence_score": 0.87,
        "gate_decision": "PASS",
        "metrics": {"accuracy": 0.87},
    })
    assert isinstance(r, str) and len(r) > 0

check("HuggingFaceProvider rule-based fallback (no API key)", _hf_fallback_produces_summary)


# Verify env-var routing to HuggingFaceProvider
def _verify_env_routing():
    os.environ["LLM_PROVIDER"] = "huggingface"
    # Re-import factory in a fresh call to confirm env is read
    from reporting_service.llm_provider import get_llm_provider, HuggingFaceProvider
    provider = get_llm_provider()   # factory reads os.environ["LLM_PROVIDER"]
    assert isinstance(provider, HuggingFaceProvider), (
        f"Expected HuggingFaceProvider, got {type(provider).__name__}"
    )
    # Restore — don't pollute downstream checks
    os.environ.pop("LLM_PROVIDER", None)

check("LLM_PROVIDER=huggingface env var routes correctly", _verify_env_routing)

# ── 3. Connectors ──────────────────────────────────────────────
print("\n[ Connector Factory ]")
from ingestion.connectors import ConnectorFactory
types = ConnectorFactory.supported_types()

for expected in ["duckdb", "clickhouse", "neo4j", "elasticsearch",
                 "elastic", "opensearch", "graph",
                 "sql", "postgresql", "mongodb", "kafka", "api"]:
    check(f"Factory: '{expected}' registered",
          lambda t=expected: (_ for _ in ()).throw(AssertionError(f"'{t}' not registered"))
          if t not in types else None)

# ── 4. DuckDB live round-trip ───────────────────────────────────
print("\n[ DuckDB Live Round-Trip ]")
def _duckdb_roundtrip():
    from ingestion.connectors.duckdb_connector import DuckDBConnector
    # Use a table-configured connector so get_schema() works on same connection
    conn = DuckDBConnector({"duckdb_path": ":memory:", "table": "dipex_test"})
    assert conn.test_connection(), "test_connection() returned False"
    db = conn._get_conn()
    db.execute("CREATE TABLE dipex_test (x INT, y FLOAT, label VARCHAR)")
    db.execute("INSERT INTO dipex_test VALUES (1, 0.5, 'A'), (2, 1.5, 'B'), (3, 2.5, 'A')")
    df = conn.extract("SELECT * FROM dipex_test WHERE label = 'A'")
    assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
    # get_schema uses same in-memory conn — table exists
    schema = conn.get_schema()
    assert "columns" in schema, f"schema missing 'columns': {schema}"
    conn.close()

check("DuckDB in-memory extract + schema", _duckdb_roundtrip)

# ── 5. Kafka Pipeline ──────────────────────────────────────────
print("\n[ Kafka Pipeline Runner ]")
check("KafkaPipelineRunner importable",
      lambda: check_import("ingestion.kafka_pipeline.KafkaPipelineRunner"))

def _kafka_process_one_batch():
    import pandas as pd
    from ingestion.kafka_pipeline import KafkaPipelineRunner
    from unittest.mock import patch
    runner = KafkaPipelineRunner({})
    df = pd.DataFrame({"id": [1, 2], "value": [10.0, 20.0]})
    with patch.object(runner, "_run_pipeline",
                      return_value={"gate_decision": "PASS", "stages": []}):
        result = runner.process_one_batch(df, dataset_id="verify_batch")
    assert result["gate_decision"] == "PASS"

check("process_one_batch() routes correctly", _kafka_process_one_batch)

# ── 6. audit/ DLQ directory ────────────────────────────────────
print("\n[ Infrastructure ]")
check("audit/ directory exists for DLQ",
      lambda: (None if os.path.isdir(os.path.join(_ROOT, "audit")) else (_ for _ in ()).throw(
          FileNotFoundError("audit/ directory missing — run mkdir audit in project root"))))

check(".env.example exists",
      lambda: (None if os.path.isfile(os.path.join(_ROOT, ".env.example")) else (_ for _ in ()).throw(
          FileNotFoundError(".env.example missing"))))

check("scripts/start_kafka_pipeline.py exists",
      lambda: (None if os.path.isfile(os.path.join(_ROOT, "scripts", "start_kafka_pipeline.py"))
               else (_ for _ in ()).throw(FileNotFoundError("startup script missing"))))

# ── Summary ────────────────────────────────────────────────────
total  = len(results)
passed = sum(results)
failed = total - passed

print("\n" + "=" * 58)
print(f"  Result: {passed}/{total} checks passed"
      + (" [ ALL OK ]" if failed == 0 else f"  [ {failed} FAILED ]"))
print("=" * 58 + "\n")

sys.exit(0 if failed == 0 else 1)
