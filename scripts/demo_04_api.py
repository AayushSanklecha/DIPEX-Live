"""
scripts/demo_04_api.py
──────────────────────────────
DIPEX Demo — HTTP API Ingestion

Fetches LIVE cryptocurrency market data from CoinGecko API (free, no key)
and runs it through the full DIPEX pipeline.

Prerequisites:
    Internet connection (no Docker needed for this demo)

Usage:
    python scripts/demo_04_api.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _demo_setup import configure_demo_environment
configure_demo_environment()

from ingestion.universal_intake import UniversalIntake, SourceConfig
from ingestion.readers.api_reader import APISourceConfig


def main():
    print("\n" + "=" * 70)
    print("  DIPEX DEMO - PATH 3: HTTP API Ingestion (Live Internet Data)")
    print("=" * 70)

    api_cfg = APISourceConfig(
        url="https://api.coingecko.com/api/v3/coins/markets",
        method="GET",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 50,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h,7d",
        },
        timeout_s=30.0,
        max_retries=3,
        backoff_base=2.0,
    )

    source = SourceConfig(
        source_type="api",
        dataset_id="crypto_market_data",
        data_mode="batch",
        api_config=api_cfg,
        require_quality_pass=False,
        block_on_schema_break=False,
    )

    print("\n-> Fetching live cryptocurrency data from CoinGecko API...")
    print("  (Top 50 coins by market cap - real-time prices)")
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
        df = snapshot.data
        key_cols = [c for c in ["name", "symbol", "current_price", "market_cap",
                                "price_change_percentage_24h"] if c in df.columns]
        if key_cols:
            print(f"\n  Top 10 Cryptocurrencies (live data):")
            print(df[key_cols].head(10).to_string(index=False))
        print(f"\n  Total columns: {len(df.columns)}")

    print("\n[OK] API ingestion complete! Live internet data processed.\n")
    return snapshot


if __name__ == "__main__":
    main()
