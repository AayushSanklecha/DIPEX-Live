"""
tests/fixtures/canary/generate_canary_fixtures.py
---------------------------------------------------
Generates reproducible canary CSV fixtures for regression and integration testing.

Run once to (re)generate all fixtures:
    python tests/fixtures/canary/generate_canary_fixtures.py

Each fixture targets a specific real-world data archetype that is known to
expose subtle pipeline bugs:

  01_financial_ohlcv.csv      — Financial OHLCV (Open/High/Low/Close/Volume)
                                 with volume spikes and weekend gaps
  02_healthcare_patient.csv   — Patient records with mixed nulls, age/BMI outliers,
                                 categorical diagnosis codes
  03_timeseries_iot.csv       — IoT sensor readings with periodic drift and
                                 occasional NaN bursts
  04_all_null_columns.csv     — Pathological: 3 fully-null columns + 1 valid ID
  05_single_row.csv           — Minimal: only 1 data row (breaks most statistics)
  06_zero_variance.csv        — All rows identical (std=0 across all columns)
  07_high_cardinality.csv     — 10 000 unique category values (label-encoding stress)
  08_wide_sparse.csv          — 200 columns, 50 rows, 40 % sparsity (null fields)
  09_mixed_encodings.csv      — UTF-8 accents, CJK chars, emojis in string columns
  10_inf_nan_mix.csv          — Intentional inf/-inf/NaN sprinkled at known positions
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ── Reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)
OUT = os.path.join(os.path.dirname(__file__))
N   = 500   # default rows


def _save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    n_bytes = os.path.getsize(path)
    print(f"  [OK]  {name:45s}  {len(df):>6} rows × {len(df.columns):>3} cols  ({n_bytes/1024:.1f} KB)")


# ── Fixture 1: Financial OHLCV ─────────────────────────────────────────────────
def financial_ohlcv() -> None:
    dates = pd.date_range("2020-01-01", periods=N, freq="B")   # business days only
    close = 100.0 + np.cumsum(RNG.normal(0, 1.5, N))
    opens = close + RNG.normal(0, 0.3, N)
    highs = np.maximum(opens, close) + RNG.uniform(0, 2, N)
    lows  = np.minimum(opens, close) - RNG.uniform(0, 2, N)
    vol   = RNG.integers(100_000, 5_000_000, N).astype(float)
    # Inject 5 volume spikes (common in real financial data)
    spike_idx = RNG.choice(N, 5, replace=False)
    vol[spike_idx] *= 20
    df = pd.DataFrame({
        "date":   dates,
        "open":   np.round(opens, 4),
        "high":   np.round(highs, 4),
        "low":    np.round(lows, 4),
        "close":  np.round(close, 4),
        "volume": vol,
        "ticker": "DIPEX",
    })
    _save(df, "01_financial_ohlcv.csv")


# ── Fixture 2: Healthcare patient records ──────────────────────────────────────
def healthcare_patient() -> None:
    diagnoses = ["T2DM", "HTN", "CKD", "COPD", "CHF", "Asthma", "Obesity", None]
    df = pd.DataFrame({
        "patient_id":  [f"P{i:05d}" for i in range(N)],
        "age":         RNG.integers(18, 95, N),
        "bmi":         np.round(RNG.uniform(16.0, 55.0, N), 1),
        "systolic_bp": RNG.integers(90, 190, N),
        "glucose":     np.round(RNG.uniform(60, 400, N), 1),
        "diagnosis":   RNG.choice(diagnoses, N),
        "readmitted":  RNG.choice([0, 1], N, p=[0.75, 0.25]),
        "days_stay":   RNG.integers(1, 30, N),
    })
    # Introduce realistic nulls: ~12 % missing glucose, ~8 % missing BMI
    null_gluc = RNG.choice(N, int(N * 0.12), replace=False)
    null_bmi  = RNG.choice(N, int(N * 0.08), replace=False)
    df.loc[null_gluc, "glucose"] = None
    df.loc[null_bmi,  "bmi"]     = None
    _save(df, "02_healthcare_patient.csv")


# ── Fixture 3: IoT time-series with drift + NaN bursts ────────────────────────
def timeseries_iot() -> None:
    timestamps = pd.date_range("2023-01-01", periods=N, freq="5min")
    temp  = 22.0 + 3 * np.sin(np.linspace(0, 4 * np.pi, N)) + RNG.normal(0, 0.5, N)
    humid = 55.0 + 10 * np.cos(np.linspace(0, 2 * np.pi, N)) + RNG.normal(0, 1, N)
    press = 1013.0 + RNG.normal(0, 2, N)
    # Introduce sensor outage (NaN burst at rows 100–115)
    temp[100:116]  = np.nan
    humid[200:206] = np.nan
    df = pd.DataFrame({
        "timestamp":     timestamps,
        "temperature_c": np.round(temp, 2),
        "humidity_pct":  np.round(np.clip(humid, 0, 100), 2),
        "pressure_hpa":  np.round(press, 2),
        "sensor_id":     RNG.choice(["S01", "S02", "S03"], N),
        "battery_pct":   RNG.integers(10, 100, N),
    })
    _save(df, "03_timeseries_iot.csv")


# ── Fixture 4: Pathological — all-null columns ────────────────────────────────
def all_null_columns() -> None:
    df = pd.DataFrame({
        "id":          range(50),
        "null_col_a":  [None] * 50,
        "null_col_b":  [None] * 50,
        "null_col_c":  [None] * 50,
    })
    _save(df, "04_all_null_columns.csv")


# ── Fixture 5: Pathological — single row ─────────────────────────────────────
def single_row() -> None:
    df = pd.DataFrame({
        "id":    [1],
        "value": [99.9],
        "label": ["target"],
    })
    _save(df, "05_single_row.csv")


# ── Fixture 6: Zero variance ─────────────────────────────────────────────────
def zero_variance() -> None:
    df = pd.DataFrame({
        "measurement": [42.0] * 200,
        "category":    ["CONST"] * 200,
        "flag":        [1] * 200,
    })
    _save(df, "06_zero_variance.csv")


# ── Fixture 7: High-cardinality categorical ───────────────────────────────────
def high_cardinality() -> None:
    n = 10_000
    df = pd.DataFrame({
        "id":       range(n),
        "category": [f"CAT_{i:06d}" for i in range(n)],
        "value":    RNG.standard_normal(n).round(4),
        "weight":   RNG.uniform(0, 1, n).round(6),
    })
    _save(df, "07_high_cardinality.csv")


# ── Fixture 8: Wide-sparse matrix ────────────────────────────────────────────
def wide_sparse() -> None:
    rows, cols = 50, 200
    data = RNG.standard_normal((rows, cols))
    # Introduce 40 % nulls
    mask = RNG.random((rows, cols)) < 0.4
    data_with_nulls = data.astype(object)
    data_with_nulls[mask] = None
    df = pd.DataFrame(
        data_with_nulls,
        columns=[f"feature_{i:03d}" for i in range(cols)],
    )
    df.insert(0, "sample_id", range(rows))
    _save(df, "08_wide_sparse.csv")


# ── Fixture 9: Mixed encodings (UTF-8, CJK, emoji) ─────────────────────────
def mixed_encodings() -> None:
    names = [
        "Müller", "García", "中村", "김민준", "Σωκράτης",
        "مريم", "Björk", "Ñoño", "café", "naïve",
        "résumé", "日本語", "한국어", "🔬 Lab", "Test\n'quoted'",
    ]
    df = pd.DataFrame({
        "id":    range(N),
        "name":  [names[i % len(names)] for i in range(N)],
        "score": RNG.uniform(0, 100, N).round(2),
        "tag":   RNG.choice(["α", "β", "γ", "δ", None], N),
    })
    _save(df, "09_mixed_encodings.csv")


# ── Fixture 10: inf/NaN mix ──────────────────────────────────────────────────
def inf_nan_mix() -> None:
    values = list(RNG.standard_normal(N - 6)) + [
        float("inf"), float("-inf"), float("nan"),
        float("inf"), float("nan"), float("-inf"),
    ]
    RNG.shuffle(values)
    df = pd.DataFrame({
        "id":     range(N),
        "value":  values,
        "ratio":  [float("inf") if i % 50 == 0 else v
                   for i, v in enumerate(RNG.uniform(0, 1, N))],
        "label":  RNG.choice(["pos", "neg", None], N),
    })
    _save(df, "10_inf_nan_mix.csv")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGenerating canary fixtures → {OUT}\n")
    financial_ohlcv()
    healthcare_patient()
    timeseries_iot()
    all_null_columns()
    single_row()
    zero_variance()
    high_cardinality()
    wide_sparse()
    mixed_encodings()
    inf_nan_mix()
    print("\nDone. All 10 canary fixtures written.\n")
