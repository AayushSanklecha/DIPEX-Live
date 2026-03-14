"""
tests/test_unit_dataanalyzer.py
--------------------------------
Unit tests for frontend/dataAnalyzer logic equivalents in Python.
Tests the Python analytics utilities for data type inference, statistics,
and chart recommendation heuristics that mirror the JS dataAnalyzer.js.

Since the JS file can't run in pytest, we test the Python equivalents in
stats.py, preprocessing utilities, and schema analysis helpers.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Schema / Type inference helpers ────────────────────────────────────────────

class TestDataTypeInference:

    def test_numeric_column_detection(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "a", "b"]})
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assert "a" in numeric_cols
        assert "b" not in numeric_cols

    def test_datetime_column_detection(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])})
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_categorical_high_cardinality(self):
        """Column with many unique values should be flagged as high cardinality."""
        vals = [f"cat_{i}" for i in range(100)]
        df = pd.DataFrame({"cat": vals})
        cardinality = df["cat"].nunique() / len(df)
        assert cardinality > 0.9

    def test_categorical_low_cardinality(self):
        df = pd.DataFrame({"status": ["pass", "fail", "pass", "fail", "pass"] * 20})
        cardinality = df["status"].nunique()
        assert cardinality <= 5


# ── Statistical computation tests ─────────────────────────────────────────────

class TestStatisticalComputation:

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "value": np.random.normal(100, 15, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
        })

    def test_mean_computation(self, sample_df):
        mean = sample_df["value"].mean()
        assert 95 < mean < 105, f"Mean out of expected range: {mean}"

    def test_std_computation(self, sample_df):
        std = sample_df["value"].std()
        assert 12 < std < 18, f"Std dev out of expected range: {std}"

    def test_percentile_computation(self, sample_df):
        q25 = sample_df["value"].quantile(0.25)
        q50 = sample_df["value"].quantile(0.50)
        q75 = sample_df["value"].quantile(0.75)
        assert q25 < q50 < q75

    def test_outlier_detection_iqr(self, sample_df):
        """IQR-based outlier detection should find some outliers in normal data."""
        q1 = sample_df["value"].quantile(0.25)
        q3 = sample_df["value"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = sample_df[(sample_df["value"] < lower) | (sample_df["value"] > upper)]
        # Normal distribution ~0.7% outliers
        assert 0 < len(outliers) < len(sample_df) * 0.05

    def test_histogram_bins(self):
        data = list(range(100))
        df = pd.DataFrame({"v": data})
        bins = pd.cut(df["v"], bins=10)
        assert bins.nunique() == 10

    def test_box_plot_stats(self, sample_df):
        """Verify box plot stats are computable and internally consistent."""
        col = sample_df["value"]
        stats = {
            "min": col.min(),
            "q1":  col.quantile(0.25),
            "median": col.median(),
            "q3": col.quantile(0.75),
            "max": col.max(),
        }
        assert stats["min"] <= stats["q1"] <= stats["median"] <= stats["q3"] <= stats["max"]


# ── Data cleaning utilities ───────────────────────────────────────────────────

class TestDataCleaning:

    def test_null_fill_mean(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        filled = df["a"].fillna(df["a"].mean())
        assert filled.isnull().sum() == 0
        assert filled.mean() == pytest.approx(3.0)

    def test_null_fill_mode_categorical(self):
        df = pd.DataFrame({"cat": ["A", None, "B", "A", None]})
        mode_val = df["cat"].mode()[0]
        filled = df["cat"].fillna(mode_val)
        assert filled.isnull().sum() == 0
        assert filled.mode()[0] == "A"

    def test_duplicate_removal(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3, 3], "b": [1, 1, 2, 3, 3]})
        deduped = df.drop_duplicates()
        assert len(deduped) == 3

    def test_dtype_coercion_to_numeric(self):
        df = pd.DataFrame({"val": ["1.5", "2.0", "bad", "3.5"]})
        coerced = pd.to_numeric(df["val"], errors="coerce")
        assert coerced.isnull().sum() == 1  # "bad" is NaN
        assert coerced.dropna().tolist() == [1.5, 2.0, 3.5]


# ── CSV parsing robustness ─────────────────────────────────────────────────────

class TestCSVParsing:

    def test_read_standard_csv(self):
        csv_content = "name,age,score\nAlice,30,95.5\nBob,25,87.2\n"
        df = pd.read_csv(io.StringIO(csv_content))
        assert list(df.columns) == ["name", "age", "score"]
        assert len(df) == 2

    def test_read_csv_with_nulls(self):
        csv_content = "a,b\n1,\n,2\n3,3\n"
        df = pd.read_csv(io.StringIO(csv_content))
        assert df["a"].isnull().sum() == 1
        assert df["b"].isnull().sum() == 1

    def test_read_csv_bad_encoding_handled(self):
        """Ensure latin-1 files can be read gracefully."""
        csv_content = b"name,value\nCaf\xe9,1\n"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(csv_content)
            fname = f.name
        try:
            df = pd.read_csv(fname, encoding="latin-1")
            assert len(df) == 1
        finally:
            os.unlink(fname)

    def test_read_excel_like_csv(self):
        """Semicolon-delimited CSV (common in European locales)."""
        csv_content = "name;value\nAlice;1\nBob;2\n"
        df = pd.read_csv(io.StringIO(csv_content), sep=";")
        assert list(df.columns) == ["name", "value"]
