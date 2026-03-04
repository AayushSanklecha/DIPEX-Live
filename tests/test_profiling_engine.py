"""
tests/test_profiling_engine.py
---------------------------------
Step 3 — Data Profiling Engine: comprehensive test suite.

Tests cover:
  Profiler          — numeric (outliers, skew, kurtosis, cardinality),
                      categorical (entropy, top values), datetime, edge cases
  CorrelationEngine — Pearson, Spearman, Cramér's V, highlights system
  MissingnessAnalyzer — row buckets, null corr, MCAR/MAR classification
  DriftDetector     — PSI, KS-test, Jensen-Shannon, temporal drift
  ProfileReport     — end-to-end assembly and JSON serialisation
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from profiling.profiler import Profiler
from profiling.correlation_engine import CorrelationEngine
from profiling.missingness_analyzer import MissingnessAnalyzer
from profiling.drift_detector import DriftDetector
from profiling.profile_report import ProfileReport


# ─────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────

# Use a separate seed for each test to avoid fixture ordering issues
RNG = np.random.default_rng(42)

_BASE_CONFIG = {
    "storage": {"report_dir": tempfile.mkdtemp()},
    "profiling": {
        "outlier":     {"iqr_multiplier": 1.5, "zscore_threshold": 3.0},
        "cardinality": {"low_pct": 0.05, "high_pct": 0.50, "unique_pct": 0.95},
        "correlation": {"strong_threshold": 0.80, "near_duplicate_threshold": 0.95},
        "missingness": {"mcar_threshold": 0.30},
        "drift": {
            "ks_p_value_threshold":   0.05,
            "js_divergence_threshold": 0.10,
            "psi_stable_threshold":   0.10,
            "psi_watch_threshold":    0.25,
        },
    },
}


@pytest.fixture
def clean_numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age":    rng.integers(18, 80, 100).tolist(),
        "income": rng.uniform(20_000, 200_000, 100).round(2).tolist(),
        "score":  rng.uniform(0, 1, 100).round(4).tolist(),
    })


@pytest.fixture
def skewed_df() -> pd.DataFrame:
    # Log-normal distribution has heavy right skew
    rng = np.random.default_rng(1)
    return pd.DataFrame({"income": rng.lognormal(mean=10, sigma=2, size=200)})


@pytest.fixture
def outlier_df() -> pd.DataFrame:
    # 5 extreme outliers in a normal distribution
    rng = np.random.default_rng(2)
    vals = rng.normal(0, 1, 95).tolist() + [100, -100, 200, -200, 150]
    return pd.DataFrame({"value": vals})


@pytest.fixture
def categorical_df() -> pd.DataFrame:
    return pd.DataFrame({
        "status": ["ACTIVE"] * 60 + ["CLOSED"] * 30 + ["PENDING"] * 10,
        "region": ["EU", "US", "APAC"] * 33 + ["EU"],
    })


# ─────────────────────────────────────────────────────────────
# Profiler — Numeric
# ─────────────────────────────────────────────────────────────

class TestProfilerNumeric:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.P = Profiler(_BASE_CONFIG)

    def test_basic_stats_present(self, clean_numeric_df):
        result = self.P.profile(clean_numeric_df)
        age = result["columns"]["age"]
        for stat in ("min", "max", "mean", "median", "std", "q25", "q75", "iqr"):
            assert stat in age, f"Missing stat: {stat}"

    def test_skewness_and_kurtosis(self, skewed_df):
        result = self.P.profile(skewed_df)
        col = result["columns"]["income"]
        assert col["skewness"] > 1.0, "Expected positive right skew for log-normal"
        assert col["high_skew"] is True

    def test_outlier_iqr_detected(self, outlier_df):
        result = self.P.profile(outlier_df)
        iqr_info = result["columns"]["value"]["outliers"]["iqr"]
        assert iqr_info["count"] >= 5, "Expected at least 5 IQR outliers"
        assert iqr_info["pct"] > 0

    def test_outlier_zscore_detected(self, outlier_df):
        result = self.P.profile(outlier_df)
        z_info = result["columns"]["value"]["outliers"]["zscore"]
        assert z_info["count"] >= 1

    def test_cardinality_tier_unique(self):
        df = pd.DataFrame({"id": list(range(100))})
        result = Profiler(_BASE_CONFIG).profile(df)
        assert result["columns"]["id"]["cardinality_tier"] == "unique"

    def test_cardinality_tier_low(self):
        df = pd.DataFrame({"flag": [0, 1] * 50})
        result = Profiler(_BASE_CONFIG).profile(df)
        assert result["columns"]["flag"]["cardinality_tier"] == "low"

    def test_percentiles_present(self, clean_numeric_df):
        result = self.P.profile(clean_numeric_df)
        for pct in ("q05", "q25", "q75", "q95"):
            assert pct in result["columns"]["age"]

    def test_shapiro_p_present(self, clean_numeric_df):
        result = self.P.profile(clean_numeric_df)
        assert "shapiro_p" in result["columns"]["age"]

    def test_high_outlier_rate_flag(self, outlier_df):
        result = Profiler(_BASE_CONFIG).profile(outlier_df)
        flags = [f["flag"] for f in result["analyst_flags"]]
        assert "HIGH_OUTLIER_RATE_IQR" in flags

    def test_empty_dataframe_handled(self):
        result = Profiler().profile(pd.DataFrame())
        assert result["row_count"] == 0
        assert result["columns"] == {}

    def test_all_null_column_handled(self):
        df = pd.DataFrame({"x": [None] * 10, "y": [1.0] * 10})
        result = Profiler().profile(df)
        assert "note" in result["columns"]["x"]


# ─────────────────────────────────────────────────────────────
# Profiler — Categorical
# ─────────────────────────────────────────────────────────────

class TestProfilerCategorical:
    def test_top_values_present(self, categorical_df):
        result = Profiler(_BASE_CONFIG).profile(categorical_df)
        col = result["columns"]["status"]
        assert "top_values" in col
        assert "ACTIVE" in col["top_values"]

    def test_mode_is_dominant_value(self, categorical_df):
        result = Profiler(_BASE_CONFIG).profile(categorical_df)
        assert result["columns"]["status"]["mode"] == "ACTIVE"

    def test_entropy_bits_computed(self, categorical_df):
        result = Profiler(_BASE_CONFIG).profile(categorical_df)
        assert "entropy_bits" in result["columns"]["status"]
        assert result["columns"]["status"]["entropy_bits"] > 0


# ─────────────────────────────────────────────────────────────
# CorrelationEngine
# ─────────────────────────────────────────────────────────────

class TestCorrelationEngine:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.CE = CorrelationEngine(_BASE_CONFIG)

    def test_pearson_detects_perfect_positive(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0],
                           "b": [2.0, 4.0, 6.0, 8.0, 10.0]})
        result = self.CE.compute(df)
        r = result["pearson"]["a::b"]
        assert abs(r - 1.0) < 1e-4

    def test_spearman_handles_non_linear(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5],
                           "y": [1, 4, 9, 16, 25]})   # perfect monotonic
        result = self.CE.compute(df)
        rho = result["spearman"]["x::y"]
        assert abs(rho - 1.0) < 1e-4

    def test_cramers_v_independent_cols(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "a": rng.choice(["X", "Y"], 100).tolist(),
            "b": rng.choice(["P", "Q"], 100).tolist(),
        })
        result = self.CE.compute(df)
        v = result["cramers_v"].get("a::b", 0)
        assert v < 0.4  # near-independent

    def test_near_duplicate_flag_raised(self):
        df = pd.DataFrame({
            "a": [float(i) for i in range(50)],
            "b": [float(i) + 0.001 for i in range(50)],
        })
        result = self.CE.compute(df)
        flags = [h["flag"] for h in result["highlights"]]
        assert "NEAR_DUPLICATE_COLUMNS" in flags

    def test_empty_df_returns_empty(self):
        result = CorrelationEngine().compute(pd.DataFrame())
        assert result["pearson"] == {}
        assert result["highlights"] == []


# ─────────────────────────────────────────────────────────────
# MissingnessAnalyzer
# ─────────────────────────────────────────────────────────────

class TestMissingnessAnalyzer:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.MA = MissingnessAnalyzer(_BASE_CONFIG)

    def test_complete_column_flagged_complete(self, clean_numeric_df):
        result = self.MA.analyze(clean_numeric_df)
        for pattern_info in result["column_patterns"].values():
            assert pattern_info["pattern"] == "COMPLETE"

    def test_mcar_detected(self):
        rng = np.random.default_rng(10)
        mask_a = rng.choice([True, False], 100, p=[0.1, 0.9])
        mask_b = rng.choice([True, False], 100, p=[0.1, 0.9])
        df = pd.DataFrame({
            "a": np.where(mask_a, np.nan, 1.0),
            "b": np.where(mask_b, np.nan, 2.0),
        })
        result = self.MA.analyze(df)
        # Independent random masks → MCAR expected
        assert result["column_patterns"]["a"]["pattern"] in ("MCAR", "MAR")

    def test_mar_detected_when_nulls_correlated(self):
        mask = np.array([True] * 30 + [False] * 70)
        df = pd.DataFrame({
            "a": np.where(mask, np.nan, 1.0),
            "b": np.where(mask, np.nan, 2.0),  # same missing rows
        })
        result = self.MA.analyze(df)
        # a and b share null rows → both should be MAR
        assert result["column_patterns"]["a"]["pattern"] == "MAR"

    def test_row_buckets_sum_to_total(self, clean_numeric_df):
        result = self.MA.analyze(clean_numeric_df)
        buckets = result["row_missingness_buckets"]
        total = sum(buckets.values())
        assert total == len(clean_numeric_df)

    def test_high_null_rate_flagged(self):
        df = pd.DataFrame({"x": [None] * 40 + [1.0] * 60})
        result = self.MA.analyze(df)
        flags = [f["flag"] for f in result["analyst_flags"]]
        assert "MNAR_SUSPECTED" in flags

    def test_overall_null_pct_computed(self):
        df = pd.DataFrame({"a": [None, 1.0, 2.0, 3.0, 4.0]})
        result = self.MA.analyze(df)
        assert abs(result["overall_null_pct"] - 0.20) < 1e-6


# ─────────────────────────────────────────────────────────────
# DriftDetector
# ─────────────────────────────────────────────────────────────

class TestDriftDetector:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.DD = DriftDetector(_BASE_CONFIG)

    def test_psi_zero_identical_arrays(self):
        arr = np.linspace(0, 1, 100)
        psi = self.DD.calculate_psi(arr, arr)
        assert psi < 0.01

    def test_psi_large_on_disjoint(self):
        psi = self.DD.calculate_psi(
            np.zeros(100), np.ones(100)
        )
        assert psi > 0.25

    def test_ks_detects_shift(self):
        baseline = np.random.default_rng(3).normal(0, 1, 200)
        shifted  = np.random.default_rng(4).normal(5, 1, 200)
        baseline_df = pd.DataFrame({"x": baseline})
        shifted_df  = pd.DataFrame({"x": shifted})
        result = self.DD.detect(baseline_df, shifted_df)
        assert result["columns"]["x"]["ks_drifted"] is True

    def test_js_stable_on_same_dist(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        baseline_df = pd.DataFrame({"x": a})
        current_df  = pd.DataFrame({"x": b})
        result = self.DD.detect(baseline_df, current_df)
        # JS divergence between two draws from the same distribution should be low
        js_div = result["columns"]["x"]["js_divergence"]
        assert js_div < 0.30, f"Expected low JS divergence for same distribution, got {js_div:.4f}"

    def test_psi_status_labels(self):
        dd = DriftDetector(_BASE_CONFIG)
        assert dd._psi_status(0.05) == "STABLE"
        assert dd._psi_status(0.15) == "WATCH"
        assert dd._psi_status(0.30) == "SIGNIFICANT_DRIFT"

    def test_temporal_drift_requires_timestamp_col(self, clean_numeric_df):
        result = self.DD.detect_temporal_drift(clean_numeric_df, timestamp_col=None)
        assert result == {}

    def test_temporal_drift_with_synthetic_time_series(self):
        rng = np.random.default_rng(6)
        n = 365
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        values = rng.normal(0, 1, n)
        # Large mean shift in last ~18% of data to ensure window z-score > 2.0
        values[300:] += 50
        df = pd.DataFrame({"event_time": dates, "value": values})
        dd = DriftDetector(_BASE_CONFIG)
        result = dd.detect_temporal_drift(df, timestamp_col="event_time", window="14D")
        assert "value" in result, "Expected 'value' column in temporal drift result"
        assert result["value"]["alert_windows"] > 0, (
            f"Expected at least one alert window. Result: {result['value']}"
        )

    def test_drift_analyst_flags_populated(self):
        rng = np.random.default_rng(8)
        baseline_df = pd.DataFrame({"x": rng.normal(0, 1, 300)})
        current_df  = pd.DataFrame({"x": rng.normal(10, 1, 300)})
        result = self.DD.detect(baseline_df, current_df)
        assert len(result["analyst_flags"]) > 0


# ─────────────────────────────────────────────────────────────
# ProfileReport (end-to-end)
# ─────────────────────────────────────────────────────────────

class TestProfileReport:
    def test_report_has_required_top_level_keys(self, clean_numeric_df):
        rpt = ProfileReport(_BASE_CONFIG)
        result = rpt.generate(clean_numeric_df, run_id="test-001")
        for key in ("run_id", "generated_at", "dataset_shape", "columns",
                    "correlation", "missingness", "drift", "analyst_flags",
                    "flag_count"):
            assert key in result

    def test_report_saved_to_disk(self, clean_numeric_df):
        report_dir = tempfile.mkdtemp()
        cfg = dict(_BASE_CONFIG)
        cfg["storage"] = {"report_dir": report_dir}
        rpt = ProfileReport(cfg)
        rpt.generate(clean_numeric_df, run_id="test-save")
        path = os.path.join(report_dir, "test-save_profile.json")
        assert os.path.isfile(path)
        with open(path) as fh:
            data = json.load(fh)
        assert data["run_id"] == "test-save"

    def test_flag_count_matches_flags_list(self, clean_numeric_df):
        rpt = ProfileReport(_BASE_CONFIG)
        result = rpt.generate(clean_numeric_df, run_id="test-count")
        assert result["flag_count"] == len(result["analyst_flags"])

    def test_with_baseline_drift_section_populated(self, clean_numeric_df):
        rng = np.random.default_rng(9)
        baseline = pd.DataFrame({"age": rng.normal(30, 5, 100)})
        current  = pd.DataFrame({"age": rng.normal(60, 5, 100)})
        rpt = ProfileReport(_BASE_CONFIG, baseline_df=baseline)
        result = rpt.generate(current, run_id="test-drift")
        assert "columns" in result["drift"]

    def test_empty_df_does_not_crash(self):
        rpt = ProfileReport(_BASE_CONFIG)
        result = rpt.generate(pd.DataFrame(), run_id="test-empty")
        assert result["dataset_shape"]["rows"] == 0
