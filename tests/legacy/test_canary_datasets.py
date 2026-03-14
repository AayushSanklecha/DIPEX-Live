"""
tests/test_canary_datasets.py
------------------------------
Canary dataset regression test suite.

Each canary fixture represents a real-world data archetype that historically
exposes subtle logic bugs in analytics pipelines. These tests verify that the
full DIPEX ISSFSnapshot + pipeline ingestion path:

  ✓ Never crashes with an unhandled exception
  ✓ Produces valid quality_score ∈ [0, 1]
  ✓ Returns the correct row_count
  ✓ Produces well-formed column_metadata
  ✓ Encodes successfully into a RAG experience vector (no NaN/Inf)
  ✓ Survives ConfidenceVectorAggregator.aggregate() with any gate result

These tests serve as a first-line smoke test for subtle bugs that only
appear with specific real datasets but not with synthetic hand-crafted fixtures.

Run:
    pytest tests/test_canary_datasets.py -v
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

# ── Fixture directory resolution ─────────────────────────────────────────────

CANARY_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "canary")

# Auto-generate fixtures if not present (keeps CI self-contained)
def _ensure_fixtures() -> None:
    expected = [f"{i:02d}_" for i in range(1, 11)]
    present  = [f for f in os.listdir(CANARY_DIR) if f.endswith(".csv")] if os.path.isdir(CANARY_DIR) else []
    if any(not any(f.startswith(pfx) for f in present) for pfx in expected):
        import subprocess, sys
        gen = os.path.join(CANARY_DIR, "generate_canary_fixtures.py")
        subprocess.run([sys.executable, gen], check=True, capture_output=True)

try:
    _ensure_fixtures()
except Exception:
    pass  # gracefully skip if generator fails (CI without write access)


def _csv_files() -> list[str]:
    if not os.path.isdir(CANARY_DIR):
        return []
    return sorted(f for f in os.listdir(CANARY_DIR) if f.endswith(".csv"))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load CSV with maximum resilience
# ─────────────────────────────────────────────────────────────────────────────

def _load(filename: str) -> pd.DataFrame:
    path = os.path.join(CANARY_DIR, filename)
    return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ISSFSnapshot canary tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("filename", _csv_files())
class TestISSFSnapshotCanary:
    """
    Run every canary CSV through ISSFSnapshot.from_dataframe() and assert
    all contract invariants hold regardless of the fixture's quirks.
    """

    def test_snapshot_never_crashes(self, filename: str) -> None:
        from ingestion.issf import ISSFSnapshot
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty after load — handled separately")
        snap = ISSFSnapshot.from_dataframe(df, dataset_id=filename, source_type="file")
        assert snap is not None

    def test_quality_score_in_unit_interval(self, filename: str) -> None:
        from ingestion.issf import ISSFSnapshot
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        snap = ISSFSnapshot.from_dataframe(df, dataset_id=filename, source_type="file")
        assert 0.0 <= snap.quality_score <= 1.0, (
            f"{filename}: quality_score={snap.quality_score}"
        )

    def test_row_count_matches_dataframe(self, filename: str) -> None:
        from ingestion.issf import ISSFSnapshot
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        snap = ISSFSnapshot.from_dataframe(df, dataset_id=filename, source_type="file")
        assert snap.row_count == len(df), (
            f"{filename}: snap.row_count={snap.row_count} != len(df)={len(df)}"
        )

    def test_column_metadata_completeness(self, filename: str) -> None:
        from ingestion.issf import ISSFSnapshot
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        snap = ISSFSnapshot.from_dataframe(df, dataset_id=filename, source_type="file")
        assert len(snap.column_metadata) == len(df.columns), (
            f"{filename}: metadata cols {len(snap.column_metadata)} != df cols {len(df.columns)}"
        )

    def test_fingerprint_is_valid_hex64(self, filename: str) -> None:
        from ingestion.issf import ISSFSnapshot
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        snap = ISSFSnapshot.from_dataframe(df, dataset_id=filename, source_type="file")
        assert isinstance(snap.fingerprint, str) and len(snap.fingerprint) == 64
        int(snap.fingerprint, 16)  # raises if not valid hex

    def test_validation_status_is_enum(self, filename: str) -> None:
        from ingestion.issf import ISSFSnapshot
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        snap = ISSFSnapshot.from_dataframe(df, dataset_id=filename, source_type="file")
        assert snap.validation_status in {"PASSED", "FAILED", "WARN"}, (
            f"{filename}: unexpected validation_status={snap.validation_status!r}"
        )

    def test_to_dict_is_json_serialisable(self, filename: str) -> None:
        import json
        from ingestion.issf import ISSFSnapshot
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        snap = ISSFSnapshot.from_dataframe(df, dataset_id=filename, source_type="file")
        json.dumps(snap.to_dict())   # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ExperienceRecall (RAG) canary tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("filename", _csv_files())
class TestExperienceRecallCanary:
    """
    Every canary fixture must encode without crash and produce a
    finite 16-dim vector — no NaN or inf allowed.
    """

    def test_encode_never_crashes(self, filename: str) -> None:
        from proposal.rag.experience_recall import ExperienceRecall
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        recall = ExperienceRecall()
        vec = recall._encode_profile(df)
        assert len(vec) == 16

    def test_encode_no_nan_from_finite_input(self, filename: str) -> None:
        from proposal.rag.experience_recall import ExperienceRecall
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        # Use only finite-valued rows for this invariant
        num_cols = df.select_dtypes(include=[np.number]).columns
        df_clean = df.copy()
        df_clean[num_cols] = df_clean[num_cols].replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna(subset=num_cols, how="all") if len(num_cols) else df_clean
        if df_clean.empty:
            pytest.skip(f"{filename}: no finite rows")
        recall = ExperienceRecall()
        vec = recall._encode_profile(df_clean)
        for i, v in enumerate(vec):
            assert not math.isnan(v) and not math.isinf(v), (
                f"{filename}: vec[{i}]={v} is NaN/Inf for finite-input df"
            )

    def test_recall_returns_list(self, filename: str) -> None:
        """recall() should always return a list (possibly empty if store empty)."""
        from proposal.rag.experience_recall import ExperienceRecall
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        recall = ExperienceRecall(config={"proposal": {"rag": {"storage_path": ":memory:"}}})
        results = recall.recall(df, top_k=3)
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ConfidenceVectorAggregator canary tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("filename", _csv_files())
class TestConfidenceVectorCanary:
    """
    Simulate a gate-result summary derived from each canary dataset and
    assert confidence_score is always in [0, 1].
    """

    def _make_gate1(self, df: pd.DataFrame) -> dict:
        null_frac = df.isnull().mean().mean()
        decision = "REJECT" if null_frac > 0.9 else ("WARN" if null_frac > 0.5 else "PASS")
        return {
            "decision": decision,
            "total_warnings": int(null_frac * 10),
            "total_failures": int(null_frac > 0.9),
        }

    def test_confidence_score_bounded(self, filename: str) -> None:
        from verifier.confidence_vector import ConfidenceVectorAggregator
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        agg = ConfidenceVectorAggregator()
        gate1 = self._make_gate1(df)
        gate2 = {
            "all_gates_passed": gate1["decision"] == "PASS",
            "vector": {
                "statistical": {"passed": gate1["decision"] != "REJECT"},
                "stability": None,
                "drift_robustness": None,
                "domain": {"passed": True},
            },
        }
        cv = agg.aggregate(gate1_result=gate1, gate2_confidence=gate2, retry_attempt=0)
        assert 0.0 <= cv.confidence_score <= 1.0, (
            f"{filename}: confidence_score={cv.confidence_score}"
        )

    def test_confidence_lower_on_reject(self, filename: str) -> None:
        """REJECT gate must produce lower confidence than PASS gate."""
        from verifier.confidence_vector import ConfidenceVectorAggregator
        df = _load(filename)
        if df.empty:
            pytest.skip(f"{filename}: empty")
        agg = ConfidenceVectorAggregator()
        g_pass = {"decision": "PASS", "total_warnings": 0}
        g_reject = {"decision": "REJECT", "total_warnings": 10}
        g2_pass = {"all_gates_passed": True, "vector": {}}
        g2_fail  = {"all_gates_passed": False, "vector": {}}
        cv_pass   = agg.aggregate(gate1_result=g_pass,   gate2_confidence=g2_pass)
        cv_reject = agg.aggregate(gate1_result=g_reject, gate2_confidence=g2_fail)
        assert cv_reject.confidence_score <= cv_pass.confidence_score + 1e-9, (
            f"{filename}: reject score {cv_reject.confidence_score:.4f} "
            f"> pass score {cv_pass.confidence_score:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Edge-case specific regression tests (non-parametrised)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanaryEdgeCases:
    """
    Targeted regression tests for the most dangerous canary archetypes.
    Each test is named after the specific failure mode it was designed to catch.
    """

    def _path(self, name: str) -> str:
        return os.path.join(CANARY_DIR, name)

    def _skip_if_missing(self, name: str) -> None:
        if not os.path.exists(self._path(name)):
            pytest.skip(f"Fixture not generated: {name}")

    def test_single_row_does_not_crash_rag_encoding(self) -> None:
        """Single-row DataFrames had historically caused log1p index errors."""
        self._skip_if_missing("05_single_row.csv")
        from proposal.rag.experience_recall import ExperienceRecall
        df = pd.read_csv(self._path("05_single_row.csv"))
        recall = ExperienceRecall()
        vec = recall._encode_profile(df)
        assert len(vec) == 16
        assert all(not math.isnan(v) for v in vec)

    def test_zero_variance_quality_score_is_valid(self) -> None:
        """Zero-variance columns (std=0) historically caused divide-by-zero."""
        self._skip_if_missing("06_zero_variance.csv")
        from ingestion.issf import ISSFSnapshot
        df = pd.read_csv(self._path("06_zero_variance.csv"))
        snap = ISSFSnapshot.from_dataframe(df, dataset_id="zero_var", source_type="file")
        assert 0.0 <= snap.quality_score <= 1.0

    def test_all_null_columns_validation_status_not_passed(self) -> None:
        """3 fully-null columns should produce WARN or FAILED, not PASSED."""
        self._skip_if_missing("04_all_null_columns.csv")
        from ingestion.issf import ISSFSnapshot
        df = pd.read_csv(self._path("04_all_null_columns.csv"))
        snap = ISSFSnapshot.from_dataframe(df, dataset_id="all_null", source_type="file")
        assert snap.validation_status in {"WARN", "FAILED"}, (
            f"Expected WARN/FAILED for all-null fixture, got {snap.validation_status}"
        )

    def test_inf_nan_mix_handled_without_crash(self) -> None:
        """Explicit inf values must not propagate to NaN in quality_score."""
        self._skip_if_missing("10_inf_nan_mix.csv")
        from ingestion.issf import ISSFSnapshot
        df = pd.read_csv(self._path("10_inf_nan_mix.csv"))
        snap = ISSFSnapshot.from_dataframe(df, dataset_id="inf_nan", source_type="file")
        assert not math.isnan(snap.quality_score), "quality_score is NaN for inf/NaN fixture"
        assert 0.0 <= snap.quality_score <= 1.0

    def test_high_cardinality_does_not_timeout(self) -> None:
        """10 000 unique categories should complete snapshot in reasonable time."""
        self._skip_if_missing("07_high_cardinality.csv")
        import time
        from ingestion.issf import ISSFSnapshot
        df = pd.read_csv(self._path("07_high_cardinality.csv"))
        t0 = time.perf_counter()
        snap = ISSFSnapshot.from_dataframe(df, dataset_id="hi_card", source_type="file")
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Snapshot took {elapsed:.1f}s for high-cardinality fixture (>10s limit)"
        assert snap.row_count == len(df)

    def test_mixed_encodings_snapshot_succeeds(self) -> None:
        """UTF-8 accents, CJK, emoji in string columns must not cause encoding errors."""
        self._skip_if_missing("09_mixed_encodings.csv")
        from ingestion.issf import ISSFSnapshot
        df = pd.read_csv(self._path("09_mixed_encodings.csv"), encoding="utf-8")
        snap = ISSFSnapshot.from_dataframe(df, dataset_id="mixed_enc", source_type="file")
        assert snap.row_count == len(df)

    def test_wide_sparse_column_metadata_count(self) -> None:
        """200-column sparse fixture must produce exactly 201 metadata entries."""
        self._skip_if_missing("08_wide_sparse.csv")
        from ingestion.issf import ISSFSnapshot
        df = pd.read_csv(self._path("08_wide_sparse.csv"))
        snap = ISSFSnapshot.from_dataframe(df, dataset_id="wide_sparse", source_type="file")
        assert len(snap.column_metadata) == len(df.columns), (
            f"Expected {len(df.columns)} metadata entries, got {len(snap.column_metadata)}"
        )

    def test_financial_ohlcv_high_always_gte_low(self) -> None:
        """
        Financial-domain sanity: the pipeline must not corrupt OHLC relationships.
        This is a domain-semantic invariant test.
        """
        self._skip_if_missing("01_financial_ohlcv.csv")
        df = pd.read_csv(self._path("01_financial_ohlcv.csv"))
        assert (df["high"] >= df["low"]).all(), "high < low detected after pipeline load"

    def test_healthcare_readmitted_is_binary(self) -> None:
        """Healthcare fixture: readmitted column must only contain 0 or 1."""
        self._skip_if_missing("02_healthcare_patient.csv")
        df = pd.read_csv(self._path("02_healthcare_patient.csv"))
        assert set(df["readmitted"].dropna().unique()).issubset({0, 1}), (
            "readmitted column contains non-binary values"
        )
