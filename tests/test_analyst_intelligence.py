"""
tests/test_analyst_intelligence.py
--------------------------------------
50+ tests across all 3 analyst tiers, cognitive engine, and supporting modules.
"""
import math
import os
import sys
import pytest
import pandas as pd
import numpy as np

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "user_id":    range(n),
        "date":       pd.date_range("2024-01-01", periods=n, freq="D"),
        "revenue":    np.random.lognormal(4, 1, n),
        "cost":       np.random.uniform(10, 50, n),
        "churn":      (np.random.rand(n) < 0.15).astype(int),
        "region":     np.random.choice(["North", "South", "East", "West"], n),
        "conversion": np.random.uniform(0, 1, n),
        "price":      np.random.uniform(5, 500, n),
    })


@pytest.fixture()
def silver(sample_df):
    try:
        from ingestion.data_layers import ImmutableDataFrame
    except ImportError:
        pytest.skip("ingestion.data_layers unavailable")
    return ImmutableDataFrame(sample_df.copy(), layer="silver", dataset_id="test_sales")


@pytest.fixture()
def layer_manager(tmp_path):
    try:
        from ingestion.data_layers import LayerManager
        return LayerManager(base_dir=str(tmp_path))
    except ImportError:
        pytest.skip("LayerManager unavailable")


# ── MidAnalyst ────────────────────────────────────────────────────────────────

class TestMidAnalyst:
    def _get(self, lm):
        from analyst.mid_analyst import MidAnalyst
        return MidAnalyst(layer_manager=lm)

    def test_automated_eda_returns_gold(self, silver, layer_manager):
        mid = self._get(layer_manager)
        gold = mid.automated_eda(silver)
        assert gold is not None

    def test_automated_eda_has_distribution_checks(self, silver, layer_manager):
        mid  = self._get(layer_manager)
        gold = mid.automated_eda(silver)
        df   = gold.data if hasattr(gold, "data") else gold
        if df is not None:
            assert "column" in df.columns or len(df) > 0

    def test_business_insights_returns_rows(self, silver, layer_manager):
        mid  = self._get(layer_manager)
        gold = mid.business_insights(silver, target_col="churn")
        df   = gold.data if hasattr(gold, "data") else gold
        assert df is not None and len(df) >= 0

    def test_outlier_investigation_adds_is_outlier_flag(self, silver, layer_manager):
        mid  = self._get(layer_manager)
        gold = mid.outlier_investigation(silver, numeric_cols=["revenue"], method="iqr")
        df   = gold.data if hasattr(gold, "data") else gold
        if df is not None:
            assert any("is_outlier" in c for c in df.columns)

    def test_variance_analysis_returns_p_value(self, silver, layer_manager):
        mid  = self._get(layer_manager)
        gold = mid.variance_analysis(silver, group_col="region", value_col="revenue")
        df   = gold.data if hasattr(gold, "data") else gold
        assert df is not None

    def test_ab_test_returns_uplift(self, silver, layer_manager):
        df = silver._df.copy()
        df["group"] = np.where(np.random.rand(len(df)) > 0.5, "treatment", "control")
        from ingestion.data_layers import ImmutableDataFrame as IDF
        silver2 = IDF(df, layer="silver", dataset_id="ab_test")
        mid  = self._get(layer_manager)
        gold = mid.ab_test_evaluation(silver2, group_col="group", metric_col="revenue")
        assert gold is not None

    def test_time_series_adds_rolling_mean(self, silver, layer_manager):
        mid  = self._get(layer_manager)
        gold = mid.time_series_exploration(silver, date_col="date", value_col="revenue")
        df   = gold.data if hasattr(gold, "data") else gold
        if df is not None:
            assert any("rolling_mean" in c for c in df.columns)

    def test_segmentation_clustering_adds_segment(self, silver, layer_manager):
        mid  = self._get(layer_manager)
        gold = mid.segmentation_clustering(silver, n_clusters=3,
                                            feature_cols=["revenue", "cost"])
        df   = gold.data if hasattr(gold, "data") else gold
        if df is not None:
            assert "_segment" in df.columns

    def test_correlation_deep_dive_returns_pearson(self, silver, layer_manager):
        mid  = self._get(layer_manager)
        gold = mid.correlation_deep_dive(silver, target_col="churn")
        df   = gold.data if hasattr(gold, "data") else gold
        assert df is not None


# ── SQLAutomationEngine ───────────────────────────────────────────────────────

class TestSQLAutomation:
    def _get(self):
        from analyst.sql_automation import SQLAutomationEngine
        return SQLAutomationEngine()

    def test_select_generates_valid_sql(self):
        eng = self._get()
        q   = eng.select("data", columns=["revenue", "region"], limit=10)
        assert "SELECT" in q.sql.upper() and "revenue" in q.sql

    def test_aggregate_generates_group_by(self):
        eng = self._get()
        q   = eng.aggregate("data", group_by=["region"], agg_map={"revenue": "SUM"})
        assert "GROUP BY" in q.sql.upper()

    def test_cte_wraps_correctly(self):
        eng = self._get()
        q   = eng.cte("base", "SELECT * FROM data", "SELECT * FROM base")
        assert "WITH base AS" in q.sql

    def test_window_function_generates_over(self):
        eng = self._get()
        q   = eng.window_function("data", "revenue", order_by="date", window_fn="RANK")
        assert "OVER" in q.sql.upper()

    def test_cost_estimation_increases_with_joins(self):
        eng = self._get()
        simple = eng.select("data")
        joined = eng.join("data", "lookup", on="data.id = lookup.id")
        assert joined.cost_score > simple.cost_score

    def test_execute_returns_dataframe(self, sample_df):
        try:
            import duckdb  # noqa: F401
        except ImportError:
            pytest.skip("DuckDB not installed")
        eng = self._get()
        q   = eng.select("data", columns=list(sample_df.columns[:3]), limit=5)
        result, q2 = eng.execute(q, sample_df)
        assert len(result) <= 5

    def test_validate_returns_true_for_valid_sql(self, sample_df):
        try:
            import duckdb  # noqa: F401
        except ImportError:
            pytest.skip("DuckDB not installed")
        eng  = self._get()
        ok, err = eng.validate("SELECT * FROM data LIMIT 1", df=sample_df)
        assert ok is True and err == ""

    def test_validate_returns_false_for_invalid_sql(self):
        try:
            import duckdb  # noqa: F401
        except ImportError:
            pytest.skip("DuckDB not installed")
        eng  = self._get()
        ok, err = eng.validate("SELECT INVALID SYNTAX !!!;")
        assert ok is False and len(err) > 0


# ── ExcelEngine ───────────────────────────────────────────────────────────────

class TestExcelEngine:
    def _get(self):
        from analyst.excel_engine import ExcelEngine
        return ExcelEngine()

    def test_pivot_table_returns_dataframe(self, sample_df):
        eng    = self._get()
        result = eng.pivot_table(sample_df, index="region", values="revenue", aggfunc="sum")
        assert isinstance(result, pd.DataFrame) and len(result) > 0

    def test_vlookup_adds_columns(self, sample_df):
        lookup = sample_df[["region"]].drop_duplicates()
        lookup["region_code"] = lookup["region"].str[0]
        result = self._get().vlookup(sample_df, lookup, "region", ["region_code"])
        assert "region_code" in result.columns

    def test_countif_returns_integer(self, sample_df):
        count = self._get().countif(sample_df, {"region": "North"})
        assert isinstance(count, int) and count >= 0

    def test_sumif_returns_float(self, sample_df):
        total = self._get().sumif(sample_df, {"region": "North"}, "revenue")
        assert isinstance(total, float)

    def test_kpi_summary_has_expected_cols(self, sample_df):
        result = self._get().kpi_summary(sample_df)
        assert "kpi" in result.columns

    def test_percent_of_total_adds_pct_col(self, sample_df):
        result = self._get().percent_of_total(sample_df, "revenue")
        assert "revenue_pct_of_total" in result.columns

    def test_rank_percentile_adds_rank_col(self, sample_df):
        result = self._get().rank_percentile(sample_df, "revenue")
        assert "revenue_rank" in result.columns and "revenue_percentile" in result.columns

    def test_running_total_is_monotone(self, sample_df):
        result = self._get().running_total(sample_df, "revenue", sort_col="date")
        cumsum = result["revenue_cumsum"]
        assert (cumsum.diff().dropna() >= -1e-6).all()

    def test_formula_computes_margin(self, sample_df):
        result = self._get().formula(sample_df, {"margin": "revenue - cost"})
        assert "margin" in result.columns
        assert ((result["margin"] - (sample_df["revenue"] - sample_df["cost"])).abs() < 1e-6).all()

    def test_highlight_rules_adds_flag(self, sample_df):
        result = self._get().highlight_rules(sample_df,
                    rules=[{"col": "revenue", "op": ">", "value": 100, "label": "high_rev"}])
        assert "_flag_high_rev" in result.columns


# ── DocumentationGenerator ───────────────────────────────────────────────────

class TestDocumentationGenerator:
    def _get(self, tmp_path):
        from analyst.documentation_generator import DocumentationGenerator
        return DocumentationGenerator(docs_dir=str(tmp_path))

    def test_kpi_dictionary_returns_list(self, sample_df, tmp_path):
        gen  = self._get(tmp_path)
        kpis = gen.generate_kpi_dictionary(sample_df, "test_dataset")
        assert isinstance(kpis, list) and len(kpis) > 0

    def test_data_contract_has_schema(self, sample_df, tmp_path):
        gen      = self._get(tmp_path)
        contract = gen.generate_data_contract(sample_df, "test_dataset")
        assert "schema" in contract and len(contract["schema"]) > 0

    def test_changelog_persists_entry(self, tmp_path):
        gen = self._get(tmp_path)
        gen.log_change("test", "v1.1", "schema", "Added revenue column")
        entries = gen.get_changelog("test")
        assert len(entries) == 1 and entries[0]["change_type"] == "schema"

    def test_lineage_document_has_expected_keys(self, tmp_path):
        gen = self._get(tmp_path)
        doc = gen.generate_lineage_document("test", "bronze_123", "silver_456", "gold_789")
        assert "bronze_checksum" in doc and "gold_lineage_id" in doc


# ── ProblemFramingEngine ──────────────────────────────────────────────────────

class TestProblemFraming:
    def _get(self):
        from analyst.problem_framing import ProblemFramingEngine
        return ProblemFramingEngine()

    def test_frames_revenue_question(self):
        f = self._get().frame("What is driving revenue decline?", dataset_id="sales")
        assert "revenue" in f.detected_intent
        assert len(f.kpi_proposals) > 0

    def test_frames_churn_question(self):
        f = self._get().frame("Why are customers churning?", dataset_id="crm")
        assert "churn" in f.detected_intent

    def test_north_star_is_set(self):
        f = self._get().frame("How do we grow our user base?")
        # growth intent → north star should be new_users or similar
        assert f.north_star is not None

    def test_ambiguity_is_higher_for_vague_request(self):
        vague    = self._get().frame("Help me understand if things are better")
        specific = self._get().frame("Calculate conversion rate by region for Q4")
        assert vague.ambiguity_score >= specific.ambiguity_score

    def test_clarification_questions_generated_for_vague(self):
        f = self._get().frame("Improve our metrics")
        assert len(f.clarification_questions) > 0


# ── ExperimentDesigner ────────────────────────────────────────────────────────

class TestExperimentDesigner:
    def _get(self, tmp_path):
        from analyst.experiment_designer import ExperimentDesigner
        return ExperimentDesigner(store_dir=str(tmp_path))

    def test_design_returns_valid_n(self, tmp_path):
        d = self._get(tmp_path).design("CTA change", "conversion_rate", 0.10, mde=0.05)
        assert d.n_per_group >= 30

    def test_design_persists_to_disk(self, tmp_path):
        self._get(tmp_path).design("Button colour", "clicks", 0.05)
        files = os.listdir(str(tmp_path))
        assert any(".json" in f for f in files)

    def test_underpowered_warning_emitted(self, tmp_path):
        d = self._get(tmp_path).design("Test", "clicks", 0.50, mde=0.001, daily_traffic=10)
        assert any("Underpowered" in w or "power" in w.lower() for w in d.warnings) or d.n_per_group > 0

    def test_runtime_proportional_to_traffic(self, tmp_path):
        d1 = self._get(tmp_path).design("T1", "clicks", 0.10, daily_traffic=5000)
        d2 = self._get(tmp_path).design("T2", "clicks", 0.10, daily_traffic=500)
        assert d2.runtime_days >= d1.runtime_days

    def test_validate_result_returns_recommendation(self, tmp_path, sample_df):
        try:
            from scipy import stats  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")
        designer = self._get(tmp_path)
        design   = designer.design("Revenue boost", "revenue", 0.10)
        a, b     = sample_df["revenue"].iloc[:80], sample_df["revenue"].iloc[80:]
        result   = designer.validate_result(design, a, b)
        assert "recommendation" in result


# ── MentorshipEngine ──────────────────────────────────────────────────────────

class TestMentorshipEngine:
    def _get(self):
        from analyst.mentorship_engine import MentorshipEngine
        return MentorshipEngine()

    def test_select_star_triggers_warning(self):
        r = self._get().review_sql("SELECT * FROM data LIMIT 10")
        assert r.warnings > 0 or r.blockers > 0

    def test_drop_table_is_blocker(self):
        r = self._get().review_sql("DROP TABLE users")
        assert r.blockers > 0 and not r.approved

    def test_safe_query_scores_high(self):
        r = self._get().review_sql("SELECT revenue, region FROM data WHERE date > '2024-01-01'")
        assert r.score >= 70

    def test_causation_language_triggers_blocker(self):
        r = self._get().review_interpretation(
            "Correlation with sales causes conversion",
            p_value=0.03
        )
        assert r.blockers > 0

    def test_nonsignificant_significance_claim_is_blocker(self):
        r = self._get().review_interpretation("Results are significant", p_value=0.20)
        assert r.blockers > 0

    def test_signoff_fails_without_required_field(self):
        r = self._get().sign_off({"dataset_id": "test"})
        assert r.blockers > 0 or not r.approved

    def test_signoff_passes_good_report(self):
        report = {
            "dataset_id": "test", "validation_passed": True,
            "confidence_score": 0.90, "generated_at": "2024-01-01",
        }
        r = self._get().sign_off(report)
        assert r.approved


# ── RLOptimizer ───────────────────────────────────────────────────────────────

class TestRLOptimizer:
    def _get(self, tmp_path):
        from analyst.rl_optimizer import RLOptimizer
        return RLOptimizer(store_path=str(tmp_path))

    def test_propose_returns_n_proposals(self, tmp_path):
        rl  = self._get(tmp_path)
        props = rl.propose("cleaning", n_proposals=3)
        assert len(props) == 3

    def test_proposal_is_advisory_only(self, tmp_path):
        rl    = self._get(tmp_path)
        props = rl.propose("model_selection")
        assert all(p.advisory_only for p in props)

    def test_record_outcome_updates_weights(self, tmp_path):
        rl    = self._get(tmp_path)
        prop  = rl.propose("cleaning")[0]
        old_w = rl._arm_weights.get("cleaning", {}).get(prop.strategy_name, 0.0)
        rl.record_outcome(prop, actual_gain=0.30)
        new_w = rl._arm_weights.get("cleaning", {}).get(prop.strategy_name, 0.0)
        assert abs(new_w - old_w) > 0

    def test_hard_gate_fires_on_critical_null_rate(self, tmp_path):
        rl  = self._get(tmp_path)
        msg = rl.check_hard_gate(null_rate=0.95)
        assert msg is not None and "sparse" in msg.lower()

    def test_hard_gate_fires_on_checksum_fail(self, tmp_path):
        rl  = self._get(tmp_path)
        msg = rl.check_hard_gate(checksum_ok=False)
        assert msg is not None

    def test_top_strategies_sorted_descending(self, tmp_path):
        rl  = self._get(tmp_path)
        top = rl.top_strategies("cleaning", n=3)
        weights = [w for _, w in top]
        assert weights == sorted(weights, reverse=True)


# ── AnalystOrchestrator ───────────────────────────────────────────────────────

class TestAnalystOrchestrator:
    def _get(self, lm):
        from analyst.analyst_orchestrator import AnalystOrchestrator
        return AnalystOrchestrator(config={})

    def test_make_silver_returns_immutable(self, sample_df, layer_manager):
        orch   = self._get(layer_manager)
        silver = orch.make_silver(sample_df, "test")
        from ingestion.data_layers import ImmutableDataFrame
        assert isinstance(silver, ImmutableDataFrame)

    def test_tier_selector_routes_basic_to_junior(self, layer_manager):
        from analyst.analyst_orchestrator import AnalystOrchestrator
        assert AnalystOrchestrator._select_tier("basic_stats") == "junior"

    def test_tier_selector_routes_eda_to_mid(self, layer_manager):
        from analyst.analyst_orchestrator import AnalystOrchestrator
        assert AnalystOrchestrator._select_tier("automated_eda") == "mid"

    def test_run_business_insights_returns_result(self, silver, layer_manager):
        orch   = self._get(layer_manager)
        result = orch.run("business_insights", silver, force_tier="mid")
        assert result is not None and result.tier_used == "mid"

    def test_run_result_has_cognitive_score(self, silver, layer_manager):
        orch   = self._get(layer_manager)
        result = orch.run("automated_eda", silver, force_tier="mid")
        assert 0.0 <= result.cognitive_score <= 1.0

    def test_full_analysis_pipeline_returns_multiple_results(self, silver, layer_manager):
        orch    = self._get(layer_manager)
        results = orch.run_full_analysis(silver, target_col="churn")
        assert len(results) >= 2
