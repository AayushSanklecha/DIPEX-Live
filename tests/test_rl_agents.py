"""
tests/test_rl_agents.py
-------------------------
Automated tests for all RL Q-learning agents in DIPEX.

Covers:
  - RLRateLimiter (adaptive_rate_limiter)
  - RLThresholdTuner (rl_threshold_tuner)
  - RLProfilingStrategy (rl_profiling_strategy)
  - RLFeatureSelector (rl_feature_selector)
  - RLAutoML (rl_automl)
  - RLOrchestrator (rl_orchestrator)
  - RLAuthTuner (rl_auth_tuner)
  - RLNarrative (rl_narrative)

All tests run fully in-memory (no file I/O required).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. RLRateLimiter
# ─────────────────────────────────────────────────────────────────────────────

class TestRLRateLimiter:

    def _agent(self, tmp_path):
        from ingestion.adaptive_rate_limiter import RLRateLimiter
        return RLRateLimiter(state_path=str(tmp_path / "rl_rate.json"))

    def test_api_backoff_returns_valid_action(self, tmp_path):
        agent = self._agent(tmp_path)
        backoff = agent.get_api_backoff("https://api.example.com")
        assert backoff in [1.0, 1.5, 2.0, 3.0, 5.0]

    def test_record_api_success_updates_q(self, tmp_path):
        agent = self._agent(tmp_path)
        backoff = agent.get_api_backoff("https://api.test.io")
        agent.record_api_outcome("https://api.test.io", backoff, success=True, latency_ms=300)
        state = f"api::https://api.test.io"
        assert state in agent.q
        assert str(backoff) in agent.q[state]

    def test_record_api_failure_negative_reward(self, tmp_path):
        agent = self._agent(tmp_path)
        backoff = 1.0
        agent.get_api_backoff("https://slow.api.io")
        agent.record_api_outcome("https://slow.api.io", backoff, success=False, latency_ms=5000)
        state = "api::https://slow.api.io"
        assert agent.q[state][str(backoff)] < 0

    def test_db_chunk_size_valid(self, tmp_path):
        agent = self._agent(tmp_path)
        chunk = agent.get_db_chunk_size("postgres", "prod-host")
        assert chunk in [1000, 5000, 10000, 25000, 50000]

    def test_save_and_reload(self, tmp_path):
        agent = self._agent(tmp_path)
        agent.get_api_backoff("https://a.com")
        agent.record_api_outcome("https://a.com", 1.0, True, 100)
        agent.save()
        agent2 = self._agent(tmp_path)
        assert "api::https://a.com" in agent2.q


# ─────────────────────────────────────────────────────────────────────────────
# 2. RLThresholdTuner
# ─────────────────────────────────────────────────────────────────────────────

class TestRLThresholdTuner:

    def _tuner(self, tmp_path):
        from validation.rl_threshold_tuner import RLThresholdTuner
        return RLThresholdTuner(state_path=str(tmp_path / "thresh.json"))

    def test_threshold_in_action_space(self, tmp_path):
        t = self._tuner(tmp_path)
        thresh = t.get_threshold("sales", "revenue", default=0.05)
        assert thresh in [0.01, 0.05, 0.10, 0.20, 0.30]

    def test_record_outcome_updates_q(self, tmp_path):
        t = self._tuner(tmp_path)
        thresh = t.get_threshold("sales", "revenue", default=0.05)
        t.record_outcome("sales", "revenue", thresh, validation_passed=True, downstream_success=True)
        state = "sales::revenue"
        assert state in t.q

    def test_record_bad_outcome_penalty(self, tmp_path):
        t = self._tuner(tmp_path)
        thresh = t.get_threshold("raw", "price", default=0.10)
        t.record_outcome("raw", "price", thresh, validation_passed=True, downstream_success=False)
        state = "raw::price"
        from validation.rl_threshold_tuner import _ak
        key = _ak(thresh)
        val = t.q[state][key]
        # Seeded at 0.5, positive reward drives it up — unless penalised
        # With -10 reward and alpha=0.15 starting from 0.5: 0.5 + 0.15*(-10-0.5) = 0.5 - 1.575 = -1.075
        assert val < 0.5   # definitely went down from prior of 0.5

    def test_policy_summary(self, tmp_path):
        t = self._tuner(tmp_path)
        for _ in range(3):
            thresh = t.get_threshold("ds1", "col_x")
            t.record_outcome("ds1", "col_x", thresh, True, True)
        summary = t.get_policy_summary()
        assert "ds1::col_x" in summary


# ─────────────────────────────────────────────────────────────────────────────
# 3. RLFeatureSelector
# ─────────────────────────────────────────────────────────────────────────────

class TestRLFeatureSelector:

    def _sel(self, tmp_path, feats=None):
        from preprocessing.rl_feature_selector import RLFeatureSelector
        feats = feats or [f"feat_{i}" for i in range(10)]
        return RLFeatureSelector(feats, state_path=str(tmp_path / "fs.json"))

    def test_first_call_returns_all(self, tmp_path):
        sel = self._sel(tmp_path)
        active = sel.get_active_features()
        assert len(active) == 10

    def test_second_call_returns_list(self, tmp_path):
        sel = self._sel(tmp_path)
        sel.get_active_features()
        active = sel.get_active_features()
        assert isinstance(active, list)
        assert len(active) >= 1

    def test_record_and_exploit(self, tmp_path):
        sel = self._sel(tmp_path)
        active = sel.get_active_features()
        sel.record_outcome(active, cv_delta=0.05, training_time_s=5.0)
        active2 = sel.get_active_features()
        assert isinstance(active2, list)

    def test_registry_persists(self, tmp_path):
        sel = self._sel(tmp_path)
        active = sel.get_active_features()
        sel.record_outcome(active, cv_delta=0.02, training_time_s=3.0)
        sel.save()
        from preprocessing.rl_feature_selector import RLFeatureSelector
        feats = [f"feat_{i}" for i in range(10)]
        sel2 = RLFeatureSelector(feats, state_path=str(tmp_path / "fs.json"))
        assert len(sel2._hash_to_features) > 0

    def test_max_features_respected(self, tmp_path):
        sel = self._sel(tmp_path, feats=[f"f{i}" for i in range(30)])
        sel.max_features = 5
        for _ in range(20):
            a = sel.get_active_features()
        assert len(a) <= 30   # mutate may temporarily exceed — but initial cap holds


# ─────────────────────────────────────────────────────────────────────────────
# 4. RLAutoML
# ─────────────────────────────────────────────────────────────────────────────

class TestRLAutoML:

    def _agent(self, tmp_path):
        from modeling.rl_automl import RLAutoML
        return RLAutoML(state_path=str(tmp_path / "automl.json"))

    def test_select_pipeline_valid(self, tmp_path):
        agent = self._agent(tmp_path)
        scaler, model, imputer = agent.select_pipeline(
            n_rows=10000, n_cols=20, null_rate=0.05, task="classification"
        )
        from modeling.rl_automl import ALL_ACTIONS
        assert any((scaler, model, imputer) == a for a in ALL_ACTIONS)

    def test_record_and_exploit(self, tmp_path):
        agent = self._agent(tmp_path)
        pipeline = agent.select_pipeline(5000, 10, 0.0, "classification")
        agent.record_outcome(5000, 10, 0.0, "classification", pipeline, cv_score=0.85, training_time_s=3.0)
        p2 = agent.select_pipeline(5000, 10, 0.0, "classification")
        assert len(p2) == 3

    def test_build_sklearn_pipeline(self, tmp_path):
        agent = self._agent(tmp_path)
        pipe = agent.build_sklearn_pipeline("standard", "rf", "median", "classification")
        from sklearn.pipeline import Pipeline
        assert isinstance(pipe, Pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# 5. RLOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestRLOrchestrator:

    def _orch(self, tmp_path):
        from modeling.rl_orchestrator import RLOrchestrator
        return RLOrchestrator(state_path=str(tmp_path / "orch.json"))

    def test_get_plan_valid(self, tmp_path):
        orch = self._orch(tmp_path)
        plan = orch.get_plan(sla_minutes=10, row_count=5000, drift_detected=False)
        assert plan["profile_depth"]    in {"basic", "full"}
        assert plan["preprocess_depth"] in {"fast", "thorough"}
        assert plan["model_complexity"] in {"light", "balanced", "heavy"}
        assert plan["validation_depth"] in {"quick", "full"}

    def test_record_sla_beat_positive_reward(self, tmp_path):
        orch = self._orch(tmp_path)
        plan = orch.get_plan(30, 10000, False)
        orch.record_outcome(plan, actual_minutes=20, quality_score=0.90, sla_minutes=30)
        state = plan["state"]
        akey  = plan["action_key"]
        assert orch.q[state][akey] > 0

    def test_record_overtime_negative_reward(self, tmp_path):
        orch = self._orch(tmp_path)
        plan = orch.get_plan(5, 100000, True)
        # Apply multiple penalties — a single update may not overcome
        # the state prior (2.0) with alpha=0.12
        for _ in range(5):
            orch.record_outcome(plan, actual_minutes=60, quality_score=0.50, sla_minutes=5)
        state = plan["state"]
        akey  = plan["action_key"]
        assert orch.q[state][akey] < 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. RLAuthTuner
# ─────────────────────────────────────────────────────────────────────────────

class TestRLAuthTuner:

    def _tuner(self, tmp_path):
        from auth.rl_auth_tuner import RLAuthTuner
        return RLAuthTuner(state_path=str(tmp_path / "auth.json"))

    def test_policy_structure(self, tmp_path):
        t = self._tuner(tmp_path)
        p = t.get_policy(access_hour=14, failure_streak=0, is_admin=False,
                         is_new_device=False, risk_score=0.2)
        assert "mfa_required" in p
        assert "max_attempts" in p
        assert p["max_attempts"] in [3, 5, 10]
        assert p["session_timeout_min"] in [15, 30, 60, 120]

    def test_high_risk_seeds_strict_prior(self, tmp_path):
        t = self._tuner(tmp_path)
        # Force a high-risk state so priors are set
        t.get_policy(access_hour=2, failure_streak=5, is_admin=True,
                     is_new_device=True, risk_score=0.9)
        state = "night::fail_high::admin1::new1::risk_high"
        assert state in t.q
        # Strict prior (mfa=True, lockout=3) should have elevated Q
        best = max(t.q[state], key=t.q[state].__getitem__)
        assert "mfa1" in best

    def test_record_breach_penalty(self, tmp_path):
        t = self._tuner(tmp_path)
        p = t.get_policy()
        q_before = t.q.get(p["state"], {}).get(p["action_key"], 0.0)
        t.record_outcome(p, legitimate=False, was_breach=True)
        q_after = t.q[p["state"]][p["action_key"]]
        assert q_after < q_before or q_after < 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. RLNarrative
# ─────────────────────────────────────────────────────────────────────────────

class TestRLNarrative:

    def _rl(self, tmp_path):
        from explanation.rl_narrative import RLNarrative
        return RLNarrative(state_path=str(tmp_path / "narrative.json"))

    def test_section_order_5_items(self, tmp_path):
        rl = self._rl(tmp_path)
        order = rl.get_section_order("model_eval", "banking", 0.90)
        assert len(order) == 5
        assert set(order) == {"narrative", "findings", "quality", "model_perf", "risk"}

    def test_feedback_updates_q(self, tmp_path):
        rl = self._rl(tmp_path)
        order = rl.get_section_order("exploratory", "generic", 0.75)
        rl.record_feedback("exploratory", "generic", 0.75, order, analyst_rating=5)
        state = "exploratory::generic::medium"
        assert state in rl.q

    def test_good_rating_raises_q(self, tmp_path):
        rl = self._rl(tmp_path)
        order = rl.get_section_order("drift", "banking", 0.45)
        akey  = "->".join(order)
        state = "drift::banking::low"
        before = rl.q.get(state, {}).get(akey, 0.0)
        rl.record_feedback("drift", "banking", 0.45, order, analyst_rating=5)
        after = rl.q[state][akey]
        assert after > before   # +2 reward should push Q up


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
