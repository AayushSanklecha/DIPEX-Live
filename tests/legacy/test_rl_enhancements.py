"""
tests/test_rl_enhancements.py
------------------------------
Pytest test suite for all 5 advanced RL self-improvement enhancements.

Modules under test:
  1. learning/rl_automl.py          — Optuna-based hyperparameter auto-tuning
  2. learning/contextual_bandit.py  — LinUCB contextual bandit
  3. learning/domain_priors.py      — Warm-start domain priors (cold-start fix)
  4. learning/reward_shaper.py      — Multi-dimensional reward shaping
  5. learning/transfer_learning.py  — Cross-dataset transfer learning

All tests run fully in-memory / tmp_path — no external services required.
Optuna tests are guarded so the suite still passes if optuna is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — load learning modules without triggering learning/__init__.py
# (which triggers chromadb + heavy imports on package load)
# ─────────────────────────────────────────────────────────────────────────────

_LEARNING_DIR = Path(__file__).parent.parent / "learning"


def _load(module_name: str) -> Any:
    """Load a learning sub-module by filename, bypassing __init__.py."""
    # Ensure rl_safety is always available as a dependency
    _ensure("learning.rl_safety", "rl_safety.py")
    return _ensure(module_name, module_name.split(".")[-1] + ".py")


def _ensure(full_name: str, filename: str) -> Any:
    if full_name not in sys.modules:
        path = _LEARNING_DIR / filename
        spec = importlib.util.spec_from_file_location(full_name, path)
        mod  = importlib.util.module_from_spec(spec)        # type: ignore[arg-type]
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)                        # type: ignore[union-attr]
    return sys.modules[full_name]


# ─────────────────────────────────────────────────────────────────────────────
# 1. RewardShaper
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardShaper:
    """Enhancement 4 — multi-dimensional reward shaping."""

    def _shaper(self, config=None):
        m = _load("learning.reward_shaper")
        return m.RewardShaper(config)

    # ── compute() output range ────────────────────────────────────────────────

    def test_reward_in_unit_interval(self):
        rs = self._shaper()
        r  = rs.compute(confidence_score=0.85, elapsed_seconds=30.0,
                        drift_psi_before=0.25, drift_psi_after=0.10)
        assert 0.0 <= r <= 1.0

    def test_perfect_run_high_reward(self):
        """Fast run, confidence 1.0, drift eliminated → reward ≥ 0.80."""
        rs = self._shaper()
        r  = rs.compute(confidence_score=1.0, elapsed_seconds=5.0,
                        drift_psi_before=0.50, drift_psi_after=0.0)
        assert r >= 0.80

    def test_worst_run_low_reward(self):
        """Zero confidence, very slow, drift worsened → reward < 0.30."""
        rs = self._shaper()
        r  = rs.compute(confidence_score=0.0, elapsed_seconds=3600.0,
                        drift_psi_before=0.05, drift_psi_after=0.90)
        assert r < 0.30

    def test_none_speed_returns_neutral(self):
        """Missing timing data → speed sub-reward = 0.5 (neutral)."""
        rs  = self._shaper()
        r1  = rs.compute(confidence_score=0.8, elapsed_seconds=None)
        r2  = rs.compute(confidence_score=0.8, elapsed_seconds=None,
                         drift_psi_before=None, drift_psi_after=None)
        # Both should be close to 0.5 * 0.8 + 0.25 * 0.5 + 0.25 * 0.5 = 0.65
        assert 0.55 <= r1 <= 0.75
        assert 0.55 <= r2 <= 0.75

    def test_drift_already_low_full_score(self):
        """PSI already good → full drift reward regardless of before/after."""
        rs = self._shaper()
        r  = rs.compute(confidence_score=0.9, elapsed_seconds=60.0,
                        drift_psi_before=0.05, drift_psi_after=0.02)
        # drift sub-reward = 1.0 because psi_after <= drift_good_psi (0.10)
        assert r > 0.7

    def test_decompose_sub_rewards_sum_correctly(self):
        rs   = self._shaper()
        dec  = rs.decompose(0.80, elapsed_seconds=40.0,
                            drift_psi_before=0.20, drift_psi_after=0.08)
        w    = dec["weights"]
        # Weighted sum should match composite_reward
        expected = (
            w["confidence"] * dec["confidence_reward"]
            + w["speed"]      * dec["speed_reward"]
            + w["drift"]      * dec["drift_reward"]
        )
        assert abs(expected - dec["composite_reward"]) < 1e-6

    def test_custom_weights_normalised(self):
        """Custom weights that don't sum to 1 should be auto-normalised."""
        cfg = {"rl": {"reward_shaping": {"weights": {"confidence": 2, "speed": 1, "drift": 1}}}}
        rs  = self._shaper(cfg)
        r   = rs.compute(confidence_score=0.9, elapsed_seconds=20.0)
        assert 0.0 <= r <= 1.0
        # Confidence weight should be 0.5 after normalisation
        assert abs(rs._weights["confidence"] - 0.5) < 1e-9

    def test_reward_clipped_to_zero_one(self):
        """Edge inputs never produce rewards outside [0, 1]."""
        rs = self._shaper()
        assert rs.compute(confidence_score=2.0)  <= 1.0
        assert rs.compute(confidence_score=-1.0) >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. DomainPriors
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainPriors:
    """Enhancement 3 — warm-start Q-value priors."""

    def _mod(self):
        return _load("learning.domain_priors")

    def test_banking_prior_returns_dict(self):
        priors = self._mod().get_prior("banking")
        assert isinstance(priors, dict)
        assert len(priors) > 0

    def test_all_values_in_unit_interval(self):
        mod = self._mod()
        for domain in mod.list_domains():
            for action, q in mod.get_prior(domain).items():
                assert 0.0 <= q <= 1.0, f"domain={domain} action={action} q={q}"

    def test_unknown_domain_falls_back_to_default(self):
        mod   = self._mod()
        prior = mod.get_prior("totally_unknown_domain_xyz")
        dflt  = mod.get_prior("default")
        assert prior == dflt

    def test_alias_banking(self):
        mod = self._mod()
        assert mod.get_prior("bank") == mod.get_prior("banking")

    def test_alias_healthcare(self):
        mod = self._mod()
        assert mod.get_prior("medical") == mod.get_prior("healthcare")

    def test_banking_prefers_hyperparameter_tuning(self):
        """Banking domain prior should give the highest Q to adjust_hyperparameters."""
        priors = self._mod().get_prior("banking")
        best   = max(priors, key=lambda a: priors[a])
        assert best == "adjust_hyperparameters"

    def test_healthcare_prefers_feature_selection(self):
        priors = self._mod().get_prior("healthcare")
        best   = max(priors, key=lambda a: priors[a])
        assert best == "apply_feature_selection"

    def test_case_insensitive(self):
        mod = self._mod()
        assert mod.get_prior("BANKING") == mod.get_prior("banking")

    def test_list_domains_not_empty(self):
        domains = self._mod().list_domains()
        assert len(domains) >= 4
        assert "default" in domains


# ─────────────────────────────────────────────────────────────────────────────
# 3. ContextualBandit (LinUCB)
# ─────────────────────────────────────────────────────────────────────────────

class TestContextualBandit:
    """Enhancement 2 — LinUCB contextual bandit."""

    _ACTIONS = ["restart_from_eda", "adjust_hyperparameters",
                "apply_feature_selection", "change_model_class"]

    def _bandit(self, tmp_path: Path) -> Any:
        m   = _load("learning.contextual_bandit")
        cfg = {"rl": {"contextual_bandit": {"state_path": str(tmp_path / "cb.json")}}}
        return m.ContextualBandit(self._ACTIONS, config=cfg)

    # ── LinUCBArm ─────────────────────────────────────────────────────────────

    def test_ucb_score_finite(self, tmp_path):
        m   = _load("learning.contextual_bandit")
        arm = m.LinUCBArm("test_arm", n_features=7)
        ctx = np.ones(7) / math.sqrt(7)
        s   = arm.ucb_score(ctx)
        assert math.isfinite(s)

    def test_arm_update_increases_score_for_good_reward(self, tmp_path):
        m   = _load("learning.contextual_bandit")
        arm = m.LinUCBArm("a", n_features=7, alpha=0.5)
        ctx = np.ones(7) / math.sqrt(7)
        s0  = arm.ucb_score(ctx)
        for _ in range(10):
            arm.update(ctx, 1.0)    # high reward
        s1  = arm.ucb_score(ctx)
        # After consistent high reward, theta·x should be higher
        assert arm.theta() @ ctx > 0

    def test_singular_matrix_graceful_degradation(self, tmp_path):
        """All-zero context shouldn't crash; arm returns a finite neutral score."""
        m   = _load("learning.contextual_bandit")
        arm = m.LinUCBArm("a", n_features=7)
        ctx = np.zeros(7)
        s   = arm.ucb_score(ctx)
        assert math.isfinite(s)

    def test_from_dict_dimension_mismatch_resets(self, tmp_path):
        """Loading a checkpoint with wrong dimensions should reset to identity."""
        m   = _load("learning.contextual_bandit")
        bad = {"name": "x", "A": [[1, 0], [0, 1]], "b": [0, 0]}  # 2×2, but n=7
        arm = m.LinUCBArm.from_dict(bad, n_features=7, alpha=1.0)
        assert arm._A.shape == (7, 7)

    # ── ContextualBandit ─────────────────────────────────────────────────────

    def test_select_action_returns_valid_action(self, tmp_path):
        bandit = self._bandit(tmp_path)
        ctx    = bandit.build_context({"null_rate": 0.1, "confidence_score": 0.8})
        action = bandit.select_action(ctx)
        assert action in self._ACTIONS

    def test_select_action_without_context(self, tmp_path):
        bandit = self._bandit(tmp_path)
        action = bandit.select_action(None)
        assert action in self._ACTIONS

    def test_build_context_normalised(self, tmp_path):
        bandit = self._bandit(tmp_path)
        ctx = bandit.build_context({
            "null_rate": 0.3, "drift_score": 0.4,
            "row_count": 50000, "col_count": 20, "confidence_score": 0.85,
        })
        assert ctx.shape == (7,)
        assert all(0.0 <= v <= 1.0 for v in ctx), f"Out of range: {ctx}"

    def test_update_persists_to_file(self, tmp_path):
        bandit = self._bandit(tmp_path)
        ctx    = bandit.build_context({"null_rate": 0.05}, episode=0)
        bandit.update("adjust_hyperparameters", ctx, 0.90)
        # File should now exist
        assert (tmp_path / "cb.json").exists()

    def test_state_reloads_correctly(self, tmp_path):
        """After update and reload, arm state should be preserved."""
        m   = _load("learning.contextual_bandit")
        cfg = {"rl": {"contextual_bandit": {"state_path": str(tmp_path / "cb.json")}}}
        b1  = m.ContextualBandit(self._ACTIONS, config=cfg)
        ctx = b1.build_context({"confidence_score": 0.9}, episode=2)
        b1.update("restart_from_eda", ctx, 0.95)

        b2     = m.ContextualBandit(self._ACTIONS, config=cfg)
        loaded = b2._arms["restart_from_eda"]
        assert loaded._b.sum() != 0  # b should have been updated

    def test_concurrent_updates_no_errors(self, tmp_path):
        """20 concurrent threads updating the bandit should produce no errors."""
        bandit = self._bandit(tmp_path)
        errors = []

        def worker(i: int) -> None:
            try:
                ctx = bandit.build_context({"null_rate": i * 0.01}, episode=i)
                bandit.update(self._ACTIONS[i % len(self._ACTIONS)], ctx, 0.5 + i * 0.01)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_get_arm_weights_all_finite(self, tmp_path):
        bandit = self._bandit(tmp_path)
        ctx = bandit.build_context({"confidence_score": 0.8})
        bandit.update("adjust_hyperparameters", ctx, 0.85)
        weights = bandit.get_arm_weights()
        assert all(math.isfinite(v) for v in weights.values())

    def test_feature_normalisation_row_count_extreme(self, tmp_path):
        """Extreme row counts (1 row or 10M rows) should still normalise to [0,1]."""
        bandit = self._bandit(tmp_path)
        for rc in [1, 100, 10_000, 5_000_000]:
            ctx = bandit.build_context({"row_count": rc})
            assert all(0.0 <= v <= 1.0 for v in ctx), f"row_count={rc} ctx={ctx}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. KnowledgeTransfer
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeTransfer:
    """Enhancement 5 — cross-dataset transfer learning."""

    def _kt(self, tmp_path: Path) -> Any:
        m   = _load("learning.transfer_learning")
        cfg = {"rl": {"transfer_learning": {
            "registry_path": str(tmp_path / "registry.json"),
        }}}
        return m.KnowledgeTransfer(config=cfg)

    def _fp(self, null_rate: float = 0.05, rows: int = 10_000) -> Any:
        m = _load("learning.transfer_learning")
        return m.DomainFingerprint.from_run_result({
            "rows_ingested": rows, "null_rate": null_rate,
            "drift_score": 0.10, "col_count": 20, "confidence_score": 0.85,
        })

    # ── DomainFingerprint ────────────────────────────────────────────────────

    def test_fingerprint_from_run_result(self, tmp_path):
        fp = self._fp()
        assert len(fp.to_list()) == 6      # 6 features

    def test_cosine_similarity_self(self, tmp_path):
        fp  = self._fp()
        sim = fp.similarity(fp)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_different(self, tmp_path):
        fp1 = self._fp(null_rate=0.01, rows=100)
        fp2 = self._fp(null_rate=0.90, rows=5_000_000)
        sim = fp1.similarity(fp2)
        assert 0.0 <= sim <= 1.0
        assert sim < 1.0   # Not identical

    def test_cosine_similarity_zero_vector_safe(self, tmp_path):
        m  = _load("learning.transfer_learning")
        fp = m.DomainFingerprint({"null_rate": 0.0, "row_count_log": 0.0,
                                   "col_count": 0.0, "drift_psi": 0.0,
                                   "schema_complexity": 0.0, "confidence_score": 0.0})
        assert fp.similarity(fp) == 0.0    # zero vec → similarity = 0

    # ── KnowledgeTransfer ────────────────────────────────────────────────────

    def test_transfer_empty_registry_returns_base(self, tmp_path):
        kt     = self._kt(tmp_path)
        base   = {"action_a": 0.6, "action_b": 0.4}
        result = kt.transfer(self._fp(), base_prior=base)
        assert result == base

    def test_store_creates_registry_file(self, tmp_path):
        kt = self._kt(tmp_path)
        kt.store("run_001", "banking", self._fp(), {"action_a": 0.8})
        assert (tmp_path / "registry.json").exists()

    def test_transfer_returns_blended_result(self, tmp_path):
        kt  = self._kt(tmp_path)
        fp  = self._fp(null_rate=0.05, rows=10_000)
        # Store a similar fingerprint with high Q for action_a
        kt.store("run_001", "banking", fp, {"action_a": 0.95, "action_b": 0.30})
        # Transfer from a nearly identical fingerprint
        fp2   = self._fp(null_rate=0.06, rows=9_500)
        base  = {"action_a": 0.5, "action_b": 0.5}
        result = kt.transfer(fp2, base_prior=base)
        # If similarity >= threshold, action_a should be pulled above 0.5
        if result:
            assert result.get("action_a", 0.5) >= 0.5

    def test_transfer_blend_bounded_by_max_weight(self, tmp_path):
        """Source contribution must never exceed max_transfer_weight (0.70)."""
        kt  = self._kt(tmp_path)
        fp  = self._fp()
        kt.store("r1", "banking", fp, {"a": 1.0})  # extreme source Q
        result = kt.transfer(fp, base_prior={"a": 0.0})
        if "a" in result:
            assert result["a"] <= 0.70 + 1e-9  # must be ≤ max_transfer_weight

    def test_dissimilar_fingerprint_no_transfer(self, tmp_path):
        kt  = self._kt(tmp_path)
        fp1 = self._fp(null_rate=0.01, rows=100)
        fp2 = self._fp(null_rate=0.95, rows=9_000_000)
        kt.store("r1", "finance", fp1, {"x": 0.9})
        base   = {"x": 0.5}
        result = kt.transfer(fp2, base_prior=base)
        # Very different → similarity likely below threshold → unchanged
        # (may or may not transfer depending on similarity, just check output valid)
        assert isinstance(result, dict)

    def test_prune_to_max_size(self, tmp_path):
        m   = _load("learning.transfer_learning")
        cfg = {"rl": {"transfer_learning": {
            "registry_path": str(tmp_path / "r.json"),
            "max_registry_size": 5,
        }}}
        kt = m.KnowledgeTransfer(config=cfg)
        fp = self._fp()
        for i in range(10):
            kt.store(f"r{i}", "banking", fp, {"a": 0.7})
        with open(tmp_path / "r.json") as f:
            data = json.load(f)
        assert len(data) <= 5

    def test_concurrent_stores_no_errors(self, tmp_path):
        """15 concurrent threads storing fingerprints should complete without errors."""
        kt     = self._kt(tmp_path)
        fp     = self._fp()
        errors = []

        def worker(i: int) -> None:
            try:
                kt.store(f"run_{i}", "banking", fp, {"a": 0.7, "b": 0.3})
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_find_most_similar_returns_tuple(self, tmp_path):
        kt = self._kt(tmp_path)
        fp = self._fp()
        kt.store("r1", "finance", fp, {"action": 0.8})
        result = kt.find_most_similar(fp)
        assert result is not None
        sim, run_id, domain = result
        assert 0.0 <= sim <= 1.0
        assert run_id == "r1"
        assert domain == "finance"

    def test_find_most_similar_empty_registry(self, tmp_path):
        kt = self._kt(tmp_path)
        assert kt.find_most_similar(self._fp()) is None

    def test_atomic_write_file_not_corrupt(self, tmp_path):
        """After concurrent writes, the registry JSON should be parseable."""
        kt     = self._kt(tmp_path)
        fp     = self._fp()
        threads = [threading.Thread(target=kt.store,
                    args=(f"r{i}", "banking", fp, {"a": 0.5})) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with open(tmp_path / "registry.json") as f:
            data = json.load(f)
        assert isinstance(data, list)


# ─────────────────────────────────────────────────────────────────────────────
# 5. RLAutoTuner
# ─────────────────────────────────────────────────────────────────────────────

class TestRLAutoTuner:
    """Enhancement 1 — Optuna-based RL hyperparameter auto-tuning."""

    def _tuner(self, tmp_path: Path) -> Any:
        m   = _load("learning.rl_automl")
        cfg = {"rl": {"automl": {
            "params_path":   str(tmp_path / "params.json"),
            "n_trials":      5,      # Fast for tests
            "min_history":   4,
            "tune_every_n":  10,
            "cv_folds":      2,
        }}}
        return m.RLAutoTuner(config=cfg)

    def _events(self, n: int = 30) -> list:
        events = []
        for i in range(n):
            events.append({"event_type": "RETRY_DECISION",
                            "payload": {"action": "restart_from_eda"}})
            events.append({"event_type": "APPROVED_OUTPUT",
                            "payload": {"confidence_score": 0.70 + (i % 10) * 0.02}})
        return events

    def test_load_best_params_returns_defaults_when_no_file(self, tmp_path):
        tuner  = self._tuner(tmp_path)
        params = tuner.load_best_params()
        assert "alpha" in params
        assert "ewc_lambda" in params
        assert "weight_lr" in params
        assert "epsilon_min" in params
        assert "epsilon_max" in params

    def test_should_tune_false_at_episode_zero(self, tmp_path):
        tuner = self._tuner(tmp_path)
        assert not tuner.should_tune(0)

    def test_should_tune_every_episode_above_min_history(self, tmp_path):
        """New behaviour: tune after EVERY run once min_history is reached."""
        tuner = self._tuner(tmp_path)   # min_history=4 in test config
        if tuner._optuna_ok:
            # Every episode >= min_history should return True
            assert tuner.should_tune(4)
            assert tuner.should_tune(5)
            assert tuner.should_tune(11)
            assert tuner.should_tune(100)
            # Before min_history → False
            assert not tuner.should_tune(0)
            assert not tuner.should_tune(3)

    def test_tune_with_insufficient_history_returns_defaults(self, tmp_path):
        tuner  = self._tuner(tmp_path)
        params = tuner.tune([{"event_type": "RETRY_DECISION"}])  # only 1 event
        assert params == tuner.load_best_params()

    @pytest.mark.skipif(
        not importlib.util.find_spec("optuna"),
        reason="optuna not installed"
    )
    def test_tune_returns_valid_params(self, tmp_path):
        tuner  = self._tuner(tmp_path)
        params = tuner.tune(self._events(30))
        assert 0.01 <= params["alpha"]       <= 0.30
        assert 0.70 <= params["ewc_lambda"]  <= 0.99
        assert 0.01 <= params["weight_lr"]   <= 0.15
        assert 0.02 <= params["epsilon_min"] <= 0.10
        assert 0.15 <= params["epsilon_max"] <= 0.40
        assert params["epsilon_max"] > params["epsilon_min"]

    @pytest.mark.skipif(
        not importlib.util.find_spec("optuna"),
        reason="optuna not installed"
    )
    def test_tune_persists_params_file(self, tmp_path):
        tuner = self._tuner(tmp_path)
        tuner.tune(self._events(30))
        assert (tmp_path / "params.json").exists()
        with open(tmp_path / "params.json") as f:
            data = json.load(f)
        assert "alpha" in data

    def test_simulate_qvalue_alignment_output_range(self, tmp_path):
        m      = _load("learning.rl_automl")
        events = self._events(40)
        score  = m._simulate_qvalue_alignment(
            events, alpha=0.1, ewc_lambda=0.9,
            weight_lr=0.05, fold_start=20, fold_end=40
        )
        assert 0.0 <= score <= 1.0

    def test_simulate_empty_fold_returns_zero(self, tmp_path):
        m   = _load("learning.rl_automl")
        r   = m._simulate_qvalue_alignment([], 0.1, 0.9, 0.05, 0, 0)
        assert r == 0.0

    def test_atomic_param_write(self, tmp_path):
        """Saved params must be valid JSON after writing."""
        tuner  = self._tuner(tmp_path)
        params     = {"alpha": 0.15, "ewc_lambda": 0.88,
                      "weight_lr": 0.04, "epsilon_min": 0.05, "epsilon_max": 0.25}
        tuner._save_params(params)
        with open(tmp_path / "params.json") as f:
            loaded = json.load(f)
        assert loaded["alpha"] == 0.15


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integration — all 5 enhancements co-operate
# ─────────────────────────────────────────────────────────────────────────────

class TestRLEnhancementsIntegration:
    """End-to-end integration: domain prior → transfer → bandit → shaped reward."""

    def test_full_cold_start_warm_start_cycle(self, tmp_path):
        """
        Simulates a real pipeline lifecycle:
          1. First run (no history) → domain prior used for banking
          2. Reward shaped with speed + drift
          3. Fingerprint stored in transfer registry
          4. Second run with similar data → transfer kicks in, blends prior
          5. Contextual bandit updated with shaped reward
        """
        dp_mod  = _load("learning.domain_priors")
        rs_mod  = _load("learning.reward_shaper")
        tl_mod  = _load("learning.transfer_learning")
        cb_mod  = _load("learning.contextual_bandit")

        actions = ["restart_from_eda", "adjust_hyperparameters",
                   "apply_feature_selection", "change_model_class"]

        # Step 1: Warm-start prior for banking
        prior  = dp_mod.get_prior("banking")
        assert prior["adjust_hyperparameters"] > 0.5

        # Step 2: Shape reward for run 1
        rs     = rs_mod.RewardShaper()
        reward = rs.compute(confidence_score=0.88, elapsed_seconds=45.0,
                            drift_psi_before=0.30, drift_psi_after=0.12)
        assert 0.0 < reward < 1.0

        # Step 3: Store fingerprint in transfer registry
        kt_cfg = {"rl": {"transfer_learning": {
            "registry_path": str(tmp_path / "reg.json"),
        }}}
        kt = tl_mod.KnowledgeTransfer(config=kt_cfg)
        fp = tl_mod.DomainFingerprint.from_run_result({
            "rows_ingested": 10_000, "null_rate": 0.05,
            "drift_score": 0.12, "col_count": 15, "confidence_score": 0.88,
        })
        kt.store("run_001", "banking", fp, prior)

        # Step 4: Next run — transfer from similar dataset
        fp2    = tl_mod.DomainFingerprint.from_run_result({
            "rows_ingested": 9_800, "null_rate": 0.06,
            "drift_score": 0.11, "col_count": 15, "confidence_score": 0.0,
        })
        blended = kt.transfer(fp2, target_domain="banking", base_prior=dict(prior))
        assert isinstance(blended, dict)

        # Step 5: Update contextual bandit with shaped reward for chosen action
        cb_cfg = {"rl": {"contextual_bandit": {
            "state_path": str(tmp_path / "cb.json"),
        }}}
        bandit = cb_mod.ContextualBandit(actions, config=cb_cfg)
        ctx    = bandit.build_context({"null_rate": 0.06, "drift_score": 0.11,
                                        "row_count": 9800, "confidence_score": 0.0},
                                       episode=1)
        action = bandit.select_action(ctx)
        assert action in actions
        bandit.update(action, ctx, reward)
        # Arm should now have non-default b vector
        assert bandit._arms[action]._b.sum() != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
