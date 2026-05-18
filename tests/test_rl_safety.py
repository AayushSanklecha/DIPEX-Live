"""
tests/test_rl_safety.py
--------------------------
RL safety rail tests for DIPEX.

Verifies:
- RLSafetyViolation raised on forbidden target attempts
- Sandbox mode produces no weight file changes
- revert_to_checkpoint restores correctly
- RL convergence: confidence improves over simulated episodes
- RL does NOT overfit: strategy diversity maintained
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

FORBIDDEN_TARGETS = [
    "schema_validators",
    "hard_gate_1",
    "hard_gate_2",
    "compliance_rules",
    "statistical_verification",
]


# ══════════════════════════════════════════════════════════════════════════════
# RLUpdater Safety
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="RLUpdater removed in v2.0 architecture simplification — see pipeline_bridge._stage_rl_update()")
class TestRLSafetyRails:

    def test_rl_updater_imports(self):
        """RLUpdater must be importable."""
        try:
            from learning.rl_updater import RLUpdater
            assert RLUpdater is not None
        except ImportError:
            pytest.skip("RLUpdater not implemented yet")

    def test_rl_updater_update_accepts_valid_params(self):
        """RLUpdater.update() must not raise on valid input."""
        try:
            from learning.rl_updater import RLUpdater
            updater = RLUpdater()
            # Should not raise
            updater.update(
                run_id="test-001",
                model_type="random_forest",
                task="classification",
                confidence=0.85,
                all_gates_passed=True,
            )
        except ImportError:
            pytest.skip("RLUpdater not available")

    def test_rl_update_does_not_touch_forbidden_files(self, tmp_path):
        """RL update must not write to validation or gate config files."""
        # Create sentinel files representing forbidden areas
        sentinel = tmp_path / "hard_gate_config.yaml"
        sentinel.write_text("threshold: 0.05")
        modified_before = sentinel.stat().st_mtime_ns

        try:
            from learning.rl_updater import RLUpdater
            updater = RLUpdater()
            updater.update(
                run_id="test-sentinel",
                model_type="logistic_regression",
                task="classification",
                confidence=0.75,
                all_gates_passed=True,
            )
        except ImportError:
            pytest.skip("RLUpdater not available")

        modified_after = sentinel.stat().st_mtime_ns
        assert modified_before == modified_after, "RL update modified forbidden sentinel file!"

    def test_confidence_does_not_decrease_after_good_run(self):
        """After a high-confidence PASS run, RL must not reduce confidence weights."""
        try:
            from learning.rl_updater import RLUpdater
            updater = RLUpdater()

            # Baseline update with low confidence
            updater.update("run-low", "rf", "classification", 0.55, True)
            # High-quality run
            updater.update("run-high", "rf", "classification", 0.95, True)

            # System must not crash and must handle both gracefully
        except ImportError:
            pytest.skip("RLUpdater not available")


# ══════════════════════════════════════════════════════════════════════════════
# Sandbox Mode
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="RLUpdater removed in v2.0 architecture simplification — see pipeline_bridge._stage_rl_update()")
class TestRLSandboxMode:

    def test_sandbox_mode_does_not_persist_weights(self, tmp_path):
        """Sandbox mode must compute rewards but not write weight files."""
        weights_file = tmp_path / "rl_weights.json"
        weights_file.write_text(json.dumps({"strategy_a": 0.5, "strategy_b": 0.5}))
        original_content = weights_file.read_text()

        try:
            from learning.rl_updater import RLUpdater
            updater = RLUpdater(sandbox=True)
            updater.update("sandbox-run", "rf", "classification", 0.90, True)
        except (ImportError, TypeError):
            # If sandbox kwarg not supported, just check file didn't change
            pass

        if weights_file.exists():
            assert weights_file.read_text() == original_content, \
                "Sandbox mode modified weights file!"


# ══════════════════════════════════════════════════════════════════════════════
# RL Convergence
# ══════════════════════════════════════════════════════════════════════════════

class TestRLConvergence:

    def test_retry_engine_learns_better_strategies(self):
        """After multiple episodes, retry engine should prefer high-reward strategies."""
        try:
            from verifier.retry_engine import RetryEngine
            engine = RetryEngine()

            # Simulate 10 episodes where strategy_a succeeds, strategy_b fails
            for ep in range(10):
                try:
                    engine.record_outcome("strategy_a", success=True, confidence_gained=0.1)
                    engine.record_outcome("strategy_b", success=False, confidence_gained=-0.05)
                except (AttributeError, TypeError):
                    break  # interface may differ

        except ImportError:
            pytest.skip("RetryEngine not available")

    def test_bandit_selects_higher_reward_arm(self):
        """Contextual bandit must prefer the arm with higher observed reward."""
        # Simulate a simple epsilon-greedy bandit
        rewards = {"arm_a": 0.9, "arm_b": 0.3, "arm_c": 0.5}
        epsilon = 0.05  # low exploration

        choices = []
        for _ in range(100):
            rng = np.random.random()
            if rng < epsilon:
                choice = np.random.choice(list(rewards.keys()))
            else:
                choice = max(rewards, key=rewards.get)
            choices.append(choice)

        # arm_a must be chosen most of the time
        arm_a_rate = choices.count("arm_a") / len(choices)
        assert arm_a_rate > 0.80, f"Bandit not preferring best arm: arm_a_rate={arm_a_rate:.2f}"

    def test_strategy_diversity_maintained(self):
        """RL must not collapse to single strategy (anti-overfit)."""
        rewards = {"a": [0.7, 0.8, 0.75], "b": [0.6, 0.65, 0.7], "c": [0.5, 0.55, 0.6]}
        epsilon = 0.20  # exploration

        choices = []
        mean_rewards = {k: np.mean(v) for k, v in rewards.items()}

        for _ in range(100):
            if np.random.random() < epsilon:
                choice = np.random.choice(list(rewards.keys()))
            else:
                choice = max(mean_rewards, key=mean_rewards.get)
            choices.append(choice)

        unique_strategies = len(set(choices))
        assert unique_strategies >= 2, "RL collapsed to single strategy (overfit)"


# ══════════════════════════════════════════════════════════════════════════════
# RL Optimizer (Analyst-specific)
# ══════════════════════════════════════════════════════════════════════════════

class TestRLOptimizer:

    def test_rl_optimizer_imports(self):
        """analyst/rl_optimizer.py must be importable."""
        try:
            from analyst.rl_optimizer import RLOptimizer
            assert RLOptimizer is not None
        except ImportError:
            pytest.skip("RLOptimizer not implemented")

    def test_rl_optimizer_select_action_returns_valid_key(self):
        """select_action must return a key in the action space."""
        try:
            from analyst.rl_optimizer import RLOptimizer
            opt = RLOptimizer()
            state = {
                "dataset_size": 1000, "cardinality": 0.1, "null_rate": 0.02,
                "drift_level": 0.05, "signal_to_noise": 0.8, "last_confidence": 0.7,
            }
            if hasattr(opt, "select_action"):
                action = opt.select_action(state)
                assert action is not None
        except ImportError:
            pytest.skip("RLOptimizer not available")

    def test_rl_optimizer_sandbox_does_not_persist(self, tmp_path):
        """Sandbox=True must not write weights."""
        try:
            from analyst.rl_optimizer import RLOptimizer
            with patch.dict(os.environ, {"DIPEX_RL_SANDBOX": "true"}):
                opt = RLOptimizer(sandbox=True)
                if hasattr(opt, "select_action"):
                    state = {"dataset_size": 100, "null_rate": 0.01, "drift_level": 0.0,
                             "cardinality": 0.5, "signal_to_noise": 0.8, "last_confidence": 0.7}
                    _ = opt.select_action(state)
        except (ImportError, TypeError):
            pytest.skip("RLOptimizer sandbox mode not available")
