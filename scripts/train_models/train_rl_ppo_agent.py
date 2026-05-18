"""
scripts/train_models/train_rl_ppo_agent.py
--------------------------------------------
Google Colab-ready training script for the PPO pipeline agent.

Warm-start strategy: Use Thompson Sampling for first 20 real episodes,
then train PPO on collected trajectories.

For Colab: Run with GPU (A100 preferred for LSTM in drift models).
Training: 500 synthetic episodes; evaluates every 50 on held-out scenarios.
Val gate: episode reward mean >= 0.65 over last 50 episodes; std < 0.10
Saves: models/rl_ppo_agent.pt + training_curves.png

Usage:
  # In Colab: run all cells in order
  # Locally: python scripts/train_models/train_rl_ppo_agent.py
"""

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# !pip install numpy matplotlib torch

# ── Cell 2: Imports and config ────────────────────────────────────────────────
import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path (for Colab)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_rl_ppo")

SEED = 42
np.random.seed(SEED)

N_EPISODES     = 1000  # raised: 500 → 1000 for stable PPO convergence
EVAL_EVERY     = 50
EVAL_EPISODES  = 30    # raised: 20 → 30 for lower-variance gate estimate
OUTPUT_DIR     = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

# Reward gate
MIN_MEAN_REWARD = 0.65
MAX_STD_REWARD  = 0.09  # tightened: 0.10 → 0.09


# ── Cell 3: Synthetic environment ────────────────────────────────────────────
class SyntheticPipelineEnv:
    """
    Parameterized simulation of the ADAP pipeline.
    Used for warm-start training before real pipeline runs are available.

    The environment generates diverse scenarios and scores actions based on
    how well they match the 'optimal' strategy for that scenario.
    """

    SCENARIO_TYPES = ["clean_small", "dirty_large", "banking_aml", "healthcare_phi",
                      "high_null", "high_drift", "ecommerce_fraud", "time_series"]

    def __init__(self, rng_seed: int = SEED) -> None:
        self.rng = np.random.default_rng(rng_seed)
        self._scenario: Optional[Dict] = None

    def reset(self) -> Dict[str, Any]:
        """Generate a new random pipeline context."""
        scenario_type = self.rng.choice(self.SCENARIO_TYPES)

        if scenario_type == "clean_small":
            ctx = dict(n_rows=int(self.rng.integers(100, 5000)),
                       n_cols=int(self.rng.integers(5, 30)),
                       null_rate=float(self.rng.uniform(0.0, 0.05)),
                       anomaly_rate=float(self.rng.uniform(0.0, 0.02)),
                       drift_psi=float(self.rng.uniform(0.0, 0.05)),
                       data_health=float(self.rng.uniform(80, 100)),
                       domain="generic")
        elif scenario_type == "dirty_large":
            ctx = dict(n_rows=int(self.rng.integers(50000, 1000000)),
                       n_cols=int(self.rng.integers(20, 100)),
                       null_rate=float(self.rng.uniform(0.15, 0.40)),
                       anomaly_rate=float(self.rng.uniform(0.05, 0.15)),
                       drift_psi=float(self.rng.uniform(0.1, 0.5)),
                       data_health=float(self.rng.uniform(30, 60)),
                       domain="generic")
        elif scenario_type == "banking_aml":
            ctx = dict(n_rows=int(self.rng.integers(10000, 500000)),
                       n_cols=int(self.rng.integers(10, 40)),
                       null_rate=float(self.rng.uniform(0.0, 0.10)),
                       anomaly_rate=float(self.rng.uniform(0.01, 0.08)),
                       drift_psi=float(self.rng.uniform(0.0, 0.20)),
                       data_health=float(self.rng.uniform(60, 85)),
                       domain="banking")
        elif scenario_type == "high_null":
            ctx = dict(n_rows=int(self.rng.integers(1000, 50000)),
                       n_cols=int(self.rng.integers(10, 50)),
                       null_rate=float(self.rng.uniform(0.30, 0.70)),
                       anomaly_rate=float(self.rng.uniform(0.0, 0.05)),
                       drift_psi=float(self.rng.uniform(0.0, 0.15)),
                       data_health=float(self.rng.uniform(20, 50)),
                       domain="generic")
        elif scenario_type == "high_drift":
            ctx = dict(n_rows=int(self.rng.integers(5000, 100000)),
                       n_cols=int(self.rng.integers(8, 30)),
                       null_rate=float(self.rng.uniform(0.02, 0.10)),
                       anomaly_rate=float(self.rng.uniform(0.05, 0.20)),
                       drift_psi=float(self.rng.uniform(0.25, 0.80)),
                       data_health=float(self.rng.uniform(40, 70)),
                       domain="finance")
        elif scenario_type == "healthcare_phi":
            ctx = dict(n_rows=int(self.rng.integers(5000, 200000)),
                       n_cols=int(self.rng.integers(15, 60)),
                       null_rate=float(self.rng.uniform(0.05, 0.20)),    # PHI often sparse
                       anomaly_rate=float(self.rng.uniform(0.01, 0.06)),
                       drift_psi=float(self.rng.uniform(0.0, 0.15)),
                       data_health=float(self.rng.uniform(55, 80)),
                       domain="healthcare")
        elif scenario_type == "ecommerce_fraud":
            ctx = dict(n_rows=int(self.rng.integers(20000, 500000)),
                       n_cols=int(self.rng.integers(10, 35)),
                       null_rate=float(self.rng.uniform(0.0, 0.08)),
                       anomaly_rate=float(self.rng.uniform(0.003, 0.05)),  # fraud is rare
                       drift_psi=float(self.rng.uniform(0.05, 0.30)),      # consumer behaviour drifts
                       data_health=float(self.rng.uniform(65, 90)),
                       domain="ecommerce")
        elif scenario_type == "time_series":
            ctx = dict(n_rows=int(self.rng.integers(1000, 50000)),
                       n_cols=int(self.rng.integers(3, 20)),
                       null_rate=float(self.rng.uniform(0.0, 0.10)),
                       anomaly_rate=float(self.rng.uniform(0.02, 0.10)),
                       drift_psi=float(self.rng.uniform(0.10, 0.50)),     # temporal drift expected
                       data_health=float(self.rng.uniform(50, 85)),
                       domain="finance")
        else:
            ctx = dict(n_rows=int(self.rng.integers(1000, 100000)),
                       n_cols=int(self.rng.integers(5, 60)),
                       null_rate=float(self.rng.uniform(0.0, 0.20)),
                       anomaly_rate=float(self.rng.uniform(0.0, 0.10)),
                       drift_psi=float(self.rng.uniform(0.0, 0.30)),
                       data_health=float(self.rng.uniform(40, 90)),
                       domain="generic")

        self._scenario = {"type": scenario_type, "context": ctx}
        return ctx

    def step(self, action_dict: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Score the action given the current scenario.
        Returns (reward, done=True for episodic).
        """
        ctx = self._scenario["context"]
        scenario_type = self._scenario["type"]

        # Base reward from data health
        reward = 0.30 * (ctx["data_health"] / 100.0)

        # Reward good action choices for the scenario
        # 1. Imputation strategy
        if ctx["null_rate"] > 0.20 and action_dict.get("imputation") in ("mice", "knn"):
            reward += 0.15  # MICE/KNN good for high-null
        elif ctx["null_rate"] < 0.05 and action_dict.get("imputation") == "median":
            reward += 0.10  # Median fine for low-null

        # 2. CV strategy
        if scenario_type in ("banking_aml", "high_drift") and action_dict.get("cv_strategy") == "temporal":
            reward += 0.15  # Temporal CV good for time-sensitive data
        elif action_dict.get("cv_strategy") == "stratified":
            reward += 0.08  # Stratified always decent

        # 3. Outlier policy
        if ctx["anomaly_rate"] > 0.10 and action_dict.get("outlier_policy") == "quarantine":
            reward += 0.10  # Quarantine high anomaly rates
        elif ctx["anomaly_rate"] < 0.03 and action_dict.get("outlier_policy") == "winsorize":
            reward += 0.08  # Winsorize low anomaly rates

        # 4. Model complexity
        if ctx["n_rows"] > 50000 and action_dict.get("model_complexity") == "high":
            reward += 0.10  # High complexity ok for big data
        elif ctx["n_rows"] < 1000 and action_dict.get("model_complexity") == "low":
            reward += 0.10  # Low complexity for small data (avoid overfitting)

        # 5. Confidence threshold
        conf = float(action_dict.get("confidence_threshold", 0.70))
        if ctx["data_health"] > 80 and conf >= 0.70:
            reward += 0.08  # High confidence for clean data
        elif ctx["data_health"] < 50 and conf <= 0.55:
            reward += 0.08  # Lower confidence for dirty data

        # Add small noise
        noise = float(self.rng.normal(0, 0.05))
        reward = float(np.clip(reward + noise, 0.0, 1.0))

        return reward, True  # Episodic


# ── Cell 4: Training loop ────────────────────────────────────────────────────

def train(n_episodes: int = N_EPISODES) -> Dict[str, Any]:
    """Run PPO training on synthetic pipeline environment."""
    from learning.rl_agent.agent import PPOAgent
    from learning.rl_agent.state_encoder import StateEncoder

    env    = SyntheticPipelineEnv(rng_seed=SEED)
    agent  = PPOAgent.from_config({})
    encoder = StateEncoder()

    # Override shadow mode for synthetic training (we can train immediately)
    agent._episode_count = 20  # Skip shadow mode for Colab training

    episode_rewards = []
    training_metrics_log = []

    logger.info("Starting PPO training: %d episodes", n_episodes)

    for ep in range(n_episodes):
        context = env.reset()
        action = agent.recommend(context, greedy=False)

        # Simulate pipeline outcome
        reward, done = env.step(action.to_dict())

        # Record outcome (simulated)
        result_summary = {
            "gate_decision": "PASS" if reward > 0.5 else "WARN",
            "model_metrics": {"roc_auc": min(reward + 0.1, 1.0)},
            "quarantine_rows": 0,
            "retry_count": 0,
        }
        analytics = {"data_health_score": context["data_health"]}
        outcome = agent.record_outcome(result_summary, analytics)

        ep_reward = outcome.get("reward", reward)
        episode_rewards.append(ep_reward)

        if (ep + 1) % EVAL_EVERY == 0:
            recent_rewards = episode_rewards[-EVAL_EVERY:]
            mean_r = float(np.mean(recent_rewards))
            std_r  = float(np.std(recent_rewards))
            logger.info(
                "Episode %d/%d | RewardMean=%.4f | Std=%.4f | Training=%s",
                ep + 1, n_episodes, mean_r, std_r,
                outcome.get("training_metrics", {})
            )
            training_metrics_log.append({
                "episode": ep + 1,
                "mean_reward": round(mean_r, 4),
                "std_reward": round(std_r, 4),
            })

    # Final evaluation
    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        context = env.reset()
        action = agent.recommend(context, greedy=True)
        reward, _ = env.step(action.to_dict())
        eval_rewards.append(reward)

    eval_mean = float(np.mean(eval_rewards))
    eval_std  = float(np.std(eval_rewards))

    logger.info("Final eval: mean=%.4f, std=%.4f", eval_mean, eval_std)

    # Quality gate
    passed = eval_mean >= MIN_MEAN_REWARD and eval_std <= MAX_STD_REWARD
    logger.info("Quality gate: %s (mean>=%.2f: %s, std<=%.2f: %s)",
                "PASS" if passed else "FAIL",
                MIN_MEAN_REWARD, eval_mean >= MIN_MEAN_REWARD,
                MAX_STD_REWARD, eval_std <= MAX_STD_REWARD)

    # Save
    agent.policy.save()
    agent.value.save()

    # Save training curve
    try:
        import matplotlib.pyplot as plt
        episodes = [m["episode"] for m in training_metrics_log]
        means    = [m["mean_reward"] for m in training_metrics_log]
        stds     = [m["std_reward"] for m in training_metrics_log]
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, means, "b-", label="Mean Reward")
        plt.fill_between(episodes,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.3, label="±1 Std")
        plt.axhline(y=MIN_MEAN_REWARD, color="r", linestyle="--", label=f"Gate ({MIN_MEAN_REWARD})")
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.title("PPO Agent Training Curves")
        plt.legend()
        plt.savefig(str(OUTPUT_DIR / "rl_training_curves.png"), dpi=100, bbox_inches="tight")
        plt.close()
        logger.info("Training curve saved to models/rl_training_curves.png")
    except Exception as e:
        logger.warning("Could not save training curve: %s", e)

    return {
        "n_episodes": n_episodes,
        "eval_mean_reward": round(eval_mean, 4),
        "eval_std_reward": round(eval_std, 4),
        "passed": passed,
        "training_log": training_metrics_log,
    }


if __name__ == "__main__":
    results = train(n_episodes=N_EPISODES)
    print("\n" + "="*60)
    print("PPO AGENT TRAINING RESULTS")
    print("="*60)
    print(f"  Eval Mean Reward : {results['eval_mean_reward']}")
    print(f"  Eval Std Reward  : {results['eval_std_reward']}")
    print(f"  Quality Gate     : {'PASS ✅' if results['passed'] else 'FAIL ❌'}")
    print("="*60)
