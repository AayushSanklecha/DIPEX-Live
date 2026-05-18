"""
learning/rl_agent/
------------------
PPO Actor-Critic Reinforcement Learning Agent for ADAP pipeline strategy optimization.

Replaces the Thompson Sampling bandit with a full PPO agent that:
  - Encodes pipeline state as a 12D normalized vector
  - Uses actor-critic networks to select 8-axis pipeline actions
  - Trains via PPO clipped objective with GAE advantage estimation
  - Falls back to Thompson Sampling until 20 real episodes are collected
"""
from .agent import PPOAgent
from .state_encoder import StateEncoder
from .action_space import ActionSpace

__all__ = ["PPOAgent", "StateEncoder", "ActionSpace"]
