"""
auth/rl_auth_tuner.py
----------------------
Production-grade RL Risk-Based Authentication Tightener.

Purpose
-------
The auth layer currently uses static MFA/lockout rules.
This RL agent learns which access patterns correlate with security
incidents and dynamically adjusts:
  - MFA required: True / False
  - Max failed attempts before lockout: 3, 5, 10
  - Session timeout (minutes): 15, 30, 60, 120

Architecture
------------
State  : (access_hour_bucket, failure_streak, is_admin, is_new_device, risk_score_bucket)
Actions: MFA_REQUIRED × LOCKOUT_THRESHOLD × SESSION_TIMEOUT (12 combinations)
Reward : +10 for blocked attack, +1 for smooth legitimate login,
         -20 for successful breach (false negative), -2 for user friction

Persistence: data/rl_auth_tuner.json

Usage
-----
    from auth.rl_auth_tuner import get_rl_auth_tuner

    tuner = get_rl_auth_tuner()
    policy = tuner.get_policy(
        access_hour=14, failure_streak=2,
        is_admin=True, is_new_device=True, risk_score=0.6
    )
    # policy = {"mfa_required": True, "max_attempts": 3, "session_timeout_min": 15}

    # After session ends:
    tuner.record_outcome(policy_action, legitimate=True, was_breach=False, friction_level=0.2)
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dipex.auth.rl_auth_tuner")

_DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "rl_auth_tuner.json"
)

# ── Action space ──────────────────────────────────────────────────────────────

_MFA_OPTIONS:     List[bool] = [True, False]
_LOCKOUT_OPTIONS: List[int]  = [3, 5, 10]
_TIMEOUT_OPTIONS: List[int]  = [15, 30, 60, 120]

# All (mfa, lockout, timeout) combos
_ALL_ACTIONS: List[Tuple[bool, int, int]] = [
    (m, l, t)
    for m in _MFA_OPTIONS
    for l in _LOCKOUT_OPTIONS
    for t in _TIMEOUT_OPTIONS
]

_ALPHA:     float = 0.15
_EPSILON:   float = 0.10
_SAVE_PROB: float = 0.12


def _state_key(
    access_hour:    int,
    failure_streak: int,
    is_admin:       bool,
    is_new_device:  bool,
    risk_score:     float,
) -> str:
    hour_bucket  = "night" if access_hour < 6 or access_hour > 22 else \
                   ("morning" if access_hour < 12 else "day")
    fail_bucket  = "high" if failure_streak >= 3 else ("low" if failure_streak == 0 else "mid")
    risk_bucket  = "high" if risk_score > 0.7 else ("low" if risk_score < 0.3 else "mid")
    return f"{hour_bucket}::fail_{fail_bucket}::admin{int(is_admin)}::new{int(is_new_device)}::risk_{risk_bucket}"


def _action_key(action: Tuple[bool, int, int]) -> str:
    return f"mfa{int(action[0])}::lock{action[1]}::timeout{action[2]}"


class RLAuthTuner:
    """Q-learning agent for adaptive, risk-based authentication policy."""

    def __init__(self, state_path: str = _DEFAULT_STATE_PATH) -> None:
        self.state_path = state_path
        self.q: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    self.q = json.load(fh)
                logger.info("RLAuthTuner: loaded %d states.", len(self.q))
            except Exception as exc:  # noqa: BLE001
                logger.warning("RLAuthTuner: Q-table load failed: %s", exc)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        try:
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(self.q, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLAuthTuner: save failed: %s", exc)

    def _init_state(self, state: str) -> None:
        if state not in self.q:
            self.q[state] = {_action_key(a): 0.0 for a in _ALL_ACTIONS}
            # Seed high-risk states with stricter priors
            if "high" in state:
                for a in _ALL_ACTIONS:
                    if a[0] is True and a[1] == 3:  # MFA=True, lockout=3
                        self.q[state][_action_key(a)] = 1.0

    def _update(self, state: str, action: str, reward: float) -> None:
        self._init_state(state)
        curr = self.q[state].get(action, 0.0)
        self.q[state][action] = round(curr + _ALPHA * (reward - curr), 6)
        if random.random() < _SAVE_PROB:
            self.save()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_policy(
        self,
        access_hour:    int   = 12,
        failure_streak: int   = 0,
        is_admin:       bool  = False,
        is_new_device:  bool  = False,
        risk_score:     float = 0.3,
    ) -> Dict[str, Any]:
        """
        Return the RL-selected auth policy for this access context.

        Returns
        -------
        {
          "mfa_required":       bool,
          "max_attempts":       int,   # lockout threshold
          "session_timeout_min": int,
          "state":              str,
          "action_key":         str,
        }
        """
        state = _state_key(access_hour, failure_streak, is_admin, is_new_device, risk_score)
        self._init_state(state)

        if random.random() < _EPSILON:
            action_tuple = random.choice(_ALL_ACTIONS)
        else:
            best_key     = max(self.q[state], key=self.q[state].__getitem__)
            # Convert key back to tuple
            parts        = best_key.split("::")
            mfa          = parts[0] == "mfa1"
            lockout      = int(parts[1].replace("lock", ""))
            timeout      = int(parts[2].replace("timeout", ""))
            action_tuple = (mfa, lockout, timeout)

        akey = _action_key(action_tuple)
        logger.debug("[RL] Auth policy for state=%s: %s", state, akey)
        return {
            "mfa_required":        action_tuple[0],
            "max_attempts":        action_tuple[1],
            "session_timeout_min": action_tuple[2],
            "state":               state,
            "action_key":          akey,
        }

    def record_outcome(
        self,
        policy:         Dict[str, Any],
        legitimate:     bool,
        was_breach:     bool  = False,
        friction_level: float = 0.0,  # [0, 1] — how much the user struggled
    ) -> None:
        """
        Update Q-table with observed auth outcome.

        Parameters
        ----------
        legitimate     : True if the session was a real valid user
        was_breach     : True if a security incident occurred (very bad)
        friction_level : [0, 1] — user friction from overly strict policy
        """
        state  = policy.get("state", "unknown")
        action = policy.get("action_key", "")

        if was_breach:
            reward = -20.0
        elif legitimate and not was_breach:
            reward = 10.0 - (friction_level * 8.0)   # reduce if too much friction
        else:
            reward = -2.0   # rejected legitimate user (false positive)

        self._update(state, action, reward)
        logger.debug(
            "[RL] Auth outcome: state=%s | action=%s | breach=%s | legit=%s | reward=%.1f",
            state, action, was_breach, legitimate, reward,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_RL_AUTH: Optional[RLAuthTuner] = None


def get_rl_auth_tuner() -> RLAuthTuner:
    global _RL_AUTH  # noqa: PLW0603
    if _RL_AUTH is None:
        _RL_AUTH = RLAuthTuner()
    return _RL_AUTH
