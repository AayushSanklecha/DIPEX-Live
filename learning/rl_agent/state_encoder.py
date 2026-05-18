"""
learning/rl_agent/state_encoder.py
------------------------------------
Encodes pipeline run context into a normalized state vector suitable for
the PPO actor-critic networks.

Base state vector (12D, indices 0–11):
  [0]  log(n_rows)/15             — dataset scale (log-normalized)
  [1]  n_cols/100                 — feature dimensionality
  [2]  null_rate                  — fraction of missing values (0-1)
  [3]  anomaly_rate               — IF anomaly fraction (0-1)
  [4]  drift_psi                  — Population Stability Index (0-1 clamped)
  [5]  data_health/100            — analyst brain health score (0-1)
  [6-11] domain_onehot(6)         — one-hot for domain (banking/healthcare/finance/ecommerce/generic/other)

Extended state vector (19D, used during training):
  [12] target_available           — 1.0 if supervised target col exists
  [13] prior_confidence           — previous run confidence score
  [14] quarantine_frac            — fraction of rows quarantined
  [15] retry_count/3              — normalised pipeline retry count
  —— Analyst Instruction Loop signals ——
  [16] instruction_given          — 1.0 if analyst provided free-text instructions
  [17] plan_rejection_count/3     — normalised plan rejection count
  [18] user_satisfaction          — 1.0 if happy, 0.0 if unhappy, 0.5 if no feedback
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("dipex.learning.rl_agent.state_encoder")

# Domain one-hot order
DOMAIN_ORDER = ["banking", "healthcare", "finance", "ecommerce", "generic", "other"]

STATE_DIM = 12       # base state dimension
STATE_DIM_EXT = 19   # extended state for training (12 base + 4 pipeline + 3 satisfaction signals)


class StateEncoder:
    """
    Converts raw pipeline context dict into a normalized numpy state vector.

    Usage::

        encoder = StateEncoder()
        state = encoder.encode({
            "n_rows": 50000,
            "n_cols": 23,
            "null_rate": 0.05,
            "anomaly_rate": 0.02,
            "drift_psi": 0.12,
            "data_health": 78.5,
            "domain": "banking",
        })
        # state.shape == (12,)
    """

    def encode(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Encodes context dict into a 12D normalized state vector.
        All values are clipped to [0, 1] for numerical stability.
        """
        n_rows = max(int(context.get("n_rows", 1)), 1)
        n_cols = max(int(context.get("n_cols", 1)), 1)
        null_rate = float(context.get("null_rate", 0.0))
        anomaly_rate = float(context.get("anomaly_rate", 0.0))
        drift_psi = float(context.get("drift_psi", 0.0))
        data_health = float(context.get("data_health", 50.0))
        domain = str(context.get("domain", "generic")).lower()

        # Encode individual features
        s0 = min(math.log(n_rows) / 15.0, 1.0)           # log-scale rows
        s1 = min(n_cols / 100.0, 1.0)                     # col dimensionality
        s2 = float(np.clip(null_rate, 0.0, 1.0))          # null rate
        s3 = float(np.clip(anomaly_rate, 0.0, 1.0))       # anomaly rate
        s4 = float(np.clip(drift_psi / 1.0, 0.0, 1.0))   # PSI normalized
        s5 = float(np.clip(data_health / 100.0, 0.0, 1.0))  # health score

        # Domain one-hot (6D)
        domain_idx = DOMAIN_ORDER.index(domain) if domain in DOMAIN_ORDER else len(DOMAIN_ORDER) - 1
        domain_oh = [0.0] * len(DOMAIN_ORDER)
        domain_oh[domain_idx] = 1.0

        state = np.array([s0, s1, s2, s3, s4, s5] + domain_oh, dtype=np.float32)
        assert state.shape[0] == STATE_DIM, f"Expected {STATE_DIM}D state, got {state.shape[0]}D"
        return state

    def encode_extended(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Encodes into a 19D state vector for training.

        Dimensions 0-11: base encode() output
        Dimensions 12-15: pipeline context signals
          [12] target_available, [13] prior_confidence, [14] quarantine_frac, [15] retry_count/3
        Dimensions 16-18: analyst instruction loop signals
          [16] instruction_given (0/1)
          [17] plan_rejection_count / 3 (normalised)
          [18] user_satisfaction (1.0=happy, 0.0=unhappy, 0.5=no feedback)
        """
        base = self.encode(context)

        # Pipeline context signals (dimensions 12-15)
        target_available  = 1.0 if context.get("target_col") else 0.0
        prior_confidence  = float(np.clip(context.get("prior_confidence", 0.5), 0.0, 1.0))
        quarantine_frac   = float(np.clip(context.get("quarantine_frac", 0.0), 0.0, 1.0))
        retry_count_norm  = float(np.clip(context.get("retry_count", 0) / 3.0, 0.0, 1.0))

        # Analyst instruction loop signals (dimensions 16-18)
        instruction_given     = 1.0 if context.get("instruction_given", False) else 0.0
        plan_rejection_norm   = float(np.clip(context.get("plan_rejection_count", 0) / 3.0, 0.0, 1.0))
        # user_satisfaction: 1.0=happy, 0.0=unhappy, 0.5=not yet provided
        _satisfaction = context.get("user_satisfaction", None)
        if _satisfaction is True:
            user_satisfaction = 1.0
        elif _satisfaction is False:
            user_satisfaction = 0.0
        else:
            user_satisfaction = 0.5  # neutral / not yet provided

        extra = np.array([
            target_available, prior_confidence, quarantine_frac, retry_count_norm,
            instruction_given, plan_rejection_norm, user_satisfaction,
        ], dtype=np.float32)

        extended = np.concatenate([base, extra])
        assert extended.shape[0] == STATE_DIM_EXT, (
            f"Expected {STATE_DIM_EXT}D extended state, got {extended.shape[0]}D"
        )
        return extended

    def from_pipeline_result(self, snapshot_meta: Dict[str, Any], analytics: Dict[str, Any]) -> np.ndarray:
        """
        Convenience method: constructs context dict from pipeline result dicts
        and encodes it.
        """
        n_rows = int(snapshot_meta.get("row_count", 1))
        n_cols = int(snapshot_meta.get("col_count", 1))
        eda = analytics.get("eda_report", {})
        summary = eda.get("summary", {})

        null_rate = float(summary.get("overall_null_pct", 0.0))
        anomaly_pct = float(summary.get("anomaly_pct", 0.0))
        drift_psi = float(analytics.get("drift_psi", 0.0) or 0.0)
        data_health = float(analytics.get("data_health_score", 50.0))
        domain = str(snapshot_meta.get("domain", "generic"))

        return self.encode({
            "n_rows": n_rows,
            "n_cols": n_cols,
            "null_rate": null_rate,
            "anomaly_rate": anomaly_pct,
            "drift_psi": drift_psi,
            "data_health": data_health,
            "domain": domain,
        })
