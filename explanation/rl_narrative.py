"""
explanation/rl_narrative.py
-----------------------------
Production-grade RL-based Adaptive Report Section Ordering.

Purpose
-------
Different analysts prefer different report layouts.  This Q-learning agent
learns which section order leads to the highest analyst engagement score
(proxy: time-on-section / click-through rate to follow-up queries).

Architecture
------------
State   : (run_type, domain, confidence_tier)
            run_type:        "exploratory" | "model_eval" | "drift"
            domain:          "banking" | "healthcare" | "retail" | "generic"
            confidence_tier: "high" | "medium" | "low"

Actions : 10 permutations of the top-5 report sections (a manageable subset)
            Sections: [narrative, findings, quality, model_perf, risk]

Reward  : analyst_rating (1-5) mapped to [-2, +2]
          default: 0 if unknown

Persistence: data/rl_narrative.json

Usage
-----
    from explanation.rl_narrative import get_rl_narrative

    rl = get_rl_narrative()
    order = rl.get_section_order(run_type="model_eval", domain="banking", confidence=0.9)
    # ["findings", "model_perf", "narrative", "quality", "risk"]

    # After analyst provides feedback:
    rl.record_feedback(run_type="model_eval", domain="banking", confidence=0.9,
                       section_order=order, analyst_rating=4)
"""

from __future__ import annotations

import json
import logging
import os
import random
from itertools import permutations
from typing import Dict, List, Optional

logger = logging.getLogger("dipex.explanation.rl_narrative")

_DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "rl_narrative.json"
)

# ── Section catalogue ──────────────────────────────────────────────────────────

SECTIONS: List[str] = ["narrative", "findings", "quality", "model_perf", "risk"]

# Pre-compute a limited canonical subset of permutations (5! = 120 — use top 20)
# to keep Q-table size manageable.  Chosen to cover diverse orderings.
_REPRESENTATIVE_ORDERS: List[tuple] = [
    ("findings", "model_perf", "narrative", "quality",   "risk"),
    ("findings", "model_perf", "risk",       "narrative", "quality"),
    ("narrative", "findings",  "model_perf", "quality",  "risk"),
    ("narrative", "findings",  "risk",       "model_perf","quality"),
    ("model_perf","findings",  "narrative",  "quality",  "risk"),
    ("model_perf","findings",  "risk",       "narrative", "quality"),
    ("quality",   "findings",  "narrative",  "model_perf","risk"),
    ("quality",   "risk",      "findings",   "narrative", "model_perf"),
    ("risk",      "findings",  "narrative",  "model_perf","quality"),
    ("risk",      "quality",   "findings",   "model_perf","narrative"),
    ("findings",  "narrative", "quality",    "model_perf","risk"),
    ("narrative", "quality",   "findings",   "risk",      "model_perf"),
    ("findings",  "quality",   "narrative",  "risk",      "model_perf"),
    ("model_perf","narrative", "findings",   "quality",   "risk"),
    ("narrative", "model_perf","findings",   "quality",   "risk"),
    ("risk",      "narrative", "findings",   "model_perf","quality"),
    ("findings",  "risk",      "narrative",  "quality",   "model_perf"),
    ("quality",   "narrative", "findings",   "risk",      "model_perf"),
    ("narrative", "risk",      "findings",   "quality",   "model_perf"),
    ("findings",  "narrative", "model_perf", "risk",      "quality"),
]

# Domain/run-type priors (seed with reasonable defaults)
_PRIORS: Dict[str, tuple] = {
    "model_eval::banking::high":      ("model_perf", "findings", "narrative", "quality", "risk"),
    "model_eval::banking::medium":    ("findings",   "model_perf","narrative","quality",  "risk"),
    "model_eval::healthcare::high":   ("risk",        "findings", "model_perf","narrative","quality"),
    "exploratory::generic::high":     ("narrative",  "findings", "quality",  "model_perf","risk"),
    "exploratory::retail::medium":    ("findings",   "narrative","quality",  "risk",      "model_perf"),
    "drift::banking::low":            ("risk",       "quality",  "findings", "narrative", "model_perf"),
    "drift::generic::medium":         ("risk",       "findings", "quality",  "narrative", "model_perf"),
}

_ALPHA:     float = 0.12
_EPSILON:   float = 0.15
_SAVE_PROB: float = 0.10


def _action_key(order: tuple) -> str:
    return "->".join(order)


def _parse_key(key: str) -> List[str]:
    return key.split("->")


def _state_key(run_type: str, domain: str, confidence: float) -> str:
    tier = "high" if confidence >= 0.80 else ("medium" if confidence >= 0.55 else "low")
    rt   = run_type if run_type in {"model_eval", "exploratory", "drift"} else "exploratory"
    dom  = domain   if domain   in {"banking", "healthcare", "retail"}     else "generic"
    return f"{rt}::{dom}::{tier}"


class RLNarrative:
    """Q-learning agent for adaptive report section ordering."""

    def __init__(self, state_path: str = _DEFAULT_STATE_PATH) -> None:
        self.state_path = state_path
        self.q: Dict[str, Dict[str, float]] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    self.q = json.load(fh)
                logger.info("RLNarrative: loaded %d states.", len(self.q))
            except Exception as exc:  # noqa: BLE001
                logger.warning("RLNarrative: load failed: %s", exc)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        try:
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(self.q, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLNarrative: save failed: %s", exc)

    def _init_state(self, state: str) -> None:
        if state not in self.q:
            self.q[state] = {_action_key(a): 0.0 for a in _REPRESENTATIVE_ORDERS}
            if state in _PRIORS:
                self.q[state][_action_key(_PRIORS[state])] = 1.0

    def _update(self, state: str, akey: str, reward: float) -> None:
        self._init_state(state)
        curr = self.q[state].get(akey, 0.0)
        self.q[state][akey] = round(curr + _ALPHA * (reward - curr), 6)
        if random.random() < _SAVE_PROB:
            self.save()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_section_order(
        self,
        run_type:   str   = "exploratory",
        domain:     str   = "generic",
        confidence: float = 0.75,
    ) -> List[str]:
        """Return RL-selected section ordering for a report."""
        state = _state_key(run_type, domain, confidence)
        self._init_state(state)

        if random.random() < _EPSILON:
            order = random.choice(_REPRESENTATIVE_ORDERS)
            logger.debug("[RL] Narrative exploring: %s", order)
        else:
            best_key = max(self.q[state], key=self.q[state].__getitem__)
            order    = tuple(_parse_key(best_key))
            logger.debug("[RL] Narrative exploiting: %s", order)

        return list(order)

    def record_feedback(
        self,
        run_type:       str,
        domain:         str,
        confidence:     float,
        section_order:  List[str],
        analyst_rating: int,   # 1-5 Likert scale
    ) -> None:
        """Update Q-table from analyst feedback."""
        state  = _state_key(run_type, domain, confidence)
        akey   = _action_key(tuple(section_order))
        # Map 1-5 rating to [-2, +2] reward
        reward = (analyst_rating - 3) * 1.0   # centre on 3 (neutral)
        self._update(state, akey, reward)
        logger.debug(
            "[RL] Narrative feedback: state=%s | order=%s | rating=%d | reward=%.1f",
            state, akey, analyst_rating, reward,
        )

    def get_policy_summary(self) -> Dict[str, Dict]:
        return {
            s: {"best_order": _parse_key(max(v, key=v.__getitem__)), "Q": max(v.values())}
            for s, v in self.q.items() if v
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_RL_NARRATIVE: Optional[RLNarrative] = None


def get_rl_narrative() -> RLNarrative:
    global _RL_NARRATIVE  # noqa: PLW0603
    if _RL_NARRATIVE is None:
        _RL_NARRATIVE = RLNarrative()
    return _RL_NARRATIVE
