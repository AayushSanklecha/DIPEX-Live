"""
preprocessing/rl_feature_selector.py
---------------------------------------
Production-grade RL Feature Selection Agent (Q-Learning).

Purpose
-------
After Deep Feature Synthesis generates many engineered features,
this agent learns which subset maximises downstream model quality
(cross-val score improvement) while minimising training time.

Architecture
------------
State   : A fingerprint of the current feature set (sorted column hash)
Action  : Toggle individual feature on/off (add/remove from active set)
Reward  : downstream_cv_delta  - training_time_penalty

The agent is given a batch of candidate features and iteratively refines
the active set over multiple pipeline runs — improving with experience.

Persistence: data/rl_feature_selector.json

Usage
-----
    from preprocessing.rl_feature_selector import RLFeatureSelector

    selector = RLFeatureSelector(candidate_features=["feat_a", "feat_b", ...])
    active = selector.get_active_features()

    # After model evaluation:
    selector.record_outcome(active, cv_delta=0.03, training_time_s=12.4)
    active_next = selector.get_active_features()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from typing import Dict, List, Optional, Set

logger = logging.getLogger("dipex.preprocessing.rl_feature_selector")

_DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "rl_feature_selector.json"
)
_ALPHA:      float = 0.12
_EPSILON:    float = 0.18
_GAMMA:      float = 0.7
_SAVE_PROB:  float = 0.15


class RLFeatureSelector:
    """
    Q-learning agent for sequential feature subset selection.

    On each call to get_active_features() the agent either:
      - Explores: randomly adds or removes one feature
      - Exploits: uses the highest-value known subset from Q-table

    Over many runs it converges to the feature set that most reliably
    improves downstream model performance.
    """

    def __init__(
        self,
        candidate_features: List[str],
        state_path: str = _DEFAULT_STATE_PATH,
        max_features: int = 50,
    ) -> None:
        self.candidates    = list(candidate_features)
        self.state_path    = state_path
        self.max_features  = max_features
        self.q: Dict[str, float] = {}        # {feature_set_hash: Q-value}
        # Bidirectional registry: hash <-> feature set
        self._hash_to_features: Dict[str, Set[str]] = {}
        self._active: Set[str]  = set(self.candidates[:max_features])
        self._last_hash: str    = ""
        self._load()

        # Ensure the initial active set is registered
        initial_hash = self._hash(self._active)
        self._register_feature_set(initial_hash, self._active)


    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    self.q = data.get("q_values", {})
                    # Convert list of features back to set
                    for h, features_list in data.get("hash_to_features", {}).items():
                        self._hash_to_features[h] = set(features_list)

                logger.info(
                    "RLFeatureSelector: loaded %d Q-values and %d feature sets.",
                    len(self.q), len(self._hash_to_features)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("RLFeatureSelector: Q-table load failed: %s", exc)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        try:
            # Convert sets to lists for JSON serialization
            serializable_hash_to_features = {
                h: sorted(list(s)) for h, s in self._hash_to_features.items()
            }
            data = {
                "q_values": self.q,
                "hash_to_features": serializable_hash_to_features,
            }
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLFeatureSelector: Save failed: %s", exc)

    # ── Hash helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _hash(feature_set: Set[str]) -> str:
        key = ",".join(sorted(feature_set))
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _register_feature_set(self, h: str, feature_set: Set[str]) -> None:
        """Registers a feature set with its hash."""
        if h not in self._hash_to_features:
            self._hash_to_features[h] = feature_set

    def _mutate(self, feature_set: Set[str]) -> Set[str]:
        """Single-step mutation: randomly add or remove one feature."""
        new_set = set(feature_set)
        if len(new_set) == 0 or (len(new_set) < len(self.candidates) and random.random() < 0.5):
            # Add a random unused feature
            unused = [f for f in self.candidates if f not in new_set]
            if unused:
                new_set.add(random.choice(unused))
        else:
            # Remove a random active feature
            new_set.discard(random.choice(list(new_set)))
        # Enforce max_features cap
        while len(new_set) > self.max_features:
            new_set.discard(random.choice(list(new_set)))

        # Register the new set
        self._register_feature_set(self._hash(new_set), new_set)
        return new_set

    # ── Public API ────────────────────────────────────────────────────────────

    def get_active_features(self) -> List[str]:
        """
        Return the current active feature subset (RL-selected).

        First call always uses all candidates (up to max_features).
        Subsequent calls apply epsilon-greedy mutation.
        """
        if not self._last_hash:
            # First call: use all candidates as starting point
            self._last_hash = self._hash(self._active)
            self._register_feature_set(self._last_hash, self._active)
            logger.debug("[RL] FeatureSelector: initial set (%d features).", len(self._active))
            return sorted(self._active)

        # Epsilon-greedy: explore or exploit
        if random.random() < _EPSILON:
            # Explore: mutate current active set
            self._active    = self._mutate(self._active)
            self._last_hash = self._hash(self._active)
            logger.debug("[RL] FeatureSelector: exploring \u2014 %d features.", len(self._active))
        else:
            # Exploit: find best known set in Q-table (or keep current)
            if self.q:
                best_hash = max(self.q, key=self.q.__getitem__)
                if best_hash != self._last_hash and best_hash in self._hash_to_features:
                    # Reconstruct and apply the best known feature set
                    self._active = self._hash_to_features[best_hash]
                    self._last_hash = best_hash
                    logger.debug(
                        "[RL] FeatureSelector: exploiting best known Q-hash=%s (%d features).",
                        best_hash, len(self._active)
                    )
                else:
                    # If best_hash is current or unknown, just re-hash current active set
                    self._last_hash = self._hash(self._active)
                    self._register_feature_set(self._last_hash, self._active)
                    logger.debug(
                        "[RL] FeatureSelector: exploiting (current set) \u2014 %d features.",
                        len(self._active)
                    )
            else:
                # No Q-values yet, just re-hash current active set
                self._last_hash = self._hash(self._active)
                self._register_feature_set(self._last_hash, self._active)
                logger.debug(
                    "[RL] FeatureSelector: exploiting (no Q-values yet) \u2014 %d features.",
                    len(self._active)
                )


        return sorted(self._active)

    def record_outcome(
        self,
        features_used:   List[str],
        cv_delta:        float,    # Δ cross-val score vs. without this feature set
        training_time_s: float,    # Wall-clock training time in seconds
    ) -> None:
        """
        Update the Q-table with the observed outcome of the selected features.

        Parameters
        ----------
        cv_delta        : Cross-validation score improvement (positive = better)
        training_time_s : Time taken to train the model with these features
        """
        current_features_set = set(features_used)
        h      = self._hash(current_features_set)
        self._register_feature_set(h, current_features_set) # Ensure this set is registered
        reward = (cv_delta * 100.0) - (training_time_s * 0.2)

        if h not in self.q:
            self.q[h] = 0.0
        self.q[h] = round(self.q[h] + _ALPHA * (reward - self.q[h]), 6)

        logger.debug(
            "[RL] FeatureSelector: hash=%s | features=%d | cv_delta=%.4f | "
            "train_s=%.1f | reward=%.2f | Q=%.4f",
            h, len(features_used), cv_delta, training_time_s, reward, self.q[h],
        )
        if random.random() < _SAVE_PROB:
            self.save()

    @property
    def active_count(self) -> int:
        return len(self._active)
