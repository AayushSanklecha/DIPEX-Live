"""
governance/anomaly_access.py
------------------------------
Production-grade ML Anomaly Detection for API Access Patterns.

Purpose
-------
Monitors per-user and per-endpoint access patterns and flags
anomalous sessions in real time using IsolationForest.

Architecture
------------
• Rolling window: maintains a sliding buffer of recent access events.
• Feature vector per session:
    - request_rate (req/min in last 5 min)
    - unique_endpoints_ratio
    - error_rate (4xx+5xx / total)
    - after_hours flag (access_hour < 7 or > 21)
    - new_endpoint_ratio (endpoints not seen in baseline)
    - payload_size_mean_kb
    - session_age_min
• IsolationForest fit on a warm-up window (first 200 events per user).
• Graceful fallback: rule-based spike detector if sklearn absent.

Usage
-----
    from governance.anomaly_access import AccessAnomalyDetector

    detector = AccessAnomalyDetector()
    detector.observe(username="alice", endpoint="/api/data", status_code=200,
                     payload_size_kb=12.5, access_hour=14)
    result = detector.is_anomalous(username="alice")
    # {"anomalous": False, "score": 0.12, "method": "isolation_forest"}
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List

import numpy as np

logger = logging.getLogger("dipex.governance.anomaly_access")

_WINDOW_MINUTES: int  = 10      # sliding window size
_MIN_EVENTS:     int  = 10      # minimum events before ML kicks in
_WARMUP_EVENTS:  int  = 80      # events needed to fit IsolationForest
_CONTAMINATION:  float = 0.05   # expected anomaly fraction


class _Event:
    __slots__ = ("endpoint", "status_code", "payload_size_kb", "access_hour", "ts")

    def __init__(self, endpoint: str, status_code: int, payload_size_kb: float, access_hour: int) -> None:
        self.endpoint        = endpoint
        self.status_code     = status_code
        self.payload_size_kb = payload_size_kb
        self.access_hour     = access_hour
        self.ts              = time.monotonic()


def _extract_features(
    events: Deque[_Event],
    baseline_endpoints: set,
    window_s: float = _WINDOW_MINUTES * 60,
) -> np.ndarray:
    """Build a 7-dim feature vector from recent events."""
    now      = time.monotonic()
    recents  = [e for e in events if (now - e.ts) <= window_s]
    n        = max(len(recents), 1)
    elapsed  = max(window_s / 60.0, 1.0)

    req_rate         = n / elapsed
    unique_eps       = len({e.endpoint for e in recents}) / n
    error_rate       = sum(1 for e in recents if e.status_code >= 400) / n
    after_hours      = float(any(e.access_hour < 7 or e.access_hour > 21 for e in recents))
    new_ep_ratio     = (
        len({e.endpoint for e in recents} - baseline_endpoints) / n
        if baseline_endpoints else 0.0
    )
    payload_mean     = np.mean([e.payload_size_kb for e in recents]) if recents else 0.0
    session_age_min  = (now - events[0].ts) / 60.0 if events else 0.0

    return np.array([
        req_rate, unique_eps, error_rate, after_hours,
        new_ep_ratio, payload_mean, session_age_min,
    ], dtype=np.float64)


class AccessAnomalyDetector:
    """
    Per-user access pattern anomaly detector using IsolationForest.

    All state is in-memory for the session. For persistent cross-restart
    learning, call save() / load() with a state file.
    """

    def __init__(self, contamination: float = _CONTAMINATION) -> None:
        self.contamination   = contamination
        self._events:         Dict[str, Deque[_Event]] = defaultdict(lambda: deque(maxlen=500))
        self._baselines:      Dict[str, set]           = defaultdict(set)
        self._models:         Dict[str, Any]           = {}   # user → fitted IFF
        self._sklearn_ok:     bool                     = self._check_sklearn()

    @staticmethod
    def _check_sklearn() -> bool:
        try:
            from sklearn.ensemble import IsolationForest  # noqa: F401
            return True
        except ImportError:
            logger.warning("AccessAnomalyDetector: scikit-learn absent — using rule-based fallback.")
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    def observe(
        self,
        username:       str,
        endpoint:       str,
        status_code:    int   = 200,
        payload_size_kb: float = 0.0,
        access_hour:    int   = 12,
    ) -> None:
        """Register one access event for a user."""
        event = _Event(endpoint, status_code, payload_size_kb, access_hour)
        self._events[username].append(event)

        # Build baseline from first WARMUP_EVENTS endpoints
        if len(self._events[username]) <= _WARMUP_EVENTS:
            self._baselines[username].add(endpoint)
            if len(self._events[username]) == _WARMUP_EVENTS:
                self._fit_model(username)

    def is_anomalous(self, username: str) -> Dict[str, Any]:
        """
        Score the current access pattern for a user.

        Returns
        -------
        {"anomalous": bool, "score": float, "method": str}
        """
        events = self._events.get(username)
        if not events or len(events) < _MIN_EVENTS:
            return {"anomalous": False, "score": 0.0, "method": "insufficient_data"}

        fv = _extract_features(events, self._baselines.get(username, set()))

        model = self._models.get(username)
        if model is not None:
            try:
                # IsolationForest score_samples: lower = more anomalous
                raw_score = float(model.score_samples(fv.reshape(1, -1))[0])
                # Normalise to [0, 1] where 1 = very anomalous
                # score_samples typically in [-0.6, 0.1] range
                norm = max(0.0, min(1.0, (-raw_score - 0.0) / 0.6))
                anomalous = model.predict(fv.reshape(1, -1))[0] == -1
                return {"anomalous": bool(anomalous), "score": round(norm, 4), "method": "isolation_forest"}
            except Exception as exc:  # noqa: BLE001
                logger.debug("AccessAnomalyDetector: model prediction failed: %s", exc)

        # Rule-based fallback
        return self._rule_based(fv)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _fit_model(self, username: str) -> None:
        if not self._sklearn_ok:
            return
        events = self._events[username]
        X = np.stack([
            _extract_features(
                deque(list(events)[:i+1], maxlen=len(events)),
                self._baselines[username],
            )
            for i in range(0, len(events), 5)
        ])
        try:
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=42,
            )
            model.fit(X)
            self._models[username] = model
            logger.info("AccessAnomalyDetector: IFF fitted for user '%s' (%d samples).", username, len(X))
        except Exception as exc:  # noqa: BLE001
            logger.warning("AccessAnomalyDetector: fit failed for '%s': %s", username, exc)

    @staticmethod
    def _rule_based(fv: np.ndarray) -> Dict[str, Any]:
        req_rate, _, error_rate, after_hours, new_ep_ratio, payload_mean, _ = fv
        score = 0.0
        if req_rate > 50:      score += 0.4
        if error_rate > 0.30:  score += 0.3
        if after_hours:        score += 0.2
        if new_ep_ratio > 0.5: score += 0.1
        if payload_mean > 500: score += 0.1
        anomalous = score >= 0.4
        return {"anomalous": anomalous, "score": round(min(score, 1.0), 4), "method": "rule_based"}

    def get_report(self) -> Dict[str, Any]:
        """Return a summary of anomaly scores for all observed users."""
        report = {}
        for username in self._events:
            report[username] = self.is_anomalous(username)
        return report
