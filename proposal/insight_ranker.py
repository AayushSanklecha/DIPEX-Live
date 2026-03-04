"""
proposal/insight_ranker.py
---------------------------
Insight Ranking module for the DIPEX Proposal Layer.

Implements:
  1. InsightRanker — produces a ranked list of data insights, each scored by a
     composite of: effect size, statistical significance, business impact,
     novelty vs. historical baseline, and temporal stability.

  2. FeatureProposer — suggests feature transformations and encodings based on
     column-level properties extracted from the profiling report.

  3. AnomalyFlagger — uses an Isolation Forest (sklearn) to flag anomalous
     records. Advisory only — returns indices + anomaly scores, never modifies data.

  4. RAGRecall — lightweight experience-memory search: given a dataset
     profile, retrieves the top-K most similar past analyses from
     ExperienceMemoryV2 via cosine similarity on a simple feature vector.

All components are PURELY ADVISORY — they return structured dicts/lists.
No data is modified, written, or published by any component in this module.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.proposal.insight_ranker")


# ══════════════════════════════════════════════════════════════════════════════
# 1. INSIGHT RANKER
# ══════════════════════════════════════════════════════════════════════════════

class InsightRanker:
    """
    Produces a ranked list of data insights from a profiling report.

    Each insight is scored by a composite of five dimensions:
      - effect_size    : standardised effect (Cohen's d proxy from skewness + range)
      - significance   : statistical robustness (sample size + null rate inverse)
      - business_impact: heuristic domain weighting (revenue/amount/rate cols score higher)
      - novelty        : deviation from historical mean (if baseline_stats provided)
      - stability      : temporal consistency (inverse of drift PSI if provided)

    Final score = weighted product in [0, 1].

    Usage::

        ranker = InsightRanker(weights={"effect_size": 0.30, "significance": 0.25, ...})
        ranked = ranker.rank(profile_report, baseline_stats=historical_profile)
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "effect_size":     0.30,
        "significance":    0.25,
        "business_impact": 0.20,
        "novelty":         0.15,
        "stability":       0.10,
    }

    # Keyword patterns that suggest business importance
    _HIGH_IMPACT_KEYWORDS = {
        "revenue", "amount", "profit", "loss", "churn", "conversion",
        "rate", "margin", "cost", "sales", "price", "value", "spend",
        "income", "growth", "retention", "arpu", "ltv", "cac", "roi",
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 10,
    ) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        # Normalise weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        self.top_k = top_k

    def rank(
        self,
        profile: Dict[str, Any],
        baseline_stats: Optional[Dict[str, Any]] = None,
        drift_report: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank insights from a profiling report.

        Parameters
        ----------
        profile        : Output of Profiler.profile() — contains 'columns' dict
        baseline_stats : Historical profiling report for novelty scoring
        drift_report   : DriftDetector report with per-column PSI scores

        Returns
        -------
        List of insight dicts sorted by score (descending), capped at top_k.
        Each dict has: column, insight_type, score, description, scores_breakdown.
        """
        columns = profile.get("columns", {})
        if not columns:
            return []

        row_count = profile.get("row_count", 1)
        psi_map   = self._extract_psi(drift_report) if drift_report else {}
        insights: List[Dict[str, Any]] = []

        for col, meta in columns.items():
            if not isinstance(meta, dict):
                continue

            col_insights = self._score_column(
                col, meta, row_count, baseline_stats, psi_map
            )
            insights.extend(col_insights)

        insights.sort(key=lambda x: x["score"], reverse=True)
        return insights[:self.top_k]

    def _score_column(
        self,
        col: str,
        meta: Dict[str, Any],
        row_count: int,
        baseline_stats: Optional[Dict[str, Any]],
        psi_map: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Generate 1-3 insights per column based on its properties."""
        results: List[Dict[str, Any]] = []

        # ── Effect size: strong skewness or large range / high std relative to mean
        mean    = meta.get("mean", 0) or 0
        std     = meta.get("std", 0) or 0
        skew    = abs(meta.get("skewness", 0) or 0)
        cv      = abs(std / mean) if mean != 0 else 0
        effect  = min((skew / 4.0 + cv / 2.0) / 1.5, 1.0)

        # ── Significance: large sample + low null rate
        null_rate    = meta.get("null_rate", 0) or 0
        sig_score    = min(math.log1p(row_count) / 12.0, 1.0) * (1.0 - null_rate)

        # ── Business impact: keyword heuristic on column name
        col_lower    = col.lower().replace("_", " ").replace("-", " ")
        biz_score    = 0.8 if any(kw in col_lower for kw in self._HIGH_IMPACT_KEYWORDS) else 0.3

        # ── Novelty: how different is current mean vs historical
        novelty_score = 0.5  # default (no baseline)
        if baseline_stats:
            base_cols = baseline_stats.get("columns", {})
            base_meta = base_cols.get(col, {})
            if base_meta and base_meta.get("mean") is not None and meta.get("mean") is not None:
                base_mean = base_meta.get("mean", 0) or 0
                cur_mean  = meta.get("mean", 0) or 0
                denom     = abs(base_mean) if base_mean != 0 else 1.0
                novelty_score = min(abs(cur_mean - base_mean) / denom, 1.0)

        # ── Stability: inverse of PSI (high PSI = low stability)
        psi           = psi_map.get(col, 0.05)
        stability     = max(0.0, 1.0 - min(psi / 0.25, 1.0))

        # ── Final score
        score = (
            self.weights["effect_size"]     * effect
            + self.weights["significance"]  * sig_score
            + self.weights["business_impact"] * biz_score
            + self.weights["novelty"]       * novelty_score
            + self.weights["stability"]     * stability
        )

        # Only surface meaningful insights
        if score < 0.15 and skew < 1.0:
            return []

        insight_type = "high_skewness" if skew > 2.0 else (
            "high_variance" if cv > 1.0 else "notable_distribution"
        )

        results.append({
            "column":     col,
            "insight_type": insight_type,
            "score":      round(score, 4),
            "description": self._describe(col, meta, skew, cv, psi, novelty_score),
            "scores_breakdown": {
                "effect_size":     round(effect, 4),
                "significance":    round(sig_score, 4),
                "business_impact": round(biz_score, 4),
                "novelty":         round(novelty_score, 4),
                "stability":       round(stability, 4),
            },
        })

        # Add null-rate finding if critical
        if null_rate > 0.10:
            null_score = (null_rate * 0.8 + biz_score * 0.2)
            results.append({
                "column":     col,
                "insight_type": "high_null_rate",
                "score":      round(null_score, 4),
                "description": (
                    f"Column '{col}' has {null_rate:.1%} null rate. "
                    "Missing data pattern may bias downstream analyses."
                ),
                "scores_breakdown": {
                    "null_rate":       round(null_rate, 4),
                    "business_impact": round(biz_score, 4),
                },
            })

        return results

    def _describe(
        self, col: str, meta: Dict[str, Any],
        skew: float, cv: float, psi: float, novelty: float,
    ) -> str:
        parts = []
        if skew > 2.0:
            parts.append(f"highly skewed (skew={skew:.2f})")
        if cv > 1.0:
            parts.append(f"high coefficient of variation (CV={cv:.2f})")
        if psi > 0.10:
            parts.append(f"drifting from baseline (PSI={psi:.3f})")
        if novelty > 0.30:
            parts.append(f"significantly different from historical mean (Δ={novelty:.1%})")
        if not parts:
            parts.append("notable distributional characteristics")
        return f"Column '{col}' exhibits: " + "; ".join(parts) + "."

    @staticmethod
    def _extract_psi(drift_report: Dict[str, Any]) -> Dict[str, float]:
        """Extract per-column PSI from a drift report."""
        col_psi: Dict[str, float] = {}
        for entry in drift_report.get("column_drifts", []):
            if isinstance(entry, dict) and "column" in entry:
                col_psi[entry["column"]] = float(entry.get("psi", 0.05))
        return col_psi


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE PROPOSER
# ══════════════════════════════════════════════════════════════════════════════

class FeatureProposer:
    """
    Suggests feature transformations and encodings based on column statistics.

    Transformation proposals:
      - log_transform    : highly right-skewed numeric columns (skew > 1.5)
      - bin_equal_freq   : high-cardinality numerics where continuous is noisy
      - one_hot_encode   : low-cardinality categoricals (<= 15 unique values)
      - target_encode    : high-cardinality categoricals (> 15 unique, < 200)
      - interaction_term : pairs of numeric cols with high mutual correlation
      - datetime_extract : datetime columns → year/month/dow/hour/is_weekend
      - flag_null        : columns with 5-30% nulls → add binary presence flag
      - polynomial_deg2  : numeric cols with near-zero skew → x² term for non-linearity

    Usage::

        proposer = FeatureProposer()
        proposals = proposer.propose(profile_report, correlation_matrix)
    """

    _LOG_SKEW_THRESHOLD    = 1.5
    _HIGH_CARDINALITY_CAT  = 16    # > this → target encode
    _BIN_CARDINALITY_NUM   = 50    # > this → consider binning
    _NULL_FLAG_MIN         = 0.05
    _NULL_FLAG_MAX         = 0.30
    _HIGH_CORR_THRESHOLD   = 0.70

    def propose(
        self,
        profile: Dict[str, Any],
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return feature transformation proposals from the profiling report.

        Returns
        -------
        List of proposal dicts with: column(s), transformation, rationale, priority.
        Sorted by priority (1 = highest).
        """
        proposals: List[Dict[str, Any]] = []
        columns = profile.get("columns", {})

        numeric_cols: List[str] = []
        for col, meta in columns.items():
            if not isinstance(meta, dict):
                continue
            dtype = meta.get("dtype", "")
            skew  = meta.get("skewness", 0) or 0
            card  = meta.get("unique_count", 0) or 0
            n_tot = max(meta.get("count", 1), 1)
            null_rate = meta.get("null_rate", 0) or 0
            is_numeric = any(t in str(dtype) for t in ["int", "float"])
            is_cat = "object" in str(dtype) or "category" in str(dtype)
            is_dt  = "datetime" in str(dtype)

            if is_numeric:
                numeric_cols.append(col)

                # Log transform
                if skew > self._LOG_SKEW_THRESHOLD:
                    proposals.append(self._proposal(
                        columns=[col],
                        transformation="log_transform",
                        rationale=(
                            f"'{col}' is right-skewed (skew={skew:.2f} > {self._LOG_SKEW_THRESHOLD}). "
                            "log1p transform normalises distribution for linear models."
                        ),
                        priority=1,
                    ))

                # Binning
                if card > self._BIN_CARDINALITY_NUM:
                    proposals.append(self._proposal(
                        columns=[col],
                        transformation="bin_equal_freq",
                        rationale=(
                            f"'{col}' has {card} unique values. "
                            "Equal-frequency binning reduces noise for tree models."
                        ),
                        priority=3,
                    ))

                # Polynomial feature for near-normal cols
                if abs(skew) < 0.5 and n_tot > 100:
                    proposals.append(self._proposal(
                        columns=[col],
                        transformation="polynomial_deg2",
                        rationale=(
                            f"'{col}' is near-normally distributed. "
                            "Polynomial x² term can capture quadratic relationships."
                        ),
                        priority=4,
                    ))

                # Null flag
                if self._NULL_FLAG_MIN <= null_rate <= self._NULL_FLAG_MAX:
                    proposals.append(self._proposal(
                        columns=[col],
                        transformation="flag_null",
                        rationale=(
                            f"'{col}' has {null_rate:.1%} null rate. "
                            "Binary null-presence flag preserves missingness signal."
                        ),
                        priority=2,
                    ))

            elif is_cat:
                unique_ratio = card / n_tot if n_tot > 0 else 0

                if card <= self._HIGH_CARDINALITY_CAT:
                    proposals.append(self._proposal(
                        columns=[col],
                        transformation="one_hot_encode",
                        rationale=(
                            f"'{col}' has {card} unique values (low cardinality). "
                            "One-hot encoding is safest choice."
                        ),
                        priority=1,
                    ))
                elif card <= 200:
                    proposals.append(self._proposal(
                        columns=[col],
                        transformation="target_encode",
                        rationale=(
                            f"'{col}' has {card} unique values (medium-high). "
                            "Target encoding reduces dimensionality vs. OHE."
                        ),
                        priority=2,
                    ))
                else:
                    proposals.append(self._proposal(
                        columns=[col],
                        transformation="hash_encode",
                        rationale=(
                            f"'{col}' has very high cardinality ({card}). "
                            "Feature hashing (MurmurHash) controls dimensionality explosion."
                        ),
                        priority=2,
                    ))

            elif is_dt:
                proposals.append(self._proposal(
                    columns=[col],
                    transformation="datetime_extract",
                    rationale=(
                        f"'{col}' is a datetime column. "
                        "Extract: year, month, day_of_week, hour, is_weekend, quarter."
                    ),
                    priority=1,
                ))

        # Interaction terms from correlation matrix
        if correlation_matrix is not None:
            try:
                corr_proposals = self._interaction_proposals(
                    numeric_cols, correlation_matrix
                )
                proposals.extend(corr_proposals)
            except Exception:
                pass

        # Deduplicate + sort
        seen: set = set()
        unique_props: List[Dict[str, Any]] = []
        for p in proposals:
            key = (tuple(p["columns"]), p["transformation"])
            if key not in seen:
                seen.add(key)
                unique_props.append(p)

        unique_props.sort(key=lambda x: x["priority"])
        return unique_props

    def _interaction_proposals(
        self,
        numeric_cols: List[str],
        corr_matrix: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        cols = [c for c in numeric_cols if c in corr_matrix.columns]
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                try:
                    r = abs(float(corr_matrix.loc[c1, c2]))
                except (KeyError, TypeError):
                    continue
                if r > self._HIGH_CORR_THRESHOLD:
                    proposals.append(self._proposal(
                        columns=[c1, c2],
                        transformation="interaction_term",
                        rationale=(
                            f"Pearson |r|={r:.2f} between '{c1}' and '{c2}'. "
                            "Multiplicative interaction term may capture joint effect."
                        ),
                        priority=3,
                    ))
        return proposals

    @staticmethod
    def _proposal(
        columns: List[str],
        transformation: str,
        rationale: str,
        priority: int,
    ) -> Dict[str, Any]:
        return {
            "columns":       columns,
            "transformation": transformation,
            "rationale":     rationale[:300],
            "priority":      priority,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANOMALY FLAGGER (Isolation Forest — advisory only)
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyFlagger:
    """
    Advisory Isolation Forest anomaly detector.

    Operates on a read-only copy of the DataFrame. Returns anomaly indices
    and scores — never modifies any data.

    Parameters
    ----------
    contamination : float
        Expected fraction of outliers in the dataset (default 0.05 = 5%).
    n_estimators : int
        Number of trees in the forest (default 100).
    max_samples : int or "auto"
        Samples per tree (default "auto").
    random_state : int
        Seed for reproducibility.

    Usage::

        flagger = AnomalyFlagger(contamination=0.05)
        result = flagger.flag(df, numeric_cols=["revenue", "cost"])
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        max_samples: Any = "auto",
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.random_state  = random_state

    def flag(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Flag anomalous rows using Isolation Forest.

        Parameters
        ----------
        df           : DataFrame (never modified)
        numeric_cols : Columns to use for anomaly detection.
                       If None, all numeric columns are used.

        Returns
        -------
        Dict with:
          - anomalous_indices : List[int] — row indices of anomalies
          - anomaly_scores    : Dict[int, float] — row idx → normalised score [0,1]
          - anomaly_count     : int
          - total_rows        : int
          - anomaly_rate      : float
          - columns_used      : List[str]
          - advisory_note     : str — reminder that this is advisory only
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning(
                "[AnomalyFlagger] scikit-learn not available. "
                "Install with: pip install scikit-learn"
            )
            return self._empty_result(len(df))

        if len(df) < 10:
            return self._empty_result(len(df))

        # Select numeric columns
        if numeric_cols:
            cols = [c for c in numeric_cols if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not cols:
            return self._empty_result(len(df))

        # Work on a copy — NEVER modify input df
        X = df[cols].copy().fillna(df[cols].median())

        clf = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        preds  = clf.fit_predict(X)
        scores = clf.decision_function(X)  # higher = more normal

        # Normalise scores to [0,1] where 1.0 = most anomalous
        norm_scores = 1.0 - (scores - scores.min()) / (scores.ptp() + 1e-10)

        anomaly_mask    = (preds == -1)
        anomalous_idx   = list(df.index[anomaly_mask])
        anomaly_scores  = {int(idx): round(float(norm_scores[i]), 4)
                           for i, idx in enumerate(df.index) if anomaly_mask[i]}

        logger.info(
            "[AnomalyFlagger] %d/%d rows flagged (contamination=%.2f) using %d cols",
            len(anomalous_idx), len(df), self.contamination, len(cols),
        )

        return {
            "anomalous_indices": anomalous_idx,
            "anomaly_scores":    anomaly_scores,
            "anomaly_count":     len(anomalous_idx),
            "total_rows":        len(df),
            "anomaly_rate":      round(len(anomalous_idx) / max(len(df), 1), 4),
            "columns_used":      cols,
            "advisory_note": (
                "Anomaly flagging is ADVISORY ONLY. "
                "Flagged rows must be reviewed by an analyst before any action. "
                "This result does NOT trigger automatic data removal."
            ),
        }

    @staticmethod
    def _empty_result(n: int) -> Dict[str, Any]:
        return {
            "anomalous_indices": [],
            "anomaly_scores":    {},
            "anomaly_count":     0,
            "total_rows":        n,
            "anomaly_rate":      0.0,
            "columns_used":      [],
            "advisory_note": "Insufficient data for anomaly detection.",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. RAG RECALL — Experience Memory similarity search
# ══════════════════════════════════════════════════════════════════════════════

class RAGRecall:
    """
    Retrieval-Augmented Generation (RAG) experience recall.

    Given a dataset's profile fingerprint, retrieves the top-K most similar
    past analyses from ExperienceMemoryV2 using cosine similarity on a
    compact feature vector derived from the profiling report.

    Feature vector (7-dim):
      [log(row_count), mean_null_rate, mean_numeric_skew,
       numeric_col_ratio, unique_col_ratio, mean_psi, confidence_score]

    This is a lightweight, dependency-free approximation of semantic RAG.
    In production, replace with a proper vector store (FAISS, Chroma, Pinecone).

    Usage::

        rag = RAGRecall(experience_memory=memory)
        similar = rag.recall(profile_report, confidence_score, top_k=5)
    """

    def __init__(
        self,
        experience_memory: Any = None,
        top_k: int = 5,
    ) -> None:
        self.experience_memory = experience_memory
        self.top_k = top_k

    def recall(
        self,
        profile: Dict[str, Any],
        confidence_score: float = 1.0,
        drift_report: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K similar past analyses from experience memory.

        Returns
        -------
        List of dicts, each containing: episode_id, similarity, winning_strategy,
        confidence_score, schema_version, source_type, summary.
        """
        if self.experience_memory is None:
            return []

        try:
            episodes = self.experience_memory.query(
                filters={"stage": "APPROVED_OUTPUT"},
                limit=200,
            )
        except Exception as exc:
            logger.warning("[RAGRecall] Memory query failed: %s", exc)
            return []

        if not episodes:
            return []

        query_vec = self._profile_to_vector(profile, confidence_score, drift_report)

        scored: List[Tuple[float, Any]] = []
        for ep in episodes:
            try:
                ep_dict  = ep.to_dict()
                ep_vec   = self._episode_to_vector(ep_dict)
                sim      = self._cosine_similarity(query_vec, ep_vec)
                scored.append((sim, ep_dict))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.top_k]

        return [
            {
                "episode_id":       ep.get("episode_id", ep.get("snapshot_id", "?")),
                "similarity":       round(sim, 4),
                "confidence_score": ep.get("confidence_score", 0),
                "winning_strategy": ep.get("winning_strategy", "N/A"),
                "schema_version":   ep.get("schema_version", "N/A"),
                "source_type":      ep.get("source_type", "unknown"),
                "summary": (
                    f"Past analysis with conf={ep.get('confidence_score', 0):.2f} "
                    f"using strategy='{ep.get('winning_strategy', 'N/A')}'. "
                    f"Similarity={sim:.3f}."
                ),
            }
            for sim, ep in top
        ]

    def _profile_to_vector(
        self,
        profile: Dict[str, Any],
        confidence: float,
        drift_report: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Convert a profile report to a compact feature vector."""
        columns = profile.get("columns", {})
        meta_list = [m for m in columns.values() if isinstance(m, dict)]
        n = max(len(meta_list), 1)

        row_count  = max(profile.get("row_count", 1), 1)
        null_rates = [m.get("null_rate", 0) or 0 for m in meta_list]
        skews      = [abs(m.get("skewness", 0) or 0) for m in meta_list]
        dtypes     = [str(m.get("dtype", "")) for m in meta_list]

        numeric_ratio = sum(1 for d in dtypes if any(t in d for t in ["int", "float"])) / n
        unique_ratio  = sum(1 for m in meta_list if m.get("cardinality_tier") == "unique") / n
        mean_psi      = 0.05
        if drift_report:
            psis = [e.get("psi", 0.05) for e in drift_report.get("column_drifts", []) if isinstance(e, dict)]
            mean_psi = float(np.mean(psis)) if psis else 0.05

        return np.array([
            math.log1p(row_count) / 15.0,
            float(np.mean(null_rates)) if null_rates else 0,
            min(float(np.mean(skews)) / 4.0, 1.0) if skews else 0,
            numeric_ratio,
            unique_ratio,
            min(mean_psi / 0.25, 1.0),
            float(confidence),
        ], dtype=float)

    def _episode_to_vector(self, ep: Dict[str, Any]) -> np.ndarray:
        """Convert a stored episode to a comparable feature vector."""
        return np.array([
            math.log1p(ep.get("row_count", 1)) / 15.0,
            ep.get("mean_null_rate", 0.1),
            ep.get("mean_skewness", 0.3) / 4.0,
            ep.get("numeric_col_ratio", 0.5),
            ep.get("unique_col_ratio", 0.2),
            ep.get("mean_psi", 0.05) / 0.25,
            float(ep.get("confidence_score", 0.7)),
        ], dtype=float)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-10:
            return 0.0
        return float(np.dot(a, b) / denom)
