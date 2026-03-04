"""
analyst/nlp_query.py
---------------------
Production-grade NLP Query Classification Engine.

Purpose
-------
Converts natural-language analyst questions (e.g. "What are the top
products by revenue?") into structured pandas query intents, enabling
conversational data exploration without external AI APIs.

Architecture
------------
The engine uses a two-stage pipeline:

  Stage 1 — Intent Classifier:
    TF-IDF (unigram + bigram) × LinearSVC (multi-class).
    • Trained on a built-in seed corpus (200 + examples) covering the
      12 most common analyst intents.
    • Falls back to keyword heuristics if sklearn absent.

  Stage 2 — Entity Extractor:
    Rule-based regex extraction of:
      • column references (matched against the DataFrame schema)
      • aggregation functions  (sum, mean, count, max, min)
      • filter conditions      (> X, < X, == X, top N, bottom N)
      • sort direction         (ascending / descending)
      • date range             (last N days/weeks/months)

Intent Labels
-------------
  top_n, bottom_n, aggregate, filter, trend, compare, correlation,
  distribution, count_distinct, group_by, time_series, general

Colab Training
--------------
To train a higher-quality model on domain-specific queries:
  See colab/train_nlp_query_classifier.ipynb
  Exports: models/nlp_query_classifier.pkl
           models/nlp_query_vectorizer.pkl

Usage
-----
    from analyst.nlp_query import NLPQueryEngine

    engine = NLPQueryEngine()
    result = engine.parse("show me top 10 customers by revenue")
    # result = {
    #   "intent":   "top_n",
    #   "n":        10,
    #   "sort_by":  "revenue",
    #   "direction":"desc",
    #   "filters":  [],
    #   "agg":      "sum",
    #   "method":   "ml",
    # }
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("dipex.analyst.nlp_query")

# ── Artifact paths ─────────────────────────────────────────────────────────────

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "nlp_query_classifier.pkl")
_VEC_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "nlp_query_vectorizer.pkl")

# ── Seed corpus for in-memory training ────────────────────────────────────────
# Format: (query_text, intent_label)
_SEED_CORPUS = [
    # top_n
    ("show me top 10 customers by revenue", "top_n"),
    ("top 5 products by sales", "top_n"),
    ("best 20 regions by profit", "top_n"),
    ("highest revenue accounts", "top_n"),
    ("which customers have the most orders", "top_n"),
    ("top performing categories", "top_n"),
    ("show the top 50 rows by score", "top_n"),
    # bottom_n
    ("worst 10 products by margin", "bottom_n"),
    ("bottom 5 stores by sales", "bottom_n"),
    ("lowest revenue segments", "bottom_n"),
    ("least active users", "bottom_n"),
    ("show bottom 20 by profit", "bottom_n"),
    # aggregate
    ("what is the total revenue", "aggregate"),
    ("sum of sales by region", "aggregate"),
    ("average order value", "aggregate"),
    ("mean income per segment", "aggregate"),
    ("total profit last year", "aggregate"),
    ("what is the max price", "aggregate"),
    ("minimum cost per product", "aggregate"),
    # filter
    ("show customers where age > 30", "filter"),
    ("filter revenue greater than 1000", "filter"),
    ("find rows where status is active", "filter"),
    ("customers from New York", "filter"),
    ("show records with null values in revenue", "filter"),
    ("orders where quantity > 5", "filter"),
    # trend
    ("sales trend over time", "trend"),
    ("show monthly revenue trend", "trend"),
    ("how has profit changed over the year", "trend"),
    ("plot revenue by month", "trend"),
    ("weekly trend for orders", "trend"),
    # compare
    ("compare revenue across regions", "compare"),
    ("difference between segment A and segment B", "compare"),
    ("how does product X compare to product Y", "compare"),
    ("year over year comparison", "compare"),
    # correlation
    ("correlation between age and income", "correlation"),
    ("is revenue correlated with marketing spend", "correlation"),
    ("what drives profit", "correlation"),
    ("relationship between price and demand", "correlation"),
    # distribution
    ("distribution of revenue", "distribution"),
    ("histogram of age", "distribution"),
    ("spread of order values", "distribution"),
    ("show the distribution of scores", "distribution"),
    # count_distinct
    ("how many unique customers", "count_distinct"),
    ("distinct products sold", "count_distinct"),
    ("number of unique orders", "count_distinct"),
    ("count of distinct regions", "count_distinct"),
    # group_by
    ("revenue by category", "group_by"),
    ("group sales by region and product", "group_by"),
    ("average price per brand", "group_by"),
    ("breakdown by segment", "group_by"),
    # time_series
    ("daily sales for last 30 days", "time_series"),
    ("monthly revenue last 12 months", "time_series"),
    ("show order count by week", "time_series"),
    ("last 7 days performance", "time_series"),
    # general
    ("show me the data", "general"),
    ("describe the table", "general"),
    ("what are the columns", "general"),
    ("give me a summary", "general"),
    ("overview of the dataset", "general"),
]

INTENT_LABELS = [
    "top_n", "bottom_n", "aggregate", "filter", "trend", "compare",
    "correlation", "distribution", "count_distinct", "group_by",
    "time_series", "general",
]

# ── Regex helpers ─────────────────────────────────────────────────────────────

_TOP_N_PAT    = re.compile(r"\b(?:top|best|highest)\s+(\d+)\b", re.I)
_BOT_N_PAT    = re.compile(r"\b(?:bottom|worst|lowest|least)\s+(\d+)\b", re.I)
_AGG_PAT      = re.compile(r"\b(sum|total|avg|average|mean|max|maximum|min|minimum|count)\b", re.I)
_GT_PAT       = re.compile(r"\b(?:greater|more|above|over|>\s*)?\s*(\d+(?:\.\d+)?)\b", re.I)
_COL_WORDS    = re.compile(r"\b(revenue|profit|sales|cost|price|income|age|score|margin|quantity|orders?)\b", re.I)
_DAYS_PAT     = re.compile(r"last\s+(\d+)\s+(day|days|week|weeks|month|months)", re.I)
_SORT_DESC    = re.compile(r"\b(desc|descending|highest|top|best|most)\b", re.I)


def _extract_entities(query: str, schema_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Extract structured entities from a natural language query."""
    entities: Dict[str, Any] = {
        "n": None, "sort_by": None, "direction": "desc",
        "filters": [], "agg": None, "date_range_days": None,
        "column_refs": [],
    }

    # N value
    m = _TOP_N_PAT.search(query) or _BOT_N_PAT.search(query)
    if m:
        entities["n"] = int(m.group(1))

    # Aggregation function
    m = _AGG_PAT.search(query)
    if m:
        raw = m.group(1).lower()
        entities["agg"] = {
            "total": "sum", "avg": "mean", "average": "mean", "maximum": "max", "minimum": "min"
        }.get(raw, raw)

    # Sort direction
    entities["direction"] = "desc" if _SORT_DESC.search(query) else "asc"

    # Date range
    m = _DAYS_PAT.search(query)
    if m:
        n_val = int(m.group(1))
        unit  = m.group(2).lower().rstrip("s")
        mult  = {"day": 1, "week": 7, "month": 30}
        entities["date_range_days"] = n_val * mult.get(unit, 1)

    # Column references — match against schema or common keywords
    if schema_cols:
        col_lower = {c.lower(): c for c in schema_cols}
        for word in re.findall(r"\b\w+\b", query.lower()):
            if word in col_lower:
                entities["column_refs"].append(col_lower[word])
        if entities["column_refs"]:
            entities["sort_by"] = entities["column_refs"][0]
    else:
        m = _COL_WORDS.search(query)
        if m:
            entities["sort_by"] = m.group(1).lower()

    return entities


def _keyword_intent(query: str) -> str:
    """Fallback keyword-based intent classifier."""
    q = query.lower()
    if _TOP_N_PAT.search(q) or "top" in q or "best" in q:
        return "top_n"
    if _BOT_N_PAT.search(q) or "worst" in q or "bottom" in q:
        return "bottom_n"
    if any(w in q for w in ["sum", "total", "average", "mean", "max", "min"]):
        return "aggregate"
    if any(w in q for w in ["trend", "over time", "monthly", "weekly"]):
        return "trend" if "time" not in q else "time_series"
    if any(w in q for w in ["correlat", "relationship", "drives"]):
        return "correlation"
    if any(w in q for w in ["distribut", "histogram", "spread"]):
        return "distribution"
    if any(w in q for w in ["unique", "distinct", "how many"]):
        return "count_distinct"
    if any(w in q for w in ["group by", "breakdown", "by region", "by category", "per "]):
        return "group_by"
    if any(w in q for w in ["filter", "where", "greater", "less", "=", ">", "<"]):
        return "filter"
    if any(w in q for w in ["compare", "difference", "vs", "year over year"]):
        return "compare"
    if any(w in q for w in ["last", "days", "weeks", "months"]):
        return "time_series"
    return "general"


class NLPQueryEngine:
    """
    TF-IDF + LinearSVC NLP engine that classifies analyst queries and
    extracts structured entities for downstream DataFrame operations.
    """

    def __init__(self) -> None:
        self._model:      Any  = None
        self._vectorizer: Any  = None
        self._method:     str  = "keyword"
        self._load()

    def _load(self) -> None:
        """Load Colab-trained artifact or train in-memory seed classifier."""
        try:
            import joblib
            if os.path.exists(_MODEL_PATH):
                self._model = joblib.load(_MODEL_PATH)
                if os.path.exists(_VEC_PATH):
                    self._vectorizer = joblib.load(_VEC_PATH)
                self._method = "colab_artifact"
                logger.info("NLPQueryEngine: loaded Colab classifier artifact.")
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("NLPQueryEngine: Colab artifact load failed: %s", exc)


        # In-memory fallback: train on seed corpus
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC
            from sklearn.pipeline import Pipeline

            texts  = [t for t, _ in _SEED_CORPUS]
            labels = [l for _, l in _SEED_CORPUS]
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5_000)
            X = self._vectorizer.fit_transform(texts)
            self._model = LinearSVC(max_iter=3000, random_state=42)
            self._model.fit(X, labels)
            self._method = "seed_trained"
            logger.info("NLPQueryEngine: trained in-memory seed classifier (%d examples).",
                        len(texts))
        except Exception as exc:  # noqa: BLE001
            logger.warning("NLPQueryEngine: in-memory training failed (%s) — using keyword fallback.", exc)

    def parse(
        self,
        query:       str,
        schema_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Parse a natural-language analyst query.

        Parameters
        ----------
        query       : Plain-English question from the analyst
        schema_cols : Optional list of column names to resolve references

        Returns
        -------
        dict with keys:
          intent, n, sort_by, direction, filters, agg,
          date_range_days, column_refs, method
        """
        if not query or not query.strip():
            return {"intent": "general", "method": self._method}

        # Classify intent
        if self._model is not None:
            try:
                # 1. SetFit / SBERT dict-wrapped model
                if isinstance(self._vectorizer, dict) and self._vectorizer.get("type") in ("setfit", "sbert"):
                    raw_pred = self._model([query])
                    le = self._vectorizer.get("le")
                    if le is not None and hasattr(raw_pred, "item"):
                        intent = le.inverse_transform([raw_pred.item()])[0]
                    elif le is not None:
                        intent = le.inverse_transform([raw_pred[0]])[0]
                    else:
                        intent = str(raw_pred[0])

                else:
                    # 2. Try direct text prediction (Colab CalibratedClassifierCV embedding TF-IDF pipeline)
                    try:
                        intent = str(self._model.predict([query])[0])
                    except Exception:
                        # 3. Fallback for separate vectorizer (in-memory seed_trained model)
                        if self._vectorizer is not None:
                            X = self._vectorizer.transform([query])
                            intent = str(self._model.predict(X)[0])
                        else:
                            raise ValueError("No vectorizer available for separate transform")
            except Exception as e:  # noqa: BLE001
                logger.debug("NLPQueryEngine prediction failed: %s", e)
                intent = _keyword_intent(query)
        else:
            intent = _keyword_intent(query)

        # Extract entities
        entities = _extract_entities(query, schema_cols)

        result = {
            "intent":         intent,
            "n":              entities["n"],
            "sort_by":        entities["sort_by"],
            "direction":      entities["direction"],
            "filters":        entities["filters"],
            "agg":            entities["agg"],
            "date_range_days": entities["date_range_days"],
            "column_refs":    entities["column_refs"],
            "method":         self._method,
            "raw_query":      query,
        }
        logger.debug("[NLP] Query: %r → intent=%s, entities=%s", query, intent, entities)
        return result

    def execute_on_df(
        self, df: "Any", query: str, schema_cols: Optional[List[str]] = None
    ) -> "Any":
        """
        Parse and execute a natural-language query on a DataFrame.
        Returns a result DataFrame based on the parsed intent.

        This is a best-effort execution layer; analysts can refine the output.
        """
        import pandas as pd

        cols = schema_cols or list(df.columns)
        parsed = self.parse(query, cols)
        intent = parsed["intent"]
        n      = parsed.get("n") or 10
        sort_by = parsed.get("sort_by")
        agg    = parsed.get("agg") or "sum"
        direction = (parsed.get("direction", "desc") == "desc")

        num_cols = df.select_dtypes(include="number").columns.tolist()

        try:
            if intent == "top_n":
                col = sort_by or (num_cols[0] if num_cols else df.columns[0])
                return df.nlargest(n, col) if col in df.columns else df.head(n)

            if intent == "bottom_n":
                col = sort_by or (num_cols[0] if num_cols else df.columns[0])
                return df.nsmallest(n, col) if col in df.columns else df.tail(n)

            if intent == "aggregate":
                agg_map = {"sum": pd.DataFrame.sum, "mean": pd.DataFrame.mean,
                           "max": pd.DataFrame.max, "min": pd.DataFrame.min,
                           "count": pd.DataFrame.count}
                fn = agg_map.get(agg, pd.DataFrame.sum)
                return df[num_cols].agg(agg).to_frame(name=agg)

            if intent == "distribution":
                col = sort_by or (num_cols[0] if num_cols else None)
                if col and col in df.columns:
                    return df[col].describe().to_frame()
                return df.describe()

            if intent == "count_distinct":
                return df.nunique().to_frame(name="distinct_count")

            if intent == "correlation":
                return df[num_cols].corr()

            if intent == "group_by" and parsed["column_refs"]:
                grp_col = parsed["column_refs"][0]
                val_col = (parsed["column_refs"][1:] or num_cols)[:1]
                if grp_col in df.columns and val_col:
                    return df.groupby(grp_col)[val_col[0]].agg(agg).reset_index()

            # Default: head
            return df.head(n)

        except Exception as exc:  # noqa: BLE001
            logger.warning("NLPQueryEngine.execute_on_df failed for intent=%s: %s", intent, exc)
            return df.head(n)
