"""
preprocessing/nlp_column_analyzer.py
======================================
Production-grade NLP engine for semantic column and row name understanding.

Used by ADAP at two points:
  1. Schema classification (replaces/augments regex with vector similarity)
  2. Domain classification (dataset-level name semantics)

Architecture:
  Primary  : sentence-transformers (all-MiniLM-L6-v2, 384-dim, 80MB)
             → cosine similarity to 21 semantic-type anchors (per class average)
             → cosine similarity to 7 domain anchors
  Fallback : spaCy (en_core_web_sm, NER + lemmatization)
  Fallback2: KeywordMatcher (regex + synonym table) — zero-dependency fallback

Why sentence-transformers?
  Unlike simple keyword matching, the model understands that:
    "txn_amt"   → amount   (abbreviation semantics)
    "dob"       → age      (acronym: date of birth)
    "cust_id"   → id       (compound)
    "px_amount" → amount   (pharmaceutical abbreviation)
  Production data catalog tools (Atlan, Collibra, Alation) use this exact approach.

Exports:
  NLPColumnAnalyzer.analyze_column(col_name, series?) -> SemanticResult
  NLPColumnAnalyzer.analyze_dataframe(df) -> Dict[col, SemanticResult]
  add_nlp_features(df, col_name, existing_feats) -> extended feature dict
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("dipex.preprocessing.nlp_column_analyzer")

# ── Semantic type anchor phrases ────────────────────────────────────────────────
# Each label has multiple anchor phrases — the embedding is the MEAN of all anchors.
# Elite practice: use diverse phrasings to capture abbreviations, jargon, synonyms.
SEMANTIC_ANCHORS: Dict[str, List[str]] = {
    "id": [
        "unique identifier column", "primary key field", "record id number",
        "customer id", "user id", "transaction id", "surrogate key",
        "uuid column", "object id", "entity identifier",
    ],
    "age": [
        "age in years", "person age", "customer age", "patient age",
        "date of birth derived age", "age at time of event",
        "years old", "duration since birth",
    ],
    "amount": [
        "monetary amount in dollars", "transaction amount", "payment amount",
        "revenue figure", "cost value", "price column", "financial amount",
        "balance amount", "expense total", "income value",
        "loan amount principal", "fee charged", "tax amount",
    ],
    "date": [
        "date column timestamp", "event date", "transaction date",
        "created at datetime", "updated at timestamp",
        "date of birth", "effective date", "expiry date",
        "reporting period date", "calendar date string",
    ],
    "category": [
        "categorical variable", "class label", "group type",
        "product category", "status column", "segment label",
        "classification bucket", "type indicator", "domain category",
    ],
    "text": [
        "free text description", "notes field", "comments column",
        "narrative text", "long text string", "remarks field",
        "open ended response", "text blob", "description paragraph",
    ],
    "phone": [
        "phone number", "mobile number", "telephone number",
        "contact number", "cell phone", "fax number",
        "international phone number with country code",
    ],
    "email": [
        "email address", "email id column", "user email",
        "contact email", "electronic mail address",
    ],
    "boolean": [
        "binary flag indicator", "yes no column", "true false flag",
        "boolean indicator", "binary column", "active inactive flag",
        "is active column", "has feature flag",
    ],
    "zipcode": [
        "zip code postal code", "pin code", "postal area code",
        "zip column", "postcode", "area code",
    ],
    "percentage": [
        "percentage value", "ratio proportion", "fractional rate",
        "percent column", "growth rate percentage", "utilization rate",
    ],
    "score": [
        "credit score", "risk score", "model score prediction",
        "rating value", "performance score", "grade point average",
        "customer satisfaction score", "propensity score",
    ],
    "count": [
        "count of occurrences", "number of items", "frequency count",
        "quantity column", "total number", "visit count",
        "transaction count", "order count",
    ],
    "name": [
        "person name", "customer name", "full name",
        "first name last name", "company name",
        "organization name", "entity name",
    ],
    "url": [
        "url web address", "hyperlink column", "website url",
        "api endpoint url", "resource url link",
    ],
    "ip_address": [
        "ip address inet", "network address", "ipv4 address",
        "ipv6 address", "server ip", "client ip address",
    ],
    "coordinates": [
        "latitude longitude coordinate", "gps coordinate",
        "geographic coordinate", "lat lon column",
        "geospatial point coordinate",
    ],
    "duration": [
        "duration in seconds", "time elapsed", "session duration",
        "call duration", "response time", "processing time seconds",
    ],
    "address": [
        "street address", "mailing address", "residential address",
        "delivery address", "physical location address",
    ],
    "currency_code": [
        "currency code iso", "currency type", "payment currency",
        "transaction currency code", "forex currency",
    ],
    "unknown": [
        "unknown column type", "unclassified column", "miscellaneous field",
        "other data column",
    ],
}

# Domain-level anchors (for dataset-level domain classification)
DOMAIN_ANCHORS: Dict[str, List[str]] = {
    "banking": [
        "bank account transaction", "loan repayment schedule", "aml kyc compliance",
        "iban swift code", "collateral mortgage", "debit credit ledger",
        "banking customer transaction history",
    ],
    "healthcare": [
        "patient diagnosis record", "icd code clinical", "drug dosage prescription",
        "bmi vital signs", "hospital admission discharge",
        "medical procedure code", "healthcare provider patient",
    ],
    "finance": [
        "stock price trading volume", "eps earnings per share",
        "market capitalization", "ebitda profit loss",
        "portfolio return investment", "equity fund nav",
    ],
    "ecommerce": [
        "product sku inventory", "shopping cart order basket",
        "customer checkout return", "product review rating",
        "delivery shipping tracking", "merchant seller storefront",
    ],
    "government": [
        "census population data", "government policy regulation",
        "public expenditure budget", "taxpayer national id",
        "municipal district census", "election voter registration",
    ],
    "insurance": [
        "insurance policy premium", "claim settlement actuarial",
        "underwriting risk assessment", "beneficiary coverage",
        "reinsurance treaty", "loss ratio reserve",
    ],
    "generic": [
        "general purpose data column", "research dataset",
        "scientific measurement tabular", "generic numeric data",
    ],
}

# ── Keyword fallback (zero-dependency) ─────────────────────────────────────────
_KW_MAP: Dict[str, List[str]] = {
    "id": ["id","uuid","key","pk","identifier","surrogate","ref","code","num"],
    "age": ["age","dob","birth","yr","year_old","years"],
    "amount": ["amount","amt","price","cost","revenue","fee","tax","balance","payment",
               "total","sum","salary","income","expense","spend","value","charge",
               "txn_amt","trx_amt","px","pay"],
    "date": ["date","dt","time","timestamp","created","updated","effective","expired",
             "period","month","quarter","year","at","on"],
    "category": ["type","cat","category","class","segment","group","tier","label","kind"],
    "text": ["text","note","comment","description","remark","narrative","memo","message"],
    "phone": ["phone","mobile","tel","cell","fax","contact_no","ph_no"],
    "email": ["email","mail","e-mail","emailid","email_id"],
    "boolean": ["flag","is_","has_","active","enabled","status","bool","binary"],
    "zipcode": ["zip","postal","pincode","postcode"],
    "percentage": ["pct","percent","ratio","rate","prop","proportion"],
    "score": ["score","rating","grade","rank","gpa","fico","nps"],
    "count": ["count","cnt","num","qty","quantity","frequency","total","n_"],
    "name": ["name","fname","lname","fullname","company","firm","org"],
    "url": ["url","link","href","website","endpoint","uri"],
    "ip_address": ["ip","inet","ipv4","ipv6","addr"],
    "coordinates": ["lat","lon","latitude","longitude","geom","coord","gps"],
    "duration": ["duration","elapsed","period","seconds","mins","hours","ttl"],
    "address": ["address","addr","street","city","state","location","place"],
    "currency_code": ["currency","ccy","curr","fx"],
}


@dataclass
class SemanticResult:
    column_name: str
    semantic_type: str                        # top predicted type
    semantic_type_probs: Dict[str, float]     # prob for each type
    domain_signals: Dict[str, float]          # domain similarity scores
    nlp_features: np.ndarray                  # 21+7 = 28 dim similarity vector
    method: str                               # 'sentence_transformers' | 'spacy' | 'keyword'
    confidence: float                         # confidence in top prediction


class NLPColumnAnalyzer:
    """
    Production-grade NLP column semantic analyzer.

    Uses sentence-transformers to embed column names and compute
    cosine similarity to manually curated semantic anchor phrases.

    Gracefully degrades through 3 backends:
      1. sentence-transformers (all-MiniLM-L6-v2)
      2. spaCy (en_core_web_sm)
      3. Keyword matching (always available)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._encoder   = None
        self._spacy_nlp = None
        self._method    = "keyword"
        self._anchor_vectors: Optional[Dict[str, np.ndarray]] = None

        # Type and domain label lists (fixed order — matches feature vector)
        self.type_labels   = list(SEMANTIC_ANCHORS.keys())
        self.domain_labels = list(DOMAIN_ANCHORS.keys())

        self._init_encoder()

    def _init_encoder(self) -> None:
        """Try to load sentence-transformers; fall back to spaCy; then keyword."""
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self._model_name)
            self._encoder.encode(["test"])  # warm up
            self._method = "sentence_transformers"
            self._precompute_anchors()
            logger.info("[NLPColumnAnalyzer] sentence-transformers loaded: %s", self._model_name)
            return
        except Exception as e:
            logger.debug("[NLPColumnAnalyzer] sentence-transformers unavailable: %s", e)

        try:
            import spacy
            self._spacy_nlp = spacy.load("en_core_web_sm")
            self._method = "spacy"
            logger.info("[NLPColumnAnalyzer] spaCy backend loaded")
            return
        except Exception as e:
            logger.debug("[NLPColumnAnalyzer] spaCy unavailable: %s", e)

        self._method = "keyword"
        logger.info("[NLPColumnAnalyzer] Using keyword fallback (no DL models available)")

    def _precompute_anchors(self) -> None:
        """Pre-compute mean anchor embeddings for all semantic types and domains."""
        from sentence_transformers import SentenceTransformer
        st: SentenceTransformer = self._encoder  # type: ignore

        self._anchor_vectors = {}

        # Semantic type anchors
        for label, phrases in SEMANTIC_ANCHORS.items():
            vecs = st.encode(phrases, normalize_embeddings=True, show_progress_bar=False)
            self._anchor_vectors[f"type_{label}"] = vecs.mean(axis=0)

        # Domain anchors
        for label, phrases in DOMAIN_ANCHORS.items():
            vecs = st.encode(phrases, normalize_embeddings=True, show_progress_bar=False)
            self._anchor_vectors[f"domain_{label}"] = vecs.mean(axis=0)

        logger.info("[NLPColumnAnalyzer] Anchor embeddings precomputed (%d types, %d domains)",
                    len(SEMANTIC_ANCHORS), len(DOMAIN_ANCHORS))

    def _normalize_col_name(self, col_name: str) -> str:
        """
        Convert raw column names to readable phrases for embedding.
        Examples:
          "txn_amt"    → "txn amt"
          "cust_DOB"   → "cust DOB"
          "totalRevenue" → "total Revenue"
          "LOAN_AMOUNT_USD" → "loan amount usd"
        """
        col = col_name.strip()
        # camelCase split
        col = re.sub(r"([a-z])([A-Z])", r"\1 \2", col)
        # snake/dash to spaces
        col = re.sub(r"[_\-/.]", " ", col)
        # multiple spaces
        col = re.sub(r"\s+", " ", col).strip().lower()
        return col

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a_n = a / (np.linalg.norm(a) + 1e-9)
        b_n = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a_n, b_n))

    def analyze_column(
        self,
        col_name: str,
        series: Optional[Any] = None,
    ) -> SemanticResult:
        """
        Analyze a single column name (+ optionally its data).

        Returns SemanticResult with:
          - semantic_type: predicted label
          - semantic_type_probs: {type: probability} over all 21 types
          - domain_signals: {domain: similarity} over 7 domains
          - nlp_features: 28-dim numpy vector (21 type sims + 7 domain sims)
        """
        if self._method == "sentence_transformers":
            return self._analyze_st(col_name)
        elif self._method == "spacy":
            return self._analyze_spacy(col_name)
        else:
            return self._analyze_keyword(col_name)

    def _analyze_st(self, col_name: str) -> SemanticResult:
        """sentence-transformers analysis."""
        readable = self._normalize_col_name(col_name)
        vec = self._encoder.encode([readable], normalize_embeddings=True,  # type: ignore
                                   show_progress_bar=False)[0]

        type_sims: Dict[str, float] = {}
        for label in self.type_labels:
            anchor = self._anchor_vectors[f"type_{label}"]  # type: ignore
            type_sims[label] = self._cosine_sim(vec, anchor)

        domain_sims: Dict[str, float] = {}
        for label in self.domain_labels:
            anchor = self._anchor_vectors[f"domain_{label}"]  # type: ignore
            domain_sims[label] = self._cosine_sim(vec, anchor)

        # Softmax over type similarities
        type_arr = np.array([type_sims[k] for k in self.type_labels])
        type_arr = np.exp(type_arr * 5.0)   # temperature=0.2 sharpening
        type_probs = type_arr / type_arr.sum()

        top_type = self.type_labels[int(np.argmax(type_probs))]
        confidence = float(type_probs.max())

        nlp_features = np.concatenate([type_probs, np.array([domain_sims[l] for l in self.domain_labels])])

        return SemanticResult(
            column_name=col_name,
            semantic_type=top_type,
            semantic_type_probs=dict(zip(self.type_labels, type_probs.tolist())),
            domain_signals=domain_sims,
            nlp_features=nlp_features.astype(np.float32),
            method="sentence_transformers",
            confidence=confidence,
        )

    def _analyze_spacy(self, col_name: str) -> SemanticResult:
        """spaCy-based analysis using token lemmas and NER."""
        readable = self._normalize_col_name(col_name)
        doc = self._spacy_nlp(readable)  # type: ignore
        lemmas = {t.lemma_.lower() for t in doc}

        scores = {label: 0.0 for label in self.type_labels}
        for label, kws in _KW_MAP.items():
            for kw in kws:
                if any(kw in lemma or lemma in kw for lemma in lemmas):
                    scores[label] = scores.get(label, 0.0) + 1.0

        # Normalize
        total = max(sum(scores.values()), 1.0)
        for k in scores:
            scores[k] /= total

        top_type = max(scores, key=scores.get)  # type: ignore
        if scores[top_type] == 0.0:
            top_type = "unknown"
            scores["unknown"] = 1.0

        arr = np.array([scores.get(k, 0.0) for k in self.type_labels], dtype=np.float32)
        domain_sims = {d: 0.0 for d in self.domain_labels}
        nlp_features = np.concatenate([arr, np.zeros(len(self.domain_labels))]).astype(np.float32)

        return SemanticResult(
            column_name=col_name,
            semantic_type=top_type,
            semantic_type_probs=scores,
            domain_signals=domain_sims,
            nlp_features=nlp_features,
            method="spacy",
            confidence=float(scores.get(top_type, 0.0)),
        )

    def _analyze_keyword(self, col_name: str) -> SemanticResult:
        """Zero-dependency keyword fallback."""
        col_l = self._normalize_col_name(col_name)
        scores = {label: 0.0 for label in self.type_labels}

        for label, kws in _KW_MAP.items():
            for kw in kws:
                if kw in col_l:
                    scores[label] = scores.get(label, 0.0) + (1.0 if kw in col_l.split() else 0.5)

        total = max(sum(scores.values()), 1.0)
        for k in scores:
            scores[k] /= total

        top_type = max(scores, key=scores.get)  # type: ignore
        if scores[top_type] < 0.1:
            top_type = "unknown"
            scores["unknown"] = 1.0

        arr = np.array([scores.get(k, 0.0) for k in self.type_labels], dtype=np.float32)
        domain_sims = {d: 0.0 for d in self.domain_labels}
        nlp_features = np.concatenate([arr, np.zeros(len(self.domain_labels))]).astype(np.float32)

        return SemanticResult(
            column_name=col_name,
            semantic_type=top_type,
            semantic_type_probs=scores,
            domain_signals=domain_sims,
            nlp_features=nlp_features,
            method="keyword",
            confidence=float(scores.get(top_type, 0.0)),
        )

    def analyze_dataframe(self, df) -> Dict[str, SemanticResult]:
        """Analyze all columns in a DataFrame."""
        results = {}
        for col in df.columns:
            try:
                results[col] = self.analyze_column(col, df[col])
            except Exception as e:
                logger.debug("[NLPColumnAnalyzer] Column '%s' failed: %s", col, e)
                results[col] = self._analyze_keyword(col)
        return results

    def get_nlp_feature_names(self) -> List[str]:
        """Return ordered list of NLP feature names (28 total)."""
        return (
            [f"nlp_type_{t}" for t in self.type_labels]
            + [f"nlp_domain_{d}" for d in self.domain_labels]
        )


def add_nlp_features(
    col_name: str,
    existing_feats: Dict[str, float],
    analyzer: Optional[NLPColumnAnalyzer] = None,
) -> Dict[str, float]:
    """
    Augment existing statistical feature dict with 28 NLP features.
    Used at both training time (in Colab) and inference time (in ADAP pipeline).

    Parameters
    ----------
    col_name       : raw column name string
    existing_feats : output of extract_column_features() — statistical features
    analyzer       : shared NLPColumnAnalyzer instance (pass to avoid re-init overhead)

    Returns
    -------
    Extended feature dict (statistical + NLP = 30 + 28 = 58 features)
    """
    if analyzer is None:
        analyzer = _get_global_analyzer()

    result = analyzer.analyze_column(col_name)
    nlp_names = analyzer.get_nlp_feature_names()

    augmented = dict(existing_feats)
    for name, val in zip(nlp_names, result.nlp_features.tolist()):
        augmented[name] = val
    return augmented


# ── Module-level lazy singleton ────────────────────────────────────────────────
_GLOBAL_ANALYZER: Optional[NLPColumnAnalyzer] = None


def _get_global_analyzer() -> NLPColumnAnalyzer:
    global _GLOBAL_ANALYZER
    if _GLOBAL_ANALYZER is None:
        _GLOBAL_ANALYZER = NLPColumnAnalyzer()
    return _GLOBAL_ANALYZER


def get_analyzer() -> NLPColumnAnalyzer:
    """Get the module-level NLPColumnAnalyzer singleton."""
    return _get_global_analyzer()


# ── Utility: domain signal for a whole DataFrame ───────────────────────────────
def infer_dataset_domain(df, analyzer: Optional[NLPColumnAnalyzer] = None) -> Dict[str, Any]:
    """
    Infer the likely domain of an entire dataset from its column names.

    Returns dict with:
      - top_domain: most likely domain
      - domain_scores: {domain: score}
      - column_types: {col: predicted_type}
      - method: backend used
    """
    if analyzer is None:
        analyzer = _get_global_analyzer()

    results = analyzer.analyze_dataframe(df)

    # Aggregate domain scores across all columns
    agg: Dict[str, float] = {d: 0.0 for d in analyzer.domain_labels}
    for r in results.values():
        for domain, score in r.domain_signals.items():
            agg[domain] = agg.get(domain, 0.0) + score

    total = max(sum(agg.values()), 1e-9)
    normalized = {k: round(v / total, 4) for k, v in agg.items()}
    top_domain = max(normalized, key=normalized.get)  # type: ignore

    return {
        "top_domain": top_domain,
        "domain_scores": normalized,
        "column_types": {col: r.semantic_type for col, r in results.items()},
        "column_confidences": {col: round(r.confidence, 3) for col, r in results.items()},
        "method": list(results.values())[0].method if results else "keyword",
    }
