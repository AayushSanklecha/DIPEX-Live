"""
proposal/__init__.py
Proposal Layer package — AI-assistive only. No data mutation.
"""
from proposal.proposal_router import ProposalRouter
from proposal.insight_ranker import (
    InsightRanker,
    FeatureProposer,
    AnomalyFlagger,
    RAGRecall,
)
from proposal.rag_retriever import RAGRetriever

__all__ = [
    "ProposalRouter",
    "InsightRanker",
    "FeatureProposer",
    "AnomalyFlagger",
    "RAGRecall",     # Lightweight: feature-vector hashing (no LLM needed)
    "RAGRetriever",  # Full: SentenceTransformer + ChromaDB embeddings
]
