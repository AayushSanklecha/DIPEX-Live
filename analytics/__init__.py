"""
analytics/__init__.py
----------------------
AI & ANALYTICS SERVICE LAYER — Analytics Orchestrator.

Sequences: AutoEDA → FeatureEngineering → InsightRanking → LLM Summarization
"""

from analytics.orchestrator import AnalyticsOrchestrator, AnalyticsResult

__all__ = ["AnalyticsOrchestrator", "AnalyticsResult"]
