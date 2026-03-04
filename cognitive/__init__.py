"""
cognitive/__init__.py
----------------------
Cognitive Reasoning Engine — Human-like analytical cognition layer.

Provides: sanity checking, assumption tracking, leakage detection,
expectation calibration, and explicit uncertainty quantification.
"""
from cognitive.reasoning_engine import CognitiveReasoningEngine, AnalysisContext, CognitiveFinding
from cognitive.sanity_checker import SanityChecker, SanityViolation
from cognitive.assumption_tracker import AssumptionTracker, Assumption
from cognitive.leakage_sentinel import LeakageSentinel, LeakageWarning
from cognitive.expectation_calibrator import ExpectationCalibrator, InsightVerdict
from cognitive.uncertainty_quantifier import UncertaintyQuantifier, UncertaintyReport

__all__ = [
    "CognitiveReasoningEngine", "AnalysisContext", "CognitiveFinding",
    "SanityChecker", "SanityViolation",
    "AssumptionTracker", "Assumption",
    "LeakageSentinel", "LeakageWarning",
    "ExpectationCalibrator", "InsightVerdict",
    "UncertaintyQuantifier", "UncertaintyReport",
]
